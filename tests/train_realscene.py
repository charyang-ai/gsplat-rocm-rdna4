#!/usr/bin/env python3
"""Train 3DGS on a Mip-NeRF 360 scene to obtain a realistic Gaussian population.

Why this exists rather than `examples/simple_trainer.py`. The intersection paper's
numbers are all measured on synthetic Gaussians, which is defensible -- the cost of
the stage is governed by pairs per Gaussian, and a sweep over that statistic says
more than any single scene -- but the reader still wants to know where real scenes
land on that curve. Getting there needs trained scenes, and simple_trainer.py cannot
run on this host: it imports `torchmetrics.image.lpip` at module scope, which needs
torchvision, and there is no torchvision wheel for torch 2.13+rocm7.2. Stubbing an
import that the trainer instantiates in `__init__` is the kind of fix that works
until it silently does not, so the loop is reproduced here instead.

What is reproduced, and therefore what "trained the same way" means: the SfM
initialisation, the per-parameter learning rates and their batch-size scaling, the
exponential decay on `means`, the SH degree warmup, the 0.8*L1 + 0.2*(1-SSIM) loss,
and `DefaultStrategy` with its stock densification thresholds -- all copied from
simple_trainer.py, which is cited in the file so the two can be diffed. What is not
reproduced: the viewer, tensorboard, video rendering, PNG compression, pose and
appearance optimisation, the bilateral grid, and the depth loss, none of which are
on by default. Evaluation keeps PSNR only; LPIPS is the thing we cannot import, and
its absence does not matter for a population of Gaussians.

The SSIM term comes from the sibling TriSSIM package, which is what the other two
papers in this series measure and is installed here as `fused_ssim_biauto`.

Usage:
    GSPLAT_TILE_SIZE=8 python tests/train_realscene.py \
        --data_dir .gsplat/examples/data/garden --result_dir results/realscene/garden

Environment:
    GSPLAT_TILE_SIZE   required, forced on every rasterization() call, as in
                       tests/run_simple_trainer.py -- an implicit tile size is
                       exactly the variable this series of papers is about
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.join(os.path.dirname(_HERE), ".gsplat", "examples")


def _install_examples_path() -> None:
    """Put gsplat's examples dir on sys.path for its dataset parser and utils."""
    if not os.path.isdir(_EXAMPLES):
        sys.exit(f"no gsplat examples at {_EXAMPLES}")
    if _EXAMPLES not in sys.path:
        sys.path.insert(0, _EXAMPLES)


def _force_tile_size() -> int:
    """Pin the rasterizer tile size for the whole process.

    Same argument as tests/run_simple_trainer.py: the fork defaults to 8 and
    upstream to 16, so leaving it implicit means not knowing which was measured.
    """
    try:
        tile_size = int(os.environ["GSPLAT_TILE_SIZE"])
    except KeyError:
        sys.exit("GSPLAT_TILE_SIZE is not set; refusing to train with an implicit "
                 "tile size")

    import gsplat.rendering

    original = gsplat.rendering.rasterization

    def rasterization(*args, **kwargs):
        kwargs["tile_size"] = tile_size
        return original(*args, **kwargs)

    gsplat.rendering.rasterization = rasterization
    return tile_size


# ----------------------------------------------------------------------------------
# Parameters and optimizers. Copied from simple_trainer.py's
# create_splats_with_optimizers() with the branches this script does not use
# (random init, appearance features, SparseAdam, SelectiveAdam, distributed)
# removed. Learning rates are simple_trainer.py's Config defaults.
# ----------------------------------------------------------------------------------
def create_splats(parser, scene_scale: float, sh_degree: int, device: str
                  ) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    from utils import knn, rgb_to_sh

    points = torch.from_numpy(parser.points).float()
    rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()

    # Initial scale is the mean distance to the 3 nearest neighbours, so a Gaussian
    # starts about as big as the gap it has to fill.
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    scales = torch.log(torch.sqrt(dist2_avg)).unsqueeze(-1).repeat(1, 3)

    N = points.shape[0]
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), 0.1))

    params = [
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = rgb_to_sh(rgbs)
    params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
    params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr, "name": name}],
            eps=1e-15,
            betas=(0.9, 0.999),
        )
        for name, _, lr in params
    }
    return splats, optimizers


def render(splats: torch.nn.ParameterDict, camtoworlds, Ks, width: int, height: int,
           sh_degree_to_use: int, absgrad: bool):
    from gsplat.rendering import rasterization

    return rasterization(
        means=splats["means"],
        quats=splats["quats"],  # rasterization normalizes internally
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=torch.cat([splats["sh0"], splats["shN"]], 1),
        viewmats=torch.linalg.inv(camtoworlds),
        Ks=Ks,
        width=width,
        height=height,
        packed=False,
        absgrad=absgrad,
        sparse_grad=False,
        rasterize_mode="classic",
        sh_degree=sh_degree_to_use,
        near_plane=0.01,
        far_plane=1e10,
    )


@torch.no_grad()
def evaluate(splats, valset, sh_degree: int, device: str) -> Dict[str, float]:
    """Mean PSNR over the held-out views.

    PSNR is a closed-form function of MSE, so it needs no torchmetrics. It is here
    only to certify that the checkpoint is a properly trained scene and not a
    diverged one -- the paper's claims are about the Gaussian population, not about
    image quality, which the sister papers measure.
    """
    total, count = 0.0, 0
    for data in valset:
        camtoworlds = data["camtoworld"].to(device).unsqueeze(0)
        Ks = data["K"].to(device).unsqueeze(0)
        pixels = data["image"].to(device).unsqueeze(0) / 255.0
        height, width = pixels.shape[1:3]
        colors, _, _ = render(splats, camtoworlds, Ks, width, height, sh_degree, False)
        mse = F.mse_loss(colors.clamp(0.0, 1.0), pixels)
        total += -10.0 * math.log10(max(mse.item(), 1e-12))
        count += 1
    return {"psnr": total / max(count, 1), "n_val": count}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--data_factor", type=int, default=4,
                    help="4 is what published Mip-NeRF 360 numbers use")
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=30_000)
    ap.add_argument("--save_steps", type=int, nargs="+", default=[7_000, 30_000])
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _install_examples_path()
    tile_size = _force_tile_size()

    from datasets.colmap import Dataset, Parser
    from gsplat.strategy import DefaultStrategy
    from utils import set_random_seed
    from fused_ssim_biauto import fused_ssim

    set_random_seed(args.seed)
    device = "cuda"
    os.makedirs(args.result_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.result_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    parser = Parser(data_dir=args.data_dir, factor=args.data_factor,
                    normalize=True, test_every=8)
    trainset = Dataset(parser, split="train", patch_size=None, load_depths=False)
    valset = Dataset(parser, split="val")
    scene_scale = parser.scene_scale * 1.1
    print(f"[train_realscene] tile_size={tile_size}  scene_scale={scene_scale:.4f}  "
          f"train={len(trainset)}  val={len(valset)}", flush=True)

    splats, optimizers = create_splats(parser, scene_scale, args.sh_degree, device)
    print(f"[train_realscene] initial Gaussians: {len(splats['means'])}", flush=True)

    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    # means decays to 1% of its initial rate; every other group is constant.
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / args.max_steps)
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=4,
        persistent_workers=True, pin_memory=True,
    )
    trainloader_iter = iter(trainloader)

    history: List[dict] = []
    tic = time.time()
    for step in range(args.max_steps):
        try:
            data = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(trainloader)
            data = next(trainloader_iter)

        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device) / 255.0
        height, width = pixels.shape[1:3]

        sh_degree_to_use = min(step // 1000, args.sh_degree)
        colors, alphas, info = render(splats, camtoworlds, Ks, width, height,
                                      sh_degree_to_use, strategy.absgrad)

        strategy.step_pre_backward(params=splats, optimizers=optimizers,
                                   state=strategy_state, step=step, info=info)

        l1loss = F.l1_loss(colors, pixels)
        ssimloss = 1.0 - fused_ssim(colors.permute(0, 3, 1, 2),
                                    pixels.permute(0, 3, 1, 2), padding="valid")
        loss = l1loss * 0.8 + ssimloss * 0.2
        loss.backward()

        if step % 100 == 0:
            # flatten_ids has one entry per (Gaussian, tile) pair, so its length is
            # the stage's output size and pairs/N is the statistic the paper's
            # density sweep is parameterised by.
            n = len(splats["means"])
            pairs = int(info["flatten_ids"].numel())
            print(f"[{step:6d}] loss={loss.item():.4f} N={n} pairs={pairs} "
                  f"pairs/N={pairs / max(n, 1):.1f} "
                  f"elapsed={time.time() - tic:.0f}s", flush=True)
            history.append({"step": step, "loss": loss.item(), "n_gaussians": n,
                            "n_pairs": pairs})

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        scheduler.step()

        strategy.step_post_backward(params=splats, optimizers=optimizers,
                                    state=strategy_state, step=step, info=info,
                                    packed=False)

        if (step + 1) in args.save_steps or (step + 1) == args.max_steps:
            metrics = evaluate(splats, valset, args.sh_degree, device)
            print(f"[train_realscene] step {step + 1}: N={len(splats['means'])} "
                  f"psnr={metrics['psnr']:.3f}", flush=True)
            torch.save(
                {"step": step + 1, "splats": splats.state_dict(),
                 "scene_scale": scene_scale, "tile_size": tile_size,
                 "data_dir": os.path.abspath(args.data_dir),
                 "data_factor": args.data_factor, "psnr": metrics["psnr"]},
                os.path.join(ckpt_dir, f"ckpt_{step + 1}.pt"),
            )
            with open(os.path.join(args.result_dir, "train.json"), "w") as f:
                json.dump({"step": step + 1, "psnr": metrics["psnr"],
                           "n_gaussians": len(splats["means"]),
                           "tile_size": tile_size, "history": history}, f, indent=2)

    print(f"[train_realscene] done in {(time.time() - tic) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
