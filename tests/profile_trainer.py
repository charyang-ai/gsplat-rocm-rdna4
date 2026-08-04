"""Profile a synthetic 3DGS training step to locate the real bottleneck kernels.

`examples/simple_trainer.py` needs a COLMAP dataset and a full CLI, which makes it awkward
to profile in isolation. This harness instead drives the SAME hot path — gsplat's
`rasterization` forward + backward, an L1 + fused-SSIM photometric loss, and an Adam
step — on a randomly generated scene (no dataset required), under `torch.profiler`, and
prints the operators/kernels sorted by GPU (self CUDA/HIP) time.

That ranked table is what tells you where the time actually goes (typically the
rasterize backward dominates, well ahead of the SSIM loss), so you can prioritize
optimization instead of guessing.

`--ssim` picks the SSIM implementation and defaults to `trissim`, the fused loss the
image actually trains with. `--ras_bwd` picks the rasterizer backward and defaults to
`baseline`, the stock HIP kernel. Everything else in the step is identical across the
choices, so running the same command twice and changing only one of those flags is a
controlled before/after experiment: whatever moves in total GPU time is attributable to
that one component.

Usage (needs a GPU exposed to the container):
  python tests/profile_trainer.py
  python tests/profile_trainer.py --num-gaussians 500000 --height 1080 --width 1920
  python tests/profile_trainer.py --iters 50 --sh-degree 3
  python tests/profile_trainer.py --ssim baseline      # pure-torch 11x11 grouped conv2d
  python tests/profile_trainer.py --ssim separable     # pure-torch two 1D passes
  python tests/profile_trainer.py --ssim off           # profile rasterize + L1 only
  python tests/profile_trainer.py --ras_bwd triton     # autotuned Triton rasterize bwd
  python tests/profile_trainer.py --trace trace.json   # also dump a chrome trace
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, schedule


def _import_rasterization():
    """gsplat exposes rasterization at the top level (newer) or under .rendering."""
    try:
        from gsplat import rasterization
        return rasterization
    except Exception:
        from gsplat.rendering import rasterization  # noqa: E402
        return rasterization


# ----------------------------------------------------------------------------------
# SSIM implementations (--ssim). All three are the classic 11x11 Gaussian SSIM
# (sigma=1.5, C1=0.01^2, C2=0.03^2, mean reduction), so swapping between them changes
# only how the five blurs are computed — never what is computed.
# ----------------------------------------------------------------------------------
_C1 = 0.01 ** 2
_C2 = 0.03 ** 2
_KERNEL_CACHE: dict = {}


def _gaussian_1d(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _blur_kernels(C: int, device, dtype, separable: bool):
    """Cached conv weights; built once so they never show up in the profiled steps."""
    key = (C, str(device), dtype, separable)
    ks = _KERNEL_CACHE.get(key)
    if ks is None:
        g = _gaussian_1d().to(device=device, dtype=dtype)
        if separable:
            ks = (g.view(1, 1, 11, 1).expand(C, 1, 11, 1).contiguous(),
                  g.view(1, 1, 1, 11).expand(C, 1, 1, 11).contiguous())
        else:
            ks = ((g[:, None] * g[None, :]).view(1, 1, 11, 11)
                  .expand(C, 1, 11, 11).contiguous(),)
        _KERNEL_CACHE[key] = ks
    return ks


def _ssim_from_stats(mu1, mu2, s11, s22, s12):
    """Assemble the mean SSIM from the five blurred quantities."""
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    ssim_map = ((2 * mu1_mu2 + _C1) * (2 * (s12 - mu1_mu2) + _C2)) / (
        (mu1_sq + mu2_sq + _C1) * ((s11 - mu1_sq) + (s22 - mu2_sq) + _C2)
    )
    return ssim_map.mean()


def _ssim_baseline(img1, img2, padding: str = "valid"):
    """One 11x11 grouped conv2d per blurred quantity — the ROCm fallback practitioners
    deploy, and the numerical reference for the other two."""
    C = img1.shape[-3]
    (k2d,) = _blur_kernels(C, img1.device, img1.dtype, separable=False)
    pad = 0 if padding == "valid" else 11 // 2

    def blur(x):
        return F.conv2d(x, k2d, padding=pad, groups=C)

    return _ssim_from_stats(blur(img1), blur(img2), blur(img1 * img1),
                            blur(img2 * img2), blur(img1 * img2))


def _ssim_separable(img1, img2, padding: str = "valid"):
    """The same window factored into 11x1 then 1x11: ~5.5x fewer MACs per pixel."""
    C = img1.shape[-3]
    k_v, k_h = _blur_kernels(C, img1.device, img1.dtype, separable=True)
    pad = 0 if padding == "valid" else 11 // 2

    def blur(x):
        x = F.conv2d(x, k_v, padding=(pad, 0), groups=C)
        return F.conv2d(x, k_h, padding=(0, pad), groups=C)

    return _ssim_from_stats(blur(img1), blur(img2), blur(img1 * img1),
                            blur(img2 * img2), blur(img1 * img2))


def _select_ssim(name: str):
    """Resolve --ssim to (fn, label). Missing TriSSIM is fatal, never silent: a run that
    quietly dropped the SSIM term would produce a plausible-looking but meaningless
    total, since the loss is exactly what we are trying to measure."""
    if name == "off":
        return None, "off (L1 only)"
    if name == "baseline":
        return _ssim_baseline, "baseline (11x11 grouped conv2d)"
    if name == "separable":
        return _ssim_separable, "separable (two 1D convs)"
    try:
        from fused_ssim import fused_ssim
    except Exception as exc:
        raise SystemExit(
            f"--ssim trissim requested but `import fused_ssim` failed: {exc}\n"
            "TriSSIM is installed by default in the Docker image; outside it run\n"
            "  pip install git+https://github.com/charyang-ai/TriSSIM.git\n"
            "or profile another implementation with --ssim baseline|separable|off."
        ) from exc
    return fused_ssim, "trissim (installed fused_ssim)"


# ----------------------------------------------------------------------------------
# Rasterizer backward (--ras_bwd). `baseline` is gsplat's stock HIP kernel; `triton`
# swaps in the autotuned Triton one from the sibling `triraster/` package. Only the
# backward changes -- the HIP forward rasterizer runs either way.
# ----------------------------------------------------------------------------------
def _select_ras_bwd(name: str) -> str:
    """Install the requested rasterizer backward and return a label for the header.

    A missing `triraster` is fatal rather than a silent fallback: a run that quietly
    profiled the HIP kernel while claiming `--ras_bwd triton` would look exactly like a
    Triton kernel that happened to be no faster."""
    if name == "baseline":
        return "baseline (gsplat HIP rasterize_to_pixels_3dgs_bwd)"

    # A sibling `triraster/src` wins over the installed package, so the plugin stays
    # editable from a bind-mounted checkout with no reinstall and no image rebuild.
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "triraster", "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
    try:
        import triraster
    except Exception as exc:
        raise SystemExit(
            f"--ras_bwd triton requested but `import triraster` failed: {exc}\n"
            "It is installed by default in the Docker image; outside it run\n"
            "  pip install --no-build-isolation ./triraster\n"
            "or profile the stock kernel with --ras_bwd baseline."
        ) from exc
    triraster.install()
    if not triraster.is_installed():
        raise SystemExit("triraster.install() did not take effect on gsplat")
    # The path is part of the label because three copies can coexist -- the working
    # tree, an editable install pointing at it, and the snapshot baked into the image
    # at build time. A timing attributed to the wrong one is worse than no timing.
    return (f"triton (triraster {triraster.__version__}, autotuned, from "
            f"{os.path.dirname(os.path.abspath(triraster.__file__))})")


def _make_scene(n: int, sh_degree: int, device, dtype):
    """Random Gaussians as trainable leaves, placed in front of a single camera."""
    means = (torch.randn(n, 3, device=device, dtype=dtype) * 0.5).requires_grad_(True)
    quats = torch.randn(n, 4, device=device, dtype=dtype).requires_grad_(True)
    scales = (torch.rand(n, 3, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    opacities = torch.randn(n, device=device, dtype=dtype).requires_grad_(True)
    if sh_degree and sh_degree > 0:
        k = (sh_degree + 1) ** 2
        colors = (torch.rand(n, k, 3, device=device, dtype=dtype)).requires_grad_(True)
    else:
        colors = (torch.rand(n, 3, device=device, dtype=dtype)).requires_grad_(True)
    return means, quats, scales, opacities, colors


def _make_camera(width: int, height: int, device, dtype):
    """One camera at the origin looking down +z; scene sits ~5 units in front."""
    focal = 0.5 * width / math.tan(0.5 * math.radians(60.0))
    K = torch.tensor([[focal, 0.0, width / 2.0],
                      [0.0, focal, height / 2.0],
                      [0.0, 0.0, 1.0]], device=device, dtype=dtype)[None]  # [1,3,3]
    viewmat = torch.eye(4, device=device, dtype=dtype)
    viewmat[2, 3] = 5.0  # push the scene to depth ~5 in camera space (in front)
    return viewmat[None], K  # [1,4,4], [1,3,3]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-gaussians", type=int, default=200_000)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--sh-degree", type=int, default=0,
                   help="0 = plain RGB colors; >0 = evaluate SH of that degree")
    p.add_argument("--iters", type=int, default=30, help="profiled training steps")
    p.add_argument("--warmup", type=int, default=10,
                   help="unprofiled warmup steps (compile/allocator/tuning settle)")
    p.add_argument("--ssim", choices=("trissim", "baseline", "separable", "off"),
                   default="trissim",
                   help="SSIM implementation for the photometric loss: "
                        "trissim = the installed fused loss (default); "
                        "baseline = pure-torch 11x11 grouped conv2d; "
                        "separable = pure-torch two 1D passes; "
                        "off = L1 only")
    p.add_argument("--no-ssim", action="store_true",
                   help="deprecated alias for --ssim off")
    p.add_argument("--tile-size", type=int, default=8,
                   help="rasterizer tile size. gsplat's ROCm fork defaults to 8 (tuned "
                        "on CDNA); 16 is upstream's. It drives tile intersection, the "
                        "radix sort and the rasterize backward at once, so it moves "
                        "most of the step")
    p.add_argument("--ras_bwd", choices=("baseline", "triton"), default="baseline",
                   help="rasterizer backward implementation: "
                        "baseline = gsplat's stock HIP kernel (default, so the "
                        "reference numbers never move silently); "
                        "triton = the autotuned Triton kernel from triraster/")
    p.add_argument("--row-limit", type=int, default=30,
                   help="rows in the sorted kernel table")
    p.add_argument("--sort-by", default="self_cuda_time_total",
                   help="profiler table sort key (e.g. self_cuda_time_total, "
                        "cuda_time_total, self_cpu_time_total)")
    p.add_argument("--trace", default=None,
                   help="optional path to also export a chrome trace (json)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_str)
    dtype = torch.float32
    if device.type != "cuda":
        print("WARNING: no CUDA/HIP device — profiling CPU only (not representative).")

    rasterization = _import_rasterization()
    if args.no_ssim:
        print("NOTE: --no-ssim is deprecated; use --ssim off", flush=True)
    ssim_fn, ssim_label = _select_ssim("off" if args.no_ssim else args.ssim)
    ras_bwd_label = _select_ras_bwd(args.ras_bwd)

    print(f"torch {torch.__version__}  hip {torch.version.hip}  device {device}",
          flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"gaussians={args.num_gaussians}  image={args.width}x{args.height}  "
          f"sh_degree={args.sh_degree}  ssim={ssim_label}  "
          f"ras_bwd={ras_bwd_label}  tile_size={args.tile_size}  "
          f"warmup={args.warmup}  iters={args.iters}", flush=True)

    means, quats, scales, opacities, colors = _make_scene(
        args.num_gaussians, args.sh_degree, device, dtype)
    viewmats, Ks = _make_camera(args.width, args.height, device, dtype)
    target = torch.rand(1, 3, args.height, args.width, device=device, dtype=dtype)

    opt = torch.optim.Adam([means, quats, scales, opacities, colors], lr=1e-3)
    sh_degree = args.sh_degree if args.sh_degree and args.sh_degree > 0 else None

    def train_step():
        opt.zero_grad(set_to_none=True)
        renders, _alphas, _meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=torch.sigmoid(opacities),
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=args.width,
            height=args.height,
            sh_degree=sh_degree,
            tile_size=args.tile_size,
        )
        img = renders[0].permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)  # [1,3,H,W]
        loss = (img - target).abs().mean()
        if ssim_fn is not None:
            loss = loss + (1.0 - ssim_fn(img, target, padding="valid"))
        loss.backward()
        opt.step()
        return loss

    # warmup (surfaces first-call compile / allocator / autotune cost)
    for _ in range(args.warmup):
        train_step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # schedule: everything active (we already warmed up above); repeat once.
    sched = schedule(wait=0, warmup=1, active=args.iters, repeat=1)
    with profile(activities=activities, schedule=sched, record_shapes=False,
                 profile_memory=False, with_stack=False) as prof:
        for _ in range(args.iters + 1):  # +1 to cover the schedule's warmup slot
            train_step()
            prof.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    print("=" * 100, flush=True)
    print(f"top {args.row_limit} operators by {args.sort_by}:", flush=True)
    print(prof.key_averages().table(sort_by=args.sort_by, row_limit=args.row_limit),
          flush=True)

    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"chrome trace written to {args.trace}", flush=True)


if __name__ == "__main__":
    main()
