"""Gradient-correctness gate for the Triton rasterizer backward (`triraster`).

The Triton kernel in `triraster` is a transliteration of gsplat's HIP
`rasterize_to_pixels_3dgs_bwd_kernel`, so the only legitimate difference between the two
is fp32 accumulation order: both scatter per-Gaussian gradients with atomics, and they
group the contributing pixels differently (the HIP kernel reduces per warp and issues
one atomic per warp, the Triton kernel reduces the whole tile and issues one). Anything
larger than that is a bug.

This drives the rasterizer directly rather than through `rasterization()`, so a
discrepancy cannot be diluted (or manufactured) by the projection/SH backward that would
otherwise sit between the kernel and the leaf gradients.

`--bench` and `--tune-report` deliberately do NOT use the synthetic field above. They
replay the exact tensors `profile_trainer.py`'s scene hands to the pixel rasterizer,
captured straight out of a `rasterization()` call. Timing a hand-rolled 2D scene instead
measures a different workload than the one being optimised: the uniform field and the
projected 3D cloud disagree by more than 2x on which backward is faster, so only the
captured one predicts what the full profile will say.

Run inside the container:
    HIP_VISIBLE_DEVICES=1 python tests/rasbwd_correctness_test.py
    HIP_VISIBLE_DEVICES=1 python tests/rasbwd_correctness_test.py --bench
    HIP_VISIBLE_DEVICES=1 python tests/rasbwd_correctness_test.py --tune-report
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, os.pardir, "triraster", "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _HERE not in sys.path:  # so `profile_trainer` resolves for the captured scene
    sys.path.insert(0, _HERE)

# Relative tolerance on the gradient tensors. Each entry is an fp32 sum over up to a few
# thousand atomically-accumulated terms whose order differs between the two kernels, so
# the floor here is reordering noise, not algorithmic slack.
_RTOL = 2e-3


def _make_case(n: int, width: int, height: int, cdim: int, device, seed: int):
    """A random field of 2D Gaussians in image space, as the projection stage would
    hand it to the rasterizer."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape):
        return torch.rand(*shape, generator=g)

    means2d = rnd(1, n, 2) * torch.tensor([float(width), float(height)])
    # covariance from random axis lengths + rotation, then invert to the conic
    sx = 1.0 + rnd(1, n) * 7.0
    sy = 1.0 + rnd(1, n) * 7.0
    th = rnd(1, n) * math.pi
    ct, st = torch.cos(th), torch.sin(th)
    a = ct * ct * sx * sx + st * st * sy * sy
    b = ct * st * (sx * sx - sy * sy)
    c = st * st * sx * sx + ct * ct * sy * sy
    det = (a * c - b * b).clamp_min(1e-6)
    conics = torch.stack([c / det, -b / det, a / det], dim=-1)  # [1, n, 3]

    radii = (3.0 * torch.maximum(sx, sy)).ceil().clamp_min(1.0)
    radii = torch.stack([radii, radii], dim=-1).to(torch.int32)  # [1, n, 2]
    depths = rnd(1, n) * 10.0 + 0.1
    colors = rnd(1, n, cdim)
    opacities = 0.05 + rnd(1, n) * 0.9

    to = lambda t: t.to(device).contiguous()  # noqa: E731
    return (to(means2d), to(conics), to(colors), to(opacities), to(radii), to(depths))


def _capture_trainer_inputs(n: int, width: int, height: int, sh_degree: int,
                            tile_size: int, device, seed: int) -> dict:
    """Replay `profile_trainer.py`'s scene and capture what it feeds the rasterizer.

    gsplat's `rasterize_to_pixels()` resolves `_RasterizeToPixels` as a module global, so
    standing a recorder in that slot for one call yields the projected means2d/conics/
    colors/opacities and the tile intersection arrays *exactly* as the profiled training
    step produces them -- no reimplementation of the projection to drift out of sync."""
    from gsplat.cuda import _wrapper
    from profile_trainer import _import_rasterization, _make_camera, _make_scene

    rasterization = _import_rasterization()
    torch.manual_seed(seed)
    means, quats, scales, opacities, colors = _make_scene(
        n, sh_degree, device, torch.float32)
    viewmats, Ks = _make_camera(width, height, device, torch.float32)

    cap: dict = {}
    original = _wrapper._RasterizeToPixels

    class _Recorder:  # duck-types the autograd Function: only `.apply` is looked up
        @staticmethod
        def apply(means2d, conics, colors_, opacities_, backgrounds, masks,
                  w, h, ts, isect_offsets, flatten_ids, absgrad):
            cap.update(means2d=means2d.detach().clone(),
                       conics=conics.detach().clone(),
                       colors=colors_.detach().clone(),
                       opacities=opacities_.detach().clone(),
                       isect_offsets=isect_offsets, flatten_ids=flatten_ids,
                       width=w, height=h, tile_size=ts)
            return original.apply(means2d, conics, colors_, opacities_, backgrounds,
                                  masks, w, h, ts, isect_offsets, flatten_ids, absgrad)

    _wrapper._RasterizeToPixels = _Recorder
    try:
        with torch.no_grad():
            rasterization(
                means=means, quats=quats, scales=scales,
                opacities=torch.sigmoid(opacities), colors=colors,
                viewmats=viewmats, Ks=Ks, width=width, height=height,
                sh_degree=sh_degree if sh_degree > 0 else None,
                tile_size=tile_size,
            )
    finally:
        _wrapper._RasterizeToPixels = original

    if "means2d" not in cap:
        raise SystemExit("rasterization() never reached the pixel rasterizer")
    return cap


def _backward_args(inputs, isect_offsets, flatten_ids, width, height, tile_size,
                   seed: int = 0):
    """Run the HIP forward once and build the full argument set the backward op takes."""
    from gsplat.cuda._wrapper import _make_lazy_cuda_func

    means2d, conics, colors, opacities = inputs
    render_colors, render_alphas, last_ids = _make_lazy_cuda_func(
        "rasterize_to_pixels_3dgs_fwd"
    )(means2d, conics, colors, opacities, None, None,
      width, height, tile_size, isect_offsets, flatten_ids)
    # A random cotangent rather than ones: a uniform one lets sign errors and misweighted
    # pixels cancel inside the per-Gaussian reduction, which is exactly what a config
    # check needs to see. Fixed by seed, and it does not affect timing.
    g = torch.Generator(device="cpu").manual_seed(seed + 104729)
    v_colors = torch.rand(render_colors.shape, generator=g).to(render_colors.device)
    v_alphas = torch.rand(render_alphas.shape, generator=g).to(render_alphas.device)
    return (
        means2d, conics, colors, opacities, None, None,
        width, height, tile_size, isect_offsets, flatten_ids,
        render_alphas, last_ids, v_colors, v_alphas, False,
    )


def _intersections(means2d, radii, depths, width, height, tile_size):
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, packed=False
    )
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
    return isect_offsets, flatten_ids


def _run(fn_cls, inputs, isect_offsets, flatten_ids, width, height, tile_size,
         backgrounds, absgrad, seed):
    """One forward+backward through `fn_cls`, returning outputs and leaf gradients."""
    means2d, conics, colors, opacities = (t.clone().requires_grad_(True)
                                          for t in inputs)
    bg = backgrounds.clone().requires_grad_(True) if backgrounds is not None else None

    render_colors, render_alphas = fn_cls.apply(
        means2d, conics, colors, opacities, bg, None,
        width, height, tile_size, isect_offsets, flatten_ids, absgrad,
    )

    # A fixed random cotangent, so both runs backprop the identical upstream gradient.
    g = torch.Generator(device="cpu").manual_seed(seed + 7919)
    gc = torch.rand(render_colors.shape, generator=g).to(render_colors.device)
    ga = torch.rand(render_alphas.shape, generator=g).to(render_alphas.device)
    ((render_colors * gc).sum() + (render_alphas * ga).sum()).backward()

    out = {
        "render_colors": render_colors.detach(),
        "render_alphas": render_alphas.detach(),
        "v_means2d": means2d.grad,
        "v_conics": conics.grad,
        "v_colors": colors.grad,
        "v_opacities": opacities.grad,
    }
    if absgrad:
        out["v_means2d_abs"] = means2d.absgrad
    if bg is not None:
        out["v_backgrounds"] = bg.grad
    return out


def _compare(ref: dict, got: dict) -> tuple[bool, list[str]]:
    lines, ok = [], True
    for k in ref:
        a, b = ref[k], got.get(k)
        if b is None:
            lines.append(f"  {k:<16} MISSING in triton output")
            ok = False
            continue
        scale = a.abs().max().item()
        err = (a - b).abs().max().item()
        rel = err / scale if scale > 0 else err
        good = rel <= _RTOL
        ok &= good
        lines.append(f"  {k:<16} max|ref|={scale:12.5e}  max|diff|={err:12.5e}  "
                     f"rel={rel:9.2e}  {'ok' if good else 'FAIL'}")
    return ok, lines


def _bench(fn_cls, inputs, isect_offsets, flatten_ids, width, height, tile_size,
           iters: int = 20) -> float:
    """Median ms for one backward pass, forward excluded."""
    means2d, conics, colors, opacities = (t.clone().requires_grad_(True)
                                          for t in inputs)
    render_colors, render_alphas = fn_cls.apply(
        means2d, conics, colors, opacities, None, None,
        width, height, tile_size, isect_offsets, flatten_ids, False,
    )
    gc = torch.ones_like(render_colors)
    ga = torch.ones_like(render_alphas)

    times = []
    for i in range(iters + 5):
        for t in (means2d, conics, colors, opacities):
            t.grad = None
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.autograd.backward(
            [render_colors, render_alphas], [gc, ga], retain_graph=True
        )
        end.record()
        torch.cuda.synchronize()
        if i >= 5:  # drop autotune + warmup
            times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-gaussians", type=int, default=20_000)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--tile-size", type=int, default=8,
                   help="gsplat's ROCm fork defaults to 8; 16 is upstream's default")
    p.add_argument("--bench", action="store_true",
                   help="also time both backwards on profile_trainer.py's scene")
    p.add_argument("--tune-report", action="store_true",
                   help="time every autotune candidate on that scene and print the "
                        "spread, which is what says whether tuning has headroom left")
    p.add_argument("--grad-bias", action="store_true",
                   help="compare the per-Gaussian screen-space gradient NORM against "
                        "HIP, which is the quantity densification thresholds on. A "
                        "max|diff|/max|ref| check cannot see a small systematic bias "
                        "here, and a bias is what would shift the final Gaussian count")
    p.add_argument("--verify-configs", action="store_true",
                   help="check EVERY autotune candidate against the HIP backward, not "
                        "just the one autotune happens to select here. Autotuning is "
                        "per shape-specialisation, so a training run at another "
                        "resolution can land on a config this test never exercised")
    p.add_argument("--sh-degree", type=int, default=3,
                   help="SH degree for the captured scene (matches the profiled run)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA/HIP device visible")
    device = torch.device("cuda")

    import triraster
    from gsplat.cuda._wrapper import _RasterizeToPixels

    if not triraster._HAS_TRITON:
        raise SystemExit("triraster imported but Triton is unavailable")

    print(f"torch {torch.__version__}  hip {torch.version.hip}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    # Which triraster this is testing. There can be three: the working tree (via the
    # sys.path insert above), an editable install pointing at it, and the snapshot the
    # image baked in at build time. Passing against the wrong one proves nothing.
    print(f"triraster {triraster.__version__} from "
          f"{os.path.dirname(os.path.abspath(triraster.__file__))}", flush=True)
    print(f"gaussians={args.num_gaussians}  image={args.width}x{args.height}  "
          f"tile_size={args.tile_size}  rtol={_RTOL:g}", flush=True)

    all_ok = True
    # cdim 3 is the training path; 1 and 5 exercise the channel masking, and the
    # background / absgrad variants cover the two optional code paths in the kernel.
    cases = [
        dict(cdim=3, absgrad=False, bg=False),
        dict(cdim=3, absgrad=True, bg=False),
        dict(cdim=3, absgrad=False, bg=True),
        dict(cdim=1, absgrad=False, bg=False),
        dict(cdim=5, absgrad=False, bg=False),
    ]
    for ci, case in enumerate(cases):
        cdim, absgrad, use_bg = case["cdim"], case["absgrad"], case["bg"]
        seed = args.seed + ci
        means2d, conics, colors, opacities, radii, depths = _make_case(
            args.num_gaussians, args.width, args.height, cdim, device, seed)
        isect_offsets, flatten_ids = _intersections(
            means2d, radii, depths, args.width, args.height, args.tile_size)

        backgrounds = None
        if use_bg:
            backgrounds = torch.rand(1, cdim, device=device)

        if not triraster.supports(colors, args.tile_size, means2d):
            raise SystemExit(
                f"case cdim={cdim} tile={args.tile_size} is outside the Triton fast "
                "path, so this test would silently compare HIP against HIP"
            )

        inputs = (means2d, conics, colors, opacities)
        kw = dict(isect_offsets=isect_offsets, flatten_ids=flatten_ids,
                  width=args.width, height=args.height, tile_size=args.tile_size,
                  backgrounds=backgrounds, absgrad=absgrad, seed=seed)
        ref = _run(_RasterizeToPixels, inputs, **kw)
        got = _run(triraster._TriRasterizeToPixels, inputs, **kw)

        ok, lines = _compare(ref, got)
        all_ok &= ok
        n_isects = flatten_ids.numel()
        print(f"\ncase {ci}: cdim={cdim} absgrad={absgrad} backgrounds={use_bg}  "
              f"n_isects={n_isects}  ->  {'PASS' if ok else 'FAIL'}", flush=True)
        print("\n".join(lines), flush=True)

    if args.bench or args.tune_report or args.verify_configs or args.grad_bias:
        cap = _capture_trainer_inputs(args.num_gaussians, args.width, args.height,
                                      args.sh_degree, args.tile_size, device, args.seed)
        inputs = (cap["means2d"], cap["conics"], cap["colors"], cap["opacities"])
        kw = dict(isect_offsets=cap["isect_offsets"], flatten_ids=cap["flatten_ids"],
                  width=cap["width"], height=cap["height"], tile_size=cap["tile_size"])
        print(f"\ncaptured profile_trainer scene: {args.num_gaussians} Gaussians  "
              f"{cap['width']}x{cap['height']}  sh_degree={args.sh_degree}  "
              f"tile_size={cap['tile_size']}  cdim={cap['colors'].shape[-1]}  "
              f"n_isects={cap['flatten_ids'].numel()}", flush=True)

    if args.bench:
        hip = _bench(_RasterizeToPixels, inputs, **kw)
        tri = _bench(triraster._TriRasterizeToPixels, inputs, **kw)
        print("\nbackward, median of 20")
        print(f"  hip     {hip:8.3f} ms")
        print(f"  triton  {tri:8.3f} ms   ({hip / tri:.2f}x)")

    if args.grad_bias:
        from gsplat.cuda._wrapper import _make_lazy_cuda_func

        bwd_args = _backward_args(inputs, seed=args.seed, **kw)
        ref_v = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(*bwd_args)[1]

        def _norm(v):
            # gsplat's DefaultStrategy scales the screen-space gradient into pixel units
            # before accumulating and thresholding it, so the bias has to be measured on
            # that same quantity rather than on the raw gradient.
            scale = torch.tensor([kw["width"] / 2.0, kw["height"] / 2.0],
                                 device=v.device, dtype=v.dtype)
            return (v * scale).norm(dim=-1).flatten()

        ref_n = _norm(ref_v)
        live = ref_n > 0
        n_live = int(live.sum())
        print(f"\nper-Gaussian |v_means2d| vs HIP  ({n_live} of {ref_n.numel()} touched)."
              "\n  bias is the mean SIGNED relative difference: a positive one means "
              "Triton\n  reports systematically larger gradients, which would densify "
              "harder.")
        for cfg in triraster.configs():
            got_n = _norm(triraster.run_config(cfg, *bwd_args)[1])
            rel = ((got_n[live] - ref_n[live]) / ref_n[live]).double()
            a = rel.abs()
            split = cfg.kwargs.get("SPLIT", 1)
            print(f"  bias={rel.mean().item():+10.3e}  "
                  f"median|d|={a.median().item():9.3e}  "
                  f"p99|d|={torch.quantile(a, 0.99).item():9.3e}  "
                  f"max|d|={a.max().item():9.3e}   "
                  f"num_warps={cfg.num_warps} SPLIT={split} "
                  f"waves_per_eu={cfg.kwargs.get('waves_per_eu', '-')}", flush=True)

    if args.verify_configs:
        from gsplat.cuda._wrapper import _make_lazy_cuda_func

        bwd_args = _backward_args(inputs, seed=args.seed, **kw)
        ref = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(*bwd_args)
        names = ("v_means2d_abs", "v_means2d", "v_conics", "v_colors", "v_opacities")

        cfgs = triraster.configs()
        print(f"\nverifying all {len(cfgs)} autotune candidates against HIP:")
        for cfg in cfgs:
            split = cfg.kwargs.get("SPLIT", 1)
            label = (f"BLOCK_G={cfg.kwargs.get('BLOCK_G'):<3} "
                     f"num_warps={cfg.num_warps} SPLIT={split} "
                     f"waves_per_eu={cfg.kwargs.get('waves_per_eu', '-')}")
            got = triraster.run_config(cfg, *bwd_args)
            worst, worst_name = 0.0, ""
            for name, a, b in zip(names, ref, got):
                if a is None or b is None:
                    continue
                scale = a.abs().max().item()
                rel = (a - b).abs().max().item() / (scale if scale > 0 else 1.0)
                if rel > worst:
                    worst, worst_name = rel, name
            good = worst <= _RTOL
            all_ok &= good
            print(f"  {'ok  ' if good else 'FAIL'}  worst rel={worst:9.2e} "
                  f"({worst_name:<13}) {label}", flush=True)

    if args.tune_report:
        rows = triraster.bench_configs(*_backward_args(inputs, seed=args.seed, **kw))
        best = next((ms for _, ms in rows if ms is not None), None)
        print(f"\nautotune candidates ({len(rows)}), fastest first:")
        for cfg, ms in rows:
            tag = f"{ms:8.3f} ms  {best / ms:5.2f}x" if ms is not None else "   failed"
            split = cfg.kwargs.get("SPLIT", 1)
            print(f"  {tag}   BLOCK_G={cfg.kwargs.get('BLOCK_G'):<3} "
                  f"num_warps={cfg.num_warps} "
                  f"SPLIT={split} (px/prog={args.tile_size ** 2 // split:<3}) "
                  f"waves_per_eu={cfg.kwargs.get('waves_per_eu', '-')}")
        timed = [ms for _, ms in rows if ms is not None]
        if timed:
            print(f"\n  spread best->worst: {max(timed) / min(timed):.2f}x  "
                  f"({min(timed):.3f} - {max(timed):.3f} ms)")

    print(f"\n{'ALL CASES PASS' if all_ok else 'FAILURES PRESENT'}", flush=True)
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
