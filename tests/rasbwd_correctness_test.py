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


def _ppl_sweep(args, device) -> None:
    """Time every candidate at several tile sizes and order the results by pixels
    per lane.

    The tile size is not part of a `triton.Config`; it changes the tile
    intersection, so each one needs its own captured scene. That is also why this
    cannot run on synthetic Gaussians: how many primitives land in a tile is a
    property of the scene, and it is exactly what the backward's cost depends on.

    A configuration is summarised by

        pi = tile^2 / (SPLIT * W * num_warps)

    and the model says pi orders the configurations on its own -- that a point's
    tile size should not matter once pi is fixed. The sweep prints the points
    sorted by pi with their tile size attached, so a violation is visible as two
    tile sizes disagreeing at the same pi.
    """
    import triraster

    tiles = [int(t) for t in args.ppl_sweep.split(",") if t.strip()]
    try:
        warp = torch.cuda.get_device_properties(device).warp_size
    except AttributeError:
        warp = 32

    print(f"\n=== pixels-per-lane sweep: tiles {tiles}, W={warp} ===", flush=True)
    rows = []
    for tile in tiles:
        try:
            cap = _capture_trainer_inputs(args.num_gaussians, args.width, args.height,
                                          args.sh_degree, tile, device, args.seed)
        except Exception as exc:  # a tile size the forward rasterizer rejects
            print(f"\ntile {tile}: capture failed, skipping -- {type(exc).__name__}: "
                  f"{exc}", flush=True)
            continue

        inputs = (cap["means2d"], cap["conics"], cap["colors"], cap["opacities"])
        bwd_args = _backward_args(
            inputs, isect_offsets=cap["isect_offsets"],
            flatten_ids=cap["flatten_ids"], width=cap["width"],
            height=cap["height"], tile_size=cap["tile_size"], seed=args.seed)
        print(f"\ntile {tile}: n_isects={cap['flatten_ids'].numel()}", flush=True)

        for cfg, ms in triraster.bench_configs(*bwd_args):
            split = cfg.kwargs.get("SPLIT", 1)
            ppl = tile * tile / (split * warp * cfg.num_warps)
            rows.append({
                "tile": tile, "split": split, "num_warps": cfg.num_warps,
                "waves_per_eu": cfg.kwargs.get("waves_per_eu", ""),
                "block_g": cfg.kwargs.get("BLOCK_G", 1), "ppl": ppl, "ms": ms,
            })
            print(f"  ppl={ppl:7.2f}  "
                  f"{'   failed' if ms is None else f'{ms:8.3f} ms'}   "
                  f"SPLIT={split} num_warps={cfg.num_warps} "
                  f"waves_per_eu={cfg.kwargs.get('waves_per_eu', '-')}", flush=True)

    timed = [r for r in rows if r["ms"] is not None]
    if not timed:
        print("\nno configuration completed; nothing to say about the model")
        return

    print(f"\n{'ppl':>7} {'ms':>9} {'tile':>5} {'SPLIT':>6} {'warps':>6}   "
          "(sorted by pixels per lane)")
    for r in sorted(timed, key=lambda r: r["ppl"]):
        print(f"{r['ppl']:>7.2f} {r['ms']:>9.3f} {r['tile']:>5} "
              f"{r['split']:>6} {r['num_warps']:>6}")

    # The model's testable content: at a shared pi, tile size should not matter.
    #
    # Conditioned on num_warps, deliberately. num_warps moves pi like SPLIT does,
    # but it also moves the reduction from wave32 cross-lane operations into LDS
    # with barriers, so two configurations at equal pi and unequal num_warps are
    # not doing the same work and are not expected to agree. Pooling over it
    # would fail the test for a reason the design already accounts for. Within a
    # num_warps, SPLIT and tile size are the only things varying, and there the
    # model makes a real prediction.
    by_key: dict[tuple[float, int], list[dict]] = {}
    for r in timed:
        by_key.setdefault((round(r["ppl"], 6), r["num_warps"]), []).append(r)
    shared = {k: v for k, v in by_key.items() if len({r["tile"] for r in v}) > 1}
    if shared:
        print("\ncollapse check -- same pi and same num_warps, different tile size:")
        worst, worst_at = 0.0, None
        for key in sorted(shared):
            ppl, nw = key
            group = shared[key]
            lo = min(r["ms"] for r in group)
            hi = max(r["ms"] for r in group)
            if hi / lo > worst:
                worst, worst_at = hi / lo, key
            detail = ", ".join(f"tile {r['tile']} (SPLIT {r['split']}): {r['ms']:.3f} ms"
                               for r in sorted(group, key=lambda r: r["tile"]))
            print(f"  ppl={ppl:6.2f} warps={nw}  spread {hi / lo:4.2f}x   {detail}")
        print(f"\n  worst disagreement at matched pi: {worst:.2f}x "
              f"(ppl={worst_at[0]:g}, num_warps={worst_at[1]})")
        print("  A collapse worth calling a model wants this near 1. If it is not,"
              "\n  weaken §3.5 to a heuristic and say so -- an overclaimed model is a"
              "\n  worse outcome than an honest heuristic.")
    else:
        print("\nno (pi, num_warps) pair is shared between tile sizes, so this sweep"
              "\ncannot test the collapse. Tile sizes two apart do share values:"
              "\ntile 8 SPLIT 1 and tile 16 SPLIT 4 are both 2 px/lane at num_warps 1.")

    if args.ppl_csv:
        import csv
        with open(args.ppl_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.ppl_csv}")


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
    p.add_argument("--ppl-sweep", metavar="TILES", default=None,
                   help="comma-separated tile sizes, e.g. 8,16,32. Times every "
                        "autotune candidate at each one and reports the results "
                        "against pixels per lane, tile^2/(SPLIT*W*num_warps). The "
                        "claim under test is that this single quantity orders the "
                        "configurations: if points from different tile sizes do not "
                        "fall on one curve, the model is a heuristic and the paper "
                        "has to say so")
    p.add_argument("--ppl-csv", default=None,
                   help="write the --ppl-sweep measurements here for plotting")
    p.add_argument("--grow-grad2d", type=float, default=2e-4,
                   help="densification threshold used by --grad-bias to count how "
                        "many Gaussians would change side. gsplat's DefaultStrategy "
                        "default is 2e-4")
    p.add_argument("--bias-csv", default=None,
                   help="write the per-Gaussian --grad-bias measurements here: one row "
                        "per live Gaussian, its reference gradient norm and the signed "
                        "relative difference under every candidate. Enough to plot the "
                        "error distribution against the margin each Gaussian has to the "
                        "densification threshold, which is what explains the flip count")
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

    if args.ppl_sweep:
        _ppl_sweep(args, device)

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
        thr = args.grow_grad2d
        print(f"\nper-Gaussian |v_means2d| vs HIP  ({n_live} of {ref_n.numel()} touched)."
              "\n  bias is the mean SIGNED relative difference: a positive one means "
              "Triton\n  reports systematically larger gradients, which would densify "
              "harder.\n  se is its standard error, so a bias smaller than se is "
              "indistinguishable\n  from zero. flips counts Gaussians that land on "
              f"opposite sides of the\n  densification threshold ({thr:g}) under the two "
              "backwards -- the decision\n  the training loop actually takes, and the "
              "only channel by which a gradient\n  difference could change the result.")
        worst_p99 = 0.0
        per_cfg = {}  # config tag -> signed relative difference, only for --bias-csv
        for cfg in triraster.configs():
            got_n = _norm(triraster.run_config(cfg, *bwd_args)[1])
            rel = ((got_n[live] - ref_n[live]) / ref_n[live]).double()
            a = rel.abs()
            se = (rel.std() / math.sqrt(rel.numel())).item()
            worst_p99 = max(worst_p99, torch.quantile(a, 0.99).item())
            ref_hi, got_hi = ref_n > thr, got_n > thr
            up = int((got_hi & ~ref_hi).sum())     # Triton would split, HIP would not
            down = int((~got_hi & ref_hi).sum())   # and the reverse
            split = cfg.kwargs.get("SPLIT", 1)
            if args.bias_csv:
                wpe = cfg.kwargs.get("waves_per_eu")
                tag = f"s{split}_nw{cfg.num_warps}" + (f"_wpe{wpe}" if wpe else "")
                per_cfg[tag] = rel.cpu()
            print(f"  bias={rel.mean().item():+10.3e} se={se:9.3e}  "
                  f"median|d|={a.median().item():9.3e}  "
                  f"p99|d|={torch.quantile(a, 0.99).item():9.3e}  "
                  f"max|d|={a.max().item():9.3e}  "
                  f"flips={up + down:>4} (+{up}/-{down})   "
                  f"num_warps={cfg.num_warps} SPLIT={split} "
                  f"waves_per_eu={cfg.kwargs.get('waves_per_eu', '-')}", flush=True)
        n_above = int((ref_n > thr).sum())
        # A Gaussian can only flip if it sits within the perturbation of the threshold,
        # so report that population: without it, flips=0 could just mean nothing was
        # ever close enough to flip, which is a much weaker statement than it looks.
        at_risk = int((live & ((ref_n - thr).abs() <= worst_p99 * thr)).sum())
        print(f"\n  {n_above} of {n_live} live Gaussians are above the threshold under "
              f"HIP.\n  {at_risk} lie within the worst p99 relative error ({worst_p99:.2e}) "
              "of it, and\n  so were at risk of flipping at all. A flip count that is small "
              "AND balanced\n  between + and - is the cleanest statement available: the "
              "substitution moves\n  no Gaussian systematically. Read it against the "
              "at-risk count, not alone.")

        if args.bias_csv:
            tags = list(per_cfg)
            norms = ref_n[live].double().cpu()
            with open(args.bias_csv, "w") as f:
                # The threshold is in the header because the margin each Gaussian has to
                # it -- not the error alone -- is what decides whether a flip is possible.
                f.write(f"# threshold={thr:.6e} n_live={n_live} n_above={n_above}\n")
                f.write("ref_norm," + ",".join(tags) + "\n")
                # .tolist() once per column: indexing the tensors per element instead
                # costs a second or two at this row count.
                cols = [norms.tolist()] + [per_cfg[t].tolist() for t in tags]
                for row in zip(*cols):
                    f.write(f"{row[0]:.9e}," +
                            ",".join(f"{v:.6e}" for v in row[1:]) + "\n")
            print(f"\nwrote {len(norms)} rows x {len(tags)} configs to {args.bias_csv}")

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
