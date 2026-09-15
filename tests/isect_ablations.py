"""Ablations and supporting measurements for the tile-intersection work.

`isect_correctness_test.py` proves the substitution is exact and times it; this file
answers the four questions a reader is entitled to ask about *why* it is built the way it
is, none of which are visible in a before/after table.

  --memory   What the stage costs in device memory. The narrowed key halves the per-pair
             key and the exact test removes pairs outright, so the peak footprint of the
             pair buffers should fall by more than either change alone. Memory is often
             the binding constraint in 3DGS training, so this is a result and not a
             footnote.

  --eps      Whether the `eps` margin in the recovered ellipse scale is doing real work.
             `_ellipse_scale` recovers `R` from the *box*, and safety requires
             `R >= 2 ln(255*opacity)` for every Gaussian. That is guaranteed in exact
             arithmetic by the `ceil` in gsplat's radius, but gsplat computes `extend`
             with a fast logarithm and we recompute `R` in fp32 from the conic, so the
             guarantee is only as good as the slack. This measures the slack directly
             over the real projection's output, which is the only honest way to defend a
             constant like 5e-4.

  --sweep    Where the emission's win comes from and where it goes away. The speedup is
             a function of pairs per Gaussian -- that is the whole content of the
             diagnosis -- so sweeping that statistic directly, by scaling the Gaussians'
             screen extent, says more than any fixed scene does. It also locates the
             crossover below which the pipeline's fixed cost is not amortized, which is
             what a dispatch rule needs to know.

  --tight    How much of the exact test's residual gap to an oracle is the price of
             recovering `R` from the box rather than from the opacity. The opacity-derived
             `R` is the tightest sound value; the ratio bounds what the API-clean choice
             gives up.

Usage:
  python tests/isect_ablations.py --memory --eps --sweep --tight
  python tests/isect_ablations.py --sweep --json sweep.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from isect_correctness_test import (  # noqa: E402
    _add_local_paths, _project_scene, _time, _tri_isect,
)

_add_local_paths()

import triisect  # noqa: E402
from gsplat.cuda._wrapper import (  # noqa: E402
    isect_offset_encode as hip_offset_encode,
    isect_tiles as hip_isect_tiles,
)


# ----------------------------------------------------------------------------------
# scene with a tunable screen extent, so pairs/Gaussian can be swept directly
# ----------------------------------------------------------------------------------
def _scene(n: int, width: int, height: int, device, seed: int, scale_mult: float = 1.0):
    """`_project_scene` with the Gaussians' world-space scale multiplied.

    Screen extent scales with world extent, and pairs per Gaussian scales with the
    square of screen extent, so this is the knob that moves the one statistic the
    diagnosis says the emission's cost depends on.
    """
    from gsplat.cuda._wrapper import fully_fused_projection

    g = torch.Generator(device="cpu").manual_seed(seed)
    means = (torch.randn(n, 3, generator=g) * 0.5).to(device)
    quats = torch.randn(n, 4, generator=g).to(device)
    scales = (torch.rand(n, 3, generator=g) * 0.05 * scale_mult).to(device)
    opacities = torch.sigmoid(torch.randn(n, generator=g)).to(device)

    focal = 0.5 * width / math.tan(0.5 * math.radians(60.0))
    K = torch.tensor([[focal, 0.0, width / 2.0],
                      [0.0, focal, height / 2.0],
                      [0.0, 0.0, 1.0]], device=device)[None]
    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = 5.0

    radii, means2d, depths, conics, _comp = fully_fused_projection(
        means, None, quats, scales, viewmat[None], K, width, height,
        opacities=opacities)
    return dict(means2d=means2d, radii=radii, depths=depths, conics=conics,
                opacities=opacities)


# ----------------------------------------------------------------------------------
# --memory
# ----------------------------------------------------------------------------------
def run_memory(args) -> dict:
    """Peak device memory attributable to the stage, per variant.

    Measured as the allocator's peak during the call minus what was already resident, so
    the inputs -- which every variant shares -- are excluded and what is left is the pair
    buffers plus whatever the sort needs for temporaries.
    """
    device = torch.device("cuda")
    print(f"\n=== memory: peak bytes attributable to the stage, "
          f"{args.num_gaussians} Gaussians\n")
    hdr = (f"{'shape':<12}{'tile':>5}{'variant':>10}{'pairs':>12}"
           f"{'B/pair':>8}{'peak MB':>10}{'vs HIP':>9}")
    print(hdr)
    print("-" * len(hdr))

    out = {}
    for (w, h) in args.shapes:
        for tile in args.tiles:
            scene = _scene(args.num_gaussians, w, h, device, args.seed)
            m, r, d, c = (scene["means2d"], scene["radii"], scene["depths"],
                          scene["conics"])
            tw, th = math.ceil(w / tile), math.ceil(h / tile)

            def measure(fn):
                # Triton's autotuner benchmarks candidate configs with `do_bench`, which
                # allocates a 256 MB buffer to flush the cache between reps. That is a
                # one-time tuning artifact, not a steady-state cost, and it dwarfs the
                # pair buffers in the smaller cells -- so tune first, measure second.
                ids, fids = fn()
                del ids, fids
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                ids, fids = fn()
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - before
                n_pairs = fids.numel()
                del ids, fids
                return peak, n_pairs

            variants = {
                "HIP": lambda: hip_isect_tiles(m, r, d, tile, tw, th)[1:3],
                "B": lambda: _tri_isect(m, r, d, tile, tw, th)[1:3],
                "B+D": lambda: _tri_isect(m, r, d, tile, tw, th,
                                          presort=True)[1:3],
                "B+D+C": lambda: _tri_isect(m, r, d, tile, tw, th,
                                            presort=True, conics=c)[1:3],
            }
            base = None
            for name, fn in variants.items():
                peak, n_pairs = measure(fn)
                if base is None:
                    base = peak
                key = f"{w}x{h}_tile{tile}_{name}"
                out[key] = dict(peak_bytes=peak, pairs=n_pairs)
                print(f"{w}x{h:<6}{tile:>5}{name:>10}{n_pairs:>12}"
                      f"{peak / max(n_pairs, 1):>8.1f}{peak / 2**20:>10.1f}"
                      f"{base / peak:>8.2f}x")
    return out


# ----------------------------------------------------------------------------------
# --eps
# ----------------------------------------------------------------------------------
def run_eps(args) -> dict:
    """Is `R` recovered from the box actually >= the rasterizer's own threshold?

    Safety needs `R >= e_true^2 = 2 ln(255*opacity)`, the squared Mahalanobis radius at
    which the rasterizer stops accepting. `R` is recovered from the integer box and the
    conic; `e_true^2` from the opacity in float64. The ratio's *minimum* over the
    population is the number that matters -- one Gaussian below 1.0 is a dropped pair
    that could have been shaded.
    """
    device = torch.device("cuda")
    print(f"\n=== eps: slack in the recovered ellipse scale, "
          f"{args.eps_gaussians} Gaussians per shape\n")
    hdr = (f"{'shape':<12}{'live':>10}{'arith':>7}{'min R/e2':>11}{'p0.01%':>10}"
           f"{'median':>9}{'   eps needed':>14}{'  safe@5e-4':>12}")
    print(hdr)
    print("-" * len(hdr))

    out = {}
    for (w, h) in args.shapes:
      scene = _scene(args.eps_gaussians, w, h, device, args.seed)
      # The kernel evaluates `_ellipse_scale` in fp32; fp64 isolates how much of any
      # shortfall is gsplat's fast logarithm rather than our own rounding.
      for arith, dt in (("fp32", torch.float32), ("fp64", torch.float64)):
        radii = scene["radii"][0].to(dt)
        conics = scene["conics"][0].to(dt)
        opac = scene["opacities"].to(dt)

        live = (radii[:, 0] > 0) & (radii[:, 1] > 0)
        rx, ry = radii[live, 0], radii[live, 1]
        A, B, C = conics[live, 0], conics[live, 1], conics[live, 2]

        det = A * C - B * B
        # what `_ellipse_scale` computes, without the eps factor
        R_box = torch.minimum(rx * rx * det / C, ry * ry * det / A)
        # what the rasterizer actually uses: sigma = Q/2 <= ln(255*opacity), in fp64
        # regardless, since it is the reference and not the thing under test
        e_true2 = torch.clamp(2.0 * torch.log(255.0 * opac[live].double()), min=0.0)
        keep = e_true2 > 0
        ratio = (R_box[keep].double() / e_true2[keep])

        rmin = ratio.min().item()
        q = torch.quantile(ratio.float(), 1e-4).item()
        med = ratio.median().item()
        # the eps that would be required if the raw recovery were short
        eps_needed = max(0.0, 1.0 / rmin - 1.0) if rmin > 0 else float("inf")
        safe = rmin * (1.0 + 5e-4) >= 1.0

        key = f"{w}x{h}_{arith}"
        out[key] = dict(live=int(keep.sum()), min_ratio=rmin, p1e4=q, median=med,
                        eps_needed=eps_needed, safe_at_5e4=bool(safe))
        print(f"{w}x{h:<6}{int(keep.sum()):>10}{arith:>7}{rmin:>11.6f}{q:>10.4f}"
              f"{med:>9.3f}{eps_needed:>14.2e}{('yes' if safe else 'NO'):>12}")
    return out


# ----------------------------------------------------------------------------------
# --sweep
# ----------------------------------------------------------------------------------
def run_sweep(args) -> dict:
    """Emission speedup as a function of pairs per Gaussian.

    The diagnosis says the win is removed divergence and scattered writes, both of which
    scale with pairs per Gaussian, so this is the curve that either supports it or does
    not. It also fixes the crossover point a dispatch rule would test against.
    """
    device = torch.device("cuda")
    print(f"\n=== sweep: emission vs pairs/Gaussian, {args.num_gaussians} Gaussians, "
          f"{args.shapes[0][0]}x{args.shapes[0][1]}\n")
    hdr = (f"{'scale':>7}{'tile':>5}{'pairs/G':>10}{'n_pairs':>12}"
           f"{'HIP':>9}{'search':>9}{'spd':>8}{'   full HIP':>12}{'forced':>11}"
           f"{'spd':>8}{'dispatch':>11}{'spd':>8}")
    print(hdr)
    print("-" * len(hdr))

    w, h = args.shapes[0]
    out = {}
    for mult in args.sweep_scales:
        scene = _scene(args.num_gaussians, w, h, device, args.seed, scale_mult=mult)
        m, r, d = scene["means2d"], scene["radii"], scene["depths"]
        for tile in args.tiles:
            tw, th = math.ceil(w / tile), math.ceil(h / tile)
            n_pairs = hip_isect_tiles(m, r, d, tile, tw, th, sort=False)[1].numel()
            if n_pairs == 0:
                continue

            e_hip = _time(lambda: hip_isect_tiles(m, r, d, tile, tw, th, sort=False))
            e_ours = _time(lambda: _tri_isect(m, r, d, tile, tw, th,
                                              sort=False, search=True))

            def hip_full():
                ids = hip_isect_tiles(m, r, d, tile, tw, th)[1]
                hip_offset_encode(ids, 1, tw, th)

            def our_full():
                ids = _tri_isect(m, r, d, tile, tw, th, presort=True)[1]
                triisect.isect_offset_encode(ids, 1, tw, th)

            def dispatched():
                """What a caller actually gets: the density rule picks the path."""
                ids = triisect.isect_tiles(m, r, d, tile, tw, th, presort=True)[1]
                triisect.isect_offset_encode(ids, 1, tw, th)

            f_hip = _time(hip_full)
            f_ours = _time(our_full)
            # the rule remembers the density per shape, and this sweep changes the scene
            # under a fixed shape, so each cell has to start from no memory of the last
            triisect.reset_dispatch_cache()
            f_disp = _time(dispatched)

            ppg = n_pairs / args.num_gaussians
            key = f"scale{mult}_tile{tile}"
            out[key] = dict(pairs_per_gauss=ppg, n_pairs=n_pairs, emit_hip=e_hip,
                            emit_ours=e_ours, full_hip=f_hip, full_ours=f_ours,
                            full_dispatched=f_disp)
            print(f"{mult:>7.2f}{tile:>5}{ppg:>10.2f}{n_pairs:>12}"
                  f"{e_hip:>9.3f}{e_ours:>9.3f}{e_hip / e_ours:>7.2f}x"
                  f"{f_hip:>12.3f}{f_ours:>11.3f}{f_hip / f_ours:>7.2f}x"
                  f"{f_disp:>11.3f}{f_hip / f_disp:>7.2f}x")
    return out


# ----------------------------------------------------------------------------------
# --tight
# ----------------------------------------------------------------------------------
def run_tight(args) -> dict:
    """How much larger the box-recovered ellipse is than the opacity-derived one.

    `R_box/R_opacity` is the factor by which the tested region is inflated by keeping the
    opacity out of the API. The area of `{Q <= R}` is `pi*R/sqrt(det)`, so for Gaussians
    large enough that the tile grid resolves them, the excess pair count is bounded by
    the same factor -- which is how much of the gap to an oracle this design choice, as
    opposed to the pixel-centre rectangle test, accounts for.
    """
    device = torch.device("cuda")
    print(f"\n=== tight: inflation of the box-recovered ellipse scale\n")
    hdr = (f"{'shape':<12}{'live':>10}{'median':>9}{'mean':>9}{'p90':>9}{'max':>9}"
           f"{'  area x':>10}")
    print(hdr)
    print("-" * len(hdr))

    out = {}
    for (w, h) in args.shapes:
        scene = _scene(args.eps_gaussians, w, h, device, args.seed)
        radii = scene["radii"][0].double()
        conics = scene["conics"][0].double()
        opac = scene["opacities"].double()
        live = (radii[:, 0] > 0) & (radii[:, 1] > 0)
        rx, ry = radii[live, 0], radii[live, 1]
        A, B, C = conics[live, 0], conics[live, 1], conics[live, 2]
        det = A * C - B * B
        R_box = torch.minimum(rx * rx * det / C, ry * ry * det / A)
        e_true2 = torch.clamp(2.0 * torch.log(255.0 * opac[live]), min=0.0)
        keep = e_true2 > 0
        ratio = R_box[keep] / e_true2[keep]
        key = f"{w}x{h}"
        out[key] = dict(median=ratio.median().item(), mean=ratio.mean().item(),
                        p90=torch.quantile(ratio.float(), 0.90).item(),
                        max=ratio.max().item())
        print(f"{w}x{h:<6}{int(keep.sum()):>10}{ratio.median():>9.3f}"
              f"{ratio.mean():>9.3f}{torch.quantile(ratio.float(), 0.90):>9.3f}"
              f"{ratio.max():>9.2f}{ratio.median():>10.2f}")
    return out


def _time_batched(fn, batch: int = 50, warmup: int = 5, rep: int = 10) -> float:
    """Median per-call time with `batch` calls inside one event window.

    Several phases of this pipeline are a few tens of microseconds, which is at or below
    what one event pair per call can resolve: measured that way, the count kernel and the
    prefix sum come out dominated by launch and event overhead rather than by their own
    work, and are not comparable to the sort in the same table. Batching amortizes that.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(batch):
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / batch)
    times.sort()
    return times[len(times) // 2]


def _time_destructive(fn, restore, rep: int = 15) -> float:
    """Median time of `fn`, with `restore` run between reps but outside the event window.

    The sort consumes its inputs as scratch, so every rep has to have them put back --
    hundreds of megabytes of copying at these sizes, comparable to a whole radix pass.
    Leaving it inside the window, or measuring it separately and subtracting, both swamp
    what is being measured.
    """
    times = []
    for i in range(rep + 5):
        restore()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        if i >= 5:
            times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def run_phases(args) -> dict:
    """Every phase of the pipeline, timed separately, on one set of tensors.

    This regenerates the per-phase table rather than leaving it transcribed. Two rules
    make the rows comparable to each other, and both matter: outputs are pre-allocated, so
    a row measures its kernel rather than the caching allocator, and the sub-100us rows are
    batched (`_time_batched`) so they are not dominated by event overhead. The rows that
    come in pairs are the substitutions -- ownership map against in-kernel search, wide
    sort against narrow, boundary scan against `searchsorted`.
    """
    device = torch.device("cuda")
    import triton

    from triisect._core import (
        _bit_width, _counts_kernel, _depth_presort, _emit_kernel, _owner_map, _sort_pairs,
    )

    print(f"\n=== phases: {args.num_gaussians} Gaussians, ms per call\n")
    out = {}
    for (w, h) in args.shapes:
        for tile in args.tiles:
            tw, th = math.ceil(w / tile), math.ceil(h / tile)
            scene = _scene(args.num_gaussians, w, h, device, args.seed)
            m, r, d = scene["means2d"], scene["radii"], scene["depths"]
            n = m.numel() // 2
            N = m.shape[-2]
            tile_bits = _bit_width(tw * th)

            counts = torch.empty(n, dtype=torch.int32, device=device)
            cum = torch.empty(n, dtype=torch.int64, device=device)

            def count():
                _counts_kernel[(triton.cdiv(n, 1024),)](
                    m, r, counts, n, TILE=tile, tile_width=tw, tile_height=th,
                    BLOCK=1024, num_warps=4)

            count()
            n_pairs = int(counts.sum().item())
            torch.cumsum(counts, 0, out=cum)
            base = cum - counts
            perm = _depth_presort(d, n)

            keys64 = torch.empty(n_pairs, dtype=torch.int64, device=device)
            keys32 = torch.empty(n_pairs, dtype=torch.int32, device=device)
            fids = torch.empty(n_pairs, dtype=torch.int32, device=device)

            def emit(key64: bool):
                keys = keys64 if key64 else keys32
                owner = base if key64 else perm
                grid = lambda meta: (triton.cdiv(n_pairs, meta["BLOCK"]),)  # noqa: E731
                _emit_kernel[grid](
                    m, r, d, base, owner, None, keys, fids, n_pairs, N, n,
                    TILE=tile, tile_width=tw, tile_height=th,
                    TILE_N_BITS=tile_bits, KEY64=key64, SEARCH=True,
                    ITERS=_bit_width(n), PERM=not key64, PACKED=False)

            rows = {}
            rows["count kernel"] = _time_batched(count)
            rows["prefix sum"] = _time_batched(
                lambda: torch.cumsum(counts, 0, out=cum))
            # the host read of the pair count. Timed on a queue that has just been
            # drained, so this is the call's own cost and not a pipeline stall.
            rows["device synchronization"] = _time_batched(
                lambda: int(counts[-1].item()), batch=20)
            rows["emit kernel"] = _time_batched(lambda: emit(True), batch=10)
            rows["depth pre-sort over N"] = _time_batched(
                lambda: _depth_presort(d, n), batch=10)
            rows["ownership map (M2)"] = _time_batched(
                lambda: _owner_map(counts, n_pairs), batch=5)

            # the sorts. `_emit_kernel` above already filled both key arrays; the sort
            # destroys them, so each rep restores from a held-back copy.
            emit(True)
            emit(False)
            src64, src32, srcf = keys64.clone(), keys32.clone(), fids.clone()

            def restore(k, s):
                def go():
                    k.copy_(s)
                    fids.copy_(srcf)
                return go

            rows["sort, 8-byte key"] = _time_destructive(
                lambda: _sort_pairs(keys64, fids, 0, 32 + tile_bits),
                restore(keys64, src64))
            rows["sort, 4-byte key"] = _time_destructive(
                lambda: _sort_pairs(keys32, fids, 0, tile_bits),
                restore(keys32, src32))

            ids64 = _sort_pairs(src64.clone(), srcf.clone(), 0, 32 + tile_bits)[0]
            ids32 = _sort_pairs(src32.clone(), srcf.clone(), 0, tile_bits)[0]
            rows["offsets, boundary scan"] = _time_batched(
                lambda: hip_offset_encode(ids64, 1, tw, th), batch=20)
            rows["offsets, searchsorted"] = _time_batched(
                lambda: triisect.isect_offset_encode(ids32, 1, tw, th), batch=20)

            print(f"-- {w}x{h}, tile {tile}: {n_pairs} pairs, "
                  f"{n_pairs/n:.1f}/Gaussian, {tile_bits} tile bits")
            for label, ms in rows.items():
                print(f"   {label:<28}{ms:9.3f}")
            print(f"   {'-> sort narrowing':<28}"
                  f"{rows['sort, 8-byte key']/rows['sort, 4-byte key']:8.2f}x")
            print(f"   {'-> offsets':<28}"
                  f"{rows['offsets, boundary scan']/rows['offsets, searchsorted']:8.2f}x")
            out[f"{w}x{h}_tile{tile}"] = dict(rows, pairs=n_pairs,
                                              tile_bits=tile_bits)
            del keys64, keys32, fids, src64, src32, srcf, ids64, ids32
            torch.cuda.empty_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-gaussians", type=int, default=500_000)
    p.add_argument("--eps-gaussians", type=int, default=2_000_000,
                   help="population for the eps/tightness statistics")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--memory", action="store_true")
    p.add_argument("--eps", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--tight", action="store_true")
    p.add_argument("--phases", action="store_true")
    p.add_argument("--shape", action="append", default=None,
                   help="WxH, repeatable")
    p.add_argument("--tile", action="append", type=int, default=None)
    p.add_argument("--sweep-scales", type=float, nargs="*", default=None)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    args.shapes = ([tuple(int(v) for v in s.split("x")) for s in args.shape]
                   if args.shape else [(1920, 1080), (618, 411)])
    args.tiles = args.tile or [8, 16]
    if args.sweep_scales is None:
        args.sweep_scales = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    if not (args.memory or args.eps or args.sweep or args.tight or args.phases):
        args.memory = args.eps = args.sweep = args.tight = args.phases = True

    torch.manual_seed(args.seed)
    results = {}
    if args.memory:
        results["memory"] = run_memory(args)
    if args.eps:
        results["eps"] = run_eps(args)
    if args.tight:
        results["tight"] = run_tight(args)
    if args.sweep:
        results["sweep"] = run_sweep(args)
    if args.phases:
        results["phases"] = run_phases(args)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(config={k: v for k, v in vars(args).items()},
                           results=results), f, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
