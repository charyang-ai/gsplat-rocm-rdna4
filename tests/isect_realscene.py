"""Where trained Mip-NeRF 360 scenes land on the intersection stage's density curve.

Everything else in this series is measured on synthetic Gaussians, and the argument for
that is in the paper: the stage's cost is governed by pairs per Gaussian, so a sweep over
that statistic (`isect_ablations.py --sweep`) characterises the method better than any one
scene can. But the sweep only tells a reader what happens *at* a density; it does not tell
them which density their scene has. That is what this script measures, on the checkpoints
`train_realscene.py` produces, over every training view rather than a hand-picked one.

Two things come out of it. `--stats` is the distribution: pairs per Gaussian per view, and
the per-Gaussian tile count whose shape the paper's limits section speculates about --
cheap, no timing, so it does not care whether the GPU is otherwise busy. `--bench` is the
speedup at those densities, timed, and it does care: run it on an idle device or the
numbers are contention, not method.

Usage:
  python tests/isect_realscene.py --stats
  python tests/isect_realscene.py --bench --json results/realscene/bench.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from isect_correctness_test import (  # noqa: E402
    _add_local_paths, _time, _tri_isect,
)

_add_local_paths()

import triisect  # noqa: E402
from gsplat.cuda._wrapper import (  # noqa: E402
    fully_fused_projection,
    isect_offset_encode as hip_offset_encode,
    isect_tiles as hip_isect_tiles,
)

_EXAMPLES = os.path.join(os.path.dirname(_HERE), ".gsplat", "examples")


# ----------------------------------------------------------------------------------
# loading a trained scene
# ----------------------------------------------------------------------------------
def _load_checkpoint(path: str, device) -> dict:
    """Gaussians and camera geometry from one `train_realscene.py` checkpoint.

    Only the parameters the projection consumes are moved to the device -- the SH
    coefficients are the bulk of the file and the intersection stage never sees a
    colour. Cameras come from the COLMAP parser rather than the Dataset so that
    nothing decodes 200 JPEGs to learn an image size.
    """
    if _EXAMPLES not in sys.path:
        sys.path.insert(0, _EXAMPLES)
    from datasets.colmap import Parser

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    splats = ckpt["splats"]
    gauss = dict(
        means=splats["means"].to(device),
        quats=splats["quats"].to(device),
        scales=torch.exp(splats["scales"]).to(device),
        opacities=torch.sigmoid(splats["opacities"]).to(device),
    )

    parser = Parser(data_dir=ckpt["data_dir"], factor=ckpt["data_factor"],
                    normalize=True, test_every=8)
    views = []
    for i in range(len(parser.camtoworlds)):
        cam_id = parser.camera_ids[i]
        w, h = parser.imsize_dict[cam_id]
        views.append((
            torch.from_numpy(parser.camtoworlds[i]).float().to(device),
            torch.from_numpy(parser.Ks_dict[cam_id]).float().to(device),
            int(w), int(h),
        ))
    return dict(gauss=gauss, views=views, step=ckpt["step"],
                tile_size=ckpt.get("tile_size"), psnr=ckpt.get("psnr"),
                n_gaussians=int(splats["means"].shape[0]))


def _project(gauss: dict, view, device, packed: bool = False):
    """Projection outputs for one view, in the layout the stage takes.

    Both layouts are here because both are real and they disagree about the one statistic
    this file is about. `rasterization()` defaults to `packed=True`, which compacts the
    frustum survivors into `[nnz, ...]`; gsplat's own `simple_trainer.py` overrides it to
    False and passes the full `[1, N, ...]`. Pairs per Gaussian is the same pair count over
    a denominator that differs by the survival rate -- and the dispatch rule reads that
    denominator, because the count kernel and the prefix sum walk the array they are given,
    dead entries included.
    """
    camtoworld, K, w, h = view
    viewmat = torch.linalg.inv(camtoworld)
    if packed:
        (_batch_ids, camera_ids, gaussian_ids, radii, means2d, depths,
         conics, _comp) = fully_fused_projection(
            gauss["means"], None, gauss["quats"], gauss["scales"],
            viewmat[None], K[None], w, h, packed=True,
            opacities=gauss["opacities"])
        kw = dict(packed=True, n_images=1, image_ids=camera_ids,
                  gaussian_ids=gaussian_ids)
    else:
        radii, means2d, depths, conics, _comp = fully_fused_projection(
            gauss["means"], None, gauss["quats"], gauss["scales"],
            viewmat[None], K[None], w, h, opacities=gauss["opacities"])
        kw = {}
    return dict(means2d=means2d, radii=radii, depths=depths, conics=conics,
                w=w, h=h, kw=kw, n_elements=means2d.numel() // 2)


def _scenes(args) -> list:
    """Checkpoints to measure, newest step per scene unless --step says otherwise."""
    out = []
    for scene in args.scenes:
        pattern = os.path.join(args.result_root, scene, "ckpts", "ckpt_*.pt")
        paths = sorted(glob.glob(pattern),
                       key=lambda p: int(os.path.basename(p)[5:-3]))
        if not paths:
            print(f"  (no checkpoint for {scene}, skipping)")
            continue
        if args.step is not None:
            paths = [p for p in paths
                     if int(os.path.basename(p)[5:-3]) == args.step]
            if not paths:
                print(f"  (no step-{args.step} checkpoint for {scene}, skipping)")
                continue
        out.append((scene, paths[-1]))
    return out


# ----------------------------------------------------------------------------------
# --stats
# ----------------------------------------------------------------------------------
def run_stats(args) -> dict:
    """Pairs per Gaussian across views, and the per-Gaussian tile-count distribution.

    Reported per view rather than averaged over the scene because the dispatch rule acts
    per call: a scene whose mean density sits above the crossover can still spend part of
    its views below it, and that is visible here and nowhere else.
    """
    device = torch.device("cuda")
    out = {}
    for scene, path in _scenes(args):
        sc = _load_checkpoint(path, device)
        print(f"\n=== stats: {scene}  step {sc['step']}  N={sc['n_gaussians']}  "
              f"psnr={sc['psnr']:.2f}\n")
        hdr = (f"{'layout':<8}{'tile':>5}{'views':>7}{'pairs/G p10':>13}{'median':>9}"
               f"{'p90':>9}{'live%':>8}{'tiles/G med':>13}{'p99':>7}{'max':>7}")
        print(hdr)
        print("-" * len(hdr))

        for packed in args.layouts:
            for tile in args.tiles:
                rec = _stats_cell(sc, tile, packed, args, device)
                out[f"{scene}_{'packed' if packed else 'dense'}_tile{tile}"] = rec
                print(f"{'packed' if packed else 'dense':<8}{tile:>5}"
                      f"{rec['n_views']:>7}{rec['ppg_p10']:>13.1f}"
                      f"{rec['ppg_median']:>9.1f}{rec['ppg_p90']:>9.1f}"
                      f"{100 * rec['live_frac']:>7.1f}%{rec['tpg_median']:>13.0f}"
                      f"{rec['tpg_p99']:>7.0f}{rec['tpg_max']:>7.0f}")
                # The tile-count distribution is over the projections the frustum kept,
                # which is the same set either way -- only the denominator of pairs per
                # Gaussian changes -- so print it once per tile rather than per layout.
                if args.histogram and packed == args.layouts[0]:
                    _print_log2(rec["hist_log2"], tile)
        del sc
        torch.cuda.empty_cache()
    return out


def _stats_cell(sc: dict, tile: int, packed: bool, args, device) -> dict:
    """One (layout, tile) cell of the stats table, over every sampled view."""
    ppg, live_frac = [], []
    # Tile counts are small non-negative integers bounded by the tile grid, so the whole
    # population's distribution fits in a bincount and the quantiles come out of its
    # cumulative sum -- exactly, and without holding 200 M values at once, which is what
    # pooling every projection of every view would otherwise need.
    hist = None
    views = sc["views"][::args.view_stride]
    for view in views:
        p = _project(sc["gauss"], view, device, packed=packed)
        tw = math.ceil(p["w"] / tile)
        th = math.ceil(p["h"] / tile)
        tiles_per_gauss, _ids, fids = hip_isect_tiles(
            p["means2d"], p["radii"], p["depths"], tile, tw, th, sort=False, **p["kw"])
        n_pairs = fids.numel()
        # The denominator is the length of the array handed to the stage, which is what
        # the count kernel and the prefix sum have to walk and what the dispatch rule
        # divides by. Under the packed layout that is the frustum survivors only; under
        # the dense one it includes the culled entries, which cost fixed work and emit
        # nothing. Hence the two rows.
        ppg.append(n_pairs / max(p["n_elements"], 1))
        live = (p["radii"][..., 0] > 0) & (p["radii"][..., 1] > 0)
        live_frac.append(live.float().mean().item())
        b = torch.bincount(tiles_per_gauss[live].to(torch.int64),
                           minlength=tw * th + 1)
        if hist is None:
            hist = b
        else:
            if b.numel() > hist.numel():
                hist, b = b, hist
            hist[:b.numel()] += b

    ppg = np.array(ppg)
    rec = dict(
        tile=tile, packed=packed, n_views=len(views),
        n_gaussians=sc["n_gaussians"], step=sc["step"], psnr=sc["psnr"],
        ppg_p10=float(np.percentile(ppg, 10)),
        ppg_median=float(np.median(ppg)),
        ppg_p90=float(np.percentile(ppg, 90)),
        ppg_min=float(ppg.min()), ppg_max=float(ppg.max()),
        live_frac=float(np.mean(live_frac)),
        **_tpg_summary(hist),
    )
    if args.histogram:
        rec["hist_log2"] = _log2_buckets(hist)
    return rec


def _tpg_summary(hist: torch.Tensor) -> dict:
    """Quantiles, mean and max of a tile-count distribution held as a bincount.

    The quantile convention is the smallest count at or below which the given fraction of
    projections falls -- no interpolation, since the variable is a number of tiles and a
    median of 6.5 tiles would not mean anything.
    """
    total = int(hist.sum())
    cdf = torch.cumsum(hist, 0)
    values = torch.arange(hist.numel(), device=hist.device, dtype=torch.float64)

    def q(f: float) -> float:
        target = torch.tensor([int(math.ceil(f * total))], device=hist.device,
                              dtype=cdf.dtype)
        i = int(torch.searchsorted(cdf, target).item())
        return float(min(i, hist.numel() - 1))

    nz = torch.nonzero(hist)
    return dict(tpg_total=total, tpg_median=q(0.5), tpg_p90=q(0.9), tpg_p99=q(0.99),
                tpg_mean=float((values * hist).sum().item() / max(total, 1)),
                tpg_max=float(nz[-1].item()) if nz.numel() else 0.0)


def _log2_buckets(hist: torch.Tensor) -> dict:
    """The same distribution on a log2 grid.

    Log2 because the tail is what matters -- the emission's load imbalance is set by how
    far the largest producers sit from the median, and a linear grid cannot show both.
    """
    edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 1 << 30]
    cum = torch.cumsum(hist, 0)
    n = hist.numel()
    at = lambda i: int(cum[min(i, n - 1)].item())  # noqa: E731
    counts = [at(hi - 1) - at(lo - 1) if lo else at(hi - 1)
              for lo, hi in zip(edges[:-1], edges[1:])]
    return dict(edges=edges, counts=counts, total=int(hist.sum()))


def _print_log2(buckets: dict, tile: int) -> None:
    edges, counts, total = buckets["edges"], buckets["counts"], buckets["total"]
    print(f"\n  tiles/Gaussian histogram (tile {tile}, {total} live projections)")
    for i, (lo, hi, c) in enumerate(zip(edges[:-1], edges[1:], counts)):
        if c == 0:
            continue
        label = f"{lo}+" if i == len(counts) - 1 else (
            f"{lo}" if hi == lo + 1 else f"{lo}-{hi - 1}")
        print(f"    {label:>10}: {100 * c / max(total, 1):5.2f}%  {c}")


# ----------------------------------------------------------------------------------
# --bench
# ----------------------------------------------------------------------------------
def run_bench(args) -> dict:
    """Stage time on real scenes, on the view whose density is the scene's median.

    The median view rather than the mean of all of them: timing every view of every scene
    would take longer than training did, and the median is the one whose density the
    `--stats` table already reports, so the two tables can be read together.
    """
    device = torch.device("cuda")
    out = {}
    print(f"\n=== bench: stage time on trained scenes (median-density view)\n")
    hdr = (f"{'scene':<10}{'layout':<8}{'tile':>5}{'elems':>10}{'pairs/G':>9}"
           f"{'n_pairs':>11}{'HIP':>9}{'ours':>9}{'spd':>7}{'disp':>9}{'spd':>7}"
           f"{'exact':>9}{'spd':>7}")
    print(hdr)
    print("-" * len(hdr))

    for scene, path in _scenes(args):
        sc = _load_checkpoint(path, device)
        for packed in args.layouts:
            for tile in args.tiles:
                view = _median_density_view(sc, tile, packed, args, device)
                p = _project(sc["gauss"], view, device, packed=packed)
                rec = _bench_cell(scene, sc, p, tile, packed)
                out[f"{scene}_{'packed' if packed else 'dense'}_tile{tile}"] = rec
                print(f"{scene:<10}{'packed' if packed else 'dense':<8}{tile:>5}"
                      f"{rec['n_elements']:>10}{rec['pairs_per_gauss']:>9.1f}"
                      f"{rec['n_pairs']:>11}"
                      f"{rec['hip']:>9.3f}{rec['ours']:>9.3f}"
                      f"{rec['hip'] / rec['ours']:>6.2f}x"
                      f"{rec['dispatched']:>9.3f}"
                      f"{rec['hip'] / rec['dispatched']:>6.2f}x"
                      f"{rec['exact']:>9.3f}{rec['hip'] / rec['exact']:>6.2f}x")
        del sc
        torch.cuda.empty_cache()
    return out


def _median_density_view(sc: dict, tile: int, packed: bool, args, device):
    """The view whose pair count is the scene's median.

    The median rather than the mean of all views: timing every view of every scene would
    take longer than the training did, and the median is the view whose density `--stats`
    already reports, so the two tables describe the same operating point.
    """
    views = sc["views"][::args.view_stride]
    dens = []
    for view in views:
        p = _project(sc["gauss"], view, device, packed=packed)
        tw, th = math.ceil(p["w"] / tile), math.ceil(p["h"] / tile)
        dens.append(hip_isect_tiles(p["means2d"], p["radii"], p["depths"],
                                    tile, tw, th, sort=False, **p["kw"])[2].numel())
    return views[int(np.argsort(dens)[len(dens) // 2])]


def _bench_cell(scene: str, sc: dict, p: dict, tile: int, packed: bool) -> dict:
    """HIP, ours, the dispatched path and the exact test, on one set of tensors."""
    m, r, d, c = p["means2d"], p["radii"], p["depths"], p["conics"]
    kw = p["kw"]
    tw, th = math.ceil(p["w"] / tile), math.ceil(p["h"] / tile)
    n_pairs = hip_isect_tiles(m, r, d, tile, tw, th, sort=False, **kw)[2].numel()

    def hip_full():
        ids = hip_isect_tiles(m, r, d, tile, tw, th, **kw)[1]
        hip_offset_encode(ids, 1, tw, th)

    def our_full():
        ids = _tri_isect(m, r, d, tile, tw, th, presort=True, **kw)[1]
        triisect.isect_offset_encode(ids, 1, tw, th)

    def dispatched():
        ids = triisect.isect_tiles(m, r, d, tile, tw, th, presort=True, **kw)[1]
        triisect.isect_offset_encode(ids, 1, tw, th)

    def exact_full():
        ids = _tri_isect(m, r, d, tile, tw, th, presort=True, conics=c, **kw)[1]
        triisect.isect_offset_encode(ids, 1, tw, th)

    t_hip = _time(hip_full)
    t_ours = _time(our_full)
    triisect.reset_dispatch_cache()
    t_disp = _time(dispatched)
    t_exact = _time(exact_full)
    n_exact = _tri_isect(m, r, d, tile, tw, th, conics=c, **kw)[2].numel()

    return dict(scene=scene, tile=tile, packed=packed, step=sc["step"],
                n_gaussians=sc["n_gaussians"], n_elements=p["n_elements"],
                n_pairs=n_pairs, pairs_per_gauss=n_pairs / max(p["n_elements"], 1),
                n_pairs_exact=n_exact, hip=t_hip, ours=t_ours,
                dispatched=t_disp, exact=t_exact)


# ----------------------------------------------------------------------------------
# --res
# ----------------------------------------------------------------------------------
def _upscale(view, s: float):
    """The same camera rendering the same scene at s times the linear resolution.

    Focal lengths and the principal point scale with the sensor, everything else is
    unchanged, so this is the camera the dataset would have provided at a smaller
    downsample factor -- not a crop and not a different viewpoint.
    """
    camtoworld, K, w, h = view
    K = K.clone()
    K[:2, :] *= s
    return (camtoworld, K, int(round(w * s)), int(round(h * s)))


def run_res(args) -> dict:
    """Density and speedup as the output resolution grows, at fixed scene.

    This is the experiment that connects the two halves of the paper. Trained scenes at
    the dataset's own resolution sit far below the crossover, so the sweep's speedups do
    not apply to them; but pairs per Gaussian is quadratic in linear resolution while the
    Gaussian population is fixed, so the same scene crosses over somewhere. Finding where
    is the difference between "our method helps real scenes" and "our method helps real
    scenes rendered at 4K and above", and only one of those is true.
    """
    device = torch.device("cuda")
    out = {}
    for scene, path in _scenes(args):
        sc = _load_checkpoint(path, device)
        print(f"\n=== res: {scene}  step {sc['step']}  N={sc['n_gaussians']}\n")
        hdr = (f"{'layout':<8}{'scale':>6}{'resolution':>12}{'MP':>7}{'tile':>5}"
               f"{'pairs/G':>9}{'n_pairs':>12}{'HIP':>9}{'ours':>9}{'spd':>7}"
               f"{'disp':>9}{'spd':>7}")
        print(hdr)
        print("-" * len(hdr))

        for packed in args.layouts:
            # One view for the whole sweep, chosen by median density at the dataset's own
            # resolution, so the rows differ in resolution and nothing else.
            base_view = _median_density_view(sc, 8, packed, args, device)
            for s in args.res_scales:
                p = _project(sc["gauss"], _upscale(base_view, s), device, packed=packed)
                for tile in args.tiles:
                    rec = _bench_cell(scene, sc, p, tile, packed)
                    rec.update(scale=s, width=p["w"], height=p["h"])
                    key = f"{scene}_{'packed' if packed else 'dense'}_s{s}_tile{tile}"
                    out[key] = rec
                    shape = f"{p['w']}x{p['h']}"
                    print(f"{'packed' if packed else 'dense':<8}{s:>6.1f}{shape:>12}"
                          f"{p['w'] * p['h'] / 1e6:>7.1f}{tile:>5}"
                          f"{rec['pairs_per_gauss']:>9.1f}{rec['n_pairs']:>12}"
                          f"{rec['hip']:>9.3f}{rec['ours']:>9.3f}"
                          f"{rec['hip'] / rec['ours']:>6.2f}x"
                          f"{rec['dispatched']:>9.3f}"
                          f"{rec['hip'] / rec['dispatched']:>6.2f}x")
                del p
                torch.cuda.empty_cache()
        del sc
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result-root", default="results/realscene")
    ap.add_argument("--scenes", nargs="+",
                    default=["bicycle", "counter", "garden", "room", "stump"])
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step to use; default is the latest present")
    ap.add_argument("--tiles", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--view-stride", type=int, default=1,
                    help="use every k-th view; 1 means all of them")
    ap.add_argument("--histogram", action="store_true",
                    help="also pool the per-Gaussian tile counts into a log2 histogram")
    ap.add_argument("--res-scales", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0],
                    help="linear resolution multipliers for --res, relative to the "
                         "dataset's own downsample factor")
    ap.add_argument("--layout", choices=["dense", "packed", "both"], default="both",
                    help="dense is what gsplat's simple_trainer.py passes; packed is "
                         "what rasterization() defaults to. They differ by the frustum "
                         "survival rate, which is the density's denominator")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--res", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    args.layouts = {"dense": [False], "packed": [True],
                    "both": [False, True]}[args.layout]

    if not (args.stats or args.bench or args.res):
        args.stats = args.bench = args.res = True

    results = {}
    if args.stats:
        results["stats"] = run_stats(args)
    if args.bench:
        results["bench"] = run_bench(args)
    if args.res:
        results["res"] = run_res(args)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
