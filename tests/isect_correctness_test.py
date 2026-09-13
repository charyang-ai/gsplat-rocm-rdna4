"""Gate for `triisect`: the Triton tile intersection against gsplat's HIP `intersect_tile`.

The claim this file checks is stronger than the rasterizer-backward one in
`rasbwd_correctness_test.py`. That kernel accumulates through atomics, so the best it can
promise is fp32 accumulation-order noise. Tile intersection produces *integers* -- tile
ids, Gaussian ids, offsets -- so the correct gate is `torch.equal`, and any difference at
all is a bug rather than a tolerance to be argued about.

Several levels, because each is blind to something another one catches:

  1. `--pairs` (default): `tiles_per_gauss`, `isect_ids`, `flatten_ids` and the encoded
     `isect_offsets`, compared exactly against the HIP op over a matrix of shapes.
  2. `--render`: the rendered image, alphas and `last_ids` through the full
     `rasterization()` call. This is what matters if an implementation ever stops
     emitting a bit-identical *list* (see `--exact`, where dropping non-contributing
     pairs is the point) -- the list changes, the image must not.
  3. `--bench`: A/B timing of the stage, split into emit and sort, since those move for
     different reasons and a single total hides which one paid.
  4. `--trainer-bench`: the same timing on tensors captured from a real training step,
     which is packed and has been through a sigmoid, unlike anything built here.
  5. `--exact`: the ellipse test against a brute-force oracle, where the list is
     deliberately *smaller* and the gate is that it stays a superset of what contributes.
  6. `--dispatch`: that the low-density fallback fires on the right side of its threshold
     and reproduces the HIP op exactly when it does.
  7. `--multiview`: batches of more than one camera. Levels 1-6 all render a single view,
     which leaves the key's image field identically zero and every way of packing it
     indistinguishable.

Usage:
  python tests/isect_correctness_test.py
  python tests/isect_correctness_test.py --render
  python tests/isect_correctness_test.py --bench
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch


def _add_local_paths() -> None:
    """A sibling `triisect/src` wins over any installed copy, so the package stays
    editable from a checkout with no reinstall."""
    here = os.path.dirname(os.path.abspath(__file__))
    for pkg in ("triisect", "triraster"):
        src = os.path.join(here, os.pardir, pkg, "src")
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)


_add_local_paths()

import triisect  # noqa: E402
from gsplat.cuda._wrapper import isect_offset_encode  # noqa: E402
from gsplat.cuda._wrapper import isect_tiles as hip_isect_tiles  # noqa: E402


# ----------------------------------------------------------------------------------
# scenes
# ----------------------------------------------------------------------------------
def _project_scene(n: int, width: int, height: int, device, seed: int,
                   sh_degree: int = 0):
    """Run gsplat's real projection so `means2d`/`radii`/`depths` have the exact
    distribution the intersection stage sees in training -- including the zero-radius
    Gaussians the near-plane and radius clips produce, which are a third of the edge
    cases in this file."""
    from gsplat.cuda._wrapper import fully_fused_projection

    g = torch.Generator(device="cpu").manual_seed(seed)
    means = (torch.randn(n, 3, generator=g) * 0.5).to(device)
    quats = torch.randn(n, 4, generator=g).to(device)
    scales = (torch.rand(n, 3, generator=g) * 0.05).to(device)
    opacities = torch.sigmoid(torch.randn(n, generator=g)).to(device)

    focal = 0.5 * width / math.tan(0.5 * math.radians(60.0))
    K = torch.tensor([[focal, 0.0, width / 2.0],
                      [0.0, focal, height / 2.0],
                      [0.0, 0.0, 1.0]], device=device)[None]
    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = 5.0
    viewmats = viewmat[None]

    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means, None, quats, scales, viewmats, K, width, height,
        opacities=opacities,
    )
    return dict(means2d=means2d, radii=radii, depths=depths, conics=conics,
                opacities=opacities, viewmats=viewmats, Ks=K,
                means=means, quats=quats, scales=scales)


def _packed_scene(n: int, width: int, height: int, device, seed: int):
    """The layout the trainer actually runs.

    `rasterization()` defaults to `packed=True`, which is easy to miss: the projection
    compacts the visible Gaussians into `[nnz, ...]` and passes an explicit `image_ids`,
    so a Triton path that only handles the dense `[I, N, ...]` layout falls back to HIP on
    every real training step while still passing every dense test."""
    from gsplat.cuda._wrapper import fully_fused_projection

    g = torch.Generator(device="cpu").manual_seed(seed)
    means = (torch.randn(n, 3, generator=g) * 0.5).to(device)
    quats = torch.randn(n, 4, generator=g).to(device)
    scales = (torch.rand(n, 3, generator=g) * 0.05).to(device)
    opacities = torch.sigmoid(torch.randn(n, generator=g)).to(device)

    focal = 0.5 * width / math.tan(0.5 * math.radians(60.0))
    K = torch.tensor([[focal, 0.0, width / 2.0],
                      [0.0, focal, height / 2.0],
                      [0.0, 0.0, 1.0]], device=device)[None]
    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = 5.0

    (_batch_ids, camera_ids, gaussian_ids, radii, means2d, depths,
     _conics, _comp) = fully_fused_projection(
        means, None, quats, scales, viewmat[None], K, width, height,
        packed=True, opacities=opacities)
    return dict(means2d=means2d, radii=radii, depths=depths, image_ids=camera_ids,
                gaussian_ids=gaussian_ids, n_images=1)


def _multiview_scene(n: int, width: int, height: int, device, seed: int,
                     n_images: int = 3, packed: bool = False):
    """Several cameras in one call, which is the layout every other scene here omits.

    `rasterization()` takes a batch of viewmats and passes the batch size straight through
    to the intersection stage, where the image id becomes the key's high field and the
    offsets become a `[I, th, tw]` array. With one camera that field is identically zero,
    so a single-camera test matrix cannot see anything about how it is packed -- which is
    how an offsets encoder that only agrees with the HIP op when the tile count is an
    exact power of two survived a full suite.

    The cameras are pushed apart along the axes rather than duplicated, so each one keeps
    a different subset of the Gaussians and the per-image pair counts differ.
    """
    from gsplat.cuda._wrapper import fully_fused_projection

    g = torch.Generator(device="cpu").manual_seed(seed)
    means = (torch.randn(n, 3, generator=g) * 0.5).to(device)
    quats = torch.randn(n, 4, generator=g).to(device)
    scales = (torch.rand(n, 3, generator=g) * 0.05).to(device)
    opacities = torch.sigmoid(torch.randn(n, generator=g)).to(device)

    focal = 0.5 * width / math.tan(0.5 * math.radians(60.0))
    K = torch.tensor([[focal, 0.0, width / 2.0],
                      [0.0, focal, height / 2.0],
                      [0.0, 0.0, 1.0]], device=device)
    Ks = K[None].repeat(n_images, 1, 1)

    viewmats = []
    for i in range(n_images):
        v = torch.eye(4, device=device)
        v[2, 3] = 5.0 + 0.5 * i
        v[0, 3] = 0.3 * i
        v[1, 3] = -0.2 * i
        viewmats.append(v)
    viewmats = torch.stack(viewmats)

    if packed:
        (_batch_ids, camera_ids, gaussian_ids, radii, means2d, depths,
         _conics, _comp) = fully_fused_projection(
            means, None, quats, scales, viewmats, Ks, width, height,
            packed=True, opacities=opacities)
        return dict(means2d=means2d, radii=radii, depths=depths,
                    image_ids=camera_ids, gaussian_ids=gaussian_ids,
                    n_images=n_images)

    radii, means2d, depths, conics, _comp = fully_fused_projection(
        means, None, quats, scales, viewmats, Ks, width, height,
        opacities=opacities)
    return dict(means2d=means2d, radii=radii, depths=depths, conics=conics,
                opacities=opacities, viewmats=viewmats, Ks=Ks, means=means,
                quats=quats, scales=scales, n_images=n_images)


def _synthetic_scene(kind: str, width: int, height: int, device):
    """Hand-built adversarial cases the random scene will not produce."""
    tw, th = width, height
    if kind == "empty":
        # every Gaussian culled by the projection (radius <= 0)
        means2d = torch.zeros(64, 2, device=device)
        radii = torch.zeros(64, 2, dtype=torch.int32, device=device)
        depths = torch.ones(64, device=device)
    elif kind == "fullscreen":
        # one Gaussian covering the whole image: the bbox is every tile
        means2d = torch.tensor([[width / 2, height / 2]], device=device)
        radii = torch.tensor([[width, height]], dtype=torch.int32, device=device)
        depths = torch.tensor([3.0], device=device)
    elif kind == "offscreen":
        # bboxes that clamp to empty on each side
        means2d = torch.tensor([[-500.0, height / 2], [width + 500.0, height / 2],
                                [width / 2, -500.0], [width / 2, height + 500.0]],
                               device=device)
        radii = torch.full((4, 2), 8, dtype=torch.int32, device=device)
        depths = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    elif kind == "depth_ties":
        # identical depths *and* identical tiles: exercises the sort's tie-break, which
        # is the one place a stable and an unstable sort disagree
        means2d = torch.full((256, 2), 100.0, device=device)
        radii = torch.full((256, 2), 4, dtype=torch.int32, device=device)
        depths = torch.full((256,), 2.5, device=device)
    elif kind == "single_pixel":
        # minimal bbox (1x1 tile) for every Gaussian
        gen = torch.Generator(device="cpu").manual_seed(7)
        means2d = (torch.rand(4096, 2, generator=gen)
                   * torch.tensor([float(width), float(height)])).to(device)
        radii = torch.ones(4096, 2, dtype=torch.int32, device=device)
        depths = (torch.rand(4096, generator=gen) * 10 + 0.1).to(device)
    else:
        raise ValueError(kind)
    return dict(means2d=means2d[None], radii=radii[None], depths=depths[None])


# ----------------------------------------------------------------------------------
# level 1: the emitted pairs
# ----------------------------------------------------------------------------------
def _compare_pairs(scene, tile_size: int, width: int, height: int, label: str,
                   presort: bool = False, verbose: bool = True) -> bool:
    """Exact comparison against the HIP op.

    `presort` deliberately changes the key encoding (int32 `image|tile` instead of int64
    `image|tile|depth`), so `isect_ids` is expected to differ and is skipped. What must
    still match bit-for-bit is everything the rasterizer actually reads: `flatten_ids`,
    in the order the offsets index into, and the offsets themselves."""
    means2d, radii, depths = scene["means2d"], scene["radii"], scene["depths"]
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    kw = dict(packed=True, n_images=scene["n_images"], image_ids=scene["image_ids"],
              gaussian_ids=scene["gaussian_ids"]) if "image_ids" in scene else {}
    # Dense inputs carry the image count in the leading dimension; packed ones state it.
    n_images = scene.get("n_images",
                         means2d.shape[0] if means2d.dim() >= 3 else 1)

    ref = hip_isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height,
                          **kw)
    got = _tri_isect(means2d, radii, depths, tile_size, tile_width, tile_height,
                     presort=presort, **kw)

    names = ("tiles_per_gauss", "isect_ids", "flatten_ids")
    ok = True
    details = []
    for name, a, b in zip(names, ref, got):
        if presort and name == "isect_ids":
            continue
        same = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
        ok &= same
        if not same:
            n_diff = (a != b).sum().item() if a.shape == b.shape else -1
            details.append(f"{name}: shape {tuple(a.shape)}/{tuple(b.shape)} "
                           f"dtype {a.dtype}/{b.dtype} n_diff {n_diff}")

    # offsets are what the rasterizer actually consumes
    off_ref = isect_offset_encode(ref[1], n_images, tile_width, tile_height)
    off_got = triisect.isect_offset_encode(got[1], n_images, tile_width, tile_height)
    same = off_ref.shape == off_got.shape and torch.equal(off_ref, off_got)
    ok &= same
    if not same:
        details.append(f"isect_offsets: n_diff {(off_ref != off_got).sum().item()}")

    if verbose:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {label:<38} tile={tile_size:<3} n_isects={ref[1].numel()}")
        for d in details:
            print(f"         {d}")
    return ok


def run_pairs(args, presort: bool = False) -> bool:
    device = torch.device("cuda")
    all_ok = True
    mode = "presort (int32 tile keys)" if presort else "baseline keys"
    print(f"level 1: emitted pairs vs HIP intersect_tile, {mode} (exact)")
    for tile_size in (8, 16):
        for (w, h) in ((1920, 1080), (618, 411), (257, 129)):
            scene = _project_scene(args.num_gaussians, w, h, device, args.seed)
            all_ok &= _compare_pairs(scene, tile_size, w, h,
                                     f"projected {args.num_gaussians} @ {w}x{h}",
                                     presort=presort)
        for (w, h) in ((1920, 1080), (618, 411)):
            scene = _packed_scene(args.num_gaussians, w, h, device, args.seed)
            all_ok &= _compare_pairs(scene, tile_size, w, h,
                                     f"packed {scene['means2d'].shape[0]} @ {w}x{h}",
                                     presort=presort)
        for kind in ("empty", "fullscreen", "offscreen", "depth_ties", "single_pixel"):
            w, h = 618, 411
            scene = _synthetic_scene(kind, w, h, device)
            all_ok &= _compare_pairs(scene, tile_size, w, h, f"synthetic {kind}",
                                     presort=presort)
    return all_ok


# ----------------------------------------------------------------------------------
# level 2: the rendered image
# ----------------------------------------------------------------------------------
def run_render(args) -> bool:
    """Whether the *renderer* agrees, which is the claim that survives an
    implementation that deliberately emits a different (smaller) list."""
    from gsplat import rasterization

    device = torch.device("cuda")
    all_ok = True
    print("level 2: rasterization() output vs baseline (exact)")
    for tile_size in (8, 16):
        for (w, h) in ((618, 411), (1920, 1080)):
            scene = _project_scene(args.num_gaussians, w, h, device, args.seed)
            kw = dict(means=scene["means"], quats=scene["quats"], scales=scene["scales"],
                      opacities=torch.sigmoid(torch.zeros(args.num_gaussians,
                                                          device=device)),
                      colors=torch.rand(args.num_gaussians, 3, device=device),
                      viewmats=scene["viewmats"], Ks=scene["Ks"], width=w, height=h,
                      tile_size=tile_size)
            triisect.uninstall()
            c_ref, a_ref, _ = rasterization(**kw)
            for presort, exact in ((False, False), (True, False), (True, True)):
                triisect.install(presort=presort, exact=exact)
                c_got, a_got, _ = rasterization(**kw)
                triisect.uninstall()
                ok = torch.equal(c_ref, c_got) and torch.equal(a_ref, a_got)
                all_ok &= ok
                mode = ("emit+presorted sort+exact ellipse" if exact
                        else "emit+presorted sort" if presort else "emit only")
                if not ok:
                    d = (c_ref - c_got).abs().max().item()
                    print(f"  [FAIL] {w}x{h} tile={tile_size} {mode}  "
                          f"max|dcolor|={d:.3e}")
                else:
                    print(f"  [OK  ] {w}x{h} tile={tile_size}  {mode}")
    return all_ok


# ----------------------------------------------------------------------------------
# level 3: timing
# ----------------------------------------------------------------------------------
def _tri_isect(*args, **kwargs):
    """`triisect.isect_tiles` with the low-density dispatch disabled.

    In production the stage falls back to the HIP op below `min_density` pairs per
    Gaussian, because its fixed cost is not amortized there. Every call in this file is
    either verifying the Triton path or measuring it, and a silent fallback would make
    the first vacuous (HIP compared against HIP) and the second meaningless. The dispatch
    itself is covered by `run_dispatch`.
    """
    kwargs.setdefault("min_density", None)
    return triisect.isect_tiles(*args, **kwargs)


def _time(fn, warmup: int = 5, rep: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def run_bench(args) -> bool:
    """A/B the stage, split into count+emit and the sort.

    The split is not cosmetic. Those two move for different reasons -- the emit is about
    write coalescing and divergence, the sort about key width and bit count -- and a
    single total for the stage hides which one a change actually paid for. `sort=False`
    is exposed by the HIP op too, so both sides can be measured the same way.

    The offsets pass is included in `full` because `presort` changes it as well (a
    `searchsorted` over 4-byte keys instead of the boundary-scanning kernel over 8-byte
    ones), and it is the only part of the stage the rasterizer's input depends on."""
    device = torch.device("cuda")
    print(f"level 3: stage timing, {args.num_gaussians} Gaussians "
          f"(median of 20, ms)\n")
    hdr = (f"{'shape':<12}{'tile':>5}{'n_isects':>11}{'pairs/G':>9}"
           f"{'   ':<3}{'emit HIP':>9}{'emit map':>9}{'emit bs':>9}{'spd':>7}"
           f"{'   ':<3}{'full HIP':>9}{'full B':>8}{'full B+D':>9}{'spd':>7}")
    print(hdr)
    print("-" * len(hdr))
    for (w, h) in args.shapes:
        for tile_size in (8, 16):
            scene = _project_scene(args.num_gaussians, w, h, device, args.seed)
            m, r, d = scene["means2d"], scene["radii"], scene["depths"]
            tw, th = math.ceil(w / tile_size), math.ceil(h / tile_size)

            def hip_full():
                ids = hip_isect_tiles(m, r, d, tile_size, tw, th)[1]
                isect_offset_encode(ids, 1, tw, th)

            def tri_full(presort):
                ids = _tri_isect(m, r, d, tile_size, tw, th, presort=presort)[1]
                triisect.isect_offset_encode(ids, 1, tw, th)

            # count+emit only, which isolates what stage B changes. `map` vs `bs` is the
            # slot->Gaussian mechanism: precomputed repeat_interleave vs in-kernel
            # binary search.
            e_hip = _time(lambda: hip_isect_tiles(m, r, d, tile_size, tw, th, sort=False))
            e_map = _time(lambda: _tri_isect(m, r, d, tile_size, tw, th,
                                             sort=False, search=False))
            e_bs = _time(lambda: _tri_isect(m, r, d, tile_size, tw, th,
                                            sort=False, search=True))
            f_hip = _time(hip_full)
            f_b = _time(lambda: tri_full(False))
            f_bd = _time(lambda: tri_full(True))
            n_isects = hip_isect_tiles(m, r, d, tile_size, tw, th, sort=False)[1].numel()
            print(f"{w}x{h:<6}{tile_size:>5}{n_isects:>11}"
                  f"{n_isects / args.num_gaussians:>9.1f}"
                  f"{'   ':<3}{e_hip:>9.3f}{e_map:>9.3f}{e_bs:>9.3f}"
                  f"{e_hip / e_bs:>6.2f}x"
                  f"{'   ':<3}{f_hip:>9.3f}{f_b:>8.3f}{f_bd:>9.3f}"
                  f"{f_hip / f_bd:>6.2f}x")
    return True


def _ideal_exact_set(scene, tile_size: int, width: int, height: int):
    """Brute force which `(Gaussian, tile)` pairs can contribute at all.

    Evaluates the rasterizer's own accept test -- `sigma >= 0` and
    `min(0.999, opacity*exp(-sigma)) >= 1/255`, with `sigma` written the same way -- at
    every pixel centre of every tile, for every Gaussian, and reduces with `any` over the
    tile. This is the set an oracle would emit, and it is the reference the exact test has
    to *contain*: dropping a pair in here would change the image.

    Deliberately independent of the implementation -- pure torch over the whole image, no
    ellipse algebra, no closed forms -- so a mistake in the row-span derivation cannot hide
    behind the same mistake in the reference.
    """
    means2d = scene["means2d"][0]          # [N, 2]
    radii = scene["radii"][0]              # [N, 2]
    conics = scene["conics"][0]            # [N, 3]
    opac = scene["opacities"]              # [N]
    n = means2d.shape[0]
    tw = math.ceil(width / tile_size)
    th = math.ceil(height / tile_size)
    dev = means2d.device

    # pad the image up to whole tiles; padded pixels are outside the image and must not
    # count, so they are masked out rather than evaluated
    ph, pw = th * tile_size, tw * tile_size
    ys = torch.arange(ph, device=dev, dtype=torch.float32) + 0.5
    xs = torch.arange(pw, device=dev, dtype=torch.float32) + 0.5
    valid = ((torch.arange(ph, device=dev)[:, None] < height)
             & (torch.arange(pw, device=dev)[None, :] < width))

    accept = torch.zeros((n, th, tw), dtype=torch.bool, device=dev)
    chunk = max(1, int(4e7 // (ph * pw)))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        mx = means2d[s:e, 0, None, None]
        my = means2d[s:e, 1, None, None]
        A = conics[s:e, 0, None, None]
        B = conics[s:e, 1, None, None]
        C = conics[s:e, 2, None, None]
        dx = mx - xs[None, None, :]
        dy = my - ys[None, :, None]
        sigma = 0.5 * (A * dx * dx + C * dy * dy) + B * dx * dy
        alpha = torch.clamp(opac[s:e, None, None] * torch.exp(-sigma), max=0.999)
        ok = (sigma >= 0) & (alpha >= 1.0 / 255.0) & valid[None]
        accept[s:e] = (ok.reshape(e - s, th, tile_size, tw, tile_size)
                       .any(dim=4).any(dim=2))

    # the rasterizer only ever sees pairs inside the bounding box, so restrict to it
    fj = means2d[:, 0] / tile_size
    fi = means2d[:, 1] / tile_size
    frx = radii[:, 0].float() / tile_size
    fry = radii[:, 1].float() / tile_size
    jmin = torch.clamp(torch.floor(fj - frx), 0, tw).int()
    jmax = torch.clamp(torch.ceil(fj + frx), 0, tw).int()
    imin = torch.clamp(torch.floor(fi - fry), 0, th).int()
    imax = torch.clamp(torch.ceil(fi + fry), 0, th).int()
    live = (radii[:, 0] > 0) & (radii[:, 1] > 0)
    jj = torch.arange(tw, device=dev)[None, None, :]
    ii = torch.arange(th, device=dev)[None, :, None]
    in_bbox = ((jj >= jmin[:, None, None]) & (jj < jmax[:, None, None])
               & (ii >= imin[:, None, None]) & (ii < imax[:, None, None])
               & live[:, None, None])
    return accept & in_bbox, in_bbox


def _emitted_set(flatten_ids, isect_ids, n, tile_size, width, height):
    """`(Gaussian, tile)` pairs as a dense bool grid, for set comparisons."""
    tw = math.ceil(width / tile_size)
    th = math.ceil(height / tile_size)
    if isect_ids.dtype == torch.int64:
        tiles = (isect_ids >> 32) & ((1 << 32) - 1)
    else:
        tiles = isect_ids.long()
    tiles = tiles % (tw * th)
    grid = torch.zeros(n * th * tw, dtype=torch.bool, device=flatten_ids.device)
    grid[flatten_ids.long() * (th * tw) + tiles] = True
    return grid.view(n, th, tw)


def run_exact(args) -> bool:
    """The exact test's gate: never drop a pair that could contribute, and never invent one.

    Both directions matter and they fail differently. Emitting a pair the bounding box
    would not have is a correctness bug in the row spans; dropping one the oracle keeps
    changes the image. The middle column is how much of the theoretical saving the
    (deliberately conservative) closed form actually gets.
    """
    device = torch.device("cuda")
    all_ok = True
    print("level 5: exact ellipse test vs a brute-force oracle")
    hdr = (f"{'shape':<12}{'tile':>5}{'bbox pairs':>12}{'exact':>11}{'oracle':>11}"
           f"{'  saved':>9}{'  of ideal':>11}{'  safe':>7}{'  subset':>9}")
    print(hdr)
    print("-" * len(hdr))
    for (w, h) in ((256, 256), (618, 411)):
        for tile_size in (8, 16):
            scene = _project_scene(args.exact_gaussians, w, h, device, args.seed)
            n = scene["means2d"].shape[1]
            tw, th = math.ceil(w / tile_size), math.ceil(h / tile_size)
            ideal, in_bbox = _ideal_exact_set(scene, tile_size, w, h)

            got = _tri_isect(
                scene["means2d"], scene["radii"], scene["depths"],
                tile_size, tw, th, conics=scene["conics"])
            mine = _emitted_set(got[2], got[1], n, tile_size, w, h)

            # every pair the oracle keeps must survive, and nothing outside the bbox
            safe = bool((ideal & ~mine).sum() == 0)
            subset = bool((mine & ~in_bbox).sum() == 0)
            n_bbox = int(in_bbox.sum())
            n_mine = int(mine.sum())
            n_ideal = int(ideal.sum())
            all_ok &= safe and subset
            print(f"{w}x{h:<6}{tile_size:>5}{n_bbox:>12}{n_mine:>11}{n_ideal:>11}"
                  f"{100 * (1 - n_mine / n_bbox):>8.1f}%"
                  f"{100 * (n_bbox - n_mine) / max(n_bbox - n_ideal, 1):>10.1f}%"
                  f"{'  yes' if safe else '  NO ':>7}"
                  f"{'  yes' if subset else '  NO ':>9}")
    return all_ok


def run_dispatch(args) -> bool:
    """The low-density dispatch: does it fire where it should, and is it transparent?

    Two things have to hold and they are independent. The rule has to *fire* on the right
    side of the threshold, which is observable in the key dtype -- the Triton path with
    `presort` returns int32 `image|tile` keys, the HIP op returns int64 ones -- and when it
    fires the result has to be the HIP op's exactly, since that is the whole point of
    falling back.

    The rule has two conditions, so both axes have to move: the Gaussians' screen extent
    scales the density, and the population size scales the pair count independently of it.
    A cell that clears one condition and fails the other is the only kind that distinguishes
    this rule from the single threshold it replaced, so the grid is chosen to contain some.

    The remembered decision is reset per cell. It is keyed on the problem *shape*, and
    this loop deliberately changes the scene while holding the shape fixed -- something a
    training run never does, and the reason `reset_dispatch_cache` exists.
    """
    device = torch.device("cuda")
    w, h = 618, 411
    density, min_pairs = 1.5, 3_000_000
    print(f"level 6: dispatch at min_density={density}, min_pairs={min_pairs}, "
          f"{w}x{h}\n")
    hdr = (f"{'N':>9}{'scale':>7}{'tile':>5}{'pairs/G':>10}{'n_pairs':>11}"
           f"{'expect':>9}{'took':>9}{'  matches HIP':>14}{'  ok':>5}")
    print(hdr)
    print("-" * len(hdr))

    all_ok = True
    for n in (100_000, 1_000_000):
        for mult in (0.25, 1.0, 2.0):
            for tile_size in (8, 16):
                triisect.reset_dispatch_cache()
                scene = _project_scene(n, w, h, device, args.seed)
                m = scene["means2d"]
                r = (scene["radii"].float() * mult).ceil().to(torch.int32)
                d = scene["depths"]
                tw, th = math.ceil(w / tile_size), math.ceil(h / tile_size)

                ref = hip_isect_tiles(m, r, d, tile_size, tw, th)
                n_pairs = ref[2].numel()
                ppg = n_pairs / n
                expect = ("triton" if ppg >= density and n_pairs >= min_pairs
                          else "HIP")

                got = triisect.isect_tiles(m, r, d, tile_size, tw, th, presort=True,
                                           min_density=density, min_pairs=min_pairs)
                took = "HIP" if got[1].dtype == torch.int64 else "triton"

                # only the fallback is required to reproduce the HIP op bit-for-bit; the
                # Triton path with presort deliberately re-encodes the key (see run_pairs)
                same = (took == "HIP"
                        and all(torch.equal(a, b) for a, b in zip(ref, got)))
                ok = (took == expect) and (took == "triton" or same)
                all_ok &= ok
                print(f"{n:>9}{mult:>7.2f}{tile_size:>5}{ppg:>10.2f}{n_pairs:>11}"
                      f"{expect:>9}{took:>9}"
                      f"{('yes' if same else '-'):>14}{('ok' if ok else 'FAIL'):>5}")
    return all_ok


def run_multiview(args) -> bool:
    """Batches of cameras, dense and packed, at tile counts either side of a power of two.

    A separate level rather than more rows in level 1 because what it guards is a property
    of the key layout, and one image cannot constrain it: the image id is the key's high
    field, so with I=1 that field is identically zero and any packing of it passes. Tile
    counts either side of a power of two are both here, not because one of them is expected
    to be the easy case -- the field is `bit_width(count)` bits wide, which overshoots every
    count, so nothing about this is power-of-two-specific -- but because the offsets encoder
    has to reconstruct that width and getting it off by one bit is the likely mistake.
    """
    device = torch.device("cuda")
    all_ok = True
    print("level 7: batched cameras, pairs and offsets vs HIP (exact)")
    for tile_size in (8, 16):
        for (w, h) in ((1920, 1080), (100, 100), (128, 128), (618, 411)):
            tw = math.ceil(w / tile_size)
            th = math.ceil(h / tile_size)
            pow2 = (tw * th) & (tw * th - 1) == 0
            for n_images in (2, 3, 5):
                for packed in (False, True):
                    scene = _multiview_scene(args.num_gaussians // 8, w, h, device,
                                             args.seed, n_images=n_images,
                                             packed=packed)
                    label = (f"{'packed' if packed else 'dense '} I={n_images} "
                             f"@ {w}x{h} {tw * th} tiles"
                             f"{' (pow2)' if pow2 else ''}")
                    all_ok &= _compare_pairs(scene, tile_size, w, h, label,
                                             presort=True)
    return all_ok


def _capture_trainer_args(n: int, width: int, height: int, tile_size: int,
                          sh_degree: int, device):
    """The exact tensors `rasterization()` hands the intersection stage.

    Timing the stage on synthetic `fully_fused_projection` output is close but not the
    same thing: the trainer runs `packed=True`, its opacities have been through a sigmoid
    (so the opacity-aware radius clip bites differently), and `image_ids` is real. Rather
    than approximate any of that, intercept the call and keep what it was given.

    The profiler alternative -- reading per-kernel rows out of a training run -- cannot
    separate the stage cleanly, because the rocprim rows all collapse to the same
    truncated name and the step's other scans (cumsum, bincount, the optimizer) land in
    the same bucket."""
    from gsplat import rendering
    from gsplat.cuda import _wrapper

    captured = {}
    original = _wrapper.isect_tiles

    def recorder(means2d, radii, depths, ts, tw, th, **kw):
        if not captured:
            captured.update(dict(means2d=means2d, radii=radii, depths=depths,
                                 tile_size=ts, tile_width=tw, tile_height=th, **kw))
        return original(means2d, radii, depths, ts, tw, th, **kw)

    scene = _project_scene(n, width, height, device, 0, sh_degree=sh_degree)
    k = (sh_degree + 1) ** 2
    colors = torch.rand(n, k, 3, device=device)
    rendering.isect_tiles = recorder
    try:
        _c, _a, meta = rendering.rasterization(
            means=scene["means"], quats=scene["quats"], scales=scene["scales"],
            opacities=torch.sigmoid(torch.randn(n, device=device)),
            colors=colors, viewmats=scene["viewmats"], Ks=scene["Ks"],
            width=width, height=height, sh_degree=sh_degree, tile_size=tile_size,
        )
    finally:
        rendering.isect_tiles = original
    # `meta` carries the conics for the same call, which is how the exact test gets them
    # here; in a patched run they come off the projection instead (see triisect._patch).
    captured["conics"] = meta["conics"]
    return captured


def run_trainer_bench(args) -> bool:
    """Stage timing on the trainer's own tensors, for the end-to-end table's breakdown."""
    device = torch.device("cuda")
    print(f"level 4: stage timing on captured trainer tensors, "
          f"{args.num_gaussians} Gaussians, sh_degree=3 (median of 20, ms)\n")
    hdr = (f"{'shape':<12}{'tile':>5}{'nnz':>10}{'n_isects':>11}{'pairs/G':>9}"
           f"{'  pairs C':>10}{'   ':<3}{'HIP':>8}{'B':>8}{'B+D':>8}{'B+D+C':>8}"
           f"{'  best':>8}")
    print(hdr)
    print("-" * len(hdr))
    for (w, h) in args.shapes:
        for tile_size in (8, 16):
            c = _capture_trainer_args(args.num_gaussians, w, h, tile_size, 3, device)
            kw = dict(packed=c["packed"], n_images=c["n_images"],
                      image_ids=c["image_ids"], gaussian_ids=c["gaussian_ids"])
            m, r, d = c["means2d"], c["radii"], c["depths"]
            tw, th = c["tile_width"], c["tile_height"]
            conics = c.get("conics")

            def hip_full():
                ids = hip_isect_tiles(m, r, d, tile_size, tw, th, **kw)[1]
                isect_offset_encode(ids, c["n_images"], tw, th)

            def tri_full(presort, exact=False):
                ids = _tri_isect(
                    m, r, d, tile_size, tw, th, presort=presort,
                    conics=conics if exact else None, **kw)[1]
                triisect.isect_offset_encode(ids, c["n_images"], tw, th)

            f_hip = _time(hip_full)
            f_b = _time(lambda: tri_full(False))
            f_bd = _time(lambda: tri_full(True))
            f_bdc = _time(lambda: tri_full(True, exact=True)) if conics is not None else 0.0
            nnz = m.shape[0]
            n_isects = hip_isect_tiles(m, r, d, tile_size, tw, th,
                                       sort=False, **kw)[1].numel()
            n_exact = (_tri_isect(m, r, d, tile_size, tw, th, sort=False,
                                  conics=conics, **kw)[1].numel()
                       if conics is not None else n_isects)
            best = min(f_bd, f_bdc) if conics is not None else f_bd
            print(f"{w}x{h:<6}{tile_size:>5}{nnz:>10}{n_isects:>11}"
                  f"{n_isects / max(nnz, 1):>9.1f}"
                  f"{100 * (1 - n_exact / max(n_isects, 1)):>9.1f}%"
                  f"{'   ':<3}{f_hip:>8.3f}{f_b:>8.3f}{f_bd:>8.3f}{f_bdc:>8.3f}"
                  f"{f_hip / best:>7.2f}x")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-gaussians", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", action="store_true", help="level 1 (default)")
    p.add_argument("--render", action="store_true", help="level 2")
    p.add_argument("--bench", action="store_true", help="level 3")
    p.add_argument("--trainer-bench", action="store_true", help="level 4")
    p.add_argument("--exact", action="store_true", help="level 5")
    p.add_argument("--dispatch", action="store_true", help="level 6")
    p.add_argument("--multiview", action="store_true", help="level 7")
    p.add_argument("--exact-gaussians", type=int, default=4096,
                   help="Gaussians for level 5; the oracle is O(N * pixels)")
    p.add_argument("--presort", action="store_true",
                   help="also check/time the depth-presorted variant")
    p.add_argument("--shape", action="append", default=None,
                   metavar="WxH", help="image size for --bench (repeatable)")
    args = p.parse_args()
    args.shapes = ([tuple(int(v) for v in s.split("x")) for s in args.shape]
                   if args.shape else [(1920, 1080), (618, 411)])

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    print(f"torch {torch.__version__}  hip {torch.version.hip}  "
          f"{torch.cuda.get_device_name(0)}")
    print(f"triisect {triisect.__version__} from {os.path.dirname(triisect.__file__)}\n")

    ran_any = False
    ok = True
    if args.exact:
        ok &= run_exact(args)
        ran_any = True
    if args.dispatch:
        ok &= run_dispatch(args)
        ran_any = True
    if args.multiview:
        ok &= run_multiview(args)
        ran_any = True
    if args.pairs or not (args.render or args.bench or args.trainer_bench
                          or args.exact or args.dispatch or args.multiview):
        ok &= run_pairs(args)
        if args.presort:
            print()
            ok &= run_pairs(args, presort=True)
        ran_any = True
    if args.render:
        ok &= run_render(args)
        ran_any = True
    if args.bench:
        ok &= run_bench(args)
        ran_any = True
    if args.trainer_bench:
        ok &= run_trainer_bench(args)
        ran_any = True

    assert ran_any
    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
