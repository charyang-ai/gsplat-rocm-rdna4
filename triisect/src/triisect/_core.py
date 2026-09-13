"""Triton replacement for gsplat's tile-intersection stage (`intersect_tile`).

What the HIP kernel does, and why it is slow on RDNA4
-----------------------------------------------------
`intersect_tile_kernel` runs twice with one thread per Gaussian. The second run walks
that Gaussian's tile bounding box and appends one `(isect_id, flatten_id)` pair per tile
at a *per-thread* cursor:

    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    for (i = tile_min.y; i < tile_max.y; ++i)
      for (j = tile_min.x; j < tile_max.x; ++j) {
        isect_ids[cur_idx] = ...; flatten_ids[cur_idx] = idx; ++cur_idx;
      }

Two things follow. The 32 lanes of a wave write to addresses `tiles_per_gauss * 8B`
apart, so one store instruction touches up to 32 cache lines instead of 4; and the trip
count varies from 1 to several hundred within a wave. At 1080p / tile 16 the pairs are
~10 M x 12 B = 120 MB, which is ~0.19 ms of writes at 644 GB/s against a measured
1.58 ms.

What this does instead
----------------------
The parallel axis becomes the *output* slot rather than the Gaussian: lane k writes pair
k, so the writes are contiguous and the load is balanced by construction. Recovering
"which Gaussian owns pair k" is a load-balancing search over the prefix sums, which
`torch.repeat_interleave` materialises in one streaming pass (see `_owner_map`).

The emitted pairs are bit-identical to the HIP kernel's, including the order within a
Gaussian (row-major over the bbox) and the key encoding, so the gate in
`tests/isect_correctness_test.py` is `torch.equal`, not a tolerance.
"""
from __future__ import annotations

import os

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton missing / CPU-only env
    _HAS_TRITON = False


_HIP_ISECT = None

# Remembered dispatch decision per problem shape.
#
# Deciding needs the count kernel and, worse, the host-side read of its sum, which is a full
# device synchronization. On the Triton path that read is needed anyway -- it sizes the pair
# buffers -- but on the fallback path it is pure overhead, and where falling back is the
# right answer the stage only takes a few tenths of a millisecond, so paying it every call
# cancels most of what falling back saves. Training runs the same shape thousands of times
# with a workload that moves slowly, so the decision is cached and re-probed occasionally
# instead. A stale decision costs performance and never correctness: both paths are exact.
_WORTH_IT: dict = {}
_PROBE_CALLS: dict = {}
_PROBE_EVERY = 64


def reset_dispatch_cache() -> None:
    """Forget the remembered decisions. For tests that switch scenes under one shape."""
    _WORTH_IT.clear()
    _PROBE_CALLS.clear()


def _worth_it(n_isects: int, n_elements: int, min_density: Optional[float],
              min_pairs: int) -> bool:
    """Whether the Triton path is expected to beat the HIP op on this problem.

    Two conditions, because there are two kinds of cost to earn back. What this path saves
    is proportional to the pair count: a narrower key over the same pairs. What it spends is
    a count kernel, a prefix sum and a depth sort over `n_elements`, plus a cost per call in
    launches and one synchronization that does not shrink with the problem at all. So the
    density has to clear the first and the absolute size has to clear the second, and one
    threshold on density cannot express both: at three pairs per element this path runs
    1.35x faster on 4.4M pairs and 0.71x on 1.9M.
    """
    if min_density is None:
        return True
    return n_isects >= min_density * n_elements and n_isects >= min_pairs


def _dispatch(dkey, min_density: Optional[float]) -> str:
    """What to do about this shape: `"hip"`, `"triton"`, or `"probe"` to measure and decide.

    Separating "take this path" from "measure first" is what lets the exact path skip the
    bounding-box count, which only the probe reads.
    """
    if min_density is None:
        return "triton"
    if dkey not in _WORTH_IT:
        return "probe"
    n = _PROBE_CALLS.get(dkey, 0) + 1
    _PROBE_CALLS[dkey] = n
    if n % _PROBE_EVERY == 0:
        return "probe"
    return "triton" if _WORTH_IT[dkey] else "hip"


def _hip_isect_tiles():
    """gsplat's own `isect_tiles`, captured so the low-density fallback cannot recurse.

    `_patch.install()` rebinds `gsplat.cuda._wrapper.isect_tiles` to the wrapper that
    calls into this module, so importing that name at fallback time would come straight
    back here. `_patch` keeps the pre-patch callable, which is the one we want.
    """
    global _HIP_ISECT
    if _HIP_ISECT is None:
        from . import _patch

        if _patch._ORIGINAL is not None:
            _HIP_ISECT = _patch._ORIGINAL[0]
        else:
            from gsplat.cuda._wrapper import isect_tiles as _hip

            _HIP_ISECT = _hip
    return _HIP_ISECT


def _bit_width(x: int) -> int:
    """gsplat's `(uint32_t)floor(log2(x)) + 1`, without the float round-trip."""
    return max(1, int(x).bit_length())


if _HAS_TRITON:

    @triton.jit
    def _tile_bbox(mx, my, rx, ry, TILE: tl.constexpr, tile_width, tile_height):
        """gsplat's tile bounding box, transliterated.

        The HIP kernel writes `min(max(0, (uint32_t)floor(v)), limit)`. The float->uint32
        conversion saturates on both NVIDIA and AMD hardware -- negatives to 0, huge
        values to 0xFFFFFFFF -- so max(0, .) is a no-op and the whole expression is a
        clamp of `floor(v)` into [0, limit]. Doing that clamp in fp32 before the integer
        conversion reproduces it exactly for every input, including the off-screen and
        overflow cases, and `TILE` is a power of two so the divisions are exact.
        """
        trx = rx / TILE
        try_ = ry / TILE
        tx = mx / TILE
        ty = my / TILE
        fw = tile_width.to(tl.float32)
        fh = tile_height.to(tl.float32)
        jmin = tl.minimum(tl.maximum(tl.floor(tx - trx), 0.0), fw).to(tl.int32)
        jmax = tl.minimum(tl.maximum(tl.ceil(tx + trx), 0.0), fw).to(tl.int32)
        imin = tl.minimum(tl.maximum(tl.floor(ty - try_), 0.0), fh).to(tl.int32)
        imax = tl.minimum(tl.maximum(tl.ceil(ty + try_), 0.0), fh).to(tl.int32)
        return jmin, jmax, imin, imax

    @triton.jit
    def _load_gauss(means2d_ptr, radii_ptr, idx, mask):
        """`means2d[idx]` and `radii[idx]`, both stored as [..., 2]."""
        mx = tl.load(means2d_ptr + 2 * idx, mask=mask, other=0.0)
        my = tl.load(means2d_ptr + 2 * idx + 1, mask=mask, other=0.0)
        rx = tl.load(radii_ptr + 2 * idx, mask=mask, other=0).to(tl.float32)
        ry = tl.load(radii_ptr + 2 * idx + 1, mask=mask, other=0).to(tl.float32)
        return mx, my, rx, ry

    @triton.jit
    def _counts_kernel(
        means2d_ptr,   # [.., N, 2] fp32
        radii_ptr,     # [.., N, 2] int32
        counts_ptr,    # [n] int32 out
        n,
        TILE: tl.constexpr,
        tile_width,
        tile_height,
        BLOCK: tl.constexpr,
    ):
        """First pass: `tiles_per_gauss`. One lane per Gaussian, fully coalesced."""
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = idx < n
        mx, my, rx, ry = _load_gauss(means2d_ptr, radii_ptr, idx, m)
        jmin, jmax, imin, imax = _tile_bbox(mx, my, rx, ry, TILE, tile_width, tile_height)
        cnt = (jmax - jmin) * (imax - imin)
        # A non-positive radius means the projection culled this Gaussian; the HIP kernel
        # returns before touching the bbox at all.
        cnt = tl.where((rx > 0) & (ry > 0), cnt, 0)
        tl.store(counts_ptr + idx, cnt, mask=m)

    @triton.jit
    def _ellipse_scale(A, B, C, rx, ry, EPS: tl.constexpr):
        """The squared Mahalanobis radius `R` of the ellipse whose bounding box is
        `[+-rx, +-ry]`, from the conic alone.

        gsplat builds the box as `radius = ceil(extend * sqrt(covar[i][i]))` with
        `extend^2 = min(3.33^2, 2 ln(opacity/ALPHA_THRESHOLD))`, and the rasterizer
        contributes to a pixel exactly when `sigma = Q(d)/2 <= ln(255 * opacity)`. Since
        opacity <= 1 the 3.33 cap never binds, so the region the rasterizer actually uses
        is `Q(d) <= extend^2` -- precisely the ellipse this box was drawn around.

        Recovering `R` from the box instead of from the opacity is what keeps the opacity
        out of this function's inputs, and it is safe in the right direction: the `ceil`
        means `rx >= extend*sigma_x`, so `rx^2/covar_xx >= extend^2` and the ellipse can
        only come out too large. `EPS` adds a little more margin, for the gap between
        gsplat's `__logf`-based `extend` and the exact logarithm.

        With `Sigma = inverse([[A,B],[B,C]])`, `covar_xx = C/det` and `covar_yy = A/det`.
        """
        det = A * C - B * B
        rx2 = rx * rx
        ry2 = ry * ry
        # R_x = rx^2 / covar_xx, R_y = ry^2 / covar_yy; the tighter is still >= extend^2
        R = tl.minimum(rx2 * det / C, ry2 * det / A)
        return R * (1.0 + EPS)

    @triton.jit
    def _row_span(A, B, C, R, dya, dyb):
        """The x-interval of `{d : Q(d) <= R}` restricted to the band `dya <= dy <= dyb`.

        A band cut through an ellipse is convex, so its shadow on the x-axis is a single
        interval -- which is what makes the exact test cost one closed-form solve per tile
        *row* rather than one per tile, and is why the surviving tiles in a row stay
        contiguous (so a run-length is enough to describe them).

        The interval's ends are either the ellipse's own x-extremes, when those fall inside
        the band, or where the ellipse boundary crosses the band's edges:
        `A dx^2 + 2B dx dy + C dy^2 = R` solved for dx gives
        `dx = (-B dy +- sqrt(A R - det dy^2)) / A`.

        Returns `(lo, hi)` with `hi < lo` when the band misses the ellipse.
        """
        det = A * C - B * B
        lo = 1e30
        hi = -1e30

        # the ellipse's extreme dx, attained at dy = -B*dx/C
        xg = tl.sqrt(tl.maximum(R * C / det, 0.0))
        dyc = -B * xg / C
        inside_pos = (dyc >= dya) & (dyc <= dyb)
        hi = tl.where(inside_pos, xg, hi)
        inside_neg = (-dyc >= dya) & (-dyc <= dyb)
        lo = tl.where(inside_neg, -xg, lo)

        for e in tl.static_range(2):
            dy = tl.where(e == 0, dya, dyb)
            disc = A * R - det * dy * dy
            ok = disc >= 0.0
            s = tl.sqrt(tl.maximum(disc, 0.0))
            lo = tl.where(ok, tl.minimum(lo, (-B * dy - s) / A), lo)
            hi = tl.where(ok, tl.maximum(hi, (-B * dy + s) / A), hi)
        return lo, hi

    @triton.jit
    def _rows_kernel(
        means2d_ptr,   # [.., N, 2] fp32
        radii_ptr,     # [.., N, 2] int32
        rows_ptr,      # [n] int32 out
        n,
        TILE: tl.constexpr,
        tile_width,
        tile_height,
        BLOCK: tl.constexpr,
    ):
        """Exact-mode first pass: how many tile *rows* each Gaussian's bbox spans."""
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = idx < n
        mx, my, rx, ry = _load_gauss(means2d_ptr, radii_ptr, idx, m)
        _jmin, _jmax, imin, imax = _tile_bbox(mx, my, rx, ry, TILE, tile_width,
                                              tile_height)
        rows = tl.where((rx > 0) & (ry > 0), imax - imin, 0)
        tl.store(rows_ptr + idx, rows, mask=m)

    @triton.autotune(configs=[triton.Config({"BLOCK": b}, num_warps=w, num_stages=1)
                              for b in (256, 512, 1024) for w in (1, 2, 4)],
                     key=["TILE", "tile_width", "tile_height"])
    @triton.jit
    def _spans_kernel(
        means2d_ptr,       # [.., N, 2] fp32
        radii_ptr,         # [.., N, 2] int32
        conics_ptr,        # [.., N, 3] fp32
        row_base_ptr,      # [n] int64, exclusive prefix of rows, in emit order
        perm_ptr,          # [n] int32 rank -> Gaussian (PERM), else unused
        cnt_ptr,           # [n_rows] int32 out: surviving tiles in this row
        tile_base_ptr,     # [n_rows] int32 out: tile id of the row's first survivor
        gauss_ptr,         # [n_rows] int32 out: the row's Gaussian
        n_rows,
        n_search,
        TILE: tl.constexpr,
        tile_width,
        tile_height,
        ITERS: tl.constexpr,
        PERM: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Exact-mode second pass: one lane per (Gaussian, tile row).

        Output-parallel over rows for the same reason the emit is output-parallel over
        pairs -- the row count per Gaussian varies from 1 to hundreds, so walking rows
        inside a per-Gaussian lane would put the divergence straight back.
        """
        r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = r < n_rows
        r64 = r.to(tl.int64)

        lo = tl.zeros([BLOCK], dtype=tl.int32)
        hi = tl.full([BLOCK], n_search - 1, dtype=tl.int32)
        for _ in tl.static_range(ITERS):
            mid = (lo + hi + 1) // 2
            v = tl.load(row_base_ptr + mid, mask=m, other=0)
            take = v <= r64
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid - 1)
        base = tl.load(row_base_ptr + lo, mask=m, other=0)
        if PERM:
            g = tl.load(perm_ptr + lo, mask=m, other=0)
        else:
            g = lo

        mx, my, rxi, ryi = _load_gauss(means2d_ptr, radii_ptr, g, m)
        jmin, jmax, imin, _imax = _tile_bbox(mx, my, rxi, ryi, TILE, tile_width,
                                             tile_height)
        i = imin + (r64 - base).to(tl.int32)

        A = tl.load(conics_ptr + 3 * g, mask=m, other=1.0)
        B = tl.load(conics_ptr + 3 * g + 1, mask=m, other=0.0)
        C = tl.load(conics_ptr + 3 * g + 2, mask=m, other=1.0)
        R = _ellipse_scale(A, B, C, rxi, ryi, EPS)

        # The band of pixel *centres* this tile row covers. Centres sit at integer + 0.5,
        # and the band is not clipped to the image height: a superset of the real centres
        # can only keep tiles, never drop one.
        fi = i.to(tl.float32)
        dya = fi * TILE + 0.5 - my
        dyb = (fi + 1.0) * TILE - 0.5 - my
        xlo, xhi = _row_span(A, B, C, R, dya, dyb)

        # tile j covers centres [j*TILE + 0.5, (j+1)*TILE - 0.5], so it survives iff
        # j*TILE + 0.5 <= xhi and (j+1)*TILE - 0.5 >= xlo
        axlo = xlo + mx
        axhi = xhi + mx
        jlo = tl.ceil((axlo + 0.5) / TILE).to(tl.int32) - 1
        jhi = tl.floor((axhi - 0.5) / TILE).to(tl.int32)
        # never emit a tile the bbox path would not have: the AABB is the outer bound
        jlo = tl.maximum(jlo, jmin)
        jhi = tl.minimum(jhi, jmax - 1)
        cnt = tl.maximum(jhi - jlo + 1, 0)
        cnt = tl.where(xhi >= xlo, cnt, 0)

        tl.store(cnt_ptr + r, cnt, mask=m)
        tl.store(tile_base_ptr + r, i * tile_width + jlo, mask=m)
        tl.store(gauss_ptr + r, g, mask=m)

    @triton.autotune(configs=[triton.Config({"BLOCK": b}, num_warps=w, num_stages=1)
                              for b in (256, 512, 1024) for w in (1, 2, 4)],
                     key=["TILE_N_BITS", "KEY64"])
    @triton.jit
    def _emit_rows_kernel(
        depths_ptr,        # [.., N] fp32
        image_ids_ptr,     # [nnz] int64, only used when PACKED
        row_prefix_ptr,    # [n_rows] int64, exclusive prefix of cnt
        tile_base_ptr,     # [n_rows] int32
        gauss_ptr,         # [n_rows] int32
        keys_ptr,          # [n_isects] int64 (KEY64) or int32 out
        flatten_ids_ptr,   # [n_isects] int32 out
        n_isects,
        N,
        n_rows,
        TILE_N_BITS: tl.constexpr,
        KEY64: tl.constexpr,
        ITERS: tl.constexpr,
        PACKED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Exact-mode third pass, one lane per emitted pair.

        Structurally the same as `_emit_kernel`, and cheaper: the run the search lands in
        already carries the tile id of its first member, so there is no bbox to recompute
        and no division -- the tile id is `tile_base[r] + (k - prefix[r])`.
        """
        k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = k < n_isects
        k64 = k.to(tl.int64)

        lo = tl.zeros([BLOCK], dtype=tl.int32)
        hi = tl.full([BLOCK], n_rows - 1, dtype=tl.int32)
        for _ in tl.static_range(ITERS):
            mid = (lo + hi + 1) // 2
            v = tl.load(row_prefix_ptr + mid, mask=m, other=0)
            take = v <= k64
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid - 1)
        base = tl.load(row_prefix_ptr + lo, mask=m, other=0)
        tbase = tl.load(tile_base_ptr + lo, mask=m, other=0)
        g = tl.load(gauss_ptr + lo, mask=m, other=0)

        tile_id = tbase + (k64 - base).to(tl.int32)
        if PACKED:
            iid = tl.load(image_ids_ptr + g, mask=m, other=0).to(tl.int32)
        else:
            iid = g // N

        if KEY64:
            d = tl.load(depths_ptr + g, mask=m, other=0.0)
            depth_bits = d.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
            key = (iid.to(tl.int64) << (32 + TILE_N_BITS)) \
                | (tile_id.to(tl.int64) << 32) | depth_bits
        else:
            key = (iid << TILE_N_BITS) | tile_id

        tl.store(keys_ptr + k, key, mask=m)
        tl.store(flatten_ids_ptr + k, g, mask=m)

    def _emit_configs():
        """The emit kernel is pure streaming -- one gather-heavy read of per-Gaussian
        state, one contiguous 12 B write per lane, no accumulator and no atomics. So
        unlike triraster's backward it is not register-pressure bound, and the only knobs
        that matter are how much work a program carries (BLOCK) and how many waves share
        the program's scheduling slot (num_warps)."""
        out = []
        for block in (256, 512, 1024):
            for nw in (1, 2, 4):
                out.append(triton.Config({"BLOCK": block}, num_warps=nw, num_stages=1))
        return out

    @triton.autotune(configs=_emit_configs(), key=["TILE", "tile_width", "tile_height"])
    @triton.jit
    def _emit_kernel(
        means2d_ptr,     # [.., N, 2] fp32
        radii_ptr,       # [.., N, 2] int32
        depths_ptr,      # [.., N] fp32
        base_ptr,        # [n] int64, exclusive first output slot, in emit order
        owner_ptr,       # [n_isects] int32 (SEARCH=False) or [n] permutation (True)
        image_ids_ptr,   # [nnz] int64, only used when PACKED
        keys_ptr,        # [n_isects] int64 (KEY64) or int32 out
        flatten_ids_ptr,  # [n_isects] int32 out
        n_isects,
        N,               # Gaussians per image (for the image id)
        n_search,        # length of base_ptr, only used when SEARCH
        TILE: tl.constexpr,
        tile_width,
        tile_height,
        TILE_N_BITS: tl.constexpr,
        KEY64: tl.constexpr,
        SEARCH: tl.constexpr,
        ITERS: tl.constexpr,
        PERM: tl.constexpr,
        PACKED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Second pass, one lane per emitted pair.

        `KEY64` selects the key layout. The 64-bit one is gsplat's:
        `image | tile | depth-bits`, which is what makes a single sort put a tile's
        Gaussians in depth order. The 32-bit one drops the depth field and keeps
        `image | tile`, which is only sortable into the same order because the pairs are
        already emitted in depth order and the radix sort is stable -- see
        `isect_tiles(presort=True)`.

        `SEARCH` selects how a lane finds the Gaussian that owns its slot. Both ways are
        kept because the choice is not obvious and the answer moved once measured:

          False -- read it from a precomputed `owner[k]`, which `repeat_interleave`
                   materialises. One extra streaming array, and at 1080p / tile 8 writing
                   and reading it costs *more* than the emit kernel it feeds (1.24 ms vs
                   0.998 ms).
          True  -- binary-search the prefix sums in-kernel. `ITERS` is
                   `bit_length(n_search)` (~20), and while that sounds expensive the
                   window a program touches is a few cache lines wide, because
                   consecutive slots belong to neighbouring Gaussians. No extra DRAM
                   traffic at all.

        Zero-count Gaussians need no special case in the search: their `base` equals their
        successor's, and taking the *last* index with `base[idx] <= k` steps over them.
        """
        k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = k < n_isects
        k64 = k.to(tl.int64)

        if SEARCH:
            lo = tl.zeros([BLOCK], dtype=tl.int32)
            hi = tl.full([BLOCK], n_search - 1, dtype=tl.int32)
            for _ in tl.static_range(ITERS):
                mid = (lo + hi + 1) // 2
                v = tl.load(base_ptr + mid, mask=m, other=0)
                take = v <= k64
                lo = tl.where(take, mid, lo)
                hi = tl.where(take, hi, mid - 1)
            base = tl.load(base_ptr + lo, mask=m, other=0)
            if PERM:
                # `lo` is a rank in depth order; the permutation maps it to a Gaussian id.
                g = tl.load(owner_ptr + lo, mask=m, other=0)
            else:
                g = lo
        else:
            g = tl.load(owner_ptr + k, mask=m, other=0)
            # Consecutive lanes mostly share `g` -- ~20 pairs per Gaussian at
            # 1080p/tile 16 -- so this gather and the ones in `_load_gauss` are L1 hits.
            base = tl.load(base_ptr + g, mask=m, other=0)
        kl = (k64 - base).to(tl.int32)

        mx, my, rx, ry = _load_gauss(means2d_ptr, radii_ptr, g, m)
        jmin, jmax, imin, imax = _tile_bbox(mx, my, rx, ry, TILE, tile_width, tile_height)
        bw = jmax - jmin
        # Row-major within the bbox, matching the HIP kernel's `for i { for j { } }`.
        bw = tl.maximum(bw, 1)  # guard the division; kl >= bw cannot occur when count > 0
        i = imin + kl // bw
        j = jmin + kl % bw
        tile_id = i * tile_width + j
        # Packed inputs are [nnz] with an explicit image id per element; dense ones are
        # [I, N] flattened, where the image id is implicit in the index.
        if PACKED:
            iid = tl.load(image_ids_ptr + g, mask=m, other=0).to(tl.int32)
        else:
            iid = g // N

        if KEY64:
            # image id | tile id (tile_n_bits) | depth (32 bits), with the depth kept as
            # the *bit pattern* of the fp32 value, zero-extended -- not its numeric
            # value. The mask is the zero-extension: `.to(tl.int64)` sign-extends.
            d = tl.load(depths_ptr + g, mask=m, other=0.0)
            depth_bits = d.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
            key = (iid.to(tl.int64) << (32 + TILE_N_BITS)) \
                | (tile_id.to(tl.int64) << 32) | depth_bits
        else:
            key = (iid << TILE_N_BITS) | tile_id

        tl.store(keys_ptr + k, key, mask=m)
        tl.store(flatten_ids_ptr + k, g, mask=m)


def supports(means2d: torch.Tensor, radii: torch.Tensor, depths: torch.Tensor,
             tile_size: int, packed: bool, segmented: bool,
             image_ids: Optional[torch.Tensor] = None) -> bool:
    """Whether the Triton path handles these inputs.

    Anything else falls back to the HIP op: a wrong-but-plausible intersection list is
    worse than a slower one, and the fallbacks here are all shapes the 3DGS trainer
    never takes."""
    if not _HAS_TRITON:
        return False
    # Segmented mode sorts per image with a different cub entry point; it is not on the
    # trainer's path and is left to the HIP op.
    if segmented:
        return False
    if not (means2d.is_cuda and radii.is_cuda and depths.is_cuda):
        return False
    if means2d.dtype != torch.float32 or depths.dtype != torch.float32:
        return False
    if radii.dtype != torch.int32:
        return False
    if packed and (image_ids is None or means2d.dim() != 2):
        return False
    if not packed and means2d.dim() < 3:
        return False
    return True


def _owner_map(counts: torch.Tensor, n_isects: int) -> torch.Tensor:
    """`owner[k]` = the Gaussian that emits pair k.

    This is the load-balancing search that lets the emit kernel be output-parallel, and
    `repeat_interleave` is the cheapest way to get it: one streaming pass writing
    n_isects int32 (40 MB at 10 M pairs), against ~21 dependent gathers per pair for an
    in-kernel binary search over the prefix sums. `output_size` is passed because
    otherwise ATen has to reduce `counts` on the device and sync to find it -- we already
    know it from the `.item()` on the prefix sum.

    Kept int32 deliberately: int64 here would double the only genuinely new DRAM traffic
    this design adds.
    """
    n = counts.numel()
    ids = torch.arange(n, dtype=torch.int32, device=counts.device)
    return torch.repeat_interleave(ids, counts.long(), output_size=n_isects)


def _sort_pairs(keys: torch.Tensor, values: torch.Tensor,
                begin_bit: int, end_bit: int):
    """gsplat's own cub `DeviceRadixSort::SortPairs`, over an explicit bit range.

    Exposed by `patches/expose_sort.gfx1201.patch`; falls back to `torch.sort` (a
    full-width sort with int64 indices, ~2x the traffic) if the op is missing, so the
    package still runs against an unpatched fork.

    The inputs are used as the sort's scratch buffers and are garbage afterwards."""
    from gsplat.cuda._wrapper import _make_lazy_cuda_func

    try:
        op = _make_lazy_cuda_func("intersect_sort_pairs")
        return op(keys, values, begin_bit, end_bit)
    except AttributeError:
        keys_sorted, order = torch.sort(keys, stable=True)
        return keys_sorted, values[order]


def _depth_presort(depths: torch.Tensor, n_elements: int) -> torch.Tensor:
    """A permutation of the Gaussians into gsplat's depth order.

    gsplat sorts on the fp32 *bit pattern* of the depth, zero-extended to 64 bits -- not
    on its numeric value. For the positive depths that survive near-plane culling the two
    agree, but reproducing the encoding rather than the intent is what makes the emitted
    order bit-identical for every input, including the culled Gaussians whose depth field
    is whatever the projection left there.

    `stable=True` matters: it fixes the order of equal depths to ascending Gaussian index,
    which is the tie-break the baseline gets for free from emitting in index order.
    """
    bits = depths.reshape(-1).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    ids = torch.arange(n_elements, dtype=torch.int32, device=depths.device)
    _, perm = _sort_pairs(bits, ids, 0, 32)
    return perm


def isect_tiles(
    means2d: torch.Tensor,   # [..., N, 2]
    radii: torch.Tensor,     # [..., N, 2]
    depths: torch.Tensor,    # [..., N]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    segmented: bool = False,
    packed: bool = False,
    n_images: Optional[int] = None,
    image_ids: Optional[torch.Tensor] = None,
    gaussian_ids: Optional[torch.Tensor] = None,
    presort: bool = False,
    search: bool = True,
    conics: Optional[torch.Tensor] = None,
    eps: float = 5e-4,
    min_density: Optional[float] = 1.5,
    min_pairs: int = 3_000_000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in for `gsplat.cuda._wrapper.isect_tiles`.

    Returns `(tiles_per_gauss, isect_ids, flatten_ids)`.

    `search` picks how the emit kernel resolves slot -> Gaussian: in-kernel binary search
    over the prefix sums (default), or a precomputed `repeat_interleave` map. See
    `_emit_kernel`; the two differ by more than the emit kernel's own cost.

    Passing `conics` switches on the exact ellipse test: a pair is emitted only if the
    Gaussian's ellipse actually reaches the tile, rather than if its bounding box does.
    The resulting list is *shorter* than the baseline's, not a permutation of it -- see
    `_isect_tiles_exact` for why the rendered image is nonetheless unchanged.

    With `presort=False` the returned `isect_ids` are bit-identical to the HIP op's.
    With `presort=True` they are int32 `image|tile` keys instead: the Gaussians are
    walked in depth order, so a stable sort over the tile bits alone lands every tile's
    list in the same depth order that the baseline gets from a 46-bit sort of a 64-bit
    key. `flatten_ids` and the encoded offsets are still bit-identical -- and those are
    the only two things the rasterizer reads. `isect_offset_encode` in `_patch` handles
    the narrower key.

    `min_density` and `min_pairs` are the two conditions under which this path is taken
    rather than the HIP op's; see `_worth_it` for why one number is not enough. The
    defaults are chosen on 80 timed operating points from five trained Mip-NeRF 360 scenes
    (`tests/isect_realscene.py`, scored by `tests/isect_fit_dispatch.py`), where they reach
    99.5% of what an oracle that always picked the faster path would get, and they are
    perfect on the synthetic sweep that was held out of that choice. The count kernel that
    supplies the test is 0.02 ms and its synchronization is one the pipeline performs
    anyway, so the check is close to free, and falling back is transparent: the HIP op
    returns 64-bit keys, and `isect_offset_encode` dispatches on key dtype and encodes
    either width. Falling back does not mean giving up the offsets, which are a search
    over the sorted keys at any width. Pass `min_density=None` to force this path.
    """
    means2d = means2d.contiguous()
    radii = radii.contiguous()
    depths = depths.contiguous()

    n_elements = means2d.numel() // 2
    n_tiles = tile_width * tile_height
    tile_n_bits = _bit_width(n_tiles)
    if packed:
        N = 0  # unused; the image id is read from image_ids
        I = int(n_images)
        image_ids = image_ids.contiguous()
    else:
        N = means2d.shape[-2]
        I = n_elements // N if N else 1
    # The image field needs enough bits for the largest id, `I - 1`, not for `I`: gsplat
    # uses `bit_width(I)`, which spends one bit on a field that is identically zero for
    # the single-image case every training step is. The bit is free wherever it lands
    # inside a radix pass and costs a whole pass wherever it crosses a boundary -- which
    # it does at any resolution whose tile count needs exactly 16 bits, 2560x1440 at
    # tile 8 among them. Dropping it cannot change the order, since the bits it covers
    # are zero.
    image_n_bits = int(I - 1).bit_length()

    key_dtype = torch.int64 if not presort else torch.int32
    if n_elements == 0:
        empty_k = torch.empty(0, dtype=key_dtype, device=means2d.device)
        empty_v = torch.empty(0, dtype=torch.int32, device=means2d.device)
        return depths.new_zeros(depths.shape, dtype=torch.int32), empty_k, empty_v

    def _fallback():
        return _hip_isect_tiles()(
            means2d, radii, depths, tile_size, tile_width, tile_height,
            sort=sort, segmented=segmented, packed=packed, n_images=n_images,
            image_ids=image_ids, gaussian_ids=gaussian_ids,
        )

    dkey = (tile_size, tile_width, tile_height, n_elements, conics is not None)
    decision = _dispatch(dkey, min_density)
    if decision == "hip":
        return _fallback()

    # The exact path counts tile *rows*, so the bounding-box count below is not one of its
    # inputs: only the dispatch probe reads it. Going straight to it when the answer is
    # already known saves an int32 per Gaussian and a synchronization on every call, which
    # at 500K Gaussians is 2 MB of peak and a few percent of the stage.
    if conics is not None and decision == "triton":
        return _isect_tiles_exact(
            means2d, radii, depths, conics, tile_size, tile_width, tile_height,
            sort=sort, presort=presort, packed=packed, image_ids=image_ids,
            N=N, I=I, tile_n_bits=tile_n_bits, image_n_bits=image_n_bits,
            n_elements=n_elements, key_dtype=key_dtype, eps=eps,
        )

    counts = torch.empty(n_elements, dtype=torch.int32, device=means2d.device)
    _counts_kernel[(triton.cdiv(n_elements, 1024),)](
        means2d, radii, counts, n_elements,
        TILE=tile_size, tile_width=tile_width, tile_height=tile_height,
        BLOCK=1024, num_warps=4,
    )
    tiles_per_gauss = counts.view(depths.shape)

    # One synchronization serves both the dispatch probe and the output size. The probe
    # reads the *bounding-box* pair count even when the exact test is on, because that is
    # the number saying how much work there is to save.
    n_isects = int(counts.sum().item())
    if min_density is not None:
        _WORTH_IT[dkey] = _worth_it(n_isects, n_elements, min_density, min_pairs)
        if not _WORTH_IT[dkey]:
            return _fallback()

    if conics is not None:
        return _isect_tiles_exact(
            means2d, radii, depths, conics, tile_size, tile_width, tile_height,
            sort=sort, presort=presort, packed=packed, image_ids=image_ids,
            N=N, I=I, tile_n_bits=tile_n_bits, image_n_bits=image_n_bits,
            n_elements=n_elements, key_dtype=key_dtype, eps=eps,
        )

    # `base` is the exclusive first output slot of each Gaussian *in emit order*. With
    # `presort` that order is depth order rather than index order, which is the only
    # difference between the two paths here.
    if presort:
        perm = _depth_presort(depths, n_elements)
        counts_o = counts[perm.long()]
    else:
        perm = None
        counts_o = counts
    cum_o = torch.cumsum(counts_o, 0)
    base = cum_o - counts_o

    keys = torch.empty(n_isects, dtype=key_dtype, device=means2d.device)
    flatten_ids = torch.empty(n_isects, dtype=torch.int32, device=means2d.device)
    if n_isects == 0:
        return tiles_per_gauss, keys, flatten_ids

    if search:
        # `perm` doubles as the rank->Gaussian map; without presort the rank *is* the id.
        owner = perm if presort else base
    else:
        if presort:
            owner = torch.repeat_interleave(perm, counts_o.long(),
                                            output_size=n_isects)
            # the emit kernel indexes `base` by Gaussian id, not by rank
            base_by_g = torch.empty_like(base)
            base_by_g.scatter_(0, perm.long(), base)
            base = base_by_g
        else:
            owner = _owner_map(counts, n_isects)

    grid = lambda meta: (triton.cdiv(n_isects, meta["BLOCK"]),)  # noqa: E731
    _emit_kernel[grid](
        means2d, radii, depths, base, owner, image_ids, keys, flatten_ids,
        n_isects, N, n_elements,
        TILE=tile_size, tile_width=tile_width, tile_height=tile_height,
        TILE_N_BITS=tile_n_bits, KEY64=not presort,
        SEARCH=search, ITERS=_bit_width(n_elements), PERM=presort, PACKED=packed,
    )

    if sort:
        # The bit range is the whole point of `presort`: tile_n_bits + image_n_bits
        # (14 at 1080p/tile 16) instead of 32 + those (46), over a 4-byte key instead of
        # an 8-byte one.
        if presort:
            keys, flatten_ids = _sort_pairs(
                keys, flatten_ids, 0, tile_n_bits + image_n_bits)
        else:
            keys, flatten_ids = _sort_pairs(
                keys, flatten_ids, 0, 32 + tile_n_bits + image_n_bits)

    return tiles_per_gauss, keys, flatten_ids


def _isect_tiles_exact(
    means2d, radii, depths, conics, tile_size, tile_width, tile_height,
    *, sort, presort, packed, image_ids, N, I, tile_n_bits, image_n_bits,
    n_elements, key_dtype, eps,
):
    """Tile intersection against the Gaussian's ellipse instead of its bounding box.

    Why the image does not change even though the list does
    -------------------------------------------------------
    A pair is dropped only when no pixel centre in the tile can reach
    `alpha >= ALPHA_THRESHOLD`. The forward rasterizer's response to such a pixel is

        if (sigma < 0.f || alpha < ALPHA_THRESHOLD) { continue; }

    -- it does not touch `T`, the colour accumulator, or `cur_idx`. So the sequence of
    *contributing* Gaussians each pixel sees, and the order it sees them in, is identical
    with those pairs removed, and `render_colors` / `render_alphas` come out bit-for-bit
    the same. `last_ids` does change, because it is a position within the tile's list and
    the list is shorter; it is an internal handoff to the backward, which reads the same
    shortened list, so the gradients stay consistent too.

    Three passes rather than two
    ----------------------------
    The exact test is one closed-form solve per (Gaussian, tile *row*), because a band cut
    through an ellipse projects onto a single x-interval -- so a row's surviving tiles are
    a contiguous run and a run-length describes them. That gives a middle pass over rows,
    parallel over rows for the same reason the emit is parallel over pairs. Its output
    doubles as the emit's index: the run a slot falls in already knows the tile id of its
    first member, so the emit needs one search and no arithmetic beyond an add.
    """
    conics = conics.contiguous()

    rows = torch.empty(n_elements, dtype=torch.int32, device=means2d.device)
    _rows_kernel[(triton.cdiv(n_elements, 1024),)](
        means2d, radii, rows, n_elements,
        TILE=tile_size, tile_width=tile_width, tile_height=tile_height,
        BLOCK=1024, num_warps=4,
    )

    if presort:
        perm = _depth_presort(depths, n_elements)
        rows_o = rows[perm.long()]
    else:
        perm = None
        rows_o = rows
    row_cum = torch.cumsum(rows_o, 0)
    n_rows = int(row_cum[-1].item())
    row_base = row_cum - rows_o

    empty_k = torch.empty(0, dtype=key_dtype, device=means2d.device)
    empty_v = torch.empty(0, dtype=torch.int32, device=means2d.device)
    if n_rows == 0:
        return (rows.view(depths.shape) * 0, empty_k, empty_v)

    row_cnt = torch.empty(n_rows, dtype=torch.int32, device=means2d.device)
    row_tile_base = torch.empty(n_rows, dtype=torch.int32, device=means2d.device)
    row_gauss = torch.empty(n_rows, dtype=torch.int32, device=means2d.device)
    _spans_kernel[lambda meta: (triton.cdiv(n_rows, meta["BLOCK"]),)](
        means2d, radii, conics, row_base, perm,
        row_cnt, row_tile_base, row_gauss, n_rows, n_elements,
        TILE=tile_size, tile_width=tile_width, tile_height=tile_height,
        ITERS=_bit_width(n_elements), PERM=presort, EPS=eps,
    )

    pair_cum = torch.cumsum(row_cnt, 0)
    n_isects = int(pair_cum[-1].item())
    row_prefix = pair_cum - row_cnt

    # `tiles_per_gauss` has to be the count actually emitted, not the bbox count, or the
    # two would disagree about how long the list is.
    tiles_per_gauss = torch.zeros(n_elements, dtype=torch.int32, device=means2d.device)
    tiles_per_gauss.scatter_add_(0, row_gauss.long(), row_cnt)
    tiles_per_gauss = tiles_per_gauss.view(depths.shape)

    if n_isects == 0:
        return tiles_per_gauss, empty_k, empty_v

    keys = torch.empty(n_isects, dtype=key_dtype, device=means2d.device)
    flatten_ids = torch.empty(n_isects, dtype=torch.int32, device=means2d.device)
    _emit_rows_kernel[lambda meta: (triton.cdiv(n_isects, meta["BLOCK"]),)](
        depths, image_ids, row_prefix, row_tile_base, row_gauss,
        keys, flatten_ids, n_isects, N, n_rows,
        TILE_N_BITS=tile_n_bits, KEY64=not presort,
        ITERS=_bit_width(n_rows), PACKED=packed,
    )

    if sort:
        end_bit = (tile_n_bits + image_n_bits if presort
                   else 32 + tile_n_bits + image_n_bits)
        keys, flatten_ids = _sort_pairs(keys, flatten_ids, 0, end_bit)
    return tiles_per_gauss, keys, flatten_ids


def isect_offset_encode(isect_ids: torch.Tensor, n_images: int,
                        tile_width: int, tile_height: int) -> torch.Tensor:
    """Drop-in for `gsplat.cuda._wrapper.isect_offset_encode`.

    For the int32 `image|tile` keys that `isect_tiles(presort=True)` produces, the offsets
    are a `searchsorted` over the sorted keys: entry t is the first slot whose key is >= t,
    which is exactly the "start of tile t's range" that the HIP kernel fills in by
    scanning for boundaries. It also reads 4 bytes per pair instead of 8.

    The values searched for are *keys*, which the output's flat index only happens to equal
    when there is a single image. Entry (i, ty, tx) sits at flat index
    `i*tile_height*tile_width + ty*tile_width + tx`, so its stride in i is the tile count,
    while its key is `i << tile_n_bits | tile`, whose stride in i is
    `2**bit_width(tile_count)`. That is strictly larger than the tile count for every count
    -- `bit_width` rounds up to a whole bit and gsplat sizes the field for the count rather
    than for the largest id -- so searching for the flat index is wrong for *any* batch of
    more than one image, and wrong silently, since the result is still monotone and still
    the right shape.

    The int64 `image|tile|depth` keys of the baseline encoding are handled the same way,
    with the tile field shifted up past the depth. The search is worth doing there too: the
    HIP kernel scans all `n_isects` keys looking for boundaries, while a search costs
    `n_tiles * log2(n_isects)` reads, and tiles are far outnumbered by pairs at any density
    worth optimising. That path is what the low-density fallback returns, so it is also the
    one case where this function sees keys it did not produce.
    """
    n_tiles = tile_height * tile_width
    tile_n_bits = _bit_width(n_tiles)
    wide = isect_ids.dtype == torch.int64
    dtype = torch.int64 if wide else torch.int32
    flat = torch.arange(n_images * n_tiles, dtype=dtype, device=isect_ids.device)

    # The value to search for is the smallest key belonging to the tile, which is the
    # output's flat index only when the fields happen to line up -- see above, they do not
    # once there is more than one image, and never in the 64-bit layout.
    if wide:
        bins = (((flat // n_tiles) << (32 + tile_n_bits))
                | ((flat % n_tiles) << 32))
    elif n_images == 1:
        bins = flat
    else:
        bins = ((flat // n_tiles) << tile_n_bits) | (flat % n_tiles)

    offsets = torch.searchsorted(isect_ids, bins, right=False)
    return offsets.to(torch.int32).view(n_images, tile_height, tile_width)
