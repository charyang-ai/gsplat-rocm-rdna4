"""Autotuned Triton replacement for gsplat's `rasterize_to_pixels_3dgs_bwd` HIP kernel.

The forward rasterizer is cheap (~0.5 ms/step at 500k Gaussians, 1080p); its backward
is the second-largest item in a 3DGS training step, right behind nothing but itself --
`rasterize_to_pixels_3dgs_bwd_kernel<3u,...>` alone is ~24% of total GPU time on
gfx1201. So only the backward is replaced here; the forward keeps using the HIP kernel.

Why a rewrite can win on RDNA4
------------------------------
The HIP kernel was written for CDNA (wave64) and carries two CDNA-shaped decisions that
cost real time on a wave32 part:

  * It stages each batch of Gaussians through LDS so the 64 threads of a tile can
    broadcast-read them, which costs two `block.sync()` barriers per batch. On AMD a
    load from a *wave-uniform* address is already a scalar (s_load) broadcast, so
    Triton reaches the same data with no LDS traffic and no barriers at all.
  * It reduces per-Gaussian gradients with a warp reduction and then has lane 0 of
    *every* warp issue its own atomicAdd. With the fork's default `tile_size=8` a block
    is 64 threads = 2 wave32 waves, so each Gaussian pays 2x9 = 18 atomics per tile.
    A single Triton program per tile with `num_warps=1` reduces across the whole tile
    inside one wave (DPP, no LDS) and issues 9.

Whether that is actually faster is an empirical question -- which is the point of
shipping this behind `--ras_bwd triton` with the HIP path still available as
`--ras_bwd baseline`.

Numerics are a transliteration of the HIP kernel, not a reformulation: same
back-to-front walk, same `T *= 1/(1-alpha)` recurrence, same `ALPHA_THRESHOLD` and
0.999 opacity clamp, same `last_ids` cutoff. Gradients should match to fp32
accumulation-order noise, nothing more.
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


# CDIM values the Triton path handles. The kernel keeps per-channel state in a
# [BLOCK_P, CPAD] tile, so wide feature channels would blow up register pressure; those
# stay on the HIP kernel.
_MAX_CDIM = 16


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1)).bit_length() if x > 1 else 1


if _HAS_TRITON:

    def _amd_option(name: str) -> bool:
        """Whether this Triton build accepts `name` as a HIP compile option. Passing an
        unrecognised one is a launch-time TypeError, so the AMD-only knobs are probed
        rather than assumed."""
        try:
            import dataclasses

            from triton.backends.amd.compiler import HIPOptions

            return name in {f.name for f in dataclasses.fields(HIPOptions)}
        except Exception:
            return False

    _HAS_WAVES_PER_EU = _amd_option("waves_per_eu")

    def _configs():
        """Full sweeps on the real training scene at both tile sizes show two competing
        effects, and the winner is wherever they cross.

        SPLIT costs work linearly. Every program re-walks its tile's whole Gaussian list
        and issues its own atomics, so SPLIT programs per tile means SPLIT times the walk.
        At tile_size=8 that is the entire story: 5.6 / 9.7 / 18.5 ms for SPLIT 1 / 2 / 4 at
        num_warps=1, near enough to a doubling each time.

        Pixels per lane, TILE*TILE / (SPLIT * 32 * num_warps), costs registers. The kernel
        keeps a [BLOCK_P, CPAD] fp32 accumulator and the loaded cotangents live across the
        whole walk, so past about 4 px/lane it spills hard -- at tile_size=16, SPLIT=1 with
        num_warps=1 is 8 px/lane and 11.4 ms, and merely halving the block to 128 px takes
        it to 6.9 ms. Below 1 px/lane it instead wastes lanes.

        So: take the smallest SPLIT that keeps px/lane out of the spilling regime. That
        rule picks both observed optima -- SPLIT=1 at tile_size=8 (2 px/lane, 5.6 ms) and
        SPLIT=2 at tile_size=16 (4 px/lane, 6.9 ms).

        num_warps moves px/lane too, but unlike SPLIT it puts the reduction across waves
        through LDS with barriers instead of keeping it in wave32 DPP. It never won at
        either tile size, and 4 was never competitive, so only 1 and 2 are kept.

        BLOCK_G is dropped: it only unrolls the walk over the tile's Gaussians, that walk
        is serial in the alpha-compositing recurrence so there is no ILP to expose, and
        duplicating the live state across unrolled steps makes the spilling worse. It lost
        monotonically at tile_size=8 and was catastrophic at tile_size=16 (21.6 ms vs
        11.6 ms at SPLIT=1). Kept as a parameter, pinned to 1.

        waves_per_eu is an AMD-only occupancy floor, worth ~3% at tile_size=8 and under 1%
        at 16, hence the probe."""
        # TRIRASTER_SPLIT pins SPLIT, so a full training run can be repeated on one value
        # without editing this list. Only for A/B-ing; unset means autotune as usual.
        pinned = os.environ.get("TRIRASTER_SPLIT")
        splits = (int(pinned),) if pinned else (1, 2, 4)

        out = []
        for split in splits:
            for nw in (1, 2):
                base = {"BLOCK_G": 1, "SPLIT": split}
                out.append(triton.Config(base, num_warps=nw, num_stages=1))
                if _HAS_WAVES_PER_EU:
                    out.append(triton.Config(dict(base, waves_per_eu=2),
                                             num_warps=nw, num_stages=1))
        return out

    @triton.autotune(
        configs=_configs(),
        key=["CDIM", "TILE", "image_width", "image_height"],
        # The kernel ACCUMULATES into its outputs with atomics, so the autotuner must
        # zero them between trial runs or the benchmarking itself corrupts the result.
        reset_to_zero=[
            "v_means2d_abs_ptr",
            "v_means2d_ptr",
            "v_conics_ptr",
            "v_colors_ptr",
            "v_opacities_ptr",
        ],
    )
    @triton.jit
    def _ras3dgs_bwd_kernel(
        # Gaussian parameters (flat, indexed by the global Gaussian id `g`)
        means2d_ptr,      # [.., 2]
        conics_ptr,       # [.., 3]
        colors_ptr,       # [.., CDIM]
        opacities_ptr,    # [..]
        backgrounds_ptr,  # [I, CDIM]
        masks_ptr,        # [I, tile_height, tile_width] (int8)
        # intersections
        tile_offsets_ptr,  # [I, tile_height, tile_width] int32
        flatten_ids_ptr,   # [n_isects] int32
        # forward outputs
        render_alphas_ptr,  # [I, H, W, 1]
        last_ids_ptr,       # [I, H, W] int32
        # upstream gradients
        v_render_colors_ptr,  # [I, H, W, CDIM]
        v_render_alphas_ptr,  # [I, H, W, 1]
        # gradient outputs (accumulated atomically)
        v_means2d_abs_ptr,
        v_means2d_ptr,
        v_conics_ptr,
        v_colors_ptr,
        v_opacities_ptr,
        # sizes
        n_isects,
        image_width,
        image_height,
        tile_width,
        tile_height,
        n_tiles,  # I * tile_height * tile_width
        CDIM: tl.constexpr,
        CPAD: tl.constexpr,   # next_pow2(CDIM)
        TILE: tl.constexpr,   # tile_size
        SPLIT: tl.constexpr,  # programs per tile; each takes TILE*TILE/SPLIT pixels
        BLOCK_G: tl.constexpr,
        HAS_BG: tl.constexpr,
        HAS_MASK: tl.constexpr,
        HAS_ABS: tl.constexpr,
    ):
        # A program owns a horizontal slice of one tile, not necessarily the whole tile.
        # This decouples how many pixels a program reduces over from the tile size the
        # intersection and sort stages want. BLOCK_P sets the size of the register-resident
        # accumulator below, so slicing is the only way to keep a 16x16 tile off the
        # spilling cliff; the price is that each slice re-walks the tile's whole Gaussian
        # list and issues its own atomics. See `_configs` for where the two cross.
        BLOCK_P: tl.constexpr = (TILE * TILE) // SPLIT

        pid = tl.program_id(0)
        tf = pid // SPLIT   # flat tile index over [I, tile_height, tile_width]
        sub = pid % SPLIT   # which slice of that tile

        if HAS_MASK:
            if tl.load(masks_ptr + tf) == 0:
                return

        tiles_per_image = tile_height * tile_width
        image_id = tf // tiles_per_image
        tile_id = tf % tiles_per_image
        ty = tile_id // tile_width
        tx = tile_id % tile_width

        range_start = tl.load(tile_offsets_ptr + tf)
        # The last tile of the last image has no successor offset; it runs to n_isects.
        nxt = tl.minimum(tf + 1, n_tiles - 1)
        range_end = tl.where(tf + 1 < n_tiles, tl.load(tile_offsets_ptr + nxt), n_isects)
        if range_end <= range_start:
            return

        # ---- per-pixel setup -------------------------------------------------------
        # sub * BLOCK_P is a multiple of TILE, so a slice is TILE/SPLIT whole rows.
        p = sub * BLOCK_P + tl.arange(0, BLOCK_P)
        gi = ty * TILE + p // TILE
        gj = tx * TILE + p % TILE
        inside = (gi < image_height) & (gj < image_width)
        px = gj.to(tl.float32) + 0.5
        py = gi.to(tl.float32) + 0.5
        # clamp to the last pixel, exactly as the HIP kernel does for out-of-image lanes
        pix = tl.minimum(gi * image_width + gj, image_width * image_height - 1)
        pix = pix.to(tl.int64) + image_id.to(tl.int64) * image_height * image_width

        T_final = 1.0 - tl.load(render_alphas_ptr + pix)
        bin_final = tl.where(inside, tl.load(last_ids_ptr + pix), 0)
        v_ra = tl.load(v_render_alphas_ptr + pix)

        ch = tl.arange(0, CPAD)
        cmask = ch < CDIM
        # Lane selectors for the per-Gaussian gradient writes. Every atomic below is a
        # *vector* atomic over one of these, never a 0-d one: a scalar `tl.atomic_add`
        # would leave it to the compiler to prove the value is wave-uniform and collapse
        # it to a single lane, and if it failed to, all 64 lanes would each add the full
        # reduced sum. That is a silent 64x gradient, not a slowdown.
        i3 = tl.arange(0, 4)
        i2 = tl.arange(0, 2)
        i1 = tl.arange(0, 1)
        vc = tl.load(
            v_render_colors_ptr + pix[:, None] * CDIM + ch[None, :],
            mask=cmask[None, :],
            other=0.0,
        )  # [BLOCK_P, CPAD]

        bg_accum = tl.zeros([BLOCK_P], dtype=tl.float32)
        if HAS_BG:
            bg = tl.load(backgrounds_ptr + image_id * CDIM + ch, mask=cmask, other=0.0)
            bg_accum = tl.sum(bg[None, :] * vc, axis=1)

        # No Gaussian past this slice's largest `last_ids` can contribute to any pixel in
        # it, which is what lets the back-to-front walk skip the tail of the list. The max
        # is over BLOCK_P pixels, so a bigger slice trims less; the HIP kernel's
        # equivalent (`warp_bin_final`) is always over 32 lanes whatever the tile size.
        # Measured to be a second-order effect next to register pressure, though.
        bf_max = tl.max(tl.where(inside, bin_final, -1))
        range_end = tl.minimum(range_end, bf_max + 1)
        if range_end <= range_start:
            return

        # ---- back-to-front walk ----------------------------------------------------
        T = T_final
        buffer = tl.zeros([BLOCK_P, CPAD], dtype=tl.float32)

        num_batches = tl.cdiv(range_end - range_start, BLOCK_G)
        for b in range(num_batches):
            batch_end = range_end - 1 - BLOCK_G * b
            for t in tl.static_range(BLOCK_G):
                idx = batch_end - t
                in_rng = idx >= range_start
                # Clamped so the scalar load stays in bounds; `in_rng` masks the result.
                g = tl.load(flatten_ids_ptr + tl.maximum(idx, range_start).to(tl.int64))

                # Wave-uniform scalar loads: the AMD scalar unit broadcasts these to all
                # lanes, which is what the HIP kernel needs its LDS staging for.
                xy_x = tl.load(means2d_ptr + 2 * g)
                xy_y = tl.load(means2d_ptr + 2 * g + 1)
                opac = tl.load(opacities_ptr + g)
                c0 = tl.load(conics_ptr + 3 * g)
                c1 = tl.load(conics_ptr + 3 * g + 1)
                c2 = tl.load(conics_ptr + 3 * g + 2)
                rgb = tl.load(colors_ptr + g * CDIM + ch, mask=cmask, other=0.0)  # [CPAD]

                dx = xy_x - px
                dy = xy_y - py
                sigma = 0.5 * (c0 * dx * dx + c2 * dy * dy) + c1 * dx * dy
                # sigma < 0 lanes are dropped below; clamping here only keeps exp finite.
                vis = tl.exp(-tl.maximum(sigma, 0.0))
                alpha = tl.minimum(0.999, opac * vis)
                valid = (
                    inside
                    & in_rng
                    & (idx <= bin_final)
                    & (sigma >= 0.0)
                    & (alpha >= 1.0 / 255.0)  # ALPHA_THRESHOLD, inlined
                )

                ra = 1.0 / (1.0 - alpha)  # alpha <= 0.999, so this is always finite
                T_new = T * ra            # transmittance BEFORE this Gaussian
                fac = alpha * T_new

                v_rgb_local = tl.where(valid[:, None], fac[:, None] * vc, 0.0)

                v_alpha = tl.sum(
                    (rgb[None, :] * T_new[:, None] - buffer * ra[:, None]) * vc, axis=1
                )
                v_alpha += T_final * ra * v_ra
                if HAS_BG:
                    v_alpha -= T_final * ra * bg_accum

                # The 0.999 clamp kills the gradient through sigma for saturated hits.
                gate = valid & ((opac * vis) <= 0.999)
                v_sigma = tl.where(gate, -opac * vis * v_alpha, 0.0)
                v_x = v_sigma * (c0 * dx + c1 * dy)
                v_y = v_sigma * (c1 * dx + c2 * dy)

                T = tl.where(valid, T_new, T)
                buffer = tl.where(
                    valid[:, None], buffer + rgb[None, :] * fac[:, None], buffer
                )

                # Lay each gradient out as a [BLOCK_P, lanes] tile so one reduction over
                # the tile's pixels yields the exact vector the atomic wants. Four
                # reductions total, matching the HIP kernel's four rocprim_warpSum calls
                # -- but over the whole tile, so one atomic per Gaussian instead of one
                # per wave.
                conic_cols = tl.where(
                    i3[None, :] == 0, (0.5 * v_sigma * dx * dx)[:, None],
                    tl.where(i3[None, :] == 1, (v_sigma * dx * dy)[:, None],
                             tl.where(i3[None, :] == 2, (0.5 * v_sigma * dy * dy)[:, None],
                                      0.0)))
                xy_cols = tl.where(i2[None, :] == 0, v_x[:, None], v_y[:, None])

                tl.atomic_add(v_colors_ptr + g * CDIM + ch,
                              tl.sum(v_rgb_local, axis=0), mask=cmask)
                tl.atomic_add(v_conics_ptr + 3 * g + i3,
                              tl.sum(conic_cols, axis=0), mask=i3 < 3)
                tl.atomic_add(v_means2d_ptr + 2 * g + i2, tl.sum(xy_cols, axis=0))
                if HAS_ABS:
                    tl.atomic_add(v_means2d_abs_ptr + 2 * g + i2,
                                  tl.sum(tl.abs(xy_cols), axis=0))
                tl.atomic_add(v_opacities_ptr + g + i1,
                              tl.sum(tl.where(gate, vis * v_alpha, 0.0)[:, None], axis=0))


def supports(
    colors: torch.Tensor,
    tile_size: int,
    means2d: torch.Tensor,
) -> bool:
    """True when the Triton path can handle this call; otherwise the caller must use the
    HIP kernel. Kept narrow on purpose -- a wrong-but-plausible gradient is worse than a
    slower one."""
    if not _HAS_TRITON:
        return False
    cdim = colors.shape[-1]
    return (
        means2d.is_cuda
        and means2d.dtype == torch.float32
        and colors.dtype == torch.float32
        and 1 <= cdim <= _MAX_CDIM
        and tile_size in (8, 16)
    )


def _marshal(
    means2d, conics, colors, opacities, backgrounds, masks,
    image_width, image_height, tile_size, tile_offsets, flatten_ids,
    render_alphas, last_ids, v_render_colors, v_render_alphas, absgrad,
):
    """Allocate the gradient buffers and lay out the kernel's arguments.

    Shared by the normal call and `bench_configs`, so a config sweep can never drift
    from what the training path actually launches."""
    cdim = colors.shape[-1]
    tile_height, tile_width = tile_offsets.shape[-2:]
    n_tiles = tile_offsets.numel()

    outs = dict(
        v_means2d=torch.zeros_like(means2d),
        v_conics=torch.zeros_like(conics),
        v_colors=torch.zeros_like(colors),
        v_opacities=torch.zeros_like(opacities),
        # Always a real tensor: `triton.autotune(reset_to_zero=...)` cannot zero a None.
        v_means2d_abs=torch.zeros_like(means2d) if absgrad else means2d.new_zeros(1),
    )

    dummy = means2d.new_zeros(1)
    # torch.bool maps to Triton's i1, which is not a sane pointee type; the reinterpret
    # to int8 is free (same 1-byte storage) and the kernel just tests != 0.
    masks_i8 = masks.contiguous().view(torch.int8) if masks is not None else dummy
    args = (
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        backgrounds.contiguous() if backgrounds is not None else dummy,
        masks_i8,
        tile_offsets.contiguous(),
        flatten_ids.contiguous(),
        # the HIP forward can hand back float64 alphas; the kernel reads fp32
        render_alphas.contiguous().float(),
        last_ids.contiguous(),
        v_render_colors.contiguous(),
        v_render_alphas.contiguous(),
        outs["v_means2d_abs"],
        outs["v_means2d"],
        outs["v_conics"],
        outs["v_colors"],
        outs["v_opacities"],
        flatten_ids.numel(),
        image_width,
        image_height,
        tile_width,
        tile_height,
        n_tiles,
    )
    meta = dict(
        CDIM=cdim,
        CPAD=_next_pow2(cdim),
        TILE=tile_size,
        HAS_BG=backgrounds is not None,
        HAS_MASK=masks is not None,
        HAS_ABS=absgrad,
    )
    # SPLIT comes from the autotune config, so the grid has to be resolved per config.
    return outs, args, meta, lambda cfg: (n_tiles * cfg["SPLIT"],)


def rasterize_to_pixels_3dgs_bwd(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    backgrounds: Optional[torch.Tensor],
    masks: Optional[torch.Tensor],
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    render_alphas: torch.Tensor,
    last_ids: torch.Tensor,
    v_render_colors: torch.Tensor,
    v_render_alphas: torch.Tensor,
    absgrad: bool,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in for gsplat's `rasterize_to_pixels_3dgs_bwd` op.

    Returns `(v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities)`."""
    outs, args, meta, grid = _marshal(
        means2d, conics, colors, opacities, backgrounds, masks,
        image_width, image_height, tile_size, tile_offsets, flatten_ids,
        render_alphas, last_ids, v_render_colors, v_render_alphas, absgrad,
    )
    if flatten_ids.numel() > 0:
        _ras3dgs_bwd_kernel[grid](*args, **meta)
    return (
        outs["v_means2d_abs"] if absgrad else None,
        outs["v_means2d"],
        outs["v_conics"],
        outs["v_colors"],
        outs["v_opacities"],
    )


def _cfg_kwargs(cfg) -> dict:
    """A `triton.Config` flattened into launch kwargs, for driving one specific
    candidate through `.fn` with the autotuner bypassed."""
    try:
        return cfg.all_kwargs()
    except AttributeError:  # older Triton
        return dict(cfg.kwargs, num_warps=cfg.num_warps, num_stages=cfg.num_stages)


def configs():
    """The autotune candidate list, so callers can enumerate what `run_config` accepts."""
    if not _HAS_TRITON:
        raise RuntimeError("Triton unavailable")
    return _configs()


def run_config(
    cfg,
    means2d, conics, colors, opacities, backgrounds, masks,
    image_width, image_height, tile_size, tile_offsets, flatten_ids,
    render_alphas, last_ids, v_render_colors, v_render_alphas, absgrad,
):
    """Run the backward under one specific autotune candidate.

    Autotuning picks a config per shape-specialisation, so the config a training run
    ends up on need not be the one a correctness test at some other resolution happened
    to validate. This exists so every candidate can be checked, not just the winner.

    Returns the same tuple as `rasterize_to_pixels_3dgs_bwd`."""
    if not _HAS_TRITON:
        raise RuntimeError("Triton unavailable")
    outs, args, meta, grid = _marshal(
        means2d, conics, colors, opacities, backgrounds, masks,
        image_width, image_height, tile_size, tile_offsets, flatten_ids,
        render_alphas, last_ids, v_render_colors, v_render_alphas, absgrad,
    )
    if flatten_ids.numel() > 0:
        _ras3dgs_bwd_kernel.fn[grid](*args, **dict(meta, **_cfg_kwargs(cfg)))
    return (
        outs["v_means2d_abs"] if absgrad else None,
        outs["v_means2d"],
        outs["v_conics"],
        outs["v_colors"],
        outs["v_opacities"],
    )


def bench_configs(
    means2d, conics, colors, opacities, backgrounds, masks,
    image_width, image_height, tile_size, tile_offsets, flatten_ids,
    render_alphas, last_ids, v_render_colors, v_render_alphas, absgrad,
    warmup: int = 3, rep: int = 10,
):
    """Time every autotune candidate explicitly and return `[(config, ms), ...]` sorted
    fastest-first, plus any that failed to compile as `(config, None)`.

    `triton.autotune` picks a winner but never tells you by how much, or why. This shows
    the whole table, which is how the pixels-per-lane behaviour in `_configs` was found.

    Read the ordering, not the spread. The list deliberately contains configs that win at
    one tile size and lose badly at the other, so best-to-worst says more about how far
    apart those regimes are than about how much tuning is left on the table -- 5.8x at
    tile_size=8 versus 1.7x at 16, for a kernel whose actual optima differ by 27%. What
    matters is which knob the ordering follows.

    It does not check the results; `run_config` exists for that."""
    if not _HAS_TRITON:
        raise RuntimeError("Triton unavailable")

    results = []
    for cfg in _configs():
        outs, args, meta, grid = _marshal(
            means2d, conics, colors, opacities, backgrounds, masks,
            image_width, image_height, tile_size, tile_offsets, flatten_ids,
            render_alphas, last_ids, v_render_colors, v_render_alphas, absgrad,
        )
        # `.fn` is the underlying JITFunction, i.e. autotune bypassed: this measures one
        # specific config rather than whatever the autotuner already cached.
        launch = _ras3dgs_bwd_kernel.fn
        kwargs = dict(meta, **_cfg_kwargs(cfg))
        try:
            for _ in range(warmup):
                launch[grid](*args, **kwargs)
            torch.cuda.synchronize()
            times = []
            for _ in range(rep):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                launch[grid](*args, **kwargs)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            times.sort()
            results.append((cfg, times[len(times) // 2]))
        except Exception:
            results.append((cfg, None))
        del outs

    ok = sorted([r for r in results if r[1] is not None], key=lambda r: r[1])
    return ok + [r for r in results if r[1] is None]
