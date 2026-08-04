"""Swap gsplat's `_RasterizeToPixels` autograd Function for one whose backward runs the
Triton kernel.

`gsplat.cuda._wrapper.rasterize_to_pixels()` resolves `_RasterizeToPixels` as a module
global at call time, so rebinding that one name is enough to redirect every rasterizer
call in the process -- including the ones `gsplat.rendering.rasterization()` makes
internally. Nothing in gsplat is edited on disk.

The forward is left on the HIP kernel: it is ~0.5 ms/step against the backward's
~11.7 ms, so there is nothing to win there and plenty of `last_ids`/`render_alphas`
semantics to get wrong.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from . import _core


class _TriRasterizeToPixels(torch.autograd.Function):
    """gsplat's `_RasterizeToPixels` with a Triton backward.

    Named distinctly so it is trivially greppable in a `torch.profiler` table: the
    backward node shows up as `_TriRasterizeToPixelsBackward`, next to (not merged
    with) the HIP `rasterize_to_pixels_3dgs_bwd_kernel` it replaces.
    """

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        conics: Tensor,
        colors: Tensor,
        opacities: Tensor,
        backgrounds: Tensor,
        masks: Tensor,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,
        flatten_ids: Tensor,
        absgrad: bool,
    ) -> Tuple[Tensor, Tensor]:
        from gsplat.cuda._wrapper import _make_lazy_cuda_func

        render_colors, render_alphas, last_ids = _make_lazy_cuda_func(
            "rasterize_to_pixels_3dgs_fwd"
        )(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad

        render_alphas = render_alphas.float()
        return render_colors, render_alphas

    @staticmethod
    def backward(ctx, v_render_colors: Tensor, v_render_alphas: Tensor):
        (
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors

        args = (
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            ctx.width,
            ctx.height,
            ctx.tile_size,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            ctx.absgrad,
        )

        if _core.supports(colors, ctx.tile_size, means2d):
            bwd = _core.rasterize_to_pixels_3dgs_bwd
        else:
            from gsplat.cuda._wrapper import _make_lazy_cuda_func

            bwd = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")

        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
        ) = bwd(*args)

        if ctx.absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[4]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(-3, -2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


_ORIGINAL: Optional[type] = None


def install() -> None:
    """Point gsplat's rasterizer at the Triton backward. Idempotent."""
    global _ORIGINAL
    if not _core._HAS_TRITON:
        raise RuntimeError(
            "triraster.install() needs Triton, but `import triton` failed. "
            "Use the HIP backward instead (--ras_bwd baseline)."
        )
    from gsplat.cuda import _wrapper

    if _ORIGINAL is None:
        _ORIGINAL = _wrapper._RasterizeToPixels
    _wrapper._RasterizeToPixels = _TriRasterizeToPixels


def uninstall() -> None:
    """Restore gsplat's stock HIP-backward autograd Function."""
    global _ORIGINAL
    if _ORIGINAL is None:
        return
    from gsplat.cuda import _wrapper

    _wrapper._RasterizeToPixels = _ORIGINAL
    _ORIGINAL = None


def is_installed() -> bool:
    from gsplat.cuda import _wrapper

    return _wrapper._RasterizeToPixels is _TriRasterizeToPixels
