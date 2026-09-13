"""Point gsplat's tile-intersection stage at the Triton implementation.

The rebinding target differs from `triraster`'s, and getting it wrong is a silent no-op.
`gsplat/rendering.py` imports the names eagerly:

    from .cuda._wrapper import (..., isect_offset_encode, isect_tiles, ...)

so `rasterization()` holds its own module-global references and patching
`gsplat.cuda._wrapper.isect_tiles` alone would never be seen. (`_RasterizeToPixels`, which
triraster swaps, *is* resolved inside `_wrapper` at call time -- hence the difference.)
Both modules are rebound here: `gsplat.rendering` for the renderer, `_wrapper` for
anything that calls the ops directly.

`isect_offset_encode` has to move together with `isect_tiles`, because in `presort` mode
the two agree on a key encoding the HIP offset kernel does not understand. Installing them
as a pair is what keeps that private.

Nothing in gsplat is edited on disk and the switch is live at any point in a process.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from . import _core

_ORIGINAL: Optional[Tuple[Callable, Callable, Callable]] = None
_PRESORT = False
_EXACT = False

# The exact ellipse test needs `conics`, and `isect_tiles`'s signature has no room for
# them -- gsplat's stage only ever sees the bounding boxes. They come from the projection
# call immediately upstream, so this catches them on the way past.
#
# Keyed on `means2d.data_ptr()`, which is only sound because the entry also holds a
# reference to `means2d` itself: that keeps the allocation alive, so the address cannot be
# recycled under a different tensor while the entry is live. A miss (or a shape that does
# not line up) falls back to the bounding-box path rather than guessing.
_PROJ_CACHE: dict = {}


def _projection(*args, **kwargs):
    out = _ORIGINAL[2](*args, **kwargs)
    if isinstance(out, tuple) and len(out) == 8:      # packed: [nnz, ...]
        means2d, conics = out[4], out[6]
    elif isinstance(out, tuple) and len(out) == 5:    # dense: [..., C, N, ...]
        means2d, conics = out[1], out[3]
    else:
        return out
    _PROJ_CACHE.clear()
    _PROJ_CACHE[means2d.data_ptr()] = (means2d, conics)
    return out


def _lookup_conics(means2d: Tensor) -> Optional[Tensor]:
    entry = _PROJ_CACHE.get(means2d.data_ptr())
    if entry is None:
        return None
    cached, conics = entry
    if cached.shape != means2d.shape or conics.shape[:-1] != means2d.shape[:-1]:
        return None
    return conics


@torch.no_grad()
def _isect_tiles(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    segmented: bool = False,
    packed: bool = False,
    n_images: Optional[int] = None,
    image_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """`isect_tiles` on Triton where supported, on the HIP op otherwise."""
    if _core.supports(means2d, radii, depths, tile_size, packed, segmented, image_ids):
        return _core.isect_tiles(
            means2d, radii, depths, tile_size, tile_width, tile_height,
            sort=sort, segmented=segmented, packed=packed, n_images=n_images,
            image_ids=image_ids, gaussian_ids=gaussian_ids, presort=_PRESORT,
            conics=_lookup_conics(means2d) if _EXACT else None,
        )
    assert _ORIGINAL is not None
    return _ORIGINAL[0](
        means2d, radii, depths, tile_size, tile_width, tile_height,
        sort=sort, segmented=segmented, packed=packed, n_images=n_images,
        image_ids=image_ids, gaussian_ids=gaussian_ids,
    )


@torch.no_grad()
def _isect_offset_encode(isect_ids: Tensor, n_images: int,
                         tile_width: int, tile_height: int) -> Tensor:
    """`searchsorted` for the narrow keys, the HIP kernel for the baseline ones."""
    return _core.isect_offset_encode(isect_ids, n_images, tile_width, tile_height)


def install(presort: bool = False, exact: bool = False) -> None:
    """Point gsplat's tile intersection at Triton. Idempotent.

    `presort=True` additionally walks the Gaussians in depth order so the radix sort only
    has to order the tile bits of a 32-bit key rather than 46 bits of a 64-bit one. The
    intersection *list* stays identical; only the key encoding it is sorted by changes.

    `exact=True` tests each tile against the Gaussian's ellipse rather than its bounding
    box, which makes the list *shorter*. The rendered image is still bit-identical -- the
    dropped pairs are ones the rasterizer would have skipped on its alpha threshold -- so
    this also removes work from the rasterizer, not just from this stage. It additionally
    patches the projection, to catch the conics the test needs.
    """
    global _ORIGINAL, _PRESORT, _EXACT
    if not _core._HAS_TRITON:
        raise RuntimeError(
            "triisect.install() needs Triton, but `import triton` failed. "
            "Use the HIP intersection instead (--isect baseline)."
        )
    from gsplat import rendering
    from gsplat.cuda import _wrapper

    if _ORIGINAL is None:
        _ORIGINAL = (_wrapper.isect_tiles, _wrapper.isect_offset_encode,
                     _wrapper.fully_fused_projection)
    _PRESORT = presort
    _EXACT = exact
    for mod in (_wrapper, rendering):
        mod.isect_tiles = _isect_tiles
        mod.isect_offset_encode = _isect_offset_encode
        if exact:
            mod.fully_fused_projection = _projection


def uninstall() -> None:
    """Restore gsplat's stock HIP intersection."""
    global _ORIGINAL, _PRESORT, _EXACT
    if _ORIGINAL is None:
        return
    from gsplat import rendering
    from gsplat.cuda import _wrapper

    for mod in (_wrapper, rendering):
        mod.isect_tiles = _ORIGINAL[0]
        mod.isect_offset_encode = _ORIGINAL[1]
        mod.fully_fused_projection = _ORIGINAL[2]
    _ORIGINAL = None
    _PRESORT = False
    _EXACT = False
    _PROJ_CACHE.clear()


def is_installed() -> bool:
    from gsplat import rendering

    return rendering.isect_tiles is _isect_tiles


def presort_enabled() -> bool:
    return _PRESORT
