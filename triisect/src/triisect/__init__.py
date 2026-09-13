"""Triton tile intersection for gsplat on AMD RDNA4 (gfx1201, wave32)."""
from __future__ import annotations

from ._core import (
    isect_offset_encode,
    isect_tiles,
    reset_dispatch_cache,
    supports,
)
from ._patch import install, is_installed, presort_enabled, uninstall

__version__ = "0.1.0"

__all__ = [
    "install",
    "uninstall",
    "is_installed",
    "presort_enabled",
    "isect_tiles",
    "isect_offset_encode",
    "reset_dispatch_cache",
    "supports",
    "__version__",
]
