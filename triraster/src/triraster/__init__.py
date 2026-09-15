"""triraster: autotuned Triton backward for gsplat's 3DGS pixel rasterizer.

Replaces the single most expensive kernel in a 3DGS training step after the loss --
`rasterize_to_pixels_3dgs_bwd_kernel` -- while leaving the HIP forward, and every other
gsplat kernel, untouched.

Usage:

    import triraster
    triraster.install()      # gsplat's rasterizer now backprops through Triton
    ...
    triraster.uninstall()    # back to the stock HIP kernel
"""
from ._core import (
    _HAS_TRITON,
    bench_configs,
    configs,
    rasterize_to_pixels_3dgs_bwd,
    run_config,
    supports,
)
from ._patch import _TriRasterizeToPixels, install, is_installed, uninstall

__all__ = [
    "install",
    "uninstall",
    "is_installed",
    "supports",
    "bench_configs",
    "configs",
    "run_config",
    "rasterize_to_pixels_3dgs_bwd",
    "_TriRasterizeToPixels",
    "_HAS_TRITON",
]
__version__ = "0.1.0"
