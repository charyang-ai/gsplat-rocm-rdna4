#!/usr/bin/env python3
"""Run gsplat's `examples/simple_trainer.py` with the rasterizer tile size forced.

`simple_trainer.py` has no `--tile_size` flag and never passes one, so it silently
takes whatever `rasterization()` defaults to -- 8 in this ROCm fork, 16 upstream.
Comparing the two therefore means either editing gsplat or wrapping it, and editing
is not an option when two runs share one `/opt/gsplat` on different GPUs. Hence the
wrapper: every setting comes from this process's environment, so concurrent runs
cannot interfere.

Usage:
    GSPLAT_TILE_SIZE=16 python tests/run_simple_trainer.py default --data_dir ... ...

Environment:
    GSPLAT_TILE_SIZE   required, the tile size to force
    GSPLAT_RAS_BWD     optional, `triton` installs the triraster backward first
    GSPLAT_TRAINER     optional, path to simple_trainer.py
                       (default /opt/gsplat/examples/simple_trainer.py)
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.environ.get("GSPLAT_TRAINER", "/opt/gsplat/examples/simple_trainer.py")


def main() -> None:
    try:
        tile_size = int(os.environ["GSPLAT_TILE_SIZE"])
    except KeyError:
        sys.exit("GSPLAT_TILE_SIZE is not set; refusing to run with an implicit "
                 "tile size, since that is the variable under test")
    if not os.path.isfile(TRAINER):
        sys.exit(f"no trainer at {TRAINER}; set GSPLAT_TRAINER")

    import gsplat
    import gsplat.rendering

    original = gsplat.rendering.rasterization

    def rasterization(*args, **kwargs):
        kwargs["tile_size"] = tile_size
        return original(*args, **kwargs)

    # Patched before the trainer is imported, so its `from gsplat... import
    # rasterization` binds the wrapper rather than the original.
    gsplat.rendering.rasterization = rasterization
    gsplat.rasterization = rasterization

    ras_bwd = os.environ.get("GSPLAT_RAS_BWD", "baseline")
    if ras_bwd == "triton":
        # Prefer the working tree over any pip-installed copy, exactly as the tests do.
        # An installed triraster is whatever was baked into the image at build time, so
        # without this a run could silently benchmark a stale kernel.
        src = os.path.join(os.path.dirname(_HERE), "triraster", "src")
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)
        import triraster

        triraster.install()
        print(f"[run_simple_trainer] triraster from {os.path.dirname(triraster.__file__)}",
              flush=True)
    elif ras_bwd != "baseline":
        sys.exit(f"GSPLAT_RAS_BWD={ras_bwd!r}, expected 'baseline' or 'triton'")

    print(f"[run_simple_trainer] tile_size={tile_size}  ras_bwd={ras_bwd}  "
          f"trainer={TRAINER}", flush=True)

    # Running the trainer directly would put its own directory on sys.path[0]; under
    # runpy that slot is this wrapper's directory instead, and the trainer's sibling
    # packages (`datasets`, `utils`) stop resolving.
    sys.path.insert(0, os.path.dirname(os.path.abspath(TRAINER)))
    sys.argv = [TRAINER] + sys.argv[1:]
    runpy.run_path(TRAINER, run_name="__main__")


if __name__ == "__main__":
    main()
