"""Collect the end-to-end table for the tile-intersection work.

Runs `profile_trainer.py` once per (tile_size, --isect) cell, holding everything else
fixed, and reports total self GPU time per step plus the per-kernel rows that belong to
the intersection stage. Everything outside that stage is invariant across the cells by
construction, so whatever moves in the total is attributable to the stage -- the same
controlled-experiment argument the `--ssim` and `--ras_bwd` flags are built on.

The stage is spread over several kernels with names that differ between the baseline and
the Triton path (`intersect_tile_kernel` and a rocprim trampoline vs `_counts_kernel`,
`_emit_kernel` and a narrower rocprim sort), so a single grep would not be comparable.
`_STAGE_PATTERNS` enumerates both sides.

One caveat on the breakdown, which is why the step total is the headline and not it: every
rocprim call in the table collapses to the same truncated name, so the `sort` column also
absorbs the step's other scans (cumsum, bincount, parts of the optimizer). That still
compares fairly *between* cells, since those other scans are identical across them, but it
is not a pure stage cost. `isect_correctness_test.py --trainer-bench` measures the stage
directly on the tensors the trainer hands it; the two agree to within a few percent.

Usage:
  python tests/isect_collect_data.py                       # the paper's matrix
  python tests/isect_collect_data.py --tile-size 16        # one tile size
  python tests/isect_collect_data.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, os.pardir)

# Rows that make up the intersection stage, per implementation. Matched against the
# profiler table's truncated `Name` column, so the patterns are prefixes.
_STAGE_PATTERNS = {
    "isect emit": (r"gsplat::intersect_tile_kernel", r"_counts_kernel", r"_emit_kernel",
                   r"_rows_kernel", r"_spans_kernel", r"_emit_rows_kernel"),
    "isect sort": (r"rocprim", r"radix", r"DeviceRadixSort"),
    "isect offsets": (r"gsplat::intersect_offset_kernel", r"searchsorted"),
}


def _parse_table(text: str):
    """`Self CUDA time total` and the per-row self-CUDA times from a profiler table."""
    total = None
    m = re.search(r"Self CUDA time total:\s*([0-9.]+)\s*([mun]?s)", text)
    if m:
        total = float(m.group(1)) * {"s": 1e3, "ms": 1.0, "us": 1e-3}[m.group(2)]

    # Columns are: Name, Self CPU %, Self CPU, CPU total %, CPU total, CPU time avg,
    # Self CUDA, Self CUDA %, CUDA total, CUDA time avg, # of Calls -- so the number we
    # want is field 6. Long kernel names are truncated with a trailing "...", which is
    # why the patterns in _STAGE_PATTERNS match prefixes.
    rows = {}
    for line in text.splitlines():
        fields = re.split(r"\s{2,}", line.strip())
        if len(fields) < 11:
            continue
        vm = re.fullmatch(r"([0-9.]+)(ms|us|s)", fields[6])
        if not vm:
            continue
        ms = float(vm.group(1)) * {"s": 1e3, "ms": 1.0, "us": 1e-3}[vm.group(2)]
        rows[fields[0]] = rows.get(fields[0], 0.0) + ms
    return total, rows


def _stage_breakdown(rows):
    out = {}
    for label, pats in _STAGE_PATTERNS.items():
        acc = 0.0
        for name, ms in rows.items():
            if any(re.search(p, name) for p in pats):
                acc += ms
        out[label] = acc
    return out


def run_cell(args, tile_size: int, isect: str):
    cmd = [
        os.path.join(_REPO, ".venv", "bin", "python"),
        os.path.join(_HERE, "profile_trainer.py"),
        "--num-gaussians", str(args.num_gaussians),
        "--width", str(args.width), "--height", str(args.height),
        "--sh-degree", str(args.sh_degree),
        "--iters", str(args.iters), "--warmup", str(args.warmup),
        "--tile-size", str(tile_size),
        "--ras_bwd", args.ras_bwd,
        "--ssim", args.ssim,
        "--isect", isect,
        "--row-limit", "60",
    ]
    env = dict(os.environ, PYTHONPATH="/tmp/pyshim")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
        raise SystemExit(f"cell tile={tile_size} isect={isect} failed")
    total, rows = _parse_table(p.stdout)
    return total, rows, p.stdout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-gaussians", type=int, default=500_000)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--ras_bwd", default="triton")
    p.add_argument("--ssim", default="trissim")
    p.add_argument("--tile-size", type=int, action="append", default=None)
    p.add_argument("--isect", action="append", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--logdir", default=None)
    args = p.parse_args()

    tiles = args.tile_size or [8, 16]
    isects = args.isect or ["baseline", "emit", "triton"]

    print(f"gaussians={args.num_gaussians} image={args.width}x{args.height} "
          f"sh={args.sh_degree} iters={args.iters} ras_bwd={args.ras_bwd} "
          f"ssim={args.ssim}\n")
    hdr = (f"{'tile':>5}{'isect':>10}{'total/step':>12}{'stage*':>9}"
           f"{'emit':>9}{'sort*':>9}{'offsets':>9}{'step spd':>10}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    base_total = {}
    for tile in tiles:
        for isect in isects:
            total, rows, log = run_cell(args, tile, isect)
            per_step = total / args.iters
            br = _stage_breakdown(rows)
            br = {k: v / args.iters for k, v in br.items()}
            stage = sum(br.values())
            if isect == isects[0]:
                base_total[tile] = per_step
            spd = base_total[tile] / per_step
            results[f"tile{tile}_{isect}"] = dict(
                total_ms_per_step=per_step, stage=br, stage_total=stage, speedup=spd)
            print(f"{tile:>5}{isect:>10}{per_step:>11.3f}{'':1}{stage:>9.3f}"
                  f"{br['isect emit']:>9.3f}{br['isect sort']:>9.3f}"
                  f"{br['isect offsets']:>9.3f}{spd:>9.2f}x")
            if args.logdir:
                os.makedirs(args.logdir, exist_ok=True)
                with open(os.path.join(args.logdir,
                                       f"tile{tile}_{isect}.log"), "w") as f:
                    f.write(log)

    print("\n* the sort column, and so the stage total, also contains the step's other "
          "rocprim scans;\n  they are identical across cells. See the module docstring.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(config=vars(args), results=results), f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
