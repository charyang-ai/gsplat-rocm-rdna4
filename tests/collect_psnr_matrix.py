#!/usr/bin/env python3
"""Collect the PSNR matrix into the numbers Table 4 of the paper needs.

The table's argument is not "which arm scored higher". It is that the spread
*within* nominally identical repeats is larger than the spread *between* arms,
which is what makes an end-to-end PSNR comparison uninformative at this sample
size. So this prints the within-arm spread first and the between-arm difference
second, and refuses to declare a winner.

Usage:
    python tests/collect_psnr_matrix.py results/psnr_matrix
"""

from __future__ import annotations

import glob
import json
import os
import re
import statistics
import sys


def _final_stats(run_dir: str) -> tuple[float, int] | None:
    """PSNR and Gaussian count at the last validation checkpoint of one run."""
    files = sorted(glob.glob(os.path.join(run_dir, "stats", "val_step*.json")))
    if not files:
        return None
    # Sort by step, not lexically: val_step6999 must come before val_step29999.
    files.sort(key=lambda f: int(re.search(r"val_step(\d+)", f).group(1)))
    with open(files[-1]) as fh:
        d = json.load(fh)
    psnr = d.get("psnr")
    n_gs = d.get("num_GS", d.get("num_gs"))
    if psnr is None:
        return None
    return float(psnr), int(n_gs) if n_gs is not None else -1


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "results/psnr_matrix"
    arms: dict[tuple[str, str], list[tuple[int, float, int]]] = {}

    for d in sorted(glob.glob(os.path.join(root, "tile*_*_r*"))):
        m = re.search(r"tile(\d+)_(\w+?)_r(\d+)$", os.path.basename(d))
        if not m:
            continue
        tile, bwd, rep = m.group(1), m.group(2), int(m.group(3))
        got = _final_stats(d)
        if got is None:
            print(f"  (no stats yet: {os.path.basename(d)})")
            continue
        arms.setdefault((tile, bwd), []).append((rep, got[0], got[1]))

    if not arms:
        sys.exit(f"no completed runs under {root}")

    print(f"\n{'arm':<18} {'rep':>3} {'PSNR (dB)':>10} {'#Gaussians':>12}")
    print("-" * 46)
    means: dict[tuple[str, str], float] = {}
    spreads: list[float] = []
    for key in sorted(arms):
        rows = sorted(arms[key])
        for rep, psnr, n_gs in rows:
            print(f"{'tile'+key[0]+' '+key[1]:<18} {rep:>3} {psnr:>10.4f} {n_gs:>12,}")
        vals = [r[1] for r in rows]
        means[key] = statistics.fmean(vals)
        if len(vals) > 1:
            spread = max(vals) - min(vals)
            spreads.append(spread)
            print(f"{'':<18} {'':>3} {'spread':>10} {spread:>11.4f} dB")
        print()

    print("=" * 46)
    if spreads:
        print(f"noise floor (max within-arm spread): {max(spreads):.4f} dB")
        print(f"                       mean spread : {statistics.fmean(spreads):.4f} dB")

    # Between-arm differences, reported only against the noise floor.
    print("\nbetween-arm differences (baseline -> triton):")
    for tile in ("8", "16"):
        b, t = means.get((tile, "baseline")), means.get((tile, "triton"))
        if b is None or t is None:
            continue
        diff = t - b
        verdict = "within noise" if spreads and abs(diff) < max(spreads) else "EXCEEDS noise floor"
        print(f"  tile {tile:<3} {diff:+.4f} dB   [{verdict}]")

    print("\nFor the paper: quote the noise floor as a measured quantity, and")
    print("report between-arm differences only relative to it.")


if __name__ == "__main__":
    main()
