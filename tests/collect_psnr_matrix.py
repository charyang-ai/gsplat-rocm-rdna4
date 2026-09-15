#!/usr/bin/env python3
"""Collect the PSNR matrix into the numbers Table 4 of the paper needs.

The table's argument is not "which arm scored higher". It is that the spread
*within* nominally identical repeats is larger than the spread *between* arms,
which is what makes an end-to-end PSNR comparison uninformative at this sample
size. So this prints the within-arm spread first and the between-arm difference
second, and refuses to declare a winner.

The three contrasts the paper tests -- the backward substitution at each tile
size, and the tile size itself pooled over backends -- are computed here rather
than by hand, against the pooled within-arm standard deviation and with a
Bonferroni threshold for the three tests. The tile-size contrast is the one
that matters: if it survives correction, tile 16 is a speed/quality trade and
not a free win.

Usage:
    python tests/collect_psnr_matrix.py results/psnr_matrix
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics
import sys

ALPHA = 0.05


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    tiny, eps = 1e-30, 3e-14
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (tiny if abs(d) < tiny else d)
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = tiny if abs(d) < tiny else d
            c = 1.0 + aa / c
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_pvalue(t: float, dof: int) -> float:
    """Two-sided p-value for Student's t. scipy is not available in the image."""
    if dof <= 0:
        return float("nan")
    return _betai(0.5 * dof, 0.5, dof / (dof + t * t))


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
    ns: dict[tuple[str, str], int] = {}
    gs_means: dict[tuple[str, str], float] = {}
    ss: list[float] = []          # within-arm sums of squared deviations
    dof = 0
    spreads: list[float] = []
    for key in sorted(arms):
        rows = sorted(arms[key])
        for rep, psnr, n_gs in rows:
            print(f"{'tile'+key[0]+' '+key[1]:<18} {rep:>3} {psnr:>10.4f} {n_gs:>12,}")
        vals = [r[1] for r in rows]
        means[key] = statistics.fmean(vals)
        ns[key] = len(vals)
        gs_means[key] = statistics.fmean([r[2] for r in rows if r[2] >= 0] or [-1])
        if len(vals) > 1:
            spread = max(vals) - min(vals)
            spreads.append(spread)
            ss.append(statistics.variance(vals) * (len(vals) - 1))
            dof += len(vals) - 1
            print(f"{'':<18} {'':>3} {'mean':>10} {means[key]:>11.4f} dB")
            print(f"{'':<18} {'':>3} {'spread':>10} {spread:>11.4f} dB")
            print(f"{'':<18} {'':>3} {'sd':>10} {statistics.stdev(vals):>11.4f} dB")
        print()

    print("=" * 46)
    if spreads:
        print(f"noise floor (max within-arm spread): {max(spreads):.4f} dB")
        print(f"                       mean spread : {statistics.fmean(spreads):.4f} dB")
    if dof == 0:
        print("\nonly one repeat per arm: no noise floor, no contrasts.")
        return

    sigma = math.sqrt(sum(ss) / dof)
    n_min = min(ns.values())
    print(f"pooled within-arm sd  sigma = {sigma:.4f} dB  ({dof} dof, "
          f"n = {n_min} per arm)")

    # The three contrasts the paper tests, all against the pooled sigma.
    contrasts = []
    for tile in ("8", "16"):
        kb, kt = (tile, "baseline"), (tile, "triton")
        if kb in means and kt in means:
            se = sigma * math.sqrt(1.0 / ns[kb] + 1.0 / ns[kt])
            contrasts.append((f"tile {tile}: HIP -> Triton", means[kt] - means[kb], se))
    t8 = [means[k] for k in means if k[0] == "8"]
    t16 = [means[k] for k in means if k[0] == "16"]
    if len(t8) == 2 and len(t16) == 2:
        n8 = sum(ns[k] for k in ns if k[0] == "8")
        n16 = sum(ns[k] for k in ns if k[0] == "16")
        se = sigma * math.sqrt(1.0 / n8 + 1.0 / n16)
        contrasts.append(("tile 8 -> tile 16 (pooled)",
                          statistics.fmean(t16) - statistics.fmean(t8), se))

    thresh = ALPHA / len(contrasts)
    print(f"\ncontrasts  (Bonferroni for {len(contrasts)} tests: "
          f"significant below p = {thresh:.3f})")
    print(f"  {'contrast':<28} {'diff (dB)':>10} {'t':>7} {'p':>8}   verdict")
    for name, diff, se in contrasts:
        t = diff / se if se else float("nan")
        p = t_pvalue(t, dof)
        if p < thresh:
            verdict = "SIGNIFICANT (survives correction)"
        elif p < ALPHA:
            verdict = "suggestive, does not survive correction"
        else:
            verdict = "not resolvable against the noise floor"
        print(f"  {name:<28} {diff:>+10.4f} {t:>7.2f} {p:>8.3f}   {verdict}")

    se_arm = sigma / math.sqrt(n_min)
    print(f"\nstandard error of one arm mean: {se_arm:.4f} dB")
    print(f"a 0.25 dB effect is {0.25 / se_arm:.1f} arm-SE wide at n = {n_min}; "
          f"n = {n_min * 4} would make it {0.25 / (sigma / math.sqrt(n_min * 4)):.1f}.")

    # Densification is the only channel a gradient perturbation could act
    # through, so a quality difference with matched primitive counts has no
    # mechanism available to it.
    if all(v > 0 for v in gs_means.values()):
        g8 = statistics.fmean([gs_means[k] for k in gs_means if k[0] == "8"])
        g16 = statistics.fmean([gs_means[k] for k in gs_means if k[0] == "16"])
        print(f"\nmean final primitives: tile 8 {g8:,.0f}   tile 16 {g16:,.0f}   "
              f"({abs(g16 - g8) / g8 * 100:.3f}% apart)")

    print("\nFor the paper: quote the noise floor as a measured quantity, and")
    print("report between-arm differences only relative to it.")


if __name__ == "__main__":
    main()
