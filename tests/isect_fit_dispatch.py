"""Fit and compare dispatch rules for the intersection stage.

The rule shipped with `isect_tiles` is a single threshold on pairs per Gaussian, calibrated
on the synthetic sweep. The real-scene measurements say that is the wrong shape of rule: at
a fixed density the Triton path wins on a large problem and loses on a small one, because
part of what it spends is fixed per call and does not shrink with the work.

So fit the thing the decision actually needs, which is the sign of the saving:

    t_hip - t_ours  ~  b*P - a*N - c

with `P` pairs, `N` the length of the array the stage was handed, and `c` the per-call
overhead the model needs to explain the small-problem losses. Then compare rules by what
they would have cost, against an oracle that always picks the faster path.

Usage:
  python tests/isect_fit_dispatch.py results/realscene/*.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def _load(paths: list) -> list:
    """Every timed cell in the given result files, de-duplicated by operating point.

    Two schemas, because the two things worth fitting were measured by different scripts:
    `isect_realscene.py` records trained scenes, and `isect_ablations.py --sweep` records
    the synthetic density sweep. The sweep does not store the array length, but pairs per
    Gaussian and the pair count between them determine it.
    """
    rows = {}
    for path in paths:
        with open(path) as f:
            blob = json.load(f)
        for section in ("bench", "res"):
            for rec in blob.get(section, {}).values():
                if "hip" not in rec:
                    continue
                key = (rec["scene"], rec["packed"], rec["tile"], rec["n_pairs"],
                       rec["n_elements"])
                rows[key] = rec
        for key, rec in blob.get("results", {}).get("sweep", {}).items():
            if not rec.get("n_pairs"):
                continue
            n_elements = int(round(rec["n_pairs"] / rec["pairs_per_gauss"]))
            rows[("sweep", path, key)] = dict(
                scene="synthetic", packed=False,
                tile=int(key.split("tile")[1]), n_pairs=rec["n_pairs"],
                n_elements=n_elements, hip=rec["full_hip"], ours=rec["full_ours"])
    return list(rows.values())


def _arrays(rows: list):
    N = np.array([r["n_elements"] for r in rows], dtype=np.float64)
    P = np.array([r["n_pairs"] for r in rows], dtype=np.float64)
    hip = np.array([r["hip"] for r in rows])
    ours = np.array([r["ours"] for r in rows])
    saving = hip - ours
    return N, P, hip, ours, saving, saving > 0


def _score(label: str, rows: list, rules: list) -> None:
    """What each rule would have cost on this set, against an always-right oracle."""
    N, P, hip, ours, _saving, win = _arrays(rows)
    oracle = np.minimum(hip, ours)
    print(f"\n{label}: {len(rows)} cells, {int(win.sum())} of them wins for Triton")
    hdr = f"  {'rule':<34}{'wrong':>7}{'total ms':>10}{'vs HIP':>8}{'of oracle':>11}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, predicate in rules:
        use = predicate(N, P)
        got = np.where(use, ours, hip)
        print(f"  {name:<34}{int((use != win).sum()):>7}{got.sum():>10.1f}"
              f"{hip.sum() / got.sum():>7.2f}x{100 * oracle.sum() / got.sum():>10.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="+", help="files the rule is chosen on")
    ap.add_argument("--holdout", nargs="*", default=[],
                    help="files scored but never used to choose anything")
    args = ap.parse_args()

    rows = _load(args.json)
    N, P, hip, ours, saving, win = _arrays(rows)

    A = np.stack([P, -N, -np.ones_like(P)], axis=1)
    (b, a, c), *_ = np.linalg.lstsq(A, saving, rcond=None)
    pred = A @ np.array([b, a, c])

    print(f"{len(rows)} timed cells, {win.sum()} of them wins for the Triton path\n")
    print("least-squares cost model for the saving:")
    print(f"  b = {b * 1e6:8.4f} us per million pairs   (what narrowing the sort buys)")
    print(f"  a = {a * 1e6:8.4f} us per million elements (count, prefix, depth sort)")
    print(f"  c = {c:8.4f} ms fixed per call             (launches and the sync)")
    print(f"  break-even density P/N = a/b + c/(b*N) = {a / b:.2f} + "
          f"{c / b / 1e6:.2f}M/N\n")
    print(f"  sign agreement {100 * (np.sign(pred) == np.sign(saving)).mean():.1f}%, "
          f"median |error| {np.median(np.abs(pred - saving)):.3f} ms")

    rules = [
        ("always HIP", lambda n, p: np.zeros(len(p), dtype=bool)),
        ("always Triton", lambda n, p: np.ones(len(p), dtype=bool)),
        ("shipped: P/N >= 6", lambda n, p: p / n >= 6.0),
        ("P/N >= 1.5 and P >= 3M", lambda n, p: (p / n >= 1.5) & (p >= 3e6)),
        ("cost model: b*P - a*N - c > 0",
         lambda n, p: b * p - a * n - c > 0),
    ]
    _score("chosen on", rows, rules)
    if args.holdout:
        _score("held out", _load(args.holdout), rules)

    oracle = np.minimum(hip, ours)
    print("\nhow sharp the two-term rule's constants are (percent of oracle):")
    print("  P >= ", end="")
    p_grid = [1e6, 2e6, 3e6, 4e6, 5e6, 6e6]
    d_grid = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    print("".join(f"{p / 1e6:>7.0f}M" for p in p_grid))
    for d in d_grid:
        line = f"  P/N >= {d:<4.2f}"
        for p in p_grid:
            use = (P / N >= d) & (P >= p)
            got = np.where(use, ours, hip)
            line += f"{100 * oracle.sum() / got.sum():>8.1f}"
        print(line)

    print("\ncells the shipped rule gets wrong, worst first:")
    shipped = P / N >= 6.0
    bad = np.where(shipped != win)[0]
    bad = bad[np.argsort(-np.abs(saving[bad]))]
    print(f"  {'scene':<9}{'layout':<8}{'tile':>5}{'P/N':>7}{'n_pairs':>11}"
          f"{'HIP':>8}{'ours':>8}{'lost ms':>9}")
    for i in bad[:12]:
        r = rows[i]
        print(f"  {r['scene']:<9}{'packed' if r['packed'] else 'dense':<8}"
              f"{r['tile']:>5}{P[i] / N[i]:>7.2f}{int(P[i]):>11}"
              f"{hip[i]:>8.3f}{ours[i]:>8.3f}{abs(saving[i]):>9.3f}")


if __name__ == "__main__":
    main()
