"""Run ssim_bench_offline timing 5 times and report mean ± std for Table 1.

Uses the same methodology as the paper: 100 timed iterations after 20 warmup,
HIP events, 1×3×1080×1920 fp32 valid padding. Each repetition uses a distinct
seed so input tensors differ across runs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from datetime import date
from pathlib import Path

import torch

# Load ssim_bench_offline from the same directory (works in repo and /opt/gsplat).
_OFFLINE = Path(__file__).resolve().parent / "ssim_bench_offline.py"
_spec = importlib.util.spec_from_file_location("ssim_bench_offline", _OFFLINE)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

IMPLS = _mod.IMPLS
ORDER = _mod.ORDER
_sync = _mod._sync
_time_iters = _mod._time_iters


def _run_once(seed: int, iters: int, warmup: int, forward_only: bool) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    shape = (1, 3, 1080, 1920)
    megapixels = shape[0] * shape[2] * shape[3] / 1e6
    padding = "valid"

    gt = torch.rand(shape, device=device, dtype=dtype)

    def make_pred():
        return torch.rand(shape, device=device, dtype=dtype, requires_grad=True)

    out: dict[str, dict[str, float]] = {}
    for name, fused_ssim in ((n, IMPLS[n]) for n in ORDER):
        def fwd(_f=fused_ssim):
            with torch.no_grad():
                _f(gt, gt, padding=padding)

        def fwd_bwd(_f=fused_ssim):
            pred = make_pred()
            loss = 1.0 - _f(pred, gt, padding=padding)
            loss.backward()

        warm_fn = fwd if forward_only else fwd_bwd
        for _ in range(warmup):
            warm_fn()
        _sync(device)

        fwd_times = _time_iters(fwd, iters, device)
        out[name] = {"forward": statistics.fmean(fwd_times)}
        if not forward_only:
            bwd_times = _time_iters(fwd_bwd, iters, device)
            out[name]["forward+backward"] = statistics.fmean(bwd_times)

    out["_meta"] = {"seed": seed, "megapixels": megapixels}
    return out


def _global_warmup(warmup: int) -> None:
    """One-shot compile / autotune before timed repetitions."""
    print("Global warmup (compile + autotune) ...", flush=True)
    _run_once(seed=9999, iters=1, warmup=warmup, forward_only=False)
    torch.cuda.synchronize()


def _aggregate(runs: list[dict], mode: str) -> dict[str, tuple[float, float, float]]:
    """Return per-impl (mean, std, variance) in milliseconds."""
    stats: dict[str, tuple[float, float, float]] = {}
    for name in ORDER:
        vals = [r[name][mode] for r in runs]
        mean = statistics.fmean(vals)
        if len(vals) > 1:
            std = statistics.stdev(vals)
            var = statistics.variance(vals)
        else:
            std = 0.0
            var = 0.0
        stats[name] = (mean, std, var)
    return stats


def _build_json_payload(
    runs: list[dict],
    *,
    repeats: int,
    iters: int,
    warmup: int,
    megapixels: float,
) -> dict:
    payload = {
        "methodology": {
            "shape": "1x3x1080x1920",
            "dtype": "float32",
            "padding": "valid",
            "iters_per_run": iters,
            "warmup_per_run": warmup,
            "repeats": repeats,
            "seeds": [r["_meta"]["seed"] for r in runs],
            "global_warmup": True,
            "gpu": torch.cuda.get_device_name(0),
            "date": date.today().isoformat(),
        },
        "runs": [
            {k: v for k, v in r.items() if k != "_meta"} for r in runs
        ],
    }
    for mode_key, mode in (
        ("forward", "forward"),
        ("forward_backward", "forward+backward"),
    ):
        agg = _aggregate(runs, mode)
        base_mean, _, _ = agg["baseline"]
        block = {}
        for name in ORDER:
            mean, std, var = agg[name]
            block[name] = {
                "mean_ms": round(mean, 3),
                "std_ms": round(std, 3),
                "var_ms2": round(var, 3),
                "mp_s": round(megapixels / (mean / 1e3), 1),
                "speedup": round(base_mean / mean, 2),
            }
        payload[mode_key] = block
    return payload


def _latex_row(label: str, mean: float, std: float, mp_s: float, speedup: float,
               bold: bool = False) -> str:
    cell = f"${mean:.2f} \\pm {std:.2f}$"
    mp = f"{mp_s:.1f}"
    sp = f"{speedup:.2f}$\\times$"
    if bold:
        return (f"    {label} & \\textbf{{{cell}}} & \\textbf{{{mp}}} & "
                f"\\textbf{{{sp}}} \\\\")
    return f"    {label:<20} & {cell} & {mp} & {sp} \\\\"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--forward-only", action="store_true")
    p.add_argument("--json-out", default=None, help="write aggregated results JSON")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP not available", file=sys.stderr)
        sys.exit(1)

    print(f"torch {torch.__version__}  hip {torch.version.hip}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"repeats={args.repeats}  iters={args.iters}  warmup={args.warmup}")
    print("-" * 80)

    _global_warmup(args.warmup)

    runs = []
    for i in range(args.repeats):
        seed = i
        print(f"Run {i + 1}/{args.repeats} (seed={seed}) ...", flush=True)
        runs.append(_run_once(seed, args.iters, args.warmup, args.forward_only))

    megapixels = runs[0]["_meta"]["megapixels"]
    modes = ["forward"] if args.forward_only else ["forward", "forward+backward"]
    labels = {
        "baseline": "Baseline",
        "separable": "Separable",
        "compiled": "Torch Compile",
        "triton": "Triton",
        "biauto": "Triton + Bi-Autotune",
    }

    for mode in modes:
        print(f"\n=== {mode} (mean ms over {args.iters} iters, "
              f"then mean ± std over {args.repeats} runs) ===")
        agg = _aggregate(runs, mode)
        base_mean, _, _ = agg["baseline"]
        for name in ORDER:
            mean, std, var = agg[name]
            mp_s = megapixels / (mean / 1e3)
            speedup = base_mean / mean
            print(f"{labels[name]:<22}  {mean:7.3f} ± {std:5.3f} ms   "
                  f"(var {var:6.3f})   {mp_s:7.1f} MP/s   {speedup:.2f}x")

    if not args.forward_only:
        payload = _build_json_payload(
            runs, repeats=args.repeats, iters=args.iters,
            warmup=args.warmup, megapixels=megapixels,
        )
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(f"\nWrote {out}", flush=True)

        print("\nLaTeX rows (tab:ssim_perf):", flush=True)
        for mode, title in (("forward", "Forward only"),
                              ("forward+backward", "Forward + backward")):
            print(f"% {title}")
            agg = _aggregate(runs, mode)
            base_mean, _, _ = agg["baseline"]
            for name in ORDER:
                mean, std, _ = agg[name]
                mp_s = megapixels / (mean / 1e3)
                speedup = base_mean / mean
                print(_latex_row(labels[name], mean, std, mp_s, speedup,
                                 bold=(name == "biauto")))


if __name__ == "__main__":
    main()
