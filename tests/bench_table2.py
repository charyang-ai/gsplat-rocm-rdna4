"""Benchmark Table 2 (tab:profiler_comparison): baseline vs TriSSIM full training step.

Matches trissim.tex: 500K Gaussians, 1920x1080, SH degree 3, tile 8 (fork default),
10 warmup + 50 profiled iterations. Five paired runs (same seed, baseline then TriSSIM).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.autograd.profiler_util import DeviceType
from torch.profiler import ProfilerActivity, profile, schedule

sys.path.insert(0, "/home/charyang/gsplat-rocm-rdna4/tests")
from profile_trainer import (  # noqa: E402
    _import_rasterization,
    _make_camera,
    _make_scene,
    _select_ras_bwd,
    _select_ssim,
)


@dataclass
class ProfileRow:
    miopen_dw: float = 0.0
    conv_bwd: float = 0.0
    fused_fwd: float = 0.0
    fused_bwd: float = 0.0
    ssim_total: float = 0.0
    ras_bwd: float = 0.0
    intersect: float = 0.0
    projection: float = 0.0
    sh_bwd: float = 0.0
    adam: float = 0.0
    total: float = 0.0

    def share(self, field: str) -> float:
        val = getattr(self, field)
        return 100.0 * val / self.total if self.total else 0.0


def _self_us(evt) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        val = getattr(evt, attr, 0) or 0
        if val:
            return float(val)
    return 0.0


def _footer_total_ms(events) -> float:
    total_us = 0.0
    for evt in events:
        if evt.is_user_annotation:
            continue
        if evt.device_type in (
            DeviceType.CUDA,
            DeviceType.PrivateUse1,
            DeviceType.MTIA,
        ):
            total_us += evt.self_device_time_total
        elif evt.device_type == DeviceType.CPU and evt.is_legacy:
            total_us += evt.self_device_time_total
    return total_us / 1000.0


def _sum_ms(events, pred: Callable[[str], bool]) -> float:
    return sum(_self_us(e) for e in events if pred(e.key)) / 1000.0


def _collect_profile(prof, ssim_mode: str) -> ProfileRow:
    events = prof.key_averages()
    row = ProfileRow(
        miopen_dw=_sum_ms(events, lambda k: "miopen_depthwise_convolution" in k),
        conv_bwd=_sum_ms(events, lambda k: "aten::convolution_backward" in k),
        fused_fwd=_sum_ms(events, lambda k: k == "_FusedBlur5TBwd"),
        fused_bwd=_sum_ms(events, lambda k: k == "_FusedBlur5TBwdBackward"),
        ras_bwd=_sum_ms(events, lambda k: "rasterize_to_pixels_3dgs_bwd_kernel" in k),
        intersect=_sum_ms(events, lambda k: "intersect_tile_kernel" in k),
        projection=_sum_ms(events, lambda k: "projection_ewa_3dgs_packed_bwd_kernel" in k),
        sh_bwd=_sum_ms(events, lambda k: "spherical_harmonics_bwd_kernel" in k),
        adam=_sum_ms(events, lambda k: "Optimizer.step#Adam.step" in k),
        total=_footer_total_ms(events),
    )
    if ssim_mode == "baseline":
        row.ssim_total = row.miopen_dw + row.conv_bwd
    else:
        row.ssim_total = row.fused_fwd + row.fused_bwd
    return row


def _profile_once(
    *,
    ssim_mode: str,
    seed: int,
    num_gaussians: int,
    width: int,
    height: int,
    sh_degree: int,
    warmup: int,
    iters: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ProfileRow:
    ssim_fn, _ = _select_ssim(ssim_mode)
    _select_ras_bwd("baseline")
    rasterization = _import_rasterization()

    torch.manual_seed(seed)
    means, quats, scales, opacities, colors = _make_scene(
        num_gaussians, sh_degree, device, dtype)
    viewmats, Ks = _make_camera(width, height, device, dtype)
    target = torch.rand(1, 3, height, width, device=device, dtype=dtype)
    opt = torch.optim.Adam([means, quats, scales, opacities, colors], lr=1e-3)
    sh = sh_degree if sh_degree > 0 else None

    def train_step():
        opt.zero_grad(set_to_none=True)
        renders, _a, _m = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=torch.sigmoid(opacities), colors=colors,
            viewmats=viewmats, Ks=Ks, width=width, height=height,
            sh_degree=sh, tile_size=8,
        )
        img = renders[0].permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
        loss = (img - target).abs().mean()
        if ssim_fn is not None:
            loss = loss + (1.0 - ssim_fn(img, target, padding="valid"))
        loss.backward()
        opt.step()

    for _ in range(warmup):
        train_step()
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    sched = schedule(wait=0, warmup=1, active=iters, repeat=1)
    with profile(activities=activities, schedule=sched, record_shapes=False,
                 profile_memory=False, with_stack=False) as prof:
        for _ in range(iters + 1):
            train_step()
            prof.step()
    torch.cuda.synchronize()
    return _collect_profile(prof, ssim_mode)


def _stats(vals: List[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(vals)
    if len(vals) > 1:
        std = statistics.stdev(vals)
        var = statistics.variance(vals)
    else:
        std = var = 0.0
    return mean, std, var


def _fmt_pm(mean: float, std: float) -> str:
    return f"${mean:.1f} \\pm {std:.1f}$"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--num-gaussians", type=int, default=500_000)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP required", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    dtype = torch.float32
    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"Table 2: {args.num_gaussians} gaussians, {args.width}x{args.height}, "
          f"sh={args.sh_degree}, warmup={args.warmup}, iters={args.iters}, "
          f"runs={args.runs}", flush=True)

    # Global warmup (compile / autotune both SSIM paths once).
    print("Global warmup ...", flush=True)
    _profile_once(ssim_mode="baseline", seed=9999, num_gaussians=args.num_gaussians,
                  width=args.width, height=args.height, sh_degree=args.sh_degree,
                  warmup=args.warmup, iters=1, device=device, dtype=dtype)
    _profile_once(ssim_mode="trissim", seed=9999, num_gaussians=args.num_gaussians,
                  width=args.width, height=args.height, sh_degree=args.sh_degree,
                  warmup=1, iters=1, device=device, dtype=dtype)

    paired: List[dict] = []
    for i in range(args.runs):
        seed = i
        print(f"\n=== run {i + 1}/{args.runs} (seed={seed}) ===", flush=True)
        base = _profile_once(
            ssim_mode="baseline", seed=seed, num_gaussians=args.num_gaussians,
            width=args.width, height=args.height, sh_degree=args.sh_degree,
            warmup=args.warmup, iters=args.iters, device=device, dtype=dtype)
        tri = _profile_once(
            ssim_mode="trissim", seed=seed, num_gaussians=args.num_gaussians,
            width=args.width, height=args.height, sh_degree=args.sh_degree,
            warmup=args.warmup, iters=args.iters, device=device, dtype=dtype)
        print(f"  baseline total={base.total:.1f} ms  ssim={base.ssim_total:.1f} ms",
              flush=True)
        print(f"  trissim  total={tri.total:.1f} ms  ssim={tri.ssim_total:.1f} ms",
              flush=True)
        paired.append({"seed": seed, "baseline": asdict(base), "trissim": asdict(tri)})

    fields = [
        "miopen_dw", "conv_bwd", "fused_fwd", "fused_bwd", "ssim_total",
        "ras_bwd", "intersect", "projection", "sh_bwd", "adam", "total",
    ]
    summary: Dict[str, Dict[str, dict]] = {"baseline": {}, "trissim": {}}
    for mode in ("baseline", "trissim"):
        for field in fields:
            vals = [p[mode][field] for p in paired]
            m, s, v = _stats(vals)
            summary[mode][field] = {
                "mean_ms": round(m, 3),
                "std_ms": round(s, 3),
                "var_ms2": round(v, 3),
            }

    b = summary["baseline"]
    t = summary["trissim"]
    bm, bs = b["ssim_total"]["mean_ms"], b["ssim_total"]["std_ms"]
    tm, ts = t["ssim_total"]["mean_ms"], t["ssim_total"]["std_ms"]
    ssim_speedup = bm / tm if tm else 0.0
    ssim_reduction = (1.0 - tm / bm) * 100.0 if bm else 0.0
    total_speedup = b["total"]["mean_ms"] / t["total"]["mean_ms"]
    total_reduction = (1.0 - t["total"]["mean_ms"] / b["total"]["mean_ms"]) * 100.0

    print("\n" + "=" * 72, flush=True)
    print(f"MEAN ± STD over {args.runs} runs:", flush=True)
    print(f"  SSIM total:  baseline {bm:.1f}±{b['ssim_total']['std_ms']:.1f} ms  "
          f"trissim {tm:.1f}±{ts:.1f} ms  speedup {ssim_speedup:.2f}x", flush=True)
    print(f"  Step total:  baseline {b['total']['mean_ms']:.1f} ms  "
          f"trissim {t['total']['mean_ms']:.1f} ms  speedup {total_speedup:.2f}x",
          flush=True)

    payload = {
        "methodology": {
            "num_gaussians": args.num_gaussians,
            "resolution": f"{args.width}x{args.height}",
            "sh_degree": args.sh_degree,
            "tile_size": 8,
            "warmup": args.warmup,
            "iters": args.iters,
            "runs": args.runs,
            "gpu": torch.cuda.get_device_name(device),
            "date": date.today().isoformat(),
        },
        "runs": paired,
        "mean": summary,
        "derived": {
            "ssim_speedup": round(ssim_speedup, 2),
            "ssim_reduction_pct": round(ssim_reduction, 1),
            "total_speedup": round(total_speedup, 2),
            "total_reduction_pct": round(total_reduction, 1),
        },
    }

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out}", flush=True)

    def row_time(mode: str, field: str) -> tuple[float, float]:
        return summary[mode][field]["mean_ms"], summary[mode][field]["std_ms"]

    print("\nLaTeX snippet (tab:profiler_comparison, times as mean±std ms):", flush=True)
    bm, bs = row_time("baseline", "miopen_dw")
    print(f"miopen & {_fmt_pm(bm, bs)} & ... \\\\")
    bm, bs = row_time("baseline", "conv_bwd")
    print(f"conv_bwd & {_fmt_pm(bm, bs)} & ... \\\\")
    tm, ts = row_time("trissim", "fused_fwd")
    print(f"FusedBlur fwd & ... & {_fmt_pm(tm, ts)} & ... \\\\")
    tm, ts = row_time("trissim", "fused_bwd")
    print(f"FusedBlur bwd & ... & {_fmt_pm(tm, ts)} & ... \\\\")
    bm, bs = row_time("baseline", "ssim_total")
    tm, ts = row_time("trissim", "ssim_total")
    print(f"SSIM total & \\textbf{{{_fmt_pm(bm, bs)}}} & ... & "
          f"\\textbf{{{_fmt_pm(tm, ts)}}} & ... & "
          f"\\textbf{{{ssim_speedup:.2f}}} & \\textbf{{{ssim_reduction:.1f}}} \\\\")


if __name__ == "__main__":
    main()
