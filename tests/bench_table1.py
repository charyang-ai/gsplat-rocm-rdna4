"""Benchmark Table 1 (tab:coupling): tile-size cross-stage coupling on RDNA4.

Matches triraster.tex: 500K Gaussians, 1920x1080, SH degree 3, TriSSIM loss,
HIP rasterizer backward, 10 warmup + 30 profiled iterations per tile size.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch
from torch.autograd.profiler_util import DeviceType
from torch.profiler import ProfilerActivity, profile, schedule

# Reuse the training harness from profile_trainer.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from profile_trainer import (  # noqa: E402
    _import_rasterization,
    _make_camera,
    _make_scene,
    _select_ras_bwd,
    _select_ssim,
)


@dataclass
class StageTimes:
    intersect_ms: float
    sort_ms: float
    forward_ms: float
    backward_ms: float
    total_ms: float

    @property
    def stages_sum_ms(self) -> float:
        return self.intersect_ms + self.sort_ms + self.forward_ms + self.backward_ms


def _us_to_ms(us: float) -> float:
    return us / 1000.0


def _self_device_us(evt) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        val = getattr(evt, attr, 0) or 0
        if val:
            return float(val)
    return 0.0


def _footer_total_ms(events) -> float:
    """Match the profiler table footer (Self CUDA/device time total)."""
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


def _collect_stage_times(prof, tile_size: int) -> StageTimes:
    """Pull accumulated self device times for the four raster-pipeline stages."""
    events = prof.key_averages()

    intersect_us = 0.0
    sort_us = 0.0
    forward_us = 0.0
    backward_us = 0.0

    for evt in events:
        name = evt.key
        self_us = _self_device_us(evt)
        if not self_us:
            continue

        if "intersect_tile_kernel" in name:
            intersect_us += self_us
        elif "rocprim::" in name and evt.count == 180:
            # Main gsplat radix sort pass (matches paper's Radix sort row).
            sort_us += self_us
        elif "rasterize_to_pixels_3dgs_fwd_kernel" in name:
            forward_us += self_us
        elif "rasterize_to_pixels_3dgs_bwd_kernel" in name:
            backward_us += self_us

    total_ms = _footer_total_ms(events)

    return StageTimes(
        intersect_ms=_us_to_ms(intersect_us),
        sort_ms=_us_to_ms(sort_us),
        forward_ms=_us_to_ms(forward_us),
        backward_ms=_us_to_ms(backward_us),
        total_ms=total_ms,
    )


def _profile_tile(
    *,
    tile_size: int,
    num_gaussians: int,
    width: int,
    height: int,
    sh_degree: int,
    warmup: int,
    iters: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> StageTimes:
    rasterization = _import_rasterization()
    ssim_fn, _ = _select_ssim("trissim")
    _select_ras_bwd("baseline")

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
            means=means,
            quats=quats,
            scales=scales,
            opacities=torch.sigmoid(opacities),
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh,
            tile_size=tile_size,
        )
        img = renders[0].permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)
        loss = (img - target).abs().mean()
        if ssim_fn is not None:
            loss = loss + (1.0 - ssim_fn(img, target, padding="valid"))
        loss.backward()
        opt.step()

    for _ in range(warmup):
        train_step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    sched = schedule(wait=0, warmup=1, active=iters, repeat=1)
    with profile(activities=activities, schedule=sched, record_shapes=False,
                 profile_memory=False, with_stack=False) as prof:
        for _ in range(iters + 1):
            train_step()
            prof.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    return _collect_stage_times(prof, tile_size)


def _mean_field(runs: List[Dict[str, StageTimes]], tile: int, field: str) -> float:
    return statistics.mean(getattr(r[tile], field) for r in runs)


def _fmt_ms(x: float, width: int = 5) -> str:
    if x >= 100:
        return f"{x:.1f}\\,ms"
    return f"\\phantom{{0}}{x:.1f}\\,ms"


def _ratio(a: float, b: float) -> str:
    if b == 0:
        return "--"
    return f"{a / b:.2f}$\\times$"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--num-gaussians", type=int, default=500_000)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if device.type != "cuda":
        print("ERROR: CUDA/HIP device required", file=sys.stderr)
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"Table 1 config: {args.num_gaussians} gaussians, "
          f"{args.width}x{args.height}, sh={args.sh_degree}, "
          f"warmup={args.warmup}, iters={args.iters}, runs={args.runs}",
          flush=True)

    all_runs: List[Dict[int, StageTimes]] = []
    for run_idx in range(args.runs):
        seed = args.seed + run_idx
        print(f"\n=== run {run_idx + 1}/{args.runs} (seed={seed}) ===", flush=True)
        row: Dict[int, StageTimes] = {}
        for tile in (8, 16):
            print(f"  tile_size={tile} ...", flush=True)
            t = _profile_tile(
                tile_size=tile,
                num_gaussians=args.num_gaussians,
                width=args.width,
                height=args.height,
                sh_degree=args.sh_degree,
                warmup=args.warmup,
                iters=args.iters,
                seed=seed,
                device=device,
                dtype=dtype,
            )
            row[tile] = t
            print(f"    int={t.intersect_ms:.1f} sort={t.sort_ms:.1f} "
                  f"fwd={t.forward_ms:.1f} bwd={t.backward_ms:.1f} "
                  f"total={t.total_ms:.1f} ms", flush=True)
        all_runs.append(row)

    tiles = (8, 16)
    summary = {}
    for tile in tiles:
        summary[tile] = {
            "intersect_ms": _mean_field(all_runs, tile, "intersect_ms"),
            "sort_ms": _mean_field(all_runs, tile, "sort_ms"),
            "forward_ms": _mean_field(all_runs, tile, "forward_ms"),
            "backward_ms": _mean_field(all_runs, tile, "backward_ms"),
            "total_ms": _mean_field(all_runs, tile, "total_ms"),
        }

    print("\n" + "=" * 72, flush=True)
    print(f"MEAN over {args.runs} runs:", flush=True)
    labels = [
        ("intersect_ms", "Tile intersection"),
        ("sort_ms", "Radix sort"),
        ("forward_ms", "Rasterization forward"),
        ("backward_ms", "Rasterization backward"),
        ("total_ms", "Total measured GPU time"),
    ]
    for key, label in labels:
        v8 = summary[8][key]
        v16 = summary[16][key]
        ratio = v8 / v16 if v16 else float("inf")
        print(f"  {label:28s}  tau=8: {v8:7.1f} ms   tau=16: {v16:7.1f} ms   "
              f"ratio: {ratio:.2f}x", flush=True)

    print("\nLaTeX rows (paste into tab:coupling):", flush=True)
    for key, label in labels[:-1]:
        v8 = summary[8][key]
        v16 = summary[16][key]
        print(f"{label:28s} & {_fmt_ms(v8)} & {_fmt_ms(v16)} & {_ratio(v8, v16)} \\\\")

    v8 = summary[8]["total_ms"]
    v16 = summary[16]["total_ms"]
    print(f"{'Total measured GPU time':28s} & {v8 / 1000:.3f}\\,s & "
          f"\\textbf{{{v16 / 1000:.3f}\\,s}} & {_ratio(v8, v16)} \\\\")

    payload = {
        "runs": [{str(k): asdict(v) for k, v in r.items()} for r in all_runs],
        "mean": {str(k): v for k, v in summary.items()},
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
