"""Profile a synthetic 3DGS training step to locate the real bottleneck kernels.

`examples/simple_trainer.py` needs a COLMAP dataset and a full CLI, which makes it awkward
to profile in isolation. This harness instead drives the SAME hot path — gsplat's
`rasterization` forward + backward, an L1 + fused-SSIM photometric loss, and an Adam
step — on a randomly generated scene (no dataset required), under `torch.profiler`, and
prints the operators/kernels sorted by GPU (self CUDA/HIP) time.

That ranked table is what tells you where the time actually goes (typically the
rasterize backward dominates, well ahead of the SSIM loss), so you can prioritize
optimization instead of guessing.

Usage (needs a GPU exposed to the container):
  python tests/profile_trainer.py
  python tests/profile_trainer.py --num-gaussians 500000 --height 1080 --width 1920
  python tests/profile_trainer.py --iters 50 --sh-degree 3
  python tests/profile_trainer.py --no-ssim            # profile rasterize + L1 only
  python tests/profile_trainer.py --trace trace.json   # also dump a chrome trace
"""
from __future__ import annotations

import argparse
import math

import torch
from torch.profiler import ProfilerActivity, profile, schedule


def _import_rasterization():
    """gsplat exposes rasterization at the top level (newer) or under .rendering."""
    try:
        from gsplat import rasterization
        return rasterization
    except Exception:
        from gsplat.rendering import rasterization  # noqa: E402
        return rasterization


def _load_fused_ssim():
    """Optional: the training SSIM loss (TriSSIM in the image). None if unavailable."""
    try:
        from fused_ssim import fused_ssim
        return fused_ssim
    except Exception:
        return None


def _make_scene(n: int, sh_degree: int, device, dtype):
    """Random Gaussians as trainable leaves, placed in front of a single camera."""
    means = (torch.randn(n, 3, device=device, dtype=dtype) * 0.5).requires_grad_(True)
    quats = torch.randn(n, 4, device=device, dtype=dtype).requires_grad_(True)
    scales = (torch.rand(n, 3, device=device, dtype=dtype) * 0.05).requires_grad_(True)
    opacities = torch.randn(n, device=device, dtype=dtype).requires_grad_(True)
    if sh_degree and sh_degree > 0:
        k = (sh_degree + 1) ** 2
        colors = (torch.rand(n, k, 3, device=device, dtype=dtype)).requires_grad_(True)
    else:
        colors = (torch.rand(n, 3, device=device, dtype=dtype)).requires_grad_(True)
    return means, quats, scales, opacities, colors


def _make_camera(width: int, height: int, device, dtype):
    """One camera at the origin looking down +z; scene sits ~5 units in front."""
    focal = 0.5 * width / math.tan(0.5 * math.radians(60.0))
    K = torch.tensor([[focal, 0.0, width / 2.0],
                      [0.0, focal, height / 2.0],
                      [0.0, 0.0, 1.0]], device=device, dtype=dtype)[None]  # [1,3,3]
    viewmat = torch.eye(4, device=device, dtype=dtype)
    viewmat[2, 3] = 5.0  # push the scene to depth ~5 in camera space (in front)
    return viewmat[None], K  # [1,4,4], [1,3,3]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-gaussians", type=int, default=200_000)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--sh-degree", type=int, default=0,
                   help="0 = plain RGB colors; >0 = evaluate SH of that degree")
    p.add_argument("--iters", type=int, default=30, help="profiled training steps")
    p.add_argument("--warmup", type=int, default=10,
                   help="unprofiled warmup steps (compile/allocator/tuning settle)")
    p.add_argument("--no-ssim", action="store_true",
                   help="use L1 only (skip the fused-SSIM term)")
    p.add_argument("--row-limit", type=int, default=30,
                   help="rows in the sorted kernel table")
    p.add_argument("--sort-by", default="self_cuda_time_total",
                   help="profiler table sort key (e.g. self_cuda_time_total, "
                        "cuda_time_total, self_cpu_time_total)")
    p.add_argument("--trace", default=None,
                   help="optional path to also export a chrome trace (json)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_str)
    dtype = torch.float32
    if device.type != "cuda":
        print("WARNING: no CUDA/HIP device — profiling CPU only (not representative).")

    rasterization = _import_rasterization()
    fused_ssim = None if args.no_ssim else _load_fused_ssim()

    print(f"torch {torch.__version__}  hip {torch.version.hip}  device {device}",
          flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"gaussians={args.num_gaussians}  image={args.width}x{args.height}  "
          f"sh_degree={args.sh_degree}  ssim={'off' if fused_ssim is None else 'on'}  "
          f"warmup={args.warmup}  iters={args.iters}", flush=True)

    means, quats, scales, opacities, colors = _make_scene(
        args.num_gaussians, args.sh_degree, device, dtype)
    viewmats, Ks = _make_camera(args.width, args.height, device, dtype)
    target = torch.rand(1, 3, args.height, args.width, device=device, dtype=dtype)

    opt = torch.optim.Adam([means, quats, scales, opacities, colors], lr=1e-3)
    sh_degree = args.sh_degree if args.sh_degree and args.sh_degree > 0 else None

    def train_step():
        opt.zero_grad(set_to_none=True)
        renders, _alphas, _meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=torch.sigmoid(opacities),
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=args.width,
            height=args.height,
            sh_degree=sh_degree,
        )
        img = renders[0].permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)  # [1,3,H,W]
        loss = (img - target).abs().mean()
        if fused_ssim is not None:
            loss = loss + (1.0 - fused_ssim(img, target, padding="valid"))
        loss.backward()
        opt.step()
        return loss

    # warmup (surfaces first-call compile / allocator / autotune cost)
    for _ in range(args.warmup):
        train_step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # schedule: everything active (we already warmed up above); repeat once.
    sched = schedule(wait=0, warmup=1, active=args.iters, repeat=1)
    with profile(activities=activities, schedule=sched, record_shapes=False,
                 profile_memory=False, with_stack=False) as prof:
        for _ in range(args.iters + 1):  # +1 to cover the schedule's warmup slot
            train_step()
            prof.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    print("=" * 100, flush=True)
    print(f"top {args.row_limit} operators by {args.sort_by}:", flush=True)
    print(prof.key_averages().table(sort_by=args.sort_by, row_limit=args.row_limit),
          flush=True)

    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"chrome trace written to {args.trace}", flush=True)


if __name__ == "__main__":
    main()
