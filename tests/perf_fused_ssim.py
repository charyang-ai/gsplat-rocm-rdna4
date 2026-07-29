"""End-to-end performance test for the TriSSIM fused-SSIM training loss.

`examples/simple_trainer.py` calls `1 - fused_ssim(pred, gt, padding="valid")` every
training step, so the cost that matters is the **forward + backward** through the SSIM
(the backward is what the optimizer drives). This script times both:

  * forward-only         — a single fused_ssim(...) call
  * forward + backward   — fused_ssim(...) followed by (1 - ssim).backward(),
                           i.e. exactly the training-loss usage

It does proper warmup and GPU synchronization (CUDA/HIP events on device, else a
perf_counter wall clock) and reports per-iteration latency (mean / median / min /
p95) plus throughput in megapixels/s.

Implementations benchmarked (all sourced from the installed `fused_ssim_biauto` /
TriSSIM package, plus a built-in pure-torch reference):

  * baseline   — built-in pure-torch 2D-conv SSIM (the numerical reference)
  * separable  — TriSSIM's pure-torch separable fallback (_fused_ssim_separable)
  * biauto     — TriSSIM's bi-directional autotuned Triton fast path (fwd AND bwd)

Every variant is checked against the baseline in BOTH value and gradient, then each
variant's speedup vs baseline is printed.

Usage (defaults to a single 3x1080x1920 image on the GPU, valid padding, all impls):
  python perf_fused_ssim.py
  python perf_fused_ssim.py --height 800 --width 800 --iters 200 --padding valid
  python perf_fused_ssim.py --batch 4 --dtype float16 --forward-only
  python perf_fused_ssim.py --impl biauto           # only the Triton fast path
  python perf_fused_ssim.py --impl separable        # only the pure-torch separable fallback
  python perf_fused_ssim.py --impl both             # baseline + biauto only
  python perf_fused_ssim.py --device cpu            # CPU reference (falls back to separable)
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F


def _gaussian_1d(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _reference_ssim(img1: torch.Tensor, img2: torch.Tensor,
                    padding: str = "same", train: bool = True) -> torch.Tensor:
    """Built-in pure-torch 2D-conv SSIM — the numerical reference for correctness.

    11x11 Gaussian window (sigma=1.5), C1=0.01^2, C2=0.03^2, data range 1.0, mean
    reduction. This is the classic (non-fused) definition every TriSSIM variant matches.
    """
    assert img1.shape == img2.shape, (img1.shape, img2.shape)
    C = img1.shape[-3]
    g = _gaussian_1d(11, 1.5).to(device=img1.device, dtype=img1.dtype)
    win2d = (g[:, None] * g[None, :]).view(1, 1, 11, 11)
    kernel = win2d.expand(C, 1, 11, 11).contiguous()
    pad = 0 if padding == "valid" else 11 // 2

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu1, mu2 = blur(img1), blur(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = blur(img1 * img1) - mu1_sq
    sigma2_sq = blur(img2 * img2) - mu2_sq
    sigma12 = blur(img1 * img2) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def _load_baseline():
    """Built-in pure-torch reference SSIM (no external dependency)."""
    return _reference_ssim


def _load_separable():
    """TriSSIM's pure-torch separable fallback (_fused_ssim_separable)."""
    from fused_ssim_biauto import _fused_ssim_separable  # noqa: E402
    return _fused_ssim_separable


def _load_biauto():
    """TriSSIM's bi-directional autotuned Triton fast path (fused fwd AND bwd)."""
    from fused_ssim_biauto import fused_ssim  # noqa: E402
    return fused_ssim


def _load_impls(which: str) -> dict:
    """Return an ordered dict of {name: fused_ssim_fn}. baseline is always first."""
    loaders = {
        "baseline": _load_baseline,
        "separable": _load_separable,
        "biauto": _load_biauto,
    }
    if which == "all":
        names = ["baseline", "separable", "biauto"]
    elif which == "both":  # baseline + the Triton fast path
        names = ["baseline", "biauto"]
    else:
        names = [which]
    return {name: loaders[name]() for name in names}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_iters(fn, iters: int, device: torch.device) -> list[float]:
    """Return per-iteration times in milliseconds."""
    times_ms: list[float] = []
    use_events = device.type == "cuda"
    for _ in range(iters):
        if use_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            times_ms.append((time.perf_counter() - t0) * 1e3)
    return times_ms


def _summarize(name: str, times_ms: list[float], megapixels: float) -> float:
    """Print a one-line summary and return the median latency (ms)."""
    mean = statistics.fmean(times_ms)
    median = statistics.median(times_ms)
    mn = min(times_ms)
    p95 = sorted(times_ms)[max(0, int(round(0.95 * len(times_ms))) - 1)]
    # throughput based on median latency (megapixels processed per second)
    mp_s = megapixels / (median / 1e3) if median > 0 else float("inf")
    print(
        f"{name:<34} mean={mean:8.3f} ms  median={median:8.3f} ms  "
        f"min={mn:8.3f} ms  p95={p95:8.3f} ms  throughput={mp_s:8.1f} MP/s"
    )
    return median


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=1, help="batch size N (default 1)")
    p.add_argument("--channels", type=int, default=3, help="channels C (default 3)")
    p.add_argument("--height", type=int, default=1080, help="image height (default 1080)")
    p.add_argument("--width", type=int, default=1920, help="image width (default 1920)")
    p.add_argument("--padding", choices=["valid", "same"], default="valid",
                   help="SSIM padding; simple_trainer uses 'valid' (default)")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"],
                   default="float32", help="input dtype (default float32)")
    p.add_argument("--iters", type=int, default=100, help="timed iterations (default 100)")
    p.add_argument("--warmup", type=int, default=20, help="warmup iterations (default 20)")
    p.add_argument("--forward-only", action="store_true",
                   help="only time the forward pass (skip backward)")
    p.add_argument("--impl",
                   choices=["baseline", "separable", "biauto", "both", "all"],
                   default="all",
                   help="which SSIM implementation(s) to benchmark (default all)")
    p.add_argument("--device", default=None,
                   help="torch device (default: cuda if available else cpu)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_str)
    dtype = getattr(torch, args.dtype)

    impls = _load_impls(args.impl)

    shape = (args.batch, args.channels, args.height, args.width)
    megapixels = args.batch * args.height * args.width / 1e6

    print(f"torch {torch.__version__}  hip {torch.version.hip}  device {device}  "
          f"dtype {args.dtype}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"input {shape}  padding={args.padding}  "
          f"warmup={args.warmup}  iters={args.iters}  impl={args.impl}", flush=True)

    gt = torch.rand(shape, device=device, dtype=dtype)

    def make_pred():
        # fresh leaf each call so backward has a clean graph (mirrors a train step)
        return torch.rand(shape, device=device, dtype=dtype, requires_grad=True)

    # --- correctness: every variant vs baseline should match in value AND gradient ---
    if "baseline" in impls and len(impls) > 1:
        probe = make_pred()
        with torch.no_grad():
            ref_a = impls["baseline"](gt, gt, padding=args.padding)
            ref_b = impls["baseline"](probe, gt, padding=args.padding)

        # reference gradient wrt the input (training uses grad of 1 - ssim)
        pred_b = probe.detach().clone().requires_grad_(True)
        (1.0 - impls["baseline"](pred_b, gt, padding=args.padding)).backward()
        ref_grad = pred_b.grad

        for name, fn in impls.items():
            if name == "baseline":
                continue
            with torch.no_grad():
                da = (fn(gt, gt, padding=args.padding) - ref_a).abs().item()
                db = (fn(probe, gt, padding=args.padding) - ref_b).abs().item()
            pred_i = probe.detach().clone().requires_grad_(True)
            (1.0 - fn(pred_i, gt, padding=args.padding)).backward()
            gd = (pred_i.grad - ref_grad).abs().max().item()
            print(f"correctness  max|{name}-baseline|  value={max(da, db):.3e}  "
                  f"grad={gd:.3e}", flush=True)

    print("-" * 100, flush=True)

    medians: dict = {}
    for name, fused_ssim in impls.items():
        def fwd(_f=fused_ssim):
            with torch.no_grad():
                _f(gt, gt, padding=args.padding)

        def fwd_bwd(_f=fused_ssim):
            pred = make_pred()
            loss = 1.0 - _f(pred, gt, padding=args.padding)
            loss.backward()

        # warmup (also surfaces first-call compile/allocator cost)
        warm_fn = fwd if args.forward_only else fwd_bwd
        for _ in range(args.warmup):
            warm_fn()
        _sync(device)

        medians[(name, "forward")] = _summarize(
            f"[{name}] forward", _time_iters(fwd, args.iters, device), megapixels)
        if not args.forward_only:
            medians[(name, "forward+backward")] = _summarize(
                f"[{name}] forward+backward",
                _time_iters(fwd_bwd, args.iters, device), megapixels)

    # --- speedup summary: each variant vs baseline ---
    if "baseline" in impls and len(impls) > 1:
        print("-" * 100, flush=True)
        modes = ["forward"] if args.forward_only else ["forward", "forward+backward"]
        for mode in modes:
            base = medians[("baseline", mode)]
            for name in impls:
                if name == "baseline":
                    continue
                cur = medians[(name, mode)]
                speedup = base / cur if cur > 0 else float("inf")
                print(f"speedup ({mode:<16}) {name:<10} vs baseline = {speedup:5.2f}x  "
                      f"({base:.3f} ms -> {cur:.3f} ms)", flush=True)


if __name__ == "__main__":
    main()
