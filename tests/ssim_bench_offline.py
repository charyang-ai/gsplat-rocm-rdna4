"""OFFLINE, standalone benchmark of five fused_ssim implementations (self-contained).

This is an OFFLINE research/comparison benchmark: it does NOT go through the Docker image's
built-in training pipeline and does NOT use the installed `fused_ssim_biauto` (TriSSIM)
package. Every SSIM variant is implemented inline, so the file is dependency-free — it
needs nothing from `shims/`, TriSSIM, or the image; just `torch` (and `triton` for the
last two variants; they fall back gracefully if it is missing). For the in-pipeline perf
test of the fused_ssim the image actually trains with, see `ssim_bench_pipeline.py`.

It mirrors the training-loss usage `1 - fused_ssim(pred, gt, padding="valid")` and times
both forward-only and forward+backward.

Variants (all match the classic 11x11 Gaussian SSIM: sigma=1.5, C1=0.01^2, C2=0.03^2,
data range 1.0, mean reduction):

  1. baseline   — pure-torch single 11x11 grouped conv2d (the numerical reference)
  2. separable  — pure-torch two 1D passes (11x1 then 1x11), ~5.5x fewer MACs/pixel
  3. compiled   — torch.compile / TorchInductor over the separable SSIM
  4. triton     — hand-written fused Triton blur (2 kernels), backward via conv_transpose2d
  5. biauto     — bi-directional autotuned Triton: autotuned fused forward AND backward

The Triton variants only accelerate the CUDA + float32 + padding="valid" fast path; any
other config transparently falls back to the separable torch implementation.

Usage:
  python tests/ssim_bench_offline.py
  python tests/ssim_bench_offline.py --height 800 --width 800 --iters 200
  python tests/ssim_bench_offline.py --impl biauto --forward-only
  python tests/ssim_bench_offline.py --impl baseline --device cpu
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton missing / CPU-only env
    _HAS_TRITON = False

C1 = 0.01 ** 2
C2 = 0.03 ** 2


def _gaussian_1d(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()  # [w]


def _ssim_from_stats(mu1, mu2, s11, s22, s12):
    """Assemble the mean SSIM from the five blurred quantities."""
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = s11 - mu1_sq
    sigma2_sq = s22 - mu2_sq
    sigma12 = s12 - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


# ======================================================================================
# 1. baseline — single 11x11 grouped conv2d
# ======================================================================================
def ssim_baseline(img1, img2, padding: str = "valid", train: bool = True):
    assert img1.shape == img2.shape, (img1.shape, img2.shape)
    C = img1.shape[-3]
    g = _gaussian_1d(11, 1.5).to(device=img1.device, dtype=img1.dtype)
    win2d = (g[:, None] * g[None, :]).view(1, 1, 11, 11)
    kernel = win2d.expand(C, 1, 11, 11).contiguous()
    pad = 0 if padding == "valid" else 11 // 2

    def blur(x):
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu1, mu2 = blur(img1), blur(img2)
    return _ssim_from_stats(mu1, mu2, blur(img1 * img1), blur(img2 * img2),
                            blur(img1 * img2))


# ======================================================================================
# 2. separable — two 1D grouped convs (11x1 then 1x11)
# ======================================================================================
def ssim_separable(img1, img2, padding: str = "valid", train: bool = True):
    assert img1.shape == img2.shape, (img1.shape, img2.shape)
    C = img1.shape[-3]
    g = _gaussian_1d(11, 1.5).to(device=img1.device, dtype=img1.dtype)
    k_v = g.view(1, 1, 11, 1).expand(C, 1, 11, 1).contiguous()
    k_h = g.view(1, 1, 1, 11).expand(C, 1, 1, 11).contiguous()
    pad = 0 if padding == "valid" else 11 // 2

    def blur(x):
        x = F.conv2d(x, k_v, padding=(pad, 0), groups=C)
        x = F.conv2d(x, k_h, padding=(0, pad), groups=C)
        return x

    mu1, mu2 = blur(img1), blur(img2)
    return _ssim_from_stats(mu1, mu2, blur(img1 * img1), blur(img2 * img2),
                            blur(img1 * img2))


# ======================================================================================
# 3. compiled — torch.compile / TorchInductor over the separable SSIM
# ======================================================================================
_COMPILED_FN = None


def ssim_compiled(img1, img2, padding: str = "valid", train: bool = True):
    global _COMPILED_FN
    # only compile the CUDA path; CPU / non-valid falls back to eager separable
    if not (img1.is_cuda and padding == "valid"):
        return ssim_separable(img1, img2, padding=padding, train=train)
    if _COMPILED_FN is None:
        _COMPILED_FN = torch.compile(ssim_separable, dynamic=False)
    return _COMPILED_FN(img1, img2, padding="valid", train=train)


# ======================================================================================
# Triton kernels (shared math; forward blur for variants 4 and 5)
# ======================================================================================
if _HAS_TRITON:

    _CONFIGS = [
        triton.Config({"BLOCK": block}, num_warps=warps)
        for block in (64, 128, 256, 512, 1024)
        for warps in (1, 2, 4, 8)
    ]

    # ---- forward horizontal (row) pass (plain body; jitted below) ----
    def _row_body(i1_ptr, i2_ptr, o_mu1, o_mu2, o_s11, o_s22, o_s12,
                  H, W, Wout, w_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        plane = pid // H
        row = pid % H
        cb = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cb < Wout
        in_base = plane * (H * W) + row * W
        out_base = plane * (H * Wout) + row * Wout
        a_mu1 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_mu2 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_s11 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_s22 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_s12 = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(0, K):
            wv = tl.load(w_ptr + k)
            c = cb + k
            cm = mask & (c < W)
            i1 = tl.load(i1_ptr + in_base + c, mask=cm, other=0.0)
            i2 = tl.load(i2_ptr + in_base + c, mask=cm, other=0.0)
            a_mu1 += wv * i1
            a_mu2 += wv * i2
            a_s11 += wv * i1 * i1
            a_s22 += wv * i2 * i2
            a_s12 += wv * i1 * i2
        tl.store(o_mu1 + out_base + cb, a_mu1, mask=mask)
        tl.store(o_mu2 + out_base + cb, a_mu2, mask=mask)
        tl.store(o_s11 + out_base + cb, a_s11, mask=mask)
        tl.store(o_s22 + out_base + cb, a_s22, mask=mask)
        tl.store(o_s12 + out_base + cb, a_s12, mask=mask)

    # ---- forward vertical (col) pass (plain body; jitted below) ----
    def _col_body(m1_ptr, m2_ptr, s11_ptr, s22_ptr, s12_ptr,
                  o_mu1, o_mu2, o_s11, o_s22, o_s12,
                  H, Wout, Hout, w_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        plane = pid // Hout
        orow = pid % Hout
        cb = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cb < Wout
        out_base = plane * (Hout * Wout) + orow * Wout
        a_mu1 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_mu2 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_s11 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_s22 = tl.zeros((BLOCK,), dtype=tl.float32)
        a_s12 = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(0, K):
            wv = tl.load(w_ptr + k)
            r = orow + k
            base = plane * (H * Wout) + r * Wout
            a_mu1 += wv * tl.load(m1_ptr + base + cb, mask=mask, other=0.0)
            a_mu2 += wv * tl.load(m2_ptr + base + cb, mask=mask, other=0.0)
            a_s11 += wv * tl.load(s11_ptr + base + cb, mask=mask, other=0.0)
            a_s22 += wv * tl.load(s22_ptr + base + cb, mask=mask, other=0.0)
            a_s12 += wv * tl.load(s12_ptr + base + cb, mask=mask, other=0.0)
        tl.store(o_mu1 + out_base + cb, a_mu1, mask=mask)
        tl.store(o_mu2 + out_base + cb, a_mu2, mask=mask)
        tl.store(o_s11 + out_base + cb, a_s11, mask=mask)
        tl.store(o_s22 + out_base + cb, a_s22, mask=mask)
        tl.store(o_s12 + out_base + cb, a_s12, mask=mask)

    # fixed-BLOCK entry points (variant 4: triton)
    _row_kernel = triton.jit(_row_body)
    _col_kernel = triton.jit(_col_body)

    # autotuned entry points (variant 5: biauto)
    _row_kernel_at = triton.autotune(configs=_CONFIGS, key=["H", "W", "Wout"])(
        triton.jit(_row_body))
    _col_kernel_at = triton.autotune(configs=_CONFIGS, key=["H", "Wout", "Hout"])(
        triton.jit(_col_body))

    # ---- backward: vertical full-correlation (adjoint of valid col conv), 5-way ----
    @triton.autotune(configs=_CONFIGS, key=["Hout", "H", "Wout"])
    @triton.jit
    def _adj_v_kernel(g1, g2, g3, g4, g5, o1, o2, o3, o4, o5,
                      Hout, H, Wout, w_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        plane = pid // H
        orow = pid % H
        cb = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cb < Wout
        out_base = plane * (H * Wout) + orow * Wout
        a1 = tl.zeros((BLOCK,), dtype=tl.float32)
        a2 = tl.zeros((BLOCK,), dtype=tl.float32)
        a3 = tl.zeros((BLOCK,), dtype=tl.float32)
        a4 = tl.zeros((BLOCK,), dtype=tl.float32)
        a5 = tl.zeros((BLOCK,), dtype=tl.float32)
        for kv in tl.static_range(0, K):
            wv = tl.load(w_ptr + kv)
            src = orow - kv
            m = mask & (src >= 0) & (src < Hout)
            base = plane * (Hout * Wout) + src * Wout
            a1 += wv * tl.load(g1 + base + cb, mask=m, other=0.0)
            a2 += wv * tl.load(g2 + base + cb, mask=m, other=0.0)
            a3 += wv * tl.load(g3 + base + cb, mask=m, other=0.0)
            a4 += wv * tl.load(g4 + base + cb, mask=m, other=0.0)
            a5 += wv * tl.load(g5 + base + cb, mask=m, other=0.0)
        tl.store(o1 + out_base + cb, a1, mask=mask)
        tl.store(o2 + out_base + cb, a2, mask=mask)
        tl.store(o3 + out_base + cb, a3, mask=mask)
        tl.store(o4 + out_base + cb, a4, mask=mask)
        tl.store(o5 + out_base + cb, a5, mask=mask)

    # ---- backward: horizontal full-correlation (adjoint of valid row conv), 5-way ----
    @triton.autotune(configs=_CONFIGS, key=["H", "Wout", "W"])
    @triton.jit
    def _adj_h_kernel(g1, g2, g3, g4, g5, o1, o2, o3, o4, o5,
                      H, Wout, W, w_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        plane = pid // H
        row = pid % H
        cb = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cb < W
        out_base = plane * (H * W) + row * W
        in_base = plane * (H * Wout) + row * Wout
        a1 = tl.zeros((BLOCK,), dtype=tl.float32)
        a2 = tl.zeros((BLOCK,), dtype=tl.float32)
        a3 = tl.zeros((BLOCK,), dtype=tl.float32)
        a4 = tl.zeros((BLOCK,), dtype=tl.float32)
        a5 = tl.zeros((BLOCK,), dtype=tl.float32)
        for kh in tl.static_range(0, K):
            wh = tl.load(w_ptr + kh)
            src = cb - kh
            m = mask & (src >= 0) & (src < Wout)
            a1 += wh * tl.load(g1 + in_base + src, mask=m, other=0.0)
            a2 += wh * tl.load(g2 + in_base + src, mask=m, other=0.0)
            a3 += wh * tl.load(g3 + in_base + src, mask=m, other=0.0)
            a4 += wh * tl.load(g4 + in_base + src, mask=m, other=0.0)
            a5 += wh * tl.load(g5 + in_base + src, mask=m, other=0.0)
        tl.store(o1 + out_base + cb, a1, mask=mask)
        tl.store(o2 + out_base + cb, a2, mask=mask)
        tl.store(o3 + out_base + cb, a3, mask=mask)
        tl.store(o4 + out_base + cb, a4, mask=mask)
        tl.store(o5 + out_base + cb, a5, mask=mask)


def _blur5_forward(img1, img2, weight, autotune: bool):
    """(mu1, mu2, s11, s22, s12) as [P,Hout,Wout] via two fused Triton passes."""
    P, H, W = img1.shape
    K = weight.numel()
    R = K // 2
    Wout, Hout = W - 2 * R, H - 2 * R
    row = [torch.empty((P, H, Wout), device=img1.device, dtype=torch.float32)
           for _ in range(5)]
    out = [torch.empty((P, Hout, Wout), device=img1.device, dtype=torch.float32)
           for _ in range(5)]
    if autotune:
        grid_r = lambda meta: (P * H, triton.cdiv(Wout, meta["BLOCK"]))      # noqa: E731
        _row_kernel_at[grid_r](img1, img2, row[0], row[1], row[2], row[3], row[4],
                               H, W, Wout, weight, K=K)
        grid_c = lambda meta: (P * Hout, triton.cdiv(Wout, meta["BLOCK"]))   # noqa: E731
        _col_kernel_at[grid_c](row[0], row[1], row[2], row[3], row[4],
                               out[0], out[1], out[2], out[3], out[4],
                               H, Wout, Hout, weight, K=K)
    else:
        block = 256
        grid_r = (P * H, triton.cdiv(Wout, block))
        _row_kernel[grid_r](img1, img2, row[0], row[1], row[2], row[3], row[4],
                            H, W, Wout, weight, K=K, BLOCK=block)
        grid_c = (P * Hout, triton.cdiv(Wout, block))
        _col_kernel[grid_c](row[0], row[1], row[2], row[3], row[4],
                            out[0], out[1], out[2], out[3], out[4],
                            H, Wout, Hout, weight, K=K, BLOCK=block)
    return out[0], out[1], out[2], out[3], out[4]


def _adjoint_blur_convT(g, kv, kh):
    """Bt via conv_transpose2d (MIOpen): g[P,Hout,Wout] -> [P,H,W]."""
    x = g.unsqueeze(1)
    x = F.conv_transpose2d(x, kv)
    x = F.conv_transpose2d(x, kh)
    return x.squeeze(1)


def _adjoint_blur5_triton(grads, H, W, weight):
    """Bt for five grads via two fused autotuned Triton passes -> five [P,H,W]."""
    P, Hout, Wout = grads[0].shape
    K = weight.numel()
    g = [x.contiguous() for x in grads]
    mid = [torch.empty((P, H, Wout), device=g[0].device, dtype=torch.float32)
           for _ in range(5)]
    grid_v = lambda meta: (P * H, triton.cdiv(Wout, meta["BLOCK"]))          # noqa: E731
    _adj_v_kernel[grid_v](g[0], g[1], g[2], g[3], g[4],
                          mid[0], mid[1], mid[2], mid[3], mid[4],
                          Hout, H, Wout, weight, K=K)
    out = [torch.empty((P, H, W), device=g[0].device, dtype=torch.float32)
           for _ in range(5)]
    grid_h = lambda meta: (P * H, triton.cdiv(W, meta["BLOCK"]))             # noqa: E731
    _adj_h_kernel[grid_h](mid[0], mid[1], mid[2], mid[3], mid[4],
                          out[0], out[1], out[2], out[3], out[4],
                          H, Wout, W, weight, K=K)
    return out


# ======================================================================================
# 4. triton — fused Triton forward, conv_transpose2d backward
# ======================================================================================
class _FusedBlur5Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img1, img2, weight):
        mu1, mu2, s11, s22, s12 = _blur5_forward(img1, img2, weight, autotune=False)
        ctx.save_for_backward(img1, img2, weight)
        return mu1, mu2, s11, s22, s12

    @staticmethod
    def backward(ctx, g_mu1, g_mu2, g_s11, g_s22, g_s12):
        img1, img2, weight = ctx.saved_tensors
        K = weight.numel()
        kv = weight.view(1, 1, K, 1)
        kh = weight.view(1, 1, 1, K)

        def bt(g):
            return _adjoint_blur_convT(g.contiguous(), kv, kh)

        a_mu1, a_s11, a_s12 = bt(g_mu1), bt(g_s11), bt(g_s12)
        a_mu2, a_s22 = bt(g_mu2), bt(g_s22)
        grad_img1 = a_mu1 + 2.0 * img1 * a_s11 + img2 * a_s12
        grad_img2 = a_mu2 + 2.0 * img2 * a_s22 + img1 * a_s12
        return grad_img1, grad_img2, None


# ======================================================================================
# 5. biauto — autotuned fused Triton forward AND autotuned fused Triton backward
# ======================================================================================
class _FusedBlur5BiAuto(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img1, img2, weight):
        mu1, mu2, s11, s22, s12 = _blur5_forward(img1, img2, weight, autotune=True)
        ctx.save_for_backward(img1, img2, weight)
        ctx.hw = (img1.shape[1], img1.shape[2])
        return mu1, mu2, s11, s22, s12

    @staticmethod
    def backward(ctx, g_mu1, g_mu2, g_s11, g_s22, g_s12):
        img1, img2, weight = ctx.saved_tensors
        H, W = ctx.hw
        a_mu1, a_mu2, a_s11, a_s22, a_s12 = _adjoint_blur5_triton(
            [g_mu1, g_mu2, g_s11, g_s22, g_s12], H, W, weight)
        grad_img1 = a_mu1 + 2.0 * img1 * a_s11 + img2 * a_s12
        grad_img2 = a_mu2 + 2.0 * img2 * a_s22 + img1 * a_s12
        return grad_img1, grad_img2, None


def _ssim_triton_like(img1, img2, fn, padding, train):
    fast = (_HAS_TRITON and img1.is_cuda and img1.dtype == torch.float32
            and padding == "valid")
    if not fast:
        return ssim_separable(img1, img2, padding=padding, train=train)
    N, C, H, W = img1.shape
    weight = _gaussian_1d(11, 1.5).to(device=img1.device,
                                      dtype=torch.float32).contiguous()
    p1 = img1.reshape(N * C, H, W).contiguous()
    p2 = img2.reshape(N * C, H, W).contiguous()
    mu1, mu2, s11, s22, s12 = fn.apply(p1, p2, weight)
    return _ssim_from_stats(mu1, mu2, s11, s22, s12)


def ssim_triton(img1, img2, padding: str = "valid", train: bool = True):
    return _ssim_triton_like(img1, img2, _FusedBlur5Triton, padding, train)


def ssim_biauto(img1, img2, padding: str = "valid", train: bool = True):
    return _ssim_triton_like(img1, img2, _FusedBlur5BiAuto, padding, train)


# ======================================================================================
# benchmark harness
# ======================================================================================
IMPLS = {
    "baseline": ssim_baseline,
    "separable": ssim_separable,
    "compiled": ssim_compiled,
    "triton": ssim_triton,
    "biauto": ssim_biauto,
}
ORDER = ["baseline", "separable", "compiled", "triton", "biauto"]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_iters(fn, iters, device):
    times_ms = []
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


def _summarize(name, times_ms, megapixels):
    mean = statistics.fmean(times_ms)
    median = statistics.median(times_ms)
    mn = min(times_ms)
    p95 = sorted(times_ms)[max(0, int(round(0.95 * len(times_ms))) - 1)]
    mp_s = megapixels / (median / 1e3) if median > 0 else float("inf")
    print(f"{name:<28} mean={mean:8.3f} ms  median={median:8.3f} ms  "
          f"min={mn:8.3f} ms  p95={p95:8.3f} ms  throughput={mp_s:8.1f} MP/s")
    return median


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--padding", choices=["valid", "same"], default="valid")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"],
                   default="float32")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--forward-only", action="store_true")
    p.add_argument("--impl", choices=ORDER + ["all"], default="all")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_str)
    dtype = getattr(torch, args.dtype)

    names = ORDER if args.impl == "all" else [args.impl]
    if "baseline" not in names:
        names = ["baseline"] + names  # always keep the reference for correctness
    impls = {n: IMPLS[n] for n in names}

    shape = (args.batch, args.channels, args.height, args.width)
    megapixels = args.batch * args.height * args.width / 1e6

    print(f"torch {torch.__version__}  hip {torch.version.hip}  device {device}  "
          f"dtype {args.dtype}  triton={_HAS_TRITON}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"input {shape}  padding={args.padding}  "
          f"warmup={args.warmup}  iters={args.iters}  impl={args.impl}", flush=True)

    gt = torch.rand(shape, device=device, dtype=dtype)

    def make_pred():
        return torch.rand(shape, device=device, dtype=dtype, requires_grad=True)

    # --- correctness: value AND gradient vs baseline ---
    if len(impls) > 1:
        probe = make_pred()
        with torch.no_grad():
            ref_a = impls["baseline"](gt, gt, padding=args.padding)
            ref_b = impls["baseline"](probe, gt, padding=args.padding)
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

    medians = {}
    for name, fused_ssim in impls.items():
        def fwd(_f=fused_ssim):
            with torch.no_grad():
                _f(gt, gt, padding=args.padding)

        def fwd_bwd(_f=fused_ssim):
            pred = make_pred()
            loss = 1.0 - _f(pred, gt, padding=args.padding)
            loss.backward()

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

    # --- speedup summary vs baseline ---
    if len(impls) > 1:
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
