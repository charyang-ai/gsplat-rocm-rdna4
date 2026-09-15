# triraster

Autotuned Triton backward for gsplat's 3DGS pixel rasterizer, targeting AMD RDNA4
(gfx1201, wave32).

`rasterize_to_pixels_3dgs_bwd_kernel` is the largest single kernel in a gsplat training
step once the SSIM loss has been dealt with — ~24% of total GPU time at 500k Gaussians
and 1080p on a Radeon AI PRO R9700. This package replaces it, and only it: the HIP
forward rasterizer and every other gsplat kernel are left alone.

## Install

```bash
pip install --no-build-isolation ./triraster
```

torch, triton and gsplat are deliberately not declared as dependencies — on ROCm they
come from the pre-installed PyTorch build, and pinning them risks pulling a CUDA wheel
over the ROCm install.

## Use

```python
import triraster
triraster.install()      # gsplat's rasterizer now backprops through Triton
triraster.uninstall()    # restore the stock HIP kernel
```

`install()` rebinds `gsplat.cuda._wrapper._RasterizeToPixels`, which
`rasterize_to_pixels()` looks up as a module global on every call. Nothing in gsplat is
edited on disk, and the switch is live at any point in a process.

## Fast-path conditions

The Triton backward runs when the inputs are CUDA/HIP float32, the colour channel count
is ≤ 16, and `tile_size` is 8 or 16. Anything else transparently falls back to the HIP
kernel — a wrong-but-plausible gradient is worse than a slower one.

## Why this can be faster on RDNA4

The HIP kernel is CDNA-shaped in two ways that cost time on a wave32 part:

- It stages each batch of Gaussians through LDS so a tile's threads can broadcast-read
  them, paying two `block.sync()` barriers per batch. A load from a wave-uniform
  address is already a scalar broadcast on AMD, so Triton needs neither.
- It warp-reduces per-Gaussian gradients and then has lane 0 of *every* warp issue its
  own `atomicAdd`. At the fork's default `tile_size=8` a block is 64 threads = 2 wave32
  waves, so each Gaussian costs 18 atomics per tile. One Triton program per tile with
  `num_warps=1` reduces inside a single wave and issues 9 — times `SPLIT`, which is why
  splitting a tile is not free (see Autotuning).

## Autotuning

`SPLIT` (how many programs share one rasterizer tile) and `num_warps` are chosen by
`triton.autotune`. Because the kernel accumulates through atomics, the autotuner is
given `reset_to_zero` for all five gradient buffers — without that, benchmarking would
corrupt the very gradients it is timing.

Sweeping the whole candidate list on a real training scene (`--tune-report`) says the
kernel is register-pressure-bound, and that two effects decide the winner. `SPLIT` costs
work linearly, because each program re-walks its tile's entire Gaussian list and issues
its own atomics. Pixels per lane — `tile_size² / (SPLIT · 32 · num_warps)` — costs
registers, because the `[BLOCK_P, CPAD]` accumulator and the loaded cotangents stay live
across the whole walk; past about 4 px/lane it spills hard, below 1 it wastes lanes. The
rule that falls out is to take the smallest `SPLIT` that stays out of the spilling
regime, and it picks both measured optima: `SPLIT=1` at `tile_size=8` and `SPLIT=2` at
`tile_size=16`.

`BLOCK_G` is kept as a parameter but pinned to 1. It only unrolls the walk over the
tile's Gaussians, that walk is serial in the alpha-compositing recurrence so there is no
ILP to expose, and duplicating the live state across unrolled steps makes the spilling
worse — it lost monotonically at `tile_size=8` and was catastrophic at 16.

## Numerics

The kernel is a transliteration of the HIP one, not a reformulation: same back-to-front
walk, same `T *= 1/(1-alpha)` recurrence, same `ALPHA_THRESHOLD` and 0.999 opacity
clamp, same `last_ids` cutoff. Differences against the HIP kernel should be fp32
accumulation-order noise only, and `tests/rasbwd_correctness_test.py` in the parent repo
checks that at three levels, because each one is blind to something the next one catches.

The default run compares all five gradients over five cases (cdim 1/3/5, `absgrad`,
`backgrounds`) as `max|diff| / max|ref|`.

`--verify-configs` repeats that against *every* autotune candidate. Autotuning is per
shape-specialisation, so the config a training run lands on need not be the one a test at
some other resolution happened to select; checking only the winner leaves most of the
search space unexercised. Worst observed over all candidates at both tile sizes: 8.3e-7.

`--grad-bias` compares the per-Gaussian screen-space gradient *norm*, scaled into pixel
units the way `DefaultStrategy` does before thresholding it. This is the level that
matters for training: densification thresholds each Gaussian's own gradient norm, so a
tiny systematic bias would shift the final Gaussian count even though `max|diff| /
max|ref|` — a single global ratio — cannot see it. Measured per-Gaussian relative error
is ~1e-7 median and ~1.8e-6 at p99, with a mean signed bias of ±3e-9, which is about the
standard error of the mean over the Gaussians in the sample and flips sign between
configs. Unbiased, in other words, and identical at `tile_size` 8 and 16.

One warning for anyone A/B-ing end-to-end quality: don't. 3DGS densification amplifies
atomic-ordering perturbations over 30k steps, and swapping this kernel in perturbs
ordering exactly as re-running the HIP kernel does. On mip-nerf360 `bicycle` at
`data_factor 8` (7.1M Gaussians for a 618x411 image, ~28 per pixel) two *identical*
baseline runs differed by 0.14 dB PSNR, so a handful of runs per arm cannot resolve
anything smaller than its own noise. Numerical equivalence is the claim this package can
support; PSNR parity follows from it rather than establishing it.
