# triisect

Triton tile intersection for gsplat's 3DGS rasterizer, targeting AMD RDNA4 (gfx1201,
wave32).

Tile intersection is not a kernel anyone expects to matter — it emits a list of
`(tile, Gaussian)` pairs and sorts it. But at 500k Gaussians and 1080p on a Radeon AI PRO
R9700 the stage is **20.99 ms of a 42.86 ms training step at `tile_size=8`** (gsplat's
ROCm default) and 5.10 ms of 27.56 ms at 16, which makes it the second largest item in the
step after the rasterizer backward. This package replaces the stage, and only it.

The pairs it emits are bit-identical to the HIP op's, so `rasterization()` returns exactly
the same image — `torch.equal`, not a tolerance. See [Correctness](#correctness).

## Install

```bash
pip install --no-build-isolation ./triisect
```

torch, triton and gsplat are deliberately not declared as dependencies — on ROCm they
come from the pre-installed PyTorch build, and pinning them risks pulling a CUDA wheel
over the ROCm install.

## Use

```python
import triisect
triisect.install()                  # output-parallel Triton emit, gsplat's 64-bit sort
triisect.install(presort=True)      # ... plus the depth-presorted 32-bit tile sort
triisect.uninstall()                # restore the stock HIP stage
```

`install()` rebinds `isect_tiles` and `isect_offset_encode` in **both**
`gsplat.cuda._wrapper` and `gsplat.rendering`. The second one is the one that matters and
is easy to miss: `rendering.py` does `from .cuda._wrapper import isect_tiles`, so it holds
its own module-global reference and patching `_wrapper` alone is a silent no-op. The two
names are installed as a pair because with `presort=True` they agree on a key encoding
the HIP offset kernel does not understand.

## Fast-path conditions

The Triton path runs for CUDA/HIP float32 inputs in either the dense `[..., N, ...]` or
the packed `[nnz, ...]` layout. `segmented=True` falls back to the HIP op.

The packed layout is not optional: `rasterization()` defaults to `packed=True`, so a
version that only handled the dense one would fall back on every real training step while
still passing every dense test.

## Results

Radeon AI PRO R9700 (gfx1201), ROCm 7.2, torch 2.13.0+rocm7.2, 500k Gaussians, 1080p,
`sh_degree=3`, TriSSIM loss, Triton rasterizer backward. `emit` is the Triton emit with
gsplat's own sort kept; `triton` adds the presorted sort and the `searchsorted` offsets.

| tile | isect | step (ms) | stage (ms) | emit | sort | step speedup |
|-----:|:---------|----------:|-----------:|------:|------:|-------------:|
| 8 | baseline | 42.86 | 20.99 | 10.57 | 9.94 | 1.00x |
| 8 | emit | 33.32 | 11.22 | 0.79 | 9.96 | 1.29x |
| 8 | **triton** | **25.74** | **3.40** | **0.77** | **2.63** | **1.67x** |
| 16 | baseline | 27.56 | 5.10 | 1.59 | 3.35 | 1.00x |
| 16 | emit | 26.24 | 3.73 | 0.26 | 3.31 | 1.05x |
| 16 | **triton** | **24.02** | **1.18** | **0.25** | **0.93** | **1.15x** |
| 32 | baseline | 31.06 | 1.25 | 0.11 | 1.13 | 1.00x |
| 32 | triton | 30.30 | 0.58 | 0.11 | 0.47 | 1.03x |

A side effect worth more than it looks: the stage is *why* small tiles were expensive.
The baseline's best and worst tile size differ by 1.55x (27.56 vs 42.86); with this stage
installed they differ by 1.07x. Tile size stops being a load-bearing tuning decision.

Stage cost in isolation, measured on the tensors captured from a real `rasterization()`
call (`--trainer-bench`), where `B` is the emit alone and `B+D` adds the sort:

| shape | tile | pairs/Gaussian | HIP | B | B+D | speedup |
|:----------|-----:|---------------:|-------:|-------:|------:|--------:|
| 1920x1080 | 8 | 69.0 | 22.940 | 11.602 | 3.756 | 6.11x |
| 1920x1080 | 16 | 21.6 | 5.385 | 3.752 | 1.495 | 3.60x |
| 618x411 | 8 | 12.0 | 2.039 | 2.077 | 1.005 | 2.03x |
| 618x411 | 16 | 5.0 | 0.601 | 0.668 | 0.698 | 0.86x |

The last row is the honest limit: at 2.5 M pairs from 500 k Gaussians there is not enough
work to cover the pipeline's fixed cost (a depth sort over all Gaussians, plus more kernel
launches than one fused HIP op needs). Raising the Gaussian count to the density a trained
scene actually reaches turns that cell into 2.03x — the fixed cost is `O(N)` while the
saving is `O(n_isects)`.

## Why this can be faster on RDNA4

### The emit is write-scatter bound, not bandwidth bound

`intersect_tile_kernel` runs one thread per Gaussian and appends that Gaussian's pairs at
a per-thread cursor. The 32 lanes of a wave therefore write to addresses
`tiles_per_gauss * 8B` apart — one store instruction touching up to 32 cache lines instead
of 4 — and the loop trip count varies from 1 to several hundred within the wave. Making
the *output slot* the parallel axis instead fixes both at once: lane k writes pair k, so
the stores are contiguous and every lane does the same amount of work.

How much that is worth is a function of pairs per Gaussian, i.e. of how divergent the
baseline's inner loop is: 11.5x on the emit at 69 pairs/Gaussian, 4.3x at 21.6, and
break-even around 12.

### The sort does not need 46 bits of a 64-bit key

gsplat sorts `image | tile | depth-bits` so that one pass leaves each tile's Gaussians in
depth order. Sorting the Gaussians by depth *first* makes the depth field redundant: a
stable sort over the tile bits alone then produces the same order. That is 14 bits of a
4-byte key instead of 46 bits of an 8-byte one, and measures 3.0–3.4x across every shape
tested. The depth sort it costs is over `N` Gaussians, not `n_isects` pairs.

Offsets come out of the same change almost free: with 32-bit `image|tile` keys the offset
array is `searchsorted(keys, arange(n_tiles))`, which is 0.018 ms flat against the HIP
boundary-scanning kernel's 0.44 ms at 34 M pairs.

### Finding the owner of a slot: search beats materialising

An output-parallel emit needs the inverse of the prefix sum — which Gaussian owns slot k.
`torch.repeat_interleave` materialises it in one streaming pass, which looks obviously
cheaper than a ~20-step binary search per pair. It is not:

| 1080p, 500k Gaussians | tile 8 | tile 16 |
|:--|--:|--:|
| repeat_interleave map | 2.335 ms | 0.872 ms |
| in-kernel binary search | 1.059 ms | 0.436 ms |

The map's cost is an extra `n_isects`-sized array written and then read — real DRAM
traffic, 1.24 ms at 34 M pairs, more than the emit kernel it feeds. The search touches
only a few cache lines per program, because consecutive slots belong to neighbouring
Gaussians, so it is L1-resident and costs almost nothing. Both are kept
(`isect_tiles(search=...)`); the search is the default.

## Correctness

```bash
python tests/isect_correctness_test.py                  # exact pairs, dense + packed
python tests/isect_correctness_test.py --presort        # ... and the presorted variant
python tests/isect_correctness_test.py --render         # exact image through rasterization()
python tests/isect_correctness_test.py --bench          # A/B timing
python tests/isect_correctness_test.py --trainer-bench  # ... on captured trainer tensors
```

The stage produces integers, so the gate is `torch.equal` rather than a tolerance: any
difference is a bug, not accumulation noise. Checked over both tile sizes, three
resolutions including non-tile-aligned ones, the dense and packed layouts, and five
adversarial cases — all-culled, one Gaussian covering every tile, bboxes clamping to empty
off each edge, 1x1 bboxes, and 256 Gaussians sharing a tile *and* a depth (which is where
a stable and an unstable sort part ways).

With `presort=True` the `isect_ids` are int32 `image|tile` keys and so differ from the
baseline's int64 ones by construction; `tiles_per_gauss`, `flatten_ids` and the encoded
offsets — everything the rasterizer reads — are still bit-identical, and `--render`
confirms the image is.

## Requires

`patches/expose_sort.gfx1201.patch`, which exposes gsplat's own cub radix sort to Python
over an explicit bit range. Two reasons it is needed rather than convenient: comparing a
Triton emit against the fused HIP op is only meaningful if both use the *same* sort, and
the bit range is the entire point of the presorted variant. Without the patch the package
still runs, falling back to `torch.sort` — a full-width sort with int64 indices, about 2x
the traffic.

`patches/cub_allocator_torch210.gfx1201.patch` is unrelated to this work: it is what makes
the pinned gsplat ref compile against torch >= 2.10 at all, where
`c10::hip::HIPCachingAllocator::get()` no longer exists.
