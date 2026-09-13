"""Where the narrowed sort's remaining cost is, and whether the key is narrow enough.

\S3.4's argument is that the sort's cost is linear in the number of significant key bits.
That is true only in steps: a radix sort with an \(r\)-bit digit costs
\(\lceil \text{bits}/r \rceil\) passes, so the cost is a staircase and what matters is
which side of a step the configuration sits on. This measures the staircase directly, by
sorting the same data over a widening bit range, and then checks the two things that
follow from it:

  * how far the sort is from its own roofline once the pass count is fixed, which bounds
    what any further key narrowing could buy;

  * whether the key still carries a bit it does not need. The image field is sized
    `bit_width(I)`, which is one bit even for the single-image case where the field is
    always zero. That bit is free wherever it lands inside a pass and costs an entire pass
    wherever it crosses a boundary -- and it crosses one at any resolution whose tile
    count needs exactly 16 bits.

Usage:
  python tests/isect_sortbits.py                 # the staircase, plus the roofline
  python tests/isect_sortbits.py --boundary      # resolutions that sit on a step edge
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from isect_correctness_test import _add_local_paths, _project_scene, _time  # noqa: E402

_add_local_paths()

from triisect._core import _bit_width, _sort_pairs  # noqa: E402

# AMD Radeon AI PRO R9700, as quoted in the paper.
_PEAK_GBPS = 640.0


def run_staircase(args) -> dict:
    """Sort the same pairs over a widening bit range and watch the pass count step."""
    device = torch.device("cuda")
    n = args.n_pairs
    print(f"\n=== staircase: {n/1e6:.1f}M pairs, int32 key + int32 payload\n")
    hdr = (f"{'end_bit':>8}{'passes(8b)':>12}{'ms':>9}{'ms/bit':>9}"
           f"{'GB/s':>9}{'of peak':>9}")
    print(hdr)
    print("-" * len(hdr))

    g = torch.Generator(device="cpu").manual_seed(0)
    keys0 = torch.randint(0, 1 << 30, (n,), generator=g, dtype=torch.int32).to(device)
    vals0 = torch.arange(n, dtype=torch.int32, device=device)
    # scratch the sort is allowed to destroy, restored between reps. Pre-allocated and
    # reused: letting each rep allocate makes the caching allocator, not the sort, the
    # thing being measured, which inflates these numbers several-fold.
    keys = torch.empty_like(keys0)
    vals = torch.empty_like(vals0)
    src = torch.empty_like(keys0)

    def time_sort(end_bit: int, rep: int = 15) -> float:
        """Median time of the sort alone.

        The op consumes its inputs, so each rep needs them restored -- 275 MB of copying
        at this size, comparable to a whole radix pass. That is done between the events
        rather than inside them.
        """
        torch.bitwise_and(keys0, (1 << end_bit) - 1, out=src)
        times = []
        for i in range(rep + 5):
            keys.copy_(src)
            vals.copy_(vals0)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _sort_pairs(keys, vals, 0, end_bit)
            e.record()
            torch.cuda.synchronize()
            if i >= 5:
                times.append(s.elapsed_time(e))
        times.sort()
        return times[len(times) // 2]

    out = {}
    prev = None
    for end_bit in args.end_bits:
        ms = time_sort(end_bit)
        # one pass reads and writes both arrays once
        passes = math.ceil(end_bit / 8)
        gb = passes * 2 * n * 8 / 1e9
        bw = gb / (ms / 1e3)
        step = "" if prev is None else f"  {ms/prev:.2f}x vs prev"
        out[end_bit] = dict(ms=ms, passes=passes, gbps=bw)
        print(f"{end_bit:>8}{passes:>12}{ms:>9.3f}{ms/end_bit:>9.4f}"
              f"{bw:>9.1f}{100*bw/_PEAK_GBPS:>8.1f}%{step}")
        prev = ms
    return out


def run_boundary(args) -> dict:
    """Which real resolutions sit on a pass boundary, and what the spare image bit costs.

    The paper's own configurations land at 16 bits, exactly filling two passes. One bit
    more is three passes, and the image field supplies exactly one bit that a single-image
    render does not use.
    """
    print("\n=== boundary: bits and passes per configuration\n")
    hdr = (f"{'resolution':<14}{'tau':>5}{'tiles':>9}{'tile bits':>11}"
           f"{'+image':>8}{'passes':>8}{'  drop image':>13}{'passes':>8}{'  saves':>8}")
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for (w, h) in args.res:
        for tile in (8, 16):
            tw, th = math.ceil(w / tile), math.ceil(h / tile)
            T = tw * th
            tb = _bit_width(T)
            ib = _bit_width(1)          # what the code uses for a single image
            p_now = math.ceil((tb + ib) / 8)
            p_drop = math.ceil(tb / 8)
            key = f"{w}x{h}_tile{tile}"
            out[key] = dict(tiles=T, tile_bits=tb, passes=p_now, passes_dropped=p_drop)
            print(f"{w}x{h:<8}{tile:>5}{T:>9}{tb:>11}{tb+ib:>8}{p_now:>8}"
                  f"{tb:>13}{p_drop:>8}"
                  f"{('a pass' if p_drop < p_now else '-'):>8}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-pairs", type=int, default=34_462_420,
                   help="default is the paper's tau=8 1080p pair count")
    p.add_argument("--end-bits", type=int, nargs="*",
                   default=[8, 13, 15, 16, 17, 20, 24, 25, 32])
    p.add_argument("--boundary", action="store_true")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    args.res = [(1920, 1080), (2560, 1440), (3840, 2160), (618, 411)]

    results = {"staircase": run_staircase(args)}
    if args.boundary or True:
        results["boundary"] = run_boundary(args)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
