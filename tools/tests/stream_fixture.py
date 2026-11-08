#!/usr/bin/env python3
"""The stream tests' fixture: the same splats as an INRIA PLY and a splats.bin, so a stream of
either can be compared, with the values the encodings have edges at (a zero and an unnormalized
rotation, -0.0, values near f16's largest, a constant group, opacity logits at both extremes).

  stream_fixture.py <out dir> [--count <n>] [--seed <n>]
      Writes <out dir>/fixture.ply and <out dir>/splats.bin.

The splats are a pure function of count and seed (with the same numpy version).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splatdata import splats  # noqa: E402

EDGE_COUNT = 4  # the edge splats first, then random ones


def fixture_splats(count: int, seed: int, zero_rotation: bool = False) -> splats.Splats:
    """`count` (at least EDGE_COUNT) splats: the edge cases, then random ones. With
    zero_rotation, splat 1's rotation is (0, 0, 0, 0), which only an f32 rotation encodes."""
    if count < EDGE_COUNT:
        raise ValueError(f"the fixture has at least {EDGE_COUNT} splats")
    rng = np.random.default_rng(seed)
    s = splats.Splats(position=rng.normal(size=(count, 3)) * 2, log_scale=rng.normal(size=(count, 3)) - 4,
                      rotation=rng.normal(size=(count, 4)), opacity_logit=rng.normal(size=count) * 3,
                      sh=rng.normal(size=(count, 16, 3)) * 0.4)
    s.rotation[0] = [1, 0, 0, 0]
    s.rotation[1] = [0, 0, 0, 0] if zero_rotation else [3, -4, 0, 12]  # unnormalized
    s.position[2] = [-0.0, 0.0, 65000.0]  # near f16's largest (65504)
    s.log_scale[2] = [-0.0, -60000.0, 1e-8]
    s.opacity_logit[2:4] = [-20.0, 20.0]
    s.sh[3] = 0.0
    s.sh[3, 0] = [-0.0, 0.5, 1.0]
    return s


def write_fixture(out_dir: str, count: int, seed: int) -> splats.Splats:
    """Writes <out_dir>/fixture.ply and <out_dir>/splats.bin of fixture_splats(count, seed)."""
    s = fixture_splats(count, seed)
    os.makedirs(out_dir, exist_ok=True)
    splats.write_ply(os.path.join(out_dir, "fixture.ply"), s)
    with open(os.path.join(out_dir, "splats.bin"), "wb") as f:
        f.write(splats.to_std430(s))
    return s


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    write_fixture(args.out, args.count, args.seed)


if __name__ == "__main__":
    main()
