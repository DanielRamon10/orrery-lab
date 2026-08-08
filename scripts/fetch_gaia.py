"""Download a Gaia DR3 sample from the ESA archive into the local cache.

One network call, cached afterwards and gitignored like every other catalogue here.

The default is a magnitude-limited sample: all sources brighter than G = 11 with a
parallax measured to better than 5 sigma. That is around a hundred thousand stars,
mostly within a kiloparsec, and it is emphatically **not** a fair sample of the
galaxy — see the module docstring of :mod:`orrery.gaia` for what it is biased towards
and why that matters for anything read off it.

Usage::

    python scripts/fetch_gaia.py
    python scripts/fetch_gaia.py --magnitude 9 --limit 40000
    python scripts/fetch_gaia.py --force
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from orrery.gaia import (
    DEFAULT_PARALLAX_SNR,
    fetch_gaia_sample,
    gaia_cache_path,
    load_gaia_sample,
)


def summarise(stars) -> None:
    """Print what came back, including what is unusable and why."""
    reliable = stars.reliable()
    distance = stars.distance_parsec()
    usable = np.isfinite(distance)

    print(f"  stars:                {len(stars):,}")
    print(
        f"  parallax SNR >= {DEFAULT_PARALLAX_SNR:.0f}:  {int(reliable.sum()):,} "
        f"({reliable.mean():.1%})"
    )
    print(f"  usable distances:     {int(usable.sum()):,}")

    if usable.any():
        finite = distance[usable]
        print(
            f"  distance range:       {finite.min():.1f} – {finite.max():,.0f} pc "
            f"(median {np.median(finite):,.0f} pc)"
        )
        absolute = stars.absolute_g()
        good = np.isfinite(absolute)
        print(f"  absolute magnitudes:  {int(good.sum()):,} computable")
        print(
            f"  G magnitude range:    {stars.g_magnitude.min():.2f} – "
            f"{stars.g_magnitude.max():.2f}"
        )
        print(
            f"  colour BP-RP range:   {np.nanmin(stars.bp_rp):.2f} – "
            f"{np.nanmax(stars.bp_rp):.2f}"
        )

    with_velocity = np.isfinite(stars.radial_velocity)
    print(f"  radial velocities:    {int(with_velocity.sum()):,} ({with_velocity.mean():.1%})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download if cached")
    parser.add_argument("--limit", type=int, default=120_000, help="maximum rows")
    parser.add_argument(
        "--magnitude", type=float, default=11.0, help="faintest apparent G magnitude"
    )
    parser.add_argument(
        "--snr", type=float, default=5.0, help="parallax signal-to-noise floor in the query"
    )
    args = parser.parse_args()

    target = gaia_cache_path()
    if target.exists() and not args.force:
        print(f"already cached: {target}")
        summarise(load_gaia_sample())
        print("\npass --force to refresh")
        return 0

    print(
        f"querying the ESA Gaia archive "
        f"(G < {args.magnitude}, parallax SNR > {args.snr}, up to {args.limit:,} rows) ..."
    )
    try:
        written = fetch_gaia_sample(
            limit=args.limit,
            magnitude_limit=args.magnitude,
            minimum_snr=args.snr,
            destination=target,
        )
    except RuntimeError as error:
        print(f"failed: {error}", file=sys.stderr)
        if error.__cause__:
            print(f"  cause: {error.__cause__}", file=sys.stderr)
        return 1

    print(f"wrote {written}  ({written.stat().st_size / 1024 / 1024:.1f} MB)\n")
    summarise(load_gaia_sample(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
