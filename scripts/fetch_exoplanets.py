"""Download a snapshot of the NASA Exoplanet Archive into the local cache.

The only part of this project that needs the network. Run once; the cache is reused
afterwards and is gitignored, because catalogue snapshots are large and change weekly.

Usage::

    python scripts/fetch_exoplanets.py
    python scripts/fetch_exoplanets.py --force     # refresh an existing snapshot
"""

from __future__ import annotations

import argparse
import sys

from orrery.exoplanets import cache_path, fetch_catalogue, load_catalogue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if a snapshot exists"
    )
    args = parser.parse_args()

    target = cache_path()
    if target.exists() and not args.force:
        catalogue = load_catalogue()
        print(f"already cached: {target}  ({len(catalogue)} planets)")
        print("pass --force to refresh")
        return 0

    print("querying the NASA Exoplanet Archive ...")
    try:
        written = fetch_catalogue(destination=target)
    except RuntimeError as error:
        print(f"failed: {error}", file=sys.stderr)
        if error.__cause__:
            print(f"  cause: {error.__cause__}", file=sys.stderr)
        return 1

    catalogue = load_catalogue(written)
    size_kb = written.stat().st_size / 1024

    print(f"wrote {written}  ({size_kb:.0f} kB, {len(catalogue)} planets)")
    print(f"  with a measured semi-major axis: {len(catalogue.with_finite('semi_major_axis_au'))}")
    print(f"  with a mass estimate:            {len(catalogue.with_finite('mass_earth'))}")
    print(f"  with a radius:                   {len(catalogue.with_finite('radius_earth'))}")
    print(f"  multi-planet systems:            {len(catalogue.multiplanet_hosts())}")

    print("\nDiscovery methods:")
    for method, count in list(catalogue.method_counts().items())[:8]:
        print(f"  {method:<26} {count:>6}  ({count / len(catalogue):.1%})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
