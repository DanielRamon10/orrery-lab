"""Download snapshots of the NASA Exoplanet Archive into the local cache.

The only part of this project that needs the network. Run once; the caches are reused
afterwards and are gitignored, because catalogue snapshots are large and change weekly.

Two tables, for two different jobs:

* the **confirmed-planet catalogue**, for the population statistics of phase 4
* the **Kepler Objects of Interest** table, which carries false positives as well as
  planets and is therefore the one that supports classification, in phase 5

Usage::

    python scripts/fetch_exoplanets.py            # confirmed planets
    python scripts/fetch_exoplanets.py --koi      # KOI table as well
    python scripts/fetch_exoplanets.py --force    # refresh existing snapshots
"""

from __future__ import annotations

import argparse
import sys

from orrery.exoplanets import cache_path, fetch_catalogue, load_catalogue
from orrery.koi import (
    LEAKY_FEATURES,
    PHYSICAL_FEATURES,
    fetch_koi_table,
    koi_cache_path,
    load_koi_table,
)


def fetch_koi(force: bool) -> int:
    """Download the KOI table and summarise it."""
    target = koi_cache_path()
    if target.exists() and not force:
        table = load_koi_table()
        print(f"\nKOI already cached: {target}  ({len(table)} objects)")
        return 0

    print("\nquerying the KOI table ...")
    try:
        written = fetch_koi_table(destination=target)
    except RuntimeError as error:
        print(f"failed: {error}", file=sys.stderr)
        return 1

    table = load_koi_table(written)
    complete = table.complete_rows()

    print(f"wrote {written}  ({written.stat().st_size / 1024:.0f} kB)")
    print(f"  objects:            {len(table)}")
    print(f"  confirmed planets:  {int(table.labels.sum())}")
    print(f"  false positives:    {int((1 - table.labels).sum())}")
    print(f"  complete rows:      {int(complete.sum())} ({complete.mean():.1%})")
    print(f"  physical features:  {len(PHYSICAL_FEATURES)}")
    print(f"  leaky features:     {len(LEAKY_FEATURES)} (never train on these)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if a snapshot exists"
    )
    parser.add_argument(
        "--koi", action="store_true", help="also fetch the Kepler Objects of Interest table"
    )
    args = parser.parse_args()

    target = cache_path()
    if target.exists() and not args.force:
        catalogue = load_catalogue()
        print(f"already cached: {target}  ({len(catalogue)} planets)")
        print("pass --force to refresh")
        return fetch_koi(args.force) if args.koi else 0

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

    return fetch_koi(args.force) if args.koi else 0


if __name__ == "__main__":
    raise SystemExit(main())
