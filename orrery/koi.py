"""Kepler Objects of Interest: the labelled dataset, and the trap inside it.

Phase 4's catalogue contains only *confirmed* planets, which makes it useless for
classification — there is nothing to classify against. The Kepler Objects of Interest
table is the counterpart: every transit signal the Kepler pipeline flagged, each one
subsequently adjudicated as a real planet or a false positive. Roughly 2,700 confirmed
against 4,800 false positives.

The leakage trap
----------------
The table ships with columns that look like excellent features and are in fact the
*output of the labelling process*:

* ``koi_score`` — a disposition confidence produced by the vetting pipeline
* ``koi_fpflag_nt`` — "not transit-like"
* ``koi_fpflag_ss`` — "stellar eclipse"
* ``koi_fpflag_co`` — "centroid offset"
* ``koi_fpflag_ec`` — "ephemeris match to a known contaminant"

Those four flags are the *reasons* a signal is called a false positive. Feeding them
to a classifier that predicts false positives is circular: the model reaches ~99%
accuracy and has learned nothing except how to read the answer key. It is the classic
target-leakage failure, and it is easy to commit here by accident because the columns
sit in the same table with unremarkable names.

So this module partitions the columns explicitly. :data:`PHYSICAL_FEATURES` are
measurements of the light curve and the host star — things known before anyone decided
what the object was. :data:`LEAKY_FEATURES` are the verdict in disguise.
:func:`orrery.models.evaluate_koi_leakage` trains on both and reports the gap, so the
trap is demonstrated rather than merely described.

Data handling matches :mod:`orrery.exoplanets`: TAP query, cached under ``data/cache/``,
gitignored, one network call.
"""

from __future__ import annotations

import csv
import io
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "PHYSICAL_FEATURES",
    "LEAKY_FEATURES",
    "KoiTable",
    "koi_cache_path",
    "fetch_koi_table",
    "load_koi_table",
    "parse_koi_csv",
]

TAP_ENDPOINT = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

#: Measurements available before the disposition was decided.
#:
#: Transit geometry and depth, signal strength, and host-star properties. A model
#: restricted to these is doing the actual job: telling a planet from an eclipsing
#: binary or an instrumental artefact using the observation alone.
PHYSICAL_FEATURES: tuple[str, ...] = (
    "koi_period",  # orbital period, days
    "koi_duration",  # transit duration, hours
    "koi_depth",  # transit depth, parts per million
    "koi_prad",  # inferred planet radius, Earth radii
    "koi_teq",  # equilibrium temperature, K
    "koi_insol",  # insolation, Earth units
    "koi_impact",  # impact parameter
    "koi_model_snr",  # signal-to-noise of the transit fit
    "koi_steff",  # host effective temperature, K
    "koi_slogg",  # host surface gravity
    "koi_srad",  # host radius, solar radii
    "koi_kepmag",  # Kepler magnitude
)

#: Columns that encode the verdict. Never train on these except to demonstrate why not.
LEAKY_FEATURES: tuple[str, ...] = (
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
)

_ALL_COLUMNS = ("kepoi_name", "koi_disposition", *PHYSICAL_FEATURES, *LEAKY_FEATURES)

DEFAULT_QUERY = (
    f"select {','.join(_ALL_COLUMNS)} from cumulative "
    "where koi_disposition in ('CONFIRMED','FALSE POSITIVE')"
)


def koi_cache_path(root: Path | None = None) -> Path:
    """Where the KOI snapshot is cached."""
    base = root if root is not None else Path(__file__).resolve().parent.parent
    return base / "data" / "cache" / "koi.csv"


@dataclass(frozen=True)
class KoiTable:
    """A parsed KOI snapshot.

    Attributes:
        names: KOI designations.
        labels: 1 for CONFIRMED, 0 for FALSE POSITIVE.
        physical: ``(N, len(PHYSICAL_FEATURES))``, ``nan`` where unmeasured.
        leaky: ``(N, len(LEAKY_FEATURES))``, ``nan`` where absent.
    """

    names: tuple[str, ...]
    labels: np.ndarray
    physical: np.ndarray
    leaky: np.ndarray

    def __len__(self) -> int:
        return len(self.names)

    @property
    def confirmed_fraction(self) -> float:
        return float(np.mean(self.labels))

    def complete_rows(self) -> np.ndarray:
        """Mask of rows where every physical feature is present.

        Dropping incomplete rows rather than imputing is the conservative choice here:
        *whether* a quantity could be measured is itself correlated with the label —
        a false positive often fails to yield a sensible planet radius — so imputation
        would smuggle a weak form of the same leakage back in.
        """
        return np.all(np.isfinite(self.physical), axis=1)


def _to_float(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return float("nan")
    try:
        return float(stripped)
    except ValueError:
        return float("nan")


def parse_koi_csv(text: str) -> KoiTable:
    """Parse the archive's CSV response for the KOI table.

    Raises:
        ValueError: If the expected columns are missing, which normally means a TAP
            error page came back instead of data.
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "koi_disposition" not in reader.fieldnames:
        preview = text[:200].replace("\n", " ")
        raise ValueError(f"response does not look like the KOI table: {preview!r}")

    rows = [row for row in reader if row["koi_disposition"].strip().strip('"')]
    if not rows:
        raise ValueError("KOI response contained no usable rows")

    def disposition(row: dict[str, str]) -> int:
        return 1 if row["koi_disposition"].strip().strip('"') == "CONFIRMED" else 0

    return KoiTable(
        names=tuple(row["kepoi_name"] for row in rows),
        labels=np.array([disposition(row) for row in rows]),
        physical=np.array(
            [[_to_float(row[column]) for column in PHYSICAL_FEATURES] for row in rows]
        ),
        leaky=np.array(
            [[_to_float(row[column]) for column in LEAKY_FEATURES] for row in rows]
        ),
    )


def fetch_koi_table(
    query: str = DEFAULT_QUERY,
    timeout: float = 180.0,
    destination: Path | None = None,
) -> Path:
    """Download the KOI table and cache it.

    Raises:
        RuntimeError: On any network failure, with the cause attached.
    """
    target = destination if destination is not None else koi_cache_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    url = f"{TAP_ENDPOINT}?{urllib.parse.urlencode({'query': query, 'format': 'csv'})}"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            "could not reach the NASA Exoplanet Archive for the KOI table. "
            "Only the exoplanet and KOI panels need this fetch."
        ) from error

    parse_koi_csv(payload)  # validate before caching
    target.write_text(payload, encoding="utf-8", newline="\n")
    return target


def load_koi_table(
    source: Path | None = None,
    download_if_missing: bool = False,
) -> KoiTable:
    """Load the cached KOI table, optionally fetching it first.

    Raises:
        FileNotFoundError: If absent and downloading was not requested.
    """
    path = source if source is not None else koi_cache_path()

    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"no KOI snapshot at {path}. Run `python scripts/fetch_exoplanets.py --koi`."
            )
        fetch_koi_table(destination=path)

    return parse_koi_csv(path.read_text(encoding="utf-8"))
