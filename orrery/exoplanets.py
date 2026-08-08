"""The confirmed-exoplanet catalogue, fetched from the NASA Exoplanet Archive.

Why this is here
----------------
Every statistic in :mod:`orrery.statistics` is computed from a sample of **one**
planetary system. That is the fundamental limit on all of it: with a single example
there is no way to know which features of the solar system are typical and which are
accidents of our own history.

Roughly six thousand confirmed exoplanets change that. Once other systems are in
view, questions that were unanswerable become ordinary: are planets usually spaced
like ours? Is a system with no planet inside Mercury's orbit unusual? (It is
strikingly so, and that turns out to be mostly a selection effect — which is the
other thing this module is for.)

Selection effects, stated up front
----------------------------------
This catalogue is **not** a random sample of planets. Two biases dominate and both
push the same way:

* **Transits** require the orbit to cross the star's disc from our line of sight, a
  geometric probability roughly proportional to ``1/a``. Close-in planets are
  enormously over-represented.
* **Radial velocity** measures the star's wobble, whose amplitude scales as
  ``M_planet / sqrt(a)``. Massive, close-in planets are easiest.

So the catalogue's crowd of hot Jupiters is largely a statement about our
instruments, not about the galaxy. Any comparison against the solar system has to
carry that caveat, and the functions here return the discovery method alongside
every planet so it can be conditioned on.

Data handling
-------------
The archive is queried over its TAP service and cached under ``data/cache/``, which
is gitignored: catalogue snapshots are large, change weekly, and do not belong in
version control. Nothing else in the project needs the network, and nothing in CI
touches this module's live path.
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
    "TAP_ENDPOINT",
    "DEFAULT_QUERY",
    "ExoplanetCatalogue",
    "cache_path",
    "fetch_catalogue",
    "load_catalogue",
    "parse_catalogue_csv",
    "transit_probability",
]

TAP_ENDPOINT = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

#: The table queried. ``pscomppars`` is the archive's *completed* parameter set: one
#: row per planet with gaps filled from the most reliable published measurement,
#: rather than one row per publication. That makes it the right table for population
#: statistics and the wrong one for tracing measurement provenance.
SOURCE_TABLE = "pscomppars"

#: Columns fetched, with the archive's names.
#:
#: ``pl_bmasse`` is the "best mass" in Earth masses — a true mass where one is known,
#: otherwise a minimum mass from radial velocity, otherwise an estimate from radius.
#: Mixing those is acceptable for a population overview and would not be for anything
#: quantitative about individual planets.
COLUMNS = (
    "pl_name",
    "hostname",
    "pl_orbper",  # orbital period, days
    "pl_orbsmax",  # semi-major axis, AU
    "pl_bmasse",  # mass, Earth masses
    "pl_rade",  # radius, Earth radii
    "pl_orbeccen",  # eccentricity
    "discoverymethod",
    "sy_dist",  # distance to the system, parsecs
)

DEFAULT_QUERY = (
    f"select {','.join(COLUMNS)} from {SOURCE_TABLE} "
    "where pl_orbper is not null and pl_orbper > 0"
)


def cache_path(root: Path | None = None) -> Path:
    """Where the catalogue snapshot is stored."""
    base = root if root is not None else Path(__file__).resolve().parent.parent
    return base / "data" / "cache" / "exoplanets.csv"


@dataclass(frozen=True)
class ExoplanetCatalogue:
    """A parsed snapshot of the confirmed-planet catalogue.

    Missing values arrive as ``nan`` rather than being dropped, because which rows
    are missing *what* is itself information about the selection function. Use
    :meth:`with_finite` to get a clean subset for a specific analysis.

    Attributes:
        names: Planet designations.
        hosts: Host star names.
        period_days: Orbital period.
        semi_major_axis_au: Semi-major axis; often ``nan``.
        mass_earth: Best-available mass in Earth masses; often ``nan``.
        radius_earth: Radius in Earth radii; often ``nan``.
        eccentricity: Orbital eccentricity; often ``nan``.
        discovery_method: e.g. ``"Transit"``, ``"Radial Velocity"``.
        distance_pc: Distance to the system in parsecs.
    """

    names: tuple[str, ...]
    hosts: tuple[str, ...]
    period_days: np.ndarray
    semi_major_axis_au: np.ndarray
    mass_earth: np.ndarray
    radius_earth: np.ndarray
    eccentricity: np.ndarray
    discovery_method: tuple[str, ...]
    distance_pc: np.ndarray

    def __len__(self) -> int:
        return len(self.names)

    def mask_by_method(self, method: str) -> np.ndarray:
        """Boolean mask selecting one discovery method, matched case-insensitively."""
        wanted = method.strip().lower()
        return np.array(
            [entry.strip().lower() == wanted for entry in self.discovery_method]
        )

    def method_counts(self) -> dict[str, int]:
        """How many planets each technique found, most productive first."""
        counts: dict[str, int] = {}
        for entry in self.discovery_method:
            key = entry.strip() or "Unknown"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))

    def with_finite(self, *fields: str) -> ExoplanetCatalogue:
        """Subset to rows where all named numeric fields are finite.

        Args:
            *fields: Attribute names, e.g. ``"mass_earth"``, ``"period_days"``.

        Raises:
            AttributeError: If a field does not exist or is not numeric.
        """
        mask = np.ones(len(self), dtype=bool)
        for field in fields:
            values = getattr(self, field)
            if not isinstance(values, np.ndarray):
                raise AttributeError(f"{field!r} is not a numeric column")
            mask &= np.isfinite(values)

        return ExoplanetCatalogue(
            names=tuple(np.asarray(self.names)[mask]),
            hosts=tuple(np.asarray(self.hosts)[mask]),
            period_days=self.period_days[mask],
            semi_major_axis_au=self.semi_major_axis_au[mask],
            mass_earth=self.mass_earth[mask],
            radius_earth=self.radius_earth[mask],
            eccentricity=self.eccentricity[mask],
            discovery_method=tuple(np.asarray(self.discovery_method)[mask]),
            distance_pc=self.distance_pc[mask],
        )

    def multiplanet_hosts(self, minimum: int = 2) -> dict[str, int]:
        """Host stars with at least ``minimum`` known planets.

        The systems where spacing and resonance questions can actually be asked, and
        therefore where the solar system can be compared to anything.
        """
        counts: dict[str, int] = {}
        for host in self.hosts:
            counts[host] = counts.get(host, 0) + 1
        return {
            host: count
            for host, count in sorted(counts.items(), key=lambda item: -item[1])
            if count >= minimum
        }


def _to_float(text: str) -> float:
    """Parse a catalogue cell, mapping blanks and junk to ``nan``."""
    stripped = text.strip()
    if not stripped:
        return float("nan")
    try:
        return float(stripped)
    except ValueError:
        return float("nan")


def parse_catalogue_csv(text: str) -> ExoplanetCatalogue:
    """Parse the archive's CSV response.

    Uses the standard library's :mod:`csv` rather than pandas so that the catalogue
    is usable from the base install; pandas stays an optional extra for phase 5.

    Raises:
        ValueError: If the expected columns are absent, which usually means the
            response is a TAP error page rather than data.
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "pl_name" not in reader.fieldnames:
        preview = text[:200].replace("\n", " ")
        raise ValueError(f"response does not look like the catalogue CSV: {preview!r}")

    rows = list(reader)
    if not rows:
        raise ValueError("catalogue response contained no rows")

    return ExoplanetCatalogue(
        names=tuple(row["pl_name"] for row in rows),
        hosts=tuple(row["hostname"] for row in rows),
        period_days=np.array([_to_float(row["pl_orbper"]) for row in rows]),
        semi_major_axis_au=np.array([_to_float(row["pl_orbsmax"]) for row in rows]),
        mass_earth=np.array([_to_float(row["pl_bmasse"]) for row in rows]),
        radius_earth=np.array([_to_float(row["pl_rade"]) for row in rows]),
        eccentricity=np.array([_to_float(row["pl_orbeccen"]) for row in rows]),
        discovery_method=tuple(row["discoverymethod"] for row in rows),
        distance_pc=np.array([_to_float(row["sy_dist"]) for row in rows]),
    )


def fetch_catalogue(
    query: str = DEFAULT_QUERY,
    timeout: float = 120.0,
    destination: Path | None = None,
) -> Path:
    """Download a catalogue snapshot and write it to the cache.

    Args:
        query: ADQL to run against the archive's TAP service.
        timeout: Seconds to wait. The archive can be slow under load.
        destination: Override the cache location.

    Returns:
        Path to the written file.

    Raises:
        RuntimeError: On any network or HTTP failure, with the cause attached. The
            message names the offline path deliberately: a missing catalogue should
            never look like a bug in the analysis code.
    """
    target = destination if destination is not None else cache_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    url = f"{TAP_ENDPOINT}?{urllib.parse.urlencode({'query': query, 'format': 'csv'})}"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            "could not reach the NASA Exoplanet Archive. Everything else in this "
            "project works offline; only the exoplanet panels need this fetch."
        ) from error

    # Validate before caching, so a TAP error page never masquerades as data.
    parse_catalogue_csv(payload)
    target.write_text(payload, encoding="utf-8", newline="\n")
    return target


def load_catalogue(
    source: Path | None = None,
    download_if_missing: bool = False,
) -> ExoplanetCatalogue:
    """Load the cached catalogue, optionally fetching it first.

    Args:
        source: Path to a CSV snapshot. Defaults to the cache location.
        download_if_missing: Fetch when the cache is absent. Left ``False`` by
            default so that no import or test can silently start using the network.

    Raises:
        FileNotFoundError: If the cache is absent and downloading was not requested.
    """
    path = source if source is not None else cache_path()

    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"no catalogue snapshot at {path}. Run "
                "`python scripts/fetch_exoplanets.py` to download one."
            )
        fetch_catalogue(destination=path)

    return parse_catalogue_csv(path.read_text(encoding="utf-8"))


def transit_probability(
    semi_major_axis_au: np.ndarray | float,
    stellar_radius_solar: float = 1.0,
) -> np.ndarray:
    r"""Geometric probability that an orbit transits, seen from a random direction.

    .. math::  p \approx \frac{R_\star}{a}

    The single most important selection effect in the catalogue. At Earth's distance
    from a Sun-like star it is about 0.5%; for a planet orbiting in three days it is
    nearer 10%. That twenty-fold difference, before any consideration of instrument
    sensitivity, is most of why the catalogue looks nothing like the solar system.

    Args:
        semi_major_axis_au: Orbital distance.
        stellar_radius_solar: Stellar radius in solar radii.

    Returns:
        Probability, clipped to at most 1.
    """
    from .constants import AU_KM, MEAN_RADIUS_KM

    stellar_radius_au = stellar_radius_solar * MEAN_RADIUS_KM["sun"] / AU_KM
    return np.minimum(stellar_radius_au / np.asarray(semi_major_axis_au, dtype=float), 1.0)
