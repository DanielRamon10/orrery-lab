r"""Gaia DR3: real stars, in three dimensions.

Everything before this phase lives inside the solar system, where distances are
measured by radar and the geometry is exact. Stepping outside means confronting the
first genuinely *uncertain* measurement in the project — and the temptation to hide
that uncertainty behind a one-line conversion.

The parallax trap
-----------------
Gaia measures **parallax**, the tiny annual wobble of a star's position caused by
Earth's orbit. Distance follows from ``d = 1/parallax``, and that formula is where
most casual uses of this catalogue go wrong. Three separate problems:

1. **Negative parallaxes exist.** Roughly a quarter of DR3 sources have one. They are
   not errors in the catalogue — parallax is a *measurement*, and for a distant star
   the true value is smaller than the noise, so noise pushes some measurements below
   zero. ``1/parallax`` on those gives a negative distance. Silently dropping them
   biases the sample towards nearby stars; silently keeping them produces nonsense.
2. **The reciprocal is biased even when the parallax is positive.** ``1/x`` is a
   non-linear function, so the expectation of ``1/parallax`` is not ``1/E[parallax]``.
   For a fractional parallax error above about 20% the bias becomes severe, and no
   amount of averaging removes it (Lutz & Kelker 1973; Bailer-Jones 2015).
3. **A magnitude-limited sample is not a volume-limited one.** Cutting on apparent
   brightness preferentially keeps intrinsically luminous stars at large distance —
   the Malmquist bias. The nearby sample looks different from the far one for reasons
   that have nothing to do with the galaxy.

This module does not solve those problems, because they are not solvable with a
formula. It does the next best thing: it refuses to convert a parallax whose
signal-to-noise is below a stated threshold, records what that discards, and puts the
caveat in the return value rather than in a footnote. See
:func:`distance_from_parallax`.

What is here
------------
* a TAP query against the ESA archive, cached like every other catalogue in this project
* the ICRS to galactic rotation, *constructed* from the pole coordinates and checked
  against the published matrix
* 3D positions relative to the Sun, in parsecs
* absolute magnitude, which turns the catalogue into a Hertzsprung-Russell diagram —
  the plot that revealed stellar evolution, and which falls straight out of two
  columns here
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
    "NORTH_GALACTIC_POLE_RA_DEG",
    "NORTH_GALACTIC_POLE_DEC_DEG",
    "GALACTIC_LONGITUDE_OF_NCP_DEG",
    "PARSEC_IN_AU",
    "GaiaStars",
    "gaia_cache_path",
    "fetch_gaia_sample",
    "load_gaia_sample",
    "parse_gaia_csv",
    "icrs_to_galactic_matrix",
    "equatorial_to_cartesian",
    "distance_from_parallax",
    "absolute_magnitude",
    "bp_rp_to_rgb",
    "DEFAULT_QUERY",
]

TAP_ENDPOINT = "https://gea.esac.esa.int/tap-server/tap/sync"

#: Direction of the north galactic pole in ICRS, degrees. From the Gaia DR3
#: documentation, inherited from the Hipparcos definition.
NORTH_GALACTIC_POLE_RA_DEG = 192.85948
NORTH_GALACTIC_POLE_DEC_DEG = 27.12825

#: Galactic longitude of the north celestial pole, degrees. The third angle needed to
#: pin the rotation — without it the galactic frame would be free to spin about its
#: own pole.
GALACTIC_LONGITUDE_OF_NCP_DEG = 122.93192

#: One parsec in astronomical units. A parsec is the distance at which one AU
#: subtends one arcsecond, so this is just the number of arcseconds in a radian.
PARSEC_IN_AU = 648_000.0 / np.pi

#: Minimum ``parallax / parallax_error`` for a distance to be reported at all.
#:
#: Ten corresponds to a 10% fractional uncertainty, below which the reciprocal's bias
#: is small enough to ignore for a visualisation. It is a judgement call, and it is a
#: parameter rather than a constant buried in a function for that reason.
DEFAULT_PARALLAX_SNR = 10.0

DEFAULT_QUERY = """
SELECT TOP {limit}
    source_id, ra, dec, parallax, parallax_error, parallax_over_error,
    phot_g_mean_mag, bp_rp, pmra, pmdec, radial_velocity
FROM gaiadr3.gaia_source
WHERE parallax_over_error > {snr}
  AND phot_g_mean_mag < {magnitude}
  AND bp_rp IS NOT NULL
ORDER BY phot_g_mean_mag
"""


def gaia_cache_path(root: Path | None = None) -> Path:
    """Where the Gaia snapshot is cached."""
    base = root if root is not None else Path(__file__).resolve().parent.parent
    return base / "data" / "cache" / "gaia.csv"


@dataclass(frozen=True)
class GaiaStars:
    """A parsed Gaia sample.

    Attributes:
        source_id: Gaia's unique identifier.
        ra: Right ascension, degrees (ICRS).
        dec: Declination, degrees.
        parallax: Parallax in milliarcseconds. **May be negative** in a general
            sample; the default query filters those out.
        parallax_error: Its standard error, mas.
        parallax_over_error: Signal-to-noise of the parallax.
        g_magnitude: Apparent magnitude in Gaia's broad G band.
        bp_rp: Colour index. Larger means redder and, for main-sequence stars,
            cooler.
        pmra, pmdec: Proper motion, mas/yr.
        radial_velocity: km/s where measured, ``nan`` otherwise.
    """

    source_id: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    parallax: np.ndarray
    parallax_error: np.ndarray
    parallax_over_error: np.ndarray
    g_magnitude: np.ndarray
    bp_rp: np.ndarray
    pmra: np.ndarray
    pmdec: np.ndarray
    radial_velocity: np.ndarray

    def __len__(self) -> int:
        return len(self.source_id)

    def reliable(self, minimum_snr: float = DEFAULT_PARALLAX_SNR) -> np.ndarray:
        """Mask of stars whose parallax is precise enough to invert."""
        return np.isfinite(self.parallax_over_error) & (
            self.parallax_over_error >= minimum_snr
        )

    def distance_parsec(self, minimum_snr: float = DEFAULT_PARALLAX_SNR) -> np.ndarray:
        """Distance in parsecs, ``nan`` where the parallax is not trustworthy."""
        return distance_from_parallax(self.parallax, self.parallax_over_error, minimum_snr)

    def absolute_g(self, minimum_snr: float = DEFAULT_PARALLAX_SNR) -> np.ndarray:
        """Absolute G magnitude, ``nan`` where the distance is unusable."""
        return absolute_magnitude(self.g_magnitude, self.distance_parsec(minimum_snr))

    def cartesian_galactic(
        self, minimum_snr: float = DEFAULT_PARALLAX_SNR
    ) -> np.ndarray:
        """``(N, 3)`` positions in parsecs, Sun at the origin, galactic axes.

        x points at the galactic centre, y along galactic rotation, z at the north
        galactic pole. Rows with an unusable parallax come back as ``nan``.
        """
        direction = equatorial_to_cartesian(self.ra, self.dec)
        galactic = np.einsum("ij,nj->ni", icrs_to_galactic_matrix(), direction)
        return galactic * self.distance_parsec(minimum_snr)[:, None]

    def brightest(self, count: int) -> GaiaStars:
        """The ``count`` apparently brightest stars, for the 3D scene."""
        order = np.argsort(self.g_magnitude)[:count]
        return GaiaStars(
            **{
                field: getattr(self, field)[order]
                for field in (
                    "source_id", "ra", "dec", "parallax", "parallax_error",
                    "parallax_over_error", "g_magnitude", "bp_rp", "pmra", "pmdec",
                    "radial_velocity",
                )
            }
        )


def icrs_to_galactic_matrix() -> np.ndarray:
    r"""Rotation taking ICRS unit vectors to galactic ones.

    **Constructed** from the three defining angles rather than pasted in as a table of
    nine numbers, so the definition is visible and a typo cannot hide in the fourteenth
    decimal place. ``tests/test_gaia.py`` checks the result against the published
    matrix and against the position of the galactic centre.

    The three rotations, right to left, all in the passive convention (they rotate the
    *frame*, not the vector):

    1. ``Rz(alpha_NGP + 90)`` swings the x-axis onto the ascending node of the galactic
       plane on the celestial equator;
    2. ``Rx(90 - delta_NGP)`` tips the equator onto the galactic plane;
    3. ``Rz(90 - l_NCP)`` spins within the galactic plane so longitude zero lands on
       the galactic centre. Without this third angle the frame would still be free to
       rotate about its own pole.

    The ``90 -`` in the third term is not decoration. Written as ``Rz(-l_NCP)``, which
    is how the angle is usually quoted, the first two steps already leave the frame a
    quarter turn ahead: the galactic centre comes out at ``l = 90`` instead of ``0``,
    and the first two rows of the matrix appear swapped and sign-flipped against the
    published one. The test suite compares all nine elements for exactly this reason.
    """
    def rotation_z(degrees: float) -> np.ndarray:
        angle = np.radians(degrees)
        cos, sin = np.cos(angle), np.sin(angle)
        return np.array([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]])

    def rotation_x(degrees: float) -> np.ndarray:
        angle = np.radians(degrees)
        cos, sin = np.cos(angle), np.sin(angle)
        return np.array([[1.0, 0.0, 0.0], [0.0, cos, sin], [0.0, -sin, cos]])

    return (
        rotation_z(90.0 - GALACTIC_LONGITUDE_OF_NCP_DEG)
        @ rotation_x(90.0 - NORTH_GALACTIC_POLE_DEC_DEG)
        @ rotation_z(NORTH_GALACTIC_POLE_RA_DEG + 90.0)
    )


def equatorial_to_cartesian(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Unit vectors from right ascension and declination, shape ``(N, 3)``."""
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    cos_dec = np.cos(dec)
    return np.stack([cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)], axis=-1)


def distance_from_parallax(
    parallax_mas: np.ndarray,
    parallax_over_error: np.ndarray,
    minimum_snr: float = DEFAULT_PARALLAX_SNR,
) -> np.ndarray:
    r"""Distance in parsecs from parallax in milliarcseconds.

    .. math::  d\,[\mathrm{pc}] = \frac{1000}{\varpi\,[\mathrm{mas}]}

    **Returns ``nan``** for any star whose parallax signal-to-noise falls below
    ``minimum_snr``, or whose parallax is not positive. That is deliberate and is the
    whole point of this function existing rather than being written inline:

    * a negative parallax is a real measurement of a distant star, not a corrupt row,
      and inverting it produces a negative distance that will silently poison
      everything downstream;
    * even for positive parallaxes the reciprocal is a biased estimator, badly so once
      the fractional error passes about 20%.

    Returning ``nan`` makes the discarded stars visible to the caller instead of
    quietly absent. Anything needing distances for low-precision parallaxes needs a
    proper Bayesian distance estimate with a galactic prior, which is out of scope
    here and is stated rather than approximated.
    """
    parallax = np.asarray(parallax_mas, dtype=float)
    snr = np.asarray(parallax_over_error, dtype=float)

    usable = np.isfinite(parallax) & (parallax > 0) & np.isfinite(snr) & (snr >= minimum_snr)
    return np.where(usable, 1000.0 / np.where(usable, parallax, 1.0), np.nan)


def absolute_magnitude(
    apparent_magnitude: np.ndarray, distance_parsec: np.ndarray
) -> np.ndarray:
    r"""Absolute magnitude from apparent magnitude and distance.

    .. math::  M = m - 5\log_{10}(d) + 5

    Absolute magnitude is what a star's brightness would be at a standard 10 parsecs,
    so it strips out distance and leaves intrinsic luminosity. Plotting it against
    colour gives the Hertzsprung-Russell diagram, in which stars fall into a narrow
    main sequence with giants above and white dwarfs below — structure that was
    invisible until distances existed.

    Note the sign convention: magnitudes run *backwards*, so a smaller number is a
    brighter star.
    """
    distance = np.asarray(distance_parsec, dtype=float)
    valid = np.isfinite(distance) & (distance > 0)
    return np.where(
        valid,
        np.asarray(apparent_magnitude, dtype=float)
        - 5.0 * np.log10(np.where(valid, distance, 1.0))
        + 5.0,
        np.nan,
    )


def bp_rp_to_rgb(bp_rp: np.ndarray) -> np.ndarray:
    """Approximate display colour from Gaia's BP-RP colour index, shape ``(N, 3)``.

    A rough perceptual mapping for rendering, **not** a calibrated conversion: real
    colour depends on temperature, reddening by interstellar dust, and the response of
    whatever is doing the looking. It interpolates blue-white through white to orange
    across the range that holds nearly all main-sequence stars.

    Values are clipped to ``[-0.5, 3.0]``, which covers hot blue stars through cool
    red dwarfs.
    """
    index = np.clip(np.asarray(bp_rp, dtype=float), -0.5, 3.0)
    # Anchors: hot blue-white, white, yellow-white, orange, deep orange-red.
    stops = np.array([-0.5, 0.4, 0.8, 1.5, 3.0])
    colours = np.array(
        [
            [0.61, 0.71, 1.00],
            [0.92, 0.94, 1.00],
            [1.00, 0.97, 0.87],
            [1.00, 0.85, 0.66],
            [1.00, 0.70, 0.50],
        ]
    )
    return np.stack(
        [np.interp(index, stops, colours[:, channel]) for channel in range(3)], axis=-1
    )


def _to_float(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return float("nan")
    try:
        return float(stripped)
    except ValueError:
        return float("nan")


def parse_gaia_csv(text: str) -> GaiaStars:
    """Parse the archive's CSV response.

    Raises:
        ValueError: If the expected columns are absent, which usually means a TAP
            error document came back instead of data.
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "source_id" not in reader.fieldnames:
        preview = text[:200].replace("\n", " ")
        raise ValueError(f"response does not look like a Gaia result: {preview!r}")

    rows = list(reader)
    if not rows:
        raise ValueError("Gaia response contained no rows")

    def column(name: str) -> np.ndarray:
        return np.array([_to_float(row.get(name, "")) for row in rows])

    return GaiaStars(
        source_id=np.array([row["source_id"].strip() for row in rows]),
        ra=column("ra"),
        dec=column("dec"),
        parallax=column("parallax"),
        parallax_error=column("parallax_error"),
        parallax_over_error=column("parallax_over_error"),
        g_magnitude=column("phot_g_mean_mag"),
        bp_rp=column("bp_rp"),
        pmra=column("pmra"),
        pmdec=column("pmdec"),
        radial_velocity=column("radial_velocity"),
    )


def fetch_gaia_sample(
    limit: int = 120_000,
    magnitude_limit: float = 11.0,
    minimum_snr: float = 5.0,
    timeout: float = 300.0,
    destination: Path | None = None,
) -> Path:
    """Download a magnitude-limited Gaia sample and cache it.

    Args:
        limit: Maximum rows. The archive caps synchronous queries, so this is also a
            politeness limit.
        magnitude_limit: Faintest apparent G magnitude to include. Brighter cuts give
            smaller, more local samples.
        minimum_snr: Parallax signal-to-noise floor applied *in the query*, which
            keeps the download small. A second, stricter cut is applied at analysis
            time by :func:`distance_from_parallax`.
        timeout: Seconds. Gaia queries can be slow.
        destination: Override the cache location.

    Returns:
        Path to the written file.

    Raises:
        RuntimeError: On any network failure, with the cause attached.
    """
    target = destination if destination is not None else gaia_cache_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    query = DEFAULT_QUERY.format(limit=limit, snr=minimum_snr, magnitude=magnitude_limit)
    body = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
    ).encode()

    try:
        request = urllib.request.Request(TAP_ENDPOINT, data=body)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            "could not reach the ESA Gaia archive. Only the Milky Way panels need it; "
            "everything else in this project works offline."
        ) from error

    parse_gaia_csv(payload)  # validate before caching
    target.write_text(payload, encoding="utf-8", newline="\n")
    return target


def load_gaia_sample(
    source: Path | None = None,
    download_if_missing: bool = False,
) -> GaiaStars:
    """Load the cached Gaia sample, optionally fetching it first.

    Raises:
        FileNotFoundError: If absent and downloading was not requested.
    """
    path = source if source is not None else gaia_cache_path()

    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"no Gaia snapshot at {path}. Run `python scripts/fetch_gaia.py`."
            )
        fetch_gaia_sample(destination=path)

    return parse_gaia_csv(path.read_text(encoding="utf-8"))
