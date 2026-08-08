"""Tests for the Gaia module.

Two things are worth testing carefully here.

The **galactic rotation** is built from three angles rather than pasted in as nine
numbers, so it needs checking against the published matrix — and it caught a real
error while being written: the third rotation angle was a quarter turn out, which put
the galactic centre at l = 90 instead of l = 0 while leaving the pole correct and the
matrix perfectly orthogonal. Nothing but an external comparison would have found it.

The **parallax handling** is the other. Its whole purpose is to *refuse* to produce a
number when the measurement cannot support one, so the tests assert what it discards
as carefully as what it returns.
"""

from __future__ import annotations

import numpy as np
import pytest

from orrery.gaia import (
    DEFAULT_PARALLAX_SNR,
    GALACTIC_LONGITUDE_OF_NCP_DEG,
    NORTH_GALACTIC_POLE_DEC_DEG,
    NORTH_GALACTIC_POLE_RA_DEG,
    PARSEC_IN_AU,
    absolute_magnitude,
    bp_rp_to_rgb,
    distance_from_parallax,
    equatorial_to_cartesian,
    gaia_cache_path,
    icrs_to_galactic_matrix,
    load_gaia_sample,
    parse_gaia_csv,
)

#: The ICRS-to-galactic matrix as published in the Gaia documentation.
PUBLISHED_MATRIX = np.array(
    [
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669],
    ]
)

needs_gaia = pytest.mark.skipif(
    not gaia_cache_path().exists(),
    reason="no Gaia snapshot; run scripts/fetch_gaia.py",
)


def galactic_coordinates(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Galactic longitude and latitude of one equatorial direction, in degrees."""
    vector = icrs_to_galactic_matrix() @ equatorial_to_cartesian(
        np.array([ra_deg]), np.array([dec_deg])
    )[0]
    longitude = float(np.degrees(np.arctan2(vector[1], vector[0])) % 360.0)
    latitude = float(np.degrees(np.arcsin(np.clip(vector[2], -1.0, 1.0))))
    return longitude, latitude


class TestGalacticRotation:
    def test_matches_the_published_matrix(self):
        """All nine elements, because a wrong angle can leave most of them right."""
        np.testing.assert_allclose(
            icrs_to_galactic_matrix(), PUBLISHED_MATRIX, atol=1e-14
        )

    def test_is_a_proper_rotation(self):
        matrix = icrs_to_galactic_matrix()
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-15)
        assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-15)

    def test_galactic_centre_lands_at_the_origin(self):
        """The check that caught the quarter-turn error.

        Orthogonality and the pole were both satisfied by the wrong matrix; only the
        centre's longitude revealed it.
        """
        longitude, latitude = galactic_coordinates(266.40510, -28.93617)
        assert min(longitude, 360.0 - longitude) < 0.01
        assert abs(latitude) < 0.01

    def test_north_galactic_pole_lands_at_the_pole(self):
        _, latitude = galactic_coordinates(
            NORTH_GALACTIC_POLE_RA_DEG, NORTH_GALACTIC_POLE_DEC_DEG
        )
        assert latitude == pytest.approx(90.0, abs=1e-4)

    def test_anticentre(self):
        longitude, latitude = galactic_coordinates(86.40510, 28.93617)
        assert longitude == pytest.approx(180.0, abs=0.01)
        assert abs(latitude) < 0.01

    def test_the_third_angle_is_load_bearing(self):
        """Dropping the ``90 -`` reproduces the original bug, so it is not cosmetic."""
        def rotation_z(degrees):
            angle = np.radians(degrees)
            cos, sin = np.cos(angle), np.sin(angle)
            return np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1.0]])

        def rotation_x(degrees):
            angle = np.radians(degrees)
            cos, sin = np.cos(angle), np.sin(angle)
            return np.array([[1.0, 0, 0], [0, cos, sin], [0, -sin, cos]])

        naive = (
            rotation_z(-GALACTIC_LONGITUDE_OF_NCP_DEG)
            @ rotation_x(90.0 - NORTH_GALACTIC_POLE_DEC_DEG)
            @ rotation_z(NORTH_GALACTIC_POLE_RA_DEG + 90.0)
        )
        centre = naive @ equatorial_to_cartesian(
            np.array([266.40510]), np.array([-28.93617])
        )[0]
        longitude = np.degrees(np.arctan2(centre[1], centre[0])) % 360.0

        assert longitude == pytest.approx(90.0, abs=0.01)


class TestEquatorialToCartesian:
    def test_produces_unit_vectors(self):
        generator = np.random.default_rng(3)
        ra = generator.uniform(0, 360, 200)
        dec = generator.uniform(-90, 90, 200)

        norms = np.linalg.norm(equatorial_to_cartesian(ra, dec), axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-15)

    @pytest.mark.parametrize(
        ("ra", "dec", "expected"),
        [
            (0.0, 0.0, [1.0, 0.0, 0.0]),  # vernal equinox
            (90.0, 0.0, [0.0, 1.0, 0.0]),
            (0.0, 90.0, [0.0, 0.0, 1.0]),  # north celestial pole
            (0.0, -90.0, [0.0, 0.0, -1.0]),
        ],
    )
    def test_known_directions(self, ra, dec, expected):
        result = equatorial_to_cartesian(np.array([ra]), np.array([dec]))[0]
        np.testing.assert_allclose(result, expected, atol=1e-15)


class TestParallaxHandling:
    def test_inverts_a_clean_parallax(self):
        """100 mas is 10 pc, by the definition of the parsec."""
        distance = distance_from_parallax(np.array([100.0]), np.array([500.0]))
        assert distance[0] == pytest.approx(10.0)

    def test_one_parsec_is_one_arcsecond(self):
        distance = distance_from_parallax(np.array([1000.0]), np.array([500.0]))
        assert distance[0] == pytest.approx(1.0)

    def test_rejects_negative_parallax(self):
        """Negative parallaxes are real measurements, and inverting them is nonsense.

        The failure mode this prevents is silent: a negative distance propagates into
        positions, magnitudes and plots without ever raising.
        """
        distance = distance_from_parallax(np.array([-0.5]), np.array([50.0]))
        assert np.isnan(distance[0])

    def test_rejects_low_signal_to_noise(self):
        """Above roughly 20% fractional error the reciprocal is badly biased."""
        distance = distance_from_parallax(
            np.array([1.0, 1.0]), np.array([DEFAULT_PARALLAX_SNR - 1, DEFAULT_PARALLAX_SNR + 1])
        )
        assert np.isnan(distance[0])
        assert np.isfinite(distance[1])

    def test_threshold_is_adjustable(self):
        relaxed = distance_from_parallax(np.array([1.0]), np.array([3.0]), minimum_snr=2.0)
        assert np.isfinite(relaxed[0])

    def test_rejected_rows_are_nan_not_absent(self):
        """Shape is preserved so callers can see *which* stars were dropped."""
        parallax = np.array([10.0, -1.0, 5.0, np.nan])
        snr = np.array([100.0, 100.0, 1.0, 100.0])

        distance = distance_from_parallax(parallax, snr)
        assert distance.shape == parallax.shape
        assert np.isfinite(distance[0])
        assert np.all(np.isnan(distance[1:]))

    def test_parsec_constant(self):
        """A parsec is the distance at which one AU subtends one arcsecond."""
        assert PARSEC_IN_AU == pytest.approx(206_264.806, rel=1e-6)


class TestAbsoluteMagnitude:
    def test_at_ten_parsecs_absolute_equals_apparent(self):
        """The definition: absolute magnitude is apparent magnitude at 10 pc."""
        result = absolute_magnitude(np.array([5.0]), np.array([10.0]))
        assert result[0] == pytest.approx(5.0)

    def test_distance_modulus(self):
        """Ten times further is five magnitudes fainter."""
        near = absolute_magnitude(np.array([5.0]), np.array([10.0]))
        far = absolute_magnitude(np.array([5.0]), np.array([100.0]))
        assert float(near[0] - far[0]) == pytest.approx(5.0)

    def test_propagates_missing_distances(self):
        result = absolute_magnitude(np.array([5.0, 5.0]), np.array([np.nan, -3.0]))
        assert np.all(np.isnan(result))


class TestColourMapping:
    def test_shape_and_range(self):
        colours = bp_rp_to_rgb(np.linspace(-1.0, 4.0, 50))
        assert colours.shape == (50, 3)
        assert np.all(colours >= 0.0) and np.all(colours <= 1.0)

    def test_red_channel_rises_and_blue_falls_with_colour_index(self):
        """Larger BP-RP means a cooler, redder star."""
        blue_end = bp_rp_to_rgb(np.array([-0.3]))[0]
        red_end = bp_rp_to_rgb(np.array([2.5]))[0]

        assert red_end[0] > blue_end[0]
        assert red_end[2] < blue_end[2]

    def test_clips_out_of_range_values(self):
        extreme = bp_rp_to_rgb(np.array([-99.0, 99.0]))
        assert np.all(np.isfinite(extreme))


class TestParsing:
    def test_rejects_a_non_gaia_response(self):
        with pytest.raises(ValueError, match="does not look like"):
            parse_gaia_csv("<VOTABLE><INFO>query error</INFO></VOTABLE>")

    def test_rejects_an_empty_result(self):
        header = "source_id,ra,dec,parallax,parallax_error,parallax_over_error,"
        header += "phot_g_mean_mag,bp_rp,pmra,pmdec,radial_velocity\n"
        with pytest.raises(ValueError, match="no rows"):
            parse_gaia_csv(header)

    def test_blank_cells_become_nan(self):
        header = "source_id,ra,dec,parallax,parallax_error,parallax_over_error,"
        header += "phot_g_mean_mag,bp_rp,pmra,pmdec,radial_velocity\n"
        row = "123,10.0,20.0,5.0,0.1,50.0,7.5,0.8,1.0,2.0,\n"

        stars = parse_gaia_csv(header + row)
        assert len(stars) == 1
        assert np.isnan(stars.radial_velocity[0])
        assert stars.parallax[0] == 5.0


@needs_gaia
class TestRealCatalogue:
    @staticmethod
    def _stars():
        return load_gaia_sample()

    def test_sample_is_substantial(self):
        assert len(self._stars()) > 10_000

    def test_distances_are_plausible(self):
        """Nothing closer than Proxima, nothing implausibly far for this cut."""
        distance = self._stars().distance_parsec()
        finite = distance[np.isfinite(distance)]

        assert finite.min() > 1.0  # Proxima Centauri is 1.30 pc
        assert np.median(finite) < 2000.0

    def test_positions_preserve_distance(self):
        """The rotation must not change how far away anything is."""
        stars = self._stars()
        positions = stars.cartesian_galactic()
        distance = stars.distance_parsec()

        usable = np.isfinite(distance)
        np.testing.assert_allclose(
            np.linalg.norm(positions[usable], axis=-1), distance[usable], rtol=1e-12
        )

    def test_the_main_sequence_appears_in_the_nearby_volume(self):
        """Redder is fainter — the diagonal that defines the main sequence.

        Checked **within 25 parsecs only**. The first version of this test used the
        whole sample and failed, correlation −0.49 instead of the expected positive:
        not a bug, but Malmquist bias. This is a magnitude-limited sample reaching
        only G = 8.6, so beyond a hundred parsecs nothing but giants is bright enough
        to appear, and giants are red *and* luminous — the opposite of the main
        sequence trend.

        Nearby, where the sample is close to volume-limited, the main sequence is
        clean. That makes this an end-to-end check of parallaxes, distances,
        magnitudes and colours together: get any one wrong and the correlation dies.
        """
        stars = self._stars()
        absolute = stars.absolute_g()
        distance = stars.distance_parsec()

        nearby = (
            np.isfinite(absolute) & np.isfinite(stars.bp_rp) & (distance < 25.0)
        )
        assert nearby.sum() > 200

        correlation = np.corrcoef(stars.bp_rp[nearby], absolute[nearby])[0, 1]
        assert correlation > 0.7

    def test_malmquist_bias_flips_the_correlation_with_distance(self):
        """The selection effect itself, measured rather than described.

        Real physics does not reverse with distance. This does, which is proof the
        trend is imposed by the survey's brightness limit — the single most important
        caveat on anything read off this catalogue.
        """
        stars = self._stars()
        absolute = stars.absolute_g()
        distance = stars.distance_parsec()
        usable = np.isfinite(absolute) & np.isfinite(stars.bp_rp)

        def correlation(low: float, high: float) -> float:
            shell = usable & (distance >= low) & (distance < high)
            return float(np.corrcoef(stars.bp_rp[shell], absolute[shell])[0, 1])

        assert correlation(0.0, 25.0) > 0.5
        assert correlation(200.0, 500.0) < 0.0

    def test_the_faint_limit_recedes_with_distance(self):
        """The mechanism behind the flip: distant faint stars are simply not in the sample."""
        stars = self._stars()
        absolute = stars.absolute_g()
        distance = stars.distance_parsec()
        usable = np.isfinite(absolute)

        near = absolute[usable & (distance < 25.0)].max()
        far = absolute[usable & (distance >= 200.0) & (distance < 500.0)].max()

        # "Faintest visible" brightens by many magnitudes as distance grows.
        assert near - far > 5.0

    def test_giants_sit_above_the_main_sequence(self):
        """Red *and* bright means a giant — the branch that broke stellar theory open."""
        stars = self._stars()
        absolute = stars.absolute_g()
        red = stars.bp_rp > 1.2

        giants = red & np.isfinite(absolute) & (absolute < 2.0)
        assert giants.sum() > 100

    def test_stars_spread_over_the_whole_sky(self):
        stars = self._stars()
        assert stars.ra.min() < 10.0 and stars.ra.max() > 350.0
        assert stars.dec.min() < -60.0 and stars.dec.max() > 60.0

    def test_sample_concentrates_towards_the_galactic_plane(self):
        """A real stellar sample is not isotropic; the disc shows up in latitude.

        Would fail if the galactic rotation were wrong, and independently of the
        matrix comparison, because it uses the actual star distribution.
        """
        stars = self._stars()
        positions = stars.cartesian_galactic()
        usable = np.isfinite(positions[:, 0])

        distance = np.linalg.norm(positions[usable], axis=-1)
        latitude = np.degrees(np.arcsin(positions[usable][:, 2] / distance))

        near_plane = np.mean(np.abs(latitude) < 20.0)
        assert near_plane > 0.5, f"only {near_plane:.1%} within 20 deg of the plane"

    def test_brightest_returns_the_brightest(self):
        stars = self._stars()
        subset = stars.brightest(100)

        assert len(subset) == 100
        assert subset.g_magnitude.max() <= np.sort(stars.g_magnitude)[99] + 1e-9
