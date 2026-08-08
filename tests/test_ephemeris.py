"""Tests for elements -> 3D state vectors.

Three independent kinds of check are used, deliberately:

1. **Conservation laws.** Angular momentum and orbital energy must be constant
   around an orbit. These catch sign errors and unit errors in the velocity.
2. **Round-tripping.** Recovering ``a``, ``e`` and ``i`` back out of the state
   vector with textbook formulas must return the inputs. This catches errors in
   the rotation matrix that conservation laws are blind to.
3. **Known values.** A handful of real published numbers (Earth's distance from
   the Sun at J2000, planetary orbital speeds, the Sun's ecliptic longitude)
   catch errors that are self-consistent but wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from orrery.constants import AU_KM, GM_SUN, J2000_JD, SECONDS_PER_DAY
from orrery.elements import PLANET_NAMES, PLANETS
from orrery.ephemeris import (
    body_state,
    ecliptic_to_equatorial,
    orbit_path,
    state_from_elements,
    system_state,
)
from orrery.timescales import julian_date


class TestGeometry:
    @pytest.mark.parametrize("body", PLANET_NAMES)
    def test_distance_lies_between_perihelion_and_aphelion(self, body):
        """The most basic sanity check: the planet is on its own ellipse."""
        elements = PLANETS[body]
        state = body_state(body, J2000_JD)
        distance = float(state.distance_au)

        assert elements.perihelion_au <= distance <= elements.aphelion_au

    @pytest.mark.parametrize("body", PLANET_NAMES)
    def test_apsides_are_reached_over_a_full_orbit(self, body):
        """Sweeping a whole orbit must actually touch perihelion and aphelion."""
        elements = PLANETS[body]
        path = orbit_path(body, J2000_JD, samples=2000)
        distances = np.linalg.norm(path, axis=-1)

        assert distances.min() == pytest.approx(elements.perihelion_au, rel=1e-5)
        assert distances.max() == pytest.approx(elements.aphelion_au, rel=1e-5)

    def test_planets_are_ordered_by_distance_at_j2000(self):
        """No two planetary orbits cross, so the mean distances stay ordered."""
        axes = [PLANETS[body].semi_major_axis_au for body in PLANET_NAMES]
        assert axes == sorted(axes)

    def test_orbit_path_is_closed(self):
        """The first and last sample of a full revolution must coincide."""
        path = orbit_path("mars", J2000_JD, samples=360)
        assert path.shape == (360, 3)
        np.testing.assert_allclose(path[0], path[-1], atol=1e-12)

    def test_earth_orbit_is_essentially_flat(self):
        """Earth defines the ecliptic, so its z-excursion must be ~zero."""
        path = orbit_path("earth", J2000_JD, samples=720)
        assert np.max(np.abs(path[:, 2])) < 1e-4  # AU

    def test_inclined_orbits_leave_the_ecliptic(self):
        """Pluto's 17-degree inclination must show up as a large z-excursion."""
        path = orbit_path("pluto", J2000_JD, samples=720)
        assert np.max(np.abs(path[:, 2])) > 5.0  # AU


class TestConservationLaws:
    @pytest.mark.parametrize("body", PLANET_NAMES)
    def test_angular_momentum_is_constant(self, body):
        """Kepler's second law, stated as a conserved vector: h = r x v."""
        elements = PLANETS[body].at(J2000_JD)
        mean_anomalies = np.linspace(0.0, 2.0 * np.pi, 200)

        state = state_from_elements(
            elements["a"],
            elements["e"],
            elements["i"],
            elements["Omega"],
            elements["omega"],
            mean_anomalies,
        )
        h = state.specific_angular_momentum

        # Constant in direction as well as magnitude: the orbit is planar.
        np.testing.assert_allclose(h, np.broadcast_to(h[0], h.shape), rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("body", PLANET_NAMES)
    def test_angular_momentum_matches_the_analytic_value(self, body):
        """|h| = sqrt(GM a (1 - e^2)) --- an independent closed form."""
        elements = PLANETS[body].at(J2000_JD)
        state = body_state(body, J2000_JD)

        expected = np.sqrt(GM_SUN * elements["a"] * (1.0 - elements["e"] ** 2))
        actual = np.linalg.norm(state.specific_angular_momentum, axis=-1)

        assert float(actual) == pytest.approx(float(expected), rel=1e-12)

    @pytest.mark.parametrize("body", PLANET_NAMES)
    def test_orbital_energy_is_constant_and_negative(self, body):
        """Energy depends only on a, so it must not vary around the orbit."""
        elements = PLANETS[body].at(J2000_JD)
        mean_anomalies = np.linspace(0.0, 2.0 * np.pi, 200)

        state = state_from_elements(
            elements["a"],
            elements["e"],
            elements["i"],
            elements["Omega"],
            elements["omega"],
            mean_anomalies,
        )
        energy = state.specific_orbital_energy()

        assert np.all(energy < 0.0)  # bound orbit
        np.testing.assert_allclose(energy, -GM_SUN / (2.0 * elements["a"]), rtol=1e-10)

    @pytest.mark.parametrize("body", PLANET_NAMES)
    def test_vis_viva_equation(self, body):
        """v^2 = GM (2/r - 1/a), the energy law written for speed."""
        elements = PLANETS[body].at(J2000_JD)
        mean_anomalies = np.linspace(0.0, 2.0 * np.pi, 100)

        state = state_from_elements(
            elements["a"],
            elements["e"],
            elements["i"],
            elements["Omega"],
            elements["omega"],
            mean_anomalies,
        )
        expected = GM_SUN * (2.0 / state.distance_au - 1.0 / elements["a"])

        np.testing.assert_allclose(state.speed_au_per_day**2, expected, rtol=1e-10)

    def test_planet_is_fastest_at_perihelion(self):
        """Kepler's second law as an inequality, on the most eccentric planet."""
        elements = PLANETS["mercury"].at(J2000_JD)
        at_perihelion = state_from_elements(
            elements["a"], elements["e"], elements["i"],
            elements["Omega"], elements["omega"], 0.0,
        )
        at_aphelion = state_from_elements(
            elements["a"], elements["e"], elements["i"],
            elements["Omega"], elements["omega"], np.pi,
        )

        assert float(at_perihelion.speed_au_per_day) > float(at_aphelion.speed_au_per_day)
        assert float(at_perihelion.distance_au) < float(at_aphelion.distance_au)


class TestRoundTrip:
    """Recover the orbital elements from the state vector and compare.

    Uses the standard inverse formulas, which are independent of the forward
    code path being tested:

        a = -GM / (2 E)                  from the energy
        e = sqrt(1 + 2 E h^2 / GM^2)
        i = atan2(hypot(h_x, h_y), h_z)

    The inclination uses ``atan2`` rather than the more commonly quoted
    ``arccos(h_z / |h|)``. They are equivalent in exact arithmetic, but ``arccos``
    has an infinite derivative at 1, so for a near-coplanar orbit like Earth's it
    amplifies the last bits of rounding error into a visible discrepancy.
    """

    @pytest.mark.parametrize("body", PLANET_NAMES)
    @pytest.mark.parametrize("mean_anomaly", [0.0, 1.0, 2.5, -2.0])
    def test_elements_are_recovered(self, body, mean_anomaly):
        elements = PLANETS[body].at(J2000_JD)
        a_in = float(elements["a"])
        e_in = float(elements["e"])
        i_in = float(elements["i"])

        state = state_from_elements(
            a_in, e_in, i_in, float(elements["Omega"]), float(elements["omega"]), mean_anomaly
        )

        energy = float(state.specific_orbital_energy())
        h_vec = state.specific_angular_momentum
        h_mag = float(np.linalg.norm(h_vec))

        a_out = -GM_SUN / (2.0 * energy)
        e_out = np.sqrt(max(0.0, 1.0 + 2.0 * energy * h_mag**2 / GM_SUN**2))
        i_out = np.arctan2(np.hypot(h_vec[0], h_vec[1]), h_vec[2])

        assert a_out == pytest.approx(a_in, rel=1e-10)
        assert e_out == pytest.approx(e_in, abs=1e-10)
        # Inclination is unsigned in this formulation, hence the abs().
        assert float(i_out) == pytest.approx(abs(i_in), abs=1e-10)


class TestAgainstPublishedValues:
    """Checks against numbers that do not come from this codebase."""

    def test_earth_sun_distance_at_j2000(self):
        """Earth was 0.9833 AU from the Sun on 2000-01-01 (just before perihelion)."""
        state = body_state("earth", J2000_JD)
        assert float(state.distance_au) == pytest.approx(0.9833, abs=0.0005)

    def test_earth_reaches_perihelion_in_early_january(self):
        """Perihelion falls on 2-5 January, not at the solstice."""
        days = julian_date(2026, 1, 1) + np.arange(0, 365, 1.0)
        distances = body_state("earth", days).distance_au

        day_of_minimum = int(np.argmin(distances))
        assert day_of_minimum <= 6, f"perihelion found {day_of_minimum} days into the year"

    def test_earth_reaches_aphelion_in_early_july(self):
        days = julian_date(2026, 1, 1) + np.arange(0, 365, 1.0)
        distances = body_state("earth", days).distance_au

        day_of_maximum = int(np.argmax(distances))
        assert 180 <= day_of_maximum <= 190  # ~4 July

    def test_suns_ecliptic_longitude_at_j2000(self):
        """Seen from Earth, the Sun sat at ecliptic longitude ~280.4 deg at J2000.

        Earth's heliocentric longitude is that value minus 180 degrees.
        """
        state = body_state("earth", J2000_JD)
        x, y, _ = state.position
        earth_longitude = np.degrees(np.arctan2(y, x)) % 360.0
        sun_longitude = (earth_longitude + 180.0) % 360.0

        assert float(sun_longitude) == pytest.approx(280.4, abs=0.2)

    @pytest.mark.parametrize(
        ("body", "low_km_s", "high_km_s"),
        [
            # Published perihelion/aphelion speed ranges, loosened slightly.
            ("mercury", 38.0, 59.0),
            ("venus", 34.0, 36.0),
            ("earth", 29.0, 30.5),
            ("mars", 21.5, 27.0),
            ("jupiter", 12.0, 13.8),
            ("saturn", 9.0, 10.3),
            ("uranus", 6.4, 7.2),
            ("neptune", 5.3, 5.6),
        ],
    )
    def test_orbital_speeds_match_published_ranges(self, body, low_km_s, high_km_s):
        state = body_state(body, J2000_JD)
        speed = float(state.speed_km_per_s)
        assert low_km_s <= speed <= high_km_s, f"{body}: {speed:.2f} km/s"

    @pytest.mark.parametrize(
        ("body", "expected_days"),
        [
            ("mercury", 87.969),
            ("venus", 224.701),
            ("earth", 365.256),
            ("mars", 686.980),
            ("jupiter", 4332.589),
            ("saturn", 10_759.22),
            ("uranus", 30_685.4),
            ("neptune", 60_189.0),
        ],
    )
    def test_orbital_periods_match_published_values(self, body, expected_days):
        """Kepler's third law reproduces the tabulated sidereal periods to <0.5%."""
        assert PLANETS[body].period_days == pytest.approx(expected_days, rel=5e-3)


class TestVectorisation:
    def test_array_of_dates_gives_array_of_states(self):
        dates = J2000_JD + np.linspace(0.0, 365.0, 50)
        state = body_state("mars", dates)

        assert state.position.shape == (50, 3)
        assert state.velocity.shape == (50, 3)
        assert state.distance_au.shape == (50,)

    def test_scalar_date_matches_the_array_result(self):
        dates = J2000_JD + np.array([0.0, 100.0, 200.0])
        batched = body_state("jupiter", dates).position

        for index, jd in enumerate(dates):
            single = body_state("jupiter", float(jd)).position
            np.testing.assert_allclose(single, batched[index], rtol=1e-13)

    def test_system_state_covers_all_planets(self):
        states = system_state(J2000_JD)
        assert set(states) == set(PLANET_NAMES)

    def test_unknown_body_raises(self):
        with pytest.raises(KeyError, match="unknown body"):
            body_state("vulcan", J2000_JD)


class TestFrameRotation:
    def test_rotation_preserves_length(self):
        """A frame change must not move anything, only relabel its coordinates."""
        vectors = body_state("mars", J2000_JD + np.arange(10.0)).position
        rotated = ecliptic_to_equatorial(vectors)

        np.testing.assert_allclose(
            np.linalg.norm(rotated, axis=-1), np.linalg.norm(vectors, axis=-1), rtol=1e-14
        )

    def test_shared_x_axis_is_unchanged(self):
        """Both frames share the vernal equinox direction, so x is invariant."""
        vectors = np.array([[1.0, 0.0, 0.0], [3.0, 2.0, -1.0]])
        rotated = ecliptic_to_equatorial(vectors)
        np.testing.assert_allclose(rotated[:, 0], vectors[:, 0], rtol=1e-14)

    def test_ecliptic_pole_tilts_by_the_obliquity(self):
        """The ecliptic pole must land 23.44 degrees from the equatorial pole."""
        pole = ecliptic_to_equatorial(np.array([0.0, 0.0, 1.0]))
        angle = np.degrees(np.arccos(np.clip(pole[2], -1.0, 1.0)))
        assert float(angle) == pytest.approx(23.4393, abs=1e-3)


class TestUnitConversions:
    def test_speed_conversion_is_consistent(self):
        state = body_state("earth", J2000_JD)
        expected = float(state.speed_au_per_day) * AU_KM / SECONDS_PER_DAY
        assert float(state.speed_km_per_s) == pytest.approx(expected, rel=1e-14)
