from __future__ import annotations

import math
import unittest

from offline.geodesic import (
    HamiltonianState,
    RayTraceOptions,
    RecordedSurfaceCrossing,
    SurfaceEventOptions,
    trace_null_geodesic,
)
from offline.kerr import (
    KerrKerrSchildMetric,
    KerrOblateTermination,
    kerr_oblate_event_to_ks_cartesian,
    kerr_zamo_camera_ray,
)
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_surface import (
    FINITE_THICKNESS_SURFACE_IDS,
    LOWER_SURFACE_ID,
    LOWER_TARGET_ID,
    SCIENTIFIC_STATUS,
    UPPER_SURFACE_ID,
    UPPER_TARGET_ID,
    KerrFiniteThicknessMultiSurface,
)


def state_at_pseudo_cylindrical_point(
    metric: KerrKerrSchildMetric,
    rho_over_mass: float,
    signed_height_over_mass: float,
) -> HamiltonianState:
    radius_over_mass = math.hypot(
        rho_over_mass,
        signed_height_over_mass,
    )
    theta = math.atan2(rho_over_mass, signed_height_over_mass)
    event = kerr_oblate_event_to_ks_cartesian(
        coordinate_time_m=0.0,
        radius_m=radius_over_mass * metric.mass_m,
        theta_rad=theta,
        phi_ks_rad=0.2,
        spin_a_m=metric.spin_a_m,
    )
    return HamiltonianState(
        event=event,
        covector=(1.0, 1.0, 0.0, 0.0),
    )


def crossing_at(state: HamiltonianState) -> RecordedSurfaceCrossing:
    return RecordedSurfaceCrossing(
        state=state,
        ray_affine_length=1.0,
        segment_index=0,
        segment_affine_length=1.0,
        orientation=-1,
        surface_value=0.0,
        bracket_affine_width=1.0e-10,
        iterations=4,
    )


class KerrFiniteThicknessSurfaceTests(unittest.TestCase):
    def make_adapter(
        self,
        *,
        spin: float = 0.7,
        accretion_ratio: float = 0.05,
        outer_radius: float = 25.0,
    ) -> KerrFiniteThicknessMultiSurface:
        metric = KerrKerrSchildMetric(spin_a_m=spin)
        calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=spin,
            eddington_scaled_mass_accretion_rate=accretion_ratio,
            outer_radius_over_mass=outer_radius,
        )
        return KerrFiniteThicknessMultiSurface(metric, calibration)

    def test_status_is_geometry_only_and_has_no_radial_sidewall(self) -> None:
        self.assertEqual(
            SCIENTIFIC_STATUS["classification"],
            "stationary Kerr finite-thickness first-visible geometry adapter",
        )
        self.assertIs(SCIENTIFIC_STATUS["includesRadialSidewall"], False)
        self.assertIs(SCIENTIFIC_STATUS["acceptsZeroThickness"], False)
        self.assertIs(SCIENTIFIC_STATUS["includesSpectrum"], False)
        self.assertIs(SCIENTIFIC_STATUS["includesReturningRadiation"], False)
        self.assertIs(SCIENTIFIC_STATUS["includesSolvedAtmosphere"], False)
        self.assertIs(
            SCIENTIFIC_STATUS["isGeneralRelativisticMagnetohydrodynamics"],
            False,
        )
        self.assertIn(
            "first-visible event geometry only",
            SCIENTIFIC_STATUS["prohibitedClaim"],
        )
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["classification"] = "mutable"

    def test_physical_upper_and_lower_faces_keep_independent_signed_fields(
        self,
    ) -> None:
        adapter = self.make_adapter()
        rho = 3.0 * adapter.calibration.isco_radius_over_mass
        height = adapter.calibration.photosphere_height_over_mass(rho)
        upper_state = state_at_pseudo_cylindrical_point(
            adapter.metric,
            rho,
            height,
        )
        lower_state = state_at_pseudo_cylindrical_point(
            adapter.metric,
            rho,
            -height,
        )
        self.assertEqual(adapter.surface_ids, FINITE_THICKNESS_SURFACE_IDS)
        self.assertAlmostEqual(
            adapter.value(UPPER_SURFACE_ID, upper_state),
            0.0,
            places=14,
        )
        self.assertLess(adapter.value(LOWER_SURFACE_ID, upper_state), 0.0)
        self.assertAlmostEqual(
            adapter.value(LOWER_SURFACE_ID, lower_state),
            0.0,
            places=14,
        )
        self.assertLess(adapter.value(UPPER_SURFACE_ID, lower_state), 0.0)

        upper_decision = adapter.classify(
            UPPER_SURFACE_ID,
            crossing_at(upper_state),
        )
        lower_decision = adapter.classify(
            LOWER_SURFACE_ID,
            crossing_at(lower_state),
        )
        self.assertTrue(upper_decision.terminates)
        self.assertTrue(lower_decision.terminates)
        self.assertEqual(upper_decision.target_id, UPPER_TARGET_ID)
        self.assertEqual(lower_decision.target_id, LOWER_TARGET_ID)

    def test_radial_extensions_are_transparent_and_do_not_make_sidewalls(
        self,
    ) -> None:
        adapter = self.make_adapter()
        inner = adapter.calibration.isco_radius_over_mass
        outer = adapter.calibration.outer_radius_over_mass
        regions = (
            (0.8 * inner, "inside-isco-transparent"),
            (1.2 * outer, "outside-outer-radius-transparent"),
        )
        for rho, expected_classification in regions:
            height = adapter.auxiliary_photosphere_height_over_mass(rho)
            for surface_id, sign in (
                (UPPER_SURFACE_ID, 1.0),
                (LOWER_SURFACE_ID, -1.0),
            ):
                state = state_at_pseudo_cylindrical_point(
                    adapter.metric,
                    rho,
                    sign * height,
                )
                with self.subTest(rho=rho, surface_id=surface_id):
                    self.assertAlmostEqual(
                        adapter.value(surface_id, state),
                        0.0,
                        places=14,
                    )
                    decision = adapter.classify(
                        surface_id,
                        crossing_at(state),
                    )
                    self.assertFalse(decision.terminates)
                    self.assertEqual(
                        decision.classification,
                        expected_classification,
                    )

        # Crossing a radial boundary through the mid-plane changes neither
        # face sign, so the adapter supplies no implicit vertical sidewall.
        for boundary in (inner, outer):
            below = boundary * (1.0 - 1.0e-6)
            above = boundary * (1.0 + 1.0e-6)
            for surface_id in FINITE_THICKNESS_SURFACE_IDS:
                values = tuple(
                    adapter.value(
                        surface_id,
                        state_at_pseudo_cylindrical_point(
                            adapter.metric,
                            rho,
                            0.0,
                        ),
                    )
                    for rho in (below, above)
                )
                with self.subTest(boundary=boundary, surface_id=surface_id):
                    self.assertLess(values[0], 0.0)
                    self.assertLess(values[1], 0.0)

    def test_spin_axis_surface_values_are_finite_and_transparent(self) -> None:
        adapter = self.make_adapter()
        rho = 0.0
        height = adapter.auxiliary_photosphere_height_over_mass(rho)
        for surface_id, signed_height in (
            (UPPER_SURFACE_ID, height),
            (LOWER_SURFACE_ID, -height),
        ):
            state = state_at_pseudo_cylindrical_point(
                adapter.metric,
                rho,
                signed_height,
            )
            with self.subTest(surface_id=surface_id):
                value = adapter.value(surface_id, state)
                self.assertTrue(math.isfinite(value))
                self.assertAlmostEqual(value, 0.0, places=14)
                decision = adapter.classify(
                    surface_id,
                    crossing_at(state),
                )
                self.assertFalse(decision.terminates)
                self.assertEqual(
                    decision.classification,
                    "inside-isco-transparent",
                )

    def test_exact_kerr_rays_certify_first_upper_and_lower_face(self) -> None:
        adapter = self.make_adapter()
        termination = KerrOblateTermination.horizon_worldtube(
            adapter.metric,
            escape_radius_m=50.0,
            offset_m=0.02,
        )
        ray_options = RayTraceOptions(
            absolute_tolerance=5.0e-10,
            relative_tolerance=5.0e-10,
            initial_step=0.05,
            maximum_step=0.25,
            maximum_affine_length=300.0,
            null_residual_limit=2.0e-7,
            record_path=True,
        )
        surface_options = SurfaceEventOptions(
            absolute_tolerance=5.0e-10,
            relative_tolerance=5.0e-10,
            null_residual_limit=2.0e-7,
            subdivisions_per_segment=4,
        )
        cases = (
            (1.1, -0.5, UPPER_SURFACE_ID, UPPER_TARGET_ID),
            (math.pi - 1.1, 0.5, LOWER_SURFACE_ID, LOWER_TARGET_ID),
        )
        for theta, screen_y, expected_surface, expected_target in cases:
            initial = kerr_zamo_camera_ray(
                adapter.metric,
                observer_radius_m=30.0,
                theta_rad=theta,
                screen_x=0.5,
                screen_y=screen_y,
            )
            result = trace_null_geodesic(
                adapter.metric,
                initial,
                termination=termination,
                multi_interior_surface=adapter,
                surface_options=surface_options,
                options=ray_options,
            )
            with self.subTest(expected_surface=expected_surface):
                self.assertEqual(
                    result.outcome,
                    "opaque-finite-thickness-disk-hit",
                    result.failure_reason,
                )
                self.assertEqual(result.terminal_target_id, expected_target)
                trace = result.multi_surface_trace
                self.assertIsNotNone(trace)
                assert trace is not None
                self.assertTrue(trace.topology_converged)
                self.assertEqual(len(trace.crossings), 1)
                self.assertEqual(trace.crossings[0].surface_id, expected_surface)
                self.assertTrue(trace.crossings[0].decision.terminates)
                self.assertLessEqual(
                    abs(trace.crossings[0].crossing.surface_value),
                    surface_options.surface_value_tolerance,
                )
                self.assertGreater(trace.probe_reintegrations, 0)

    def test_zero_thickness_and_mismatched_metric_fail_before_tracing(self) -> None:
        metric = KerrKerrSchildMetric(spin_a_m=0.7)
        zero = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=0.0,
            outer_radius_over_mass=25.0,
        )
        with self.assertRaisesRegex(ValueError, "faces coincide"):
            KerrFiniteThicknessMultiSurface(metric, zero)

        mismatch = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.5,
            eddington_scaled_mass_accretion_rate=0.05,
            outer_radius_over_mass=25.0,
        )
        with self.assertRaisesRegex(ValueError, "spin are inconsistent"):
            KerrFiniteThicknessMultiSurface(metric, mismatch)

    def test_bad_ids_states_and_radii_fail_closed(self) -> None:
        adapter = self.make_adapter()
        rho = 3.0 * adapter.calibration.isco_radius_over_mass
        height = adapter.calibration.photosphere_height_over_mass(rho)
        state = state_at_pseudo_cylindrical_point(adapter.metric, rho, height)
        with self.assertRaises(ValueError):
            adapter.value("unknown", state)
        with self.assertRaises(TypeError):
            adapter.value(UPPER_SURFACE_ID, object())
        for invalid in (-1.0, math.nan, math.inf, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                adapter.auxiliary_photosphere_height_over_mass(invalid)


if __name__ == "__main__":
    unittest.main()
