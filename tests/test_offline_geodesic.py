from __future__ import annotations

import math
import unittest
from dataclasses import replace

from offline.geodesic import (
    InteriorSurfaceDecision,
    HamiltonianState,
    RadialTermination,
    RayPathSegment,
    RayTraceOptions,
    SurfaceEventError,
    SurfaceEventOptions,
    TerminationCrossing,
    hamiltonian_null_residual,
    locate_recorded_surface_crossings,
    static_schwarzschild_camera_ray,
    trace_null_geodesic,
    trace_refined_null_geodesic,
)
from offline.spacetime import (
    MetricSample,
    MinkowskiMetric,
    SchwarzschildKerrSchildMetric,
    bilinear,
)


class RecordingTimeDependentMinkowski:
    source_id = "test-time-dependent-minkowski"
    time_dependent = True

    def __init__(self) -> None:
        self.events: list[tuple[float, float, float, float]] = []
        self._metric = MinkowskiMetric()

    def sample(self, event: tuple[float, float, float, float]) -> MetricSample:
        self.events.append(event)
        return self._metric.sample(event)


class ExcessInterpolationErrorMinkowski:
    source_id = "test-excess-interpolation-error"
    time_dependent = True

    def sample(self, event: tuple[float, float, float, float]) -> MetricSample:
        exact = MinkowskiMetric().sample(event)
        return MetricSample(
            covariant=exact.covariant,
            inverse=exact.inverse,
            inverse_derivatives=exact.inverse_derivatives,
            interpolation_error=2.0e-4,
        )


class StepSensitiveTargetTermination:
    def classify_initial(
        self,
        state: HamiltonianState,
    ) -> tuple[str, str] | None:
        return None

    def crossing(
        self,
        previous: HamiltonianState,
        current: HamiltonianState,
    ) -> TerminationCrossing | None:
        before = previous.event[1] - 1.0
        after = current.event[1] - 1.0
        if before < 0.0 <= after:
            target = (
                "fine-target"
                if current.event[1] - previous.event[1] < 0.5
                else "coarse-target"
            )
            return TerminationCrossing("escaped", target, before, after)
        return None

    def value(
        self,
        state: HamiltonianState,
        crossing: TerminationCrossing,
    ) -> float:
        return state.event[1] - 1.0

    def needs_refinement(
        self,
        previous: HamiltonianState,
        current: HamiltonianState,
    ) -> bool:
        return False


class PolynomialInteriorSurface:
    def __init__(
        self,
        roots: tuple[float, ...],
        *,
        terminal_root: float,
    ) -> None:
        self.roots = roots
        self.terminal_root = terminal_root

    def value(self, state: HamiltonianState) -> float:
        return math.prod(state.event[1] - root for root in self.roots)

    def classify(self, crossing):
        coordinate = crossing.state.event[1]
        if abs(coordinate - self.terminal_root) < 0.025:
            return InteriorSurfaceDecision(
                "opaque",
                "surface-hit",
                "test-opaque-surface",
            )
        return InteriorSurfaceDecision(
            "transparent-inner" if coordinate < 0.3 else "transparent-outer"
        )


class TangentInteriorSurface:
    def value(self, state: HamiltonianState) -> float:
        return (state.event[1] - 0.25) ** 2

    def classify(self, crossing):
        return InteriorSurfaceDecision(
            "opaque",
            "surface-hit",
            "test-tangent-surface",
        )


class EndpointPlaneSurface:
    def __init__(self, *, opaque: bool) -> None:
        self.opaque = opaque

    def value(self, state: HamiltonianState) -> float:
        return state.event[3]

    def classify(self, crossing):
        if self.opaque:
            return InteriorSurfaceDecision(
                "opaque",
                "surface-hit",
                "test-endpoint-opaque",
            )
        return InteriorSurfaceDecision("transparent")


class XWorldtubeTermination:
    def __init__(self, outcome: str) -> None:
        self.outcome = outcome
        self.target_id = f"test-{outcome}-worldtube"

    def classify_initial(self, state: HamiltonianState):
        return None

    def crossing(self, previous: HamiltonianState, current: HamiltonianState):
        before = previous.event[1] - 1.0
        after = current.event[1] - 1.0
        if before < 0.0 <= after:
            return TerminationCrossing(
                self.outcome,
                self.target_id,
                before,
                after,
            )
        return None

    def value(self, state: HamiltonianState, crossing: TerminationCrossing):
        return state.event[1] - 1.0

    def needs_refinement(
        self,
        previous: HamiltonianState,
        current: HamiltonianState,
    ) -> bool:
        return False


class OfflineSpacetimeTests(unittest.TestCase):
    @staticmethod
    def _recorded_minkowski_path(
        *,
        maximum_affine_length: float = 2.0,
        step: float = 2.0,
        initial: HamiltonianState = HamiltonianState(
            event=(0.0, 0.0, 0.0, 0.0),
            covector=(1.0, 1.0, 0.0, 0.0),
        ),
    ) -> tuple[RayPathSegment, ...]:
        result = trace_null_geodesic(
            MinkowskiMetric(),
            initial,
            options=RayTraceOptions(
                initial_step=step,
                maximum_step=step,
                maximum_affine_length=maximum_affine_length,
                record_path=True,
            ),
        )
        if result.outcome != "completed":
            raise AssertionError(result.failure_reason)
        return result.segments

    def test_schwarzschild_metric_inverse_and_analytic_derivatives(self) -> None:
        provider = SchwarzschildKerrSchildMetric()
        event = (3.0, 13.0, -7.0, 5.0)
        sample = provider.sample(event)
        for row in range(4):
            for column in range(4):
                product = math.fsum(
                    sample.covariant[row][inner] * sample.inverse[inner][column]
                    for inner in range(4)
                )
                self.assertAlmostEqual(product, float(row == column), delta=2.0e-14)

        epsilon = 1.0e-5
        for derivative_axis in range(1, 4):
            plus = list(event)
            minus = list(event)
            plus[derivative_axis] += epsilon
            minus[derivative_axis] -= epsilon
            inverse_plus = provider.sample(tuple(plus)).inverse
            inverse_minus = provider.sample(tuple(minus)).inverse
            for row in range(4):
                for column in range(4):
                    finite_difference = (
                        inverse_plus[row][column] - inverse_minus[row][column]
                    ) / (2.0 * epsilon)
                    self.assertAlmostEqual(
                        sample.inverse_derivatives[derivative_axis][row][column],
                        finite_difference,
                        delta=2.0e-10,
                    )

    def test_static_camera_tetrad_produces_past_directed_null_rays(self) -> None:
        provider = SchwarzschildKerrSchildMetric()
        for screen_x, screen_y in ((0.0, 0.0), (0.2, -0.1), (-0.4, 0.3)):
            ray = static_schwarzschild_camera_ray(
                provider,
                observer_radius_m=40.0,
                screen_x=screen_x,
                screen_y=screen_y,
            )
            contravariant = provider.sample(ray.event).inverse
            self.assertLess(
                sum(contravariant[0][index] * ray.covector[index] for index in range(4)),
                0.0,
            )
            self.assertLess(hamiltonian_null_residual(provider, ray), 2.0e-15)

    def test_minkowski_full_hamiltonian_is_a_straight_null_line(self) -> None:
        provider = MinkowskiMetric()
        initial = HamiltonianState(
            event=(0.0, 2.0, -3.0, 5.0),
            covector=(1.0, 0.6, 0.8, 0.0),
        )
        self.assertAlmostEqual(
            bilinear(
                initial.covector,
                provider.sample(initial.event).inverse,
                initial.covector,
            ),
            0.0,
            delta=2.0e-16,
        )
        result = trace_null_geodesic(
            provider,
            initial,
            options=RayTraceOptions(
                maximum_affine_length=12.0,
                maximum_step=2.0,
                record_path=True,
            ),
        )
        self.assertEqual(result.outcome, "completed")
        self.assertAlmostEqual(result.terminal_state.event[0], -12.0, places=11)
        self.assertAlmostEqual(result.terminal_state.event[1], 9.2, places=11)
        self.assertAlmostEqual(result.terminal_state.event[2], 6.6, places=11)
        self.assertAlmostEqual(result.terminal_state.event[3], 5.0, places=11)
        self.assertEqual(result.terminal_state.covector, initial.covector)
        self.assertLess(result.maximum_null_residual, 1.0e-14)
        self.assertGreater(len(result.segments), 0)

    def test_recorded_surface_locator_finds_multiple_crossings_in_order(self) -> None:
        segments = self._recorded_minkowski_path()
        crossings = locate_recorded_surface_crossings(
            MinkowskiMetric(),
            segments,
            lambda state: (state.event[1] - 0.5) * (state.event[1] - 1.5),
        )

        self.assertEqual(len(crossings), 2)
        self.assertEqual(tuple(event.orientation for event in crossings), (-1, 1))
        self.assertEqual(
            tuple(event.segment_index for event in crossings),
            (0, 0),
        )
        for crossing, expected_affine in zip(crossings, (0.5, 1.5)):
            self.assertAlmostEqual(
                crossing.ray_affine_length,
                expected_affine,
                delta=1.0e-12,
            )
            self.assertAlmostEqual(
                crossing.state.event[1],
                expected_affine,
                delta=1.0e-12,
            )
            self.assertEqual(crossing.state.covector, (1.0, 1.0, 0.0, 0.0))
            self.assertLessEqual(abs(crossing.surface_value), 1.0e-9)

    def test_accepted_step_surface_refines_when_2n_exposes_a_double_root(
        self,
    ) -> None:
        result = trace_null_geodesic(
            MinkowskiMetric(),
            HamiltonianState(
                event=(0.0, 0.0, 0.0, 1.0),
                covector=(1.0, 1.0, 0.0, 0.0),
            ),
            interior_surface=PolynomialInteriorSurface(
                (0.2, 0.3),
                terminal_root=0.3,
            ),
            surface_options=SurfaceEventOptions(subdivisions_per_segment=2),
            options=RayTraceOptions(
                initial_step=1.0,
                maximum_step=1.0,
                maximum_affine_length=2.0,
                record_path=True,
            ),
        )

        self.assertEqual(result.outcome, "surface-hit", result.failure_reason)
        self.assertEqual(result.terminal_target_id, "test-opaque-surface")
        self.assertAlmostEqual(result.affine_length, 0.3, delta=2.0e-10)
        self.assertGreaterEqual(result.rejected_steps, 1)
        self.assertEqual(len(result.segments), 1)
        trace = result.interior_surface_trace
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertTrue(trace.topology_converged)
        self.assertEqual(trace.base_subdivisions_per_step, 2)
        self.assertEqual(trace.verification_subdivisions_per_step, 4)
        self.assertEqual(
            tuple(entry.decision.classification for entry in trace.crossings),
            ("transparent-inner", "opaque"),
        )
        self.assertEqual(
            tuple(entry.crossing.orientation for entry in trace.crossings),
            (-1, 1),
        )

    def test_accepted_step_surface_continues_across_transparent_regions(self) -> None:
        result = trace_null_geodesic(
            MinkowskiMetric(),
            HamiltonianState(
                event=(0.0, 0.0, 0.0, 1.0),
                covector=(1.0, 1.0, 0.0, 0.0),
            ),
            interior_surface=PolynomialInteriorSurface(
                (0.2, 0.4, 0.8),
                terminal_root=0.8,
            ),
            surface_options=SurfaceEventOptions(subdivisions_per_segment=8),
            options=RayTraceOptions(
                initial_step=1.0,
                maximum_step=1.0,
                maximum_affine_length=2.0,
                record_path=True,
            ),
        )

        self.assertEqual(result.outcome, "surface-hit", result.failure_reason)
        self.assertAlmostEqual(result.affine_length, 0.8, delta=2.0e-10)
        trace = result.interior_surface_trace
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(
            tuple(entry.decision.classification for entry in trace.crossings),
            ("transparent-inner", "transparent-outer", "opaque"),
        )

    def test_boundary_and_interior_surface_are_ordered_within_one_step(self) -> None:
        initial = HamiltonianState(
            event=(0.0, 1.0, 0.0, 0.0),
            covector=(1.0, 1.0, 0.0, 0.0),
        )
        options = RayTraceOptions(
            initial_step=4.0,
            maximum_step=4.0,
            maximum_affine_length=6.0,
            record_path=True,
        )
        boundary_first = trace_null_geodesic(
            MinkowskiMetric(),
            initial,
            termination=RadialTermination(0.5, 2.0),
            interior_surface=PolynomialInteriorSurface(
                (3.0,),
                terminal_root=3.0,
            ),
            options=options,
        )
        self.assertEqual(boundary_first.outcome, "escaped")
        self.assertAlmostEqual(boundary_first.terminal_state.event[1], 2.0, delta=2e-9)
        assert boundary_first.interior_surface_trace is not None
        self.assertEqual(boundary_first.interior_surface_trace.crossings, ())

        surface_first = trace_null_geodesic(
            MinkowskiMetric(),
            initial,
            termination=RadialTermination(0.5, 2.0),
            interior_surface=PolynomialInteriorSurface(
                (1.5,),
                terminal_root=1.5,
            ),
            options=options,
        )
        self.assertEqual(surface_first.outcome, "surface-hit")
        self.assertAlmostEqual(surface_first.terminal_state.event[1], 1.5, delta=2e-9)

    def test_exact_transparent_plane_contact_at_worldtube_endpoint_is_allowed(
        self,
    ) -> None:
        initial = HamiltonianState(
            event=(0.0, 0.0, 0.0, -1.0),
            covector=(math.sqrt(2.0), 1.0, 0.0, 1.0),
        )
        options = RayTraceOptions(
            initial_step=2.0,
            maximum_step=2.0,
            maximum_affine_length=3.0,
            record_path=True,
        )
        for outcome in ("captured", "escaped"):
            with self.subTest(outcome=outcome):
                result = trace_null_geodesic(
                    MinkowskiMetric(),
                    initial,
                    termination=XWorldtubeTermination(outcome),
                    interior_surface=EndpointPlaneSurface(opaque=False),
                    options=options,
                )
                self.assertEqual(result.outcome, outcome, result.failure_reason)
                self.assertAlmostEqual(result.terminal_state.event[1], 1.0, delta=2e-9)
                self.assertAlmostEqual(result.terminal_state.event[3], 0.0, delta=2e-9)
                self.assertIsNotNone(result.interior_surface_trace)
                assert result.interior_surface_trace is not None
                self.assertEqual(result.interior_surface_trace.crossings, ())

        opaque = trace_null_geodesic(
            MinkowskiMetric(),
            initial,
            termination=XWorldtubeTermination("escaped"),
            interior_surface=EndpointPlaneSurface(opaque=True),
            options=options,
        )
        self.assertEqual(opaque.outcome, "integrator-failure")
        self.assertIn("unbracketed", opaque.failure_reason or "")

    def test_accepted_step_tangent_contact_fails_closed(self) -> None:
        result = trace_null_geodesic(
            MinkowskiMetric(),
            HamiltonianState(
                event=(0.0, 0.0, 0.0, 1.0),
                covector=(1.0, 1.0, 0.0, 0.0),
            ),
            interior_surface=TangentInteriorSurface(),
            surface_options=SurfaceEventOptions(subdivisions_per_segment=2),
            options=RayTraceOptions(
                initial_step=1.0,
                maximum_step=1.0,
                maximum_affine_length=1.0,
            ),
        )
        self.assertEqual(result.outcome, "integrator-failure")
        self.assertIn("tangent", result.failure_reason or "")
        self.assertIsNotNone(result.interior_surface_trace)
        assert result.interior_surface_trace is not None
        self.assertFalse(result.interior_surface_trace.topology_converged)

    def test_recorded_surface_locator_deduplicates_a_segment_endpoint(self) -> None:
        initial = HamiltonianState(
            event=(0.0, 0.0, 0.0, -1.0),
            covector=(1.0, 0.0, 0.0, 1.0),
        )
        segments = self._recorded_minkowski_path(step=1.0, initial=initial)
        self.assertEqual(len(segments), 2)

        crossings = locate_recorded_surface_crossings(
            MinkowskiMetric(),
            segments,
            lambda state: state.event[3],
        )

        self.assertEqual(len(crossings), 1)
        crossing = crossings[0]
        self.assertEqual(crossing.orientation, 1)
        self.assertEqual(crossing.segment_index, 0)
        self.assertEqual(crossing.iterations, 0)
        self.assertAlmostEqual(crossing.ray_affine_length, 1.0, delta=1.0e-13)
        self.assertAlmostEqual(crossing.state.event[3], 0.0, delta=1.0e-13)

    def test_near_zero_probe_keeps_a_real_root_bracket(self) -> None:
        ray = trace_null_geodesic(
            MinkowskiMetric(),
            HamiltonianState(
                event=(0.0, 0.0, 0.0, -1.6e-9),
                covector=(5.0e-9, 0.0, 0.0, 5.0e-9),
            ),
            options=RayTraceOptions(
                initial_step=1.0,
                maximum_step=1.0,
                maximum_affine_length=1.0,
                record_path=True,
                absolute_tolerance=1.0e-12,
                relative_tolerance=1.0e-12,
            ),
        )
        crossings = locate_recorded_surface_crossings(
            MinkowskiMetric(),
            ray.segments,
            lambda state: state.event[3],
            options=SurfaceEventOptions(
                absolute_tolerance=1.0e-12,
                relative_tolerance=1.0e-12,
                surface_value_tolerance=1.0e-9,
                affine_tolerance=1.0e-10,
            ),
        )
        self.assertEqual(len(crossings), 1)
        self.assertAlmostEqual(crossings[0].ray_affine_length, 0.32, delta=1.0e-9)
        self.assertLessEqual(abs(crossings[0].surface_value), 1.0e-9)
        self.assertGreater(crossings[0].bracket_affine_width, 0.0)
        self.assertLessEqual(crossings[0].bracket_affine_width, 1.0e-10)

    def test_explicit_policy_ignores_only_unbracketed_path_endpoints(self) -> None:
        ray = trace_null_geodesic(
            MinkowskiMetric(),
            HamiltonianState(
                event=(0.0, 0.0, 0.0, 0.0),
                covector=(1.0, math.sqrt(0.75), 0.0, 0.5),
            ),
            options=RayTraceOptions(
                initial_step=0.5,
                maximum_step=0.5,
                maximum_affine_length=1.0,
                record_path=True,
            ),
        )
        with self.assertRaisesRegex(SurfaceEventError, "unbracketed"):
            locate_recorded_surface_crossings(
                MinkowskiMetric(),
                ray.segments,
                lambda state: state.event[3],
            )
        self.assertEqual(
            locate_recorded_surface_crossings(
                MinkowskiMetric(),
                ray.segments,
                lambda state: state.event[3],
                ignore_unbracketed_path_endpoints=True,
            ),
            (),
        )
        with self.assertRaises(TypeError):
            locate_recorded_surface_crossings(
                MinkowskiMetric(),
                ray.segments,
                lambda state: state.event[3],
                ignore_unbracketed_path_endpoints=1,  # type: ignore[arg-type]
            )

    def test_recorded_surface_locator_preserves_order_across_segments(self) -> None:
        segments = self._recorded_minkowski_path(
            maximum_affine_length=3.0,
            step=1.0,
        )
        crossings = locate_recorded_surface_crossings(
            MinkowskiMetric(),
            segments,
            lambda state: (
                (state.event[1] - 0.25)
                * (state.event[1] - 1.25)
                * (state.event[1] - 2.25)
            ),
        )

        self.assertEqual(tuple(event.segment_index for event in crossings), (0, 1, 2))
        self.assertEqual(
            tuple(round(event.ray_affine_length, 12) for event in crossings),
            (0.25, 1.25, 2.25),
        )
        self.assertEqual(
            tuple(event.orientation for event in crossings),
            (1, -1, 1),
        )

    def test_recorded_surface_locator_rejects_tangent_and_boundary_contacts(
        self,
    ) -> None:
        segments = self._recorded_minkowski_path(step=1.0)
        with self.assertRaisesRegex(SurfaceEventError, "tangent"):
            locate_recorded_surface_crossings(
                MinkowskiMetric(),
                segments,
                lambda state: (state.event[1] - 1.0) ** 2,
            )
        with self.assertRaisesRegex(SurfaceEventError, "unbracketed path endpoint"):
            locate_recorded_surface_crossings(
                MinkowskiMetric(),
                segments,
                lambda state: state.event[1],
            )
        self.assertEqual(
            locate_recorded_surface_crossings(
                MinkowskiMetric(),
                segments,
                lambda state: state.event[1] + 1.0,
            ),
            (),
        )

    def test_recorded_surface_locator_fails_closed_on_a_discontinuous_sign_flip(
        self,
    ) -> None:
        segments = self._recorded_minkowski_path()
        with self.assertRaisesRegex(SurfaceEventError, "without a value root"):
            locate_recorded_surface_crossings(
                MinkowskiMetric(),
                segments,
                lambda state: -1.0 if state.event[1] < 0.7 else 1.0,
                options=SurfaceEventOptions(affine_tolerance=1.0e-6),
            )

    def test_recorded_surface_locator_enforces_work_and_iteration_budgets(
        self,
    ) -> None:
        segments = self._recorded_minkowski_path()
        with self.assertRaisesRegex(SurfaceEventError, "reintegration budget"):
            locate_recorded_surface_crossings(
                MinkowskiMetric(),
                segments,
                lambda state: state.event[1] - 0.7,
                options=SurfaceEventOptions(maximum_reintegrations=1),
            )
        with self.assertRaisesRegex(SurfaceEventError, "iteration budget"):
            locate_recorded_surface_crossings(
                MinkowskiMetric(),
                segments,
                lambda state: state.event[1] - 0.7,
                options=SurfaceEventOptions(maximum_iterations=1),
            )

    def test_recorded_surface_locator_reintegrates_recorded_states(self) -> None:
        segments = self._recorded_minkowski_path()
        segment = segments[0]
        corrupted = replace(
            segment,
            midpoint=HamiltonianState(
                event=(
                    segment.midpoint.event[0],
                    segment.midpoint.event[1] + 1.0e-4,
                    segment.midpoint.event[2],
                    segment.midpoint.event[3],
                ),
                covector=segment.midpoint.covector,
            ),
        )
        with self.assertRaisesRegex(
            SurfaceEventError,
            "midpoint does not match Hamiltonian reintegration",
        ):
            locate_recorded_surface_crossings(
                MinkowskiMetric(),
                (corrupted,),
                lambda state: state.event[1] - 0.5,
            )

    def test_recorded_surface_locator_rejects_non_finite_surface_values(self) -> None:
        segments = self._recorded_minkowski_path()
        with self.assertRaisesRegex(SurfaceEventError, "non-finite"):
            locate_recorded_surface_crossings(
                MinkowskiMetric(),
                segments,
                lambda _state: math.nan,
            )
        with self.assertRaisesRegex(SurfaceEventError, "interpolation error"):
            locate_recorded_surface_crossings(
                ExcessInterpolationErrorMinkowski(),
                segments,
                lambda state: state.event[1] - 0.5,
                options=SurfaceEventOptions(
                    metric_interpolation_error_limit=1.0e-5,
                ),
            )

    def test_null_residual_is_invariant_across_momentum_scales(self) -> None:
        provider = MinkowskiMetric()
        residuals = tuple(
            hamiltonian_null_residual(
                provider,
                HamiltonianState(
                    event=(0.0, 0.0, 0.0, 0.0),
                    covector=(scale, scale, 0.0, 0.0),
                ),
            )
            for scale in (1.0e-320, 1.0, 1.0e300)
        )
        self.assertEqual(residuals, (0.0, 0.0, 0.0))

    def test_time_dependent_provider_is_sampled_at_ray_event_time(self) -> None:
        provider = RecordingTimeDependentMinkowski()
        result = trace_null_geodesic(
            provider,
            HamiltonianState(
                event=(7.0, 0.0, 0.0, 0.0),
                covector=(1.0, 1.0, 0.0, 0.0),
            ),
            options=RayTraceOptions(
                maximum_affine_length=3.0,
                maximum_step=0.5,
            ),
        )
        self.assertEqual(result.outcome, "completed")
        sampled_times = [event[0] for event in provider.events]
        self.assertGreater(max(sampled_times), min(sampled_times))
        self.assertLess(min(sampled_times), 4.1)
        self.assertAlmostEqual(result.terminal_state.event[0], 4.0, places=11)

    def test_schwarzschild_capture_and_escape_recover_critical_impact(self) -> None:
        provider = SchwarzschildKerrSchildMetric()
        observer_radius = 40.0
        lapse = math.sqrt(1.0 - 2.0 / observer_radius)

        def screen_radius_for_impact(impact: float) -> float:
            sine = impact * lapse / observer_radius
            return sine / math.sqrt(1.0 - sine * sine)

        termination = RadialTermination(2.02, 120.0)
        options = RayTraceOptions(
            absolute_tolerance=5.0e-10,
            relative_tolerance=5.0e-10,
            maximum_step=0.5,
            maximum_affine_length=1_000.0,
            null_residual_limit=2.0e-7,
            record_path=True,
        )
        captured = trace_null_geodesic(
            provider,
            static_schwarzschild_camera_ray(
                provider,
                observer_radius_m=observer_radius,
                screen_x=screen_radius_for_impact(4.8),
                screen_y=0.0,
            ),
            termination=termination,
            options=options,
        )
        escaped = trace_null_geodesic(
            provider,
            static_schwarzschild_camera_ray(
                provider,
                observer_radius_m=observer_radius,
                screen_x=screen_radius_for_impact(5.6),
                screen_y=0.0,
            ),
            termination=termination,
            options=options,
        )
        self.assertEqual(captured.outcome, "captured", captured.failure_reason)
        self.assertEqual(escaped.outcome, "escaped", escaped.failure_reason)
        self.assertEqual(captured.terminal_target_id, "analytic-capture-sphere")
        self.assertEqual(escaped.terminal_target_id, "analytic-escape-sphere")
        self.assertLess(captured.terminal_state.event[0], 0.0)
        self.assertLess(escaped.terminal_state.event[0], 0.0)
        self.assertLess(captured.maximum_null_residual, options.null_residual_limit)
        self.assertLess(escaped.maximum_null_residual, options.null_residual_limit)
        self.assertAlmostEqual(
            termination.radius(captured.terminal_state),
            termination.capture_radius_m,
            delta=2.0e-9,
        )
        self.assertAlmostEqual(
            termination.radius(escaped.terminal_state),
            termination.escape_radius_m,
            delta=2.0e-9,
        )
        self.assertGreater(len(captured.segments), 0)
        self.assertGreater(len(escaped.segments), 0)

    def test_affine_budget_with_terminal_surface_is_unresolved(self) -> None:
        provider = SchwarzschildKerrSchildMetric()
        result = trace_null_geodesic(
            provider,
            static_schwarzschild_camera_ray(
                provider,
                observer_radius_m=40.0,
                screen_x=0.0,
                screen_y=0.0,
            ),
            termination=RadialTermination(2.02, 120.0),
            options=RayTraceOptions(maximum_affine_length=1.0),
        )
        self.assertEqual(result.outcome, "unresolved")
        self.assertEqual(
            result.failure_reason,
            "affine-parameter budget exhausted",
        )

    def test_initial_state_inside_capture_domain_terminates_immediately(self) -> None:
        result = trace_null_geodesic(
            MinkowskiMetric(),
            HamiltonianState(
                event=(0.0, 0.5, 0.0, 0.0),
                covector=(1.0, 1.0, 0.0, 0.0),
            ),
            termination=RadialTermination(1.0, 2.0),
        )
        self.assertEqual(result.outcome, "captured")
        self.assertEqual(result.affine_length, 0.0)
        self.assertEqual(result.accepted_steps, 0)
        self.assertEqual(result.terminal_target_id, "analytic-capture-sphere")

    def test_radial_event_refinement_catches_an_interior_leap(self) -> None:
        result = trace_null_geodesic(
            MinkowskiMetric(),
            HamiltonianState(
                event=(0.0, -2.0, 0.0, 0.0),
                covector=(1.0, 1.0, 0.0, 0.0),
            ),
            termination=RadialTermination(0.5, 10.0),
            options=RayTraceOptions(
                initial_step=4.0,
                maximum_step=4.0,
                maximum_affine_length=6.0,
            ),
        )
        self.assertEqual(result.outcome, "captured", result.failure_reason)
        self.assertEqual(result.terminal_target_id, "analytic-capture-sphere")
        self.assertAlmostEqual(
            result.terminal_state.event[1],
            -0.5,
            delta=2.0e-9,
        )

    def test_whole_ray_coarse_fine_refinement_compares_terminal_state(self) -> None:
        provider = SchwarzschildKerrSchildMetric()
        observer_radius = 40.0
        lapse = math.sqrt(1.0 - 2.0 / observer_radius)
        impact = 5.8
        sine = impact * lapse / observer_radius
        screen_radius = sine / math.sqrt(1.0 - sine * sine)
        result = trace_refined_null_geodesic(
            provider,
            static_schwarzschild_camera_ray(
                provider,
                observer_radius_m=observer_radius,
                screen_x=screen_radius,
                screen_y=0.0,
            ),
            termination=RadialTermination(2.02, 120.0),
            fine_options=RayTraceOptions(
                absolute_tolerance=1.0e-10,
                relative_tolerance=1.0e-10,
                maximum_step=0.4,
                maximum_affine_length=1_000.0,
            ),
            terminal_event_tolerance=1.0e-4,
            terminal_covector_tolerance=1.0e-4,
        )
        self.assertTrue(result.outcome_agrees)
        self.assertTrue(result.discretizations_differ)
        self.assertTrue(result.terminal_target_agrees)
        self.assertEqual(result.fine.outcome, "escaped")
        self.assertTrue(result.converged)
        self.assertLess(result.terminal_event_difference, 1.0e-4)
        self.assertLess(result.terminal_covector_difference, 1.0e-4)

    def test_refinement_coarse_path_recording_defaults_to_disabled(self) -> None:
        initial = HamiltonianState(
            event=(0.0, 0.0, 0.0, 0.0),
            covector=(1.0, 1.0, 0.0, 0.0),
        )
        options = RayTraceOptions(
            initial_step=0.25,
            maximum_step=0.25,
            maximum_affine_length=4.0,
            record_path=True,
        )
        default = trace_refined_null_geodesic(
            MinkowskiMetric(),
            initial,
            fine_options=options,
        )
        explicit_false = trace_refined_null_geodesic(
            MinkowskiMetric(),
            initial,
            fine_options=options,
            record_coarse_path=False,
        )

        self.assertEqual(default, explicit_false)
        self.assertGreater(len(default.fine.segments), 0)
        self.assertEqual(default.coarse.segments, ())

    def test_refinement_can_record_complete_fine_and_coarse_paths(self) -> None:
        initial = HamiltonianState(
            event=(0.0, 0.0, 0.0, 0.0),
            covector=(1.0, 1.0, 0.0, 0.0),
        )
        result = trace_refined_null_geodesic(
            MinkowskiMetric(),
            initial,
            fine_options=RayTraceOptions(
                initial_step=0.25,
                maximum_step=0.25,
                maximum_affine_length=4.0,
                record_path=True,
            ),
            record_coarse_path=True,
        )

        for trace in (result.fine, result.coarse):
            self.assertEqual(len(trace.segments), trace.accepted_steps)
            self.assertEqual(trace.segments[0].start, initial)
            self.assertEqual(trace.segments[-1].end, trace.terminal_state)
            self.assertEqual(
                math.fsum(segment.affine_length for segment in trace.segments),
                trace.affine_length,
            )
            for previous, current in zip(trace.segments, trace.segments[1:]):
                self.assertEqual(previous.end, current.start)

        self.assertEqual(result.fine.terminal_state, result.coarse.terminal_state)
        self.assertEqual(result.fine.affine_length, result.coarse.affine_length)
        self.assertNotEqual(len(result.fine.segments), len(result.coarse.segments))

    def test_refinement_coarse_path_recording_is_strictly_gated(self) -> None:
        initial = HamiltonianState(
            event=(0.0, 0.0, 0.0, 0.0),
            covector=(1.0, 1.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(
            ValueError,
            "requires fine_options.record_path=True",
        ):
            trace_refined_null_geodesic(
                MinkowskiMetric(),
                initial,
                record_coarse_path=True,
            )

        recorded_options = RayTraceOptions(
            initial_step=0.25,
            maximum_step=0.25,
            maximum_affine_length=1.0,
            record_path=True,
        )
        for invalid in (None, 0, 1, "true", (), []):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "must be a bool"):
                    trace_refined_null_geodesic(
                        MinkowskiMetric(),
                        initial,
                        fine_options=recorded_options,
                        record_coarse_path=invalid,  # type: ignore[arg-type]
                    )

    def test_refinement_rejects_disagreement_on_terminal_target(self) -> None:
        result = trace_refined_null_geodesic(
            MinkowskiMetric(),
            HamiltonianState(
                event=(0.0, 0.0, 0.0, 0.0),
                covector=(1.0, 1.0, 0.0, 0.0),
            ),
            termination=StepSensitiveTargetTermination(),
            fine_options=RayTraceOptions(
                initial_step=0.1,
                maximum_step=0.1,
                maximum_affine_length=2.0,
            ),
        )
        self.assertTrue(result.outcome_agrees)
        self.assertFalse(result.terminal_target_agrees)
        self.assertFalse(result.converged)
        self.assertLess(result.terminal_event_difference, 1.0e-4)
        self.assertLess(result.terminal_covector_difference, 1.0e-4)

    def test_refinement_uses_distinct_step_hierarchies_when_max_step_dominates(self) -> None:
        result = trace_refined_null_geodesic(
            MinkowskiMetric(),
            HamiltonianState(
                event=(0.0, 0.0, 0.0, 0.0),
                covector=(1.0, 1.0, 0.0, 0.0),
            ),
            fine_options=RayTraceOptions(
                initial_step=0.25,
                maximum_step=0.25,
                maximum_affine_length=4.0,
            ),
        )
        self.assertTrue(result.discretizations_differ)
        self.assertNotEqual(
            result.fine.accepted_steps,
            result.coarse.accepted_steps,
        )
        self.assertTrue(result.converged)

    def test_metric_sample_shape_is_fail_closed(self) -> None:
        identity = MinkowskiMetric().sample((0.0, 0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "exactly four rows"):
            MetricSample(
                covariant=identity.covariant[:3],  # type: ignore[arg-type]
                inverse=identity.inverse,
                inverse_derivatives=identity.inverse_derivatives,
            )
        with self.assertRaisesRegex(ValueError, "four coordinate derivatives"):
            MetricSample(
                covariant=identity.covariant,
                inverse=identity.inverse,
                inverse_derivatives=identity.inverse_derivatives[:3],  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MetricSample(
                covariant=identity.covariant,
                inverse=tuple(
                    tuple(2.0 * value for value in row)
                    for row in identity.inverse
                ),  # type: ignore[arg-type]
                inverse_derivatives=identity.inverse_derivatives,
            )
        epsilon = 1.0e-5
        determinant = -epsilon
        exact_inverse_block = (
            ((1.0 - epsilon) / determinant, -1.0 / determinant),
            (-1.0 / determinant, 1.0 / determinant),
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MetricSample(
                covariant=(
                    (1.0, 1.0, 0.0, 0.0),
                    (1.0, 1.0 - epsilon, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                inverse=(
                    (
                        exact_inverse_block[0][0] + 1.0,
                        exact_inverse_block[0][1],
                        0.0,
                        0.0,
                    ),
                    (
                        exact_inverse_block[1][0],
                        exact_inverse_block[1][1],
                        0.0,
                        0.0,
                    ),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                inverse_derivatives=identity.inverse_derivatives,
            )
        with self.assertRaisesRegex(ValueError, r"Lorentzian -\+\+\+"):
            MetricSample(
                covariant=(
                    (-1.0, 0.0, 0.0, 0.0),
                    (0.0, -1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                inverse=(
                    (-1.0, 0.0, 0.0, 0.0),
                    (0.0, -1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                inverse_derivatives=identity.inverse_derivatives,
            )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            MetricSample(
                covariant=(
                    (-1.0e100, 0.0, 0.0, 0.0),
                    (0.0, 1.0e-100, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                inverse=(
                    (-1.0e-100, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                inverse_derivatives=identity.inverse_derivatives,
            )

    def test_step_and_event_budgets_require_real_positive_integers(self) -> None:
        for value in (True, 1.5, math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    RayTraceOptions(maximum_accepted_steps=value)  # type: ignore[arg-type]

    def test_fail_closed_rejects_non_null_initial_covector(self) -> None:
        with self.assertRaisesRegex(ValueError, "not null"):
            trace_null_geodesic(
                MinkowskiMetric(),
                HamiltonianState(
                    event=(0.0, 0.0, 0.0, 0.0),
                    covector=(1.0, 0.5, 0.0, 0.0),
                ),
            )

    def test_metric_interpolation_error_is_a_hard_ray_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "interpolation error"):
            trace_null_geodesic(
                ExcessInterpolationErrorMinkowski(),
                HamiltonianState(
                    event=(0.0, 0.0, 0.0, 0.0),
                    covector=(1.0, 1.0, 0.0, 0.0),
                ),
                options=RayTraceOptions(
                    metric_interpolation_error_limit=1.0e-5,
                ),
            )

        with self.assertRaisesRegex(ValueError, "not null"):
            trace_null_geodesic(
                MinkowskiMetric(),
                HamiltonianState(
                    event=(0.0, 0.0, 0.0, 0.0),
                    covector=(0.0, 0.0, 0.0, 0.0),
                ),
            )


if __name__ == "__main__":
    unittest.main()
