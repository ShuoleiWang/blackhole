from __future__ import annotations

import math
import unittest

from offline.geodesic import (
    HamiltonianState,
    InitialMultiSurfaceContact,
    InteriorSurfaceDecision,
    RadialTermination,
    RayTraceOptions,
    SurfaceEventOptions,
    trace_null_geodesic,
    trace_refined_null_geodesic,
)
from offline.spacetime import MinkowskiMetric


INITIAL_X_RAY = HamiltonianState(
    event=(0.0, 0.0, 0.0, 0.0),
    covector=(1.0, 1.0, 0.0, 0.0),
)


class PlaneMultiSurface:
    def __init__(
        self,
        planes: dict[str, float],
        *,
        terminal_ids: frozenset[str] = frozenset(),
        returned_id_order: tuple[str, ...] | None = None,
    ) -> None:
        self.planes = dict(planes)
        self.terminal_ids = terminal_ids
        self._surface_ids = (
            tuple(planes)
            if returned_id_order is None
            else returned_id_order
        )

    @property
    def surface_ids(self) -> tuple[str, ...]:
        return self._surface_ids

    def value(self, surface_id: str, state: HamiltonianState) -> float:
        return state.event[1] - self.planes[surface_id]

    def classify(self, surface_id: str, crossing):
        if surface_id in self.terminal_ids:
            return InteriorSurfaceDecision(
                f"opaque-{surface_id}",
                "surface-hit",
                f"target-{surface_id}",
            )
        return InteriorSurfaceDecision(f"transparent-{surface_id}")


class PolynomialMultiSurface:
    surface_ids = ("polynomial",)

    def value(self, surface_id: str, state: HamiltonianState) -> float:
        if surface_id != "polynomial":
            raise KeyError(surface_id)
        coordinate = state.event[1]
        return (coordinate - 0.2) * (coordinate - 0.3)

    def classify(self, surface_id: str, crossing):
        coordinate = crossing.state.event[1]
        if abs(coordinate - 0.3) < 0.025:
            return InteriorSurfaceDecision(
                "opaque-polynomial",
                "surface-hit",
                "target-polynomial",
            )
        return InteriorSurfaceDecision("transparent-polynomial")


class InitialReentryMultiSurface:
    surface_ids = ("reentry",)

    def value(self, surface_id: str, state: HamiltonianState) -> float:
        if surface_id != "reentry":
            raise KeyError(surface_id)
        coordinate = state.event[1]
        return coordinate * (coordinate - 0.02)

    def classify(self, surface_id: str, crossing):
        return InteriorSurfaceDecision(
            "opaque-reentry",
            "surface-hit",
            "target-reentry",
        )


class TangentMultiSurface:
    surface_ids = ("tangent",)

    def value(self, surface_id: str, state: HamiltonianState) -> float:
        return (state.event[1] - 0.5) ** 2

    def classify(self, surface_id: str, crossing):
        return InteriorSurfaceDecision(
            "opaque-tangent",
            "surface-hit",
            "target-tangent",
        )


class MutableIdMultiSurface(PlaneMultiSurface):
    def value(self, surface_id: str, state: HamiltonianState) -> float:
        self._surface_ids = ("changed",)
        return super().value(surface_id, state)


class SinglePlaneSurface:
    def value(self, state: HamiltonianState) -> float:
        return state.event[1] - 0.5

    def classify(self, crossing):
        return InteriorSurfaceDecision("transparent")


def one_step_options(*, maximum_affine_length: float = 1.0) -> RayTraceOptions:
    return RayTraceOptions(
        initial_step=maximum_affine_length,
        maximum_step=maximum_affine_length,
        maximum_affine_length=maximum_affine_length,
        record_path=True,
    )


class OfflineMultiSurfaceTests(unittest.TestCase):
    def test_authenticated_exact_initial_contact_assigns_only_affine_zero_side(
        self,
    ) -> None:
        result = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=PlaneMultiSurface(
                {"initial": 0.0},
                terminal_ids=frozenset(("initial",)),
            ),
            initial_multi_surface_contact=InitialMultiSurfaceContact(
                "initial",
                1,
            ),
            surface_options=SurfaceEventOptions(subdivisions_per_segment=4),
            options=one_step_options(),
        )

        self.assertEqual(result.outcome, "completed", result.failure_reason)
        trace = result.multi_surface_trace
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.crossings, ())
        self.assertIsNotNone(trace.initial_contact)
        assert trace.initial_contact is not None
        self.assertEqual(trace.initial_contact.surface_id, "initial")
        self.assertEqual(trace.initial_contact.side, 1)
        self.assertEqual(trace.initial_contact.actual_surface_value, 0.0)

    def test_authenticated_near_zero_initial_contact_uses_declared_tolerance(
        self,
    ) -> None:
        result = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=PlaneMultiSurface(
                {"near": -5.0e-10},
                terminal_ids=frozenset(("near",)),
            ),
            initial_multi_surface_contact=InitialMultiSurfaceContact("near", 1),
            surface_options=SurfaceEventOptions(
                surface_value_tolerance=1.0e-9,
                subdivisions_per_segment=4,
            ),
            options=one_step_options(),
        )

        self.assertEqual(result.outcome, "completed", result.failure_reason)
        assert result.multi_surface_trace is not None
        contact = result.multi_surface_trace.initial_contact
        self.assertIsNotNone(contact)
        assert contact is not None
        self.assertEqual(contact.actual_surface_value.hex(), (5.0e-10).hex())
        self.assertEqual(contact.surface_value_tolerance.hex(), (1.0e-9).hex())

    def test_initial_contact_rejects_unknown_id_residual_and_illegal_side(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, r"side must be -1 or \+1"):
            InitialMultiSurfaceContact("initial", 0)
        with self.assertRaisesRegex(ValueError, "not declared"):
            trace_null_geodesic(
                MinkowskiMetric(),
                INITIAL_X_RAY,
                multi_interior_surface=PlaneMultiSurface({"initial": 0.0}),
                initial_multi_surface_contact=InitialMultiSurfaceContact(
                    "unknown",
                    1,
                ),
            )
        with self.assertRaisesRegex(ValueError, "not within"):
            trace_null_geodesic(
                MinkowskiMetric(),
                INITIAL_X_RAY,
                multi_interior_surface=PlaneMultiSurface({"far": 0.1}),
                initial_multi_surface_contact=InitialMultiSurfaceContact("far", -1),
                surface_options=SurfaceEventOptions(
                    surface_value_tolerance=1.0e-9
                ),
            )

    def test_initial_contact_token_preserves_fast_subsequent_reentry(self) -> None:
        result = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=InitialReentryMultiSurface(),
            initial_multi_surface_contact=InitialMultiSurfaceContact(
                "reentry",
                -1,
            ),
            surface_options=SurfaceEventOptions(subdivisions_per_segment=2),
            options=one_step_options(),
        )

        self.assertEqual(result.outcome, "surface-hit", result.failure_reason)
        self.assertAlmostEqual(result.affine_length, 0.02, delta=2.0e-9)
        assert result.multi_surface_trace is not None
        self.assertEqual(len(result.multi_surface_trace.crossings), 1)
        crossing = result.multi_surface_trace.crossings[0].crossing
        self.assertGreater(crossing.ray_affine_length, 0.0)

    def test_undeclared_exact_initial_contact_keeps_strict_legacy_behavior(
        self,
    ) -> None:
        result = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=PlaneMultiSurface(
                {"initial": 0.0},
                terminal_ids=frozenset(("initial",)),
            ),
            surface_options=SurfaceEventOptions(subdivisions_per_segment=4),
            options=one_step_options(),
        )

        self.assertEqual(result.outcome, "integrator-failure")
        assert result.multi_surface_trace is not None
        self.assertIsNone(result.multi_surface_trace.initial_contact)

    def test_two_planes_in_one_step_are_globally_ordered_observer_to_source(
        self,
    ) -> None:
        surfaces = PlaneMultiSurface(
            {"far": 0.75, "near": 0.25},
            returned_id_order=("far", "near"),
        )
        result = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=surfaces,
            surface_options=SurfaceEventOptions(subdivisions_per_segment=4),
            options=one_step_options(),
        )

        self.assertEqual(result.outcome, "completed", result.failure_reason)
        self.assertIsNone(result.interior_surface_trace)
        trace = result.multi_surface_trace
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.surface_ids, ("far", "near"))
        self.assertEqual(
            tuple(entry.surface_id for entry in trace.crossings),
            ("near", "far"),
        )
        self.assertEqual(
            tuple(entry.crossing.ray_affine_length for entry in trace.crossings),
            (0.25, 0.75),
        )
        self.assertTrue(trace.topology_converged)

    def test_first_opaque_member_stops_before_later_members(self) -> None:
        surfaces = PlaneMultiSurface(
            {"later": 0.75, "first-opaque": 0.25},
            terminal_ids=frozenset(("first-opaque", "later")),
        )
        result = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=surfaces,
            surface_options=SurfaceEventOptions(subdivisions_per_segment=4),
            options=one_step_options(),
        )

        self.assertEqual(result.outcome, "surface-hit", result.failure_reason)
        self.assertEqual(result.terminal_target_id, "target-first-opaque")
        self.assertEqual(result.affine_length, 0.25)
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].affine_length, 0.25)
        trace = result.multi_surface_trace
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(
            tuple(entry.surface_id for entry in trace.crossings),
            ("first-opaque",),
        )
        self.assertTrue(trace.crossings[0].decision.terminates)

    def test_transparent_members_continue_to_later_terminal_member(self) -> None:
        surfaces = PlaneMultiSurface(
            {"transparent-a": 0.2, "transparent-b": 0.4, "opaque": 0.8},
            terminal_ids=frozenset(("opaque",)),
        )
        result = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=surfaces,
            surface_options=SurfaceEventOptions(subdivisions_per_segment=10),
            options=one_step_options(),
        )

        self.assertEqual(result.outcome, "surface-hit", result.failure_reason)
        self.assertAlmostEqual(result.affine_length, 0.8, delta=2.0e-10)
        trace = result.multi_surface_trace
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(
            tuple(entry.surface_id for entry in trace.crossings),
            ("transparent-a", "transparent-b", "opaque"),
        )
        self.assertEqual(
            tuple(entry.decision.terminates for entry in trace.crossings),
            (False, False, True),
        )

    def test_n_vs_2n_topology_mismatch_refines_the_geodesic_step(self) -> None:
        result = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=PolynomialMultiSurface(),
            surface_options=SurfaceEventOptions(subdivisions_per_segment=2),
            options=one_step_options(maximum_affine_length=1.0),
        )

        self.assertEqual(result.outcome, "surface-hit", result.failure_reason)
        self.assertGreaterEqual(result.rejected_steps, 1)
        self.assertAlmostEqual(result.affine_length, 0.3, delta=2.0e-10)
        trace = result.multi_surface_trace
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertTrue(trace.topology_converged)
        self.assertEqual(
            tuple(entry.decision.classification for entry in trace.crossings),
            ("transparent-polynomial", "opaque-polynomial"),
        )

    def test_simultaneous_different_member_roots_fail_independent_of_id_order(
        self,
    ) -> None:
        for returned_order in (("a", "b"), ("b", "a")):
            with self.subTest(returned_order=returned_order):
                result = trace_null_geodesic(
                    MinkowskiMetric(),
                    INITIAL_X_RAY,
                    multi_interior_surface=PlaneMultiSurface(
                        {"a": 0.5, "b": 0.5},
                        terminal_ids=frozenset(("a", "b")),
                        returned_id_order=returned_order,
                    ),
                    surface_options=SurfaceEventOptions(
                        subdivisions_per_segment=2
                    ),
                    options=one_step_options(),
                )
                self.assertEqual(result.outcome, "integrator-failure")
                self.assertIn(
                    "simultaneous or unresolved-order",
                    result.failure_reason or "",
                )
                trace = result.multi_surface_trace
                self.assertIsNotNone(trace)
                assert trace is not None
                self.assertFalse(trace.topology_converged)
                self.assertEqual(trace.crossings, ())

    def test_hidden_simultaneous_roots_after_first_opaque_do_not_poison_prefix(
        self,
    ) -> None:
        surface_options = SurfaceEventOptions(subdivisions_per_segment=10)
        result = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=PlaneMultiSurface(
                {
                    "visible-opaque": 0.2,
                    "hidden-a": 0.81337,
                    "hidden-b": 0.81337,
                },
                terminal_ids=frozenset(("visible-opaque", "hidden-a", "hidden-b")),
            ),
            surface_options=surface_options,
            options=one_step_options(),
        )
        baseline = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=PlaneMultiSurface(
                {"visible-opaque": 0.2},
                terminal_ids=frozenset(("visible-opaque",)),
            ),
            surface_options=surface_options,
            options=one_step_options(),
        )
        self.assertEqual(result.outcome, "surface-hit", result.failure_reason)
        self.assertEqual(result.terminal_target_id, "target-visible-opaque")
        self.assertAlmostEqual(result.affine_length, 0.2, delta=2.0e-10)
        assert result.multi_surface_trace is not None
        self.assertEqual(
            tuple(
                entry.surface_id
                for entry in result.multi_surface_trace.crossings
            ),
            ("visible-opaque",),
        )
        assert baseline.multi_surface_trace is not None
        self.assertEqual(
            result.multi_surface_trace.probe_reintegrations,
            baseline.multi_surface_trace.probe_reintegrations,
        )

    def test_simultaneous_roots_before_first_opaque_still_fail_closed(self) -> None:
        result = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=PlaneMultiSurface(
                {"ambiguous-a": 0.2, "ambiguous-b": 0.2, "opaque": 0.8},
                terminal_ids=frozenset(("opaque",)),
            ),
            surface_options=SurfaceEventOptions(subdivisions_per_segment=10),
            options=one_step_options(),
        )
        self.assertEqual(result.outcome, "integrator-failure")
        self.assertIn(
            "simultaneous or unresolved-order",
            result.failure_reason or "",
        )

    def test_tangent_member_contact_fails_closed(self) -> None:
        result = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=TangentMultiSurface(),
            surface_options=SurfaceEventOptions(subdivisions_per_segment=2),
            options=one_step_options(),
        )
        self.assertEqual(result.outcome, "integrator-failure")
        self.assertIn("tangent multi-surface", result.failure_reason or "")

    def test_worldtube_and_multi_surface_are_ordered_within_one_step(self) -> None:
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
            multi_interior_surface=PlaneMultiSurface(
                {"after-boundary": 3.0},
                terminal_ids=frozenset(("after-boundary",)),
            ),
            options=options,
        )
        self.assertEqual(boundary_first.outcome, "escaped")
        self.assertAlmostEqual(
            boundary_first.terminal_state.event[1],
            2.0,
            delta=2.0e-9,
        )
        assert boundary_first.multi_surface_trace is not None
        self.assertEqual(boundary_first.multi_surface_trace.crossings, ())

        surface_first = trace_null_geodesic(
            MinkowskiMetric(),
            initial,
            termination=RadialTermination(0.5, 2.0),
            multi_interior_surface=PlaneMultiSurface(
                {"before-boundary": 1.5},
                terminal_ids=frozenset(("before-boundary",)),
            ),
            options=options,
        )
        self.assertEqual(surface_first.outcome, "surface-hit")
        self.assertAlmostEqual(
            surface_first.terminal_state.event[1],
            1.5,
            delta=2.0e-9,
        )

    def test_shared_probe_reintegration_count_does_not_scale_with_members(
        self,
    ) -> None:
        options = one_step_options(maximum_affine_length=1.0)
        surface_options = SurfaceEventOptions(subdivisions_per_segment=4)
        one = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=PlaneMultiSurface({"one": 10.0}),
            surface_options=surface_options,
            options=options,
        )
        two = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=PlaneMultiSurface(
                {"one": 10.0, "two": 20.0}
            ),
            surface_options=surface_options,
            options=options,
        )
        assert one.multi_surface_trace is not None
        assert two.multi_surface_trace is not None
        self.assertEqual(one.outcome, "completed")
        self.assertEqual(two.outcome, "completed")
        self.assertEqual(one.multi_surface_trace.probe_reintegrations, 8)
        self.assertEqual(
            two.multi_surface_trace.probe_reintegrations,
            one.multi_surface_trace.probe_reintegrations,
        )
        self.assertEqual(one.multi_surface_trace.surface_value_evaluations, 9)
        self.assertEqual(two.multi_surface_trace.surface_value_evaluations, 18)

    def test_multi_surface_work_budget_is_global_and_fail_closed(self) -> None:
        result = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=PlaneMultiSurface(
                {"one": 10.0, "two": 20.0}
            ),
            surface_options=SurfaceEventOptions(
                subdivisions_per_segment=2,
                maximum_reintegrations=3,
            ),
            options=one_step_options(),
        )
        self.assertEqual(result.outcome, "integrator-failure")
        self.assertIn("work budget exhausted", result.failure_reason or "")
        trace = result.multi_surface_trace
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.probe_reintegrations, 3)
        self.assertFalse(trace.topology_converged)

    def test_ids_are_unique_stable_and_single_api_is_mutually_exclusive(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "must be unique"):
            trace_null_geodesic(
                MinkowskiMetric(),
                INITIAL_X_RAY,
                multi_interior_surface=PlaneMultiSurface(
                    {"a": 0.5},
                    returned_id_order=("a", "a"),
                ),
            )
        changing = trace_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=MutableIdMultiSurface({"a": 10.0}),
            options=one_step_options(),
        )
        self.assertEqual(changing.outcome, "integrator-failure")
        self.assertIn("ids changed", changing.failure_reason or "")
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            trace_null_geodesic(
                MinkowskiMetric(),
                INITIAL_X_RAY,
                interior_surface=SinglePlaneSurface(),
                multi_interior_surface=PlaneMultiSurface({"a": 0.5}),
            )

    def test_whole_ray_refinement_compares_multi_surface_terminal_member(
        self,
    ) -> None:
        result = trace_refined_null_geodesic(
            MinkowskiMetric(),
            INITIAL_X_RAY,
            multi_interior_surface=PlaneMultiSurface(
                {"transparent": 0.2, "opaque": 0.7},
                terminal_ids=frozenset(("opaque",)),
            ),
            surface_options=SurfaceEventOptions(
                subdivisions_per_segment=4
            ),
            fine_options=RayTraceOptions(
                initial_step=0.25,
                maximum_step=0.25,
                maximum_affine_length=1.0,
            ),
            terminal_event_tolerance=1.0e-9,
            terminal_covector_tolerance=1.0e-9,
        )

        self.assertTrue(result.outcome_agrees)
        self.assertTrue(result.discretizations_differ)
        self.assertTrue(result.terminal_target_agrees)
        self.assertEqual(result.fine.outcome, "surface-hit")
        self.assertEqual(result.fine.terminal_target_id, "target-opaque")
        self.assertTrue(result.converged)
        for trace_result in (result.fine, result.coarse):
            trace = trace_result.multi_surface_trace
            self.assertIsNotNone(trace)
            assert trace is not None
            self.assertEqual(
                tuple(entry.surface_id for entry in trace.crossings),
                ("transparent", "opaque"),
            )


if __name__ == "__main__":
    unittest.main()
