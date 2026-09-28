"""Byte-exact whole-ray replay against the authenticated nested16 corpus."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import struct
import unittest

from native.cpu.tests.binding import (
    BH_CPU_ABI_VERSION,
    BH_CPU_CAPACITY_EXCEEDED,
    BH_CPU_INVALID_ARGUMENT,
    BH_CPU_OK,
    BH_CPU_RAY_FAILURE_NONE,
    BH_CPU_RAY_OUTCOME_CAPTURED,
    BH_CPU_RAY_OUTCOME_ESCAPED,
    BH_CPU_RAY_OUTCOME_NONE,
    BH_CPU_RAY_OUTCOME_PLUNGE_SINK,
    BH_CPU_RAY_OUTCOME_RETURNED,
    BH_CPU_SURFACE_CLASSIFICATION_NONE,
    BH_CPU_SURFACE_INSIDE_ISCO_TRANSPARENT,
    BH_CPU_SURFACE_INWARD_LOWER_PLUNGE_ENTRY,
    BH_CPU_SURFACE_INWARD_UPPER_PLUNGE_ENTRY,
    BH_CPU_SURFACE_LOWER,
    BH_CPU_SURFACE_OUTSIDE_OUTER_TRANSPARENT,
    BH_CPU_SURFACE_OUTWARD_LOWER_PLUNGE_EXIT,
    BH_CPU_SURFACE_OUTWARD_UPPER_PLUNGE_EXIT,
    BH_CPU_SURFACE_SUBSEQUENT_LOWER_CONTACT,
    BH_CPU_SURFACE_SUBSEQUENT_UPPER_CONTACT,
    BH_CPU_SURFACE_UPPER,
    BH_CPU_TARGET_KERR_ESCAPE_WORLDTUBE,
    BH_CPU_TARGET_KERR_STRETCHED_HORIZON,
    BH_CPU_TARGET_LOWER_PLUNGE_ENTRY,
    BH_CPU_TARGET_NONE,
    BH_CPU_TARGET_OPAQUE_LOWER_FACE,
    BH_CPU_TARGET_OPAQUE_UPPER_FACE,
    BH_CPU_TARGET_UPPER_PLUNGE_ENTRY,
    KerrRayOptions,
    KerrRayResult,
    KerrReturningModel,
    KerrSurfaceOptions,
    RayPathSegment,
    RecordedSurfaceCrossing,
    double_array,
    load_library,
)
from offline.kerr_finite_thickness import (
    StationaryKerrFiniteThicknessCalibration,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = (
    REPOSITORY_ROOT / "tools/native/nested16_phase_space_golden.json"
)

OUTCOMES = {
    "captured": BH_CPU_RAY_OUTCOME_CAPTURED,
    "escaped": BH_CPU_RAY_OUTCOME_ESCAPED,
    "entered-unmodelled-plunge-sink": BH_CPU_RAY_OUTCOME_PLUNGE_SINK,
    "returned-to-finite-thickness-photosphere": BH_CPU_RAY_OUTCOME_RETURNED,
}
TARGETS = {
    None: BH_CPU_TARGET_NONE,
    "analytic-kerr-escape-worldtube": BH_CPU_TARGET_KERR_ESCAPE_WORLDTUBE,
    "analytic-kerr-stretched-horizon": BH_CPU_TARGET_KERR_STRETCHED_HORIZON,
    "finite-thickness-lower-plunge-entry": BH_CPU_TARGET_LOWER_PLUNGE_ENTRY,
    "finite-thickness-upper-plunge-entry": BH_CPU_TARGET_UPPER_PLUNGE_ENTRY,
    "kerr-finite-thickness-opaque-lower-face": BH_CPU_TARGET_OPAQUE_LOWER_FACE,
    "kerr-finite-thickness-opaque-upper-face": BH_CPU_TARGET_OPAQUE_UPPER_FACE,
}
SURFACES = {
    "lower": BH_CPU_SURFACE_LOWER,
    "upper": BH_CPU_SURFACE_UPPER,
    "kerr-finite-thickness-lower-photosphere": BH_CPU_SURFACE_LOWER,
    "kerr-finite-thickness-upper-photosphere": BH_CPU_SURFACE_UPPER,
}
CLASSIFICATIONS = {
    None: BH_CPU_SURFACE_CLASSIFICATION_NONE,
    "inside-isco-transparent": BH_CPU_SURFACE_INSIDE_ISCO_TRANSPARENT,
    "outside-outer-radius-transparent": BH_CPU_SURFACE_OUTSIDE_OUTER_TRANSPARENT,
    "outward-lower-continuation-plunge-exit-transparent": (
        BH_CPU_SURFACE_OUTWARD_LOWER_PLUNGE_EXIT
    ),
    "outward-upper-continuation-plunge-exit-transparent": (
        BH_CPU_SURFACE_OUTWARD_UPPER_PLUNGE_EXIT
    ),
    "subsequent-lower-photosphere-contact": (
        BH_CPU_SURFACE_SUBSEQUENT_LOWER_CONTACT
    ),
    "subsequent-upper-photosphere-contact": (
        BH_CPU_SURFACE_SUBSEQUENT_UPPER_CONTACT
    ),
    "inward-lower-continuation-plunge-entry": (
        BH_CPU_SURFACE_INWARD_LOWER_PLUNGE_ENTRY
    ),
    "inward-upper-continuation-plunge-entry": (
        BH_CPU_SURFACE_INWARD_UPPER_PLUNGE_ENTRY
    ),
}


def bits(value: float) -> bytes:
    return struct.pack(">d", value)


def packed_state(document: dict[str, object]) -> tuple[float, ...]:
    return tuple(document["event"]) + tuple(document["covector"])  # type: ignore[arg-type]


def ray_options(document: dict[str, object]) -> KerrRayOptions:
    result = KerrRayOptions()
    result.abi_version = BH_CPU_ABI_VERSION
    result.struct_size = ctypes.sizeof(result)
    for name in (
        "absolute_tolerance",
        "relative_tolerance",
        "initial_step",
        "minimum_step",
        "maximum_step",
        "maximum_affine_length",
        "null_residual_limit",
        "metric_interpolation_error_limit",
        "event_value_tolerance",
        "event_affine_tolerance",
        "maximum_accepted_steps",
        "maximum_rejected_steps",
        "event_maximum_iterations",
    ):
        setattr(result, name, document[name])
    result.record_path = int(document["record_path"] is True)
    return result


def surface_options(document: dict[str, object]) -> KerrSurfaceOptions:
    result = KerrSurfaceOptions()
    result.abi_version = BH_CPU_ABI_VERSION
    result.struct_size = ctypes.sizeof(result)
    for name in (
        "absolute_tolerance",
        "relative_tolerance",
        "null_residual_limit",
        "metric_interpolation_error_limit",
        "surface_value_tolerance",
        "affine_tolerance",
        "maximum_reintegrations",
        "maximum_iterations",
        "subdivisions_per_segment",
    ):
        setattr(result, name, document[name])
    return result


class WholeRayParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.library = load_library()
        cls.golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        identity = cls.golden["reference"]["scientificDocument"][
            "scientificIdentity"
        ]
        cls.identity = identity
        calibration_document = identity["surface"]["calibration"]
        cls.calibration = StationaryKerrFiniteThicknessCalibration(
            **calibration_document
        )

    def test_abi3_struct_layouts_are_frozen(self) -> None:
        self.assertEqual(ctypes.sizeof(KerrRayOptions), 112)
        self.assertEqual(ctypes.sizeof(KerrSurfaceOptions), 72)
        self.assertEqual(ctypes.sizeof(KerrReturningModel), 80)
        self.assertEqual(ctypes.sizeof(RayPathSegment), 208)
        self.assertEqual(ctypes.sizeof(RecordedSurfaceCrossing), 128)
        self.assertEqual(ctypes.sizeof(KerrRayResult), 216)

    def assert_float_exact(
        self,
        actual: float,
        expected: float,
        label: str,
    ) -> None:
        self.assertEqual(bits(actual), bits(expected), label)

    def make_model(self, record: dict[str, object]) -> KerrReturningModel:
        identity = self.identity
        metric = identity["metric"]
        termination = identity["termination"]
        model = KerrReturningModel()
        model.abi_version = BH_CPU_ABI_VERSION
        model.struct_size = ctypes.sizeof(model)
        model.mass_m = metric["mass_m"]
        model.spin_a_m = metric["spin_a_m"]
        model.singularity_guard_m = metric["singularity_guard_m"]
        model.capture_radius_m = termination["capture_radius_m"]
        model.escape_radius_m = termination["escape_radius_m"]
        model.isco_radius_over_mass = self.calibration.isco_radius_over_mass
        model.outer_radius_over_mass = self.calibration.outer_radius_over_mass
        model.asymptotic_pressure_scale_height_over_mass = (
            self.calibration.asymptotic_pressure_scale_height_over_mass
        )
        model.emitting_surface_id = SURFACES[record["sample"]["sourceFace"]]
        model.initial_contact_side = 1
        return model

    def trace(
        self,
        record: dict[str, object],
        resolution: str,
        *,
        segment_capacity: int | None = None,
        crossing_capacity: int | None = None,
    ):
        expected = record["phaseSpace"][resolution]
        policy = record["referenceEvidence"]["primitiveDescriptor"][
            "numericalPolicy"
        ]
        prefix = "fine" if resolution == "fine" else "coarse"
        ray = ray_options(policy[f"{prefix}RayOptions"])
        surface = surface_options(policy[f"{prefix}SurfaceOptions"])
        initial_state = double_array(
            packed_state(expected["segments"][0]["start"])
        )
        if segment_capacity is None:
            segment_capacity = len(expected["segments"])
        expected_crossings = expected["surfaceTrace"]["crossings"]
        if crossing_capacity is None:
            crossing_capacity = len(expected_crossings)
        segments = (
            (RayPathSegment * segment_capacity)()
            if segment_capacity > 0
            else None
        )
        crossings = (
            (RecordedSurfaceCrossing * crossing_capacity)()
            if crossing_capacity > 0
            else None
        )
        result = KerrRayResult()
        status = self.library.bh_cpu_trace_kerr_returning_ray(
            initial_state,
            ctypes.byref(self.make_model(record)),
            ctypes.byref(ray),
            ctypes.byref(surface),
            segments,
            segment_capacity,
            crossings,
            crossing_capacity,
            ctypes.byref(result),
        )
        return status, result, segments, crossings, expected

    def assert_state_exact(
        self,
        actual: ctypes.Array[ctypes.c_double],
        expected: dict[str, object],
        label: str,
    ) -> None:
        packed = packed_state(expected)
        self.assertEqual(len(actual), len(packed), label)
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, packed)
        ):
            self.assert_float_exact(
                actual_value,
                expected_value,
                f"{label}[{index}]",
            )

    def assert_whole_ray_exact(
        self,
        result: KerrRayResult,
        segments,
        crossings,
        expected: dict[str, object],
        label: str,
    ) -> None:
        surface_trace = expected["surfaceTrace"]
        expected_crossings = surface_trace["crossings"]
        self.assertEqual(result.abi_version, BH_CPU_ABI_VERSION, label)
        self.assertEqual(result.struct_size, ctypes.sizeof(result), label)
        self.assertEqual(result.outcome, OUTCOMES[expected["outcome"]], label)
        self.assertEqual(
            result.terminal_target,
            TARGETS[expected["terminalTargetId"]],
            label,
        )
        self.assertIsNone(expected["failureReason"], label)
        self.assertEqual(result.failure, BH_CPU_RAY_FAILURE_NONE, label)
        self.assertEqual(result.topology_converged, 1, label)
        self.assertEqual(
            result.initial_contact_surface_id,
            SURFACES[surface_trace["initialContact"]["surfaceId"]],
            label,
        )
        self.assertEqual(
            result.initial_contact_side,
            surface_trace["initialContact"]["side"],
            label,
        )
        self.assert_state_exact(
            result.terminal_state,
            expected["terminalState"],
            f"{label}.terminalState",
        )
        for field, expected_name in (
            ("affine_length", "affineLength"),
            ("maximum_null_residual", "maximumNullResidual"),
            (
                "maximum_metric_interpolation_error",
                "maximumMetricInterpolationError",
            ),
        ):
            self.assert_float_exact(
                getattr(result, field),
                expected[expected_name],
                f"{label}.{field}",
            )
        for field, expected_name in (
            ("accepted_steps", "acceptedSteps"),
            ("rejected_steps", "rejectedSteps"),
        ):
            self.assertEqual(
                getattr(result, field),
                expected[expected_name],
                f"{label}.{field}",
            )
        self.assertEqual(result.segment_count, len(expected["segments"]), label)
        self.assertEqual(result.crossing_count, len(expected_crossings), label)
        self.assertEqual(
            result.probe_reintegrations,
            surface_trace["probeReintegrations"],
            label,
        )
        self.assertEqual(
            result.surface_value_evaluations,
            surface_trace["surfaceValueEvaluations"],
            label,
        )
        self.assertGreater(result.metric_sample_evaluations, 0, label)
        self.assertGreater(result.hamiltonian_rhs_evaluations, 0, label)
        for field, expected_name in (
            ("maximum_probe_event_difference", "maximumProbeEventDifference"),
            (
                "maximum_probe_covector_relative_difference",
                "maximumProbeCovectorRelativeDifference",
            ),
            (
                "initial_contact_actual_surface_value",
                "actualSurfaceValue",
            ),
            (
                "initial_contact_surface_value_tolerance",
                "surfaceValueTolerance",
            ),
        ):
            source = (
                surface_trace
                if expected_name.startswith("maximum")
                else surface_trace["initialContact"]
            )
            self.assert_float_exact(
                getattr(result, field),
                source[expected_name],
                f"{label}.{field}",
            )

        for segment_index, (actual, golden) in enumerate(
            zip(segments, expected["segments"])
        ):
            segment_label = f"{label}.segments[{segment_index}]"
            self.assert_state_exact(actual.start, golden["start"], segment_label)
            self.assert_state_exact(actual.end, golden["end"], segment_label)
            self.assert_state_exact(
                actual.midpoint,
                golden["midpoint"],
                segment_label,
            )
            self.assert_float_exact(
                actual.affine_length,
                golden["affineLength"],
                f"{segment_label}.affineLength",
            )
            self.assert_float_exact(
                actual.midpoint_null_residual,
                golden["midpointNullResidual"],
                f"{segment_label}.midpointNullResidual",
            )

        for crossing_index, (actual, golden) in enumerate(
            zip(crossings or (), expected_crossings)
        ):
            crossing_label = f"{label}.crossings[{crossing_index}]"
            self.assert_state_exact(actual.state, golden["state"], crossing_label)
            for field, expected_name in (
                ("ray_affine_length", "rayAffineLength"),
                ("segment_affine_length", "segmentAffineLength"),
                ("surface_value", "surfaceValue"),
                ("bracket_affine_width", "bracketAffineWidth"),
            ):
                self.assert_float_exact(
                    getattr(actual, field),
                    golden[expected_name],
                    f"{crossing_label}.{field}",
                )
            self.assertEqual(actual.segment_index, golden["segmentIndex"], label)
            self.assertEqual(actual.iterations, golden["iterations"], label)
            self.assertEqual(actual.orientation, golden["orientation"], label)
            self.assertEqual(actual.surface_id, SURFACES[golden["surfaceId"]], label)
            decision = golden["decision"]
            self.assertEqual(
                actual.classification,
                CLASSIFICATIONS[decision["classification"]],
                label,
            )
            self.assertEqual(
                actual.outcome,
                OUTCOMES.get(decision["outcome"], BH_CPU_RAY_OUTCOME_NONE),
                label,
            )
            self.assertEqual(actual.target, TARGETS[decision["targetId"]], label)

    def test_all_frozen_whole_rays_are_byte_exact(self) -> None:
        self.assertEqual(
            self.golden["selectedOrdinals"],
            [369, 433, 726, 732, 736, 31263, 37331, 75785],
        )
        for record in self.golden["records"]:
            ordinal = record["coordinate"]["ordinal"]
            for resolution in ("fine", "coarse"):
                with self.subTest(ordinal=ordinal, resolution=resolution):
                    status, result, segments, crossings, expected = self.trace(
                        record,
                        resolution,
                    )
                    self.assertEqual(status, BH_CPU_OK)
                    self.assert_whole_ray_exact(
                        result,
                        segments,
                        crossings,
                        expected,
                        f"ordinal={ordinal}/{resolution}",
                    )

    def test_capacity_failure_publishes_nothing(self) -> None:
        record = self.golden["records"][0]
        expected = record["phaseSpace"]["fine"]
        policy = record["referenceEvidence"]["primitiveDescriptor"][
            "numericalPolicy"
        ]
        ray = ray_options(policy["fineRayOptions"])
        surface = surface_options(policy["fineSurfaceOptions"])
        initial_state = double_array(
            packed_state(expected["segments"][0]["start"])
        )
        result = KerrRayResult()
        ctypes.memset(ctypes.byref(result), 0xA5, ctypes.sizeof(result))
        before = bytes(result)
        status = self.library.bh_cpu_trace_kerr_returning_ray(
            initial_state,
            ctypes.byref(self.make_model(record)),
            ctypes.byref(ray),
            ctypes.byref(surface),
            None,
            0,
            None,
            0,
            ctypes.byref(result),
        )
        self.assertEqual(status, BH_CPU_CAPACITY_EXCEEDED)
        self.assertEqual(bytes(result), before)

    def test_auxiliary_emitter_contact_is_rejected_without_publication(self) -> None:
        record = self.golden["records"][0]
        policy = record["referenceEvidence"]["primitiveDescriptor"][
            "numericalPolicy"
        ]
        ray = ray_options(policy["fineRayOptions"])
        surface = surface_options(policy["fineSurfaceOptions"])
        state = double_array(
            tuple(
                float.fromhex(value)
                for value in (
                    "0x0.0p+0",
                    "0x1.9800000000000p+5",
                    "0x1.6657a7d2d77b3p-1",
                    "0x1.d42f6586af96ep-1",
                    "-0x1.ffff2c96e67dap-1",
                    "0x1.14e3ad1d70908p+0",
                    "0x1.0713c0b07305fp-62",
                    "0x1.7f886a0432140p-10",
                )
            )
        )
        model = self.make_model(record)
        model.emitting_surface_id = BH_CPU_SURFACE_UPPER
        result = KerrRayResult()
        ctypes.memset(ctypes.byref(result), 0xA5, ctypes.sizeof(result))
        before = bytes(result)
        status = self.library.bh_cpu_trace_kerr_returning_ray(
            state,
            ctypes.byref(model),
            ctypes.byref(ray),
            ctypes.byref(surface),
            None,
            0,
            None,
            0,
            ctypes.byref(result),
        )
        self.assertEqual(status, BH_CPU_INVALID_ARGUMENT)
        self.assertEqual(bytes(result), before)


if __name__ == "__main__":
    unittest.main()
