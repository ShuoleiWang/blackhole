from __future__ import annotations

import hashlib
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.job import canonical_json_bytes
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_finite_thickness import (
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
from offline.kerr_native_cpu_backend import KerrNativeCpuRuntimeBinding
from offline.kerr_returning_radiation_kernel import (
    KerrReturningRadiationKernelPolicy,
)
from offline.kerr_returning_radiation_kernel_cached import (
    build_forward_kerr_returning_radiation_kernel_cache_definition,
)
import offline.kerr_returning_radiation_kernel_jobs as jobs_module
from offline.kerr_returning_radiation_kernel_jobs import (
    FORWARD,
    KerrKernelDirectionCoordinate,
    KerrKernelDirectionTaskPlan,
    KerrKernelScientificContext,
    build_forward_kernel_scientific_identity,
    kernel_direction_evaluator_process_local_key,
    make_kernel_direction_cache_definition,
    make_kernel_direction_evaluator_input,
    make_kernel_direction_evaluator_runtime_binding,
    revalidate_kernel_direction_evaluator_runtime_binding,
    run_kernel_direction_cache,
)


EVALUATOR_ID = "tests.native_runtime_binding.forward/v1"
DRIFT_EVALUATOR_ID = "tests.native_runtime_binding.drifting/v1"
LEGACY_SCIENTIFIC_KEY = (
    "1d9256788df8664fd9910df2edadeed70b7be6a9a2f2fc935917d83dedddc8ad"
)
LEGACY_CACHE_KEY = (
    "7ec29e2d1e8e9619d98ac91b31160fffbe0e5bd5864085ea3d9dfcdb55a6b91b"
)
FROZEN_NESTED16_SCIENTIFIC_KEY = (
    "19c8c3ebd465a4f66d6e893935dc18134ecd47d8842b2e77809823d4148ec980"
)
FROZEN_NESTED16_CACHE_KEY = (
    "d8286face863a0755d70c20c1638c6a5dc4d9d9037c0227626fb805bb4d0f751"
)
_DRIFT_LIBRARY: Path | None = None


def valid_evaluator(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate,
) -> dict[str, object]:
    return {}


def drifting_evaluator(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate,
) -> dict[str, object]:
    if _DRIFT_LIBRARY is None:
        raise AssertionError("drift target is not configured")
    _DRIFT_LIBRARY.write_bytes(b"changed-after-worker-preflight")
    return {}


def evaluator_with_default(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate | None = None,
) -> dict[str, object]:
    return {}


def evaluator_with_keyword_only(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate,
    *,
    backend: object,
) -> dict[str, object]:
    return {}


def evaluator_with_varargs(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate,
    *extra: object,
) -> dict[str, object]:
    return {}


def generator_evaluator(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate,
):
    yield {}


async def coroutine_evaluator(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate,
) -> dict[str, object]:
    return {}


def scientific_definition_inputs():
    metric = KerrKerrSchildMetric(1.0, 0.7, 1.0e-9)
    calibration = StationaryKerrFiniteThicknessCalibration(
        0.7,
        0.05,
        "prograde",
        25.0,
        0.25,
    )
    surface = KerrFiniteThicknessMultiSurface(metric, calibration)
    termination = KerrOblateTermination(
        0.7,
        1.7341428428542849,
        50.0,
        "analytic-kerr-stretched-horizon",
        "analytic-kerr-escape-worldtube",
    )
    ray = RayTraceOptions(
        absolute_tolerance=6.25e-11,
        relative_tolerance=6.25e-11,
        initial_step=0.025,
        minimum_step=1.25e-9,
        maximum_step=0.125,
        maximum_affine_length=300.0,
        maximum_accepted_steps=100_000,
        maximum_rejected_steps=100_000,
        null_residual_limit=2.5e-8,
        metric_interpolation_error_limit=1.25e-8,
        event_value_tolerance=1.25e-10,
        event_affine_tolerance=1.25e-11,
        event_maximum_iterations=64,
        record_path=True,
    )
    surface_options = SurfaceEventOptions(
        absolute_tolerance=7.8125e-12,
        relative_tolerance=7.8125e-12,
        null_residual_limit=3.125e-9,
        metric_interpolation_error_limit=1.5625e-9,
        surface_value_tolerance=1.5625e-11,
        affine_tolerance=1.5625e-12,
        maximum_iterations=64,
        maximum_reintegrations=100_000,
        subdivisions_per_segment=16,
    )
    plan = KerrKernelDirectionTaskPlan(FORWARD, 1, 4, 4, 4, 4)
    identity = build_forward_kernel_scientific_identity(
        surface=surface,
        termination=termination,
        annulus_edges_over_mass=(calibration.isco_radius_over_mass, 25.0),
        fine_ray_options=ray,
        fine_surface_options=surface_options,
        coarse_ray_options=None,
        coarse_surface_options=None,
        source_closure_sha256=("0" * 64,),
    )
    return plan, identity


def backend_descriptor() -> dict[str, object]:
    return {
        "abiVersion": 3,
        "artifacts": {
            "library": {
                "logicalPath": "runtime/libblackhole_cpu.dylib",
                "sha256Semantics": "exact-file-bytes",
            }
        },
        "implementationId": "strict-kerr-cpu-whole-ray/abi-v3",
    }


def runtime_definition(
    library: Path,
    *,
    evaluator=valid_evaluator,
    evaluator_id: str = EVALUATOR_ID,
):
    binding = make_kernel_direction_evaluator_runtime_binding(
        evaluator_implementation_id=evaluator_id,
        library_path=library,
        backend_descriptor=backend_descriptor(),
        segment_capacity=100_000,
        crossing_capacity=100_000,
    )
    evaluator_input = make_kernel_direction_evaluator_input(
        evaluator,
        implementation_id=evaluator_id,
    )
    plan, identity = scientific_definition_inputs()
    definition = make_kernel_direction_cache_definition(
        plan,
        identity=identity,
        inputs=(evaluator_input, binding.descriptor_input),
        evaluator_runtime_binding=binding,
    )
    return definition, binding


class NativeRuntimeBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        global _DRIFT_LIBRARY
        _DRIFT_LIBRARY = None
        self.temporary.cleanup()

    def library(
        self,
        parent: str,
        payload: bytes = b"strict-runtime",
        name: str = "libblackhole_cpu.dylib",
    ) -> Path:
        directory = self.root / parent
        directory.mkdir()
        path = directory / name
        path.write_bytes(payload)
        return path

    def test_legacy_documents_and_keys_are_byte_identical(self) -> None:
        plan, identity = scientific_definition_inputs()
        definition = make_kernel_direction_cache_definition(plan, identity=identity)
        self.assertIsNone(definition.scientific_context.evaluator_runtime_binding)
        self.assertEqual(definition.scientific_job_key, LEGACY_SCIENTIFIC_KEY)
        self.assertEqual(definition.job_spec.job_key, LEGACY_CACHE_KEY)
        self.assertNotIn(
            "evaluatorRuntimeBinding",
            definition.scientific_context.scientific_document(),
        )

    def test_current_source_closure_cannot_reuse_frozen_nested16_cache(self) -> None:
        plan, identity = scientific_definition_inputs()
        policy = KerrReturningRadiationKernelPolicy(
            rho_order=16,
            mu_order=32,
            psi_count=64,
            absolute_tolerance=2.0e-2,
            relative_tolerance=5.0e-2,
            symmetry_absolute_tolerance=2.0e-8,
            symmetry_relative_tolerance=2.0e-7,
            maximum_direction_evaluations=2_000_000,
            maximum_whole_ray_traces=8_000_000,
        )
        definition = build_forward_kerr_returning_radiation_kernel_cache_definition(
            identity.surface,
            termination=identity.termination,
            annulus_edges_over_mass=identity.annulus_edges_over_mass,
            ray_options=identity.fine_ray_options,
            surface_options=identity.fine_surface_options,
            coarse_ray_options=identity.coarse_ray_options,
            coarse_surface_options=identity.coarse_surface_options,
            policy=policy,
            area_policy=KerrFiniteThicknessAreaQuadraturePolicy(),
            directions_per_task=64,
        )
        self.assertEqual(plan.formulation, definition.plan.formulation)
        self.assertNotEqual(
            definition.scientific_job_key,
            FROZEN_NESTED16_SCIENTIFIC_KEY,
        )
        self.assertNotEqual(definition.job_spec.job_key, FROZEN_NESTED16_CACHE_KEY)

    def test_binding_identity_is_path_independent_and_documents_omit_paths(
        self,
    ) -> None:
        first_path = self.library("first")
        second_path = self.library("second", name="renamed-native-core.dylib")
        first, first_binding = runtime_definition(first_path)
        second, second_binding = runtime_definition(second_path)
        self.assertNotEqual(first_binding.library_path, second_binding.library_path)
        self.assertEqual(first_binding.descriptor_json, second_binding.descriptor_json)
        self.assertEqual(first_binding.descriptor_input, second_binding.descriptor_input)
        self.assertEqual(first.scientific_job_key, second.scientific_job_key)
        self.assertEqual(first.job_spec.job_key, second.job_spec.job_key)
        for document in (
            first.scientific_context.scientific_document(),
            first.job_spec.as_dict(),
        ):
            payload = canonical_json_bytes(document)
            self.assertNotIn(os.fspath(first_path).encode("utf-8"), payload)
            self.assertNotIn(os.fspath(second_path).encode("utf-8"), payload)

    def test_binding_is_spawn_pickleable_and_exposes_process_hook(self) -> None:
        path = self.library("pickle")
        _definition, binding = runtime_definition(path)
        restored = pickle.loads(pickle.dumps(binding))
        self.assertEqual(restored, binding)
        self.assertIsInstance(restored, KerrNativeCpuRuntimeBinding)
        self.assertEqual(
            revalidate_kernel_direction_evaluator_runtime_binding(restored),
            binding,
        )
        expected_backend_sha = hashlib.sha256(
            canonical_json_bytes(backend_descriptor())
        ).hexdigest()
        self.assertEqual(binding.backend_descriptor_sha256, expected_backend_sha)
        self.assertEqual(
            binding.path_free_descriptor(),
            {
                "backendDescriptorSha256": expected_backend_sha,
                "callerOwnedCapacities": {
                    "crossings": 100_000,
                    "segments": 100_000,
                },
                "implementationId": (
                    "strict-kerr-cpu-process-runtime-binding/v1"
                ),
            },
        )
        process_key = kernel_direction_evaluator_process_local_key(restored)
        self.assertEqual(process_key[0], os.getpid())
        self.assertEqual(process_key[1], os.fspath(path))
        self.assertEqual(process_key[2], binding.descriptor_input.sha256)

    def test_bad_callable_shapes_fail_the_descriptor_gate(self) -> None:
        for evaluator in (
            evaluator_with_default,
            evaluator_with_keyword_only,
            evaluator_with_varargs,
            generator_evaluator,
            coroutine_evaluator,
        ):
            with self.subTest(evaluator=evaluator.__name__):
                with self.assertRaisesRegex(ValueError, "reserved evaluator"):
                    make_kernel_direction_evaluator_input(
                        evaluator,
                        implementation_id=EVALUATOR_ID,
                    )

    def test_runtime_input_mismatch_fails_context_construction(self) -> None:
        path = self.library("mismatch")
        binding = make_kernel_direction_evaluator_runtime_binding(
            evaluator_implementation_id=EVALUATOR_ID,
            library_path=path,
            backend_descriptor=backend_descriptor(),
            segment_capacity=1,
            crossing_capacity=1,
        )
        plan, identity = scientific_definition_inputs()
        evaluator_input = make_kernel_direction_evaluator_input(
            valid_evaluator,
            implementation_id=EVALUATOR_ID,
        )
        with self.assertRaisesRegex(ValueError, "runtime binding differs"):
            make_kernel_direction_cache_definition(
                plan,
                identity=identity,
                inputs=(evaluator_input,),
                evaluator_runtime_binding=binding,
            )

    def test_bad_signature_and_runtime_drift_fail_before_cache_session(self) -> None:
        path = self.library("pre-cache")
        definition, _binding = runtime_definition(path)
        never_cache = self.root / "must-not-open-cache"
        with patch.object(
            jobs_module,
            "_secure_cache_session",
            side_effect=AssertionError("cache session opened"),
        ) as session:
            with self.assertRaisesRegex(
                jobs_module.KerrReturningRadiationKernelJobError,
                "reserved evaluator",
            ):
                run_kernel_direction_cache(
                    definition,
                    evaluator_with_keyword_only,
                    never_cache,
                )
        session.assert_not_called()

        path.write_bytes(b"drifted-before-cache-open")
        with patch.object(
            jobs_module,
            "_secure_cache_session",
            side_effect=AssertionError("cache session opened"),
        ) as session:
            with self.assertRaisesRegex(ValueError, "runtime library"):
                run_kernel_direction_cache(
                    definition,
                    valid_evaluator,
                    never_cache,
                )
        session.assert_not_called()

    def test_worker_rehashes_runtime_after_task_before_returning_payload(self) -> None:
        global _DRIFT_LIBRARY
        path = self.library("worker")
        definition, _binding = runtime_definition(
            path,
            evaluator=drifting_evaluator,
            evaluator_id=DRIFT_EVALUATOR_ID,
        )
        producer = jobs_module._KernelDirectionTaskProducer(
            definition.plan,
            definition.scientific_context,
            definition.job_spec.job_key,
            drifting_evaluator,
        )
        _DRIFT_LIBRARY = path
        with self.assertRaisesRegex(ValueError, "runtime library"):
            producer(definition.job_spec.tasks[0])

    def test_worker_rejects_mismatched_evaluator_before_coordinate_work(self) -> None:
        path = self.library("worker-evaluator")
        definition, _binding = runtime_definition(path)
        producer = jobs_module._KernelDirectionTaskProducer(
            definition.plan,
            definition.scientific_context,
            definition.job_spec.job_key,
            evaluator_with_default,
        )
        with self.assertRaisesRegex(
            jobs_module.KerrReturningRadiationKernelJobError,
            "reserved evaluator",
        ):
            producer(definition.job_spec.tasks[0])


if __name__ == "__main__":
    unittest.main()
