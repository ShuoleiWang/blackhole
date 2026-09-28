from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from offline.job import canonical_json_bytes
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_returning_radiation_convergence_v2 import (
    KerrReturningRadiationConvergenceV2Policy,
)
from offline.kerr_returning_radiation_kernel import (
    KerrReturningRadiationKernelPolicy,
)
import offline.kerr_returning_radiation_kernel_cached as cached_module
import offline.kerr_returning_radiation_refinement_checkpoint as checkpoint_module
from offline.kerr_returning_radiation_refinement_checkpoint import (
    NATIVE_CPU_EVALUATOR_MODE,
    PYTHON_EVALUATOR_MODE,
    KerrReturningRadiationRefinementPlan,
)
import scripts.run_offline_kerr_returning_radiation_refinement as cli_module
from tests.test_offline_kerr_native_runtime_binding import (
    scientific_definition_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = (ROOT / "native/cpu/build/libblackhole_cpu.dylib").absolute()
OLD_CACHE_KEY = (
    "d8286face863a0755d70c20c1638c6a5dc4d9d9037c0227626fb805bb4d0f751"
)


def policy() -> KerrReturningRadiationKernelPolicy:
    return KerrReturningRadiationKernelPolicy(
        rho_order=4,
        mu_order=4,
        psi_count=4,
        absolute_tolerance=0.25,
        relative_tolerance=0.25,
        symmetry_absolute_tolerance=0.25,
        symmetry_relative_tolerance=0.25,
        maximum_direction_evaluations=10_000,
        maximum_whole_ray_traces=20_000,
    )


def plan_arguments(output: Path, cache: Path) -> dict[str, object]:
    _task_plan, identity = scientific_definition_inputs()
    return {
        "output_directory": output,
        "cache_root": cache,
        "surface": identity.surface,
        "termination": identity.termination,
        "annulus_edges_over_mass": identity.annulus_edges_over_mass,
        "ray_options": identity.fine_ray_options,
        "surface_options": identity.fine_surface_options,
        "kernel_policy": policy(),
        "area_policy": KerrFiniteThicknessAreaQuadraturePolicy(),
        "convergence_policy": KerrReturningRadiationConvergenceV2Policy(),
        "directions_per_task": 16,
        "jobs": 1,
    }


class NativeRefinementRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temporary.name)
        self.output = self.root / "checkpoint"
        self.cache = self.root / "cache"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def construct_plan(self, **updates) -> KerrReturningRadiationRefinementPlan:
        arguments = plan_arguments(self.output, self.cache)
        arguments.update(updates)
        with (
            patch.object(checkpoint_module, "_require_exact_source_module_origins"),
            patch.object(checkpoint_module, "_assert_runtime_bindings"),
        ):
            return KerrReturningRadiationRefinementPlan(**arguments)

    def test_plan_modes_are_exact_and_native_controls_never_cross_modes(self) -> None:
        python = self.construct_plan()
        self.assertEqual(python.evaluator_mode, PYTHON_EVALUATOR_MODE)
        self.assertIsNone(python.native_library_path)
        native = self.construct_plan(
            evaluator_mode=NATIVE_CPU_EVALUATOR_MODE,
            native_library_path=LIBRARY,
            native_segment_capacity=100_000,
            native_crossing_capacity=100_000,
        )
        self.assertEqual(native.evaluator_mode, NATIVE_CPU_EVALUATOR_MODE)
        self.assertEqual(native.native_library_path, LIBRARY)

        for updates, message in (
            (
                {"native_library_path": LIBRARY},
                "python evaluator mode cannot carry native",
            ),
            (
                {"evaluator_mode": NATIVE_CPU_EVALUATOR_MODE},
                "requires an exact absolute dylib",
            ),
            (
                {
                    "evaluator_mode": NATIVE_CPU_EVALUATOR_MODE,
                    "native_library_path": Path("relative.dylib"),
                    "native_segment_capacity": 1,
                    "native_crossing_capacity": 1,
                },
                "exact absolute",
            ),
        ):
            with self.subTest(updates=updates):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    self.construct_plan(**updates)

    def test_definition_and_integrator_route_consistently_without_fallback(self) -> None:
        python = self.construct_plan()
        native = self.construct_plan(
            evaluator_mode=NATIVE_CPU_EVALUATOR_MODE,
            native_library_path=LIBRARY,
            native_segment_capacity=123,
            native_crossing_capacity=456,
        )
        python_definition = object()
        native_definition = object()
        python_execution = object()
        native_execution = object()
        with (
            patch.object(
                checkpoint_module,
                "_CACHE_DEFINITION_CALL_ENTRY",
                return_value=python_definition,
            ) as python_builder,
            patch.object(
                checkpoint_module,
                "_NATIVE_CACHE_DEFINITION_CALL_ENTRY",
                return_value=native_definition,
            ) as native_builder,
            patch.object(
                checkpoint_module,
                "_CACHED_INTEGRATOR_CALL_ENTRY",
                return_value=python_execution,
            ) as python_integrator,
            patch.object(
                checkpoint_module,
                "_NATIVE_CACHED_INTEGRATOR_CALL_ENTRY",
                return_value=native_execution,
            ) as native_integrator,
        ):
            self.assertIs(
                checkpoint_module._cache_definition_for_plan(python),
                python_definition,
            )
            self.assertIs(
                checkpoint_module._cache_definition_for_plan(native),
                native_definition,
            )
            self.assertIs(
                checkpoint_module._cached_execution_for_plan(python, self.cache),
                python_execution,
            )
            self.assertIs(
                checkpoint_module._cached_execution_for_plan(native, self.cache),
                native_execution,
            )
        python_builder.assert_called_once()
        python_integrator.assert_called_once()
        native_builder.assert_called_once()
        native_integrator.assert_called_once()
        native_keywords = native_integrator.call_args.kwargs
        self.assertEqual(native_keywords["native_library_path"], LIBRARY)
        self.assertEqual(native_keywords["native_segment_capacity"], 123)
        self.assertEqual(native_keywords["native_crossing_capacity"], 456)

    def test_cli_library_flag_selects_native_without_environment_or_fallback(self) -> None:
        arguments = cli_module.parse_args(
            [
                str(self.output),
                "--kernel-cache",
                str(self.cache),
                "--native-kerr-cpu-library",
                str(LIBRARY),
            ]
        )
        with (
            patch.object(cli_module, "_require_exact_cli_module_origins"),
            patch.object(checkpoint_module, "_require_exact_source_module_origins"),
            patch.object(checkpoint_module, "_assert_runtime_bindings"),
        ):
            plan = cli_module.build_refinement_plan(arguments)
        self.assertEqual(plan.evaluator_mode, NATIVE_CPU_EVALUATOR_MODE)
        self.assertEqual(plan.native_library_path, LIBRARY)
        self.assertEqual(plan.native_segment_capacity, 100_000)
        self.assertEqual(plan.native_crossing_capacity, 100_000)

        relative = cli_module.parse_args(
            [
                str(self.output),
                "--kernel-cache",
                str(self.cache),
                "--native-kerr-cpu-library",
                "relative.dylib",
            ]
        )
        with (
            patch.object(cli_module, "_require_exact_cli_module_origins"),
            self.assertRaisesRegex(ValueError, "exact absolute"),
        ):
            cli_module.build_refinement_plan(relative)

        capacity_only = cli_module.parse_args(
            [
                str(self.output),
                "--kernel-cache",
                str(self.cache),
                "--native-kerr-cpu-segment-capacity",
                "123",
            ]
        )
        with (
            patch.object(cli_module, "_require_exact_cli_module_origins"),
            self.assertRaisesRegex(ValueError, "require --native-kerr-cpu-library"),
        ):
            cli_module.build_refinement_plan(capacity_only)

    def test_native_definition_runtime_snapshot_and_manifest_field_are_path_free(
        self,
    ) -> None:
        native = self.construct_plan(
            evaluator_mode=NATIVE_CPU_EVALUATOR_MODE,
            native_library_path=LIBRARY,
            native_segment_capacity=100_000,
            native_crossing_capacity=100_000,
        )
        definition = checkpoint_module._cache_definition_for_plan(native)
        self.assertEqual(
            checkpoint_module._evaluator_mode_from_definition(definition),
            NATIVE_CPU_EVALUATOR_MODE,
        )
        self.assertNotEqual(definition.job_spec.job_key, OLD_CACHE_KEY)
        runtime = checkpoint_module._runtime_snapshot(definition)
        self.assertEqual(
            runtime["descriptor"]["implementationId"],
            "cpython-control-plane+strict-kerr-cpu-whole-ray/abi-v3",
        )
        for document in (
            definition.scientific_context.scientific_document(),
            definition.job_spec.as_dict(),
            runtime,
        ):
            self.assertNotIn(
                str(LIBRARY).encode("utf-8"),
                canonical_json_bytes(document),
            )

    def test_manifest_schema_keeps_python_v1_and_uses_native_v2(self) -> None:
        python_plan = self.construct_plan()
        native_plan = self.construct_plan(
            evaluator_mode=NATIVE_CPU_EVALUATOR_MODE,
            native_library_path=LIBRARY,
            native_segment_capacity=100_000,
            native_crossing_capacity=100_000,
        )
        for selected, expected_schema, expected_implementation, has_mode in (
            (
                python_plan,
                checkpoint_module.MANIFEST_SCHEMA,
                checkpoint_module.IMPLEMENTATION_ID,
                False,
            ),
            (
                native_plan,
                checkpoint_module.NATIVE_MANIFEST_SCHEMA,
                checkpoint_module.NATIVE_IMPLEMENTATION_ID,
                True,
            ),
        ):
            definition = checkpoint_module._cache_definition_for_plan(selected)
            closure = cached_module._source_closure_for_definition(definition)
            reduction = cached_module.KerrKernelReductionConfiguration(
                selected.kernel_policy,
                selected.area_policy,
            )
            evaluator_id = cached_module._evaluator_id_from_definition(definition)
            audit = cached_module.KerrCachedKernelExecutionAudit(
                "forward",
                evaluator_id,
                definition.scientific_job_key,
                definition.job_spec.job_key,
                definition.plan.direction_count,
                0,
                definition.plan.direction_count,
                0,
                definition.plan.task_count,
                cached_module._source_closure_manifest_sha256(closure),
                True,
                True,
                False,
                False,
            )
            kernel_descriptor = {"testKernel": evaluator_id}
            kernel_sha = hashlib.sha256(
                canonical_json_bytes(kernel_descriptor)[:-1]
            ).hexdigest()
            execution = SimpleNamespace(
                cache_definition=definition,
                execution_audit=audit,
                source_closure=closure,
                kernel=SimpleNamespace(
                    model_descriptor=lambda: kernel_descriptor,
                    model_descriptor_sha256=kernel_sha,
                ),
                job_run=SimpleNamespace(
                    max_in_flight_observed=0,
                    results=tuple(
                        SimpleNamespace(reused=True, key=key)
                        for key in definition.job_spec.tasks
                    ),
                ),
                reduction_configuration=reduction,
                reduction_configuration_sha256=reduction.descriptor_sha256,
            )
            binding_json, binding_sha = (
                cached_module._production_forward_scientific_binding(execution)
            )
            authentication = {
                "descriptor": {
                    "provenance": {
                        "scientificBindingSha256": binding_sha,
                    }
                },
                "qualified": False,
            }
            with patch.object(
                checkpoint_module,
                "_authenticated_documents",
                return_value=authentication,
            ):
                document, qualified = checkpoint_module._checkpoint_document(
                    execution,
                    object(),
                    {"mode": selected.evaluator_mode},
                    checkpoint_module._runtime_snapshot(definition),
                    binding_json,
                    binding_sha,
                )
            self.assertFalse(qualified)
            self.assertEqual(document["schema"], expected_schema)
            self.assertEqual(document["implementationId"], expected_implementation)
            self.assertEqual("evaluatorMode" in document, has_mode)
            if has_mode:
                self.assertEqual(
                    document["evaluatorMode"],
                    NATIVE_CPU_EVALUATOR_MODE,
                )
                self.assertNotIn(
                    str(LIBRARY).encode("utf-8"),
                    canonical_json_bytes(document),
                )

    def test_manifest_runtime_helper_requires_matching_native_library(self) -> None:
        native = self.construct_plan(
            evaluator_mode=NATIVE_CPU_EVALUATOR_MODE,
            native_library_path=LIBRARY,
            native_segment_capacity=100_000,
            native_crossing_capacity=100_000,
        )
        definition = checkpoint_module._cache_definition_for_plan(native)
        binding = definition.scientific_context.evaluator_runtime_binding
        self.assertIsNotNone(binding)
        document = {
            "evaluatorMode": NATIVE_CPU_EVALUATOR_MODE,
            "producer": {
                "scientificBinding": {
                    "descriptor": {
                        "evaluator": {
                            "runtimeBinding": {
                                "artifact": binding.descriptor_input.as_dict(),
                                "backendDescriptorSha256": (
                                    binding.backend_descriptor_sha256
                                ),
                                "pathFreeDescriptor": (
                                    binding.path_free_descriptor()
                                ),
                            }
                        }
                    }
                }
            },
        }
        current = checkpoint_module._current_runtime_snapshot_for_manifest(
            document,
            LIBRARY,
        )
        self.assertEqual(current, checkpoint_module._runtime_snapshot(definition))
        with self.assertRaisesRegex(TypeError, "requires an exact absolute"):
            checkpoint_module._current_runtime_snapshot_for_manifest(document, None)

        python_document = {"evaluatorMode": PYTHON_EVALUATOR_MODE}
        with self.assertRaisesRegex(ValueError, "cannot carry a native"):
            checkpoint_module._current_runtime_snapshot_for_manifest(
                python_document,
                LIBRARY,
            )

    def test_native_snapshot_drift_fails_the_prepublication_gate(self) -> None:
        native = self.construct_plan(
            evaluator_mode=NATIVE_CPU_EVALUATOR_MODE,
            native_library_path=LIBRARY,
            native_segment_capacity=100_000,
            native_crossing_capacity=100_000,
        )
        definition = checkpoint_module._cache_definition_for_plan(native)
        source = {"stable": "source"}
        runtime = checkpoint_module._runtime_snapshot(definition)
        with (
            patch.object(
                checkpoint_module,
                "_source_snapshot",
                return_value=source,
            ),
            patch.object(
                checkpoint_module,
                "_runtime_snapshot",
                return_value={"changed": True},
            ),
            self.assertRaisesRegex(
                checkpoint_module.KerrReturningRadiationRefinementCheckpointError,
                "numeric backend changed before publication",
            ),
        ):
            checkpoint_module._assert_snapshots_stable(
                source,
                runtime,
                "before publication",
                evaluator_mode=NATIVE_CPU_EVALUATOR_MODE,
                definition=definition,
            )


if __name__ == "__main__":
    unittest.main()
