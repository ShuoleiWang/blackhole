from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from offline.job import canonical_json_bytes
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_returning_radiation_kernel import (
    KerrReturningRadiationKernelPolicy,
)
import offline.kerr_native_cpu_backend as backend_module
import offline.kerr_returning_radiation_kernel_cached as cached_module
import offline.kerr_returning_radiation_kernel_jobs as jobs_module
from tests.test_offline_kerr_native_runtime_binding import (
    scientific_definition_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = (ROOT / "native/cpu/build/libblackhole_cpu.dylib").absolute()
GOLDEN = ROOT / "tools/native/nested16_golden_corpus.json"
OLD_SCIENTIFIC_KEY = (
    "19c8c3ebd465a4f66d6e893935dc18134ecd47d8842b2e77809823d4148ec980"
)
OLD_CACHE_KEY = (
    "d8286face863a0755d70c20c1638c6a5dc4d9d9037c0227626fb805bb4d0f751"
)


def kernel_policy() -> KerrReturningRadiationKernelPolicy:
    return KerrReturningRadiationKernelPolicy(
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


def common_arguments() -> dict[str, object]:
    _plan, identity = scientific_definition_inputs()
    return {
        "surface": identity.surface,
        "termination": identity.termination,
        "annulus_edges_over_mass": identity.annulus_edges_over_mass,
        "ray_options": identity.fine_ray_options,
        "surface_options": identity.fine_surface_options,
        "coarse_ray_options": identity.coarse_ray_options,
        "coarse_surface_options": identity.coarse_surface_options,
        "policy": kernel_policy(),
        "area_policy": KerrFiniteThicknessAreaQuadraturePolicy(),
    }


def synthetic_native_evaluator(context, coordinate) -> dict[str, object]:
    node = cached_module._forward_coordinate_node(context, coordinate)
    return cached_module._forward_transport_document(
        node,
        cached_module._synthetic_forward_transport(node),
        context.identity.annulus_edges_over_mass,
    )


def synthetic_task_payload(definition, key) -> bytes:
    records = [
        {
            "coordinate": coordinate.as_dict(),
            "transport": synthetic_native_evaluator(
                definition.scientific_context,
                coordinate,
            ),
        }
        for coordinate in definition.plan.coordinates_for_task(key)
    ]
    return canonical_json_bytes(
        {
            "cacheJobKey": definition.job_spec.job_key,
            "records": records,
            "schema": jobs_module.TASK_PAYLOAD_SCHEMA,
            "scientificJobKey": definition.scientific_job_key,
            "scientificPlanSha256": (
                definition.scientific_context.scientific_plan_sha256
            ),
            "task": key.as_dict(),
        }
    )


@unittest.skipUnless(LIBRARY.is_file(), "native CPU ABI-v3 library is not built")
class NativeCachedBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        backend_module._reset_process_local_backend_cache_for_tests()
        self.temporary.cleanup()

    def copy_library(self, directory: str, name: str) -> Path:
        parent = self.root / directory
        parent.mkdir()
        target = parent / name
        shutil.copyfile(LIBRARY, target)
        return target.absolute()

    def native_definition(
        self,
        library: Path = LIBRARY,
        *,
        directions_per_task: int = 64,
    ):
        arguments = common_arguments()
        surface = arguments.pop("surface")
        return cached_module.build_native_forward_kerr_returning_radiation_kernel_cache_definition(
            surface,
            native_library_path=library,
            directions_per_task=directions_per_task,
            **arguments,
        )

    def python_definition(self):
        arguments = common_arguments()
        surface = arguments.pop("surface")
        return cached_module.build_forward_kerr_returning_radiation_kernel_cache_definition(
            surface,
            directions_per_task=64,
            **arguments,
        )

    def test_native_identity_is_fresh_path_free_and_evaluator_specific(self) -> None:
        first_path = self.copy_library("first", "first-name.dylib")
        second_path = self.copy_library("second", "second-name.dylib")
        first = self.native_definition(first_path)
        second = self.native_definition(second_path)
        python = self.python_definition()

        self.assertEqual(first.scientific_job_key, second.scientific_job_key)
        self.assertEqual(first.job_spec.job_key, second.job_spec.job_key)
        self.assertNotEqual(first.scientific_job_key, python.scientific_job_key)
        self.assertNotEqual(first.job_spec.job_key, python.job_spec.job_key)
        self.assertNotEqual(first.scientific_job_key, OLD_SCIENTIFIC_KEY)
        self.assertNotEqual(first.job_spec.job_key, OLD_CACHE_KEY)
        self.assertEqual(
            cached_module._evaluator_id_from_definition(first),
            cached_module._NATIVE_FORWARD_EVALUATOR_ID,
        )
        self.assertEqual(len(first.scientific_context.inputs), 2)
        self.assertIsNotNone(first.scientific_context.evaluator_runtime_binding)

        for document in (
            first.scientific_context.scientific_document(),
            first.job_spec.as_dict(),
        ):
            payload = canonical_json_bytes(document)
            self.assertNotIn(str(first_path).encode("utf-8"), payload)
            self.assertNotIn(str(second_path).encode("utf-8"), payload)

        closure = cached_module._source_closure_for_definition(first)
        paths = tuple(item.logical_path for item in closure)
        for required in (
            "native/cpu/Makefile",
            "native/cpu/include/blackhole_cpu.h",
            "native/cpu/src/blackhole_cpu.c",
            "native/cpu/tools/audit_binary.py",
            "offline/kerr_native_cpu_backend.py",
            "offline/kerr_returning_radiation_native_cpu.py",
            "runtime/numeric-backend.json",
        ):
            self.assertIn(required, paths)
        self.assertNotIn(str(first_path), paths)

    def test_native_production_binding_is_distinct_and_path_free(self) -> None:
        definition = self.native_definition()
        closure = cached_module._source_closure_for_definition(definition)
        reduction = cached_module.KerrKernelReductionConfiguration(
            kernel_policy(),
            KerrFiniteThicknessAreaQuadraturePolicy(),
        )
        execution = SimpleNamespace(
            cache_definition=definition,
            execution_audit=SimpleNamespace(
                source_closure_rechecked_before_and_after_reduction=True,
                is_task_reuse_history_cryptographically_authenticated=False,
            ),
            source_closure=closure,
            kernel=SimpleNamespace(model_descriptor_sha256="a" * 64),
            reduction_configuration=reduction,
        )
        descriptor_json, descriptor_sha = (
            cached_module._production_forward_scientific_binding(execution)
        )
        document = json.loads(descriptor_json)
        self.assertEqual(
            document["implementationId"],
            cached_module.CACHED_NATIVE_FORWARD_SCIENTIFIC_BINDING_IMPLEMENTATION_ID,
        )
        self.assertEqual(
            document["evaluator"]["implementationId"],
            cached_module._NATIVE_FORWARD_EVALUATOR_ID,
        )
        self.assertIn("runtimeBinding", document["evaluator"])
        self.assertEqual(
            descriptor_sha,
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        )
        self.assertNotIn(
            str(LIBRARY).encode("utf-8"),
            canonical_json_bytes(document),
        )

    def test_path_and_runtime_drift_fail_before_any_cache_session(self) -> None:
        arguments = common_arguments()
        surface = arguments.pop("surface")
        never_cache = self.root / "never-cache"
        with (
            patch.object(
                jobs_module,
                "_secure_cache_session",
                side_effect=AssertionError("cache session opened"),
            ) as session,
            self.assertRaisesRegex(TypeError, "exact absolute"),
        ):
            cached_module.integrate_cached_kerr_returning_radiation_energy_kernel_native_cpu(
                surface,
                native_library_path=Path("relative/libblackhole_cpu.dylib"),
                cache_root=never_cache,
                **arguments,
            )
        session.assert_not_called()

        copied = self.copy_library("drift", "runtime.dylib")
        definition = self.native_definition(copied)
        copied.write_bytes(b"changed-after-definition")
        with (
            patch.object(
                jobs_module,
                "_secure_cache_session",
                side_effect=AssertionError("cache session opened"),
            ) as session,
            self.assertRaisesRegex(ValueError, "runtime library"),
        ):
            jobs_module.run_kernel_direction_cache(
                definition,
                cached_module._evaluate_forward_direction_native_cpu,
                never_cache,
            )
        session.assert_not_called()

    def test_existing_cache_entry_targets_only_the_fresh_native_key(self) -> None:
        definition = self.native_definition()
        observed: list[tuple[Path, str]] = []

        @contextmanager
        def reject_new_cache(cache_root: Path, job_key: str):
            observed.append((cache_root, job_key))
            raise FileNotFoundError("fresh native cache is absent")
            yield

        arguments = common_arguments()
        surface = arguments.pop("surface")
        with (
            patch.object(
                jobs_module,
                "_secure_existing_cache_session",
                side_effect=reject_new_cache,
            ),
            self.assertRaisesRegex(FileNotFoundError, "fresh native cache"),
        ):
            cached_module.integrate_existing_cached_kerr_returning_radiation_energy_kernel_native_cpu(
                surface,
                native_library_path=LIBRARY,
                cache_root=self.root / "only-old-d8286",
                **arguments,
            )
        self.assertEqual(
            observed,
            [(self.root / "only-old-d8286", definition.job_spec.job_key)],
        )
        self.assertNotEqual(observed[0][1], OLD_CACHE_KEY)

    def test_ordinal_736_direct_producer_is_exact_and_reuses_one_backend(self) -> None:
        definition = self.native_definition(directions_per_task=8)
        key = next(
            key
            for key in definition.job_spec.tasks
            if any(
                coordinate.ordinal == 736
                for coordinate in definition.plan.coordinates_for_task(key)
            )
        )
        producer = jobs_module._KernelDirectionTaskProducer(
            definition.plan,
            definition.scientific_context,
            definition.job_spec.job_key,
            cached_module._evaluate_forward_direction_native_cpu,
        )
        backend_module._reset_process_local_backend_cache_for_tests()
        original = backend_module.StrictKerrCpuBackend
        with patch.object(
            backend_module,
            "StrictKerrCpuBackend",
            wraps=original,
        ) as constructor:
            payload = producer(key)
        self.assertEqual(constructor.call_count, 1)

        document = json.loads(payload)
        actual = next(
            item["transport"]
            for item in document["records"]
            if item["coordinate"]["ordinal"] == 736
        )
        golden_document = json.loads(GOLDEN.read_bytes())
        expected = next(
            item["transport"]
            for item in golden_document["records"]
            if item["coordinate"]["ordinal"] == 736
        )
        self.assertEqual(
            canonical_json_bytes(actual),
            canonical_json_bytes(expected),
        )

    def test_jobs1_one_missing_task_and_existing_only_roundtrip(self) -> None:
        arguments = common_arguments()
        surface = arguments.pop("surface")
        small_policy = KerrReturningRadiationKernelPolicy(
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
        arguments["policy"] = small_policy
        definition = cached_module.build_native_forward_kerr_returning_radiation_kernel_cache_definition(
            surface,
            native_library_path=LIBRARY,
            directions_per_task=8,
            **arguments,
        )
        cache_root = (self.root / "fresh-one-task-cache").absolute()
        missing = definition.job_spec.tasks[0]
        with jobs_module._secure_cache_session(
            cache_root,
            definition.job_spec.job_key,
        ) as session:
            jobs_module._write_job_document(session, definition.job_spec)
            for key in definition.job_spec.tasks[1:]:
                jobs_module._publish_task_payload(
                    session,
                    definition,
                    key,
                    synthetic_task_payload(definition, key),
                )

        backend_module._reset_process_local_backend_cache_for_tests()
        original = backend_module.StrictKerrCpuBackend
        with patch.object(
            backend_module,
            "StrictKerrCpuBackend",
            wraps=original,
        ) as constructor:
            run = jobs_module.run_kernel_direction_cache(
                definition,
                cached_module._evaluate_forward_direction_native_cpu,
                cache_root,
                jobs=1,
            )
        self.assertEqual(run.executed_tasks, 1)
        self.assertEqual(run.reused_tasks, len(definition.job_spec.tasks) - 1)
        self.assertEqual(constructor.call_count, 1)

        with jobs_module._secure_cache_session(
            cache_root,
            definition.job_spec.job_key,
        ) as session:
            jobs_module._publish_task_payload(
                session,
                definition,
                missing,
                synthetic_task_payload(definition, missing),
            )

        existing = cached_module.integrate_existing_cached_kerr_returning_radiation_energy_kernel_native_cpu(
            surface,
            native_library_path=LIBRARY,
            cache_root=cache_root,
            directions_per_task=8,
            **arguments,
        )
        self.assertEqual(existing.job_run.executed_tasks, 0)
        self.assertEqual(
            existing.job_run.reused_tasks,
            len(definition.job_spec.tasks),
        )
        validated = cached_module.validate_and_reduce_cached_kerr_returning_radiation_energy_kernel(
            existing
        )
        binding = json.loads(validated.scientific_binding_json)
        self.assertEqual(
            binding["implementationId"],
            cached_module.CACHED_NATIVE_FORWARD_SCIENTIFIC_BINDING_IMPLEMENTATION_ID,
        )

    def test_spawn_workers_consume_runtime_binding_without_live_backend_pickle(
        self,
    ) -> None:
        arguments = common_arguments()
        surface = arguments.pop("surface")
        arguments["policy"] = KerrReturningRadiationKernelPolicy(
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
        definition = cached_module.build_native_forward_kerr_returning_radiation_kernel_cache_definition(
            surface,
            native_library_path=LIBRARY,
            directions_per_task=8,
            **arguments,
        )
        missing = tuple(definition.job_spec.tasks[:2])
        cache_root = (self.root / "spawn-native-cache").absolute()
        with jobs_module._secure_cache_session(
            cache_root,
            definition.job_spec.job_key,
        ) as session:
            jobs_module._write_job_document(session, definition.job_spec)
            for key in definition.job_spec.tasks[2:]:
                jobs_module._publish_task_payload(
                    session,
                    definition,
                    key,
                    synthetic_task_payload(definition, key),
                )
        backend_module._reset_process_local_backend_cache_for_tests()
        try:
            run = jobs_module.run_kernel_direction_cache(
                definition,
                cached_module._evaluate_forward_direction_native_cpu,
                cache_root,
                jobs=2,
                max_in_flight=2,
            )
        except PermissionError as error:
            if error.errno != errno.EPERM:
                raise
            self.skipTest("sandbox forbids multiprocessing semaphores")
        self.assertEqual(run.executed_tasks, len(missing))
        self.assertEqual(
            run.reused_tasks,
            len(definition.job_spec.tasks) - len(missing),
        )
        by_key = {item.key: item for item in run.results}
        for key in missing:
            self.assertFalse(by_key[key].reused)


if __name__ == "__main__":
    unittest.main()
