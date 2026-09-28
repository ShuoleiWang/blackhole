from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import FunctionType
from unittest.mock import patch

from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.job import InputArtifact, JobSpec, canonical_json_bytes
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_finite_thickness import (
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
import offline.kerr_returning_radiation_kernel as forward_module
from offline.kerr_returning_radiation_kernel import (
    KerrReturningRadiationKernelPolicy,
)
import offline.kerr_returning_radiation_kernel_cached as cached_module
import offline.kerr_returning_radiation_kernel_jobs as jobs_module
from offline.kerr_returning_radiation_kernel_jobs import (
    FORWARD,
    RECEIVER,
    SCIENTIFIC_STATUS,
    KerrKernelDirectionCacheDefinition,
    KerrKernelDirectionTaskPlan,
    KerrKernelScientificContext,
    KerrKernelScientificIdentity,
    KerrReturningRadiationKernelJobError,
    build_forward_kernel_scientific_identity,
    build_receiver_kernel_scientific_identity,
    iter_cached_kernel_direction_records,
    make_kernel_direction_evaluator_input,
    make_kernel_direction_cache_definition,
    require_complete_existing_kernel_direction_cache,
    run_kernel_direction_cache,
)


SOURCE_A = "a" * 64
SOURCE_B = "b" * 64
SCHEMA_VALID_WRONG_CALLS = 0
STREAMING_SOURCE_EVALUATOR_CALLS = 0


def thread_executor(max_workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=max_workers)


def deterministic_transport(context: KerrKernelScientificContext, coordinate) -> dict:
    return {
        "formulation": context.identity.formulation,
        "g2": coordinate.ordinal / 1024.0,
        "outcome": "synthetic-test-only",
        "primitiveDescriptorSha256": hashlib.sha256(
            str(coordinate.ordinal).encode("ascii")
        ).hexdigest(),
    }


def deterministic_production_forward_trace(*arguments):
    source_radius = arguments[7]
    tangent_azimuth = arguments[9]
    receiver_face = "upper" if tangent_azimuth < 3.141592653589793 else "lower"
    ratio = 2.0
    return forward_module._DirectionTransport(
        f"return-{receiver_face}",
        receiver_face,
        source_radius,
        ratio,
        ratio * ratio,
        hashlib.sha256(repr(arguments[6:]).encode("utf-8")).hexdigest(),
        receiver_face,
        source_radius,
    )


def schema_valid_wrong_transport(
    context: KerrKernelScientificContext,
    coordinate,
) -> dict:
    global SCHEMA_VALID_WRONG_CALLS
    SCHEMA_VALID_WRONG_CALLS += 1
    return {
        "formulation": context.identity.formulation,
        "g2": coordinate.ordinal / 1024.0,
        "outcome": "synthetic-test-only",
        "primitiveDescriptorSha256": hashlib.sha256(
            str(coordinate.ordinal).encode("ascii")
        ).hexdigest(),
    }


def metadata_spoof_transport(
    context: KerrKernelScientificContext,
    coordinate,
) -> dict:
    return schema_valid_wrong_transport(context, coordinate)


def streaming_source_transport(
    context: KerrKernelScientificContext,
    coordinate,
) -> dict:
    global STREAMING_SOURCE_EVALUATOR_CALLS
    STREAMING_SOURCE_EVALUATOR_CALLS += 1
    return deterministic_transport(context, coordinate)


class CallableTransport:
    def __call__(self, context, coordinate):
        return schema_valid_wrong_transport(context, coordinate)


class IntSubclass(int):
    pass


class KerrReturningRadiationKernelJobTests(unittest.TestCase):
    def setUp(self) -> None:
        # macOS exposes /tmp and /var as symlinks.  The secure writer rejects
        # every symlink in the absolute cache-root chain, so anchor tests in
        # the real /private hierarchy.
        self.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def plan(
        self,
        *,
        formulation: str = FORWARD,
        directions_per_task: int = 16,
    ) -> KerrKernelDirectionTaskPlan:
        return KerrKernelDirectionTaskPlan(
            formulation=formulation,
            annulus_count=1,
            rho_order=4,
            mu_order=4,
            psi_count=4,
            directions_per_task=directions_per_task,
        )

    def identity(
        self,
        formulation: str = FORWARD,
        *,
        spin: float = 0.7,
        rho_order: int = 4,
        mu_order: int = 4,
        psi_count: int = 4,
        maximum_direction_evaluations: int = 10_000,
        maximum_whole_ray_traces: int = 40_000,
        source_closure: tuple[str, ...] = (SOURCE_A,),
    ) -> KerrKernelScientificIdentity:
        metric = KerrKerrSchildMetric(spin_a_m=spin)
        calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=spin,
            eddington_scaled_mass_accretion_rate=0.08,
            outer_radius_over_mass=8.0,
        )
        surface = KerrFiniteThicknessMultiSurface(metric, calibration)
        termination = KerrOblateTermination.horizon_worldtube(
            metric,
            escape_radius_m=40.0,
        )
        kwargs = {
            "surface": surface,
            "termination": termination,
            "annulus_edges_over_mass": (
                float(calibration.isco_radius_over_mass),
                8.0,
            ),
            "fine_ray_options": RayTraceOptions(),
            "fine_surface_options": SurfaceEventOptions(),
            "coarse_ray_options": None,
            "coarse_surface_options": None,
            "source_closure_sha256": source_closure,
        }
        builder = (
            build_forward_kernel_scientific_identity
            if formulation == FORWARD
            else build_receiver_kernel_scientific_identity
        )
        return builder(**kwargs)

    def definition(self, plan: KerrKernelDirectionTaskPlan):
        return make_kernel_direction_cache_definition(
            plan,
            identity=self.identity(plan.formulation),
        )

    def bound_forward_definition(self, evaluator_id: str):
        identity = self.identity()
        policy = KerrReturningRadiationKernelPolicy(
            rho_order=4,
            mu_order=4,
            psi_count=4,
            absolute_tolerance=0.25,
            relative_tolerance=0.25,
            symmetry_absolute_tolerance=0.25,
            symmetry_relative_tolerance=0.25,
            maximum_direction_evaluations=10_000,
            maximum_whole_ray_traces=40_000,
        )
        area_policy = KerrFiniteThicknessAreaQuadraturePolicy(
            gauss_legendre_order=24,
            relative_tolerance=1.0e-8,
            absolute_tolerance_over_mass_squared=1.0e-8,
            maximum_point_evaluations=384,
        )
        definition, _closure, _reduction = cached_module._build_cache_definition(
            FORWARD,
            surface=identity.surface,
            termination=identity.termination,
            annulus_edges_over_mass=identity.annulus_edges_over_mass,
            ray_options=identity.fine_ray_options,
            surface_options=identity.fine_surface_options,
            coarse_ray_options=None,
            coarse_surface_options=None,
            policy=policy,
            area_policy=area_policy,
            directions_per_task=16,
            evaluator_id=evaluator_id,
        )
        return definition

    def reserved_source_fixture(
        self,
        name: str,
        payload: bytes,
    ):
        fixture_root = self.root / name
        fake_jobs_path = fixture_root / "offline" / "jobs.py"
        fake_jobs_path.parent.mkdir(parents=True)
        fake_jobs_path.write_bytes(b"# jobs fixture\n")
        source_path = fixture_root / "tests" / "source.py"
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(payload)
        evaluator = FunctionType(
            streaming_source_transport.__code__.replace(
                co_filename=str(source_path)
            ),
            globals(),
            streaming_source_transport.__name__,
        )
        implementation_id = f"tests.streaming-source-{name}/v1"
        with patch.object(jobs_module, "__file__", str(fake_jobs_path)):
            artifact = make_kernel_direction_evaluator_input(
                evaluator,
                implementation_id=implementation_id,
            )
        definition = make_kernel_direction_cache_definition(
            self.plan(),
            identity=self.identity(),
            inputs=(artifact,),
        )
        return fake_jobs_path, source_path, evaluator, definition

    def test_plan_exactly_matches_current_five_grid_work_accounting(self) -> None:
        plan = self.plan(directions_per_task=3)
        self.assertEqual(
            [item.name for item in plan.passes],
            ["full", "half-rho", "half-mu", "half-psi", "phase-shifted"],
        )
        self.assertEqual(
            [
                (item.rho_order, item.mu_order, item.psi_count, item.phase_cells)
                for item in plan.passes
            ],
            [
                (4, 4, 4, 0.0),
                (2, 4, 4, 0.0),
                (4, 2, 4, 0.0),
                (4, 4, 2, 0.0),
                (4, 4, 4, 0.5),
            ],
        )
        self.assertEqual(plan.direction_count, 7 * 1 * 4 * 4 * 4)
        self.assertEqual(
            plan.direction_count,
            plan.expected_current_kernel_direction_count,
        )

        coordinates = tuple(
            coordinate
            for task in plan.task_keys()
            for coordinate in plan.coordinates_for_task(task)
        )
        self.assertEqual(
            [coordinate.ordinal for coordinate in coordinates],
            list(range(plan.direction_count)),
        )
        for task in plan.task_keys():
            grid_pass = plan.passes[task.sample_index]
            start_rho = task.x // grid_pass.angular_direction_count
            end_rho = (task.x + task.width - 1) // grid_pass.angular_direction_count
            self.assertEqual(start_rho, end_rho)

    def test_chunk_layout_changes_cache_key_but_not_scientific_identity(self) -> None:
        fine = self.definition(self.plan(directions_per_task=1))
        chunky = self.definition(self.plan(directions_per_task=16))
        self.assertEqual(
            fine.plan.scientific_plan_sha256,
            chunky.plan.scientific_plan_sha256,
        )
        self.assertEqual(
            fine.scientific_job_key,
            chunky.scientific_job_key,
        )
        self.assertNotEqual(fine.job_spec.job_key, chunky.job_spec.job_key)
        self.assertGreater(len(fine.job_spec.tasks), len(chunky.job_spec.tasks))

        receiver = self.definition(
            self.plan(formulation=RECEIVER, directions_per_task=16)
        )
        self.assertNotEqual(
            chunky.scientific_job_key,
            receiver.scientific_job_key,
        )

    def test_worker_count_does_not_change_payloads_or_reduction_order(self) -> None:
        definition = self.definition(self.plan())
        serial = run_kernel_direction_cache(
            definition,
            deterministic_transport,
            self.root / "serial",
            jobs=1,
        )
        parallel = run_kernel_direction_cache(
            definition,
            deterministic_transport,
            self.root / "parallel",
            jobs=4,
            max_in_flight=3,
            executor_factory=thread_executor,
        )
        self.assertEqual(
            [item.sha256 for item in serial.results],
            [item.sha256 for item in parallel.results],
        )
        serial_records = tuple(iter_cached_kernel_direction_records(definition, serial))
        parallel_records = tuple(
            iter_cached_kernel_direction_records(definition, parallel)
        )
        self.assertEqual(
            [item.coordinate.ordinal for item in serial_records],
            list(range(definition.plan.direction_count)),
        )
        self.assertEqual(
            [item.transport_sha256 for item in serial_records],
            [item.transport_sha256 for item in parallel_records],
        )
        self.assertTrue(
            all(
                item.transport()["outcome"] == "synthetic-test-only"
                for item in serial_records
            )
        )

        different_layout = self.definition(self.plan(directions_per_task=4))
        layout_run = run_kernel_direction_cache(
            different_layout,
            deterministic_transport,
            self.root / "different-layout",
            jobs=2,
            executor_factory=thread_executor,
        )
        layout_records = tuple(
            iter_cached_kernel_direction_records(different_layout, layout_run)
        )
        self.assertEqual(
            [item.transport_sha256 for item in serial_records],
            [item.transport_sha256 for item in layout_records],
        )

    def test_scheduler_controls_are_hard_bounded_before_cache_or_evaluator(
        self,
    ) -> None:
        self.assertEqual(jobs_module.MAXIMUM_WORKERS, 64)
        self.assertEqual(jobs_module.MAXIMUM_IN_FLIGHT_TASKS, 256)
        self.assertEqual(
            jobs_module._validated_execution_controls(1, None, 1),
            (1, 1),
        )
        definition = self.definition(
            self.plan(directions_per_task=jobs_module._MAXIMUM_DIRECTIONS_PER_TASK)
        )
        calls = 0

        def counted(context, coordinate) -> dict:
            nonlocal calls
            calls += 1
            return deterministic_transport(context, coordinate)

        cases = (
            ("huge-jobs", 10**9, None, "jobs must not exceed"),
            (
                "huge-in-flight",
                1,
                10**9,
                "max_in_flight must not exceed",
            ),
            (
                "jobs-over-tasks",
                definition.plan.task_count + 1,
                None,
                "jobs must not exceed",
            ),
            (
                "in-flight-over-tasks",
                1,
                definition.plan.task_count + 1,
                "max_in_flight must not exceed",
            ),
            ("bool-jobs", True, None, "positive exact int"),
            ("subclass-jobs", IntSubclass(1), None, "positive exact int"),
            ("bool-in-flight", 1, True, "positive exact int"),
            (
                "subclass-in-flight",
                1,
                IntSubclass(1),
                "positive exact int",
            ),
        )
        for name, worker_count, in_flight, message in cases:
            cache_root = self.root / name
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                run_kernel_direction_cache(
                    definition,
                    counted,
                    cache_root,
                    jobs=worker_count,
                    max_in_flight=in_flight,
                )
            self.assertFalse(cache_root.exists())
            self.assertEqual(calls, 0)

        scientific_document = canonical_json_bytes(
            definition.scientific_context.scientific_document()
        )
        self.assertNotIn(b'"jobs"', scientific_document)
        self.assertNotIn(b'"maxInFlight"', scientific_document)
        self.assertNotIn(b'"maximumWorkers"', scientific_document)

    def test_resume_reuses_valid_records_and_repairs_one_chunk(self) -> None:
        definition = self.definition(self.plan())
        first = run_kernel_direction_cache(
            definition,
            deterministic_transport,
            self.root / "cache",
            jobs=1,
        )
        calls = 0
        lock = threading.Lock()

        def counted(context: KerrKernelScientificContext, coordinate) -> dict:
            nonlocal calls
            with lock:
                calls += 1
            return deterministic_transport(context, coordinate)

        resumed = run_kernel_direction_cache(
            definition,
            counted,
            self.root / "cache",
            jobs=3,
            executor_factory=thread_executor,
        )
        self.assertEqual(calls, 0)
        self.assertEqual(resumed.executed_tasks, 0)
        self.assertEqual(resumed.reused_tasks, len(definition.job_spec.tasks))

        damaged = first.results[len(first.results) // 2]
        damaged.payload_path.write_bytes(b"corrupt")
        repaired = run_kernel_direction_cache(
            definition,
            counted,
            self.root / "cache",
            jobs=1,
        )
        self.assertEqual(repaired.executed_tasks, 1)
        self.assertEqual(calls, damaged.key.width)
        self.assertEqual(
            len(tuple(iter_cached_kernel_direction_records(definition, repaired))),
            definition.plan.direction_count,
        )

    def test_complete_existing_cache_returns_exact_full_reuse_without_writes(
        self,
    ) -> None:
        for name, evaluator_id, evaluator in (
            (
                "production-bound",
                cached_module._FORWARD_EVALUATOR_ID,
                cached_module._evaluate_forward_direction,
            ),
            (
                "synthetic-bound",
                cached_module._SYNTHETIC_FORWARD_EVALUATOR_ID,
                cached_module._synthetic_forward_direction_evaluator,
            ),
        ):
            with self.subTest(name=name):
                definition = self.bound_forward_definition(evaluator_id)
                cache_root = self.root / f"complete-{name}"
                if name == "production-bound":
                    with patch.object(
                        forward_module,
                        "_trace_direction",
                        side_effect=deterministic_production_forward_trace,
                    ):
                        created = run_kernel_direction_cache(
                            definition,
                            evaluator,
                            cache_root,
                            jobs=1,
                        )
                else:
                    created = run_kernel_direction_cache(
                        definition,
                        evaluator,
                        cache_root,
                        jobs=1,
                    )
                before = self._cache_tree_snapshot(cache_root)
                with (
                    patch.object(
                        jobs_module,
                        "_KernelDirectionTaskProducer",
                        side_effect=AssertionError(
                            "complete-cache authentication constructed an evaluator"
                        ),
                    ) as producer,
                    patch.object(
                        jobs_module,
                        "_atomic_write_at",
                        side_effect=AssertionError(
                            "complete-cache authentication attempted a write"
                        ),
                    ) as writer,
                ):
                    reused = require_complete_existing_kernel_direction_cache(
                        definition,
                        cache_root,
                    )
                self.assertEqual(producer.call_count, 0)
                self.assertEqual(writer.call_count, 0)
                self.assertIs(type(reused), type(created))
                self.assertEqual(reused.job_key, definition.job_spec.job_key)
                self.assertEqual(
                    tuple(item.key for item in reused.results),
                    tuple(definition.job_spec.tasks),
                )
                self.assertEqual(
                    tuple(item.sha256 for item in reused.results),
                    tuple(item.sha256 for item in created.results),
                )
                self.assertTrue(all(item.reused is True for item in reused.results))
                self.assertEqual(reused.reused_tasks, definition.plan.task_count)
                self.assertEqual(reused.executed_tasks, 0)
                self.assertEqual(reused.max_in_flight_observed, 0)
                self.assertEqual(self._cache_tree_snapshot(cache_root), before)

    @staticmethod
    def _cache_tree_snapshot(
        cache_root: Path,
    ) -> tuple[tuple[str, str, int, str], ...]:
        if not cache_root.exists() and not cache_root.is_symlink():
            return ()
        snapshot: list[tuple[str, str, int, str]] = []
        for path in sorted(cache_root.rglob("*")):
            relative = path.relative_to(cache_root).as_posix()
            stat_result = path.lstat()
            if path.is_symlink():
                snapshot.append(
                    (relative, "symlink", stat_result.st_mode, os.readlink(path))
                )
            elif path.is_file():
                payload = path.read_bytes()
                snapshot.append(
                    (
                        relative,
                        "file",
                        stat_result.st_mode,
                        hashlib.sha256(payload).hexdigest(),
                    )
                )
            else:
                snapshot.append((relative, "directory", stat_result.st_mode, ""))
        return tuple(snapshot)

    def test_complete_existing_cache_fails_closed_for_missing_or_corrupt_tree(
        self,
    ) -> None:
        definition = self.definition(self.plan())
        source_root = self.root / "complete-source"
        source_run = run_kernel_direction_cache(
            definition,
            deterministic_transport,
            source_root,
            jobs=1,
        )
        first = source_run.results[0]
        task_directory = first.payload_path.parent
        job_directory = task_directory.parent

        cases = (
            ("missing-root", lambda root: None),
            ("missing-job", lambda root: root.mkdir()),
            (
                "missing-tasks",
                lambda root: (root / definition.job_spec.job_key).mkdir(parents=True),
            ),
            (
                "missing-job-json",
                lambda root: self._copy_cache_then_remove(
                    source_root,
                    root,
                    job_directory / "job.json",
                ),
            ),
            (
                "missing-payload",
                lambda root: self._copy_cache_then_remove(
                    source_root,
                    root,
                    first.payload_path,
                ),
            ),
            (
                "missing-receipt",
                lambda root: self._copy_cache_then_remove(
                    source_root,
                    root,
                    first.receipt_path,
                ),
            ),
            (
                "corrupt-job-json",
                lambda root: self._copy_cache_then_write(
                    source_root,
                    root,
                    job_directory / "job.json",
                    b"{}\n",
                ),
            ),
            (
                "corrupt-payload",
                lambda root: self._copy_cache_then_write(
                    source_root,
                    root,
                    first.payload_path,
                    b"corrupt",
                ),
            ),
            (
                "corrupt-receipt",
                lambda root: self._copy_cache_then_write(
                    source_root,
                    root,
                    first.receipt_path,
                    b"{}\n",
                ),
            ),
            (
                "symlink-job",
                lambda root: self._copy_cache_then_symlink(
                    source_root,
                    root,
                    job_directory,
                ),
            ),
            (
                "symlink-tasks",
                lambda root: self._copy_cache_then_symlink(
                    source_root,
                    root,
                    task_directory,
                ),
            ),
            (
                "symlink-payload",
                lambda root: self._copy_cache_then_symlink(
                    source_root,
                    root,
                    first.payload_path,
                ),
            ),
            (
                "symlink-receipt",
                lambda root: self._copy_cache_then_symlink(
                    source_root,
                    root,
                    first.receipt_path,
                ),
            ),
            (
                "irregular-job",
                lambda root: self._copy_cache_then_irregular(
                    source_root,
                    root,
                    job_directory,
                ),
            ),
            (
                "irregular-tasks",
                lambda root: self._copy_cache_then_irregular(
                    source_root,
                    root,
                    task_directory,
                ),
            ),
            (
                "irregular-payload",
                lambda root: self._copy_cache_then_irregular(
                    source_root,
                    root,
                    first.payload_path,
                ),
            ),
            (
                "irregular-receipt",
                lambda root: self._copy_cache_then_irregular(
                    source_root,
                    root,
                    first.receipt_path,
                ),
            ),
        )
        for name, prepare in cases:
            cache_root = self.root / f"invalid-complete-{name}"
            prepare(cache_root)
            before = self._cache_tree_snapshot(cache_root)
            with (
                self.subTest(name=name),
                patch.object(
                    jobs_module,
                    "_KernelDirectionTaskProducer",
                    side_effect=AssertionError(
                        "invalid complete cache constructed an evaluator"
                    ),
                ) as producer,
                patch.object(
                    jobs_module,
                    "_atomic_write_at",
                    side_effect=AssertionError(
                        "invalid complete cache attempted a write"
                    ),
                ) as writer,
                self.assertRaises(KerrReturningRadiationKernelJobError),
            ):
                require_complete_existing_kernel_direction_cache(
                    definition,
                    cache_root,
                )
            self.assertEqual(producer.call_count, 0)
            self.assertEqual(writer.call_count, 0)
            self.assertEqual(self._cache_tree_snapshot(cache_root), before)

    @staticmethod
    def _cache_relative_path(
        source_root: Path,
        destination_root: Path,
        path: Path,
    ) -> Path:
        return destination_root / path.relative_to(source_root)

    def _copy_cache_then_remove(
        self,
        source_root: Path,
        destination_root: Path,
        path: Path,
    ) -> None:
        shutil.copytree(source_root, destination_root)
        target = self._cache_relative_path(source_root, destination_root, path)
        if target.is_dir():
            target.rmdir()
        else:
            target.unlink()

    def _copy_cache_then_write(
        self,
        source_root: Path,
        destination_root: Path,
        path: Path,
        payload: bytes,
    ) -> None:
        shutil.copytree(source_root, destination_root)
        target = self._cache_relative_path(source_root, destination_root, path)
        target.write_bytes(payload)

    def _copy_cache_then_symlink(
        self,
        source_root: Path,
        destination_root: Path,
        path: Path,
    ) -> None:
        shutil.copytree(source_root, destination_root)
        target = self._cache_relative_path(source_root, destination_root, path)
        real = target.with_name(target.name + "-real")
        target.rename(real)
        target.symlink_to(real, target_is_directory=real.is_dir())

    def _copy_cache_then_irregular(
        self,
        source_root: Path,
        destination_root: Path,
        path: Path,
    ) -> None:
        shutil.copytree(source_root, destination_root)
        target = self._cache_relative_path(source_root, destination_root, path)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        os.mkfifo(target)

    def test_reserved_evaluator_descriptor_rejects_substitution_before_cache_access(
        self,
    ) -> None:
        global SCHEMA_VALID_WRONG_CALLS
        SCHEMA_VALID_WRONG_CALLS = 0
        implementation_id = "tests.deterministic-transport/v1"
        definition = make_kernel_direction_cache_definition(
            self.plan(),
            identity=self.identity(),
            inputs=(
                make_kernel_direction_evaluator_input(
                    deterministic_transport,
                    implementation_id=implementation_id,
                ),
            ),
        )
        status = definition.scientific_context.scientific_document()[
            "scientificStatus"
        ]
        self.assertEqual(
            canonical_json_bytes(
                definition.job_spec.parameters["scientificStatus"]
            ),
            canonical_json_bytes(status),
        )
        self.assertEqual(
            status["evaluatorIdentityBindingMode"],
            "reserved-exact-function-source-code-descriptor/v2",
        )
        self.assertTrue(status["authenticatedEvaluatorIdentity"])
        self.assertEqual(status["reservedEvaluatorArtifactCount"], 1)
        self.assertEqual(
            status["reservedEvaluatorArtifactUri"],
            (
                "urn:blackhole:returning-radiation-kernel-direction-evaluator:"
                f"v2:{implementation_id}"
            ),
        )
        self.assertFalse(
            status["genericUnboundCacheMayBeReusedAcrossCallerEvaluators"]
        )
        self.assertTrue(
            status[
                "evaluatorImplementationAndSourceIdentityBoundAsScientificInput"
            ]
        )
        # SOURCE_A is an opaque digest unrelated to this test-file evaluator.
        # A reserved input binds the descriptor but cannot prove closure
        # membership at this generic jobs layer.
        self.assertFalse(
            status["evaluatorImplementationAndSourceIdentityInSourceClosure"]
        )
        self.assertTrue(
            status["reservedEvaluatorCallableDescriptorCheckedBeforeCacheAccess"]
        )
        cache_root = self.root / "reserved-evaluator"
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelJobError,
            "callable descriptor differs",
        ):
            run_kernel_direction_cache(
                definition,
                schema_valid_wrong_transport,
                cache_root,
                jobs=1,
            )
        self.assertFalse(cache_root.exists())
        self.assertEqual(SCHEMA_VALID_WRONG_CALLS, 0)

        original_metadata = (
            metadata_spoof_transport.__module__,
            metadata_spoof_transport.__name__,
            metadata_spoof_transport.__qualname__,
        )
        try:
            metadata_spoof_transport.__module__ = deterministic_transport.__module__
            metadata_spoof_transport.__name__ = deterministic_transport.__name__
            metadata_spoof_transport.__qualname__ = (
                deterministic_transport.__qualname__
            )
            with self.assertRaisesRegex(
                KerrReturningRadiationKernelJobError,
                "callable descriptor could not be authenticated",
            ):
                run_kernel_direction_cache(
                    definition,
                    metadata_spoof_transport,
                    cache_root,
                    jobs=1,
                )
        finally:
            (
                metadata_spoof_transport.__module__,
                metadata_spoof_transport.__name__,
                metadata_spoof_transport.__qualname__,
            ) = original_metadata
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelJobError,
            "exact Python function",
        ):
            run_kernel_direction_cache(
                definition,
                CallableTransport(),
                cache_root,
                jobs=1,
            )
        try:
            deterministic_transport.claimed_implementation_id = implementation_id
            with self.assertRaisesRegex(
                KerrReturningRadiationKernelJobError,
                "custom function attributes",
            ):
                run_kernel_direction_cache(
                    definition,
                    deterministic_transport,
                    cache_root,
                    jobs=1,
                )
        finally:
            del deterministic_transport.claimed_implementation_id
        self.assertFalse(cache_root.exists())

        first = run_kernel_direction_cache(
            definition,
            deterministic_transport,
            cache_root,
            jobs=1,
        )
        self.assertEqual(first.executed_tasks, definition.plan.task_count)
        resumed = run_kernel_direction_cache(
            definition,
            deterministic_transport,
            cache_root,
            jobs=1,
        )
        self.assertEqual(resumed.executed_tasks, 0)
        self.assertEqual(resumed.reused_tasks, definition.plan.task_count)

    def test_reserved_evaluator_streams_source_without_path_read_bytes(self) -> None:
        implementation_id = "tests.streaming-evaluator/v1"
        expected_artifact = make_kernel_direction_evaluator_input(
            deterministic_transport,
            implementation_id=implementation_id,
        )
        source_path = Path(deterministic_transport.__code__.co_filename)
        source_payload = source_path.read_bytes()
        expected_source_sha256 = hashlib.sha256(source_payload).hexdigest()
        with patch.object(
            Path,
            "read_bytes",
            side_effect=AssertionError("Path.read_bytes is forbidden in full run"),
        ):
            actual_artifact = make_kernel_direction_evaluator_input(
                deterministic_transport,
                implementation_id=implementation_id,
            )
            descriptor = jobs_module._evaluator_callable_descriptor(
                deterministic_transport,
                implementation_id,
            )
            definition = make_kernel_direction_cache_definition(
                self.plan(),
                identity=self.identity(),
                inputs=(actual_artifact,),
            )
            run = run_kernel_direction_cache(
                definition,
                deterministic_transport,
                self.root / "streaming-full-run",
                jobs=1,
            )
        self.assertEqual(actual_artifact, expected_artifact)
        self.assertEqual(descriptor["source"]["byteLength"], len(source_payload))
        self.assertEqual(descriptor["source"]["sha256"], expected_source_sha256)
        self.assertEqual(run.executed_tasks, definition.plan.task_count)
        self.assertEqual(
            len(tuple(iter_cached_kernel_direction_records(definition, run))),
            definition.plan.direction_count,
        )

    def test_reserved_evaluator_source_limits_fail_before_cache_or_evaluation(
        self,
    ) -> None:
        global STREAMING_SOURCE_EVALUATOR_CALLS

        fake_jobs, source, evaluator, definition = self.reserved_source_fixture(
            "source-oversize",
            b"12345678",
        )
        source.write_bytes(b"123456789")
        cache_root = self.root / "oversize-cache"
        STREAMING_SOURCE_EVALUATOR_CALLS = 0
        with (
            patch.object(jobs_module, "__file__", str(fake_jobs)),
            patch.object(jobs_module, "_MAXIMUM_EVALUATOR_SOURCE_BYTES", 8),
            patch.object(jobs_module.os, "read", wraps=os.read) as source_read,
        ):
            with self.assertRaisesRegex(
                KerrReturningRadiationKernelJobError,
                "could not be authenticated",
            ):
                run_kernel_direction_cache(
                    definition,
                    evaluator,
                    cache_root,
                    jobs=1,
                )
        self.assertEqual(source_read.call_count, 0)
        self.assertEqual(STREAMING_SOURCE_EVALUATOR_CALLS, 0)
        self.assertFalse(cache_root.exists())

        fake_jobs, source, evaluator, definition = self.reserved_source_fixture(
            "source-symlink",
            b"original source",
        )
        target = source.with_name("target.py")
        target.write_bytes(b"original source")
        source.unlink()
        source.symlink_to(target)
        cache_root = self.root / "symlink-cache"
        STREAMING_SOURCE_EVALUATOR_CALLS = 0
        with patch.object(jobs_module, "__file__", str(fake_jobs)):
            with self.assertRaisesRegex(
                KerrReturningRadiationKernelJobError,
                "could not be authenticated",
            ):
                run_kernel_direction_cache(
                    definition,
                    evaluator,
                    cache_root,
                    jobs=1,
                )
        self.assertEqual(STREAMING_SOURCE_EVALUATOR_CALLS, 0)
        self.assertFalse(cache_root.exists())

    def test_reserved_evaluator_source_toctou_and_growth_fail_closed(self) -> None:
        global STREAMING_SOURCE_EVALUATOR_CALLS

        fake_jobs, source, evaluator, definition = self.reserved_source_fixture(
            "source-toctou",
            b"a" * 32,
        )
        original_read = os.read
        mutated = False

        def mutate_after_read(descriptor: int, byte_count: int) -> bytes:
            nonlocal mutated
            block = original_read(descriptor, byte_count)
            if block and not mutated:
                mutated = True
                with source.open("r+b") as handle:
                    handle.write(b"b")
                    handle.flush()
                    os.fsync(handle.fileno())
            return block

        cache_root = self.root / "toctou-cache"
        STREAMING_SOURCE_EVALUATOR_CALLS = 0
        with (
            patch.object(jobs_module, "__file__", str(fake_jobs)),
            patch.object(jobs_module.os, "read", side_effect=mutate_after_read),
        ):
            with self.assertRaisesRegex(
                KerrReturningRadiationKernelJobError,
                "could not be authenticated",
            ):
                run_kernel_direction_cache(
                    definition,
                    evaluator,
                    cache_root,
                    jobs=1,
                )
        self.assertEqual(STREAMING_SOURCE_EVALUATOR_CALLS, 0)
        self.assertFalse(cache_root.exists())

        fake_jobs, source, evaluator, definition = self.reserved_source_fixture(
            "source-growth",
            b"12345678",
        )
        grew = False

        def grow_after_read(descriptor: int, byte_count: int) -> bytes:
            nonlocal grew
            block = original_read(descriptor, byte_count)
            if block and not grew:
                grew = True
                with source.open("ab") as handle:
                    handle.write(b"9")
                    handle.flush()
                    os.fsync(handle.fileno())
            return block

        cache_root = self.root / "growth-cache"
        STREAMING_SOURCE_EVALUATOR_CALLS = 0
        with (
            patch.object(jobs_module, "__file__", str(fake_jobs)),
            patch.object(jobs_module, "_MAXIMUM_EVALUATOR_SOURCE_BYTES", 8),
            patch.object(jobs_module.os, "read", side_effect=grow_after_read),
        ):
            with self.assertRaisesRegex(
                KerrReturningRadiationKernelJobError,
                "could not be authenticated",
            ):
                run_kernel_direction_cache(
                    definition,
                    evaluator,
                    cache_root,
                    jobs=1,
                )
        self.assertEqual(STREAMING_SOURCE_EVALUATOR_CALLS, 0)
        self.assertFalse(cache_root.exists())

    def test_consumer_rehashes_cache_after_publication(self) -> None:
        definition = self.definition(self.plan())
        run = run_kernel_direction_cache(
            definition,
            deterministic_transport,
            self.root / "cache",
            jobs=1,
        )
        run.results[0].payload_path.write_bytes(b"{}\n")
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelJobError,
            "differs from its authenticated receipt",
        ):
            tuple(iter_cached_kernel_direction_records(definition, run))

    def test_default_spawn_executor_round_trip(self) -> None:
        definition = self.definition(self.plan())
        try:
            run = run_kernel_direction_cache(
                definition,
                deterministic_transport,
                self.root / "process-cache",
                jobs=2,
                max_in_flight=2,
            )
        except PermissionError as error:
            if error.errno != errno.EPERM:
                raise
            self.skipTest("sandbox forbids multiprocessing semaphores")
        self.assertEqual(run.executed_tasks, definition.plan.task_count)
        self.assertEqual(
            len(tuple(iter_cached_kernel_direction_records(definition, run))),
            definition.plan.direction_count,
        )

    def test_fail_closed_exact_types_and_identity_changes(self) -> None:
        with self.assertRaisesRegex(ValueError, "rho_order"):
            KerrKernelDirectionTaskPlan(FORWARD, 1, True, 4, 4)
        with self.assertRaisesRegex(ValueError, "directions_per_task"):
            self.plan(directions_per_task=True)
        plan = self.plan()
        with self.assertRaisesRegex(TypeError, "exact KerrKernelScientificIdentity"):
            make_kernel_direction_cache_definition(
                plan,
                identity={},
            )
        with self.assertRaisesRegex(ValueError, "hard direction budget"):
            KerrKernelDirectionTaskPlan(FORWARD, 1, 64, 64, 256)
        changed = make_kernel_direction_cache_definition(
            plan,
            identity=self.identity(spin=0.8),
        )
        self.assertNotEqual(
            self.definition(plan).scientific_job_key,
            changed.scientific_job_key,
        )
        changed_closure = make_kernel_direction_cache_definition(
            plan,
            identity=self.identity(source_closure=(SOURCE_B,)),
        )
        self.assertNotEqual(
            self.definition(plan).scientific_job_key,
            changed_closure.scientific_job_key,
        )

        with self.assertRaisesRegex(TypeError, "source closure"):
            self.identity(source_closure=())
        identity = self.identity()
        with self.assertRaisesRegex(TypeError, "fine_ray_options"):
            replace(
                identity,
                fine_ray_options=RayTraceOptions(absolute_tolerance=1),
            )
        with self.assertRaisesRegex(ValueError, "appear together"):
            replace(
                identity,
                coarse_ray_options=RayTraceOptions(),
                coarse_surface_options=None,
            )
        with self.assertRaisesRegex(TypeError, "source closure"):
            replace(identity, source_closure_sha256=[SOURCE_A])

    def test_strong_identity_has_exact_required_projection(self) -> None:
        identity = self.identity()
        self.assertEqual(
            set(identity.as_dict()),
            {
                "annulusEdgesOverMass",
                "coarseRayOptions",
                "coarseSurfaceOptions",
                "fineRayOptions",
                "fineSurfaceOptions",
                "formulation",
                "metric",
                "sourceClosureSha256",
                "surface",
                "termination",
            },
        )
        definition = self.definition(self.plan())
        context = definition.scientific_context
        self.assertIs(type(context), KerrKernelScientificContext)
        self.assertFalse(hasattr(context, "directions_per_task"))
        self.assertFalse(hasattr(context, "job_spec"))
        self.assertFalse(hasattr(context, "cacheLayout"))
        self.assertEqual(context.identity, identity)
        self.assertEqual(
            canonical_json_bytes(
                definition.job_spec.parameters["scientificStatus"]
            ),
            canonical_json_bytes(dict(SCIENTIFIC_STATUS)),
        )
        self.assertEqual(
            context.scientific_document()["scientificStatus"],
            dict(SCIENTIFIC_STATUS),
        )
        self.assertTrue(SCIENTIFIC_STATUS["allScientificStateComesFromTypedContext"])
        self.assertEqual(
            SCIENTIFIC_STATUS["evaluatorIdentityBindingMode"],
            "caller-trusted-unbound",
        )
        self.assertFalse(SCIENTIFIC_STATUS["authenticatedEvaluatorIdentity"])
        self.assertTrue(
            SCIENTIFIC_STATUS[
                "transportScientificIdentityExcludesAreaQuadrature"
            ]
        )
        self.assertTrue(
            SCIENTIFIC_STATUS[
                "transportScientificIdentityExcludesReductionConvergence"
            ]
        )
        self.assertTrue(
            SCIENTIFIC_STATUS[
                "transportScientificIdentityExcludesReductionSymmetryThresholds"
            ]
        )
        self.assertTrue(
            SCIENTIFIC_STATUS[
                "transportScientificIdentityExcludesReductionWorkBudgets"
            ]
        )
        self.assertTrue(SCIENTIFIC_STATUS["quadratureOrdersBoundByScientificPlan"])
        self.assertFalse(
            SCIENTIFIC_STATUS[
                "reductionConfigurationAuthenticatedByThisSubstrate"
            ]
        )
        self.assertIn(
            "not authenticated",
            SCIENTIFIC_STATUS["evaluatorIdentityEvidence"],
        )
        self.assertEqual(SCIENTIFIC_STATUS["reservedEvaluatorArtifactCount"], 0)
        self.assertIsNone(SCIENTIFIC_STATUS["reservedEvaluatorArtifactUri"])
        self.assertTrue(
            SCIENTIFIC_STATUS[
                "genericUnboundCacheMayBeReusedAcrossCallerEvaluators"
            ]
        )
        self.assertFalse(
            SCIENTIFIC_STATUS[
                "evaluatorImplementationAndSourceIdentityBoundAsScientificInput"
            ]
        )
        self.assertFalse(
            SCIENTIFIC_STATUS[
                "evaluatorImplementationAndSourceIdentityInSourceClosure"
            ]
        )
        self.assertFalse(
            SCIENTIFIC_STATUS[
                "claimsProtectionFromArbitraryHiddenEvaluatorClosure"
            ]
        )
        self.assertFalse(
            SCIENTIFIC_STATUS[
                "reservedEvaluatorCallableDescriptorCheckedBeforeCacheAccess"
            ]
        )
        self.assertFalse(
            SCIENTIFIC_STATUS[
                "claimsProtectionFromMaliciousSameProcessCodeExecution"
            ]
        )
        self.assertFalse(SCIENTIFIC_STATUS["isIntegratedPublicKernelAcceleration"])

        malformed_reserved = InputArtifact(
            "urn:blackhole:returning-radiation-kernel-direction-evaluator:v1:old",
            0,
            "0" * 64,
        )
        malformed_definition = make_kernel_direction_cache_definition(
            self.plan(),
            identity=identity,
            inputs=(malformed_reserved,),
        )
        malformed_status = malformed_definition.scientific_context.scientific_document()[
            "scientificStatus"
        ]
        self.assertEqual(
            malformed_status["evaluatorIdentityBindingMode"],
            "invalid-reserved-artifact-set",
        )
        self.assertFalse(malformed_status["authenticatedEvaluatorIdentity"])
        self.assertFalse(
            malformed_status[
                "evaluatorImplementationAndSourceIdentityBoundAsScientificInput"
            ]
        )
        self.assertFalse(
            malformed_status[
                "evaluatorImplementationAndSourceIdentityInSourceClosure"
            ]
        )
        malformed_root = self.root / "malformed-reserved"
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelJobError,
            "unsupported reserved evaluator artifact",
        ):
            run_kernel_direction_cache(
                malformed_definition,
                deterministic_transport,
                malformed_root,
            )
        self.assertFalse(malformed_root.exists())

        with self.assertRaisesRegex(ValueError, "formulations differ"):
            make_kernel_direction_cache_definition(
                self.plan(formulation=RECEIVER),
                identity=identity,
            )
        with self.assertRaisesRegex(ValueError, "scientific job key"):
            KerrKernelScientificContext(
                context.plan,
                context.identity,
                context.inputs,
                "b" * 64,
            )

    def test_cache_definition_rejects_foreign_job_spec_components(self) -> None:
        definition = self.definition(self.plan())
        original = definition.job_spec
        parameters = json.loads(canonical_json_bytes(original.parameters))
        parameter_variants = []
        for field, value in (
            ("scientificPlanSha256", "b" * 64),
            ("scientificJobKey", "b" * 64),
        ):
            variant = json.loads(canonical_json_bytes(parameters))
            variant[field] = value
            parameter_variants.append((field, variant))
        layout_variant = json.loads(canonical_json_bytes(parameters))
        layout_variant["cacheLayout"]["taskCount"] += 1
        parameter_variants.append(("cacheLayout", layout_variant))
        document_variant = json.loads(canonical_json_bytes(parameters))
        document_variant["scientificDocument"]["plan"]["formulation"] = RECEIVER
        parameter_variants.append(("scientificDocument", document_variant))

        variants = [
            (
                "producer",
                {
                    "producer": "foreign-producer",
                },
            ),
            (
                "algorithmVersion",
                {
                    "algorithm_version": "9.9.9",
                },
            ),
            (
                "tasks",
                {
                    "tasks": original.tasks[:-1],
                },
            ),
            *(
                (name, {"parameters": value})
                for name, value in parameter_variants
            ),
        ]
        for name, override in variants:
            kwargs = {
                "producer": original.producer,
                "algorithm_version": original.algorithm_version,
                "tasks": original.tasks,
                "parameters": original.parameters,
                "inputs": original.inputs,
                "producer_source_hashes": original.producer_source_hashes,
                "record_bytes": original.record_bytes,
            }
            kwargs.update(override)
            foreign = JobSpec(**kwargs)
            with self.subTest(name=name), self.assertRaisesRegex(
                (TypeError, ValueError),
                "job_spec",
            ):
                KerrKernelDirectionCacheDefinition(
                    definition.plan,
                    definition.scientific_context,
                    foreign,
                )

    def test_hard_task_transport_payload_depth_and_total_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "task-count"):
            KerrKernelDirectionTaskPlan(FORWARD, 1, 16, 16, 64, 1)

        definition = self.definition(self.plan())

        def oversized_transport(_context, _coordinate) -> dict:
            return {"blob": "x" * jobs_module._MAXIMUM_TRANSPORT_JSON_BYTES}

        with self.assertRaisesRegex(ValueError, "per-record byte limit"):
            run_kernel_direction_cache(
                definition,
                oversized_transport,
                self.root / "oversized-transport",
            )

        def oversized_task(_context, _coordinate) -> dict:
            return {"blob": "x" * 20_000}

        with self.assertRaisesRegex(ValueError, "task payload limit"):
            run_kernel_direction_cache(
                definition,
                oversized_task,
                self.root / "oversized-task",
            )

        nested = {"leaf": True}
        for _index in range(jobs_module._MAXIMUM_JSON_DEPTH + 2):
            nested = {"nested": nested}

        def too_deep(_context, _coordinate) -> dict:
            return nested

        with self.assertRaisesRegex(ValueError, "JSON depth limit"):
            run_kernel_direction_cache(
                definition,
                too_deep,
                self.root / "too-deep",
            )

        valid = run_kernel_direction_cache(
            definition,
            deterministic_transport,
            self.root / "total-limit",
        )
        forged_first = replace(
            valid.results[0],
            byte_length=jobs_module._MAXIMUM_TOTAL_CACHE_BYTES + 1,
        )
        forged_run = replace(
            valid,
            results=(forged_first, *valid.results[1:]),
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelJobError,
            "total cache byte limit",
        ):
            tuple(iter_cached_kernel_direction_records(definition, forged_run))

    def test_minimum_canonical_payload_is_rejected_before_evaluation(self) -> None:
        plan = KerrKernelDirectionTaskPlan(
            formulation=FORWARD,
            annulus_count=1,
            rho_order=4,
            mu_order=64,
            psi_count=64,
            directions_per_task=4096,
        )
        calls = 0

        def never_reached(_context, _coordinate) -> dict:
            nonlocal calls
            calls += 1
            return {}

        def construct_and_run() -> None:
            definition = make_kernel_direction_cache_definition(
                plan,
                identity=self.identity(
                    rho_order=4,
                    mu_order=64,
                    psi_count=64,
                    maximum_direction_evaluations=2_000_000,
                    maximum_whole_ray_traces=8_000_000,
                ),
            )
            run_kernel_direction_cache(
                definition,
                never_reached,
                self.root / "minimum-payload-attack",
            )

        with self.assertRaisesRegex(
            ValueError,
            "canonical transport=\\{\\} records",
        ):
            construct_and_run()
        self.assertEqual(calls, 0)

    def test_writer_rejects_symlink_in_higher_cache_root_ancestor(self) -> None:
        definition = self.definition(self.plan())
        target = self.root / "higher-ancestor-attack-target"
        target.mkdir()
        linked_ancestor = self.root / "linked-higher-ancestor"
        linked_ancestor.symlink_to(target, target_is_directory=True)
        calls = 0

        def counted(context, coordinate) -> dict:
            nonlocal calls
            calls += 1
            return deterministic_transport(context, coordinate)

        with self.assertRaisesRegex(
            KerrReturningRadiationKernelJobError,
            "absolute path contains a symlink",
        ):
            run_kernel_direction_cache(
                definition,
                counted,
                linked_ancestor / "nested-cache-root",
            )
        self.assertEqual(calls, 0)
        self.assertEqual(tuple(target.iterdir()), ())

    def test_writer_detects_job_directory_swap_before_publication(self) -> None:
        definition = self.definition(self.plan(directions_per_task=1))
        cache_root = self.root / "swap-cache"
        target = self.root / "job-directory-attack-target"
        target.mkdir()
        displaced = self.root / "displaced-job-directory"
        calls = 0

        def swap_to_symlink(context, coordinate) -> dict:
            nonlocal calls
            calls += 1
            if calls == 1:
                job_directory = cache_root / definition.job_spec.job_key
                job_directory.rename(displaced)
                job_directory.symlink_to(target, target_is_directory=True)
            return deterministic_transport(context, coordinate)

        with self.assertRaisesRegex(
            KerrReturningRadiationKernelJobError,
            "cache path became a symlink|path changed",
        ):
            run_kernel_direction_cache(
                definition,
                swap_to_symlink,
                cache_root,
                jobs=1,
            )
        self.assertEqual(calls, 1)
        self.assertEqual(tuple(target.iterdir()), ())

    def test_cache_paths_reject_symlink_parents_and_oversized_receipts(self) -> None:
        definition = self.definition(self.plan())
        run = run_kernel_direction_cache(
            definition,
            deterministic_transport,
            self.root / "cache",
        )
        task_directory = run.results[0].payload_path.parent
        actual_directory = task_directory.with_name("tasks-real")
        task_directory.rename(actual_directory)
        task_directory.symlink_to(actual_directory, target_is_directory=True)
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelJobError,
            "symlink or non-directory",
        ):
            tuple(iter_cached_kernel_direction_records(definition, run))

        second = run_kernel_direction_cache(
            definition,
            deterministic_transport,
            self.root / "receipt-cache",
        )
        second.results[0].receipt_path.write_bytes(
            b"x" * (jobs_module._MAXIMUM_RECEIPT_BYTES + 1)
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelJobError,
            "receipt.*hard byte limit",
        ):
            tuple(iter_cached_kernel_direction_records(definition, second))

        real_parent = self.root / "real-parent"
        real_parent.mkdir()
        linked_parent = self.root / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelJobError,
            "absolute path contains a symlink",
        ):
            run_kernel_direction_cache(
                definition,
                deterministic_transport,
                linked_parent / "new-cache",
            )


if __name__ == "__main__":
    unittest.main()
