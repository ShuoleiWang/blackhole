from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import offline.job as job_module
from offline.job import (
    InputArtifact,
    JobSpec,
    MAXIMUM_IN_FLIGHT_TASKS,
    MAXIMUM_JOB_DOCUMENT_BYTES,
    MAXIMUM_TASK_COUNT,
    MAXIMUM_TASK_PAYLOAD_BYTES,
    MAXIMUM_WORKERS,
    OfflineJobCacheError,
    TaskKey,
    canonical_json_bytes,
    job_key,
    run_job,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
SOURCE_A = "1" * 64
SOURCE_B = "2" * 64


class AlwaysEqualStr(str):
    def __eq__(self, other: object) -> bool:
        del other
        return True

    __hash__ = str.__hash__


class PolicyBypassInt(int):
    def __lt__(self, other: object) -> bool:
        del other
        return False

    __hash__ = int.__hash__


def thread_executor(max_workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=max_workers)


def deterministic_payload(spec: JobSpec, key: TaskKey) -> bytes:
    value = {
        "algorithm": spec.algorithm_version,
        "parameter": spec.parameters["value"],
        "task": key.as_dict(),
    }
    return canonical_json_bytes(value)


def deterministic_memoryview_payload(spec: JobSpec, key: TaskKey) -> memoryview:
    return memoryview(deterministic_payload(spec, key))


class OfflineJobTests(unittest.TestCase):
    def setUp(self) -> None:
        # Generic job cache roots reject symlinked ancestors.  On macOS the
        # default /var temporary path is itself a symlink to /private/var.
        self.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.tasks = tuple(
            TaskKey(0, 0, x, 1, 1)
            for x in range(6)
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def spec(
        self,
        *,
        value: int = 7,
        algorithm: str = "1.0.0",
        input_hash: str = HASH_A,
        source_hash: str = SOURCE_A,
        tasks: tuple[TaskKey, ...] | None = None,
    ) -> JobSpec:
        return JobSpec(
            producer="tests.deterministic-payload",
            algorithm_version=algorithm,
            tasks=tasks or self.tasks,
            parameters={"nested": {"enabled": True}, "value": value},
            inputs=(InputArtifact("input.bin", 12, input_hash),),
            producer_source_hashes=(source_hash,),
            record_bytes=1,
        )

    def test_job_key_is_canonical_and_excludes_scheduler_controls(self) -> None:
        first = self.spec()
        second = JobSpec(
            producer=first.producer,
            algorithm_version=first.algorithm_version,
            tasks=tuple(reversed(first.tasks)),
            parameters={"value": 7, "nested": {"enabled": True}},
            inputs=tuple(reversed(first.inputs)),
            producer_source_hashes=tuple(reversed(first.producer_source_hashes)),
            record_bytes=1,
        )
        self.assertEqual(job_key(first), job_key(second))

        initial = run_job(first, deterministic_payload, self.cache, jobs=1)
        resumed = run_job(
            first,
            deterministic_payload,
            self.cache,
            jobs=4,
            max_in_flight=1,
            executor_factory=thread_executor,
        )
        self.assertEqual(initial.job_key, resumed.job_key)
        self.assertEqual(resumed.reused_tasks, len(self.tasks))
        self.assertEqual(resumed.executed_tasks, 0)

    def test_failure_resumes_only_tasks_without_valid_receipts(self) -> None:
        spec = self.spec(tasks=self.tasks[:3])
        attempted: list[TaskKey] = []

        def fails_on_second(_spec: JobSpec, key: TaskKey) -> bytes:
            attempted.append(key)
            if key == spec.tasks[1]:
                raise RuntimeError("injected task failure")
            return deterministic_payload(_spec, key)

        with self.assertRaisesRegex(RuntimeError, "injected task failure"):
            run_job(spec, fails_on_second, self.cache, jobs=1)
        self.assertEqual(attempted, list(spec.tasks[:2]))

        resumed_attempts: list[TaskKey] = []

        def records_attempts(_spec: JobSpec, key: TaskKey) -> bytes:
            resumed_attempts.append(key)
            return deterministic_payload(_spec, key)

        resumed = run_job(spec, records_attempts, self.cache, jobs=1)
        self.assertEqual(resumed.reused_tasks, 1)
        self.assertEqual(resumed.executed_tasks, 2)
        self.assertEqual(resumed_attempts, list(spec.tasks[1:]))

    def test_orphan_partial_and_payload_without_receipt_are_not_resumed(self) -> None:
        spec = self.spec(tasks=self.tasks[:1])
        task_directory = self.cache / spec.job_key / "tasks"
        task_directory.mkdir(parents=True)
        stem = spec.tasks[0].file_stem
        (task_directory / f".{stem}.bin.partial-stale").write_bytes(b"stale")
        (task_directory / f"{stem}.bin").write_bytes(b"orphan")

        calls = 0

        def counted(_spec: JobSpec, key: TaskKey) -> bytes:
            nonlocal calls
            calls += 1
            return deterministic_payload(_spec, key)

        report = run_job(spec, counted, self.cache, jobs=1)
        self.assertEqual(calls, 1)
        self.assertEqual(report.executed_tasks, 1)
        self.assertEqual(
            report.results[0].payload_path.read_bytes(),
            deterministic_payload(spec, spec.tasks[0]),
        )

    def test_corrupt_payload_is_recomputed_while_other_tasks_are_reused(self) -> None:
        spec = self.spec(tasks=self.tasks[:3])
        first = run_job(spec, deterministic_payload, self.cache, jobs=1)
        first.results[1].payload_path.write_bytes(b"corrupt")

        recomputed: list[TaskKey] = []

        def counted(_spec: JobSpec, key: TaskKey) -> bytes:
            recomputed.append(key)
            return deterministic_payload(_spec, key)

        second = run_job(spec, counted, self.cache, jobs=1)
        self.assertEqual(recomputed, [spec.tasks[1]])
        self.assertEqual(second.reused_tasks, 2)
        self.assertEqual(second.executed_tasks, 1)
        for result in second.results:
            self.assertEqual(result.sha256, hashlib.sha256(
                deterministic_payload(spec, result.key)
            ).hexdigest())

    def test_completion_order_does_not_change_result_order_or_payload_hashes(self) -> None:
        spec = self.spec()
        serial_cache = Path(self.temporary.name) / "serial"
        parallel_cache = Path(self.temporary.name) / "parallel"
        serial = run_job(spec, deterministic_payload, serial_cache, jobs=1)
        completion_order: list[TaskKey] = []
        lock = threading.Lock()

        def reverse_completion(_spec: JobSpec, key: TaskKey) -> bytes:
            time.sleep(0.006 * (len(spec.tasks) - key.x))
            with lock:
                completion_order.append(key)
            return deterministic_payload(_spec, key)

        parallel = run_job(
            spec,
            reverse_completion,
            parallel_cache,
            jobs=3,
            max_in_flight=3,
            executor_factory=thread_executor,
        )
        self.assertNotEqual(completion_order, list(spec.tasks))
        self.assertEqual(
            [result.key for result in parallel.results],
            list(spec.tasks),
        )
        self.assertEqual(
            [result.sha256 for result in serial.results],
            [result.sha256 for result in parallel.results],
        )

    def test_job_key_changes_with_every_scientific_identity_input(self) -> None:
        baseline = self.spec()
        variants = (
            self.spec(value=8),
            self.spec(algorithm="1.0.1"),
            self.spec(input_hash=HASH_B),
            self.spec(source_hash=SOURCE_B),
            self.spec(tasks=self.tasks[:-1]),
        )
        self.assertEqual(len({baseline.job_key, *(item.job_key for item in variants)}), 6)

    def test_max_in_flight_bounds_submitted_and_active_work(self) -> None:
        spec = self.spec()
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def measured(_spec: JobSpec, key: TaskKey) -> bytes:
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                time.sleep(0.015)
                return deterministic_payload(_spec, key)
            finally:
                with lock:
                    active -= 1

        report = run_job(
            spec,
            measured,
            self.cache,
            jobs=4,
            max_in_flight=2,
            executor_factory=thread_executor,
        )
        self.assertEqual(report.max_in_flight_observed, 2)
        self.assertLessEqual(maximum_active, 2)
        self.assertEqual(report.executed_tasks, len(spec.tasks))

    def test_default_process_executor_round_trip(self) -> None:
        spec = self.spec(tasks=self.tasks[:2])
        try:
            report = run_job(
                spec,
                deterministic_memoryview_payload,
                self.cache,
                jobs=2,
                max_in_flight=2,
            )
        except PermissionError as error:
            if error.errno != errno.EPERM:
                raise
            self.skipTest("sandbox forbids multiprocessing semaphores")
        self.assertEqual(report.executed_tasks, 2)
        self.assertEqual(
            [result.payload_path.read_bytes() for result in report.results],
            [deterministic_payload(spec, key) for key in spec.tasks],
        )

    def test_receipts_are_canonical_and_bind_payload_hash_and_task(self) -> None:
        spec = self.spec(tasks=self.tasks[:1])
        report = run_job(spec, deterministic_payload, self.cache, jobs=1)
        result = report.results[0]
        receipt_bytes = result.receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        self.assertEqual(receipt_bytes, canonical_json_bytes(receipt))
        self.assertEqual(receipt["jobKey"], spec.job_key)
        self.assertEqual(receipt["task"], result.key.as_dict())
        self.assertEqual(receipt["byteLength"], result.payload_path.stat().st_size)
        self.assertEqual(receipt["recordCount"], result.record_count)
        self.assertEqual(result.record_count, result.byte_length)
        self.assertEqual(receipt["sha256"], hashlib.sha256(
            result.payload_path.read_bytes()
        ).hexdigest())

    def test_concurrent_runners_publish_one_authenticated_task_pair(self) -> None:
        spec = self.spec(tasks=self.tasks[:1])
        calls = 0
        lock = threading.Lock()

        def slow_counted(_spec: JobSpec, key: TaskKey) -> bytes:
            nonlocal calls
            with lock:
                calls += 1
            time.sleep(0.03)
            return deterministic_payload(_spec, key)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    run_job,
                    spec,
                    slow_counted,
                    self.cache,
                    jobs=1,
                )
                for _ in range(2)
            ]
            reports = [future.result() for future in futures]

        self.assertEqual(calls, 1)
        self.assertEqual(
            sorted(report.reused_tasks for report in reports),
            [0, 1],
        )
        for report in reports:
            result = report.results[0]
            expected_payload = (
                self.cache
                / spec.job_key
                / "tasks"
                / f"{result.key.file_stem}.bin"
            )
            expected_receipt = expected_payload.with_name(
                f"{result.key.file_stem}.receipt.json"
            )
            self.assertTrue(result.payload_path.is_absolute())
            self.assertTrue(result.receipt_path.is_absolute())
            self.assertEqual(result.payload_path, expected_payload)
            self.assertEqual(result.receipt_path, expected_receipt)
            self.assertEqual(
                hashlib.sha256(result.payload_path.read_bytes()).hexdigest(),
                result.sha256,
            )

    def test_resource_controls_fail_before_cache_or_producer(self) -> None:
        self.assertEqual(MAXIMUM_WORKERS, 64)
        self.assertEqual(MAXIMUM_IN_FLIGHT_TASKS, 256)
        self.assertEqual(MAXIMUM_TASK_COUNT, 65_536)
        self.assertEqual(MAXIMUM_JOB_DOCUMENT_BYTES, 16 * 1024 * 1024)
        self.assertEqual(MAXIMUM_TASK_PAYLOAD_BYTES, 256 * 1024 * 1024)
        one_task = self.spec(tasks=self.tasks[:1])
        cases = (
            ("huge-workers", self.spec(), 10**9, 1),
            ("huge-in-flight", self.spec(), 1, 10**9),
            ("workers-exceed-task-count", one_task, 2, 1),
            ("in-flight-exceeds-task-count", one_task, 1, 2),
        )
        for name, spec, jobs, in_flight in cases:
            with self.subTest(name=name):
                cache = self.root / name
                calls = 0

                def counted(_spec: JobSpec, key: TaskKey) -> bytes:
                    nonlocal calls
                    calls += 1
                    return deterministic_payload(_spec, key)

                with self.assertRaisesRegex(ValueError, "must not exceed"):
                    run_job(
                        spec,
                        counted,
                        cache,
                        jobs=jobs,
                        max_in_flight=in_flight,
                        executor_factory=thread_executor,
                    )
                self.assertEqual(calls, 0)
                self.assertFalse(cache.exists())

        invalid_controls = (
            ("bool-workers", True, 1, thread_executor),
            ("float-workers", 1.0, 1, thread_executor),
            ("bool-in-flight", 1, True, thread_executor),
            ("float-in-flight", 1, 1.0, thread_executor),
            ("noncallable-executor", 1, 1, object()),
        )
        for name, jobs, in_flight, factory in invalid_controls:
            with self.subTest(name=name):
                cache = self.root / name
                with self.assertRaises((TypeError, ValueError)):
                    run_job(
                        one_task,
                        lambda _spec, _key: self.fail("invalid control ran"),
                        cache,
                        jobs=jobs,  # type: ignore[arg-type]
                        max_in_flight=in_flight,  # type: ignore[arg-type]
                        executor_factory=factory,  # type: ignore[arg-type]
                    )
                self.assertFalse(cache.exists())

        metadata_cache = self.root / "oversized-metadata"
        calls = 0

        def metadata_counted(_spec: JobSpec, key: TaskKey) -> bytes:
            nonlocal calls
            calls += 1
            return deterministic_payload(_spec, key)

        with (
            mock.patch.object(job_module, "MAXIMUM_JOB_DOCUMENT_BYTES", 128),
            self.assertRaisesRegex(ValueError, "metadata byte limit"),
        ):
            run_job(self.spec(), metadata_counted, metadata_cache, jobs=1)
        self.assertEqual(calls, 0)
        self.assertFalse(metadata_cache.exists())

        task_cap_cache = self.root / "task-cap"
        oversized_task_spec = self.spec(
            tasks=tuple(
                TaskKey(0, 0, x, 1, 1)
                for x in range(MAXIMUM_TASK_COUNT + 1)
            )
        )
        with self.assertRaisesRegex(ValueError, "task count"):
            run_job(
                oversized_task_spec,
                metadata_counted,
                task_cap_cache,
                jobs=1,
            )
        self.assertEqual(calls, 0)
        self.assertFalse(task_cap_cache.exists())

    def test_deep_exact_spec_schema_fails_before_cache_or_producer(self) -> None:
        task = self.tasks[0]
        ordinary = {
            "producer": "tests.deterministic-payload",
            "algorithm_version": "1.0.0",
            "tasks": (task,),
            "parameters": {"value": 7},
            "inputs": (InputArtifact("input.bin", 12, HASH_A),),
            "producer_source_hashes": (SOURCE_A,),
            "record_bytes": 1,
        }
        cases = {
            "producer-string-subclass": {
                **ordinary,
                "producer": AlwaysEqualStr("tests.deterministic-payload"),
            },
            "version-string-subclass": {
                **ordinary,
                "algorithm_version": AlwaysEqualStr("1.0.0"),
            },
            "task-field-int-subclass": {
                **ordinary,
                "tasks": (TaskKey(PolicyBypassInt(0), 0, 0, 1, 1),),
            },
            "input-uri-string-subclass": {
                **ordinary,
                "inputs": (
                    InputArtifact(AlwaysEqualStr("input.bin"), 12, HASH_A),
                ),
            },
            "input-length-int-subclass": {
                **ordinary,
                "inputs": (
                    InputArtifact("input.bin", PolicyBypassInt(12), HASH_A),
                ),
            },
            "input-hash-string-subclass": {
                **ordinary,
                "inputs": (
                    InputArtifact("input.bin", 12, AlwaysEqualStr(HASH_A)),
                ),
            },
            "source-hash-string-subclass": {
                **ordinary,
                "producer_source_hashes": (AlwaysEqualStr(SOURCE_A),),
            },
            "parameter-value-int-subclass": {
                **ordinary,
                "parameters": {"value": PolicyBypassInt(7)},
            },
            "parameter-key-string-subclass": {
                **ordinary,
                "parameters": {AlwaysEqualStr("value"): 7},
            },
            "record-bytes-int-subclass": {
                **ordinary,
                "record_bytes": PolicyBypassInt(1),
            },
        }
        for name, arguments in cases.items():
            with self.subTest(name=name):
                spec = JobSpec(**arguments)
                cache = self.root / f"exact-{name}"
                calls = 0

                def counted(_spec: JobSpec, key: TaskKey) -> bytes:
                    nonlocal calls
                    calls += 1
                    return deterministic_payload(_spec, key)

                with self.assertRaisesRegex(TypeError, "exact|Exact"):
                    run_job(spec, counted, cache, jobs=1)
                self.assertEqual(calls, 0)
                self.assertFalse(cache.exists())

        class ForeignJobSpec(JobSpec):
            pass

        foreign = ForeignJobSpec(**ordinary)
        foreign_cache = self.root / "exact-jobspec-subclass"
        with self.assertRaisesRegex(TypeError, "exact JobSpec"):
            run_job(
                foreign,
                lambda _spec, _key: self.fail("foreign JobSpec ran"),
                foreign_cache,
                jobs=1,
            )
        self.assertFalse(foreign_cache.exists())

    def test_spec_mutation_by_producer_fails_before_task_publication(self) -> None:
        spec = self.spec(tasks=self.tasks[:1])

        def mutates_spec(_spec: JobSpec, key: TaskKey) -> bytes:
            payload = deterministic_payload(_spec, key)
            object.__setattr__(_spec, "producer", "mutated-producer")
            return payload

        with self.assertRaisesRegex(OfflineJobCacheError, "JobSpec changed"):
            run_job(spec, mutates_spec, self.cache, jobs=1)
        task_directory = self.cache / job_key(self.spec(tasks=self.tasks[:1])) / "tasks"
        self.assertEqual(tuple(task_directory.glob("*.bin")), ())
        self.assertEqual(tuple(task_directory.glob("*.receipt.json")), ())

    def test_worker_payload_cap_prevents_parent_cache_publication(self) -> None:
        spec = self.spec(tasks=self.tasks[:2])
        cache = self.root / "oversized-worker-payload"
        calls = 0
        lock = threading.Lock()

        def oversized(_spec: JobSpec, key: TaskKey) -> bytes:
            nonlocal calls
            del _spec, key
            with lock:
                calls += 1
            return b"x" * 33

        with (
            mock.patch.object(job_module, "MAXIMUM_TASK_PAYLOAD_BYTES", 32),
            self.assertRaisesRegex(ValueError, "payload byte limit"),
        ):
            run_job(
                spec,
                oversized,
                cache,
                jobs=2,
                max_in_flight=2,
                executor_factory=thread_executor,
            )
        self.assertGreaterEqual(calls, 1)
        self.assertEqual(tuple(cache.rglob("*.bin")), ())
        self.assertEqual(tuple(cache.rglob("*.receipt.json")), ())

    def test_cache_root_requires_exact_absolute_nonsymlink_path(self) -> None:
        spec = self.spec(tasks=self.tasks[:1])
        calls = 0

        def counted(_spec: JobSpec, key: TaskKey) -> bytes:
            nonlocal calls
            calls += 1
            return deterministic_payload(_spec, key)

        with self.assertRaisesRegex(TypeError, "exact absolute"):
            run_job(spec, counted, Path("relative-cache"), jobs=1)
        with self.assertRaisesRegex(TypeError, "exact absolute"):
            run_job(spec, counted, str(self.cache), jobs=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(OfflineJobCacheError, "non-root"):
            run_job(spec, counted, Path("/"), jobs=1)
        self.assertEqual(calls, 0)
        self.assertFalse(self.cache.exists())

    def test_rejects_missing_or_symlinked_cache_root_components(self) -> None:
        spec = self.spec(tasks=self.tasks[:1])
        calls = 0

        def counted(_spec: JobSpec, key: TaskKey) -> bytes:
            nonlocal calls
            calls += 1
            return deterministic_payload(_spec, key)

        missing_parent = self.root / "missing-parent" / "cache"
        with self.assertRaisesRegex(OfflineJobCacheError, "missing ancestor"):
            run_job(spec, counted, missing_parent, jobs=1)
        self.assertFalse(self.root.joinpath("missing-parent").exists())

        ancestor_target = self.root / "ancestor-target"
        ancestor_target.mkdir()
        linked_ancestor = self.root / "linked-ancestor"
        linked_ancestor.symlink_to(ancestor_target, target_is_directory=True)
        with self.assertRaisesRegex(OfflineJobCacheError, "symlink"):
            run_job(spec, counted, linked_ancestor / "cache", jobs=1)
        self.assertEqual(tuple(ancestor_target.iterdir()), ())

        final_target = self.root / "final-target"
        final_target.mkdir()
        final_link = self.root / "final-link"
        final_link.symlink_to(final_target, target_is_directory=True)
        with self.assertRaisesRegex(OfflineJobCacheError, "symlink"):
            run_job(spec, counted, final_link, jobs=1)
        self.assertEqual(tuple(final_target.iterdir()), ())
        self.assertEqual(calls, 0)

    def test_rejects_symlinked_job_tasks_and_final_entries(self) -> None:
        spec = self.spec(tasks=self.tasks[:1])

        for layout in ("job-directory", "task-directory"):
            with self.subTest(layout=layout):
                cache = self.root / f"layout-{layout}"
                cache.mkdir()
                target = self.root / f"target-{layout}"
                target.mkdir()
                job_directory = cache / spec.job_key
                if layout == "job-directory":
                    job_directory.symlink_to(target, target_is_directory=True)
                else:
                    job_directory.mkdir()
                    (job_directory / "tasks").symlink_to(
                        target,
                        target_is_directory=True,
                    )
                calls = 0

                def counted(_spec: JobSpec, key: TaskKey) -> bytes:
                    nonlocal calls
                    calls += 1
                    return deterministic_payload(_spec, key)

                with self.assertRaisesRegex(
                    OfflineJobCacheError,
                    "symlink|non-directory",
                ):
                    run_job(spec, counted, cache, jobs=1)
                self.assertEqual(calls, 0)
                self.assertEqual(tuple(target.iterdir()), ())

        for entry in ("job.json", "payload", "receipt", "lock"):
            with self.subTest(entry=entry):
                cache = self.root / f"entry-{entry.replace('.', '-')}"
                tasks = cache / spec.job_key / "tasks"
                tasks.mkdir(parents=True)
                sentinel = self.root / f"sentinel-{entry.replace('.', '-')}"
                sentinel.write_bytes(b"sentinel")
                stem = spec.tasks[0].file_stem
                if entry == "job.json":
                    (tasks.parent / "job.json").symlink_to(sentinel)
                elif entry == "payload":
                    (tasks / f"{stem}.bin").symlink_to(sentinel)
                elif entry == "receipt":
                    (tasks / f"{stem}.receipt.json").symlink_to(sentinel)
                else:
                    (tasks / f"{stem}.lock").symlink_to(sentinel)
                calls = 0

                def counted(_spec: JobSpec, key: TaskKey) -> bytes:
                    nonlocal calls
                    calls += 1
                    return deterministic_payload(_spec, key)

                with self.assertRaisesRegex(OfflineJobCacheError, "symlink"):
                    run_job(spec, counted, cache, jobs=1)
                self.assertEqual(calls, 0)
                self.assertEqual(sentinel.read_bytes(), b"sentinel")

    def test_job_and_task_directory_swaps_fail_before_publication(self) -> None:
        spec = self.spec(tasks=self.tasks[:1])
        for mode in (
            "cache-real-directory",
            "job-symlink",
            "job-real-directory",
            "tasks-symlink",
            "tasks-real-directory",
        ):
            with self.subTest(mode=mode):
                cache = self.root / f"swap-{mode}"
                attack_target = self.root / f"attack-{mode}"
                displaced = self.root / f"displaced-{mode}"
                calls = 0

                def swaps_path(_spec: JobSpec, key: TaskKey) -> bytes:
                    nonlocal calls
                    calls += 1
                    job_directory = cache / _spec.job_key
                    task_directory = job_directory / "tasks"
                    if mode == "cache-real-directory":
                        cache.rename(displaced)
                        cache.mkdir()
                    elif mode.startswith("job-"):
                        job_directory.rename(displaced)
                        if mode == "job-symlink":
                            attack_target.mkdir()
                            job_directory.symlink_to(
                                attack_target,
                                target_is_directory=True,
                            )
                        else:
                            job_directory.mkdir()
                    else:
                        task_directory.rename(displaced)
                        if mode == "tasks-symlink":
                            attack_target.mkdir()
                            task_directory.symlink_to(
                                attack_target,
                                target_is_directory=True,
                            )
                        else:
                            task_directory.mkdir()
                    return deterministic_payload(_spec, key)

                with self.assertRaisesRegex(
                    OfflineJobCacheError,
                    "symlink|path changed|cache path",
                ):
                    run_job(spec, swaps_path, cache, jobs=1)
                self.assertEqual(calls, 1)
                if attack_target.exists():
                    self.assertEqual(tuple(attack_target.iterdir()), ())
                if mode == "cache-real-directory":
                    self.assertEqual(tuple(cache.iterdir()), ())
                elif mode == "job-real-directory":
                    self.assertEqual(tuple((cache / spec.job_key).iterdir()), ())
                elif mode == "tasks-real-directory":
                    self.assertEqual(
                        tuple((cache / spec.job_key / "tasks").iterdir()),
                        (),
                    )
                self.assertEqual(tuple(displaced.rglob("*.bin")), ())
                self.assertEqual(tuple(displaced.rglob("*.receipt.json")), ())

    def test_end_gate_detects_all_hit_task_directory_inode_swap(self) -> None:
        spec = self.spec(tasks=self.tasks[:1])
        run_job(spec, deterministic_payload, self.cache, jobs=1)
        original = job_module._secure_cached_result
        calls = 0

        def swaps_after_final_read(*arguments, **keywords):
            nonlocal calls
            calls += 1
            result = original(*arguments, **keywords)
            if calls == 2:
                task_directory = self.cache / spec.job_key / "tasks"
                task_directory.rename(self.root / "all-hit-old-tasks")
                task_directory.mkdir()
            return result

        with (
            mock.patch.object(
                job_module,
                "_secure_cached_result",
                side_effect=swaps_after_final_read,
            ),
            self.assertRaisesRegex(OfflineJobCacheError, "path changed"),
        ):
            run_job(
                spec,
                lambda _spec, _key: self.fail("all-hit run called producer"),
                self.cache,
                jobs=1,
            )
        self.assertEqual(calls, 2)
        self.assertEqual(tuple((self.cache / spec.job_key / "tasks").iterdir()), ())

    def test_lock_inode_swap_fails_before_payload_publication(self) -> None:
        spec = self.spec(tasks=self.tasks[:1])
        calls = 0

        def swaps_lock(_spec: JobSpec, key: TaskKey) -> bytes:
            nonlocal calls
            calls += 1
            tasks = self.cache / _spec.job_key / "tasks"
            lock_path = tasks / f"{key.file_stem}.lock"
            lock_path.rename(tasks / f"{key.file_stem}.lock.displaced")
            lock_path.write_bytes(b"replacement-lock")
            return deterministic_payload(_spec, key)

        with self.assertRaisesRegex(OfflineJobCacheError, "lock inode"):
            run_job(spec, swaps_lock, self.cache, jobs=1)
        self.assertEqual(calls, 1)
        task_directory = self.cache / spec.job_key / "tasks"
        self.assertEqual(tuple(task_directory.glob("*.bin")), ())
        self.assertEqual(tuple(task_directory.glob("*.receipt.json")), ())

    def test_stable_fd_read_rejects_same_inode_payload_mutation(self) -> None:
        spec = self.spec(tasks=self.tasks[:1])
        first = run_job(spec, deterministic_payload, self.cache, jobs=1)
        payload_path = first.results[0].payload_path
        payload_stat = payload_path.stat()
        original_read = job_module.os.read
        mutated = False

        def mutating_read(descriptor: int, count: int) -> bytes:
            nonlocal mutated
            chunk = original_read(descriptor, count)
            current = os.fstat(descriptor)
            if (
                not mutated
                and chunk
                and current.st_dev == payload_stat.st_dev
                and current.st_ino == payload_stat.st_ino
            ):
                mutated = True
                payload_path.write_bytes(b"x" * payload_stat.st_size)
            return chunk

        with (
            mock.patch.object(job_module.os, "read", side_effect=mutating_read),
            self.assertRaisesRegex(OfflineJobCacheError, "changed while being read"),
        ):
            run_job(
                spec,
                lambda _spec, _key: self.fail("unstable hit called producer"),
                self.cache,
                jobs=1,
            )
        self.assertTrue(mutated)


if __name__ == "__main__":
    unittest.main()
