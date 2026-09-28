#!/usr/bin/env python3
"""Replay an authenticated copied cache through the native CPU evaluator.

The supplied cache is opened read-only through the frozen golden-cache
extractor.  Every selected direction is evaluated exactly once under the new
native scientific/runtime identity and compared byte-for-byte with its cached
transport.  The tool never invokes a cache runner and never writes below the
supplied job directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import marshal
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import stat
import sys
import time
from types import CodeType, FunctionType, ModuleType
from typing import Any, Final, Mapping, Sequence

if __package__ in (None, ""):
    _BOOTSTRAP_ROOT = Path(__file__).absolute().parents[2]
    if str(_BOOTSTRAP_ROOT) not in sys.path:
        sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from offline.job import canonical_json_bytes  # noqa: E402
from offline.kerr_finite_thickness_area import (  # noqa: E402
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_returning_radiation_kernel import (  # noqa: E402
    KerrReturningRadiationKernelPolicy,
)
import offline.kerr_returning_radiation_kernel_cached as _cached  # noqa: E402
import offline.kerr_returning_radiation_kernel_jobs as _jobs  # noqa: E402
import tools.native.golden_cache as _golden_cache  # noqa: E402
import tools.native.phase_space_golden as _phase_space  # noqa: E402


SCHEMA: Final = "blackhole.native-cpu-copied-cache-replay/v2"
VERSION: Final = 2
FULL_DIRECTION_COUNT: Final = 38_144
MAXIMUM_WORKERS: Final = 8
TASK_WIDTH: Final = 64
MAXIMUM_REPLAY_ARTIFACT_BYTES: Final = 256 * 1024 * 1024
DEFAULT_REPLAY_DEADLINE_SECONDS: Final = 2 * 60 * 60
MAXIMUM_REPLAY_DEADLINE_SECONDS: Final = 24 * 60 * 60
_PATH_TYPE: Final = type(Path())
_REPLAY_SOURCE_ROOT: Final = Path(__file__).absolute().parents[2]
FROZEN_NESTED16_TRUST: Final = _golden_cache.FROZEN_NESTED16_TRUST
_EXTRACT_GOLDEN_CORPUS_ENTRY: Final = _golden_cache.extract_golden_corpus
_BUILD_NUMERIC_REPLAY_CONTEXT_ENTRY: Final = (
    _phase_space.build_nested16_numeric_replay_context
)
_HELPER_BINDINGS: Final = (
    (
        "tools.native.golden_cache",
        _golden_cache,
        _REPLAY_SOURCE_ROOT / "tools/native/golden_cache.py",
        "extract_golden_corpus",
        _EXTRACT_GOLDEN_CORPUS_ENTRY,
        "93a737d039884041d69faa8a766a92ed5600e3072091ef35e4badad4685e7d46",
        9016,
        "a4e9493525fdf0331bf6fa643d3c0d71074674a713e10deaa960eb3449e74b0f",
        964,
    ),
    (
        "tools.native.phase_space_golden",
        _phase_space,
        _REPLAY_SOURCE_ROOT / "tools/native/phase_space_golden.py",
        "build_nested16_numeric_replay_context",
        _BUILD_NUMERIC_REPLAY_CONTEXT_ENTRY,
        "bbe9cfa993f63b4f6dddf3073da83e0b408797a8ac167c595bbc713dd4671cb0",
        560,
        "49bf7066d1463a92d615b440f28100d5612787a1276c17e0a4b3e0735e04c60f",
        464,
    ),
)


class NativeCpuCacheReplayError(RuntimeError):
    """Raised when replay evidence cannot be produced exactly."""


_WORKER_CONTEXT: _jobs.KerrKernelScientificContext | None = None
_WORKER_PLAN: _jobs.KerrKernelDirectionTaskPlan | None = None


def _absolute_path(value: Path, label: str) -> Path:
    if type(value) is not _PATH_TYPE or not value.is_absolute():
        raise NativeCpuCacheReplayError(
            f"{label} must be an explicit absolute platform Path"
        )
    return Path(os.path.abspath(os.fspath(value)))


def _stable_artifact(
    path: Path,
    label: str,
    maximum_bytes: int = MAXIMUM_REPLAY_ARTIFACT_BYTES,
) -> dict[str, Any]:
    absolute = _absolute_path(path, label)
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise TypeError("artifact maximum_bytes must be a positive exact int")
    descriptor = -1
    try:
        path_before = os.lstat(absolute)
        if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(
            path_before.st_mode
        ):
            raise NativeCpuCacheReplayError(
                f"{label} must be a regular non-symlink file"
            )
        descriptor = os.open(
            absolute,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if before.st_size > maximum_bytes:
            raise NativeCpuCacheReplayError(
                f"{label} exceeds its {maximum_bytes}-byte limit"
            )
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        digest = hashlib.sha256()
        length = 0
        while True:
            block = os.read(
                descriptor,
                min(1024 * 1024, maximum_bytes - length + 1),
            )
            if not block:
                break
            digest.update(block)
            length += len(block)
            if length > maximum_bytes:
                raise NativeCpuCacheReplayError(
                    f"{label} grew beyond its byte limit"
                )
        after = os.fstat(descriptor)
        path_after = os.lstat(absolute)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        path_identity = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_mode,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        if identity != after_identity or identity != path_identity:
            raise NativeCpuCacheReplayError(f"{label} changed while read")
        if length != before.st_size:
            raise NativeCpuCacheReplayError(f"{label} length changed while read")
        return {
            "artifactName": absolute.name,
            "byteLength": length,
            "sha256": digest.hexdigest(),
        }
    except OSError as error:
        raise NativeCpuCacheReplayError(f"cannot authenticate {label}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _normalized_helper_code(code: CodeType, logical_path: str) -> CodeType:
    if type(code) is not CodeType:
        raise TypeError("replay helper code must have exact CodeType")
    return code.replace(
        co_filename=logical_path,
        co_consts=tuple(
            _normalized_helper_code(item, logical_path)
            if type(item) is CodeType
            else item
            for item in code.co_consts
        ),
    )


def _helper_source_artifacts() -> dict[str, dict[str, Any]]:
    """Freeze helper module origins, callables, and exact source bytes."""

    result: dict[str, dict[str, Any]] = {}
    for (
        module_name,
        module,
        path,
        callable_name,
        expected_callable,
        expected_source_sha256,
        expected_code_bytes,
        expected_code_sha256,
        expected_first_line,
    ) in _HELPER_BINDINGS:
        current_module = sys.modules.get(module_name)
        if (
            type(module) is not ModuleType
            or current_module is not module
            or type(module.__file__) is not str
            or Path(os.path.abspath(module.__file__)) != path
        ):
            raise NativeCpuCacheReplayError(
                f"replay helper module {module_name} has a foreign origin"
            )
        current_callable = getattr(module, callable_name, None)
        if (
            type(expected_callable) is not FunctionType
            or current_callable is not expected_callable
            or expected_callable.__module__ != module_name
            or expected_callable.__name__ != callable_name
            or type(expected_callable.__code__.co_filename) is not str
            or Path(os.path.abspath(expected_callable.__code__.co_filename))
            != path
        ):
            raise NativeCpuCacheReplayError(
                f"replay helper callable {module_name}.{callable_name} changed"
            )
        source_artifact = _stable_artifact(
            path,
            f"replay helper source {module_name}",
        )
        logical_path = path.relative_to(_REPLAY_SOURCE_ROOT).as_posix()
        code_payload = marshal.dumps(
            _normalized_helper_code(expected_callable.__code__, logical_path)
        )
        if (
            source_artifact["sha256"] != expected_source_sha256
            or len(code_payload) != expected_code_bytes
            or hashlib.sha256(code_payload).hexdigest()
            != expected_code_sha256
            or expected_callable.__code__.co_firstlineno != expected_first_line
        ):
            raise NativeCpuCacheReplayError(
                f"replay helper {module_name} differs from reviewed source/code"
            )
        result[module_name] = {
            **source_artifact,
            "callableCodeByteLength": len(code_payload),
            "callableCodeSha256": expected_code_sha256,
            "callableFirstLineNumber": expected_first_line,
            "logicalPath": logical_path,
        }
    if (
        _EXTRACT_GOLDEN_CORPUS_ENTRY is not _HELPER_BINDINGS[0][4]
        or _BUILD_NUMERIC_REPLAY_CONTEXT_ENTRY is not _HELPER_BINDINGS[1][4]
    ):
        raise NativeCpuCacheReplayError("invoked replay helper binding changed")
    if _golden_cache.FROZEN_NESTED16_TRUST is not FROZEN_NESTED16_TRUST:
        raise NativeCpuCacheReplayError("frozen copied-cache trust binding changed")
    return result


_FROZEN_REPLAY_HELPER_ARTIFACTS: Final = _helper_source_artifacts()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _strict_policy() -> KerrReturningRadiationKernelPolicy:
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


def _native_definition(library: Path):
    replay = _BUILD_NUMERIC_REPLAY_CONTEXT_ENTRY()
    identity = replay.scientific_context.identity
    return _cached.build_native_forward_kerr_returning_radiation_kernel_cache_definition(
        identity.surface,
        native_library_path=library,
        termination=identity.termination,
        annulus_edges_over_mass=identity.annulus_edges_over_mass,
        ray_options=identity.fine_ray_options,
        surface_options=identity.fine_surface_options,
        coarse_ray_options=identity.coarse_ray_options,
        coarse_surface_options=identity.coarse_surface_options,
        policy=_strict_policy(),
        area_policy=KerrFiniteThicknessAreaQuadraturePolicy(),
        directions_per_task=TASK_WIDTH,
    )


def _coordinate(document: Mapping[str, Any]) -> _jobs.KerrKernelDirectionCoordinate:
    if type(document) is not dict:
        raise NativeCpuCacheReplayError("cached coordinate is not an exact dict")
    return _jobs.KerrKernelDirectionCoordinate(
        document["ordinal"],
        document["passIndex"],
        document["passName"],
        document["faceIndex"],
        document["face"],
        document["annulusIndex"],
        document["rhoIndex"],
        document["muIndex"],
        document["psiIndex"],
    )


def _evaluator_artifact(
    context: _jobs.KerrKernelScientificContext,
) -> Any:
    matches = tuple(
        item
        for item in context.inputs
        if item.uri.startswith(_cached._EVALUATOR_ARTIFACT_URI_PREFIX)
    )
    if len(matches) != 1:
        raise NativeCpuCacheReplayError(
            "native context lacks one evaluator artifact"
        )
    return matches[0]


def _revalidate_worker_identity() -> None:
    context = _WORKER_CONTEXT
    if type(context) is not _jobs.KerrKernelScientificContext:
        raise NativeCpuCacheReplayError("worker context is not initialized")
    rebuilt = _jobs.KerrKernelScientificContext(
        context.plan,
        context.identity,
        context.inputs,
        context.scientific_job_key,
        context.evaluator_runtime_binding,
    )
    if _cached._evaluator_id_from_context(rebuilt) != (
        _cached._NATIVE_FORWARD_EVALUATOR_ID
    ):
        raise NativeCpuCacheReplayError("worker evaluator identity is not native")
    expected = _cached._evaluator_input(_cached._NATIVE_FORWARD_EVALUATOR_ID)
    if _evaluator_artifact(rebuilt) != expected:
        raise NativeCpuCacheReplayError("worker evaluator source changed")
    binding = rebuilt.evaluator_runtime_binding
    if type(binding) is not _jobs.KerrKernelEvaluatorRuntimeBinding:
        raise NativeCpuCacheReplayError("worker runtime binding is missing")
    _jobs.revalidate_kernel_direction_evaluator_runtime_binding(binding)


def _worker_initialize(
    context: _jobs.KerrKernelScientificContext,
    plan: _jobs.KerrKernelDirectionTaskPlan,
) -> None:
    global _WORKER_CONTEXT, _WORKER_PLAN
    _WORKER_CONTEXT = _jobs.KerrKernelScientificContext(
        context.plan,
        context.identity,
        context.inputs,
        context.scientific_job_key,
        context.evaluator_runtime_binding,
    )
    _WORKER_PLAN = _jobs.KerrKernelDirectionTaskPlan(
        plan.formulation,
        plan.annulus_count,
        plan.rho_order,
        plan.mu_order,
        plan.psi_count,
        plan.directions_per_task,
    )
    _revalidate_worker_identity()


def _worker_replay_chunk(
    chunk: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    context = _WORKER_CONTEXT
    plan = _WORKER_PLAN
    if (
        type(context) is not _jobs.KerrKernelScientificContext
        or type(plan) is not _jobs.KerrKernelDirectionTaskPlan
    ):
        raise NativeCpuCacheReplayError("worker was not initialized")
    _revalidate_worker_identity()
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    fate_distribution: Counter[str] = Counter()
    stream = hashlib.sha256()
    first_ordinal: int | None = None
    last_ordinal: int | None = None
    for record in chunk:
        if type(record) is not dict or set(record) != {"coordinate", "transport"}:
            raise NativeCpuCacheReplayError("worker record has a stale schema")
        coordinate = _coordinate(record["coordinate"])
        if first_ordinal is None:
            first_ordinal = coordinate.ordinal
        if last_ordinal is not None and coordinate.ordinal != last_ordinal + 1:
            raise NativeCpuCacheReplayError("worker chunk is not contiguous")
        last_ordinal = coordinate.ordinal
        actual = _cached._evaluate_forward_direction_native_cpu(
            context,
            coordinate,
        )
        expected = record["transport"]
        actual_payload = canonical_json_bytes(actual)
        expected_payload = canonical_json_bytes(expected)
        if actual_payload != expected_payload:
            raise NativeCpuCacheReplayError(
                "byte mismatch at ordinal "
                f"{coordinate.ordinal}: expected="
                f"{hashlib.sha256(expected_payload).hexdigest()} actual="
                f"{hashlib.sha256(actual_payload).hexdigest()}"
            )
        canonical_record = canonical_json_bytes(
            {"coordinate": coordinate.as_dict(), "transport": actual}
        )
        stream.update(canonical_record)
        fate_distribution[actual["transport"]["fate"]] += 1
    _revalidate_worker_identity()
    if first_ordinal is None or last_ordinal is None:
        raise NativeCpuCacheReplayError("worker received an empty chunk")
    return {
        "count": len(chunk),
        "cpuSeconds": time.process_time() - started_cpu,
        "fateDistribution": dict(sorted(fate_distribution.items())),
        "firstOrdinal": first_ordinal,
        "lastOrdinal": last_ordinal,
        "peakRssBytes": _peak_rss_bytes(),
        "pid": os.getpid(),
        "streamSha256": stream.hexdigest(),
        "wallSeconds": time.perf_counter() - started_wall,
    }


def _chunks(records: Sequence[dict[str, Any]]) -> tuple[tuple[dict[str, Any], ...], ...]:
    return tuple(
        tuple(records[index : index + TASK_WIDTH])
        for index in range(0, len(records), TASK_WIDTH)
    )


def _selection(value: str) -> tuple[int, int]:
    try:
        first, last = (int(item) for item in value.split(":", 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("selection must be START:STOP") from error
    if not 0 <= first < last <= FULL_DIRECTION_COUNT:
        raise argparse.ArgumentTypeError(
            f"selection must satisfy 0 <= START < STOP <= {FULL_DIRECTION_COUNT}"
        )
    return first, last


def _deadline_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("deadline must be finite seconds") from error
    if (
        not 1.0 <= parsed <= float(MAXIMUM_REPLAY_DEADLINE_SECONDS)
        or not parsed < float("inf")
    ):
        raise argparse.ArgumentTypeError(
            "deadline must lie in [1, 86400] seconds"
        )
    return parsed


def _open_secure_directory(path: Path, label: str) -> int:
    absolute = _absolute_path(path, label)
    if absolute == Path("/"):
        try:
            return os.open(
                "/",
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
        except OSError as error:
            raise NativeCpuCacheReplayError(f"cannot open {label}") from error
    descriptor = -1
    try:
        descriptor = os.open(
            "/",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        for component in absolute.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            snapshot = os.fstat(next_descriptor)
            if not stat.S_ISDIR(snapshot.st_mode):
                os.close(next_descriptor)
                raise NativeCpuCacheReplayError(f"{label} contains a non-directory")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException as error:
        if descriptor >= 0:
            os.close(descriptor)
        if isinstance(error, NativeCpuCacheReplayError):
            raise
        raise NativeCpuCacheReplayError(
            f"{label} contains a symlink or unreadable ancestor"
        ) from error


def _directory_is_at_or_below(
    candidate_descriptor: int,
    ancestor_descriptor: int,
) -> bool:
    ancestor = os.fstat(ancestor_descriptor)
    ancestor_identity = (ancestor.st_dev, ancestor.st_ino)
    current = os.dup(candidate_descriptor)
    seen: set[tuple[int, int]] = set()
    try:
        for _depth in range(1024):
            snapshot = os.fstat(current)
            identity = (snapshot.st_dev, snapshot.st_ino)
            if identity == ancestor_identity:
                return True
            if identity in seen:
                raise NativeCpuCacheReplayError(
                    "output parent ancestry contains a directory cycle"
                )
            seen.add(identity)
            parent = os.open(
                "..",
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current,
            )
            parent_snapshot = os.fstat(parent)
            parent_identity = (parent_snapshot.st_dev, parent_snapshot.st_ino)
            os.close(current)
            current = parent
            if parent_identity == identity:
                return False
        raise NativeCpuCacheReplayError("output parent ancestry exceeds hard depth")
    finally:
        os.close(current)


def _safe_write_new(path: Path, payload: bytes, job_directory: Path) -> str:
    output = _absolute_path(path, "output")
    job = _absolute_path(job_directory, "job_directory")
    if output == job or job in output.parents:
        raise NativeCpuCacheReplayError("output must remain outside the cache job")
    job_descriptor = _open_secure_directory(job, "job_directory")
    try:
        parent_descriptor = _open_secure_directory(output.parent, "output parent")
    except BaseException:
        os.close(job_descriptor)
        raise
    if _directory_is_at_or_below(parent_descriptor, job_descriptor):
        os.close(parent_descriptor)
        os.close(job_descriptor)
        raise NativeCpuCacheReplayError(
            "output directory identity is at or below the cache job"
        )
    descriptor = -1
    created = False
    try:
        descriptor = os.open(
            output.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        created = True
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short replay-report write")
            view = view[written:]
        os.fsync(descriptor)
        written_status = os.fstat(descriptor)
        if not stat.S_ISREG(written_status.st_mode):
            raise NativeCpuCacheReplayError("replay report is not a regular file")
        os.close(descriptor)
        descriptor = -1

        reopened_parent = _open_secure_directory(output.parent, "output parent")
        try:
            held_parent = os.fstat(parent_descriptor)
            reopened_status = os.fstat(reopened_parent)
            if (held_parent.st_dev, held_parent.st_ino) != (
                reopened_status.st_dev,
                reopened_status.st_ino,
            ):
                raise NativeCpuCacheReplayError(
                    "output parent path changed during publication"
                )
        finally:
            os.close(reopened_parent)

        descriptor = os.open(
            output.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        read_status = os.fstat(descriptor)
        if (read_status.st_dev, read_status.st_ino) != (
            written_status.st_dev,
            written_status.st_ino,
        ):
            raise NativeCpuCacheReplayError("replay report path changed")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        written = b"".join(blocks)
        if written != payload:
            raise NativeCpuCacheReplayError("written replay report changed")
        return hashlib.sha256(written).hexdigest()
    except BaseException:
        if created:
            try:
                os.unlink(output.name, dir_fd=parent_descriptor)
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
        os.close(job_descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_directory", type=Path)
    parser.add_argument("--dylib", required=True, type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, MAXIMUM_WORKERS + 1), default=8)
    parser.add_argument(
        "--deadline-seconds",
        type=_deadline_seconds,
        default=float(DEFAULT_REPLAY_DEADLINE_SECONDS),
        help="hard wall deadline; timed-out workers are terminated",
    )
    parser.add_argument(
        "--ordinal-range",
        type=_selection,
        default=(0, FULL_DIRECTION_COUNT),
        help="half-open replay selection; default is the complete copied cache",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def run_replay(
    arguments: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    job_directory = _absolute_path(arguments.job_directory, "job_directory")
    dylib = _absolute_path(arguments.dylib, "dylib")
    output = _absolute_path(arguments.output, "output")
    if type(arguments.workers) is not int or not 1 <= arguments.workers <= MAXIMUM_WORKERS:
        raise NativeCpuCacheReplayError("workers lies outside the hard bound")
    if (
        type(arguments.deadline_seconds) is not float
        or not 1.0
        <= arguments.deadline_seconds
        <= float(MAXIMUM_REPLAY_DEADLINE_SECONDS)
    ):
        raise NativeCpuCacheReplayError("deadline_seconds lies outside the hard bound")
    selection = arguments.ordinal_range
    if (
        type(selection) is not tuple
        or len(selection) != 2
        or any(type(item) is not int for item in selection)
        or not 0 <= selection[0] < selection[1] <= FULL_DIRECTION_COUNT
    ):
        raise NativeCpuCacheReplayError("ordinal selection is malformed")
    ordinals = tuple(range(selection[0], selection[1]))
    tool_before = _stable_artifact(Path(__file__).absolute(), "replay tool")
    helpers_before = _helper_source_artifacts()
    if helpers_before != _FROZEN_REPLAY_HELPER_ARTIFACTS:
        raise NativeCpuCacheReplayError(
            "replay helper source changed after module import"
        )
    dylib_before = _stable_artifact(dylib, "dylib")

    authentication_started = time.perf_counter()
    golden_before = _EXTRACT_GOLDEN_CORPUS_ENTRY(
        job_directory,
        selected_ordinals=ordinals,
        trust=FROZEN_NESTED16_TRUST,
    )
    authentication_seconds = time.perf_counter() - authentication_started
    records = golden_before["records"]
    if (
        type(records) is not list
        or len(records) != len(ordinals)
        or tuple(item["coordinate"]["ordinal"] for item in records) != ordinals
    ):
        raise NativeCpuCacheReplayError("authenticated selection is incomplete")

    definition_started = time.perf_counter()
    definition = _native_definition(dylib)
    closure_before = _cached._source_closure_for_definition(definition)
    closure_manifest = _cached._source_closure_manifest_sha256(closure_before)
    context = definition.scientific_context
    binding = context.evaluator_runtime_binding
    if type(binding) is not _jobs.KerrKernelEvaluatorRuntimeBinding:
        raise NativeCpuCacheReplayError("native definition lacks runtime binding")
    _jobs.revalidate_kernel_direction_evaluator_runtime_binding(binding)
    if _cached._evaluator_id_from_definition(definition) != (
        _cached._NATIVE_FORWARD_EVALUATOR_ID
    ):
        raise NativeCpuCacheReplayError("definition did not select native evaluator")
    definition_seconds = time.perf_counter() - definition_started

    expected_stream = hashlib.sha256()
    for record in records:
        expected_stream.update(canonical_json_bytes(record))
    chunks = _chunks(records)
    replay_started = time.perf_counter()
    deadline_at = time.monotonic() + arguments.deadline_seconds
    context_spawn = multiprocessing.get_context("spawn")
    pool = context_spawn.Pool(
        processes=arguments.workers,
        initializer=_worker_initialize,
        initargs=(context, definition.plan),
    )
    try:
        pending = tuple(
            pool.apply_async(_worker_replay_chunk, (chunk,))
            for chunk in chunks
        )
        reports: list[dict[str, Any]] = []
        for result in pending:
            remaining = deadline_at - time.monotonic()
            if remaining <= 0.0:
                raise multiprocessing.TimeoutError
            reports.append(result.get(timeout=remaining))
        pool.close()
        pool.join()
        chunk_reports = tuple(reports)
    except multiprocessing.TimeoutError as error:
        pool.terminate()
        pool.join()
        raise NativeCpuCacheReplayError(
            "native copied-cache replay exceeded its hard wall deadline"
        ) from error
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    replay_seconds = time.perf_counter() - replay_started

    next_ordinal = selection[0]
    fate_distribution: Counter[str] = Counter()
    worker_pids: set[int] = set()
    task_inventory: list[dict[str, Any]] = []
    for item in chunk_reports:
        if item["firstOrdinal"] != next_ordinal:
            raise NativeCpuCacheReplayError("worker results are out of task order")
        next_ordinal = item["lastOrdinal"] + 1
        fate_distribution.update(item["fateDistribution"])
        worker_pids.add(item["pid"])
        task_inventory.append(
            {
                "count": item["count"],
                "firstOrdinal": item["firstOrdinal"],
                "lastOrdinal": item["lastOrdinal"],
                "streamSha256": item["streamSha256"],
            }
        )
    if next_ordinal != selection[1]:
        raise NativeCpuCacheReplayError("worker results do not cover selection")

    _jobs.revalidate_kernel_direction_evaluator_runtime_binding(binding)
    closure_after = _cached._source_closure_for_definition(definition)
    if closure_after != closure_before:
        raise NativeCpuCacheReplayError("native source/runtime closure changed")
    golden_after = _EXTRACT_GOLDEN_CORPUS_ENTRY(
        job_directory,
        selected_ordinals=ordinals,
        trust=FROZEN_NESTED16_TRUST,
    )
    if golden_after != golden_before:
        raise NativeCpuCacheReplayError("copied cache changed during replay")
    if _stable_artifact(dylib, "dylib") != dylib_before:
        raise NativeCpuCacheReplayError("dylib changed during replay")
    if _stable_artifact(Path(__file__).absolute(), "replay tool") != tool_before:
        raise NativeCpuCacheReplayError("replay tool changed during execution")
    if (
        _helper_source_artifacts() != helpers_before
        or helpers_before != _FROZEN_REPLAY_HELPER_ARTIFACTS
    ):
        raise NativeCpuCacheReplayError("replay helper source closure changed")

    complete = selection == (0, FULL_DIRECTION_COUNT)
    evidence = golden_before["evidence"]
    report_without_hash = {
        "artifacts": {
            "backendDescriptorSha256": binding.backend_descriptor_sha256,
            "dylib": dylib_before,
            "evaluatorRuntimeInput": binding.descriptor_input.as_dict(),
            "librarySha256": binding.library_sha256,
            "replayHelperSources": helpers_before,
            "sourceRuntimeClosureManifestSha256": closure_manifest,
            "tool": tool_before,
        },
        "authentication": {
            "cacheJobKey": FROZEN_NESTED16_TRUST.cache_job_key,
            "cacheReadOnlyStableBeforeAfter": True,
            "directionStreamSha256": evidence["authenticatedDirectionStreamSha256"],
            "payloadSetSha256": evidence["authenticatedPayloadSetSha256"],
            "scientificJobKey": FROZEN_NESTED16_TRUST.scientific_job_key,
            "scientificPlanSha256": (
                FROZEN_NESTED16_TRUST.scientific_plan_sha256
            ),
            "copiedCacheIsComplete": evidence["isCompleteCache"],
            "plannedDirectionCount": definition.plan.direction_count,
            "plannedTaskCount": definition.plan.task_count,
            "presentAuthenticatedDirectionCount": evidence[
                "authenticatedDirectionCount"
            ],
            "presentAuthenticatedTaskCount": evidence["authenticatedTaskCount"],
        },
        "classification": "read-only same-code copied-cache native differential replay",
        "fateDistribution": dict(sorted(fate_distribution.items())),
        "identity": {
            "nativeCacheJobKey": definition.job_spec.job_key,
            "nativeEvaluatorId": _cached._NATIVE_FORWARD_EVALUATOR_ID,
            "nativeScientificJobKey": definition.scientific_job_key,
            "oldCacheKeyReused": False,
        },
        "qualification": {
            "allSelectedDirectionsByteExact": True,
            "completeAuthenticatedPresentPrefixReplay": complete,
            "independentPhysicsOrGeodesicOracle": False,
            "productionQualifiedForUncachedDirections": False,
            "rayWorkCountersAvailableFromNarrowTransport": False,
            "qualifiedScope": (
                "all 38144 authenticated present-prefix directions out of "
                "229376 planned directions"
                if complete
                else "bounded selected-direction smoke replay only"
            ),
        },
        "schema": SCHEMA,
        "selection": {
            "directionCount": len(records),
            "firstOrdinal": selection[0],
            "lastOrdinal": selection[1] - 1,
            "taskChunkCount": len(chunks),
        },
        "timing": {
            "cacheAuthenticationWallSeconds": authentication_seconds,
            "deadlineSeconds": arguments.deadline_seconds,
            "definitionWallSeconds": definition_seconds,
            "directionsPerSecond": len(records) / replay_seconds,
            "replayWallSeconds": replay_seconds,
            "sumWorkerCpuSeconds": sum(item["cpuSeconds"] for item in chunk_reports),
        },
        "workerEvidence": {
            "maximumWorkerPeakRssBytes": max(
                item["peakRssBytes"] for item in chunk_reports
            ),
            "requestedWorkers": arguments.workers,
            "taskInventorySha256": hashlib.sha256(
                canonical_json_bytes(task_inventory)
            ).hexdigest(),
            "uniqueWorkerPids": len(worker_pids),
        },
        "selectedDirectionStreamSha256": expected_stream.hexdigest(),
        "version": VERSION,
    }
    basis_sha = hashlib.sha256(canonical_json_bytes(report_without_hash)).hexdigest()
    report = {**report_without_hash, "reportBasisSha256": basis_sha}
    payload = canonical_json_bytes(report)
    output_sha = _safe_write_new(output, payload, job_directory)
    return report, output_sha


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        report, output_sha = run_replay(arguments)
    except (NativeCpuCacheReplayError, OSError, TypeError, ValueError) as error:
        print(f"native copied-cache replay failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "completeAuthenticatedPresentPrefixReplay": report[
                    "qualification"
                ][
                    "completeAuthenticatedPresentPrefixReplay"
                ],
                "directionCount": report["selection"]["directionCount"],
                "output": str(arguments.output),
                "outputSha256": output_sha,
                "schema": report["schema"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
