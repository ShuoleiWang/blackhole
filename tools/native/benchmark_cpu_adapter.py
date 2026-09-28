#!/usr/bin/env python3
"""Authenticate, byte-verify, and time the strict native CPU ray adapter.

This tool never invokes a cache runner and never writes into the supplied cache
job directory.  It authenticates the copied partial nested16 cache, rebuilds
the exact typed non-production numeric-replay context, evaluates selected
canonical coordinates directly, compares every full transport document
byte-exactly, and writes one new canonical JSON report.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import stat
import sys
import time
from typing import Any, Final, Mapping, Sequence

if __package__ in (None, ""):
    _BOOTSTRAP_ROOT = Path(__file__).absolute().parents[2]
    if str(_BOOTSTRAP_ROOT) not in sys.path:
        sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from offline.kerr_finite_thickness import LOWER, UPPER  # noqa: E402
from offline.kerr_finite_thickness_emitter import (  # noqa: E402
    KerrFiniteThicknessFaceEmitter,
)
from offline.kerr_finite_thickness_launch import (  # noqa: E402
    KerrFiniteThicknessEmissionLaunch,
    KerrFiniteThicknessSurfaceFrame,
)
from offline.kerr_native_cpu_backend import (  # noqa: E402
    StrictKerrCpuBackend,
)
import offline.kerr_returning_radiation_kernel as _forward  # noqa: E402
import offline.kerr_returning_radiation_kernel_cached as _cached  # noqa: E402
import offline.kerr_returning_radiation_rays as _rays  # noqa: E402
import offline.kerr_returning_radiation_native_cpu as _native  # noqa: E402
from tools.native.golden_cache import (  # noqa: E402
    CANDIDATE_SCHEMA,
    FROZEN_NESTED16_TRUST,
    canonical_json_bytes,
    compare_corpus_documents,
    extract_golden_corpus,
)
from tools.native.phase_space_golden import (  # noqa: E402
    _coordinate_for_ordinal,
    build_nested16_numeric_replay_context,
)


SCHEMA: Final = "blackhole.native-cpu-adapter-benchmark/v1"
VERSION: Final = 1
DEFAULT_NATIVE_TASK_START: Final = 704
TASK_DIRECTION_COUNT: Final = 64
MAXIMUM_NATIVE_DIRECTIONS: Final = 512
MAXIMUM_NATIVE_EVALUATIONS: Final = 2048
MAXIMUM_PYTHON_EVALUATIONS: Final = 64
MAXIMUM_REPEATS: Final = 32
_PATH_TYPE: Final = type(Path())


class CpuAdapterBenchmarkError(RuntimeError):
    """Raised when authentication, qualification, or bounds fail closed."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _ordinal_range(value: str) -> tuple[int, int]:
    try:
        first, last = value.split(":", 1)
        start = int(first)
        stop = int(last)
    except (AttributeError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "ordinal range must use half-open START:STOP integers"
        ) from error
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError(
            "ordinal range must satisfy 0 <= START < STOP"
        )
    return start, stop


def _absolute_path(value: Path, label: str) -> Path:
    if type(value) is not _PATH_TYPE or not value.is_absolute():
        raise CpuAdapterBenchmarkError(
            f"{label} must be an explicit absolute platform Path"
        )
    return Path(os.path.abspath(os.fspath(value)))


def _stable_regular_artifact(path: Path, label: str) -> dict[str, Any]:
    absolute = _absolute_path(path, label)
    descriptor = -1
    try:
        path_before = os.lstat(absolute)
        if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
            raise CpuAdapterBenchmarkError(
                f"{label} must be a regular non-symlink file"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        descriptor = os.open(absolute, flags)
        before = os.fstat(descriptor)
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
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            length += len(block)
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
        if identity != after_identity or identity != path_identity or length != before.st_size:
            raise CpuAdapterBenchmarkError(f"{label} changed while authenticated")
        return {
            "artifactName": absolute.name,
            "byteLength": length,
            "sha256": digest.hexdigest(),
        }
    except OSError as error:
        raise CpuAdapterBenchmarkError(f"cannot authenticate {label}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _selected_ordinals(arguments: argparse.Namespace) -> tuple[int, ...]:
    ranges = arguments.ordinal_range
    starts = arguments.task_start
    if ranges is not None and starts is not None:
        raise CpuAdapterBenchmarkError(
            "ordinal ranges and canonical task starts are mutually exclusive"
        )
    if ranges is None and starts is None:
        if arguments.backend != "native":
            raise CpuAdapterBenchmarkError(
                "Python mode requires an explicit ordinal range or task start"
            )
        starts = [DEFAULT_NATIVE_TASK_START]
    selected: list[int] = []
    if ranges is not None:
        if ranges != sorted(ranges):
            raise CpuAdapterBenchmarkError("ordinal ranges must be sorted")
        previous_stop = -1
        for start, stop in ranges:
            if start < previous_stop:
                raise CpuAdapterBenchmarkError(
                    "ordinal ranges must not overlap"
                )
            selected.extend(range(start, stop))
            previous_stop = stop
    else:
        assert starts is not None
        if starts != sorted(starts) or len(set(starts)) != len(starts):
            raise CpuAdapterBenchmarkError(
                "canonical task starts must be sorted and unique"
            )
        for start in starts:
            if start % TASK_DIRECTION_COUNT:
                raise CpuAdapterBenchmarkError(
                    "canonical 64-direction ordinal starts must be multiples of 64"
                )
            selected.extend(range(start, start + TASK_DIRECTION_COUNT))
    result = tuple(selected)
    if not result or tuple(sorted(set(result))) != result:
        raise CpuAdapterBenchmarkError(
            "selected ordinals must be unique, non-empty, and strictly ordered"
        )
    if len(result) > MAXIMUM_NATIVE_DIRECTIONS:
        raise CpuAdapterBenchmarkError(
            f"selection exceeds the hard {MAXIMUM_NATIVE_DIRECTIONS}-direction limit"
        )
    evaluations = len(result) * arguments.repeat
    if arguments.backend == "python":
        limit = arguments.python_direction_limit
        if limit is None:
            raise CpuAdapterBenchmarkError(
                "Python mode requires --python-direction-limit"
            )
        if limit > MAXIMUM_PYTHON_EVALUATIONS or evaluations > limit:
            raise CpuAdapterBenchmarkError(
                "Python direction evaluations exceed the explicit/hard bound"
            )
    elif arguments.python_direction_limit is not None:
        raise CpuAdapterBenchmarkError(
            "--python-direction-limit is valid only for explicit Python mode"
        )
    if arguments.backend == "native" and evaluations > MAXIMUM_NATIVE_EVALUATIONS:
        raise CpuAdapterBenchmarkError(
            "native direction evaluations exceed the hard benchmark bound"
        )
    return result


def _issued_transport(issued: Any) -> Any:
    fate = issued.fate
    if fate.startswith("return-"):
        receiver_face = issued.receiver_face
        receiver_radius = issued.receiver_radius_over_mass
        ratio = issued.emitter_to_receiver_frequency_ratio
        if (
            type(receiver_face) is not str
            or receiver_face not in (LOWER, UPPER)
            or type(receiver_radius) is not float
            or type(ratio) is not float
            or receiver_radius <= 0.0
            or ratio <= 0.0
        ):
            raise CpuAdapterBenchmarkError(
                "returned primitive has invalid receiver evidence"
            )
        coarse_face = issued.coarse_receiver_face
        coarse_radius = issued.coarse_receiver_radius_over_mass
        if type(coarse_face) is not str or type(coarse_radius) is not float:
            raise CpuAdapterBenchmarkError(
                "coarse returned primitive lacks receiver evidence"
            )
        g2 = ratio * ratio
        if not math.isfinite(g2) or g2 <= 0.0:
            raise CpuAdapterBenchmarkError("returned primitive has invalid g^2")
        return _forward._DirectionTransport(
            fate,
            receiver_face,
            receiver_radius,
            ratio,
            g2,
            issued.primitive_descriptor_sha256,
            coarse_face,
            coarse_radius,
        )
    return _forward._DirectionTransport(
        fate,
        None,
        None,
        None,
        0.0,
        issued.primitive_descriptor_sha256,
        None,
        None,
    )


def _evaluate_record(
    definition: Any,
    coordinate: Any,
    *,
    backend_mode: str,
    native_backend: StrictKerrCpuBackend | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    context = definition.scientific_context
    identity = context.identity
    node = _cached._forward_coordinate_node(context, coordinate)
    emitter = KerrFiniteThicknessFaceEmitter(
        metric=identity.surface.metric,
        calibration=identity.surface.calibration,
        pseudo_cylindrical_radius_over_mass=node.source_radius_over_mass,
        face=node.source_face,
    )
    launch = KerrFiniteThicknessEmissionLaunch(
        KerrFiniteThicknessSurfaceFrame(emitter),
        node.emission_angle_cosine,
        node.tangent_azimuth_rad,
        1.0,
    )
    keywords = {
        "termination": identity.termination,
        "ray_options": identity.fine_ray_options,
        "surface_options": identity.fine_surface_options,
        "coarse_ray_options": identity.coarse_ray_options,
        "coarse_surface_options": identity.coarse_surface_options,
    }
    if backend_mode == "native":
        if type(native_backend) is not StrictKerrCpuBackend:
            raise CpuAdapterBenchmarkError("native backend was not constructed")
        primitive, issue_token = (
            _native._trace_issued_kerr_returning_radiation_direction_native_cpu(
                launch,
                identity.surface,
                backend=native_backend,
                **keywords,
            )
        )
        issued = _native._consume_issued_kerr_returning_radiation_direction_native_cpu(
            primitive,
            issue_token,
        )
    else:
        if native_backend is not None:
            raise CpuAdapterBenchmarkError("Python mode unexpectedly owns native state")
        primitive, issue_token = (
            _rays._trace_issued_kerr_returning_radiation_direction(
                launch,
                identity.surface,
                **keywords,
            )
        )
        issued = _rays._consume_issued_kerr_returning_radiation_direction(
            primitive,
            issue_token,
        )
    transport = _issued_transport(issued)
    document = _cached._forward_transport_document(
        node,
        transport,
        identity.annulus_edges_over_mass,
    )
    fine_trace = primitive.ray.multi_surface_trace
    coarse_trace = primitive.coarse_ray.multi_surface_trace
    counters = {
        "coarseAcceptedSteps": primitive.coarse_ray.accepted_steps,
        "coarseProbeReintegrations": (
            0 if coarse_trace is None else coarse_trace.probe_reintegrations
        ),
        "coarseRejectedSteps": primitive.coarse_ray.rejected_steps,
        "fineAcceptedSteps": primitive.ray.accepted_steps,
        "fineProbeReintegrations": (
            0 if fine_trace is None else fine_trace.probe_reintegrations
        ),
        "fineRejectedSteps": primitive.ray.rejected_steps,
    }
    return document, counters


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _safe_write_new(path: Path, payload: bytes, job_directory: Path) -> str:
    output = _absolute_path(path, "output")
    job = _absolute_path(job_directory, "job_directory")
    try:
        if output.is_relative_to(job):
            raise CpuAdapterBenchmarkError("output must remain outside the cache job")
    except AttributeError:
        if job == output or job in output.parents:
            raise CpuAdapterBenchmarkError("output must remain outside the cache job")
    if output.exists() or output.is_symlink():
        raise CpuAdapterBenchmarkError("refusing to overwrite existing output")
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir():
        raise CpuAdapterBenchmarkError("output parent must be a regular directory")
    descriptor = -1
    created = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(output, flags, 0o600)
        created = True
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short benchmark report write")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        if created:
            try:
                output.unlink()
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    written = output.read_bytes()
    if written != payload:
        raise CpuAdapterBenchmarkError("written benchmark report changed")
    return hashlib.sha256(written).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_directory", type=Path)
    parser.add_argument("--dylib", required=True, type=Path)
    parser.add_argument("--backend", choices=("native", "python"), default="native")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--ordinal-range",
        action="append",
        type=_ordinal_range,
        help="half-open START:STOP; may be repeated in sorted non-overlapping order",
    )
    selection.add_argument(
        "--task-start",
        action="append",
        type=_non_negative_int,
        help="canonical ordinal start of one 64-direction task; may be repeated",
    )
    parser.add_argument("--repeat", type=_positive_int, default=1)
    parser.add_argument(
        "--python-direction-limit",
        type=_positive_int,
        help="required explicit total direction-evaluation budget in Python mode",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def run_benchmark(arguments: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if arguments.repeat > MAXIMUM_REPEATS:
        raise CpuAdapterBenchmarkError(
            f"repeat exceeds hard limit {MAXIMUM_REPEATS}"
        )
    job_directory = _absolute_path(arguments.job_directory, "job_directory")
    dylib = _absolute_path(arguments.dylib, "dylib")
    ordinals = _selected_ordinals(arguments)
    dylib_artifact = _stable_regular_artifact(dylib, "dylib")
    tool_artifact = _stable_regular_artifact(
        Path(__file__).absolute(), "benchmark tool"
    )

    authentication_wall_start = time.perf_counter()
    golden_before = extract_golden_corpus(
        job_directory,
        selected_ordinals=ordinals,
        trust=FROZEN_NESTED16_TRUST,
    )
    authentication_wall = time.perf_counter() - authentication_wall_start
    golden_sha256 = hashlib.sha256(canonical_json_bytes(golden_before)).hexdigest()
    expected_by_ordinal = {
        item["coordinate"]["ordinal"]: item for item in golden_before["records"]
    }
    if tuple(expected_by_ordinal) != ordinals:
        raise CpuAdapterBenchmarkError(
            "authenticated cache records differ from selected ordinal order"
        )

    context_wall_start = time.perf_counter()
    numeric_replay = build_nested16_numeric_replay_context()
    definition = numeric_replay.definition
    numeric_replay_descriptor = numeric_replay.descriptor()
    closure_before = _cached._source_closure_manifest()
    coordinates = tuple(
        _coordinate_for_ordinal(definition.plan, ordinal) for ordinal in ordinals
    )
    context_wall = time.perf_counter() - context_wall_start

    backend_wall_start = time.perf_counter()
    if arguments.backend == "native":
        native_backend: StrictKerrCpuBackend | None = StrictKerrCpuBackend(dylib)
        backend_descriptor = native_backend.model_descriptor()
        backend_descriptor_sha256 = native_backend.model_descriptor_sha256
        no_fallback = True
    else:
        native_backend = None
        backend_descriptor = _cached._numeric_backend_descriptor()
        backend_descriptor_sha256 = hashlib.sha256(
            canonical_json_bytes(backend_descriptor)
        ).hexdigest()
        no_fallback = None
    backend_wall = time.perf_counter() - backend_wall_start

    repeat_reports: list[dict[str, Any]] = []
    candidate_sha256_values: list[str] = []
    comparison_sha256_values: list[str] = []
    final_candidate: dict[str, Any] | None = None
    fate_distribution: Counter[str] = Counter()
    for repeat_index in range(arguments.repeat):
        counters = Counter()
        records: list[dict[str, Any]] = []
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        for coordinate in coordinates:
            transport, one_counters = _evaluate_record(
                definition,
                coordinate,
                backend_mode=arguments.backend,
                native_backend=native_backend,
            )
            records.append(
                {
                    "coordinate": coordinate.as_dict(),
                    "transport": transport,
                }
            )
            counters.update(one_counters)
        candidate = {
            "expanded": {
                "backendDescriptorSha256": backend_descriptor_sha256,
                "backendMode": arguments.backend,
                "numericReplayContext": numeric_replay_descriptor,
            },
            "records": records,
            "schema": CANDIDATE_SCHEMA,
            "version": 1,
        }
        candidate_payload = canonical_json_bytes(candidate)
        cpu_seconds = time.process_time() - cpu_start
        wall_seconds = time.perf_counter() - wall_start
        comparison = compare_corpus_documents(
            golden_before,
            candidate,
            require_byte_exact=True,
        )
        if comparison["qualified"] is not True or comparison["byteExact"] is not True:
            raise CpuAdapterBenchmarkError(
                f"repeat {repeat_index} differs from authenticated cache records"
            )
        comparison_payload = canonical_json_bytes(comparison)
        candidate_sha = hashlib.sha256(candidate_payload).hexdigest()
        comparison_sha = hashlib.sha256(comparison_payload).hexdigest()
        candidate_sha256_values.append(candidate_sha)
        comparison_sha256_values.append(comparison_sha)
        repeat_fates = Counter(
            item["transport"]["transport"]["fate"] for item in records
        )
        if repeat_index == 0:
            fate_distribution = repeat_fates
        elif repeat_fates != fate_distribution:
            raise CpuAdapterBenchmarkError("repeat fate distribution changed")
        repeat_reports.append(
            {
                "byteExact": True,
                "candidateSha256": candidate_sha,
                "comparisonReportSha256": comparison_sha,
                "cpuSeconds": cpu_seconds,
                "directionsPerSecond": len(ordinals) / wall_seconds,
                "qualification": True,
                "repeatIndex": repeat_index,
                "transportCounters": dict(sorted(counters.items())),
                "wallSeconds": wall_seconds,
            }
        )
        final_candidate = candidate
    if len(set(candidate_sha256_values)) != 1:
        raise CpuAdapterBenchmarkError("repeated canonical transport bytes changed")
    if len(set(comparison_sha256_values)) != 1:
        raise CpuAdapterBenchmarkError("repeated comparison evidence changed")
    if type(native_backend) is StrictKerrCpuBackend:
        native_backend.revalidate()

    closure_after = _cached._source_closure_manifest()
    closure_before_tree = tuple(
        (item.logical_path, item.byte_length, item.sha256, item.binding_sha256)
        for item in closure_before
    )
    closure_after_tree = tuple(
        (item.logical_path, item.byte_length, item.sha256, item.binding_sha256)
        for item in closure_after
    )
    if closure_after_tree != closure_before_tree:
        raise CpuAdapterBenchmarkError("Python source/runtime closure changed")
    golden_after = extract_golden_corpus(
        job_directory,
        selected_ordinals=ordinals,
        trust=FROZEN_NESTED16_TRUST,
    )
    if golden_after != golden_before:
        raise CpuAdapterBenchmarkError("copied cache evidence changed during benchmark")
    if _stable_regular_artifact(dylib, "dylib") != dylib_artifact:
        raise CpuAdapterBenchmarkError("dylib changed during benchmark")
    if _stable_regular_artifact(
        Path(__file__).absolute(), "benchmark tool"
    ) != tool_artifact:
        raise CpuAdapterBenchmarkError("benchmark tool changed during benchmark")

    wall_values = sorted(item["wallSeconds"] for item in repeat_reports)
    cpu_values = sorted(item["cpuSeconds"] for item in repeat_reports)
    middle = len(wall_values) // 2
    median_wall = (
        wall_values[middle]
        if len(wall_values) % 2
        else 0.5 * (wall_values[middle - 1] + wall_values[middle])
    )
    median_cpu = (
        cpu_values[middle]
        if len(cpu_values) % 2
        else 0.5 * (cpu_values[middle - 1] + cpu_values[middle])
    )
    assert final_candidate is not None
    report_without_hash = {
        "artifacts": {
            "backendDescriptorSha256": backend_descriptor_sha256,
            "dylib": dylib_artifact,
            "tool": tool_artifact,
        },
        "authentication": {
            "authenticatedDirectionCount": golden_before["evidence"][
                "authenticatedDirectionCount"
            ],
            "authenticatedDirectionStreamSha256": golden_before["evidence"][
                "authenticatedDirectionStreamSha256"
            ],
            "authenticatedPayloadSetSha256": golden_before["evidence"][
                "authenticatedPayloadSetSha256"
            ],
            "authenticatedTaskCount": golden_before["evidence"][
                "authenticatedTaskCount"
            ],
            "cacheJobKey": FROZEN_NESTED16_TRUST.cache_job_key,
            "cacheReadOnlyStableBeforeAfter": True,
            "goldenSelectionSha256": golden_sha256,
            "jobDocumentSha256": FROZEN_NESTED16_TRUST.job_document_sha256,
            "scientificJobKey": FROZEN_NESTED16_TRUST.scientific_job_key,
            "scientificPlanSha256": FROZEN_NESTED16_TRUST.scientific_plan_sha256,
            "scope": (
                "read-only authenticated copied-cache evidence; these frozen keys "
                "are not the temporary numeric-replay context identity"
            ),
        },
        "backend": {
            "descriptor": backend_descriptor,
            "mode": arguments.backend,
            "nativeNeverFallsBack": no_fallback,
        },
        "classification": (
            "non-production numeric-replay direct-evaluator authenticated-cache "
            "byte-exact throughput benchmark"
        ),
        "construction": {
            "backendWallSeconds": backend_wall,
            "cacheAuthenticationWallSeconds": authentication_wall,
            "contextWallSeconds": context_wall,
        },
        "fateDistribution": dict(sorted(fate_distribution.items())),
        "peakRssBytes": _peak_rss_bytes(),
        "numericReplayContext": numeric_replay_descriptor,
        "qualification": {
            "allRepeatsByteExact": True,
            "allRepeatsQualified": True,
            "cacheRunnerInvoked": False,
            "cacheWrites": False,
            "candidateSha256Stable": True,
            "dylibStable": True,
            "sourceRuntimeClosureStable": True,
            "toolSourceStable": True,
            "numericReplayContextProductionQualified": False,
            "temporaryScientificOrJobKeyReported": False,
        },
        "repeats": repeat_reports,
        "schema": SCHEMA,
        "selection": {
            "directionCount": len(ordinals),
            "ordinals": list(ordinals),
            "repeatCount": arguments.repeat,
            "totalDirectionEvaluations": len(ordinals) * arguments.repeat,
        },
        "sourceRuntimeClosureManifestSha256": (
            _cached._source_closure_manifest_sha256(closure_before)
        ),
        "sourceRuntimeClosureScope": (
            "current numeric-replay checkout only; not the frozen d828 source identity"
        ),
        "summary": {
            "medianCpuSeconds": median_cpu,
            "medianDirectionsPerSecond": len(ordinals) / median_wall,
            "medianWallSeconds": median_wall,
        },
        "version": VERSION,
    }
    basis_sha256 = hashlib.sha256(
        canonical_json_bytes(report_without_hash)
    ).hexdigest()
    report = {
        **report_without_hash,
        "reportBasisSha256": basis_sha256,
    }
    payload = canonical_json_bytes(report)
    output_sha256 = _safe_write_new(
        arguments.output,
        payload,
        job_directory,
    )
    return report, output_sha256


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        report, output_sha256 = run_benchmark(arguments)
    except (CpuAdapterBenchmarkError, OSError, TypeError, ValueError) as error:
        print(f"native CPU adapter benchmark failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "directionCount": report["selection"]["directionCount"],
                "output": str(arguments.output),
                "outputSha256": output_sha256,
                "qualified": report["qualification"]["allRepeatsQualified"],
                "schema": report["schema"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
