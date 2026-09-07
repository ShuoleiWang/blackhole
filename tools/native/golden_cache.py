"""Read-only golden-cache extraction and native-result comparison.

This module has a deliberately narrow trust boundary:

* it imports only the Python standard library;
* it opens every cache path component with ``O_NOFOLLOW|O_DIRECTORY``;
* it authenticates the frozen job document against an external SHA-256 trust
  anchor before trusting task names;
* it validates every present payload/receipt pair, canonical JSON schema, and
  canonical record coordinate/order itself; and
* it accepts an incomplete cache as evidence only.  It never imports or calls
  the production cache runner, so a partial cache can never start ray work.

The checked-in corpus is intentionally narrow.  A native candidate uses the
same ordered record shape as the cache: ``{coordinate, transport}``.  Optional
expanded implementation evidence must live under a separate top-level
``expanded`` leaf and is never required by the narrow comparison.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Any, Final, Iterator, Mapping, Sequence
import uuid


CORPUS_SCHEMA: Final = "blackhole.native-kerr-golden-corpus/v1"
CANDIDATE_SCHEMA: Final = "blackhole.native-kerr-golden-candidate/v1"
COMPARISON_SCHEMA: Final = "blackhole.native-kerr-golden-comparison/v1"
CORPUS_VERSION: Final = 1
# Updated whenever the checked-in, externally reviewed corpus bytes change.
FROZEN_NESTED16_CORPUS_SHA256: Final = (
    "15758851df05d4a6126eee00995ea99a5d76261bbb71266015b023b83f191606"
)

RECEIPT_SCHEMA: Final = "blackhole.offline-task-receipt/v1"
TASK_PAYLOAD_SCHEMA: Final = (
    "blackhole.returning-radiation-kernel-direction-task/v1"
)
TASK_PLAN_SCHEMA: Final = "blackhole.returning-radiation-kernel-task-plan/v1"
FORWARD_TRANSPORT_SCHEMA: Final = (
    "blackhole.kerr-forward-returning-direction-transport/v1"
)

DEFAULT_SELECTED_ORDINALS: Final = (369, 433, 726, 732, 736, 31263, 37331)

_MAXIMUM_JOB_BYTES: Final = 32 * 1024 * 1024
_MAXIMUM_TASK_PAYLOAD_BYTES: Final = 256 * 1024
_MAXIMUM_RECEIPT_BYTES: Final = 16 * 1024
_MAXIMUM_DOCUMENT_BYTES: Final = 16 * 1024 * 1024
_MAXIMUM_JSON_DEPTH: Final = 32
_MAXIMUM_TASK_COUNT: Final = 65_536
_MAXIMUM_RECORD_COUNT: Final = 2_000_000
_SHA256_HEX_LENGTH: Final = 64

_PASS_NAMES: Final = (
    "full",
    "half-rho",
    "half-mu",
    "half-psi",
    "phase-shifted",
)
_FACES: Final = ("upper", "lower")
_NON_RETURN_FATES: Final = ("captured", "escaped", "plunge-sink")
_RETURN_FATES: Final = ("return-upper", "return-lower")
_ALL_FATES: Final = _NON_RETURN_FATES + _RETURN_FATES


class GoldenCacheError(RuntimeError):
    """Raised when golden evidence is incomplete, stale, or unauthenticated."""


@dataclass(frozen=True, slots=True)
class CacheTrust:
    """External trust anchors for one frozen cache snapshot."""

    cache_job_key: str
    scientific_job_key: str
    scientific_plan_sha256: str
    job_document_sha256: str
    authenticated_task_count: int | None = None
    authenticated_direction_count: int | None = None
    authenticated_payload_set_sha256: str | None = None
    authenticated_direction_stream_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "cache_job_key",
            "scientific_job_key",
            "scientific_plan_sha256",
            "job_document_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in ("authenticated_task_count", "authenticated_direction_count"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise GoldenCacheError(f"{name} must be an exact positive int")
        for name in (
            "authenticated_payload_set_sha256",
            "authenticated_direction_stream_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(value, name)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's canonical JSON spelling."""

    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise GoldenCacheError("document cannot be represented as canonical JSON") from error


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or len(value) != _SHA256_HEX_LENGTH:
        raise GoldenCacheError(f"{label} is not a lowercase SHA-256 digest")
    if value.lower() != value:
        raise GoldenCacheError(f"{label} is not a lowercase SHA-256 digest")
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise GoldenCacheError(f"{label} is not a lowercase SHA-256 digest") from error
    if len(raw) != 32:
        raise GoldenCacheError(f"{label} is not a lowercase SHA-256 digest")
    return value


FROZEN_NESTED16_TRUST: Final = CacheTrust(
    cache_job_key=(
        "d8286face863a0755d70c20c1638c6a5dc4d9d9037c0227626fb805bb4d0f751"
    ),
    scientific_job_key=(
        "19c8c3ebd465a4f66d6e893935dc18134ecd47d8842b2e77809823d4148ec980"
    ),
    scientific_plan_sha256=(
        "a04fb6a9180a69e8cab8a6fb96bb615715408982ba5f25ad0ac8844cc8f6c0f7"
    ),
    job_document_sha256=(
        "d66a1c7f5de21af2492747fc5fa7cf2835c7977d9abce659454cdf4469898a1f"
    ),
    authenticated_task_count=596,
    authenticated_direction_count=38_144,
    authenticated_payload_set_sha256=(
        "010f6a21dcd774eba15afa1760d8a5df76415b9260c231d3cea715a58c8f8d91"
    ),
    authenticated_direction_stream_sha256=(
        "0bf7548ea6f93cd1986a04adca2e42090db5ed4848c51b3f4289348db472bf4e"
    ),
)


def _require_exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise GoldenCacheError(f"{label} must be an exact int >= {minimum}")
    return value


def _require_finite_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise GoldenCacheError(f"{label} must be an exact finite float")
    return value


def _validate_json_nesting(payload: bytes, label: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for byte in payload:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):
            depth += 1
            if depth > _MAXIMUM_JSON_DEPTH:
                raise GoldenCacheError(f"{label} exceeds the hard JSON depth limit")
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth < 0:
                raise GoldenCacheError(f"{label} has invalid JSON nesting")


def _reject_non_finite_tree(value: Any, label: str) -> None:
    if type(value) is float:
        if not math.isfinite(value):
            raise GoldenCacheError(f"{label} contains a non-finite float")
        return
    if type(value) is dict:
        for key, item in value.items():
            _reject_non_finite_tree(item, f"{label}.{key}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _reject_non_finite_tree(item, f"{label}[{index}]")


def strict_json_bytes(payload: bytes, label: str, maximum_bytes: int) -> Any:
    """Decode bounded, duplicate-free, canonical, finite JSON bytes."""

    if type(payload) is not bytes or len(payload) > maximum_bytes:
        raise GoldenCacheError(f"{label} exceeds its hard byte limit")
    _validate_json_nesting(payload, label)

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise GoldenCacheError(f"{label} is not strict JSON") from error
    _reject_non_finite_tree(value, label)
    if canonical_json_bytes(value) != payload:
        raise GoldenCacheError(f"{label} is not canonical JSON")
    return value


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if type(nofollow) is not int or type(directory) is not int:
        raise GoldenCacheError("platform lacks O_NOFOLLOW/O_DIRECTORY")
    return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


def _stable_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_absolute_path(path: Path, label: str) -> Path:
    if type(path) is not type(Path()) or not path.is_absolute():
        raise GoldenCacheError(f"{label} must be an exact absolute platform Path")
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute == absolute.parent:
        raise GoldenCacheError(f"{label} must not be the filesystem root")
    return absolute


def _open_absolute_directory(path: Path, label: str) -> tuple[Path, int]:
    absolute = _canonical_absolute_path(path, label)
    descriptor = os.open(os.sep, _directory_flags())
    try:
        for component in absolute.parts[1:]:
            if component in ("", ".", "..") or os.sep in component:
                raise GoldenCacheError(f"{label} has a non-canonical component")
            following = os.open(component, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        return absolute, descriptor
    except OSError as error:
        os.close(descriptor)
        raise GoldenCacheError(
            f"{label} contains a missing, symlink, or non-directory component"
        ) from error
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_file_at(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    if type(name) is not str or not name or os.sep in name or name in (".", ".."):
        raise GoldenCacheError(f"{label} has an invalid filename")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = -1
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise GoldenCacheError(
                f"{label} is not regular or exceeds its hard byte limit"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise GoldenCacheError(f"{label} changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise GoldenCacheError(f"{label} grew while being read")
        after = os.fstat(descriptor)
        if _stable_stat_identity(before) != _stable_stat_identity(after):
            raise GoldenCacheError(f"{label} changed while being read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise GoldenCacheError(f"{label} stable-size check failed")
        return payload
    except OSError as error:
        raise GoldenCacheError(
            f"{label} cannot be opened without following symlinks"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _require_regular_entry(directory_fd: int, name: str, label: str) -> None:
    try:
        snapshot = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise GoldenCacheError(f"{label} cannot be inspected") from error
    if stat.S_ISLNK(snapshot.st_mode) or not stat.S_ISREG(snapshot.st_mode):
        raise GoldenCacheError(f"{label} must be a regular non-symlink file")


@dataclass(slots=True)
class _SecureJobDirectory:
    path: Path
    job_fd: int
    task_fd: int


@contextmanager
def _secure_job_directory(path: Path, trust: CacheTrust) -> Iterator[_SecureJobDirectory]:
    absolute, job_fd = _open_absolute_directory(path, "cache job directory")
    task_fd = -1
    try:
        if absolute.name != trust.cache_job_key:
            raise GoldenCacheError("cache job directory name differs from trust anchor")
        task_fd = os.open("tasks", _directory_flags(), dir_fd=job_fd)
        yield _SecureJobDirectory(absolute, job_fd, task_fd)
    except OSError as error:
        raise GoldenCacheError("cache tasks path is missing, a symlink, or not a directory") from error
    finally:
        if task_fd >= 0:
            os.close(task_fd)
        os.close(job_fd)


def _stable_directory_names(directory_fd: int, label: str) -> tuple[str, ...]:
    before = os.fstat(directory_fd)
    try:
        names = os.listdir(directory_fd)
    except OSError as error:
        raise GoldenCacheError(f"{label} cannot be listed") from error
    after = os.fstat(directory_fd)
    if _stable_stat_identity(before) != _stable_stat_identity(after):
        raise GoldenCacheError(f"{label} changed while being listed")
    if any(type(name) is not str or not name for name in names):
        raise GoldenCacheError(f"{label} contains a non-string or empty name")
    return tuple(sorted(names))


def _task_stem(task: Mapping[str, Any]) -> str:
    return (
        f"t{task['sampleIndex']:06d}-y{task['y']:06d}-x{task['x']:06d}-"
        f"w{task['width']:06d}-h{task['height']:06d}"
    )


def _validated_task_document(value: Any, label: str) -> dict[str, int]:
    keys = {"height", "sampleIndex", "width", "x", "y"}
    if type(value) is not dict or set(value) != keys:
        raise GoldenCacheError(f"{label} has a non-exact schema")
    result: dict[str, int] = {}
    for key in sorted(keys):
        result[key] = _require_exact_int(value[key], f"{label}.{key}")
    if result["height"] != 1 or result["width"] < 1:
        raise GoldenCacheError(f"{label} has an invalid task extent")
    return result


def _validated_plan(value: Any) -> tuple[dict[str, Any], ...]:
    expected_keys = {
        "annulusCount",
        "canonicalOrder",
        "directionCount",
        "faces",
        "formulation",
        "passes",
        "schema",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise GoldenCacheError("scientific plan has a non-exact schema")
    if (
        value["schema"] != TASK_PLAN_SCHEMA
        or value["canonicalOrder"] != "pass/face/annulus/rho/mu/psi"
        or value["formulation"] != "forward"
        or value["faces"] != list(_FACES)
    ):
        raise GoldenCacheError("scientific plan identity is unsupported")
    annulus_count = _require_exact_int(
        value["annulusCount"], "scientific plan annulusCount", minimum=1
    )
    direction_count = _require_exact_int(
        value["directionCount"], "scientific plan directionCount", minimum=1
    )
    raw_passes = value["passes"]
    if type(raw_passes) is not list or len(raw_passes) != len(_PASS_NAMES):
        raise GoldenCacheError("scientific plan must contain the exact five passes")
    passes: list[dict[str, Any]] = []
    total = 0
    pass_keys = {"muOrder", "name", "phaseCells", "psiCount", "rhoOrder"}
    for index, (raw, expected_name) in enumerate(zip(raw_passes, _PASS_NAMES)):
        if type(raw) is not dict or set(raw) != pass_keys:
            raise GoldenCacheError(f"scientific plan pass {index} has a non-exact schema")
        if raw["name"] != expected_name:
            raise GoldenCacheError(f"scientific plan pass {index} has the wrong name")
        mu = _require_exact_int(raw["muOrder"], f"pass {index} muOrder", minimum=1)
        psi = _require_exact_int(raw["psiCount"], f"pass {index} psiCount", minimum=1)
        rho = _require_exact_int(raw["rhoOrder"], f"pass {index} rhoOrder", minimum=1)
        phase = _require_finite_float(raw["phaseCells"], f"pass {index} phaseCells")
        passes.append(
            {
                "muOrder": mu,
                "name": expected_name,
                "phaseCells": phase,
                "psiCount": psi,
                "rhoOrder": rho,
            }
        )
        total += 2 * annulus_count * rho * mu * psi
    if total != direction_count or total > _MAXIMUM_RECORD_COUNT:
        raise GoldenCacheError("scientific plan direction accounting is inconsistent")
    return tuple(passes)


def _expected_tasks(
    plan: Mapping[str, Any],
    passes: Sequence[Mapping[str, Any]],
    directions_per_task: int,
) -> tuple[dict[str, int], ...]:
    annulus_count = plan["annulusCount"]
    result: list[dict[str, int]] = []
    for pass_index, grid_pass in enumerate(passes):
        angular_count = grid_pass["muOrder"] * grid_pass["psiCount"]
        for face_index in range(2):
            for annulus_index in range(annulus_count):
                y = face_index * annulus_count + annulus_index
                for rho_index in range(grid_pass["rhoOrder"]):
                    radial_start = rho_index * angular_count
                    for angular_start in range(0, angular_count, directions_per_task):
                        width = min(directions_per_task, angular_count - angular_start)
                        result.append(
                            {
                                "height": 1,
                                "sampleIndex": pass_index,
                                "width": width,
                                "x": radial_start + angular_start,
                                "y": y,
                            }
                        )
    if len(result) > _MAXIMUM_TASK_COUNT:
        raise GoldenCacheError("scientific plan exceeds the task-count bound")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _JobDefinition:
    job_document_sha256: str
    plan: Mapping[str, Any]
    passes: tuple[Mapping[str, Any], ...]
    tasks: tuple[Mapping[str, int], ...]
    directions_per_task: int
    annulus_edges_over_mass: tuple[float, ...]
    scientific_job_key: str
    scientific_plan_sha256: str


def _validated_job_document(payload: bytes, trust: CacheTrust) -> _JobDefinition:
    digest = hashlib.sha256(payload).hexdigest()
    if digest != trust.job_document_sha256:
        raise GoldenCacheError("cache job document differs from its frozen SHA-256")
    document = strict_json_bytes(payload, "cache job document", _MAXIMUM_JOB_BYTES)
    if type(document) is not dict or set(document) != {"jobKey", "spec"}:
        raise GoldenCacheError("cache job document has a non-exact schema")
    if document["jobKey"] != trust.cache_job_key:
        raise GoldenCacheError("cache job key differs from trust anchor")
    spec = document["spec"]
    spec_keys = {
        "algorithmVersion",
        "inputs",
        "parameters",
        "producer",
        "producerSourceHashes",
        "recordBytes",
        "schema",
        "tasks",
    }
    if type(spec) is not dict or set(spec) != spec_keys:
        raise GoldenCacheError("cache job specification has a non-exact schema")
    if (
        spec["schema"] != "blackhole.offline-job/v1"
        or spec["producer"] != "offline.kerr-returning-radiation-direction-cache"
        or spec["algorithmVersion"] != "1.1.0"
        or type(spec["recordBytes"]) is not int
        or spec["recordBytes"] != 1
    ):
        raise GoldenCacheError("cache job producer identity is unsupported")
    if hashlib.sha256(canonical_json_bytes(spec)).hexdigest() != trust.cache_job_key:
        raise GoldenCacheError("cache job key does not authenticate its specification")
    parameters = spec["parameters"]
    parameter_keys = {
        "cacheLayout",
        "payloadBudget",
        "scientificDocument",
        "scientificJobKey",
        "scientificPlanSha256",
        "scientificStatus",
    }
    if type(parameters) is not dict or set(parameters) != parameter_keys:
        raise GoldenCacheError("cache parameters have a non-exact schema")
    if (
        parameters["scientificJobKey"] != trust.scientific_job_key
        or parameters["scientificPlanSha256"] != trust.scientific_plan_sha256
    ):
        raise GoldenCacheError("cache scientific identity differs from trust anchor")
    scientific_document = parameters["scientificDocument"]
    if type(scientific_document) is not dict or "plan" not in scientific_document:
        raise GoldenCacheError("scientific document lacks its plan")
    if (
        hashlib.sha256(canonical_json_bytes(scientific_document)).hexdigest()
        != trust.scientific_job_key
    ):
        raise GoldenCacheError("scientific job key does not authenticate its document")
    plan = scientific_document["plan"]
    passes = _validated_plan(plan)
    if hashlib.sha256(canonical_json_bytes(plan)).hexdigest() != trust.scientific_plan_sha256:
        raise GoldenCacheError("scientific plan SHA-256 does not authenticate its plan")
    identity = scientific_document.get("scientificIdentity")
    if type(identity) is not dict or "annulusEdgesOverMass" not in identity:
        raise GoldenCacheError("scientific identity lacks annulus edges")
    raw_edges = identity["annulusEdgesOverMass"]
    if type(raw_edges) is not list or len(raw_edges) != plan["annulusCount"] + 1:
        raise GoldenCacheError("scientific annulus edges have an invalid extent")
    annulus_edges = tuple(
        _require_finite_float(value, f"scientific annulus edge {index}")
        for index, value in enumerate(raw_edges)
    )
    if any(right <= left for left, right in zip(annulus_edges, annulus_edges[1:])):
        raise GoldenCacheError("scientific annulus edges are not strictly increasing")
    layout = parameters["cacheLayout"]
    if type(layout) is not dict or set(layout) != {
        "chunkBoundary",
        "directionsPerTaskMaximum",
        "taskCount",
    }:
        raise GoldenCacheError("cache layout has a non-exact schema")
    if layout["chunkBoundary"] != "never crosses pass/face/annulus/rho":
        raise GoldenCacheError("cache layout has an unsupported chunk boundary")
    directions_per_task = _require_exact_int(
        layout["directionsPerTaskMaximum"],
        "directionsPerTaskMaximum",
        minimum=1,
    )
    raw_tasks = spec["tasks"]
    if type(raw_tasks) is not list:
        raise GoldenCacheError("cache tasks must be an exact list")
    tasks = tuple(
        _validated_task_document(item, f"cache task {index}")
        for index, item in enumerate(raw_tasks)
    )
    expected_tasks = _expected_tasks(plan, passes, directions_per_task)
    if tasks != expected_tasks:
        raise GoldenCacheError("cache task list is incomplete or out of canonical order")
    if layout["taskCount"] != len(tasks):
        raise GoldenCacheError("cache layout task count differs from its task list")
    return _JobDefinition(
        digest,
        plan,
        passes,
        tasks,
        directions_per_task,
        annulus_edges,
        trust.scientific_job_key,
        trust.scientific_plan_sha256,
    )


def _expected_coordinates(
    definition: _JobDefinition,
    task: Mapping[str, int],
) -> tuple[dict[str, Any], ...]:
    pass_index = task["sampleIndex"]
    if pass_index >= len(definition.passes):
        raise GoldenCacheError("task pass index exceeds scientific plan")
    grid_pass = definition.passes[pass_index]
    annulus_count = definition.plan["annulusCount"]
    if task["y"] >= 2 * annulus_count:
        raise GoldenCacheError("task y coordinate exceeds scientific plan")
    angular_count = grid_pass["muOrder"] * grid_pass["psiCount"]
    rho_index, angular_start = divmod(task["x"], angular_count)
    if rho_index >= grid_pass["rhoOrder"]:
        raise GoldenCacheError("task rho coordinate exceeds scientific plan")
    if angular_start + task["width"] > angular_count:
        raise GoldenCacheError("task crosses a canonical rho boundary")
    expected_width = min(
        definition.directions_per_task, angular_count - angular_start
    )
    if (
        angular_start % definition.directions_per_task
        or task["width"] != expected_width
    ):
        raise GoldenCacheError("task is not a canonical cache-layout chunk")
    face_index, annulus_index = divmod(task["y"], annulus_count)
    pass_offset = 2 * annulus_count * sum(
        item["rhoOrder"] * item["muOrder"] * item["psiCount"]
        for item in definition.passes[:pass_index]
    )
    local_prefix = (
        (face_index * annulus_count + annulus_index)
        * grid_pass["rhoOrder"]
        * angular_count
        + rho_index * angular_count
    )
    result: list[dict[str, Any]] = []
    for angular_index in range(angular_start, angular_start + task["width"]):
        mu_index, psi_index = divmod(angular_index, grid_pass["psiCount"])
        result.append(
            {
                "annulusIndex": annulus_index,
                "face": _FACES[face_index],
                "faceIndex": face_index,
                "muIndex": mu_index,
                "ordinal": pass_offset + local_prefix + angular_index,
                "passIndex": pass_index,
                "passName": grid_pass["name"],
                "psiIndex": psi_index,
                "rhoIndex": rho_index,
            }
        )
    return tuple(result)


def _validated_coordinate(value: Any, label: str) -> dict[str, Any]:
    keys = {
        "annulusIndex",
        "face",
        "faceIndex",
        "muIndex",
        "ordinal",
        "passIndex",
        "passName",
        "psiIndex",
        "rhoIndex",
    }
    if type(value) is not dict or set(value) != keys:
        raise GoldenCacheError(f"{label} has a non-exact schema")
    result = dict(value)
    for key in (
        "annulusIndex",
        "faceIndex",
        "muIndex",
        "ordinal",
        "passIndex",
        "psiIndex",
        "rhoIndex",
    ):
        _require_exact_int(result[key], f"{label}.{key}")
    if type(result["face"]) is not str or result["face"] not in _FACES:
        raise GoldenCacheError(f"{label}.face is unsupported")
    if type(result["passName"]) is not str or result["passName"] not in _PASS_NAMES:
        raise GoldenCacheError(f"{label}.passName is unsupported")
    return result


def _validated_forward_transport(
    value: Any,
    coordinate: Mapping[str, Any],
    label: str,
    *,
    annulus_edges_over_mass: Sequence[float] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "formulation",
        "sample",
        "schema",
        "transport",
    }:
        raise GoldenCacheError(f"{label} has a non-exact evaluator schema")
    if value["formulation"] != "forward" or value["schema"] != FORWARD_TRANSPORT_SCHEMA:
        raise GoldenCacheError(f"{label} has an unsupported evaluator identity")
    sample = value["sample"]
    sample_keys = {
        "emissionAngleCosine",
        "muIndex",
        "normalizedEmittedFluxWeight",
        "passIndex",
        "passName",
        "psiIndex",
        "rhoIndex",
        "sourceAnnulusIndex",
        "sourceFace",
        "sourceRadiusOverMass",
        "tangentAzimuthRad",
    }
    if type(sample) is not dict or set(sample) != sample_keys:
        raise GoldenCacheError(f"{label}.sample has a non-exact schema")
    exact_matches = {
        "muIndex": "muIndex",
        "passIndex": "passIndex",
        "passName": "passName",
        "psiIndex": "psiIndex",
        "rhoIndex": "rhoIndex",
        "sourceAnnulusIndex": "annulusIndex",
        "sourceFace": "face",
    }
    for sample_key, coordinate_key in exact_matches.items():
        if type(sample[sample_key]) is not type(coordinate[coordinate_key]) or (
            sample[sample_key] != coordinate[coordinate_key]
        ):
            raise GoldenCacheError(
                f"{label}.sample.{sample_key} differs from its coordinate"
            )
    for key in (
        "emissionAngleCosine",
        "normalizedEmittedFluxWeight",
        "sourceRadiusOverMass",
        "tangentAzimuthRad",
    ):
        _require_finite_float(sample[key], f"{label}.sample.{key}")
    narrow = value["transport"]
    narrow_keys = {
        "coarseReceiverAnnulusIndex",
        "coarseReceiverFace",
        "coarseReceiverRadiusOverMass",
        "fate",
        "frequencyRatio",
        "g2",
        "primitiveDescriptorSha256",
        "receiverAnnulusIndex",
        "receiverFace",
        "receiverRadiusOverMass",
    }
    if type(narrow) is not dict or set(narrow) != narrow_keys:
        raise GoldenCacheError(f"{label}.transport has a non-exact schema")
    fate = narrow["fate"]
    if type(fate) is not str or fate not in _ALL_FATES:
        raise GoldenCacheError(f"{label}.transport.fate is unsupported")
    _require_sha256(
        narrow["primitiveDescriptorSha256"],
        f"{label}.transport.primitiveDescriptorSha256",
    )
    _require_finite_float(narrow["g2"], f"{label}.transport.g2")
    receiver_keys = (
        "coarseReceiverAnnulusIndex",
        "coarseReceiverFace",
        "coarseReceiverRadiusOverMass",
        "frequencyRatio",
        "receiverAnnulusIndex",
        "receiverFace",
        "receiverRadiusOverMass",
    )
    if fate in _RETURN_FATES:
        for key in ("coarseReceiverAnnulusIndex", "receiverAnnulusIndex"):
            _require_exact_int(narrow[key], f"{label}.transport.{key}")
        for key in ("coarseReceiverFace", "receiverFace"):
            expected_face = fate.removeprefix("return-")
            if type(narrow[key]) is not str or narrow[key] != expected_face:
                raise GoldenCacheError(f"{label}.transport.{key} disagrees with fate")
        for key in (
            "coarseReceiverRadiusOverMass",
            "frequencyRatio",
            "receiverRadiusOverMass",
        ):
            if _require_finite_float(narrow[key], f"{label}.transport.{key}") <= 0.0:
                raise GoldenCacheError(f"{label}.transport.{key} must be positive")
        if narrow["g2"] <= 0.0:
            raise GoldenCacheError(f"{label}.transport.g2 must be positive for return")
        if narrow["g2"].hex() != (
            narrow["frequencyRatio"] * narrow["frequencyRatio"]
        ).hex():
            raise GoldenCacheError(
                f"{label}.transport.g2 is not the exact squared frequency ratio"
            )
        if narrow["receiverAnnulusIndex"] != narrow["coarseReceiverAnnulusIndex"]:
            raise GoldenCacheError(
                f"{label}.transport fine/coarse receiver bins disagree"
            )
        if annulus_edges_over_mass is not None:
            edges = tuple(annulus_edges_over_mass)
            if len(edges) < 2 or any(
                type(item) is not float or not math.isfinite(item) for item in edges
            ) or any(right <= left for left, right in zip(edges, edges[1:])):
                raise GoldenCacheError(f"{label} received invalid annulus edges")

            def receiver_bin(radius: float) -> int:
                index = bisect_right(edges, radius) - 1
                if index == len(edges) - 1 and radius.hex() == edges[-1].hex():
                    index -= 1
                if index < 0 or index >= len(edges) - 1:
                    raise GoldenCacheError(
                        f"{label}.transport receiver radius lies outside annulus edges"
                    )
                return index

            expected_fine = receiver_bin(narrow["receiverRadiusOverMass"])
            expected_coarse = receiver_bin(narrow["coarseReceiverRadiusOverMass"])
            if (
                narrow["receiverAnnulusIndex"] != expected_fine
                or narrow["coarseReceiverAnnulusIndex"] != expected_coarse
            ):
                raise GoldenCacheError(
                    f"{label}.transport receiver bin differs from exact annulus edges"
                )
    else:
        if any(narrow[key] is not None for key in receiver_keys):
            raise GoldenCacheError(f"{label}.transport has receiver data for non-return fate")
        if narrow["g2"].hex() != 0.0.hex():
            raise GoldenCacheError(f"{label}.transport.g2 must be +0.0 for non-return fate")
    return dict(value)


def _validated_receipt(
    payload: bytes,
    receipt_payload: bytes,
    *,
    task: Mapping[str, int],
    payload_name: str,
    trust: CacheTrust,
) -> tuple[str, int]:
    receipt = strict_json_bytes(
        receipt_payload, "task receipt", _MAXIMUM_RECEIPT_BYTES
    )
    receipt_keys = {
        "byteLength",
        "jobKey",
        "payload",
        "recordCount",
        "schema",
        "sha256",
        "task",
    }
    if type(receipt) is not dict or set(receipt) != receipt_keys:
        raise GoldenCacheError("task receipt has a non-exact schema")
    receipt_task = _validated_task_document(receipt["task"], "task receipt task")
    digest = hashlib.sha256(payload).hexdigest()
    if (
        receipt["schema"] != RECEIPT_SCHEMA
        or receipt["jobKey"] != trust.cache_job_key
        or receipt["payload"] != payload_name
        or receipt_task != task
        or type(receipt["byteLength"]) is not int
        or type(receipt["recordCount"]) is not int
        or receipt["byteLength"] != len(payload)
        or receipt["recordCount"] != len(payload)
        or receipt["sha256"] != digest
    ):
        raise GoldenCacheError("task receipt does not authenticate its exact payload")
    _require_sha256(receipt["sha256"], "task receipt sha256")
    return digest, len(receipt_payload)


def _validated_payload_records(
    payload: bytes,
    *,
    definition: _JobDefinition,
    task: Mapping[str, int],
    trust: CacheTrust,
) -> tuple[dict[str, Any], ...]:
    document = strict_json_bytes(
        payload, "direction task payload", _MAXIMUM_TASK_PAYLOAD_BYTES
    )
    keys = {
        "cacheJobKey",
        "records",
        "schema",
        "scientificJobKey",
        "scientificPlanSha256",
        "task",
    }
    if type(document) is not dict or set(document) != keys:
        raise GoldenCacheError("direction task payload has a non-exact schema")
    payload_task = _validated_task_document(
        document["task"], "direction task payload task"
    )
    if (
        document["schema"] != TASK_PAYLOAD_SCHEMA
        or document["cacheJobKey"] != trust.cache_job_key
        or document["scientificJobKey"] != trust.scientific_job_key
        or document["scientificPlanSha256"] != trust.scientific_plan_sha256
        or payload_task != task
        or type(document["records"]) is not list
    ):
        raise GoldenCacheError("direction task payload has stale scientific identity")
    expected = _expected_coordinates(definition, task)
    if len(document["records"]) != len(expected):
        raise GoldenCacheError("direction task payload record count is inconsistent")
    result: list[dict[str, Any]] = []
    for index, (raw_record, expected_coordinate) in enumerate(
        zip(document["records"], expected)
    ):
        if type(raw_record) is not dict or set(raw_record) != {"coordinate", "transport"}:
            raise GoldenCacheError(
                f"direction task record {index} has a non-exact schema"
            )
        coordinate = _validated_coordinate(
            raw_record["coordinate"], f"direction task record {index}.coordinate"
        )
        if coordinate != expected_coordinate:
            raise GoldenCacheError(
                f"direction task record {index} is out of canonical order"
            )
        transport = _validated_forward_transport(
            raw_record["transport"],
            coordinate,
            f"direction task record {index}.transport",
            annulus_edges_over_mass=definition.annulus_edges_over_mass,
        )
        result.append({"coordinate": coordinate, "transport": transport})
    return tuple(result)


def _payload_inventory_sha256(items: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(items))).hexdigest()


def extract_golden_corpus(
    job_directory: Path,
    *,
    selected_ordinals: Sequence[int] = DEFAULT_SELECTED_ORDINALS,
    trust: CacheTrust = FROZEN_NESTED16_TRUST,
) -> dict[str, Any]:
    """Authenticate a frozen partial cache and return a deterministic corpus.

    Missing task pairs are evidence of incompleteness and are not sent to a
    runner.  A one-sided pair, unknown task file, symlink, or invalid present
    pair is rejected rather than silently treated as missing.
    """

    if type(trust) is not CacheTrust:
        raise GoldenCacheError("trust must be an exact CacheTrust")
    raw_ordinals = tuple(selected_ordinals)
    if not raw_ordinals or any(type(item) is not int or item < 0 for item in raw_ordinals):
        raise GoldenCacheError("selected ordinals must be non-empty exact non-negative ints")
    if raw_ordinals != tuple(sorted(set(raw_ordinals))):
        raise GoldenCacheError("selected ordinals must be unique and sorted")
    selected = set(raw_ordinals)

    with _secure_job_directory(job_directory, trust) as session:
        job_payload = _read_regular_file_at(
            session.job_fd,
            "job.json",
            maximum_bytes=_MAXIMUM_JOB_BYTES,
            label="cache job document",
        )
        definition = _validated_job_document(job_payload, trust)
        directory_before = os.fstat(session.task_fd)
        names = _stable_directory_names(session.task_fd, "cache task directory")
        name_set = set(names)
        expected_by_stem = {_task_stem(task): task for task in definition.tasks}
        payload_names = {f"{stem}.bin" for stem in expected_by_stem}
        receipt_names = {f"{stem}.receipt.json" for stem in expected_by_stem}
        lock_names = {f"{stem}.lock" for stem in expected_by_stem}
        allowed_names = payload_names | receipt_names | lock_names
        unexpected = sorted(name_set - allowed_names)
        if unexpected:
            raise GoldenCacheError(
                f"cache task directory contains unexpected entry {unexpected[0]!r}"
            )
        for name in names:
            _require_regular_entry(session.task_fd, name, f"cache task entry {name!r}")

        authenticated_tasks = 0
        authenticated_records = 0
        authenticated_payload_bytes = 0
        authenticated_receipt_bytes = 0
        fate_distribution: Counter[str] = Counter()
        selected_fates: Counter[str] = Counter()
        selected_records: dict[int, dict[str, Any]] = {}
        payload_inventory: list[dict[str, Any]] = []
        authenticated_ordinals: list[int] = []
        stream_digest = hashlib.sha256()
        task_prefix_count = 0
        prefix_open = True

        for task in definition.tasks:
            stem = _task_stem(task)
            payload_name = f"{stem}.bin"
            receipt_name = f"{stem}.receipt.json"
            has_payload = payload_name in name_set
            has_receipt = receipt_name in name_set
            if has_payload != has_receipt:
                raise GoldenCacheError(
                    f"cache task {stem} has a one-sided payload/receipt pair"
                )
            if not has_payload:
                prefix_open = False
                continue
            if prefix_open:
                task_prefix_count += 1
            payload = _read_regular_file_at(
                session.task_fd,
                payload_name,
                maximum_bytes=_MAXIMUM_TASK_PAYLOAD_BYTES,
                label=f"task payload {payload_name}",
            )
            receipt_payload = _read_regular_file_at(
                session.task_fd,
                receipt_name,
                maximum_bytes=_MAXIMUM_RECEIPT_BYTES,
                label=f"task receipt {receipt_name}",
            )
            payload_sha256, receipt_length = _validated_receipt(
                payload,
                receipt_payload,
                task=task,
                payload_name=payload_name,
                trust=trust,
            )
            records = _validated_payload_records(
                payload,
                definition=definition,
                task=task,
                trust=trust,
            )
            authenticated_tasks += 1
            authenticated_records += len(records)
            authenticated_payload_bytes += len(payload)
            authenticated_receipt_bytes += receipt_length
            payload_inventory.append(
                {
                    "byteLength": len(payload),
                    "name": payload_name,
                    "recordCount": len(records),
                    "sha256": payload_sha256,
                }
            )
            for record in records:
                ordinal = record["coordinate"]["ordinal"]
                if authenticated_ordinals and ordinal <= authenticated_ordinals[-1]:
                    raise GoldenCacheError("authenticated record stream is not strictly ordered")
                authenticated_ordinals.append(ordinal)
                stream_digest.update(canonical_json_bytes(record))
                fate = record["transport"]["transport"]["fate"]
                fate_distribution[fate] += 1
                if ordinal in selected:
                    if ordinal in selected_records:
                        raise GoldenCacheError(f"selected ordinal {ordinal} is duplicated")
                    selected_records[ordinal] = record
                    selected_fates[fate] += 1

        directory_after = os.fstat(session.task_fd)
        if _stable_stat_identity(directory_before) != _stable_stat_identity(directory_after):
            raise GoldenCacheError("cache task directory changed during extraction")

    missing = tuple(item for item in raw_ordinals if item not in selected_records)
    if missing:
        raise GoldenCacheError(f"selected ordinals are absent from cache: {missing}")
    ordered_records = [selected_records[item] for item in raw_ordinals]
    continuous_prefix = 0
    for ordinal in authenticated_ordinals:
        if ordinal != continuous_prefix:
            break
        continuous_prefix += 1
    if sum(fate_distribution.values()) != authenticated_records:
        raise GoldenCacheError("fate distribution does not cover authenticated records")
    payload_set_sha256 = _payload_inventory_sha256(payload_inventory)
    direction_stream_sha256 = stream_digest.hexdigest()
    snapshot_checks = (
        (
            "authenticated task count",
            trust.authenticated_task_count,
            authenticated_tasks,
        ),
        (
            "authenticated direction count",
            trust.authenticated_direction_count,
            authenticated_records,
        ),
        (
            "authenticated payload-set SHA-256",
            trust.authenticated_payload_set_sha256,
            payload_set_sha256,
        ),
        (
            "authenticated direction-stream SHA-256",
            trust.authenticated_direction_stream_sha256,
            direction_stream_sha256,
        ),
    )
    for label, expected, actual in snapshot_checks:
        if expected is not None and actual != expected:
            raise GoldenCacheError(f"frozen {label} differs from its trust anchor")
    selected_record_sha256 = hashlib.sha256(
        canonical_json_bytes(ordered_records)
    ).hexdigest()
    evidence = {
        "authenticatedDirectionCount": authenticated_records,
        "authenticatedDirectionStreamSha256": direction_stream_sha256,
        "authenticatedPayloadBytes": authenticated_payload_bytes,
        "authenticatedPayloadSetSha256": payload_set_sha256,
        "authenticatedReceiptBytes": authenticated_receipt_bytes,
        "authenticatedTaskCount": authenticated_tasks,
        "authenticatedTaskPrefixCount": task_prefix_count,
        "continuousOrdinalPrefixCount": continuous_prefix,
        "fateDistribution": {
            fate: fate_distribution.get(fate, 0) for fate in sorted(_ALL_FATES)
        },
        "isCompleteCache": authenticated_tasks == len(definition.tasks),
        "isContinuousOrdinalPrefix": continuous_prefix == authenticated_records,
        "ordinalRange": (
            None
            if not authenticated_ordinals
            else [authenticated_ordinals[0], authenticated_ordinals[-1]]
        ),
        "selectedFateDistribution": {
            fate: selected_fates.get(fate, 0) for fate in sorted(_ALL_FATES)
        },
        "selectedRecordSetSha256": selected_record_sha256,
    }
    return {
        "cache": {
            "annulusEdgesOverMass": list(definition.annulus_edges_over_mass),
            "cacheJobKey": trust.cache_job_key,
            "jobDocumentSha256": trust.job_document_sha256,
            "plannedDirectionCount": definition.plan["directionCount"],
            "plannedTaskCount": len(definition.tasks),
            "scientificJobKey": trust.scientific_job_key,
            "scientificPlanSha256": trust.scientific_plan_sha256,
        },
        "evidence": evidence,
        "records": ordered_records,
        "schema": CORPUS_SCHEMA,
        "selectedOrdinals": list(raw_ordinals),
        "version": CORPUS_VERSION,
    }


def _read_document_path(path: Path, label: str) -> bytes:
    absolute = _canonical_absolute_path(path, label)
    parent, descriptor = _open_absolute_directory(absolute.parent, f"{label} parent")
    try:
        if parent != absolute.parent:
            raise GoldenCacheError(f"{label} parent path drifted")
        return _read_regular_file_at(
            descriptor,
            absolute.name,
            maximum_bytes=_MAXIMUM_DOCUMENT_BYTES,
            label=label,
        )
    finally:
        os.close(descriptor)


def load_golden_corpus(
    path: Path,
    *,
    expected_sha256: str = FROZEN_NESTED16_CORPUS_SHA256,
) -> dict[str, Any]:
    payload = _read_document_path(path, "golden corpus")
    _require_sha256(expected_sha256, "expected golden corpus SHA-256")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise GoldenCacheError("golden corpus differs from its external SHA-256 anchor")
    document = strict_json_bytes(payload, "golden corpus", _MAXIMUM_DOCUMENT_BYTES)
    expected_keys = {
        "cache",
        "evidence",
        "records",
        "schema",
        "selectedOrdinals",
        "version",
    }
    if type(document) is not dict or set(document) != expected_keys:
        raise GoldenCacheError("golden corpus has a non-exact schema")
    if document["schema"] != CORPUS_SCHEMA or document["version"] != CORPUS_VERSION:
        raise GoldenCacheError("golden corpus schema/version is unsupported")
    cache = document["cache"]
    cache_keys = {
        "annulusEdgesOverMass",
        "cacheJobKey",
        "jobDocumentSha256",
        "plannedDirectionCount",
        "plannedTaskCount",
        "scientificJobKey",
        "scientificPlanSha256",
    }
    if type(cache) is not dict or set(cache) != cache_keys:
        raise GoldenCacheError("golden corpus cache identity has a non-exact schema")
    for key in (
        "cacheJobKey",
        "jobDocumentSha256",
        "scientificJobKey",
        "scientificPlanSha256",
    ):
        _require_sha256(cache[key], f"golden corpus cache.{key}")
    planned_directions = _require_exact_int(
        cache["plannedDirectionCount"], "golden corpus plannedDirectionCount", minimum=1
    )
    planned_tasks = _require_exact_int(
        cache["plannedTaskCount"], "golden corpus plannedTaskCount", minimum=1
    )
    raw_edges = cache["annulusEdgesOverMass"]
    if type(raw_edges) is not list or len(raw_edges) < 2:
        raise GoldenCacheError("golden corpus annulus edges are invalid")
    annulus_edges = tuple(
        _require_finite_float(value, f"golden corpus annulus edge {index}")
        for index, value in enumerate(raw_edges)
    )
    if any(right <= left for left, right in zip(annulus_edges, annulus_edges[1:])):
        raise GoldenCacheError("golden corpus annulus edges are not strictly increasing")
    records = document["records"]
    ordinals = document["selectedOrdinals"]
    if type(records) is not list or type(ordinals) is not list or not records:
        raise GoldenCacheError("golden corpus must contain non-empty record/ordinal lists")
    if any(type(item) is not int or item < 0 for item in ordinals):
        raise GoldenCacheError("golden corpus selected ordinals are invalid")
    if ordinals != sorted(set(ordinals)) or len(records) != len(ordinals):
        raise GoldenCacheError("golden corpus selected ordinals are not unique and ordered")
    validated: list[dict[str, Any]] = []
    for index, (record, ordinal) in enumerate(zip(records, ordinals)):
        if type(record) is not dict or set(record) != {"coordinate", "transport"}:
            raise GoldenCacheError(f"golden record {index} has a non-exact schema")
        coordinate = _validated_coordinate(record["coordinate"], f"golden record {index}.coordinate")
        if coordinate["ordinal"] != ordinal:
            raise GoldenCacheError(f"golden record {index} ordinal differs from selection")
        transport = _validated_forward_transport(
            record["transport"],
            coordinate,
            f"golden record {index}.transport",
            annulus_edges_over_mass=annulus_edges,
        )
        validated.append({"coordinate": coordinate, "transport": transport})
    evidence = document["evidence"]
    if type(evidence) is not dict or set(evidence) != {
        "authenticatedDirectionCount",
        "authenticatedDirectionStreamSha256",
        "authenticatedPayloadBytes",
        "authenticatedPayloadSetSha256",
        "authenticatedReceiptBytes",
        "authenticatedTaskCount",
        "authenticatedTaskPrefixCount",
        "continuousOrdinalPrefixCount",
        "fateDistribution",
        "isCompleteCache",
        "isContinuousOrdinalPrefix",
        "ordinalRange",
        "selectedFateDistribution",
        "selectedRecordSetSha256",
    }:
        raise GoldenCacheError("golden corpus evidence has a non-exact schema")
    for key in (
        "authenticatedDirectionStreamSha256",
        "authenticatedPayloadSetSha256",
        "selectedRecordSetSha256",
    ):
        _require_sha256(evidence[key], f"golden corpus evidence.{key}")
    count_keys = (
        "authenticatedDirectionCount",
        "authenticatedPayloadBytes",
        "authenticatedReceiptBytes",
        "authenticatedTaskCount",
        "authenticatedTaskPrefixCount",
        "continuousOrdinalPrefixCount",
    )
    counts = {
        key: _require_exact_int(evidence[key], f"golden corpus evidence.{key}")
        for key in count_keys
    }
    for key in ("isCompleteCache", "isContinuousOrdinalPrefix"):
        if type(evidence[key]) is not bool:
            raise GoldenCacheError(f"golden corpus evidence.{key} must be an exact bool")
    if (
        counts["authenticatedTaskCount"] > planned_tasks
        or counts["authenticatedTaskPrefixCount"] > counts["authenticatedTaskCount"]
        or counts["authenticatedDirectionCount"] > planned_directions
        or counts["continuousOrdinalPrefixCount"] > counts["authenticatedDirectionCount"]
    ):
        raise GoldenCacheError("golden corpus evidence counts exceed their bounds")
    if evidence["isCompleteCache"] is not (
        counts["authenticatedTaskCount"] == planned_tasks
    ):
        raise GoldenCacheError("golden corpus complete-cache flag disagrees with counts")
    if evidence["isContinuousOrdinalPrefix"] is not (
        counts["continuousOrdinalPrefixCount"]
        == counts["authenticatedDirectionCount"]
    ):
        raise GoldenCacheError("golden corpus prefix flag disagrees with counts")
    ordinal_range = evidence["ordinalRange"]
    if (
        type(ordinal_range) is not list
        or len(ordinal_range) != 2
        or any(type(item) is not int or item < 0 for item in ordinal_range)
        or ordinal_range[1] < ordinal_range[0]
        or ordinal_range[1] >= planned_directions
    ):
        raise GoldenCacheError("golden corpus ordinal range is invalid")
    if counts["authenticatedDirectionCount"] < 1:
        raise GoldenCacheError("golden corpus has no authenticated directions")
    if evidence["isContinuousOrdinalPrefix"] and ordinal_range != [
        0,
        counts["authenticatedDirectionCount"] - 1,
    ]:
        raise GoldenCacheError("golden corpus continuous prefix has the wrong range")
    for key, expected_total in (
        ("fateDistribution", counts["authenticatedDirectionCount"]),
        ("selectedFateDistribution", len(records)),
    ):
        distribution = evidence[key]
        if type(distribution) is not dict or set(distribution) != set(_ALL_FATES):
            raise GoldenCacheError(f"golden corpus evidence.{key} has an invalid schema")
        if any(type(value) is not int or value < 0 for value in distribution.values()):
            raise GoldenCacheError(f"golden corpus evidence.{key} has an invalid count")
        if sum(distribution.values()) != expected_total:
            raise GoldenCacheError(f"golden corpus evidence.{key} does not close")
    derived_selected_fates = Counter(
        record["transport"]["transport"]["fate"] for record in validated
    )
    if evidence["selectedFateDistribution"] != {
        fate: derived_selected_fates.get(fate, 0) for fate in sorted(_ALL_FATES)
    }:
        raise GoldenCacheError("golden corpus selected fate distribution is stale")
    if any(ordinal > ordinal_range[1] for ordinal in ordinals):
        raise GoldenCacheError("golden corpus selection lies outside authenticated range")
    expected_record_hash = hashlib.sha256(canonical_json_bytes(validated)).hexdigest()
    if evidence["selectedRecordSetSha256"] != expected_record_hash:
        raise GoldenCacheError("golden corpus selected-record hash is stale")
    return document


def load_candidate_document(path: Path) -> dict[str, Any]:
    payload = _read_document_path(path, "native candidate")
    document = strict_json_bytes(payload, "native candidate", _MAXIMUM_DOCUMENT_BYTES)
    if type(document) is not dict or set(document) not in (
        {"records", "schema", "version"},
        {"expanded", "records", "schema", "version"},
    ):
        raise GoldenCacheError("native candidate has a non-exact top-level schema")
    if document["schema"] != CANDIDATE_SCHEMA or document["version"] != CORPUS_VERSION:
        raise GoldenCacheError("native candidate schema/version is unsupported")
    if "expanded" in document and type(document["expanded"]) is not dict:
        raise GoldenCacheError("native candidate expanded leaf must be an exact object")
    if type(document["records"]) is not list:
        raise GoldenCacheError("native candidate records must be an exact list")
    for index, record in enumerate(document["records"]):
        if type(record) is not dict or set(record) != {"coordinate", "transport"}:
            raise GoldenCacheError(f"native candidate record {index} has a non-exact schema")
        _validated_coordinate(record["coordinate"], f"native candidate record {index}.coordinate")
        if type(record["transport"]) is not dict:
            raise GoldenCacheError(f"native candidate record {index}.transport must be an object")
    return document


def _ordered_float_bits(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def _ulp_distance(left: float, right: float) -> int:
    return abs(_ordered_float_bits(left) - _ordered_float_bits(right))


def _scalar_summary(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int, float):
        return value
    return {"type": type(value).__name__}


def compare_corpus_documents(
    golden: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    maximum_ulp: int = 0,
    require_byte_exact: bool = False,
) -> dict[str, Any]:
    """Compare one canonical narrow candidate with explicit numeric limits."""

    if type(absolute_tolerance) is not float or not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise GoldenCacheError("absolute_tolerance must be an exact finite non-negative float")
    if type(relative_tolerance) is not float or not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise GoldenCacheError("relative_tolerance must be an exact finite non-negative float")
    if type(maximum_ulp) is not int or maximum_ulp < 0:
        raise GoldenCacheError("maximum_ulp must be an exact non-negative int")
    if type(require_byte_exact) is not bool:
        raise GoldenCacheError("require_byte_exact must be an exact bool")
    if type(golden) is not dict or type(candidate) is not dict:
        raise GoldenCacheError("golden and candidate documents must be exact dicts")
    golden_records = golden.get("records")
    candidate_records = candidate.get("records")
    if type(golden_records) is not list or type(candidate_records) is not list:
        raise GoldenCacheError("golden and candidate records must be exact lists")
    if len(candidate_records) != len(golden_records):
        raise GoldenCacheError("native candidate record count differs from golden corpus")
    golden_ordinals = [item["coordinate"]["ordinal"] for item in golden_records]
    candidate_ordinals = [item["coordinate"]["ordinal"] for item in candidate_records]
    if candidate_ordinals != golden_ordinals:
        raise GoldenCacheError("native candidate ordinals are missing, duplicated, or reordered")

    categorical: list[dict[str, Any]] = []
    numeric: list[dict[str, Any]] = []
    numeric_comparisons = 0
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    maximum_ulp_error = 0

    def compare(expected: Any, actual: Any, path: str) -> None:
        nonlocal numeric_comparisons
        nonlocal maximum_absolute_error, maximum_relative_error, maximum_ulp_error
        if type(expected) is float and type(actual) is float:
            if not math.isfinite(expected) or not math.isfinite(actual):
                categorical.append(
                    {"actual": actual, "expected": expected, "path": path, "reason": "non-finite-float"}
                )
                return
            numeric_comparisons += 1
            if expected == actual and expected.hex() != actual.hex():
                categorical.append(
                    {
                        "actual": actual.hex(),
                        "expected": expected.hex(),
                        "path": path,
                        "reason": "signed-zero-mismatch",
                    }
                )
                return
            absolute_error = abs(actual - expected)
            scale = max(abs(expected), abs(actual))
            relative_error = absolute_error if scale == 0.0 else absolute_error / scale
            ulp_error = _ulp_distance(expected, actual)
            maximum_absolute_error = max(maximum_absolute_error, absolute_error)
            maximum_relative_error = max(maximum_relative_error, relative_error)
            maximum_ulp_error = max(maximum_ulp_error, ulp_error)
            if expected.hex() != actual.hex():
                within_abs_rel = absolute_error <= max(
                    absolute_tolerance, relative_tolerance * scale
                )
                within_ulp = ulp_error <= maximum_ulp
                numeric.append(
                    {
                        "absoluteError": absolute_error,
                        "actual": actual,
                        "expected": expected,
                        "path": path,
                        "relativeError": relative_error,
                        "ulpError": ulp_error,
                        "withinLimits": within_abs_rel or within_ulp,
                    }
                )
            return
        if type(expected) is not type(actual):
            categorical.append(
                {
                    "actual": _scalar_summary(actual),
                    "actualType": type(actual).__name__,
                    "expected": _scalar_summary(expected),
                    "expectedType": type(expected).__name__,
                    "path": path,
                    "reason": "type-mismatch",
                }
            )
            return
        if type(expected) is dict:
            expected_keys = set(expected)
            actual_keys = set(actual)
            for key in sorted(expected_keys - actual_keys):
                categorical.append(
                    {"path": f"{path}.{key}", "reason": "missing-key"}
                )
            for key in sorted(actual_keys - expected_keys):
                categorical.append(
                    {"path": f"{path}.{key}", "reason": "extra-key"}
                )
            for key in sorted(expected_keys & actual_keys):
                compare(expected[key], actual[key], f"{path}.{key}")
            return
        if type(expected) is list:
            if len(expected) != len(actual):
                categorical.append(
                    {
                        "actual": len(actual),
                        "expected": len(expected),
                        "path": path,
                        "reason": "list-length-mismatch",
                    }
                )
            for index, (left, right) in enumerate(zip(expected, actual)):
                compare(left, right, f"{path}[{index}]")
            return
        if expected != actual or (
            type(expected) is str
            and expected.encode("utf-8") != actual.encode("utf-8")
        ):
            categorical.append(
                {
                    "actual": _scalar_summary(actual),
                    "expected": _scalar_summary(expected),
                    "path": path,
                    "reason": "categorical-mismatch",
                }
            )

    byte_exact_count = 0
    golden_cache = golden.get("cache")
    if type(golden_cache) is not dict or type(
        golden_cache.get("annulusEdgesOverMass")
    ) is not list:
        raise GoldenCacheError("golden corpus lacks comparison annulus edges")
    comparison_edges = tuple(golden_cache["annulusEdgesOverMass"])
    for index, (expected, actual) in enumerate(zip(golden_records, candidate_records)):
        if canonical_json_bytes(expected) == canonical_json_bytes(actual):
            byte_exact_count += 1
        if expected["coordinate"] != actual["coordinate"]:
            raise GoldenCacheError(
                f"native candidate coordinate {index} differs from golden coordinate"
            )
        _validated_forward_transport(
            actual["transport"],
            actual["coordinate"],
            f"native candidate record {index}.transport",
            annulus_edges_over_mass=comparison_edges,
        )
        compare(expected["transport"], actual["transport"], f"$.records[{index}].transport")

    violations = sum(not item["withinLimits"] for item in numeric)
    byte_exact = byte_exact_count == len(golden_records)
    qualified = not categorical and violations == 0 and (
        byte_exact or not require_byte_exact
    )
    return {
        "byteExact": byte_exact,
        "byteExactRecordCount": byte_exact_count,
        "categoricalDrift": categorical,
        "categoricalDriftCount": len(categorical),
        "goldenCorpusSha256": None,
        "limits": {
            "absoluteTolerance": absolute_tolerance,
            "maximumUlp": maximum_ulp,
            "relativeTolerance": relative_tolerance,
            "requireByteExact": require_byte_exact,
        },
        "maximumAbsoluteError": maximum_absolute_error,
        "maximumRelativeError": maximum_relative_error,
        "maximumUlpError": maximum_ulp_error,
        "numericComparisonCount": numeric_comparisons,
        "numericDrift": numeric,
        "numericDriftCount": len(numeric),
        "numericLimitViolationCount": violations,
        "qualified": qualified,
        "recordCount": len(golden_records),
        "schema": COMPARISON_SCHEMA,
        "scope": {
            "classification": "narrow-selected-record-comparison",
            "provesFullAuthenticatedCorpusParity": False,
            "provesNativeExecutionProvenance": False,
            "selectedOrdinalCount": len(golden_records),
        },
        "version": CORPUS_VERSION,
    }


def compare_corpus_files(
    golden_path: Path,
    candidate_path: Path,
    *,
    expected_golden_sha256: str = FROZEN_NESTED16_CORPUS_SHA256,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    maximum_ulp: int = 0,
    require_byte_exact: bool = False,
) -> dict[str, Any]:
    golden = load_golden_corpus(
        golden_path,
        expected_sha256=expected_golden_sha256,
    )
    candidate = load_candidate_document(candidate_path)
    report = compare_corpus_documents(
        golden,
        candidate,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        maximum_ulp=maximum_ulp,
        require_byte_exact=require_byte_exact,
    )
    report["goldenCorpusSha256"] = expected_golden_sha256
    return report


def candidate_document_from_corpus(corpus: Mapping[str, Any]) -> dict[str, Any]:
    """Build the canonical narrow candidate shape for harness smoke tests."""

    if type(corpus) is not dict or type(corpus.get("records")) is not list:
        raise GoldenCacheError("corpus must contain an exact records list")
    return {
        "records": corpus["records"],
        "schema": CANDIDATE_SCHEMA,
        "version": CORPUS_VERSION,
    }


def _write_canonical_output(path: Path, payload: bytes) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.name or absolute.name in (".", ".."):
        raise GoldenCacheError("output filename is invalid")
    parent, parent_fd = _open_absolute_directory(absolute.parent, "output parent")
    if parent != absolute.parent:
        os.close(parent_fd)
        raise GoldenCacheError("output parent path drifted")
    temporary = f".{absolute.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written < 1:
                raise GoldenCacheError("output write made no forward progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary,
            absolute.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def _write_or_print(document: Mapping[str, Any], output: str | None) -> None:
    payload = canonical_json_bytes(document)
    if output is None or output == "-":
        sys.stdout.buffer.write(payload)
        return
    _write_canonical_output(Path(output), payload)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticate frozen nested16 golden evidence or compare a native candidate."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser(
        "extract", help="read-only authenticate and extract the fixed narrow corpus"
    )
    extract.add_argument("job_directory", type=Path)
    extract.add_argument("--output", default="-")
    compare = subparsers.add_parser(
        "compare", help="compare an ordered native candidate with a golden corpus"
    )
    compare.add_argument("golden", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--absolute-tolerance", type=float, default=0.0)
    compare.add_argument("--relative-tolerance", type=float, default=0.0)
    compare.add_argument("--maximum-ulp", type=int, default=0)
    compare.add_argument("--require-byte-exact", action="store_true")
    compare.add_argument("--output", default="-")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "extract":
            job_directory = _canonical_absolute_path(
                args.job_directory, "cache job directory"
            )
            if args.output not in (None, "-"):
                output = Path(os.path.abspath(args.output))
                try:
                    output.relative_to(job_directory)
                except ValueError:
                    pass
                else:
                    raise GoldenCacheError("output must not be inside the golden cache")
            document = extract_golden_corpus(job_directory)
            _write_or_print(document, args.output)
            return 0
        report = compare_corpus_files(
            _canonical_absolute_path(args.golden, "golden corpus"),
            _canonical_absolute_path(args.candidate, "native candidate"),
            absolute_tolerance=float(args.absolute_tolerance),
            relative_tolerance=float(args.relative_tolerance),
            maximum_ulp=args.maximum_ulp,
            require_byte_exact=args.require_byte_exact,
        )
        _write_or_print(report, args.output)
        return 0 if report["qualified"] else 2
    except GoldenCacheError as error:
        print(f"golden-cache error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
