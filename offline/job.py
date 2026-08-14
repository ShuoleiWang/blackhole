"""Deterministic, resumable execution primitives for offline render jobs.

The module deliberately knows nothing about a particular metric, ray solver,
or output manifest.  A producer receives an immutable :class:`JobSpec` and one
canonical :class:`TaskKey`, then returns the task payload as bytes.  The runner
stores that payload and its receipt atomically in a content-addressed cache.

Scheduling controls such as ``jobs`` and ``max_in_flight`` are intentionally
absent from ``JobSpec``.  They may change throughput, but must never change the
scientific identity of a job or the canonical order of its results.
"""

from __future__ import annotations

import hashlib
import errno
import json
import math
import multiprocessing
import os
import re
import stat
import uuid
import fcntl
from contextlib import contextmanager
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final


JOB_SCHEMA: Final = "blackhole.offline-job/v1"
RECEIPT_SCHEMA: Final = "blackhole.offline-task-receipt/v1"
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
MAXIMUM_WORKERS: Final = 64
MAXIMUM_IN_FLIGHT_TASKS: Final = 256
MAXIMUM_TASK_COUNT: Final = 65_536
MAXIMUM_JOB_DOCUMENT_BYTES: Final = 16 * 1024 * 1024
MAXIMUM_TASK_PAYLOAD_BYTES: Final = 256 * 1024 * 1024
MAXIMUM_RECEIPT_BYTES: Final = 64 * 1024
_PATH_TYPE: Final = type(Path())
Producer = Callable[["JobSpec", "TaskKey"], bytes | bytearray | memoryview]
ExecutorFactory = Callable[[int], Executor]


class OfflineJobCacheError(RuntimeError):
    """Raised when the anchored cache path or cache entry is unsafe."""


class _InvalidCachedEntry(ValueError):
    """Internal signal for ordinary corrupt cache data that may be replaced."""


@dataclass(frozen=True)
class _FrozenJsonObject(Mapping[str, Any]):
    """Small pickle-safe immutable mapping used inside a frozen JobSpec."""

    _items: tuple[tuple[str, Any], ...]

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)


def _freeze_json(value: Any, path: str = "$parameters") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        items: list[tuple[str, Any]] = []
        keys = tuple(value)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError(f"{path} has a non-string object key")
        for key in sorted(keys):
            items.append((key, _freeze_json(value[key], f"{path}.{key}")))
        return _FrozenJsonObject(tuple(items))
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} contains unsupported canonical JSON value {type(value).__name__}"
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Encode finite JSON data with stable key order and no insignificant space."""

    frozen = _freeze_json(value, "$canonical")
    return (
        json.dumps(
            _json_value(frozen),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path, block_bytes: int = 1024 * 1024) -> str:
    """Hash a file without retaining it in memory."""

    if block_bytes < 1:
        raise ValueError("block_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, order=True)
class InputArtifact:
    """An immutable input identity; locations alone never identify inputs."""

    uri: str
    byte_length: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri:
            raise ValueError("input artifact URI must be non-empty")
        if (
            isinstance(self.byte_length, bool)
            or not isinstance(self.byte_length, int)
            or self.byte_length < 0
        ):
            raise ValueError("input artifact byte_length must be non-negative")
        if (
            not isinstance(self.sha256, str)
            or not SHA256_PATTERN.fullmatch(self.sha256)
        ):
            raise ValueError("input artifact sha256 must be lowercase hexadecimal")

    @classmethod
    def from_path(cls, uri: str, path: Path) -> "InputArtifact":
        return cls(uri, path.stat().st_size, sha256_file(path))

    def as_dict(self) -> dict[str, Any]:
        return {
            "byteLength": self.byte_length,
            "sha256": self.sha256,
            "uri": self.uri,
        }


@dataclass(frozen=True, order=True)
class TaskKey:
    """Canonical tile coordinate and ordering key for one independent task."""

    sample_index: int
    y: int
    x: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("sample_index", "y", "x", "width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"TaskKey.{name} must be an integer")
        if self.sample_index < 0 or self.x < 0 or self.y < 0:
            raise ValueError("task sample and origin coordinates must be non-negative")
        if self.width < 1 or self.height < 1:
            raise ValueError("task dimensions must be positive")

    @property
    def file_stem(self) -> str:
        return (
            f"t{self.sample_index:06d}-y{self.y:06d}-x{self.x:06d}"
            f"-w{self.width:06d}-h{self.height:06d}"
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "height": self.height,
            "sampleIndex": self.sample_index,
            "width": self.width,
            "x": self.x,
            "y": self.y,
        }


@dataclass(frozen=True)
class JobSpec:
    """Canonical scientific identity for a collection of independent tasks."""

    producer: str
    algorithm_version: str
    tasks: Sequence[TaskKey]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    inputs: Sequence[InputArtifact] = ()
    producer_source_hashes: Sequence[str] = ()
    record_bytes: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.producer, str) or not self.producer:
            raise ValueError("producer must be a non-empty string")
        if not isinstance(self.algorithm_version, str) or not self.algorithm_version:
            raise ValueError("algorithm_version must be a non-empty string")
        if isinstance(self.record_bytes, bool) or not isinstance(self.record_bytes, int):
            raise TypeError("record_bytes must be an integer")
        if self.record_bytes < 1:
            raise ValueError("record_bytes must be positive")

        raw_tasks = tuple(self.tasks)
        if any(not isinstance(task, TaskKey) for task in raw_tasks):
            raise TypeError("tasks must contain TaskKey values")
        tasks = tuple(sorted(raw_tasks))
        if not tasks:
            raise ValueError("a job must contain at least one task")
        if len(set(tasks)) != len(tasks):
            raise ValueError("job tasks must be unique")

        raw_inputs = tuple(self.inputs)
        if any(not isinstance(item, InputArtifact) for item in raw_inputs):
            raise TypeError("inputs must contain InputArtifact values")
        inputs = tuple(sorted(raw_inputs))
        input_uris = [item.uri for item in inputs]
        if len(set(input_uris)) != len(input_uris):
            raise ValueError("input artifact URIs must be unique")

        raw_source_hashes = tuple(self.producer_source_hashes)
        if any(not isinstance(value, str) for value in raw_source_hashes):
            raise ValueError(
                "producer source hashes must be lowercase SHA-256 digests"
            )
        source_hashes = tuple(sorted(raw_source_hashes))
        if len(set(source_hashes)) != len(source_hashes):
            raise ValueError("producer source hashes must be unique")
        if any(not SHA256_PATTERN.fullmatch(value) for value in source_hashes):
            raise ValueError(
                "producer source hashes must be lowercase SHA-256 digests"
            )

        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "parameters", _freeze_json(self.parameters))
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "producer_source_hashes", source_hashes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithmVersion": self.algorithm_version,
            "inputs": [item.as_dict() for item in self.inputs],
            "parameters": _json_value(self.parameters),
            "producer": self.producer,
            "producerSourceHashes": list(self.producer_source_hashes),
            "recordBytes": self.record_bytes,
            "schema": JOB_SCHEMA,
            "tasks": [task.as_dict() for task in self.tasks],
        }

    @property
    def job_key(self) -> str:
        return job_key(self)


_EXACT_FROZEN_JSON_OBJECT_TYPE: Final = _FrozenJsonObject
_EXACT_INPUT_ARTIFACT_TYPE: Final = InputArtifact
_EXACT_JOB_SPEC_TYPE: Final = JobSpec
_EXACT_TASK_KEY_TYPE: Final = TaskKey
_MAXIMUM_CANONICAL_JSON_DEPTH: Final = 64


def _validated_frozen_json(
    value: Any,
    path: str,
    *,
    depth: int = 0,
) -> Any:
    """Return plain JSON only from the exact immutable JobSpec representation."""

    if depth > _MAXIMUM_CANONICAL_JSON_DEPTH:
        raise ValueError(
            f"{path} exceeds the hard canonical JSON nesting limit"
        )
    if value is None:
        return None
    if type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if type(value) is tuple:
        return [
            _validated_frozen_json(item, f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if type(value) is _EXACT_FROZEN_JSON_OBJECT_TYPE:
        items = object.__getattribute__(value, "_items")
        if type(items) is not tuple:
            raise TypeError(f"{path} lost its exact immutable object shape")
        result: dict[str, Any] = {}
        previous: str | None = None
        for index, pair in enumerate(items):
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError(
                    f"{path} object item {index} lost its exact pair shape"
                )
            key, item = pair
            if type(key) is not str:
                raise TypeError(f"{path} has a non-exact string object key")
            if previous is not None and key <= previous:
                raise ValueError(
                    f"{path} object keys are not unique canonical order"
                )
            previous = key
            result[key] = _validated_frozen_json(
                item,
                f"{path}.{key}",
                depth=depth + 1,
            )
        return result
    raise TypeError(
        f"{path} contains non-exact canonical JSON value "
        f"{type(value).__name__}"
    )


def _validated_task_key(key: Any, path: str) -> tuple[int, int, int, int, int]:
    if type(key) is not _EXACT_TASK_KEY_TYPE:
        raise TypeError(f"{path} must be an exact TaskKey")
    values = tuple(
        object.__getattribute__(key, name)
        for name in ("sample_index", "y", "x", "width", "height")
    )
    if any(type(value) is not int for value in values):
        raise TypeError(f"{path} fields must be exact integers")
    sample_index, y, x, width, height = values
    if sample_index < 0 or y < 0 or x < 0:
        raise ValueError(f"{path} sample and origin must be non-negative")
    if width < 1 or height < 1:
        raise ValueError(f"{path} dimensions must be positive")
    return sample_index, y, x, width, height


def _task_document_from_values(
    values: tuple[int, int, int, int, int],
) -> dict[str, int]:
    sample_index, y, x, width, height = values
    return {
        "height": height,
        "sampleIndex": sample_index,
        "width": width,
        "x": x,
        "y": y,
    }


def _task_file_stem(key: TaskKey) -> str:
    sample_index, y, x, width, height = _validated_task_key(key, "$task")
    return (
        f"t{sample_index:06d}-y{y:06d}-x{x:06d}"
        f"-w{width:06d}-h{height:06d}"
    )


def _validated_job_spec_document(spec: Any) -> dict[str, Any]:
    """Deeply authenticate a JobSpec without invoking its public methods."""

    if (
        JobSpec is not _EXACT_JOB_SPEC_TYPE
        or TaskKey is not _EXACT_TASK_KEY_TYPE
        or InputArtifact is not _EXACT_INPUT_ARTIFACT_TYPE
        or _FrozenJsonObject is not _EXACT_FROZEN_JSON_OBJECT_TYPE
    ):
        raise RuntimeError("offline job public type bindings changed at runtime")
    if type(spec) is not _EXACT_JOB_SPEC_TYPE:
        raise TypeError("run_job expects an exact JobSpec")

    producer = object.__getattribute__(spec, "producer")
    algorithm_version = object.__getattribute__(spec, "algorithm_version")
    record_bytes = object.__getattribute__(spec, "record_bytes")
    if type(producer) is not str or not producer:
        raise TypeError("JobSpec.producer must be an exact non-empty string")
    if type(algorithm_version) is not str or not algorithm_version:
        raise TypeError(
            "JobSpec.algorithm_version must be an exact non-empty string"
        )
    if type(record_bytes) is not int or record_bytes < 1:
        raise TypeError("JobSpec.record_bytes must be an exact positive integer")

    tasks = object.__getattribute__(spec, "tasks")
    if type(tasks) is not tuple or not tasks:
        raise TypeError("JobSpec.tasks must be an exact non-empty tuple")
    if len(tasks) > MAXIMUM_TASK_COUNT:
        raise ValueError("task count exceeds the hard generic-job limit")
    task_values = tuple(
        _validated_task_key(key, f"$spec.tasks[{index}]")
        for index, key in enumerate(tasks)
    )
    if task_values != tuple(sorted(task_values)):
        raise ValueError("JobSpec.tasks are not in canonical order")
    if len(set(task_values)) != len(task_values):
        raise ValueError("JobSpec.tasks are not unique")

    inputs = object.__getattribute__(spec, "inputs")
    if type(inputs) is not tuple:
        raise TypeError("JobSpec.inputs must be an exact tuple")
    input_values: list[tuple[str, int, str]] = []
    for index, item in enumerate(inputs):
        if type(item) is not _EXACT_INPUT_ARTIFACT_TYPE:
            raise TypeError(
                f"$spec.inputs[{index}] must be an exact InputArtifact"
            )
        uri = object.__getattribute__(item, "uri")
        byte_length = object.__getattribute__(item, "byte_length")
        digest = object.__getattribute__(item, "sha256")
        if type(uri) is not str or not uri:
            raise TypeError(
                f"$spec.inputs[{index}].uri must be an exact non-empty string"
            )
        if type(byte_length) is not int or byte_length < 0:
            raise TypeError(
                f"$spec.inputs[{index}].byte_length must be an exact "
                "non-negative integer"
            )
        if type(digest) is not str or not SHA256_PATTERN.fullmatch(digest):
            raise TypeError(
                f"$spec.inputs[{index}].sha256 must be an exact SHA-256 string"
            )
        input_values.append((uri, byte_length, digest))
    if tuple(input_values) != tuple(sorted(input_values)):
        raise ValueError("JobSpec.inputs are not in canonical order")
    if len({uri for uri, _length, _digest in input_values}) != len(input_values):
        raise ValueError("JobSpec input URIs are not unique")

    source_hashes = object.__getattribute__(spec, "producer_source_hashes")
    if type(source_hashes) is not tuple:
        raise TypeError("JobSpec.producer_source_hashes must be an exact tuple")
    if any(
        type(value) is not str or not SHA256_PATTERN.fullmatch(value)
        for value in source_hashes
    ):
        raise TypeError(
            "JobSpec producer source hashes must be exact SHA-256 strings"
        )
    if source_hashes != tuple(sorted(source_hashes)):
        raise ValueError("JobSpec producer source hashes are not canonical order")
    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError("JobSpec producer source hashes are not unique")

    parameters = object.__getattribute__(spec, "parameters")
    if type(parameters) is not _EXACT_FROZEN_JSON_OBJECT_TYPE:
        raise TypeError("JobSpec.parameters lost its exact frozen object shape")
    parameter_document = _validated_frozen_json(parameters, "$spec.parameters")
    return {
        "algorithmVersion": algorithm_version,
        "inputs": [
            {"byteLength": length, "sha256": digest, "uri": uri}
            for uri, length, digest in input_values
        ],
        "parameters": parameter_document,
        "producer": producer,
        "producerSourceHashes": list(source_hashes),
        "recordBytes": record_bytes,
        "schema": JOB_SCHEMA,
        "tasks": [
            _task_document_from_values(values)
            for values in task_values
        ],
    }


def _require_job_spec_snapshot(spec: JobSpec, expected: bytes) -> None:
    current = canonical_json_bytes(_validated_job_spec_document(spec))
    if current != expected:
        raise OfflineJobCacheError("JobSpec changed during secure execution")


def job_key(spec: JobSpec) -> str:
    """Return the content identity of a JobSpec (never its scheduling policy)."""

    if not isinstance(spec, JobSpec):
        raise TypeError("job_key expects a JobSpec")
    return hashlib.sha256(canonical_json_bytes(spec.as_dict())).hexdigest()


@dataclass(frozen=True)
class TaskResult:
    key: TaskKey
    payload_path: Path
    receipt_path: Path
    record_count: int
    byte_length: int
    sha256: str
    reused: bool


@dataclass(frozen=True)
class JobRun:
    job_key: str
    results: tuple[TaskResult, ...]
    reused_tasks: int
    executed_tasks: int
    max_in_flight_observed: int


@dataclass(frozen=True, slots=True)
class _SecureCacheSession:
    cache_root: Path
    job_directory: Path
    task_directory: Path
    cache_fd: int
    job_fd: int
    task_fd: int


def _directory_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise OfflineJobCacheError(
            "platform lacks O_NOFOLLOW/O_DIRECTORY secure cache primitives"
        )
    return os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)


def _open_absolute_cache_root(
    path: Path,
    *,
    create_final: bool,
) -> tuple[Path, int]:
    if type(path) is not _PATH_TYPE or not path.is_absolute():
        raise TypeError("cache_root must be an exact absolute platform Path")
    if path == path.parent:
        raise OfflineJobCacheError(
            "cache_root must name a non-root absolute directory"
        )
    components = path.parts[1:]
    flags = _directory_open_flags()
    descriptor = os.open(os.sep, flags)
    try:
        for index, component in enumerate(components):
            if component in ("", ".", "..") or os.sep in component:
                raise OfflineJobCacheError(
                    "cache_root has a non-canonical path component"
                )
            final = index == len(components) - 1
            try:
                following = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                if (
                    error.errno != errno.ENOENT
                    or not final
                    or not create_final
                ):
                    raise OfflineJobCacheError(
                        "cache_root absolute path contains a symlink, missing "
                        "ancestor, or non-directory component"
                    ) from error
                created = False
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    # A concurrent secure runner may have created the final
                    # directory after our failed open.  Reopen it with the same
                    # no-follow directory flags below.
                    pass
                except OSError as create_error:
                    raise OfflineJobCacheError(
                        "cache_root final directory cannot be created securely"
                    ) from create_error
                try:
                    if created:
                        os.fsync(descriptor)
                    following = os.open(component, flags, dir_fd=descriptor)
                except OSError as open_error:
                    raise OfflineJobCacheError(
                        "cache_root final directory cannot be opened securely"
                    ) from open_error
            os.close(descriptor)
            descriptor = following
        return path, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_or_create_directory_at(parent_fd: int, name: str, label: str) -> int:
    if type(name) is not str or not name or os.sep in name or name in (".", ".."):
        raise OfflineJobCacheError(f"{label} name is invalid")
    flags = _directory_open_flags()
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise OfflineJobCacheError(
                f"{label} is a symlink or non-directory"
            ) from error
    created = False
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        created = True
    except FileExistsError:
        pass
    except OSError as error:
        raise OfflineJobCacheError(
            f"{label} cannot be created securely"
        ) from error
    try:
        if created:
            os.fsync(parent_fd)
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise OfflineJobCacheError(
            f"{label} cannot be opened securely"
        ) from error


@contextmanager
def _secure_cache_session(
    cache_root: Path,
    cache_key: str,
) -> Iterator[_SecureCacheSession]:
    absolute, cache_fd = _open_absolute_cache_root(
        cache_root,
        create_final=True,
    )
    job_fd = -1
    task_fd = -1
    try:
        job_fd = _open_or_create_directory_at(
            cache_fd,
            cache_key,
            "cache job directory",
        )
        task_fd = _open_or_create_directory_at(
            job_fd,
            "tasks",
            "cache task directory",
        )
        yield _SecureCacheSession(
            absolute,
            absolute / cache_key,
            absolute / cache_key / "tasks",
            cache_fd,
            job_fd,
            task_fd,
        )
    finally:
        if task_fd >= 0:
            os.close(task_fd)
        if job_fd >= 0:
            os.close(job_fd)
        os.close(cache_fd)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written < 1:
            raise OfflineJobCacheError(
                "secure cache write made no forward progress"
            )
        offset += written


def _atomic_write_at(directory_fd: int, name: str, payload: bytes) -> None:
    temporary = f".{name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except OSError as error:
        raise OfflineJobCacheError(
            f"cache entry {name!r} cannot be published securely"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _entry_kind_at(directory_fd: int, name: str, label: str) -> str:
    try:
        snapshot = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return "missing"
    except OSError as error:
        raise OfflineJobCacheError(f"{label} cannot be inspected") from error
    if stat.S_ISLNK(snapshot.st_mode):
        raise OfflineJobCacheError(f"{label} must not be a symlink")
    if not stat.S_ISREG(snapshot.st_mode):
        raise OfflineJobCacheError(f"{label} must be a regular file")
    return "regular"


def _read_regular_file_at(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise OfflineJobCacheError(
            f"{label} cannot be opened without following symlinks"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OfflineJobCacheError(f"{label} must be a regular file")
        if before.st_size > maximum_bytes:
            raise _InvalidCachedEntry(f"{label} exceeds its hard byte limit")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise OfflineJobCacheError(f"{label} changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise OfflineJobCacheError(f"{label} grew while being read")
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field)
            for field in stable_fields
        ):
            raise OfflineJobCacheError(f"{label} changed while being read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise OfflineJobCacheError(f"{label} stable-size check failed")
        return payload
    finally:
        os.close(descriptor)


def _strict_json_payload(payload: bytes, label: str) -> Any:
    if type(payload) is not bytes or len(payload) > MAXIMUM_RECEIPT_BYTES:
        raise _InvalidCachedEntry(f"{label} exceeds its hard byte limit")

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
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise _InvalidCachedEntry(f"{label} is not strict JSON") from error
    if canonical_json_bytes(value) != payload:
        raise _InvalidCachedEntry(f"{label} is not canonical JSON")
    return value


def _task_names(key: TaskKey) -> tuple[str, str]:
    stem = _task_file_stem(key)
    return (
        f"{stem}.bin",
        f"{stem}.receipt.json",
    )


def _matches_task_document(value: Any, key: TaskKey) -> bool:
    return (
        type(value) is dict
        and set(value) == {"height", "sampleIndex", "width", "x", "y"}
        and all(type(item) is int for item in value.values())
        and value == _task_document_from_values(
            _validated_task_key(key, "$task")
        )
    )


def _secure_cached_result(
    session: _SecureCacheSession,
    cache_key: str,
    key: TaskKey,
    record_bytes: int,
) -> TaskResult | None:
    payload_name, receipt_name = _task_names(key)
    payload_kind = _entry_kind_at(session.task_fd, payload_name, "task payload")
    receipt_kind = _entry_kind_at(session.task_fd, receipt_name, "task receipt")
    if payload_kind == "missing" or receipt_kind == "missing":
        return None
    try:
        receipt_payload = _read_regular_file_at(
            session.task_fd,
            receipt_name,
            maximum_bytes=MAXIMUM_RECEIPT_BYTES,
            label="task receipt",
        )
        receipt = _strict_json_payload(receipt_payload, "task receipt")
    except _InvalidCachedEntry:
        return None
    if type(receipt) is not dict or set(receipt) != {
        "byteLength",
        "jobKey",
        "payload",
        "recordCount",
        "schema",
        "sha256",
        "task",
    }:
        return None
    if (
        receipt["schema"] != RECEIPT_SCHEMA
        or receipt["jobKey"] != cache_key
        or receipt["payload"] != payload_name
        or not _matches_task_document(receipt["task"], key)
        or type(receipt["recordCount"]) is not int
        or receipt["recordCount"] < 0
        or type(receipt["byteLength"]) is not int
        or receipt["byteLength"] < 0
        or receipt["byteLength"] > MAXIMUM_TASK_PAYLOAD_BYTES
        or type(receipt["sha256"]) is not str
        or not SHA256_PATTERN.fullmatch(receipt["sha256"])
    ):
        return None
    try:
        payload = _read_regular_file_at(
            session.task_fd,
            payload_name,
            maximum_bytes=MAXIMUM_TASK_PAYLOAD_BYTES,
            label="task payload",
        )
    except _InvalidCachedEntry:
        return None
    digest = hashlib.sha256(payload).hexdigest()
    if (
        receipt["byteLength"] != len(payload)
        or len(payload) % record_bytes
        or receipt["recordCount"] != len(payload) // record_bytes
        or receipt["sha256"] != digest
    ):
        return None
    return TaskResult(
        key=key,
        payload_path=session.task_directory / payload_name,
        receipt_path=session.task_directory / receipt_name,
        record_count=receipt["recordCount"],
        byte_length=len(payload),
        sha256=digest,
        reused=True,
    )


def _write_job_document(
    session: _SecureCacheSession,
    expected: bytes,
) -> None:
    kind = _entry_kind_at(session.job_fd, "job.json", "cache job document")
    if kind == "regular":
        try:
            current = _read_regular_file_at(
                session.job_fd,
                "job.json",
                maximum_bytes=MAXIMUM_JOB_DOCUMENT_BYTES,
                label="cache job document",
            )
        except _InvalidCachedEntry:
            current = b""
        if current == expected:
            return
    _atomic_write_at(session.job_fd, "job.json", expected)


def _require_job_document(
    session: _SecureCacheSession,
    expected: bytes,
) -> None:
    if _entry_kind_at(
        session.job_fd,
        "job.json",
        "cache job document",
    ) != "regular":
        raise OfflineJobCacheError("cache job document disappeared")
    try:
        current = _read_regular_file_at(
            session.job_fd,
            "job.json",
            maximum_bytes=MAXIMUM_JOB_DOCUMENT_BYTES,
            label="cache job document",
        )
    except _InvalidCachedEntry as error:
        raise OfflineJobCacheError(
            "cache job document became oversized"
        ) from error
    if current != expected:
        raise OfflineJobCacheError("cache job document changed during execution")


def _verify_lock_path(
    session: _SecureCacheSession,
    key: TaskKey,
    descriptor: int,
) -> None:
    name = f"{_task_file_stem(key)}.lock"
    try:
        named = os.stat(name, dir_fd=session.task_fd, follow_symlinks=False)
        held = os.fstat(descriptor)
    except OSError as error:
        raise OfflineJobCacheError("task lock path changed") from error
    if (
        not stat.S_ISREG(named.st_mode)
        or not stat.S_ISREG(held.st_mode)
        or named.st_dev != held.st_dev
        or named.st_ino != held.st_ino
    ):
        raise OfflineJobCacheError("task lock inode changed during execution")


def _acquire_task_lock(session: _SecureCacheSession, key: TaskKey) -> int:
    name = f"{_task_file_stem(key)}.lock"
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=session.task_fd)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OfflineJobCacheError("task lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _verify_lock_path(session, key, descriptor)
        return descriptor
    except OfflineJobCacheError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise OfflineJobCacheError(
            "task lock cannot be acquired without following symlinks"
        ) from error


def _release_task_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _verify_secure_session_path(
    session: _SecureCacheSession,
    cache_key: str,
) -> None:
    absolute, cache_fd = _open_absolute_cache_root(
        session.cache_root,
        create_final=False,
    )
    job_fd = -1
    task_fd = -1
    try:
        if absolute != session.cache_root:
            raise OfflineJobCacheError("cache_root identity changed")
        job_fd = os.open(cache_key, _directory_open_flags(), dir_fd=cache_fd)
        task_fd = os.open("tasks", _directory_open_flags(), dir_fd=job_fd)
        for held, reopened, label in (
            (session.cache_fd, cache_fd, "cache root"),
            (session.job_fd, job_fd, "cache job directory"),
            (session.task_fd, task_fd, "cache task directory"),
        ):
            held_stat = os.fstat(held)
            reopened_stat = os.fstat(reopened)
            if (
                held_stat.st_dev != reopened_stat.st_dev
                or held_stat.st_ino != reopened_stat.st_ino
            ):
                raise OfflineJobCacheError(
                    f"{label} path changed during secure execution"
                )
    except OSError as error:
        raise OfflineJobCacheError(
            "cache path became a symlink or changed during secure execution"
        ) from error
    finally:
        if task_fd >= 0:
            os.close(task_fd)
        if job_fd >= 0:
            os.close(job_fd)
        os.close(cache_fd)


def _validated_payload(
    produced: Any,
    record_bytes: int,
    key: TaskKey,
) -> bytes:
    if not isinstance(produced, (bytes, bytearray, memoryview)):
        raise TypeError("offline task producers must return a bytes-like payload")
    # Authenticate the buffer size before materializing an immutable copy.
    # In particular, a huge memoryview must not force an equally huge parent
    # allocation merely so the post-copy limit can reject it.
    view = memoryview(produced)
    if view.nbytes > MAXIMUM_TASK_PAYLOAD_BYTES:
        raise ValueError(
            f"task {_task_file_stem(key)} exceeds the hard task payload byte limit"
        )
    payload = bytes(view)
    if len(payload) != view.nbytes:
        raise ValueError("bytes-like task payload changed while being copied")
    if type(record_bytes) is not int or record_bytes < 1:
        raise TypeError("record_bytes snapshot must be an exact positive integer")
    if len(payload) % record_bytes:
        raise ValueError(
            f"task {_task_file_stem(key)} produced {len(payload)} bytes, which "
            f"is not a multiple of record_bytes={record_bytes}"
        )
    return payload


def _produce_task_payload(
    producer: Producer,
    spec: JobSpec,
    key: TaskKey,
    record_bytes: int,
) -> bytes:
    """Pickle-safe worker entry; workers never receive a cache path or fd."""

    return _validated_payload(producer(spec, key), record_bytes, key)


def _publish_task_payload(
    session: _SecureCacheSession,
    cache_key: str,
    record_bytes: int,
    key: TaskKey,
    payload: bytes,
) -> TaskResult:
    payload = _validated_payload(payload, record_bytes, key)
    payload_name, receipt_name = _task_names(key)
    digest = hashlib.sha256(payload).hexdigest()
    receipt = canonical_json_bytes(
        {
            "byteLength": len(payload),
            "jobKey": cache_key,
            "payload": payload_name,
            "recordCount": len(payload) // record_bytes,
            "schema": RECEIPT_SCHEMA,
            "sha256": digest,
            "task": _task_document_from_values(
                _validated_task_key(key, "$task")
            ),
        }
    )
    if len(receipt) > MAXIMUM_RECEIPT_BYTES:
        raise ValueError("generated task receipt exceeds its hard byte limit")
    # The receipt is the commit marker, so it is always published second.
    _atomic_write_at(session.task_fd, payload_name, payload)
    _atomic_write_at(session.task_fd, receipt_name, receipt)
    published = _secure_cached_result(
        session,
        cache_key,
        key,
        record_bytes,
    )
    if published is None or published.sha256 != digest:
        raise OfflineJobCacheError(
            f"task {_task_file_stem(key)} cache publication did not authenticate"
        )
    return TaskResult(
        key=published.key,
        payload_path=published.payload_path,
        receipt_path=published.receipt_path,
        record_count=published.record_count,
        byte_length=published.byte_length,
        sha256=published.sha256,
        reused=False,
    )


def _default_executor(max_workers: int) -> Executor:
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
    )


def run_job(
    spec: JobSpec,
    producer: Producer,
    cache_root: Path,
    *,
    jobs: int = 1,
    max_in_flight: int | None = None,
    executor_factory: ExecutorFactory | None = None,
) -> JobRun:
    """Execute/reuse tasks through one anchored no-symlink cache session.

    Workers only evaluate the producer and return bounded bytes.  The parent
    process owns every cache descriptor, task lock, stable read, and atomic
    publication.  A valid receipt and matching payload are the only resume
    authority; ordinary corrupt regular entries are recomputed, while symlink,
    directory-identity, and read-time mutation attacks fail closed.
    """

    spec_document = _validated_job_spec_document(spec)
    expected_spec_bytes = canonical_json_bytes(spec_document)
    tasks = object.__getattribute__(spec, "tasks")
    task_count = len(tasks)
    if task_count > MAXIMUM_TASK_COUNT:
        raise ValueError("task count exceeds the hard generic-job limit")
    record_bytes = object.__getattribute__(spec, "record_bytes")
    if not callable(producer):
        raise TypeError("producer must be callable")
    if type(cache_root) is not _PATH_TYPE or not cache_root.is_absolute():
        raise TypeError("cache_root must be an exact absolute platform Path")
    if type(jobs) is not int or jobs < 1:
        raise ValueError("jobs must be a positive exact integer")
    maximum_jobs = min(MAXIMUM_WORKERS, task_count)
    if jobs > maximum_jobs:
        raise ValueError(
            "jobs must not exceed min(MAXIMUM_WORKERS, task_count) "
            f"({maximum_jobs})"
        )
    if max_in_flight is None:
        max_in_flight = min(task_count, 2 * jobs)
    elif type(max_in_flight) is not int or max_in_flight < 1:
        raise ValueError("max_in_flight must be a positive exact integer")
    maximum_in_flight = min(MAXIMUM_IN_FLIGHT_TASKS, task_count)
    if max_in_flight > maximum_in_flight:
        raise ValueError(
            "max_in_flight must not exceed "
            "min(MAXIMUM_IN_FLIGHT_TASKS, task_count) "
            f"({maximum_in_flight})"
        )
    if executor_factory is not None and not callable(executor_factory):
        raise TypeError("executor_factory must be callable or None")

    # Compute and bound all metadata before the cache root is opened or the
    # producer becomes reachable.
    cache_key = hashlib.sha256(expected_spec_bytes).hexdigest()
    job_document = {"jobKey": cache_key, "spec": spec_document}
    expected_job_bytes = canonical_json_bytes(job_document)
    if len(expected_job_bytes) > MAXIMUM_JOB_DOCUMENT_BYTES:
        raise ValueError("job document exceeds the hard metadata byte limit")

    completed_run: JobRun | None = None
    with _secure_cache_session(cache_root, cache_key) as session:
        _write_job_document(session, expected_job_bytes)
        _verify_secure_session_path(session, cache_key)

        results: list[TaskResult] = []
        missing: list[TaskKey] = []
        for key in tasks:
            cached = _secure_cached_result(
                session,
                cache_key,
                key,
                record_bytes,
            )
            if cached is None:
                missing.append(key)
            else:
                results.append(cached)

        maximum_observed = 0
        if missing and jobs == 1:
            for key in missing:
                lock_fd = _acquire_task_lock(session, key)
                try:
                    cached = _secure_cached_result(
                        session,
                        cache_key,
                        key,
                        record_bytes,
                    )
                    if cached is not None:
                        results.append(cached)
                        continue
                    _verify_secure_session_path(session, cache_key)
                    _verify_lock_path(session, key, lock_fd)
                    payload = _produce_task_payload(
                        producer,
                        spec,
                        key,
                        record_bytes,
                    )
                    payload = _validated_payload(payload, record_bytes, key)
                    _require_job_spec_snapshot(spec, expected_spec_bytes)
                    _verify_secure_session_path(session, cache_key)
                    _verify_lock_path(session, key, lock_fd)
                    results.append(
                        _publish_task_payload(
                            session,
                            cache_key,
                            record_bytes,
                            key,
                            payload,
                        )
                    )
                    _verify_lock_path(session, key, lock_fd)
                    maximum_observed = 1
                finally:
                    _release_task_lock(lock_fd)
        elif missing:
            factory = executor_factory or _default_executor
            pending: dict[Future[bytes], tuple[TaskKey, int]] = {}
            iterator = iter(missing)
            with factory(jobs) as executor:

                def release_pending() -> None:
                    for future in pending:
                        future.cancel()
                    # Keep every persistent lock until its producer has stopped;
                    # otherwise another runner could execute the same key while
                    # an uncancellable worker is still active.
                    for future, (_key, descriptor) in tuple(pending.items()):
                        try:
                            future.result()
                        except BaseException:
                            pass
                        _release_task_lock(descriptor)
                    pending.clear()

                def fill() -> None:
                    nonlocal maximum_observed
                    while len(pending) < max_in_flight:
                        try:
                            key = next(iterator)
                        except StopIteration:
                            break
                        lock_fd = _acquire_task_lock(session, key)
                        try:
                            cached = _secure_cached_result(
                                session,
                                cache_key,
                                key,
                                record_bytes,
                            )
                            if cached is not None:
                                results.append(cached)
                                _release_task_lock(lock_fd)
                                lock_fd = -1
                                continue
                            _verify_secure_session_path(session, cache_key)
                            _verify_lock_path(session, key, lock_fd)
                            future = executor.submit(
                                _produce_task_payload,
                                producer,
                                spec,
                                key,
                                record_bytes,
                            )
                            pending[future] = (key, lock_fd)
                            lock_fd = -1
                            maximum_observed = max(
                                maximum_observed,
                                len(pending),
                            )
                        finally:
                            if lock_fd >= 0:
                                _release_task_lock(lock_fd)

                try:
                    fill()
                    while pending:
                        completed, _remaining = wait(
                            pending,
                            return_when=FIRST_COMPLETED,
                        )
                        for future in sorted(
                            completed,
                            key=lambda item: pending[item][0],
                        ):
                            key, lock_fd = pending.pop(future)
                            try:
                                payload = _validated_payload(
                                    future.result(),
                                    record_bytes,
                                    key,
                                )
                                _require_job_spec_snapshot(
                                    spec,
                                    expected_spec_bytes,
                                )
                                _verify_secure_session_path(session, cache_key)
                                _verify_lock_path(session, key, lock_fd)
                                results.append(
                                    _publish_task_payload(
                                        session,
                                        cache_key,
                                        record_bytes,
                                        key,
                                        payload,
                                    )
                                )
                                _verify_lock_path(session, key, lock_fd)
                            finally:
                                _release_task_lock(lock_fd)
                        fill()
                except BaseException:
                    release_pending()
                    raise

        ordered = tuple(sorted(results, key=lambda item: item.key))
        if tuple(result.key for result in ordered) != tasks:
            raise OfflineJobCacheError(
                "offline runner did not produce exactly one result per task"
            )

        _require_job_spec_snapshot(spec, expected_spec_bytes)
        _require_job_document(session, expected_job_bytes)
        _verify_secure_session_path(session, cache_key)
        authenticated: list[TaskResult] = []
        for result in ordered:
            current = _secure_cached_result(
                session,
                cache_key,
                result.key,
                record_bytes,
            )
            if (
                current is None
                or current.payload_path != result.payload_path
                or current.receipt_path != result.receipt_path
                or not current.payload_path.is_absolute()
                or not current.receipt_path.is_absolute()
                or current.record_count != result.record_count
                or current.byte_length != result.byte_length
                or current.sha256 != result.sha256
            ):
                raise OfflineJobCacheError(
                    "task result changed before the secure run completed"
                )
            authenticated.append(
                TaskResult(
                    current.key,
                    current.payload_path,
                    current.receipt_path,
                    current.record_count,
                    current.byte_length,
                    current.sha256,
                    result.reused,
                )
            )
        _verify_secure_session_path(session, cache_key)
        reused = sum(result.reused for result in authenticated)
        completed_run = JobRun(
            job_key=cache_key,
            results=tuple(authenticated),
            reused_tasks=reused,
            executed_tasks=len(authenticated) - reused,
            max_in_flight_observed=maximum_observed,
        )
    if completed_run is None:
        raise AssertionError("secure cache session returned no JobRun")
    return completed_run


__all__ = [
    "InputArtifact",
    "JOB_SCHEMA",
    "JobRun",
    "JobSpec",
    "MAXIMUM_IN_FLIGHT_TASKS",
    "MAXIMUM_JOB_DOCUMENT_BYTES",
    "MAXIMUM_RECEIPT_BYTES",
    "MAXIMUM_TASK_COUNT",
    "MAXIMUM_TASK_PAYLOAD_BYTES",
    "MAXIMUM_WORKERS",
    "OfflineJobCacheError",
    "RECEIPT_SCHEMA",
    "TaskKey",
    "TaskResult",
    "canonical_json_bytes",
    "job_key",
    "run_job",
    "sha256_file",
]
