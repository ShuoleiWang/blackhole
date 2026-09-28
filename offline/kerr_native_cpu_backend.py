"""Strict ABI-v3 loader and one-resolution Kerr whole-ray adapter.

The native library is an explicitly selected numerical implementation, never a
fallback.  This module authenticates the exact library/header/source bytes,
checks the ABI and floating-point runtime contract, invokes one whole ray, and
reconstructs the repository's existing immutable geodesic evidence types.

It deliberately does not own launch construction, fine/coarse comparison,
receiver physics, public descriptors, cache records, or kernel reduction.
"""

from __future__ import annotations

import ctypes
import _ctypes
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import struct
import subprocess
import sys
import threading
from typing import Any, Final, Iterator, Mapping, Protocol, runtime_checkable

from offline.geodesic import (
    AuthenticatedInitialMultiSurfaceContact,
    ClassifiedMultiInteriorSurfaceCrossing,
    HamiltonianState,
    InteriorSurfaceDecision,
    MultiInteriorSurfaceTrace,
    RayPathSegment,
    RayTraceOptions,
    RayTraceResult,
    RecordedSurfaceCrossing,
    SurfaceEventOptions,
)
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_finite_thickness import LOWER, UPPER
from offline.kerr_finite_thickness_surface import (
    FINITE_THICKNESS_SURFACE_IDS,
    LOWER_SURFACE_ID,
    LOWER_TARGET_ID,
    UPPER_SURFACE_ID,
    UPPER_TARGET_ID,
    KerrFiniteThicknessMultiSurface,
)
from offline.kerr_returning_radiation_rays import (
    PLUNGE_LOWER_TARGET_ID,
    PLUNGE_OUTCOME,
    PLUNGE_UPPER_TARGET_ID,
    RETURNED_OUTCOME,
)
import offline.kerr_returning_radiation_rays as _rays


ABI_VERSION: Final = 3
IMPLEMENTATION_ID: Final = "strict-kerr-cpu-whole-ray/abi-v3"
_PATH_TYPE: Final = type(Path())
_MODULE_ROOT: Final = Path(__file__).absolute().parents[1]
DEFAULT_HEADER_PATH: Final = (
    _MODULE_ROOT / "native/cpu/include/blackhole_cpu.h"
).absolute()
DEFAULT_SOURCE_PATH: Final = (
    _MODULE_ROOT / "native/cpu/src/blackhole_cpu.c"
).absolute()
DEFAULT_MAKEFILE_PATH: Final = (_MODULE_ROOT / "native/cpu/Makefile").absolute()
DEFAULT_AUDIT_TOOL_PATH: Final = (
    _MODULE_ROOT / "native/cpu/tools/audit_binary.py"
).absolute()
_BACKEND_SOURCE_PATH: Final = Path(__file__).absolute()
_WRAPPER_SOURCE_PATH: Final = (
    _MODULE_ROOT / "offline/kerr_returning_radiation_native_cpu.py"
).absolute()

BH_CPU_OK: Final = 0
BH_CPU_INVALID_ARGUMENT: Final = 1
BH_CPU_INVALID_FP_ENVIRONMENT: Final = 2
BH_CPU_NONFINITE_INPUT: Final = 3
BH_CPU_NUMERIC_OVERFLOW: Final = 4
BH_CPU_INCONSISTENT_METRIC: Final = 5
BH_CPU_TOO_MANY_TERMS: Final = 6
BH_CPU_GUARDED_SINGULARITY: Final = 7
BH_CPU_CAPACITY_EXCEEDED: Final = 8

_OUTCOME_NONE: Final = 0
_OUTCOME_CAPTURED: Final = 1
_OUTCOME_ESCAPED: Final = 2
_OUTCOME_RETURNED: Final = 3
_OUTCOME_PLUNGE: Final = 4
_OUTCOME_INTEGRATOR_FAILURE: Final = 5
_OUTCOME_UNRESOLVED: Final = 6
_OUTCOME_COMPLETED: Final = 7

_TARGET_NONE: Final = 0
_TARGET_STRETCHED_HORIZON: Final = 1
_TARGET_ESCAPE: Final = 2
_TARGET_LOWER_FACE: Final = 3
_TARGET_UPPER_FACE: Final = 4
_TARGET_LOWER_PLUNGE: Final = 5
_TARGET_UPPER_PLUNGE: Final = 6

_SURFACE_LOWER: Final = 0
_SURFACE_UPPER: Final = 1

_CLASSIFICATION_NONE: Final = 0
_CLASSIFICATION_INSIDE_ISCO: Final = 1
_CLASSIFICATION_OUTSIDE_OUTER: Final = 2
_CLASSIFICATION_OUTWARD_LOWER_PLUNGE: Final = 3
_CLASSIFICATION_OUTWARD_UPPER_PLUNGE: Final = 4
_CLASSIFICATION_SUBSEQUENT_LOWER: Final = 5
_CLASSIFICATION_SUBSEQUENT_UPPER: Final = 6
_CLASSIFICATION_INWARD_LOWER_PLUNGE: Final = 7
_CLASSIFICATION_INWARD_UPPER_PLUNGE: Final = 8

_FAILURE_NONE: Final = 0
_FAILURE_REASONS: Final[Mapping[int, str]] = {
    1: "initial covector is not null within the declared residual limit",
    2: "minimum affine step reached",
    3: "accepted-step budget exhausted",
    4: "rejected-step budget exhausted",
    5: "event-refinement budget exhausted",
    6: "surface-refinement budget exhausted",
    7: "native surface-event certification failed",
    8: "native Kerr whole-ray numerical evaluation failed",
    9: "null residual exceeded declared limit",
    10: "affine-parameter budget exhausted",
}


class KerrNativeCpuError(RuntimeError):
    """Base fail-closed native CPU adapter error."""


class KerrNativeCpuAuthenticationError(KerrNativeCpuError):
    """Raised when an artifact, ABI, or runtime contract is not exact."""


class KerrNativeCpuCapacityError(KerrNativeCpuError):
    """Raised when caller-owned output capacity is insufficient."""


class KerrNativeCpuExecutionError(KerrNativeCpuError):
    """Raised when the native call does not return one authenticated result."""


class _RuntimeContract(ctypes.Structure):
    _fields_ = (
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("double_size", ctypes.c_uint32),
        ("double_mantissa_bits", ctypes.c_uint32),
        ("float_radix", ctypes.c_uint32),
        ("float_evaluation_method", ctypes.c_int32),
        ("rounding_mode", ctypes.c_int32),
        ("iec_60559_binary64", ctypes.c_uint32),
        ("fast_math_enabled", ctypes.c_uint32),
        ("little_endian", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    )


class _KerrRayOptions(ctypes.Structure):
    _fields_ = (
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("absolute_tolerance", ctypes.c_double),
        ("relative_tolerance", ctypes.c_double),
        ("initial_step", ctypes.c_double),
        ("minimum_step", ctypes.c_double),
        ("maximum_step", ctypes.c_double),
        ("maximum_affine_length", ctypes.c_double),
        ("null_residual_limit", ctypes.c_double),
        ("metric_interpolation_error_limit", ctypes.c_double),
        ("event_value_tolerance", ctypes.c_double),
        ("event_affine_tolerance", ctypes.c_double),
        ("maximum_accepted_steps", ctypes.c_uint64),
        ("maximum_rejected_steps", ctypes.c_uint64),
        ("event_maximum_iterations", ctypes.c_uint32),
        ("record_path", ctypes.c_uint32),
    )


class _KerrSurfaceOptions(ctypes.Structure):
    _fields_ = (
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("absolute_tolerance", ctypes.c_double),
        ("relative_tolerance", ctypes.c_double),
        ("null_residual_limit", ctypes.c_double),
        ("metric_interpolation_error_limit", ctypes.c_double),
        ("surface_value_tolerance", ctypes.c_double),
        ("affine_tolerance", ctypes.c_double),
        ("maximum_reintegrations", ctypes.c_uint64),
        ("maximum_iterations", ctypes.c_uint32),
        ("subdivisions_per_segment", ctypes.c_uint32),
    )


class _KerrReturningModel(ctypes.Structure):
    _fields_ = (
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("mass_m", ctypes.c_double),
        ("spin_a_m", ctypes.c_double),
        ("singularity_guard_m", ctypes.c_double),
        ("capture_radius_m", ctypes.c_double),
        ("escape_radius_m", ctypes.c_double),
        ("isco_radius_over_mass", ctypes.c_double),
        ("outer_radius_over_mass", ctypes.c_double),
        ("asymptotic_pressure_scale_height_over_mass", ctypes.c_double),
        ("emitting_surface_id", ctypes.c_uint32),
        ("initial_contact_side", ctypes.c_int32),
    )


class _RayPathSegment(ctypes.Structure):
    _fields_ = (
        ("start", ctypes.c_double * 8),
        ("end", ctypes.c_double * 8),
        ("midpoint", ctypes.c_double * 8),
        ("affine_length", ctypes.c_double),
        ("midpoint_null_residual", ctypes.c_double),
    )


class _RecordedSurfaceCrossing(ctypes.Structure):
    _fields_ = (
        ("state", ctypes.c_double * 8),
        ("ray_affine_length", ctypes.c_double),
        ("segment_affine_length", ctypes.c_double),
        ("surface_value", ctypes.c_double),
        ("bracket_affine_width", ctypes.c_double),
        ("segment_index", ctypes.c_uint64),
        ("iterations", ctypes.c_uint32),
        ("orientation", ctypes.c_int32),
        ("surface_id", ctypes.c_uint32),
        ("classification", ctypes.c_uint32),
        ("outcome", ctypes.c_uint32),
        ("target", ctypes.c_uint32),
    )


class _KerrRayResult(ctypes.Structure):
    _fields_ = (
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("outcome", ctypes.c_uint32),
        ("terminal_target", ctypes.c_uint32),
        ("failure", ctypes.c_uint32),
        ("topology_converged", ctypes.c_uint32),
        ("initial_contact_surface_id", ctypes.c_uint32),
        ("initial_contact_side", ctypes.c_int32),
        ("terminal_state", ctypes.c_double * 8),
        ("affine_length", ctypes.c_double),
        ("maximum_null_residual", ctypes.c_double),
        ("maximum_metric_interpolation_error", ctypes.c_double),
        ("maximum_probe_event_difference", ctypes.c_double),
        ("maximum_probe_covector_relative_difference", ctypes.c_double),
        ("initial_contact_actual_surface_value", ctypes.c_double),
        ("initial_contact_surface_value_tolerance", ctypes.c_double),
        ("accepted_steps", ctypes.c_uint64),
        ("rejected_steps", ctypes.c_uint64),
        ("segment_count", ctypes.c_uint64),
        ("crossing_count", ctypes.c_uint64),
        ("probe_reintegrations", ctypes.c_uint64),
        ("surface_value_evaluations", ctypes.c_uint64),
        ("metric_sample_evaluations", ctypes.c_uint64),
        ("hamiltonian_rhs_evaluations", ctypes.c_uint64),
    )


_STRUCT_TYPES: Final = (
    _RuntimeContract,
    _KerrRayOptions,
    _KerrSurfaceOptions,
    _KerrReturningModel,
    _RayPathSegment,
    _RecordedSurfaceCrossing,
    _KerrRayResult,
)
_ABI3_STRUCTURE_SIZES: Final[Mapping[str, int]] = {
    "RuntimeContract": 44,
    "KerrRayOptions": 112,
    "KerrSurfaceOptions": 72,
    "KerrReturningModel": 80,
    "RayPathSegment": 208,
    "RecordedSurfaceCrossing": 128,
    "KerrRayResult": 216,
}


@dataclass(frozen=True, slots=True)
class _ArtifactSnapshot:
    path: Path
    byte_length: int
    sha256: str
    stat_identity: tuple[int, int, int, int, int, int]

    def descriptor(self, logical_path: str) -> dict[str, Any]:
        if type(logical_path) is not str or not logical_path:
            raise TypeError("artifact logical_path must be a non-empty exact string")
        return {
            "artifactName": Path(logical_path).name,
            "byteLength": self.byte_length,
            "logicalPath": logical_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class KerrNativeCpuWorkDiagnostics:
    metric_sample_evaluations: int
    hamiltonian_rhs_evaluations: int
    backend_descriptor_sha256: str


@dataclass(frozen=True, slots=True)
class KerrNativeCpuOneResolution:
    ray: RayTraceResult
    work: KerrNativeCpuWorkDiagnostics


@runtime_checkable
class KerrNativeCpuRuntimeBinding(Protocol):
    """Path-explicit, path-free-identity request for one worker backend."""

    library_path: Path
    library_sha256: str
    backend_descriptor_sha256: str
    segment_capacity: int
    crossing_capacity: int

    def path_free_descriptor(self) -> Mapping[str, Any]:
        """Return the scientific/runtime identity without the host path."""


@dataclass(frozen=True, slots=True)
class FrozenKerrNativeCpuRuntimeBinding:
    """Canonical spawn-safe configuration; it never contains a live backend."""

    library_path: Path
    library_sha256: str
    backend_descriptor_sha256: str
    segment_capacity: int = 100_000
    crossing_capacity: int = 100_000

    def __post_init__(self) -> None:
        path = _exact_absolute_path(self.library_path, "runtime-binding library")
        for name in ("library_sha256", "backend_descriptor_sha256"):
            digest = getattr(self, name)
            if (
                type(digest) is not str
                or len(digest) != 64
                or digest.lower() != digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
            try:
                decoded = bytes.fromhex(digest)
            except ValueError as error:
                raise ValueError(f"{name} must be hexadecimal") from error
            if len(decoded) != 32:
                raise ValueError(f"{name} must be a lowercase SHA-256")
        for name in ("segment_capacity", "crossing_capacity"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be an exact non-negative integer")
        object.__setattr__(self, "library_path", path)

    def path_free_descriptor(self) -> Mapping[str, Any]:
        return {
            "backendDescriptorSha256": self.backend_descriptor_sha256,
            "callerOwnedCapacities": {
                "crossings": self.crossing_capacity,
                "segments": self.segment_capacity,
            },
            "implementationId": "strict-kerr-cpu-process-runtime-binding/v1",
        }

    @property
    def path_free_descriptor_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json_bytes(self.path_free_descriptor())
        ).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


_NATIVE_IMAGE_LOCK: Final = threading.RLock()
_NATIVE_IMAGE_REGISTRY_PID: int | None = None
_NATIVE_IMAGE_SHA_BY_PATH: dict[str, str] = {}
_NATIVE_IMAGE_POISONED_PATHS: set[str] = set()


def _prepare_native_image_registry() -> None:
    global _NATIVE_IMAGE_REGISTRY_PID
    process_id = os.getpid()
    if _NATIVE_IMAGE_REGISTRY_PID == process_id:
        return
    if _NATIVE_IMAGE_REGISTRY_PID is not None:
        _NATIVE_IMAGE_POISONED_PATHS.update(_NATIVE_IMAGE_SHA_BY_PATH)
    _NATIVE_IMAGE_SHA_BY_PATH.clear()
    _NATIVE_IMAGE_REGISTRY_PID = process_id


def _poison_native_image_path(path: Path) -> None:
    with _NATIVE_IMAGE_LOCK:
        _prepare_native_image_registry()
        _NATIVE_IMAGE_POISONED_PATHS.add(os.fspath(path))
        _NATIVE_IMAGE_SHA_BY_PATH.pop(os.fspath(path), None)


def _exact_absolute_path(value: Path, label: str) -> Path:
    if type(value) is not _PATH_TYPE or not value.is_absolute():
        raise TypeError(f"{label} must be an exact absolute platform Path")
    path = Path(os.path.abspath(os.fspath(value)))
    if path == path.parent:
        raise KerrNativeCpuAuthenticationError(f"{label} cannot be the filesystem root")
    for parent in (path.parent, *path.parents[1:-1]):
        try:
            snapshot = os.lstat(parent)
        except OSError as error:
            raise KerrNativeCpuAuthenticationError(
                f"{label} has an unreadable ancestor"
            ) from error
        if stat.S_ISLNK(snapshot.st_mode) or not stat.S_ISDIR(snapshot.st_mode):
            raise KerrNativeCpuAuthenticationError(
                f"{label} has a symlink or non-directory ancestor"
            )
    return path


def _authenticate_artifact(path: Path, label: str) -> _ArtifactSnapshot:
    exact = _exact_absolute_path(path, label)
    descriptor = -1
    try:
        before_path = os.lstat(exact)
        if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
            raise KerrNativeCpuAuthenticationError(
                f"{label} must be a regular non-symlink file"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        descriptor = os.open(exact, flags)
        before = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(before_path):
            raise KerrNativeCpuAuthenticationError(f"{label} changed while opened")
        digest = hashlib.sha256()
        byte_length = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            byte_length += len(block)
        after = os.fstat(descriptor)
        after_path = os.lstat(exact)
        identity = _stat_identity(before)
        if (
            identity != _stat_identity(after)
            or identity != _stat_identity(after_path)
            or byte_length != before.st_size
        ):
            raise KerrNativeCpuAuthenticationError(
                f"{label} changed while authenticated"
            )
        return _ArtifactSnapshot(
            exact,
            byte_length,
            digest.hexdigest(),
            identity,
        )
    except OSError as error:
        raise KerrNativeCpuAuthenticationError(f"cannot authenticate {label}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _load_authenticated_native_library(
    artifact: _ArtifactSnapshot,
) -> ctypes.CDLL:
    """Load one stable image and permanently bind its path to these bytes."""

    path_key = os.fspath(artifact.path)
    with _NATIVE_IMAGE_LOCK:
        _prepare_native_image_registry()
        if path_key in _NATIVE_IMAGE_POISONED_PATHS:
            raise KerrNativeCpuAuthenticationError(
                "native library path was poisoned after image or artifact drift"
            )
        previous_sha = _NATIVE_IMAGE_SHA_BY_PATH.get(path_key)
        if previous_sha is not None and previous_sha != artifact.sha256:
            _NATIVE_IMAGE_POISONED_PATHS.add(path_key)
            _NATIVE_IMAGE_SHA_BY_PATH.pop(path_key, None)
            raise KerrNativeCpuAuthenticationError(
                "native library path cannot load different bytes in one process"
            )
        try:
            library = ctypes.CDLL(str(artifact.path))
        except (OSError, TypeError, ValueError) as error:
            raise KerrNativeCpuAuthenticationError(
                "native library could not be loaded"
            ) from error
        try:
            after = _authenticate_artifact(artifact.path, "native library")
        except BaseException:
            _NATIVE_IMAGE_POISONED_PATHS.add(path_key)
            _NATIVE_IMAGE_SHA_BY_PATH.pop(path_key, None)
            raise
        if after != artifact:
            _NATIVE_IMAGE_POISONED_PATHS.add(path_key)
            _NATIVE_IMAGE_SHA_BY_PATH.pop(path_key, None)
            raise KerrNativeCpuAuthenticationError(
                "native library changed while its image was loaded"
            )
        _NATIVE_IMAGE_SHA_BY_PATH[path_key] = artifact.sha256
        return library


def _canonical_json_bytes(value: Any) -> bytes:
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


def _validated_binary_audit_report(value: str) -> tuple[str, dict[str, Any]]:
    prefix = "binary audit passed: "
    if type(value) is not str or not value.startswith(prefix):
        raise KerrNativeCpuAuthenticationError(
            "native binary audit returned an unsupported report"
        )
    encoded = value[len(prefix) :]
    try:
        document = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise KerrNativeCpuAuthenticationError(
            "native binary audit report is not exact JSON"
        ) from error
    if type(document) is not dict or set(document) != {
        "architecture",
        "exports",
        "fusedFpInstructions",
        "ltoPayloads",
        "tools",
    }:
        raise KerrNativeCpuAuthenticationError(
            "native binary audit report has an unsupported schema"
        )
    tools = document["tools"]
    if (
        document["architecture"] != platform.machine()
        or document["exports"] != 16
        or document["fusedFpInstructions"] != 0
        or document["ltoPayloads"] != 0
        or type(tools) is not dict
        or set(tools) != {"file", "nm", "otool"}
    ):
        raise KerrNativeCpuAuthenticationError(
            "native binary audit report differs from the strict contract"
        )
    for name, artifact in tools.items():
        if (
            type(name) is not str
            or type(artifact) is not dict
            or set(artifact) != {"artifactName", "byteLength", "sha256"}
            or artifact["artifactName"] != name
            or type(artifact["byteLength"]) is not int
            or artifact["byteLength"] <= 0
            or type(artifact["sha256"]) is not str
            or len(artifact["sha256"]) != 64
            or artifact["sha256"].lower() != artifact["sha256"]
        ):
            raise KerrNativeCpuAuthenticationError(
                "native binary audit tool artifact is malformed"
            )
        try:
            decoded = bytes.fromhex(artifact["sha256"])
        except ValueError as error:
            raise KerrNativeCpuAuthenticationError(
                "native binary audit tool SHA-256 is malformed"
            ) from error
        if len(decoded) != 32:
            raise KerrNativeCpuAuthenticationError(
                "native binary audit tool SHA-256 is malformed"
            )
    canonical = prefix + json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if value != canonical:
        raise KerrNativeCpuAuthenticationError(
            "native binary audit report is not canonical"
        )
    return canonical, document


def _state(values: Any) -> HamiltonianState:
    packed = tuple(float(values[index]) for index in range(8))
    return HamiltonianState.unpack(packed)


def _ray_options(value: RayTraceOptions) -> _KerrRayOptions:
    if type(value) is not RayTraceOptions:
        raise TypeError("ray_options must be the exact RayTraceOptions")
    return _KerrRayOptions(
        ABI_VERSION,
        ctypes.sizeof(_KerrRayOptions),
        value.absolute_tolerance,
        value.relative_tolerance,
        value.initial_step,
        value.minimum_step,
        value.maximum_step,
        value.maximum_affine_length,
        value.null_residual_limit,
        value.metric_interpolation_error_limit,
        value.event_value_tolerance,
        value.event_affine_tolerance,
        value.maximum_accepted_steps,
        value.maximum_rejected_steps,
        value.event_maximum_iterations,
        int(value.record_path),
    )


def _surface_options(value: SurfaceEventOptions) -> _KerrSurfaceOptions:
    if type(value) is not SurfaceEventOptions:
        raise TypeError("surface_options must be the exact SurfaceEventOptions")
    return _KerrSurfaceOptions(
        ABI_VERSION,
        ctypes.sizeof(_KerrSurfaceOptions),
        value.absolute_tolerance,
        value.relative_tolerance,
        value.null_residual_limit,
        value.metric_interpolation_error_limit,
        value.surface_value_tolerance,
        value.affine_tolerance,
        value.maximum_reintegrations,
        value.maximum_iterations,
        value.subdivisions_per_segment,
    )


def _surface_id(value: str) -> int:
    if type(value) is not str or value not in (LOWER, UPPER):
        raise ValueError("emitting_face must be exact 'lower' or 'upper'")
    return _SURFACE_LOWER if value == LOWER else _SURFACE_UPPER


def _surface_name(value: int) -> str:
    if value == _SURFACE_LOWER:
        return LOWER_SURFACE_ID
    if value == _SURFACE_UPPER:
        return UPPER_SURFACE_ID
    raise KerrNativeCpuExecutionError("native crossing names an unknown surface")


def _outcome(value: int) -> str:
    mapping = {
        _OUTCOME_CAPTURED: "captured",
        _OUTCOME_ESCAPED: "escaped",
        _OUTCOME_RETURNED: RETURNED_OUTCOME,
        _OUTCOME_PLUNGE: PLUNGE_OUTCOME,
        _OUTCOME_INTEGRATOR_FAILURE: "integrator-failure",
        _OUTCOME_UNRESOLVED: "unresolved",
        _OUTCOME_COMPLETED: "completed",
    }
    try:
        return mapping[value]
    except KeyError as error:
        raise KerrNativeCpuExecutionError("native ray has an unknown outcome") from error


def _target(value: int, termination: KerrOblateTermination) -> str | None:
    mapping = {
        _TARGET_NONE: None,
        _TARGET_STRETCHED_HORIZON: termination.capture_target_id,
        _TARGET_ESCAPE: termination.escape_target_id,
        _TARGET_LOWER_FACE: LOWER_TARGET_ID,
        _TARGET_UPPER_FACE: UPPER_TARGET_ID,
        _TARGET_LOWER_PLUNGE: PLUNGE_LOWER_TARGET_ID,
        _TARGET_UPPER_PLUNGE: PLUNGE_UPPER_TARGET_ID,
    }
    try:
        return mapping[value]
    except KeyError as error:
        raise KerrNativeCpuExecutionError("native ray has an unknown target") from error


def _classification(value: int) -> str:
    mapping = {
        _CLASSIFICATION_INSIDE_ISCO: "inside-isco-transparent",
        _CLASSIFICATION_OUTSIDE_OUTER: "outside-outer-radius-transparent",
        _CLASSIFICATION_OUTWARD_LOWER_PLUNGE: (
            "outward-lower-continuation-plunge-exit-transparent"
        ),
        _CLASSIFICATION_OUTWARD_UPPER_PLUNGE: (
            "outward-upper-continuation-plunge-exit-transparent"
        ),
        _CLASSIFICATION_SUBSEQUENT_LOWER: (
            "subsequent-lower-photosphere-contact"
        ),
        _CLASSIFICATION_SUBSEQUENT_UPPER: (
            "subsequent-upper-photosphere-contact"
        ),
        _CLASSIFICATION_INWARD_LOWER_PLUNGE: (
            "inward-lower-continuation-plunge-entry"
        ),
        _CLASSIFICATION_INWARD_UPPER_PLUNGE: (
            "inward-upper-continuation-plunge-entry"
        ),
    }
    if value == _CLASSIFICATION_NONE:
        raise KerrNativeCpuExecutionError("native crossing lacks a classification")
    try:
        return mapping[value]
    except KeyError as error:
        raise KerrNativeCpuExecutionError(
            "native crossing has an unknown classification"
        ) from error


class StrictKerrCpuBackend:
    """One explicitly authenticated ABI-v3 native backend instance."""

    def __init__(
        self,
        library_path: Path,
        *,
        header_path: Path = DEFAULT_HEADER_PATH,
        source_path: Path = DEFAULT_SOURCE_PATH,
        makefile_path: Path = DEFAULT_MAKEFILE_PATH,
        audit_tool_path: Path = DEFAULT_AUDIT_TOOL_PATH,
        segment_capacity: int = 100_000,
        crossing_capacity: int = 100_000,
    ) -> None:
        for name, value in (
            ("segment_capacity", segment_capacity),
            ("crossing_capacity", crossing_capacity),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be an exact non-negative integer")
        self._segment_capacity = segment_capacity
        self._crossing_capacity = crossing_capacity
        self._library_artifact = _authenticate_artifact(library_path, "native library")
        self._header_artifact = _authenticate_artifact(header_path, "native ABI header")
        self._source_artifact = _authenticate_artifact(source_path, "native source")
        self._makefile_artifact = _authenticate_artifact(
            makefile_path, "native build Makefile"
        )
        self._audit_tool_artifact = _authenticate_artifact(
            audit_tool_path, "native binary audit tool"
        )
        self._backend_source_artifact = _authenticate_artifact(
            _BACKEND_SOURCE_PATH, "native Python backend source"
        )
        self._wrapper_source_artifact = _authenticate_artifact(
            _WRAPPER_SOURCE_PATH, "native returning-ray wrapper source"
        )
        try:
            python_runtime = Path(sys.executable).resolve(strict=True).absolute()
            ctypes_runtime = Path(_ctypes.__file__).resolve(strict=True).absolute()
            math_runtime = Path(math.__file__).resolve(strict=True).absolute()
        except (AttributeError, OSError, TypeError, ValueError) as error:
            raise KerrNativeCpuAuthenticationError(
                "cannot resolve Python numerical runtime artifacts"
            ) from error
        self._python_runtime_artifact = _authenticate_artifact(
            python_runtime, "Python executable"
        )
        self._ctypes_runtime_artifact = _authenticate_artifact(
            ctypes_runtime, "_ctypes runtime extension"
        )
        self._math_runtime_artifact = _authenticate_artifact(
            math_runtime, "math runtime extension"
        )
        actual_structure_sizes = {
            value.__name__.removeprefix("_"): ctypes.sizeof(value)
            for value in _STRUCT_TYPES
        }
        if actual_structure_sizes != _ABI3_STRUCTURE_SIZES:
            raise KerrNativeCpuAuthenticationError(
                "ctypes layouts differ from the exact ABI-v3 structure sizes"
            )
        try:
            audit = subprocess.run(
                (
                    str(self._python_runtime_artifact.path),
                    "-I",
                    "-S",
                    str(self._audit_tool_artifact.path),
                    str(self._library_artifact.path),
                ),
                check=False,
                env={
                    "LANG": "C",
                    "LC_ALL": "C",
                    "PATH": "",
                    "TMPDIR": "/private/tmp",
                },
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60.0,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise KerrNativeCpuAuthenticationError(
                "native binary audit could not run"
            ) from error
        if audit.returncode != 0:
            raise KerrNativeCpuAuthenticationError(
                "native binary failed its bound export/FP/LTO audit: "
                + audit.stdout.strip()
            )
        audit_report, _audit_document = _validated_binary_audit_report(
            audit.stdout.strip()
        )
        try:
            library = _load_authenticated_native_library(
                self._library_artifact
            )
            library.bh_cpu_get_runtime_contract.argtypes = (
                ctypes.POINTER(_RuntimeContract),
            )
            library.bh_cpu_get_runtime_contract.restype = ctypes.c_int
            library.bh_cpu_require_strict_fp.argtypes = ()
            library.bh_cpu_require_strict_fp.restype = ctypes.c_int
            library.bh_cpu_trace_kerr_returning_ray.argtypes = (
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(_KerrReturningModel),
                ctypes.POINTER(_KerrRayOptions),
                ctypes.POINTER(_KerrSurfaceOptions),
                ctypes.POINTER(_RayPathSegment),
                ctypes.c_size_t,
                ctypes.POINTER(_RecordedSurfaceCrossing),
                ctypes.c_size_t,
                ctypes.POINTER(_KerrRayResult),
            )
            library.bh_cpu_trace_kerr_returning_ray.restype = ctypes.c_int
            library.bh_cpu_status_string.argtypes = (ctypes.c_int,)
            library.bh_cpu_status_string.restype = ctypes.c_char_p
        except (AttributeError, OSError, TypeError, ValueError) as error:
            raise KerrNativeCpuAuthenticationError(
                "native library does not expose the exact ABI-v3 surface"
            ) from error
        contract = _RuntimeContract()
        status = library.bh_cpu_get_runtime_contract(ctypes.byref(contract))
        if status != BH_CPU_OK or library.bh_cpu_require_strict_fp() != BH_CPU_OK:
            raise KerrNativeCpuAuthenticationError(
                "native library rejected the strict floating-point runtime"
            )
        expected_contract = (
            contract.abi_version == ABI_VERSION
            and contract.struct_size == ctypes.sizeof(_RuntimeContract)
            and contract.double_size == ctypes.sizeof(ctypes.c_double) == 8
            and contract.double_mantissa_bits == sys.float_info.mant_dig == 53
            and contract.float_radix == sys.float_info.radix == 2
            and contract.float_evaluation_method == 0
            and contract.iec_60559_binary64 == 1
            and contract.fast_math_enabled == 0
            and contract.little_endian == int(sys.byteorder == "little") == 1
            and contract.reserved == 0
        )
        if not expected_contract:
            raise KerrNativeCpuAuthenticationError(
                "native ABI/runtime descriptor is not strict binary64 ABI v3"
            )
        self._library = library
        self._contract = contract
        self._process_id = os.getpid()
        descriptor = {
            "abiVersion": ABI_VERSION,
            "artifacts": {
                "auditTool": self._audit_tool_artifact.descriptor(
                    "native/cpu/tools/audit_binary.py"
                ),
                "backendPython": self._backend_source_artifact.descriptor(
                    "offline/kerr_native_cpu_backend.py"
                ),
                "ctypesRuntime": self._ctypes_runtime_artifact.descriptor(
                    "runtime/_ctypes-extension"
                ),
                "header": self._header_artifact.descriptor(
                    "native/cpu/include/blackhole_cpu.h"
                ),
                "library": self._library_artifact.descriptor(
                    "runtime/libblackhole_cpu.dylib"
                ),
                "makefile": self._makefile_artifact.descriptor(
                    "native/cpu/Makefile"
                ),
                "mathRuntime": self._math_runtime_artifact.descriptor(
                    "runtime/math-extension"
                ),
                "pythonRuntime": self._python_runtime_artifact.descriptor(
                    "runtime/python-executable"
                ),
                "source": self._source_artifact.descriptor(
                    "native/cpu/src/blackhole_cpu.c"
                ),
                "wrapperPython": self._wrapper_source_artifact.descriptor(
                    "offline/kerr_returning_radiation_native_cpu.py"
                ),
            },
            "binaryAudit": {
                "report": audit_report,
                "reportSha256": hashlib.sha256(
                    audit_report.encode("utf-8")
                ).hexdigest(),
            },
            "callerOwnedCapacities": {
                "crossings": crossing_capacity,
                "segments": segment_capacity,
            },
            "implementationId": IMPLEMENTATION_ID,
            "platform": {
                "architectureBits": 8 * struct.calcsize("P"),
                "byteOrder": sys.byteorder,
                "machine": platform.machine(),
                "operatingSystem": platform.system(),
                "operatingSystemRelease": platform.release(),
                "pythonImplementation": platform.python_implementation(),
                "pythonVersion": platform.python_version(),
            },
            "runtimeContract": {
                name: int(getattr(contract, name))
                for name, _ctype in _RuntimeContract._fields_
            },
            "structureSizes": actual_structure_sizes,
        }
        self._descriptor_json = _canonical_json_bytes(descriptor)
        self._descriptor_sha256 = hashlib.sha256(self._descriptor_json).hexdigest()

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    @property
    def library_sha256(self) -> str:
        return self._library_artifact.sha256

    def model_descriptor(self) -> dict[str, Any]:
        return json.loads(self._descriptor_json)

    def runtime_binding(self) -> FrozenKerrNativeCpuRuntimeBinding:
        """Return a path-explicit request containing no live native state."""

        self.revalidate()
        return FrozenKerrNativeCpuRuntimeBinding(
            library_path=self._library_artifact.path,
            library_sha256=self._library_artifact.sha256,
            backend_descriptor_sha256=self._descriptor_sha256,
            segment_capacity=self._segment_capacity,
            crossing_capacity=self._crossing_capacity,
        )

    def __reduce__(self) -> Any:
        raise TypeError("live native CPU backends cannot be pickled")

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise TypeError("live native CPU backends cannot be pickled")

    def __copy__(self) -> Any:
        raise TypeError("live native CPU backends cannot be copied")

    def __deepcopy__(self, memo: Any) -> Any:
        del memo
        raise TypeError("live native CPU backends cannot be copied")

    def __getstate__(self) -> Any:
        raise TypeError("live native CPU backends cannot be serialized")

    def revalidate(self) -> None:
        """Fail if the process or any bound artifact changed after loading."""

        if os.getpid() != self._process_id:
            _poison_native_image_path(self._library_artifact.path)
            raise KerrNativeCpuAuthenticationError(
                "native backend may not be inherited across fork"
            )
        for artifact, label in (
            (self._library_artifact, "native library"),
            (self._header_artifact, "native ABI header"),
            (self._source_artifact, "native source"),
            (self._makefile_artifact, "native build Makefile"),
            (self._audit_tool_artifact, "native binary audit tool"),
            (self._backend_source_artifact, "native Python backend source"),
            (self._wrapper_source_artifact, "native returning-ray wrapper source"),
            (self._python_runtime_artifact, "Python executable"),
            (self._ctypes_runtime_artifact, "_ctypes runtime extension"),
            (self._math_runtime_artifact, "math runtime extension"),
        ):
            try:
                current = os.lstat(artifact.path)
            except OSError as error:
                if artifact is self._library_artifact:
                    _poison_native_image_path(artifact.path)
                raise KerrNativeCpuAuthenticationError(
                    f"cannot revalidate {label}"
                ) from error
            if _stat_identity(current) != artifact.stat_identity:
                if artifact is self._library_artifact:
                    _poison_native_image_path(artifact.path)
                raise KerrNativeCpuAuthenticationError(
                    f"{label} changed after backend construction"
                )
        if self._library.bh_cpu_require_strict_fp() != BH_CPU_OK:
            raise KerrNativeCpuAuthenticationError(
                "calling thread no longer satisfies the strict FP contract"
            )

    def _status_message(self, status: int) -> str:
        raw = self._library.bh_cpu_status_string(status)
        if not raw:
            return f"unknown native status {status}"
        try:
            return raw.decode("ascii")
        except (UnicodeError, AttributeError):
            return f"non-ASCII native status {status}"

    def trace_one_resolution(
        self,
        initial_state: HamiltonianState,
        surface: KerrFiniteThicknessMultiSurface,
        termination: KerrOblateTermination,
        ray_options: RayTraceOptions,
        surface_options: SurfaceEventOptions,
        *,
        emitting_face: str,
    ) -> KerrNativeCpuOneResolution:
        """Trace exactly one fine *or* coarse resolution, without fallback."""

        self.revalidate()
        if type(initial_state) is not HamiltonianState:
            raise TypeError("initial_state must be the exact HamiltonianState")
        if type(surface) is not KerrFiniteThicknessMultiSurface:
            raise TypeError("surface must be the exact KerrFiniteThicknessMultiSurface")
        if type(surface.metric) is not KerrKerrSchildMetric:
            raise TypeError("native backend requires exact analytic Kerr metric")
        if type(termination) is not KerrOblateTermination:
            raise TypeError("termination must be the exact KerrOblateTermination")
        surface = _rays._validated_surface(surface)
        termination = _rays._validated_termination(
            termination,
            surface,
            initial_state,
        )
        ray_options, surface_options = _rays._validated_options(
            surface.metric,
            ray_options,
            surface_options,
        )
        if termination.capture_target_id != "analytic-kerr-stretched-horizon":
            raise ValueError("native ABI v3 supports only the stretched Kerr horizon")
        if termination.escape_target_id != "analytic-kerr-escape-worldtube":
            raise ValueError("native ABI v3 requires the exact Kerr escape target")
        if termination.spin_a_m.hex() != surface.metric.spin_a_m.hex():
            raise ValueError("termination spin differs from the surface metric")
        if surface.calibration.outer_radius_over_mass <= (
            surface.calibration.isco_radius_over_mass
        ):
            raise ValueError("native surface calibration is stale")
        native_ray_options = _ray_options(ray_options)
        native_surface_options = _surface_options(surface_options)
        native_model = _KerrReturningModel(
            ABI_VERSION,
            ctypes.sizeof(_KerrReturningModel),
            surface.metric.mass_m,
            surface.metric.spin_a_m,
            surface.metric.singularity_guard_m,
            termination.capture_radius_m,
            termination.escape_radius_m,
            surface.calibration.isco_radius_over_mass,
            surface.calibration.outer_radius_over_mass,
            surface.calibration.asymptotic_pressure_scale_height_over_mass,
            _surface_id(emitting_face),
            1,
        )
        packed = (ctypes.c_double * 8)(*initial_state.packed())
        segment_capacity = self._segment_capacity if ray_options.record_path else 0
        segments = (
            (_RayPathSegment * segment_capacity)()
            if segment_capacity > 0
            else None
        )
        crossings = (
            (_RecordedSurfaceCrossing * self._crossing_capacity)()
            if self._crossing_capacity > 0
            else None
        )
        result = _KerrRayResult()
        status = int(
            self._library.bh_cpu_trace_kerr_returning_ray(
                packed,
                ctypes.byref(native_model),
                ctypes.byref(native_ray_options),
                ctypes.byref(native_surface_options),
                segments,
                segment_capacity,
                crossings,
                self._crossing_capacity,
                ctypes.byref(result),
            )
        )
        self.revalidate()
        if status == BH_CPU_CAPACITY_EXCEEDED:
            raise KerrNativeCpuCapacityError(
                "native whole ray exceeded caller-owned segment/crossing capacity"
            )
        if status != BH_CPU_OK:
            raise KerrNativeCpuExecutionError(
                f"native whole ray failed closed: {self._status_message(status)}"
            )
        if (
            result.abi_version != ABI_VERSION
            or result.struct_size != ctypes.sizeof(_KerrRayResult)
            or result.topology_converged not in (0, 1)
            or result.segment_count > segment_capacity
            or result.crossing_count > self._crossing_capacity
        ):
            raise KerrNativeCpuExecutionError("native whole-ray result has stale ABI fields")
        if ray_options.record_path:
            if result.segment_count != result.accepted_steps:
                raise KerrNativeCpuExecutionError(
                    "native recorded segment count differs from accepted steps"
                )
        elif result.segment_count != 0:
            raise KerrNativeCpuExecutionError(
                "native unrecorded ray unexpectedly returned path segments"
            )
        expected_surface_id = _surface_id(emitting_face)
        expected_residual = surface.value(
            UPPER_SURFACE_ID if emitting_face == UPPER else LOWER_SURFACE_ID,
            initial_state,
        )
        if (
            result.initial_contact_surface_id != expected_surface_id
            or result.initial_contact_side != 1
            or result.initial_contact_actual_surface_value.hex()
            != float(expected_residual).hex()
            or result.initial_contact_surface_value_tolerance.hex()
            != surface_options.surface_value_tolerance.hex()
        ):
            raise KerrNativeCpuExecutionError(
                "native initial-contact evidence differs from Python ownership"
            )

        python_segments = tuple(
            RayPathSegment(
                start=_state(segments[index].start),
                end=_state(segments[index].end),
                midpoint=_state(segments[index].midpoint),
                affine_length=float(segments[index].affine_length),
                midpoint_null_residual=float(
                    segments[index].midpoint_null_residual
                ),
            )
            for index in range(int(result.segment_count))
        )
        entries: list[ClassifiedMultiInteriorSurfaceCrossing] = []
        assert crossings is not None or result.crossing_count == 0
        for index in range(int(result.crossing_count)):
            raw = crossings[index]
            outcome = None if raw.outcome == _OUTCOME_NONE else _outcome(raw.outcome)
            target = _target(raw.target, termination)
            if (outcome is None) != (target is None):
                raise KerrNativeCpuExecutionError(
                    "native crossing outcome/target ownership is inconsistent"
                )
            crossing = RecordedSurfaceCrossing(
                state=_state(raw.state),
                ray_affine_length=float(raw.ray_affine_length),
                segment_index=int(raw.segment_index),
                segment_affine_length=float(raw.segment_affine_length),
                orientation=int(raw.orientation),
                surface_value=float(raw.surface_value),
                bracket_affine_width=float(raw.bracket_affine_width),
                iterations=int(raw.iterations),
            )
            entries.append(
                ClassifiedMultiInteriorSurfaceCrossing(
                    surface_id=_surface_name(raw.surface_id),
                    crossing=crossing,
                    decision=InteriorSurfaceDecision(
                        _classification(raw.classification),
                        outcome,
                        target,
                    ),
                )
            )
        trace = MultiInteriorSurfaceTrace(
            surface_ids=tuple(sorted(FINITE_THICKNESS_SURFACE_IDS)),
            crossings=tuple(entries),
            base_subdivisions_per_step=surface_options.subdivisions_per_segment,
            verification_subdivisions_per_step=(
                2 * surface_options.subdivisions_per_segment
            ),
            topology_converged=bool(result.topology_converged),
            maximum_probe_event_difference=float(
                result.maximum_probe_event_difference
            ),
            maximum_probe_covector_relative_difference=float(
                result.maximum_probe_covector_relative_difference
            ),
            probe_reintegrations=int(result.probe_reintegrations),
            surface_value_evaluations=int(result.surface_value_evaluations),
            initial_contact=AuthenticatedInitialMultiSurfaceContact(
                _surface_name(result.initial_contact_surface_id),
                int(result.initial_contact_side),
                float(result.initial_contact_actual_surface_value),
                float(result.initial_contact_surface_value_tolerance),
            ),
        )
        ray_outcome = _outcome(result.outcome)
        terminal_target = _target(result.terminal_target, termination)
        failure_reason = None
        if result.failure != _FAILURE_NONE:
            failure_reason = _FAILURE_REASONS.get(
                int(result.failure),
                f"unknown native ray failure {int(result.failure)}",
            )
        elif ray_outcome in ("integrator-failure", "unresolved"):
            raise KerrNativeCpuExecutionError(
                "native failed ray omitted its failure classification"
            )
        if ray_outcome in ("captured", "escaped", RETURNED_OUTCOME, PLUNGE_OUTCOME):
            if failure_reason is not None or terminal_target is None:
                raise KerrNativeCpuExecutionError(
                    "native successful ray has inconsistent failure/target evidence"
                )
        ray = RayTraceResult(
            outcome=ray_outcome,
            terminal_state=_state(result.terminal_state),
            affine_length=float(result.affine_length),
            accepted_steps=int(result.accepted_steps),
            rejected_steps=int(result.rejected_steps),
            maximum_null_residual=float(result.maximum_null_residual),
            maximum_metric_interpolation_error=float(
                result.maximum_metric_interpolation_error
            ),
            segments=python_segments,
            failure_reason=failure_reason,
            terminal_target_id=terminal_target,
            multi_surface_trace=trace,
        )
        return KerrNativeCpuOneResolution(
            ray=ray,
            work=KerrNativeCpuWorkDiagnostics(
                metric_sample_evaluations=int(result.metric_sample_evaluations),
                hamiltonian_rhs_evaluations=int(
                    result.hamiltonian_rhs_evaluations
                ),
                backend_descriptor_sha256=self._descriptor_sha256,
            ),
        )


_PROCESS_LOCAL_BACKEND_LOCK: Final = threading.RLock()
_PROCESS_LOCAL_BACKEND_PID: int | None = None
_PROCESS_LOCAL_BACKENDS: dict[
    tuple[int, str, str, str, int, int],
    StrictKerrCpuBackend,
] = {}


def _canonical_runtime_binding(
    value: KerrNativeCpuRuntimeBinding,
) -> FrozenKerrNativeCpuRuntimeBinding:
    if not isinstance(value, KerrNativeCpuRuntimeBinding):
        raise TypeError("runtime_binding must implement KerrNativeCpuRuntimeBinding")
    try:
        rebuilt = FrozenKerrNativeCpuRuntimeBinding(
            library_path=value.library_path,
            library_sha256=value.library_sha256,
            backend_descriptor_sha256=value.backend_descriptor_sha256,
            segment_capacity=value.segment_capacity,
            crossing_capacity=value.crossing_capacity,
        )
        declared = value.path_free_descriptor()
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("runtime binding fields are malformed") from error
    if type(declared) is not dict or declared != rebuilt.path_free_descriptor():
        raise KerrNativeCpuAuthenticationError(
            "runtime binding path-free descriptor is stale"
        )
    return rebuilt


def _acquire_process_local_backend(
    runtime_binding: KerrNativeCpuRuntimeBinding,
) -> StrictKerrCpuBackend:
    """Construct at most one matching live backend in the current process."""

    global _PROCESS_LOCAL_BACKEND_PID
    binding = _canonical_runtime_binding(runtime_binding)
    process_id = os.getpid()
    key = (
        process_id,
        os.fspath(binding.library_path),
        binding.library_sha256,
        binding.backend_descriptor_sha256,
        binding.segment_capacity,
        binding.crossing_capacity,
    )
    with _PROCESS_LOCAL_BACKEND_LOCK:
        if _PROCESS_LOCAL_BACKEND_PID != process_id:
            _PROCESS_LOCAL_BACKENDS.clear()
            _PROCESS_LOCAL_BACKEND_PID = process_id
        cached = _PROCESS_LOCAL_BACKENDS.get(key)
        if cached is not None:
            try:
                cached.revalidate()
            except BaseException:
                _PROCESS_LOCAL_BACKENDS.pop(key, None)
                raise
            if (
                cached.library_sha256 != binding.library_sha256
                or cached.model_descriptor_sha256
                != binding.backend_descriptor_sha256
            ):
                _PROCESS_LOCAL_BACKENDS.pop(key, None)
                raise KerrNativeCpuAuthenticationError(
                    "cached backend differs from its runtime binding"
                )
            return cached
        backend = StrictKerrCpuBackend(
            binding.library_path,
            segment_capacity=binding.segment_capacity,
            crossing_capacity=binding.crossing_capacity,
        )
        backend.revalidate()
        if (
            backend.library_sha256 != binding.library_sha256
            or backend.model_descriptor_sha256
            != binding.backend_descriptor_sha256
        ):
            raise KerrNativeCpuAuthenticationError(
                "explicit library path differs from the bound backend descriptor"
            )
        rebound = backend.runtime_binding()
        if rebound.path_free_descriptor() != binding.path_free_descriptor():
            raise KerrNativeCpuAuthenticationError(
                "constructed backend capacities/descriptor differ from binding"
            )
        _PROCESS_LOCAL_BACKENDS[key] = backend
        return backend


@contextmanager
def consume_kerr_native_cpu_runtime_binding(
    runtime_binding: KerrNativeCpuRuntimeBinding,
) -> Iterator[StrictKerrCpuBackend]:
    """Yield one PID-local backend and authenticate it before and after use."""

    binding = _canonical_runtime_binding(runtime_binding)
    backend = _acquire_process_local_backend(binding)
    backend.revalidate()
    if (
        backend.library_sha256 != binding.library_sha256
        or backend.model_descriptor_sha256 != binding.backend_descriptor_sha256
    ):
        raise KerrNativeCpuAuthenticationError(
            "process-local backend does not match runtime binding"
        )
    try:
        yield backend
    finally:
        backend.revalidate()
        if (
            backend.library_sha256 != binding.library_sha256
            or backend.model_descriptor_sha256
            != binding.backend_descriptor_sha256
        ):
            raise KerrNativeCpuAuthenticationError(
                "process-local backend changed during bound use"
            )


def _reset_process_local_backend_cache_for_tests() -> None:
    """Drop process-local references; intentionally private to test code."""

    global _NATIVE_IMAGE_REGISTRY_PID, _PROCESS_LOCAL_BACKEND_PID
    with _PROCESS_LOCAL_BACKEND_LOCK:
        _PROCESS_LOCAL_BACKENDS.clear()
        _PROCESS_LOCAL_BACKEND_PID = None
    with _NATIVE_IMAGE_LOCK:
        _NATIVE_IMAGE_SHA_BY_PATH.clear()
        _NATIVE_IMAGE_POISONED_PATHS.clear()
        _NATIVE_IMAGE_REGISTRY_PID = None


__all__ = (
    "ABI_VERSION",
    "DEFAULT_HEADER_PATH",
    "DEFAULT_AUDIT_TOOL_PATH",
    "DEFAULT_MAKEFILE_PATH",
    "DEFAULT_SOURCE_PATH",
    "FrozenKerrNativeCpuRuntimeBinding",
    "IMPLEMENTATION_ID",
    "KerrNativeCpuAuthenticationError",
    "KerrNativeCpuCapacityError",
    "KerrNativeCpuError",
    "KerrNativeCpuExecutionError",
    "KerrNativeCpuOneResolution",
    "KerrNativeCpuRuntimeBinding",
    "KerrNativeCpuWorkDiagnostics",
    "StrictKerrCpuBackend",
    "consume_kerr_native_cpu_runtime_binding",
)
