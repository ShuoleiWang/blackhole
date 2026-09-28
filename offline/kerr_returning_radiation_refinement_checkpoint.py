"""Publish a kernel-only Kerr returning-radiation refinement checkpoint.

The checkpoint runs or resumes the fixed production-forward direction cache,
authenticates its five producer-owned grids through convergence v2, and
publishes the complete finite-grid evidence whether it is qualified or not.
It deliberately stops before every thermal, spectral, colour, frame, tile, or
display stage.

This is same-code finite-grid evidence.  It is neither an independent
geodesic/physics oracle nor a rigorous continuum error bound.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import errno
import hashlib
import hmac
import importlib
from importlib.machinery import ModuleSpec, SourceFileLoader
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any, Final, Mapping
from types import ModuleType
import uuid

from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.job import canonical_json_bytes
from offline.kerr import KerrOblateTermination
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
import offline.kerr_returning_radiation_convergence_v2 as _convergence
from offline.kerr_returning_radiation_convergence_v2 import (
    KerrAuthenticatedReturningRadiationConvergenceV2,
    KerrReturningRadiationConvergenceV2Policy,
    KerrReturningRadiationGridSummaryV2,
    authenticate_cached_kerr_returning_radiation_convergence_v2,
    compare_kerr_returning_radiation_grids_v2,
    verify_authenticated_kerr_returning_radiation_convergence_v2,
)
from offline.kerr_returning_radiation_kernel import (
    KerrReturningRadiationKernelPolicy,
)
import offline.kerr_returning_radiation_kernel_cached as _cached
from offline.kerr_returning_radiation_kernel_cached import (
    KerrCachedReturningRadiationKernelExecution,
    build_forward_kerr_returning_radiation_kernel_cache_definition,
    build_native_forward_kerr_returning_radiation_kernel_cache_definition,
    integrate_cached_kerr_returning_radiation_energy_kernel,
    integrate_cached_kerr_returning_radiation_energy_kernel_native_cpu,
    validate_and_reduce_cached_kerr_returning_radiation_energy_kernel,
)
from offline.kerr_returning_radiation_kernel_jobs import (
    MAXIMUM_IN_FLIGHT_TASKS,
    MAXIMUM_WORKERS,
    KerrKernelDirectionTaskPlan,
)
import offline.kerr_returning_radiation_kernel_jobs as _jobs


IMPLEMENTATION_ID: Final = (
    "kerr-returning-radiation-refinement-checkpoint/v1"
)
NATIVE_IMPLEMENTATION_ID: Final = (
    "kerr-returning-radiation-refinement-checkpoint/native-cpu-v2"
)
MANIFEST_SCHEMA: Final = (
    "blackhole.kerr-returning-radiation-refinement-checkpoint/v1"
)
NATIVE_MANIFEST_SCHEMA: Final = (
    "blackhole.kerr-returning-radiation-refinement-checkpoint/v2"
)
MANIFEST_NAME: Final = "manifest.json"
SIDECAR_NAME: Final = "manifest.sha256"
MAXIMUM_MANIFEST_BYTES: Final = 64 * 1024 * 1024
MAXIMUM_SOURCE_FILE_BYTES: Final = 16 * 1024 * 1024
MAXIMUM_SOURCE_TOTAL_BYTES: Final = 64 * 1024 * 1024
MAXIMUM_SOURCE_ARTIFACTS: Final = 64
MAXIMUM_DIRECTIONS_PER_TASK: Final = 4096
MAXIMUM_JSON_NESTING_DEPTH: Final = 64
STRICT_CONVERGENCE_POLICY_SHA256: Final = (
    KerrReturningRadiationConvergenceV2Policy().model_descriptor_sha256
)
PYTHON_EVALUATOR_MODE: Final = "python"
NATIVE_CPU_EVALUATOR_MODE: Final = "native-cpu"
_EVALUATOR_MODES: Final = (PYTHON_EVALUATOR_MODE, NATIVE_CPU_EVALUATOR_MODE)

_PATH_TYPE: Final = type(Path())
_MODULE_NAME: Final = "offline.kerr_returning_radiation_refinement_checkpoint"
_TRUSTED_CHECKOUT_ROOT: Final = Path("/Users/shuolei/Documents/blackhole")
_OFFLINE_PACKAGE_OWNER: Final = sys.modules.get("offline")
_EXPECTED_MODULE_FILE: Final = (
    _TRUSTED_CHECKOUT_ROOT
    / "offline/kerr_returning_radiation_refinement_checkpoint.py"
)
_MODULE_FILE: Final = Path(os.path.abspath(__file__))
_MODULE_OWNER: Final = sys.modules.get(_MODULE_NAME)
_MODULE_SPEC: Final = globals().get("__spec__")
_MODULE_LOADER: Final = globals().get("__loader__")
_SOURCE_ROOT: Final = _TRUSTED_CHECKOUT_ROOT
_ADAPTER_AND_CHECKPOINT_SOURCE_PATHS: Final = (
    "offline/kerr_returning_radiation_convergence_v2.py",
    "offline/kerr_returning_radiation_refinement_checkpoint.py",
    "scripts/run_offline_kerr_returning_radiation_refinement.py",
)
_CHECKPOINT_SOURCE_PATHS: Final = tuple(
    sorted(
        set(
            (*_cached._SOURCE_CLOSURE_PATHS, *_ADAPTER_AND_CHECKPOINT_SOURCE_PATHS)
        )
    )
)
_NATIVE_CHECKPOINT_SOURCE_PATHS: Final = tuple(
    sorted(
        set(
            (
                *_cached._NATIVE_FORWARD_SOURCE_CLOSURE_PATHS,
                *_ADAPTER_AND_CHECKPOINT_SOURCE_PATHS,
            )
        )
    )
)
_FROZEN_CHECKPOINT_SOURCE_PATHS: Final = _CHECKPOINT_SOURCE_PATHS
_FROZEN_NATIVE_CHECKPOINT_SOURCE_PATHS: Final = _NATIVE_CHECKPOINT_SOURCE_PATHS
_KERNEL_SOURCE_CLOSURE_PATHS: Final = tuple(_cached._SOURCE_CLOSURE_PATHS)
_NATIVE_KERNEL_SOURCE_CLOSURE_PATHS: Final = tuple(
    _cached._NATIVE_FORWARD_SOURCE_CLOSURE_PATHS
)
_KERNEL_NUMERIC_BACKEND_LOGICAL_PATH: Final = (
    _cached._NUMERIC_BACKEND_LOGICAL_PATH
)


def _source_module_name(logical_path: str) -> str | None:
    if logical_path == "offline/__init__.py":
        return "offline"
    if (
        logical_path.startswith("offline/")
        and logical_path.endswith(".py")
        and "/" not in logical_path[len("offline/") : -len(".py")]
    ):
        return logical_path[: -len(".py")].replace("/", ".")
    if logical_path == _ADAPTER_AND_CHECKPOINT_SOURCE_PATHS[-1]:
        return "scripts.run_offline_kerr_returning_radiation_refinement"
    return None


_CHECKPOINT_SOURCE_MODULE_OWNERS: Final = tuple(
    (
        module_name,
        sys.modules.get(module_name),
        _SOURCE_ROOT / logical_path,
    )
    for logical_path in _CHECKPOINT_SOURCE_PATHS
    for module_name in (_source_module_name(logical_path),)
    if module_name is not None and module_name in sys.modules
)

_CACHED_INTEGRATOR_PUBLIC_ENTRY = (
    integrate_cached_kerr_returning_radiation_energy_kernel
)
_CACHED_INTEGRATOR_CALL_ENTRY = _CACHED_INTEGRATOR_PUBLIC_ENTRY
_NATIVE_CACHED_INTEGRATOR_PUBLIC_ENTRY = (
    integrate_cached_kerr_returning_radiation_energy_kernel_native_cpu
)
_NATIVE_CACHED_INTEGRATOR_CALL_ENTRY = _NATIVE_CACHED_INTEGRATOR_PUBLIC_ENTRY
_CACHE_DEFINITION_PUBLIC_ENTRY = (
    build_forward_kerr_returning_radiation_kernel_cache_definition
)
_CACHE_DEFINITION_CALL_ENTRY = _CACHE_DEFINITION_PUBLIC_ENTRY
_NATIVE_CACHE_DEFINITION_PUBLIC_ENTRY = (
    build_native_forward_kerr_returning_radiation_kernel_cache_definition
)
_NATIVE_CACHE_DEFINITION_CALL_ENTRY = _NATIVE_CACHE_DEFINITION_PUBLIC_ENTRY
_AUTHENTICATE_PUBLIC_ENTRY = (
    authenticate_cached_kerr_returning_radiation_convergence_v2
)
_AUTHENTICATE_CALL_ENTRY = _AUTHENTICATE_PUBLIC_ENTRY
_VERIFY_AUTHENTICATED_PUBLIC_ENTRY = (
    verify_authenticated_kerr_returning_radiation_convergence_v2
)
_VERIFY_AUTHENTICATED_CALL_ENTRY = _VERIFY_AUTHENTICATED_PUBLIC_ENTRY
_VALIDATE_CACHED_PUBLIC_ENTRY = (
    validate_and_reduce_cached_kerr_returning_radiation_energy_kernel
)
_VALIDATE_CACHED_CALL_ENTRY = _VALIDATE_CACHED_PUBLIC_ENTRY
_PRODUCTION_BINDING_CALL_ENTRY = _cached._production_forward_scientific_binding
_NUMERIC_BACKEND_CALL_ENTRY = _cached._numeric_backend_descriptor
_SOURCE_CLOSURE_FOR_DEFINITION_CALL_ENTRY = _cached._source_closure_for_definition
_POST_PUBLICATION_VERIFIER_PUBLIC_ENTRY = None
_CLI_ORIGIN_GUARD_CALL_ENTRY = None


class KerrReturningRadiationRefinementCheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be authenticated or published safely."""


@dataclass(frozen=True, slots=True)
class KerrReturningRadiationRefinementPlan:
    """Kernel-only scientific inputs plus non-scientific execution controls."""

    output_directory: Path
    cache_root: Path
    surface: KerrFiniteThicknessMultiSurface
    termination: KerrOblateTermination
    annulus_edges_over_mass: tuple[float, ...]
    ray_options: RayTraceOptions
    surface_options: SurfaceEventOptions
    kernel_policy: KerrReturningRadiationKernelPolicy
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy
    convergence_policy: KerrReturningRadiationConvergenceV2Policy
    evaluator_mode: str = PYTHON_EVALUATOR_MODE
    native_library_path: Path | None = None
    native_segment_capacity: int | None = None
    native_crossing_capacity: int | None = None
    directions_per_task: int = 64
    jobs: int = 1
    max_in_flight: int | None = None

    def __post_init__(self) -> None:
        if type(self.evaluator_mode) is not str or self.evaluator_mode not in (
            _EVALUATOR_MODES
        ):
            raise ValueError("evaluator_mode must be exact 'python' or 'native-cpu'")
        if self.evaluator_mode == PYTHON_EVALUATOR_MODE:
            _require_exact_source_module_origins()
        else:
            _require_exact_source_module_origins(self.evaluator_mode)
        _assert_runtime_bindings()
        if type(self.output_directory) is not _PATH_TYPE:
            raise TypeError("output_directory must be an exact platform Path")
        if type(self.cache_root) is not _PATH_TYPE:
            raise TypeError("cache_root must be an exact platform Path")
        _validate_checkpoint_paths(self.output_directory, self.cache_root)
        if type(self.surface) is not KerrFiniteThicknessMultiSurface:
            raise TypeError("surface must have its exact finite-thickness type")
        if type(self.termination) is not KerrOblateTermination:
            raise TypeError("termination must have its exact Kerr type")
        if type(self.annulus_edges_over_mass) is not tuple or len(
            self.annulus_edges_over_mass
        ) < 2:
            raise TypeError("annulus edges must be an exact tuple with at least two values")
        if any(type(value) is not float for value in self.annulus_edges_over_mass):
            raise TypeError("annulus edges must contain exact floats")
        if type(self.ray_options) is not RayTraceOptions:
            raise TypeError("ray_options must have its exact type")
        if type(self.surface_options) is not SurfaceEventOptions:
            raise TypeError("surface_options must have its exact type")
        if type(self.kernel_policy) is not KerrReturningRadiationKernelPolicy:
            raise TypeError("kernel_policy must have its exact type")
        if type(self.area_policy) is not KerrFiniteThicknessAreaQuadraturePolicy:
            raise TypeError("area_policy must have its exact type")
        if type(self.convergence_policy) is not KerrReturningRadiationConvergenceV2Policy:
            raise TypeError("convergence_policy must have its exact type")
        native_fields = (
            self.native_library_path,
            self.native_segment_capacity,
            self.native_crossing_capacity,
        )
        if self.evaluator_mode == PYTHON_EVALUATOR_MODE:
            if any(value is not None for value in native_fields):
                raise ValueError(
                    "python evaluator mode cannot carry native runtime controls"
                )
        else:
            if type(self.native_library_path) is not _PATH_TYPE:
                raise TypeError(
                    "native-cpu evaluator mode requires an exact absolute dylib Path"
                )
            library = _validate_absolute_path(
                self.native_library_path,
                "native CPU library",
            )
            try:
                library_status = os.lstat(library)
            except OSError as error:
                raise ValueError("native CPU library is not readable") from error
            if stat.S_ISLNK(library_status.st_mode) or not stat.S_ISREG(
                library_status.st_mode
            ):
                raise ValueError(
                    "native CPU library must be a regular non-symlink file"
                )
            for name in (
                "native_segment_capacity",
                "native_crossing_capacity",
            ):
                value = getattr(self, name)
                if type(value) is not int or not 1 <= value <= 2_000_000:
                    raise ValueError(
                        f"{name} must be an exact integer in [1, 2000000]"
                    )
            object.__setattr__(self, "native_library_path", library)
        if (
            self.convergence_policy.model_descriptor_sha256
            != STRICT_CONVERGENCE_POLICY_SHA256
        ):
            raise ValueError(
                "convergence policy must be the canonical strict v2 default"
            )
        for name in ("directions_per_task", "jobs"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise TypeError(f"{name} must be a positive exact int")
        if self.directions_per_task > MAXIMUM_DIRECTIONS_PER_TASK:
            raise ValueError("directions_per_task exceeds the hard maximum")
        if self.jobs > MAXIMUM_WORKERS:
            raise ValueError("jobs exceeds the hard worker maximum")
        if self.max_in_flight is not None and (
            type(self.max_in_flight) is not int or self.max_in_flight < 1
        ):
            raise TypeError("max_in_flight must be None or a positive exact int")
        self.convergence_policy.revalidate()


@dataclass(frozen=True, slots=True)
class KerrReturningRadiationRefinementPublication:
    output_directory: Path
    manifest_path: Path
    manifest_sha256: str
    checkpoint_id: str
    qualified: bool


@dataclass(frozen=True, slots=True)
class KerrReturningRadiationVerifiedRefinementCheckpoint:
    output_directory: Path
    manifest_path: Path
    manifest_sha256: str
    checkpoint_id: str
    qualified: bool
    document: Mapping[str, Any]


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _validate_absolute_path(path: Path, label: str) -> Path:
    if type(path) is not _PATH_TYPE or not path.is_absolute():
        raise TypeError(f"{label} must be an exact absolute platform Path")
    if path != _lexical_absolute(path) or path == path.parent or path.name in (
        "",
        ".",
        "..",
    ):
        raise ValueError(f"{label} must be a canonical non-root absolute path")
    current = Path(path.anchor)
    for index, component in enumerate(path.parts[1:]):
        if component in ("", ".", ".."):
            raise ValueError(f"{label} contains a non-canonical component")
        current = current / component
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{label} contains an existing symbolic link")
        if index + 1 < len(path.parts[1:]) and not stat.S_ISDIR(status.st_mode):
            raise NotADirectoryError(f"{label} contains a non-directory ancestor")
    return path


def _paths_equal_or_nested(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_checkpoint_paths(output: Path, cache: Path) -> tuple[Path, Path]:
    output = _validate_absolute_path(output, "checkpoint output")
    cache = _validate_absolute_path(cache, "kernel cache")
    if _paths_equal_or_nested(output, cache):
        raise ValueError("checkpoint output and kernel cache must be non-nested")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing checkpoint {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError("checkpoint output parent must already exist")
    if not cache.parent.is_dir():
        raise FileNotFoundError("kernel cache parent must already exist")
    if cache.exists() and not cache.is_dir():
        raise NotADirectoryError("kernel cache final path must be a directory")
    return output, cache


def _directory_open_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(getattr(os, name, None) is None for name in required):
        raise KerrReturningRadiationRefinementCheckpointError(
            "platform lacks O_DIRECTORY/O_NOFOLLOW secure path primitives"
        )
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_absolute_directory_chain(path: Path, label: str) -> list[int]:
    _validate_absolute_path(path, label)
    descriptors = [os.open(os.sep, _directory_open_flags())]
    try:
        for component in path.parts[1:]:
            descriptors.append(
                os.open(component, _directory_open_flags(), dir_fd=descriptors[-1])
            )
        return descriptors
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _directory_identity(descriptor: int) -> tuple[int, int, int]:
    status = os.fstat(descriptor)
    if not stat.S_ISDIR(status.st_mode):
        raise KerrReturningRadiationRefinementCheckpointError(
            "checkpoint directory descriptor changed type"
        )
    return status.st_dev, status.st_ino, status.st_mode


def _require_same_directory_path(
    path: Path,
    descriptor: int,
    label: str,
) -> None:
    reopened = _open_absolute_directory_chain(path, label)
    try:
        if _directory_identity(reopened[-1]) != _directory_identity(descriptor):
            raise KerrReturningRadiationRefinementCheckpointError(
                f"{label} path identity changed"
            )
    finally:
        for item in reversed(reopened):
            os.close(item)


def _read_source_at(root_descriptor: int, logical_path: str) -> bytes:
    pure = PurePosixPath(logical_path)
    if (
        pure.is_absolute()
        or len(pure.parts) < 2
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise KerrReturningRadiationRefinementCheckpointError(
            "checkpoint source closure contains an invalid logical path"
        )
    parent_descriptors: list[int] = []
    current = root_descriptor
    descriptor = -1
    try:
        for component in pure.parts[:-1]:
            following = os.open(component, _directory_open_flags(), dir_fd=current)
            parent_descriptors.append(following)
            current = following
        descriptor = os.open(
            pure.name,
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=current,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise KerrReturningRadiationRefinementCheckpointError(
                f"checkpoint source {logical_path} is not a regular file"
            )
        if before.st_size < 0 or before.st_size > MAXIMUM_SOURCE_FILE_BYTES:
            raise KerrReturningRadiationRefinementCheckpointError(
                f"checkpoint source {logical_path} exceeds its byte limit"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise KerrReturningRadiationRefinementCheckpointError(
                    f"checkpoint source {logical_path} ended early"
                )
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise KerrReturningRadiationRefinementCheckpointError(
                f"checkpoint source {logical_path} grew while hashing"
            )
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise KerrReturningRadiationRefinementCheckpointError(
                f"checkpoint source {logical_path} changed while hashing"
            )
        return b"".join(chunks)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        for parent in reversed(parent_descriptors):
            os.close(parent)


def _require_exact_source_module_origins(
    evaluator_mode: str = PYTHON_EVALUATOR_MODE,
) -> None:
    """Bind every loaded checkpoint dependency to the exact tree being hashed."""

    if type(evaluator_mode) is not str or evaluator_mode not in _EVALUATOR_MODES:
        raise ValueError("checkpoint evaluator mode is unsupported")
    module_file = globals().get("__file__")
    module = sys.modules.get(_MODULE_NAME)
    offline_package = sys.modules.get("offline")
    if (
        type(globals().get("__name__")) is not str
        or globals()["__name__"] != _MODULE_NAME
        or type(module_file) is not str
        or Path(os.path.abspath(module_file)) != _MODULE_FILE
        or _MODULE_FILE != _EXPECTED_MODULE_FILE
        or type(offline_package) is not ModuleType
        or offline_package is not _OFFLINE_PACKAGE_OWNER
        or type(object.__getattribute__(offline_package, "__file__")) is not str
        or Path(object.__getattribute__(offline_package, "__file__"))
        != _SOURCE_ROOT / "offline/__init__.py"
        or type(module) is not ModuleType
        or module is not _MODULE_OWNER
        or globals().get("__spec__") is not _MODULE_SPEC
        or globals().get("__loader__") is not _MODULE_LOADER
        or type(_MODULE_SPEC) is not ModuleSpec
        or type(_MODULE_LOADER) is not SourceFileLoader
        or _MODULE_SPEC.loader is not _MODULE_LOADER
        or type(_MODULE_SPEC.name) is not str
        or _MODULE_SPEC.name != _MODULE_NAME
        or type(_MODULE_SPEC.origin) is not str
        or Path(_MODULE_SPEC.origin) != _EXPECTED_MODULE_FILE
        or type(_MODULE_LOADER.name) is not str
        or _MODULE_LOADER.name != _MODULE_NAME
        or type(_MODULE_LOADER.path) is not str
        or Path(_MODULE_LOADER.path) != _EXPECTED_MODULE_FILE
    ):
        raise KerrReturningRadiationRefinementCheckpointError(
            "checkpoint module has no frozen source identity"
        )
    closure_is_frozen = (
        _CHECKPOINT_SOURCE_PATHS is _FROZEN_CHECKPOINT_SOURCE_PATHS
        if evaluator_mode == PYTHON_EVALUATOR_MODE
        else _NATIVE_CHECKPOINT_SOURCE_PATHS
        is _FROZEN_NATIVE_CHECKPOINT_SOURCE_PATHS
    )
    if not closure_is_frozen:
        raise KerrReturningRadiationRefinementCheckpointError(
            "checkpoint source closure binding changed"
        )
    try:
        _convergence._require_exact_module_origin()
    except Exception as error:
        raise KerrReturningRadiationRefinementCheckpointError(
            "authenticated convergence adapter was loaded from another tree"
        ) from error
    if evaluator_mode == NATIVE_CPU_EVALUATOR_MODE:
        importlib.import_module("offline.kerr_native_cpu_backend")
        importlib.import_module("offline.kerr_returning_radiation_native_cpu")
        try:
            _cached._require_exact_source_module_origins(
                _source_paths=_cached._NATIVE_FORWARD_SOURCE_CLOSURE_PATHS,
                _module_owners=(
                    _cached._NATIVE_FORWARD_SOURCE_CLOSURE_MODULE_OWNERS
                ),
            )
        except Exception as error:
            raise KerrReturningRadiationRefinementCheckpointError(
                "native cached source closure was loaded from another tree"
            ) from error
    current_cli = sys.modules.get(
        "scripts.run_offline_kerr_returning_radiation_refinement"
    )
    if current_cli is not None:
        _require_exact_loaded_source_module(
            "scripts.run_offline_kerr_returning_radiation_refinement",
            current_cli,
            _SOURCE_ROOT
            / "scripts/run_offline_kerr_returning_radiation_refinement.py",
        )
    for module_name, expected_module, expected_file in (
        _CHECKPOINT_SOURCE_MODULE_OWNERS
    ):
        _require_exact_loaded_source_module(
            module_name,
            expected_module,
            expected_file,
        )


def _require_exact_loaded_source_module(
    module_name: str,
    expected_module: Any,
    expected_file: Path,
) -> None:
    module = sys.modules.get(module_name)
    if type(module) is not ModuleType or module is not expected_module:
        raise KerrReturningRadiationRefinementCheckpointError(
            f"checkpoint source module {module_name} is not the exact loaded module"
        )
    try:
        actual_file = object.__getattribute__(module, "__file__")
        module_spec = object.__getattribute__(module, "__spec__")
        module_loader = object.__getattribute__(module, "__loader__")
    except AttributeError as error:
        raise KerrReturningRadiationRefinementCheckpointError(
            f"checkpoint source module {module_name} has incomplete import identity"
        ) from error
    if (
        type(actual_file) is not str
        or not Path(actual_file).is_absolute()
        or Path(actual_file) != expected_file
        or type(module_spec) is not ModuleSpec
        or type(module_loader) is not SourceFileLoader
        or module_spec.loader is not module_loader
        or type(module_spec.name) is not str
        or module_spec.name != module_name
        or type(module_spec.origin) is not str
        or Path(module_spec.origin) != expected_file
        or type(module_loader.name) is not str
        or module_loader.name != module_name
        or type(module_loader.path) is not str
        or Path(module_loader.path) != expected_file
    ):
        raise KerrReturningRadiationRefinementCheckpointError(
            f"checkpoint source module {module_name} has a foreign import identity"
        )


def _source_snapshot(
    evaluator_mode: str = PYTHON_EVALUATOR_MODE,
) -> dict[str, Any]:
    source_paths = (
        _CHECKPOINT_SOURCE_PATHS
        if evaluator_mode == PYTHON_EVALUATOR_MODE
        else _NATIVE_CHECKPOINT_SOURCE_PATHS
    )
    if len(source_paths) > MAXIMUM_SOURCE_ARTIFACTS:
        raise KerrReturningRadiationRefinementCheckpointError(
            "checkpoint source closure exceeds its artifact-count limit"
        )
    if evaluator_mode == PYTHON_EVALUATOR_MODE:
        _require_exact_source_module_origins()
    else:
        _require_exact_source_module_origins(evaluator_mode)
    chain = _open_absolute_directory_chain(_SOURCE_ROOT, "checkpoint source root")
    try:
        entries: list[dict[str, Any]] = []
        total = 0
        for logical_path in source_paths:
            payload = _read_source_at(chain[-1], logical_path)
            total += len(payload)
            if total > MAXIMUM_SOURCE_TOTAL_BYTES:
                raise KerrReturningRadiationRefinementCheckpointError(
                    "checkpoint source closure exceeds its aggregate byte limit"
                )
            entries.append(
                {
                    "byteLength": len(payload),
                    "logicalPath": logical_path,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    finally:
        for descriptor in reversed(chain):
            os.close(descriptor)
    frozen = list(entries)
    adapter = list(
        item
        for item in frozen
        if item["logicalPath"] in _ADAPTER_AND_CHECKPOINT_SOURCE_PATHS
    )
    return {
        "adapterAndCheckpointArtifacts": adapter,
        "artifacts": frozen,
        "artifactCount": len(frozen),
        "totalByteLength": total,
        "manifestSha256": _canonical_sha256(frozen),
    }


def _evaluator_mode_from_definition(definition: Any) -> str:
    if type(definition) is not _cached.KerrKernelDirectionCacheDefinition:
        raise TypeError("cache definition must have its exact type")
    evaluator_id = _cached._evaluator_id_from_definition(definition)
    if evaluator_id == _cached._FORWARD_EVALUATOR_ID:
        return PYTHON_EVALUATOR_MODE
    if evaluator_id == _cached._NATIVE_FORWARD_EVALUATOR_ID:
        return NATIVE_CPU_EVALUATOR_MODE
    raise KerrReturningRadiationRefinementCheckpointError(
        "checkpoint definition uses a non-production evaluator"
    )


def _runtime_snapshot(definition: Any | None = None) -> dict[str, Any]:
    if definition is None:
        descriptor = _NUMERIC_BACKEND_CALL_ENTRY()
    else:
        mode = _evaluator_mode_from_definition(definition)
        if mode == PYTHON_EVALUATOR_MODE:
            descriptor = _NUMERIC_BACKEND_CALL_ENTRY()
        else:
            binding = definition.scientific_context.evaluator_runtime_binding
            if type(binding) is not _jobs.KerrKernelEvaluatorRuntimeBinding:
                raise KerrReturningRadiationRefinementCheckpointError(
                    "native definition lacks its exact runtime binding"
                )
            descriptor = _cached._native_numeric_backend_descriptor(
                _cached._native_backend_descriptor_from_binding(binding)
            )
        closure = _SOURCE_CLOSURE_FOR_DEFINITION_CALL_ENTRY(definition)
        numeric_entry = closure[-1]
        descriptor_payload = canonical_json_bytes(descriptor)
        if (
            numeric_entry.logical_path != _KERNEL_NUMERIC_BACKEND_LOGICAL_PATH
            or numeric_entry.byte_length != len(descriptor_payload)
            or numeric_entry.sha256
            != hashlib.sha256(descriptor_payload).hexdigest()
        ):
            raise KerrReturningRadiationRefinementCheckpointError(
                "definition numeric backend differs from its source closure"
            )
    payload = canonical_json_bytes(descriptor)
    return {
        "descriptor": json.loads(payload),
        "descriptorSha256": hashlib.sha256(payload).hexdigest(),
    }


def _require_loaded_checkpoint_function_identity(function: Any, name: str) -> None:
    if type(function) is not type(_require_loaded_checkpoint_function_identity):
        raise KerrReturningRadiationRefinementCheckpointError(
            f"checkpoint {name} has a foreign callable type"
        )
    code = object.__getattribute__(function, "__code__")
    if (
        type(code.co_filename) is not str
        or Path(os.path.abspath(code.co_filename)) != _EXPECTED_MODULE_FILE
    ):
        raise KerrReturningRadiationRefinementCheckpointError(
            f"checkpoint {name} was not executed from the trusted source path"
        )


def _assert_snapshots_stable(
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    label: str,
    *,
    evaluator_mode: str = PYTHON_EVALUATOR_MODE,
    definition: Any | None = None,
) -> None:
    current_source = (
        _source_snapshot()
        if evaluator_mode == PYTHON_EVALUATOR_MODE
        else _source_snapshot(evaluator_mode)
    )
    if current_source != source:
        raise KerrReturningRadiationRefinementCheckpointError(
            f"checkpoint source closure changed {label}"
        )
    if _runtime_snapshot(definition) != runtime:
        raise KerrReturningRadiationRefinementCheckpointError(
            f"numeric backend changed {label}"
        )


def _assert_runtime_bindings() -> None:
    bindings = (
        (
            _cached.integrate_cached_kerr_returning_radiation_energy_kernel,
            _CACHED_INTEGRATOR_PUBLIC_ENTRY,
            "cached integrator",
        ),
        (
            _cached.integrate_cached_kerr_returning_radiation_energy_kernel_native_cpu,
            _NATIVE_CACHED_INTEGRATOR_PUBLIC_ENTRY,
            "native cached integrator",
        ),
        (
            _cached.build_forward_kerr_returning_radiation_kernel_cache_definition,
            _CACHE_DEFINITION_PUBLIC_ENTRY,
            "cache-definition builder",
        ),
        (
            _cached.build_native_forward_kerr_returning_radiation_kernel_cache_definition,
            _NATIVE_CACHE_DEFINITION_PUBLIC_ENTRY,
            "native cache-definition builder",
        ),
        (
            _convergence.authenticate_cached_kerr_returning_radiation_convergence_v2,
            _AUTHENTICATE_PUBLIC_ENTRY,
            "authenticated-v2 adapter",
        ),
        (
            _convergence.verify_authenticated_kerr_returning_radiation_convergence_v2,
            _VERIFY_AUTHENTICATED_PUBLIC_ENTRY,
            "authenticated-v2 verifier",
        ),
        (
            _cached.validate_and_reduce_cached_kerr_returning_radiation_energy_kernel,
            _VALIDATE_CACHED_PUBLIC_ENTRY,
            "cached-forward validator",
        ),
        (
            verify_kerr_returning_radiation_refinement_checkpoint
            if "verify_kerr_returning_radiation_refinement_checkpoint" in globals()
            else _POST_PUBLICATION_VERIFIER_PUBLIC_ENTRY,
            _POST_PUBLICATION_VERIFIER_PUBLIC_ENTRY,
            "post-publication checkpoint verifier",
        ),
    )
    for current, frozen, label in bindings:
        if frozen is None:
            continue
        if current is not frozen:
            raise KerrReturningRadiationRefinementCheckpointError(
                f"public {label} binding changed"
            )
    for name in (
        "_require_exact_source_module_origins",
        "execute_kerr_returning_radiation_refinement_checkpoint",
        "verify_kerr_returning_radiation_refinement_checkpoint",
    ):
        function = globals().get(name)
        if function is not None:
            _require_loaded_checkpoint_function_identity(function, name)


def _authenticated_documents(
    result: KerrAuthenticatedReturningRadiationConvergenceV2,
) -> dict[str, Any]:
    if type(result) is not KerrAuthenticatedReturningRadiationConvergenceV2:
        raise TypeError("authenticated convergence result has a foreign type")
    _VERIFY_AUTHENTICATED_CALL_ENTRY(result)
    descriptor_json = object.__getattribute__(result, "canonical_descriptor_json")
    descriptor_sha = object.__getattribute__(result, "model_descriptor_sha256")
    if type(descriptor_json) is not str or type(descriptor_sha) is not str:
        raise KerrReturningRadiationRefinementCheckpointError(
            "authenticated convergence descriptor has a non-exact type"
        )
    if hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest() != descriptor_sha:
        raise KerrReturningRadiationRefinementCheckpointError(
            "authenticated convergence descriptor SHA-256 differs"
        )
    summaries = object.__getattribute__(result, "summaries")
    reports = object.__getattribute__(result, "comparisons")
    if type(summaries) is not tuple or len(summaries) != 5:
        raise KerrReturningRadiationRefinementCheckpointError(
            "authenticated convergence must contain five summaries"
        )
    if type(reports) is not tuple or len(reports) != 4:
        raise KerrReturningRadiationRefinementCheckpointError(
            "authenticated convergence must contain four reports"
        )
    summary_documents = tuple(
        {
            "descriptor": dict(item.descriptor()),
            "descriptorSha256": item.model_descriptor_sha256,
        }
        for item in summaries
    )
    report_documents = tuple(
        {
            "descriptor": dict(item.descriptor()),
            "descriptorSha256": item.model_descriptor_sha256,
        }
        for item in reports
    )
    qualified = all(item.converged for item in reports)
    if result.converged is not qualified:
        raise KerrReturningRadiationRefinementCheckpointError(
            "authenticated aggregate qualification differs from its reports"
        )
    return {
        "descriptor": json.loads(descriptor_json),
        "descriptorSha256": descriptor_sha,
        "policy": {
            "descriptor": dict(result.policy.descriptor()),
            "descriptorSha256": result.policy.model_descriptor_sha256,
        },
        "summaries": summary_documents,
        "comparisonReports": report_documents,
        "qualified": qualified,
    }


def _checkpoint_document(
    execution: KerrCachedReturningRadiationKernelExecution,
    authenticated: KerrAuthenticatedReturningRadiationConvergenceV2,
    source_snapshot: Mapping[str, Any],
    runtime_snapshot: Mapping[str, Any],
    scientific_binding_json: str,
    scientific_binding_sha256: str,
) -> tuple[dict[str, Any], bool]:
    definition = execution.cache_definition
    context = definition.scientific_context
    evaluator_mode = _evaluator_mode_from_definition(definition)
    authentication = _authenticated_documents(authenticated)
    qualified = authentication["qualified"]
    binding = json.loads(scientific_binding_json)
    canonical_binding = canonical_json_bytes(binding)
    if (
        hashlib.sha256(canonical_binding[:-1]).hexdigest()
        != scientific_binding_sha256
    ):
        raise KerrReturningRadiationRefinementCheckpointError(
            "cached-forward scientific binding SHA-256 differs"
        )
    expected_binding_id = (
        _cached.CACHED_NATIVE_FORWARD_SCIENTIFIC_BINDING_IMPLEMENTATION_ID
        if evaluator_mode == NATIVE_CPU_EVALUATOR_MODE
        else _cached.CACHED_FORWARD_SCIENTIFIC_BINDING_IMPLEMENTATION_ID
    )
    if binding.get("implementationId") != expected_binding_id:
        raise KerrReturningRadiationRefinementCheckpointError(
            "cached-forward scientific binding evaluator mode differs"
        )
    provenance = authentication["descriptor"].get("provenance")
    if not isinstance(provenance, dict) or provenance.get(
        "scientificBindingSha256"
    ) != scientific_binding_sha256:
        raise KerrReturningRadiationRefinementCheckpointError(
            "authenticated convergence and cached scientific binding differ"
        )
    audit = execution.execution_audit
    operational = {
        "cacheLayout": definition.plan.cache_layout_descriptor(),
        "cacheJobKey": definition.job_spec.job_key,
        "executedDirectionRecords": audit.executed_direction_records,
        "executedTasks": audit.executed_tasks,
        "maximumInFlightObserved": execution.job_run.max_in_flight_observed,
        "reusedDirectionRecords": audit.reused_direction_records,
        "reusedTasks": audit.reused_tasks,
        "taskCount": definition.plan.task_count,
        "taskPayloadPathsPublished": False,
        "taskReuseEvidence": [
            {
                "reused": item.reused,
                "task": item.key.as_dict(),
            }
            for item in execution.job_run.results
        ],
    }
    scientific_identity = context.identity.as_dict()
    body = {
        "authenticatedConvergenceV2": authentication,
        "classification": (
            "source-current-v2-qualified-finite-grid-checkpoint"
            if qualified
            else "source-current-v2-non-qualified-finite-grid-checkpoint"
        ),
        "implementationId": (
            NATIVE_IMPLEMENTATION_ID
            if evaluator_mode == NATIVE_CPU_EVALUATOR_MODE
            else IMPLEMENTATION_ID
        ),
        "kernel": {
            "descriptor": execution.kernel.model_descriptor(),
            "descriptorSha256": execution.kernel.model_descriptor_sha256,
        },
        "operationalExecution": operational,
        "producer": {
            "cacheDefinition": {
                "cacheJobKey": definition.job_spec.job_key,
                "jobSpec": definition.job_spec.as_dict(),
                "jobSpecSha256": _canonical_sha256(definition.job_spec.as_dict()),
                "scientificDocument": context.scientific_document(),
                "scientificIdentity": scientific_identity,
                "scientificIdentitySha256": _canonical_sha256(scientific_identity),
                "scientificJobKey": definition.scientific_job_key,
                "scientificPlan": definition.plan.scientific_descriptor(),
                "scientificPlanSha256": definition.plan.scientific_plan_sha256,
            },
            "executionAudit": asdict(audit),
            "reductionConfiguration": {
                "descriptor": execution.reduction_configuration.as_dict(),
                "descriptorSha256": execution.reduction_configuration_sha256,
            },
            "scientificBinding": {
                "descriptor": binding,
                "descriptorSha256": scientific_binding_sha256,
            },
            "sourceRuntimeClosure": binding["sourceRuntimeClosure"],
        },
        "qualified": qualified,
        "runtimeNumericBackend": dict(runtime_snapshot),
        "schema": (
            NATIVE_MANIFEST_SCHEMA
            if evaluator_mode == NATIVE_CPU_EVALUATOR_MODE
            else MANIFEST_SCHEMA
        ),
        "scientificBoundary": {
            "cachedAuthenticationWholeRayTraces": 0,
            "containsCieOrColourInputs": False,
            "containsFrameOrTileInputs": False,
            "containsThermalOrSpectralInputs": False,
            "externalExpectedManifestSha256RequiredForUntrustedTransport": True,
            "isIndependentGeodesicOrPhysicsOracle": False,
            "isRigorousContinuumErrorBound": False,
            "isSameCodeFiniteGridEvidence": True,
            "nonQualifiedCheckpointsArePublished": True,
            "publicationSuccessUsesExternalExpectedManifestSha256": True,
            "prohibitedClaim": (
                "Qualification is finite-grid same-code evidence, not an "
                "independent oracle or rigorous continuum error bound."
            ),
            "sidecarIsExternalTrustAnchor": False,
            "standaloneTwoFileVerificationResistsMaliciousResealing": False,
        },
        "sourceFreeze": {
            "afterAuthentication": dict(source_snapshot),
            "beforeExecution": dict(source_snapshot),
            "stable": True,
        },
    }
    if evaluator_mode == NATIVE_CPU_EVALUATOR_MODE:
        body["evaluatorMode"] = evaluator_mode
    checkpoint_sha = _canonical_sha256(body)
    document = {
        **body,
        "id": f"kerr-returning-radiation-refinement-{checkpoint_sha[:24]}",
        "integrity": {
            "checkpointSha256": checkpoint_sha,
            "manifestSidecar": SIDECAR_NAME,
        },
    }
    return document, qualified


def _write_exclusive_at(directory_descriptor: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=directory_descriptor,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written < 1:
                raise KerrReturningRadiationRefinementCheckpointError(
                    "short write while publishing refinement checkpoint"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _promote_no_replace_at(
    parent_descriptor: int,
    staging_name: str,
    output_name: str,
    output: Path,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        function = library.renameatx_np
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            os.fsencode(staging_name),
            parent_descriptor,
            os.fsencode(output_name),
            0x00000004,
        )
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            os.fsencode(staging_name),
            parent_descriptor,
            os.fsencode(output_name),
            1,
        )
    else:
        raise KerrReturningRadiationRefinementCheckpointError(
            "platform lacks atomic no-replace directory publication"
        )
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(f"refusing to overwrite existing checkpoint {output}")
    raise KerrReturningRadiationRefinementCheckpointError(
        f"cannot publish refinement checkpoint atomically: {os.strerror(number)}"
    )


def _read_published_at(
    directory_descriptor: int, name: str, maximum_bytes: int
) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum_bytes:
            raise KerrReturningRadiationRefinementCheckpointError(
                "published checkpoint member has an invalid type or size"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise KerrReturningRadiationRefinementCheckpointError(
                    "published checkpoint member ended early"
                )
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise KerrReturningRadiationRefinementCheckpointError(
                "published checkpoint member grew while reading"
            )
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
            raise KerrReturningRadiationRefinementCheckpointError(
                "published checkpoint member changed while reading"
            )
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise KerrReturningRadiationRefinementCheckpointError(
                "published checkpoint member stable-size check failed"
            )
        return payload
    finally:
        os.close(descriptor)


def _validate_json_nesting(payload: bytes) -> None:
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
            if depth > MAXIMUM_JSON_NESTING_DEPTH:
                raise KerrReturningRadiationRefinementCheckpointError(
                    "refinement manifest exceeds its JSON nesting limit"
                )
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth < 0:
                raise KerrReturningRadiationRefinementCheckpointError(
                    "refinement manifest has invalid JSON nesting"
                )
    if in_string or escaped or depth != 0:
        raise KerrReturningRadiationRefinementCheckpointError(
            "refinement manifest has incomplete JSON nesting"
        )


def _strict_canonical_manifest(payload: bytes) -> dict[str, Any]:
    if not payload or len(payload) > MAXIMUM_MANIFEST_BYTES:
        raise KerrReturningRadiationRefinementCheckpointError(
            "refinement manifest is empty or exceeds its hard byte limit"
        )
    def unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if type(key) is not str or key in result:
                raise ValueError("duplicate or foreign JSON object key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"forbidden non-finite JSON number {value!r}")

    _validate_json_nesting(payload)
    try:
        text = payload.decode("utf-8")
        document = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise KerrReturningRadiationRefinementCheckpointError(
            "refinement manifest is not valid UTF-8 JSON"
        ) from error
    try:
        canonical = canonical_json_bytes(document)
    except (TypeError, ValueError, RecursionError) as error:
        raise KerrReturningRadiationRefinementCheckpointError(
            "refinement manifest is not finite canonical JSON"
        ) from error
    if type(document) is not dict or canonical != payload:
        raise KerrReturningRadiationRefinementCheckpointError(
            "refinement manifest is not canonical JSON"
        )
    return document


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise KerrReturningRadiationRefinementCheckpointError(
            f"{label} is not a canonical SHA-256"
        )
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise KerrReturningRadiationRefinementCheckpointError(
            f"{label} is not a canonical SHA-256"
        ) from error
    if value.lower() != value:
        raise KerrReturningRadiationRefinementCheckpointError(
            f"{label} is not a lowercase SHA-256"
        )
    return value


def _descriptor_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} keys")
    return value


def _require_exact_json_tree(actual: Any, expected: Any, label: str) -> None:
    """Compare decoded canonical JSON without accepting Python subclasses."""

    if type(actual) is not type(expected):
        raise TypeError(f"{label} type")
    if type(expected) is dict:
        if set(actual) != set(expected):
            raise ValueError(f"{label} keys")
        for key in expected:
            if type(key) is not str:
                raise TypeError(f"{label} key type")
            _require_exact_json_tree(actual[key], expected[key], f"{label}.{key}")
        return
    if type(expected) is list:
        if len(actual) != len(expected):
            raise ValueError(f"{label} length")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _require_exact_json_tree(
                actual_item,
                expected_item,
                f"{label}[{index}]",
            )
        return
    if type(expected) is float:
        if not math.isfinite(actual) or actual.hex() != expected.hex():
            raise ValueError(f"{label} value")
        return
    if type(expected) is bool:
        if actual is not expected:
            raise ValueError(f"{label} value")
        return
    if type(expected) in (int, str) or expected is None:
        if actual != expected:
            raise ValueError(f"{label} value")
        return
    raise TypeError(f"{label} unsupported type")


def _require_strict_convergence_policy(policy: Any) -> str:
    expected = KerrReturningRadiationConvergenceV2Policy()
    expected_descriptor = dict(expected.descriptor())
    _require_exact_json_tree(
        policy,
        expected_descriptor,
        "canonical strict v2 convergence policy",
    )
    if expected.model_descriptor_sha256 != STRICT_CONVERGENCE_POLICY_SHA256:
        raise KerrReturningRadiationRefinementCheckpointError(
            "canonical strict v2 convergence policy binding changed"
        )
    return STRICT_CONVERGENCE_POLICY_SHA256


def _summary_from_full_kernel(kernel: Mapping[str, Any]) -> dict[str, Any]:
    annuli = kernel["annuli"]
    area = kernel["area"]
    result = kernel["result"]
    upper_areas = area["upperAnnulusAreasOverMassSquared"]
    lower_areas = area["lowerAnnulusAreasOverMassSquared"]
    receiver_areas = [*upper_areas, *lower_areas]
    matrices = result["matrices"]
    coefficients = [
        [*uu_row, *ul_row]
        for uu_row, ul_row in zip(matrices["UU"], matrices["UL"])
    ] + [
        [*lu_row, *ll_row]
        for lu_row, ll_row in zip(matrices["LU"], matrices["LL"])
    ]
    g2 = result["g2ReturnedPowerColumns"]
    fate_order = (
        ("return_upper", "return_lower", "captured", "escaped", "plunge_sink")
    )
    fates = result["fates"]
    fate_rows = [
        [row[name] for name in fate_order]
        for row in (*fates["upperEmitters"], *fates["lowerEmitters"])
    ]
    return {
        "coefficientIndexOrder": "K[receiverCell][emitterColumn]",
        "emitterAreas": receiver_areas,
        "fateFractionOrder": list(_convergence.FATE_NAMES),
        "fateFractions": fate_rows,
        "g2ReturnedPowerColumns": [
            *g2["upperEmitters"],
            *g2["lowerEmitters"],
        ],
        "gridId": "cached-full",
        "matrices": coefficients,
        "maximumNormalizedSampleWeight": None,
        "properPowerEquation": "P[i,j]=A_receiver[i]*K[i,j]/A_emitter[j]",
        "receiverAreas": receiver_areas,
        "schema": "blackhole.kerr-returning-radiation-grid-summary/v2",
        "_annulusEdgesOverMass": annuli["edgesOverMass"],
    }


def _expected_task_documents(
    plan: Mapping[str, Any], directions_per_task: int
) -> list[dict[str, int]]:
    rebuilt = KerrKernelDirectionTaskPlan(
        plan["formulation"],
        plan["annulusCount"],
        plan["rhoOrder"] if "rhoOrder" in plan else plan["passes"][0]["rhoOrder"],
        plan["muOrder"] if "muOrder" in plan else plan["passes"][0]["muOrder"],
        plan["psiCount"] if "psiCount" in plan else plan["passes"][0]["psiCount"],
        directions_per_task,
    )
    expected_plan = json.loads(canonical_json_bytes(rebuilt.scientific_descriptor()))
    _require_exact_json_tree(plan, expected_plan, "scientific plan reconstruction")
    return json.loads(
        canonical_json_bytes([item.as_dict() for item in rebuilt.task_keys()])
    )


def _verify_checkpoint_document(document: dict[str, Any]) -> bool:
    try:
        schema = document.get("schema")
        if schema == MANIFEST_SCHEMA:
            evaluator_mode = PYTHON_EVALUATOR_MODE
            expected_implementation_id = IMPLEMENTATION_ID
            mode_fields: set[str] = set()
        elif schema == NATIVE_MANIFEST_SCHEMA:
            evaluator_mode = document.get("evaluatorMode")
            if evaluator_mode != NATIVE_CPU_EVALUATOR_MODE:
                raise ValueError("native manifest evaluator mode")
            expected_implementation_id = NATIVE_IMPLEMENTATION_ID
            mode_fields = {"evaluatorMode"}
        else:
            raise KeyError("schema")
        _require_exact_keys(
            document,
            mode_fields
            | {
                "authenticatedConvergenceV2",
                "classification",
                "id",
                "implementationId",
                "integrity",
                "kernel",
                "operationalExecution",
                "producer",
                "qualified",
                "runtimeNumericBackend",
                "schema",
                "scientificBoundary",
                "sourceFreeze",
            },
            "manifest",
        )
        if document["implementationId"] != expected_implementation_id:
            raise KeyError("implementationId")
        expected_evaluator_id = (
            _cached._NATIVE_FORWARD_EVALUATOR_ID
            if evaluator_mode == NATIVE_CPU_EVALUATOR_MODE
            else _cached._FORWARD_EVALUATOR_ID
        )
        expected_binding_implementation_id = (
            _cached.CACHED_NATIVE_FORWARD_SCIENTIFIC_BINDING_IMPLEMENTATION_ID
            if evaluator_mode == NATIVE_CPU_EVALUATOR_MODE
            else _cached.CACHED_FORWARD_SCIENTIFIC_BINDING_IMPLEMENTATION_ID
        )
        expected_kernel_closure_paths = (
            _NATIVE_KERNEL_SOURCE_CLOSURE_PATHS
            if evaluator_mode == NATIVE_CPU_EVALUATOR_MODE
            else _KERNEL_SOURCE_CLOSURE_PATHS
        )
        qualified = document["qualified"]
        if type(qualified) is not bool:
            raise TypeError("qualified")
        authentication = _require_exact_keys(
            document["authenticatedConvergenceV2"],
            {
                "comparisonReports",
                "descriptor",
                "descriptorSha256",
                "policy",
                "qualified",
                "summaries",
            },
            "authenticated convergence",
        )
        summaries = authentication["summaries"]
        reports = authentication["comparisonReports"]
        if type(summaries) is not list or len(summaries) != 5:
            raise TypeError("summaries")
        if type(reports) is not list or len(reports) != 4:
            raise TypeError("reports")
        if type(authentication["qualified"]) is not bool or (
            authentication["qualified"] is not qualified
        ):
            raise TypeError("authenticated qualification")
        descriptor = authentication["descriptor"]
        descriptor_sha = _require_sha256(
            authentication["descriptorSha256"],
            "authenticated convergence descriptor SHA-256",
        )
        canonical_descriptor = _descriptor_bytes(descriptor)
        if hashlib.sha256(canonical_descriptor).hexdigest() != descriptor_sha:
            raise ValueError("authenticated descriptor digest")
        policy = _require_exact_keys(
            authentication["policy"],
            {"descriptor", "descriptorSha256"},
            "convergence policy",
        )
        policy_sha = _require_sha256(
            policy["descriptorSha256"], "convergence policy SHA-256"
        )
        strict_policy_sha = _require_strict_convergence_policy(
            policy["descriptor"]
        )
        if (
            hashlib.sha256(_descriptor_bytes(policy["descriptor"])).hexdigest()
            != policy_sha
            or descriptor.get("policySha256") != policy_sha
            or policy_sha != strict_policy_sha
        ):
            raise ValueError("convergence policy binding")
        leaf_hashes: dict[str, list[str]] = {"summary": [], "report": []}
        for label, items in (("summary", summaries), ("report", reports)):
            for index, item in enumerate(items):
                _require_exact_keys(
                    item,
                    {"descriptor", "descriptorSha256"},
                    f"{label} leaf",
                )
                digest = _require_sha256(
                    item["descriptorSha256"], f"{label} descriptor SHA-256"
                )
                raw = _descriptor_bytes(item["descriptor"])
                if hashlib.sha256(raw).hexdigest() != digest:
                    raise ValueError(f"{label} descriptor digest")
                leaf_hashes[label].append(digest)
                if label == "summary":
                    expected_grid = (
                        "cached-full",
                        "cached-half-rho",
                        "cached-half-mu",
                        "cached-half-psi",
                        "cached-phase-shifted",
                    )[index]
                    if item["descriptor"].get("gridId") != expected_grid:
                        raise ValueError("summary grid identity")
                else:
                    report = item["descriptor"]
                    expected_comparison = (
                        "cached-half-rho",
                        "cached-half-mu",
                        "cached-half-psi",
                        "cached-phase-shifted",
                    )[index]
                    if (
                        report.get("fineGridId") != "cached-full"
                        or report.get("comparisonGridId") != expected_comparison
                        or report.get("fineGridSummarySha256")
                        != leaf_hashes["summary"][0]
                        or report.get("comparisonGridSummarySha256")
                        != leaf_hashes["summary"][index + 1]
                    ):
                        raise ValueError("report/summary cross-binding")
        if (
            descriptor.get("sourceKind") != "cached"
            or descriptor.get("summarySha256") != leaf_hashes["summary"]
            or descriptor.get("comparisonReportSha256") != leaf_hashes["report"]
        ):
            raise ValueError("authenticated descriptor leaf binding")
        rebuilt_summaries = tuple(
            KerrReturningRadiationGridSummaryV2(
                item["descriptor"]["gridId"],
                tuple(item["descriptor"]["receiverAreas"]),
                tuple(item["descriptor"]["emitterAreas"]),
                tuple(tuple(row) for row in item["descriptor"]["matrices"]),
                tuple(item["descriptor"]["g2ReturnedPowerColumns"]),
                tuple(tuple(row) for row in item["descriptor"]["fateFractions"]),
                item["descriptor"]["maximumNormalizedSampleWeight"],
            )
            for item in summaries
        )
        rebuilt_policy = KerrReturningRadiationConvergenceV2Policy(
            g2_column_relative_tolerance=policy["descriptor"][
                "g2ColumnRelativeTolerance"
            ],
            g2_column_absolute_tolerance=policy["descriptor"][
                "g2ColumnAbsoluteTolerance"
            ],
            g2_column_relative_floor=policy["descriptor"][
                "g2ColumnRelativeFloor"
            ],
            column_normalized_l1_tolerance=policy["descriptor"][
                "columnNormalizedL1Tolerance"
            ],
            significant_cell_fraction_of_column=policy["descriptor"][
                "significantCellFractionOfColumn"
            ],
            significant_cell_symmetric_relative_tolerance=policy["descriptor"][
                "significantCellSymmetricRelativeTolerance"
            ],
            insignificant_tail_normalized_tolerance=policy["descriptor"][
                "insignificantTailNormalizedTolerance"
            ],
            support_flip_absolute_tolerance=policy["descriptor"][
                "supportFlipAbsoluteTolerance"
            ],
            fate_total_variation_tolerance=policy["descriptor"][
                "fateTotalVariationTolerance"
            ],
            fate_component_absolute_tolerance=policy["descriptor"][
                "fateComponentAbsoluteTolerance"
            ],
            fate_component_relative_tolerance=policy["descriptor"][
                "fateComponentRelativeTolerance"
            ],
            fate_component_relative_floor=policy["descriptor"][
                "fateComponentRelativeFloor"
            ],
            maximum_normalized_sample_weight=policy["descriptor"][
                "maximumNormalizedSampleWeight"
            ],
            maximum_receiver_cells=policy["descriptor"]["maximumReceiverCells"],
            maximum_source_columns=policy["descriptor"]["maximumSourceColumns"],
            maximum_matrix_cells=policy["descriptor"]["maximumMatrixCells"],
        )
        if rebuilt_policy.model_descriptor_sha256 != policy_sha:
            raise ValueError("convergence policy reconstruction")
        rebuilt_reports = tuple(
            compare_kerr_returning_radiation_grids_v2(
                rebuilt_summaries[0], item, rebuilt_policy
            )
            for item in rebuilt_summaries[1:]
        )
        if [item.model_descriptor_sha256 for item in rebuilt_reports] != (
            leaf_hashes["report"]
        ):
            raise ValueError("comparison report reconstruction")
        report_qualification = all(
            type(item["descriptor"].get("converged")) is bool
            and item["descriptor"]["converged"] is True
            for item in reports
        )
        if report_qualification is not qualified:
            raise ValueError("report qualification")

        kernel = _require_exact_keys(
            document["kernel"],
            {"descriptor", "descriptorSha256"},
            "kernel",
        )
        kernel_sha = _require_sha256(
            kernel["descriptorSha256"], "kernel descriptor SHA-256"
        )
        kernel_descriptor = _require_exact_keys(
            kernel["descriptor"],
            {
                "annuli",
                "area",
                "capabilities",
                "coefficient",
                "convergence",
                "implementationId",
                "modelOwnership",
                "quadrature",
                "result",
                "sampleAuditSha256",
                "workBudget",
            },
            "kernel descriptor",
        )
        kernel_json = _descriptor_bytes(kernel_descriptor)
        if hashlib.sha256(kernel_json).hexdigest() != kernel_sha:
            raise ValueError("kernel descriptor digest")
        if descriptor["kernelDescriptorSha256"] != kernel_sha:
            raise ValueError("authenticated/kernel descriptor binding")

        producer = _require_exact_keys(
            document["producer"],
            {
                "cacheDefinition",
                "executionAudit",
                "reductionConfiguration",
                "scientificBinding",
                "sourceRuntimeClosure",
            },
            "producer",
        )
        definition = _require_exact_keys(
            producer["cacheDefinition"],
            {
                "cacheJobKey",
                "jobSpec",
                "jobSpecSha256",
                "scientificDocument",
                "scientificIdentity",
                "scientificIdentitySha256",
                "scientificJobKey",
                "scientificPlan",
                "scientificPlanSha256",
            },
            "cache definition",
        )
        plan_sha = _require_sha256(
            definition["scientificPlanSha256"], "scientific plan SHA-256"
        )
        if _canonical_sha256(definition["scientificPlan"]) != plan_sha:
            raise ValueError("scientific plan digest")
        identity_sha = _require_sha256(
            definition["scientificIdentitySha256"],
            "scientific identity SHA-256",
        )
        if _canonical_sha256(definition["scientificIdentity"]) != identity_sha:
            raise ValueError("scientific identity digest")
        scientific_job_key = _require_sha256(
            definition["scientificJobKey"], "scientific job key"
        )
        cache_job_key = _require_sha256(
            definition["cacheJobKey"], "cache job key"
        )
        job_spec_sha = _require_sha256(
            definition["jobSpecSha256"], "JobSpec SHA-256"
        )
        scientific_document = definition["scientificDocument"]
        if _canonical_sha256(scientific_document) != scientific_job_key:
            raise ValueError("scientific document/job key binding")
        if _canonical_sha256(definition["jobSpec"]) != job_spec_sha:
            raise ValueError("JobSpec digest")
        if hashlib.sha256(canonical_json_bytes(definition["jobSpec"])).hexdigest() != (
            cache_job_key
        ):
            raise ValueError("JobSpec/cache key binding")
        if (
            scientific_document.get("scientificIdentity")
            != definition["scientificIdentity"]
            or scientific_document.get("plan") != definition["scientificPlan"]
        ):
            raise ValueError("scientific document component binding")
        scientific_plan = definition["scientificPlan"]
        scientific_identity = definition["scientificIdentity"]
        job_spec = definition["jobSpec"]
        job_parameters = job_spec["parameters"]
        operational = document["operationalExecution"]
        directions_per_task = operational["cacheLayout"][
            "directionsPerTaskMaximum"
        ]
        expected_tasks = _expected_task_documents(
            scientific_plan,
            directions_per_task,
        )
        if (
            job_parameters["scientificDocument"] != scientific_document
            or job_parameters["scientificJobKey"] != scientific_job_key
            or job_parameters["scientificPlanSha256"] != plan_sha
            or job_parameters["cacheLayout"] != operational["cacheLayout"]
            or job_spec["tasks"] != expected_tasks
            or len(expected_tasks) != operational["taskCount"]
            or scientific_document["producerSourceHashes"]
            != scientific_identity["sourceClosureSha256"]
            or job_spec["producerSourceHashes"]
            != scientific_identity["sourceClosureSha256"]
            or job_spec["inputs"] != scientific_document["inputs"]
            or job_spec["producer"] != scientific_document["producer"]
            or job_spec["algorithmVersion"]
            != scientific_document["algorithmVersion"]
            or scientific_plan["formulation"] != "forward"
            or scientific_identity["formulation"] != "forward"
            or scientific_identity["annulusEdgesOverMass"]
            != kernel_descriptor["annuli"]["edgesOverMass"]
        ):
            raise ValueError("JobSpec/scientific plan closed-tree binding")

        reduction = _require_exact_keys(
            producer["reductionConfiguration"],
            {"descriptor", "descriptorSha256"},
            "reduction configuration",
        )
        reduction_sha = _require_sha256(
            reduction["descriptorSha256"], "reduction configuration SHA-256"
        )
        if _canonical_sha256(reduction["descriptor"]) != reduction_sha:
            raise ValueError("reduction configuration digest")
        binding = _require_exact_keys(
            producer["scientificBinding"],
            {"descriptor", "descriptorSha256"},
            "scientific binding",
        )
        binding_sha = _require_sha256(
            binding["descriptorSha256"], "scientific binding SHA-256"
        )
        binding_bytes = canonical_json_bytes(binding["descriptor"])
        if hashlib.sha256(binding_bytes[:-1]).hexdigest() != binding_sha:
            raise ValueError("scientific binding digest")
        binding_descriptor = _require_exact_keys(
            binding["descriptor"],
            {
                "authentication",
                "evaluator",
                "excludedOperationalIdentity",
                "forwardKernelDescriptorSha256",
                "implementationId",
                "reductionConfiguration",
                "reductionConfigurationSha256",
                "scientificJobKey",
                "scientificPlanSha256",
                "sourceRuntimeClosure",
                "transportScientificJobKey",
            },
            "scientific binding descriptor",
        )
        if binding_descriptor["implementationId"] != (
            expected_binding_implementation_id
        ):
            raise ValueError("scientific binding evaluator mode")
        evaluator_document = _require_exact_keys(
            binding_descriptor["evaluator"],
            (
                {
                    "artifact",
                    "implementationId",
                    "productionForwardOnly",
                    "runtimeBinding",
                }
                if evaluator_mode == NATIVE_CPU_EVALUATOR_MODE
                else {
                    "artifact",
                    "implementationId",
                    "productionForwardOnly",
                }
            ),
            "scientific binding evaluator",
        )
        if (
            evaluator_document["implementationId"] != expected_evaluator_id
            or evaluator_document["productionForwardOnly"] is not True
            or evaluator_document["artifact"] not in scientific_document["inputs"]
        ):
            raise ValueError("scientific binding evaluator artifact")
        if evaluator_mode == NATIVE_CPU_EVALUATOR_MODE:
            runtime_binding_document = _require_exact_keys(
                evaluator_document["runtimeBinding"],
                {
                    "artifact",
                    "backendDescriptorSha256",
                    "pathFreeDescriptor",
                },
                "native runtime binding",
            )
            if (
                runtime_binding_document["artifact"]
                not in scientific_document["inputs"]
                or runtime_binding_document["artifact"]
                == evaluator_document["artifact"]
                or len(scientific_document["inputs"]) != 2
            ):
                raise ValueError("native runtime input artifact")
        elif len(scientific_document["inputs"]) != 1:
            raise ValueError("python evaluator input artifact count")
        if (
            binding_descriptor["forwardKernelDescriptorSha256"] != kernel_sha
            or binding_descriptor["scientificPlanSha256"] != plan_sha
            or binding_descriptor["transportScientificJobKey"]
            != scientific_job_key
            or binding_descriptor["reductionConfigurationSha256"]
            != reduction_sha
            or descriptor["provenance"]["scientificBindingSha256"]
            != binding_sha
            or descriptor["provenance"]["transportScientificJobKey"]
            != scientific_job_key
            or descriptor["provenance"]["forwardKernelDescriptorSha256"]
            != kernel_sha
            or descriptor["provenance"]["reductionConfigurationSha256"]
            != reduction_sha
            or descriptor["provenance"]["scientificBindingJsonSha256"]
            != hashlib.sha256(
                json.dumps(
                    binding_descriptor,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
        ):
            raise ValueError("producer provenance cross-binding")
        if (
            producer["sourceRuntimeClosure"]
            != binding_descriptor["sourceRuntimeClosure"]
            or binding_descriptor["reductionConfiguration"]
            != reduction["descriptor"]
        ):
            raise ValueError("source/runtime closure binding")
        closure = _require_exact_keys(
            producer["sourceRuntimeClosure"],
            {"entries", "manifestSha256", "numericBackendLogicalPath"},
            "source/runtime closure",
        )
        entries = closure["entries"]
        if type(entries) is not list or not entries:
            raise TypeError("source/runtime closure entries")
        expected_closure_paths = (
            *expected_kernel_closure_paths,
            _KERNEL_NUMERIC_BACKEND_LOGICAL_PATH,
        )
        actual_closure_paths = tuple(
            item.get("logicalPath") if type(item) is dict else None
            for item in entries
        )
        if actual_closure_paths != expected_closure_paths:
            raise ValueError("fixed source/runtime closure path order")
        expected_manifest = hashlib.sha256(
            "\n".join(
                f"{item['logicalPath']}\0{item['byteLength']}\0{item['sha256']}"
                for item in entries
            ).encode("utf-8")
        ).hexdigest()
        expected_bindings: list[str] = []
        for item in entries:
            _require_exact_keys(
                item,
                {"bindingSha256", "byteLength", "logicalPath", "sha256"},
                "source/runtime closure entry",
            )
            source_digest = _require_sha256(
                item["sha256"], "source/runtime closure entry SHA-256"
            )
            if (
                type(item["logicalPath"]) is not str
                or not item["logicalPath"]
                or type(item["byteLength"]) is not int
                or item["byteLength"] < 0
            ):
                raise TypeError("source/runtime closure entry identity")
            binding_digest = hashlib.sha256(
                (
                    f"{item['logicalPath']}\0{item['byteLength']}\0"
                    f"{source_digest}"
                ).encode("utf-8")
            ).hexdigest()
            if item["bindingSha256"] != binding_digest:
                raise ValueError("source/runtime closure ownership binding")
            expected_bindings.append(binding_digest)
        if (
            closure["manifestSha256"] != expected_manifest
            or scientific_identity["sourceClosureSha256"]
            != sorted(expected_bindings)
            or len(set(expected_bindings)) != len(expected_bindings)
        ):
            raise ValueError("source/runtime closure manifest binding")
        freeze = document["sourceFreeze"]
        frozen_artifacts = freeze["beforeExecution"]["artifacts"]
        frozen_by_path = {
            item["logicalPath"]: item for item in frozen_artifacts
        }
        runtime = document["runtimeNumericBackend"]
        runtime_payload = canonical_json_bytes(runtime["descriptor"])
        runtime_path = closure["numericBackendLogicalPath"]
        if runtime_path != _KERNEL_NUMERIC_BACKEND_LOGICAL_PATH:
            raise ValueError("numeric backend closure path")
        for item in entries:
            if item["logicalPath"] == runtime_path:
                if (
                    item["byteLength"] != len(runtime_payload)
                    or item["sha256"]
                    != hashlib.sha256(runtime_payload).hexdigest()
                    or runtime["descriptorSha256"] != item["sha256"]
                ):
                    raise ValueError("numeric backend closure binding")
            else:
                frozen_item = frozen_by_path.get(item["logicalPath"])
                if frozen_item is None or (
                    item["byteLength"] != frozen_item["byteLength"]
                    or item["sha256"] != frozen_item["sha256"]
                ):
                    raise ValueError("source closure/current freeze binding")
        if evaluator_mode == NATIVE_CPU_EVALUATOR_MODE:
            runtime_descriptor = _require_exact_keys(
                runtime["descriptor"],
                {
                    "implementationId",
                    "nativeWholeRay",
                    "nativeWholeRayDescriptorSha256",
                    "pythonControlPlane",
                },
                "native runtime numeric backend",
            )
            path_free_runtime = _require_exact_keys(
                runtime_binding_document["pathFreeDescriptor"],
                {
                    "backendDescriptorSha256",
                    "callerOwnedCapacities",
                    "implementationId",
                },
                "native path-free runtime binding",
            )
            if (
                runtime_descriptor["implementationId"]
                != "cpython-control-plane+strict-kerr-cpu-whole-ray/abi-v3"
                or runtime_descriptor["nativeWholeRayDescriptorSha256"]
                != runtime_binding_document["backendDescriptorSha256"]
                or runtime_descriptor["nativeWholeRayDescriptorSha256"]
                != path_free_runtime["backendDescriptorSha256"]
                or path_free_runtime["implementationId"]
                != "strict-kerr-cpu-process-runtime-binding/v1"
                or hashlib.sha256(
                    canonical_json_bytes(runtime_descriptor["nativeWholeRay"])
                ).hexdigest()
                != runtime_descriptor["nativeWholeRayDescriptorSha256"]
            ):
                raise ValueError("native runtime/backend cross-binding")
        elif runtime["descriptor"].get("implementationId") != (
            "cpython-binary64-struct-libm/v2"
        ):
            raise ValueError("python runtime backend identity")

        audit = _require_exact_keys(
            producer["executionAudit"],
            {
                "authenticated_direction_records",
                "cache_job_key",
                "evaluator_id",
                "executed_direction_records",
                "executed_tasks",
                "formulation",
                "is_independent_physics_or_geodesic_oracle",
                "is_same_code_cache_evidence",
                "is_task_reuse_history_cryptographically_authenticated",
                "reused_direction_records",
                "reused_tasks",
                "scientific_job_key",
                "source_closure_manifest_sha256",
                "source_closure_rechecked_before_and_after_reduction",
            },
            "execution audit",
        )
        operational = _require_exact_keys(
            operational,
            {
                "cacheJobKey",
                "cacheLayout",
                "executedDirectionRecords",
                "executedTasks",
                "maximumInFlightObserved",
                "reusedDirectionRecords",
                "reusedTasks",
                "taskCount",
                "taskPayloadPathsPublished",
                "taskReuseEvidence",
            },
            "operational execution",
        )
        for name in (
            "executedDirectionRecords",
            "executedTasks",
            "maximumInFlightObserved",
            "reusedDirectionRecords",
            "reusedTasks",
            "taskCount",
        ):
            if type(operational[name]) is not int or operational[name] < 0:
                raise TypeError(f"operational counter {name}")
        for name in (
            "authenticated_direction_records",
            "executed_direction_records",
            "executed_tasks",
            "reused_direction_records",
            "reused_tasks",
        ):
            if type(audit[name]) is not int or audit[name] < 0:
                raise TypeError(f"execution audit counter {name}")
        expected_layout = KerrKernelDirectionTaskPlan(
            scientific_plan["formulation"],
            scientific_plan["annulusCount"],
            scientific_plan["passes"][0]["rhoOrder"],
            scientific_plan["passes"][0]["muOrder"],
            scientific_plan["passes"][0]["psiCount"],
            directions_per_task,
        ).cache_layout_descriptor()
        task_reuse_evidence = operational["taskReuseEvidence"]
        if (
            type(task_reuse_evidence) is not list
            or len(task_reuse_evidence) != len(expected_tasks)
        ):
            raise TypeError("task reuse evidence")
        exact_executed_tasks = 0
        exact_reused_tasks = 0
        exact_executed_directions = 0
        exact_reused_directions = 0
        for index, (item, expected_task) in enumerate(
            zip(task_reuse_evidence, expected_tasks)
        ):
            item = _require_exact_keys(
                item,
                {"reused", "task"},
                f"task reuse evidence {index}",
            )
            if type(item["reused"]) is not bool:
                raise TypeError("task reuse flag")
            _require_exact_json_tree(
                item["task"],
                expected_task,
                f"task reuse evidence {index} task",
            )
            if item["reused"] is True:
                exact_reused_tasks += 1
                exact_reused_directions += expected_task["width"]
            else:
                exact_executed_tasks += 1
                exact_executed_directions += expected_task["width"]
        maximum_in_flight = operational["maximumInFlightObserved"]
        if (
            audit["authenticated_direction_records"]
            != operational["executedDirectionRecords"]
            + operational["reusedDirectionRecords"]
            or audit["executed_direction_records"]
            != operational["executedDirectionRecords"]
            or audit["reused_direction_records"]
            != operational["reusedDirectionRecords"]
            or audit["executed_tasks"] != operational["executedTasks"]
            or audit["reused_tasks"] != operational["reusedTasks"]
            or audit["cache_job_key"] != operational["cacheJobKey"]
            or audit["cache_job_key"] != cache_job_key
            or audit["scientific_job_key"] != scientific_job_key
            or audit["formulation"] != "forward"
            or audit["evaluator_id"]
            != expected_evaluator_id
            or audit["source_closure_manifest_sha256"] != expected_manifest
            or descriptor["provenance"]["sourceClosureManifestSha256"]
            != expected_manifest
            or audit["source_closure_rechecked_before_and_after_reduction"]
            is not True
            or audit["is_same_code_cache_evidence"] is not True
            or audit["is_independent_physics_or_geodesic_oracle"] is not False
            or audit["is_task_reuse_history_cryptographically_authenticated"]
            is not False
            or audit["authenticated_direction_records"]
            != definition["scientificPlan"]["directionCount"]
            or operational["taskCount"]
            != operational["executedTasks"] + operational["reusedTasks"]
            or operational["taskCount"]
            != operational["cacheLayout"]["taskCount"]
            or operational["cacheLayout"] != expected_layout
            or operational["executedTasks"] != exact_executed_tasks
            or operational["reusedTasks"] != exact_reused_tasks
            or operational["executedDirectionRecords"]
            != exact_executed_directions
            or operational["reusedDirectionRecords"] != exact_reused_directions
            or (
                operational["executedTasks"] == 0
                and maximum_in_flight != 0
            )
            or (
                operational["executedTasks"] > 0
                and not 1
                <= maximum_in_flight
                <= min(MAXIMUM_IN_FLIGHT_TASKS, operational["executedTasks"])
            )
            or operational["taskPayloadPathsPublished"] is not False
        ):
            raise ValueError("operational counters")

        auth_expected = {
            "acceptsCallerConstructedSummaries": False,
            "isIndependentGeodesicOrPhysicsOracle": False,
            "isRigorousContinuumErrorBound": False,
            "revalidationRebuildsFromRetainedSource": True,
            "sameCodeEvidence": True,
        }
        semantics_expected = {
            "coefficientIndexOrder": "K[receiverCell][emitterColumn]",
            "emitterCellOrder": "upper annuli then lower annuli",
            "matrixBlockRows": "UU|UL then LU|LL",
            "receiverCellOrder": "upper annuli then lower annuli",
        }
        _require_exact_json_tree(
            descriptor["authentication"], auth_expected, "authentication semantics"
        )
        _require_exact_json_tree(
            descriptor["cellSemantics"], semantics_expected, "cell semantics"
        )
        if (
            descriptor["implementationId"]
            != _convergence.AUTHENTICATED_IMPLEMENTATION_ID
            or descriptor["schema"]
            != "blackhole.kerr-returning-radiation-authenticated-convergence/v2"
        ):
            raise ValueError("authenticated convergence implementation identity")

        geometry = descriptor["gridGeometry"]
        upper_areas = geometry["upperAnnulusAreasOverMassSquared"]
        lower_areas = geometry["lowerAnnulusAreasOverMassSquared"]
        if (
            geometry["annulusEdgesOverMass"]
            != kernel_descriptor["annuli"]["edgesOverMass"]
            or geometry["annulusEdgesOverMass"]
            != scientific_identity["annulusEdgesOverMass"]
            or upper_areas
            != kernel_descriptor["area"]["upperAnnulusAreasOverMassSquared"]
            or lower_areas
            != kernel_descriptor["area"]["lowerAnnulusAreasOverMassSquared"]
            or scientific_identity["metric"]
            != kernel_descriptor["modelOwnership"]["metric"]
            or scientific_identity["surface"]["calibration"]
            != kernel_descriptor["modelOwnership"]["calibration"]
            or scientific_identity["termination"]
            != kernel_descriptor["modelOwnership"]["termination"]
        ):
            raise ValueError("authenticated grid geometry binding")

        full_expected = _summary_from_full_kernel(kernel_descriptor)
        full_edges = full_expected.pop("_annulusEdgesOverMass")
        full_expected["maximumNormalizedSampleWeight"] = descriptor[
            "passEvidence"
        ][0]["maximumNormalizedSampleWeight"]
        _require_exact_json_tree(
            summaries[0]["descriptor"],
            full_expected,
            "full summary/kernel binding",
        )
        if full_edges != geometry["annulusEdgesOverMass"]:
            raise ValueError("full summary annulus binding")

        pass_evidence = descriptor["passEvidence"]
        passes = scientific_plan["passes"]
        if type(pass_evidence) is not list or len(pass_evidence) != 5:
            raise TypeError("pass evidence")
        sample_keys = ("full", "halfRho", "halfMu", "halfPsi", "phaseShifted")
        sample_audits = kernel_descriptor["sampleAuditSha256"]
        direction_total = 0
        annulus_count = scientific_plan["annulusCount"]
        for index, (evidence, pass_plan, summary) in enumerate(
            zip(pass_evidence, passes, summaries)
        ):
            expected_directions = (
                2
                * annulus_count
                * pass_plan["rhoOrder"]
                * pass_plan["muOrder"]
                * pass_plan["psiCount"]
            )
            direction_total += expected_directions
            witness = evidence["maximumNormalizedSampleWeightWitness"]
            source_areas = (
                upper_areas if witness["sourceFace"] == "upper" else lower_areas
            )
            reconstructed_weight = (
                witness["rhoAreaOverMassSquared"]
                * witness["normalizedEmittedFluxDirectionWeight"]
                / source_areas[witness["sourceAnnulusIndex"]]
            )
            if (
                evidence["passIndex"] != index
                or evidence["passName"] != pass_plan["name"]
                or evidence["rhoOrder"] != pass_plan["rhoOrder"]
                or evidence["muOrder"] != pass_plan["muOrder"]
                or evidence["psiCount"] != pass_plan["psiCount"]
                or evidence["phaseCells"] != pass_plan["phaseCells"]
                or evidence["directionEvaluations"] != expected_directions
                or evidence["sampleAuditSha256"] != sample_audits[sample_keys[index]]
                or evidence["maximumNormalizedSampleWeight"]
                != summary["descriptor"]["maximumNormalizedSampleWeight"]
                or witness["passIndex"] != index
                or witness["passName"] != pass_plan["name"]
                or witness["normalizedSampleWeight"]
                != evidence["maximumNormalizedSampleWeight"]
                or reconstructed_weight.hex()
                != evidence["maximumNormalizedSampleWeight"].hex()
            ):
                raise ValueError("producer pass evidence binding")
        full_result = kernel_descriptor["result"]
        kernel_policy = reduction["descriptor"]["kernelPolicy"]
        area_policy = reduction["descriptor"]["areaPolicy"]
        if (
            pass_evidence[0]["upperEmitterG2ColumnClosureResiduals"]
            != full_result["g2ColumnClosureResiduals"]["upperEmitters"]
            or pass_evidence[0]["lowerEmitterG2ColumnClosureResiduals"]
            != full_result["g2ColumnClosureResiduals"]["lowerEmitters"]
            or direction_total != scientific_plan["directionCount"]
            or direction_total
            != kernel_descriptor["workBudget"]["directionEvaluationsConsumed"]
            or kernel_descriptor["workBudget"]["wholeRayTracesConsumed"]
            != 2 * direction_total
            or kernel_descriptor["workBudget"]["maximumDirectionEvaluations"]
            != kernel_policy["maximum_direction_evaluations"]
            or kernel_descriptor["workBudget"]["maximumWholeRayTraces"]
            != kernel_policy["maximum_whole_ray_traces"]
            or kernel_policy["rho_order"] != passes[0]["rhoOrder"]
            or kernel_policy["mu_order"] != passes[0]["muOrder"]
            or kernel_policy["psi_count"] != passes[0]["psiCount"]
            or kernel_descriptor["quadrature"]["fullRhoOrderPerSourceCell"]
            != kernel_policy["rho_order"]
            or kernel_descriptor["quadrature"]["fullMuOrder"]
            != kernel_policy["mu_order"]
            or kernel_descriptor["quadrature"]["fullPsiCount"]
            != kernel_policy["psi_count"]
            or kernel_descriptor["convergence"]["absoluteTolerance"]
            != kernel_policy["absolute_tolerance"]
            or kernel_descriptor["convergence"]["relativeTolerance"]
            != kernel_policy["relative_tolerance"]
            or kernel_descriptor["area"]["quadraturePolicy"]
            != {
                "absoluteToleranceOverMassSquared": area_policy[
                    "absolute_tolerance_over_mass_squared"
                ],
                "coarseOrder": area_policy["gauss_legendre_order"],
                "fineOrder": 2 * area_policy["gauss_legendre_order"],
                "maximumPointEvaluations": area_policy[
                    "maximum_point_evaluations"
                ],
                "relativeTolerance": area_policy["relative_tolerance"],
                "requiredPointEvaluations": 3
                * area_policy["gauss_legendre_order"],
            }
        ):
            raise ValueError("kernel pass evidence/work-budget binding")
        expected_boundary = {
            "cachedAuthenticationWholeRayTraces": 0,
            "containsCieOrColourInputs": False,
            "containsFrameOrTileInputs": False,
            "containsThermalOrSpectralInputs": False,
            "externalExpectedManifestSha256RequiredForUntrustedTransport": True,
            "isIndependentGeodesicOrPhysicsOracle": False,
            "isRigorousContinuumErrorBound": False,
            "isSameCodeFiniteGridEvidence": True,
            "nonQualifiedCheckpointsArePublished": True,
            "publicationSuccessUsesExternalExpectedManifestSha256": True,
            "prohibitedClaim": (
                "Qualification is finite-grid same-code evidence, not an "
                "independent oracle or rigorous continuum error bound."
            ),
            "sidecarIsExternalTrustAnchor": False,
            "standaloneTwoFileVerificationResistsMaliciousResealing": False,
        }
        _require_exact_json_tree(
            document["scientificBoundary"],
            expected_boundary,
            "scientific boundary",
        )
        expected_classification = (
            "source-current-v2-qualified-finite-grid-checkpoint"
            if qualified
            else "source-current-v2-non-qualified-finite-grid-checkpoint"
        )
        if document["classification"] != expected_classification:
            raise ValueError("classification")
        integrity = document["integrity"]
        if integrity["manifestSidecar"] != SIDECAR_NAME:
            raise ValueError("sidecar name")
        checkpoint_sha = _require_sha256(
            integrity["checkpointSha256"], "checkpoint SHA-256"
        )
        body = {
            key: value
            for key, value in document.items()
            if key not in ("id", "integrity")
        }
        if (
            _canonical_sha256(body) != checkpoint_sha
            or document["id"]
            != f"kerr-returning-radiation-refinement-{checkpoint_sha[:24]}"
        ):
            raise ValueError("checkpoint identity")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise KerrReturningRadiationRefinementCheckpointError(
            f"refinement checkpoint manifest is internally inconsistent: {error}"
        ) from error
    return qualified


def _current_runtime_snapshot_for_manifest(
    document: Mapping[str, Any],
    native_library_path: Path | None,
) -> dict[str, Any]:
    evaluator_mode = document.get("evaluatorMode", PYTHON_EVALUATOR_MODE)
    if evaluator_mode == PYTHON_EVALUATOR_MODE:
        if native_library_path is not None:
            raise ValueError(
                "python checkpoint verification cannot carry a native dylib path"
            )
        return _runtime_snapshot()
    if evaluator_mode != NATIVE_CPU_EVALUATOR_MODE:
        raise ValueError("checkpoint evaluator mode is unsupported")
    if type(native_library_path) is not _PATH_TYPE:
        raise TypeError(
            "native checkpoint verification requires an exact absolute dylib Path"
        )
    runtime_document = document["producer"]["scientificBinding"]["descriptor"][
        "evaluator"
    ]["runtimeBinding"]
    path_free = runtime_document["pathFreeDescriptor"]
    capacities = path_free["callerOwnedCapacities"]
    binding, backend_descriptor = _cached._native_forward_runtime_binding(
        native_library_path,
        segment_capacity=capacities["segments"],
        crossing_capacity=capacities["crossings"],
    )
    if (
        binding.path_free_descriptor() != path_free
        or binding.descriptor_input.as_dict() != runtime_document["artifact"]
        or binding.backend_descriptor_sha256
        != runtime_document["backendDescriptorSha256"]
    ):
        raise KerrReturningRadiationRefinementCheckpointError(
            "current native dylib differs from checkpoint runtime binding"
        )
    descriptor = _cached._native_numeric_backend_descriptor(backend_descriptor)
    payload = canonical_json_bytes(descriptor)
    return {
        "descriptor": json.loads(payload),
        "descriptorSha256": hashlib.sha256(payload).hexdigest(),
    }


def verify_kerr_returning_radiation_refinement_checkpoint(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str | None = None,
    native_library_path: Path | None = None,
) -> KerrReturningRadiationVerifiedRefinementCheckpoint:
    """Verify a current closed checkpoint, optionally against an external digest.

    Without ``expected_manifest_sha256`` this proves canonical closed-tree
    consistency and current local source/runtime identity, but not authenticity
    against whole-document malicious resealing.  Untrusted transport therefore
    requires an externally retained expected digest.
    """

    _assert_runtime_bindings()
    if type(manifest_path) is not _PATH_TYPE:
        raise TypeError("manifest_path must be an exact platform Path")
    if expected_manifest_sha256 is not None:
        expected_manifest_sha256 = _require_sha256(
            expected_manifest_sha256,
            "expected manifest SHA-256",
        )
    manifest_path = _validate_absolute_path(manifest_path, "checkpoint manifest")
    if manifest_path.name != MANIFEST_NAME:
        raise ValueError(f"checkpoint manifest must be named {MANIFEST_NAME}")
    output = manifest_path.parent
    chain = _open_absolute_directory_chain(output, "checkpoint output")
    try:
        names = tuple(sorted(os.listdir(chain[-1])))
        if names != (MANIFEST_NAME, SIDECAR_NAME):
            raise KerrReturningRadiationRefinementCheckpointError(
                "checkpoint output must be a closed two-file directory"
            )
        manifest = _read_published_at(
            chain[-1], MANIFEST_NAME, MAXIMUM_MANIFEST_BYTES
        )
        sidecar = _read_published_at(chain[-1], SIDECAR_NAME, 256)
        digest = hashlib.sha256(manifest).hexdigest()
        if expected_manifest_sha256 is not None and not hmac.compare_digest(
            digest,
            expected_manifest_sha256,
        ):
            raise KerrReturningRadiationRefinementCheckpointError(
                "checkpoint manifest differs from the external expected SHA-256"
            )
        if sidecar != f"{digest}  {MANIFEST_NAME}\n".encode("ascii"):
            raise KerrReturningRadiationRefinementCheckpointError(
                "checkpoint manifest sidecar differs"
            )
        document = _strict_canonical_manifest(manifest)
        qualified = _verify_checkpoint_document(document)
        evaluator_mode = document.get("evaluatorMode", PYTHON_EVALUATOR_MODE)
        if evaluator_mode == PYTHON_EVALUATOR_MODE:
            _require_exact_source_module_origins()
        else:
            _require_exact_source_module_origins(evaluator_mode)
        if qualified and expected_manifest_sha256 is None:
            raise KerrReturningRadiationRefinementCheckpointError(
                "qualified checkpoint acceptance requires an external expected "
                "manifest SHA-256"
            )
        source = _source_snapshot(evaluator_mode)
        runtime = _current_runtime_snapshot_for_manifest(
            document,
            native_library_path,
        )
        freeze = document.get("sourceFreeze")
        if (
            type(freeze) is not dict
            or freeze.get("stable") is not True
            or freeze.get("beforeExecution") != source
            or freeze.get("afterAuthentication") != source
            or document.get("runtimeNumericBackend") != runtime
        ):
            raise KerrReturningRadiationRefinementCheckpointError(
                "checkpoint is not source/runtime-current"
            )
        closing_manifest = _read_published_at(
            chain[-1], MANIFEST_NAME, MAXIMUM_MANIFEST_BYTES
        )
        closing_sidecar = _read_published_at(chain[-1], SIDECAR_NAME, 256)
        if closing_manifest != manifest or closing_sidecar != sidecar:
            raise KerrReturningRadiationRefinementCheckpointError(
                "checkpoint bytes changed during verification"
            )
        closing_names = tuple(sorted(os.listdir(chain[-1])))
        if closing_names != (MANIFEST_NAME, SIDECAR_NAME):
            raise KerrReturningRadiationRefinementCheckpointError(
                "checkpoint output changed from its closed two-file tree"
            )
        _require_same_directory_path(
            output,
            chain[-1],
            "checkpoint output",
        )
    finally:
        for descriptor in reversed(chain):
            os.close(descriptor)
    return KerrReturningRadiationVerifiedRefinementCheckpoint(
        output,
        manifest_path,
        digest,
        document["id"],
        qualified,
        document,
    )


_POST_PUBLICATION_VERIFIER_PUBLIC_ENTRY = (
    verify_kerr_returning_radiation_refinement_checkpoint
)


def _publish_document(
    output: Path, document: Mapping[str, Any], qualified: bool
) -> KerrReturningRadiationRefinementPublication:
    payload = canonical_json_bytes(document)
    if len(payload) > MAXIMUM_MANIFEST_BYTES:
        raise KerrReturningRadiationRefinementCheckpointError(
            "refinement checkpoint manifest exceeds its hard byte limit"
        )
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = f"{digest}  {MANIFEST_NAME}\n".encode("ascii")
    parent_chain = _open_absolute_directory_chain(output.parent, "checkpoint parent")
    parent_descriptor = parent_chain[-1]
    staging_name = f".{output.name}.staging-{uuid.uuid4().hex}"
    staging_descriptor = -1
    staging_owned = False
    promoted = False
    try:
        try:
            os.stat(output.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite existing checkpoint {output}")
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_descriptor)
        staging_owned = True
        try:
            status = os.stat(
                staging_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as error:
            raise KerrReturningRadiationRefinementCheckpointError(
                "checkpoint staging directory vanished after creation"
            ) from error
        if not stat.S_ISDIR(status.st_mode):
            raise KerrReturningRadiationRefinementCheckpointError(
                "checkpoint staging path is not a directory"
            )
        staging_descriptor = os.open(
            staging_name, _directory_open_flags(), dir_fd=parent_descriptor
        )
        _write_exclusive_at(staging_descriptor, MANIFEST_NAME, payload)
        _write_exclusive_at(staging_descriptor, SIDECAR_NAME, sidecar)
        os.fsync(staging_descriptor)
        if (
            _read_published_at(staging_descriptor, MANIFEST_NAME, MAXIMUM_MANIFEST_BYTES)
            != payload
            or _read_published_at(staging_descriptor, SIDECAR_NAME, 256) != sidecar
        ):
            raise KerrReturningRadiationRefinementCheckpointError(
                "staged checkpoint bytes differ from their canonical payload"
            )
        _promote_no_replace_at(
            parent_descriptor, staging_name, output.name, output
        )
        staging_owned = False
        promoted = True
        os.fsync(parent_descriptor)
        output_descriptor = os.open(
            output.name, _directory_open_flags(), dir_fd=parent_descriptor
        )
        try:
            if (
                _read_published_at(
                    output_descriptor, MANIFEST_NAME, MAXIMUM_MANIFEST_BYTES
                )
                != payload
                or _read_published_at(output_descriptor, SIDECAR_NAME, 256)
                != sidecar
            ):
                raise KerrReturningRadiationRefinementCheckpointError(
                    "published checkpoint bytes differ from their canonical payload"
                )
        finally:
            os.close(output_descriptor)
        _require_same_directory_path(
            output.parent,
            parent_descriptor,
            "checkpoint parent",
        )
        output_chain = _open_absolute_directory_chain(output, "checkpoint output")
        try:
            output_status = os.stat(
                output.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                output_status.st_dev,
                output_status.st_ino,
                output_status.st_mode,
            ) != _directory_identity(output_chain[-1]):
                raise KerrReturningRadiationRefinementCheckpointError(
                    "published checkpoint path identity changed"
                )
        finally:
            for descriptor in reversed(output_chain):
                os.close(descriptor)
    finally:
        try:
            if staging_owned and not promoted and staging_descriptor >= 0:
                for member in (MANIFEST_NAME, SIDECAR_NAME):
                    try:
                        os.unlink(member, dir_fd=staging_descriptor)
                    except FileNotFoundError:
                        pass
        finally:
            try:
                if staging_descriptor >= 0:
                    os.close(staging_descriptor)
            finally:
                try:
                    if staging_owned and not promoted:
                        try:
                            os.rmdir(staging_name, dir_fd=parent_descriptor)
                        except FileNotFoundError:
                            pass
                finally:
                    for descriptor in reversed(parent_chain):
                        os.close(descriptor)
    return KerrReturningRadiationRefinementPublication(
        output,
        output / MANIFEST_NAME,
        digest,
        str(document["id"]),
        qualified,
    )


def _cache_definition_for_plan(
    plan: KerrReturningRadiationRefinementPlan,
):
    common = {
        "termination": plan.termination,
        "annulus_edges_over_mass": plan.annulus_edges_over_mass,
        "ray_options": plan.ray_options,
        "surface_options": plan.surface_options,
        "policy": plan.kernel_policy,
        "area_policy": plan.area_policy,
        "directions_per_task": plan.directions_per_task,
    }
    if plan.evaluator_mode == PYTHON_EVALUATOR_MODE:
        return _CACHE_DEFINITION_CALL_ENTRY(plan.surface, **common)
    assert plan.native_library_path is not None
    assert plan.native_segment_capacity is not None
    assert plan.native_crossing_capacity is not None
    return _NATIVE_CACHE_DEFINITION_CALL_ENTRY(
        plan.surface,
        native_library_path=plan.native_library_path,
        native_segment_capacity=plan.native_segment_capacity,
        native_crossing_capacity=plan.native_crossing_capacity,
        **common,
    )


def _cached_execution_for_plan(
    plan: KerrReturningRadiationRefinementPlan,
    cache: Path,
) -> KerrCachedReturningRadiationKernelExecution:
    common = {
        "termination": plan.termination,
        "annulus_edges_over_mass": plan.annulus_edges_over_mass,
        "cache_root": cache,
        "ray_options": plan.ray_options,
        "surface_options": plan.surface_options,
        "policy": plan.kernel_policy,
        "area_policy": plan.area_policy,
        "directions_per_task": plan.directions_per_task,
        "jobs": plan.jobs,
        "max_in_flight": plan.max_in_flight,
    }
    if plan.evaluator_mode == PYTHON_EVALUATOR_MODE:
        return _CACHED_INTEGRATOR_CALL_ENTRY(plan.surface, **common)
    assert plan.native_library_path is not None
    assert plan.native_segment_capacity is not None
    assert plan.native_crossing_capacity is not None
    return _NATIVE_CACHED_INTEGRATOR_CALL_ENTRY(
        plan.surface,
        native_library_path=plan.native_library_path,
        native_segment_capacity=plan.native_segment_capacity,
        native_crossing_capacity=plan.native_crossing_capacity,
        **common,
    )


def execute_kerr_returning_radiation_refinement_checkpoint(
    plan: KerrReturningRadiationRefinementPlan,
) -> KerrReturningRadiationRefinementPublication:
    """Run/resume K, authenticate five grids, and publish qualified or not."""

    _assert_runtime_bindings()
    if _CLI_ORIGIN_GUARD_CALL_ENTRY is not None:
        _CLI_ORIGIN_GUARD_CALL_ENTRY()
    if type(plan) is not KerrReturningRadiationRefinementPlan:
        raise TypeError("plan must have its exact refinement type")
    if plan.evaluator_mode == PYTHON_EVALUATOR_MODE:
        _require_exact_source_module_origins()
    else:
        _require_exact_source_module_origins(plan.evaluator_mode)
    output, cache = _validate_checkpoint_paths(
        plan.output_directory, plan.cache_root
    )
    definition = _cache_definition_for_plan(plan)
    if _evaluator_mode_from_definition(definition) != plan.evaluator_mode:
        raise KerrReturningRadiationRefinementCheckpointError(
            "planned cache definition evaluator mode differs"
        )
    source_before = _source_snapshot(plan.evaluator_mode)
    runtime_before = _runtime_snapshot(definition)
    if plan.jobs > min(MAXIMUM_WORKERS, definition.plan.task_count):
        raise ValueError("jobs exceeds min(hard maximum, task count)")
    if plan.max_in_flight is not None and plan.max_in_flight > min(
        MAXIMUM_IN_FLIGHT_TASKS, definition.plan.task_count
    ):
        raise ValueError("max_in_flight exceeds min(hard maximum, task count)")
    _assert_snapshots_stable(
        source_before,
        runtime_before,
        "during planning",
        evaluator_mode=plan.evaluator_mode,
        definition=definition,
    )

    execution = _cached_execution_for_plan(plan, cache)
    if type(execution) is not KerrCachedReturningRadiationKernelExecution:
        raise TypeError("cached integrator returned a foreign execution type")
    if (
        _evaluator_mode_from_definition(execution.cache_definition)
        != plan.evaluator_mode
        or
        execution.cache_definition.scientific_job_key
        != definition.scientific_job_key
        or execution.cache_definition.job_spec.job_key
        != definition.job_spec.job_key
        or canonical_json_bytes(execution.cache_definition.job_spec.as_dict())
        != canonical_json_bytes(definition.job_spec.as_dict())
    ):
        raise KerrReturningRadiationRefinementCheckpointError(
            "cached execution differs from the planned definition"
        )
    _assert_snapshots_stable(
        source_before,
        runtime_before,
        "during cache execution",
        evaluator_mode=plan.evaluator_mode,
        definition=definition,
    )

    authenticated = _AUTHENTICATE_CALL_ENTRY(
        execution, plan.convergence_policy
    )
    authenticated_documents = _authenticated_documents(authenticated)
    validated = _VALIDATE_CACHED_CALL_ENTRY(execution)
    rebuilt = validated.execution
    if (
        rebuilt.kernel.model_descriptor_sha256
        != authenticated_documents["descriptor"]["kernelDescriptorSha256"]
    ):
        raise KerrReturningRadiationRefinementCheckpointError(
            "validated kernel and authenticated convergence descriptors differ"
        )
    binding_json = validated.scientific_binding_json
    binding_sha = validated.scientific_binding_sha256
    if _PRODUCTION_BINDING_CALL_ENTRY(rebuilt) != (binding_json, binding_sha):
        raise KerrReturningRadiationRefinementCheckpointError(
            "cached-forward scientific binding changed after validation"
        )
    _assert_snapshots_stable(
        source_before,
        runtime_before,
        "during authentication",
        evaluator_mode=plan.evaluator_mode,
        definition=definition,
    )

    document, qualified = _checkpoint_document(
        rebuilt,
        authenticated,
        source_before,
        runtime_before,
        binding_json,
        binding_sha,
    )
    if qualified is not authenticated_documents["qualified"]:
        raise AssertionError("checkpoint qualification changed during construction")
    _assert_snapshots_stable(
        source_before,
        runtime_before,
        "before publication",
        evaluator_mode=plan.evaluator_mode,
        definition=definition,
    )
    publication = _publish_document(output, document, qualified)
    _assert_snapshots_stable(
        source_before,
        runtime_before,
        "after publication",
        evaluator_mode=plan.evaluator_mode,
        definition=definition,
    )
    verification_keywords: dict[str, Any] = {
        "expected_manifest_sha256": publication.manifest_sha256,
    }
    if plan.evaluator_mode == NATIVE_CPU_EVALUATOR_MODE:
        verification_keywords["native_library_path"] = plan.native_library_path
    verified = _POST_PUBLICATION_VERIFIER_PUBLIC_ENTRY(
        publication.manifest_path,
        **verification_keywords,
    )
    _assert_snapshots_stable(
        source_before,
        runtime_before,
        "after verification",
        evaluator_mode=plan.evaluator_mode,
        definition=definition,
    )
    verified = _POST_PUBLICATION_VERIFIER_PUBLIC_ENTRY(
        publication.manifest_path,
        **verification_keywords,
    )
    if (
        type(verified) is not KerrReturningRadiationVerifiedRefinementCheckpoint
        or verified.output_directory != publication.output_directory
        or verified.manifest_path != publication.manifest_path
        or verified.manifest_sha256 != publication.manifest_sha256
        or verified.checkpoint_id != publication.checkpoint_id
        or verified.qualified is not publication.qualified
    ):
        raise KerrReturningRadiationRefinementCheckpointError(
            "published checkpoint differs from secure post-publication verification"
        )
    return KerrReturningRadiationRefinementPublication(
        verified.output_directory,
        verified.manifest_path,
        verified.manifest_sha256,
        verified.checkpoint_id,
        verified.qualified,
    )


# Owner-frozen identity lets command-line and other public consumers reject an
# execute-entry rebinding that happened before those consumers were imported.
# The private canonical handle remains inside the declared same-process trust
# boundary; public and owning-module rebinding do not.
_EXECUTE_REFINEMENT_CHECKPOINT_CANONICAL_ENTRY: Final = (
    execute_kerr_returning_radiation_refinement_checkpoint
)


__all__ = (
    "IMPLEMENTATION_ID",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "NATIVE_IMPLEMENTATION_ID",
    "NATIVE_MANIFEST_SCHEMA",
    "MAXIMUM_MANIFEST_BYTES",
    "MAXIMUM_JSON_NESTING_DEPTH",
    "NATIVE_CPU_EVALUATOR_MODE",
    "PYTHON_EVALUATOR_MODE",
    "SIDECAR_NAME",
    "KerrReturningRadiationRefinementCheckpointError",
    "KerrReturningRadiationRefinementPlan",
    "KerrReturningRadiationRefinementPublication",
    "KerrReturningRadiationVerifiedRefinementCheckpoint",
    "execute_kerr_returning_radiation_refinement_checkpoint",
    "verify_kerr_returning_radiation_refinement_checkpoint",
)
