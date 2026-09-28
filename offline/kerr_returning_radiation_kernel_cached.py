"""Same-code resumable execution for the two Kerr returning kernels.

This module connects the authenticated direction cache in
``kerr_returning_radiation_kernel_jobs`` to the existing forward and
receiver-centred reducers.  Cache records contain only a narrow, publicly
replayed direction transport plus exact quadrature-node evidence.  Matrix
weights, convergence gates, audit hashes, descriptors, and public result
construction remain owned by the original kernel modules.

The cache is execution evidence, not an independent ray or physics oracle.
The public kernel descriptor is intentionally byte-identical to a direct run
with the same direction transports; worker count and chunk layout are absent
from that scientific descriptor.  Each fixed evaluator is additionally bound
by an exact callable/source/code descriptor which the jobs runner checks before
cache access.  This prevents accidental public-runner substitution; it is not
protection from malicious code already executing in the same Python process.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
from functools import lru_cache
import hashlib
import importlib
import json
import marshal
import math
import os
import platform
from pathlib import Path
import stat
import struct
import sys
from types import CodeType, FunctionType, MappingProxyType, ModuleType
from typing import Any, Final

from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.job import (
    InputArtifact,
    JobRun,
    TaskKey,
    TaskResult,
    canonical_json_bytes,
)
from offline.kerr import KerrOblateTermination
from offline.kerr_finite_thickness import LOWER, UPPER
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
import offline.kerr_returning_radiation_kernel as _forward
from offline.kerr_returning_radiation_kernel import (
    KerrForwardReturningRadiationKernel,
    KerrReturningRadiationKernelPolicy,
)
from offline.kerr_returning_radiation_kernel_jobs import (
    EVALUATOR_DESCRIPTOR_SCHEMA,
    FORWARD,
    RECEIVER,
    KerrCachedKernelDirectionRecord,
    KerrKernelDirectionCacheDefinition,
    KerrKernelDirectionCoordinate,
    KerrKernelDirectionTaskPlan,
    KerrKernelEvaluatorRuntimeBinding,
    KerrKernelScientificContext,
    build_forward_kernel_scientific_identity,
    build_receiver_kernel_scientific_identity,
    iter_cached_kernel_direction_records,
    make_kernel_direction_evaluator_runtime_binding,
    make_kernel_direction_cache_definition,
    require_complete_existing_kernel_direction_cache,
    run_kernel_direction_cache,
)
import offline.kerr_returning_radiation_kernel_jobs as _jobs
import offline.kerr_returning_radiation_receiver_kernel as _receiver
from offline.kerr_returning_radiation_receiver_kernel import (
    KerrReceiverReturningRadiationKernel,
)
from offline.returning_radiation import AxisymmetricReturningRadiationKernel


IMPLEMENTATION_ID: Final = "kerr-returning-radiation-kernel-cache-bridge/v2"
CACHED_FORWARD_SCIENTIFIC_BINDING_IMPLEMENTATION_ID: Final = (
    f"{IMPLEMENTATION_ID}/validated-production-forward-source/v2"
)
CACHED_NATIVE_FORWARD_SCIENTIFIC_BINDING_IMPLEMENTATION_ID: Final = (
    f"{IMPLEMENTATION_ID}/validated-production-native-forward-source/v1"
)
REDUCTION_CONFIGURATION_SCHEMA: Final = (
    "blackhole.returning-radiation-kernel-reduction-configuration/v1"
)
FORWARD_TRANSPORT_SCHEMA: Final = (
    "blackhole.kerr-forward-returning-direction-transport/v1"
)
RECEIVER_TRANSPORT_SCHEMA: Final = (
    "blackhole.kerr-receiver-returning-direction-transport/v1"
)
_FORWARD_EVALUATOR_ID: Final = (
    "offline.kerr_returning_radiation_kernel_cached._evaluate_forward_direction/v1"
)
_NATIVE_FORWARD_EVALUATOR_ID: Final = (
    "offline.kerr_returning_radiation_kernel_cached."
    "_evaluate_forward_direction_native_cpu/v1"
)
_RECEIVER_EVALUATOR_ID: Final = (
    "offline.kerr_returning_radiation_kernel_cached._evaluate_receiver_direction/v1"
)
_SYNTHETIC_FORWARD_EVALUATOR_ID: Final = (
    "offline.kerr_returning_radiation_kernel_cached."
    "_synthetic_forward_direction_evaluator/test-only-v1"
)
_SYNTHETIC_RECEIVER_EVALUATOR_ID: Final = (
    "offline.kerr_returning_radiation_kernel_cached."
    "_synthetic_receiver_direction_evaluator/test-only-v1"
)
_EVALUATOR_IDS: Final = (
    _FORWARD_EVALUATOR_ID,
    _NATIVE_FORWARD_EVALUATOR_ID,
    _RECEIVER_EVALUATOR_ID,
    _SYNTHETIC_FORWARD_EVALUATOR_ID,
    _SYNTHETIC_RECEIVER_EVALUATOR_ID,
)
_REQUIRE_COMPLETE_EXISTING_CACHE_PUBLIC_ENTRY: Final = (
    _jobs._REQUIRE_COMPLETE_EXISTING_CACHE_CANONICAL_ENTRY
)

_SOURCE_CLOSURE_PATHS: Final = (
    "offline/__init__.py",
    "offline/disk_atmosphere.py",
    "offline/geodesic.py",
    "offline/job.py",
    "offline/kerr.py",
    "offline/kerr_disk.py",
    "offline/kerr_finite_thickness.py",
    "offline/kerr_finite_thickness_area.py",
    "offline/kerr_finite_thickness_emitter.py",
    "offline/kerr_finite_thickness_launch.py",
    "offline/kerr_finite_thickness_surface.py",
    "offline/kerr_returning_radiation_kernel.py",
    "offline/kerr_returning_radiation_kernel_cached.py",
    "offline/kerr_returning_radiation_kernel_jobs.py",
    "offline/kerr_returning_radiation_rays.py",
    "offline/kerr_returning_radiation_receiver_kernel.py",
    "offline/kerr_returning_radiation_receiver_rays.py",
    "offline/novikov_thorne.py",
    "offline/radiative_transfer.py",
    "offline/returning_radiation.py",
    "offline/returning_radiation_fate_quadrature.py",
    "offline/spacetime.py",
)
_NATIVE_FORWARD_ADDITIONAL_SOURCE_CLOSURE_PATHS: Final = (
    "native/cpu/Makefile",
    "native/cpu/include/blackhole_cpu.h",
    "native/cpu/src/blackhole_cpu.c",
    "native/cpu/tools/audit_binary.py",
    "offline/kerr_native_cpu_backend.py",
    "offline/kerr_returning_radiation_native_cpu.py",
)
_NATIVE_FORWARD_SOURCE_CLOSURE_PATHS: Final = tuple(
    sorted(
        set(
            (
                *_SOURCE_CLOSURE_PATHS,
                *_NATIVE_FORWARD_ADDITIONAL_SOURCE_CLOSURE_PATHS,
            )
        )
    )
)
_FROZEN_SOURCE_MODULE_FILE: Final = Path(os.path.abspath(__file__))
_FROZEN_SOURCE_ROOT: Final = _FROZEN_SOURCE_MODULE_FILE.parents[1]
_NUMERIC_BACKEND_LOGICAL_PATH: Final = "runtime/numeric-backend.json"
_SOURCE_AUTHENTICATION_READ_CHUNK_BYTES: Final = 1024 * 1024
_SOURCE_CLOSURE_MAXIMUM_FILE_BYTES: Final = 16 * 1024 * 1024
_SOURCE_CLOSURE_MAXIMUM_TOTAL_BYTES: Final = 64 * 1024 * 1024
_EVALUATOR_SOURCE_MAXIMUM_BYTES: Final = _SOURCE_CLOSURE_MAXIMUM_FILE_BYTES
_EVALUATOR_ARTIFACT_URI_PREFIX: Final = (
    "urn:blackhole:returning-radiation-kernel-direction-evaluator:v2:"
)

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": "same-code authenticated direction-cache execution bridge",
        "implementationId": IMPLEMENTATION_ID,
        "reusesOriginalKernelReducer": True,
        "reusesOriginalKernelResultBuilder": True,
        "cacheLayoutExcludedFromKernelScientificIdentity": True,
        "workerCountExcludedFromKernelScientificIdentity": True,
        "transportScientificJobKeyExcludesAreaQuadrature": True,
        "transportScientificJobKeyExcludesReductionConvergence": True,
        "transportScientificJobKeyExcludesReductionSymmetryThresholds": True,
        "transportScientificJobKeyExcludesReductionWorkBudgets": True,
        "quadratureOrdersBoundByTransportScientificPlan": True,
        "cachedExecutionBindsExactReductionConfiguration": True,
        "reductionConfigurationReplayedByOriginalKernelReducer": True,
        "fixedTransitiveSourceClosureBound": True,
        "sourceModuleOriginsRecheckedBeforeAuthentication": True,
        "sourceModuleExpectedPathsFrozenAtImport": True,
        "numericBackendAndNativeArtifactsBound": True,
        "fixedEvaluatorIdentityArtifactBound": True,
        "fixedEvaluatorCallableDescriptorCheckedBeforeCacheAccess": True,
        "claimsProtectionFromMaliciousSameProcessCodeExecution": False,
        "sourceClosureRecheckedBeforeAndAfterReduction": True,
        "acceptsArbitraryPublicExecutorFactory": False,
        "kernelWorkBudgetRetainsDirectScientificReplayAccounting": True,
        "executionAuditSeparatesExecutedAndReusedDirections": True,
        "taskReuseHistoryIsCryptographicallyAuthenticated": False,
        "cachedReplayRetracesRays": False,
        "directKernelRevalidationStillRetracesRays": True,
        "isSameCodeCacheEvidence": True,
        "hasIndependentGeodesicOracle": False,
        "hasIndependentPhysicsOracle": False,
        "prohibitedClaim": (
            "Do not describe cache authentication or cached same-code replay as "
            "an independent geodesic/physics oracle, continuum validation, or "
            "protection from malicious same-process code execution."
        ),
    }
)


def _source_module_name(logical_path: str) -> str | None:
    """Map one fixed closure path to its import-system owner."""

    if logical_path == "offline/__init__.py":
        return "offline"
    if logical_path in _NATIVE_FORWARD_ADDITIONAL_SOURCE_CLOSURE_PATHS and not (
        logical_path.startswith("offline/") and logical_path.endswith(".py")
    ):
        return None
    if (
        type(logical_path) is not str
        or not logical_path.startswith("offline/")
        or not logical_path.endswith(".py")
        or "/" in logical_path[len("offline/") : -len(".py")]
    ):
        raise KerrReturningRadiationKernelCacheError(
            "source closure contains an unsupported Python module path"
        )
    return logical_path[: -len(".py")].replace("/", ".")


_SOURCE_CLOSURE_MODULE_OWNERS: Final = tuple(
    (
        _source_module_name(logical_path),
        sys.modules.get(_source_module_name(logical_path)),
        _FROZEN_SOURCE_ROOT / logical_path,
    )
    for logical_path in _SOURCE_CLOSURE_PATHS
)
_NATIVE_FORWARD_SOURCE_CLOSURE_MODULE_OWNERS: Final = tuple(
    (
        module_name,
        None if module_name is None else sys.modules.get(module_name),
        _FROZEN_SOURCE_ROOT / logical_path,
    )
    for logical_path in _NATIVE_FORWARD_SOURCE_CLOSURE_PATHS
    for module_name in (_source_module_name(logical_path),)
)


def _require_exact_source_module_origins(
    *,
    _source_module_file: Path = _FROZEN_SOURCE_MODULE_FILE,
    _source_root: Path = _FROZEN_SOURCE_ROOT,
    _source_paths: tuple[str, ...] = _SOURCE_CLOSURE_PATHS,
    _module_owners: tuple[tuple[str, Any, Path], ...] = (
        _SOURCE_CLOSURE_MODULE_OWNERS
    ),
) -> Path:
    """Bind every executed closure module to the exact tree being hashed."""

    module_file = globals().get("__file__")
    legacy = (
        _source_paths is _SOURCE_CLOSURE_PATHS
        and _module_owners is _SOURCE_CLOSURE_MODULE_OWNERS
    )
    native = (
        _source_paths is _NATIVE_FORWARD_SOURCE_CLOSURE_PATHS
        and _module_owners is _NATIVE_FORWARD_SOURCE_CLOSURE_MODULE_OWNERS
    )
    if (
        type(module_file) is not str
        or Path(os.path.abspath(module_file)) != _source_module_file
        or not (legacy or native)
    ):
        raise KerrReturningRadiationKernelCacheError(
            "cached kernel module or closure has no frozen source identity"
        )
    for logical_path, (module_name, expected_module, expected_file) in zip(
        _source_paths,
        _module_owners,
    ):
        if module_name is None:
            if (
                not native
                or logical_path
                not in _NATIVE_FORWARD_ADDITIONAL_SOURCE_CLOSURE_PATHS
            ):
                raise KerrReturningRadiationKernelCacheError(
                    "source closure contains an unauthenticated non-module path"
            )
            continue
        module = sys.modules.get(module_name)
        lazy_native_module = (
            native
            and expected_module is None
            and logical_path
            in _NATIVE_FORWARD_ADDITIONAL_SOURCE_CLOSURE_PATHS
        )
        if type(module) is not ModuleType or (
            not lazy_native_module and module is not expected_module
        ):
            raise KerrReturningRadiationKernelCacheError(
                f"source-closure module {module_name} is not the exact loaded module"
            )
        actual_file = getattr(module, "__file__", None)
        if (
            type(actual_file) is not str
            or Path(os.path.abspath(actual_file)) != expected_file
        ):
            raise KerrReturningRadiationKernelCacheError(
                f"source-closure module {module_name} was loaded from another tree"
            )
    return _source_root


def _require_cache_only_runtime_binding() -> None:
    """Reject a replaced existing-cache authenticator before cache access."""

    if (
        _jobs._REQUIRE_COMPLETE_EXISTING_CACHE_CANONICAL_ENTRY
        is not _REQUIRE_COMPLETE_EXISTING_CACHE_PUBLIC_ENTRY
        or require_complete_existing_kernel_direction_cache
        is not _REQUIRE_COMPLETE_EXISTING_CACHE_PUBLIC_ENTRY
        or _jobs.require_complete_existing_kernel_direction_cache
        is not _REQUIRE_COMPLETE_EXISTING_CACHE_PUBLIC_ENTRY
    ):
        raise KerrReturningRadiationKernelCacheError(
            "complete existing-cache authenticator runtime binding changed"
        )


class KerrReturningRadiationKernelCacheError(RuntimeError):
    """Raised when cached direction execution cannot be reduced safely."""


class KerrReturningRadiationKernelCacheVerificationError(
    KerrReturningRadiationKernelCacheError
):
    """Raised when cached replay differs from its exact authenticated tree."""


@dataclass(frozen=True, slots=True)
class KerrKernelSourceClosureEntry:
    logical_path: str
    byte_length: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.logical_path) is not str or not self.logical_path:
            raise TypeError("source-closure logical_path must be a non-empty exact str")
        if type(self.byte_length) is not int or self.byte_length < 0:
            raise TypeError("source-closure byte_length must be a non-negative exact int")
        _lowercase_sha256(self.sha256, "source-closure sha256")

    @property
    def binding_sha256(self) -> str:
        """Bind dependency ownership, size, and content into one source hash."""

        payload = (
            f"{self.logical_path}\0{self.byte_length}\0{self.sha256}"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class KerrKernelReductionConfiguration:
    """Exact non-transport inputs required to rebuild one public kernel."""

    policy: KerrReturningRadiationKernelPolicy
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy

    def __post_init__(self) -> None:
        if type(self.policy) is not KerrReturningRadiationKernelPolicy:
            raise TypeError("reduction policy must have its exact type")
        if type(self.area_policy) is not KerrFiniteThicknessAreaQuadraturePolicy:
            raise TypeError("reduction area_policy must have its exact type")
        rebuilt_policy = KerrReturningRadiationKernelPolicy(**asdict(self.policy))
        rebuilt_area = KerrFiniteThicknessAreaQuadraturePolicy(
            **asdict(self.area_policy)
        )
        _require_trusted_exact_tree(
            self.policy,
            rebuilt_policy,
            "reduction_configuration.policy",
        )
        _require_trusted_exact_tree(
            self.area_policy,
            rebuilt_area,
            "reduction_configuration.area_policy",
        )
        object.__setattr__(self, "policy", rebuilt_policy)
        object.__setattr__(self, "area_policy", rebuilt_area)

    def as_dict(self) -> dict[str, Any]:
        return {
            "areaPolicy": asdict(self.area_policy),
            "kernelPolicy": asdict(self.policy),
            "schema": REDUCTION_CONFIGURATION_SCHEMA,
        }

    @property
    def descriptor_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class KerrCachedKernelExecutionAudit:
    formulation: str
    evaluator_id: str
    scientific_job_key: str
    cache_job_key: str
    authenticated_direction_records: int
    executed_direction_records: int
    reused_direction_records: int
    executed_tasks: int
    reused_tasks: int
    source_closure_manifest_sha256: str
    source_closure_rechecked_before_and_after_reduction: bool
    is_same_code_cache_evidence: bool
    is_independent_physics_or_geodesic_oracle: bool
    is_task_reuse_history_cryptographically_authenticated: bool

    def __post_init__(self) -> None:
        if type(self.formulation) is not str or self.formulation not in (
            FORWARD,
            RECEIVER,
        ):
            raise ValueError("audit formulation must be exact forward or receiver")
        if type(self.evaluator_id) is not str or not self.evaluator_id:
            raise TypeError("audit evaluator_id must be a non-empty exact str")
        for name in (
            "scientific_job_key",
            "cache_job_key",
            "source_closure_manifest_sha256",
        ):
            _lowercase_sha256(getattr(self, name), name)
        for name in (
            "authenticated_direction_records",
            "executed_direction_records",
            "reused_direction_records",
            "executed_tasks",
            "reused_tasks",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"audit {name} must be a non-negative exact int")
        if (
            self.executed_direction_records + self.reused_direction_records
            != self.authenticated_direction_records
        ):
            raise ValueError("audit direction accounting does not close")
        if self.source_closure_rechecked_before_and_after_reduction is not True:
            raise ValueError("cached execution requires pre/post source-closure gates")
        if self.is_same_code_cache_evidence is not True:
            raise ValueError("cached execution must declare same-code cache evidence")
        if self.is_independent_physics_or_geodesic_oracle is not False:
            raise ValueError("cached execution cannot claim an independent oracle")
        if self.is_task_reuse_history_cryptographically_authenticated is not False:
            raise ValueError("cache receipts do not authenticate prior reuse history")


KernelResult = (
    KerrForwardReturningRadiationKernel | KerrReceiverReturningRadiationKernel
)


@dataclass(frozen=True, slots=True)
class KerrCachedReturningRadiationKernelExecution:
    """Kernel plus exact cache definition, run receipts, and execution audit."""

    formulation: str
    kernel: KernelResult
    cache_definition: KerrKernelDirectionCacheDefinition
    reduction_configuration: KerrKernelReductionConfiguration
    job_run: JobRun
    source_closure: tuple[KerrKernelSourceClosureEntry, ...]
    execution_audit: KerrCachedKernelExecutionAudit

    def __post_init__(self) -> None:
        if type(self.formulation) is not str or self.formulation not in (
            FORWARD,
            RECEIVER,
        ):
            raise ValueError("execution formulation must be forward or receiver")
        expected_kernel_type = (
            KerrForwardReturningRadiationKernel
            if self.formulation == FORWARD
            else KerrReceiverReturningRadiationKernel
        )
        if type(self.kernel) is not expected_kernel_type:
            raise TypeError("execution kernel type disagrees with formulation")
        if type(self.cache_definition) is not KerrKernelDirectionCacheDefinition:
            raise TypeError("cache_definition must have its exact type")
        reduction_configuration = _rebuilt_reduction_configuration(
            self.reduction_configuration,
            "execution.reduction_configuration",
        )
        _require_reduction_configuration_covers_plan(
            reduction_configuration,
            self.cache_definition.plan,
        )
        kernel_configuration = KerrKernelReductionConfiguration(
            _trusted_attribute(self.kernel, "policy", "execution.kernel"),
            _trusted_attribute(
                self.kernel,
                "area_policy",
                "execution.kernel",
            ),
        )
        _require_trusted_exact_tree(
            reduction_configuration,
            kernel_configuration,
            "execution.reduction_configuration_vs_kernel",
        )
        if type(self.job_run) is not JobRun:
            raise TypeError("job_run must have its exact type")
        if type(self.source_closure) is not tuple or not self.source_closure:
            raise TypeError("source_closure must be a non-empty exact tuple")
        if any(type(item) is not KerrKernelSourceClosureEntry for item in self.source_closure):
            raise TypeError("source_closure contains a non-exact entry")
        if tuple(item.logical_path for item in self.source_closure) != tuple(
            sorted(item.logical_path for item in self.source_closure)
        ):
            raise ValueError("source_closure paths are not canonical")
        if type(self.execution_audit) is not KerrCachedKernelExecutionAudit:
            raise TypeError("execution_audit must have its exact type")
        if (
            self.cache_definition.plan.formulation != self.formulation
            or self.execution_audit.formulation != self.formulation
        ):
            raise ValueError("execution formulation evidence disagrees")
        object.__setattr__(
            self,
            "reduction_configuration",
            reduction_configuration,
        )

    @property
    def kernel_descriptor_sha256(self) -> str:
        return self.kernel.model_descriptor_sha256

    @property
    def reduction_configuration_sha256(self) -> str:
        return self.reduction_configuration.descriptor_sha256

    def revalidate_cached(self) -> None:
        verify_cached_kerr_returning_radiation_kernel_execution(self)


@dataclass(frozen=True, slots=True, init=False)
class KerrValidatedCachedForwardReduction:
    """Canonical production-forward cache proof plus its symmetry reduction.

    The scientific binding deliberately excludes cache roots, task chunking,
    worker scheduling, and unauthenticated reuse history.  ``execution`` is
    retained separately so downstream revalidation can still re-open every
    payload and receipt.
    """

    execution: KerrCachedReturningRadiationKernelExecution
    forward_kernel: KerrForwardReturningRadiationKernel
    axisymmetric_kernel: AxisymmetricReturningRadiationKernel
    scientific_binding_json: str
    scientific_binding_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "KerrValidatedCachedForwardReduction is built only by the "
            "production cached-forward validator"
        )

    def scientific_binding(self) -> Mapping[str, Any]:
        if type(self.scientific_binding_json) is not str:
            raise KerrReturningRadiationKernelCacheVerificationError(
                "cached-forward scientific binding has a non-exact type"
            )
        value = json.loads(self.scientific_binding_json)
        if type(value) is not dict:
            raise KerrReturningRadiationKernelCacheVerificationError(
                "cached-forward scientific binding is not an object"
            )
        return MappingProxyType(value)


@dataclass(frozen=True, slots=True)
class _ForwardCoordinateNode:
    pass_index: int
    pass_name: str
    source_face: str
    source_annulus_index: int
    rho_index: int
    mu_index: int
    psi_index: int
    source_radius_over_mass: float
    emission_angle_cosine: float
    tangent_azimuth_rad: float
    normalized_emitted_flux_weight: float


@dataclass(frozen=True, slots=True)
class _ReceiverCoordinateNode:
    pass_index: int
    pass_name: str
    receiver_face: str
    receiver_annulus_index: int
    rho_index: int
    mu_index: int
    psi_index: int
    receiver_radius_over_mass: float
    incidence_cosine: float
    mu_weight: float
    tangent_azimuth_rad: float
    phase_cells: float


def _lowercase_sha256(value: Any, label: str) -> str:
    if type(value) is not str or len(value) != 64 or value.lower() != value:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest") from error
    if len(decoded) != 32:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _trusted_attribute(value: Any, name: str, path: str) -> Any:
    try:
        return object.__getattribute__(value, name)
    except (AttributeError, TypeError) as error:
        raise KerrReturningRadiationKernelCacheVerificationError(
            f"{path}.{name} is missing"
        ) from error


def _require_trusted_exact_tree(actual: Any, expected: Any, path: str) -> None:
    """Compare immutable evidence without attacker-controlled equality hooks."""

    if type(actual) is not type(expected):
        raise KerrReturningRadiationKernelCacheVerificationError(
            f"{path} has non-exact type {type(actual).__name__}; "
            f"expected {type(expected).__name__}"
        )
    if is_dataclass(expected) and not isinstance(expected, type):
        for item in fields(expected):
            _require_trusted_exact_tree(
                _trusted_attribute(actual, item.name, path),
                _trusted_attribute(expected, item.name, path),
                f"{path}.{item.name}",
            )
        return
    if type(expected) is tuple:
        if len(actual) != len(expected):
            raise KerrReturningRadiationKernelCacheVerificationError(
                f"{path} tuple length differs"
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _require_trusted_exact_tree(
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
        return
    if type(expected) is float:
        differs = not math.isfinite(actual) or actual.hex() != expected.hex()
    elif type(expected) is str:
        differs = actual.encode("utf-8") != expected.encode("utf-8")
    elif type(expected) is int:
        differs = actual != expected
    elif type(expected) is bool:
        differs = actual is not expected
    elif expected is None:
        differs = False
    else:
        raise KerrReturningRadiationKernelCacheVerificationError(
            f"{path} uses unsupported trusted type {type(expected).__name__}"
        )
    if differs:
        raise KerrReturningRadiationKernelCacheVerificationError(
            f"{path} differs from exact reconstruction"
        )


def _rebuilt_reduction_configuration(
    value: Any,
    path: str,
) -> KerrKernelReductionConfiguration:
    if type(value) is not KerrKernelReductionConfiguration:
        raise TypeError(f"{path} must have its exact reduction-configuration type")
    rebuilt = KerrKernelReductionConfiguration(
        _trusted_attribute(value, "policy", path),
        _trusted_attribute(value, "area_policy", path),
    )
    _require_trusted_exact_tree(value, rebuilt, path)
    return rebuilt


def _require_reduction_configuration_covers_plan(
    configuration: KerrKernelReductionConfiguration,
    plan: KerrKernelDirectionTaskPlan,
) -> None:
    if type(configuration) is not KerrKernelReductionConfiguration:
        raise TypeError("reduction configuration must have its exact type")
    if type(plan) is not KerrKernelDirectionTaskPlan:
        raise TypeError("transport plan must have its exact type")
    policy = configuration.policy
    if (
        policy.rho_order != plan.rho_order
        or policy.mu_order != plan.mu_order
        or policy.psi_count != plan.psi_count
    ):
        raise ValueError("reduction policy orders differ from transport plan")
    if plan.direction_count > policy.maximum_direction_evaluations:
        raise ValueError(
            "reduction direction budget does not cover the transport plan"
        )
    whole_rays_per_direction = 2 if plan.formulation == FORWARD else 4
    if (
        whole_rays_per_direction * plan.direction_count
        > policy.maximum_whole_ray_traces
    ):
        raise ValueError(
            "reduction whole-ray budget does not cover the transport plan"
        )


def _source_stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return the metadata that must stay fixed across one source read."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _authenticated_regular_file_sha256(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[int, str]:
    """Hash one stable regular non-symlink through a single bounded fd.

    The extra byte in the read budget makes concurrent growth fail closed even
    when the initial ``st_size`` is exactly at the limit.  Successful reads
    bind both path ownership and the pre/post state of the open file.
    """

    if type(path) is not type(Path()):
        raise TypeError("authenticated source path must be an exact platform Path")
    if type(label) is not str or not label:
        raise TypeError("authenticated source label must be a non-empty exact str")
    if type(maximum_bytes) is not int or maximum_bytes < 0:
        raise TypeError("authenticated source byte limit must be a non-negative int")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int or nofollow == 0:
        raise KerrReturningRadiationKernelCacheError(
            f"cannot authenticate {label}: O_NOFOLLOW is unavailable"
        )
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    descriptor = -1
    try:
        path_before = os.lstat(path)
        if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(
            path_before.st_mode
        ):
            raise ValueError("source path is not a regular non-symlink file")
        descriptor = os.open(path, flags)
        opened_before = os.fstat(descriptor)
        before_identity = _source_stat_identity(opened_before)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or before_identity != _source_stat_identity(path_before)
        ):
            raise ValueError("source path changed while it was opened")
        if opened_before.st_size > maximum_bytes:
            raise ValueError("source file exceeds its fixed byte limit")

        digest = hashlib.sha256()
        byte_length = 0
        while True:
            remaining_with_sentinel = maximum_bytes - byte_length + 1
            request_bytes = min(
                _SOURCE_AUTHENTICATION_READ_CHUNK_BYTES,
                remaining_with_sentinel,
            )
            block = os.read(descriptor, request_bytes)
            if not block:
                break
            byte_length += len(block)
            if byte_length > maximum_bytes:
                raise ValueError("source file grew beyond its fixed byte limit")
            digest.update(block)

        opened_after = os.fstat(descriptor)
        path_after = os.lstat(path)
        after_identity = _source_stat_identity(opened_after)
        if (
            not stat.S_ISREG(opened_after.st_mode)
            or stat.S_ISLNK(path_after.st_mode)
            or not stat.S_ISREG(path_after.st_mode)
            or before_identity != after_identity
            or after_identity != _source_stat_identity(path_after)
            or byte_length != opened_after.st_size
        ):
            raise ValueError("source file changed while it was authenticated")
        return byte_length, digest.hexdigest()
    except KerrReturningRadiationKernelCacheError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise KerrReturningRadiationKernelCacheError(
            f"cannot authenticate {label}: {error}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _bounded_payload_sha256(
    payload: bytes,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[int, str]:
    """Apply the same fixed-size envelope to test-only in-memory sources."""

    if type(payload) is not bytes:
        raise TypeError("authenticated source payload must be exact bytes")
    if len(payload) > maximum_bytes:
        raise KerrReturningRadiationKernelCacheError(
            f"cannot authenticate {label}: source file exceeds its fixed byte limit"
        )
    return len(payload), hashlib.sha256(payload).hexdigest()


def _source_closure_manifest(
    byte_overrides: Mapping[str, bytes] | None = None,
    numeric_backend_override: dict[str, Any] | None = None,
    *,
    source_paths: tuple[str, ...] | None = None,
    module_owners: tuple[tuple[str | None, Any, Path], ...] | None = None,
) -> tuple[KerrKernelSourceClosureEntry, ...]:
    """Hash the fixed production closure; overrides exist only for unit proof."""

    if source_paths is None:
        source_paths = _SOURCE_CLOSURE_PATHS
        if module_owners is None:
            module_owners = _SOURCE_CLOSURE_MODULE_OWNERS
    elif module_owners is None:
        raise TypeError("explicit source paths require exact module owners")
    if byte_overrides is not None:
        if type(byte_overrides) is not dict or any(
            type(key) is not str or type(value) is not bytes
            for key, value in byte_overrides.items()
        ):
            raise TypeError("source byte overrides must be an exact str->bytes dict")
        unknown = set(byte_overrides) - set(source_paths)
        if unknown:
            raise ValueError("source byte override names an unknown dependency")
    root = _require_exact_source_module_origins(
        _source_paths=source_paths,
        _module_owners=module_owners,
    )
    entries: list[KerrKernelSourceClosureEntry] = []
    total_byte_length = 0
    for logical_path in source_paths:
        path = root / logical_path
        maximum_bytes = min(
            _SOURCE_CLOSURE_MAXIMUM_FILE_BYTES,
            _SOURCE_CLOSURE_MAXIMUM_TOTAL_BYTES - total_byte_length,
        )
        if maximum_bytes < 0:
            raise KerrReturningRadiationKernelCacheError(
                "source closure exceeds its fixed total byte limit"
            )
        if byte_overrides is not None and logical_path in byte_overrides:
            byte_length, sha256 = _bounded_payload_sha256(
                byte_overrides[logical_path],
                label=f"source-closure dependency {logical_path}",
                maximum_bytes=maximum_bytes,
            )
        else:
            byte_length, sha256 = _authenticated_regular_file_sha256(
                path,
                label=f"source-closure dependency {logical_path}",
                maximum_bytes=maximum_bytes,
            )
        total_byte_length += byte_length
        if total_byte_length > _SOURCE_CLOSURE_MAXIMUM_TOTAL_BYTES:
            raise KerrReturningRadiationKernelCacheError(
                "source closure exceeds its fixed total byte limit"
            )
        entries.append(
            KerrKernelSourceClosureEntry(
                logical_path,
                byte_length,
                sha256,
            )
        )
    backend = (
        _numeric_backend_descriptor()
        if numeric_backend_override is None
        else numeric_backend_override
    )
    if type(backend) is not dict:
        raise TypeError("numeric backend override must be an exact dict")
    try:
        backend_payload = canonical_json_bytes(backend)
    except (TypeError, ValueError) as error:
        raise KerrReturningRadiationKernelCacheError(
            "numeric backend descriptor is not finite canonical JSON"
        ) from error
    total_byte_length += len(backend_payload)
    if (
        len(backend_payload) > _SOURCE_CLOSURE_MAXIMUM_FILE_BYTES
        or total_byte_length > _SOURCE_CLOSURE_MAXIMUM_TOTAL_BYTES
    ):
        raise KerrReturningRadiationKernelCacheError(
            "source closure exceeds its fixed byte limits"
        )
    entries.append(
        KerrKernelSourceClosureEntry(
            _NUMERIC_BACKEND_LOGICAL_PATH,
            len(backend_payload),
            hashlib.sha256(backend_payload).hexdigest(),
        )
    )
    result = tuple(entries)
    if len({item.binding_sha256 for item in result}) != len(result):
        raise KerrReturningRadiationKernelCacheError(
            "source closure contains duplicate ownership-bound digests"
        )
    return result


def _runtime_artifact_descriptor(path: str, label: str) -> dict[str, Any]:
    try:
        artifact = Path(path).resolve(strict=True)
        if not artifact.is_file():
            raise ValueError(f"{label} is not a regular file")
        digest = hashlib.sha256()
        byte_length = 0
        with artifact.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                byte_length += len(block)
    except (OSError, TypeError, ValueError) as error:
        raise KerrReturningRadiationKernelCacheError(
            f"cannot authenticate the {label} runtime artifact"
        ) from error
    return {
        "artifactName": artifact.name,
        "byteLength": byte_length,
        "sha256": digest.hexdigest(),
    }


def _numeric_backend_descriptor() -> dict[str, Any]:
    math_extension = getattr(math, "__file__", None)
    if type(math_extension) is not str or not math_extension:
        raise KerrReturningRadiationKernelCacheError(
            "numeric backend requires an authenticated native math extension"
        )
    build_tag, build_date = platform.python_build()
    libc_name, libc_version = platform.libc_ver()
    return {
        "architectureBits": 8 * struct.calcsize("P"),
        "binary64MantissaBits": sys.float_info.mant_dig,
        "byteOrder": sys.byteorder,
        "floatRadix": sys.float_info.radix,
        "implementationId": "cpython-binary64-struct-libm/v2",
        "libc": {
            "implementation": libc_name,
            "version": libc_version,
        },
        "machine": platform.machine(),
        "mathExtension": _runtime_artifact_descriptor(
            math_extension,
            "math extension",
        ),
        "operatingSystem": platform.system(),
        "operatingSystemRelease": platform.release(),
        "operatingSystemVersion": platform.version(),
        "processor": platform.processor(),
        "pythonBuild": {
            "date": build_date,
            "tag": build_tag,
        },
        "pythonCacheTag": sys.implementation.cache_tag,
        "pythonCompiler": platform.python_compiler(),
        "pythonExecutable": _runtime_artifact_descriptor(
            sys.executable,
            "Python executable",
        ),
        "pythonImplementation": platform.python_implementation(),
        "pythonVersion": platform.python_version(),
        "structDoubleBytes": struct.calcsize("d"),
    }


def _native_numeric_backend_descriptor(
    backend_descriptor: dict[str, Any],
) -> dict[str, Any]:
    """Bind the Python control plane and exact path-free native backend."""

    if type(backend_descriptor) is not dict:
        raise TypeError("native backend descriptor must be an exact dict")
    try:
        backend_payload = canonical_json_bytes(backend_descriptor)
        rebuilt_backend = json.loads(backend_payload)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise KerrReturningRadiationKernelCacheError(
            "native backend descriptor is not finite canonical JSON"
        ) from error
    if type(rebuilt_backend) is not dict:
        raise TypeError("native backend descriptor must be an exact JSON object")
    return {
        "implementationId": (
            "cpython-control-plane+strict-kerr-cpu-whole-ray/abi-v3"
        ),
        "nativeWholeRay": rebuilt_backend,
        "nativeWholeRayDescriptorSha256": hashlib.sha256(
            backend_payload
        ).hexdigest(),
        "pythonControlPlane": _numeric_backend_descriptor(),
    }


def _native_forward_runtime_binding(
    native_library_path: Path,
    *,
    segment_capacity: int,
    crossing_capacity: int,
) -> tuple[KerrKernelEvaluatorRuntimeBinding, dict[str, Any]]:
    """Authenticate a parent-selected dylib without retaining its live backend."""

    backend_module = importlib.import_module("offline.kerr_native_cpu_backend")
    native_wrapper = importlib.import_module(
        "offline.kerr_returning_radiation_native_cpu"
    )

    if native_wrapper.__name__ != "offline.kerr_returning_radiation_native_cpu":
        raise KerrReturningRadiationKernelCacheError(
            "native returning-radiation wrapper has a foreign identity"
        )
    if type(native_library_path) is not type(Path()) or not (
        native_library_path.is_absolute()
    ):
        raise TypeError("native_library_path must be an exact absolute Path")
    backend_type = getattr(backend_module, "StrictKerrCpuBackend", None)
    if type(backend_type) is not type:
        raise KerrReturningRadiationKernelCacheError(
            "native backend module lacks its exact backend type"
        )
    backend = backend_type(
        native_library_path,
        segment_capacity=segment_capacity,
        crossing_capacity=crossing_capacity,
    )
    backend.revalidate()
    descriptor = backend.model_descriptor()
    binding = make_kernel_direction_evaluator_runtime_binding(
        evaluator_implementation_id=_NATIVE_FORWARD_EVALUATOR_ID,
        library_path=native_library_path,
        backend_descriptor=descriptor,
        segment_capacity=segment_capacity,
        crossing_capacity=crossing_capacity,
    )
    if (
        binding.backend_descriptor_sha256
        != backend.model_descriptor_sha256
        or descriptor.get("artifacts", {}).get("library", {}).get("sha256")
        != binding.library_sha256
    ):
        raise KerrReturningRadiationKernelCacheError(
            "native runtime binding differs from its authenticated backend"
        )
    return binding, descriptor


def _source_closure_manifest_sha256(
    entries: tuple[KerrKernelSourceClosureEntry, ...],
) -> str:
    payload = "\n".join(
        f"{item.logical_path}\0{item.byte_length}\0{item.sha256}"
        for item in entries
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_evaluator_code(code: CodeType, logical_path: str) -> CodeType:
    """Normalize filenames exactly as the jobs-layer v2 descriptor does."""

    if type(code) is not CodeType:
        raise TypeError("evaluator code must have the exact CodeType")
    constants = tuple(
        _normalized_evaluator_code(item, logical_path)
        if type(item) is CodeType
        else item
        for item in code.co_consts
    )
    return code.replace(co_consts=constants, co_filename=logical_path)


def _authenticated_evaluator_callable_descriptor(
    evaluator: Any,
    implementation_id: str,
) -> dict[str, Any]:
    """Build the jobs-layer v2 descriptor with bounded stable source I/O."""

    if type(evaluator) is not FunctionType:
        raise TypeError("a reserved evaluator must be an exact Python function")
    module = object.__getattribute__(evaluator, "__module__")
    name = object.__getattribute__(evaluator, "__name__")
    qualname = object.__getattribute__(evaluator, "__qualname__")
    code = object.__getattribute__(evaluator, "__code__")
    defaults = object.__getattribute__(evaluator, "__defaults__")
    keyword_defaults = object.__getattribute__(evaluator, "__kwdefaults__")
    closure = object.__getattribute__(evaluator, "__closure__")
    annotations = object.__getattribute__(evaluator, "__annotations__")
    custom_attributes = object.__getattribute__(evaluator, "__dict__")
    if any(type(item) is not str or not item for item in (module, name, qualname)):
        raise TypeError("reserved evaluator metadata must use non-empty exact strings")
    if "<locals>" in qualname:
        raise ValueError("a reserved evaluator must be a module-level function")
    if type(code) is not CodeType:
        raise TypeError("reserved evaluator __code__ must have the exact CodeType")
    if (
        type(code.co_name) is not str
        or type(code.co_qualname) is not str
        or code.co_name.encode("utf-8") != name.encode("utf-8")
        or code.co_qualname.encode("utf-8") != qualname.encode("utf-8")
    ):
        raise ValueError("reserved evaluator metadata disagrees with its code object")
    if defaults is not None or keyword_defaults is not None:
        raise ValueError("reserved evaluator defaults are forbidden")
    if closure is not None or code.co_freevars:
        raise ValueError("reserved evaluator closures are forbidden")
    if type(custom_attributes) is not dict or custom_attributes:
        raise ValueError("reserved evaluator custom function attributes are forbidden")
    if type(annotations) is not dict:
        raise TypeError("reserved evaluator annotations must be an exact dict")
    annotation_document: dict[str, str] = {}
    for key in sorted(annotations):
        value = annotations[key]
        if type(key) is not str or type(value) is not str:
            raise TypeError(
                "reserved evaluator annotations must be exact string pairs"
            )
        annotation_document[key] = value

    filename = code.co_filename
    if type(filename) is not str or not filename:
        raise TypeError("reserved evaluator code filename must be a non-empty exact str")
    source_root = _require_exact_source_module_origins()
    try:
        source_path = Path(os.path.abspath(filename))
        logical_path = source_path.relative_to(source_root).as_posix()
        source_byte_length, source_sha256 = _authenticated_regular_file_sha256(
            source_path,
            label="reserved evaluator project source",
            maximum_bytes=_EVALUATOR_SOURCE_MAXIMUM_BYTES,
        )
    except (KerrReturningRadiationKernelCacheError, TypeError, ValueError) as error:
        raise ValueError(
            "reserved evaluator source must be a stable readable project file"
        ) from error
    if type(logical_path) is not str or not logical_path:
        raise ValueError("reserved evaluator source logical path is empty")
    normalized_code = _normalized_evaluator_code(code, logical_path)
    try:
        marshalled_code = marshal.dumps(normalized_code)
    except (TypeError, ValueError) as error:
        raise ValueError("reserved evaluator code is not marshalable") from error
    return {
        "annotations": annotation_document,
        "callableType": "types.FunctionType/exact",
        "closurePolicy": "no-free-variables-or-closure-cells",
        "code": {
            "byteLength": len(marshalled_code),
            "firstLineNumber": code.co_firstlineno,
            "marshalSha256": hashlib.sha256(marshalled_code).hexdigest(),
            "marshalVersion": marshal.version,
            "name": code.co_name,
            "qualifiedName": code.co_qualname,
        },
        "customFunctionAttributePolicy": "empty-only",
        "defaultsPolicy": "no-positional-or-keyword-defaults",
        "implementationId": implementation_id,
        "module": module,
        "name": name,
        "qualifiedName": qualname,
        "schema": EVALUATOR_DESCRIPTOR_SCHEMA,
        "source": {
            "byteLength": source_byte_length,
            "logicalPath": logical_path,
            "sha256": source_sha256,
        },
    }


def _evaluator_input(evaluator_id: str) -> InputArtifact:
    if type(evaluator_id) is not str or evaluator_id not in _EVALUATOR_IDS:
        raise ValueError("evaluator_id is not a fixed source-bound implementation")
    evaluator = (
        _evaluate_forward_direction
        if evaluator_id == _FORWARD_EVALUATOR_ID
        else _evaluate_forward_direction_native_cpu
        if evaluator_id == _NATIVE_FORWARD_EVALUATOR_ID
        else _evaluate_receiver_direction
        if evaluator_id == _RECEIVER_EVALUATOR_ID
        else _synthetic_forward_direction_evaluator
        if evaluator_id == _SYNTHETIC_FORWARD_EVALUATOR_ID
        else _synthetic_receiver_direction_evaluator
    )
    payload = canonical_json_bytes(
        _authenticated_evaluator_callable_descriptor(evaluator, evaluator_id)
    )
    return InputArtifact(
        f"{_EVALUATOR_ARTIFACT_URI_PREFIX}{evaluator_id}",
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def _evaluator_id_from_definition(
    definition: KerrKernelDirectionCacheDefinition,
) -> str:
    return _evaluator_id_from_context(definition.scientific_context)


def _declared_evaluator_id_from_context(
    context: KerrKernelScientificContext,
) -> str:
    """Read the narrow artifact declaration without replaying source I/O.

    This is used only inside a reserved evaluator after the jobs runner has
    authenticated the complete callable descriptor before cache access.  All
    definition, reduction, and persistence boundaries use the stronger
    ``_evaluator_id_from_context`` replay below.
    """

    if type(context) is not KerrKernelScientificContext:
        raise KerrReturningRadiationKernelCacheError(
            "evaluator context has a non-exact type"
        )
    inputs = context.inputs
    if type(inputs) is not tuple:
        raise KerrReturningRadiationKernelCacheError(
            "cache definition lacks its fixed evaluator identity artifact"
        )
    evaluator_artifacts = tuple(
        item
        for item in inputs
        if type(item) is InputArtifact
        and type(item.uri) is str
        and item.uri.startswith(_EVALUATOR_ARTIFACT_URI_PREFIX)
    )
    if len(evaluator_artifacts) != 1:
        raise KerrReturningRadiationKernelCacheError(
            "cache definition lacks one fixed evaluator identity artifact"
        )
    actual = evaluator_artifacts[0]
    if type(actual) is not InputArtifact:
        raise KerrReturningRadiationKernelCacheError(
            "cache evaluator identity artifact has a non-exact type"
        )
    if type(actual.uri) is not str or not actual.uri.startswith(
        _EVALUATOR_ARTIFACT_URI_PREFIX
    ):
        raise KerrReturningRadiationKernelCacheError(
            "cache evaluator identity artifact is unsupported or stale"
        )
    evaluator_id = actual.uri[len(_EVALUATOR_ARTIFACT_URI_PREFIX) :]
    if evaluator_id not in _EVALUATOR_IDS:
        raise KerrReturningRadiationKernelCacheError(
            "cache evaluator identity artifact is unsupported or stale"
        )
    if type(actual.byte_length) is not int or actual.byte_length <= 0:
        raise KerrReturningRadiationKernelCacheError(
            "cache evaluator identity artifact is unsupported or stale"
        )
    try:
        _lowercase_sha256(actual.sha256, "cache evaluator artifact sha256")
    except ValueError as error:
        raise KerrReturningRadiationKernelCacheError(
            "cache evaluator identity artifact is unsupported or stale"
        ) from error
    runtime_binding = context.evaluator_runtime_binding
    if evaluator_id == _NATIVE_FORWARD_EVALUATOR_ID:
        if (
            type(runtime_binding) is not KerrKernelEvaluatorRuntimeBinding
            or len(inputs) != 2
            or runtime_binding.descriptor_input not in inputs
            or runtime_binding.evaluator_implementation_id != evaluator_id
        ):
            raise KerrReturningRadiationKernelCacheError(
                "native cache evaluator lacks its exact runtime binding"
            )
    elif runtime_binding is not None or len(inputs) != 1:
        raise KerrReturningRadiationKernelCacheError(
            "non-native cache evaluator has unexpected runtime inputs"
        )
    return evaluator_id


def _evaluator_id_from_context(context: KerrKernelScientificContext) -> str:
    evaluator_id = _declared_evaluator_id_from_context(context)
    actual = next(
        item
        for item in context.inputs
        if item.uri.startswith(_EVALUATOR_ARTIFACT_URI_PREFIX)
    )
    expected = _evaluator_input(evaluator_id)
    if (
        actual.uri.encode("utf-8") == expected.uri.encode("utf-8")
        and actual.byte_length == expected.byte_length
        and actual.sha256.encode("ascii") == expected.sha256.encode("ascii")
    ):
        return evaluator_id
    raise KerrReturningRadiationKernelCacheError(
        "cache evaluator identity artifact is unsupported or stale"
    )


def _validate_definition_execution_identity(
    definition: KerrKernelDirectionCacheDefinition,
    closure: tuple[KerrKernelSourceClosureEntry, ...],
) -> None:
    if type(definition) is not KerrKernelDirectionCacheDefinition:
        raise KerrReturningRadiationKernelCacheError(
            "cache definition has a non-exact type"
        )
    expected_hashes = tuple(sorted(item.binding_sha256 for item in closure))
    actual_hashes = definition.scientific_context.identity.source_closure_sha256
    if type(actual_hashes) is not tuple or len(actual_hashes) != len(expected_hashes):
        raise KerrReturningRadiationKernelCacheError(
            "cache definition does not bind the fixed source closure"
        )
    for actual, expected in zip(actual_hashes, expected_hashes):
        if (
            type(actual) is not str
            or actual.encode("ascii") != expected.encode("ascii")
        ):
            raise KerrReturningRadiationKernelCacheError(
                "cache definition does not bind the fixed source closure"
            )
    evaluator_id = _evaluator_id_from_definition(definition)
    formulation = definition.plan.formulation
    allowed = (
        (
            _FORWARD_EVALUATOR_ID,
            _NATIVE_FORWARD_EVALUATOR_ID,
            _SYNTHETIC_FORWARD_EVALUATOR_ID,
        )
        if formulation == FORWARD
        else (_RECEIVER_EVALUATOR_ID, _SYNTHETIC_RECEIVER_EVALUATOR_ID)
    )
    if evaluator_id not in allowed:
        raise KerrReturningRadiationKernelCacheError(
            "cache evaluator identity disagrees with formulation"
        )


def _validated_context_and_coordinate(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate,
    formulation: str,
) -> tuple[KerrKernelScientificContext, Any]:
    if type(context) is not KerrKernelScientificContext:
        raise TypeError("evaluator context must have its exact type")
    context = KerrKernelScientificContext(
        context.plan,
        context.identity,
        context.inputs,
        context.scientific_job_key,
        context.evaluator_runtime_binding,
    )
    if context.identity.formulation != formulation:
        raise ValueError("evaluator context formulation is stale")
    if type(coordinate) is not KerrKernelDirectionCoordinate:
        raise TypeError("coordinate must have its exact type")
    if coordinate.pass_index >= len(context.plan.passes):
        raise ValueError("coordinate pass lies outside the scientific plan")
    grid_pass = context.plan.passes[coordinate.pass_index]
    if (
        coordinate.pass_name != grid_pass.name
        or coordinate.face_index >= 2
        or coordinate.face not in (UPPER, LOWER)
        or coordinate.face != (UPPER, LOWER)[coordinate.face_index]
        or coordinate.annulus_index >= context.plan.annulus_count
        or coordinate.rho_index >= grid_pass.rho_order
        or coordinate.mu_index >= grid_pass.mu_order
        or coordinate.psi_index >= grid_pass.psi_count
    ):
        raise ValueError("coordinate lies outside its canonical grid pass")
    pass_offset = 2 * context.plan.annulus_count * sum(
        item.rho_order * item.mu_order * item.psi_count
        for item in context.plan.passes[: coordinate.pass_index]
    )
    ordinal = pass_offset + (
        (
            (
                (
                    coordinate.face_index * context.plan.annulus_count
                    + coordinate.annulus_index
                )
                * grid_pass.rho_order
                + coordinate.rho_index
            )
            * grid_pass.mu_order
            + coordinate.mu_index
        )
        * grid_pass.psi_count
        + coordinate.psi_index
    )
    if coordinate.ordinal != ordinal:
        raise ValueError("coordinate ordinal is not canonical")
    return context, grid_pass


def _forward_coordinate_node(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate,
) -> _ForwardCoordinateNode:
    context, grid_pass = _validated_context_and_coordinate(
        context,
        coordinate,
        FORWARD,
    )
    identity = context.identity
    inner = identity.annulus_edges_over_mass[coordinate.annulus_index]
    outer = identity.annulus_edges_over_mass[coordinate.annulus_index + 1]
    rho = _cached_forward_rho_nodes(
        inner,
        outer,
        grid_pass.rho_order,
    )[coordinate.rho_index]
    directions = _cached_forward_direction_nodes(
        grid_pass.mu_order,
        grid_pass.psi_count,
        grid_pass.phase_cells,
    )
    node = directions[
        coordinate.mu_index * grid_pass.psi_count + coordinate.psi_index
    ]
    return _ForwardCoordinateNode(
        coordinate.pass_index,
        coordinate.pass_name,
        coordinate.face,
        coordinate.annulus_index,
        coordinate.rho_index,
        coordinate.mu_index,
        coordinate.psi_index,
        rho,
        node.emission_angle_cosine,
        node.tangent_azimuth_rad,
        node.normalized_emitted_flux_weight,
    )


def _receiver_coordinate_node(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate,
) -> _ReceiverCoordinateNode:
    context, grid_pass = _validated_context_and_coordinate(
        context,
        coordinate,
        RECEIVER,
    )
    identity = context.identity
    inner = identity.annulus_edges_over_mass[coordinate.annulus_index]
    outer = identity.annulus_edges_over_mass[coordinate.annulus_index + 1]
    rho = _cached_receiver_rho_nodes(
        inner,
        outer,
        grid_pass.rho_order,
    )[coordinate.rho_index]
    node = _cached_receiver_direction_nodes(
        grid_pass.mu_order,
        grid_pass.psi_count,
        grid_pass.phase_cells,
    )[coordinate.mu_index * grid_pass.psi_count + coordinate.psi_index]
    return _ReceiverCoordinateNode(
        coordinate.pass_index,
        coordinate.pass_name,
        coordinate.face,
        coordinate.annulus_index,
        coordinate.rho_index,
        coordinate.mu_index,
        coordinate.psi_index,
        rho,
        node[0],
        node[1],
        node[2],
        grid_pass.phase_cells,
    )


@lru_cache(maxsize=512)
def _cached_forward_rho_nodes(
    inner: float,
    outer: float,
    order: int,
) -> tuple[float, ...]:
    return _forward._rho_coordinate_nodes(inner, outer, order)


@lru_cache(maxsize=32)
def _cached_forward_direction_nodes(
    mu_order: int,
    psi_count: int,
    phase_cells: float,
) -> tuple[Any, ...]:
    return _forward.kerrbb_d20_emitted_flux_direction_nodes(
        mu_order,
        psi_count,
        phase_cells=phase_cells,
    )


@lru_cache(maxsize=512)
def _cached_receiver_rho_nodes(
    inner: float,
    outer: float,
    order: int,
) -> tuple[float, ...]:
    return _receiver._rho_coordinate_nodes(inner, outer, order)


@lru_cache(maxsize=32)
def _cached_receiver_direction_nodes(
    mu_order: int,
    psi_count: int,
    phase_cells: float,
) -> tuple[tuple[float, float, float], ...]:
    return _receiver._receiver_sky_direction_nodes(
        mu_order,
        psi_count,
        phase_cells,
    )


def _forward_sample_document(node: _ForwardCoordinateNode) -> dict[str, Any]:
    return {
        "emissionAngleCosine": node.emission_angle_cosine,
        "muIndex": node.mu_index,
        "normalizedEmittedFluxWeight": node.normalized_emitted_flux_weight,
        "passIndex": node.pass_index,
        "passName": node.pass_name,
        "psiIndex": node.psi_index,
        "rhoIndex": node.rho_index,
        "sourceAnnulusIndex": node.source_annulus_index,
        "sourceFace": node.source_face,
        "sourceRadiusOverMass": node.source_radius_over_mass,
        "tangentAzimuthRad": node.tangent_azimuth_rad,
    }


def _receiver_sample_document(node: _ReceiverCoordinateNode) -> dict[str, Any]:
    return {
        "incidenceCosine": node.incidence_cosine,
        "muIndex": node.mu_index,
        "muWeight": node.mu_weight,
        "passIndex": node.pass_index,
        "passName": node.pass_name,
        "phaseCells": node.phase_cells,
        "psiIndex": node.psi_index,
        "receiverAnnulusIndex": node.receiver_annulus_index,
        "receiverFace": node.receiver_face,
        "receiverRadiusOverMass": node.receiver_radius_over_mass,
        "rhoIndex": node.rho_index,
        "tangentAzimuthRad": node.tangent_azimuth_rad,
    }


def _forward_transport_document(
    node: _ForwardCoordinateNode,
    transport: _forward._DirectionTransport,
    edges: tuple[float, ...],
) -> dict[str, Any]:
    if type(transport) is not _forward._DirectionTransport:
        raise TypeError("forward evaluator transport has a non-exact type")
    if transport.fate.startswith("return-"):
        receiver_bin = _forward._receiver_bin_index(
            transport.receiver_radius_over_mass,
            edges,
        )
        coarse_bin = _forward._receiver_bin_index(
            transport.coarse_receiver_radius_over_mass,
            edges,
        )
    else:
        receiver_bin = None
        coarse_bin = None
    return {
        "formulation": FORWARD,
        "sample": _forward_sample_document(node),
        "schema": FORWARD_TRANSPORT_SCHEMA,
        "transport": {
            "coarseReceiverAnnulusIndex": coarse_bin,
            "coarseReceiverFace": transport.coarse_receiver_face,
            "coarseReceiverRadiusOverMass": (
                transport.coarse_receiver_radius_over_mass
            ),
            "fate": transport.fate,
            "frequencyRatio": transport.frequency_ratio,
            "g2": transport.g2,
            "primitiveDescriptorSha256": transport.primitive_descriptor_sha256,
            "receiverAnnulusIndex": receiver_bin,
            "receiverFace": transport.receiver_face,
            "receiverRadiusOverMass": transport.receiver_radius_over_mass,
        },
    }


def _receiver_transport_document(
    node: _ReceiverCoordinateNode,
    transport: _receiver._ReceiverDirectionTransport,
    edges: tuple[float, ...],
) -> dict[str, Any]:
    if type(transport) is not _receiver._ReceiverDirectionTransport:
        raise TypeError("receiver evaluator transport has a non-exact type")
    if transport.outcome.startswith("source-"):
        source_bin = _receiver._bin_index(
            transport.source_radius_over_mass,
            edges,
            "fine source radius",
        )
        coarse_bin = _receiver._bin_index(
            transport.coarse_source_radius_over_mass,
            edges,
            "coarse source radius",
        )
    else:
        source_bin = None
        coarse_bin = None
    return {
        "formulation": RECEIVER,
        "sample": _receiver_sample_document(node),
        "schema": RECEIVER_TRANSPORT_SCHEMA,
        "transport": {
            "coarseSourceAnnulusIndex": coarse_bin,
            "coarseSourceRadiusOverMass": transport.coarse_source_radius_over_mass,
            "outcome": transport.outcome,
            "primitiveDescriptorSha256": transport.primitive_descriptor_sha256,
            "receiverIntegrand": transport.receiver_integrand,
            "sourceAnnulusIndex": source_bin,
            "sourceFace": transport.source_face,
            "sourceRadiusOverMass": transport.source_radius_over_mass,
        },
    }


def _evaluate_forward_direction(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate,
) -> dict[str, Any]:
    if _declared_evaluator_id_from_context(context) != _FORWARD_EVALUATOR_ID:
        raise KerrReturningRadiationKernelCacheError(
            "forward production evaluator identity is not bound by context"
        )
    node = _forward_coordinate_node(context, coordinate)
    identity = context.identity
    transport = _forward._trace_direction(
        identity.surface,
        identity.termination,
        identity.fine_ray_options,
        identity.fine_surface_options,
        identity.coarse_ray_options,
        identity.coarse_surface_options,
        node.source_face,
        node.source_radius_over_mass,
        node.emission_angle_cosine,
        node.tangent_azimuth_rad,
    )
    return _forward_transport_document(
        node,
        transport,
        identity.annulus_edges_over_mass,
    )


def _evaluate_forward_direction_native_cpu(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate,
) -> dict[str, Any]:
    """Evaluate one forward coordinate through its explicit runtime binding."""

    native_wrapper = importlib.import_module(
        "offline.kerr_returning_radiation_native_cpu"
    )
    consume_forward_direction_native_cpu_runtime_binding = getattr(
        native_wrapper,
        "consume_forward_direction_native_cpu_runtime_binding",
        None,
    )
    if not callable(consume_forward_direction_native_cpu_runtime_binding):
        raise KerrReturningRadiationKernelCacheError(
            "native wrapper lacks its runtime-binding consumer"
        )

    if (
        _declared_evaluator_id_from_context(context)
        != _NATIVE_FORWARD_EVALUATOR_ID
    ):
        raise KerrReturningRadiationKernelCacheError(
            "native forward evaluator identity is not bound by context"
        )
    node = _forward_coordinate_node(context, coordinate)
    identity = context.identity
    runtime_binding = context.evaluator_runtime_binding
    if type(runtime_binding) is not KerrKernelEvaluatorRuntimeBinding:
        raise KerrReturningRadiationKernelCacheError(
            "native forward evaluator lacks its exact runtime binding"
        )
    transport = consume_forward_direction_native_cpu_runtime_binding(
        runtime_binding,
        identity.surface,
        identity.termination,
        identity.fine_ray_options,
        identity.fine_surface_options,
        identity.coarse_ray_options,
        identity.coarse_surface_options,
        node.source_face,
        node.source_radius_over_mass,
        node.emission_angle_cosine,
        node.tangent_azimuth_rad,
    )
    return _forward_transport_document(
        node,
        transport,
        identity.annulus_edges_over_mass,
    )


def _evaluate_receiver_direction(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate,
) -> dict[str, Any]:
    if _declared_evaluator_id_from_context(context) != _RECEIVER_EVALUATOR_ID:
        raise KerrReturningRadiationKernelCacheError(
            "receiver production evaluator identity is not bound by context"
        )
    node = _receiver_coordinate_node(context, coordinate)
    identity = context.identity
    transport = _receiver._trace_direction(
        identity.surface,
        identity.termination,
        identity.fine_ray_options,
        identity.fine_surface_options,
        identity.coarse_ray_options,
        identity.coarse_surface_options,
        node.receiver_face,
        node.receiver_radius_over_mass,
        node.incidence_cosine,
        node.tangent_azimuth_rad,
    )
    return _receiver_transport_document(
        node,
        transport,
        identity.annulus_edges_over_mass,
    )


def _synthetic_forward_direction_evaluator(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate,
) -> dict[str, Any]:
    """Fixed source-bound evaluator used only by fast bridge tests."""

    if (
        _declared_evaluator_id_from_context(context)
        != _SYNTHETIC_FORWARD_EVALUATOR_ID
    ):
        raise KerrReturningRadiationKernelCacheError(
            "synthetic forward evaluator identity is not bound by context"
        )
    node = _forward_coordinate_node(context, coordinate)
    transport = _synthetic_forward_transport(node)
    return _forward_transport_document(
        node,
        transport,
        context.identity.annulus_edges_over_mass,
    )


def _synthetic_receiver_direction_evaluator(
    context: KerrKernelScientificContext,
    coordinate: KerrKernelDirectionCoordinate,
) -> dict[str, Any]:
    """Fixed source-bound evaluator used only by fast bridge tests."""

    if (
        _declared_evaluator_id_from_context(context)
        != _SYNTHETIC_RECEIVER_EVALUATOR_ID
    ):
        raise KerrReturningRadiationKernelCacheError(
            "synthetic receiver evaluator identity is not bound by context"
        )
    node = _receiver_coordinate_node(context, coordinate)
    transport = _synthetic_receiver_transport(node)
    return _receiver_transport_document(
        node,
        transport,
        context.identity.annulus_edges_over_mass,
    )


def _synthetic_forward_transport(
    node: _ForwardCoordinateNode,
) -> _forward._DirectionTransport:
    receiver_face = UPPER if node.tangent_azimuth_rad < math.pi else LOWER
    ratio = 2.0
    identity = hashlib.sha256(repr(asdict(node)).encode("utf-8")).hexdigest()
    return _forward._DirectionTransport(
        "return-upper" if receiver_face == UPPER else "return-lower",
        receiver_face,
        node.source_radius_over_mass,
        ratio,
        ratio * ratio,
        identity,
        receiver_face,
        node.source_radius_over_mass,
    )


def _synthetic_receiver_transport(
    node: _ReceiverCoordinateNode,
) -> _receiver._ReceiverDirectionTransport:
    source_face = UPPER if node.tangent_azimuth_rad < math.pi else LOWER
    identity = hashlib.sha256(repr(asdict(node)).encode("utf-8")).hexdigest()
    return _receiver._ReceiverDirectionTransport(
        "source-upper" if source_face == UPPER else "source-lower",
        source_face,
        node.receiver_radius_over_mass,
        node.receiver_radius_over_mass,
        2.0,
        identity,
    )


def _synthetic_forward_transport_for_sample(
    sample: _forward._ForwardDirectionSample,
) -> _forward._DirectionTransport:
    return _synthetic_forward_transport(
        _ForwardCoordinateNode(
            sample.pass_index,
            sample.pass_name,
            sample.source_face,
            sample.source_annulus_index,
            sample.rho_index,
            sample.mu_index,
            sample.psi_index,
            sample.source_radius_over_mass,
            sample.emission_angle_cosine,
            sample.tangent_azimuth_rad,
            sample.normalized_emitted_flux_weight,
        )
    )


def _synthetic_receiver_transport_for_sample(
    sample: _receiver._ReceiverDirectionSample,
) -> _receiver._ReceiverDirectionTransport:
    return _synthetic_receiver_transport(
        _ReceiverCoordinateNode(
            sample.pass_index,
            sample.pass_name,
            sample.receiver_face,
            sample.receiver_annulus_index,
            sample.rho_index,
            sample.mu_index,
            sample.psi_index,
            sample.receiver_radius_over_mass,
            sample.incidence_cosine,
            sample.mu_weight,
            sample.tangent_azimuth_rad,
            sample.phase_cells,
        )
    )


def _require_exact_json(actual: Any, expected: Any, path: str) -> None:
    if type(actual) is not type(expected):
        raise KerrReturningRadiationKernelCacheError(
            f"{path} has non-exact type {type(actual).__name__}"
        )
    if type(expected) is dict:
        if set(actual) != set(expected):
            raise KerrReturningRadiationKernelCacheError(
                f"{path} has an unknown or missing key"
            )
        for key in sorted(expected):
            _require_exact_json(actual[key], expected[key], f"{path}.{key}")
        return
    if type(expected) is float:
        differs = not math.isfinite(actual) or actual.hex() != expected.hex()
    elif type(expected) in (str, int):
        differs = actual != expected
    elif expected is None:
        differs = False
    else:
        raise KerrReturningRadiationKernelCacheError(
            f"{path} has unsupported exact JSON type {type(expected).__name__}"
        )
    if differs:
        raise KerrReturningRadiationKernelCacheError(
            f"{path} differs from the canonical quadrature evidence"
        )


def _forward_transport_from_record(
    record: KerrCachedKernelDirectionRecord,
    sample: _forward._ForwardDirectionSample,
    edges: tuple[float, ...],
) -> _forward._DirectionTransport:
    node = _ForwardCoordinateNode(
        sample.pass_index,
        sample.pass_name,
        sample.source_face,
        sample.source_annulus_index,
        sample.rho_index,
        sample.mu_index,
        sample.psi_index,
        sample.source_radius_over_mass,
        sample.emission_angle_cosine,
        sample.tangent_azimuth_rad,
        sample.normalized_emitted_flux_weight,
    )
    document = dict(record.transport())
    if type(document) is not dict or set(document) != {
        "formulation",
        "sample",
        "schema",
        "transport",
    }:
        raise KerrReturningRadiationKernelCacheError(
            "forward cached transport has a non-exact schema"
        )
    if document["formulation"] != FORWARD or document["schema"] != FORWARD_TRANSPORT_SCHEMA:
        raise KerrReturningRadiationKernelCacheError(
            "forward cached transport identity is stale"
        )
    _require_exact_json(
        document["sample"],
        _forward_sample_document(node),
        "$transport.sample",
    )
    raw = document["transport"]
    expected_keys = {
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
    if type(raw) is not dict or set(raw) != expected_keys:
        raise KerrReturningRadiationKernelCacheError(
            "forward narrow transport has an unknown or missing key"
        )
    try:
        transport = _forward._DirectionTransport(
            raw["fate"],
            raw["receiverFace"],
            raw["receiverRadiusOverMass"],
            raw["frequencyRatio"],
            raw["g2"],
            raw["primitiveDescriptorSha256"],
            raw["coarseReceiverFace"],
            raw["coarseReceiverRadiusOverMass"],
        )
    except (TypeError, ValueError) as error:
        raise KerrReturningRadiationKernelCacheError(
            "forward narrow transport is invalid"
        ) from error
    if transport.fate.startswith("return-"):
        receiver_bin = _forward._receiver_bin_index(
            transport.receiver_radius_over_mass,
            edges,
        )
        coarse_bin = _forward._receiver_bin_index(
            transport.coarse_receiver_radius_over_mass,
            edges,
        )
    else:
        receiver_bin = None
        coarse_bin = None
    _require_exact_json(
        raw["receiverAnnulusIndex"],
        receiver_bin,
        "$transport.transport.receiverAnnulusIndex",
    )
    _require_exact_json(
        raw["coarseReceiverAnnulusIndex"],
        coarse_bin,
        "$transport.transport.coarseReceiverAnnulusIndex",
    )
    return transport


def _receiver_transport_from_record(
    record: KerrCachedKernelDirectionRecord,
    sample: _receiver._ReceiverDirectionSample,
    edges: tuple[float, ...],
) -> _receiver._ReceiverDirectionTransport:
    node = _ReceiverCoordinateNode(
        sample.pass_index,
        sample.pass_name,
        sample.receiver_face,
        sample.receiver_annulus_index,
        sample.rho_index,
        sample.mu_index,
        sample.psi_index,
        sample.receiver_radius_over_mass,
        sample.incidence_cosine,
        sample.mu_weight,
        sample.tangent_azimuth_rad,
        sample.phase_cells,
    )
    document = dict(record.transport())
    if type(document) is not dict or set(document) != {
        "formulation",
        "sample",
        "schema",
        "transport",
    }:
        raise KerrReturningRadiationKernelCacheError(
            "receiver cached transport has a non-exact schema"
        )
    if (
        document["formulation"] != RECEIVER
        or document["schema"] != RECEIVER_TRANSPORT_SCHEMA
    ):
        raise KerrReturningRadiationKernelCacheError(
            "receiver cached transport identity is stale"
        )
    _require_exact_json(
        document["sample"],
        _receiver_sample_document(node),
        "$transport.sample",
    )
    raw = document["transport"]
    expected_keys = {
        "coarseSourceAnnulusIndex",
        "coarseSourceRadiusOverMass",
        "outcome",
        "primitiveDescriptorSha256",
        "receiverIntegrand",
        "sourceAnnulusIndex",
        "sourceFace",
        "sourceRadiusOverMass",
    }
    if type(raw) is not dict or set(raw) != expected_keys:
        raise KerrReturningRadiationKernelCacheError(
            "receiver narrow transport has an unknown or missing key"
        )
    try:
        transport = _receiver._ReceiverDirectionTransport(
            raw["outcome"],
            raw["sourceFace"],
            raw["sourceRadiusOverMass"],
            raw["coarseSourceRadiusOverMass"],
            raw["receiverIntegrand"],
            raw["primitiveDescriptorSha256"],
        )
    except (TypeError, ValueError) as error:
        raise KerrReturningRadiationKernelCacheError(
            "receiver narrow transport is invalid"
        ) from error
    if transport.outcome.startswith("source-"):
        source_bin = _receiver._bin_index(
            transport.source_radius_over_mass,
            edges,
            "fine source radius",
        )
        coarse_bin = _receiver._bin_index(
            transport.coarse_source_radius_over_mass,
            edges,
            "coarse source radius",
        )
    else:
        source_bin = None
        coarse_bin = None
    _require_exact_json(
        raw["sourceAnnulusIndex"],
        source_bin,
        "$transport.transport.sourceAnnulusIndex",
    )
    _require_exact_json(
        raw["coarseSourceAnnulusIndex"],
        coarse_bin,
        "$transport.transport.coarseSourceAnnulusIndex",
    )
    return transport


class _ForwardCachedTransportStream:
    def __init__(
        self,
        records: Iterator[KerrCachedKernelDirectionRecord],
        edges: tuple[float, ...],
    ) -> None:
        self._records = records
        self._edges = edges
        self._ordinal = 0

    def __call__(
        self,
        sample: _forward._ForwardDirectionSample,
    ) -> _forward._DirectionTransport:
        try:
            record = next(self._records)
        except StopIteration as error:
            raise KerrReturningRadiationKernelCacheError(
                "forward cached transport stream ended early"
            ) from error
        coordinate = record.coordinate
        expected = (
            self._ordinal,
            sample.pass_index,
            sample.pass_name,
            sample.source_face,
            sample.source_annulus_index,
            sample.rho_index,
            sample.mu_index,
            sample.psi_index,
        )
        actual = (
            coordinate.ordinal,
            coordinate.pass_index,
            coordinate.pass_name,
            coordinate.face,
            coordinate.annulus_index,
            coordinate.rho_index,
            coordinate.mu_index,
            coordinate.psi_index,
        )
        if actual != expected:
            raise KerrReturningRadiationKernelCacheError(
                "forward cached coordinate differs from reducer order"
            )
        self._ordinal += 1
        return _forward_transport_from_record(record, sample, self._edges)

    def finish(self) -> None:
        try:
            next(self._records)
        except StopIteration:
            return
        raise KerrReturningRadiationKernelCacheError(
            "forward cached transport stream has trailing records"
        )


class _ReceiverCachedTransportStream:
    def __init__(
        self,
        records: Iterator[KerrCachedKernelDirectionRecord],
        edges: tuple[float, ...],
    ) -> None:
        self._records = records
        self._edges = edges
        self._ordinal = 0

    def __call__(
        self,
        sample: _receiver._ReceiverDirectionSample,
    ) -> _receiver._ReceiverDirectionTransport:
        try:
            record = next(self._records)
        except StopIteration as error:
            raise KerrReturningRadiationKernelCacheError(
                "receiver cached transport stream ended early"
            ) from error
        coordinate = record.coordinate
        expected = (
            self._ordinal,
            sample.pass_index,
            sample.pass_name,
            sample.receiver_face,
            sample.receiver_annulus_index,
            sample.rho_index,
            sample.mu_index,
            sample.psi_index,
        )
        actual = (
            coordinate.ordinal,
            coordinate.pass_index,
            coordinate.pass_name,
            coordinate.face,
            coordinate.annulus_index,
            coordinate.rho_index,
            coordinate.mu_index,
            coordinate.psi_index,
        )
        if actual != expected:
            raise KerrReturningRadiationKernelCacheError(
                "receiver cached coordinate differs from reducer order"
            )
        self._ordinal += 1
        return _receiver_transport_from_record(record, sample, self._edges)

    def finish(self) -> None:
        try:
            next(self._records)
        except StopIteration:
            return
        raise KerrReturningRadiationKernelCacheError(
            "receiver cached transport stream has trailing records"
        )


def _build_cache_definition(
    formulation: str,
    *,
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    annulus_edges_over_mass: tuple[float, ...],
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
    coarse_ray_options: RayTraceOptions | None,
    coarse_surface_options: SurfaceEventOptions | None,
    policy: KerrReturningRadiationKernelPolicy,
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy,
    directions_per_task: int,
    evaluator_id: str,
    evaluator_runtime_binding: KerrKernelEvaluatorRuntimeBinding | None = None,
    native_backend_descriptor: dict[str, Any] | None = None,
) -> tuple[
    KerrKernelDirectionCacheDefinition,
    tuple[KerrKernelSourceClosureEntry, ...],
    KerrKernelReductionConfiguration,
]:
    allowed_evaluators = (
        (
            _FORWARD_EVALUATOR_ID,
            _NATIVE_FORWARD_EVALUATOR_ID,
            _SYNTHETIC_FORWARD_EVALUATOR_ID,
        )
        if formulation == FORWARD
        else (_RECEIVER_EVALUATOR_ID, _SYNTHETIC_RECEIVER_EVALUATOR_ID)
    )
    if type(evaluator_id) is not str or evaluator_id not in allowed_evaluators:
        raise ValueError("evaluator identity disagrees with kernel formulation")
    if evaluator_id == _NATIVE_FORWARD_EVALUATOR_ID:
        if (
            formulation != FORWARD
            or type(evaluator_runtime_binding)
            is not KerrKernelEvaluatorRuntimeBinding
            or type(native_backend_descriptor) is not dict
        ):
            raise ValueError(
                "native forward evaluator requires one exact runtime binding"
            )
    elif evaluator_runtime_binding is not None or native_backend_descriptor is not None:
        raise ValueError("non-native evaluator cannot carry a native runtime binding")
    reduction_configuration = KerrKernelReductionConfiguration(policy, area_policy)
    if type(annulus_edges_over_mass) is not tuple or len(
        annulus_edges_over_mass
    ) < 2:
        raise TypeError("annulus edges must be an exact tuple with at least two values")
    plan = KerrKernelDirectionTaskPlan(
        formulation,
        len(annulus_edges_over_mass) - 1,
        reduction_configuration.policy.rho_order,
        reduction_configuration.policy.mu_order,
        reduction_configuration.policy.psi_count,
        directions_per_task,
    )
    _require_reduction_configuration_covers_plan(
        reduction_configuration,
        plan,
    )
    # Policy/area exactness and both work budgets are closed before source or
    # cache filesystem access.  Only then authenticate the transport closure.
    if evaluator_id == _NATIVE_FORWARD_EVALUATOR_ID:
        assert native_backend_descriptor is not None
        closure = _source_closure_manifest(
            numeric_backend_override=_native_numeric_backend_descriptor(
                native_backend_descriptor
            ),
            source_paths=_NATIVE_FORWARD_SOURCE_CLOSURE_PATHS,
            module_owners=_NATIVE_FORWARD_SOURCE_CLOSURE_MODULE_OWNERS,
        )
        assert evaluator_runtime_binding is not None
        inputs = (
            _evaluator_input(evaluator_id),
            evaluator_runtime_binding.descriptor_input,
        )
    else:
        closure = _source_closure_manifest()
        inputs = (_evaluator_input(evaluator_id),)
    builder = (
        build_forward_kernel_scientific_identity
        if formulation == FORWARD
        else build_receiver_kernel_scientific_identity
    )
    identity = builder(
        surface=surface,
        termination=termination,
        annulus_edges_over_mass=annulus_edges_over_mass,
        fine_ray_options=ray_options,
        fine_surface_options=surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
        source_closure_sha256=tuple(item.binding_sha256 for item in closure),
    )
    return (
        make_kernel_direction_cache_definition(
            plan,
            identity=identity,
            inputs=inputs,
            evaluator_runtime_binding=evaluator_runtime_binding,
        ),
        closure,
        reduction_configuration,
    )


def _native_backend_descriptor_from_binding(
    binding: KerrKernelEvaluatorRuntimeBinding,
) -> dict[str, Any]:
    if type(binding) is not KerrKernelEvaluatorRuntimeBinding:
        raise TypeError("native evaluator runtime binding must have its exact type")
    try:
        document = json.loads(binding.descriptor_json)
        backend = document["backendDescriptor"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise KerrReturningRadiationKernelCacheError(
            "native evaluator runtime descriptor is malformed"
        ) from error
    if type(document) is not dict or type(backend) is not dict:
        raise TypeError("native evaluator backend descriptor must be an exact dict")
    if hashlib.sha256(canonical_json_bytes(backend)).hexdigest() != (
        binding.backend_descriptor_sha256
    ):
        raise KerrReturningRadiationKernelCacheError(
            "native evaluator backend descriptor SHA-256 differs"
        )
    return backend


def _source_closure_for_definition(
    definition: KerrKernelDirectionCacheDefinition,
) -> tuple[KerrKernelSourceClosureEntry, ...]:
    if type(definition) is not KerrKernelDirectionCacheDefinition:
        raise TypeError("cache definition must have its exact type")
    evaluator_id = _evaluator_id_from_definition(definition)
    if evaluator_id == _NATIVE_FORWARD_EVALUATOR_ID:
        binding = definition.scientific_context.evaluator_runtime_binding
        if type(binding) is not KerrKernelEvaluatorRuntimeBinding:
            raise KerrReturningRadiationKernelCacheError(
                "native definition lacks its exact runtime binding"
            )
        return _source_closure_manifest(
            numeric_backend_override=_native_numeric_backend_descriptor(
                _native_backend_descriptor_from_binding(binding)
            ),
            source_paths=_NATIVE_FORWARD_SOURCE_CLOSURE_PATHS,
            module_owners=_NATIVE_FORWARD_SOURCE_CLOSURE_MODULE_OWNERS,
        )
    return _source_closure_manifest()


def build_forward_kerr_returning_radiation_kernel_cache_definition(
    surface: KerrFiniteThicknessMultiSurface,
    *,
    termination: KerrOblateTermination,
    annulus_edges_over_mass: tuple[float, ...],
    ray_options: RayTraceOptions = RayTraceOptions(),
    surface_options: SurfaceEventOptions = SurfaceEventOptions(
        subdivisions_per_segment=4
    ),
    coarse_ray_options: RayTraceOptions | None = None,
    coarse_surface_options: SurfaceEventOptions | None = None,
    policy: KerrReturningRadiationKernelPolicy | None = None,
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy | None = None,
    directions_per_task: int = 64,
) -> KerrKernelDirectionCacheDefinition:
    """Build the fixed production forward cache identity without tracing."""

    selected_policy = KerrReturningRadiationKernelPolicy() if policy is None else policy
    selected_area = (
        KerrFiniteThicknessAreaQuadraturePolicy()
        if area_policy is None
        else area_policy
    )
    definition, _closure, _reduction_configuration = _build_cache_definition(
        FORWARD,
        surface=surface,
        termination=termination,
        annulus_edges_over_mass=annulus_edges_over_mass,
        ray_options=ray_options,
        surface_options=surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
        policy=selected_policy,
        area_policy=selected_area,
        directions_per_task=directions_per_task,
        evaluator_id=_FORWARD_EVALUATOR_ID,
    )
    return definition


def build_native_forward_kerr_returning_radiation_kernel_cache_definition(
    surface: KerrFiniteThicknessMultiSurface,
    *,
    native_library_path: Path,
    termination: KerrOblateTermination,
    annulus_edges_over_mass: tuple[float, ...],
    ray_options: RayTraceOptions = RayTraceOptions(),
    surface_options: SurfaceEventOptions = SurfaceEventOptions(
        subdivisions_per_segment=4
    ),
    coarse_ray_options: RayTraceOptions | None = None,
    coarse_surface_options: SurfaceEventOptions | None = None,
    policy: KerrReturningRadiationKernelPolicy | None = None,
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy | None = None,
    directions_per_task: int = 64,
    native_segment_capacity: int = 100_000,
    native_crossing_capacity: int = 100_000,
) -> KerrKernelDirectionCacheDefinition:
    """Build a fresh native-forward identity without tracing or cache access."""

    selected_policy = KerrReturningRadiationKernelPolicy() if policy is None else policy
    selected_area = (
        KerrFiniteThicknessAreaQuadraturePolicy()
        if area_policy is None
        else area_policy
    )
    runtime_binding, backend_descriptor = _native_forward_runtime_binding(
        native_library_path,
        segment_capacity=native_segment_capacity,
        crossing_capacity=native_crossing_capacity,
    )
    definition, _closure, _reduction_configuration = _build_cache_definition(
        FORWARD,
        surface=surface,
        termination=termination,
        annulus_edges_over_mass=annulus_edges_over_mass,
        ray_options=ray_options,
        surface_options=surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
        policy=selected_policy,
        area_policy=selected_area,
        directions_per_task=directions_per_task,
        evaluator_id=_NATIVE_FORWARD_EVALUATOR_ID,
        evaluator_runtime_binding=runtime_binding,
        native_backend_descriptor=backend_descriptor,
    )
    return definition


def build_receiver_kerr_returning_radiation_kernel_cache_definition(
    surface: KerrFiniteThicknessMultiSurface,
    *,
    termination: KerrOblateTermination,
    annulus_edges_over_mass: tuple[float, ...],
    ray_options: RayTraceOptions = RayTraceOptions(),
    surface_options: SurfaceEventOptions = SurfaceEventOptions(
        subdivisions_per_segment=4
    ),
    coarse_ray_options: RayTraceOptions | None = None,
    coarse_surface_options: SurfaceEventOptions | None = None,
    policy: KerrReturningRadiationKernelPolicy | None = None,
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy | None = None,
    directions_per_task: int = 64,
) -> KerrKernelDirectionCacheDefinition:
    """Build the fixed production receiver cache identity without tracing."""

    selected_policy = KerrReturningRadiationKernelPolicy() if policy is None else policy
    selected_area = (
        KerrFiniteThicknessAreaQuadraturePolicy()
        if area_policy is None
        else area_policy
    )
    definition, _closure, _reduction_configuration = _build_cache_definition(
        RECEIVER,
        surface=surface,
        termination=termination,
        annulus_edges_over_mass=annulus_edges_over_mass,
        ray_options=ray_options,
        surface_options=surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
        policy=selected_policy,
        area_policy=selected_area,
        directions_per_task=directions_per_task,
        evaluator_id=_RECEIVER_EVALUATOR_ID,
    )
    return definition


def _execution_audit(
    definition: KerrKernelDirectionCacheDefinition,
    job_run: JobRun,
    closure: tuple[KerrKernelSourceClosureEntry, ...],
) -> KerrCachedKernelExecutionAudit:
    if type(job_run.reused_tasks) is not int or type(job_run.executed_tasks) is not int:
        raise KerrReturningRadiationKernelCacheError(
            "JobRun task counters have non-exact types"
        )
    if (
        type(job_run.max_in_flight_observed) is not int
        or job_run.max_in_flight_observed < 0
    ):
        raise KerrReturningRadiationKernelCacheError(
            "JobRun max-in-flight evidence has a non-exact type or value"
        )
    if any(type(item.reused) is not bool for item in job_run.results):
        raise KerrReturningRadiationKernelCacheError(
            "TaskResult reuse evidence has a non-exact type"
        )
    exact_executed_tasks = sum(item.reused is False for item in job_run.results)
    exact_reused_tasks = sum(item.reused is True for item in job_run.results)
    if (
        job_run.executed_tasks != exact_executed_tasks
        or job_run.reused_tasks != exact_reused_tasks
    ):
        raise KerrReturningRadiationKernelCacheError(
            "JobRun task counters disagree with exact result evidence"
        )
    executed_directions = sum(
        item.key.width for item in job_run.results if item.reused is False
    )
    reused_directions = sum(
        item.key.width for item in job_run.results if item.reused is True
    )
    formulation = definition.plan.formulation
    evaluator_id = _evaluator_id_from_definition(definition)
    return KerrCachedKernelExecutionAudit(
        formulation,
        evaluator_id,
        definition.scientific_job_key,
        definition.job_spec.job_key,
        definition.plan.direction_count,
        executed_directions,
        reused_directions,
        exact_executed_tasks,
        exact_reused_tasks,
        _source_closure_manifest_sha256(closure),
        True,
        True,
        False,
        False,
    )


def _validate_job_run_evidence(
    definition: KerrKernelDirectionCacheDefinition,
    job_run: JobRun,
) -> None:
    """Bind a possibly forged JobRun before the jobs-layer reader sees it."""

    if type(job_run) is not JobRun:
        raise KerrReturningRadiationKernelCacheError(
            "JobRun must have its exact type"
        )
    job_key = _trusted_attribute(job_run, "job_key", "job_run")
    if (
        type(job_key) is not str
        or job_key.encode("ascii")
        != definition.job_spec.job_key.encode("ascii")
    ):
        raise KerrReturningRadiationKernelCacheError(
            "JobRun belongs to a different cache definition"
        )
    results = _trusted_attribute(job_run, "results", "job_run")
    expected_tasks = tuple(definition.job_spec.tasks)
    if type(results) is not tuple or len(results) != len(expected_tasks):
        raise KerrReturningRadiationKernelCacheError(
            "JobRun results are not an exact complete tuple"
        )
    path_type = type(Path())
    for index, (result, expected_key) in enumerate(zip(results, expected_tasks)):
        if type(result) is not TaskResult:
            raise KerrReturningRadiationKernelCacheError(
                "JobRun contains a non-exact TaskResult"
            )
        key = _trusted_attribute(result, "key", f"job_run.results[{index}]")
        if type(key) is not TaskKey:
            raise KerrReturningRadiationKernelCacheError(
                "TaskResult key has a non-exact type"
            )
        try:
            _require_trusted_exact_tree(
                key,
                expected_key,
                f"job_run.results[{index}].key",
            )
        except KerrReturningRadiationKernelCacheVerificationError as error:
            raise KerrReturningRadiationKernelCacheError(
                "TaskResult key differs from canonical task order"
            ) from error
        for name in ("payload_path", "receipt_path"):
            path_value = _trusted_attribute(
                result,
                name,
                f"job_run.results[{index}]",
            )
            if type(path_value) is not path_type:
                raise KerrReturningRadiationKernelCacheError(
                    f"TaskResult {name} has a non-exact type"
                )
        for name in ("record_count", "byte_length"):
            value = _trusted_attribute(result, name, f"job_run.results[{index}]")
            if type(value) is not int or value < 0:
                raise KerrReturningRadiationKernelCacheError(
                    f"TaskResult {name} has a non-exact type or value"
                )
        digest = _trusted_attribute(
            result,
            "sha256",
            f"job_run.results[{index}]",
        )
        try:
            _lowercase_sha256(digest, "TaskResult sha256")
        except ValueError as error:
            raise KerrReturningRadiationKernelCacheError(
                "TaskResult sha256 has a non-exact type or value"
            ) from error
        if type(_trusted_attribute(result, "reused", f"job_run.results[{index}]")) is not bool:
            raise KerrReturningRadiationKernelCacheError(
                "TaskResult reused has a non-exact type"
            )
    for name in ("reused_tasks", "executed_tasks", "max_in_flight_observed"):
        value = _trusted_attribute(job_run, name, "job_run")
        if type(value) is not int or value < 0:
            raise KerrReturningRadiationKernelCacheError(
                f"JobRun {name} has a non-exact type or value"
            )


def _reduce_cached_execution_with_forward_evidence(
    definition: KerrKernelDirectionCacheDefinition,
    job_run: JobRun,
    closure: tuple[KerrKernelSourceClosureEntry, ...],
    reduction_configuration: KerrKernelReductionConfiguration,
) -> tuple[
    KerrCachedReturningRadiationKernelExecution,
    _forward.KerrForwardReturningRadiationConvergenceEvidence | None,
]:
    reduction_configuration = _rebuilt_reduction_configuration(
        reduction_configuration,
        "reduction_configuration.before_reduction",
    )
    _require_reduction_configuration_covers_plan(
        reduction_configuration,
        definition.plan,
    )
    try:
        _require_trusted_exact_tree(
            closure,
            _source_closure_for_definition(definition),
            "source_closure.before_reduction",
        )
    except KerrReturningRadiationKernelCacheVerificationError as error:
        raise KerrReturningRadiationKernelCacheError(
            "source closure changed between cache definition and reduction"
        ) from error
    _validate_definition_execution_identity(definition, closure)
    identity = definition.scientific_context.identity
    _validate_job_run_evidence(definition, job_run)
    records = iter_cached_kernel_direction_records(definition, job_run)
    common = {
        "surface": identity.surface,
        "termination": identity.termination,
        "annulus_edges_over_mass": identity.annulus_edges_over_mass,
        "ray_options": identity.fine_ray_options,
        "surface_options": identity.fine_surface_options,
        "coarse_ray_options": identity.coarse_ray_options,
        "coarse_surface_options": identity.coarse_surface_options,
        "policy": reduction_configuration.policy,
        "area_policy": reduction_configuration.area_policy,
    }
    if definition.plan.formulation == FORWARD:
        stream = _ForwardCachedTransportStream(
            records,
            identity.annulus_edges_over_mass,
        )
        kernel, forward_evidence = (
            _forward._integrate_kerr_returning_radiation_energy_kernel_with_transport_provider_and_evidence(
                **common,
                direction_transport_provider=stream,
            )
        )
    else:
        stream = _ReceiverCachedTransportStream(
            records,
            identity.annulus_edges_over_mass,
        )
        kernel = (
            _receiver
            ._integrate_kerr_returning_radiation_receiver_energy_kernel_with_transport_provider(
                **common,
                direction_transport_provider=stream,
            )
        )
        forward_evidence = None
    stream.finish()
    try:
        _require_trusted_exact_tree(
            closure,
            _source_closure_for_definition(definition),
            "source_closure.after_reduction",
        )
    except KerrReturningRadiationKernelCacheVerificationError as error:
        raise KerrReturningRadiationKernelCacheError(
            "source closure changed while cached reduction was running"
        ) from error
    execution = KerrCachedReturningRadiationKernelExecution(
        definition.plan.formulation,
        kernel,
        definition,
        reduction_configuration,
        job_run,
        closure,
        _execution_audit(definition, job_run, closure),
    )
    return execution, forward_evidence


def _reduce_cached_execution(
    definition: KerrKernelDirectionCacheDefinition,
    job_run: JobRun,
    closure: tuple[KerrKernelSourceClosureEntry, ...],
    reduction_configuration: KerrKernelReductionConfiguration,
) -> KerrCachedReturningRadiationKernelExecution:
    execution, _evidence = _reduce_cached_execution_with_forward_evidence(
        definition,
        job_run,
        closure,
        reduction_configuration,
    )
    return execution


def integrate_cached_kerr_returning_radiation_energy_kernel(
    surface: KerrFiniteThicknessMultiSurface,
    *,
    termination: KerrOblateTermination,
    annulus_edges_over_mass: tuple[float, ...],
    cache_root: Path,
    ray_options: RayTraceOptions = RayTraceOptions(),
    surface_options: SurfaceEventOptions = SurfaceEventOptions(
        subdivisions_per_segment=4
    ),
    coarse_ray_options: RayTraceOptions | None = None,
    coarse_surface_options: SurfaceEventOptions | None = None,
    policy: KerrReturningRadiationKernelPolicy | None = None,
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy | None = None,
    directions_per_task: int = 64,
    jobs: int = 1,
    max_in_flight: int | None = None,
) -> KerrCachedReturningRadiationKernelExecution:
    """Run/resume forward directions, then reduce through the original kernel."""

    selected_policy = KerrReturningRadiationKernelPolicy() if policy is None else policy
    selected_area = (
        KerrFiniteThicknessAreaQuadraturePolicy()
        if area_policy is None
        else area_policy
    )
    definition, closure, reduction_configuration = _build_cache_definition(
        FORWARD,
        surface=surface,
        termination=termination,
        annulus_edges_over_mass=annulus_edges_over_mass,
        ray_options=ray_options,
        surface_options=surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
        policy=selected_policy,
        area_policy=selected_area,
        directions_per_task=directions_per_task,
        evaluator_id=_FORWARD_EVALUATOR_ID,
    )
    _require_exact_source_module_origins()
    job_run = run_kernel_direction_cache(
        definition,
        _evaluate_forward_direction,
        cache_root,
        jobs=jobs,
        max_in_flight=max_in_flight,
    )
    return _reduce_cached_execution(
        definition,
        job_run,
        closure,
        reduction_configuration,
    )


def integrate_cached_kerr_returning_radiation_energy_kernel_native_cpu(
    surface: KerrFiniteThicknessMultiSurface,
    *,
    native_library_path: Path,
    termination: KerrOblateTermination,
    annulus_edges_over_mass: tuple[float, ...],
    cache_root: Path,
    ray_options: RayTraceOptions = RayTraceOptions(),
    surface_options: SurfaceEventOptions = SurfaceEventOptions(
        subdivisions_per_segment=4
    ),
    coarse_ray_options: RayTraceOptions | None = None,
    coarse_surface_options: SurfaceEventOptions | None = None,
    policy: KerrReturningRadiationKernelPolicy | None = None,
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy | None = None,
    directions_per_task: int = 64,
    jobs: int = 1,
    max_in_flight: int | None = None,
    native_segment_capacity: int = 100_000,
    native_crossing_capacity: int = 100_000,
) -> KerrCachedReturningRadiationKernelExecution:
    """Run/resume one explicitly selected strict-CPU forward cache."""

    selected_policy = KerrReturningRadiationKernelPolicy() if policy is None else policy
    selected_area = (
        KerrFiniteThicknessAreaQuadraturePolicy()
        if area_policy is None
        else area_policy
    )
    runtime_binding, backend_descriptor = _native_forward_runtime_binding(
        native_library_path,
        segment_capacity=native_segment_capacity,
        crossing_capacity=native_crossing_capacity,
    )
    definition, closure, reduction_configuration = _build_cache_definition(
        FORWARD,
        surface=surface,
        termination=termination,
        annulus_edges_over_mass=annulus_edges_over_mass,
        ray_options=ray_options,
        surface_options=surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
        policy=selected_policy,
        area_policy=selected_area,
        directions_per_task=directions_per_task,
        evaluator_id=_NATIVE_FORWARD_EVALUATOR_ID,
        evaluator_runtime_binding=runtime_binding,
        native_backend_descriptor=backend_descriptor,
    )
    _require_exact_source_module_origins(
        _source_paths=_NATIVE_FORWARD_SOURCE_CLOSURE_PATHS,
        _module_owners=_NATIVE_FORWARD_SOURCE_CLOSURE_MODULE_OWNERS,
    )
    job_run = run_kernel_direction_cache(
        definition,
        _evaluate_forward_direction_native_cpu,
        cache_root,
        jobs=jobs,
        max_in_flight=max_in_flight,
    )
    return _reduce_cached_execution(
        definition,
        job_run,
        closure,
        reduction_configuration,
    )


def integrate_existing_cached_kerr_returning_radiation_energy_kernel(
    surface: KerrFiniteThicknessMultiSurface,
    *,
    termination: KerrOblateTermination,
    annulus_edges_over_mass: tuple[float, ...],
    cache_root: Path,
    ray_options: RayTraceOptions = RayTraceOptions(),
    surface_options: SurfaceEventOptions = SurfaceEventOptions(
        subdivisions_per_segment=4
    ),
    coarse_ray_options: RayTraceOptions | None = None,
    coarse_surface_options: SurfaceEventOptions | None = None,
    policy: KerrReturningRadiationKernelPolicy | None = None,
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy | None = None,
    directions_per_task: int = 64,
) -> KerrCachedReturningRadiationKernelExecution:
    """Reduce only a complete pre-existing forward cache; never trace rays."""

    _require_cache_only_runtime_binding()
    selected_policy = KerrReturningRadiationKernelPolicy() if policy is None else policy
    selected_area = (
        KerrFiniteThicknessAreaQuadraturePolicy()
        if area_policy is None
        else area_policy
    )
    definition, closure, reduction_configuration = _build_cache_definition(
        FORWARD,
        surface=surface,
        termination=termination,
        annulus_edges_over_mass=annulus_edges_over_mass,
        ray_options=ray_options,
        surface_options=surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
        policy=selected_policy,
        area_policy=selected_area,
        directions_per_task=directions_per_task,
        evaluator_id=_FORWARD_EVALUATOR_ID,
    )
    _require_exact_source_module_origins()
    job_run = require_complete_existing_kernel_direction_cache(
        definition,
        cache_root,
    )
    return _reduce_cached_execution(
        definition,
        job_run,
        closure,
        reduction_configuration,
    )


def integrate_existing_cached_kerr_returning_radiation_energy_kernel_native_cpu(
    surface: KerrFiniteThicknessMultiSurface,
    *,
    native_library_path: Path,
    termination: KerrOblateTermination,
    annulus_edges_over_mass: tuple[float, ...],
    cache_root: Path,
    ray_options: RayTraceOptions = RayTraceOptions(),
    surface_options: SurfaceEventOptions = SurfaceEventOptions(
        subdivisions_per_segment=4
    ),
    coarse_ray_options: RayTraceOptions | None = None,
    coarse_surface_options: SurfaceEventOptions | None = None,
    policy: KerrReturningRadiationKernelPolicy | None = None,
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy | None = None,
    directions_per_task: int = 64,
    native_segment_capacity: int = 100_000,
    native_crossing_capacity: int = 100_000,
) -> KerrCachedReturningRadiationKernelExecution:
    """Reduce only a complete cache under the explicit native identity."""

    _require_cache_only_runtime_binding()
    selected_policy = KerrReturningRadiationKernelPolicy() if policy is None else policy
    selected_area = (
        KerrFiniteThicknessAreaQuadraturePolicy()
        if area_policy is None
        else area_policy
    )
    runtime_binding, backend_descriptor = _native_forward_runtime_binding(
        native_library_path,
        segment_capacity=native_segment_capacity,
        crossing_capacity=native_crossing_capacity,
    )
    definition, closure, reduction_configuration = _build_cache_definition(
        FORWARD,
        surface=surface,
        termination=termination,
        annulus_edges_over_mass=annulus_edges_over_mass,
        ray_options=ray_options,
        surface_options=surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
        policy=selected_policy,
        area_policy=selected_area,
        directions_per_task=directions_per_task,
        evaluator_id=_NATIVE_FORWARD_EVALUATOR_ID,
        evaluator_runtime_binding=runtime_binding,
        native_backend_descriptor=backend_descriptor,
    )
    _require_exact_source_module_origins(
        _source_paths=_NATIVE_FORWARD_SOURCE_CLOSURE_PATHS,
        _module_owners=_NATIVE_FORWARD_SOURCE_CLOSURE_MODULE_OWNERS,
    )
    job_run = require_complete_existing_kernel_direction_cache(
        definition,
        cache_root,
    )
    return _reduce_cached_execution(
        definition,
        job_run,
        closure,
        reduction_configuration,
    )


def integrate_cached_kerr_returning_radiation_receiver_energy_kernel(
    surface: KerrFiniteThicknessMultiSurface,
    *,
    termination: KerrOblateTermination,
    annulus_edges_over_mass: tuple[float, ...],
    cache_root: Path,
    ray_options: RayTraceOptions = RayTraceOptions(),
    surface_options: SurfaceEventOptions = SurfaceEventOptions(
        subdivisions_per_segment=4
    ),
    coarse_ray_options: RayTraceOptions | None = None,
    coarse_surface_options: SurfaceEventOptions | None = None,
    policy: KerrReturningRadiationKernelPolicy | None = None,
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy | None = None,
    directions_per_task: int = 64,
    jobs: int = 1,
    max_in_flight: int | None = None,
) -> KerrCachedReturningRadiationKernelExecution:
    """Run/resume receiver directions, then use the original reducer."""

    selected_policy = KerrReturningRadiationKernelPolicy() if policy is None else policy
    selected_area = (
        KerrFiniteThicknessAreaQuadraturePolicy()
        if area_policy is None
        else area_policy
    )
    definition, closure, reduction_configuration = _build_cache_definition(
        RECEIVER,
        surface=surface,
        termination=termination,
        annulus_edges_over_mass=annulus_edges_over_mass,
        ray_options=ray_options,
        surface_options=surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
        policy=selected_policy,
        area_policy=selected_area,
        directions_per_task=directions_per_task,
        evaluator_id=_RECEIVER_EVALUATOR_ID,
    )
    _require_exact_source_module_origins()
    job_run = run_kernel_direction_cache(
        definition,
        _evaluate_receiver_direction,
        cache_root,
        jobs=jobs,
        max_in_flight=max_in_flight,
    )
    return _reduce_cached_execution(
        definition,
        job_run,
        closure,
        reduction_configuration,
    )


def _rebuild_verified_cached_execution(
    execution: KerrCachedReturningRadiationKernelExecution,
) -> KerrCachedReturningRadiationKernelExecution:
    """Re-authenticate cache records and return the canonical rebuilt wrapper."""

    if type(execution) is not KerrCachedReturningRadiationKernelExecution:
        raise TypeError("execution must have its exact cached wrapper type")
    formulation = _trusted_attribute(execution, "formulation", "execution")
    if type(formulation) is not str or formulation not in (FORWARD, RECEIVER):
        raise KerrReturningRadiationKernelCacheVerificationError(
            "cached execution formulation has a non-exact type or value"
        )
    raw_definition = _trusted_attribute(
        execution,
        "cache_definition",
        "execution",
    )
    if type(raw_definition) is not KerrKernelDirectionCacheDefinition:
        raise KerrReturningRadiationKernelCacheVerificationError(
            "cached execution definition has a non-exact type"
        )
    definition = KerrKernelDirectionCacheDefinition(
        _trusted_attribute(raw_definition, "plan", "execution.cache_definition"),
        _trusted_attribute(
            raw_definition,
            "scientific_context",
            "execution.cache_definition",
        ),
        _trusted_attribute(
            raw_definition,
            "job_spec",
            "execution.cache_definition",
        ),
    )
    if definition.plan.formulation.encode("utf-8") != formulation.encode("utf-8"):
        raise KerrReturningRadiationKernelCacheVerificationError(
            "cached execution formulation differs from its definition"
        )
    raw_reduction_configuration = _trusted_attribute(
        execution,
        "reduction_configuration",
        "execution",
    )
    try:
        reduction_configuration = _rebuilt_reduction_configuration(
            raw_reduction_configuration,
            "execution.reduction_configuration",
        )
        _require_reduction_configuration_covers_plan(
            reduction_configuration,
            definition.plan,
        )
    except KerrReturningRadiationKernelCacheVerificationError:
        raise
    except Exception as error:
        raise KerrReturningRadiationKernelCacheVerificationError(
            "cached execution reduction configuration is invalid"
        ) from error
    # The exact reduction configuration and both work budgets are pure data
    # gates.  Reject them before hashing any source file or opening cache
    # records so a forged wrapper cannot force I/O ahead of its resource limits.
    source_closure = _trusted_attribute(
        execution,
        "source_closure",
        "execution",
    )
    current_closure = _source_closure_for_definition(definition)
    _require_trusted_exact_tree(
        source_closure,
        current_closure,
        "execution.source_closure",
    )
    expected_hashes = tuple(
        sorted(item.binding_sha256 for item in current_closure)
    )
    actual_hashes = definition.scientific_context.identity.source_closure_sha256
    if type(actual_hashes) is not tuple or len(actual_hashes) != len(expected_hashes):
        raise KerrReturningRadiationKernelCacheVerificationError(
            "cache scientific identity does not bind the fixed source closure"
        )
    for actual_hash, expected_hash in zip(actual_hashes, expected_hashes):
        if (
            type(actual_hash) is not str
            or actual_hash.encode("ascii") != expected_hash.encode("ascii")
        ):
            raise KerrReturningRadiationKernelCacheVerificationError(
                "cache scientific identity does not bind the fixed source closure"
            )
    job_run = _trusted_attribute(execution, "job_run", "execution")
    if type(job_run) is not JobRun:
        raise KerrReturningRadiationKernelCacheVerificationError(
            "cached execution JobRun has a non-exact type"
        )
    kernel = _trusted_attribute(execution, "kernel", "execution")
    expected_kernel_type = (
        KerrForwardReturningRadiationKernel
        if formulation == FORWARD
        else KerrReceiverReturningRadiationKernel
    )
    if type(kernel) is not expected_kernel_type:
        raise KerrReturningRadiationKernelCacheVerificationError(
            "cached execution kernel type differs from formulation"
        )
    audit = _trusted_attribute(execution, "execution_audit", "execution")
    if type(audit) is not KerrCachedKernelExecutionAudit:
        raise KerrReturningRadiationKernelCacheVerificationError(
            "cached execution audit has a non-exact type"
        )
    try:
        rebuilt = _reduce_cached_execution(
            definition,
            job_run,
            current_closure,
            reduction_configuration,
        )
        if formulation == FORWARD:
            _forward._require_trusted_exact_tree(
                kernel,
                rebuilt.kernel,
                "cached_execution.kernel",
            )
        else:
            _receiver._require_trusted_exact_tree(
                kernel,
                rebuilt.kernel,
                "cached_execution.kernel",
            )
        _require_trusted_exact_tree(
            audit,
            rebuilt.execution_audit,
            "execution.execution_audit",
        )
        _require_trusted_exact_tree(
            raw_reduction_configuration,
            rebuilt.reduction_configuration,
            "execution.reduction_configuration",
        )
    except KerrReturningRadiationKernelCacheVerificationError:
        raise
    except Exception as error:
        raise KerrReturningRadiationKernelCacheVerificationError(
            "cached execution differs from authenticated same-code replay"
        ) from error
    return rebuilt


def _rebuild_verified_cached_forward_execution_with_evidence(
    execution: KerrCachedReturningRadiationKernelExecution,
) -> tuple[
    KerrCachedReturningRadiationKernelExecution,
    _forward.KerrForwardReturningRadiationConvergenceEvidence,
]:
    """Authenticate one production-forward cache and retain reducer evidence."""

    if type(execution) is not KerrCachedReturningRadiationKernelExecution:
        raise TypeError("execution must have its exact cached wrapper type")
    formulation = _trusted_attribute(execution, "formulation", "execution")
    if type(formulation) is not str or formulation.encode("utf-8") != FORWARD.encode(
        "utf-8"
    ):
        raise KerrReturningRadiationKernelCacheVerificationError(
            "authenticated convergence requires a forward cached execution"
        )
    raw_definition = _trusted_attribute(
        execution, "cache_definition", "execution"
    )
    if type(raw_definition) is not KerrKernelDirectionCacheDefinition:
        raise KerrReturningRadiationKernelCacheVerificationError(
            "cached execution definition has a non-exact type"
        )
    definition = KerrKernelDirectionCacheDefinition(
        _trusted_attribute(raw_definition, "plan", "execution.cache_definition"),
        _trusted_attribute(
            raw_definition, "scientific_context", "execution.cache_definition"
        ),
        _trusted_attribute(
            raw_definition, "job_spec", "execution.cache_definition"
        ),
    )
    context = definition.scientific_context
    if _evaluator_id_from_context(context) not in (
        _FORWARD_EVALUATOR_ID,
        _NATIVE_FORWARD_EVALUATOR_ID,
    ):
        raise KerrReturningRadiationKernelCacheVerificationError(
            "authenticated convergence requires the fixed production evaluator"
        )
    reduction_configuration = _rebuilt_reduction_configuration(
        _trusted_attribute(
            execution, "reduction_configuration", "execution"
        ),
        "execution.reduction_configuration",
    )
    _require_reduction_configuration_covers_plan(
        reduction_configuration,
        definition.plan,
    )
    source_closure = _trusted_attribute(execution, "source_closure", "execution")
    current_closure = _source_closure_for_definition(definition)
    _require_trusted_exact_tree(
        source_closure, current_closure, "execution.source_closure"
    )
    expected_hashes = tuple(sorted(item.binding_sha256 for item in current_closure))
    _require_trusted_exact_tree(
        definition.scientific_context.identity.source_closure_sha256,
        expected_hashes,
        "execution.scientific_identity.source_closure_sha256",
    )
    job_run = _trusted_attribute(execution, "job_run", "execution")
    if type(job_run) is not JobRun:
        raise KerrReturningRadiationKernelCacheVerificationError(
            "cached execution JobRun has a non-exact type"
        )
    rebuilt, evidence = _reduce_cached_execution_with_forward_evidence(
        definition,
        job_run,
        current_closure,
        reduction_configuration,
    )
    if evidence is None:
        raise KerrReturningRadiationKernelCacheVerificationError(
            "forward cached replay omitted convergence evidence"
        )
    _forward._require_trusted_exact_tree(
        _trusted_attribute(execution, "kernel", "execution"),
        rebuilt.kernel,
        "cached_execution.kernel",
    )
    _require_trusted_exact_tree(
        _trusted_attribute(execution, "execution_audit", "execution"),
        rebuilt.execution_audit,
        "execution.execution_audit",
    )
    _require_trusted_exact_tree(
        _trusted_attribute(execution, "reduction_configuration", "execution"),
        rebuilt.reduction_configuration,
        "execution.reduction_configuration",
    )
    return rebuilt, evidence


def verify_cached_kerr_returning_radiation_kernel_execution(
    execution: KerrCachedReturningRadiationKernelExecution,
) -> None:
    """Re-authenticate cache records and rebuild the exact kernel without rays."""

    _rebuild_verified_cached_execution(execution)


def _production_forward_scientific_binding(
    execution: KerrCachedReturningRadiationKernelExecution,
) -> tuple[str, str]:
    """Build the layout-free evidence bound by certified thermal consumers."""

    definition = execution.cache_definition
    context = definition.scientific_context
    evaluator_id = _evaluator_id_from_context(context)
    if evaluator_id not in (
        _FORWARD_EVALUATOR_ID,
        _NATIVE_FORWARD_EVALUATOR_ID,
    ):
        raise KerrReturningRadiationKernelCacheVerificationError(
            "certified cached forward sources require the fixed production evaluator"
        )
    expected_input_count = (
        2 if evaluator_id == _NATIVE_FORWARD_EVALUATOR_ID else 1
    )
    if type(context.inputs) is not tuple or len(context.inputs) != expected_input_count:
        raise KerrReturningRadiationKernelCacheVerificationError(
            "production cached forward source has an invalid input artifact set"
        )
    evaluator_artifacts = tuple(
        item
        for item in context.inputs
        if item.uri.startswith(_EVALUATOR_ARTIFACT_URI_PREFIX)
    )
    if len(evaluator_artifacts) != 1:
        raise KerrReturningRadiationKernelCacheVerificationError(
            "production cached forward source lacks one evaluator artifact"
        )
    evaluator_artifact = evaluator_artifacts[0]
    if type(evaluator_artifact) is not InputArtifact:
        raise KerrReturningRadiationKernelCacheVerificationError(
            "production cached forward evaluator artifact has a non-exact type"
        )
    expected_artifact = _evaluator_input(evaluator_id)
    _require_trusted_exact_tree(
        evaluator_artifact,
        expected_artifact,
        "cached_forward.evaluator_artifact",
    )
    evaluator_document: dict[str, Any] = {
        "artifact": evaluator_artifact.as_dict(),
        "implementationId": evaluator_id,
        "productionForwardOnly": True,
    }
    binding_implementation_id = (
        CACHED_NATIVE_FORWARD_SCIENTIFIC_BINDING_IMPLEMENTATION_ID
        if evaluator_id == _NATIVE_FORWARD_EVALUATOR_ID
        else CACHED_FORWARD_SCIENTIFIC_BINDING_IMPLEMENTATION_ID
    )
    if evaluator_id == _NATIVE_FORWARD_EVALUATOR_ID:
        runtime_binding = context.evaluator_runtime_binding
        if type(runtime_binding) is not KerrKernelEvaluatorRuntimeBinding:
            raise KerrReturningRadiationKernelCacheVerificationError(
                "native cached forward source lacks its runtime binding"
            )
        _require_trusted_exact_tree(
            runtime_binding.descriptor_input,
            next(
                item
                for item in context.inputs
                if item is not evaluator_artifact
            ),
            "cached_forward.runtime_artifact",
        )
        evaluator_document["runtimeBinding"] = {
            "artifact": runtime_binding.descriptor_input.as_dict(),
            "backendDescriptorSha256": (
                runtime_binding.backend_descriptor_sha256
            ),
            "pathFreeDescriptor": runtime_binding.path_free_descriptor(),
        }
    audit = execution.execution_audit
    closure = execution.source_closure
    binding = {
        "authentication": {
            "cachedReplayRetracesRays": False,
            "hasIndependentGeodesicOrPhysicsOracle": False,
            "isSameCodeCacheEvidence": True,
            "sourceClosureRecheckedBeforeAndAfterReduction": (
                audit.source_closure_rechecked_before_and_after_reduction
            ),
            "taskReuseHistoryIsCryptographicallyAuthenticated": (
                audit.is_task_reuse_history_cryptographically_authenticated
            ),
        },
        "evaluator": evaluator_document,
        "excludedOperationalIdentity": {
            "cacheAbsolutePaths": True,
            "cacheJobKeyAndChunkLayout": True,
            "executedOrReusedCounters": True,
            "reuseHistory": True,
            "workerScheduling": True,
        },
        "forwardKernelDescriptorSha256": execution.kernel.model_descriptor_sha256,
        "implementationId": binding_implementation_id,
        "reductionConfiguration": execution.reduction_configuration.as_dict(),
        "reductionConfigurationSha256": (
            execution.reduction_configuration.descriptor_sha256
        ),
        "scientificPlanSha256": context.scientific_plan_sha256,
        # Retained as a compatibility alias; the explicit field below names
        # the narrowed semantics after transport/reduction identity splitting.
        "scientificJobKey": context.scientific_job_key,
        "transportScientificJobKey": context.scientific_job_key,
        "sourceRuntimeClosure": {
            "entries": tuple(
                {
                    "bindingSha256": item.binding_sha256,
                    "byteLength": item.byte_length,
                    "logicalPath": item.logical_path,
                    "sha256": item.sha256,
                }
                for item in closure
            ),
            "manifestSha256": _source_closure_manifest_sha256(closure),
            "numericBackendLogicalPath": _NUMERIC_BACKEND_LOGICAL_PATH,
        },
    }
    payload = canonical_json_bytes(binding)
    if not payload.endswith(b"\n"):
        raise AssertionError("canonical JSON encoder omitted its terminal newline")
    descriptor_json = payload[:-1].decode("utf-8")
    return descriptor_json, hashlib.sha256(
        descriptor_json.encode("utf-8")
    ).hexdigest()


def validate_and_reduce_cached_kerr_returning_radiation_energy_kernel(
    execution: KerrCachedReturningRadiationKernelExecution,
    *,
    require_equatorial_symmetry: bool = True,
) -> KerrValidatedCachedForwardReduction:
    """Authenticate a production forward cache once and reduce it without rays."""

    if type(require_equatorial_symmetry) is not bool:
        raise TypeError("require_equatorial_symmetry must be an exact bool")
    if require_equatorial_symmetry is not True:
        raise ValueError(
            "four-face cached reduction is prohibited without equatorial symmetry"
        )
    rebuilt = _rebuild_verified_cached_execution(execution)
    if rebuilt.formulation.encode("utf-8") != FORWARD.encode("utf-8"):
        raise KerrReturningRadiationKernelCacheVerificationError(
            "certified thermal reduction requires a forward cached execution"
        )
    if type(rebuilt.kernel) is not KerrForwardReturningRadiationKernel:
        raise KerrReturningRadiationKernelCacheVerificationError(
            "forward cached execution rebuilt a non-forward kernel"
        )
    binding_json, binding_sha = _production_forward_scientific_binding(rebuilt)
    axisymmetric = _forward._reduce_verified_four_face_kernel(
        rebuilt.kernel,
        producer_id=(
            f"{_forward.IMPLEMENTATION_ID}:"
            f"{rebuilt.kernel.model_descriptor_sha256}"
        ),
    )
    result = object.__new__(KerrValidatedCachedForwardReduction)
    for name, value in (
        ("execution", rebuilt),
        ("forward_kernel", rebuilt.kernel),
        ("axisymmetric_kernel", axisymmetric),
        ("scientific_binding_json", binding_json),
        ("scientific_binding_sha256", binding_sha),
    ):
        object.__setattr__(result, name, value)
    return result


def verify_and_reduce_cached_kerr_returning_radiation_energy_kernel(
    execution: KerrCachedReturningRadiationKernelExecution,
    *,
    require_equatorial_symmetry: bool = True,
) -> AxisymmetricReturningRadiationKernel:
    """Return the direct-descriptor-equivalent cached axisymmetric kernel."""

    return validate_and_reduce_cached_kerr_returning_radiation_energy_kernel(
        execution,
        require_equatorial_symmetry=require_equatorial_symmetry,
    ).axisymmetric_kernel


__all__ = (
    "CACHED_FORWARD_SCIENTIFIC_BINDING_IMPLEMENTATION_ID",
    "CACHED_NATIVE_FORWARD_SCIENTIFIC_BINDING_IMPLEMENTATION_ID",
    "FORWARD_TRANSPORT_SCHEMA",
    "IMPLEMENTATION_ID",
    "REDUCTION_CONFIGURATION_SCHEMA",
    "RECEIVER_TRANSPORT_SCHEMA",
    "SCIENTIFIC_STATUS",
    "KerrCachedKernelExecutionAudit",
    "KerrCachedReturningRadiationKernelExecution",
    "KerrValidatedCachedForwardReduction",
    "KerrKernelSourceClosureEntry",
    "KerrKernelReductionConfiguration",
    "KerrReturningRadiationKernelCacheError",
    "KerrReturningRadiationKernelCacheVerificationError",
    "build_forward_kerr_returning_radiation_kernel_cache_definition",
    "build_native_forward_kerr_returning_radiation_kernel_cache_definition",
    "build_receiver_kerr_returning_radiation_kernel_cache_definition",
    "integrate_cached_kerr_returning_radiation_energy_kernel",
    "integrate_cached_kerr_returning_radiation_energy_kernel_native_cpu",
    "integrate_existing_cached_kerr_returning_radiation_energy_kernel",
    "integrate_existing_cached_kerr_returning_radiation_energy_kernel_native_cpu",
    "integrate_cached_kerr_returning_radiation_receiver_energy_kernel",
    "validate_and_reduce_cached_kerr_returning_radiation_energy_kernel",
    "verify_and_reduce_cached_kerr_returning_radiation_energy_kernel",
    "verify_cached_kerr_returning_radiation_kernel_execution",
)
