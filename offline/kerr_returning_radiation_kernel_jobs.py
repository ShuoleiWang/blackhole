"""Deterministic resumable task planning for returning-radiation kernels.

This module is deliberately an execution/cache substrate, not another kernel
integrator.  It does not evaluate a ray, change a quadrature weight, sum a
coefficient, or construct either public kernel result type.  Instead it gives
the existing forward- and receiver-centred five-pass quadratures a canonical
integer direction lattice and stores evaluator-produced narrow transport
records through :mod:`offline.job`.

The transport scientific identity excludes ``directions_per_task``, worker
controls, area quadrature, reduction convergence/symmetry thresholds, and
reduction work budgets.  It is built from the exact Kerr surface,
termination, annulus edges, fine/coarse trace options, five-pass plan, and a
non-empty transitive source-closure digest set; arbitrary JSON identity
documents are not accepted.  Evaluators receive an authenticated layout-free
transport context, never the cache JobSpec.  Reduction policy is deliberately
owned and authenticated by the consuming bridge rather than this transport
cache substrate.  A reducer must consume records
from :func:`iter_cached_kernel_direction_records`, whose order is the canonical
pass/face/annulus/rho/mu/psi order, rather than future completion order.  This
is the boundary needed for bitwise-stable ``math.fsum`` inputs.

An evaluator may trace and publicly replay a direction once before returning a
record; a valid cache receipt then avoids repeating that work after restart.
The cache proves byte identity and producer provenance only.  An unbound
generic definition treats its evaluator as a caller-trusted deterministic
pure-function boundary and may reuse that cache with a different caller
evaluator; it therefore makes no authenticated evaluator-identity claim.
Reserved bridge inputs instead bind a canonical exact-function/source/code
descriptor which the public runner rechecks before any cache access; the
jobs layer records that descriptor as a scientific input but cannot infer its
membership in an opaque source-closure digest set.  A production bridge may
prove closure membership separately.  This closes accidental callable
substitution, but arbitrary
same-process code execution and hidden Python global state are not sandboxed or
proved absent.  This is same-code execution evidence, never an independent
geodesic or physics oracle.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, is_dataclass
import errno
import fcntl
import hashlib
import json
import marshal
import math
import multiprocessing
import os
from pathlib import Path
import stat
from types import CodeType, FunctionType, MappingProxyType
from typing import Any, Final
import uuid

from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.job import (
    InputArtifact,
    JobRun,
    JobSpec,
    RECEIPT_SCHEMA,
    TaskKey,
    TaskResult,
    canonical_json_bytes,
)
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface


TASK_PLAN_SCHEMA: Final = "blackhole.returning-radiation-kernel-task-plan/v1"
SCIENTIFIC_JOB_SCHEMA: Final = (
    "blackhole.returning-radiation-kernel-scientific-job/v2"
)
TASK_PAYLOAD_SCHEMA: Final = (
    "blackhole.returning-radiation-kernel-direction-task/v1"
)
PRODUCER_ID: Final = "offline.kerr-returning-radiation-direction-cache"
ALGORITHM_VERSION: Final = "1.1.0"
EVALUATOR_DESCRIPTOR_SCHEMA: Final = (
    "blackhole.returning-radiation-kernel-direction-evaluator/v2"
)
MAXIMUM_WORKERS: Final = 64
MAXIMUM_IN_FLIGHT_TASKS: Final = 256

FORWARD: Final = "forward"
RECEIVER: Final = "receiver"
_FORMULATIONS: Final = (FORWARD, RECEIVER)
_FACES: Final = (UPPER, LOWER)
_PASS_NAMES: Final = (
    "full",
    "half-rho",
    "half-mu",
    "half-psi",
    "phase-shifted",
)
_SHA256_LENGTH: Final = 64
_MAXIMUM_ORDER: Final = 64
_MAXIMUM_PSI_COUNT: Final = 256
_MAXIMUM_DIRECTIONS_PER_TASK: Final = 4096
_MAXIMUM_DIRECTION_EVALUATIONS: Final = 2_000_000
_MAXIMUM_TASK_COUNT: Final = 65_536
_MAXIMUM_TRANSPORT_JSON_BYTES: Final = 64 * 1024
_MAXIMUM_TASK_PAYLOAD_BYTES: Final = 256 * 1024
_MAXIMUM_RECEIPT_BYTES: Final = 16 * 1024
_MAXIMUM_TOTAL_CACHE_BYTES: Final = (
    _MAXIMUM_TASK_COUNT * _MAXIMUM_TASK_PAYLOAD_BYTES
)
_MAXIMUM_JOB_SPEC_BYTES: Final = 32 * 1024 * 1024
_MAXIMUM_JSON_DEPTH: Final = 32
_PATH_TYPE: Final = type(Path())
_EVALUATOR_ARTIFACT_URI_ROOT: Final = (
    "urn:blackhole:returning-radiation-kernel-direction-evaluator"
)
_EVALUATOR_ARTIFACT_URI_PREFIX: Final = f"{_EVALUATOR_ARTIFACT_URI_ROOT}:v2:"
EVALUATOR_RUNTIME_DESCRIPTOR_SCHEMA: Final = (
    "blackhole.returning-radiation-kernel-evaluator-runtime/v1"
)
_EVALUATOR_RUNTIME_ARTIFACT_URI_ROOT: Final = (
    "urn:blackhole:returning-radiation-kernel-evaluator-runtime"
)
_EVALUATOR_RUNTIME_ARTIFACT_URI_PREFIX: Final = (
    f"{_EVALUATOR_RUNTIME_ARTIFACT_URI_ROOT}:v1:"
)
_MAXIMUM_EVALUATOR_IMPLEMENTATION_ID_BYTES: Final = 512
_EVALUATOR_SOURCE_READ_CHUNK_BYTES: Final = 1024 * 1024
_MAXIMUM_EVALUATOR_SOURCE_BYTES: Final = 16 * 1024 * 1024
_MAXIMUM_EVALUATOR_RUNTIME_DESCRIPTOR_BYTES: Final = 4 * 1024 * 1024
_MAXIMUM_EVALUATOR_RUNTIME_LIBRARY_BYTES: Final = 256 * 1024 * 1024
_MAXIMUM_EVALUATOR_RUNTIME_CAPACITY: Final = 2_000_000
_EVALUATOR_RUNTIME_LIBRARY_ARTIFACT_NAME: Final = "libblackhole_cpu.dylib"

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": "same-code deterministic direction-task cache substrate",
        "allScientificStateComesFromTypedContext": True,
        "evaluatorContract": (
            "trusted deterministic pure function of exact typed scientific "
            "context and canonical direction coordinate"
        ),
        "evaluatorIdentityBindingMode": "caller-trusted-unbound",
        "authenticatedEvaluatorIdentity": False,
        "evaluatorIdentityEvidence": (
            "none; caller-supplied evaluator identity is not authenticated"
        ),
        "reservedEvaluatorArtifactCount": 0,
        "reservedEvaluatorArtifactUri": None,
        "genericUnboundCacheMayBeReusedAcrossCallerEvaluators": True,
        "transportScientificIdentityExcludesAreaQuadrature": True,
        "transportScientificIdentityExcludesReductionConvergence": True,
        "transportScientificIdentityExcludesReductionSymmetryThresholds": True,
        "transportScientificIdentityExcludesReductionWorkBudgets": True,
        "quadratureOrdersBoundByScientificPlan": True,
        "reductionConfigurationAuthenticatedByThisSubstrate": False,
        "evaluatorImplementationAndSourceIdentityBoundAsScientificInput": False,
        "evaluatorImplementationAndSourceIdentityInSourceClosure": False,
        "reservedEvaluatorCallableDescriptorCheckedBeforeCacheAccess": False,
        "claimsProtectionFromArbitraryHiddenEvaluatorClosure": False,
        "claimsProtectionFromMaliciousSameProcessCodeExecution": False,
        "hasIndependentGeodesicOracle": False,
        "hasIndependentPhysicsOracle": False,
        "isIntegratedPublicKernelAcceleration": False,
        "prohibitedClaim": (
            "Do not claim independent physics, existing-kernel acceleration, "
            "protection from arbitrary evaluator hidden state/closures, or "
            "protection from malicious same-process code execution."
        ),
    }
)

ExecutorFactory = Callable[[int], Executor]


class KerrReturningRadiationKernelJobError(RuntimeError):
    """Raised when a cached direction job is incomplete or unauthenticated."""


def _exact_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive exact int")
    return value


def _exact_even_order(value: Any, label: str, maximum: int) -> int:
    value = _exact_positive_int(value, label)
    if value < 4 or value > maximum or value % 2:
        raise ValueError(f"{label} must be even and lie in [4, {maximum}]")
    return value


def _exact_even_grid_order(value: Any, label: str, maximum: int) -> int:
    """Validate a realized pass, whose mandatory half grid may have order 2."""

    value = _exact_positive_int(value, label)
    if value < 2 or value > maximum or value % 2:
        raise ValueError(f"{label} must be even and lie in [2, {maximum}]")
    return value


def _lowercase_sha256(value: Any, label: str) -> str:
    if type(value) is not str or len(value) != _SHA256_LENGTH:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    if value.lower() != value:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest") from error
    if len(decoded) != 32:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_json_copy(
    value: Any,
    path: str = "$transport",
    depth: int = 0,
) -> Any:
    """Copy an exact finite JSON tree without invoking subclass hooks."""

    if depth > _MAXIMUM_JSON_DEPTH:
        raise ValueError(f"{path} exceeds the hard JSON depth limit")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if type(value) is dict:
        result: dict[str, Any] = {}
        keys = tuple(value)
        if any(type(key) is not str for key in keys):
            raise TypeError(f"{path} has a non-exact string key")
        for key in sorted(keys):
            result[key] = _strict_json_copy(
                value[key],
                f"{path}.{key}",
                depth + 1,
            )
        return result
    if type(value) in (list, tuple):
        return [
            _strict_json_copy(item, f"{path}[{index}]", depth + 1)
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains unsupported exact JSON value {type(value).__name__}"
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_json_nesting(payload: bytes, label: str) -> None:
    """Bound nesting before handing untrusted bytes to the JSON decoder."""

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
                raise KerrReturningRadiationKernelJobError(
                    f"{label} exceeds the hard JSON depth limit"
                )
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth < 0:
                raise KerrReturningRadiationKernelJobError(
                    f"{label} has invalid JSON nesting"
                )


def _strict_json_payload(
    payload: bytes,
    label: str = "cached direction payload",
    maximum_bytes: int = _MAXIMUM_TASK_PAYLOAD_BYTES,
) -> Any:
    if type(payload) is not bytes or len(payload) > maximum_bytes:
        raise KerrReturningRadiationKernelJobError(
            f"{label} exceeds its hard byte limit"
        )
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
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise KerrReturningRadiationKernelJobError(
            f"{label} is not strict JSON"
        ) from error
    if canonical_json_bytes(value) != payload:
        raise KerrReturningRadiationKernelJobError(
            f"{label} is not canonical JSON"
        )
    return value


def _matches_task_document(value: Any, key: TaskKey) -> bool:
    if type(value) is not dict or set(value) != {
        "height",
        "sampleIndex",
        "width",
        "x",
        "y",
    }:
        return False
    return all(type(item) is int for item in value.values()) and value == key.as_dict()


def _trusted_attribute(value: Any, name: str, path: str) -> Any:
    try:
        return object.__getattribute__(value, name)
    except (AttributeError, TypeError) as error:
        raise TypeError(f"{path}.{name} is missing") from error


def _require_exact_schema_types(actual: Any, template: Any, path: str) -> None:
    if type(actual) is not type(template):
        raise TypeError(
            f"{path} has non-exact type {type(actual).__name__}; "
            f"expected {type(template).__name__}"
        )
    if is_dataclass(template) and not isinstance(template, type):
        for item in fields(template):
            _require_exact_schema_types(
                _trusted_attribute(actual, item.name, path),
                _trusted_attribute(template, item.name, path),
                f"{path}.{item.name}",
            )
        return
    if type(template) is tuple:
        if len(actual) != len(template):
            raise TypeError(f"{path} tuple length differs from its exact schema")
        for index, (actual_item, template_item) in enumerate(zip(actual, template)):
            _require_exact_schema_types(
                actual_item,
                template_item,
                f"{path}[{index}]",
            )
        return
    if type(template) not in (
        float,
        int,
        bool,
        str,
        bytes,
        _PATH_TYPE,
        type(None),
    ):
        raise TypeError(f"{path} uses unsupported exact schema type")


def _require_exact_tree(actual: Any, expected: Any, path: str) -> None:
    _require_exact_schema_types(actual, expected, path)
    if is_dataclass(expected) and not isinstance(expected, type):
        for item in fields(expected):
            _require_exact_tree(
                _trusted_attribute(actual, item.name, path),
                _trusted_attribute(expected, item.name, path),
                f"{path}.{item.name}",
            )
        return
    if type(expected) is tuple:
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _require_exact_tree(actual_item, expected_item, f"{path}[{index}]")
        return
    if type(expected) is float:
        differs = actual.hex() != expected.hex()
    elif type(expected) is bool:
        differs = actual is not expected
    elif type(expected) is int:
        differs = actual != expected
    elif type(expected) is str:
        differs = actual.encode("utf-8") != expected.encode("utf-8")
    elif type(expected) is bytes:
        differs = actual != expected
    elif type(expected) is _PATH_TYPE:
        differs = os.fspath(actual).encode("utf-8") != os.fspath(expected).encode(
            "utf-8"
        )
    elif expected is None:
        differs = False
    else:
        raise TypeError(f"{path} uses unsupported exact comparison type")
    if differs:
        raise ValueError(f"{path} differs from its exact reconstruction")


def _rebuilt_dataclass(value: Any, template: Any, label: str) -> Any:
    _require_exact_schema_types(value, template, label)
    rebuilt = type(template)(**asdict(value))
    _require_exact_tree(value, rebuilt, label)
    return rebuilt


def _rebuilt_surface(
    surface: KerrFiniteThicknessMultiSurface,
) -> KerrFiniteThicknessMultiSurface:
    template_calibration = StationaryKerrFiniteThicknessCalibration(
        dimensionless_spin=0.0,
        eddington_scaled_mass_accretion_rate=0.01,
        outer_radius_over_mass=10.0,
    )
    template = KerrFiniteThicknessMultiSurface(
        KerrKerrSchildMetric(),
        template_calibration,
    )
    _require_exact_schema_types(surface, template, "identity.surface")
    metric = _rebuilt_dataclass(
        surface.metric,
        KerrKerrSchildMetric(),
        "identity.surface.metric",
    )
    calibration = _rebuilt_dataclass(
        surface.calibration,
        template_calibration,
        "identity.surface.calibration",
    )
    rebuilt = KerrFiniteThicknessMultiSurface(metric, calibration)
    _require_exact_tree(surface, rebuilt, "identity.surface")
    return rebuilt


@dataclass(frozen=True, slots=True)
class KerrKernelScientificIdentity:
    """Exact trace identity shared with no reduction or cache-layout controls."""

    formulation: str
    surface: KerrFiniteThicknessMultiSurface
    termination: KerrOblateTermination
    annulus_edges_over_mass: tuple[float, ...]
    fine_ray_options: RayTraceOptions
    fine_surface_options: SurfaceEventOptions
    coarse_ray_options: RayTraceOptions | None
    coarse_surface_options: SurfaceEventOptions | None
    source_closure_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.formulation) is not str or self.formulation not in _FORMULATIONS:
            raise ValueError("identity formulation must be exact forward or receiver")
        surface = _rebuilt_surface(self.surface)
        termination = _rebuilt_dataclass(
            self.termination,
            KerrOblateTermination(0.0, 1.0, 2.0),
            "identity.termination",
        )
        fine_ray = _rebuilt_dataclass(
            self.fine_ray_options,
            RayTraceOptions(),
            "identity.fine_ray_options",
        )
        fine_surface = _rebuilt_dataclass(
            self.fine_surface_options,
            SurfaceEventOptions(),
            "identity.fine_surface_options",
        )
        if (self.coarse_ray_options is None) != (
            self.coarse_surface_options is None
        ):
            raise ValueError("coarse ray and surface options must appear together")
        if self.coarse_ray_options is None:
            coarse_ray = None
            coarse_surface = None
        else:
            coarse_ray = _rebuilt_dataclass(
                self.coarse_ray_options,
                RayTraceOptions(),
                "identity.coarse_ray_options",
            )
            coarse_surface = _rebuilt_dataclass(
                self.coarse_surface_options,
                SurfaceEventOptions(),
                "identity.coarse_surface_options",
            )
        if type(self.annulus_edges_over_mass) is not tuple or len(
            self.annulus_edges_over_mass
        ) < 2:
            raise TypeError("annulus edges must be an exact non-empty tuple")
        edges = tuple(
            value
            for value in self.annulus_edges_over_mass
            if type(value) is float and math.isfinite(value)
        )
        if len(edges) != len(self.annulus_edges_over_mass):
            raise TypeError("annulus edges must contain finite exact floats")
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise ValueError("annulus edges must be strictly increasing")
        if (
            edges[0].hex()
            != float(surface.calibration.isco_radius_over_mass).hex()
            or edges[-1].hex()
            != float(surface.calibration.outer_radius_over_mass).hex()
        ):
            raise ValueError("annulus edges must exactly cover ISCO through R_out")
        if termination.spin_a_m.hex() != surface.metric.spin_a_m.hex():
            raise ValueError("termination and surface metric spins differ")

        if type(self.source_closure_sha256) is not tuple or not (
            self.source_closure_sha256
        ):
            raise TypeError("source closure must be a non-empty exact tuple")
        source_closure = tuple(
            _lowercase_sha256(item, "source closure digest")
            for item in self.source_closure_sha256
        )
        if source_closure != tuple(sorted(source_closure)):
            raise ValueError("source closure digests must be in canonical order")
        if len(set(source_closure)) != len(source_closure):
            raise ValueError("source closure digests must be unique")

        object.__setattr__(self, "surface", surface)
        object.__setattr__(self, "termination", termination)
        object.__setattr__(self, "annulus_edges_over_mass", edges)
        object.__setattr__(self, "fine_ray_options", fine_ray)
        object.__setattr__(self, "fine_surface_options", fine_surface)
        object.__setattr__(self, "coarse_ray_options", coarse_ray)
        object.__setattr__(self, "coarse_surface_options", coarse_surface)
        object.__setattr__(self, "source_closure_sha256", source_closure)

    @property
    def annulus_count(self) -> int:
        return len(self.annulus_edges_over_mass) - 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "annulusEdgesOverMass": self.annulus_edges_over_mass,
            "coarseRayOptions": (
                None
                if self.coarse_ray_options is None
                else asdict(self.coarse_ray_options)
            ),
            "coarseSurfaceOptions": (
                None
                if self.coarse_surface_options is None
                else asdict(self.coarse_surface_options)
            ),
            "fineRayOptions": asdict(self.fine_ray_options),
            "fineSurfaceOptions": asdict(self.fine_surface_options),
            "formulation": self.formulation,
            "metric": asdict(self.surface.metric),
            "sourceClosureSha256": self.source_closure_sha256,
            "surface": {
                "calibration": asdict(self.surface.calibration),
                "surfaceIds": self.surface.surface_ids,
            },
            "termination": asdict(self.termination),
        }


def _build_kernel_scientific_identity(
    formulation: str,
    *,
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    annulus_edges_over_mass: tuple[float, ...],
    fine_ray_options: RayTraceOptions,
    fine_surface_options: SurfaceEventOptions,
    coarse_ray_options: RayTraceOptions | None,
    coarse_surface_options: SurfaceEventOptions | None,
    source_closure_sha256: Sequence[str],
) -> KerrKernelScientificIdentity:
    raw_closure = tuple(source_closure_sha256)
    closure = tuple(
        sorted(
            _lowercase_sha256(item, "source closure digest")
            for item in raw_closure
        )
    )
    return KerrKernelScientificIdentity(
        formulation,
        surface,
        termination,
        annulus_edges_over_mass,
        fine_ray_options,
        fine_surface_options,
        coarse_ray_options,
        coarse_surface_options,
        closure,
    )


def build_forward_kernel_scientific_identity(
    *,
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    annulus_edges_over_mass: tuple[float, ...],
    fine_ray_options: RayTraceOptions,
    fine_surface_options: SurfaceEventOptions,
    coarse_ray_options: RayTraceOptions | None,
    coarse_surface_options: SurfaceEventOptions | None,
    source_closure_sha256: Sequence[str],
) -> KerrKernelScientificIdentity:
    return _build_kernel_scientific_identity(
        FORWARD,
        surface=surface,
        termination=termination,
        annulus_edges_over_mass=annulus_edges_over_mass,
        fine_ray_options=fine_ray_options,
        fine_surface_options=fine_surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
        source_closure_sha256=source_closure_sha256,
    )


def build_receiver_kernel_scientific_identity(
    *,
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    annulus_edges_over_mass: tuple[float, ...],
    fine_ray_options: RayTraceOptions,
    fine_surface_options: SurfaceEventOptions,
    coarse_ray_options: RayTraceOptions | None,
    coarse_surface_options: SurfaceEventOptions | None,
    source_closure_sha256: Sequence[str],
) -> KerrKernelScientificIdentity:
    return _build_kernel_scientific_identity(
        RECEIVER,
        surface=surface,
        termination=termination,
        annulus_edges_over_mass=annulus_edges_over_mass,
        fine_ray_options=fine_ray_options,
        fine_surface_options=fine_surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
        source_closure_sha256=source_closure_sha256,
    )


@dataclass(frozen=True, slots=True)
class KerrKernelGridPass:
    """One of the exact five convergence grids used by both kernel APIs."""

    name: str
    rho_order: int
    mu_order: int
    psi_count: int
    phase_cells: float

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in _PASS_NAMES:
            raise ValueError("grid-pass name is unsupported")
        _exact_even_grid_order(self.rho_order, "rho_order", _MAXIMUM_ORDER)
        _exact_even_grid_order(self.mu_order, "mu_order", _MAXIMUM_ORDER)
        _exact_even_grid_order(self.psi_count, "psi_count", _MAXIMUM_PSI_COUNT)
        if type(self.phase_cells) is not float or self.phase_cells.hex() not in (
            0.0.hex(),
            0.5.hex(),
        ):
            raise ValueError("phase_cells must be exact float 0.0 or 0.5")
        expected_phase = 0.5 if self.name == "phase-shifted" else 0.0
        if self.phase_cells.hex() != expected_phase.hex():
            raise ValueError("grid-pass name and phase_cells disagree")

    @property
    def angular_direction_count(self) -> int:
        return self.mu_order * self.psi_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "muOrder": self.mu_order,
            "name": self.name,
            "phaseCells": self.phase_cells,
            "psiCount": self.psi_count,
            "rhoOrder": self.rho_order,
        }


def _five_grid_passes(
    rho_order: int,
    mu_order: int,
    psi_count: int,
) -> tuple[KerrKernelGridPass, ...]:
    return (
        KerrKernelGridPass("full", rho_order, mu_order, psi_count, 0.0),
        KerrKernelGridPass("half-rho", rho_order // 2, mu_order, psi_count, 0.0),
        KerrKernelGridPass("half-mu", rho_order, mu_order // 2, psi_count, 0.0),
        KerrKernelGridPass("half-psi", rho_order, mu_order, psi_count // 2, 0.0),
        KerrKernelGridPass("phase-shifted", rho_order, mu_order, psi_count, 0.5),
    )


@dataclass(frozen=True, slots=True)
class KerrKernelScientificPlan:
    """Layout-free five-pass topology visible to scientific evaluators."""

    formulation: str
    annulus_count: int
    rho_order: int
    mu_order: int
    psi_count: int

    def __post_init__(self) -> None:
        if type(self.formulation) is not str or self.formulation not in _FORMULATIONS:
            raise ValueError("formulation must be exact forward or receiver")
        _exact_positive_int(self.annulus_count, "annulus_count")
        _exact_even_order(self.rho_order, "rho_order", _MAXIMUM_ORDER)
        _exact_even_order(self.mu_order, "mu_order", _MAXIMUM_ORDER)
        _exact_even_order(self.psi_count, "psi_count", _MAXIMUM_PSI_COUNT)
        if self.direction_count > _MAXIMUM_DIRECTION_EVALUATIONS:
            raise ValueError(
                "scientific plan exceeds the current kernel hard direction budget"
            )

    @property
    def passes(self) -> tuple[KerrKernelGridPass, ...]:
        return _five_grid_passes(self.rho_order, self.mu_order, self.psi_count)

    @property
    def direction_count(self) -> int:
        return 2 * self.annulus_count * sum(
            item.rho_order * item.mu_order * item.psi_count
            for item in self.passes
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "annulusCount": self.annulus_count,
            "canonicalOrder": "pass/face/annulus/rho/mu/psi",
            "directionCount": self.direction_count,
            "faces": _FACES,
            "formulation": self.formulation,
            "passes": tuple(item.as_dict() for item in self.passes),
            "schema": TASK_PLAN_SCHEMA,
        }

    @property
    def descriptor_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())


@dataclass(frozen=True, order=True, slots=True)
class KerrKernelDirectionCoordinate:
    """Canonical integer address for one expensive direction evaluation."""

    ordinal: int
    pass_index: int
    pass_name: str
    face_index: int
    face: str
    annulus_index: int
    rho_index: int
    mu_index: int
    psi_index: int

    def __post_init__(self) -> None:
        for name in (
            "ordinal",
            "pass_index",
            "face_index",
            "annulus_index",
            "rho_index",
            "mu_index",
            "psi_index",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact int")
        if type(self.pass_name) is not str or self.pass_name not in _PASS_NAMES:
            raise ValueError("pass_name is unsupported")
        if (
            self.pass_index >= len(_PASS_NAMES)
            or self.pass_name != _PASS_NAMES[self.pass_index]
        ):
            raise ValueError("pass_name and pass_index disagree")
        if type(self.face) is not str or self.face not in _FACES:
            raise ValueError("face is unsupported")
        if self.face_index >= len(_FACES) or self.face != _FACES[self.face_index]:
            raise ValueError("face and face_index disagree")

    def as_dict(self) -> dict[str, Any]:
        return {
            "annulusIndex": self.annulus_index,
            "face": self.face,
            "faceIndex": self.face_index,
            "muIndex": self.mu_index,
            "ordinal": self.ordinal,
            "passIndex": self.pass_index,
            "passName": self.pass_name,
            "psiIndex": self.psi_index,
            "rhoIndex": self.rho_index,
        }


@dataclass(frozen=True, slots=True)
class KerrKernelDirectionTaskPlan:
    """Five-pass direction topology plus a non-scientific chunk layout."""

    formulation: str
    annulus_count: int
    rho_order: int
    mu_order: int
    psi_count: int
    directions_per_task: int = 64

    def __post_init__(self) -> None:
        if type(self.formulation) is not str or self.formulation not in _FORMULATIONS:
            raise ValueError("formulation must be exact 'forward' or 'receiver'")
        _exact_positive_int(self.annulus_count, "annulus_count")
        _exact_even_order(self.rho_order, "rho_order", _MAXIMUM_ORDER)
        _exact_even_order(self.mu_order, "mu_order", _MAXIMUM_ORDER)
        _exact_even_order(self.psi_count, "psi_count", _MAXIMUM_PSI_COUNT)
        chunk = _exact_positive_int(
            self.directions_per_task,
            "directions_per_task",
        )
        if chunk > _MAXIMUM_DIRECTIONS_PER_TASK:
            raise ValueError(
                "directions_per_task exceeds the hard cache-layout maximum"
            )
        if self.task_count > _MAXIMUM_TASK_COUNT:
            raise ValueError("task plan exceeds the hard task-count limit")

    @property
    def scientific_plan(self) -> KerrKernelScientificPlan:
        return KerrKernelScientificPlan(
            self.formulation,
            self.annulus_count,
            self.rho_order,
            self.mu_order,
            self.psi_count,
        )

    @property
    def passes(self) -> tuple[KerrKernelGridPass, ...]:
        return self.scientific_plan.passes

    @property
    def direction_count(self) -> int:
        return self.scientific_plan.direction_count

    @property
    def expected_current_kernel_direction_count(self) -> int:
        """Closed form used by both current public kernel work budgets."""

        return (
            7
            * self.annulus_count
            * self.rho_order
            * self.mu_order
            * self.psi_count
        )

    def scientific_descriptor(self) -> dict[str, Any]:
        """Return the task topology identity, excluding storage chunking."""

        return self.scientific_plan.as_dict()

    @property
    def scientific_plan_sha256(self) -> str:
        return self.scientific_plan.descriptor_sha256

    def cache_layout_descriptor(self) -> dict[str, Any]:
        return {
            "chunkBoundary": "never crosses pass/face/annulus/rho",
            "directionsPerTaskMaximum": self.directions_per_task,
            "taskCount": self.task_count,
        }

    @property
    def task_count(self) -> int:
        return 2 * self.annulus_count * sum(
            grid_pass.rho_order
            * (
                (
                    grid_pass.angular_direction_count
                    + self.directions_per_task
                    - 1
                )
                // self.directions_per_task
            )
            for grid_pass in self.passes
        )

    def iter_task_keys(self) -> Iterator[TaskKey]:
        """Yield cache chunks in canonical scientific reduction order."""

        for pass_index, grid_pass in enumerate(self.passes):
            angular_count = grid_pass.angular_direction_count
            for face_index in range(len(_FACES)):
                for annulus_index in range(self.annulus_count):
                    y = face_index * self.annulus_count + annulus_index
                    for rho_index in range(grid_pass.rho_order):
                        radial_start = rho_index * angular_count
                        for angular_start in range(
                            0,
                            angular_count,
                            self.directions_per_task,
                        ):
                            width = min(
                                self.directions_per_task,
                                angular_count - angular_start,
                            )
                            yield TaskKey(
                                pass_index,
                                y,
                                radial_start + angular_start,
                                width,
                                1,
                            )

    def task_keys(self) -> tuple[TaskKey, ...]:
        """Materialize canonical chunks for :class:`offline.job.JobSpec`."""

        tasks = tuple(self.iter_task_keys())
        if tasks != tuple(sorted(tasks)) or len(set(tasks)) != len(tasks):
            raise KerrReturningRadiationKernelJobError(
                "kernel task planner did not produce unique canonical keys"
            )
        if sum(item.width for item in tasks) != self.direction_count:
            raise KerrReturningRadiationKernelJobError(
                "kernel task planner direction accounting is inconsistent"
            )
        return tasks

    def coordinates_for_task(
        self,
        key: TaskKey,
    ) -> tuple[KerrKernelDirectionCoordinate, ...]:
        if type(key) is not TaskKey:
            raise TypeError("key must be exact TaskKey")
        if key.height != 1 or key.sample_index >= len(self.passes):
            raise ValueError("task key does not belong to this kernel plan")
        grid_pass = self.passes[key.sample_index]
        if key.y >= 2 * self.annulus_count:
            raise ValueError("task key does not belong to this kernel plan")
        angular_count = grid_pass.angular_direction_count
        rho_index, angular_start = divmod(key.x, angular_count)
        if rho_index >= grid_pass.rho_order:
            raise ValueError("task key does not belong to this kernel plan")
        if angular_start + key.width > angular_count:
            raise ValueError("task chunk crosses a canonical radial-node boundary")
        expected_width = min(
            self.directions_per_task,
            angular_count - angular_start,
        )
        if angular_start % self.directions_per_task or key.width != expected_width:
            raise ValueError("task key is not a canonical cache-layout chunk")

        face_index, annulus_index = divmod(key.y, self.annulus_count)
        pass_offset = 2 * self.annulus_count * sum(
            item.rho_order * item.angular_direction_count
            for item in self.passes[: key.sample_index]
        )
        local_prefix = (
            (
                face_index * self.annulus_count
                + annulus_index
            )
            * grid_pass.rho_order
            * angular_count
            + rho_index * angular_count
        )
        result: list[KerrKernelDirectionCoordinate] = []
        for angular_index in range(angular_start, angular_start + key.width):
            mu_index, psi_index = divmod(angular_index, grid_pass.psi_count)
            result.append(
                KerrKernelDirectionCoordinate(
                    pass_offset + local_prefix + angular_index,
                    key.sample_index,
                    grid_pass.name,
                    face_index,
                    _FACES[face_index],
                    annulus_index,
                    rho_index,
                    mu_index,
                    psi_index,
                )
            )
        return tuple(result)


def _validated_inputs(inputs: Sequence[InputArtifact]) -> tuple[InputArtifact, ...]:
    raw = tuple(inputs)
    rebuilt: list[InputArtifact] = []
    for item in raw:
        if type(item) is not InputArtifact:
            raise TypeError("inputs must contain exact InputArtifact values")
        if (
            type(item.uri) is not str
            or type(item.byte_length) is not int
            or type(item.sha256) is not str
        ):
            raise TypeError("input artifact fields must have exact schema types")
        reconstructed = InputArtifact(item.uri, item.byte_length, item.sha256)
        if item != reconstructed:
            raise ValueError("input artifact differs from exact reconstruction")
        rebuilt.append(reconstructed)
    ordered = tuple(sorted(rebuilt))
    if len({item.uri for item in ordered}) != len(ordered):
        raise ValueError("input artifact URIs must be unique")
    return ordered


def _reserved_evaluator_artifacts(
    inputs: tuple[InputArtifact, ...],
) -> tuple[InputArtifact, ...]:
    """Return every artifact in the evaluator-reserved URI namespace."""

    return tuple(
        item
        for item in inputs
        if item.uri == _EVALUATOR_ARTIFACT_URI_ROOT
        or item.uri.startswith(f"{_EVALUATOR_ARTIFACT_URI_ROOT}:")
    )


def _reserved_evaluator_runtime_artifacts(
    inputs: tuple[InputArtifact, ...],
) -> tuple[InputArtifact, ...]:
    """Return artifacts in the explicit evaluator-runtime URI namespace."""

    return tuple(
        item
        for item in inputs
        if item.uri == _EVALUATOR_RUNTIME_ARTIFACT_URI_ROOT
        or item.uri.startswith(f"{_EVALUATOR_RUNTIME_ARTIFACT_URI_ROOT}:")
    )


def _scientific_status(
    inputs: tuple[InputArtifact, ...],
) -> dict[str, Any]:
    """Describe the evaluator trust boundary encoded by exact inputs.

    A generic definition deliberately remains usable with arbitrary callables,
    so an absent binding cannot authenticate the callable which produced or
    reused its records.  Exactly one syntactically valid v2 reserved artifact
    selects the bridge boundary whose descriptor is re-derived and compared by
    :func:`run_kernel_direction_cache` before the cache root is opened.
    """

    status = dict(SCIENTIFIC_STATUS)
    reserved = _reserved_evaluator_artifacts(inputs)
    bound: InputArtifact | None = None
    if len(reserved) == 1 and reserved[0].uri.startswith(
        _EVALUATOR_ARTIFACT_URI_PREFIX
    ):
        implementation_id = reserved[0].uri[len(_EVALUATOR_ARTIFACT_URI_PREFIX) :]
        try:
            _validated_evaluator_implementation_id(implementation_id)
        except (TypeError, ValueError):
            pass
        else:
            bound = reserved[0]

    status["reservedEvaluatorArtifactCount"] = len(reserved)
    if bound is not None:
        status.update(
            {
                "evaluatorIdentityBindingMode": (
                    "reserved-exact-function-source-code-descriptor/v2"
                ),
                "authenticatedEvaluatorIdentity": True,
                "evaluatorIdentityEvidence": (
                    "one reserved v2 descriptor artifact is bound in exact "
                    "scientific inputs and rechecked before cache access"
                ),
                "reservedEvaluatorArtifactUri": bound.uri,
                "genericUnboundCacheMayBeReusedAcrossCallerEvaluators": False,
                "evaluatorImplementationAndSourceIdentityBoundAsScientificInput": True,
                "reservedEvaluatorCallableDescriptorCheckedBeforeCacheAccess": True,
            }
        )
    elif reserved:
        status.update(
            {
                "evaluatorIdentityBindingMode": "invalid-reserved-artifact-set",
                "evaluatorIdentityEvidence": (
                    "reserved evaluator artifacts are not exactly one supported "
                    "v2 binding; execution rejects them before cache access"
                ),
                "genericUnboundCacheMayBeReusedAcrossCallerEvaluators": False,
            }
        )
    return status


def _rebuilt_identity(
    identity: KerrKernelScientificIdentity,
) -> KerrKernelScientificIdentity:
    if type(identity) is not KerrKernelScientificIdentity:
        raise TypeError("identity must be exact KerrKernelScientificIdentity")
    rebuilt = KerrKernelScientificIdentity(
        identity.formulation,
        identity.surface,
        identity.termination,
        identity.annulus_edges_over_mass,
        identity.fine_ray_options,
        identity.fine_surface_options,
        identity.coarse_ray_options,
        identity.coarse_surface_options,
        identity.source_closure_sha256,
    )
    _require_exact_tree(identity, rebuilt, "scientific_context.identity")
    return rebuilt


def _scientific_document(
    plan: KerrKernelScientificPlan,
    identity: KerrKernelScientificIdentity,
    inputs: tuple[InputArtifact, ...],
) -> dict[str, Any]:
    return {
        "algorithmVersion": ALGORITHM_VERSION,
        "inputs": tuple(item.as_dict() for item in inputs),
        "plan": plan.as_dict(),
        "producer": PRODUCER_ID,
        "producerSourceHashes": identity.source_closure_sha256,
        "schema": SCIENTIFIC_JOB_SCHEMA,
        "scientificStatus": _scientific_status(inputs),
        "scientificIdentity": identity.as_dict(),
    }


@dataclass(frozen=True, slots=True)
class KerrKernelScientificContext:
    """Authenticated layout-free context passed to every evaluator call."""

    plan: KerrKernelScientificPlan
    identity: KerrKernelScientificIdentity
    inputs: tuple[InputArtifact, ...]
    scientific_job_key: str
    evaluator_runtime_binding: KerrKernelEvaluatorRuntimeBinding | None = None

    def __post_init__(self) -> None:
        if type(self.plan) is not KerrKernelScientificPlan:
            raise TypeError("context plan must be exact KerrKernelScientificPlan")
        rebuilt_plan = KerrKernelScientificPlan(
            self.plan.formulation,
            self.plan.annulus_count,
            self.plan.rho_order,
            self.plan.mu_order,
            self.plan.psi_count,
        )
        _require_exact_tree(self.plan, rebuilt_plan, "scientific_context.plan")
        identity = _rebuilt_identity(self.identity)
        inputs = _validated_inputs(self.inputs)
        if type(self.inputs) is not tuple or self.inputs != inputs:
            raise TypeError("context inputs must be a canonical exact tuple")
        if identity.formulation != rebuilt_plan.formulation:
            raise ValueError("identity and scientific plan formulations differ")
        if identity.annulus_count != rebuilt_plan.annulus_count:
            raise ValueError("identity edges and scientific plan annuli differ")
        key = _lowercase_sha256(self.scientific_job_key, "scientific_job_key")
        expected = _canonical_sha256(
            _scientific_document(rebuilt_plan, identity, inputs)
        )
        if key.encode("ascii") != expected.encode("ascii"):
            raise ValueError("scientific job key differs from exact reconstruction")
        runtime_binding = _validated_evaluator_runtime_binding_for_inputs(
            self.evaluator_runtime_binding,
            inputs,
        )
        object.__setattr__(self, "plan", rebuilt_plan)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "evaluator_runtime_binding", runtime_binding)

    @property
    def scientific_plan_sha256(self) -> str:
        return self.plan.descriptor_sha256

    def scientific_document(self) -> dict[str, Any]:
        return _scientific_document(self.plan, self.identity, self.inputs)


def _make_scientific_context(
    plan: KerrKernelScientificPlan,
    identity: KerrKernelScientificIdentity,
    inputs: Sequence[InputArtifact],
    evaluator_runtime_binding: KerrKernelEvaluatorRuntimeBinding | None = None,
) -> KerrKernelScientificContext:
    checked_inputs = _validated_inputs(inputs)
    key = _canonical_sha256(_scientific_document(plan, identity, checked_inputs))
    return KerrKernelScientificContext(
        plan,
        identity,
        checked_inputs,
        key,
        evaluator_runtime_binding,
    )


def _maximum_minimum_record_payload_bytes(
    plan: KerrKernelDirectionTaskPlan,
    context: KerrKernelScientificContext,
) -> int:
    """Conservative canonical size with the smallest legal ``transport={}``."""

    maximum_width = max(
        min(plan.directions_per_task, item.angular_direction_count)
        for item in plan.passes
    )
    worst_coordinate = KerrKernelDirectionCoordinate(
        plan.direction_count - 1,
        len(_PASS_NAMES) - 1,
        _PASS_NAMES[-1],
        len(_FACES) - 1,
        _FACES[-1],
        plan.annulus_count - 1,
        plan.rho_order - 1,
        plan.mu_order - 1,
        plan.psi_count - 1,
    )
    record_bytes = canonical_json_bytes(
        {
            "coordinate": worst_coordinate.as_dict(),
            "transport": {},
        }
    )
    record_length = len(record_bytes) - 1
    maximum_angular = max(item.angular_direction_count for item in plan.passes)
    maximum_x = plan.rho_order * maximum_angular - 1
    worst_task = TaskKey(
        len(_PASS_NAMES) - 1,
        2 * plan.annulus_count - 1,
        maximum_x,
        maximum_width,
        1,
    )
    empty_envelope = canonical_json_bytes(
        {
            "cacheJobKey": "0" * 64,
            "records": [],
            "schema": TASK_PAYLOAD_SCHEMA,
            "scientificJobKey": context.scientific_job_key,
            "scientificPlanSha256": context.scientific_plan_sha256,
            "task": worst_task.as_dict(),
        }
    )
    # ``[]`` in the empty envelope is replaced by N canonical records and
    # N-1 commas.  The trailing newline is already present in the envelope.
    return (
        len(empty_envelope)
        + maximum_width * record_length
        + max(0, maximum_width - 1)
    )


def _expected_job_spec(
    plan: KerrKernelDirectionTaskPlan,
    context: KerrKernelScientificContext,
) -> JobSpec:
    minimum_record_payload = _maximum_minimum_record_payload_bytes(plan, context)
    if minimum_record_payload > _MAXIMUM_TASK_PAYLOAD_BYTES:
        raise ValueError(
            "directions_per_task cannot fit even canonical transport={} records "
            "inside the hard task payload limit"
        )
    parameters = {
        "cacheLayout": plan.cache_layout_descriptor(),
        "payloadBudget": {
            "maximumJsonDepth": _MAXIMUM_JSON_DEPTH,
            "maximumMinimumRecordPayloadBytes": minimum_record_payload,
            "maximumTaskPayloadBytes": _MAXIMUM_TASK_PAYLOAD_BYTES,
            "maximumTransportJsonBytes": _MAXIMUM_TRANSPORT_JSON_BYTES,
        },
        "scientificStatus": _scientific_status(context.inputs),
        "scientificDocument": context.scientific_document(),
        "scientificJobKey": context.scientific_job_key,
        "scientificPlanSha256": context.scientific_plan_sha256,
    }
    return JobSpec(
        producer=PRODUCER_ID,
        algorithm_version=ALGORITHM_VERSION,
        tasks=plan.task_keys(),
        parameters=parameters,
        inputs=context.inputs,
        producer_source_hashes=context.identity.source_closure_sha256,
        record_bytes=1,
    )


@dataclass(frozen=True, slots=True)
class KerrKernelDirectionCacheDefinition:
    """A cache JobSpec authenticated against a layout-free context."""

    plan: KerrKernelDirectionTaskPlan
    scientific_context: KerrKernelScientificContext
    job_spec: JobSpec

    def __post_init__(self) -> None:
        if type(self.plan) is not KerrKernelDirectionTaskPlan:
            raise TypeError("plan must be exact KerrKernelDirectionTaskPlan")
        rebuilt_plan = KerrKernelDirectionTaskPlan(
            self.plan.formulation,
            self.plan.annulus_count,
            self.plan.rho_order,
            self.plan.mu_order,
            self.plan.psi_count,
            self.plan.directions_per_task,
        )
        _require_exact_tree(self.plan, rebuilt_plan, "cache_definition.plan")
        if type(self.scientific_context) is not KerrKernelScientificContext:
            raise TypeError("scientific_context must have its exact type")
        context = KerrKernelScientificContext(
            self.scientific_context.plan,
            self.scientific_context.identity,
            self.scientific_context.inputs,
            self.scientific_context.scientific_job_key,
            self.scientific_context.evaluator_runtime_binding,
        )
        _require_exact_tree(
            self.scientific_context,
            context,
            "cache_definition.scientific_context",
        )
        if rebuilt_plan.scientific_plan != context.plan:
            raise ValueError("cache layout plan differs from scientific context")
        if type(self.job_spec) is not JobSpec:
            raise TypeError("job_spec must be exact JobSpec")
        expected = _expected_job_spec(rebuilt_plan, context)
        _require_exact_tree(self.job_spec, expected, "cache_definition.job_spec")
        if (
            canonical_json_bytes(self.job_spec.as_dict())
            != canonical_json_bytes(expected.as_dict())
            or self.job_spec.job_key.encode("ascii")
            != expected.job_key.encode("ascii")
        ):
            raise ValueError(
                "JobSpec producer/version/parameters/tasks differ from reconstruction"
            )
        if len(canonical_json_bytes(expected.as_dict())) > _MAXIMUM_JOB_SPEC_BYTES:
            raise ValueError("JobSpec exceeds the hard metadata byte limit")
        object.__setattr__(self, "plan", rebuilt_plan)
        object.__setattr__(self, "scientific_context", context)
        object.__setattr__(self, "job_spec", expected)

    @property
    def scientific_job_key(self) -> str:
        return self.scientific_context.scientific_job_key


def make_kernel_direction_cache_definition(
    plan: KerrKernelDirectionTaskPlan,
    *,
    identity: KerrKernelScientificIdentity,
    inputs: Sequence[InputArtifact] = (),
    evaluator_runtime_binding: KerrKernelEvaluatorRuntimeBinding | None = None,
) -> KerrKernelDirectionCacheDefinition:
    """Build a fail-closed cache definition from a strong scientific identity."""

    if type(plan) is not KerrKernelDirectionTaskPlan:
        raise TypeError("plan must be exact KerrKernelDirectionTaskPlan")
    identity = _rebuilt_identity(identity)
    context = _make_scientific_context(
        plan.scientific_plan,
        identity,
        inputs,
        evaluator_runtime_binding,
    )
    spec = _expected_job_spec(plan, context)
    return KerrKernelDirectionCacheDefinition(plan, context, spec)


KernelDirectionEvaluator = Callable[
    [KerrKernelScientificContext, KerrKernelDirectionCoordinate],
    Mapping[str, Any],
]


def _validated_evaluator_implementation_id(value: Any) -> str:
    if type(value) is not str or not value:
        raise TypeError("evaluator implementation_id must be a non-empty exact str")
    encoded = value.encode("utf-8")
    if len(encoded) > _MAXIMUM_EVALUATOR_IMPLEMENTATION_ID_BYTES or any(
        ord(character) > 0x7F
        or not (character.isalnum() or character in "._/-")
        for character in value
    ):
        raise ValueError(
            "evaluator implementation_id must use bounded URI-safe ASCII"
        )
    return value


def _exact_runtime_library_path(value: Any) -> Path:
    """Return one canonical absolute regular-file locator without following links."""

    if type(value) is not _PATH_TYPE or not value.is_absolute():
        raise TypeError("evaluator runtime library must be an exact absolute Path")
    path = Path(os.path.abspath(os.fspath(value)))
    if value != path or path == path.parent:
        raise ValueError("evaluator runtime library path is not canonical")
    for ancestor in path.parents:
        try:
            snapshot = os.lstat(ancestor)
        except OSError as error:
            raise ValueError(
                "evaluator runtime library has an unreadable ancestor"
            ) from error
        if stat.S_ISLNK(snapshot.st_mode) or not stat.S_ISDIR(snapshot.st_mode):
            raise ValueError(
                "evaluator runtime library has a symlink or non-directory ancestor"
            )
    return path


def _authenticated_evaluator_runtime_library(
    path: Path,
) -> tuple[Path, int, str]:
    """Authenticate one bounded dylib locator through a stable no-follow fd."""

    exact = _exact_runtime_library_path(path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int or nofollow == 0:
        raise ValueError("evaluator runtime authentication requires O_NOFOLLOW")
    descriptor = -1
    try:
        path_before = os.lstat(exact)
        if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(
            path_before.st_mode
        ):
            raise ValueError(
                "evaluator runtime library must be a regular non-symlink file"
            )
        descriptor = os.open(
            exact,
            os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0),
        )
        opened_before = os.fstat(descriptor)
        before_identity = _evaluator_source_stat_identity(opened_before)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or before_identity != _evaluator_source_stat_identity(path_before)
        ):
            raise ValueError("evaluator runtime library changed while opened")
        if opened_before.st_size > _MAXIMUM_EVALUATOR_RUNTIME_LIBRARY_BYTES:
            raise ValueError("evaluator runtime library exceeds its byte limit")
        digest = hashlib.sha256()
        byte_length = 0
        while True:
            remaining_with_sentinel = (
                _MAXIMUM_EVALUATOR_RUNTIME_LIBRARY_BYTES - byte_length + 1
            )
            request_bytes = min(
                _EVALUATOR_SOURCE_READ_CHUNK_BYTES,
                remaining_with_sentinel,
            )
            block = os.read(descriptor, request_bytes)
            if not block:
                break
            byte_length += len(block)
            if byte_length > _MAXIMUM_EVALUATOR_RUNTIME_LIBRARY_BYTES:
                raise ValueError("evaluator runtime library exceeded its byte limit")
            digest.update(block)
        opened_after = os.fstat(descriptor)
        path_after = os.lstat(exact)
        after_identity = _evaluator_source_stat_identity(opened_after)
        if (
            not stat.S_ISREG(opened_after.st_mode)
            or stat.S_ISLNK(path_after.st_mode)
            or not stat.S_ISREG(path_after.st_mode)
            or before_identity != after_identity
            or after_identity != _evaluator_source_stat_identity(path_after)
            or byte_length != opened_after.st_size
        ):
            raise ValueError(
                "evaluator runtime library changed while authenticated"
            )
        return exact, byte_length, digest.hexdigest()
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"cannot authenticate evaluator runtime library: {error}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _json_contains_absolute_path(value: Any) -> bool:
    if type(value) is str:
        return os.path.isabs(value)
    if type(value) is dict:
        return any(
            _json_contains_absolute_path(key)
            or _json_contains_absolute_path(item)
            for key, item in value.items()
        )
    if type(value) in (list, tuple):
        return any(_json_contains_absolute_path(item) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class KerrKernelEvaluatorRuntimeBinding:
    """Pickle-safe operational locator bound to path-free scientific evidence."""

    evaluator_implementation_id: str
    library_path: Path
    descriptor_json: bytes
    descriptor_input: InputArtifact
    library_byte_length: int
    library_sha256: str
    segment_capacity: int
    crossing_capacity: int

    def __post_init__(self) -> None:
        implementation_id = _validated_evaluator_implementation_id(
            self.evaluator_implementation_id
        )
        path, byte_length, sha256 = _authenticated_evaluator_runtime_library(
            self.library_path
        )
        if type(self.descriptor_json) is not bytes or len(
            self.descriptor_json
        ) > _MAXIMUM_EVALUATOR_RUNTIME_DESCRIPTOR_BYTES:
            raise TypeError("evaluator runtime descriptor must be bounded exact bytes")
        document = _strict_json_payload(
            self.descriptor_json,
            "evaluator runtime descriptor",
            _MAXIMUM_EVALUATOR_RUNTIME_DESCRIPTOR_BYTES,
        )
        if type(document) is not dict or set(document) != {
            "backendDescriptor",
            "capacities",
            "evaluatorImplementationId",
            "library",
            "schema",
        }:
            raise ValueError("evaluator runtime descriptor has a non-exact schema")
        if _json_contains_absolute_path(document):
            raise ValueError("evaluator runtime descriptor contains an absolute path")
        if document["schema"] != EVALUATOR_RUNTIME_DESCRIPTOR_SCHEMA:
            raise ValueError("evaluator runtime descriptor schema is unsupported")
        if document["evaluatorImplementationId"] != implementation_id:
            raise ValueError("evaluator runtime descriptor identity differs")
        capacities = document["capacities"]
        if type(capacities) is not dict or set(capacities) != {
            "crossings",
            "segments",
        }:
            raise ValueError("evaluator runtime capacities have a non-exact schema")
        for name, value in (
            ("segment_capacity", self.segment_capacity),
            ("crossing_capacity", self.crossing_capacity),
        ):
            if (
                type(value) is not int
                or value < 0
                or value > _MAXIMUM_EVALUATOR_RUNTIME_CAPACITY
            ):
                raise ValueError(
                    f"{name} must be an exact bounded non-negative integer"
                )
        if capacities != {
            "crossings": self.crossing_capacity,
            "segments": self.segment_capacity,
        }:
            raise ValueError("evaluator runtime capacity evidence differs")
        library = document["library"]
        if type(library) is not dict or set(library) != {
            "artifactName",
            "byteLength",
            "sha256",
        }:
            raise ValueError("evaluator runtime library has a non-exact schema")
        if (
            library["artifactName"] != _EVALUATOR_RUNTIME_LIBRARY_ARTIFACT_NAME
            or type(self.library_byte_length) is not int
            or self.library_byte_length != byte_length
            or library["byteLength"] != byte_length
            or _lowercase_sha256(
                self.library_sha256,
                "evaluator runtime library sha256",
            )
            != sha256
            or library["sha256"] != sha256
        ):
            raise ValueError("evaluator runtime library evidence differs")
        if type(document["backendDescriptor"]) is not dict:
            raise TypeError("evaluator runtime backend descriptor must be an exact dict")
        if type(self.descriptor_input) is not InputArtifact:
            raise TypeError("evaluator runtime descriptor input must be exact")
        expected_input = InputArtifact(
            f"{_EVALUATOR_RUNTIME_ARTIFACT_URI_PREFIX}{implementation_id}",
            len(self.descriptor_json),
            hashlib.sha256(self.descriptor_json).hexdigest(),
        )
        if self.descriptor_input != expected_input:
            raise ValueError("evaluator runtime descriptor input differs")
        object.__setattr__(self, "evaluator_implementation_id", implementation_id)
        object.__setattr__(self, "library_path", path)

    @property
    def backend_descriptor_sha256(self) -> str:
        document = _strict_json_payload(
            self.descriptor_json,
            "evaluator runtime descriptor",
            _MAXIMUM_EVALUATOR_RUNTIME_DESCRIPTOR_BYTES,
        )
        return hashlib.sha256(
            canonical_json_bytes(document["backendDescriptor"])
        ).hexdigest()

    def path_free_descriptor(self) -> dict[str, Any]:
        """Match the backend-owned spawn-safe runtime-binding protocol."""

        return {
            "backendDescriptorSha256": self.backend_descriptor_sha256,
            "callerOwnedCapacities": {
                "crossings": self.crossing_capacity,
                "segments": self.segment_capacity,
            },
            "implementationId": "strict-kerr-cpu-process-runtime-binding/v1",
        }


def make_kernel_direction_evaluator_runtime_binding(
    *,
    evaluator_implementation_id: str,
    library_path: Path,
    backend_descriptor: Mapping[str, Any],
    segment_capacity: int,
    crossing_capacity: int,
) -> KerrKernelEvaluatorRuntimeBinding:
    """Bind an explicit dylib locator without serializing its absolute path."""

    implementation_id = _validated_evaluator_implementation_id(
        evaluator_implementation_id
    )
    if type(backend_descriptor) is not dict:
        raise TypeError("backend_descriptor must be an exact dict")
    backend_document = _strict_json_copy(
        backend_descriptor,
        "$evaluatorRuntime.backendDescriptor",
    )
    if type(backend_document) is not dict:
        raise TypeError("backend_descriptor must be an exact JSON object")
    if _json_contains_absolute_path(backend_document):
        raise ValueError("backend_descriptor contains an absolute path")
    path, byte_length, sha256 = _authenticated_evaluator_runtime_library(
        library_path
    )
    for name, value in (
        ("segment_capacity", segment_capacity),
        ("crossing_capacity", crossing_capacity),
    ):
        if (
            type(value) is not int
            or value < 0
            or value > _MAXIMUM_EVALUATOR_RUNTIME_CAPACITY
        ):
            raise ValueError(
                f"{name} must be an exact bounded non-negative integer"
            )
    document = {
        "backendDescriptor": backend_document,
        "capacities": {
            "crossings": crossing_capacity,
            "segments": segment_capacity,
        },
        "evaluatorImplementationId": implementation_id,
        "library": {
            "artifactName": _EVALUATOR_RUNTIME_LIBRARY_ARTIFACT_NAME,
            "byteLength": byte_length,
            "sha256": sha256,
        },
        "schema": EVALUATOR_RUNTIME_DESCRIPTOR_SCHEMA,
    }
    payload = canonical_json_bytes(document)
    if len(payload) > _MAXIMUM_EVALUATOR_RUNTIME_DESCRIPTOR_BYTES:
        raise ValueError("evaluator runtime descriptor exceeds its byte limit")
    descriptor_input = InputArtifact(
        f"{_EVALUATOR_RUNTIME_ARTIFACT_URI_PREFIX}{implementation_id}",
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )
    return KerrKernelEvaluatorRuntimeBinding(
        implementation_id,
        path,
        payload,
        descriptor_input,
        byte_length,
        sha256,
        segment_capacity,
        crossing_capacity,
    )


def revalidate_kernel_direction_evaluator_runtime_binding(
    binding: KerrKernelEvaluatorRuntimeBinding,
) -> KerrKernelEvaluatorRuntimeBinding:
    """Re-hash the explicit dylib and return the exact unchanged binding."""

    if type(binding) is not KerrKernelEvaluatorRuntimeBinding:
        raise TypeError("evaluator runtime binding must have its exact type")
    rebuilt = KerrKernelEvaluatorRuntimeBinding(
        binding.evaluator_implementation_id,
        binding.library_path,
        binding.descriptor_json,
        binding.descriptor_input,
        binding.library_byte_length,
        binding.library_sha256,
        binding.segment_capacity,
        binding.crossing_capacity,
    )
    if rebuilt != binding:
        raise ValueError("evaluator runtime binding differs from reconstruction")
    return rebuilt


def kernel_direction_evaluator_process_local_key(
    binding: KerrKernelEvaluatorRuntimeBinding,
) -> tuple[int, str, str]:
    """Return the PID-scoped key a bridge may use for backend construction."""

    checked = revalidate_kernel_direction_evaluator_runtime_binding(binding)
    return (
        os.getpid(),
        os.fspath(checked.library_path),
        checked.descriptor_input.sha256,
    )


def _validated_evaluator_runtime_binding_for_inputs(
    binding: KerrKernelEvaluatorRuntimeBinding | None,
    inputs: tuple[InputArtifact, ...],
) -> KerrKernelEvaluatorRuntimeBinding | None:
    runtime_artifacts = _reserved_evaluator_runtime_artifacts(inputs)
    if binding is None:
        if runtime_artifacts:
            raise ValueError(
                "evaluator runtime artifacts require an explicit runtime binding"
            )
        return None
    checked = revalidate_kernel_direction_evaluator_runtime_binding(binding)
    if runtime_artifacts != (checked.descriptor_input,):
        raise ValueError(
            "evaluator runtime binding differs from exact scientific inputs"
        )
    evaluator_artifacts = _reserved_evaluator_artifacts(inputs)
    expected_evaluator_uri = (
        f"{_EVALUATOR_ARTIFACT_URI_PREFIX}"
        f"{checked.evaluator_implementation_id}"
    )
    if (
        len(evaluator_artifacts) != 1
        or evaluator_artifacts[0].uri != expected_evaluator_uri
    ):
        raise ValueError(
            "evaluator runtime binding lacks its matching evaluator input"
        )
    return checked


def _normalized_evaluator_code(
    code: CodeType,
    logical_path: str,
) -> CodeType:
    if type(code) is not CodeType:
        raise TypeError("evaluator code must have the exact CodeType")
    constants = tuple(
        _normalized_evaluator_code(item, logical_path)
        if type(item) is CodeType
        else item
        for item in code.co_consts
    )
    return code.replace(co_consts=constants, co_filename=logical_path)


def _evaluator_source_stat_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _authenticated_evaluator_source_sha256(path: Path) -> tuple[int, str]:
    """Hash one stable, bounded, regular evaluator source through one fd."""

    if type(path) is not _PATH_TYPE:
        raise TypeError("evaluator source path must be an exact platform Path")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int or nofollow == 0:
        raise ValueError("evaluator source authentication requires O_NOFOLLOW")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    descriptor = -1
    try:
        path_before = os.lstat(path)
        if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(
            path_before.st_mode
        ):
            raise ValueError("evaluator source is not a regular non-symlink file")
        descriptor = os.open(path, flags)
        opened_before = os.fstat(descriptor)
        before_identity = _evaluator_source_stat_identity(opened_before)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or before_identity != _evaluator_source_stat_identity(path_before)
        ):
            raise ValueError("evaluator source changed while it was opened")
        if opened_before.st_size > _MAXIMUM_EVALUATOR_SOURCE_BYTES:
            raise ValueError("evaluator source exceeds its fixed byte limit")

        digest = hashlib.sha256()
        byte_length = 0
        while True:
            remaining_with_sentinel = (
                _MAXIMUM_EVALUATOR_SOURCE_BYTES - byte_length + 1
            )
            request_bytes = min(
                _EVALUATOR_SOURCE_READ_CHUNK_BYTES,
                remaining_with_sentinel,
            )
            block = os.read(descriptor, request_bytes)
            if not block:
                break
            byte_length += len(block)
            if byte_length > _MAXIMUM_EVALUATOR_SOURCE_BYTES:
                raise ValueError(
                    "evaluator source grew beyond its fixed byte limit"
                )
            digest.update(block)

        opened_after = os.fstat(descriptor)
        path_after = os.lstat(path)
        after_identity = _evaluator_source_stat_identity(opened_after)
        if (
            not stat.S_ISREG(opened_after.st_mode)
            or stat.S_ISLNK(path_after.st_mode)
            or not stat.S_ISREG(path_after.st_mode)
            or before_identity != after_identity
            or after_identity != _evaluator_source_stat_identity(path_after)
            or byte_length != opened_after.st_size
        ):
            raise ValueError("evaluator source changed while it was authenticated")
        return byte_length, digest.hexdigest()
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"cannot authenticate evaluator source: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _evaluator_callable_descriptor(
    evaluator: Any,
    implementation_id: str,
) -> dict[str, Any]:
    """Describe a closure-free source function without trusting custom hooks."""

    implementation_id = _validated_evaluator_implementation_id(
        implementation_id
    )
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
    forbidden_flags = 0x04 | 0x08 | 0x20 | 0x80 | 0x200
    if (
        code.co_argcount != 2
        or code.co_posonlyargcount != 0
        or code.co_kwonlyargcount != 0
        or code.co_flags & forbidden_flags
    ):
        raise ValueError(
            "reserved evaluator must have exactly two positional parameters and "
            "cannot be variadic, a generator, or a coroutine"
        )
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
    source_root = Path(__file__).resolve().parents[1]
    try:
        source_path = Path(os.path.abspath(filename))
        logical_path = source_path.relative_to(source_root).as_posix()
        source_byte_length, source_sha256 = _authenticated_evaluator_source_sha256(
            source_path
        )
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            "reserved evaluator source must be a readable project file"
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


def make_kernel_direction_evaluator_input(
    evaluator: KernelDirectionEvaluator,
    *,
    implementation_id: str,
) -> InputArtifact:
    """Bind one exact callable descriptor into a reserved cache input.

    The binding prevents accidental substitution through the public jobs API.
    It is not a sandbox and does not defend against malicious code already
    executing in the same Python process.
    """

    implementation_id = _validated_evaluator_implementation_id(
        implementation_id
    )
    payload = canonical_json_bytes(
        _evaluator_callable_descriptor(evaluator, implementation_id)
    )
    return InputArtifact(
        f"{_EVALUATOR_ARTIFACT_URI_PREFIX}{implementation_id}",
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def _validate_reserved_evaluator_binding(
    inputs: tuple[InputArtifact, ...],
    evaluator: Any,
) -> None:
    reserved = _reserved_evaluator_artifacts(inputs)
    if not reserved:
        return
    if len(reserved) != 1:
        raise KerrReturningRadiationKernelJobError(
            "cache definition must bind exactly one reserved evaluator"
        )
    expected = reserved[0]
    if not expected.uri.startswith(_EVALUATOR_ARTIFACT_URI_PREFIX):
        raise KerrReturningRadiationKernelJobError(
            "cache definition uses an unsupported reserved evaluator artifact"
        )
    implementation_id = expected.uri[len(_EVALUATOR_ARTIFACT_URI_PREFIX) :]
    try:
        actual = make_kernel_direction_evaluator_input(
            evaluator,
            implementation_id=implementation_id,
        )
    except (OSError, TypeError, ValueError) as error:
        raise KerrReturningRadiationKernelJobError(
            "reserved evaluator callable descriptor could not be authenticated: "
            f"{error}"
        ) from error
    if (
        type(expected.uri) is not str
        or actual.uri.encode("utf-8") != expected.uri.encode("utf-8")
        or type(expected.byte_length) is not int
        or actual.byte_length != expected.byte_length
        or type(expected.sha256) is not str
        or actual.sha256.encode("ascii") != expected.sha256.encode("ascii")
    ):
        raise KerrReturningRadiationKernelJobError(
            "reserved evaluator callable descriptor differs from the definition"
        )


@dataclass(frozen=True, slots=True)
class _KernelDirectionTaskProducer:
    plan: KerrKernelDirectionTaskPlan
    scientific_context: KerrKernelScientificContext
    cache_job_key: str
    evaluator: KernelDirectionEvaluator

    def __call__(self, key: TaskKey) -> bytes:
        _lowercase_sha256(self.cache_job_key, "cache_job_key")
        coordinates = self.plan.coordinates_for_task(key)
        context = KerrKernelScientificContext(
            self.scientific_context.plan,
            self.scientific_context.identity,
            self.scientific_context.inputs,
            self.scientific_context.scientific_job_key,
            self.scientific_context.evaluator_runtime_binding,
        )
        _validate_reserved_evaluator_binding(
            context.inputs,
            self.evaluator,
        )
        runtime_binding = context.evaluator_runtime_binding
        if runtime_binding is not None:
            revalidate_kernel_direction_evaluator_runtime_binding(runtime_binding)
        records: list[dict[str, Any]] = []
        transport_bytes_consumed = 0
        for coordinate in coordinates:
            evaluated = self.evaluator(
                context,
                coordinate,
            )
            if type(evaluated) is not dict:
                raise TypeError("direction evaluator must return an exact dict")
            transport = _strict_json_copy(evaluated)
            transport_bytes = canonical_json_bytes(transport)
            if len(transport_bytes) > _MAXIMUM_TRANSPORT_JSON_BYTES:
                raise ValueError(
                    "direction transport exceeds the hard per-record byte limit"
                )
            transport_bytes_consumed += len(transport_bytes)
            if transport_bytes_consumed > _MAXIMUM_TASK_PAYLOAD_BYTES:
                raise ValueError(
                    "direction transports exceed the hard task payload limit"
                )
            records.append(
                {
                    "coordinate": coordinate.as_dict(),
                    "transport": transport,
                }
            )
        if runtime_binding is not None:
            revalidate_kernel_direction_evaluator_runtime_binding(runtime_binding)
        _validate_reserved_evaluator_binding(
            context.inputs,
            self.evaluator,
        )
        KerrKernelScientificContext(
            context.plan,
            context.identity,
            context.inputs,
            context.scientific_job_key,
            context.evaluator_runtime_binding,
        )
        document = {
            "cacheJobKey": self.cache_job_key,
            "records": records,
            "schema": TASK_PAYLOAD_SCHEMA,
            "scientificJobKey": context.scientific_job_key,
            "scientificPlanSha256": context.scientific_plan_sha256,
            "task": key.as_dict(),
        }
        payload = canonical_json_bytes(document)
        if len(payload) > _MAXIMUM_TASK_PAYLOAD_BYTES:
            raise ValueError("direction task exceeds the hard payload byte limit")
        _validate_json_nesting(payload, "generated direction task")
        return payload


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
        raise KerrReturningRadiationKernelJobError(
            "platform lacks O_NOFOLLOW/O_DIRECTORY secure cache primitives"
        )
    return os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)


def _open_absolute_cache_root(
    path: Path,
    *,
    create_final: bool,
) -> tuple[Path, int]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.is_absolute() or absolute == absolute.parent:
        raise KerrReturningRadiationKernelJobError(
            "cache_root must name a non-root absolute directory"
        )
    components = absolute.parts[1:]
    flags = _directory_open_flags()
    descriptor = os.open(os.sep, flags)
    try:
        for index, component in enumerate(components):
            if component in ("", ".", "..") or os.sep in component:
                raise KerrReturningRadiationKernelJobError(
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
                    raise KerrReturningRadiationKernelJobError(
                        "cache_root absolute path contains a symlink, missing "
                        "ancestor, or non-directory component"
                    ) from error
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                    following = os.open(component, flags, dir_fd=descriptor)
                except OSError as create_error:
                    raise KerrReturningRadiationKernelJobError(
                        "cache_root final directory cannot be created securely"
                    ) from create_error
            os.close(descriptor)
            descriptor = following
        return absolute, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_or_create_directory_at(parent_fd: int, name: str, label: str) -> int:
    if type(name) is not str or not name or os.sep in name or name in (".", ".."):
        raise KerrReturningRadiationKernelJobError(f"{label} name is invalid")
    flags = _directory_open_flags()
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise KerrReturningRadiationKernelJobError(
                f"{label} is a symlink or non-directory"
            ) from error
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise KerrReturningRadiationKernelJobError(
            f"{label} cannot be created securely"
        ) from error


@contextmanager
def _secure_cache_session(
    cache_root: Path,
    job_key: str,
) -> Iterator[_SecureCacheSession]:
    absolute, cache_fd = _open_absolute_cache_root(
        cache_root,
        create_final=True,
    )
    job_fd = -1
    task_fd = -1
    try:
        job_fd = _open_or_create_directory_at(cache_fd, job_key, "cache job directory")
        task_fd = _open_or_create_directory_at(job_fd, "tasks", "cache task directory")
        yield _SecureCacheSession(
            absolute,
            absolute / job_key,
            absolute / job_key / "tasks",
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


@contextmanager
def _secure_existing_cache_session(
    cache_root: Path,
    job_key: str,
) -> Iterator[_SecureCacheSession]:
    absolute, cache_fd = _open_absolute_cache_root(
        cache_root,
        create_final=False,
    )
    job_fd = -1
    task_fd = -1
    try:
        job_fd = os.open(job_key, _directory_open_flags(), dir_fd=cache_fd)
        task_fd = os.open("tasks", _directory_open_flags(), dir_fd=job_fd)
        yield _SecureCacheSession(
            absolute,
            absolute / job_key,
            absolute / job_key / "tasks",
            cache_fd,
            job_fd,
            task_fd,
        )
    except OSError as error:
        raise KerrReturningRadiationKernelJobError(
            "existing cache layout contains a symlink or non-directory"
        ) from error
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
            raise KerrReturningRadiationKernelJobError(
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
        raise KerrReturningRadiationKernelJobError(
            f"{label} cannot be inspected"
        ) from error
    if stat.S_ISLNK(snapshot.st_mode):
        raise KerrReturningRadiationKernelJobError(f"{label} must not be a symlink")
    if not stat.S_ISREG(snapshot.st_mode):
        raise KerrReturningRadiationKernelJobError(f"{label} must be a regular file")
    return "regular"


def _write_job_document(
    session: _SecureCacheSession,
    spec: JobSpec,
) -> None:
    payload = canonical_json_bytes({"jobKey": spec.job_key, "spec": spec.as_dict()})
    if len(payload) > _MAXIMUM_JOB_SPEC_BYTES:
        raise KerrReturningRadiationKernelJobError(
            "cache job document exceeds the hard metadata byte limit"
        )
    kind = _entry_kind_at(session.job_fd, "job.json", "cache job document")
    if kind == "regular":
        existing = _read_regular_file_at(
            session.job_fd,
            "job.json",
            maximum_bytes=_MAXIMUM_JOB_SPEC_BYTES,
            label="cache job document",
        )
        if existing == payload:
            return
    _atomic_write_at(session.job_fd, "job.json", payload)


def _acquire_task_lock(session: _SecureCacheSession, key: TaskKey) -> int:
    name = f"{key.file_stem}.lock"
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=session.task_fd)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise KerrReturningRadiationKernelJobError(
                "task lock must be a regular file"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return descriptor
    except KerrReturningRadiationKernelJobError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise KerrReturningRadiationKernelJobError(
            "task lock cannot be acquired without following symlinks"
        ) from error


def _release_task_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _task_names(key: TaskKey) -> tuple[str, str]:
    return f"{key.file_stem}.bin", f"{key.file_stem}.receipt.json"


def _secure_cached_result(
    session: _SecureCacheSession,
    definition: KerrKernelDirectionCacheDefinition,
    key: TaskKey,
) -> TaskResult | None:
    payload_name, receipt_name = _task_names(key)
    payload_kind = _entry_kind_at(session.task_fd, payload_name, "task payload")
    receipt_kind = _entry_kind_at(session.task_fd, receipt_name, "task receipt")
    if payload_kind == "missing" or receipt_kind == "missing":
        return None
    payload = _read_regular_file_at(
        session.task_fd,
        payload_name,
        maximum_bytes=_MAXIMUM_TASK_PAYLOAD_BYTES,
        label="task payload",
    )
    receipt_payload = _read_regular_file_at(
        session.task_fd,
        receipt_name,
        maximum_bytes=_MAXIMUM_RECEIPT_BYTES,
        label="task receipt",
    )
    try:
        receipt = _strict_json_payload(
            receipt_payload,
            "task receipt",
            _MAXIMUM_RECEIPT_BYTES,
        )
    except KerrReturningRadiationKernelJobError:
        return None
    digest = hashlib.sha256(payload).hexdigest()
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
        or receipt["jobKey"] != definition.job_spec.job_key
        or receipt["payload"] != payload_name
        or not _matches_task_document(receipt["task"], key)
        or type(receipt["byteLength"]) is not int
        or type(receipt["recordCount"]) is not int
        or type(receipt["sha256"]) is not str
        or receipt["byteLength"] != len(payload)
        or receipt["recordCount"] != len(payload)
        or receipt["sha256"] != digest
    ):
        return None
    try:
        _validated_cached_task_records(definition, key, payload)
    except KerrReturningRadiationKernelJobError:
        return None
    return TaskResult(
        key,
        session.task_directory / payload_name,
        session.task_directory / receipt_name,
        len(payload),
        len(payload),
        digest,
        True,
    )


def _publish_task_payload(
    session: _SecureCacheSession,
    definition: KerrKernelDirectionCacheDefinition,
    key: TaskKey,
    payload: bytes,
) -> TaskResult:
    if type(payload) is not bytes or len(payload) > _MAXIMUM_TASK_PAYLOAD_BYTES:
        raise KerrReturningRadiationKernelJobError(
            "producer payload exceeds the hard task byte limit"
        )
    _validated_cached_task_records(definition, key, payload)
    payload_name, receipt_name = _task_names(key)
    digest = hashlib.sha256(payload).hexdigest()
    receipt = canonical_json_bytes(
        {
            "byteLength": len(payload),
            "jobKey": definition.job_spec.job_key,
            "payload": payload_name,
            "recordCount": len(payload),
            "schema": RECEIPT_SCHEMA,
            "sha256": digest,
            "task": key.as_dict(),
        }
    )
    if len(receipt) > _MAXIMUM_RECEIPT_BYTES:
        raise KerrReturningRadiationKernelJobError(
            "generated task receipt exceeds its hard byte limit"
        )
    _atomic_write_at(session.task_fd, payload_name, payload)
    _atomic_write_at(session.task_fd, receipt_name, receipt)
    published = _secure_cached_result(session, definition, key)
    if published is None or published.sha256 != digest:
        raise KerrReturningRadiationKernelJobError(
            "secure task publication did not authenticate"
        )
    return TaskResult(
        published.key,
        published.payload_path,
        published.receipt_path,
        published.record_count,
        published.byte_length,
        published.sha256,
        False,
    )


def _verify_secure_session_path(session: _SecureCacheSession, job_key: str) -> None:
    absolute, cache_fd = _open_absolute_cache_root(
        session.cache_root,
        create_final=False,
    )
    job_fd = -1
    task_fd = -1
    try:
        if absolute != session.cache_root:
            raise KerrReturningRadiationKernelJobError("cache_root identity changed")
        job_fd = os.open(job_key, _directory_open_flags(), dir_fd=cache_fd)
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
                raise KerrReturningRadiationKernelJobError(
                    f"{label} path changed during secure execution"
                )
    except OSError as error:
        raise KerrReturningRadiationKernelJobError(
            "cache path became a symlink or changed during secure execution"
        ) from error
    finally:
        if task_fd >= 0:
            os.close(task_fd)
        if job_fd >= 0:
            os.close(job_fd)
        os.close(cache_fd)


def _default_executor(max_workers: int) -> Executor:
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
    )


def _validated_execution_controls(
    jobs: Any,
    max_in_flight: Any,
    task_count: Any,
) -> tuple[int, int]:
    """Bound non-scientific scheduling controls before evaluator/cache access."""

    task_count = _exact_positive_int(task_count, "task_count")
    jobs = _exact_positive_int(jobs, "jobs")
    maximum_jobs = min(MAXIMUM_WORKERS, task_count)
    if jobs > maximum_jobs:
        raise ValueError(
            "jobs must not exceed min(MAXIMUM_WORKERS, task_count) "
            f"({maximum_jobs})"
        )
    maximum_in_flight = min(MAXIMUM_IN_FLIGHT_TASKS, task_count)
    if max_in_flight is None:
        selected_in_flight = min(task_count, 2 * jobs)
    else:
        selected_in_flight = _exact_positive_int(
            max_in_flight,
            "max_in_flight",
        )
        if selected_in_flight > maximum_in_flight:
            raise ValueError(
                "max_in_flight must not exceed "
                "min(MAXIMUM_IN_FLIGHT_TASKS, task_count) "
                f"({maximum_in_flight})"
            )
    return jobs, selected_in_flight


def run_kernel_direction_cache(
    definition: KerrKernelDirectionCacheDefinition,
    evaluator: KernelDirectionEvaluator,
    cache_root: Path,
    *,
    jobs: int = 1,
    max_in_flight: int | None = None,
    executor_factory: ExecutorFactory | None = None,
) -> JobRun:
    """Execute/reuse chunks through an anchored no-symlink dirfd writer."""

    if type(definition) is not KerrKernelDirectionCacheDefinition:
        raise TypeError("definition must be exact KerrKernelDirectionCacheDefinition")
    definition = KerrKernelDirectionCacheDefinition(
        definition.plan,
        definition.scientific_context,
        definition.job_spec,
    )
    jobs, max_in_flight = _validated_execution_controls(
        jobs,
        max_in_flight,
        definition.plan.task_count,
    )
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    # Reserved bridge definitions bind a canonical exact-function descriptor.
    # Authenticate it before opening the cache root, including all-hit runs.
    # Generic jobs definitions without this reserved input remain extensible.
    _validate_reserved_evaluator_binding(
        definition.scientific_context.inputs,
        evaluator,
    )
    runtime_binding = definition.scientific_context.evaluator_runtime_binding
    if runtime_binding is not None:
        revalidate_kernel_direction_evaluator_runtime_binding(runtime_binding)
    if type(cache_root) is not _PATH_TYPE:
        raise TypeError("cache_root must be an exact platform Path")

    # Recompute the canonical minimum-record envelope before any evaluator is
    # reachable, including on resumed jobs.
    minimum_payload = _maximum_minimum_record_payload_bytes(
        definition.plan,
        definition.scientific_context,
    )
    if minimum_payload > _MAXIMUM_TASK_PAYLOAD_BYTES:
        raise ValueError(
            "directions_per_task cannot fit canonical minimum records"
        )
    producer = _KernelDirectionTaskProducer(
        definition.plan,
        definition.scientific_context,
        definition.job_spec.job_key,
        evaluator,
    )
    results: list[TaskResult] = []
    maximum_observed = 0
    with _secure_cache_session(cache_root, definition.job_spec.job_key) as session:
        _write_job_document(session, definition.job_spec)
        missing: list[TaskKey] = []
        for key in definition.job_spec.tasks:
            cached = _secure_cached_result(session, definition, key)
            if cached is None:
                missing.append(key)
            else:
                results.append(cached)

        if missing and jobs == 1:
            for key in missing:
                lock_fd = _acquire_task_lock(session, key)
                try:
                    cached = _secure_cached_result(session, definition, key)
                    if cached is not None:
                        results.append(cached)
                    else:
                        maximum_observed = 1
                        payload = producer(key)
                        _verify_secure_session_path(
                            session,
                            definition.job_spec.job_key,
                        )
                        results.append(
                            _publish_task_payload(
                                session,
                                definition,
                                key,
                                payload,
                            )
                        )
                finally:
                    _release_task_lock(lock_fd)
        elif missing:
            factory = executor_factory or _default_executor
            with factory(jobs) as executor:
                iterator = iter(missing)
                pending: dict[Future[bytes], tuple[TaskKey, int]] = {}

                def fill() -> None:
                    nonlocal maximum_observed
                    while len(pending) < max_in_flight:
                        try:
                            key = next(iterator)
                        except StopIteration:
                            break
                        lock_fd = _acquire_task_lock(session, key)
                        try:
                            cached = _secure_cached_result(session, definition, key)
                        except BaseException:
                            _release_task_lock(lock_fd)
                            raise
                        if cached is not None:
                            results.append(cached)
                            _release_task_lock(lock_fd)
                            continue
                        try:
                            future = executor.submit(producer, key)
                        except BaseException:
                            _release_task_lock(lock_fd)
                            raise
                        pending[future] = (key, lock_fd)
                        maximum_observed = max(maximum_observed, len(pending))

                try:
                    fill()
                    while pending:
                        completed, _remaining = wait(
                            pending,
                            return_when=FIRST_COMPLETED,
                        )
                        ordered_completed = sorted(
                            completed,
                            key=lambda item: pending[item][0],
                        )
                        for future in ordered_completed:
                            key, lock_fd = pending.pop(future)
                            try:
                                payload = future.result()
                                _verify_secure_session_path(
                                    session,
                                    definition.job_spec.job_key,
                                )
                                results.append(
                                    _publish_task_payload(
                                        session,
                                        definition,
                                        key,
                                        payload,
                                    )
                                )
                            finally:
                                _release_task_lock(lock_fd)
                        fill()
                except BaseException:
                    for future, (_key, lock_fd) in pending.items():
                        future.cancel()
                        _release_task_lock(lock_fd)
                    raise

        _verify_secure_session_path(session, definition.job_spec.job_key)

    ordered = tuple(sorted(results, key=lambda item: item.key))
    if tuple(item.key for item in ordered) != tuple(definition.job_spec.tasks):
        raise KerrReturningRadiationKernelJobError(
            "secure runner did not produce exactly one result per task"
        )
    reused = sum(item.reused for item in ordered)
    return JobRun(
        definition.job_spec.job_key,
        ordered,
        reused,
        len(ordered) - reused,
        maximum_observed,
    )


def require_complete_existing_kernel_direction_cache(
    definition: KerrKernelDirectionCacheDefinition,
    cache_root: Path,
) -> JobRun:
    """Authenticate every existing task without creating or evaluating work."""

    if type(definition) is not KerrKernelDirectionCacheDefinition:
        raise TypeError("definition must be exact KerrKernelDirectionCacheDefinition")
    definition = KerrKernelDirectionCacheDefinition(
        definition.plan,
        definition.scientific_context,
        definition.job_spec,
    )
    runtime_binding = definition.scientific_context.evaluator_runtime_binding
    if runtime_binding is not None:
        revalidate_kernel_direction_evaluator_runtime_binding(runtime_binding)
    if type(cache_root) is not _PATH_TYPE or not cache_root.is_absolute():
        raise TypeError("cache_root must be an exact absolute platform Path")
    expected_job_payload = canonical_json_bytes(
        {
            "jobKey": definition.job_spec.job_key,
            "spec": definition.job_spec.as_dict(),
        }
    )
    results: list[TaskResult] = []
    with _secure_existing_cache_session(
        cache_root,
        definition.job_spec.job_key,
    ) as session:
        if _entry_kind_at(
            session.job_fd,
            "job.json",
            "cache job document",
        ) != "regular":
            raise KerrReturningRadiationKernelJobError(
                "complete existing cache lacks its job document"
            )
        actual_job_payload = _read_regular_file_at(
            session.job_fd,
            "job.json",
            maximum_bytes=_MAXIMUM_JOB_SPEC_BYTES,
            label="cache job document",
        )
        if actual_job_payload != expected_job_payload:
            raise KerrReturningRadiationKernelJobError(
                "existing cache job document differs from the exact definition"
            )
        for key in definition.job_spec.tasks:
            cached = _secure_cached_result(session, definition, key)
            if cached is None:
                raise KerrReturningRadiationKernelJobError(
                    "product rendering requires every kernel cache task to exist"
                )
            results.append(cached)
        _verify_secure_session_path(session, definition.job_spec.job_key)
    ordered = tuple(results)
    if tuple(item.key for item in ordered) != tuple(definition.job_spec.tasks):
        raise KerrReturningRadiationKernelJobError(
            "complete existing cache task order differs from its definition"
        )
    return JobRun(
        definition.job_spec.job_key,
        ordered,
        len(ordered),
        0,
        0,
    )


# Owner-frozen identity lets downstream cache-only consumers reject a public
# rebinding that happened before those consumers were first imported.  The
# private canonical handle remains inside the declared same-process trust
# boundary; public and owning-module rebinding do not.
_REQUIRE_COMPLETE_EXISTING_CACHE_CANONICAL_ENTRY: Final = (
    require_complete_existing_kernel_direction_cache
)


@dataclass(frozen=True, slots=True)
class KerrCachedKernelDirectionRecord:
    """One canonical cache record; transport bytes retain float spellings."""

    coordinate: KerrKernelDirectionCoordinate
    transport_json: bytes
    transport_sha256: str

    def __post_init__(self) -> None:
        if type(self.coordinate) is not KerrKernelDirectionCoordinate:
            raise TypeError("coordinate must be exact KerrKernelDirectionCoordinate")
        if type(self.transport_json) is not bytes:
            raise TypeError("transport_json must be exact bytes")
        if len(self.transport_json) > _MAXIMUM_TRANSPORT_JSON_BYTES:
            raise ValueError("transport_json exceeds the hard per-record byte limit")
        _lowercase_sha256(self.transport_sha256, "transport_sha256")
        if hashlib.sha256(self.transport_json).hexdigest() != self.transport_sha256:
            raise ValueError("transport SHA-256 does not authenticate its bytes")
        _strict_json_payload(
            self.transport_json,
            "cached direction transport",
            _MAXIMUM_TRANSPORT_JSON_BYTES,
        )

    def transport(self) -> Mapping[str, Any]:
        value = _strict_json_payload(self.transport_json)
        if type(value) is not dict:
            raise KerrReturningRadiationKernelJobError(
                "cached direction transport is not a JSON object"
            )
        return MappingProxyType(value)


def _coordinate_from_document(value: Any) -> KerrKernelDirectionCoordinate:
    if type(value) is not dict or set(value) != {
        "annulusIndex",
        "face",
        "faceIndex",
        "muIndex",
        "ordinal",
        "passIndex",
        "passName",
        "psiIndex",
        "rhoIndex",
    }:
        raise KerrReturningRadiationKernelJobError(
            "cached direction coordinate has a non-exact schema"
        )
    try:
        return KerrKernelDirectionCoordinate(
            value["ordinal"],
            value["passIndex"],
            value["passName"],
            value["faceIndex"],
            value["face"],
            value["annulusIndex"],
            value["rhoIndex"],
            value["muIndex"],
            value["psiIndex"],
        )
    except (TypeError, ValueError) as error:
        raise KerrReturningRadiationKernelJobError(
            "cached direction coordinate is invalid"
        ) from error


def _validated_cached_task_records(
    definition: KerrKernelDirectionCacheDefinition,
    key: TaskKey,
    payload: bytes,
) -> tuple[KerrCachedKernelDirectionRecord, ...]:
    """Validate one canonical task document independent of filesystem paths."""

    document = _strict_json_payload(
        payload,
        "cached direction task",
        _MAXIMUM_TASK_PAYLOAD_BYTES,
    )
    if type(document) is not dict or set(document) != {
        "cacheJobKey",
        "records",
        "schema",
        "scientificJobKey",
        "scientificPlanSha256",
        "task",
    }:
        raise KerrReturningRadiationKernelJobError(
            "cached direction task has a non-exact schema"
        )
    if (
        document["schema"] != TASK_PAYLOAD_SCHEMA
        or document["cacheJobKey"] != definition.job_spec.job_key
        or document["scientificJobKey"] != definition.scientific_job_key
        or document["scientificPlanSha256"]
        != definition.plan.scientific_plan_sha256
        or not _matches_task_document(document["task"], key)
        or type(document["records"]) is not list
    ):
        raise KerrReturningRadiationKernelJobError(
            "cached direction task identity is stale"
        )
    expected_coordinates = definition.plan.coordinates_for_task(key)
    if len(document["records"]) != len(expected_coordinates):
        raise KerrReturningRadiationKernelJobError(
            "cached direction task record count is inconsistent"
        )
    records: list[KerrCachedKernelDirectionRecord] = []
    for raw_record, expected_coordinate in zip(
        document["records"],
        expected_coordinates,
    ):
        if type(raw_record) is not dict or set(raw_record) != {
            "coordinate",
            "transport",
        }:
            raise KerrReturningRadiationKernelJobError(
                "cached direction record has a non-exact schema"
            )
        coordinate = _coordinate_from_document(raw_record["coordinate"])
        if coordinate != expected_coordinate:
            raise KerrReturningRadiationKernelJobError(
                "cached direction record is out of canonical reduction order"
            )
        transport = raw_record["transport"]
        if type(transport) is not dict:
            raise KerrReturningRadiationKernelJobError(
                "cached narrow transport record must be a JSON object"
            )
        transport_bytes = canonical_json_bytes(transport)
        if len(transport_bytes) > _MAXIMUM_TRANSPORT_JSON_BYTES:
            raise KerrReturningRadiationKernelJobError(
                "cached transport exceeds the hard per-record byte limit"
            )
        records.append(
            KerrCachedKernelDirectionRecord(
                coordinate,
                transport_bytes,
                hashlib.sha256(transport_bytes).hexdigest(),
            )
        )
    return tuple(records)


def _read_regular_file_at(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise KerrReturningRadiationKernelJobError(
            f"{label} cannot be opened without following symlinks"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise KerrReturningRadiationKernelJobError(
                f"{label} is not regular or exceeds its hard byte limit"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise KerrReturningRadiationKernelJobError(
                    f"{label} changed while being read"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise KerrReturningRadiationKernelJobError(
                f"{label} grew while being read"
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
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise KerrReturningRadiationKernelJobError(
                f"{label} changed while being read"
            )
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise KerrReturningRadiationKernelJobError(
                f"{label} stable-size check failed"
            )
        return payload
    finally:
        os.close(descriptor)


def _validated_task_payload(
    definition: KerrKernelDirectionCacheDefinition,
    result: TaskResult,
) -> bytes:
    """Re-authenticate one payload/receipt pair at reduction time."""

    payload_path = result.payload_path
    receipt_path = result.receipt_path
    if type(payload_path) is not _PATH_TYPE or type(receipt_path) is not _PATH_TYPE:
        raise KerrReturningRadiationKernelJobError(
            "cached task paths have non-exact types"
        )
    expected_payload_name = f"{result.key.file_stem}.bin"
    expected_receipt_name = f"{result.key.file_stem}.receipt.json"
    task_directory = payload_path.parent
    job_directory = task_directory.parent
    cache_root = job_directory.parent
    if (
        not payload_path.is_absolute()
        or not receipt_path.is_absolute()
        or payload_path.name != expected_payload_name
        or receipt_path.name != expected_receipt_name
        or task_directory != receipt_path.parent
        or task_directory.name != "tasks"
        or job_directory.name != definition.job_spec.job_key
    ):
        raise KerrReturningRadiationKernelJobError(
            "cached task paths do not form the canonical cache layout"
        )
    with _secure_existing_cache_session(
        cache_root,
        definition.job_spec.job_key,
    ) as session:
        if (
            session.task_directory != task_directory
            or payload_path
            != session.task_directory / expected_payload_name
            or receipt_path
            != session.task_directory / expected_receipt_name
        ):
            raise KerrReturningRadiationKernelJobError(
                "cached task paths are not the exact anchored absolute layout"
            )
        payload = _read_regular_file_at(
            session.task_fd,
            expected_payload_name,
            maximum_bytes=_MAXIMUM_TASK_PAYLOAD_BYTES,
            label="cached task payload",
        )
        receipt_payload = _read_regular_file_at(
            session.task_fd,
            expected_receipt_name,
            maximum_bytes=_MAXIMUM_RECEIPT_BYTES,
            label="cached task receipt",
        )
        _verify_secure_session_path(session, definition.job_spec.job_key)
    receipt = _strict_json_payload(
        receipt_payload,
        "cached task receipt",
        _MAXIMUM_RECEIPT_BYTES,
    )
    if type(receipt) is not dict or set(receipt) != {
        "byteLength",
        "jobKey",
        "payload",
        "recordCount",
        "schema",
        "sha256",
        "task",
    }:
        raise KerrReturningRadiationKernelJobError(
            "cached task receipt has a non-exact schema"
        )
    digest = hashlib.sha256(payload).hexdigest()
    if (
        receipt["schema"] != RECEIPT_SCHEMA
        or receipt["jobKey"] != definition.job_spec.job_key
        or receipt["payload"] != expected_payload_name
        or not _matches_task_document(receipt["task"], result.key)
        or type(receipt["byteLength"]) is not int
        or type(receipt["recordCount"]) is not int
        or type(receipt["sha256"]) is not str
        or receipt["byteLength"] != len(payload)
        or receipt["recordCount"] != len(payload)
        or receipt["sha256"] != digest
        or type(result.byte_length) is not int
        or type(result.record_count) is not int
        or type(result.sha256) is not str
        or result.byte_length != len(payload)
        or result.record_count != len(payload)
        or result.sha256 != digest
    ):
        raise KerrReturningRadiationKernelJobError(
            "cached direction payload differs from its authenticated receipt"
        )
    return payload


def iter_cached_kernel_direction_records(
    definition: KerrKernelDirectionCacheDefinition,
    job_run: JobRun,
) -> Iterator[KerrCachedKernelDirectionRecord]:
    """Yield authenticated records in canonical reduction order.

    The function re-hashes payloads at consumption time.  This closes the gap
    between task publication and a later reduction if cache bytes drift.
    """

    if type(definition) is not KerrKernelDirectionCacheDefinition:
        raise TypeError("definition must be exact KerrKernelDirectionCacheDefinition")
    definition = KerrKernelDirectionCacheDefinition(
        definition.plan,
        definition.scientific_context,
        definition.job_spec,
    )
    if type(job_run) is not JobRun:
        raise TypeError("job_run must be exact JobRun")
    if job_run.job_key != definition.job_spec.job_key:
        raise KerrReturningRadiationKernelJobError(
            "JobRun belongs to a different direction-cache job"
        )
    expected_tasks = tuple(definition.job_spec.tasks)
    if len(job_run.results) != len(expected_tasks) or any(
        type(item) is not TaskResult or type(item.key) is not TaskKey
        for item in job_run.results
    ):
        raise KerrReturningRadiationKernelJobError(
            "JobRun contains non-exact or incomplete task results"
        )
    if tuple(item.key for item in job_run.results) != expected_tasks:
        raise KerrReturningRadiationKernelJobError(
            "JobRun is incomplete or not in canonical task order"
        )
    total_bytes = 0
    for item in job_run.results:
        if type(item.byte_length) is not int or item.byte_length < 0:
            raise KerrReturningRadiationKernelJobError(
                "JobRun task byte length has a non-exact type"
            )
        total_bytes += item.byte_length
        if total_bytes > _MAXIMUM_TOTAL_CACHE_BYTES:
            raise KerrReturningRadiationKernelJobError(
                "JobRun exceeds the hard total cache byte limit"
            )

    expected_ordinal = 0
    for result, expected_task in zip(job_run.results, expected_tasks):
        if type(result) is not TaskResult or type(result.key) is not TaskKey:
            raise KerrReturningRadiationKernelJobError(
                "cached task result has a non-exact type"
            )
        if result.key != expected_task:
            raise KerrReturningRadiationKernelJobError(
                "cached task result key is not canonical"
            )
        payload = _validated_task_payload(definition, result)
        records = _validated_cached_task_records(
            definition,
            expected_task,
            payload,
        )
        for record in records:
            if record.coordinate.ordinal != expected_ordinal:
                raise KerrReturningRadiationKernelJobError(
                    "cached direction record is out of canonical reduction order"
                )
            yield record
            expected_ordinal += 1
    if expected_ordinal != definition.plan.direction_count:
        raise KerrReturningRadiationKernelJobError(
            "cached direction stream is incomplete"
        )


__all__ = (
    "ALGORITHM_VERSION",
    "EVALUATOR_DESCRIPTOR_SCHEMA",
    "EVALUATOR_RUNTIME_DESCRIPTOR_SCHEMA",
    "FORWARD",
    "MAXIMUM_IN_FLIGHT_TASKS",
    "MAXIMUM_WORKERS",
    "PRODUCER_ID",
    "RECEIVER",
    "SCIENTIFIC_STATUS",
    "SCIENTIFIC_JOB_SCHEMA",
    "TASK_PAYLOAD_SCHEMA",
    "TASK_PLAN_SCHEMA",
    "KerrCachedKernelDirectionRecord",
    "KerrKernelDirectionCacheDefinition",
    "KerrKernelDirectionCoordinate",
    "KerrKernelDirectionTaskPlan",
    "KerrKernelEvaluatorRuntimeBinding",
    "KerrKernelGridPass",
    "KerrKernelScientificContext",
    "KerrKernelScientificIdentity",
    "KerrKernelScientificPlan",
    "KerrReturningRadiationKernelJobError",
    "build_forward_kernel_scientific_identity",
    "build_receiver_kernel_scientific_identity",
    "iter_cached_kernel_direction_records",
    "kernel_direction_evaluator_process_local_key",
    "make_kernel_direction_cache_definition",
    "make_kernel_direction_evaluator_input",
    "make_kernel_direction_evaluator_runtime_binding",
    "revalidate_kernel_direction_evaluator_runtime_binding",
    "require_complete_existing_kernel_direction_cache",
    "run_kernel_direction_cache",
)
