"""Receiver-centred finite-volume returning-radiation energy kernel.

For every receiver face and annulus this module averages over the receiver's
comoving proper area and integrates its incoming local sky.  A disk-source
direction contributes

``dK[i,j] = (dA_i/A_i) (2 w_mu/N_psi)``
``            * mu_i g**4 D20(mu_e)``

to ``K[receiver_annulus][emitter_annulus]``.  This is the quadrature form of

``K = (1/pi) integral mu_i g**4 D20(mu_e) dOmega_i``.

The first source face and source radius are supplied by the separately
authenticated receiver-direction primitive.  Past-worldtube directions have
no disk source and contribute zero.  Independent fine/coarse whole rays must
place every source in the same user-declared source annulus before deposition.

This receiver construction and the repository's forward finite-volume
construction use the same exact Kerr metric, finite-thickness surface, event
layer, and geodesic integrator.  Their matrix comparison is therefore a useful
same-code-family bidirectional diagnostic, not an independent geodesic oracle
or a continuum-error theorem.  The operator is local-comoving, bolometric and
energy-only; it omits KERRBB's returning-radiation stress/work term ``F_S``,
spectral redistribution, scattering, polarization, an atmosphere solution,
GRMHD, and complete-KERRBB claims.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Final, Mapping

from offline.geodesic import RayTraceOptions, RayTraceResult, SurfaceEventOptions
from offline.kerr import (
    KerrKerrSchildMetric,
    KerrOblateTermination,
    kerr_ks_event_to_oblate,
)
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
    integrate_kerr_finite_thickness_annulus_area,
    kerr_finite_thickness_area_density,
)
from offline.kerr_finite_thickness_emitter import KerrFiniteThicknessFaceEmitter
from offline.kerr_finite_thickness_launch import KerrFiniteThicknessSurfaceFrame
from offline.kerr_finite_thickness_surface import (
    LOWER_SURFACE_ID,
    LOWER_TARGET_ID,
    OPAQUE_OUTCOME,
    UPPER_SURFACE_ID,
    UPPER_TARGET_ID,
    KerrFiniteThicknessMultiSurface,
)
from offline.kerr_returning_radiation_kernel import (
    KerrForwardReturningRadiationKernel,
    KerrReturningRadiationKernelPolicy,
    verify_kerr_returning_radiation_energy_kernel,
)
from offline.kerr_returning_radiation_receiver_rays import (
    PAST_WORLDTUBE_NO_SOURCE,
    KerrReturningRadiationReceiverRayPrimitive,
    trace_kerr_returning_radiation_receiver_direction,
    verify_kerr_returning_radiation_receiver_direction,
)
from offline.returning_radiation import AxisymmetricReturningRadiationKernel


IMPLEMENTATION_ID: Final = (
    "kerr-finite-thickness-receiver-centred-finite-volume-energy-kernel/v1"
)
_FACES: Final = (UPPER, LOWER)
_OUTCOMES: Final = (
    "source-upper",
    "source-lower",
    PAST_WORLDTUBE_NO_SOURCE,
)
_MAXIMUM_TOLERANCE: Final = 0.25
_GAUSS_MAXIMUM_ITERATIONS: Final = 64

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "same-code-family finite-grid receiver-centred local-comoving "
            "bolometric returning-radiation energy kernel"
        ),
        "implementationId": IMPLEMENTATION_ID,
        "coefficientEquation": (
            "K=(1/pi) integral mu_i*g^4*D20(mu_e) dOmega_i"
        ),
        "matrixIndexOrder": (
            "K[receiverFace,receiverAnnulus][sourceFace,sourceAnnulus]"
        ),
        "directionQuadratureWeight": "2*w_mu/N_psi",
        "usesAuthenticatedReceiverDirectionPrimitive": True,
        "fineCoarseSourceBinTopologyRequired": True,
        "isFiniteGridEnergyOnlyKernel": True,
        "sharesExactKerrGeodesicCodeFamilyWithForwardKernel": True,
        "isIndependentGeodesicOracle": False,
        "hasIndependentPhysicsOracle": False,
        "hasPerCoefficientForwardReceiverDiagnostic": True,
        "rigorousContinuumErrorBound": False,
        "includesReturningRadiationStressWorkFS": False,
        "includesSpectralRedistribution": False,
        "includesScattering": False,
        "includesPolarization": False,
        "includesSolvedAtmosphere": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "isCompleteKerrbb": False,
        "prohibitedClaim": (
            "Do not describe this shared-code finite-grid comparison as an "
            "independent geodesic oracle, continuum theorem, complete KERRBB, "
            "F_S, atmosphere, spectral/polarized transfer, or GRMHD."
        ),
    }
)


class KerrReturningRadiationReceiverKernelError(RuntimeError):
    """Base class for fail-closed receiver-kernel failures."""


class KerrReturningRadiationReceiverKernelConvergenceError(
    KerrReturningRadiationReceiverKernelError
):
    """Raised when a mandatory grid, topology, or symmetry gate fails."""


class KerrReturningRadiationReceiverKernelVerificationError(
    KerrReturningRadiationReceiverKernelError
):
    """Raised when an immutable result differs from exact same-code replay."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise KerrReturningRadiationReceiverKernelError(
            "receiver-kernel descriptor is not finite canonical JSON"
        ) from error


def _trusted_attribute(value: Any, name: str, path: str) -> Any:
    try:
        return object.__getattribute__(value, name)
    except (AttributeError, TypeError) as error:
        raise KerrReturningRadiationReceiverKernelVerificationError(
            f"{path}.{name} is missing"
        ) from error


def _require_exact_schema_types(actual: Any, template: Any, path: str) -> None:
    if type(actual) is not type(template):
        raise KerrReturningRadiationReceiverKernelVerificationError(
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
            raise KerrReturningRadiationReceiverKernelVerificationError(
                f"{path} tuple length differs from the trusted schema"
            )
        for index, (actual_item, template_item) in enumerate(zip(actual, template)):
            _require_exact_schema_types(
                actual_item,
                template_item,
                f"{path}[{index}]",
            )
        return
    if type(template) not in (float, int, bool, str, type(None)):
        raise KerrReturningRadiationReceiverKernelVerificationError(
            f"{path} uses unsupported schema type {type(template).__name__}"
        )


def _require_trusted_exact_tree(actual: Any, expected: Any, path: str) -> None:
    _require_exact_schema_types(actual, expected, path)
    if is_dataclass(expected) and not isinstance(expected, type):
        for item in fields(expected):
            _require_trusted_exact_tree(
                _trusted_attribute(actual, item.name, path),
                _trusted_attribute(expected, item.name, path),
                f"{path}.{item.name}",
            )
        return
    if type(expected) is tuple:
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _require_trusted_exact_tree(
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
        return
    if type(expected) is float:
        differs = actual.hex() != expected.hex()
    elif type(expected) is int:
        differs = actual != expected
    elif type(expected) is bool:
        differs = actual is not expected
    elif type(expected) is str:
        differs = actual.encode("utf-8") != expected.encode("utf-8")
    elif expected is None:
        differs = False
    else:
        raise KerrReturningRadiationReceiverKernelVerificationError(
            f"{path} has unsupported trusted type {type(expected).__name__}"
        )
    if differs:
        raise KerrReturningRadiationReceiverKernelVerificationError(
            f"{path} differs from exact same-code replay"
        )


def _exact_finite_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite exact float")
    return value


def _exact_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive exact int")
    return value


@dataclass(frozen=True, slots=True)
class ReceiverSkySourceFractions:
    """Receiver-area averaged cosine-sky partition, normalized by ``pi``."""

    source_upper: float
    source_lower: float
    past_worldtube_no_disk_source: float

    def __post_init__(self) -> None:
        for name in (
            "source_upper",
            "source_lower",
            "past_worldtube_no_disk_source",
        ):
            value = _exact_finite_float(getattr(self, name), name)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.total.hex() != 1.0.hex():
            raise ValueError("receiver-sky source fractions must sum exactly to one")

    @property
    def total(self) -> float:
        return math.fsum(self.as_tuple())

    def as_tuple(self) -> tuple[float, float, float]:
        return (
            self.source_upper,
            self.source_lower,
            self.past_worldtube_no_disk_source,
        )


@dataclass(frozen=True, slots=True)
class KerrReceiverReturningRadiationGridDifference:
    matrix_maximum_absolute_difference: float
    matrix_maximum_scaled_difference: float
    row_total_maximum_absolute_difference: float
    row_total_maximum_scaled_difference: float
    sky_fraction_maximum_absolute_difference: float
    sky_fraction_maximum_scaled_difference: float
    converged: bool

    def __post_init__(self) -> None:
        for name in (
            "matrix_maximum_absolute_difference",
            "matrix_maximum_scaled_difference",
            "row_total_maximum_absolute_difference",
            "row_total_maximum_scaled_difference",
            "sky_fraction_maximum_absolute_difference",
            "sky_fraction_maximum_scaled_difference",
        ):
            value = _exact_finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if type(self.converged) is not bool:
            raise TypeError("converged must be an exact bool")
        expected = max(
            self.matrix_maximum_scaled_difference,
            self.row_total_maximum_scaled_difference,
            self.sky_fraction_maximum_scaled_difference,
        ) <= 1.0
        if self.converged is not expected:
            raise ValueError("grid convergence flag disagrees with its gates")


@dataclass(frozen=True, slots=True)
class KerrReceiverReturningRadiationKernelConvergence:
    half_receiver_rho: KerrReceiverReturningRadiationGridDifference
    half_mu: KerrReceiverReturningRadiationGridDifference
    half_psi: KerrReceiverReturningRadiationGridDifference
    phase_shifted: KerrReceiverReturningRadiationGridDifference
    converged: bool

    def __post_init__(self) -> None:
        for name in (
            "half_receiver_rho",
            "half_mu",
            "half_psi",
            "phase_shifted",
        ):
            if type(getattr(self, name)) is not KerrReceiverReturningRadiationGridDifference:
                raise TypeError(f"{name} must be an exact grid difference")
        if type(self.converged) is not bool:
            raise TypeError("converged must be an exact bool")
        expected = all(
            item.converged
            for item in (
                self.half_receiver_rho,
                self.half_mu,
                self.half_psi,
                self.phase_shifted,
            )
        )
        if self.converged is not expected:
            raise ValueError("aggregate convergence disagrees with component gates")


@dataclass(frozen=True, slots=True)
class _ReceiverDirectionTransport:
    outcome: str
    source_face: str | None
    source_radius_over_mass: float | None
    coarse_source_radius_over_mass: float | None
    receiver_integrand: float
    primitive_descriptor_sha256: str

    def __post_init__(self) -> None:
        if type(self.outcome) is not str or self.outcome not in _OUTCOMES:
            raise ValueError("receiver direction outcome is unsupported")
        integrand = _exact_finite_float(
            self.receiver_integrand,
            "receiver_integrand",
        )
        if self.outcome.startswith("source-"):
            expected = UPPER if self.outcome == "source-upper" else LOWER
            if (
                type(self.source_face) is not str
                or self.source_face.encode("utf-8") != expected.encode("utf-8")
            ):
                raise ValueError("source outcome and face disagree")
            fine = _exact_finite_float(
                self.source_radius_over_mass,
                "source_radius_over_mass",
            )
            coarse = _exact_finite_float(
                self.coarse_source_radius_over_mass,
                "coarse_source_radius_over_mass",
            )
            if fine <= 0.0 or coarse <= 0.0 or integrand <= 0.0:
                raise ValueError("source radii and receiver integrand must be positive")
        elif (
            self.source_face is not None
            or self.source_radius_over_mass is not None
            or self.coarse_source_radius_over_mass is not None
            or integrand.hex() != 0.0.hex()
        ):
            raise ValueError("no-source direction cannot carry source transport")
        identity = self.primitive_descriptor_sha256
        if (
            type(identity) is not str
            or len(identity) != 64
            or identity.lower() != identity
        ):
            raise ValueError("primitive identity must be lowercase SHA-256")
        try:
            bytes.fromhex(identity)
        except ValueError as error:
            raise ValueError("primitive identity must be hexadecimal SHA-256") from error


@dataclass(frozen=True, slots=True)
class _ReceiverDirectionSample:
    """Exact canonical address and quadrature node presented to a provider."""

    pass_index: int
    pass_name: str
    receiver_face: str
    receiver_annulus_index: int
    rho_index: int
    mu_index: int
    psi_index: int
    receiver_radius_over_mass: float
    rho_area_over_mass_squared: float
    incidence_cosine: float
    mu_weight: float
    tangent_azimuth_rad: float
    phase_cells: float


_ReceiverDirectionTransportProvider = Callable[
    [_ReceiverDirectionSample],
    _ReceiverDirectionTransport,
]


def _coarse_source_radius(
    primitive: KerrReturningRadiationReceiverRayPrimitive,
    surface: KerrFiniteThicknessMultiSurface,
    source_face: str,
) -> float:
    coarse_ray = object.__getattribute__(primitive, "coarse_ray")
    if type(coarse_ray) is not RayTraceResult:
        raise KerrReturningRadiationReceiverKernelError(
            "revalidated primitive has non-exact coarse ray type"
        )
    trace = object.__getattribute__(coarse_ray, "multi_surface_trace")
    if trace is None or not trace.crossings:
        raise KerrReturningRadiationReceiverKernelError(
            "revalidated source primitive lacks a coarse terminal crossing"
        )
    terminating_entries = tuple(
        entry for entry in trace.crossings if entry.decision.terminates
    )
    if len(terminating_entries) != 1:
        raise KerrReturningRadiationReceiverKernelError(
            "coarse ray must expose exactly one first terminal surface entry"
        )
    terminal = terminating_entries[0]
    if terminal is not trace.crossings[-1]:
        raise KerrReturningRadiationReceiverKernelError(
            "coarse terminal source is not the final recorded crossing"
        )
    expected_surface_id = UPPER_SURFACE_ID if source_face == UPPER else LOWER_SURFACE_ID
    expected_target_id = UPPER_TARGET_ID if source_face == UPPER else LOWER_TARGET_ID
    expected_public_outcome = (
        "source-upper" if source_face == UPPER else "source-lower"
    )
    if (
        type(primitive.outcome) is not str
        or primitive.outcome.encode("utf-8")
        != expected_public_outcome.encode("utf-8")
        or type(primitive.source_face) is not str
        or primitive.source_face.encode("utf-8") != source_face.encode("utf-8")
        or type(primitive.source_surface_id) is not str
        or primitive.source_surface_id.encode("utf-8")
        != expected_surface_id.encode("utf-8")
        or type(coarse_ray.outcome) is not str
        or coarse_ray.outcome.encode("utf-8") != OPAQUE_OUTCOME.encode("utf-8")
        or type(coarse_ray.terminal_target_id) is not str
        or coarse_ray.terminal_target_id.encode("utf-8")
        != expected_target_id.encode("utf-8")
        or type(terminal.surface_id) is not str
        or type(terminal.decision.outcome) is not str
        or type(terminal.decision.target_id) is not str
        or terminal.surface_id.encode("utf-8") != expected_surface_id.encode("utf-8")
        or not terminal.decision.terminates
        or type(terminal.crossing.orientation) is not int
        or terminal.crossing.orientation != -1
        or terminal.decision.outcome.encode("utf-8") != OPAQUE_OUTCOME.encode("utf-8")
        or terminal.decision.target_id.encode("utf-8")
        != expected_target_id.encode("utf-8")
    ):
        raise KerrReturningRadiationReceiverKernelError(
            "coarse terminal source face disagrees with the revalidated primitive"
        )
    _require_trusted_exact_tree(
        terminal.crossing.state,
        coarse_ray.terminal_state,
        "primitive.coarse_ray.terminal_state",
    )
    oblate = kerr_ks_event_to_oblate(
        surface.metric,
        terminal.crossing.state.event,
    )
    rho = oblate.radius_m * math.sin(oblate.theta_rad) / surface.metric.mass_m
    if type(rho) is not float or not math.isfinite(rho) or rho <= 0.0:
        raise KerrReturningRadiationReceiverKernelError(
            "coarse source pseudo-cylindrical radius is invalid"
        )
    difference = primitive.convergence.source_radius_difference_over_mass
    fine = primitive.source_radius_over_mass
    if type(difference) is not float or type(fine) is not float:
        raise KerrReturningRadiationReceiverKernelError(
            "source-radius convergence has a non-exact schema"
        )
    roundoff = 32.0 * math.ulp(max(1.0, abs(rho), abs(fine)))
    if abs(abs(fine - rho) - difference) > roundoff:
        raise KerrReturningRadiationReceiverKernelError(
            "coarse source radius disagrees with public convergence evidence"
        )
    return rho


def _trace_direction(
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
    coarse_ray_options: RayTraceOptions | None,
    coarse_surface_options: SurfaceEventOptions | None,
    receiver_face: str,
    receiver_radius_over_mass: float,
    incidence_cosine: float,
    tangent_azimuth_rad: float,
) -> _ReceiverDirectionTransport:
    receiver = KerrFiniteThicknessFaceEmitter(
        metric=surface.metric,
        calibration=surface.calibration,
        pseudo_cylindrical_radius_over_mass=receiver_radius_over_mass,
        face=receiver_face,
    )
    frame = KerrFiniteThicknessSurfaceFrame(receiver)
    primitive = trace_kerr_returning_radiation_receiver_direction(
        frame,
        surface,
        incidence_cosine,
        tangent_azimuth_rad,
        termination=termination,
        ray_options=ray_options,
        surface_options=surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
    )
    if type(primitive) is not KerrReturningRadiationReceiverRayPrimitive:
        raise TypeError("receiver tracer returned a non-exact primitive")
    verify_kerr_returning_radiation_receiver_direction(primitive)
    outcome = object.__getattribute__(primitive, "outcome")
    identity = primitive.model_descriptor_sha256
    if type(outcome) is not str or outcome not in _OUTCOMES:
        raise KerrReturningRadiationReceiverKernelError(
            "revalidated receiver primitive has unsupported outcome"
        )
    if outcome == PAST_WORLDTUBE_NO_SOURCE:
        return _ReceiverDirectionTransport(outcome, None, None, None, 0.0, identity)
    source_face = object.__getattribute__(primitive, "source_face")
    source_radius = object.__getattribute__(primitive, "source_radius_over_mass")
    integrand = object.__getattribute__(primitive, "receiver_directional_integrand")
    if type(source_face) is not str or source_face not in _FACES:
        raise KerrReturningRadiationReceiverKernelError(
            "revalidated primitive lacks an exact source face"
        )
    source_radius = _exact_finite_float(source_radius, "source_radius_over_mass")
    integrand = _exact_finite_float(integrand, "receiver_directional_integrand")
    coarse_radius = _coarse_source_radius(primitive, surface, source_face)
    return _ReceiverDirectionTransport(
        outcome,
        source_face,
        source_radius,
        coarse_radius,
        integrand,
        identity,
    )


def _gauss_legendre_unit_interval(order: int) -> tuple[tuple[float, float], ...]:
    order = _exact_positive_int(order, "Gauss-Legendre order")
    nodes = [0.0] * order
    weights = [0.0] * order
    half = (order + 1) // 2
    for index in range(half):
        root = math.cos(math.pi * (index + 0.75) / (order + 0.5))
        derivative = 0.0
        visited_iterates: list[tuple[float, float, float]] = []
        for _iteration in range(_GAUSS_MAXIMUM_ITERATIONS):
            previous = 1.0
            current = root
            for degree in range(2, order + 1):
                following = (
                    (2.0 * degree - 1.0) * root * current
                    - (degree - 1.0) * previous
                ) / degree
                previous, current = current, following
            derivative = order * (root * current - previous) / (root * root - 1.0)
            update = current / derivative
            next_root = root - update
            if next_root == root or abs(update) <= 2.0 * math.ulp(root):
                root = next_root
                break
            visited_iterates.append((root, current, derivative))
            cycle_start = next(
                (
                    item_index
                    for item_index, (item_root, _residual, _derivative) in enumerate(
                        visited_iterates
                    )
                    if next_root == item_root
                ),
                None,
            )
            if cycle_start is not None:
                root, _residual, derivative = min(
                    visited_iterates[cycle_start:],
                    key=lambda item: (abs(item[1]), item[0]),
                )
                break
            root = next_root
        else:
            raise KerrReturningRadiationReceiverKernelError(
                "Gauss-Legendre root solve did not converge"
            )
        weight = 1.0 / ((1.0 - root * root) * derivative * derivative)
        nodes[index] = 0.5 * (1.0 - root)
        nodes[order - 1 - index] = 0.5 * (1.0 + root)
        weights[index] = weight
        weights[order - 1 - index] = weight
    rule = tuple(zip(nodes, weights))
    if any(
        not (0.0 < node < 1.0 and math.isfinite(weight) and weight > 0.0)
        for node, weight in rule
    ):
        raise KerrReturningRadiationReceiverKernelError(
            "Gauss-Legendre rule is not interior-positive"
        )
    if abs(math.fsum(weight for _node, weight in rule) - 1.0) > 64.0 * math.ulp(1.0):
        raise KerrReturningRadiationReceiverKernelError(
            "Gauss-Legendre rule does not integrate a constant"
        )
    return rule


def _validated_edges(
    values: tuple[float, ...],
    surface: KerrFiniteThicknessMultiSurface,
) -> tuple[float, ...]:
    if type(values) is not tuple or len(values) < 2:
        raise TypeError("annulus edges must be an exact tuple with at least two values")
    edges = tuple(
        _exact_finite_float(value, f"annulus edge {index}")
        for index, value in enumerate(values)
    )
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("annulus edges must be strictly increasing")
    inner = float(surface.calibration.isco_radius_over_mass)
    outer = float(surface.calibration.outer_radius_over_mass)
    if edges[0].hex() != inner.hex() or edges[-1].hex() != outer.hex():
        raise ValueError(
            "annulus edges must exactly and completely cover ISCO through R_out"
        )
    return edges


def _bin_index(radius: float, edges: tuple[float, ...], label: str) -> int:
    radius = _exact_finite_float(radius, label)
    index = bisect_right(edges, radius) - 1
    if index == len(edges) - 1 and radius.hex() == edges[-1].hex():
        index -= 1
    if index < 0 or index >= len(edges) - 1:
        raise KerrReturningRadiationReceiverKernelError(
            f"{label} lies outside the declared finite annulus grid"
        )
    return index


def _validated_source_bin(
    transport: _ReceiverDirectionTransport,
    edges: tuple[float, ...],
) -> int:
    if type(transport) is not _ReceiverDirectionTransport:
        raise TypeError("transport must be the exact internal receiver direction")
    if not transport.outcome.startswith("source-"):
        raise ValueError("source-bin validation requires a disk source")
    fine = _bin_index(
        transport.source_radius_over_mass,
        edges,
        "fine source radius",
    )
    coarse = _bin_index(
        transport.coarse_source_radius_over_mass,
        edges,
        "coarse source radius",
    )
    if fine != coarse:
        raise KerrReturningRadiationReceiverKernelConvergenceError(
            "fine/coarse backward rays land in different source annuli: "
            f"fine rho/M={transport.source_radius_over_mass:.17g} -> bin {fine}, "
            f"coarse rho/M={transport.coarse_source_radius_over_mass:.17g} -> bin {coarse}"
        )
    return fine


def _validated_surface(
    surface: KerrFiniteThicknessMultiSurface,
) -> KerrFiniteThicknessMultiSurface:
    if type(surface) is not KerrFiniteThicknessMultiSurface:
        raise TypeError("surface must be exact KerrFiniteThicknessMultiSurface")
    metric = object.__getattribute__(surface, "metric")
    calibration = object.__getattribute__(surface, "calibration")
    if type(metric) is not KerrKerrSchildMetric:
        raise TypeError("surface metric must be exact KerrKerrSchildMetric")
    if type(calibration) is not StationaryKerrFiniteThicknessCalibration:
        raise TypeError("surface calibration must be exact built-in type")
    _require_exact_schema_types(metric, KerrKerrSchildMetric(), "surface.metric")
    _require_exact_schema_types(
        calibration,
        StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.0,
            eddington_scaled_mass_accretion_rate=0.01,
            outer_radius_over_mass=10.0,
        ),
        "surface.calibration",
    )
    rebuilt_metric = KerrKerrSchildMetric(**asdict(metric))
    rebuilt_calibration = StationaryKerrFiniteThicknessCalibration(**asdict(calibration))
    rebuilt = KerrFiniteThicknessMultiSurface(rebuilt_metric, rebuilt_calibration)
    _require_trusted_exact_tree(surface, rebuilt, "surface")
    return rebuilt


def _validated_auxiliary_inputs(
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
    coarse_ray_options: RayTraceOptions | None,
    coarse_surface_options: SurfaceEventOptions | None,
) -> tuple[
    KerrOblateTermination,
    RayTraceOptions,
    SurfaceEventOptions,
    RayTraceOptions | None,
    SurfaceEventOptions | None,
]:
    if type(termination) is not KerrOblateTermination:
        raise TypeError("termination must be exact KerrOblateTermination")
    if type(ray_options) is not RayTraceOptions:
        raise TypeError("ray_options must be exact RayTraceOptions")
    if type(surface_options) is not SurfaceEventOptions:
        raise TypeError("surface_options must be exact SurfaceEventOptions")
    if (coarse_ray_options is None) != (coarse_surface_options is None):
        raise ValueError("coarse ray and surface options must be supplied together")
    if coarse_ray_options is not None and type(coarse_ray_options) is not RayTraceOptions:
        raise TypeError("coarse_ray_options must be exact RayTraceOptions or None")
    if coarse_surface_options is not None and type(coarse_surface_options) is not SurfaceEventOptions:
        raise TypeError(
            "coarse_surface_options must be exact SurfaceEventOptions or None"
        )
    _require_exact_schema_types(
        termination,
        KerrOblateTermination(0.0, 1.0, 2.0),
        "termination",
    )
    _require_exact_schema_types(ray_options, RayTraceOptions(), "ray_options")
    _require_exact_schema_types(
        surface_options,
        SurfaceEventOptions(),
        "surface_options",
    )
    if coarse_ray_options is not None:
        assert coarse_surface_options is not None
        _require_exact_schema_types(
            coarse_ray_options,
            RayTraceOptions(),
            "coarse_ray_options",
        )
        _require_exact_schema_types(
            coarse_surface_options,
            SurfaceEventOptions(),
            "coarse_surface_options",
        )
    rebuilt_termination = KerrOblateTermination(**asdict(termination))
    rebuilt_ray = RayTraceOptions(**asdict(ray_options))
    rebuilt_surface = SurfaceEventOptions(**asdict(surface_options))
    rebuilt_coarse_ray = (
        None
        if coarse_ray_options is None
        else RayTraceOptions(**asdict(coarse_ray_options))
    )
    rebuilt_coarse_surface = (
        None
        if coarse_surface_options is None
        else SurfaceEventOptions(**asdict(coarse_surface_options))
    )
    _require_trusted_exact_tree(termination, rebuilt_termination, "termination")
    _require_trusted_exact_tree(ray_options, rebuilt_ray, "ray_options")
    _require_trusted_exact_tree(surface_options, rebuilt_surface, "surface_options")
    if coarse_ray_options is not None:
        assert coarse_surface_options is not None
        assert rebuilt_coarse_ray is not None
        assert rebuilt_coarse_surface is not None
        _require_trusted_exact_tree(
            coarse_ray_options,
            rebuilt_coarse_ray,
            "coarse_ray_options",
        )
        _require_trusted_exact_tree(
            coarse_surface_options,
            rebuilt_coarse_surface,
            "coarse_surface_options",
        )
    return (
        rebuilt_termination,
        rebuilt_ray,
        rebuilt_surface,
        rebuilt_coarse_ray,
        rebuilt_coarse_surface,
    )


def _validated_policy(
    policy: KerrReturningRadiationKernelPolicy | None,
) -> KerrReturningRadiationKernelPolicy:
    selected = KerrReturningRadiationKernelPolicy() if policy is None else policy
    if type(selected) is not KerrReturningRadiationKernelPolicy:
        raise TypeError("policy must be exact KerrReturningRadiationKernelPolicy")
    _require_exact_schema_types(
        selected,
        KerrReturningRadiationKernelPolicy(),
        "policy",
    )
    rebuilt = KerrReturningRadiationKernelPolicy(**asdict(selected))
    _require_trusted_exact_tree(selected, rebuilt, "policy")
    return rebuilt


def _validated_area_policy(
    policy: KerrFiniteThicknessAreaQuadraturePolicy | None,
) -> KerrFiniteThicknessAreaQuadraturePolicy:
    selected = KerrFiniteThicknessAreaQuadraturePolicy() if policy is None else policy
    if type(selected) is not KerrFiniteThicknessAreaQuadraturePolicy:
        raise TypeError("area_policy must be exact built-in area policy")
    _require_exact_schema_types(
        selected,
        KerrFiniteThicknessAreaQuadraturePolicy(),
        "area_policy",
    )
    rebuilt = KerrFiniteThicknessAreaQuadraturePolicy(
        gauss_legendre_order=selected.gauss_legendre_order,
        relative_tolerance=selected.relative_tolerance,
        absolute_tolerance_over_mass_squared=(
            selected.absolute_tolerance_over_mass_squared
        ),
        maximum_point_evaluations=selected.maximum_point_evaluations,
    )
    _require_trusted_exact_tree(selected, rebuilt, "area_policy")
    return rebuilt


def _annulus_areas(
    surface: KerrFiniteThicknessMultiSurface,
    edges: tuple[float, ...],
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[str, ...], tuple[str, ...]]:
    values: dict[str, list[float]] = {UPPER: [], LOWER: []}
    identities: dict[str, list[str]] = {UPPER: [], LOWER: []}
    for face in _FACES:
        for inner, outer in zip(edges, edges[1:]):
            result = integrate_kerr_finite_thickness_annulus_area(
                metric=surface.metric,
                calibration=surface.calibration,
                inner_radius_over_mass=inner,
                outer_radius_over_mass=outer,
                face=face,
                policy=area_policy,
            )
            result.revalidate()
            values[face].append(result.proper_area_over_mass_squared)
            identities[face].append(result.model_descriptor_sha256)
    return (
        tuple(values[UPPER]),
        tuple(values[LOWER]),
        tuple(identities[UPPER]),
        tuple(identities[LOWER]),
    )


def _receiver_rho_area_nodes(
    surface: KerrFiniteThicknessMultiSurface,
    inner: float,
    outer: float,
    face: str,
    order: int,
    exact_annulus_area: float,
) -> tuple[tuple[float, float], ...]:
    width = outer - inner
    provisional: list[tuple[float, float]] = []
    rule = _gauss_legendre_unit_interval(order)
    rho_nodes = _rho_coordinate_nodes_from_rule(inner, outer, rule)
    for rho, (_node, weight) in zip(rho_nodes, rule):
        density = kerr_finite_thickness_area_density(
            metric=surface.metric,
            calibration=surface.calibration,
            pseudo_cylindrical_radius_over_mass=rho,
            face=face,
        )
        density.revalidate()
        raw_area = (
            2.0
            * math.pi
            * width
            * weight
            * density.proper_area_density_over_mass_squared
        )
        if not math.isfinite(raw_area) or raw_area <= 0.0:
            raise KerrReturningRadiationReceiverKernelError(
                "receiver radial proper-area weight is invalid"
            )
        provisional.append((rho, raw_area))
    raw_total = math.fsum(weight for _rho, weight in provisional)
    scale = exact_annulus_area / raw_total
    normalized = [(rho, weight * scale) for rho, weight in provisional]
    for _iteration in range(16):
        current = math.fsum(weight for _rho, weight in normalized)
        if current.hex() == exact_annulus_area.hex():
            break
        rho, weight = normalized[-1]
        candidate = weight + (exact_annulus_area - current)
        if candidate.hex() == weight.hex():
            candidate = math.nextafter(
                weight,
                math.inf if current < exact_annulus_area else -math.inf,
            )
        normalized[-1] = (rho, candidate)
    total = math.fsum(weight for _rho, weight in normalized)
    if (
        abs(total - exact_annulus_area) > 2.0 * math.ulp(exact_annulus_area)
        or any(weight <= 0.0 or not math.isfinite(weight) for _rho, weight in normalized)
    ):
        raise KerrReturningRadiationReceiverKernelError(
            "receiver radial weights do not close to proper annulus area"
        )
    return tuple(normalized)


def _rho_coordinate_nodes(
    inner: float,
    outer: float,
    order: int,
) -> tuple[float, ...]:
    """Return the exact radial coordinates shared by tracing and cache jobs."""

    return _rho_coordinate_nodes_from_rule(
        inner,
        outer,
        _gauss_legendre_unit_interval(order),
    )


def _rho_coordinate_nodes_from_rule(
    inner: float,
    outer: float,
    rule: tuple[tuple[float, float], ...],
) -> tuple[float, ...]:
    width = outer - inner
    return tuple(
        math.fsum((inner, width * node))
        for node, _weight in rule
    )


def _receiver_sky_direction_nodes(
    mu_order: int,
    psi_count: int,
    phase_cells: float,
) -> tuple[tuple[float, float, float], ...]:
    """Canonical receiver-sky nodes shared by direct and cached execution."""

    phase = _exact_finite_float(phase_cells, "phase_cells")
    return tuple(
        (
            mu_i,
            mu_weight,
            (
                (psi_index + 0.5 + phase)
                * 2.0
                * math.pi
                / psi_count
            )
            % (2.0 * math.pi),
        )
        for mu_i, mu_weight in _gauss_legendre_unit_interval(mu_order)
        for psi_index in range(psi_count)
    )


def _empty_contributions(count: int) -> list[list[list[float]]]:
    return [[[] for _source in range(count)] for _receiver in range(count)]


def _freeze_matrix(values: list[list[list[float]]]) -> tuple[tuple[float, ...], ...]:
    matrix = tuple(tuple(math.fsum(cell) for cell in row) for row in values)
    if any(value < 0.0 or not math.isfinite(value) for row in matrix for value in row):
        raise KerrReturningRadiationReceiverKernelError(
            "receiver kernel matrix is not finite/non-negative"
        )
    return matrix


@dataclass(frozen=True, slots=True)
class _ReceiverKernelGridEstimate:
    uu: tuple[tuple[float, ...], ...]
    ul: tuple[tuple[float, ...], ...]
    lu: tuple[tuple[float, ...], ...]
    ll: tuple[tuple[float, ...], ...]
    upper_row_totals: tuple[float, ...]
    lower_row_totals: tuple[float, ...]
    upper_row_closure_residuals: tuple[float, ...]
    lower_row_closure_residuals: tuple[float, ...]
    upper_sky_fractions: tuple[ReceiverSkySourceFractions, ...]
    lower_sky_fractions: tuple[ReceiverSkySourceFractions, ...]
    direction_evaluations: int
    sample_audit_sha256: str

    def matrix_vector(self) -> tuple[float, ...]:
        return tuple(
            value
            for block in (self.uu, self.ul, self.lu, self.ll)
            for row in block
            for value in row
        )

    def row_vector(self) -> tuple[float, ...]:
        return (*self.upper_row_totals, *self.lower_row_totals)

    def sky_vector(self) -> tuple[float, ...]:
        return tuple(
            value
            for fractions in (*self.upper_sky_fractions, *self.lower_sky_fractions)
            for value in fractions.as_tuple()
        )


def _integrate_grid(
    *,
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
    coarse_ray_options: RayTraceOptions | None,
    coarse_surface_options: SurfaceEventOptions | None,
    edges: tuple[float, ...],
    upper_areas: tuple[float, ...],
    lower_areas: tuple[float, ...],
    receiver_rho_order: int,
    mu_order: int,
    psi_count: int,
    phase_cells: float,
    pass_index: int,
    pass_name: str,
    direction_transport_provider: _ReceiverDirectionTransportProvider | None,
    rho_node_cache: dict[
        tuple[str, str, str, int, str],
        tuple[tuple[float, float], ...],
    ],
) -> _ReceiverKernelGridEstimate:
    count = len(edges) - 1
    blocks = {
        (UPPER, UPPER): _empty_contributions(count),
        (UPPER, LOWER): _empty_contributions(count),
        (LOWER, UPPER): _empty_contributions(count),
        (LOWER, LOWER): _empty_contributions(count),
    }
    areas = {UPPER: upper_areas, LOWER: lower_areas}
    row_contributions: dict[str, list[list[float]]] = {
        face: [[] for _receiver in range(count)] for face in _FACES
    }
    sky_contributions: dict[str, list[list[list[float]]]] = {
        face: [[[] for _outcome in _OUTCOMES] for _receiver in range(count)]
        for face in _FACES
    }
    final_outcome: dict[str, list[str | None]] = {
        face: [None] * count for face in _FACES
    }
    phase = _exact_finite_float(phase_cells, "phase_cells")
    direction_nodes = _receiver_sky_direction_nodes(
        mu_order,
        psi_count,
        phase,
    )
    audit_hash = hashlib.sha256()
    evaluations = 0
    for receiver_face in _FACES:
        for receiver_index, (inner, outer) in enumerate(zip(edges, edges[1:])):
            receiver_area = areas[receiver_face][receiver_index]
            cache_key = (
                receiver_face,
                inner.hex(),
                outer.hex(),
                receiver_rho_order,
                receiver_area.hex(),
            )
            rho_nodes = rho_node_cache.get(cache_key)
            if rho_nodes is None:
                rho_nodes = _receiver_rho_area_nodes(
                    surface,
                    inner,
                    outer,
                    receiver_face,
                    receiver_rho_order,
                    receiver_area,
                )
                rho_node_cache[cache_key] = rho_nodes
            for rho_index, (rho, rho_area) in enumerate(rho_nodes):
                receiver_area_weight = rho_area / receiver_area
                if (
                    not math.isfinite(receiver_area_weight)
                    or receiver_area_weight <= 0.0
                ):
                    raise KerrReturningRadiationReceiverKernelError(
                        "receiver proper-area averaging weight is invalid"
                    )
                for direction_index, (mu_i, mu_weight, psi_i) in enumerate(
                    direction_nodes
                ):
                    mu_index, psi_index = divmod(direction_index, psi_count)
                    sample = _ReceiverDirectionSample(
                        pass_index,
                        pass_name,
                        receiver_face,
                        receiver_index,
                        rho_index,
                        mu_index,
                        psi_index,
                        rho,
                        rho_area,
                        mu_i,
                        mu_weight,
                        psi_i,
                        phase,
                    )
                    if direction_transport_provider is None:
                        transport = _trace_direction(
                            surface,
                            termination,
                            ray_options,
                            surface_options,
                            coarse_ray_options,
                            coarse_surface_options,
                            receiver_face,
                            rho,
                            mu_i,
                            psi_i,
                        )
                    else:
                        transport = direction_transport_provider(sample)
                    if type(transport) is not _ReceiverDirectionTransport:
                        raise TypeError(
                            "receiver direction must have exact internal type"
                        )
                    sky_weight = (
                        receiver_area_weight
                        * 2.0
                        * mu_i
                        * mu_weight
                        / psi_count
                    )
                    sky_contributions[receiver_face][receiver_index][
                        _OUTCOMES.index(transport.outcome)
                    ].append(sky_weight)
                    final_outcome[receiver_face][receiver_index] = transport.outcome
                    coefficient = 0.0
                    source_index = None
                    if transport.outcome.startswith("source-"):
                        source_face = transport.source_face
                        assert source_face is not None
                        source_index = _validated_source_bin(transport, edges)
                        coefficient = (
                            receiver_area_weight
                            * (2.0 * mu_weight / psi_count)
                            * transport.receiver_integrand
                        )
                        if not math.isfinite(coefficient) or coefficient <= 0.0:
                            raise KerrReturningRadiationReceiverKernelError(
                                "receiver-centred coefficient is invalid"
                            )
                        blocks[(receiver_face, source_face)][receiver_index][
                            source_index
                        ].append(coefficient)
                        row_contributions[receiver_face][receiver_index].append(
                            coefficient
                        )
                    audit_hash.update(
                        _canonical_json(
                            {
                                "coarseSourceRadiusOverMass": (
                                    transport.coarse_source_radius_over_mass
                                ),
                                "directionCoefficient": coefficient,
                                "mu": mu_i,
                                "muWeight": mu_weight,
                                "outcome": transport.outcome,
                                "phaseCells": phase,
                                "primitiveDescriptorSha256": (
                                    transport.primitive_descriptor_sha256
                                ),
                                "psi": psi_i,
                                "receiverAreaOverMassSquared": rho_area,
                                "receiverFace": receiver_face,
                                "receiverRadiusOverMass": rho,
                                "receiverRow": receiver_index,
                                "sourceColumn": source_index,
                                "sourceFace": transport.source_face,
                                "sourceRadiusOverMass": (
                                    transport.source_radius_over_mass
                                ),
                                "receiverDirectionalIntegrand": (
                                    transport.receiver_integrand
                                ),
                            }
                        ).encode("utf-8")
                    )
                    audit_hash.update(b"\n")
                    evaluations += 1

    frozen_blocks = {key: _freeze_matrix(value) for key, value in blocks.items()}
    row_totals: dict[str, tuple[float, ...]] = {}
    row_residuals: dict[str, tuple[float, ...]] = {}
    sky_fractions: dict[str, tuple[ReceiverSkySourceFractions, ...]] = {}
    for receiver_face in _FACES:
        totals: list[float] = []
        residuals: list[float] = []
        fractions: list[ReceiverSkySourceFractions] = []
        for receiver_index in range(count):
            direct = math.fsum(row_contributions[receiver_face][receiver_index])
            reconstructed = math.fsum(
                frozen_blocks[(receiver_face, source_face)][receiver_index][source_index]
                for source_face in _FACES
                for source_index in range(count)
            )
            residual = abs(direct - reconstructed)
            if residual > 4096.0 * math.ulp(max(1.0, direct, reconstructed)):
                raise KerrReturningRadiationReceiverKernelError(
                    "receiver row does not close against direct sky integration"
                )
            totals.append(direct)
            residuals.append(residual)

            values = [
                math.fsum(entries)
                for entries in sky_contributions[receiver_face][receiver_index]
            ]
            total = math.fsum(values)
            closure = 1.0 - total
            if abs(closure) > 2048.0 * math.ulp(max(1.0, total)):
                raise KerrReturningRadiationReceiverKernelError(
                    "receiver sky partition exceeds float64 closure allowance"
                )
            last = final_outcome[receiver_face][receiver_index]
            if last is None:
                raise KerrReturningRadiationReceiverKernelError(
                    "receiver annulus produced no sky directions"
                )
            values[_OUTCOMES.index(last)] += closure
            fractions.append(ReceiverSkySourceFractions(*values))
        row_totals[receiver_face] = tuple(totals)
        row_residuals[receiver_face] = tuple(residuals)
        sky_fractions[receiver_face] = tuple(fractions)

    return _ReceiverKernelGridEstimate(
        frozen_blocks[(UPPER, UPPER)],
        frozen_blocks[(UPPER, LOWER)],
        frozen_blocks[(LOWER, UPPER)],
        frozen_blocks[(LOWER, LOWER)],
        row_totals[UPPER],
        row_totals[LOWER],
        row_residuals[UPPER],
        row_residuals[LOWER],
        sky_fractions[UPPER],
        sky_fractions[LOWER],
        evaluations,
        audit_hash.hexdigest(),
    )


def _vector_difference(
    full: tuple[float, ...],
    comparison: tuple[float, ...],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[float, float]:
    if len(full) != len(comparison) or not full:
        raise KerrReturningRadiationReceiverKernelError(
            "convergence vectors have incompatible shapes"
        )
    absolute: list[float] = []
    scaled: list[float] = []
    for full_value, comparison_value in zip(full, comparison):
        difference = abs(full_value - comparison_value)
        threshold = max(
            absolute_tolerance,
            relative_tolerance * max(abs(full_value), abs(comparison_value)),
        )
        if not math.isfinite(difference) or not math.isfinite(threshold):
            raise KerrReturningRadiationReceiverKernelError(
                "grid convergence diagnostic is non-finite"
            )
        absolute.append(difference)
        scaled.append(difference / threshold)
    return max(absolute), max(scaled)


def _grid_difference(
    full: _ReceiverKernelGridEstimate,
    comparison: _ReceiverKernelGridEstimate,
    policy: KerrReturningRadiationKernelPolicy,
) -> KerrReceiverReturningRadiationGridDifference:
    matrix_abs, matrix_scaled = _vector_difference(
        full.matrix_vector(),
        comparison.matrix_vector(),
        policy.absolute_tolerance,
        policy.relative_tolerance,
    )
    row_abs, row_scaled = _vector_difference(
        full.row_vector(),
        comparison.row_vector(),
        policy.absolute_tolerance,
        policy.relative_tolerance,
    )
    sky_abs, sky_scaled = _vector_difference(
        full.sky_vector(),
        comparison.sky_vector(),
        policy.absolute_tolerance,
        policy.relative_tolerance,
    )
    return KerrReceiverReturningRadiationGridDifference(
        matrix_abs,
        matrix_scaled,
        row_abs,
        row_scaled,
        sky_abs,
        sky_scaled,
        max(matrix_scaled, row_scaled, sky_scaled) <= 1.0,
    )


@dataclass(frozen=True, slots=True, init=False)
class KerrReceiverReturningRadiationKernel:
    """Authenticated four-face receiver-centred finite-volume energy operator."""

    annulus_edges_over_mass: tuple[float, ...]
    annulus_representative_radii_over_mass: tuple[float, ...]
    upper_annulus_areas_over_mass_squared: tuple[float, ...]
    lower_annulus_areas_over_mass_squared: tuple[float, ...]
    upper_receiver_upper_source_coefficients: tuple[tuple[float, ...], ...]
    upper_receiver_lower_source_coefficients: tuple[tuple[float, ...], ...]
    lower_receiver_upper_source_coefficients: tuple[tuple[float, ...], ...]
    lower_receiver_lower_source_coefficients: tuple[tuple[float, ...], ...]
    upper_receiver_returning_row_totals: tuple[float, ...]
    lower_receiver_returning_row_totals: tuple[float, ...]
    upper_receiver_row_closure_residuals: tuple[float, ...]
    lower_receiver_row_closure_residuals: tuple[float, ...]
    upper_receiver_sky_source_fractions: tuple[ReceiverSkySourceFractions, ...]
    lower_receiver_sky_source_fractions: tuple[ReceiverSkySourceFractions, ...]
    convergence: KerrReceiverReturningRadiationKernelConvergence
    fine_coarse_source_bin_topology_verified: bool
    direction_evaluations_consumed: int
    whole_ray_traces_consumed: int
    full_grid_sample_audit_sha256: str
    half_receiver_rho_sample_audit_sha256: str
    half_mu_sample_audit_sha256: str
    half_psi_sample_audit_sha256: str
    phase_shifted_sample_audit_sha256: str
    upper_annulus_area_descriptor_sha256: tuple[str, ...]
    lower_annulus_area_descriptor_sha256: tuple[str, ...]
    surface: KerrFiniteThicknessMultiSurface
    termination: KerrOblateTermination
    ray_options: RayTraceOptions
    surface_options: SurfaceEventOptions
    coarse_ray_options: RayTraceOptions | None
    coarse_surface_options: SurfaceEventOptions | None
    policy: KerrReturningRadiationKernelPolicy
    area_policy: KerrFiniteThicknessAreaQuadraturePolicy
    _descriptor_json: str
    _descriptor_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "KerrReceiverReturningRadiationKernel is built only by its integrator"
        )

    @property
    def annulus_count(self) -> int:
        return len(self.annulus_edges_over_mass) - 1

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        if type(self._descriptor_json) is not str:
            raise KerrReturningRadiationReceiverKernelVerificationError(
                "receiver-kernel descriptor has non-exact type"
            )
        return json.loads(self._descriptor_json)

    def revalidate(self) -> None:
        verify_kerr_returning_radiation_receiver_energy_kernel(self)

    def to_axisymmetric_energy_kernel(
        self,
        require_equatorial_symmetry: bool = True,
    ) -> AxisymmetricReturningRadiationKernel:
        if type(require_equatorial_symmetry) is not bool:
            raise TypeError("require_equatorial_symmetry must be an exact bool")
        if require_equatorial_symmetry is not True:
            raise ValueError("four-face reduction requires equatorial symmetry")
        verify_kerr_returning_radiation_receiver_energy_kernel(self)
        _require_equatorial_symmetry(self)
        count = self.annulus_count
        reduced = tuple(
            tuple(
                0.5
                * math.fsum(
                    (
                        self.upper_receiver_upper_source_coefficients[i][j],
                        self.upper_receiver_lower_source_coefficients[i][j],
                        self.lower_receiver_upper_source_coefficients[i][j],
                        self.lower_receiver_lower_source_coefficients[i][j],
                    )
                )
                for j in range(count)
            )
            for i in range(count)
        )
        return AxisymmetricReturningRadiationKernel(
            annulus_radii_over_mass=self.annulus_representative_radii_over_mass,
            receiver_emitter_coefficients=reduced,
            ray_kernel_producer_id=f"{IMPLEMENTATION_ID}:{self.model_descriptor_sha256}",
        )


def _require_equatorial_symmetry(result: KerrReceiverReturningRadiationKernel) -> None:
    policy = result.policy
    for upper, lower in zip(
        result.upper_annulus_areas_over_mass_squared,
        result.lower_annulus_areas_over_mass_squared,
    ):
        difference = abs(upper - lower)
        threshold = max(
            policy.symmetry_absolute_tolerance,
            policy.symmetry_relative_tolerance * max(abs(upper), abs(lower)),
        )
        if difference > threshold:
            raise KerrReturningRadiationReceiverKernelConvergenceError(
                "upper/lower receiver annulus areas fail equatorial symmetry"
            )
    for left, right, label in (
        (
            result.upper_receiver_upper_source_coefficients,
            result.lower_receiver_lower_source_coefficients,
            "UU versus LL",
        ),
        (
            result.upper_receiver_lower_source_coefficients,
            result.lower_receiver_upper_source_coefficients,
            "UL versus LU",
        ),
    ):
        for left_row, right_row in zip(left, right):
            for left_value, right_value in zip(left_row, right_row):
                difference = abs(left_value - right_value)
                threshold = max(
                    policy.symmetry_absolute_tolerance,
                    policy.symmetry_relative_tolerance
                    * max(abs(left_value), abs(right_value)),
                )
                if difference > threshold:
                    raise KerrReturningRadiationReceiverKernelConvergenceError(
                        f"{label} blocks fail equatorial symmetry"
                    )


def _integrate_kerr_returning_radiation_receiver_energy_kernel_with_transport_provider(
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
    direction_transport_provider: _ReceiverDirectionTransportProvider | None = None,
) -> KerrReceiverReturningRadiationKernel:
    """Integrate the four receiver-centred finite-volume face blocks."""

    surface = _validated_surface(surface)
    (
        termination,
        ray_options,
        surface_options,
        coarse_ray_options,
        coarse_surface_options,
    ) = _validated_auxiliary_inputs(
        termination,
        ray_options,
        surface_options,
        coarse_ray_options,
        coarse_surface_options,
    )
    selected_policy = _validated_policy(policy)
    selected_area_policy = _validated_area_policy(area_policy)
    if direction_transport_provider is not None and not callable(
        direction_transport_provider
    ):
        raise TypeError("direction_transport_provider must be callable or None")
    edges = _validated_edges(annulus_edges_over_mass, surface)
    count = len(edges) - 1
    expected_directions = (
        7
        * count
        * selected_policy.rho_order
        * selected_policy.mu_order
        * selected_policy.psi_count
    )
    expected_whole_rays = 4 * expected_directions
    if expected_directions > selected_policy.maximum_direction_evaluations:
        raise ValueError(
            f"receiver kernel requires {expected_directions} direction evaluations "
            f"but budget is {selected_policy.maximum_direction_evaluations}"
        )
    if expected_whole_rays > selected_policy.maximum_whole_ray_traces:
        raise ValueError(
            f"receiver kernel requires {expected_whole_rays} whole rays but budget "
            f"is {selected_policy.maximum_whole_ray_traces}"
        )
    (
        upper_areas,
        lower_areas,
        upper_area_hashes,
        lower_area_hashes,
    ) = _annulus_areas(surface, edges, selected_area_policy)
    common = {
        "surface": surface,
        "termination": termination,
        "ray_options": ray_options,
        "surface_options": surface_options,
        "coarse_ray_options": coarse_ray_options,
        "coarse_surface_options": coarse_surface_options,
        "edges": edges,
        "upper_areas": upper_areas,
        "lower_areas": lower_areas,
        "rho_node_cache": {},
        "direction_transport_provider": direction_transport_provider,
    }
    full = _integrate_grid(
        **common,
        receiver_rho_order=selected_policy.rho_order,
        mu_order=selected_policy.mu_order,
        psi_count=selected_policy.psi_count,
        phase_cells=0.0,
        pass_index=0,
        pass_name="full",
    )
    half_rho = _integrate_grid(
        **common,
        receiver_rho_order=selected_policy.rho_order // 2,
        mu_order=selected_policy.mu_order,
        psi_count=selected_policy.psi_count,
        phase_cells=0.0,
        pass_index=1,
        pass_name="half-rho",
    )
    half_mu = _integrate_grid(
        **common,
        receiver_rho_order=selected_policy.rho_order,
        mu_order=selected_policy.mu_order // 2,
        psi_count=selected_policy.psi_count,
        phase_cells=0.0,
        pass_index=2,
        pass_name="half-mu",
    )
    half_psi = _integrate_grid(
        **common,
        receiver_rho_order=selected_policy.rho_order,
        mu_order=selected_policy.mu_order,
        psi_count=selected_policy.psi_count // 2,
        phase_cells=0.0,
        pass_index=3,
        pass_name="half-psi",
    )
    phase_shifted = _integrate_grid(
        **common,
        receiver_rho_order=selected_policy.rho_order,
        mu_order=selected_policy.mu_order,
        psi_count=selected_policy.psi_count,
        phase_cells=0.5,
        pass_index=4,
        pass_name="phase-shifted",
    )
    estimates = (full, half_rho, half_mu, half_psi, phase_shifted)
    actual_direction_count = sum(estimate.direction_evaluations for estimate in estimates)
    if type(actual_direction_count) is not int or actual_direction_count != expected_directions:
        raise KerrReturningRadiationReceiverKernelError(
            "receiver direction work accounting disagrees with declared grids"
        )
    differences = tuple(
        _grid_difference(full, comparison, selected_policy)
        for comparison in (half_rho, half_mu, half_psi, phase_shifted)
    )
    convergence = KerrReceiverReturningRadiationKernelConvergence(
        differences[0],
        differences[1],
        differences[2],
        differences[3],
        all(item.converged for item in differences),
    )
    if not convergence.converged:
        raise KerrReturningRadiationReceiverKernelConvergenceError(
            "receiver K failed full/half-rho, full/half-mu, full/half-psi, "
            "or periodic-phase convergence"
        )
    representative = tuple(
        0.5 * math.fsum((inner, outer))
        for inner, outer in zip(edges, edges[1:])
    )
    descriptor = {
        "annuli": {
            "edgesOverMass": edges,
            "representativeRadiusPolicy": "arithmetic edge midpoint",
            "representativeRadiiOverMass": representative,
        },
        "area": {
            "measure": "receiver actual-face comoving proper area",
            "quadraturePolicy": dict(selected_area_policy.descriptor()),
            "upperAnnulusAreaDescriptorSha256": upper_area_hashes,
            "upperAnnulusAreasOverMassSquared": upper_areas,
            "lowerAnnulusAreaDescriptorSha256": lower_area_hashes,
            "lowerAnnulusAreasOverMassSquared": lower_areas,
        },
        "capabilities": dict(SCIENTIFIC_STATUS),
        "coefficient": {
            "equation": "K=(1/pi) integral mu_i*g^4*D20(mu_e) dOmega_i",
            "directionWeight": "2*w_mu/N_psi",
            "matrixIndexOrder": "K[receiverAnnulus][sourceAnnulus]",
            "surfaceBlockOrder": {
                "LL": "lower receiver, lower source",
                "LU": "lower receiver, upper source",
                "UL": "upper receiver, lower source",
                "UU": "upper receiver, upper source",
            },
            "receiverAreaAverage": "sum (dA_i/A_i) local-sky K",
            "fineCoarseSourceTopologyGate": (
                "every source direction must land in the same user annulus for "
                "independent fine and coarse whole rays"
            ),
            "coarseSourceRadiusProvenance": (
                "public revalidated exact-tree-bound coarse_ray; require its "
                "unique first terminal multi-surface entry to match the fine "
                "source outcome/face/target, reconstruct rho with the public "
                "Kerr oblate transform, and cross-check the primitive's "
                "published source-radius difference before binning"
            ),
            "pastWorldtubeContribution": 0,
        },
        "convergence": {
            "actual": asdict(convergence),
            "absoluteTolerance": selected_policy.absolute_tolerance,
            "relativeTolerance": selected_policy.relative_tolerance,
            "matrixRowsAndSkyFractionsGated": True,
            "fineCoarseSourceBinTopologyVerified": True,
            "rigorousErrorBound": False,
        },
        "implementationId": IMPLEMENTATION_ID,
        "modelOwnership": {
            "calibration": asdict(surface.calibration),
            "metric": asdict(surface.metric),
            "termination": asdict(termination),
        },
        "numericalPolicy": {
            "kernel": asdict(selected_policy),
            "fineRayOptions": asdict(ray_options),
            "fineSurfaceOptions": asdict(surface_options),
            "coarseRayOptions": (
                None if coarse_ray_options is None else asdict(coarse_ray_options)
            ),
            "coarseSurfaceOptions": (
                None
                if coarse_surface_options is None
                else asdict(coarse_surface_options)
            ),
        },
        "quadrature": {
            "fullReceiverRhoOrderPerCell": selected_policy.rho_order,
            "fullMuOrder": selected_policy.mu_order,
            "fullPsiCount": selected_policy.psi_count,
            "periodicPhaseShiftCells": 0.5,
            "muDomain": "incoming receiver hemisphere mu_i in (0,1)",
        },
        "result": {
            "matrices": {
                "LL": full.ll,
                "LU": full.lu,
                "UL": full.ul,
                "UU": full.uu,
            },
            "receiverRowTotals": {
                "lower": full.lower_row_totals,
                "upper": full.upper_row_totals,
            },
            "receiverRowClosureResiduals": {
                "lower": full.lower_row_closure_residuals,
                "upper": full.upper_row_closure_residuals,
            },
            "receiverSkySourceFractions": {
                "measure": "(mu_i/pi) dOmega_i",
                "lower": tuple(asdict(item) for item in full.lower_sky_fractions),
                "upper": tuple(asdict(item) for item in full.upper_sky_fractions),
            },
        },
        "sampleAuditSha256": {
            "full": full.sample_audit_sha256,
            "halfReceiverRho": half_rho.sample_audit_sha256,
            "halfMu": half_mu.sample_audit_sha256,
            "halfPsi": half_psi.sample_audit_sha256,
            "phaseShifted": phase_shifted.sample_audit_sha256,
        },
        "workBudget": {
            "directionEvaluationsConsumed": actual_direction_count,
            "maximumDirectionEvaluations": selected_policy.maximum_direction_evaluations,
            "primitiveWholeRaysPerDirection": 2,
            "publicPrimitiveReplayWholeRaysPerDirection": 2,
            "wholeRayTracesConsumed": 4 * actual_direction_count,
            "maximumWholeRayTraces": selected_policy.maximum_whole_ray_traces,
        },
    }
    descriptor_json = _canonical_json(descriptor)
    result = object.__new__(KerrReceiverReturningRadiationKernel)
    values = (
        ("annulus_edges_over_mass", edges),
        ("annulus_representative_radii_over_mass", representative),
        ("upper_annulus_areas_over_mass_squared", upper_areas),
        ("lower_annulus_areas_over_mass_squared", lower_areas),
        ("upper_receiver_upper_source_coefficients", full.uu),
        ("upper_receiver_lower_source_coefficients", full.ul),
        ("lower_receiver_upper_source_coefficients", full.lu),
        ("lower_receiver_lower_source_coefficients", full.ll),
        ("upper_receiver_returning_row_totals", full.upper_row_totals),
        ("lower_receiver_returning_row_totals", full.lower_row_totals),
        ("upper_receiver_row_closure_residuals", full.upper_row_closure_residuals),
        ("lower_receiver_row_closure_residuals", full.lower_row_closure_residuals),
        ("upper_receiver_sky_source_fractions", full.upper_sky_fractions),
        ("lower_receiver_sky_source_fractions", full.lower_sky_fractions),
        ("convergence", convergence),
        ("fine_coarse_source_bin_topology_verified", True),
        ("direction_evaluations_consumed", actual_direction_count),
        ("whole_ray_traces_consumed", 4 * actual_direction_count),
        ("full_grid_sample_audit_sha256", full.sample_audit_sha256),
        ("half_receiver_rho_sample_audit_sha256", half_rho.sample_audit_sha256),
        ("half_mu_sample_audit_sha256", half_mu.sample_audit_sha256),
        ("half_psi_sample_audit_sha256", half_psi.sample_audit_sha256),
        ("phase_shifted_sample_audit_sha256", phase_shifted.sample_audit_sha256),
        ("upper_annulus_area_descriptor_sha256", upper_area_hashes),
        ("lower_annulus_area_descriptor_sha256", lower_area_hashes),
        ("surface", surface),
        ("termination", termination),
        ("ray_options", ray_options),
        ("surface_options", surface_options),
        ("coarse_ray_options", coarse_ray_options),
        ("coarse_surface_options", coarse_surface_options),
        ("policy", selected_policy),
        ("area_policy", selected_area_policy),
        ("_descriptor_json", descriptor_json),
        (
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        ),
    )
    for name, value in values:
        object.__setattr__(result, name, value)
    return result


def integrate_kerr_returning_radiation_receiver_energy_kernel(
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
) -> KerrReceiverReturningRadiationKernel:
    """Integrate directly; every direction uses the production ray tracer."""

    return (
        _integrate_kerr_returning_radiation_receiver_energy_kernel_with_transport_provider(
            surface,
            termination=termination,
            annulus_edges_over_mass=annulus_edges_over_mass,
            ray_options=ray_options,
            surface_options=surface_options,
            coarse_ray_options=coarse_ray_options,
            coarse_surface_options=coarse_surface_options,
            policy=policy,
            area_policy=area_policy,
            direction_transport_provider=None,
        )
    )


def verify_kerr_returning_radiation_receiver_energy_kernel(
    result: KerrReceiverReturningRadiationKernel,
) -> None:
    """Replay every receiver area and sky direction and compare exact fields."""

    if type(result) is not KerrReceiverReturningRadiationKernel:
        raise TypeError("result must be exact KerrReceiverReturningRadiationKernel")
    descriptor_json = object.__getattribute__(result, "_descriptor_json")
    descriptor_sha = object.__getattribute__(result, "_descriptor_sha256")
    if type(descriptor_json) is not str or type(descriptor_sha) is not str:
        raise KerrReturningRadiationReceiverKernelVerificationError(
            "receiver-kernel descriptor identity has non-exact type"
        )
    try:
        parsed = json.loads(descriptor_json)
    except (json.JSONDecodeError, TypeError) as error:
        raise KerrReturningRadiationReceiverKernelVerificationError(
            "receiver-kernel descriptor is malformed"
        ) from error
    if _canonical_json(parsed).encode("utf-8") != descriptor_json.encode("utf-8"):
        raise KerrReturningRadiationReceiverKernelVerificationError(
            "receiver-kernel descriptor is not canonical"
        )
    expected_sha = hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest()
    if expected_sha.encode("ascii") != descriptor_sha.encode("ascii"):
        raise KerrReturningRadiationReceiverKernelVerificationError(
            "receiver-kernel descriptor SHA-256 is inconsistent"
        )
    rebuilt = integrate_kerr_returning_radiation_receiver_energy_kernel(
        result.surface,
        termination=result.termination,
        annulus_edges_over_mass=result.annulus_edges_over_mass,
        ray_options=result.ray_options,
        surface_options=result.surface_options,
        coarse_ray_options=result.coarse_ray_options,
        coarse_surface_options=result.coarse_surface_options,
        policy=result.policy,
        area_policy=result.area_policy,
    )
    _require_trusted_exact_tree(result, rebuilt, "result")


@dataclass(frozen=True, slots=True)
class KerrForwardReceiverBlockDifference:
    maximum_absolute_difference: float
    maximum_scaled_difference: float
    converged: bool

    def __post_init__(self) -> None:
        absolute = _exact_finite_float(
            self.maximum_absolute_difference,
            "maximum_absolute_difference",
        )
        scaled = _exact_finite_float(
            self.maximum_scaled_difference,
            "maximum_scaled_difference",
        )
        if absolute < 0.0 or scaled < 0.0:
            raise ValueError("comparison differences must be non-negative")
        if type(self.converged) is not bool or self.converged is not (scaled <= 1.0):
            raise ValueError("comparison convergence flag disagrees with scaled gate")


@dataclass(frozen=True, slots=True, init=False)
class KerrForwardReceiverKernelComparison:
    upper_receiver_upper_source: KerrForwardReceiverBlockDifference
    upper_receiver_lower_source: KerrForwardReceiverBlockDifference
    lower_receiver_upper_source: KerrForwardReceiverBlockDifference
    lower_receiver_lower_source: KerrForwardReceiverBlockDifference
    upper_source_column_energy: KerrForwardReceiverBlockDifference
    lower_source_column_energy: KerrForwardReceiverBlockDifference
    receiver_reconstructed_upper_source_g2_columns: tuple[float, ...]
    receiver_reconstructed_lower_source_g2_columns: tuple[float, ...]
    maximum_absolute_difference: float
    maximum_scaled_difference: float
    converged: bool
    absolute_tolerance: float
    relative_tolerance: float
    shares_exact_kerr_geodesic_code_family: bool
    is_independent_geodesic_oracle: bool
    forward_kernel_descriptor_sha256: str
    receiver_kernel_descriptor_sha256: str
    _forward: KerrForwardReturningRadiationKernel
    _receiver: KerrReceiverReturningRadiationKernel
    _descriptor_json: str
    _descriptor_sha256: str

    def __init__(self) -> None:
        raise TypeError("comparison is built only by compare_kerr_returning_radiation_kernels")

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        return json.loads(self._descriptor_json)

    def revalidate(self) -> None:
        verify_kerr_returning_radiation_kernel_comparison(self)


def _block_difference(
    forward: tuple[tuple[float, ...], ...],
    receiver: tuple[tuple[float, ...], ...],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> KerrForwardReceiverBlockDifference:
    forward_vector = tuple(value for row in forward for value in row)
    receiver_vector = tuple(value for row in receiver for value in row)
    absolute, scaled = _vector_difference(
        forward_vector,
        receiver_vector,
        absolute_tolerance,
        relative_tolerance,
    )
    return KerrForwardReceiverBlockDifference(absolute, scaled, scaled <= 1.0)


def _receiver_reconstructed_g2_columns(
    receiver: KerrReceiverReturningRadiationKernel,
    source_face: str,
) -> tuple[float, ...]:
    count = receiver.annulus_count
    source_areas = (
        receiver.upper_annulus_areas_over_mass_squared
        if source_face == UPPER
        else receiver.lower_annulus_areas_over_mass_squared
    )
    upper_block = (
        receiver.upper_receiver_upper_source_coefficients
        if source_face == UPPER
        else receiver.upper_receiver_lower_source_coefficients
    )
    lower_block = (
        receiver.lower_receiver_upper_source_coefficients
        if source_face == UPPER
        else receiver.lower_receiver_lower_source_coefficients
    )
    return tuple(
        math.fsum(
            (
                *(
                    receiver.upper_annulus_areas_over_mass_squared[i]
                    * upper_block[i][j]
                    / source_areas[j]
                    for i in range(count)
                ),
                *(
                    receiver.lower_annulus_areas_over_mass_squared[i]
                    * lower_block[i][j]
                    / source_areas[j]
                    for i in range(count)
                ),
            )
        )
        for j in range(count)
    )


def compare_kerr_returning_radiation_kernels(
    forward: KerrForwardReturningRadiationKernel,
    receiver: KerrReceiverReturningRadiationKernel,
    *,
    absolute_tolerance: float = 2.0e-2,
    relative_tolerance: float = 5.0e-2,
) -> KerrForwardReceiverKernelComparison:
    """Compare all four K blocks; this is not an independent geodesic oracle."""

    if type(forward) is not KerrForwardReturningRadiationKernel:
        raise TypeError("forward must be exact KerrForwardReturningRadiationKernel")
    if type(receiver) is not KerrReceiverReturningRadiationKernel:
        raise TypeError("receiver must be exact KerrReceiverReturningRadiationKernel")
    absolute = _exact_finite_float(absolute_tolerance, "absolute_tolerance")
    relative = _exact_finite_float(relative_tolerance, "relative_tolerance")
    if (
        absolute <= 0.0
        or relative <= 0.0
        or absolute > _MAXIMUM_TOLERANCE
        or relative > _MAXIMUM_TOLERANCE
    ):
        raise ValueError("comparison tolerances must lie in (0, 0.25]")
    verify_kerr_returning_radiation_energy_kernel(forward)
    verify_kerr_returning_radiation_receiver_energy_kernel(receiver)
    try:
        _require_trusted_exact_tree(
            forward.surface,
            receiver.surface,
            "comparison.surface",
        )
        _require_trusted_exact_tree(
            forward.termination,
            receiver.termination,
            "comparison.termination",
        )
    except KerrReturningRadiationReceiverKernelVerificationError as error:
        raise ValueError(
            "forward and receiver kernels must use the same exact physical model"
        ) from error
    if len(forward.annulus_edges_over_mass) != len(receiver.annulus_edges_over_mass) or any(
        left.hex() != right.hex()
        for left, right in zip(
            forward.annulus_edges_over_mass,
            receiver.annulus_edges_over_mass,
        )
    ):
        raise ValueError("forward and receiver kernels require identical annulus edges")
    pairs = (
        (
            forward.upper_receiver_upper_emitter_coefficients,
            receiver.upper_receiver_upper_source_coefficients,
        ),
        (
            forward.upper_receiver_lower_emitter_coefficients,
            receiver.upper_receiver_lower_source_coefficients,
        ),
        (
            forward.lower_receiver_upper_emitter_coefficients,
            receiver.lower_receiver_upper_source_coefficients,
        ),
        (
            forward.lower_receiver_lower_emitter_coefficients,
            receiver.lower_receiver_lower_source_coefficients,
        ),
    )
    differences = tuple(
        _block_difference(left, right, absolute, relative) for left, right in pairs
    )
    receiver_upper_g2 = _receiver_reconstructed_g2_columns(receiver, UPPER)
    receiver_lower_g2 = _receiver_reconstructed_g2_columns(receiver, LOWER)
    upper_column_difference = _block_difference(
        (forward.upper_emitter_g2_returned_power_columns,),
        (receiver_upper_g2,),
        absolute,
        relative,
    )
    lower_column_difference = _block_difference(
        (forward.lower_emitter_g2_returned_power_columns,),
        (receiver_lower_g2,),
        absolute,
        relative,
    )
    all_differences = (*differences, upper_column_difference, lower_column_difference)
    maximum_absolute = max(
        item.maximum_absolute_difference for item in all_differences
    )
    maximum_scaled = max(item.maximum_scaled_difference for item in all_differences)
    converged = all(item.converged for item in all_differences)
    descriptor = {
        "absoluteTolerance": absolute,
        "capabilities": {
            "sharesExactKerrGeodesicCodeFamily": True,
            "isIndependentGeodesicOracle": False,
            "rigorousContinuumErrorBound": False,
        },
        "columnEnergyDiagnostic": (
            "C_Sj=(1/A_Sj) sum_Ri A_Ri K_RSij compared with the forward "
            "direct g^2 returned-power column"
        ),
        "differences": {
            "UU": asdict(differences[0]),
            "UL": asdict(differences[1]),
            "LU": asdict(differences[2]),
            "LL": asdict(differences[3]),
            "upperSourceColumnEnergy": asdict(upper_column_difference),
            "lowerSourceColumnEnergy": asdict(lower_column_difference),
            "receiverReconstructedUpperSourceG2Columns": receiver_upper_g2,
            "receiverReconstructedLowerSourceG2Columns": receiver_lower_g2,
            "maximumAbsolute": maximum_absolute,
            "maximumScaled": maximum_scaled,
            "converged": converged,
        },
        "forwardKernelDescriptorSha256": forward.model_descriptor_sha256,
        "implementationId": f"{IMPLEMENTATION_ID}/forward-comparison/v1",
        "receiverKernelDescriptorSha256": receiver.model_descriptor_sha256,
        "relativeTolerance": relative,
    }
    descriptor_json = _canonical_json(descriptor)
    result = object.__new__(KerrForwardReceiverKernelComparison)
    values = (
        ("upper_receiver_upper_source", differences[0]),
        ("upper_receiver_lower_source", differences[1]),
        ("lower_receiver_upper_source", differences[2]),
        ("lower_receiver_lower_source", differences[3]),
        ("upper_source_column_energy", upper_column_difference),
        ("lower_source_column_energy", lower_column_difference),
        ("receiver_reconstructed_upper_source_g2_columns", receiver_upper_g2),
        ("receiver_reconstructed_lower_source_g2_columns", receiver_lower_g2),
        ("maximum_absolute_difference", maximum_absolute),
        ("maximum_scaled_difference", maximum_scaled),
        ("converged", converged),
        ("absolute_tolerance", absolute),
        ("relative_tolerance", relative),
        ("shares_exact_kerr_geodesic_code_family", True),
        ("is_independent_geodesic_oracle", False),
        ("forward_kernel_descriptor_sha256", forward.model_descriptor_sha256),
        ("receiver_kernel_descriptor_sha256", receiver.model_descriptor_sha256),
        ("_forward", forward),
        ("_receiver", receiver),
        ("_descriptor_json", descriptor_json),
        (
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        ),
    )
    for name, value in values:
        object.__setattr__(result, name, value)
    return result


def verify_kerr_returning_radiation_kernel_comparison(
    result: KerrForwardReceiverKernelComparison,
) -> None:
    if type(result) is not KerrForwardReceiverKernelComparison:
        raise TypeError("result must be exact KerrForwardReceiverKernelComparison")
    descriptor_json = object.__getattribute__(result, "_descriptor_json")
    descriptor_sha = object.__getattribute__(result, "_descriptor_sha256")
    if type(descriptor_json) is not str or type(descriptor_sha) is not str:
        raise KerrReturningRadiationReceiverKernelVerificationError(
            "comparison descriptor identity has non-exact type"
        )
    try:
        parsed = json.loads(descriptor_json)
    except (json.JSONDecodeError, TypeError) as error:
        raise KerrReturningRadiationReceiverKernelVerificationError(
            "comparison descriptor is malformed"
        ) from error
    if (
        _canonical_json(parsed).encode("utf-8") != descriptor_json.encode("utf-8")
        or hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest().encode("ascii")
        != descriptor_sha.encode("ascii")
    ):
        raise KerrReturningRadiationReceiverKernelVerificationError(
            "comparison descriptor identity is stale"
        )
    rebuilt = compare_kerr_returning_radiation_kernels(
        result._forward,
        result._receiver,
        absolute_tolerance=result.absolute_tolerance,
        relative_tolerance=result.relative_tolerance,
    )
    _require_trusted_exact_tree(result, rebuilt, "comparison")


__all__ = (
    "IMPLEMENTATION_ID",
    "PAST_WORLDTUBE_NO_SOURCE",
    "SCIENTIFIC_STATUS",
    "KerrForwardReceiverBlockDifference",
    "KerrForwardReceiverKernelComparison",
    "KerrReceiverReturningRadiationGridDifference",
    "KerrReceiverReturningRadiationKernel",
    "KerrReceiverReturningRadiationKernelConvergence",
    "KerrReturningRadiationReceiverKernelConvergenceError",
    "KerrReturningRadiationReceiverKernelError",
    "KerrReturningRadiationReceiverKernelVerificationError",
    "ReceiverSkySourceFractions",
    "compare_kerr_returning_radiation_kernels",
    "integrate_kerr_returning_radiation_receiver_energy_kernel",
    "verify_kerr_returning_radiation_kernel_comparison",
    "verify_kerr_returning_radiation_receiver_energy_kernel",
)
