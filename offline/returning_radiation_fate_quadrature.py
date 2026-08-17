"""Emitter-local bolometric emitted-flux fate quadrature.

For one authenticated finite-thickness Kerr photosphere emitter, this module
integrates the *outgoing local flux measure*

``dF / F = 2 mu f(mu) dmu dpsi / (2 pi)``

over the outward hemisphere.  ``f(mu) = 1/2 + 3 mu/4`` is the built-in,
flux-conserving KERRBB D20 angular prescription.  Gauss--Legendre nodes cover
``mu in (0, 1)`` and an exactly periodic midpoint rule covers ``psi``.  Every
node is classified by the independently certified single-direction primitive
in :mod:`offline.kerr_returning_radiation_rays`.

The result is an emitter-local *energy-flux fate fraction*: return, capture,
escape, or entry into the declared plunge sink.  It is not a photon-number
probability, a receiver-centred ``F_in`` or ``K`` coefficient, and it contains
no receiver area/solid-angle Jacobian.  In particular, neither the returned
ray's frequency shift ``g`` nor receiver incidence cosine ``mu_i`` is used as
a Jacobian here.  This module cannot produce an energy kernel and does not
include the KERRBB returning-radiation stress/work term ``F_S``.

The reported ``N_mu``/``N_psi`` grid is compared independently with
``N_mu/2`` at fixed ``N_psi``, with ``N_psi/2`` at fixed ``N_mu``, and with a
half-cell phase shift of the full periodic grid.  This is a finite,
caller-declared absolute convergence gate; it is not an asymptotic error
proof.  A hard whole-ray budget includes both the fine/coarse rays inside each
primitive and the mandatory public replay before a primitive is consumed.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Final, Literal, Mapping

from offline.disk_atmosphere import (
    FluxConservingLinearLimbDarkening,
    IsotropicAngularEmission,
)
from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.kerr import KerrOblateTermination
from offline.kerr_finite_thickness import LOWER, UPPER
from offline.kerr_finite_thickness_launch import (
    KerrFiniteThicknessEmissionLaunch,
    KerrFiniteThicknessSurfaceFrame,
)
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
from offline.kerr_returning_radiation_rays import (
    KerrReturningRadiationRayPrimitive,
    trace_kerr_returning_radiation_direction,
    verify_kerr_returning_radiation_direction,
)


IMPLEMENTATION_ID: Final = "kerr-emitter-local-bolometric-fate-quadrature/v1"
D20_QUADRATURE_IMPLEMENTATION_ID: Final = (
    "kerrbb-d20-emitted-flux-gauss-periodic/v1"
)
KERRBB_SOURCE_URL: Final = "https://arxiv.org/abs/astro-ph/0411583"
_D20_IMPLEMENTATION_ID: Final = "flux-conserving-linear-limb-darkening/v1"
_FATES: Final = (
    "return-upper",
    "return-lower",
    "captured",
    "escaped",
    "plunge-sink",
)
_MAXIMUM_WHOLE_RAY_BUDGET: Final = 16_384
_MAXIMUM_CONVERGENCE_TOLERANCE: Final = 0.1
_GAUSS_MAXIMUM_ITERATIONS: Final = 64

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "finite-grid point estimate of emitter-local one-face bolometric "
            "emitted-flux fate fractions"
        ),
        "implementationId": IMPLEMENTATION_ID,
        "primarySource": KERRBB_SOURCE_URL,
        "primarySourceCitation": "Li et al. 2005, Appendix D, equation D20",
        "measure": "dF/F = 2 mu f(mu) dmu dpsi/(2 pi)",
        "angularLaw": "KERRBB D20 f(mu)=1/2+3 mu/4",
        "directionPrimitive": (
            "finite-thickness-kerr-returning-ray/v1; mandatory public replay "
            "before fate consumption"
        ),
        "receiverBinSemantics": (
            "diagnostic partition of returned emitter-local flux only; bin "
            "closure residual is published and bounded at float64 roundoff, "
            "not claimed bit-exact"
        ),
        "uncertaintySemantics": (
            "published N versus N/2 mu and psi differences and periodic-phase "
            "difference are finite-grid diagnostics, not rigorous error bounds"
        ),
        "isPhotonNumberProbability": False,
        "isReceiverCentredIncidentFlux": False,
        "isIndependentRayKernel": False,
        "outputsReturningRadiationKernelK": False,
        "usesFrequencyShiftAsJacobian": False,
        "usesReceiverIncidenceCosineAsJacobian": False,
        "includesReturningRadiationStressWorkFS": False,
        "includesSpectralRedistribution": False,
        "isCompleteKerrbb": False,
        "prohibitedClaim": (
            "Do not describe these emitter-local energy-fate fractions as "
            "photon-number probabilities, receiver F_in, a K coefficient, "
            "complete KERRBB, or the F_S stress/work term."
        ),
    }
)

D20_EMITTED_FLUX_QUADRATURE_DESCRIPTOR: Final[Mapping[str, Any]] = (
    MappingProxyType(
        {
            "angularLaw": "KERRBB D20 f(mu)=1/2+3 mu/4",
            "fluxMeasure": "2 mu f(mu) dmu dpsi/(2 pi)",
            "implementationId": D20_QUADRATURE_IMPLEMENTATION_ID,
            "muRule": "float64 Gauss-Legendre on open interval (0,1)",
            "normalization": (
                "analytic unit flux; only float64 roundoff residual closure allowed"
            ),
            "psiRule": "periodic equal-weight midpoint with phase in cell units",
            "source": KERRBB_SOURCE_URL,
        }
    )
)


class KerrReturningRadiationFateQuadratureError(RuntimeError):
    """Raised when fate quadrature provenance or closure fails closed."""


class KerrReturningRadiationFateConvergenceError(
    KerrReturningRadiationFateQuadratureError
):
    """Raised when resolution or periodic-phase agreement is insufficient."""


class KerrReturningRadiationFateVerificationError(
    KerrReturningRadiationFateQuadratureError
):
    """Raised when a public result cannot be reproduced exactly."""


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
        raise KerrReturningRadiationFateQuadratureError(
            "fate-quadrature descriptor is not finite canonical JSON"
        ) from error


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _exact_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive exact int")
    return value


def _exact_finite_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite exact float")
    return value


def _trusted_attribute(value: Any, name: str, path: str) -> Any:
    try:
        return object.__getattribute__(value, name)
    except (AttributeError, TypeError) as error:
        raise KerrReturningRadiationFateVerificationError(
            f"{path}.{name} is missing"
        ) from error


def _require_trusted_exact_tree(actual: Any, expected: Any, path: str) -> None:
    """Compare trusted reconstructed data without untrusted equality hooks."""

    if type(actual) is not type(expected):
        raise KerrReturningRadiationFateVerificationError(
            f"{path} has non-exact type {type(actual).__name__}; "
            f"expected {type(expected).__name__}"
        )
    if is_dataclass(expected) and not isinstance(expected, type):
        for field in fields(expected):
            _require_trusted_exact_tree(
                _trusted_attribute(actual, field.name, path),
                _trusted_attribute(expected, field.name, path),
                f"{path}.{field.name}",
            )
        return
    if type(expected) is tuple:
        if len(actual) != len(expected):
            raise KerrReturningRadiationFateVerificationError(
                f"{path} tuple length differs"
            )
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
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
        raise KerrReturningRadiationFateVerificationError(
            f"{path} has unsupported trusted type {type(expected).__name__}"
        )
    if differs:
        raise KerrReturningRadiationFateVerificationError(
            f"{path} differs from the trusted replay"
        )


@dataclass(frozen=True, slots=True)
class EmittedFluxDirectionNode:
    """One deterministic local direction and normalized flux weight."""

    emission_angle_cosine: float
    tangent_azimuth_rad: float
    normalized_emitted_flux_weight: float

    def __post_init__(self) -> None:
        for name in (
            "emission_angle_cosine",
            "tangent_azimuth_rad",
            "normalized_emitted_flux_weight",
        ):
            _exact_finite_float(getattr(self, name), name)
        if not 0.0 < self.emission_angle_cosine < 1.0:
            raise ValueError("quadrature mu must lie strictly inside (0, 1)")
        if not 0.0 <= self.tangent_azimuth_rad < 2.0 * math.pi:
            raise ValueError("quadrature psi must lie in [0, 2 pi)")
        if self.normalized_emitted_flux_weight <= 0.0:
            raise ValueError("quadrature flux weight must be positive")


def _gauss_legendre_unit_interval(order: int) -> tuple[tuple[float, float], ...]:
    """Return deterministic float64 Gauss--Legendre ``(node, weight)`` pairs."""

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
            if order == 1:
                current = root
                previous = 1.0
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
                # Binary64 Newton iteration can settle into a short finite
                # cycle around an otherwise resolved root.  Pick the visited
                # endpoint with the smallest directly evaluated residual;
                # non-cycling orders retain their historical bit patterns.
                root, _residual, derivative = min(
                    visited_iterates[cycle_start:],
                    key=lambda item: (abs(item[1]), item[0]),
                )
                break
            root = next_root
        else:
            raise KerrReturningRadiationFateQuadratureError(
                "Gauss-Legendre root solve did not converge"
            )
        weight = 1.0 / ((1.0 - root * root) * derivative * derivative)
        lower = 0.5 * (1.0 - root)
        upper = 0.5 * (1.0 + root)
        nodes[index] = lower
        nodes[order - 1 - index] = upper
        weights[index] = weight
        weights[order - 1 - index] = weight
    result = tuple(zip(nodes, weights))
    if any(
        not (0.0 < node < 1.0 and math.isfinite(weight) and weight > 0.0)
        for node, weight in result
    ):
        raise KerrReturningRadiationFateQuadratureError(
            "Gauss-Legendre rule is not finite and interior-positive"
        )
    if abs(math.fsum(weight for _node, weight in result) - 1.0) > 64.0 * math.ulp(1.0):
        raise KerrReturningRadiationFateQuadratureError(
            "Gauss-Legendre weights do not integrate a constant"
        )
    return result


def _emitted_flux_nodes(
    mu_order: int,
    psi_count: int,
    *,
    phase_cells: float,
    angular_law: FluxConservingLinearLimbDarkening | IsotropicAngularEmission,
) -> tuple[EmittedFluxDirectionNode, ...]:
    """Build a strictly normalized tensor-product local flux rule.

    ``angular_law`` is deliberately limited to the repository's two exact
    built-ins so analytic tests cannot introduce a duck-typed atmosphere.
    Production always supplies the exact KERRBB D20 instance.
    """

    mu_order = _exact_positive_int(mu_order, "mu_order")
    psi_count = _exact_positive_int(psi_count, "psi_count")
    phase = _exact_finite_float(phase_cells, "phase_cells")
    if type(angular_law) not in (
        FluxConservingLinearLimbDarkening,
        IsotropicAngularEmission,
    ):
        raise TypeError("angular_law must be an exact built-in angular law")
    if (
        type(angular_law) is FluxConservingLinearLimbDarkening
        and (
            angular_law.coefficient.hex() != float(1.5).hex()
            or angular_law.implementation_id.encode("utf-8")
            != _D20_IMPLEMENTATION_ID.encode("utf-8")
        )
    ):
        raise ValueError("production linear law must be the exact KERRBB D20 law")

    provisional: list[tuple[float, float, float]] = []
    two_pi = 2.0 * math.pi
    for mu, mu_weight in _gauss_legendre_unit_interval(mu_order):
        multiplier = angular_law.intensity_multiplier(mu)
        if type(multiplier) is not float or not math.isfinite(multiplier) or multiplier <= 0.0:
            raise KerrReturningRadiationFateQuadratureError(
                "angular law returned a non-exact or invalid multiplier"
            )
        direction_weight = 2.0 * mu_weight * mu * multiplier / psi_count
        for psi_index in range(psi_count):
            psi = ((psi_index + 0.5 + phase) * two_pi / psi_count) % two_pi
            provisional.append((mu, psi, direction_weight))

    raw_sum = math.fsum(weight for _mu, _psi, weight in provisional)
    normalization_tolerance = 128.0 * math.ulp(1.0)
    if (
        not math.isfinite(raw_sum)
        or raw_sum <= 0.0
        or abs(raw_sum - 1.0) > normalization_tolerance
    ):
        raise KerrReturningRadiationFateQuadratureError(
            "built-in angular-law quadrature does not reproduce analytic unit "
            "emitted flux within the float64 roundoff gate"
        )
    # D20 and isotropic flux normalization are analytic facts.  Never hide
    # quadrature under-resolution by dividing through an inaccurate raw sum
    # (for example, one D20 mu node gives 0.875).  Only close the already
    # roundoff-level residual so the one-hot public partition sums exactly.
    normalized = [weight for _mu, _psi, weight in provisional]
    original_last_weight = normalized[-1]
    for _closure_iteration in range(16):
        current_total = math.fsum(normalized)
        if current_total == 1.0:
            break
        candidate = normalized[-1] + (1.0 - current_total)
        if candidate == normalized[-1]:
            candidate = math.nextafter(
                normalized[-1],
                math.inf if current_total < 1.0 else -math.inf,
            )
        normalized[-1] = candidate
    correction = abs(normalized[-1] - original_last_weight)
    if math.fsum(normalized) != 1.0 or any(weight <= 0.0 for weight in normalized):
        raise KerrReturningRadiationFateQuadratureError(
            "quadrature flux weights cannot be normalized exactly"
        )
    if correction > normalization_tolerance:
        raise KerrReturningRadiationFateQuadratureError(
            "quadrature exact closure exceeded the float64 roundoff gate"
        )
    return tuple(
        EmittedFluxDirectionNode(mu, psi, weight)
        for (mu, psi, _raw), weight in zip(provisional, normalized)
    )


def kerrbb_d20_emitted_flux_direction_nodes(
    mu_order: int,
    psi_count: int,
    *,
    phase_cells: float = 0.0,
) -> tuple[EmittedFluxDirectionNode, ...]:
    """Return the stable public KERRBB-D20 local emitted-flux direction rule.

    This narrow helper is shared by fate integration and future energy-kernel
    geometry.  It owns the D20 formula, exact built-in provenance, periodic
    phase convention, and hard analytic-normalization gate so consumers never
    import a private helper or duplicate emission weights.
    """

    return _emitted_flux_nodes(
        mu_order,
        psi_count,
        phase_cells=phase_cells,
        angular_law=FluxConservingLinearLimbDarkening(),
    )


@dataclass(frozen=True, slots=True)
class _DirectionClassification:
    fate: str
    receiver_radius_over_mass: float | None
    primitive_descriptor_sha256: str

    def __post_init__(self) -> None:
        if type(self.fate) is not str or self.fate not in _FATES:
            raise ValueError("direction classification has an unsupported fate")
        if self.fate.startswith("return-"):
            _exact_finite_float(
                self.receiver_radius_over_mass,
                "receiver_radius_over_mass",
            )
            if self.receiver_radius_over_mass <= 0.0:
                raise ValueError("returned direction radius must be positive")
        elif self.receiver_radius_over_mass is not None:
            raise ValueError("non-returning direction cannot own a receiver radius")
        if (
            type(self.primitive_descriptor_sha256) is not str
            or len(self.primitive_descriptor_sha256) != 64
        ):
            raise ValueError("direction classification needs a SHA-256 identity")
        try:
            bytes.fromhex(self.primitive_descriptor_sha256)
        except ValueError as error:
            raise ValueError(
                "direction classification SHA-256 must be lowercase hexadecimal"
            ) from error
        if self.primitive_descriptor_sha256 != self.primitive_descriptor_sha256.lower():
            raise ValueError(
                "direction classification SHA-256 must be lowercase hexadecimal"
            )


@dataclass(frozen=True, slots=True)
class EmittedFluxDirectionAudit:
    emission_angle_cosine: float
    tangent_azimuth_rad: float
    normalized_emitted_flux_weight: float
    fate: str
    receiver_radius_over_mass: float | None
    primitive_descriptor_sha256: str


@dataclass(frozen=True, slots=True)
class EmittedFluxFateFractions:
    """One normalized local emitted-flux partition, excluding all ``g`` factors."""

    return_upper: float
    return_lower: float
    captured: float
    escaped: float
    plunge_sink: float

    @property
    def returning(self) -> float:
        return math.fsum((self.return_upper, self.return_lower))

    @property
    def total(self) -> float:
        return math.fsum(
            (
                self.return_upper,
                self.return_lower,
                self.captured,
                self.escaped,
                self.plunge_sink,
            )
        )

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.return_upper,
            self.return_lower,
            self.captured,
            self.escaped,
            self.plunge_sink,
        )

    def __post_init__(self) -> None:
        for name in (
            "return_upper",
            "return_lower",
            "captured",
            "escaped",
            "plunge_sink",
        ):
            value = _exact_finite_float(getattr(self, name), name)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.total != 1.0:
            raise ValueError("emitted-flux fate fractions must sum exactly to one")


@dataclass(frozen=True, slots=True)
class _GridIntegration:
    fractions: EmittedFluxFateFractions
    return_upper_by_receiver_bin: tuple[float, ...]
    return_lower_by_receiver_bin: tuple[float, ...]
    return_upper_bin_closure_residual: float
    return_lower_bin_closure_residual: float
    direction_audit: tuple[EmittedFluxDirectionAudit, ...]

    def convergence_vector(self) -> tuple[float, ...]:
        return (
            *self.fractions.as_tuple(),
            *self.return_upper_by_receiver_bin,
            *self.return_lower_by_receiver_bin,
        )


def _receiver_bin_index(radius: float, edges: tuple[float, ...]) -> int:
    index = bisect_right(edges, radius) - 1
    if index == len(edges) - 1 and radius.hex() == edges[-1].hex():
        index -= 1
    if index < 0 or index >= len(edges) - 1:
        raise KerrReturningRadiationFateQuadratureError(
            "returned ray lies outside the declared receiver bins"
        )
    return index


def _integrate_classification_grid(
    nodes: tuple[EmittedFluxDirectionNode, ...],
    receiver_bin_edges: tuple[float, ...],
    classifier: Callable[[float, float], _DirectionClassification],
) -> _GridIntegration:
    """Pure one-hot accumulator used by both production and analytic tests."""

    if type(nodes) is not tuple or not nodes:
        raise TypeError("nodes must be a non-empty exact tuple")
    if type(receiver_bin_edges) is not tuple or len(receiver_bin_edges) < 2:
        raise TypeError("receiver_bin_edges must be an exact tuple of edges")
    category_weights: dict[str, list[float]] = {fate: [] for fate in _FATES}
    upper_bins: list[list[float]] = [
        [] for _index in range(len(receiver_bin_edges) - 1)
    ]
    lower_bins: list[list[float]] = [
        [] for _index in range(len(receiver_bin_edges) - 1)
    ]
    audit: list[EmittedFluxDirectionAudit] = []
    for node in nodes:
        if type(node) is not EmittedFluxDirectionNode:
            raise TypeError("nodes must contain exact EmittedFluxDirectionNode values")
        classification = classifier(
            node.emission_angle_cosine,
            node.tangent_azimuth_rad,
        )
        if type(classification) is not _DirectionClassification:
            raise TypeError("classifier must return exact direction classifications")
        weight = node.normalized_emitted_flux_weight
        category_weights[classification.fate].append(weight)
        if classification.fate == "return-upper":
            upper_bins[
                _receiver_bin_index(
                    classification.receiver_radius_over_mass,
                    receiver_bin_edges,
                )
            ].append(weight)
        elif classification.fate == "return-lower":
            lower_bins[
                _receiver_bin_index(
                    classification.receiver_radius_over_mass,
                    receiver_bin_edges,
                )
            ].append(weight)
        audit.append(
            EmittedFluxDirectionAudit(
                emission_angle_cosine=node.emission_angle_cosine,
                tangent_azimuth_rad=node.tangent_azimuth_rad,
                normalized_emitted_flux_weight=weight,
                fate=classification.fate,
                receiver_radius_over_mass=(
                    classification.receiver_radius_over_mass
                ),
                primitive_descriptor_sha256=(
                    classification.primitive_descriptor_sha256
                ),
            )
        )

    fractions_raw = [math.fsum(category_weights[fate]) for fate in _FATES]
    residual = 1.0 - math.fsum(fractions_raw)
    # The one-hot partition contains every strictly normalized direction once.
    # Apply only the final roundoff residual to the category owning the final
    # quadrature node; this makes the public closure exactly auditable.
    final_fate_index = _FATES.index(audit[-1].fate)
    fractions_raw[final_fate_index] += residual
    fractions = EmittedFluxFateFractions(*fractions_raw)
    upper = [math.fsum(entries) for entries in upper_bins]
    lower = [math.fsum(entries) for entries in lower_bins]
    upper_residual = abs(math.fsum(upper) - fractions.return_upper)
    lower_residual = abs(math.fsum(lower) - fractions.return_lower)
    closure_scale = max(
        1.0,
        fractions.return_upper,
        fractions.return_lower,
    )
    if (
        upper_residual > 16.0 * math.ulp(closure_scale)
        or lower_residual > 16.0 * math.ulp(closure_scale)
        or any(value < 0.0 for value in (*upper, *lower))
    ):
        raise KerrReturningRadiationFateQuadratureError(
            "receiver-bin diagnostic does not close against returned flux"
        )
    return _GridIntegration(
        fractions,
        tuple(upper),
        tuple(lower),
        upper_residual,
        lower_residual,
        tuple(audit),
    )


@dataclass(frozen=True, slots=True)
class KerrReturningRadiationFateConvergence:
    mu_resolution_maximum_absolute_difference: float
    psi_resolution_maximum_absolute_difference: float
    resolution_maximum_absolute_difference: float
    periodic_phase_maximum_absolute_difference: float
    absolute_tolerance: float
    converged: bool

    def __post_init__(self) -> None:
        for name in (
            "mu_resolution_maximum_absolute_difference",
            "psi_resolution_maximum_absolute_difference",
            "resolution_maximum_absolute_difference",
            "periodic_phase_maximum_absolute_difference",
            "absolute_tolerance",
        ):
            value = _exact_finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if type(self.converged) is not bool:
            raise TypeError("converged must be an exact bool")
        expected_resolution = max(
            self.mu_resolution_maximum_absolute_difference,
            self.psi_resolution_maximum_absolute_difference,
        )
        if self.resolution_maximum_absolute_difference.hex() != expected_resolution.hex():
            raise ValueError(
                "aggregate resolution difference disagrees with mu/psi gates"
            )
        expected = max(
            self.resolution_maximum_absolute_difference,
            self.periodic_phase_maximum_absolute_difference,
        ) <= self.absolute_tolerance
        if self.converged is not expected:
            raise ValueError("convergence flag disagrees with the declared gate")


@dataclass(frozen=True, slots=True, init=False)
class KerrReturningRadiationFateQuadrature:
    """Authenticated local emitted-flux fate partition at one emitter."""

    frame: KerrFiniteThicknessSurfaceFrame
    surface: KerrFiniteThicknessMultiSurface
    termination: KerrOblateTermination
    ray_options: RayTraceOptions
    surface_options: SurfaceEventOptions
    coarse_ray_options: RayTraceOptions | None
    coarse_surface_options: SurfaceEventOptions | None
    mu_order: int
    psi_count: int
    receiver_radius_bin_edges_over_mass: tuple[float, ...]
    convergence_absolute_tolerance: float
    maximum_whole_ray_traces: int
    whole_ray_traces_consumed: int
    fractions: EmittedFluxFateFractions
    return_upper_by_receiver_bin: tuple[float, ...]
    return_lower_by_receiver_bin: tuple[float, ...]
    convergence: KerrReturningRadiationFateConvergence
    full_grid_direction_audit: tuple[EmittedFluxDirectionAudit, ...]
    half_mu_grid_direction_audit: tuple[EmittedFluxDirectionAudit, ...]
    half_psi_grid_direction_audit: tuple[EmittedFluxDirectionAudit, ...]
    phase_shifted_direction_audit: tuple[EmittedFluxDirectionAudit, ...]
    _descriptor_json: str
    _descriptor_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "KerrReturningRadiationFateQuadrature is built only by the "
            "certified integrator"
        )

    @property
    def model_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def model_descriptor(self) -> Mapping[str, Any]:
        if type(self._descriptor_json) is not str:
            raise KerrReturningRadiationFateVerificationError(
                "quadrature descriptor has a non-exact type"
            )
        return json.loads(self._descriptor_json)

    def revalidate(self) -> None:
        verify_kerr_returning_radiation_fate_quadrature(self)


def _validated_receiver_edges(
    values: tuple[float, ...] | None,
    surface: KerrFiniteThicknessMultiSurface,
) -> tuple[float, ...]:
    calibration = surface.calibration
    if values is None:
        return (
            float(calibration.isco_radius_over_mass),
            float(calibration.outer_radius_over_mass),
        )
    if type(values) is not tuple or len(values) < 2:
        raise TypeError("receiver radius bin edges must be an exact tuple")
    edges = tuple(
        _exact_finite_float(value, f"receiver bin edge {index}")
        for index, value in enumerate(values)
    )
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("receiver radius bin edges must be strictly increasing")
    if edges[0] <= 0.0:
        raise ValueError("receiver radius bin edges must be positive")
    if (
        edges[0] > calibration.isco_radius_over_mass
        or edges[-1] < calibration.outer_radius_over_mass
    ):
        raise ValueError("receiver radius bins must cover the whole photosphere")
    return edges


def _validate_inputs(
    frame: KerrFiniteThicknessSurfaceFrame,
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
    coarse_ray_options: RayTraceOptions | None,
    coarse_surface_options: SurfaceEventOptions | None,
) -> KerrFiniteThicknessSurfaceFrame:
    if type(frame) is not KerrFiniteThicknessSurfaceFrame:
        raise TypeError("frame must be the exact finite-thickness surface frame")
    if type(surface) is not KerrFiniteThicknessMultiSurface:
        raise TypeError("surface must be the exact finite-thickness multi-surface")
    if type(termination) is not KerrOblateTermination:
        raise TypeError("termination must be the exact KerrOblateTermination")
    if type(ray_options) is not RayTraceOptions:
        raise TypeError("ray_options must be the exact RayTraceOptions")
    if type(surface_options) is not SurfaceEventOptions:
        raise TypeError("surface_options must be the exact SurfaceEventOptions")
    if (coarse_ray_options is None) != (coarse_surface_options is None):
        raise ValueError("coarse ray and surface options must be supplied together")
    if coarse_ray_options is not None and type(coarse_ray_options) is not RayTraceOptions:
        raise TypeError("coarse_ray_options must be exact RayTraceOptions or None")
    if (
        coarse_surface_options is not None
        and type(coarse_surface_options) is not SurfaceEventOptions
    ):
        raise TypeError(
            "coarse_surface_options must be exact SurfaceEventOptions or None"
        )
    rebuilt_frame = KerrFiniteThicknessSurfaceFrame(frame.emitter)
    _require_trusted_exact_tree(frame, rebuilt_frame, "frame")
    if (
        rebuilt_frame.emitter.metric != surface.metric
        or rebuilt_frame.emitter.calibration != surface.calibration
    ):
        raise KerrReturningRadiationFateQuadratureError(
            "frame emitter and traced photosphere do not share exact ownership"
        )
    return rebuilt_frame


def _trace_classification(
    frame: KerrFiniteThicknessSurfaceFrame,
    surface: KerrFiniteThicknessMultiSurface,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions,
    surface_options: SurfaceEventOptions,
    coarse_ray_options: RayTraceOptions | None,
    coarse_surface_options: SurfaceEventOptions | None,
    mu: float,
    psi: float,
) -> _DirectionClassification:
    launch = KerrFiniteThicknessEmissionLaunch(frame, mu, psi, 1.0)
    primitive = trace_kerr_returning_radiation_direction(
        launch,
        surface,
        termination=termination,
        ray_options=ray_options,
        surface_options=surface_options,
        coarse_ray_options=coarse_ray_options,
        coarse_surface_options=coarse_surface_options,
    )
    if type(primitive) is not KerrReturningRadiationRayPrimitive:
        raise TypeError("direction tracer returned a non-exact primitive")
    # This module is a downstream scientific consumer.  The primitive's
    # contract requires the public replay before reading even its fate.
    verify_kerr_returning_radiation_direction(primitive)
    fate = primitive.fate
    if type(fate) is not str or fate not in _FATES:
        raise KerrReturningRadiationFateQuadratureError(
            "revalidated primitive has an unsupported fate"
        )
    receiver_radius = primitive.receiver_radius_over_mass
    if fate.startswith("return-"):
        if type(receiver_radius) is not float or not math.isfinite(receiver_radius):
            raise KerrReturningRadiationFateQuadratureError(
                "returned primitive lacks an exact finite receiver radius"
            )
        expected_face = UPPER if fate == "return-upper" else LOWER
        if primitive.receiver_face != expected_face:
            raise KerrReturningRadiationFateQuadratureError(
                "returned primitive fate and receiver face disagree"
            )
    elif receiver_radius is not None:
        raise KerrReturningRadiationFateQuadratureError(
            "non-returning primitive unexpectedly owns a receiver radius"
        )
    return _DirectionClassification(
        fate,
        receiver_radius,
        primitive.model_descriptor_sha256,
    )


def _maximum_absolute_difference(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float:
    if len(first) != len(second):
        raise KerrReturningRadiationFateQuadratureError(
            "convergence vectors have different dimensions"
        )
    return max(abs(left - right) for left, right in zip(first, second))


def integrate_kerr_returning_radiation_fates(
    frame: KerrFiniteThicknessSurfaceFrame,
    surface: KerrFiniteThicknessMultiSurface,
    *,
    termination: KerrOblateTermination,
    ray_options: RayTraceOptions = RayTraceOptions(),
    surface_options: SurfaceEventOptions = SurfaceEventOptions(
        subdivisions_per_segment=4
    ),
    coarse_ray_options: RayTraceOptions | None = None,
    coarse_surface_options: SurfaceEventOptions | None = None,
    mu_order: int = 4,
    psi_count: int = 8,
    receiver_radius_bin_edges_over_mass: tuple[float, ...] | None = None,
    convergence_absolute_tolerance: float = 0.05,
    maximum_whole_ray_traces: int = 512,
) -> KerrReturningRadiationFateQuadrature:
    """Integrate and certify one emitter's local bolometric flux fates.

    ``mu_order`` and ``psi_count`` are the full ``N`` grid and must both be
    even.  Mu and psi resolution are halved independently while holding the
    other coordinate at full resolution.  The phase test shifts the full
    periodic grid by half a ``psi`` cell.  One direction consumes four whole
    geodesic integrations: two in the primitive and two in its mandatory
    downstream replay.
    """

    frame = _validate_inputs(
        frame,
        surface,
        termination,
        ray_options,
        surface_options,
        coarse_ray_options,
        coarse_surface_options,
    )
    mu_order = _exact_positive_int(mu_order, "mu_order")
    psi_count = _exact_positive_int(psi_count, "psi_count")
    if mu_order < 4 or mu_order % 2:
        raise ValueError(
            "production mu_order must be an even integer at least 4 so its "
            "N/2 rule has at least two Gauss-Legendre nodes"
        )
    if psi_count < 4 or psi_count % 2:
        raise ValueError("psi_count must be an even integer at least 4")
    tolerance = _exact_finite_float(
        convergence_absolute_tolerance,
        "convergence_absolute_tolerance",
    )
    if tolerance <= 0.0 or tolerance > _MAXIMUM_CONVERGENCE_TOLERANCE:
        raise ValueError(
            "convergence tolerance must lie in (0, 0.1]"
        )
    budget = _exact_positive_int(
        maximum_whole_ray_traces,
        "maximum_whole_ray_traces",
    )
    if budget > _MAXIMUM_WHOLE_RAY_BUDGET:
        raise ValueError("whole-ray budget exceeds the hard policy maximum")
    direction_evaluations = 3 * mu_order * psi_count
    whole_ray_traces = 4 * direction_evaluations
    if whole_ray_traces > budget:
        raise ValueError(
            f"quadrature requires {whole_ray_traces} whole rays but budget is {budget}"
        )
    edges = _validated_receiver_edges(
        receiver_radius_bin_edges_over_mass,
        surface,
    )
    angular_law = FluxConservingLinearLimbDarkening()
    if (
        type(angular_law) is not FluxConservingLinearLimbDarkening
        or angular_law.coefficient.hex() != float(1.5).hex()
    ):
        raise KerrReturningRadiationFateQuadratureError(
            "built-in KERRBB D20 angular law identity is unavailable"
        )

    def classifier(mu: float, psi: float) -> _DirectionClassification:
        return _trace_classification(
            frame,
            surface,
            termination,
            ray_options,
            surface_options,
            coarse_ray_options,
            coarse_surface_options,
            mu,
            psi,
        )

    full = _integrate_classification_grid(
        kerrbb_d20_emitted_flux_direction_nodes(
            mu_order,
            psi_count,
            phase_cells=0.0,
        ),
        edges,
        classifier,
    )
    half_mu = _integrate_classification_grid(
        kerrbb_d20_emitted_flux_direction_nodes(
            mu_order // 2,
            psi_count,
            phase_cells=0.0,
        ),
        edges,
        classifier,
    )
    half_psi = _integrate_classification_grid(
        kerrbb_d20_emitted_flux_direction_nodes(
            mu_order,
            psi_count // 2,
            phase_cells=0.0,
        ),
        edges,
        classifier,
    )
    phase_shifted = _integrate_classification_grid(
        kerrbb_d20_emitted_flux_direction_nodes(
            mu_order,
            psi_count,
            phase_cells=0.5,
        ),
        edges,
        classifier,
    )
    mu_resolution_difference = _maximum_absolute_difference(
        full.convergence_vector(),
        half_mu.convergence_vector(),
    )
    psi_resolution_difference = _maximum_absolute_difference(
        full.convergence_vector(),
        half_psi.convergence_vector(),
    )
    resolution_difference = max(
        mu_resolution_difference,
        psi_resolution_difference,
    )
    phase_difference = _maximum_absolute_difference(
        full.convergence_vector(),
        phase_shifted.convergence_vector(),
    )
    converged = max(resolution_difference, phase_difference) <= tolerance
    convergence = KerrReturningRadiationFateConvergence(
        mu_resolution_difference,
        psi_resolution_difference,
        resolution_difference,
        phase_difference,
        tolerance,
        converged,
    )
    if not converged:
        raise KerrReturningRadiationFateConvergenceError(
            "emitter-local fate fractions fail the declared mu N/2N, psi "
            f"N/2N, or periodic phase gate (mu={mu_resolution_difference:.17g}, "
            f"psi={psi_resolution_difference:.17g}, "
            f"phase={phase_difference:.17g}, tolerance={tolerance:.17g})"
        )

    descriptor = {
        "angularEmission": angular_law.descriptor(),
        "capabilities": dict(SCIENTIFIC_STATUS),
        "convergence": {
            "actual": asdict(convergence),
            "fullGridFractions": asdict(full.fractions),
            "halfMuGridFractions": asdict(half_mu.fractions),
            "halfPsiGridFractions": asdict(half_psi.fractions),
            "maximumAllowedAbsoluteTolerance": (
                _MAXIMUM_CONVERGENCE_TOLERANCE
            ),
            "phaseShiftedGridFractions": asdict(phase_shifted.fractions),
            "resolutionPairs": (
                "reported N_mu versus N_mu/2 at fixed N_psi; reported "
                "N_psi versus N_psi/2 at fixed N_mu"
            ),
            "periodicPhaseTest": "reported N shifted by half one psi cell",
        },
        "emitterFrameDescriptorSha256": frame.model_descriptor_sha256,
        "implementationId": IMPLEMENTATION_ID,
        "measure": {
            "denominator": "one-face local emitted bolometric flux",
            "formula": "2 mu f(mu) dmu dpsi/(2 pi)",
            "isPhotonNumber": False,
            "usesFrequencyShiftG": False,
            "usesReceiverIncidenceCosine": False,
        },
        "modelOwnership": {
            "calibration": asdict(surface.calibration),
            "metric": asdict(surface.metric),
            "termination": asdict(termination),
        },
        "quadrature": {
            "d20Rule": dict(D20_EMITTED_FLUX_QUADRATURE_DESCRIPTOR),
            "fullMuOrder": mu_order,
            "fullPsiCount": psi_count,
            "halfMuOrder": mu_order // 2,
            "halfPsiCount": psi_count // 2,
            "muRule": "float64 Gauss-Legendre on open interval (0,1)",
            "psiRule": "periodic equal-weight midpoint",
            "receiverRadiusBinEdgesOverMass": edges,
        },
        "result": {
            "estimateKind": "finite-grid-point-estimate",
            "fractions": asdict(full.fractions),
            "publishedFiniteGridUncertaintyDiagnostics": {
                "muResolutionMaximumAbsoluteDifference": (
                    convergence.mu_resolution_maximum_absolute_difference
                ),
                "periodicPhaseMaximumAbsoluteDifference": (
                    convergence.periodic_phase_maximum_absolute_difference
                ),
                "psiResolutionMaximumAbsoluteDifference": (
                    convergence.psi_resolution_maximum_absolute_difference
                ),
                "rigorousErrorBound": False,
            },
            "returnUpperByReceiverBin": full.return_upper_by_receiver_bin,
            "returnUpperBinClosureResidual": (
                full.return_upper_bin_closure_residual
            ),
            "returnLowerByReceiverBin": full.return_lower_by_receiver_bin,
            "returnLowerBinClosureResidual": (
                full.return_lower_bin_closure_residual
            ),
        },
        "sampleAuditSha256": {
            "full": _sha256_json(tuple(asdict(item) for item in full.direction_audit)),
            "halfMu": _sha256_json(
                tuple(asdict(item) for item in half_mu.direction_audit)
            ),
            "halfPsi": _sha256_json(
                tuple(asdict(item) for item in half_psi.direction_audit)
            ),
            "phaseShifted": _sha256_json(
                tuple(asdict(item) for item in phase_shifted.direction_audit)
            ),
        },
        "workBudget": {
            "directionEvaluations": direction_evaluations,
            "hardPolicyMaximumWholeRayTraces": _MAXIMUM_WHOLE_RAY_BUDGET,
            "maximumWholeRayTraces": budget,
            "primitiveWholeRaysPerDirection": 2,
            "publicReplayWholeRaysPerDirection": 2,
            "wholeRayTracesConsumed": whole_ray_traces,
        },
    }
    descriptor_json = _canonical_json(descriptor)
    result = object.__new__(KerrReturningRadiationFateQuadrature)
    for name, value in (
        ("frame", frame),
        ("surface", surface),
        ("termination", termination),
        ("ray_options", ray_options),
        ("surface_options", surface_options),
        ("coarse_ray_options", coarse_ray_options),
        ("coarse_surface_options", coarse_surface_options),
        ("mu_order", mu_order),
        ("psi_count", psi_count),
        ("receiver_radius_bin_edges_over_mass", edges),
        ("convergence_absolute_tolerance", tolerance),
        ("maximum_whole_ray_traces", budget),
        ("whole_ray_traces_consumed", whole_ray_traces),
        ("fractions", full.fractions),
        ("return_upper_by_receiver_bin", full.return_upper_by_receiver_bin),
        ("return_lower_by_receiver_bin", full.return_lower_by_receiver_bin),
        ("convergence", convergence),
        ("full_grid_direction_audit", full.direction_audit),
        ("half_mu_grid_direction_audit", half_mu.direction_audit),
        ("half_psi_grid_direction_audit", half_psi.direction_audit),
        ("phase_shifted_direction_audit", phase_shifted.direction_audit),
        ("_descriptor_json", descriptor_json),
        (
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        ),
    ):
        object.__setattr__(result, name, value)
    return result


def verify_kerr_returning_radiation_fate_quadrature(
    result: KerrReturningRadiationFateQuadrature,
) -> None:
    """Replay every direction and compare the complete immutable result tree."""

    if type(result) is not KerrReturningRadiationFateQuadrature:
        raise TypeError(
            "result must be the exact KerrReturningRadiationFateQuadrature"
        )
    if type(result._descriptor_json) is not str or type(result._descriptor_sha256) is not str:
        raise KerrReturningRadiationFateVerificationError(
            "quadrature descriptor identity has a non-exact type"
        )
    try:
        parsed_descriptor = json.loads(result._descriptor_json)
    except (json.JSONDecodeError, TypeError) as error:
        raise KerrReturningRadiationFateVerificationError(
            "quadrature descriptor is malformed"
        ) from error
    if _canonical_json(parsed_descriptor).encode("utf-8") != result._descriptor_json.encode(
        "utf-8"
    ):
        raise KerrReturningRadiationFateVerificationError(
            "quadrature descriptor is not canonical"
        )
    expected_sha = hashlib.sha256(result._descriptor_json.encode("utf-8")).hexdigest()
    if expected_sha.encode("ascii") != result._descriptor_sha256.encode("ascii"):
        raise KerrReturningRadiationFateVerificationError(
            "quadrature descriptor SHA-256 is inconsistent"
        )
    rebuilt = integrate_kerr_returning_radiation_fates(
        result.frame,
        result.surface,
        termination=result.termination,
        ray_options=result.ray_options,
        surface_options=result.surface_options,
        coarse_ray_options=result.coarse_ray_options,
        coarse_surface_options=result.coarse_surface_options,
        mu_order=result.mu_order,
        psi_count=result.psi_count,
        receiver_radius_bin_edges_over_mass=(
            result.receiver_radius_bin_edges_over_mass
        ),
        convergence_absolute_tolerance=result.convergence_absolute_tolerance,
        maximum_whole_ray_traces=result.maximum_whole_ray_traces,
    )
    _require_trusted_exact_tree(result, rebuilt, "result")


__all__ = (
    "D20_EMITTED_FLUX_QUADRATURE_DESCRIPTOR",
    "D20_QUADRATURE_IMPLEMENTATION_ID",
    "EmittedFluxDirectionAudit",
    "EmittedFluxDirectionNode",
    "EmittedFluxFateFractions",
    "IMPLEMENTATION_ID",
    "KERRBB_SOURCE_URL",
    "KerrReturningRadiationFateConvergence",
    "KerrReturningRadiationFateConvergenceError",
    "KerrReturningRadiationFateQuadrature",
    "KerrReturningRadiationFateQuadratureError",
    "KerrReturningRadiationFateVerificationError",
    "SCIENTIFIC_STATUS",
    "integrate_kerr_returning_radiation_fates",
    "kerrbb_d20_emitted_flux_direction_nodes",
    "verify_kerr_returning_radiation_fate_quadrature",
)
