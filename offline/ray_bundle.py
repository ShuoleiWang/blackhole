"""Finite-difference screen-to-source ray-bundle diagnostics.

This module differentiates a *certified endpoint map* by tracing nine
independent rays: the centre, four neighbours at ``h``, and four neighbours at
``h/2``.  The two centred-difference Jacobians must agree within declared
absolute and relative tolerances before a diagnostic is returned.

The result is a local screen-space finite-difference diagnostic.  It is not a
Sachs/Jacobi geodesic-deviation integration along the ray, does not provide
optical scalars, wavefront curvature, or a time-delay Hessian, and a finite
stencil cannot prove that every caustic has been found.

``solid_angle_magnification`` is only meaningful because every endpoint
sample carries the local solid-angle density of its declared source-coordinate
chart.  For a map ``s(x)`` it is

    dOmega_observer / dOmega_source
      = rho_screen(x) / (rho_source(s) * abs(det(ds/dx))).

Near a rank-deficient Jacobian the finite magnification field is deliberately
``None``.  The determinant, singular values, and criticality flag remain
available without pretending that a numerically unstable reciprocal is a
resolved physical observable.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Mapping, Protocol, Sequence, runtime_checkable


RAY_BUNDLE_DIAGNOSTIC_VERSION: Final = (
    "screen-finite-difference-endpoint-bundle-h-h2/v1"
)
SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "local screen-space finite-difference endpoint-map ray-bundle "
            "diagnostic"
        ),
        "stencil": "centre plus axis neighbours at h and h/2",
        "isSachsJacobiRayBundle": False,
        "integratesGeodesicDeviation": False,
        "providesOpticalScalarsAlongRay": False,
        "providesWavefrontCurvature": False,
        "providesTimeDelayHessian": False,
        "isCausticComplete": False,
        "prohibitedClaim": (
            "Do not describe this finite endpoint stencil as Sachs/Jacobi "
            "geodesic deviation, a wavefront-curvature or time-delay-Hessian "
            "solver, or a caustic-complete proof."
        ),
    }
)

Matrix2 = tuple[tuple[float, float], tuple[float, float]]


class RayBundleDiagnosticError(RuntimeError):
    """Raised when a finite-difference bundle cannot be certified."""


class RayBundleWorkBudgetExceeded(RayBundleDiagnosticError):
    """Raised before the endpoint mapper would exceed its declared budget."""


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _positive_number(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _non_negative_number(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _finite_pair(values: Sequence[float], label: str) -> tuple[float, float]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must contain two finite numbers")
    try:
        entries = tuple(values)
    except TypeError as error:
        raise ValueError(f"{label} must contain two finite numbers") from error
    if len(entries) != 2:
        raise ValueError(f"{label} must contain two finite numbers")
    return (
        _finite_number(entries[0], f"{label}[0]"),
        _finite_number(entries[1], f"{label}[1]"),
    )


def _finite_matrix(value: Sequence[Sequence[float]], label: str) -> Matrix2:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a finite 2x2 matrix")
    try:
        rows = tuple(value)
    except TypeError as error:
        raise ValueError(f"{label} must be a finite 2x2 matrix") from error
    if len(rows) != 2:
        raise ValueError(f"{label} must be a finite 2x2 matrix")
    return (
        _finite_pair(rows[0], f"{label}[0]"),
        _finite_pair(rows[1], f"{label}[1]"),
    )


def pinhole_chart_solid_angle_density(
    chart_x: float,
    chart_y: float,
) -> float:
    """Return ``dOmega / (dx dy)`` for a unit-focal-length pinhole chart."""

    x_value = _finite_number(chart_x, "chart_x")
    y_value = _finite_number(chart_y, "chart_y")
    radius = math.hypot(1.0, x_value, y_value)
    inverse = 1.0 / radius
    density = inverse * inverse * inverse
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("pinhole solid-angle density is not representable")
    return density


@dataclass(frozen=True, slots=True)
class RayEndpointConvergenceAudit:
    """Numerical gates attached to one independently traced endpoint."""

    maximum_null_residual: float = 0.0
    maximum_source_coordinate_error: float = 0.0
    accepted_steps: int = 0
    rejected_steps: int = 0
    ray_gate_passed: bool = False
    source_topology_gate_passed: bool = False
    source_coordinate_gate_passed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "maximum_null_residual",
            "maximum_source_coordinate_error",
        ):
            object.__setattr__(
                self,
                name,
                _non_negative_number(getattr(self, name), name),
            )
        for name in ("accepted_steps", "rejected_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "ray_gate_passed",
            "source_topology_gate_passed",
            "source_coordinate_gate_passed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if self.ray_gate_passed and self.accepted_steps < 1:
            raise ValueError("a passed ray gate requires an accepted ray step")

    @property
    def all_gates_passed(self) -> bool:
        return (
            self.ray_gate_passed
            and self.source_topology_gate_passed
            and self.source_coordinate_gate_passed
        )


@dataclass(frozen=True, slots=True)
class RayEndpointSample:
    """One certified screen-ray endpoint in a declared source angular chart."""

    source_kind: str
    topology_signature: str
    source_chart_id: str
    source_coordinates: tuple[float, float]
    source_solid_angle_density_sr_per_coordinate_area: float
    endpoint_converged: bool = False
    convergence_audit: RayEndpointConvergenceAudit = RayEndpointConvergenceAudit()

    def __post_init__(self) -> None:
        for name in ("source_kind", "topology_signature", "source_chart_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        coordinates = _finite_pair(self.source_coordinates, "source_coordinates")
        density = _positive_number(
            self.source_solid_angle_density_sr_per_coordinate_area,
            "source_solid_angle_density_sr_per_coordinate_area",
        )
        if type(self.endpoint_converged) is not bool:
            raise TypeError("endpoint_converged must be a bool")
        if not isinstance(self.convergence_audit, RayEndpointConvergenceAudit):
            raise TypeError(
                "convergence_audit must be a RayEndpointConvergenceAudit"
            )
        if self.endpoint_converged and not self.convergence_audit.all_gates_passed:
            raise ValueError("a converged endpoint requires every audit gate to pass")
        object.__setattr__(self, "source_coordinates", coordinates)
        object.__setattr__(
            self,
            "source_solid_angle_density_sr_per_coordinate_area",
            density,
        )


@runtime_checkable
class CertifiedRayEndpointMapper(Protocol):
    """Named screen-to-source endpoint map consumed by this diagnostic."""

    implementation_id: str

    def map_endpoint(
        self,
        screen_x: float,
        screen_y: float,
    ) -> RayEndpointSample:
        """Trace one independent ray and return a certified endpoint."""


@dataclass(frozen=True, slots=True)
class RayBundleOptions:
    """Numerical and work limits for the fixed nine-ray stencil."""

    finite_difference_step: float = 1.0e-4
    jacobian_absolute_tolerance: float = 1.0e-9
    jacobian_relative_tolerance: float = 1.0e-5
    minimum_singular_value_ratio: float = 1.0e-10
    maximum_endpoint_evaluations: int = 9
    diagnostic_version: str = field(
        default=RAY_BUNDLE_DIAGNOSTIC_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "finite_difference_step",
            _positive_number(
                self.finite_difference_step,
                "finite_difference_step",
            ),
        )
        for name in (
            "jacobian_absolute_tolerance",
            "jacobian_relative_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _non_negative_number(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "minimum_singular_value_ratio",
            _positive_number(
                self.minimum_singular_value_ratio,
                "minimum_singular_value_ratio",
            ),
        )
        if self.minimum_singular_value_ratio >= 1.0:
            raise ValueError("minimum_singular_value_ratio must be less than one")
        if (
            type(self.maximum_endpoint_evaluations) is not int
            or self.maximum_endpoint_evaluations < 1
        ):
            raise ValueError(
                "maximum_endpoint_evaluations must be a positive integer"
            )


@dataclass(frozen=True, slots=True)
class StableSvd2x2:
    """Scale-normalized singular-value and determinant decomposition."""

    singular_values: tuple[float, float]
    normalized_singular_values: tuple[float, float]
    matrix_scale: float
    normalized_determinant: float
    determinant: float | None
    determinant_sign: int
    log_absolute_determinant: float | None
    condition_number: float


def stable_svd_2x2(matrix: Sequence[Sequence[float]]) -> StableSvd2x2:
    """Compute a stable 2x2 SVD summary without squaring raw matrix entries.

    The matrix is first divided by its largest absolute entry.  The larger
    singular value comes from the symmetric eigenproblem of the normalized
    Gram matrix; the smaller value uses ``|det| / sigma_max`` to retain
    relative accuracy near rank loss.
    """

    value = _finite_matrix(matrix, "matrix")
    entries = (value[0][0], value[0][1], value[1][0], value[1][1])
    scale = max(abs(entry) for entry in entries)
    if scale == 0.0:
        return StableSvd2x2(
            singular_values=(0.0, 0.0),
            normalized_singular_values=(0.0, 0.0),
            matrix_scale=0.0,
            normalized_determinant=0.0,
            determinant=0.0,
            determinant_sign=0,
            log_absolute_determinant=None,
            condition_number=math.inf,
        )

    a_value, b_value, c_value, d_value = (
        entry / scale for entry in entries
    )
    determinant_normalized = math.fsum(
        (a_value * d_value, -b_value * c_value)
    )
    first_norm_squared = math.fsum((a_value * a_value, c_value * c_value))
    second_norm_squared = math.fsum((b_value * b_value, d_value * d_value))
    cross = math.fsum((a_value * b_value, c_value * d_value))
    half_trace = 0.5 * (first_norm_squared + second_norm_squared)
    half_difference = 0.5 * (first_norm_squared - second_norm_squared)
    largest_eigenvalue = half_trace + math.hypot(half_difference, cross)
    largest_normalized = math.sqrt(max(0.0, largest_eigenvalue))
    if largest_normalized == 0.0:
        smallest_normalized = 0.0
    else:
        smallest_normalized = abs(determinant_normalized) / largest_normalized

    largest = scale * largest_normalized
    smallest = scale * smallest_normalized
    if not all(math.isfinite(item) for item in (largest, smallest)):
        raise ValueError("matrix singular values are not representable")
    sign = (
        1
        if determinant_normalized > 0.0
        else -1
        if determinant_normalized < 0.0
        else 0
    )
    if sign == 0:
        determinant: float | None = 0.0
        log_determinant: float | None = None
    else:
        log_determinant = (
            2.0 * math.log(scale) + math.log(abs(determinant_normalized))
        )
        maximum_log = math.log(sys.float_info.max)
        minimum_log = math.log(math.ulp(0.0))
        if minimum_log <= log_determinant <= maximum_log:
            determinant = math.copysign(math.exp(log_determinant), sign)
        else:
            determinant = None
    condition = (
        largest_normalized / smallest_normalized
        if smallest_normalized > 0.0
        else math.inf
    )
    return StableSvd2x2(
        singular_values=(largest, smallest),
        normalized_singular_values=(largest_normalized, smallest_normalized),
        matrix_scale=scale,
        normalized_determinant=determinant_normalized,
        determinant=determinant,
        determinant_sign=sign,
        log_absolute_determinant=log_determinant,
        condition_number=condition,
    )


@dataclass(frozen=True, slots=True)
class RayBundleDiagnostic:
    """Certified local derivative and caustic indicators for one screen ray."""

    mapper_implementation_id: str
    screen_coordinates: tuple[float, float]
    source_kind: str
    topology_signature: str
    source_chart_id: str
    source_coordinates: tuple[float, float]
    coarse_step: float
    fine_step: float
    coarse_jacobian: Matrix2
    fine_jacobian: Matrix2
    jacobian: Matrix2
    jacobian_difference_norm: float
    jacobian_relative_difference: float
    estimated_jacobian_error_norm: float
    singular_values: tuple[float, float]
    condition_number: float
    determinant: float | None
    determinant_sign: int
    log_absolute_determinant: float | None
    parity: int
    critical_singular_value_ratio_threshold: float
    near_critical: bool
    screen_solid_angle_density_sr_per_coordinate_area: float
    source_solid_angle_density_sr_per_coordinate_area: float
    solid_angle_magnification: float | None
    log_solid_angle_magnification: float | None
    sample_count: int
    maximum_null_residual: float
    maximum_source_coordinate_error: float
    diagnostic_version: str = field(
        default=RAY_BUNDLE_DIAGNOSTIC_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "mapper_implementation_id",
            "source_kind",
            "topology_signature",
            "source_chart_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(
            self,
            "screen_coordinates",
            _finite_pair(self.screen_coordinates, "screen_coordinates"),
        )
        object.__setattr__(
            self,
            "source_coordinates",
            _finite_pair(self.source_coordinates, "source_coordinates"),
        )
        coarse_step = _positive_number(self.coarse_step, "coarse_step")
        fine_step = _positive_number(self.fine_step, "fine_step")
        if fine_step != 0.5 * coarse_step:
            raise ValueError("fine_step must be exactly half coarse_step")
        object.__setattr__(self, "coarse_step", coarse_step)
        object.__setattr__(self, "fine_step", fine_step)
        coarse = _finite_matrix(self.coarse_jacobian, "coarse_jacobian")
        fine = _finite_matrix(self.fine_jacobian, "fine_jacobian")
        jacobian = _finite_matrix(self.jacobian, "jacobian")
        if jacobian != fine:
            raise ValueError("reported Jacobian must be the h/2 Jacobian")
        object.__setattr__(self, "coarse_jacobian", coarse)
        object.__setattr__(self, "fine_jacobian", fine)
        object.__setattr__(self, "jacobian", jacobian)

        difference_norm = _non_negative_number(
            self.jacobian_difference_norm,
            "jacobian_difference_norm",
        )
        relative_difference = _non_negative_number(
            self.jacobian_relative_difference,
            "jacobian_relative_difference",
        )
        estimated_error = _non_negative_number(
            self.estimated_jacobian_error_norm,
            "estimated_jacobian_error_norm",
        )
        expected_difference = _matrix_norm(_matrix_difference(fine, coarse))
        reference = max(_matrix_norm(fine), _matrix_norm(coarse))
        expected_relative = expected_difference / reference if reference else 0.0
        if (
            difference_norm != expected_difference
            or relative_difference != expected_relative
            or estimated_error != expected_difference / 3.0
        ):
            raise ValueError("reported Jacobian convergence evidence is inconsistent")

        singular = _finite_pair(self.singular_values, "singular_values")
        if singular[0] < singular[1] or singular[1] < 0.0:
            raise ValueError("singular_values must be ordered and non-negative")
        decomposition = stable_svd_2x2(jacobian)
        if singular != decomposition.singular_values:
            raise ValueError("reported singular values disagree with the Jacobian")
        condition = self.condition_number
        if (
            isinstance(condition, bool)
            or not isinstance(condition, (int, float))
            or math.isnan(float(condition))
            or float(condition) < 1.0
        ):
            raise ValueError("condition_number must be at least one or infinity")
        condition = float(condition)
        if condition != decomposition.condition_number:
            raise ValueError("reported condition number disagrees with the Jacobian")
        if type(self.determinant_sign) is not int or self.determinant_sign not in (
            -1,
            0,
            1,
        ):
            raise ValueError("determinant_sign must be -1, 0, or 1")
        if self.determinant_sign != decomposition.determinant_sign:
            raise ValueError("reported determinant sign disagrees with the Jacobian")
        if self.determinant != decomposition.determinant:
            raise ValueError("reported determinant disagrees with the Jacobian")
        if self.log_absolute_determinant != (
            decomposition.log_absolute_determinant
        ):
            raise ValueError(
                "reported log determinant disagrees with the Jacobian"
            )

        critical_threshold = _positive_number(
            self.critical_singular_value_ratio_threshold,
            "critical_singular_value_ratio_threshold",
        )
        if critical_threshold >= 1.0:
            raise ValueError(
                "critical_singular_value_ratio_threshold must be less than one"
            )
        singular_ratio = (
            decomposition.normalized_singular_values[1]
            / decomposition.normalized_singular_values[0]
            if decomposition.normalized_singular_values[0] > 0.0
            else 0.0
        )
        expected_critical = (
            decomposition.determinant_sign == 0
            or singular_ratio <= critical_threshold
        )
        if type(self.near_critical) is not bool:
            raise TypeError("near_critical must be a bool")
        if self.near_critical != expected_critical:
            raise ValueError(
                "near_critical disagrees with the declared singular-value gate"
            )
        if type(self.parity) is not int or self.parity not in (-1, 0, 1):
            raise ValueError("parity must be -1, 0, or 1")
        if self.near_critical:
            if (
                self.parity != 0
                or self.solid_angle_magnification is not None
                or self.log_solid_angle_magnification is not None
            ):
                raise ValueError(
                    "near-critical diagnostics may not report finite magnification"
                )
        elif self.parity != self.determinant_sign or self.parity == 0:
            raise ValueError("non-critical parity must match the determinant sign")

        screen_density = _positive_number(
            self.screen_solid_angle_density_sr_per_coordinate_area,
            "screen_solid_angle_density_sr_per_coordinate_area",
        )
        source_density = _positive_number(
            self.source_solid_angle_density_sr_per_coordinate_area,
            "source_solid_angle_density_sr_per_coordinate_area",
        )
        magnification = self.solid_angle_magnification
        log_magnification = self.log_solid_angle_magnification
        if not self.near_critical:
            if log_magnification is None:
                raise ValueError(
                    "non-critical diagnostic requires log magnification"
                )
            if decomposition.log_absolute_determinant is None:
                raise ValueError(
                    "non-critical diagnostic requires a non-zero determinant"
                )
            log_magnification = _finite_number(
                log_magnification,
                "log_solid_angle_magnification",
            )
            expected_log = (
                math.log(screen_density)
                - math.log(source_density)
                - decomposition.log_absolute_determinant
            )
            if log_magnification != expected_log:
                raise ValueError("reported solid-angle magnification is inconsistent")
            if magnification is not None:
                magnification = _positive_number(
                    magnification,
                    "solid_angle_magnification",
                )
                if not math.isclose(
                    math.log(magnification),
                    log_magnification,
                    rel_tol=2.0e-15,
                    abs_tol=2.0e-15,
                ):
                    raise ValueError(
                        "finite and logarithmic magnification disagree"
                    )
        if type(self.sample_count) is not int or self.sample_count != 9:
            raise ValueError("sample_count must bind the fixed nine-ray stencil")
        for name in (
            "maximum_null_residual",
            "maximum_source_coordinate_error",
        ):
            object.__setattr__(
                self,
                name,
                _non_negative_number(getattr(self, name), name),
            )
        object.__setattr__(self, "jacobian_difference_norm", difference_norm)
        object.__setattr__(
            self,
            "jacobian_relative_difference",
            relative_difference,
        )
        object.__setattr__(
            self,
            "estimated_jacobian_error_norm",
            estimated_error,
        )
        object.__setattr__(self, "singular_values", singular)
        object.__setattr__(self, "condition_number", condition)
        object.__setattr__(
            self,
            "critical_singular_value_ratio_threshold",
            critical_threshold,
        )
        object.__setattr__(
            self,
            "screen_solid_angle_density_sr_per_coordinate_area",
            screen_density,
        )
        object.__setattr__(
            self,
            "source_solid_angle_density_sr_per_coordinate_area",
            source_density,
        )
        object.__setattr__(self, "solid_angle_magnification", magnification)
        object.__setattr__(
            self,
            "log_solid_angle_magnification",
            log_magnification,
        )


def _safe_central_difference(
    positive: float,
    negative: float,
    step: float,
) -> float:
    try:
        difference = math.fsum((positive, -negative))
    except OverflowError as error:
        raise RayBundleDiagnosticError(
            "source-coordinate difference overflowed"
        ) from error
    derivative = difference / (2.0 * step)
    if not math.isfinite(derivative):
        raise RayBundleDiagnosticError(
            "source-coordinate derivative is not representable"
        )
    return derivative


def _jacobian(
    x_positive: RayEndpointSample,
    x_negative: RayEndpointSample,
    y_positive: RayEndpointSample,
    y_negative: RayEndpointSample,
    step: float,
) -> Matrix2:
    return (
        (
            _safe_central_difference(
                x_positive.source_coordinates[0],
                x_negative.source_coordinates[0],
                step,
            ),
            _safe_central_difference(
                y_positive.source_coordinates[0],
                y_negative.source_coordinates[0],
                step,
            ),
        ),
        (
            _safe_central_difference(
                x_positive.source_coordinates[1],
                x_negative.source_coordinates[1],
                step,
            ),
            _safe_central_difference(
                y_positive.source_coordinates[1],
                y_negative.source_coordinates[1],
                step,
            ),
        ),
    )


def _matrix_difference(first: Matrix2, second: Matrix2) -> Matrix2:
    try:
        return (
            (
                math.fsum((first[0][0], -second[0][0])),
                math.fsum((first[0][1], -second[0][1])),
            ),
            (
                math.fsum((first[1][0], -second[1][0])),
                math.fsum((first[1][1], -second[1][1])),
            ),
        )
    except OverflowError as error:
        raise RayBundleDiagnosticError("Jacobian difference overflowed") from error


def _matrix_norm(matrix: Matrix2) -> float:
    return math.hypot(matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1])


def _tolerances_accept(
    difference: float,
    reference: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    if reference == 0.0:
        return difference <= absolute_tolerance
    return difference / reference <= (
        relative_tolerance + absolute_tolerance / reference
    )


def _finite_stencil_points(
    screen_x: float,
    screen_y: float,
    step: float,
) -> tuple[tuple[str, tuple[float, float]], ...]:
    half_step = 0.5 * step
    points = (
        ("centre", (screen_x, screen_y)),
        ("coarse+x", (screen_x + step, screen_y)),
        ("coarse-x", (screen_x - step, screen_y)),
        ("coarse+y", (screen_x, screen_y + step)),
        ("coarse-y", (screen_x, screen_y - step)),
        ("fine+x", (screen_x + half_step, screen_y)),
        ("fine-x", (screen_x - half_step, screen_y)),
        ("fine+y", (screen_x, screen_y + half_step)),
        ("fine-y", (screen_x, screen_y - half_step)),
    )
    if any(not all(math.isfinite(value) for value in point) for _, point in points):
        raise RayBundleDiagnosticError("finite-difference stencil is not finite")
    if len({point for _, point in points}) != len(points):
        raise RayBundleDiagnosticError(
            "finite-difference step is not distinct at this screen-coordinate scale"
        )
    return points


def diagnose_screen_ray_bundle(
    mapper: CertifiedRayEndpointMapper,
    screen_x: float,
    screen_y: float,
    *,
    options: RayBundleOptions = RayBundleOptions(),
) -> RayBundleDiagnostic:
    """Differentiate a certified endpoint map with an ``h``/``h/2`` gate.

    Exactly nine endpoint evaluations are required.  If the declared budget is
    smaller, the function raises before invoking the mapper.  A failed endpoint
    audit, topology/chart mismatch, non-finite value, or unconverged Jacobian
    also raises instead of returning a plausible-looking diagnostic.
    """

    if not isinstance(options, RayBundleOptions):
        raise TypeError("options must be RayBundleOptions")
    if not isinstance(mapper, CertifiedRayEndpointMapper):
        raise TypeError("mapper must implement CertifiedRayEndpointMapper")
    implementation_id = mapper.implementation_id
    if not isinstance(implementation_id, str) or not implementation_id.strip():
        raise ValueError("mapper implementation_id must be a non-empty string")
    x_value = _finite_number(screen_x, "screen_x")
    y_value = _finite_number(screen_y, "screen_y")
    points = _finite_stencil_points(
        x_value,
        y_value,
        options.finite_difference_step,
    )
    if options.maximum_endpoint_evaluations < len(points):
        raise RayBundleWorkBudgetExceeded(
            "nine-ray endpoint stencil exceeds maximum_endpoint_evaluations"
        )

    samples: dict[str, RayEndpointSample] = {}
    evaluations = 0
    for label, point in points:
        if evaluations >= options.maximum_endpoint_evaluations:
            raise RayBundleWorkBudgetExceeded(
                "endpoint-mapper evaluation budget exhausted"
            )
        evaluations += 1
        try:
            sample = mapper.map_endpoint(*point)
        except Exception as error:
            raise RayBundleDiagnosticError(
                f"endpoint mapper failed at stencil point {label!r}"
            ) from error
        if not isinstance(sample, RayEndpointSample):
            raise RayBundleDiagnosticError(
                f"endpoint mapper returned an invalid sample at {label!r}"
            )
        if not sample.endpoint_converged or not (
            sample.convergence_audit.all_gates_passed
        ):
            raise RayBundleDiagnosticError(
                f"endpoint convergence gate failed at {label!r}"
            )
        samples[label] = sample

    if mapper.implementation_id != implementation_id:
        raise RayBundleDiagnosticError(
            "mapper implementation_id changed during endpoint evaluation"
        )
    centre = samples["centre"]
    topology = (
        centre.source_kind,
        centre.topology_signature,
        centre.source_chart_id,
    )
    for label, sample in samples.items():
        if (
            sample.source_kind,
            sample.topology_signature,
            sample.source_chart_id,
        ) != topology:
            raise RayBundleDiagnosticError(
                f"mixed source topology or chart at stencil point {label!r}"
            )

    coarse = _jacobian(
        samples["coarse+x"],
        samples["coarse-x"],
        samples["coarse+y"],
        samples["coarse-y"],
        options.finite_difference_step,
    )
    fine_step = 0.5 * options.finite_difference_step
    fine = _jacobian(
        samples["fine+x"],
        samples["fine-x"],
        samples["fine+y"],
        samples["fine-y"],
        fine_step,
    )
    difference = _matrix_norm(_matrix_difference(fine, coarse))
    reference = max(_matrix_norm(fine), _matrix_norm(coarse))
    relative_difference = difference / reference if reference > 0.0 else 0.0
    if not _tolerances_accept(
        difference,
        reference,
        options.jacobian_absolute_tolerance,
        options.jacobian_relative_tolerance,
    ):
        raise RayBundleDiagnosticError(
            "h versus h/2 Jacobian convergence gate failed"
        )

    decomposition = stable_svd_2x2(fine)
    largest_normalized, smallest_normalized = (
        decomposition.normalized_singular_values
    )
    singular_ratio = (
        smallest_normalized / largest_normalized
        if largest_normalized > 0.0
        else 0.0
    )
    near_critical = (
        decomposition.determinant_sign == 0
        or singular_ratio <= options.minimum_singular_value_ratio
    )
    parity = 0 if near_critical else decomposition.determinant_sign
    screen_density = pinhole_chart_solid_angle_density(x_value, y_value)
    source_density = (
        centre.source_solid_angle_density_sr_per_coordinate_area
    )
    magnification: float | None = None
    log_magnification: float | None = None
    if not near_critical:
        if decomposition.log_absolute_determinant is None:
            raise RayBundleDiagnosticError(
                "non-critical Jacobian has no determinant magnitude"
            )
        log_magnification = (
            math.log(screen_density)
            - math.log(source_density)
            - decomposition.log_absolute_determinant
        )
        minimum_log = math.log(math.ulp(0.0))
        maximum_log = math.log(sys.float_info.max)
        if minimum_log <= log_magnification <= maximum_log:
            magnification = math.exp(log_magnification)

    return RayBundleDiagnostic(
        mapper_implementation_id=implementation_id,
        screen_coordinates=(x_value, y_value),
        source_kind=centre.source_kind,
        topology_signature=centre.topology_signature,
        source_chart_id=centre.source_chart_id,
        source_coordinates=centre.source_coordinates,
        coarse_step=options.finite_difference_step,
        fine_step=fine_step,
        coarse_jacobian=coarse,
        fine_jacobian=fine,
        jacobian=fine,
        jacobian_difference_norm=difference,
        jacobian_relative_difference=relative_difference,
        estimated_jacobian_error_norm=difference / 3.0,
        singular_values=decomposition.singular_values,
        condition_number=decomposition.condition_number,
        determinant=decomposition.determinant,
        determinant_sign=decomposition.determinant_sign,
        log_absolute_determinant=decomposition.log_absolute_determinant,
        parity=parity,
        critical_singular_value_ratio_threshold=(
            options.minimum_singular_value_ratio
        ),
        near_critical=near_critical,
        screen_solid_angle_density_sr_per_coordinate_area=screen_density,
        source_solid_angle_density_sr_per_coordinate_area=source_density,
        solid_angle_magnification=magnification,
        log_solid_angle_magnification=log_magnification,
        sample_count=evaluations,
        maximum_null_residual=max(
            sample.convergence_audit.maximum_null_residual
            for sample in samples.values()
        ),
        maximum_source_coordinate_error=max(
            sample.convergence_audit.maximum_source_coordinate_error
            for sample in samples.values()
        ),
    )


__all__ = (
    "CertifiedRayEndpointMapper",
    "RAY_BUNDLE_DIAGNOSTIC_VERSION",
    "RayBundleDiagnostic",
    "RayBundleDiagnosticError",
    "RayBundleOptions",
    "RayBundleWorkBudgetExceeded",
    "RayEndpointConvergenceAudit",
    "RayEndpointSample",
    "SCIENTIFIC_STATUS",
    "StableSvd2x2",
    "diagnose_screen_ray_bundle",
    "pinhole_chart_solid_angle_density",
    "stable_svd_2x2",
)
