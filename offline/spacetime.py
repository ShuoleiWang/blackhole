"""Metric-provider boundary for the high-fidelity offline renderer.

The realtime renderer intentionally owns a different, frame-frozen contract.
Offline rays use the full four-dimensional Hamiltonian and ask a provider for
``g^{mu nu}`` and all four coordinate derivatives at every sampled event.  A
future NR adapter can therefore interpolate a time-dependent metric without
changing the geodesic integrator.

The analytic providers in this module are calibration sources.  They are not
numerical-relativity data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable


Vector4: TypeAlias = tuple[float, float, float, float]
Matrix4: TypeAlias = tuple[Vector4, Vector4, Vector4, Vector4]
MetricDerivatives: TypeAlias = tuple[Matrix4, Matrix4, Matrix4, Matrix4]
METRIC_CONSISTENCY_TOLERANCE = 2.0e-10
METRIC_CONSISTENCY_ABSOLUTE_TOLERANCE = 2.0e-12

MINKOWSKI_COVARIANT: Matrix4 = (
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
ZERO_MATRIX4: Matrix4 = (
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
)
ZERO_DERIVATIVES: MetricDerivatives = (
    ZERO_MATRIX4,
    ZERO_MATRIX4,
    ZERO_MATRIX4,
    ZERO_MATRIX4,
)


def _finite_vector(values: Vector4) -> bool:
    return all(math.isfinite(value) for value in values)


def matrix_vector(matrix: Matrix4, vector: Vector4) -> Vector4:
    """Multiply a row-major 4x4 matrix by a four-vector."""
    return tuple(  # type: ignore[return-value]
        math.fsum(matrix[row][column] * vector[column] for column in range(4))
        for row in range(4)
    )


def bilinear(left: Vector4, matrix: Matrix4, right: Vector4) -> float:
    """Return ``left^T matrix right`` with compensated summation."""
    product = matrix_vector(matrix, right)
    return math.fsum(left[index] * product[index] for index in range(4))


def _require_lorentzian_minus_plus_signature(matrix: Matrix4) -> None:
    """Verify one negative and three positive eigenvalues by Jacobi sweeps."""

    scale = max(abs(value) for row in matrix for value in row)
    if not math.isfinite(scale) or scale == 0.0:
        raise ValueError("metric tensor must be non-degenerate Lorentzian")
    working = [
        [matrix[row][column] / scale for column in range(4)]
        for row in range(4)
    ]
    for _sweep in range(64):
        off_diagonal, first, second = max(
            (
                (abs(working[row][column]), row, column)
                for row in range(4)
                for column in range(row + 1, 4)
            ),
            key=lambda item: item[0],
        )
        if off_diagonal <= 2.0e-14:
            break
        coupling = working[first][second]
        tau = (
            working[second][second] - working[first][first]
        ) / (2.0 * coupling)
        tangent = (
            1.0 if tau >= 0.0 else -1.0
        ) / (abs(tau) + math.sqrt(1.0 + tau * tau))
        cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine
        first_diagonal = working[first][first]
        second_diagonal = working[second][second]
        working[first][first] = (
            cosine * cosine * first_diagonal
            - 2.0 * sine * cosine * coupling
            + sine * sine * second_diagonal
        )
        working[second][second] = (
            sine * sine * first_diagonal
            + 2.0 * sine * cosine * coupling
            + cosine * cosine * second_diagonal
        )
        working[first][second] = 0.0
        working[second][first] = 0.0
        for index in range(4):
            if index in (first, second):
                continue
            old_first = working[index][first]
            old_second = working[index][second]
            new_first = cosine * old_first - sine * old_second
            new_second = sine * old_first + cosine * old_second
            working[index][first] = new_first
            working[first][index] = new_first
            working[index][second] = new_second
            working[second][index] = new_second
    else:
        raise ValueError("metric signature eigensolve did not converge")

    eigenvalues = tuple(working[index][index] for index in range(4))
    if any(abs(value) <= 2.0e-12 for value in eigenvalues):
        raise ValueError("metric tensor must be non-degenerate Lorentzian")
    if sum(value < 0.0 for value in eigenvalues) != 1:
        raise ValueError("metric tensor must use Lorentzian -+++ signature")


@dataclass(frozen=True)
class MetricSample:
    """Metric tensors and analytic derivatives at one coordinate event.

    ``inverse_derivatives[mu][alpha][beta]`` is
    ``partial_mu g^{alpha beta}``.  ``interpolation_error`` is a provider-owned
    finite, non-negative dimensionless error indicator; analytic providers set
    it to zero.  Offline NR adapters must normalize and expose their worst
    interpolation error instead of hiding it in a ray budget.
    """

    covariant: Matrix4
    inverse: Matrix4
    inverse_derivatives: MetricDerivatives
    interpolation_error: float = 0.0

    def __post_init__(self) -> None:
        if len(self.covariant) != 4 or len(self.inverse) != 4:
            raise ValueError("metric tensors must have exactly four rows")
        if len(self.inverse_derivatives) != 4:
            raise ValueError("metric must provide exactly four coordinate derivatives")
        if any(len(derivative) != 4 for derivative in self.inverse_derivatives):
            raise ValueError("each metric derivative must have exactly four rows")
        rows = (*self.covariant, *self.inverse)
        derivative_rows = tuple(
            row for derivative in self.inverse_derivatives for row in derivative
        )
        if any(len(row) != 4 for row in (*rows, *derivative_rows)):
            raise ValueError("metric tensors must be 4x4")
        if not all(_finite_vector(row) for row in (*rows, *derivative_rows)):
            raise ValueError("metric sample must contain only finite values")
        if (
            not math.isfinite(self.interpolation_error)
            or self.interpolation_error < 0.0
        ):
            raise ValueError("metric interpolation error must be finite and non-negative")

        matrices = (self.covariant, self.inverse, *self.inverse_derivatives)
        for matrix in matrices:
            symmetry_is_valid = all(
                abs(matrix[row][column] - matrix[column][row])
                <= METRIC_CONSISTENCY_ABSOLUTE_TOLERANCE
                + METRIC_CONSISTENCY_TOLERANCE
                * max(
                    abs(matrix[row][column]),
                    abs(matrix[column][row]),
                )
                for row in range(4)
                for column in range(4)
            )
            if not symmetry_is_valid:
                raise ValueError("metric tensors and derivatives must be symmetric")

        for row in range(4):
            for column in range(4):
                terms = tuple(
                    self.covariant[row][inner] * self.inverse[inner][column]
                    for inner in range(4)
                )
                if any(not math.isfinite(term) for term in terms):
                    raise ValueError(
                        "covariant and inverse metric product is non-finite"
                    )
                product = math.fsum(terms)
                sum_absolute = math.fsum(abs(term) for term in terms)
                if not math.isfinite(product) or not math.isfinite(sum_absolute):
                    raise ValueError(
                        "covariant and inverse metric product is non-finite"
                    )
                expected = float(row == column)
                tolerance = (
                    METRIC_CONSISTENCY_ABSOLUTE_TOLERANCE
                    + METRIC_CONSISTENCY_TOLERANCE
                    * max(1.0, abs(expected))
                )
                if abs(product - expected) > tolerance:
                    raise ValueError(
                        "covariant and inverse metric tensors are inconsistent"
                    )

        _require_lorentzian_minus_plus_signature(self.covariant)


@runtime_checkable
class MetricProvider(Protocol):
    """Four-dimensional metric source consumed by the offline integrator."""

    source_id: str
    time_dependent: bool

    def sample(self, event: Vector4) -> MetricSample:
        """Return the metric and derivatives at ``(t, x, y, z)``."""


@dataclass(frozen=True)
class MinkowskiMetric:
    """Exact flat-spacetime calibration provider."""

    source_id: str = "analytic-minkowski"
    time_dependent: bool = False

    def sample(self, event: Vector4) -> MetricSample:
        if not _finite_vector(event):
            raise ValueError("metric event must be finite")
        return MetricSample(
            covariant=MINKOWSKI_COVARIANT,
            inverse=MINKOWSKI_COVARIANT,
            inverse_derivatives=ZERO_DERIVATIVES,
        )


@dataclass(frozen=True)
class SchwarzschildKerrSchildMetric:
    """Exact Schwarzschild metric in ingoing Cartesian Kerr-Schild form.

    With signature ``-+++`` and ``H=M/r`` the metric is

    ``g_mn = eta_mn + 2 H l_m l_n`` and
    ``g^mn = eta^mn - 2 H l^m l^n``,

    where ``l_m=(1,n_i)`` and ``l^m=(-1,n_i)``.  This horizon-penetrating
    chart avoids the Schwarzschild-coordinate singularity and is also a useful
    stationary oracle for future NR adapters.
    """

    mass_m: float = 1.0
    singularity_guard_m: float = 1.0e-9
    source_id: str = "analytic-schwarzschild-kerr-schild"
    time_dependent: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.mass_m) or self.mass_m <= 0.0:
            raise ValueError("Schwarzschild mass must be finite and positive")
        if (
            not math.isfinite(self.singularity_guard_m)
            or self.singularity_guard_m <= 0.0
        ):
            raise ValueError("singularity guard must be finite and positive")

    def sample(self, event: Vector4) -> MetricSample:
        if not _finite_vector(event):
            raise ValueError("metric event must be finite")
        _time, x, y, z = event
        radius = math.sqrt(x * x + y * y + z * z)
        if radius <= self.singularity_guard_m:
            raise ValueError("Schwarzschild metric sampled at guarded singularity")

        normal = (x / radius, y / radius, z / radius)
        h = self.mass_m / radius
        l_covariant: Vector4 = (1.0, *normal)
        l_contravariant: Vector4 = (-1.0, *normal)

        covariant = tuple(  # type: ignore[assignment]
            tuple(
                MINKOWSKI_COVARIANT[row][column]
                + 2.0 * h * l_covariant[row] * l_covariant[column]
                for column in range(4)
            )
            for row in range(4)
        )
        inverse = tuple(  # type: ignore[assignment]
            tuple(
                MINKOWSKI_COVARIANT[row][column]
                - 2.0 * h * l_contravariant[row] * l_contravariant[column]
                for column in range(4)
            )
            for row in range(4)
        )

        derivatives: list[Matrix4] = [ZERO_MATRIX4]
        for spatial_derivative in range(3):
            normal_component = normal[spatial_derivative]
            derivative_h = -h * normal_component / radius
            derivative_l: Vector4 = (
                0.0,
                *tuple(
                    (
                        (1.0 if component == spatial_derivative else 0.0)
                        - normal[component] * normal_component
                    )
                    / radius
                    for component in range(3)
                ),
            )
            derivative = tuple(  # type: ignore[assignment]
                tuple(
                    -2.0
                    * (
                        derivative_h
                        * l_contravariant[row]
                        * l_contravariant[column]
                        + h
                        * derivative_l[row]
                        * l_contravariant[column]
                        + h
                        * l_contravariant[row]
                        * derivative_l[column]
                    )
                    for column in range(4)
                )
                for row in range(4)
            )
            derivatives.append(derivative)

        return MetricSample(
            covariant=covariant,
            inverse=inverse,
            inverse_derivatives=tuple(derivatives),  # type: ignore[arg-type]
        )
