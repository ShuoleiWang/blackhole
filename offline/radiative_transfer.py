"""Covariant polarized radiative transfer for offline reference rendering.

The transported state is the invariant Stokes vector

    S = (I, Q, U, V) / nu**3.

Segments MUST be supplied in physical propagation order from the source
boundary toward the observer.  Coefficients are constant within one segment
and are expressed in the segment's parallel-transported polarization basis.
For affine path parameter ``lambda`` increasing source-to-observer, this
module solves

    dS/dlambda = J - K S,

with invariant emissivity ``J = j_nu / nu**2`` and invariant propagation
coefficients ``K = nu * K_nu``.  ``TransferSegment.length`` is therefore an
affine-parameter interval.  A caller using another monotonically increasing
parameter must include the corresponding Jacobian in both ``J`` and ``K``.

``TransferCoefficients`` uses the standard passive-medium propagation matrix

        [aI, aQ,  aU,  aV]
    K = [aQ, aI,  rV, -rU]
        [aU,-rV,  aI,  rQ]
        [aV, rU, -rQ,  aI],

where ``aI`` is invariant scalar absorption, ``(aQ,aU,aV)`` is invariant
dichroism, and ``(rQ,rU,rV)`` contains invariant Faraday conversion/rotation.
The sign convention is part of this module's public contract.

Python ``float`` is IEEE-754 binary64 on supported CPython builds.  This module
uses only the standard library and rejects non-finite inputs and intermediate
states rather than returning a plausible partial radiance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence


_STOKES_COMPONENTS = 4
_COUPLING_COMPONENTS = 3
_PHYSICAL_TOLERANCE = 1.0e-12
_PIVOT_TOLERANCE = 2.0e-14


class RadiativeTransferError(ValueError):
    """Base class for fail-closed radiative-transfer errors."""


class TransferValidationError(RadiativeTransferError):
    """Raised when the transfer contract receives invalid physical input."""


class TransferIntegrationError(RadiativeTransferError):
    """Raised when a finite, stable transfer update cannot be completed."""


class StepBudgetExceeded(TransferIntegrationError):
    """Raised before a segment would exceed the declared work budget."""


def _finite_float(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise TransferValidationError(f"{label} must be a finite float")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TransferValidationError(
            f"{label} must be a finite float"
        ) from error
    if not math.isfinite(result):
        raise TransferValidationError(f"{label} must be finite")
    return result


def _finite_vector(
    value: Sequence[float],
    size: int,
    label: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TransferValidationError(f"{label} must contain {size} floats")
    try:
        entries = tuple(value)
    except TypeError as error:
        raise TransferValidationError(
            f"{label} must contain {size} floats"
        ) from error
    if len(entries) != size:
        raise TransferValidationError(f"{label} must contain {size} floats")
    return tuple(
        _finite_float(entry, f"{label}[{index}]")
        for index, entry in enumerate(entries)
    )


def _checked_sum(first: float, second: float, label: str) -> float:
    result = first + second
    if not math.isfinite(result):
        raise TransferIntegrationError(f"{label} overflowed binary64")
    return result


def _checked_product(first: float, second: float, label: str) -> float:
    result = first * second
    if not math.isfinite(result):
        raise TransferIntegrationError(f"{label} overflowed binary64")
    return result


@dataclass(frozen=True, slots=True)
class StokesInvariant:
    """Invariant Stokes vector ``(I,Q,U,V)/nu**3``.

    ``I`` must be non-negative and the polarization magnitude may not exceed
    ``I`` beyond a small binary64 roundoff allowance.
    """

    i: float = 0.0
    q: float = 0.0
    u: float = 0.0
    v: float = 0.0

    def __post_init__(self) -> None:
        values = tuple(
            _finite_float(value, f"StokesInvariant.{name}")
            for name, value in zip(("i", "q", "u", "v"), self.as_tuple())
        )
        for name, value in zip(("i", "q", "u", "v"), values):
            object.__setattr__(self, name, value)
        if self.i < 0.0:
            raise TransferValidationError(
                "StokesInvariant.i must be non-negative"
            )
        polarization = math.hypot(self.q, self.u, self.v)
        allowance = _PHYSICAL_TOLERANCE * self.i
        if polarization > self.i + allowance:
            raise TransferValidationError(
                "Stokes polarization magnitude may not exceed intensity"
            )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.i, self.q, self.u, self.v)

    @property
    def polarization_norm(self) -> float:
        return math.hypot(self.q, self.u, self.v)


@dataclass(frozen=True, slots=True)
class TransferCoefficients:
    """Constant invariant transfer coefficients for one path segment.

    The four components of ``invariant_emissivity`` mean ``j_nu / nu**2``;
    ``StokesInvariant`` is reused as their finite, physically admissible
    four-vector carrier, not as an assertion that emissivity scales as
    ``nu**-3``.  ``invariant_absorption`` is ``aI``.  Passive dichroism requires
    ``aI >= hypot(aQ,aU,aV)``; maser/gain media are intentionally outside this
    first fail-closed contract.  Faraday coefficients may have either sign.
    """

    invariant_emissivity: StokesInvariant = field(
        default_factory=StokesInvariant
    )
    invariant_absorption: float = 0.0
    invariant_dichroism: tuple[float, float, float] = (0.0, 0.0, 0.0)
    invariant_faraday: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.invariant_emissivity, StokesInvariant):
            raise TransferValidationError(
                "invariant_emissivity must be a StokesInvariant"
            )
        absorption = _finite_float(
            self.invariant_absorption,
            "TransferCoefficients.invariant_absorption",
        )
        if absorption < 0.0:
            raise TransferValidationError(
                "invariant_absorption must be non-negative"
            )
        dichroism = _finite_vector(
            self.invariant_dichroism,
            _COUPLING_COMPONENTS,
            "TransferCoefficients.invariant_dichroism",
        )
        faraday = _finite_vector(
            self.invariant_faraday,
            _COUPLING_COMPONENTS,
            "TransferCoefficients.invariant_faraday",
        )
        dichroism_norm = math.hypot(*dichroism)
        allowance = _PHYSICAL_TOLERANCE * absorption
        if dichroism_norm > absorption + allowance:
            raise TransferValidationError(
                "passive transfer requires absorption >= dichroism magnitude"
            )
        object.__setattr__(self, "invariant_absorption", absorption)
        object.__setattr__(self, "invariant_dichroism", dichroism)
        object.__setattr__(self, "invariant_faraday", faraday)

    @property
    def is_uncoupled(self) -> bool:
        """Whether all four Stokes components share one scalar absorption."""

        return (
            self.invariant_dichroism == (0.0, 0.0, 0.0)
            and self.invariant_faraday == (0.0, 0.0, 0.0)
        )

    def propagation_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Return the documented invariant 4x4 propagation matrix ``K``."""

        absorption = self.invariant_absorption
        a_q, a_u, a_v = self.invariant_dichroism
        r_q, r_u, r_v = self.invariant_faraday
        return (
            (absorption, a_q, a_u, a_v),
            (a_q, absorption, r_v, -r_u),
            (a_u, -r_v, absorption, r_q),
            (a_v, r_u, -r_q, absorption),
        )


@dataclass(frozen=True, slots=True)
class TransferSegment:
    """One constant-coefficient affine interval in source-to-observer order."""

    length: float
    coefficients: TransferCoefficients = field(
        default_factory=TransferCoefficients
    )

    def __post_init__(self) -> None:
        length = _finite_float(self.length, "TransferSegment.length")
        if length < 0.0:
            raise TransferValidationError(
                "TransferSegment.length must be non-negative"
            )
        if not isinstance(self.coefficients, TransferCoefficients):
            raise TransferValidationError(
                "TransferSegment.coefficients must be TransferCoefficients"
            )
        object.__setattr__(self, "length", length)


@dataclass(frozen=True, slots=True)
class TransferDiagnostics:
    """Deterministic audit data for one completed transfer."""

    ordering: str
    segment_count: int
    total_length: float
    scalar_optical_depth: float
    exact_slab_segments: int
    implicit_midpoint_segments: int
    implicit_midpoint_steps: int
    zero_length_segments: int
    total_steps: int
    maximum_matrix_norm: float
    maximum_step_matrix_norm: float
    maximum_convergence_error_norm: float
    minimum_pivot: float | None


@dataclass(frozen=True, slots=True)
class TransferResult:
    """Final observer-side invariant Stokes vector and integration diagnostics."""

    stokes: StokesInvariant
    diagnostics: TransferDiagnostics


def _matrix_infinity_norm(matrix: Sequence[Sequence[float]]) -> float:
    result = max(sum(abs(entry) for entry in row) for row in matrix)
    if not math.isfinite(result):
        raise TransferIntegrationError("propagation matrix norm is non-finite")
    return result


def _matrix_vector(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> tuple[float, ...]:
    result = tuple(
        sum(matrix[row][column] * vector[column] for column in range(4))
        for row in range(4)
    )
    if not all(math.isfinite(entry) for entry in result):
        raise TransferIntegrationError(
            "propagation matrix-vector product is non-finite"
        )
    return result


def _solve_4x4(
    matrix: Sequence[Sequence[float]],
    right_hand_side: Sequence[float],
) -> tuple[tuple[float, ...], float]:
    """Solve a finite 4x4 system with scaled partial pivoting."""

    augmented = [
        [float(matrix[row][column]) for column in range(4)]
        + [float(right_hand_side[row])]
        for row in range(4)
    ]
    scale = max(abs(entry) for row in augmented for entry in row[:-1])
    if not math.isfinite(scale) or scale <= 0.0:
        raise TransferIntegrationError("implicit midpoint matrix is invalid")
    minimum_pivot = math.inf
    for column in range(4):
        pivot_row = max(
            range(column, 4),
            key=lambda row: abs(augmented[row][column]),
        )
        pivot = augmented[pivot_row][column]
        if (
            not math.isfinite(pivot)
            or abs(pivot) <= _PIVOT_TOLERANCE * scale
        ):
            raise TransferIntegrationError(
                "implicit midpoint system is singular or ill-conditioned"
            )
        minimum_pivot = min(minimum_pivot, abs(pivot))
        if pivot_row != column:
            augmented[column], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[column],
            )
        pivot = augmented[column][column]
        for row in range(column + 1, 4):
            factor = augmented[row][column] / pivot
            if not math.isfinite(factor):
                raise TransferIntegrationError(
                    "implicit midpoint elimination became non-finite"
                )
            augmented[row][column] = 0.0
            for entry in range(column + 1, 5):
                augmented[row][entry] -= factor * augmented[column][entry]

    solution = [0.0] * 4
    for row in range(3, -1, -1):
        remainder = augmented[row][4] - sum(
            augmented[row][column] * solution[column]
            for column in range(row + 1, 4)
        )
        pivot = augmented[row][row]
        value = remainder / pivot
        if not math.isfinite(value):
            raise TransferIntegrationError(
                "implicit midpoint solution became non-finite"
            )
        solution[row] = value
    return tuple(solution), minimum_pivot


def _exact_uncoupled_slab(
    state: Sequence[float],
    coefficients: TransferCoefficients,
    length: float,
) -> tuple[float, ...]:
    absorption = coefficients.invariant_absorption
    emissivity = coefficients.invariant_emissivity.as_tuple()
    if absorption == 0.0:
        result = tuple(
            state[index] + emissivity[index] * length
            for index in range(_STOKES_COMPONENTS)
        )
    else:
        optical_depth = _checked_product(
            absorption,
            length,
            "segment optical depth",
        )
        if optical_depth == 0.0:
            # A finite positive absorption can underflow in the product a*L.
            # The exact limiting formal solution is transparent propagation
            # plus j*L, not zero emitted intensity.
            attenuation = 1.0
            emission_weight = length
        else:
            attenuation = math.exp(-optical_depth)
            emission_weight = -math.expm1(-optical_depth) / absorption
        result = tuple(
            state[index] * attenuation
            + emissivity[index] * emission_weight
            for index in range(_STOKES_COMPONENTS)
        )
    if not all(math.isfinite(entry) for entry in result):
        raise TransferIntegrationError(
            "exact slab solution produced a non-finite Stokes vector"
        )
    return result


def _implicit_midpoint_segment(
    state: Sequence[float],
    coefficients: TransferCoefficients,
    length: float,
    steps: int,
) -> tuple[tuple[float, ...], float, float]:
    matrix = coefficients.propagation_matrix()
    matrix_norm = _matrix_infinity_norm(matrix)
    step_length = length / steps
    half_step = 0.5 * step_length
    left_matrix = tuple(
        tuple(
            (1.0 if row == column else 0.0)
            + half_step * matrix[row][column]
            for column in range(4)
        )
        for row in range(4)
    )
    emissivity = coefficients.invariant_emissivity.as_tuple()
    current = tuple(state)
    minimum_pivot = math.inf
    for _ in range(steps):
        matrix_state = _matrix_vector(matrix, current)
        right_hand_side = tuple(
            current[index]
            - half_step * matrix_state[index]
            + step_length * emissivity[index]
            for index in range(4)
        )
        if not all(math.isfinite(entry) for entry in right_hand_side):
            raise TransferIntegrationError(
                "implicit midpoint right-hand side became non-finite"
            )
        current, pivot = _solve_4x4(left_matrix, right_hand_side)
        minimum_pivot = min(minimum_pivot, pivot)
    return current, minimum_pivot, step_length * matrix_norm


def _validated_positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TransferValidationError(f"{label} must be a positive integer")
    return value


def _ceil_positive_float_ratio(
    numerator: float,
    denominator: float,
) -> int:
    """Return ``ceil(numerator / denominator)`` without float overflow.

    Both arguments are already-validated, positive finite binary64 values.
    Using their exact integer ratios keeps an extreme interaction-to-step-limit
    ratio from becoming ``inf`` before it can be rejected by the step budget.
    """

    numerator_integer, numerator_denominator = numerator.as_integer_ratio()
    denominator_integer, denominator_denominator = denominator.as_integer_ratio()
    dividend = numerator_integer * denominator_denominator
    divisor = numerator_denominator * denominator_integer
    return (dividend + divisor - 1) // divisor


def _integrate_coupled_with_local_convergence(
    state: Sequence[float],
    coefficients: TransferCoefficients,
    length: float,
    total_transfer_length: float,
    initial_intervals: int,
    maximum_work_steps: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[tuple[float, ...], int, float, float, float]:
    """Integrate one constant segment with local midpoint step doubling.

    Comparing only whole-segment endpoints can alias an oscillatory Faraday
    phase by an integer number of turns.  Here every interval is compared
    locally (one full step versus two half steps) before the state advances.
    The per-interval allowance is weighted by its fraction of total affine
    length, so accepted local errors sum to the declared segment tolerance.
    """

    work_steps = 0
    minimum_pivot = math.inf
    maximum_step_norm = 0.0
    maximum_accepted_error_norm = 0.0

    def integrate_interval(
        interval_input: tuple[float, ...],
        interval_length: float,
    ) -> tuple[float, ...]:
        nonlocal work_steps
        nonlocal minimum_pivot
        nonlocal maximum_step_norm
        nonlocal maximum_accepted_error_norm

        if work_steps + 3 > maximum_work_steps:
            raise StepBudgetExceeded(
                "coupled transfer failed its local convergence gate before "
                "the declared step budget"
            )
        full, full_pivot, full_step_norm = _implicit_midpoint_segment(
            interval_input,
            coefficients,
            interval_length,
            1,
        )
        fine, fine_pivot, fine_step_norm = _implicit_midpoint_segment(
            interval_input,
            coefficients,
            interval_length,
            2,
        )
        work_steps += 3
        minimum_pivot = min(minimum_pivot, full_pivot, fine_pivot)
        maximum_step_norm = max(
            maximum_step_norm,
            full_step_norm,
            fine_step_norm,
        )

        affine_fraction = interval_length / total_transfer_length
        state_scale = max(
            max(abs(value) for value in fine),
            max(abs(value) for value in full),
        )
        error_norm = (
            max(
                abs(fine[index] - full[index])
                for index in range(_STOKES_COMPONENTS)
            )
            / 3.0
            / (
                affine_fraction
                * (
                    absolute_tolerance
                    + relative_tolerance * state_scale
                )
            )
        )
        if not math.isfinite(error_norm):
            raise TransferIntegrationError(
                "local coupled-transfer convergence estimate is non-finite"
            )
        if error_norm <= 1.0:
            maximum_accepted_error_norm = max(
                maximum_accepted_error_norm,
                error_norm,
            )
            return fine

        half_length = 0.5 * interval_length
        if half_length == 0.0 or half_length == interval_length:
            raise TransferIntegrationError(
                "coupled transfer interval cannot be refined further"
            )
        midpoint_state = integrate_interval(interval_input, half_length)
        return integrate_interval(midpoint_state, half_length)

    current = tuple(state)
    interval_length = length / initial_intervals
    if interval_length <= 0.0 or not math.isfinite(interval_length):
        raise TransferIntegrationError("coupled transfer interval is invalid")
    for _interval in range(initial_intervals):
        current = integrate_interval(current, interval_length)
    return (
        current,
        work_steps,
        minimum_pivot,
        maximum_step_norm,
        maximum_accepted_error_norm,
    )


def propagate_source_to_observer(
    source_stokes: StokesInvariant,
    segments: Iterable[TransferSegment],
    *,
    maximum_steps: int = 100_000,
    maximum_step_matrix_norm: float = 0.25,
    absolute_tolerance: float = 1.0e-10,
    relative_tolerance: float = 1.0e-5,
) -> TransferResult:
    """Propagate invariant Stokes data from the source boundary to the observer.

    ``segments`` is consumed exactly in the supplied order.  The first segment
    is nearest the source boundary and the last is nearest the observer; the
    function never sorts or reverses them.

    Uncoupled scalar-absorption segments use the exact homogeneous-slab formal
    solution.  Coupled dichroic/Faraday segments use the A-stable implicit
    midpoint rule, subdivided so ``h * ||K||_inf`` does not exceed
    ``maximum_step_matrix_norm``.  Every local interval is compared against two
    half steps and recursively refined until its second-order Richardson
    estimate satisfies ``absolute_tolerance`` and ``relative_tolerance``.  This
    local gate cannot hide a Faraday phase error behind a whole-turn endpoint
    alias.  ``maximum_steps`` counts every convergence attempt and is checked
    before each solve, so failures never return a partial result.
    """

    if not isinstance(source_stokes, StokesInvariant):
        raise TransferValidationError(
            "source_stokes must be a StokesInvariant"
        )
    step_budget = _validated_positive_integer(maximum_steps, "maximum_steps")
    step_norm_limit = _finite_float(
        maximum_step_matrix_norm,
        "maximum_step_matrix_norm",
    )
    if step_norm_limit <= 0.0:
        raise TransferValidationError(
            "maximum_step_matrix_norm must be positive"
        )
    absolute_error_limit = _finite_float(
        absolute_tolerance,
        "absolute_tolerance",
    )
    relative_error_limit = _finite_float(
        relative_tolerance,
        "relative_tolerance",
    )
    if absolute_error_limit <= 0.0 or relative_error_limit <= 0.0:
        raise TransferValidationError(
            "transfer absolute and relative tolerances must be positive"
        )
    try:
        iterator = iter(segments)
    except TypeError as error:
        raise TransferValidationError("segments must be iterable") from error

    materialized_segments: list[TransferSegment] = []
    materialized_total_length = 0.0
    for segment_index, segment in enumerate(iterator):
        if segment_index >= step_budget:
            raise StepBudgetExceeded(
                "segment count exceeds the declared transfer step budget"
            )
        if not isinstance(segment, TransferSegment):
            raise TransferValidationError(
                f"segments[{segment_index}] must be a TransferSegment"
            )
        materialized_segments.append(segment)
        materialized_total_length = _checked_sum(
            materialized_total_length,
            segment.length,
            "total path length",
        )

    state = source_stokes.as_tuple()
    segment_count = 0
    total_length = 0.0
    scalar_optical_depth = 0.0
    exact_slab_segments = 0
    implicit_midpoint_segments = 0
    implicit_midpoint_steps = 0
    zero_length_segments = 0
    total_steps = 0
    maximum_matrix_norm = 0.0
    maximum_actual_step_norm = 0.0
    maximum_convergence_error_norm = 0.0
    minimum_pivot: float | None = None

    for segment_index, segment in enumerate(materialized_segments):
        segment_count += 1
        total_length = _checked_sum(
            total_length,
            segment.length,
            "total path length",
        )
        segment_optical_depth = _checked_product(
            segment.coefficients.invariant_absorption,
            segment.length,
            f"segments[{segment_index}] scalar optical depth",
        )
        scalar_optical_depth = _checked_sum(
            scalar_optical_depth,
            segment_optical_depth,
            "total scalar optical depth",
        )
        if segment.length == 0.0:
            required_steps = 1
            if total_steps + required_steps > step_budget:
                raise StepBudgetExceeded(
                    f"maximum_steps={step_budget} exhausted before "
                    f"segments[{segment_index}]"
                )
            total_steps += required_steps
            zero_length_segments += 1
            continue

        matrix = segment.coefficients.propagation_matrix()
        matrix_norm = _matrix_infinity_norm(matrix)
        maximum_matrix_norm = max(maximum_matrix_norm, matrix_norm)

        if segment.coefficients.is_uncoupled:
            required_steps = 1
            if total_steps + required_steps > step_budget:
                raise StepBudgetExceeded(
                    f"maximum_steps={step_budget} exhausted before "
                    f"segments[{segment_index}]"
                )
            state = _exact_uncoupled_slab(
                state,
                segment.coefficients,
                segment.length,
            )
            total_steps += required_steps
            exact_slab_segments += 1
            continue

        interaction_norm = _checked_product(
            segment.length,
            matrix_norm,
            f"segments[{segment_index}] interaction norm",
        )
        required_steps = max(
            1,
            _ceil_positive_float_ratio(
                interaction_norm,
                step_norm_limit,
            ),
        )
        first_attempt_steps = 3 * required_steps
        if total_steps + first_attempt_steps > step_budget:
            raise StepBudgetExceeded(
                f"segments[{segment_index}] needs at least "
                f"{first_attempt_steps} implicit convergence steps but only "
                f"{step_budget - total_steps} remain"
            )
        try:
            (
                state,
                segment_work_steps,
                segment_minimum_pivot,
                actual_step_norm,
                segment_convergence_error,
            ) = _integrate_coupled_with_local_convergence(
                state,
                segment.coefficients,
                segment.length,
                materialized_total_length,
                required_steps,
                step_budget - total_steps,
                absolute_error_limit,
                relative_error_limit,
            )
        except StepBudgetExceeded as error:
            raise StepBudgetExceeded(
                f"segments[{segment_index}] failed the local transfer "
                f"convergence gate before maximum_steps={step_budget}"
            ) from error
        total_steps += segment_work_steps
        implicit_midpoint_steps += segment_work_steps
        maximum_convergence_error_norm = max(
            maximum_convergence_error_norm,
            segment_convergence_error,
        )

        implicit_midpoint_segments += 1
        maximum_actual_step_norm = max(
            maximum_actual_step_norm,
            actual_step_norm,
        )
        minimum_pivot = (
            segment_minimum_pivot
            if minimum_pivot is None
            else min(minimum_pivot, segment_minimum_pivot)
        )

    try:
        observer_stokes = StokesInvariant(*state)
    except TransferValidationError as error:
        raise TransferIntegrationError(
            "transfer produced a non-physical observer Stokes vector"
        ) from error
    diagnostics = TransferDiagnostics(
        ordering="source-to-observer",
        segment_count=segment_count,
        total_length=total_length,
        scalar_optical_depth=scalar_optical_depth,
        exact_slab_segments=exact_slab_segments,
        implicit_midpoint_segments=implicit_midpoint_segments,
        implicit_midpoint_steps=implicit_midpoint_steps,
        zero_length_segments=zero_length_segments,
        total_steps=total_steps,
        maximum_matrix_norm=maximum_matrix_norm,
        maximum_step_matrix_norm=maximum_actual_step_norm,
        maximum_convergence_error_norm=maximum_convergence_error_norm,
        minimum_pivot=minimum_pivot,
    )
    return TransferResult(stokes=observer_stokes, diagnostics=diagnostics)
