"""Float64 full four-dimensional null-Hamiltonian integration.

Unlike the realtime fast-light shader, this module never assumes that ``p_t``
is conserved.  Every derivative evaluation samples the metric at the ray's
current coordinate event, including coordinate time.  That is the required
ownership boundary for future slow-light NR providers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import (
    Callable,
    Iterable,
    Protocol,
    Sequence,
    TypeAlias,
    runtime_checkable,
)

from offline.spacetime import (
    MetricProvider,
    Vector4,
    bilinear,
    matrix_vector,
)


State8: TypeAlias = tuple[float, float, float, float, float, float, float, float]


@dataclass(frozen=True)
class HamiltonianState:
    event: Vector4
    covector: Vector4

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (*self.event, *self.covector)):
            raise ValueError("Hamiltonian state must be finite")

    def packed(self) -> State8:
        return (*self.event, *self.covector)

    @classmethod
    def unpack(cls, values: Sequence[float]) -> "HamiltonianState":
        if len(values) != 8:
            raise ValueError("Hamiltonian state needs eight scalars")
        return cls(
            event=tuple(float(value) for value in values[:4]),  # type: ignore[arg-type]
            covector=tuple(float(value) for value in values[4:]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class RayPathSegment:
    """One accepted observer-to-source affine segment."""

    start: HamiltonianState
    end: HamiltonianState
    midpoint: HamiltonianState
    affine_length: float
    midpoint_null_residual: float


@dataclass(frozen=True)
class RayTraceOptions:
    absolute_tolerance: float = 2.0e-10
    relative_tolerance: float = 2.0e-10
    initial_step: float = 0.05
    minimum_step: float = 1.0e-8
    maximum_step: float = 1.0
    maximum_affine_length: float = 4_000.0
    maximum_accepted_steps: int = 100_000
    maximum_rejected_steps: int = 100_000
    null_residual_limit: float = 1.0e-7
    metric_interpolation_error_limit: float = 1.0e-7
    event_value_tolerance: float = 1.0e-9
    event_affine_tolerance: float = 1.0e-10
    event_maximum_iterations: int = 64
    record_path: bool = False

    def __post_init__(self) -> None:
        positive = (
            self.absolute_tolerance,
            self.relative_tolerance,
            self.initial_step,
            self.minimum_step,
            self.maximum_step,
            self.maximum_affine_length,
            self.null_residual_limit,
            self.metric_interpolation_error_limit,
            self.event_value_tolerance,
            self.event_affine_tolerance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("ray tolerances, steps, budgets, and limits must be positive")
        if self.minimum_step > self.initial_step or self.initial_step > self.maximum_step:
            raise ValueError("ray step bounds must satisfy min <= initial <= max")
        integer_budgets = (
            self.maximum_accepted_steps,
            self.maximum_rejected_steps,
            self.event_maximum_iterations,
        )
        if any(type(value) is not int or value < 1 for value in integer_budgets):
            raise ValueError("ray step budgets must be positive integers")


class RecordedPathSamplingError(RuntimeError):
    """Raised when a recorded ray state cannot be certified by reintegration."""


class SurfaceEventError(RecordedPathSamplingError):
    """Raised when a recorded-path surface event cannot be proven safely."""


@dataclass(frozen=True)
class RecordedPathSamplingOptions:
    """Accuracy and work limits for certified recorded-path state sampling.

    Samples are reconstructed from an accepted segment's start with the same
    four-dimensional Hamiltonian and Dormand--Prince 5(4) stepper used by the
    ray tracer.  They are never coordinate or covector interpolations.
    ``maximum_reintegrations`` belongs to one sampler instance and therefore
    remains a global work budget across every segment sampled through it.
    """

    absolute_tolerance: float = 2.0e-10
    relative_tolerance: float = 2.0e-10
    null_residual_limit: float = 1.0e-7
    metric_interpolation_error_limit: float = 1.0e-7
    maximum_reintegrations: int = 100_000

    def __post_init__(self) -> None:
        positive = (
            self.absolute_tolerance,
            self.relative_tolerance,
            self.null_residual_limit,
            self.metric_interpolation_error_limit,
        )
        if any(
            isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0.0
            for value in positive
        ):
            raise ValueError("recorded-path sampling limits must be positive")
        if (
            type(self.maximum_reintegrations) is not int
            or self.maximum_reintegrations < 1
        ):
            raise ValueError(
                "maximum_reintegrations must be a positive integer"
            )


@dataclass(frozen=True)
class RecordedPathSamplingDiagnostics:
    """Audit counters for one certified recorded-path sampler."""

    reintegrations: int
    maximum_null_residual: float
    maximum_metric_interpolation_error: float


@dataclass(frozen=True)
class SurfaceEventOptions:
    """Accuracy and work limits for recorded-path surface localization.

    ``subdivisions_per_segment`` is an even number so every recorded midpoint
    is independently reconstructed and checked.  More subdivisions permit more
    than two signed crossings to be bracketed inside one accepted geodesic
    segment; no unbracketed contact is ever promoted to a crossing.
    """

    absolute_tolerance: float = 2.0e-10
    relative_tolerance: float = 2.0e-10
    null_residual_limit: float = 1.0e-7
    metric_interpolation_error_limit: float = 1.0e-7
    surface_value_tolerance: float = 1.0e-9
    affine_tolerance: float = 1.0e-10
    maximum_iterations: int = 64
    maximum_reintegrations: int = 100_000
    subdivisions_per_segment: int = 2

    def __post_init__(self) -> None:
        positive = (
            self.absolute_tolerance,
            self.relative_tolerance,
            self.null_residual_limit,
            self.metric_interpolation_error_limit,
            self.surface_value_tolerance,
            self.affine_tolerance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("surface-event tolerances and limits must be positive")
        integer_budgets = (
            self.maximum_iterations,
            self.maximum_reintegrations,
            self.subdivisions_per_segment,
        )
        if any(type(value) is not int or value < 1 for value in integer_budgets):
            raise ValueError("surface-event budgets must be positive integers")
        if self.subdivisions_per_segment % 2:
            raise ValueError("surface-event subdivisions must be even")


@dataclass(frozen=True)
class InitialMultiSurfaceContact:
    """Declare the signed side of one authenticated initial member contact.

    The real member scalar is first required to lie within the declared
    ``surface_value_tolerance``.  Only the affine-zero probe is assigned this
    side; every positive-affine probe evaluates the physical scalar.
    """

    surface_id: str
    side: int

    def __post_init__(self) -> None:
        if type(self.surface_id) is not str or not self.surface_id:
            raise ValueError(
                "initial multi-surface contact id must be an exact non-empty string"
            )
        if type(self.side) is not int or self.side not in (-1, 1):
            raise ValueError("initial multi-surface contact side must be -1 or +1")


@dataclass(frozen=True)
class AuthenticatedInitialMultiSurfaceContact:
    """Trace-owned evidence for one applied affine-zero side declaration."""

    surface_id: str
    side: int
    actual_surface_value: float
    surface_value_tolerance: float

    def __post_init__(self) -> None:
        InitialMultiSurfaceContact(self.surface_id, self.side)
        if (
            type(self.actual_surface_value) is not float
            or not math.isfinite(self.actual_surface_value)
        ):
            raise ValueError("authenticated initial surface value must be finite")
        if (
            type(self.surface_value_tolerance) is not float
            or not math.isfinite(self.surface_value_tolerance)
            or self.surface_value_tolerance <= 0.0
        ):
            raise ValueError(
                "authenticated initial surface tolerance must be positive"
            )
        if abs(self.actual_surface_value) > self.surface_value_tolerance:
            raise ValueError(
                "authenticated initial contact exceeds its surface tolerance"
            )


@dataclass(frozen=True)
class RecordedSurfaceCrossing:
    """One proven signed surface crossing in observer-to-source order."""

    state: HamiltonianState
    ray_affine_length: float
    segment_index: int
    segment_affine_length: float
    orientation: int
    surface_value: float
    bracket_affine_width: float
    iterations: int

    def __post_init__(self) -> None:
        if type(self.segment_index) is not int or self.segment_index < 0:
            raise ValueError("surface crossing segment_index must be non-negative")
        if type(self.iterations) is not int or self.iterations < 0:
            raise ValueError("surface crossing iterations must be non-negative")
        if self.orientation not in (-1, 1):
            raise ValueError("surface crossing orientation must be -1 or +1")
        finite_non_negative = (
            self.ray_affine_length,
            self.segment_affine_length,
            self.bracket_affine_width,
        )
        if any(
            not math.isfinite(value) or value < 0.0
            for value in finite_non_negative
        ):
            raise ValueError("surface crossing affine diagnostics must be finite")
        if not math.isfinite(self.surface_value):
            raise ValueError("surface crossing value must be finite")


@dataclass(frozen=True)
class InteriorSurfaceDecision:
    """Classification of one signed crossing inside an accepted ray step.

    A decision with no ``outcome`` is transparent and integration continues.
    Supplying both ``outcome`` and ``target_id`` makes the crossing terminal.
    The generic integrator owns only ordering and numerical localization; a
    product adapter owns the physical meaning of ``classification``.
    """

    classification: str
    outcome: str | None = None
    target_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.classification, str) or not self.classification:
            raise ValueError("interior surface classification must be non-empty")
        if (self.outcome is None) != (self.target_id is None):
            raise ValueError(
                "interior surface outcome and target_id must be supplied together"
            )
        if self.outcome is not None and (
            not isinstance(self.outcome, str)
            or not self.outcome
            or not isinstance(self.target_id, str)
            or not self.target_id
        ):
            raise ValueError("terminal interior surface decision is malformed")

    @property
    def terminates(self) -> bool:
        return self.outcome is not None


@runtime_checkable
class InteriorSurface(Protocol):
    """Signed scalar surface classified after Hamiltonian root localization."""

    def value(self, state: HamiltonianState) -> float:
        """Return a finite signed scalar; zero defines the surface."""

    def classify(
        self,
        crossing: RecordedSurfaceCrossing,
    ) -> InteriorSurfaceDecision:
        """Classify a proven crossing as transparent or terminal."""


@runtime_checkable
class MultiInteriorSurface(Protocol):
    """A stable-id family of independently signed interior surfaces.

    Every member retains its own signed scalar.  The accepted-step scanner
    reconstructs one shared Hamiltonian probe grid and evaluates all member
    scalars on those states; it never multiplies member surfaces into a
    degenerate composite scalar.  ``surface_ids`` must remain a non-empty set
    of unique, non-empty strings for the lifetime of a trace.  Their returned
    order has no physical meaning.
    """

    @property
    def surface_ids(self) -> tuple[str, ...]:
        """Return stable surface identifiers in any order."""

    def value(
        self,
        surface_id: str,
        state: HamiltonianState,
    ) -> float:
        """Return the finite signed scalar owned by ``surface_id``."""

    def classify(
        self,
        surface_id: str,
        crossing: RecordedSurfaceCrossing,
    ) -> InteriorSurfaceDecision:
        """Classify one proven member crossing."""


@dataclass(frozen=True)
class ClassifiedInteriorSurfaceCrossing:
    """A localized crossing and its product-owned physical classification."""

    crossing: RecordedSurfaceCrossing
    decision: InteriorSurfaceDecision

    def __post_init__(self) -> None:
        if not isinstance(self.crossing, RecordedSurfaceCrossing):
            raise TypeError("classified crossing needs a RecordedSurfaceCrossing")
        if not isinstance(self.decision, InteriorSurfaceDecision):
            raise TypeError("classified crossing needs an InteriorSurfaceDecision")


@dataclass(frozen=True)
class ClassifiedMultiInteriorSurfaceCrossing:
    """One stable-id member crossing in globally ordered ray sequence."""

    surface_id: str
    crossing: RecordedSurfaceCrossing
    decision: InteriorSurfaceDecision

    def __post_init__(self) -> None:
        if not isinstance(self.surface_id, str) or not self.surface_id:
            raise ValueError("multi-surface crossing needs a non-empty surface id")
        if not isinstance(self.crossing, RecordedSurfaceCrossing):
            raise TypeError("multi-surface crossing needs a RecordedSurfaceCrossing")
        if not isinstance(self.decision, InteriorSurfaceDecision):
            raise TypeError(
                "multi-surface crossing needs an InteriorSurfaceDecision"
            )


@dataclass(frozen=True)
class InteriorSurfaceTrace:
    """Accepted observer-to-source prefix certified at declared probe resolution.

    ``topology_converged`` means that independent ``N`` and ``2N`` probe grids
    agreed on every signed crossing up to the first terminal interior surface.
    It is deliberately not a claim of mathematical surface completeness: a
    finite grid cannot exclude arbitrarily high-frequency even root pairs.
    """

    crossings: tuple[ClassifiedInteriorSurfaceCrossing, ...]
    base_subdivisions_per_step: int
    verification_subdivisions_per_step: int
    topology_converged: bool
    maximum_probe_event_difference: float
    maximum_probe_covector_relative_difference: float

    def __post_init__(self) -> None:
        if (
            type(self.base_subdivisions_per_step) is not int
            or self.base_subdivisions_per_step < 2
            or self.base_subdivisions_per_step % 2
        ):
            raise ValueError("base surface subdivisions must be a positive even N")
        if self.verification_subdivisions_per_step != (
            2 * self.base_subdivisions_per_step
        ):
            raise ValueError("verification surface subdivisions must equal 2N")
        if type(self.topology_converged) is not bool:
            raise TypeError("topology_converged must be a bool")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.maximum_probe_event_difference,
                self.maximum_probe_covector_relative_difference,
            )
        ):
            raise ValueError("surface convergence differences must be finite")
        previous_affine = -math.inf
        terminal_seen = False
        for entry in self.crossings:
            if not isinstance(entry, ClassifiedInteriorSurfaceCrossing):
                raise TypeError("surface trace contains an invalid crossing")
            affine = entry.crossing.ray_affine_length
            if affine <= previous_affine:
                raise ValueError("surface trace crossings must be strictly ordered")
            if terminal_seen:
                raise ValueError("surface trace may not continue beyond a terminal hit")
            terminal_seen = entry.decision.terminates
            previous_affine = affine


@dataclass(frozen=True)
class MultiInteriorSurfaceTrace:
    """Certified accepted-step prefix for several independent signed surfaces.

    ``probe_reintegrations`` counts Hamiltonian reconstructions, while
    ``surface_value_evaluations`` counts cheap member-scalar evaluations.
    Shared probe states mean adding members does not repeat the common N/2N
    Hamiltonian grid.  Root refinement may still need additional unique
    affine states when different members have different brackets.
    """

    surface_ids: tuple[str, ...]
    crossings: tuple[ClassifiedMultiInteriorSurfaceCrossing, ...]
    base_subdivisions_per_step: int
    verification_subdivisions_per_step: int
    topology_converged: bool
    maximum_probe_event_difference: float
    maximum_probe_covector_relative_difference: float
    probe_reintegrations: int
    surface_value_evaluations: int
    initial_contact: AuthenticatedInitialMultiSurfaceContact | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.surface_ids, tuple)
            or not self.surface_ids
            or any(
                not isinstance(surface_id, str) or not surface_id
                for surface_id in self.surface_ids
            )
            or len(set(self.surface_ids)) != len(self.surface_ids)
            or self.surface_ids != tuple(sorted(self.surface_ids))
        ):
            raise ValueError(
                "multi-surface trace ids must be unique non-empty canonical strings"
            )
        if (
            type(self.base_subdivisions_per_step) is not int
            or self.base_subdivisions_per_step < 2
            or self.base_subdivisions_per_step % 2
        ):
            raise ValueError("base multi-surface subdivisions must be even N >= 2")
        if self.verification_subdivisions_per_step != (
            2 * self.base_subdivisions_per_step
        ):
            raise ValueError("multi-surface verification subdivisions must equal 2N")
        if type(self.topology_converged) is not bool:
            raise TypeError("multi-surface topology_converged must be a bool")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.maximum_probe_event_difference,
                self.maximum_probe_covector_relative_difference,
            )
        ):
            raise ValueError("multi-surface convergence differences must be finite")
        if (
            type(self.probe_reintegrations) is not int
            or self.probe_reintegrations < 0
            or type(self.surface_value_evaluations) is not int
            or self.surface_value_evaluations < 0
        ):
            raise ValueError("multi-surface work diagnostics must be non-negative")
        if self.initial_contact is not None and type(self.initial_contact) is not (
            AuthenticatedInitialMultiSurfaceContact
        ):
            raise TypeError(
                "multi-surface trace initial contact has an invalid type"
            )
        declared_ids = set(self.surface_ids)
        if (
            self.initial_contact is not None
            and self.initial_contact.surface_id not in declared_ids
        ):
            raise ValueError(
                "multi-surface trace initial contact id was not declared"
            )
        previous_affine = -math.inf
        terminal_seen = False
        for entry in self.crossings:
            if not isinstance(entry, ClassifiedMultiInteriorSurfaceCrossing):
                raise TypeError("multi-surface trace contains an invalid crossing")
            if entry.surface_id not in declared_ids:
                raise ValueError("multi-surface trace crossing id was not declared")
            affine = entry.crossing.ray_affine_length
            if affine <= previous_affine:
                raise ValueError(
                    "multi-surface trace crossings must be strictly ordered"
                )
            if terminal_seen:
                raise ValueError(
                    "multi-surface trace may not continue after a terminal hit"
                )
            terminal_seen = entry.decision.terminates
            previous_affine = affine


@dataclass(frozen=True)
class TerminationCrossing:
    """One bracketed terminal surface crossing."""

    outcome: str
    target_id: str
    value_before: float
    value_after: float

    def __post_init__(self) -> None:
        if not self.outcome or not self.target_id:
            raise ValueError("termination crossing needs an outcome and target id")
        if not math.isfinite(self.value_before) or not math.isfinite(self.value_after):
            raise ValueError("termination bracket values must be finite")
        if self.value_before == 0.0 or self.value_after == 0.0:
            return
        if math.copysign(1.0, self.value_before) == math.copysign(
            1.0,
            self.value_after,
        ):
            raise ValueError("termination crossing must bracket a zero")


@runtime_checkable
class TerminationSurface(Protocol):
    """Time-dependent terminal-worldtube boundary.

    Implementations may evaluate moving NR horizon worldtubes at each state's
    coordinate time.  The analytic radial calibration below is only one
    implementation; the integrator does not treat coordinate spheres as
    physical horizons by default.
    """

    def crossing(
        self,
        previous: HamiltonianState,
        current: HamiltonianState,
    ) -> TerminationCrossing | None:
        """Return a bracket when an accepted segment crosses a surface."""

    def classify_initial(
        self,
        state: HamiltonianState,
    ) -> tuple[str, str] | None:
        """Classify an observer event already on or beyond a terminal surface."""

    def value(
        self,
        state: HamiltonianState,
        crossing: TerminationCrossing,
    ) -> float:
        """Return the signed event function for root localization."""

    def needs_refinement(
        self,
        previous: HamiltonianState,
        current: HamiltonianState,
    ) -> bool:
        """Whether endpoint tests could miss an interior surface hit."""


@dataclass(frozen=True)
class RadialTermination:
    """Capture and escape surfaces centred on one spatial point."""

    capture_radius_m: float
    escape_radius_m: float
    centre_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.capture_radius_m)
            or not math.isfinite(self.escape_radius_m)
            or self.capture_radius_m <= 0.0
            or self.escape_radius_m <= self.capture_radius_m
        ):
            raise ValueError("radial surfaces require 0 < capture < escape")
        if not all(math.isfinite(value) for value in self.centre_m):
            raise ValueError("radial surface centre must be finite")

    def radius(self, state: HamiltonianState) -> float:
        return math.sqrt(
            math.fsum(
                (state.event[index + 1] - self.centre_m[index]) ** 2
                for index in range(3)
            )
        )

    capture_target_id: str = "analytic-capture-sphere"
    escape_target_id: str = "analytic-escape-sphere"

    def classify_initial(
        self,
        state: HamiltonianState,
    ) -> tuple[str, str] | None:
        radius = self.radius(state)
        if radius <= self.capture_radius_m:
            return ("captured", self.capture_target_id)
        if radius >= self.escape_radius_m:
            return ("escaped", self.escape_target_id)
        return None

    def value(
        self,
        state: HamiltonianState,
        crossing: TerminationCrossing,
    ) -> float:
        radius = self.radius(state)
        if crossing.outcome == "captured":
            return radius - self.capture_radius_m
        if crossing.outcome == "escaped":
            return radius - self.escape_radius_m
        raise ValueError(f"unsupported radial outcome {crossing.outcome!r}")

    def crossing(
        self,
        previous: HamiltonianState,
        current: HamiltonianState,
    ) -> TerminationCrossing | None:
        previous_radius = self.radius(previous)
        current_radius = self.radius(current)
        capture_before = previous_radius - self.capture_radius_m
        capture_after = current_radius - self.capture_radius_m
        if capture_before > 0.0 and capture_after <= 0.0:
            return TerminationCrossing(
                "captured",
                self.capture_target_id,
                capture_before,
                capture_after,
            )
        if (
            previous_radius < self.escape_radius_m
            and current_radius >= self.escape_radius_m
        ):
            return TerminationCrossing(
                "escaped",
                self.escape_target_id,
                previous_radius - self.escape_radius_m,
                current_radius - self.escape_radius_m,
            )
        return None

    def needs_refinement(
        self,
        previous: HamiltonianState,
        current: HamiltonianState,
    ) -> bool:
        """Detect a chord that enters and exits the analytic capture sphere."""

        start = tuple(
            previous.event[index + 1] - self.centre_m[index]
            for index in range(3)
        )
        delta = tuple(
            current.event[index + 1] - previous.event[index + 1]
            for index in range(3)
        )
        length_squared = math.fsum(value * value for value in delta)
        if length_squared == 0.0:
            return False
        fraction = -math.fsum(
            start[index] * delta[index] for index in range(3)
        ) / length_squared
        fraction = min(1.0, max(0.0, fraction))
        closest_squared = math.fsum(
            (start[index] + fraction * delta[index]) ** 2
            for index in range(3)
        )
        return (
            self.radius(previous) > self.capture_radius_m
            and self.radius(current) > self.capture_radius_m
            and closest_squared <= self.capture_radius_m**2
        )


@dataclass(frozen=True)
class RayTraceResult:
    outcome: str
    terminal_state: HamiltonianState
    affine_length: float
    accepted_steps: int
    rejected_steps: int
    maximum_null_residual: float
    maximum_metric_interpolation_error: float
    segments: tuple[RayPathSegment, ...]
    failure_reason: str | None = None
    terminal_target_id: str | None = None
    interior_surface_trace: InteriorSurfaceTrace | None = None
    multi_surface_trace: MultiInteriorSurfaceTrace | None = None


@dataclass(frozen=True)
class RayRefinementResult:
    """Independent coarse/fine whole-ray convergence evidence."""

    fine: RayTraceResult
    coarse: RayTraceResult
    outcome_agrees: bool
    terminal_event_difference: float
    terminal_covector_difference: float
    discretizations_differ: bool
    terminal_target_agrees: bool
    converged: bool


@dataclass
class _MetricAudit:
    """Collect provider interpolation evidence from every RHS evaluation."""

    provider: MetricProvider
    maximum_interpolation_error: float = 0.0

    def sample(self, event: Vector4):
        sample = self.provider.sample(event)
        self.maximum_interpolation_error = max(
            self.maximum_interpolation_error,
            sample.interpolation_error,
        )
        return sample


def _normalized_null_residual_from_sample(sample, covector: Vector4) -> float:
    momentum_scale = max(abs(value) for value in covector)
    metric_scale = max(
        abs(value)
        for row in sample.inverse
        for value in row
    )
    if (
        not math.isfinite(momentum_scale)
        or momentum_scale == 0.0
        or not math.isfinite(metric_scale)
        or metric_scale == 0.0
    ):
        return math.inf
    normalized_covector = tuple(value / momentum_scale for value in covector)
    terms = tuple(
        sample.inverse[row][column]
        / metric_scale
        * normalized_covector[row]
        * normalized_covector[column]
        for row in range(4)
        for column in range(4)
    )
    numerator = abs(math.fsum(terms))
    denominator = math.fsum(abs(value) for value in terms)
    if denominator == 0.0:
        return math.inf
    result = numerator / denominator
    return result if math.isfinite(result) else math.inf


def hamiltonian_null_residual(
    provider: MetricProvider,
    state: HamiltonianState,
) -> float:
    """Return a momentum-scale-invariant null-constraint residual."""
    sample = provider.sample(state.event)
    return _normalized_null_residual_from_sample(sample, state.covector)


def _derivative(audit: _MetricAudit, values: Sequence[float]) -> State8:
    state = HamiltonianState.unpack(values)
    sample = audit.sample(state.event)
    coordinate_derivative = matrix_vector(sample.inverse, state.covector)
    momentum_derivative = tuple(
        -0.5
        * bilinear(
            state.covector,
            sample.inverse_derivatives[coordinate],
            state.covector,
        )
        for coordinate in range(4)
    )
    result = (*coordinate_derivative, *momentum_derivative)
    if not all(math.isfinite(value) for value in result):
        raise ArithmeticError("non-finite Hamiltonian derivative")
    return result  # type: ignore[return-value]


def _linear_combination(
    base: Sequence[float],
    step: float,
    terms: Iterable[tuple[float, Sequence[float]]],
) -> State8:
    weighted = tuple(terms)
    return tuple(  # type: ignore[return-value]
        base[index]
        + step
        * math.fsum(
            coefficient * vector[index] for coefficient, vector in weighted
        )
        for index in range(8)
    )


def _dormand_prince_step(
    derivative: Callable[[Sequence[float]], State8],
    state: Sequence[float],
    step: float,
) -> tuple[State8, State8]:
    """One Dormand-Prince 5(4) step and its embedded error."""
    k1 = derivative(state)
    k2 = derivative(_linear_combination(state, step, ((1 / 5, k1),)))
    k3 = derivative(
        _linear_combination(state, step, ((3 / 40, k1), (9 / 40, k2)))
    )
    k4 = derivative(
        _linear_combination(
            state,
            step,
            ((44 / 45, k1), (-56 / 15, k2), (32 / 9, k3)),
        )
    )
    k5 = derivative(
        _linear_combination(
            state,
            step,
            (
                (19372 / 6561, k1),
                (-25360 / 2187, k2),
                (64448 / 6561, k3),
                (-212 / 729, k4),
            ),
        )
    )
    k6 = derivative(
        _linear_combination(
            state,
            step,
            (
                (9017 / 3168, k1),
                (-355 / 33, k2),
                (46732 / 5247, k3),
                (49 / 176, k4),
                (-5103 / 18656, k5),
            ),
        )
    )
    fifth = _linear_combination(
        state,
        step,
        (
            (35 / 384, k1),
            (500 / 1113, k3),
            (125 / 192, k4),
            (-2187 / 6784, k5),
            (11 / 84, k6),
        ),
    )
    k7 = derivative(fifth)
    fourth = _linear_combination(
        state,
        step,
        (
            (5179 / 57600, k1),
            (7571 / 16695, k3),
            (393 / 640, k4),
            (-92097 / 339200, k5),
            (187 / 2100, k6),
            (1 / 40, k7),
        ),
    )
    return fifth, tuple(  # type: ignore[return-value]
        fifth[index] - fourth[index] for index in range(8)
    )


def _error_norm(
    before: Sequence[float],
    after: Sequence[float],
    error: Sequence[float],
    options: RayTraceOptions,
) -> float:
    return _scaled_state_error_norm(
        before,
        after,
        error,
        absolute_tolerance=options.absolute_tolerance,
        relative_tolerance=options.relative_tolerance,
    )


def _scaled_state_error_norm(
    before: Sequence[float],
    after: Sequence[float],
    error: Sequence[float],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    return max(
        abs(error[index])
        / (
            absolute_tolerance
            + relative_tolerance
            * max(abs(before[index]), abs(after[index]))
        )
        for index in range(8)
    )


@dataclass(frozen=True)
class _SurfaceProbe:
    state: HamiltonianState
    value: float
    ray_affine_length: float
    segment_index: int
    segment_affine_length: float


@dataclass(frozen=True)
class _SurfaceInterval:
    lower: _SurfaceProbe
    upper: _SurfaceProbe
    segment_index: int
    segment_start_affine: float


class _RecordedPathReintegrator:
    """Reconstruct states from a segment start with the canonical Hamiltonian."""

    def __init__(
        self,
        provider: MetricProvider,
        options: SurfaceEventOptions | RecordedPathSamplingOptions,
    ) -> None:
        self.options = options
        self.audit = _MetricAudit(provider)
        self.reintegrations = 0
        self.maximum_null_residual = 0.0
        self.derivative = lambda values: _derivative(self.audit, values)

    def _check_metric_error(self) -> None:
        if (
            self.audit.maximum_interpolation_error
            > self.options.metric_interpolation_error_limit
        ):
            raise SurfaceEventError(
                "metric interpolation error exceeded the recorded-path limit"
            )

    def validate_state(self, state: HamiltonianState, label: str) -> None:
        try:
            sample = self.audit.sample(state.event)
            residual = _normalized_null_residual_from_sample(
                sample,
                state.covector,
            )
        except (
            ArithmeticError,
            IndexError,
            TypeError,
            ValueError,
            OverflowError,
        ) as error:
            raise SurfaceEventError(
                f"unable to validate {label}: {error}"
            ) from error
        self._check_metric_error()
        if residual > self.options.null_residual_limit:
            raise SurfaceEventError(
                f"{label} exceeds the recorded-path null-residual limit"
            )
        self.maximum_null_residual = max(
            self.maximum_null_residual,
            residual,
        )

    def state_at(
        self,
        segment: RayPathSegment,
        affine_offset: float,
        *,
        label: str,
        expected: HamiltonianState | None = None,
    ) -> HamiltonianState:
        if (
            not math.isfinite(affine_offset)
            or affine_offset <= 0.0
            or affine_offset > segment.affine_length
        ):
            raise SurfaceEventError(
                f"{label} has an invalid segment affine offset"
            )
        if self.reintegrations >= self.options.maximum_reintegrations:
            raise SurfaceEventError(
                "recorded-path reintegration budget exhausted"
            )
        self.reintegrations += 1
        try:
            values, error = _dormand_prince_step(
                self.derivative,
                segment.start.packed(),
                affine_offset,
            )
            error_norm = _scaled_state_error_norm(
                segment.start.packed(),
                values,
                error,
                absolute_tolerance=self.options.absolute_tolerance,
                relative_tolerance=self.options.relative_tolerance,
            )
            state = HamiltonianState.unpack(values)
        except (
            ArithmeticError,
            IndexError,
            TypeError,
            ValueError,
            OverflowError,
        ) as error_value:
            raise SurfaceEventError(
                f"Hamiltonian reintegration failed for {label}: {error_value}"
            ) from error_value
        self._check_metric_error()
        if not math.isfinite(error_norm) or error_norm > 1.0:
            raise SurfaceEventError(
                f"Hamiltonian reintegration error exceeded tolerance for {label}"
            )
        self.validate_state(state, label)
        if expected is not None:
            difference = tuple(
                state.packed()[index] - expected.packed()[index]
                for index in range(8)
            )
            mismatch = _scaled_state_error_norm(
                expected.packed(),
                state.packed(),
                difference,
                absolute_tolerance=self.options.absolute_tolerance,
                relative_tolerance=self.options.relative_tolerance,
            )
            if not math.isfinite(mismatch) or mismatch > 1.0:
                raise SurfaceEventError(
                    f"recorded {label} does not match Hamiltonian reintegration"
                )
        return state


class CertifiedRecordedPathSampler:
    """Public certified sampler for an already accepted ray path.

    A fraction of zero returns the recorded segment start after independently
    checking its null constraint.  Every positive fraction is reconstructed
    from that start with the canonical Hamiltonian DOPRI5(4) step and may be
    compared with an expected recorded state.  The class intentionally keeps
    a single reintegration counter so downstream adaptive physics cannot reset
    its work budget once per segment or recursion branch.
    """

    def __init__(
        self,
        provider: MetricProvider,
        options: RecordedPathSamplingOptions = RecordedPathSamplingOptions(),
    ) -> None:
        if not isinstance(options, RecordedPathSamplingOptions):
            raise TypeError("options must be a RecordedPathSamplingOptions")
        self._options = options
        self._reintegrator = _RecordedPathReintegrator(provider, options)

    @property
    def options(self) -> RecordedPathSamplingOptions:
        return self._options

    @property
    def diagnostics(self) -> RecordedPathSamplingDiagnostics:
        return RecordedPathSamplingDiagnostics(
            reintegrations=self._reintegrator.reintegrations,
            maximum_null_residual=(
                self._reintegrator.maximum_null_residual
            ),
            maximum_metric_interpolation_error=(
                self._reintegrator.audit.maximum_interpolation_error
            ),
        )

    def sample(
        self,
        segment: RayPathSegment,
        affine_fraction: float,
        *,
        expected: HamiltonianState | None = None,
        label: str = "recorded path sample",
    ) -> HamiltonianState:
        """Return one certified state at ``fraction * segment.length``."""

        if not isinstance(segment, RayPathSegment):
            raise TypeError("segment must be a RayPathSegment")
        if (
            not math.isfinite(segment.affine_length)
            or segment.affine_length <= 0.0
        ):
            raise RecordedPathSamplingError(
                "recorded segment must have positive finite affine length"
            )
        if isinstance(affine_fraction, bool):
            raise TypeError("affine_fraction must be a finite float")
        try:
            fraction = float(affine_fraction)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError("affine_fraction must be a finite float") from error
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("affine_fraction must lie in [0, 1]")
        if not isinstance(label, str) or not label:
            raise ValueError("sample label must be a non-empty string")
        if expected is not None and not isinstance(expected, HamiltonianState):
            raise TypeError("expected must be a HamiltonianState or None")

        try:
            if fraction == 0.0:
                self._reintegrator.validate_state(segment.start, label)
                state = segment.start
                if expected is not None and state != expected:
                    difference = tuple(
                        state.packed()[index] - expected.packed()[index]
                        for index in range(8)
                    )
                    mismatch = _scaled_state_error_norm(
                        expected.packed(),
                        state.packed(),
                        difference,
                        absolute_tolerance=self.options.absolute_tolerance,
                        relative_tolerance=self.options.relative_tolerance,
                    )
                    if not math.isfinite(mismatch) or mismatch > 1.0:
                        raise SurfaceEventError(
                            f"recorded {label} does not match its expected state"
                        )
                return state
            return self._reintegrator.state_at(
                segment,
                fraction * segment.affine_length,
                label=label,
                expected=expected,
            )
        except SurfaceEventError as error:
            raise RecordedPathSamplingError(str(error)) from error

    def certify_segment(
        self,
        segment: RayPathSegment,
        *,
        label: str = "recorded segment",
    ) -> None:
        """Reconstruct and bind a segment's start, midpoint, and endpoint."""

        if not isinstance(segment, RayPathSegment):
            raise TypeError("segment must be a RayPathSegment")
        if (
            not math.isfinite(segment.midpoint_null_residual)
            or segment.midpoint_null_residual < 0.0
        ):
            raise RecordedPathSamplingError(
                "recorded segment midpoint residual is invalid"
            )
        self.sample(segment, 0.0, expected=segment.start, label=f"{label}.start")
        self.sample(
            segment,
            0.5,
            expected=segment.midpoint,
            label=f"{label}.midpoint",
        )
        self.sample(
            segment,
            1.0,
            expected=segment.end,
            label=f"{label}.end",
        )


def _checked_surface_value(
    surface: Callable[[HamiltonianState], float],
    state: HamiltonianState,
) -> float:
    try:
        raw_value = surface(state)
        if isinstance(raw_value, bool):
            raise TypeError("boolean is not a signed scalar")
        value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SurfaceEventError(f"signed surface evaluation failed: {error}") from error
    if not math.isfinite(value):
        raise SurfaceEventError("signed surface returned a non-finite value")
    return value


def _locate_bracketed_recorded_surface_crossing(
    segment: RayPathSegment,
    interval: _SurfaceInterval,
    surface: Callable[[HamiltonianState], float],
    reintegrator: _RecordedPathReintegrator,
) -> RecordedSurfaceCrossing:
    options = reintegrator.options
    lower_offset = interval.lower.ray_affine_length - interval.segment_start_affine
    upper_offset = interval.upper.ray_affine_length - interval.segment_start_affine
    lower_state = interval.lower.state
    upper_state = interval.upper.state
    lower_value = interval.lower.value
    upper_value = interval.upper.value
    if (
        lower_value == 0.0
        or upper_value == 0.0
        or math.copysign(1.0, lower_value) == math.copysign(1.0, upper_value)
    ):
        raise SurfaceEventError("surface root localization requires a strict bracket")
    orientation = 1 if lower_value < 0.0 < upper_value else -1

    for iteration in range(1, options.maximum_iterations + 1):
        middle_offset = 0.5 * (lower_offset + upper_offset)
        if middle_offset <= lower_offset or middle_offset >= upper_offset:
            raise SurfaceEventError("surface root bracket cannot be refined further")
        middle_state = reintegrator.state_at(
            segment,
            middle_offset,
            label=f"surface root iteration {iteration}",
        )
        middle_value = _checked_surface_value(surface, middle_state)
        bracket_width = upper_offset - lower_offset
        if middle_value == 0.0:
            return RecordedSurfaceCrossing(
                state=middle_state,
                ray_affine_length=(
                    interval.segment_start_affine + middle_offset
                ),
                segment_index=interval.segment_index,
                segment_affine_length=middle_offset,
                orientation=orientation,
                surface_value=middle_value,
                bracket_affine_width=bracket_width,
                iterations=iteration,
            )
        if math.copysign(1.0, middle_value) == math.copysign(
            1.0,
            lower_value,
        ):
            lower_offset = middle_offset
            lower_state = middle_state
            lower_value = middle_value
        else:
            upper_offset = middle_offset
            upper_state = middle_state
            upper_value = middle_value

        if upper_offset - lower_offset <= options.affine_tolerance:
            candidates = (
                (lower_state, lower_offset, lower_value),
                (upper_state, upper_offset, upper_value),
            )
            chosen_state, chosen_offset, chosen_value = min(
                candidates,
                key=lambda candidate: abs(candidate[2]),
            )
            if abs(chosen_value) > options.surface_value_tolerance:
                raise SurfaceEventError(
                    "surface root reached affine tolerance without a value root"
                )
            return RecordedSurfaceCrossing(
                state=chosen_state,
                ray_affine_length=(
                    interval.segment_start_affine + chosen_offset
                ),
                segment_index=interval.segment_index,
                segment_affine_length=chosen_offset,
                orientation=orientation,
                surface_value=chosen_value,
                bracket_affine_width=upper_offset - lower_offset,
                iterations=iteration,
            )

    raise SurfaceEventError("surface root-localization iteration budget exhausted")


def locate_recorded_surface_crossings(
    provider: MetricProvider,
    segments: Iterable[RayPathSegment],
    surface: Callable[[HamiltonianState], float],
    *,
    options: SurfaceEventOptions = SurfaceEventOptions(),
    ignore_unbracketed_path_endpoints: bool = False,
) -> tuple[RecordedSurfaceCrossing, ...]:
    """Locate signed scalar-surface crossings on an observer-traced path.

    The input segments remain in observer-to-source order.  Every probe after a
    segment start is reconstructed by the same Hamiltonian RHS and metric
    provider used by :func:`trace_null_geodesic`; states are never linearly
    interpolated.  Only finite sign-changing brackets are accepted.  An exact
    interior endpoint is emitted once, while a tangent, a zero plateau, or a
    zero at an unbracketed path boundary raises :class:`SurfaceEventError`.

    ``ignore_unbracketed_path_endpoints`` is reserved for product layers whose
    physical emitting surface excludes both ray endpoints even though the
    signed auxiliary surface touches them.  For example, an equatorial Kerr
    observer outside a finite disk annulus starts on the auxiliary ``z=0``
    plane but not on the disk.  Interior contacts remain strict.

    A finite subdivision grid cannot prove the absence of arbitrarily many
    oscillatory roots.  Callers must choose ``subdivisions_per_segment`` for the
    smallest physical surface scale they intend to resolve; exhausted accuracy
    or work budgets fail closed.
    """

    if not isinstance(options, SurfaceEventOptions):
        raise TypeError("options must be a SurfaceEventOptions")
    if not callable(surface):
        raise TypeError("surface must be a callable signed scalar field")
    if type(ignore_unbracketed_path_endpoints) is not bool:
        raise TypeError("ignore_unbracketed_path_endpoints must be a bool")
    try:
        recorded = tuple(segments)
    except TypeError as error:
        raise TypeError("segments must be iterable") from error
    if not recorded:
        return ()

    reintegrator = _RecordedPathReintegrator(provider, options)
    probes: list[_SurfaceProbe] = []
    intervals: list[_SurfaceInterval] = []
    cumulative_affine = 0.0
    previous_end: HamiltonianState | None = None

    for segment_index, segment in enumerate(recorded):
        if not isinstance(segment, RayPathSegment):
            raise TypeError(f"segments[{segment_index}] must be a RayPathSegment")
        if (
            not math.isfinite(segment.affine_length)
            or segment.affine_length <= 0.0
        ):
            raise SurfaceEventError(
                f"segments[{segment_index}] must have positive finite length"
            )
        if (
            not math.isfinite(segment.midpoint_null_residual)
            or segment.midpoint_null_residual < 0.0
        ):
            raise SurfaceEventError(
                f"segments[{segment_index}] has an invalid midpoint residual"
            )
        if previous_end is not None and segment.start != previous_end:
            raise SurfaceEventError("recorded path segments are not contiguous")
        reintegrator.validate_state(
            segment.start,
            f"segments[{segment_index}].start",
        )

        if not probes:
            probes.append(
                _SurfaceProbe(
                    state=segment.start,
                    value=_checked_surface_value(surface, segment.start),
                    ray_affine_length=cumulative_affine,
                    segment_index=segment_index,
                    segment_affine_length=0.0,
                )
            )
        start_probe = probes[-1]
        segment_start_affine = cumulative_affine
        subdivisions = options.subdivisions_per_segment
        for subdivision in range(1, subdivisions + 1):
            affine_offset = segment.affine_length * subdivision / subdivisions
            expected: HamiltonianState | None = None
            label = (
                f"segments[{segment_index}] subdivision {subdivision}/"
                f"{subdivisions}"
            )
            if subdivision == subdivisions // 2:
                expected = segment.midpoint
                label = f"segments[{segment_index}].midpoint"
            if subdivision == subdivisions:
                expected = segment.end
                label = f"segments[{segment_index}].end"
            state = reintegrator.state_at(
                segment,
                affine_offset,
                label=label,
                expected=expected,
            )
            ray_affine = segment_start_affine + affine_offset
            if not math.isfinite(ray_affine):
                raise SurfaceEventError("recorded path affine length overflowed")
            probe = _SurfaceProbe(
                state=state,
                value=_checked_surface_value(surface, state),
                ray_affine_length=ray_affine,
                segment_index=segment_index,
                segment_affine_length=affine_offset,
            )
            intervals.append(
                _SurfaceInterval(
                    lower=start_probe,
                    upper=probe,
                    segment_index=segment_index,
                    segment_start_affine=segment_start_affine,
                )
            )
            probes.append(probe)
            start_probe = probe

        cumulative_affine += segment.affine_length
        if not math.isfinite(cumulative_affine):
            raise SurfaceEventError("recorded path affine length overflowed")
        previous_end = segment.end

    ignored_initial_contact = (
        ignore_unbracketed_path_endpoints
        and abs(probes[0].value) <= options.surface_value_tolerance
    )
    ignored_terminal_contact = (
        ignore_unbracketed_path_endpoints
        and abs(probes[-1].value) <= options.surface_value_tolerance
    )
    zero_indices = {
        index for index, probe in enumerate(probes)
        if probe.value == 0.0
        and not (index == 0 and ignored_initial_contact)
        and not (index == len(probes) - 1 and ignored_terminal_contact)
    }
    events: list[RecordedSurfaceCrossing] = []
    cursor = 0
    while cursor < len(probes):
        if cursor not in zero_indices:
            cursor += 1
            continue
        end = cursor + 1
        while end < len(probes) and end in zero_indices:
            end += 1
        if end - cursor > 1:
            raise SurfaceEventError(
                "surface remains within zero tolerance over a finite path interval"
            )
        if cursor == 0 or cursor == len(probes) - 1:
            if ignore_unbracketed_path_endpoints:
                cursor = end
                continue
            raise SurfaceEventError("surface contact at an unbracketed path endpoint")
        left = probes[cursor - 1]
        root = probes[cursor]
        right = probes[cursor + 1]
        if math.copysign(1.0, left.value) == math.copysign(1.0, right.value):
            raise SurfaceEventError("tangent surface contact has no signed bracket")
        orientation = 1 if left.value < 0.0 < right.value else -1
        events.append(
            RecordedSurfaceCrossing(
                state=root.state,
                ray_affine_length=root.ray_affine_length,
                segment_index=root.segment_index,
                segment_affine_length=root.segment_affine_length,
                orientation=orientation,
                surface_value=root.value,
                bracket_affine_width=0.0,
                iterations=0,
            )
        )
        cursor = end

    for interval in intervals:
        if (
            (ignored_initial_contact and interval.lower is probes[0])
            or (ignored_terminal_contact and interval.upper is probes[-1])
        ):
            continue
        if (
            interval.lower.value == 0.0
            or interval.upper.value == 0.0
        ):
            continue
        if math.copysign(1.0, interval.lower.value) == math.copysign(
            1.0,
            interval.upper.value,
        ):
            continue
        events.append(
            _locate_bracketed_recorded_surface_crossing(
                recorded[interval.segment_index],
                interval,
                surface,
                reintegrator,
            )
        )

    ordered = sorted(events, key=lambda event: event.ray_affine_length)
    deduplicated: list[RecordedSurfaceCrossing] = []
    for event in ordered:
        if (
            deduplicated
            and event.ray_affine_length
            - deduplicated[-1].ray_affine_length
            <= options.affine_tolerance
        ):
            previous = deduplicated[-1]
            if previous.orientation != event.orientation:
                raise SurfaceEventError(
                    "opposite surface crossings are unresolved within affine tolerance"
                )
            if abs(event.surface_value) < abs(previous.surface_value):
                deduplicated[-1] = event
            continue
        deduplicated.append(event)
    return tuple(deduplicated)


class _InteriorSurfaceNeedsRefinement(RuntimeError):
    """Internal signal that the accepted geodesic step must be retried smaller."""


@dataclass(frozen=True)
class _StepSurfaceProbe:
    state: HamiltonianState
    value: float
    affine_offset: float


class _StepSurfaceReintegrator:
    """Reconstruct accepted-step probes with the trace's Hamiltonian and audit."""

    def __init__(
        self,
        audit: _MetricAudit,
        options: SurfaceEventOptions,
    ) -> None:
        self.audit = audit
        self.options = options
        self.reintegrations = 0
        self.maximum_null_residual = 0.0
        self.derivative = lambda values: _derivative(self.audit, values)

    def state_at(
        self,
        start: HamiltonianState,
        affine_offset: float,
        *,
        label: str,
        expected: HamiltonianState | None = None,
    ) -> HamiltonianState:
        if not math.isfinite(affine_offset) or affine_offset <= 0.0:
            raise SurfaceEventError(f"{label} has an invalid affine offset")
        if self.reintegrations >= self.options.maximum_reintegrations:
            raise SurfaceEventError("accepted-step surface work budget exhausted")
        self.reintegrations += 1
        try:
            values, error = _dormand_prince_step(
                self.derivative,
                start.packed(),
                affine_offset,
            )
            error_norm = _scaled_state_error_norm(
                start.packed(),
                values,
                error,
                absolute_tolerance=self.options.absolute_tolerance,
                relative_tolerance=self.options.relative_tolerance,
            )
            state = HamiltonianState.unpack(values)
            sample = self.audit.sample(state.event)
            residual = _normalized_null_residual_from_sample(
                sample,
                state.covector,
            )
        except (
            ArithmeticError,
            IndexError,
            TypeError,
            ValueError,
            OverflowError,
        ) as error_value:
            raise SurfaceEventError(
                f"Hamiltonian surface probe failed for {label}: {error_value}"
            ) from error_value
        if (
            self.audit.maximum_interpolation_error
            > self.options.metric_interpolation_error_limit
        ):
            raise SurfaceEventError(
                "metric interpolation error exceeded the surface-event limit"
            )
        if not math.isfinite(error_norm):
            raise SurfaceEventError("surface probe integration error is non-finite")
        if error_norm > 1.0:
            raise _InteriorSurfaceNeedsRefinement(
                f"surface probe {label} exceeded its integration tolerance"
            )
        if residual > self.options.null_residual_limit:
            raise SurfaceEventError(
                f"surface probe {label} exceeded its null-residual limit"
            )
        self.maximum_null_residual = max(
            self.maximum_null_residual,
            residual,
        )
        if expected is not None:
            difference = tuple(
                state.packed()[index] - expected.packed()[index]
                for index in range(8)
            )
            mismatch = _scaled_state_error_norm(
                expected.packed(),
                state.packed(),
                difference,
                absolute_tolerance=self.options.absolute_tolerance,
                relative_tolerance=self.options.relative_tolerance,
            )
            if not math.isfinite(mismatch):
                raise SurfaceEventError("surface probe consistency is non-finite")
            if mismatch > 1.0:
                raise _InteriorSurfaceNeedsRefinement(
                    f"surface probe {label} disagrees with the accepted step"
                )
        return state


def _checked_interior_surface_value(
    surface: InteriorSurface,
    state: HamiltonianState,
) -> float:
    return _checked_surface_value(surface.value, state)


def _locate_step_surface_root(
    surface: InteriorSurface,
    reintegrator: _StepSurfaceReintegrator,
    start: HamiltonianState,
    lower: _StepSurfaceProbe,
    upper: _StepSurfaceProbe,
    *,
    ray_affine_start: float,
    segment_index: int,
) -> RecordedSurfaceCrossing:
    options = reintegrator.options
    lower_offset = lower.affine_offset
    upper_offset = upper.affine_offset
    lower_state = lower.state
    upper_state = upper.state
    lower_value = lower.value
    upper_value = upper.value
    if (
        lower_value == 0.0
        or upper_value == 0.0
        or math.copysign(1.0, lower_value) == math.copysign(1.0, upper_value)
    ):
        raise SurfaceEventError("accepted-step root needs a strict signed bracket")
    orientation = 1 if lower_value < 0.0 < upper_value else -1

    for iteration in range(1, options.maximum_iterations + 1):
        middle_offset = 0.5 * (lower_offset + upper_offset)
        if middle_offset <= lower_offset or middle_offset >= upper_offset:
            raise SurfaceEventError("accepted-step root bracket cannot be refined")
        middle_state = reintegrator.state_at(
            start,
            middle_offset,
            label=f"surface root iteration {iteration}",
        )
        middle_value = _checked_interior_surface_value(surface, middle_state)
        bracket_width = upper_offset - lower_offset
        if middle_value == 0.0:
            return RecordedSurfaceCrossing(
                state=middle_state,
                ray_affine_length=ray_affine_start + middle_offset,
                segment_index=segment_index,
                segment_affine_length=middle_offset,
                orientation=orientation,
                surface_value=0.0,
                bracket_affine_width=bracket_width,
                iterations=iteration,
            )
        if math.copysign(1.0, middle_value) == math.copysign(
            1.0,
            lower_value,
        ):
            lower_offset = middle_offset
            lower_state = middle_state
            lower_value = middle_value
        else:
            upper_offset = middle_offset
            upper_state = middle_state
            upper_value = middle_value

        if upper_offset - lower_offset <= options.affine_tolerance:
            chosen_state, chosen_offset, chosen_value = min(
                (
                    (lower_state, lower_offset, lower_value),
                    (upper_state, upper_offset, upper_value),
                ),
                key=lambda entry: abs(entry[2]),
            )
            if abs(chosen_value) > options.surface_value_tolerance:
                raise SurfaceEventError(
                    "accepted-step root reached affine tolerance without a value root"
                )
            return RecordedSurfaceCrossing(
                state=chosen_state,
                ray_affine_length=ray_affine_start + chosen_offset,
                segment_index=segment_index,
                segment_affine_length=chosen_offset,
                orientation=orientation,
                surface_value=chosen_value,
                bracket_affine_width=upper_offset - lower_offset,
                iterations=iteration,
            )
    raise SurfaceEventError("accepted-step root-localization budget exhausted")


def _classify_interior_crossing(
    surface: InteriorSurface,
    crossing: RecordedSurfaceCrossing,
) -> ClassifiedInteriorSurfaceCrossing:
    try:
        decision = surface.classify(crossing)
    except (
        ArithmeticError,
        IndexError,
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise SurfaceEventError(
            f"interior surface classification failed: {error}"
        ) from error
    if not isinstance(decision, InteriorSurfaceDecision):
        raise SurfaceEventError(
            "interior surface classifier returned an invalid decision"
        )
    return ClassifiedInteriorSurfaceCrossing(crossing, decision)


def _terminal_endpoint_contact_is_transparent(
    surface: InteriorSurface,
    probe: _StepSurfaceProbe,
    *,
    ray_affine_start: float,
    segment_index: int,
) -> bool:
    """Classify an exact worldtube-end contact without inventing orientation.

    Both possible signed orientations must produce the same transparent
    product decision.  An opaque, orientation-dependent, or malformed contact
    remains unbracketed and therefore fails closed.
    """

    decisions: list[InteriorSurfaceDecision] = []
    for orientation in (-1, 1):
        crossing = RecordedSurfaceCrossing(
            state=probe.state,
            ray_affine_length=ray_affine_start + probe.affine_offset,
            segment_index=segment_index,
            segment_affine_length=probe.affine_offset,
            orientation=orientation,
            surface_value=0.0,
            bracket_affine_width=0.0,
            iterations=0,
        )
        decisions.append(
            _classify_interior_crossing(surface, crossing).decision
        )
    return (
        decisions[0] == decisions[1]
        and not decisions[0].terminates
    )


def _scan_accepted_step_surface(
    surface: InteriorSurface,
    reintegrator: _StepSurfaceReintegrator,
    start: HamiltonianState,
    end: HamiltonianState,
    full_step: float,
    *,
    subdivisions: int,
    ray_affine_start: float,
    segment_index: int,
    allow_transparent_terminal_contact: bool,
) -> tuple[ClassifiedInteriorSurfaceCrossing, ...]:
    """Locate one declared-resolution crossing prefix inside an accepted step."""

    probes = [
        _StepSurfaceProbe(
            state=start,
            value=_checked_interior_surface_value(surface, start),
            affine_offset=0.0,
        )
    ]
    for subdivision in range(1, subdivisions + 1):
        offset = full_step * subdivision / subdivisions
        expected = end if subdivision == subdivisions else None
        state = reintegrator.state_at(
            start,
            offset,
            label=f"accepted step probe {subdivision}/{subdivisions}",
            expected=expected,
        )
        probes.append(
            _StepSurfaceProbe(
                state=state,
                value=_checked_interior_surface_value(surface, state),
                affine_offset=offset,
            )
        )

    all_zero_indices = tuple(
        index for index, probe in enumerate(probes) if probe.value == 0.0
    )
    ignored_zero_indices: set[int] = set()
    if (
        allow_transparent_terminal_contact
        and subdivisions in all_zero_indices
        and _terminal_endpoint_contact_is_transparent(
            surface,
            probes[-1],
            ray_affine_start=ray_affine_start,
            segment_index=segment_index,
        )
    ):
        ignored_zero_indices.add(subdivisions)
    zero_indices = tuple(
        index
        for index in all_zero_indices
        if index not in ignored_zero_indices
    )
    if zero_indices and (
        zero_indices[0] == 0 or zero_indices[-1] == subdivisions
    ):
        raise SurfaceEventError(
            "surface contact at an unbracketed accepted-step endpoint"
        )
    for left, right in zip(zero_indices, zero_indices[1:]):
        if right == left + 1:
            raise SurfaceEventError(
                "surface remains exactly zero over a finite accepted-step interval"
            )

    roots: list[RecordedSurfaceCrossing] = []
    zero_set = set(zero_indices)
    for index in zero_indices:
        left = probes[index - 1]
        root = probes[index]
        right = probes[index + 1]
        if math.copysign(1.0, left.value) == math.copysign(1.0, right.value):
            raise SurfaceEventError("tangent accepted-step surface contact")
        roots.append(
            RecordedSurfaceCrossing(
                state=root.state,
                ray_affine_length=ray_affine_start + root.affine_offset,
                segment_index=segment_index,
                segment_affine_length=root.affine_offset,
                orientation=1 if left.value < 0.0 < right.value else -1,
                surface_value=0.0,
                bracket_affine_width=0.0,
                iterations=0,
            )
        )

    for index, (lower, upper) in enumerate(zip(probes, probes[1:])):
        if (
            index in zero_set
            or index + 1 in zero_set
            or index in ignored_zero_indices
            or index + 1 in ignored_zero_indices
        ):
            continue
        if math.copysign(1.0, lower.value) == math.copysign(
            1.0,
            upper.value,
        ):
            continue
        roots.append(
            _locate_step_surface_root(
                surface,
                reintegrator,
                start,
                lower,
                upper,
                ray_affine_start=ray_affine_start,
                segment_index=segment_index,
            )
        )

    classified: list[ClassifiedInteriorSurfaceCrossing] = []
    for crossing in sorted(roots, key=lambda item: item.ray_affine_length):
        entry = _classify_interior_crossing(surface, crossing)
        classified.append(entry)
        if entry.decision.terminates:
            break
    return tuple(classified)


def _surface_prefix_key(
    entries: tuple[ClassifiedInteriorSurfaceCrossing, ...],
) -> tuple[tuple[int, str, str | None, str | None], ...]:
    return tuple(
        (
            entry.crossing.orientation,
            entry.decision.classification,
            entry.decision.outcome,
            entry.decision.target_id,
        )
        for entry in entries
    )


def _probe_grid_difference(
    coarse: ClassifiedInteriorSurfaceCrossing,
    fine: ClassifiedInteriorSurfaceCrossing,
) -> tuple[float, float]:
    event_difference = math.sqrt(
        math.fsum(
            (
                coarse.crossing.state.event[index]
                - fine.crossing.state.event[index]
            )
            ** 2
            for index in range(4)
        )
    )
    covector_scale = max(
        math.sqrt(
            math.fsum(value * value for value in coarse.crossing.state.covector)
        ),
        math.sqrt(
            math.fsum(value * value for value in fine.crossing.state.covector)
        ),
        1.0e-300,
    )
    covector_difference = math.sqrt(
        math.fsum(
            (
                coarse.crossing.state.covector[index]
                - fine.crossing.state.covector[index]
            )
            ** 2
            for index in range(4)
        )
    ) / covector_scale
    if not math.isfinite(event_difference) or not math.isfinite(covector_difference):
        raise SurfaceEventError("surface probe convergence difference is non-finite")
    return event_difference, covector_difference


def _locate_converged_step_surface_prefix(
    surface: InteriorSurface,
    reintegrator: _StepSurfaceReintegrator,
    start: HamiltonianState,
    end: HamiltonianState,
    full_step: float,
    *,
    ray_affine_start: float,
    segment_index: int,
    allow_transparent_terminal_contact: bool,
) -> tuple[
    tuple[ClassifiedInteriorSurfaceCrossing, ...],
    float,
    float,
]:
    base_subdivisions = reintegrator.options.subdivisions_per_segment
    base = _scan_accepted_step_surface(
        surface,
        reintegrator,
        start,
        end,
        full_step,
        subdivisions=base_subdivisions,
        ray_affine_start=ray_affine_start,
        segment_index=segment_index,
        allow_transparent_terminal_contact=(
            allow_transparent_terminal_contact
        ),
    )
    verified = _scan_accepted_step_surface(
        surface,
        reintegrator,
        start,
        end,
        full_step,
        subdivisions=2 * base_subdivisions,
        ray_affine_start=ray_affine_start,
        segment_index=segment_index,
        allow_transparent_terminal_contact=(
            allow_transparent_terminal_contact
        ),
    )
    if _surface_prefix_key(base) != _surface_prefix_key(verified):
        raise _InteriorSurfaceNeedsRefinement(
            "N/2N accepted-step surface topologies disagree"
        )
    maximum_event_difference = 0.0
    maximum_covector_difference = 0.0
    for base_entry, verified_entry in zip(base, verified):
        affine_difference = abs(
            base_entry.crossing.ray_affine_length
            - verified_entry.crossing.ray_affine_length
        )
        if affine_difference > 2.0 * reintegrator.options.affine_tolerance:
            raise _InteriorSurfaceNeedsRefinement(
                "N/2N accepted-step surface roots have not converged"
            )
        event_difference, covector_difference = _probe_grid_difference(
            base_entry,
            verified_entry,
        )
        maximum_event_difference = max(
            maximum_event_difference,
            event_difference,
        )
        maximum_covector_difference = max(
            maximum_covector_difference,
            covector_difference,
        )
    return verified, maximum_event_difference, maximum_covector_difference


def _canonical_multi_surface_ids(
    surface: MultiInteriorSurface,
) -> tuple[str, ...]:
    try:
        raw_ids = surface.surface_ids
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("multi-surface ids could not be read") from error
    if not isinstance(raw_ids, tuple) or not raw_ids:
        raise ValueError("multi-surface ids must be a non-empty tuple")
    if any(
        not isinstance(surface_id, str) or not surface_id
        for surface_id in raw_ids
    ):
        raise ValueError("multi-surface ids must be non-empty strings")
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("multi-surface ids must be unique")
    return tuple(sorted(raw_ids))


class _ValidatedMultiInteriorSurface:
    """Freeze stable ids and count member-scalar work for one ray."""

    def __init__(
        self,
        surface: MultiInteriorSurface,
        initial_state: HamiltonianState,
        initial_contact: InitialMultiSurfaceContact | None,
        surface_value_tolerance: float,
    ) -> None:
        self.surface = surface
        self.surface_ids = _canonical_multi_surface_ids(surface)
        self.surface_value_evaluations = 0
        self.authenticated_initial_contact = self._authenticate_initial_contact(
            initial_state,
            initial_contact,
            surface_value_tolerance,
        )

    def _one_value(self, surface_id: str, state: HamiltonianState) -> float:
        try:
            raw_value = self.surface.value(surface_id, state)
        except (
            ArithmeticError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
        ) as error:
            raise SurfaceEventError(
                f"multi-surface '{surface_id}' evaluation failed: {error}"
            ) from error
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(float(raw_value))
        ):
            raise SurfaceEventError(
                f"multi-surface '{surface_id}' returned a non-finite value"
            )
        self.surface_value_evaluations += 1
        return float(raw_value)

    def _authenticate_initial_contact(
        self,
        initial_state: HamiltonianState,
        contact: InitialMultiSurfaceContact | None,
        surface_value_tolerance: float,
    ) -> AuthenticatedInitialMultiSurfaceContact | None:
        if contact is None:
            return None
        if type(contact) is not InitialMultiSurfaceContact:
            raise TypeError(
                "initial_multi_surface_contact must be the exact declaration type"
            )
        if contact.surface_id not in self.surface_ids:
            raise ValueError(
                "initial multi-surface contact id is not declared by the surface"
            )
        actual = self._one_value(contact.surface_id, initial_state)
        if abs(actual) > surface_value_tolerance:
            raise ValueError(
                "initial multi-surface contact is not within the declared "
                "surface-value tolerance"
            )
        return AuthenticatedInitialMultiSurfaceContact(
            surface_id=contact.surface_id,
            side=contact.side,
            actual_surface_value=actual,
            surface_value_tolerance=float(surface_value_tolerance),
        )

    def require_stable_ids(self) -> None:
        if _canonical_multi_surface_ids(self.surface) != self.surface_ids:
            raise SurfaceEventError("multi-surface ids changed during the ray trace")

    def values(
        self,
        state: HamiltonianState,
        *,
        apply_initial_contact: bool = False,
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for surface_id in self.surface_ids:
            values[surface_id] = self._one_value(surface_id, state)
        contact = self.authenticated_initial_contact
        if apply_initial_contact:
            if contact is None:
                raise SurfaceEventError(
                    "an undeclared initial multi-surface side was requested"
                )
            values[contact.surface_id] = float(contact.side)
        return values

    def classify(
        self,
        surface_id: str,
        crossing: RecordedSurfaceCrossing,
    ) -> ClassifiedMultiInteriorSurfaceCrossing:
        if surface_id not in self.surface_ids:
            raise SurfaceEventError("multi-surface crossing id was not declared")
        try:
            decision = self.surface.classify(surface_id, crossing)
        except (
            ArithmeticError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
        ) as error:
            raise SurfaceEventError(
                f"multi-surface '{surface_id}' classification failed: {error}"
            ) from error
        if not isinstance(decision, InteriorSurfaceDecision):
            raise SurfaceEventError(
                f"multi-surface '{surface_id}' classifier returned an invalid decision"
            )
        return ClassifiedMultiInteriorSurfaceCrossing(
            surface_id=surface_id,
            crossing=crossing,
            decision=decision,
        )


@dataclass(frozen=True)
class _MultiStepSurfaceProbe:
    state: HamiltonianState
    values: dict[str, float]
    affine_offset: float


class _SharedMultiSurfaceStepProbes:
    """Cache Hamiltonian states and all member values by affine offset."""

    def __init__(
        self,
        evaluator: _ValidatedMultiInteriorSurface,
        reintegrator: _StepSurfaceReintegrator,
        start: HamiltonianState,
        *,
        apply_initial_contact: bool,
    ) -> None:
        self.evaluator = evaluator
        self.reintegrator = reintegrator
        self.start = start
        self._probes = {
            0.0: _MultiStepSurfaceProbe(
                state=start,
                values=evaluator.values(
                    start,
                    apply_initial_contact=apply_initial_contact,
                ),
                affine_offset=0.0,
            )
        }

    def _check_cached_expected(
        self,
        state: HamiltonianState,
        expected: HamiltonianState,
        label: str,
    ) -> None:
        difference = tuple(
            state.packed()[index] - expected.packed()[index]
            for index in range(8)
        )
        mismatch = _scaled_state_error_norm(
            expected.packed(),
            state.packed(),
            difference,
            absolute_tolerance=self.reintegrator.options.absolute_tolerance,
            relative_tolerance=self.reintegrator.options.relative_tolerance,
        )
        if not math.isfinite(mismatch):
            raise SurfaceEventError(
                f"cached multi-surface probe {label} consistency is non-finite"
            )
        if mismatch > 1.0:
            raise _InteriorSurfaceNeedsRefinement(
                f"cached multi-surface probe {label} disagrees with accepted step"
            )

    def at(
        self,
        affine_offset: float,
        *,
        label: str,
        expected: HamiltonianState | None = None,
    ) -> _MultiStepSurfaceProbe:
        cached = self._probes.get(affine_offset)
        if cached is not None:
            if expected is not None:
                self._check_cached_expected(cached.state, expected, label)
            return cached
        state = self.reintegrator.state_at(
            self.start,
            affine_offset,
            label=label,
            expected=expected,
        )
        probe = _MultiStepSurfaceProbe(
            state=state,
            values=self.evaluator.values(state),
            affine_offset=affine_offset,
        )
        self._probes[affine_offset] = probe
        return probe


def _multi_probe_as_single(
    probe: _MultiStepSurfaceProbe,
    surface_id: str,
) -> _StepSurfaceProbe:
    return _StepSurfaceProbe(
        state=probe.state,
        value=probe.values[surface_id],
        affine_offset=probe.affine_offset,
    )


def _locate_multi_step_surface_root(
    surface_id: str,
    probes: _SharedMultiSurfaceStepProbes,
    lower: _MultiStepSurfaceProbe,
    upper: _MultiStepSurfaceProbe,
    *,
    ray_affine_start: float,
    segment_index: int,
) -> RecordedSurfaceCrossing:
    options = probes.reintegrator.options
    lower_offset = lower.affine_offset
    upper_offset = upper.affine_offset
    lower_state = lower.state
    upper_state = upper.state
    lower_value = lower.values[surface_id]
    upper_value = upper.values[surface_id]
    if (
        lower_value == 0.0
        or upper_value == 0.0
        or math.copysign(1.0, lower_value)
        == math.copysign(1.0, upper_value)
    ):
        raise SurfaceEventError(
            f"multi-surface '{surface_id}' root needs a strict signed bracket"
        )
    orientation = 1 if lower_value < 0.0 < upper_value else -1

    for iteration in range(1, options.maximum_iterations + 1):
        middle_offset = 0.5 * (lower_offset + upper_offset)
        if middle_offset <= lower_offset or middle_offset >= upper_offset:
            raise SurfaceEventError(
                f"multi-surface '{surface_id}' root bracket cannot be refined"
            )
        middle = probes.at(
            middle_offset,
            label=f"multi-surface '{surface_id}' root iteration {iteration}",
        )
        middle_value = middle.values[surface_id]
        bracket_width = upper_offset - lower_offset
        if middle_value == 0.0:
            return RecordedSurfaceCrossing(
                state=middle.state,
                ray_affine_length=ray_affine_start + middle_offset,
                segment_index=segment_index,
                segment_affine_length=middle_offset,
                orientation=orientation,
                surface_value=0.0,
                bracket_affine_width=bracket_width,
                iterations=iteration,
            )
        if math.copysign(1.0, middle_value) == math.copysign(
            1.0,
            lower_value,
        ):
            lower_offset = middle_offset
            lower_state = middle.state
            lower_value = middle_value
        else:
            upper_offset = middle_offset
            upper_state = middle.state
            upper_value = middle_value
        if upper_offset - lower_offset <= options.affine_tolerance:
            chosen_state, chosen_offset, chosen_value = min(
                (
                    (lower_state, lower_offset, lower_value),
                    (upper_state, upper_offset, upper_value),
                ),
                key=lambda entry: abs(entry[2]),
            )
            if abs(chosen_value) > options.surface_value_tolerance:
                raise SurfaceEventError(
                    f"multi-surface '{surface_id}' reached affine tolerance "
                    "without a value root"
                )
            return RecordedSurfaceCrossing(
                state=chosen_state,
                ray_affine_length=ray_affine_start + chosen_offset,
                segment_index=segment_index,
                segment_affine_length=chosen_offset,
                orientation=orientation,
                surface_value=chosen_value,
                bracket_affine_width=upper_offset - lower_offset,
                iterations=iteration,
            )
    raise SurfaceEventError(
        f"multi-surface '{surface_id}' root-localization budget exhausted"
    )


def _multi_terminal_endpoint_contact_is_transparent(
    evaluator: _ValidatedMultiInteriorSurface,
    surface_id: str,
    probe: _MultiStepSurfaceProbe,
    *,
    ray_affine_start: float,
    segment_index: int,
) -> bool:
    decisions: list[InteriorSurfaceDecision] = []
    for orientation in (-1, 1):
        crossing = RecordedSurfaceCrossing(
            state=probe.state,
            ray_affine_length=ray_affine_start + probe.affine_offset,
            segment_index=segment_index,
            segment_affine_length=probe.affine_offset,
            orientation=orientation,
            surface_value=0.0,
            bracket_affine_width=0.0,
            iterations=0,
        )
        decisions.append(
            evaluator.classify(surface_id, crossing).decision
        )
    return decisions[0] == decisions[1] and not decisions[0].terminates


def _multi_surface_root_pair_order_is_resolved(
    left_id: str,
    left: RecordedSurfaceCrossing,
    right_id: str,
    right: RecordedSurfaceCrossing,
    options: SurfaceEventOptions,
) -> bool:
    if left_id == right_id:
        return True
    separation = right.ray_affine_length - left.ray_affine_length
    uncertainty = max(
        2.0 * options.affine_tolerance,
        left.bracket_affine_width + right.bracket_affine_width,
    )
    return separation > uncertainty


@dataclass(frozen=True)
class _MultiSurfaceRootCandidate:
    """One exact grid root or one strict member-specific sign bracket."""

    surface_id: str
    lower: _MultiStepSurfaceProbe
    upper: _MultiStepSurfaceProbe
    exact_crossing: RecordedSurfaceCrossing | None = None

    @property
    def lower_affine_offset(self) -> float:
        return self.lower.affine_offset

    @property
    def upper_affine_offset(self) -> float:
        return self.upper.affine_offset


def _multi_surface_candidate_groups(
    candidates: list[_MultiSurfaceRootCandidate],
    options: SurfaceEventOptions,
) -> tuple[tuple[_MultiSurfaceRootCandidate, ...], ...]:
    """Group only brackets whose affine ordering is not already separated."""

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.lower_affine_offset,
            candidate.upper_affine_offset,
        ),
    )
    groups: list[list[_MultiSurfaceRootCandidate]] = []
    group_upper = -math.inf
    for candidate in ordered:
        if (
            not groups
            or candidate.lower_affine_offset
            > group_upper + 2.0 * options.affine_tolerance
        ):
            groups.append([candidate])
            group_upper = candidate.upper_affine_offset
        else:
            groups[-1].append(candidate)
            group_upper = max(group_upper, candidate.upper_affine_offset)
    return tuple(tuple(group) for group in groups)


def _scan_multi_surface_probe_grid(
    evaluator: _ValidatedMultiInteriorSurface,
    shared_probes: _SharedMultiSurfaceStepProbes,
    grid: tuple[_MultiStepSurfaceProbe, ...],
    *,
    ray_affine_start: float,
    segment_index: int,
    allow_transparent_terminal_contact: bool,
) -> tuple[ClassifiedMultiInteriorSurfaceCrossing, ...]:
    subdivisions = len(grid) - 1
    candidates: list[_MultiSurfaceRootCandidate] = []
    for surface_id in evaluator.surface_ids:
        all_zero_indices = tuple(
            index
            for index, probe in enumerate(grid)
            if probe.values[surface_id] == 0.0
        )
        ignored_zero_indices: set[int] = set()
        if (
            allow_transparent_terminal_contact
            and subdivisions in all_zero_indices
            and _multi_terminal_endpoint_contact_is_transparent(
                evaluator,
                surface_id,
                grid[-1],
                ray_affine_start=ray_affine_start,
                segment_index=segment_index,
            )
        ):
            ignored_zero_indices.add(subdivisions)
        zero_indices = tuple(
            index
            for index in all_zero_indices
            if index not in ignored_zero_indices
        )
        if zero_indices and (
            zero_indices[0] == 0 or zero_indices[-1] == subdivisions
        ):
            raise SurfaceEventError(
                f"multi-surface '{surface_id}' contact at an unbracketed "
                "accepted-step endpoint"
            )
        for left, right in zip(zero_indices, zero_indices[1:]):
            if right == left + 1:
                raise SurfaceEventError(
                    f"multi-surface '{surface_id}' remains zero over a finite "
                    "accepted-step interval"
                )

        zero_set = set(zero_indices)
        for index in zero_indices:
            left = grid[index - 1]
            root = grid[index]
            right = grid[index + 1]
            left_value = left.values[surface_id]
            right_value = right.values[surface_id]
            if math.copysign(1.0, left_value) == math.copysign(
                1.0,
                right_value,
            ):
                raise SurfaceEventError(
                    f"tangent multi-surface '{surface_id}' contact"
                )
            exact_crossing = RecordedSurfaceCrossing(
                state=root.state,
                ray_affine_length=(
                    ray_affine_start + root.affine_offset
                ),
                segment_index=segment_index,
                segment_affine_length=root.affine_offset,
                orientation=(
                    1 if left_value < 0.0 < right_value else -1
                ),
                surface_value=0.0,
                bracket_affine_width=0.0,
                iterations=0,
            )
            candidates.append(
                _MultiSurfaceRootCandidate(
                    surface_id=surface_id,
                    lower=root,
                    upper=root,
                    exact_crossing=exact_crossing,
                )
            )

        for index, (lower, upper) in enumerate(zip(grid, grid[1:])):
            if (
                index in zero_set
                or index + 1 in zero_set
                or index in ignored_zero_indices
                or index + 1 in ignored_zero_indices
            ):
                continue
            lower_value = lower.values[surface_id]
            upper_value = upper.values[surface_id]
            if math.copysign(1.0, lower_value) == math.copysign(
                1.0,
                upper_value,
            ):
                continue
            candidates.append(
                _MultiSurfaceRootCandidate(
                    surface_id=surface_id,
                    lower=lower,
                    upper=upper,
                )
            )

    classified: list[ClassifiedMultiInteriorSurfaceCrossing] = []
    for group in _multi_surface_candidate_groups(
        candidates,
        shared_probes.reintegrator.options,
    ):
        roots: list[tuple[str, RecordedSurfaceCrossing]] = []
        for candidate in group:
            crossing = candidate.exact_crossing
            if crossing is None:
                crossing = _locate_multi_step_surface_root(
                    candidate.surface_id,
                    shared_probes,
                    candidate.lower,
                    candidate.upper,
                    ray_affine_start=ray_affine_start,
                    segment_index=segment_index,
                )
            roots.append((candidate.surface_id, crossing))
        ordered_roots = sorted(
            roots,
            key=lambda entry: entry[1].ray_affine_length,
        )
        for index, (surface_id, crossing) in enumerate(ordered_roots):
            if index + 1 < len(ordered_roots):
                next_id, next_crossing = ordered_roots[index + 1]
                if not _multi_surface_root_pair_order_is_resolved(
                    surface_id,
                    crossing,
                    next_id,
                    next_crossing,
                    shared_probes.reintegrator.options,
                ):
                    raise SurfaceEventError(
                        "simultaneous or unresolved-order multi-surface "
                        f"intersection between '{surface_id}' and '{next_id}'"
                    )
            entry = evaluator.classify(surface_id, crossing)
            classified.append(entry)
            if entry.decision.terminates:
                return tuple(classified)
    return tuple(classified)


def _multi_surface_prefix_key(
    entries: tuple[ClassifiedMultiInteriorSurfaceCrossing, ...],
) -> tuple[tuple[str, int, str, str | None, str | None], ...]:
    return tuple(
        (
            entry.surface_id,
            entry.crossing.orientation,
            entry.decision.classification,
            entry.decision.outcome,
            entry.decision.target_id,
        )
        for entry in entries
    )


def _multi_probe_grid_difference(
    coarse: ClassifiedMultiInteriorSurfaceCrossing,
    fine: ClassifiedMultiInteriorSurfaceCrossing,
) -> tuple[float, float]:
    if coarse.surface_id != fine.surface_id:
        raise SurfaceEventError("multi-surface convergence compared different ids")
    event_difference = math.sqrt(
        math.fsum(
            (
                coarse.crossing.state.event[index]
                - fine.crossing.state.event[index]
            )
            ** 2
            for index in range(4)
        )
    )
    covector_scale = max(
        math.sqrt(
            math.fsum(value * value for value in coarse.crossing.state.covector)
        ),
        math.sqrt(
            math.fsum(value * value for value in fine.crossing.state.covector)
        ),
        1.0e-300,
    )
    covector_difference = math.sqrt(
        math.fsum(
            (
                coarse.crossing.state.covector[index]
                - fine.crossing.state.covector[index]
            )
            ** 2
            for index in range(4)
        )
    ) / covector_scale
    if not math.isfinite(event_difference) or not math.isfinite(
        covector_difference
    ):
        raise SurfaceEventError(
            "multi-surface probe convergence difference is non-finite"
        )
    return event_difference, covector_difference


def _locate_converged_step_multi_surface_prefix(
    evaluator: _ValidatedMultiInteriorSurface,
    reintegrator: _StepSurfaceReintegrator,
    start: HamiltonianState,
    end: HamiltonianState,
    full_step: float,
    *,
    ray_affine_start: float,
    segment_index: int,
    allow_transparent_terminal_contact: bool,
) -> tuple[
    tuple[ClassifiedMultiInteriorSurfaceCrossing, ...],
    float,
    float,
]:
    evaluator.require_stable_ids()
    shared = _SharedMultiSurfaceStepProbes(
        evaluator,
        reintegrator,
        start,
        apply_initial_contact=(
            ray_affine_start == 0.0
            and segment_index == 0
            and evaluator.authenticated_initial_contact is not None
        ),
    )
    base_subdivisions = reintegrator.options.subdivisions_per_segment
    verified_subdivisions = 2 * base_subdivisions
    verified_grid = tuple(
        shared.at(
            full_step * subdivision / verified_subdivisions,
            label=(
                "accepted multi-surface step probe "
                f"{subdivision}/{verified_subdivisions}"
            ),
            expected=end if subdivision == verified_subdivisions else None,
        )
        for subdivision in range(verified_subdivisions + 1)
    )
    base_grid = verified_grid[::2]
    base = _scan_multi_surface_probe_grid(
        evaluator,
        shared,
        base_grid,
        ray_affine_start=ray_affine_start,
        segment_index=segment_index,
        allow_transparent_terminal_contact=allow_transparent_terminal_contact,
    )
    verified = _scan_multi_surface_probe_grid(
        evaluator,
        shared,
        verified_grid,
        ray_affine_start=ray_affine_start,
        segment_index=segment_index,
        allow_transparent_terminal_contact=allow_transparent_terminal_contact,
    )
    evaluator.require_stable_ids()
    if _multi_surface_prefix_key(base) != _multi_surface_prefix_key(verified):
        raise _InteriorSurfaceNeedsRefinement(
            "N/2N accepted-step multi-surface topologies disagree"
        )
    maximum_event_difference = 0.0
    maximum_covector_difference = 0.0
    for base_entry, verified_entry in zip(base, verified):
        affine_difference = abs(
            base_entry.crossing.ray_affine_length
            - verified_entry.crossing.ray_affine_length
        )
        if affine_difference > 2.0 * reintegrator.options.affine_tolerance:
            raise _InteriorSurfaceNeedsRefinement(
                "N/2N accepted-step multi-surface roots have not converged"
            )
        event_difference, covector_difference = _multi_probe_grid_difference(
            base_entry,
            verified_entry,
        )
        maximum_event_difference = max(
            maximum_event_difference,
            event_difference,
        )
        maximum_covector_difference = max(
            maximum_covector_difference,
            covector_difference,
        )
    return verified, maximum_event_difference, maximum_covector_difference


def _locate_termination_event(
    derivative: Callable[[Sequence[float]], State8],
    previous: HamiltonianState,
    candidate: HamiltonianState,
    full_step: float,
    termination: TerminationSurface,
    crossing: TerminationCrossing,
    options: RayTraceOptions,
) -> tuple[HamiltonianState, float]:
    """Reintegrate a bracketed accepted step to its terminal worldtube."""

    lower_step = 0.0
    upper_step = full_step
    lower_state = previous
    upper_state = candidate
    lower_value = termination.value(lower_state, crossing)
    upper_value = termination.value(upper_state, crossing)
    if not math.isfinite(lower_value) or not math.isfinite(upper_value):
        raise ArithmeticError("termination event function is non-finite")
    if abs(lower_value) <= options.event_value_tolerance:
        return lower_state, lower_step
    if abs(upper_value) <= options.event_value_tolerance:
        return upper_state, upper_step
    if math.copysign(1.0, lower_value) == math.copysign(1.0, upper_value):
        raise ArithmeticError("termination event lost its root bracket")

    for _iteration in range(options.event_maximum_iterations):
        middle_step = 0.5 * (lower_step + upper_step)
        middle_values, _error = _dormand_prince_step(
            derivative,
            previous.packed(),
            middle_step,
        )
        middle_state = HamiltonianState.unpack(middle_values)
        middle_value = termination.value(middle_state, crossing)
        if not math.isfinite(middle_value):
            raise ArithmeticError("termination event function is non-finite")
        if (
            abs(middle_value) <= options.event_value_tolerance
            or upper_step - lower_step <= options.event_affine_tolerance
        ):
            return middle_state, middle_step
        if math.copysign(1.0, lower_value) == math.copysign(
            1.0,
            middle_value,
        ):
            lower_step = middle_step
            lower_state = middle_state
            lower_value = middle_value
        else:
            upper_step = middle_step
            upper_state = middle_state
            upper_value = middle_value

    if abs(lower_value) <= abs(upper_value):
        chosen_state, chosen_step, chosen_value = (
            lower_state,
            lower_step,
            lower_value,
        )
    else:
        chosen_state, chosen_step, chosen_value = (
            upper_state,
            upper_step,
            upper_value,
        )
    if abs(chosen_value) > options.event_value_tolerance:
        raise ArithmeticError("termination root-localization budget exhausted")
    return chosen_state, chosen_step


def trace_null_geodesic(
    provider: MetricProvider,
    initial_state: HamiltonianState,
    *,
    termination: TerminationSurface | None = None,
    interior_surface: InteriorSurface | None = None,
    multi_interior_surface: MultiInteriorSurface | None = None,
    initial_multi_surface_contact: InitialMultiSurfaceContact | None = None,
    surface_options: SurfaceEventOptions = SurfaceEventOptions(),
    options: RayTraceOptions = RayTraceOptions(),
) -> RayTraceResult:
    """Trace one past- or future-directed null ray with fail-closed outcomes.

    The optional ``interior_surface`` retains the original single-surface
    behavior.  ``multi_interior_surface`` instead evaluates a stable-id family
    on one shared Hamiltonian probe cache and globally orders member roots.
    Both paths compare ``N`` and ``2N`` topology, continue through transparent
    classifications, and stop at the first proven terminal root.  An optional
    :class:`InitialMultiSurfaceContact` assigns a sign only to one authenticated
    affine-zero member contact; every positive-affine probe remains physical.
    The single- and multi-surface paths are mutually exclusive; omitting both
    preserves the original full-path path.
    """
    if not isinstance(options, RayTraceOptions):
        raise TypeError("options must be a RayTraceOptions")
    if not isinstance(surface_options, SurfaceEventOptions):
        raise TypeError("surface_options must be a SurfaceEventOptions")
    if interior_surface is not None and not isinstance(
        interior_surface,
        InteriorSurface,
    ):
        raise TypeError("interior_surface must implement InteriorSurface")
    if multi_interior_surface is not None and not isinstance(
        multi_interior_surface,
        MultiInteriorSurface,
    ):
        raise TypeError(
            "multi_interior_surface must implement MultiInteriorSurface"
        )
    if interior_surface is not None and multi_interior_surface is not None:
        raise ValueError(
            "interior_surface and multi_interior_surface are mutually exclusive"
        )
    if initial_multi_surface_contact is not None:
        if type(initial_multi_surface_contact) is not InitialMultiSurfaceContact:
            raise TypeError(
                "initial_multi_surface_contact must be the exact declaration type"
            )
        if multi_interior_surface is None:
            raise ValueError(
                "initial_multi_surface_contact requires multi_interior_surface"
            )
    state = initial_state
    packed = state.packed()
    affine_length = 0.0
    step = options.initial_step
    accepted = 0
    rejected = 0
    audit = _MetricAudit(provider)
    initial_sample = audit.sample(state.event)
    maximum_null = _normalized_null_residual_from_sample(
        initial_sample,
        state.covector,
    )
    segments: list[RayPathSegment] = []
    if maximum_null > options.null_residual_limit:
        raise ValueError(
            "initial covector is not null within the declared residual limit"
        )
    if (
        audit.maximum_interpolation_error
        > options.metric_interpolation_error_limit
    ):
        raise ValueError(
            "initial metric interpolation error exceeds the declared limit"
        )

    derivative = lambda values: _derivative(audit, values)  # noqa: E731
    multi_surface_evaluator = (
        None
        if multi_interior_surface is None
        else _ValidatedMultiInteriorSurface(
            multi_interior_surface,
            initial_state,
            initial_multi_surface_contact,
            surface_options.surface_value_tolerance,
        )
    )
    surface_reintegrator = (
        None
        if interior_surface is None and multi_interior_surface is None
        else _StepSurfaceReintegrator(audit, surface_options)
    )
    interior_crossings: list[ClassifiedInteriorSurfaceCrossing] = []
    multi_surface_crossings: list[
        ClassifiedMultiInteriorSurfaceCrossing
    ] = []
    maximum_probe_event_difference = 0.0
    maximum_probe_covector_difference = 0.0

    def finish(
        outcome: str,
        failure_reason: str | None = None,
        *,
        terminal_state: HamiltonianState | None = None,
        terminal_affine_length: float | None = None,
        accepted_steps: int | None = None,
        terminal_target_id: str | None = None,
    ) -> RayTraceResult:
        surface_trace = None
        if interior_surface is not None:
            surface_trace = InteriorSurfaceTrace(
                crossings=tuple(interior_crossings),
                base_subdivisions_per_step=(
                    surface_options.subdivisions_per_segment
                ),
                verification_subdivisions_per_step=(
                    2 * surface_options.subdivisions_per_segment
                ),
                topology_converged=(
                    outcome not in ("integrator-failure", "unresolved")
                ),
                maximum_probe_event_difference=(
                    maximum_probe_event_difference
                ),
                maximum_probe_covector_relative_difference=(
                    maximum_probe_covector_difference
                ),
            )
        multi_surface_trace = None
        if multi_surface_evaluator is not None:
            multi_surface_trace = MultiInteriorSurfaceTrace(
                surface_ids=multi_surface_evaluator.surface_ids,
                crossings=tuple(multi_surface_crossings),
                base_subdivisions_per_step=(
                    surface_options.subdivisions_per_segment
                ),
                verification_subdivisions_per_step=(
                    2 * surface_options.subdivisions_per_segment
                ),
                topology_converged=(
                    outcome not in ("integrator-failure", "unresolved")
                ),
                maximum_probe_event_difference=(
                    maximum_probe_event_difference
                ),
                maximum_probe_covector_relative_difference=(
                    maximum_probe_covector_difference
                ),
                probe_reintegrations=(
                    0
                    if surface_reintegrator is None
                    else surface_reintegrator.reintegrations
                ),
                surface_value_evaluations=(
                    multi_surface_evaluator.surface_value_evaluations
                ),
                initial_contact=(
                    multi_surface_evaluator.authenticated_initial_contact
                ),
            )
        return RayTraceResult(
            outcome=outcome,
            terminal_state=terminal_state or state,
            affine_length=(
                affine_length
                if terminal_affine_length is None
                else terminal_affine_length
            ),
            accepted_steps=accepted if accepted_steps is None else accepted_steps,
            rejected_steps=rejected,
            maximum_null_residual=maximum_null,
            maximum_metric_interpolation_error=audit.maximum_interpolation_error,
            segments=tuple(segments),
            failure_reason=failure_reason,
            terminal_target_id=terminal_target_id,
            interior_surface_trace=surface_trace,
            multi_surface_trace=multi_surface_trace,
        )

    if termination is not None:
        try:
            initial_terminal = termination.classify_initial(state)
        except (ArithmeticError, IndexError, TypeError, ValueError, OverflowError) as error:
            return finish("integrator-failure", str(error))
        if initial_terminal is not None:
            initial_outcome, initial_target = initial_terminal
            if not initial_outcome or not initial_target:
                return finish(
                    "integrator-failure",
                    "initial termination classification is malformed",
                )
            return finish(
                initial_outcome,
                terminal_target_id=initial_target,
            )

    while affine_length < options.maximum_affine_length:
        if accepted >= options.maximum_accepted_steps:
            return finish("unresolved", "accepted-step budget exhausted")
        remaining = options.maximum_affine_length - affine_length
        step = min(step, remaining)
        if step < options.minimum_step:
            return finish("integrator-failure", "minimum affine step reached")
        try:
            candidate, error = _dormand_prince_step(derivative, packed, step)
            error_norm = _error_norm(packed, candidate, error, options)
        except (
            ArithmeticError,
            IndexError,
            TypeError,
            ValueError,
            OverflowError,
        ) as error_value:
            return finish("integrator-failure", str(error_value))
        if (
            audit.maximum_interpolation_error
            > options.metric_interpolation_error_limit
        ):
            return finish(
                "integrator-failure",
                "metric interpolation error exceeded declared limit",
            )
        if not math.isfinite(error_norm):
            error_norm = math.inf
        if error_norm > 1.0:
            rejected += 1
            if rejected > options.maximum_rejected_steps:
                return finish(
                    "integrator-failure",
                    "rejected-step budget exhausted",
                )
            factor = max(0.1, 0.9 * error_norm ** -0.2)
            step *= factor
            continue

        previous = state
        candidate_state = HamiltonianState.unpack(candidate)
        accepted_step = step
        crossing: TerminationCrossing | None = None
        step_surface_entries: tuple[
            ClassifiedInteriorSurfaceCrossing,
            ...,
        ] = ()
        step_multi_surface_entries: tuple[
            ClassifiedMultiInteriorSurfaceCrossing,
            ...,
        ] = ()
        interior_terminal: InteriorSurfaceDecision | None = None
        try:
            if termination is not None:
                crossing = termination.crossing(previous, candidate_state)
                if (
                    crossing is None
                    and termination.needs_refinement(previous, candidate_state)
                ):
                    rejected += 1
                    if rejected > options.maximum_rejected_steps:
                        return finish(
                            "integrator-failure",
                            "event-refinement budget exhausted",
                        )
                    step *= 0.5
                    continue
                if crossing is not None:
                    candidate_state, accepted_step = _locate_termination_event(
                        derivative,
                        previous,
                        candidate_state,
                        step,
                        termination,
                        crossing,
                        options,
                    )
            if interior_surface is not None:
                if surface_reintegrator is None:
                    raise AssertionError("interior surface reintegrator is missing")
                (
                    step_surface_entries,
                    step_probe_event_difference,
                    step_probe_covector_difference,
                ) = _locate_converged_step_surface_prefix(
                    interior_surface,
                    surface_reintegrator,
                    previous,
                    candidate_state,
                    accepted_step,
                    ray_affine_start=affine_length,
                    segment_index=accepted,
                    allow_transparent_terminal_contact=(crossing is not None),
                )
                maximum_probe_event_difference = max(
                    maximum_probe_event_difference,
                    step_probe_event_difference,
                )
                maximum_probe_covector_difference = max(
                    maximum_probe_covector_difference,
                    step_probe_covector_difference,
                )
                if (
                    step_surface_entries
                    and step_surface_entries[-1].decision.terminates
                ):
                    terminal_entry = step_surface_entries[-1]
                    interior_terminal = terminal_entry.decision
                    candidate_state = terminal_entry.crossing.state
                    accepted_step = terminal_entry.crossing.segment_affine_length
                    if accepted_step <= 0.0:
                        raise SurfaceEventError(
                            "terminal interior surface has no positive ray prefix"
                        )
            if multi_surface_evaluator is not None:
                if surface_reintegrator is None:
                    raise AssertionError(
                        "multi-surface reintegrator is missing"
                    )
                (
                    step_multi_surface_entries,
                    step_probe_event_difference,
                    step_probe_covector_difference,
                ) = _locate_converged_step_multi_surface_prefix(
                    multi_surface_evaluator,
                    surface_reintegrator,
                    previous,
                    candidate_state,
                    accepted_step,
                    ray_affine_start=affine_length,
                    segment_index=accepted,
                    allow_transparent_terminal_contact=(crossing is not None),
                )
                maximum_probe_event_difference = max(
                    maximum_probe_event_difference,
                    step_probe_event_difference,
                )
                maximum_probe_covector_difference = max(
                    maximum_probe_covector_difference,
                    step_probe_covector_difference,
                )
                if (
                    step_multi_surface_entries
                    and step_multi_surface_entries[-1].decision.terminates
                ):
                    terminal_entry = step_multi_surface_entries[-1]
                    interior_terminal = terminal_entry.decision
                    candidate_state = terminal_entry.crossing.state
                    accepted_step = terminal_entry.crossing.segment_affine_length
                    if accepted_step <= 0.0:
                        raise SurfaceEventError(
                            "terminal multi-surface has no positive ray prefix"
                        )
            endpoint_sample = audit.sample(candidate_state.event)
            endpoint_null = _normalized_null_residual_from_sample(
                endpoint_sample,
                candidate_state.covector,
            )
            midpoint_values, _midpoint_error = _dormand_prince_step(
                derivative,
                previous.packed(),
                0.5 * accepted_step,
            )
            midpoint = HamiltonianState.unpack(midpoint_values)
            midpoint_sample = audit.sample(midpoint.event)
            midpoint_null = _normalized_null_residual_from_sample(
                midpoint_sample,
                midpoint.covector,
            )
        except _InteriorSurfaceNeedsRefinement as error_value:
            rejected += 1
            if rejected > options.maximum_rejected_steps:
                return finish(
                    "integrator-failure",
                    f"surface-refinement budget exhausted: {error_value}",
                    terminal_state=previous,
                )
            step *= 0.5
            continue
        except (
            ArithmeticError,
            IndexError,
            SurfaceEventError,
            TypeError,
            ValueError,
            OverflowError,
        ) as error_value:
            return finish(
                "integrator-failure",
                str(error_value),
                terminal_state=previous,
            )

        if (
            audit.maximum_interpolation_error
            > options.metric_interpolation_error_limit
        ):
            return finish(
                "integrator-failure",
                "metric interpolation error exceeded declared limit",
                terminal_state=previous,
            )

        state = candidate_state
        packed = state.packed()
        affine_length += accepted_step
        accepted += 1
        maximum_null = max(maximum_null, endpoint_null)
        maximum_null = max(maximum_null, midpoint_null)
        if surface_reintegrator is not None:
            maximum_null = max(
                maximum_null,
                surface_reintegrator.maximum_null_residual,
            )
        if maximum_null > options.null_residual_limit:
            return finish(
                "integrator-failure",
                "null residual exceeded declared limit",
            )
        if options.record_path:
            segments.append(
                RayPathSegment(
                    start=previous,
                    end=state,
                    midpoint=midpoint,
                    affine_length=accepted_step,
                    midpoint_null_residual=midpoint_null,
                )
            )
        interior_crossings.extend(step_surface_entries)
        multi_surface_crossings.extend(step_multi_surface_entries)

        if interior_terminal is not None:
            if interior_terminal.outcome is None or interior_terminal.target_id is None:
                raise AssertionError("terminal interior surface decision is incomplete")
            return finish(
                interior_terminal.outcome,
                terminal_target_id=interior_terminal.target_id,
            )
        if crossing is not None:
            return finish(
                crossing.outcome,
                terminal_target_id=crossing.target_id,
            )
        if error_norm == 0.0:
            factor = 5.0
        else:
            factor = min(5.0, max(0.2, 0.9 * error_norm ** -0.2))
        step = min(options.maximum_step, step * factor)

    if termination is not None:
        return finish("unresolved", "affine-parameter budget exhausted")
    return finish("completed")


def trace_refined_null_geodesic(
    provider: MetricProvider,
    initial_state: HamiltonianState,
    *,
    termination: TerminationSurface | None = None,
    interior_surface: InteriorSurface | None = None,
    multi_interior_surface: MultiInteriorSurface | None = None,
    initial_multi_surface_contact: InitialMultiSurfaceContact | None = None,
    surface_options: SurfaceEventOptions = SurfaceEventOptions(),
    fine_options: RayTraceOptions = RayTraceOptions(),
    record_coarse_path: bool = False,
    coarse_tolerance_multiplier: float = 32.0,
    terminal_event_tolerance: float = 2.0e-5,
    terminal_covector_tolerance: float = 2.0e-5,
) -> RayRefinementResult:
    """Trace the visible ray prefix twice and compare terminal observables.

    The embedded DP5(4) estimate controls local steps; it is not a global image
    error.  This independent second trace is the minimum whole-ray convergence
    gate.  Product-specific orchestration should additionally compare escape
    directions, frequency transfer, Jacobi fields, and radiative observables.
    """

    if type(record_coarse_path) is not bool:
        raise ValueError("record_coarse_path must be a bool")
    if not isinstance(surface_options, SurfaceEventOptions):
        raise TypeError("surface_options must be a SurfaceEventOptions")
    if interior_surface is not None and not isinstance(
        interior_surface,
        InteriorSurface,
    ):
        raise TypeError("interior_surface must implement InteriorSurface")
    if multi_interior_surface is not None and not isinstance(
        multi_interior_surface,
        MultiInteriorSurface,
    ):
        raise TypeError(
            "multi_interior_surface must implement MultiInteriorSurface"
        )
    if interior_surface is not None and multi_interior_surface is not None:
        raise ValueError(
            "interior_surface and multi_interior_surface are mutually exclusive"
        )
    if initial_multi_surface_contact is not None:
        if type(initial_multi_surface_contact) is not InitialMultiSurfaceContact:
            raise TypeError(
                "initial_multi_surface_contact must be the exact declaration type"
            )
        if multi_interior_surface is None:
            raise ValueError(
                "initial_multi_surface_contact requires multi_interior_surface"
            )
    if record_coarse_path and fine_options.record_path is not True:
        raise ValueError(
            "record_coarse_path requires fine_options.record_path=True"
        )
    finite_positive = (
        coarse_tolerance_multiplier,
        terminal_event_tolerance,
        terminal_covector_tolerance,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
        raise ValueError("refinement multipliers and tolerances must be positive")
    if coarse_tolerance_multiplier <= 1.0:
        raise ValueError("coarse tolerance multiplier must be greater than one")

    coarse_step_multiplier = min(
        8.0,
        math.sqrt(coarse_tolerance_multiplier),
    )
    coarse_options = replace(
        fine_options,
        absolute_tolerance=(
            fine_options.absolute_tolerance * coarse_tolerance_multiplier
        ),
        relative_tolerance=(
            fine_options.relative_tolerance * coarse_tolerance_multiplier
        ),
        initial_step=(
            fine_options.initial_step * coarse_step_multiplier
        ),
        maximum_step=(
            fine_options.maximum_step * coarse_step_multiplier
        ),
        record_path=record_coarse_path,
    )
    coarse_surface_options = replace(
        surface_options,
        absolute_tolerance=(
            surface_options.absolute_tolerance * coarse_tolerance_multiplier
        ),
        relative_tolerance=(
            surface_options.relative_tolerance * coarse_tolerance_multiplier
        ),
    )
    fine = trace_null_geodesic(
        provider,
        initial_state,
        termination=termination,
        interior_surface=interior_surface,
        multi_interior_surface=multi_interior_surface,
        initial_multi_surface_contact=initial_multi_surface_contact,
        surface_options=surface_options,
        options=fine_options,
    )
    coarse = trace_null_geodesic(
        provider,
        initial_state,
        termination=termination,
        interior_surface=interior_surface,
        multi_interior_surface=multi_interior_surface,
        initial_multi_surface_contact=initial_multi_surface_contact,
        surface_options=coarse_surface_options,
        options=coarse_options,
    )
    event_difference = math.sqrt(
        math.fsum(
            (fine.terminal_state.event[index] - coarse.terminal_state.event[index])
            ** 2
            for index in range(4)
        )
    )
    covector_scale = max(
        math.sqrt(math.fsum(value * value for value in fine.terminal_state.covector)),
        math.sqrt(math.fsum(value * value for value in coarse.terminal_state.covector)),
        1.0e-300,
    )
    covector_difference = math.sqrt(
        math.fsum(
            (
                fine.terminal_state.covector[index]
                - coarse.terminal_state.covector[index]
            )
            ** 2
            for index in range(4)
        )
    ) / covector_scale
    outcome_agrees = fine.outcome == coarse.outcome
    discretizations_differ = (
        fine.accepted_steps,
        fine.rejected_steps,
    ) != (
        coarse.accepted_steps,
        coarse.rejected_steps,
    )
    acceptable_outcomes = {"captured", "escaped", "completed"}
    surface_topology_agrees = (
        interior_surface is None and multi_interior_surface is None
    )
    surface_terminal_success = False
    if interior_surface is not None:
        fine_surface = fine.interior_surface_trace
        coarse_surface = coarse.interior_surface_trace
        if fine_surface is not None and coarse_surface is not None:
            surface_topology_agrees = (
                fine_surface.topology_converged
                and coarse_surface.topology_converged
                and _surface_prefix_key(fine_surface.crossings)
                == _surface_prefix_key(coarse_surface.crossings)
            )
            if fine_surface.crossings and coarse_surface.crossings:
                fine_last = fine_surface.crossings[-1]
                coarse_last = coarse_surface.crossings[-1]
                surface_terminal_success = (
                    fine_last.decision.terminates
                    and coarse_last.decision.terminates
                    and fine_last.decision.outcome == fine.outcome
                    and coarse_last.decision.outcome == coarse.outcome
                    and fine_last.decision.target_id == fine.terminal_target_id
                    and coarse_last.decision.target_id == coarse.terminal_target_id
                )
    if multi_interior_surface is not None:
        fine_surface = fine.multi_surface_trace
        coarse_surface = coarse.multi_surface_trace
        if fine_surface is not None and coarse_surface is not None:
            surface_topology_agrees = (
                fine_surface.surface_ids == coarse_surface.surface_ids
                and (
                    (
                        fine_surface.initial_contact is None
                        and coarse_surface.initial_contact is None
                    )
                    or (
                        fine_surface.initial_contact is not None
                        and coarse_surface.initial_contact is not None
                        and fine_surface.initial_contact.surface_id
                        == coarse_surface.initial_contact.surface_id
                        and fine_surface.initial_contact.side
                        == coarse_surface.initial_contact.side
                        and fine_surface.initial_contact.actual_surface_value
                        == coarse_surface.initial_contact.actual_surface_value
                    )
                )
                and fine_surface.topology_converged
                and coarse_surface.topology_converged
                and _multi_surface_prefix_key(fine_surface.crossings)
                == _multi_surface_prefix_key(coarse_surface.crossings)
            )
            if fine_surface.crossings and coarse_surface.crossings:
                fine_last = fine_surface.crossings[-1]
                coarse_last = coarse_surface.crossings[-1]
                surface_terminal_success = (
                    fine_last.surface_id == coarse_last.surface_id
                    and fine_last.decision.terminates
                    and coarse_last.decision.terminates
                    and fine_last.decision.outcome == fine.outcome
                    and coarse_last.decision.outcome == coarse.outcome
                    and fine_last.decision.target_id
                    == fine.terminal_target_id
                    and coarse_last.decision.target_id
                    == coarse.terminal_target_id
                )
    terminal_target_agrees = fine.terminal_target_id == coarse.terminal_target_id
    terminal_target_is_valid = (
        fine.outcome == "completed"
        or bool(fine.terminal_target_id)
    )
    converged = (
        outcome_agrees
        and discretizations_differ
        and terminal_target_agrees
        and terminal_target_is_valid
        and (
            fine.outcome in acceptable_outcomes
            or surface_terminal_success
        )
        and surface_topology_agrees
        and event_difference <= terminal_event_tolerance
        and covector_difference <= terminal_covector_tolerance
    )
    return RayRefinementResult(
        fine=fine,
        coarse=coarse,
        outcome_agrees=outcome_agrees,
        terminal_event_difference=event_difference,
        terminal_covector_difference=covector_difference,
        discretizations_differ=discretizations_differ,
        terminal_target_agrees=terminal_target_agrees,
        converged=converged,
    )


def static_schwarzschild_camera_ray(
    provider: MetricProvider,
    *,
    observer_radius_m: float,
    screen_x: float,
    screen_y: float,
) -> HamiltonianState:
    """Create a past-directed pinhole ray for a static +X-axis observer.

    This initializer is specific to the exact Schwarzschild Kerr-Schild chart.
    The offline integrator itself remains provider-independent.
    """
    mass = getattr(provider, "mass_m", None)
    if not isinstance(mass, (int, float)) or isinstance(mass, bool):
        raise TypeError("Schwarzschild camera initializer needs a mass_m provider")
    values = (observer_radius_m, screen_x, screen_y, float(mass))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("camera ray parameters must be finite")
    if observer_radius_m <= 2.0 * mass:
        raise ValueError("static Schwarzschild observer must be outside 2M")

    lapse_squared = 1.0 - 2.0 * mass / observer_radius_m
    lapse = math.sqrt(lapse_squared)
    h2 = 2.0 * mass / observer_radius_m
    observer: Vector4 = (1.0 / lapse, 0.0, 0.0, 0.0)
    right: Vector4 = (0.0, 0.0, 1.0, 0.0)
    up: Vector4 = (0.0, 0.0, 0.0, 1.0)
    forward: Vector4 = (-h2 / lapse, -lapse, 0.0, 0.0)
    inverse_norm = 1.0 / math.sqrt(1.0 + screen_x * screen_x + screen_y * screen_y)
    contravariant = tuple(  # type: ignore[assignment]
        -observer[index]
        + inverse_norm
        * (
            screen_x * right[index]
            + screen_y * up[index]
            + forward[index]
        )
        for index in range(4)
    )
    event: Vector4 = (0.0, observer_radius_m, 0.0, 0.0)
    metric = provider.sample(event)
    covector = matrix_vector(metric.covariant, contravariant)
    return HamiltonianState(event=event, covector=covector)
