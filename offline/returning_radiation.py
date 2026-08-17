"""Axisymmetric returning-radiation energy-kernel fixed point.

This module consumes an externally generated, receiver-centred bolometric
flux kernel and solves only

``F_in[i] = sum_j K[i,j] F_out[j]``

and

``F_out = F_0 + F_in = F_0 + K F_out``.

Every non-negative dimensionless coefficient ``K[i,j]`` is defined to already
contain the emitter angular law, the bolometric ``g**4`` shift, receiver
incidence cosine, receiver solid-angle/Jacobian factor, and the discretised
emitting-annulus weight.  It therefore maps local comoving outgoing flux on
annulus ``j`` to local comoving incident flux on annulus ``i``.  These factors
must not be applied a second time by this solver.

The equation is the absorbed-and-thermally-reradiated, energy-only ``F_in``
part of Appendix D of Li et al. (2005, KERRBB).  It deliberately omits the
returning-radiation stress/work term ``F_S`` in their equation D17.  It also
does not perform ray tracing or independently validate a supplied kernel, and
does not implement spectral redistribution, scattering, polarization, finite
thickness, an atmosphere, GRMHD, or a complete KERRBB model.  Cunningham
(1976) is the transfer-function provenance for the receiver-centred framing;
it is not a claim that this numerical kernel was independently reproduced.

Photon return/capture/escape probabilities, when provided, live in a separate
type.  They are photon-number fate diagnostics and are never substituted for
the local energy-flux kernel.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence


RETURNING_RADIATION_IMPLEMENTATION_ID: Final = (
    "axisymmetric-receiver-centred-bolometric-returning-radiation/v1"
)
KERRBB_SOURCE_URL: Final = "https://arxiv.org/abs/astro-ph/0411583"
CUNNINGHAM_SOURCE_URL: Final = (
    "https://ui.adsabs.harvard.edu/abs/1976ApJ...208..534C/abstract"
)

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": (
            "axisymmetric receiver-centred local-comoving bolometric "
            "returning-radiation energy-kernel fixed point"
        ),
        "implementationId": RETURNING_RADIATION_IMPLEMENTATION_ID,
        "primarySource": KERRBB_SOURCE_URL,
        "primarySourceCitation": (
            "Li et al. 2005, ApJS 157, 335, Appendix D, especially D17"
        ),
        "transferFunctionSource": CUNNINGHAM_SOURCE_URL,
        "transferFunctionSourceCitation": "Cunningham 1976, ApJ 208, 534",
        "equation": "F_out = F_0 + F_in; F_in = K F_out",
        "kernelCoefficientSemantics": (
            "K[i,j] already includes emitter angular law, g^4, receiver "
            "incidence cosine, receiver solid-angle/Jacobian, and the "
            "discretised emitting-annulus weight"
        ),
        "isIndependentRayKernel": False,
        "includesReturningRadiationStressWorkFS": False,
        "includesSpectralRedistribution": False,
        "includesScattering": False,
        "includesPolarization": False,
        "includesFiniteThickness": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "isCompleteKerrbb": False,
        "prohibitedClaim": (
            "Do not describe this energy-only fixed point as a complete "
            "KERRBB, independent ray-kernel, finite-thickness, atmosphere, "
            "polarized transfer, or GRMHD calculation."
        ),
    }
)


class ReturningRadiationError(RuntimeError):
    """Base class for fail-closed returning-radiation errors."""


class ReturningRadiationConvergenceError(ReturningRadiationError):
    """Raised when subcriticality or the fixed point is not certified."""


class ReturningRadiationVerificationError(ReturningRadiationError):
    """Raised when a returned solution cannot be reproduced exactly."""


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
    # Canonicalize negative IEEE zero so physically identical inputs seal to
    # the same descriptor and digest.
    return 0.0 if result == 0.0 else result


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    try:
        return tuple(value)
    except TypeError as error:
        raise ValueError(f"{label} must be a sequence") from error


def _positive_strictly_increasing_radii(
    values: Sequence[float],
    label: str,
) -> tuple[float, ...]:
    entries = _sequence(values, label)
    if not entries:
        raise ValueError(f"{label} must not be empty")
    result = tuple(
        _positive_number(value, f"{label}[{index}]")
        for index, value in enumerate(entries)
    )
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError(f"{label} must be strictly increasing")
    return result


def _non_negative_vector(
    values: Sequence[float],
    label: str,
    *,
    expected_length: int | None = None,
) -> tuple[float, ...]:
    entries = _sequence(values, label)
    if expected_length is not None and len(entries) != expected_length:
        raise ValueError(
            f"{label} must contain exactly {expected_length} entries"
        )
    return tuple(
        _non_negative_number(value, f"{label}[{index}]")
        for index, value in enumerate(entries)
    )


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
        raise ValueError("descriptor must be finite canonical JSON data") from error


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _matvec(
    rows: tuple[tuple[float, ...], ...],
    vector: tuple[float, ...],
    *,
    label: str,
) -> tuple[float, ...]:
    result: list[float] = []
    try:
        for receiver_index, row in enumerate(rows):
            value = math.fsum(
                coefficient * vector[emitter_index]
                for emitter_index, coefficient in enumerate(row)
            )
            if not math.isfinite(value) or value < 0.0:
                raise ReturningRadiationConvergenceError(
                    f"{label}[{receiver_index}] is not finite and non-negative"
                )
            result.append(value)
    except OverflowError as error:
        raise ReturningRadiationConvergenceError(
            f"{label} overflowed the finite numerical domain"
        ) from error
    return tuple(result)


@dataclass(frozen=True, slots=True)
class AxisymmetricReturningRadiationKernel:
    """Externally supplied local-energy kernel ``K[receiver][emitter]``.

    ``annulus_radii_over_mass`` are strictly increasing representative radii.
    The discretised annulus widths/areas belong inside the coefficients, not
    in this radius vector.  ``ray_kernel_producer_id`` identifies the external
    producer; this class validates numbers and semantics but does not retrace
    any ray or establish that the coefficients are physically correct.
    """

    annulus_radii_over_mass: tuple[float, ...]
    receiver_emitter_coefficients: tuple[tuple[float, ...], ...]
    ray_kernel_producer_id: str
    _descriptor_json: str = field(init=False, repr=False, compare=False)
    _descriptor_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        radii = _positive_strictly_increasing_radii(
            self.annulus_radii_over_mass,
            "annulus_radii_over_mass",
        )
        producer = _non_empty_string(
            self.ray_kernel_producer_id,
            "ray_kernel_producer_id",
        )
        input_rows = _sequence(
            self.receiver_emitter_coefficients,
            "receiver_emitter_coefficients",
        )
        if len(input_rows) != len(radii):
            raise ValueError(
                "receiver_emitter_coefficients must have one row per receiver "
                "annulus"
            )
        rows = tuple(
            _non_negative_vector(
                row,
                f"receiver_emitter_coefficients[{receiver_index}]",
                expected_length=len(radii),
            )
            for receiver_index, row in enumerate(input_rows)
        )
        descriptor = {
            "annulusRadiiOverMass": radii,
            "coefficientIndexOrder": "K[receiverAnnulus][emitterAnnulus]",
            "coefficientSemantics": SCIENTIFIC_STATUS[
                "kernelCoefficientSemantics"
            ],
            "implementationId": RETURNING_RADIATION_IMPLEMENTATION_ID,
            "isIndependentRayKernel": False,
            "rayKernelProducerId": producer,
            "receiverEmitterCoefficients": rows,
        }
        descriptor_json = _canonical_json(descriptor)
        object.__setattr__(self, "annulus_radii_over_mass", radii)
        object.__setattr__(self, "receiver_emitter_coefficients", rows)
        object.__setattr__(self, "ray_kernel_producer_id", producer)
        object.__setattr__(self, "_descriptor_json", descriptor_json)
        object.__setattr__(
            self,
            "_descriptor_sha256",
            hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest(),
        )

    @property
    def annulus_count(self) -> int:
        return len(self.annulus_radii_over_mass)

    @property
    def canonical_descriptor_json(self) -> str:
        return self._descriptor_json

    @property
    def canonical_descriptor_sha256(self) -> str:
        return self._descriptor_sha256

    def canonical_descriptor(self) -> Mapping[str, Any]:
        """Return a fresh JSON-compatible copy of the canonical descriptor."""

        return json.loads(self._descriptor_json)


@dataclass(frozen=True, slots=True)
class ReturningRadiationFixedPointPolicy:
    """Finite convergence gates for the monotone energy-only fixed point."""

    maximum_iterations: int = 10_000
    absolute_residual_tolerance: float = 0.0
    relative_residual_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if type(self.maximum_iterations) is not int or self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be a positive integer")
        absolute = _non_negative_number(
            self.absolute_residual_tolerance,
            "absolute_residual_tolerance",
        )
        relative = _non_negative_number(
            self.relative_residual_tolerance,
            "relative_residual_tolerance",
        )
        if absolute == 0.0 and relative == 0.0:
            raise ValueError("at least one residual tolerance must be positive")
        object.__setattr__(self, "absolute_residual_tolerance", absolute)
        object.__setattr__(self, "relative_residual_tolerance", relative)

    def canonical_descriptor(self) -> Mapping[str, Any]:
        return {
            "absoluteResidualTolerance": self.absolute_residual_tolerance,
            "maximumIterations": self.maximum_iterations,
            "relativeResidualTolerance": self.relative_residual_tolerance,
        }

    @property
    def canonical_descriptor_sha256(self) -> str:
        return _sha256_json(self.canonical_descriptor())


DEFAULT_FIXED_POINT_POLICY: Final = ReturningRadiationFixedPointPolicy()


@dataclass(frozen=True, slots=True)
class ReturningRadiationFluxSolution:
    """One converged, kernel-bound, reproducible fixed-point result."""

    intrinsic_flux: tuple[float, ...]
    incident_returning_flux: tuple[float, ...]
    outgoing_flux: tuple[float, ...]
    equation_residual: tuple[float, ...]
    residual_tolerances: tuple[float, ...]
    iterations: int
    monotonic_fixed_point_verified: bool
    kernel_descriptor_sha256: str
    policy_descriptor_sha256: str

    def __post_init__(self) -> None:
        intrinsic = _non_negative_vector(self.intrinsic_flux, "intrinsic_flux")
        count = len(intrinsic)
        if count < 1:
            raise ValueError("intrinsic_flux must not be empty")
        incident = _non_negative_vector(
            self.incident_returning_flux,
            "incident_returning_flux",
            expected_length=count,
        )
        outgoing = _non_negative_vector(
            self.outgoing_flux,
            "outgoing_flux",
            expected_length=count,
        )
        residual_entries = _sequence(self.equation_residual, "equation_residual")
        if len(residual_entries) != count:
            raise ValueError("equation_residual has the wrong shape")
        residual = tuple(
            _finite_number(value, f"equation_residual[{index}]")
            for index, value in enumerate(residual_entries)
        )
        tolerances = _non_negative_vector(
            self.residual_tolerances,
            "residual_tolerances",
            expected_length=count,
        )
        if any(
            abs(residual_value) > tolerance
            for residual_value, tolerance in zip(residual, tolerances)
        ):
            raise ValueError("equation residual exceeds its declared tolerance")
        if type(self.iterations) is not int or self.iterations < 0:
            raise ValueError("iterations must be a non-negative integer")
        if self.monotonic_fixed_point_verified is not True:
            raise ValueError("a solution must certify monotonic fixed-point updates")
        for name in ("kernel_descriptor_sha256", "policy_descriptor_sha256"):
            digest = getattr(self, name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        object.__setattr__(self, "intrinsic_flux", intrinsic)
        object.__setattr__(self, "incident_returning_flux", incident)
        object.__setattr__(self, "outgoing_flux", outgoing)
        object.__setattr__(self, "equation_residual", residual)
        object.__setattr__(self, "residual_tolerances", tolerances)

    @property
    def maximum_absolute_equation_residual(self) -> float:
        return max(abs(value) for value in self.equation_residual)

    def canonical_descriptor(self) -> Mapping[str, Any]:
        return {
            "equationResidual": self.equation_residual,
            "incidentReturningFlux": self.incident_returning_flux,
            "intrinsicFlux": self.intrinsic_flux,
            "iterations": self.iterations,
            "kernelDescriptorSha256": self.kernel_descriptor_sha256,
            "monotonicFixedPointVerified": self.monotonic_fixed_point_verified,
            "outgoingFlux": self.outgoing_flux,
            "policyDescriptorSha256": self.policy_descriptor_sha256,
            "residualTolerances": self.residual_tolerances,
        }

    @property
    def canonical_descriptor_sha256(self) -> str:
        return _sha256_json(self.canonical_descriptor())


def _residual_and_tolerances(
    intrinsic_flux: tuple[float, ...],
    incident_flux: tuple[float, ...],
    outgoing_flux: tuple[float, ...],
    policy: ReturningRadiationFixedPointPolicy,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    residual: list[float] = []
    tolerances: list[float] = []
    for intrinsic, incident, outgoing in zip(
        intrinsic_flux,
        incident_flux,
        outgoing_flux,
    ):
        try:
            right_hand_side = math.fsum((intrinsic, incident))
        except OverflowError as error:
            raise ReturningRadiationConvergenceError(
                "fixed-point right-hand side overflowed"
            ) from error
        if not math.isfinite(right_hand_side):
            raise ReturningRadiationConvergenceError(
                "fixed-point right-hand side is not finite"
            )
        difference = right_hand_side - outgoing
        tolerance = math.fsum(
            (
                policy.absolute_residual_tolerance,
                policy.relative_residual_tolerance
                * max(abs(right_hand_side), abs(outgoing)),
            )
        )
        if not math.isfinite(difference) or not math.isfinite(tolerance):
            raise ReturningRadiationConvergenceError(
                "fixed-point residual is not finite"
            )
        residual.append(difference)
        tolerances.append(tolerance)
    return tuple(residual), tuple(tolerances)


def _residual_gate_passed(
    residual: tuple[float, ...],
    tolerances: tuple[float, ...],
) -> bool:
    return all(
        abs(value) <= tolerance
        for value, tolerance in zip(residual, tolerances)
    )


def _certify_subcritical_kernel(
    kernel: AxisymmetricReturningRadiationKernel,
    maximum_iterations: int,
) -> None:
    """Find a positive vector ``v`` with ``K v < v``.

    For a non-negative matrix this is a finite Collatz--Wielandt certificate
    that its spectral radius is below one.  The search itself is bounded and
    fails closed; it is not an eigensolver.  Stable matrices with a certificate
    that is too slow to expose within the declared budget are rejected.
    """

    vector = (1.0,) * kernel.annulus_count
    for _ in range(maximum_iterations):
        mapped = _matvec(
            kernel.receiver_emitter_coefficients,
            vector,
            label="subcriticality_probe",
        )
        if all(mapped_value < value for mapped_value, value in zip(mapped, vector)):
            return
        if all(
            mapped_value >= value
            for mapped_value, value in zip(mapped, vector)
        ):
            raise ReturningRadiationConvergenceError(
                "kernel is certified critical or supercritical; "
                "subcriticality was not certified"
            )
        try:
            vector = tuple(
                math.fsum((1.0, mapped_value)) for mapped_value in mapped
            )
        except OverflowError as error:
            raise ReturningRadiationConvergenceError(
                "subcriticality certificate overflowed"
            ) from error
        if any(not math.isfinite(value) for value in vector):
            raise ReturningRadiationConvergenceError(
                "subcriticality certificate left the finite numerical domain"
            )
    raise ReturningRadiationConvergenceError(
        "kernel subcriticality was not certified within maximum_iterations; "
        "the kernel may be critical or divergent"
    )


def solve_absorbed_returning_radiation(
    kernel: AxisymmetricReturningRadiationKernel,
    intrinsic_flux: Sequence[float],
    policy: ReturningRadiationFixedPointPolicy = DEFAULT_FIXED_POINT_POLICY,
) -> ReturningRadiationFluxSolution:
    """Solve the absorbed, thermally reradiated energy-only fixed point.

    Iteration starts at ``F_0`` and adds the non-negative Neumann-series terms
    ``K F_0, K**2 F_0, ...``.  Every public iterate is therefore componentwise
    monotone.  A result is returned only after both a finite subcriticality
    certificate and the direct equation-residual gate pass.  Exhausting either
    bounded search raises :class:`ReturningRadiationConvergenceError`.
    """

    checked_kernel = _validated_live_kernel(kernel)
    checked_policy = _validated_live_policy(policy)
    source = _non_negative_vector(
        intrinsic_flux,
        "intrinsic_flux",
        expected_length=checked_kernel.annulus_count,
    )
    _certify_subcritical_kernel(
        checked_kernel,
        checked_policy.maximum_iterations,
    )

    outgoing = source
    incident = _matvec(
        checked_kernel.receiver_emitter_coefficients,
        outgoing,
        label="incident_returning_flux",
    )
    residual, tolerances = _residual_and_tolerances(
        source,
        incident,
        outgoing,
        checked_policy,
    )
    iterations = 0
    if not _residual_gate_passed(residual, tolerances):
        increment = incident
        for iteration in range(1, checked_policy.maximum_iterations + 1):
            try:
                next_outgoing = tuple(
                    math.fsum((value, delta))
                    for value, delta in zip(outgoing, increment)
                )
            except OverflowError as error:
                raise ReturningRadiationConvergenceError(
                    "fixed-point iterate overflowed"
                ) from error
            if any(not math.isfinite(value) for value in next_outgoing):
                raise ReturningRadiationConvergenceError(
                    "fixed-point iterate left the finite numerical domain"
                )
            if any(
                next_value < previous_value
                for next_value, previous_value in zip(next_outgoing, outgoing)
            ):
                raise ReturningRadiationConvergenceError(
                    "non-negative kernel violated monotone fixed-point iteration"
                )
            outgoing = next_outgoing
            incident = _matvec(
                checked_kernel.receiver_emitter_coefficients,
                outgoing,
                label="incident_returning_flux",
            )
            residual, tolerances = _residual_and_tolerances(
                source,
                incident,
                outgoing,
                checked_policy,
            )
            iterations = iteration
            if _residual_gate_passed(residual, tolerances):
                break
            increment = _matvec(
                checked_kernel.receiver_emitter_coefficients,
                increment,
                label="fixed_point_increment",
            )
        else:
            raise ReturningRadiationConvergenceError(
                "fixed-point residual did not converge within maximum_iterations"
            )

    return ReturningRadiationFluxSolution(
        intrinsic_flux=source,
        incident_returning_flux=incident,
        outgoing_flux=outgoing,
        equation_residual=residual,
        residual_tolerances=tolerances,
        iterations=iterations,
        monotonic_fixed_point_verified=True,
        kernel_descriptor_sha256=checked_kernel.canonical_descriptor_sha256,
        policy_descriptor_sha256=checked_policy.canonical_descriptor_sha256,
    )


def _require_exact_solution_schema(
    solution: ReturningRadiationFluxSolution,
) -> None:
    if type(solution) is not ReturningRadiationFluxSolution:
        raise TypeError("solution must be a ReturningRadiationFluxSolution")
    for name in (
        "intrinsic_flux",
        "incident_returning_flux",
        "outgoing_flux",
        "equation_residual",
        "residual_tolerances",
    ):
        values = object.__getattribute__(solution, name)
        if type(values) is not tuple or any(
            type(value) is not float for value in values
        ):
            raise ReturningRadiationVerificationError(
                f"solution.{name} must be an exact tuple of binary64 floats"
            )
    if type(object.__getattribute__(solution, "iterations")) is not int:
        raise ReturningRadiationVerificationError(
            "solution.iterations must be an exact integer"
        )
    if type(
        object.__getattribute__(solution, "monotonic_fixed_point_verified")
    ) is not bool:
        raise ReturningRadiationVerificationError(
            "solution.monotonic_fixed_point_verified must be an exact boolean"
        )
    for name in (
        "kernel_descriptor_sha256",
        "policy_descriptor_sha256",
    ):
        if type(object.__getattribute__(solution, name)) is not str:
            raise ReturningRadiationVerificationError(
                f"solution.{name} must be an exact string"
            )


def _same_float(left: float, right: float) -> bool:
    return left.hex().encode("ascii") == right.hex().encode("ascii")


def _same_float_tuple(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> bool:
    return len(left) == len(right) and all(
        _same_float(left_value, right_value)
        for left_value, right_value in zip(left, right)
    )


def _solution_fields_exactly_equal(
    left: ReturningRadiationFluxSolution,
    right: ReturningRadiationFluxSolution,
) -> bool:
    for name in (
        "intrinsic_flux",
        "incident_returning_flux",
        "outgoing_flux",
        "equation_residual",
        "residual_tolerances",
    ):
        if not _same_float_tuple(
            object.__getattribute__(left, name),
            object.__getattribute__(right, name),
        ):
            return False
    return (
        left.iterations == right.iterations
        and left.monotonic_fixed_point_verified
        is right.monotonic_fixed_point_verified
        and left.kernel_descriptor_sha256.encode("ascii")
        == right.kernel_descriptor_sha256.encode("ascii")
        and left.policy_descriptor_sha256.encode("ascii")
        == right.policy_descriptor_sha256.encode("ascii")
    )


def _validated_live_kernel(
    kernel: AxisymmetricReturningRadiationKernel,
) -> AxisymmetricReturningRadiationKernel:
    if type(kernel) is not AxisymmetricReturningRadiationKernel:
        raise TypeError("kernel must be an AxisymmetricReturningRadiationKernel")
    radii = object.__getattribute__(kernel, "annulus_radii_over_mass")
    rows = object.__getattribute__(kernel, "receiver_emitter_coefficients")
    producer = object.__getattribute__(kernel, "ray_kernel_producer_id")
    descriptor_json = object.__getattribute__(kernel, "_descriptor_json")
    descriptor_sha = object.__getattribute__(kernel, "_descriptor_sha256")
    if (
        type(radii) is not tuple
        or any(type(value) is not float for value in radii)
        or type(rows) is not tuple
        or any(
            type(row) is not tuple
            or any(type(value) is not float for value in row)
            for row in rows
        )
        or type(producer) is not str
        or type(descriptor_json) is not str
        or type(descriptor_sha) is not str
    ):
        raise ReturningRadiationVerificationError(
            "kernel live fields do not have the canonical exact schema"
        )
    try:
        rebuilt = AxisymmetricReturningRadiationKernel(
            annulus_radii_over_mass=radii,
            receiver_emitter_coefficients=rows,
            ray_kernel_producer_id=producer,
        )
    except (TypeError, ValueError) as error:
        raise ReturningRadiationVerificationError(
            f"kernel live fields cannot be reconstructed: {error}"
        ) from error
    if (
        not _same_float_tuple(radii, rebuilt.annulus_radii_over_mass)
        or len(rows) != len(rebuilt.receiver_emitter_coefficients)
        or any(
            not _same_float_tuple(left, right)
            for left, right in zip(rows, rebuilt.receiver_emitter_coefficients)
        )
        or producer.encode("utf-8")
        != rebuilt.ray_kernel_producer_id.encode("utf-8")
        or descriptor_json.encode("utf-8")
        != rebuilt.canonical_descriptor_json.encode("utf-8")
        or descriptor_sha.encode("ascii")
        != rebuilt.canonical_descriptor_sha256.encode("ascii")
    ):
        raise ReturningRadiationVerificationError(
            "kernel live fields or canonical descriptor are stale"
        )
    return rebuilt


def _validated_live_policy(
    policy: ReturningRadiationFixedPointPolicy,
) -> ReturningRadiationFixedPointPolicy:
    if type(policy) is not ReturningRadiationFixedPointPolicy:
        raise TypeError("policy must be a ReturningRadiationFixedPointPolicy")
    maximum_iterations = object.__getattribute__(policy, "maximum_iterations")
    absolute = object.__getattribute__(policy, "absolute_residual_tolerance")
    relative = object.__getattribute__(policy, "relative_residual_tolerance")
    if (
        type(maximum_iterations) is not int
        or type(absolute) is not float
        or type(relative) is not float
    ):
        raise ReturningRadiationVerificationError(
            "fixed-point policy live fields do not have exact canonical types"
        )
    try:
        rebuilt = ReturningRadiationFixedPointPolicy(
            maximum_iterations=maximum_iterations,
            absolute_residual_tolerance=absolute,
            relative_residual_tolerance=relative,
        )
    except (TypeError, ValueError) as error:
        raise ReturningRadiationVerificationError(
            f"fixed-point policy cannot be reconstructed: {error}"
        ) from error
    if (
        rebuilt.maximum_iterations != maximum_iterations
        or not _same_float(rebuilt.absolute_residual_tolerance, absolute)
        or not _same_float(rebuilt.relative_residual_tolerance, relative)
    ):
        raise ReturningRadiationVerificationError(
            "fixed-point policy live fields are not canonical"
        )
    return rebuilt


def validate_returning_radiation_solution(
    kernel: AxisymmetricReturningRadiationKernel,
    expected_intrinsic_flux: Sequence[float],
    solution: ReturningRadiationFluxSolution,
    policy: ReturningRadiationFixedPointPolicy = DEFAULT_FIXED_POINT_POLICY,
) -> None:
    """Validate a computed fixed point without rerunning its iteration.

    This is the cheap boundary for a caller that has just received ``solution``
    from :func:`solve_absorbed_returning_radiation`.  It independently binds
    the exact public schema, source, kernel, policy, incident flux, equation
    residual, and residual tolerances.  It does not claim deterministic replay
    of the iteration history; use :func:`verify_returning_radiation_solution`
    at an untrusted persistence boundary.
    """

    checked_kernel = _validated_live_kernel(kernel)
    checked_policy = _validated_live_policy(policy)
    _require_exact_solution_schema(solution)
    try:
        reconstructed = ReturningRadiationFluxSolution(
            intrinsic_flux=solution.intrinsic_flux,
            incident_returning_flux=solution.incident_returning_flux,
            outgoing_flux=solution.outgoing_flux,
            equation_residual=solution.equation_residual,
            residual_tolerances=solution.residual_tolerances,
            iterations=solution.iterations,
            monotonic_fixed_point_verified=(
                solution.monotonic_fixed_point_verified
            ),
            kernel_descriptor_sha256=solution.kernel_descriptor_sha256,
            policy_descriptor_sha256=solution.policy_descriptor_sha256,
        )
    except (TypeError, ValueError) as error:
        raise ReturningRadiationVerificationError(
            f"solution public fields are not canonical: {error}"
        ) from error
    if not _solution_fields_exactly_equal(solution, reconstructed):
        raise ReturningRadiationVerificationError(
            "solution public fields are not the canonical binary64 schema"
        )
    expected_source = _non_negative_vector(
        expected_intrinsic_flux,
        "expected_intrinsic_flux",
        expected_length=checked_kernel.annulus_count,
    )
    if solution.intrinsic_flux != expected_source:
        raise ReturningRadiationVerificationError(
            "solution intrinsic flux does not match the expected input"
        )
    if solution.kernel_descriptor_sha256.encode("ascii") != (
        checked_kernel.canonical_descriptor_sha256.encode("ascii")
    ):
        raise ReturningRadiationVerificationError(
            "solution kernel descriptor binding is stale"
        )
    if solution.policy_descriptor_sha256.encode("ascii") != (
        checked_policy.canonical_descriptor_sha256.encode("ascii")
    ):
        raise ReturningRadiationVerificationError(
            "solution fixed-point policy binding is stale"
        )
    if solution.monotonic_fixed_point_verified is not True:
        raise ReturningRadiationVerificationError(
            "solution does not certify monotonic fixed-point updates"
        )
    if (
        solution.iterations < 0
        or solution.iterations > checked_policy.maximum_iterations
    ):
        raise ReturningRadiationVerificationError(
            "solution iteration count is outside the fixed-point policy"
        )
    if any(
        outgoing < intrinsic
        for intrinsic, outgoing in zip(
            expected_source,
            solution.outgoing_flux,
        )
    ):
        raise ReturningRadiationVerificationError(
            "solution outgoing flux is below the intrinsic monotone lower bound"
        )
    try:
        expected_incident = _matvec(
            checked_kernel.receiver_emitter_coefficients,
            solution.outgoing_flux,
            label="verified_incident_returning_flux",
        )
        expected_residual, expected_tolerances = _residual_and_tolerances(
            expected_source,
            expected_incident,
            solution.outgoing_flux,
            checked_policy,
        )
    except ReturningRadiationConvergenceError as error:
        raise ReturningRadiationVerificationError(
            f"solution algebra could not be validated: {error}"
        ) from error
    if not _same_float_tuple(
        solution.incident_returning_flux,
        expected_incident,
    ):
        raise ReturningRadiationVerificationError(
            "solution incident flux is not K times its outgoing flux"
        )
    if not _same_float_tuple(solution.equation_residual, expected_residual):
        raise ReturningRadiationVerificationError(
            "solution equation residual is not the direct residual"
        )
    if not _same_float_tuple(
        solution.residual_tolerances,
        expected_tolerances,
    ):
        raise ReturningRadiationVerificationError(
            "solution residual tolerances do not match the fixed-point policy"
        )
    if not _residual_gate_passed(expected_residual, expected_tolerances):
        raise ReturningRadiationVerificationError(
            "solution does not pass the direct fixed-point equation gate"
        )
    if solution.iterations == 0 and not _same_float_tuple(
        solution.outgoing_flux,
        expected_source,
    ):
        raise ReturningRadiationVerificationError(
            "a zero-iteration solution must equal the intrinsic source"
        )


def verify_returning_radiation_solution(
    kernel: AxisymmetricReturningRadiationKernel,
    expected_intrinsic_flux: Sequence[float],
    solution: ReturningRadiationFluxSolution,
    policy: ReturningRadiationFixedPointPolicy = DEFAULT_FIXED_POINT_POLICY,
) -> None:
    """Reproduce a solution and reject changed inputs, outputs, or metadata."""

    validate_returning_radiation_solution(
        kernel,
        expected_intrinsic_flux,
        solution,
        policy,
    )
    expected_source = solution.intrinsic_flux
    try:
        reproduced = solve_absorbed_returning_radiation(
            kernel,
            expected_source,
            policy,
        )
    except (TypeError, ValueError, ReturningRadiationConvergenceError) as error:
        raise ReturningRadiationVerificationError(
            f"solution could not be reproduced: {error}"
        ) from error
    _require_exact_solution_schema(reproduced)
    if not _solution_fields_exactly_equal(solution, reproduced):
        raise ReturningRadiationVerificationError(
            "solution fields are not the deterministic fixed-point result"
        )


@dataclass(frozen=True, slots=True)
class PhotonFateProbabilityTriple:
    """Separate photon-number fate probabilities for one emitting annulus."""

    return_probability: float
    capture_probability: float
    escape_probability: float

    def __post_init__(self) -> None:
        values = tuple(
            _non_negative_number(getattr(self, name), name)
            for name in (
                "return_probability",
                "capture_probability",
                "escape_probability",
            )
        )
        if any(value > 1.0 for value in values):
            raise ValueError("photon fate probabilities may not exceed one")
        total = math.fsum(values)
        if abs(total - 1.0) > 8.0 * math.ulp(1.0):
            raise ValueError("photon fate probabilities must sum to one")
        object.__setattr__(self, "return_probability", values[0])
        object.__setattr__(self, "capture_probability", values[1])
        object.__setattr__(self, "escape_probability", values[2])

    @property
    def probability_sum(self) -> float:
        return math.fsum(
            (
                self.return_probability,
                self.capture_probability,
                self.escape_probability,
            )
        )


@dataclass(frozen=True, slots=True)
class AnnulusPhotonFateTable:
    """Photon-number fates, intentionally separate from local energy ``K``."""

    annulus_radii_over_mass: tuple[float, ...]
    probability_triples: tuple[PhotonFateProbabilityTriple, ...]
    ray_fate_producer_id: str
    _descriptor_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        radii = _positive_strictly_increasing_radii(
            self.annulus_radii_over_mass,
            "annulus_radii_over_mass",
        )
        producer = _non_empty_string(
            self.ray_fate_producer_id,
            "ray_fate_producer_id",
        )
        triples = _sequence(self.probability_triples, "probability_triples")
        if len(triples) != len(radii):
            raise ValueError("probability_triples must have one entry per annulus")
        if any(type(value) is not PhotonFateProbabilityTriple for value in triples):
            raise TypeError(
                "probability_triples must contain PhotonFateProbabilityTriple values"
            )
        descriptor = {
            "annulusRadiiOverMass": radii,
            "isLocalEnergyFluxKernel": False,
            "probabilityOrder": ("return", "capture", "escape"),
            "probabilityTriples": tuple(
                (
                    value.return_probability,
                    value.capture_probability,
                    value.escape_probability,
                )
                for value in triples
            ),
            "rayFateProducerId": producer,
        }
        object.__setattr__(self, "annulus_radii_over_mass", radii)
        object.__setattr__(self, "probability_triples", triples)
        object.__setattr__(self, "ray_fate_producer_id", producer)
        object.__setattr__(self, "_descriptor_json", _canonical_json(descriptor))

    @property
    def canonical_descriptor_json(self) -> str:
        return self._descriptor_json

    @property
    def canonical_descriptor_sha256(self) -> str:
        return hashlib.sha256(self._descriptor_json.encode("utf-8")).hexdigest()

    def canonical_descriptor(self) -> Mapping[str, Any]:
        return json.loads(self._descriptor_json)
