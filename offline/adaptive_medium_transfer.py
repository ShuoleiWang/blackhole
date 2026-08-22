"""Adaptive scalar radiative transfer along a certified recorded geodesic.

The geodesic is stored in observer-to-source order, but radiative transfer is
performed explicitly in source-to-observer order.  Every medium stencil state
inside an accepted ray segment is reconstructed by
:class:`offline.geodesic.CertifiedRecordedPathSampler`; no coordinate or
covector is linearly interpolated.

This is deliberately a scalar ``I_nu / nu**3`` reference integrator.  It
rejects Q/U/V emissivity, polarized boundaries, dichroism, and Faraday terms.
Its convergence evidence is finite-stencil whole-versus-halves step doubling.
It cannot prove the absence of an arbitrarily thin feature missed by every
sample, and it is not a claim of GRMHD plasma evolution or complete GRRT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from offline.geodesic import (
    CertifiedRecordedPathSampler,
    HamiltonianState,
    RayPathSegment,
    RayTraceResult,
    RecordedPathSamplingError,
    RecordedPathSamplingOptions,
    hamiltonian_null_residual,
)
from offline.pipeline import AffineMediumProvider, BoundarySpectrum
from offline.radiative_transfer import StokesInvariant, TransferCoefficients
from offline.spacetime import MetricProvider


FINITE_STENCIL_CAPABILITY = (
    "scalar-I_nu-only; certified-Hamiltonian affine sampling; "
    "finite-stencil intensity-and-optical-depth whole-vs-halves convergence; "
    "no arbitrary-subgrid "
    "completeness; no polarization; no GRMHD; no complete-GRRT"
)


class AdaptiveMediumTransferError(ValueError):
    """Base fail-closed error for adaptive scalar medium transfer."""


class AdaptiveMediumValidationError(AdaptiveMediumTransferError):
    """Raised when a ray, provider, boundary, or option violates the contract."""


class AdaptiveMediumIntegrationError(AdaptiveMediumTransferError):
    """Raised when finite scalar transfer cannot pass its convergence gate."""


class AdaptiveMediumBudgetExceeded(AdaptiveMediumIntegrationError):
    """Raised before declared work, depth, or minimum-step bounds are crossed."""


@dataclass(frozen=True)
class AdaptiveScalarTransferOptions:
    """Accuracy and hard work limits for one observer-frequency bin.

    Local error allowances are multiplied by the interval's fraction of the
    *whole recorded affine path*.  Accepted intervals partition that path, so
    arbitrary recorded-segment splitting cannot multiply the global absolute
    or relative tolerance.
    """

    absolute_tolerance: float = 1.0e-10
    relative_tolerance: float = 1.0e-5
    optical_depth_absolute_tolerance: float = 1.0e-10
    optical_depth_relative_tolerance: float = 1.0e-5
    maximum_coefficient_evaluations: int = 100_000
    maximum_refinement_depth: int = 24
    minimum_affine_step: float = 1.0e-12
    sampling: RecordedPathSamplingOptions = field(
        default_factory=RecordedPathSamplingOptions
    )

    def __post_init__(self) -> None:
        positive = (
            self.absolute_tolerance,
            self.relative_tolerance,
            self.optical_depth_absolute_tolerance,
            self.optical_depth_relative_tolerance,
            self.minimum_affine_step,
        )
        if any(
            isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0.0
            for value in positive
        ):
            raise ValueError("adaptive transfer tolerances and step must be positive")
        if (
            type(self.maximum_coefficient_evaluations) is not int
            or self.maximum_coefficient_evaluations < 1
        ):
            raise ValueError(
                "maximum_coefficient_evaluations must be a positive integer"
            )
        if (
            type(self.maximum_refinement_depth) is not int
            or self.maximum_refinement_depth < 0
        ):
            raise ValueError(
                "maximum_refinement_depth must be a non-negative integer"
            )
        if not isinstance(self.sampling, RecordedPathSamplingOptions):
            raise TypeError("sampling must be a RecordedPathSamplingOptions")


@dataclass(frozen=True)
class AdaptiveScalarTransferDiagnostics:
    """Deterministic convergence and work evidence for a completed transfer."""

    ordering: str
    capability: str
    metric_source_id: str
    medium_source_id: str
    boundary_source_id: str
    recorded_segment_count: int
    total_affine_length: float
    accepted_intervals: int
    refined_intervals: int
    coefficient_evaluations: int
    geodesic_reintegrations: int
    maximum_refinement_depth: int
    minimum_accepted_affine_step: float
    estimated_global_absolute_error: float
    estimated_global_optical_depth_error: float
    error_scale_invariant_intensity: float
    global_error_limit: float
    optical_depth_global_error_limit: float
    maximum_local_error_norm: float
    maximum_local_optical_depth_error_norm: float
    scalar_optical_depth: float
    maximum_null_residual: float
    maximum_metric_interpolation_error: float


@dataclass(frozen=True)
class AdaptiveScalarTransferResult:
    """Observer-side invariant scalar intensity and its convergence evidence."""

    observer_frequency_hz: float
    source_invariant_intensity: float
    observer_invariant_intensity: float
    diagnostics: AdaptiveScalarTransferDiagnostics


@dataclass(frozen=True)
class _ScalarCoefficients:
    emissivity: float
    absorption: float


@dataclass(frozen=True)
class _IntervalResult:
    intensity: float
    estimated_error: float
    estimated_optical_depth_error: float
    optical_depth: float
    accepted_intervals: int
    maximum_depth: int
    minimum_step: float
    maximum_local_error_norm: float
    maximum_local_optical_depth_error_norm: float
    maximum_radiance_scale: float


def _checked_finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise AdaptiveMediumValidationError(f"{label} must be a finite float")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AdaptiveMediumValidationError(
            f"{label} must be a finite float"
        ) from error
    if not math.isfinite(result):
        raise AdaptiveMediumValidationError(f"{label} must be finite")
    return result


def _checked_add(first: float, second: float, label: str) -> float:
    result = first + second
    if not math.isfinite(result):
        raise AdaptiveMediumIntegrationError(f"{label} overflowed binary64")
    return result


def _scalar_slab(
    incoming: float,
    coefficients: _ScalarCoefficients,
    length: float,
) -> tuple[float, float]:
    """Apply the exact constant scalar slab and return ``(I_out, tau)``."""

    optical_depth = coefficients.absorption * length
    if not math.isfinite(optical_depth):
        raise AdaptiveMediumIntegrationError("scalar optical depth is non-finite")
    if coefficients.absorption == 0.0 or optical_depth == 0.0:
        emitted = coefficients.emissivity * length
        if not math.isfinite(emitted):
            raise AdaptiveMediumIntegrationError(
                "scalar emission increment is non-finite"
            )
        result = incoming + emitted
    else:
        attenuation = math.exp(-optical_depth)
        emission_weight = -math.expm1(-optical_depth) / coefficients.absorption
        result = incoming * attenuation + coefficients.emissivity * emission_weight
    if not math.isfinite(result) or result < 0.0:
        raise AdaptiveMediumIntegrationError(
            "scalar slab produced a non-finite or negative intensity"
        )
    return result, optical_depth


def _validate_ray(ray: RayTraceResult) -> tuple[RayPathSegment, ...]:
    if type(ray) is not RayTraceResult:
        raise TypeError("ray must be the exact RayTraceResult")
    if type(ray.outcome) is not str:
        raise AdaptiveMediumValidationError("ray outcome must be an exact string")
    if type(ray.terminal_state) is not HamiltonianState:
        raise AdaptiveMediumValidationError(
            "ray terminal_state must be the exact HamiltonianState"
        )
    for value, label in (
        (ray.accepted_steps, "ray accepted_steps"),
        (ray.rejected_steps, "ray rejected_steps"),
    ):
        if type(value) is not int or value < 0:
            raise AdaptiveMediumValidationError(
                f"{label} must be a non-negative exact integer"
            )
    maximum_null_residual = _checked_finite(
        ray.maximum_null_residual,
        "ray maximum_null_residual",
    )
    maximum_metric_error = _checked_finite(
        ray.maximum_metric_interpolation_error,
        "ray maximum_metric_interpolation_error",
    )
    if maximum_null_residual < 0.0 or maximum_metric_error < 0.0:
        raise AdaptiveMediumValidationError(
            "ray numerical diagnostics must be non-negative"
        )
    if ray.outcome not in {"captured", "escaped", "completed"}:
        raise AdaptiveMediumValidationError(
            f"ray outcome {ray.outcome!r} is not usable for transfer"
        )
    if ray.failure_reason is not None:
        raise AdaptiveMediumValidationError("failed ray may not enter transfer")
    if ray.outcome in {"captured", "escaped"} and (
        type(ray.terminal_target_id) is not str or not ray.terminal_target_id
    ):
        raise AdaptiveMediumValidationError(
            "terminal ray outcome requires an exact non-empty target id"
        )
    affine_length = _checked_finite(ray.affine_length, "ray affine_length")
    if affine_length < 0.0:
        raise AdaptiveMediumValidationError(
            "ray affine length must be finite and non-negative"
        )
    if type(ray.segments) is not tuple:
        raise AdaptiveMediumValidationError("ray segments must be an exact tuple")
    segments = ray.segments
    if ray.affine_length > 0.0 and not segments:
        raise AdaptiveMediumValidationError(
            "ray path was not recorded; trace with record_path=True"
        )
    previous_end: HamiltonianState | None = None
    for index, segment in enumerate(segments):
        if type(segment) is not RayPathSegment:
            raise AdaptiveMediumValidationError(
                f"ray.segments[{index}] must be the exact RayPathSegment"
            )
        for state, state_label in (
            (segment.start, "start"),
            (segment.midpoint, "midpoint"),
            (segment.end, "end"),
        ):
            if type(state) is not HamiltonianState:
                raise AdaptiveMediumValidationError(
                    f"ray.segments[{index}].{state_label} must be the exact "
                    "HamiltonianState"
                )
        segment_length = _checked_finite(
            segment.affine_length,
            f"ray.segments[{index}].affine_length",
        )
        if segment_length <= 0.0:
            raise AdaptiveMediumValidationError(
                f"ray.segments[{index}] must have positive finite length"
            )
        midpoint_null = _checked_finite(
            segment.midpoint_null_residual,
            f"ray.segments[{index}].midpoint_null_residual",
        )
        if midpoint_null < 0.0:
            raise AdaptiveMediumValidationError(
                f"ray.segments[{index}] midpoint null residual must be non-negative"
            )
        if previous_end is not None and segment.start != previous_end:
            raise AdaptiveMediumValidationError(
                "recorded ray segments are not contiguous"
            )
        previous_end = segment.end
    if segments and segments[-1].end != ray.terminal_state:
        raise AdaptiveMediumValidationError(
            "recorded ray does not end at its terminal state"
        )
    recorded_length = math.fsum(segment.affine_length for segment in segments)
    if not math.isfinite(recorded_length) or not math.isclose(
        recorded_length,
        ray.affine_length,
        rel_tol=2.0e-13,
        abs_tol=2.0e-13,
    ):
        raise AdaptiveMediumValidationError(
            "recorded ray segments do not cover its affine length"
        )
    if ray.accepted_steps != len(segments):
        raise AdaptiveMediumValidationError(
            "recorded ray segment count does not match accepted steps"
        )
    if not segments and ray.rejected_steps != 0:
        raise AdaptiveMediumValidationError(
            "zero-length ray rejected_steps must be exactly zero"
        )
    return segments


def _validate_scalar_stokes(stokes: StokesInvariant, label: str) -> float:
    if not isinstance(stokes, StokesInvariant):
        raise AdaptiveMediumValidationError(f"{label} must be a StokesInvariant")
    values = tuple(
        _checked_finite(value, f"{label}[{index}]")
        for index, value in enumerate(stokes.as_tuple())
    )
    if values[0] < 0.0:
        raise AdaptiveMediumValidationError(f"{label} intensity must be non-negative")
    if values[1] != 0.0 or values[2] != 0.0 or values[3] != 0.0:
        raise AdaptiveMediumValidationError(
            f"{label} polarization requires a transported screen basis"
        )
    return values[0]


def _validate_scalar_coefficients(
    coefficients: TransferCoefficients,
) -> _ScalarCoefficients:
    if not isinstance(coefficients, TransferCoefficients):
        raise AdaptiveMediumValidationError(
            "medium must return TransferCoefficients"
        )
    emissivity = _validate_scalar_stokes(
        coefficients.invariant_emissivity,
        "medium invariant emissivity",
    )
    absorption = _checked_finite(
        coefficients.invariant_absorption,
        "medium invariant absorption",
    )
    if absorption < 0.0:
        raise AdaptiveMediumValidationError(
            "medium invariant absorption must be non-negative"
        )
    if (
        coefficients.invariant_dichroism != (0.0, 0.0, 0.0)
        or coefficients.invariant_faraday != (0.0, 0.0, 0.0)
    ):
        raise AdaptiveMediumValidationError(
            "adaptive scalar transfer forbids dichroism and Faraday terms"
        )
    return _ScalarCoefficients(emissivity, absorption)


def propagate_adaptive_scalar_recorded_ray(
    ray: RayTraceResult,
    provider: MetricProvider,
    medium: AffineMediumProvider,
    boundary: BoundarySpectrum,
    observer_frequency_hz: float,
    *,
    options: AdaptiveScalarTransferOptions = AdaptiveScalarTransferOptions(),
) -> AdaptiveScalarTransferResult:
    """Propagate one scalar invariant-intensity bin source-to-observer.

    For each recorded observer-to-source segment, one midpoint slab is compared
    with two half slabs sampled at the physical affine quarter points.  Failed
    intervals recurse in physical propagation order.  The Richardson estimate
    of every accepted interval is charged a whole-path affine-weighted share of
    one global absolute/relative error budget.
    """

    if not isinstance(options, AdaptiveScalarTransferOptions):
        raise TypeError("options must be an AdaptiveScalarTransferOptions")
    if not isinstance(provider, MetricProvider):
        raise TypeError("provider must implement MetricProvider")
    frequency = _checked_finite(observer_frequency_hz, "observer_frequency_hz")
    if frequency <= 0.0:
        raise AdaptiveMediumValidationError(
            "observer_frequency_hz must be positive"
        )
    if not isinstance(medium, AffineMediumProvider):
        raise TypeError("medium must implement AffineMediumProvider")
    if not isinstance(boundary, BoundarySpectrum):
        raise TypeError("boundary must implement BoundarySpectrum")
    metric_source_id = getattr(provider, "source_id", None)
    medium_source_id = getattr(medium, "source_id", None)
    boundary_source_id = getattr(boundary, "source_id", None)
    for value, label in (
        (metric_source_id, "metric source_id"),
        (medium_source_id, "medium source_id"),
        (boundary_source_id, "boundary source_id"),
    ):
        if not isinstance(value, str) or not value:
            raise AdaptiveMediumValidationError(f"{label} must be non-empty")

    segments = _validate_ray(ray)
    ray_maximum_null = _checked_finite(
        ray.maximum_null_residual,
        "ray maximum_null_residual",
    )
    ray_maximum_metric_error = _checked_finite(
        ray.maximum_metric_interpolation_error,
        "ray maximum_metric_interpolation_error",
    )
    if ray_maximum_null > options.sampling.null_residual_limit:
        raise AdaptiveMediumValidationError(
            "ray maximum null residual exceeds the sampling limit"
        )
    if (
        ray_maximum_metric_error
        > options.sampling.metric_interpolation_error_limit
    ):
        raise AdaptiveMediumValidationError(
            "ray metric interpolation error exceeds the sampling limit"
        )
    sampler = CertifiedRecordedPathSampler(provider, options.sampling)
    certified_states: dict[tuple[int, float], HamiltonianState] = {}
    try:
        for index, segment in enumerate(segments):
            sampler.certify_segment(segment, label=f"ray.segments[{index}]")
            certified_states[(index, 0.0)] = segment.start
            certified_states[(index, 0.5)] = segment.midpoint
            certified_states[(index, 1.0)] = segment.end
    except RecordedPathSamplingError as error:
        raise AdaptiveMediumIntegrationError(
            f"recorded path failed Hamiltonian certification: {error}"
        ) from error

    empty_path_null_residual = 0.0
    empty_path_metric_error = 0.0
    if not segments:
        try:
            terminal_sample = provider.sample(ray.terminal_state.event)
            empty_path_null_residual = hamiltonian_null_residual(
                provider,
                ray.terminal_state,
            )
            empty_path_metric_error = float(terminal_sample.interpolation_error)
        except (
            ArithmeticError,
            IndexError,
            TypeError,
            ValueError,
            OverflowError,
        ) as error:
            raise AdaptiveMediumValidationError(
                f"zero-length ray terminal state could not be authenticated: {error}"
            ) from error
        if (
            not math.isfinite(empty_path_null_residual)
            or empty_path_null_residual > options.sampling.null_residual_limit
        ):
            raise AdaptiveMediumValidationError(
                "zero-length ray terminal state exceeds the null-residual limit"
            )
        if (
            not math.isfinite(empty_path_metric_error)
            or empty_path_metric_error
            > options.sampling.metric_interpolation_error_limit
        ):
            raise AdaptiveMediumValidationError(
                "zero-length ray terminal metric exceeds the interpolation limit"
            )
        if (
            ray_maximum_null.hex() != empty_path_null_residual.hex()
            or ray_maximum_metric_error.hex() != empty_path_metric_error.hex()
        ):
            raise AdaptiveMediumValidationError(
                "zero-length ray diagnostics disagree with current metric authentication"
            )

    if ray.outcome == "captured":
        source_intensity = 0.0
    else:
        try:
            source = boundary.invariant_stokes(ray.terminal_state, frequency)
        except Exception as error:
            raise AdaptiveMediumValidationError(
                f"boundary spectrum evaluation failed: {error}"
            ) from error
        source_intensity = _validate_scalar_stokes(source, "boundary spectrum")

    if not segments:
        sampling_diagnostics = sampler.diagnostics
        global_error_limit = (
            options.absolute_tolerance
            + options.relative_tolerance * source_intensity
        )
        optical_depth_global_error_limit = (
            options.optical_depth_absolute_tolerance
        )
        if not math.isfinite(global_error_limit):
            raise AdaptiveMediumIntegrationError(
                "global scalar-transfer error budget is non-finite"
            )
        diagnostics = AdaptiveScalarTransferDiagnostics(
            ordering="source-to-observer",
            capability=FINITE_STENCIL_CAPABILITY,
            metric_source_id=metric_source_id,
            medium_source_id=medium_source_id,
            boundary_source_id=boundary_source_id,
            recorded_segment_count=0,
            total_affine_length=0.0,
            accepted_intervals=0,
            refined_intervals=0,
            coefficient_evaluations=0,
            geodesic_reintegrations=sampling_diagnostics.reintegrations,
            maximum_refinement_depth=0,
            minimum_accepted_affine_step=0.0,
            estimated_global_absolute_error=0.0,
            estimated_global_optical_depth_error=0.0,
            error_scale_invariant_intensity=source_intensity,
            global_error_limit=global_error_limit,
            optical_depth_global_error_limit=(
                optical_depth_global_error_limit
            ),
            maximum_local_error_norm=0.0,
            maximum_local_optical_depth_error_norm=0.0,
            scalar_optical_depth=0.0,
            maximum_null_residual=max(
                ray_maximum_null,
                empty_path_null_residual,
                sampling_diagnostics.maximum_null_residual,
            ),
            maximum_metric_interpolation_error=max(
                ray_maximum_metric_error,
                empty_path_metric_error,
                sampling_diagnostics.maximum_metric_interpolation_error
            ),
        )
        return AdaptiveScalarTransferResult(
            observer_frequency_hz=frequency,
            source_invariant_intensity=source_intensity,
            observer_invariant_intensity=source_intensity,
            diagnostics=diagnostics,
        )

    total_length = math.fsum(segment.affine_length for segment in segments)
    coefficient_cache: dict[tuple[int, float], _ScalarCoefficients] = {}
    coefficient_evaluations = 0
    refined_intervals = 0

    def coefficients_at(index: int, fraction: float) -> _ScalarCoefficients:
        nonlocal coefficient_evaluations
        key = (index, fraction)
        cached = coefficient_cache.get(key)
        if cached is not None:
            return cached
        if coefficient_evaluations >= options.maximum_coefficient_evaluations:
            raise AdaptiveMediumBudgetExceeded(
                "medium coefficient-evaluation budget exhausted"
            )
        state = certified_states.get(key)
        if state is None:
            try:
                state = sampler.sample(
                    segments[index],
                    fraction,
                    label=f"ray.segments[{index}] affine fraction {fraction:.17g}",
                )
            except RecordedPathSamplingError as error:
                raise AdaptiveMediumIntegrationError(
                    f"Hamiltonian medium sampling failed: {error}"
                ) from error
            certified_states[key] = state
        coefficient_evaluations += 1
        try:
            raw = medium.coefficients(state, frequency)
        except Exception as error:
            raise AdaptiveMediumValidationError(
                f"medium coefficient evaluation failed: {error}"
            ) from error
        result = _validate_scalar_coefficients(raw)
        coefficient_cache[key] = result
        return result

    def integrate_interval(
        segment_index: int,
        lower_fraction: float,
        upper_fraction: float,
        incoming: float,
        depth: int,
        force_refinement: bool = False,
    ) -> _IntervalResult:
        nonlocal refined_intervals
        segment = segments[segment_index]
        fraction_width = upper_fraction - lower_fraction
        length = segment.affine_length * fraction_width
        if not math.isfinite(length) or length <= 0.0:
            raise AdaptiveMediumIntegrationError(
                "adaptive interval has invalid affine length"
            )
        middle = 0.5 * (lower_fraction + upper_fraction)
        observer_quarter = 0.5 * (lower_fraction + middle)
        source_quarter = 0.5 * (middle + upper_fraction)

        whole_coefficients = coefficients_at(segment_index, middle)
        source_coefficients = coefficients_at(segment_index, source_quarter)
        observer_coefficients = coefficients_at(
            segment_index,
            observer_quarter,
        )
        whole, whole_tau = _scalar_slab(
            incoming,
            whole_coefficients,
            length,
        )
        source_half, source_tau = _scalar_slab(
            incoming,
            source_coefficients,
            0.5 * length,
        )
        fine, observer_tau = _scalar_slab(
            source_half,
            observer_coefficients,
            0.5 * length,
        )
        fine_tau = _checked_add(
            source_tau,
            observer_tau,
            "fine scalar optical depth",
        )
        estimated_error = abs(fine - whole) / 3.0
        if not math.isfinite(estimated_error):
            raise AdaptiveMediumIntegrationError(
                "local transfer error estimate is non-finite"
            )
        radiance_scale = max(incoming, whole, source_half, fine)
        path_fraction = length / total_length
        local_limit = path_fraction * (
            options.absolute_tolerance
            + options.relative_tolerance * radiance_scale
        )
        if not math.isfinite(local_limit) or local_limit <= 0.0:
            raise AdaptiveMediumIntegrationError(
                "local transfer error budget is invalid"
            )
        local_error_norm = estimated_error / local_limit
        if not math.isfinite(local_error_norm):
            raise AdaptiveMediumIntegrationError(
                "local transfer convergence norm is non-finite"
            )
        estimated_optical_depth_error = abs(fine_tau - whole_tau) / 3.0
        optical_depth_scale = max(whole_tau, fine_tau)
        optical_depth_limit = (
            path_fraction * options.optical_depth_absolute_tolerance
            + options.optical_depth_relative_tolerance * optical_depth_scale
        )
        if (
            not math.isfinite(estimated_optical_depth_error)
            or not math.isfinite(optical_depth_limit)
            or optical_depth_limit <= 0.0
        ):
            raise AdaptiveMediumIntegrationError(
                "local optical-depth error budget is invalid"
            )
        optical_depth_error_norm = (
            estimated_optical_depth_error / optical_depth_limit
        )
        if not math.isfinite(optical_depth_error_norm):
            raise AdaptiveMediumIntegrationError(
                "local optical-depth convergence norm is non-finite"
            )
        needs_refinement = (
            local_error_norm > 1.0 or optical_depth_error_norm > 1.0
        )
        if not needs_refinement and not force_refinement:
            return _IntervalResult(
                intensity=fine,
                estimated_error=estimated_error,
                estimated_optical_depth_error=(
                    estimated_optical_depth_error
                ),
                optical_depth=fine_tau,
                accepted_intervals=1,
                maximum_depth=depth,
                minimum_step=length,
                maximum_local_error_norm=local_error_norm,
                maximum_local_optical_depth_error_norm=(
                    optical_depth_error_norm
                ),
                maximum_radiance_scale=radiance_scale,
            )

        if depth >= options.maximum_refinement_depth:
            raise AdaptiveMediumBudgetExceeded(
                "adaptive transfer refinement-depth budget exhausted"
            )
        half_length = 0.5 * length
        if (
            not math.isfinite(half_length)
            or half_length < options.minimum_affine_step
            or half_length == 0.0
            or half_length == length
        ):
            raise AdaptiveMediumBudgetExceeded(
                "adaptive transfer reached its minimum affine step"
            )
        refined_intervals += 1
        source_result = integrate_interval(
            segment_index,
            middle,
            upper_fraction,
            incoming,
            depth + 1,
            needs_refinement,
        )
        observer_result = integrate_interval(
            segment_index,
            lower_fraction,
            middle,
            source_result.intensity,
            depth + 1,
            needs_refinement,
        )
        return _IntervalResult(
            intensity=observer_result.intensity,
            estimated_error=_checked_add(
                source_result.estimated_error,
                observer_result.estimated_error,
                "global transfer error estimate",
            ),
            estimated_optical_depth_error=_checked_add(
                source_result.estimated_optical_depth_error,
                observer_result.estimated_optical_depth_error,
                "global optical-depth error estimate",
            ),
            optical_depth=_checked_add(
                source_result.optical_depth,
                observer_result.optical_depth,
                "scalar optical depth",
            ),
            accepted_intervals=(
                source_result.accepted_intervals
                + observer_result.accepted_intervals
            ),
            maximum_depth=max(
                source_result.maximum_depth,
                observer_result.maximum_depth,
            ),
            minimum_step=min(
                source_result.minimum_step,
                observer_result.minimum_step,
            ),
            maximum_local_error_norm=max(
                source_result.maximum_local_error_norm,
                observer_result.maximum_local_error_norm,
            ),
            maximum_local_optical_depth_error_norm=max(
                source_result.maximum_local_optical_depth_error_norm,
                observer_result.maximum_local_optical_depth_error_norm,
            ),
            maximum_radiance_scale=max(
                source_result.maximum_radiance_scale,
                observer_result.maximum_radiance_scale,
            ),
        )

    intensity = source_intensity
    segment_results: list[_IntervalResult] = []
    for segment_index in range(len(segments) - 1, -1, -1):
        segment_result = integrate_interval(
            segment_index,
            0.0,
            1.0,
            intensity,
            0,
        )
        segment_results.append(segment_result)
        intensity = segment_result.intensity

    estimated_global_error = math.fsum(
        result.estimated_error for result in segment_results
    )
    estimated_global_optical_depth_error = math.fsum(
        result.estimated_optical_depth_error for result in segment_results
    )
    optical_depth = math.fsum(result.optical_depth for result in segment_results)
    maximum_radiance_scale = max(
        source_intensity,
        intensity,
        *(result.maximum_radiance_scale for result in segment_results),
    )
    global_error_limit = (
        options.absolute_tolerance
        + options.relative_tolerance * maximum_radiance_scale
    )
    optical_depth_global_error_limit = (
        options.optical_depth_absolute_tolerance
        + options.optical_depth_relative_tolerance * optical_depth
    )
    if (
        not math.isfinite(estimated_global_error)
        or not math.isfinite(global_error_limit)
        or estimated_global_error > global_error_limit * (1.0 + 8.0e-15)
    ):
        raise AdaptiveMediumIntegrationError(
            "global scalar-transfer convergence budget was not satisfied"
        )
    if (
        not math.isfinite(estimated_global_optical_depth_error)
        or not math.isfinite(optical_depth)
        or not math.isfinite(optical_depth_global_error_limit)
        or estimated_global_optical_depth_error
        > optical_depth_global_error_limit * (1.0 + 8.0e-15)
    ):
        raise AdaptiveMediumIntegrationError(
            "global optical-depth convergence budget was not satisfied"
        )

    sampling_diagnostics = sampler.diagnostics
    diagnostics = AdaptiveScalarTransferDiagnostics(
        ordering="source-to-observer",
        capability=FINITE_STENCIL_CAPABILITY,
        metric_source_id=metric_source_id,
        medium_source_id=medium_source_id,
        boundary_source_id=boundary_source_id,
        recorded_segment_count=len(segments),
        total_affine_length=total_length,
        accepted_intervals=sum(
            result.accepted_intervals for result in segment_results
        ),
        refined_intervals=refined_intervals,
        coefficient_evaluations=coefficient_evaluations,
        geodesic_reintegrations=sampling_diagnostics.reintegrations,
        maximum_refinement_depth=max(
            result.maximum_depth for result in segment_results
        ),
        minimum_accepted_affine_step=min(
            result.minimum_step for result in segment_results
        ),
        estimated_global_absolute_error=estimated_global_error,
        estimated_global_optical_depth_error=(
            estimated_global_optical_depth_error
        ),
        error_scale_invariant_intensity=maximum_radiance_scale,
        global_error_limit=global_error_limit,
        optical_depth_global_error_limit=(
            optical_depth_global_error_limit
        ),
        maximum_local_error_norm=max(
            result.maximum_local_error_norm for result in segment_results
        ),
        maximum_local_optical_depth_error_norm=max(
            result.maximum_local_optical_depth_error_norm
            for result in segment_results
        ),
        scalar_optical_depth=optical_depth,
        maximum_null_residual=max(
            ray_maximum_null,
            sampling_diagnostics.maximum_null_residual,
        ),
        maximum_metric_interpolation_error=max(
            ray_maximum_metric_error,
            sampling_diagnostics.maximum_metric_interpolation_error
        ),
    )
    return AdaptiveScalarTransferResult(
        observer_frequency_hz=frequency,
        source_invariant_intensity=source_intensity,
        observer_invariant_intensity=intensity,
        diagnostics=diagnostics,
    )


__all__ = (
    "AdaptiveMediumBudgetExceeded",
    "AdaptiveMediumIntegrationError",
    "AdaptiveMediumTransferError",
    "AdaptiveMediumValidationError",
    "AdaptiveScalarTransferDiagnostics",
    "AdaptiveScalarTransferOptions",
    "AdaptiveScalarTransferResult",
    "FINITE_STENCIL_CAPABILITY",
    "propagate_adaptive_scalar_recorded_ray",
)
