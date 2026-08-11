"""First-visible-surface transfer for an exact-Kerr Novikov--Thorne disk.

The generic geodesic integrator owns accepted-step Hamiltonian localization.
This adapter owns the product rule: equatorial crossings inside the ISCO and
outside the declared outer edge are transparent, while the first crossing in
the finite Novikov--Thorne annulus is opaque and terminates the visible ray.

The attached surface trace proves agreement of declared ``N`` and ``2N`` probe
grids on the visible prefix.  It is not described as mathematically
surface-complete, because no finite grid can exclude arbitrary oscillatory
even root pairs.  Transfer remains scalar ``I_nu`` only: no polarization,
returning radiation, solved atmosphere, or GRMHD claim is made here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, cast

from offline.disk_atmosphere import (
    AngularEmissionLaw,
    equatorial_emission_angle_cosine,
)
from offline.geodesic import (
    ClassifiedInteriorSurfaceCrossing,
    HamiltonianState,
    InteriorSurfaceDecision,
    RayTraceResult,
    RecordedSurfaceCrossing,
    SurfaceEventOptions,
)
from offline.kerr import kerr_ks_event_to_oblate
from offline.kerr_disk import (
    KerrDiskEmitter,
    StationaryNovikovThorneDisk,
    observer_to_emitter_frequency_shift_g,
)
from offline.kerr_disk_transfer import (
    EscapedObserverSpecificIntensity,
    KerrDiskCrossingRegion,
    KerrDiskCrossingSignatureEntry,
    KerrDiskEmissionPayload,
    KerrDiskSourceKind,
)


KERR_DISK_OPAQUE_HIT_OUTCOME = "opaque-disk-hit"
KERR_DISK_OPAQUE_HIT_TARGET_ID = "exact-kerr-nt-opaque-annulus"
KERR_DISK_SURFACE_RESOLUTION_GATE = "accepted-step-N-vs-2N-visible-prefix"


class KerrDiskEarlyStopError(RuntimeError):
    """Raised when a first-visible-surface ray cannot be certified."""


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _positive_frequencies(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("observer_frequencies_hz must be a sequence")
    try:
        frequencies = tuple(
            _finite_number(value, f"observer_frequencies_hz[{index}]")
            for index, value in enumerate(values)
        )
    except TypeError as error:
        raise ValueError("observer_frequencies_hz must be a sequence") from error
    if not frequencies or any(value <= 0.0 for value in frequencies):
        raise ValueError("observer frequencies must be non-empty and positive")
    if any(right <= left for left, right in zip(frequencies, frequencies[1:])):
        raise ValueError("observer frequencies must be strictly increasing")
    return frequencies


def _finite_four_velocity(
    values: Sequence[float],
) -> tuple[float, float, float, float]:
    if isinstance(values, (str, bytes)):
        raise ValueError("observer_four_velocity must contain four finite numbers")
    try:
        entries = tuple(values)
    except TypeError as error:
        raise ValueError(
            "observer_four_velocity must contain four finite numbers"
        ) from error
    if len(entries) != 4:
        raise ValueError("observer_four_velocity must contain four finite numbers")
    return tuple(  # type: ignore[return-value]
        _finite_number(value, f"observer_four_velocity[{index}]")
        for index, value in enumerate(entries)
    )


@dataclass(frozen=True, slots=True)
class KerrDiskAnnulusSurface:
    """Product classifier for the equatorial finite opaque disk annulus."""

    disk: StationaryNovikovThorneDisk
    outer_radius_m: float
    opaque_outcome: str = KERR_DISK_OPAQUE_HIT_OUTCOME
    opaque_target_id: str = KERR_DISK_OPAQUE_HIT_TARGET_ID

    def __post_init__(self) -> None:
        if not isinstance(self.disk, StationaryNovikovThorneDisk):
            raise TypeError("disk must be a StationaryNovikovThorneDisk")
        outer = _finite_number(self.outer_radius_m, "outer_radius_m")
        if outer < self.disk.isco_radius_m:
            raise ValueError("outer_radius_m must be at or outside the disk ISCO")
        if not isinstance(self.opaque_outcome, str) or not self.opaque_outcome:
            raise ValueError("opaque_outcome must be non-empty")
        if not isinstance(self.opaque_target_id, str) or not self.opaque_target_id:
            raise ValueError("opaque_target_id must be non-empty")
        object.__setattr__(self, "outer_radius_m", outer)

    def value(self, state: HamiltonianState) -> float:
        if not isinstance(state, HamiltonianState):
            raise TypeError("surface state must be a HamiltonianState")
        return state.event[3]

    def classify(
        self,
        crossing: RecordedSurfaceCrossing,
    ) -> InteriorSurfaceDecision:
        if not isinstance(crossing, RecordedSurfaceCrossing):
            raise TypeError("crossing must be a RecordedSurfaceCrossing")
        radius = kerr_ks_event_to_oblate(
            self.disk.metric,
            crossing.state.event,
        ).radius_m
        if radius < self.disk.isco_radius_m:
            return InteriorSurfaceDecision("inside-isco")
        if radius > self.outer_radius_m:
            return InteriorSurfaceDecision("outside-outer-radius")
        return InteriorSurfaceDecision(
            "opaque-annulus",
            self.opaque_outcome,
            self.opaque_target_id,
        )


@dataclass(frozen=True, slots=True)
class KerrDiskVisibleSpectrumResult:
    """Scalar spectrum from an early disk hit or a reached finite boundary."""

    observer_frequencies_hz: tuple[float, ...]
    observed_specific_intensities_nu: tuple[float, ...]
    source_kind: KerrDiskSourceKind
    ray_boundary_outcome: Literal["captured", "escaped"] | None
    ray_boundary_target_id: str | None
    crossing_signature: tuple[KerrDiskCrossingSignatureEntry, ...]
    crossing_bracket_affine_widths: tuple[float, ...]
    first_opaque_crossing_index: int | None
    terminated_at_opaque_disk: bool
    crossing: RecordedSurfaceCrossing | None = None
    disk_radius_m: float | None = None
    emitter: KerrDiskEmitter | None = None
    frequency_shift_g: float | None = None
    emitted_frequencies_hz: tuple[float, ...] | None = None
    isotropic_emitted_specific_intensities_nu: tuple[float, ...] | None = None
    emitted_specific_intensities_nu: tuple[float, ...] | None = None
    emission_angle_cosine: float | None = None
    angular_emission_multiplier: float | None = None
    emitter_event_tolerance_m: float | None = None
    surface_resolution_gate: str = field(
        default=KERR_DISK_SURFACE_RESOLUTION_GATE,
        init=False,
    )

    def __post_init__(self) -> None:
        frequencies = _positive_frequencies(self.observer_frequencies_hz)
        intensities = tuple(
            _finite_number(value, f"observed_specific_intensities_nu[{index}]")
            for index, value in enumerate(self.observed_specific_intensities_nu)
        )
        if len(intensities) != len(frequencies) or any(
            value < 0.0 for value in intensities
        ):
            raise ValueError("observed spectrum is malformed")
        signature = tuple(self.crossing_signature)
        widths = tuple(
            _finite_number(value, f"crossing_bracket_affine_widths[{index}]")
            for index, value in enumerate(self.crossing_bracket_affine_widths)
        )
        if len(signature) != len(widths) or any(value < 0.0 for value in widths):
            raise ValueError("crossing signature diagnostics are malformed")
        if any(
            not isinstance(entry, KerrDiskCrossingSignatureEntry)
            for entry in signature
        ):
            raise TypeError("crossing signature contains an invalid entry")
        opaque_indices = tuple(
            index
            for index, entry in enumerate(signature)
            if entry.radial_region == "opaque-annulus"
        )
        expected_opaque = opaque_indices[0] if opaque_indices else None
        if self.first_opaque_crossing_index != expected_opaque:
            raise ValueError("first opaque crossing index is inconsistent")
        if not isinstance(self.surface_resolution_gate, str) or not (
            self.surface_resolution_gate
        ):
            raise ValueError("surface_resolution_gate must be non-empty")

        if self.source_kind == "disk":
            if (
                not self.terminated_at_opaque_disk
                or self.ray_boundary_outcome is not None
                or self.ray_boundary_target_id is not None
                or expected_opaque is None
                or expected_opaque != len(signature) - 1
                or not isinstance(self.crossing, RecordedSurfaceCrossing)
                or not isinstance(self.emitter, KerrDiskEmitter)
            ):
                raise ValueError("early disk spectrum has inconsistent source semantics")
            if (
                self.crossing.orientation
                != signature[expected_opaque].orientation
                or self.crossing.bracket_affine_width
                != widths[expected_opaque]
            ):
                raise ValueError(
                    "visible disk crossing disagrees with its signature diagnostics"
                )
            KerrDiskEmissionPayload(
                observer_frequencies_hz=frequencies,
                observed_specific_intensities_nu=intensities,
                crossing=self.crossing,
                disk_radius_m=self.disk_radius_m,  # type: ignore[arg-type]
                emitter=self.emitter,
                frequency_shift_g=self.frequency_shift_g,  # type: ignore[arg-type]
                emitted_frequencies_hz=self.emitted_frequencies_hz,  # type: ignore[arg-type]
                isotropic_emitted_specific_intensities_nu=(
                    self.isotropic_emitted_specific_intensities_nu  # type: ignore[arg-type]
                ),
                emitted_specific_intensities_nu=(
                    self.emitted_specific_intensities_nu  # type: ignore[arg-type]
                ),
                emission_angle_cosine=self.emission_angle_cosine,
                angular_emission_multiplier=self.angular_emission_multiplier,
                emitter_event_tolerance_m=(
                    self.emitter_event_tolerance_m  # type: ignore[arg-type]
                ),
            )
            radius = _finite_number(self.disk_radius_m, "disk_radius_m")
            shift = _finite_number(self.frequency_shift_g, "frequency_shift_g")
            event_tolerance = _finite_number(
                self.emitter_event_tolerance_m,
                "emitter_event_tolerance_m",
            )
            if radius <= 0.0 or shift <= 0.0:
                raise ValueError("disk radius and frequency shift must be positive")
            if event_tolerance < 0.0:
                raise ValueError("emitter event tolerance must be non-negative")
            if (
                self.emitted_frequencies_hz is None
                or self.isotropic_emitted_specific_intensities_nu is None
                or self.emitted_specific_intensities_nu is None
            ):
                raise ValueError("disk spectrum needs emitter-frame bins")
            emitted_frequencies = tuple(
                _finite_number(value, f"emitted_frequencies_hz[{index}]")
                for index, value in enumerate(self.emitted_frequencies_hz)
            )
            emitted_intensities = tuple(
                _finite_number(value, f"emitted_specific_intensities_nu[{index}]")
                for index, value in enumerate(self.emitted_specific_intensities_nu)
            )
            isotropic_intensities = tuple(
                _finite_number(
                    value,
                    f"isotropic_emitted_specific_intensities_nu[{index}]",
                )
                for index, value in enumerate(
                    self.isotropic_emitted_specific_intensities_nu
                )
            )
            if (
                len(emitted_frequencies) != len(frequencies)
                or len(isotropic_intensities) != len(frequencies)
                or len(emitted_intensities) != len(frequencies)
                or any(value <= 0.0 for value in emitted_frequencies)
                or any(
                    value < 0.0
                    for value in (*isotropic_intensities, *emitted_intensities)
                )
            ):
                raise ValueError("emitter-frame spectrum is malformed")
            if (self.emission_angle_cosine is None) != (
                self.angular_emission_multiplier is None
            ):
                raise ValueError("angular-emission diagnostics must be paired")
            if self.emission_angle_cosine is not None:
                cosine = _finite_number(
                    self.emission_angle_cosine,
                    "emission_angle_cosine",
                )
                multiplier = _finite_number(
                    self.angular_emission_multiplier,
                    "angular_emission_multiplier",
                )
                if not 0.0 <= cosine <= 1.0 or multiplier <= 0.0:
                    raise ValueError("angular-emission diagnostics are unphysical")
            effective_multiplier = (
                1.0
                if self.angular_emission_multiplier is None
                else self.angular_emission_multiplier
            )
            for isotropic_value, emitted_value in zip(
                isotropic_intensities,
                emitted_intensities,
            ):
                if not math.isclose(
                    emitted_value,
                    isotropic_value * effective_multiplier,
                    rel_tol=8.0e-14,
                    abs_tol=0.0,
                ):
                    raise ValueError(
                        "emitted intensity is inconsistent with angular emission"
                    )
            shift_cubed = shift * shift * shift
            for observed_frequency, emitted_frequency in zip(
                frequencies,
                emitted_frequencies,
            ):
                if not math.isclose(
                    emitted_frequency,
                    observed_frequency / shift,
                    rel_tol=8.0e-14,
                    abs_tol=0.0,
                ):
                    raise ValueError("emitted frequency is inconsistent with g")
            for observed, emitted in zip(intensities, emitted_intensities):
                if not math.isclose(
                    observed,
                    shift_cubed * emitted,
                    rel_tol=8.0e-14,
                    abs_tol=0.0,
                ):
                    raise ValueError("observed intensity is inconsistent with g^3")
            object.__setattr__(self, "disk_radius_m", radius)
            object.__setattr__(self, "frequency_shift_g", shift)
            object.__setattr__(self, "emitted_frequencies_hz", emitted_frequencies)
            object.__setattr__(
                self,
                "isotropic_emitted_specific_intensities_nu",
                isotropic_intensities,
            )
            object.__setattr__(
                self,
                "emitted_specific_intensities_nu",
                emitted_intensities,
            )
            if not math.isclose(
                radius,
                self.emitter.radius_m,
                rel_tol=2.0e-13,
                abs_tol=0.0,
            ):
                raise ValueError("disk radius disagrees with its emitter")
            if any(
                abs(actual - expected) > event_tolerance
                for actual, expected in zip(
                    self.crossing.state.event,
                    self.emitter.event,
                )
            ):
                raise ValueError("disk crossing event disagrees with its emitter")
            object.__setattr__(
                self,
                "emitter_event_tolerance_m",
                event_tolerance,
            )
        else:
            if self.source_kind not in ("captured-boundary", "escaped-boundary"):
                raise ValueError("visible source kind is invalid")
            expected_outcome = self.source_kind.removesuffix("-boundary")
            if (
                self.terminated_at_opaque_disk
                or expected_opaque is not None
                or self.ray_boundary_outcome != expected_outcome
                or not isinstance(self.ray_boundary_target_id, str)
                or not self.ray_boundary_target_id
                or any(
                    value is not None
                    for value in (
                        self.crossing,
                        self.disk_radius_m,
                        self.emitter,
                        self.frequency_shift_g,
                        self.emitted_frequencies_hz,
                        self.isotropic_emitted_specific_intensities_nu,
                        self.emitted_specific_intensities_nu,
                        self.emission_angle_cosine,
                        self.angular_emission_multiplier,
                        self.emitter_event_tolerance_m,
                    )
                )
            ):
                raise ValueError("boundary spectrum has inconsistent source semantics")
            if self.source_kind == "captured-boundary" and any(intensities):
                raise ValueError("captured boundary must be exactly black")
        object.__setattr__(self, "observer_frequencies_hz", frequencies)
        object.__setattr__(self, "observed_specific_intensities_nu", intensities)
        object.__setattr__(self, "crossing_signature", signature)
        object.__setattr__(self, "crossing_bracket_affine_widths", widths)


def _validated_visible_prefix(
    ray: RayTraceResult,
    surface: KerrDiskAnnulusSurface,
    surface_options: SurfaceEventOptions,
) -> tuple[ClassifiedInteriorSurfaceCrossing, ...]:
    if not isinstance(ray, RayTraceResult):
        raise TypeError("ray must be a RayTraceResult")
    if ray.failure_reason is not None:
        raise ValueError("failed ray may not enter early disk transfer")
    if ray.outcome not in (
        "captured",
        "escaped",
        surface.opaque_outcome,
    ):
        raise ValueError("ray has no certified visible terminal source")
    if (
        type(ray.accepted_steps) is not int
        or ray.accepted_steps < 1
        or type(ray.rejected_steps) is not int
        or ray.rejected_steps < 0
        or not math.isfinite(ray.affine_length)
        or ray.affine_length <= 0.0
    ):
        raise ValueError("ray step and affine diagnostics are invalid")
    if (
        not math.isfinite(ray.maximum_null_residual)
        or ray.maximum_null_residual < 0.0
        or ray.maximum_null_residual > surface_options.null_residual_limit
        or not math.isfinite(ray.maximum_metric_interpolation_error)
        or ray.maximum_metric_interpolation_error < 0.0
        or ray.maximum_metric_interpolation_error
        > surface_options.metric_interpolation_error_limit
    ):
        raise ValueError("ray numerical diagnostics exceed the surface limits")
    if (
        not isinstance(ray.segments, tuple)
        or len(ray.segments) != ray.accepted_steps
        or not ray.segments
        or ray.segments[-1].end != ray.terminal_state
        or any(
            previous.end != current.start
            for previous, current in zip(ray.segments, ray.segments[1:])
        )
        or not math.isclose(
            math.fsum(segment.affine_length for segment in ray.segments),
            ray.affine_length,
            rel_tol=2.0e-13,
            abs_tol=surface_options.affine_tolerance,
        )
    ):
        raise ValueError("ray path does not completely record its visible prefix")
    trace = ray.interior_surface_trace
    if trace is None or not trace.topology_converged:
        raise ValueError("ray lacks a converged accepted-step surface trace")
    if (
        trace.base_subdivisions_per_step
        != surface_options.subdivisions_per_segment
        or trace.verification_subdivisions_per_step
        != 2 * surface_options.subdivisions_per_segment
    ):
        raise ValueError(
            "ray surface probe resolution disagrees with surface_options"
        )
    entries = tuple(trace.crossings)
    segment_start_affines: list[float] = []
    cumulative_affine = 0.0
    for segment in ray.segments:
        segment_start_affines.append(cumulative_affine)
        cumulative_affine += segment.affine_length
    previous_affine = -math.inf
    for index, entry in enumerate(entries):
        if not isinstance(entry, ClassifiedInteriorSurfaceCrossing):
            raise ValueError("surface trace contains an invalid entry")
        crossing = entry.crossing
        if (
            crossing.ray_affine_length <= previous_affine
            or crossing.ray_affine_length <= 0.0
            or crossing.ray_affine_length > ray.affine_length
            or crossing.segment_index < 0
            or crossing.segment_index >= len(ray.segments)
            or crossing.segment_affine_length <= 0.0
            or crossing.segment_affine_length
            > ray.segments[crossing.segment_index].affine_length
            or abs(surface.value(crossing.state))
            > surface_options.surface_value_tolerance
        ):
            raise ValueError("surface trace crossing diagnostics are invalid")
        expected_ray_affine = (
            segment_start_affines[crossing.segment_index]
            + crossing.segment_affine_length
        )
        if not math.isclose(
            crossing.ray_affine_length,
            expected_ray_affine,
            rel_tol=2.0e-13,
            abs_tol=surface_options.affine_tolerance,
        ):
            raise ValueError(
                "surface trace crossing is not bound to its ray segment"
            )
        expected = surface.classify(crossing)
        if expected != entry.decision:
            raise ValueError("surface trace classification disagrees with the disk")
        if entry.decision.terminates and index != len(entries) - 1:
            raise ValueError("surface trace continues behind an opaque crossing")
        previous_affine = crossing.ray_affine_length

    if ray.outcome == surface.opaque_outcome:
        if (
            ray.terminal_target_id != surface.opaque_target_id
            or not entries
            or entries[-1].decision.classification != "opaque-annulus"
            or entries[-1].crossing.state != ray.terminal_state
            or entries[-1].crossing.segment_index != len(ray.segments) - 1
            or not math.isclose(
                entries[-1].crossing.segment_affine_length,
                ray.segments[-1].affine_length,
                rel_tol=2.0e-13,
                abs_tol=surface_options.affine_tolerance,
            )
            or entries[-1].crossing.state != ray.segments[-1].end
            or not math.isclose(
                entries[-1].crossing.ray_affine_length,
                ray.affine_length,
                rel_tol=2.0e-13,
                abs_tol=surface_options.affine_tolerance,
            )
        ):
            raise ValueError("opaque disk terminal event is inconsistent")
    elif (
        not isinstance(ray.terminal_target_id, str)
        or not ray.terminal_target_id
        or any(entry.decision.terminates for entry in entries)
    ):
        raise ValueError("boundary ray carries inconsistent surface semantics")
    return entries


def _checked_intensity(value: Any, label: str) -> float:
    intensity = _finite_number(value, label)
    if intensity < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return intensity


def transfer_early_stopped_kerr_disk_spectrum(
    surface: KerrDiskAnnulusSurface,
    ray: RayTraceResult,
    observer_four_velocity: Sequence[float],
    observer_frequencies_hz: Sequence[float],
    *,
    escaped_observer_specific_intensity_nu: EscapedObserverSpecificIntensity,
    surface_options: SurfaceEventOptions = SurfaceEventOptions(),
    frequency_null_residual_limit: float = 1.0e-7,
    conserved_quantity_tolerance: float = 1.0e-7,
    emitter_event_tolerance_m: float | None = None,
    angular_emission_law: AngularEmissionLaw | None = None,
) -> KerrDiskVisibleSpectrumResult:
    """Transfer a ray using its already-localized visible surface prefix."""

    if not isinstance(surface, KerrDiskAnnulusSurface):
        raise TypeError("surface must be a KerrDiskAnnulusSurface")
    if not isinstance(surface_options, SurfaceEventOptions):
        raise TypeError("surface_options must be a SurfaceEventOptions")
    if not callable(escaped_observer_specific_intensity_nu):
        raise TypeError("escaped observer intensity must be callable")
    if angular_emission_law is not None and not isinstance(
        angular_emission_law,
        AngularEmissionLaw,
    ):
        raise TypeError("angular_emission_law must implement AngularEmissionLaw")
    frequencies = _positive_frequencies(observer_frequencies_hz)
    observer_velocity = _finite_four_velocity(observer_four_velocity)
    null_limit = _finite_number(
        frequency_null_residual_limit,
        "frequency_null_residual_limit",
    )
    conserved_limit = _finite_number(
        conserved_quantity_tolerance,
        "conserved_quantity_tolerance",
    )
    if null_limit <= 0.0 or conserved_limit <= 0.0:
        raise ValueError("frequency-shift tolerances must be positive")
    event_tolerance = (
        1.0e-8 * surface.disk.metric.mass_m
        if emitter_event_tolerance_m is None
        else _finite_number(
            emitter_event_tolerance_m,
            "emitter_event_tolerance_m",
        )
    )
    if event_tolerance < 0.0:
        raise ValueError("emitter_event_tolerance_m must be non-negative")

    entries = _validated_visible_prefix(ray, surface, surface_options)
    signature = tuple(
        KerrDiskCrossingSignatureEntry(
            entry.crossing.orientation,
            cast(KerrDiskCrossingRegion, entry.decision.classification),
        )
        for entry in entries
    )
    widths = tuple(
        entry.crossing.bracket_affine_width for entry in entries
    )
    opaque_index = next(
        (
            index
            for index, entry in enumerate(signature)
            if entry.radial_region == "opaque-annulus"
        ),
        None,
    )

    if ray.outcome != surface.opaque_outcome:
        boundary_outcome = cast(Literal["captured", "escaped"], ray.outcome)
        boundary_target = cast(str, ray.terminal_target_id)
        if boundary_outcome == "captured":
            intensities = tuple(0.0 for _frequency in frequencies)
            source_kind: KerrDiskSourceKind = "captured-boundary"
        else:
            intensities = tuple(
                _checked_intensity(
                    escaped_observer_specific_intensity_nu(
                        ray.terminal_state,
                        frequency,
                        boundary_target,
                    ),
                    f"escaped specific intensity at bin {index}",
                )
                for index, frequency in enumerate(frequencies)
            )
            source_kind = "escaped-boundary"
        return KerrDiskVisibleSpectrumResult(
            observer_frequencies_hz=frequencies,
            observed_specific_intensities_nu=intensities,
            source_kind=source_kind,
            ray_boundary_outcome=boundary_outcome,
            ray_boundary_target_id=boundary_target,
            crossing_signature=signature,
            crossing_bracket_affine_widths=widths,
            first_opaque_crossing_index=None,
            terminated_at_opaque_disk=False,
        )

    if not entries or opaque_index is None:
        raise KerrDiskEarlyStopError("opaque ray lacks a visible disk crossing")
    crossing = entries[-1].crossing
    oblate = kerr_ks_event_to_oblate(surface.disk.metric, crossing.state.event)
    disk_radius = oblate.radius_m
    emitter = surface.disk.emitter(
        disk_radius,
        phi_ks_rad=oblate.phi_ks_rad,
        coordinate_time_m=oblate.coordinate_time_m,
    )
    shift = _finite_number(
        observer_to_emitter_frequency_shift_g(
            surface.disk.metric,
            ray.segments[0].start,
            observer_velocity,
            crossing.state,
            emitter,
            null_residual_limit=null_limit,
            conserved_quantity_tolerance=conserved_limit,
            emitter_event_tolerance_m=event_tolerance,
        ),
        "frequency_shift_g",
    )
    if shift <= 0.0:
        raise KerrDiskEarlyStopError("frequency shift g must be positive")
    emitted_frequencies = tuple(frequency / shift for frequency in frequencies)
    if any(
        not math.isfinite(frequency) or frequency <= 0.0
        for frequency in emitted_frequencies
    ):
        raise KerrDiskEarlyStopError("emitter-frame frequency is invalid")
    isotropic = tuple(
        _checked_intensity(
            surface.disk.emitted_specific_intensity_nu(disk_radius, frequency),
            f"emitted specific intensity at bin {index}",
        )
        for index, frequency in enumerate(emitted_frequencies)
    )
    emission_angle: float | None = None
    angular_multiplier: float | None = None
    emitted = isotropic
    if angular_emission_law is not None:
        emission_angle = _finite_number(
            equatorial_emission_angle_cosine(
                surface.disk.metric,
                crossing.state,
                emitter,
                null_residual_limit=null_limit,
                emitter_event_tolerance_m=event_tolerance,
            ),
            "emission_angle_cosine",
        )
        angular_multiplier = _finite_number(
            angular_emission_law.intensity_multiplier(emission_angle),
            "angular_emission_multiplier",
        )
        if angular_multiplier <= 0.0:
            raise KerrDiskEarlyStopError(
                "angular emission multiplier must be positive"
            )
        emitted = tuple(
            _checked_intensity(
                value * angular_multiplier,
                f"angular emitted specific intensity at bin {index}",
            )
            for index, value in enumerate(isotropic)
        )
    shift_cubed = shift * shift * shift
    if not math.isfinite(shift_cubed) or shift_cubed <= 0.0:
        raise KerrDiskEarlyStopError("g^3 transfer factor is invalid")
    observed = tuple(shift_cubed * value for value in emitted)
    if any(not math.isfinite(value) or value < 0.0 for value in observed):
        raise KerrDiskEarlyStopError("observer-frame disk intensity is invalid")
    return KerrDiskVisibleSpectrumResult(
        observer_frequencies_hz=frequencies,
        observed_specific_intensities_nu=observed,
        source_kind="disk",
        ray_boundary_outcome=None,
        ray_boundary_target_id=None,
        crossing_signature=signature,
        crossing_bracket_affine_widths=widths,
        first_opaque_crossing_index=opaque_index,
        terminated_at_opaque_disk=True,
        crossing=crossing,
        disk_radius_m=disk_radius,
        emitter=emitter,
        frequency_shift_g=shift,
        emitted_frequencies_hz=emitted_frequencies,
        isotropic_emitted_specific_intensities_nu=isotropic,
        emitted_specific_intensities_nu=emitted,
        emission_angle_cosine=emission_angle,
        angular_emission_multiplier=angular_multiplier,
        emitter_event_tolerance_m=event_tolerance,
    )


__all__ = (
    "KERR_DISK_OPAQUE_HIT_OUTCOME",
    "KERR_DISK_OPAQUE_HIT_TARGET_ID",
    "KERR_DISK_SURFACE_RESOLUTION_GATE",
    "KerrDiskAnnulusSurface",
    "KerrDiskEarlyStopError",
    "KerrDiskVisibleSpectrumResult",
    "transfer_early_stopped_kerr_disk_spectrum",
)
