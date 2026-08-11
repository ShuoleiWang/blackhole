"""Scalar spectral transfer for an opaque equatorial Kerr thin disk.

This module composes already-recorded observer-to-source null rays with the
stationary Novikov--Thorne surface emitter.  It deliberately stops at scalar
specific intensity: no screen basis is available for polarization, and no
returning-radiation iteration is performed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any,
    Final,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
    cast,
)

from offline.geodesic import (
    HamiltonianState,
    RayPathSegment,
    RayTraceResult,
    RecordedSurfaceCrossing,
    SurfaceEventOptions,
    locate_recorded_surface_crossings,
)
from offline.kerr import KerrOblateEvent, kerr_ks_event_to_oblate
from offline.kerr_disk import (
    KerrDiskEmitter,
    StationaryNovikovThorneDisk,
    observer_to_emitter_frequency_shift_g,
)
from offline.disk_atmosphere import (
    AngularEmissionLaw,
    equatorial_emission_angle_cosine,
)


KERR_DISK_TRANSFER_SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "observable": "scalar observer-frame specific intensity I_nu",
        "transferInvariant": "I_nu / nu^3",
        "surface": "opaque stationary equatorial Novikov-Thorne thin disk",
        "insideISCO": "transparent plunge-region gap",
        "includesPolarization": False,
        "includesReturningRadiation": False,
        "angularEmission": (
            "optional declared flux-normalized local law; absent means isotropic"
        ),
        "includesSolvedAtmosphere": False,
        "escapedBoundary": (
            "callable-owned already-observer-frame I_nu; its finite-boundary "
            "tetrad and redshift are outside this module"
        ),
        "prohibitedClaim": (
            "Do not describe this scalar first-surface transfer as polarized "
            "transport, returning radiation, an atmosphere, or GRMHD."
        ),
    }
)


KerrDiskSourceKind: TypeAlias = Literal[
    "disk",
    "captured-boundary",
    "escaped-boundary",
]
KerrDiskCrossingRegion: TypeAlias = Literal[
    "inside-isco",
    "opaque-annulus",
    "outside-outer-radius",
]


class EscapedObserverSpecificIntensity(Protocol):
    """Callable returning already-observer-frame escaped-boundary ``I_nu``.

    The callable owns any finite-boundary tetrad, source-frame spectrum, and
    boundary-to-observer redshift.  This module passes an observer frequency
    and does not apply another ``g^3`` factor to the returned intensity.
    """

    def __call__(
        self,
        terminal_state: HamiltonianState,
        observer_frequency_hz: float,
        boundary_target_id: str,
    ) -> float:
        """Return observer-frame ``I_nu`` in ``W m^-2 sr^-1 Hz^-1``."""

        ...


class KerrDiskTransferError(RuntimeError):
    """Raised when a nominal transfer cannot produce a finite physical value."""


@dataclass(frozen=True, slots=True)
class KerrDiskCrossingSignatureEntry:
    """Compact topology class for one ordered equatorial crossing."""

    orientation: int
    radial_region: KerrDiskCrossingRegion

    def __post_init__(self) -> None:
        if type(self.orientation) is not int or self.orientation not in (-1, 1):
            raise ValueError("crossing orientation must be -1 or +1")
        if self.radial_region not in (
            "inside-isco",
            "opaque-annulus",
            "outside-outer-radius",
        ):
            raise ValueError("crossing radial region is invalid")


@dataclass(frozen=True, slots=True)
class KerrDiskEmissionPayload:
    """Jointly validated scalar disk-hit observables shared by transfer paths."""

    observer_frequencies_hz: tuple[float, ...]
    observed_specific_intensities_nu: tuple[float, ...]
    crossing: RecordedSurfaceCrossing
    disk_radius_m: float
    emitter: KerrDiskEmitter
    frequency_shift_g: float
    emitted_frequencies_hz: tuple[float, ...]
    isotropic_emitted_specific_intensities_nu: tuple[float, ...]
    emitted_specific_intensities_nu: tuple[float, ...]
    emission_angle_cosine: float | None
    angular_emission_multiplier: float | None
    emitter_event_tolerance_m: float

    def __post_init__(self) -> None:
        frequencies = _positive_frequencies(self.observer_frequencies_hz)
        observed = tuple(
            _finite_number(value, f"observed_specific_intensities_nu[{index}]")
            for index, value in enumerate(self.observed_specific_intensities_nu)
        )
        if len(observed) != len(frequencies) or any(value < 0.0 for value in observed):
            raise ValueError("observed disk spectrum is malformed")
        if not isinstance(self.crossing, RecordedSurfaceCrossing):
            raise TypeError("disk emission payload needs a recorded crossing")
        if not isinstance(self.emitter, KerrDiskEmitter):
            raise TypeError("disk emission payload needs a Kerr disk emitter")
        radius = _finite_number(self.disk_radius_m, "disk_radius_m")
        shift = _finite_number(self.frequency_shift_g, "frequency_shift_g")
        event_tolerance = _finite_number(
            self.emitter_event_tolerance_m,
            "emitter_event_tolerance_m",
        )
        if radius <= 0.0 or shift <= 0.0 or event_tolerance < 0.0:
            raise ValueError("disk radius, g, or emitter-event tolerance is invalid")
        emitted_frequencies = tuple(
            _finite_number(value, f"emitted_frequencies_hz[{index}]")
            for index, value in enumerate(self.emitted_frequencies_hz)
        )
        isotropic = tuple(
            _finite_number(
                value,
                f"isotropic_emitted_specific_intensities_nu[{index}]",
            )
            for index, value in enumerate(
                self.isotropic_emitted_specific_intensities_nu
            )
        )
        emitted = tuple(
            _finite_number(value, f"emitted_specific_intensities_nu[{index}]")
            for index, value in enumerate(self.emitted_specific_intensities_nu)
        )
        if (
            len(emitted_frequencies) != len(frequencies)
            or len(isotropic) != len(frequencies)
            or len(emitted) != len(frequencies)
            or any(value <= 0.0 for value in emitted_frequencies)
            or any(value < 0.0 for value in (*isotropic, *emitted))
        ):
            raise ValueError("emitter-frame disk spectrum is malformed")
        if (self.emission_angle_cosine is None) != (
            self.angular_emission_multiplier is None
        ):
            raise ValueError("disk angular-emission diagnostics must be paired")
        cosine: float | None = None
        multiplier: float | None = None
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
        effective_multiplier = 1.0 if multiplier is None else multiplier
        for isotropic_value, emitted_value in zip(isotropic, emitted):
            expected = isotropic_value * effective_multiplier
            if not math.isfinite(expected) or not math.isclose(
                emitted_value,
                expected,
                rel_tol=8.0e-14,
                abs_tol=0.0,
            ):
                raise ValueError(
                    "emitted intensity is inconsistent with angular emission"
                )
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
        shift_cubed = shift * shift * shift
        if not math.isfinite(shift_cubed) or shift_cubed <= 0.0:
            raise ValueError("g^3 transfer factor is invalid")
        for observed_value, emitted_value in zip(observed, emitted):
            expected = shift_cubed * emitted_value
            if not math.isfinite(expected) or not math.isclose(
                observed_value,
                expected,
                rel_tol=8.0e-14,
                abs_tol=0.0,
            ):
                raise ValueError("observed intensity is inconsistent with g^3")
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
        object.__setattr__(self, "observer_frequencies_hz", frequencies)
        object.__setattr__(self, "observed_specific_intensities_nu", observed)
        object.__setattr__(self, "disk_radius_m", radius)
        object.__setattr__(self, "frequency_shift_g", shift)
        object.__setattr__(self, "emitted_frequencies_hz", emitted_frequencies)
        object.__setattr__(
            self,
            "isotropic_emitted_specific_intensities_nu",
            isotropic,
        )
        object.__setattr__(self, "emitted_specific_intensities_nu", emitted)
        object.__setattr__(self, "emission_angle_cosine", cosine)
        object.__setattr__(self, "angular_emission_multiplier", multiplier)
        object.__setattr__(self, "emitter_event_tolerance_m", event_tolerance)


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
    if not frequencies:
        raise ValueError("at least one observer frequency is required")
    if any(value <= 0.0 for value in frequencies):
        raise ValueError("observer frequencies must be positive")
    if any(right <= left for left, right in zip(frequencies, frequencies[1:])):
        raise ValueError("observer frequencies must be strictly increasing")
    return frequencies


def _finite_four_velocity(values: Sequence[float]) -> tuple[float, float, float, float]:
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
class KerrDiskSpectrumResult:
    """One ray's scalar spectrum and its first opaque source classification.

    Specific intensities use ``W m^-2 sr^-1 Hz^-1``.  Boundary diagnostics are
    retained even when an opaque disk hit hides that farther boundary.
    """

    observer_frequencies_hz: tuple[float, ...]
    observed_specific_intensities_nu: tuple[float, ...]
    source_kind: KerrDiskSourceKind
    ray_boundary_outcome: Literal["captured", "escaped"]
    ray_boundary_target_id: str
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
    crossing_signature: tuple[KerrDiskCrossingSignatureEntry, ...] = ()
    crossing_bracket_affine_widths: tuple[float, ...] = ()
    first_opaque_crossing_index: int | None = None

    def __post_init__(self) -> None:
        frequencies = _positive_frequencies(self.observer_frequencies_hz)
        intensities = tuple(
            _finite_number(value, f"observed_specific_intensities_nu[{index}]")
            for index, value in enumerate(self.observed_specific_intensities_nu)
        )
        if len(intensities) != len(frequencies):
            raise ValueError(
                "observer frequencies and intensities must have equal length"
            )
        if any(value < 0.0 for value in intensities):
            raise ValueError("observed specific intensities must be non-negative")
        if self.source_kind not in (
            "disk",
            "captured-boundary",
            "escaped-boundary",
        ):
            raise ValueError("Kerr disk spectrum source kind is invalid")
        if self.ray_boundary_outcome not in ("captured", "escaped"):
            raise ValueError("ray boundary outcome must be captured or escaped")
        if (
            not isinstance(self.ray_boundary_target_id, str)
            or not self.ray_boundary_target_id.strip()
        ):
            raise ValueError("ray boundary target id must be non-empty")
        signature = tuple(self.crossing_signature)
        if any(
            not isinstance(entry, KerrDiskCrossingSignatureEntry)
            for entry in signature
        ):
            raise ValueError("crossing signature contains an invalid entry")
        bracket_widths = tuple(
            _finite_number(
                value,
                f"crossing_bracket_affine_widths[{index}]",
            )
            for index, value in enumerate(self.crossing_bracket_affine_widths)
        )
        if len(bracket_widths) != len(signature):
            raise ValueError(
                "crossing signature and bracket widths must have equal length"
            )
        if any(value < 0.0 for value in bracket_widths):
            raise ValueError("crossing bracket widths must be non-negative")
        opaque_indices = tuple(
            index
            for index, entry in enumerate(signature)
            if entry.radial_region == "opaque-annulus"
        )
        expected_opaque_index = opaque_indices[0] if opaque_indices else None
        if self.first_opaque_crossing_index != expected_opaque_index:
            raise ValueError("first opaque crossing index is inconsistent")

        disk_fields = (
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
        if self.source_kind != "disk":
            if expected_opaque_index is not None:
                raise ValueError("opaque crossing signature requires a disk source")
            if any(value is not None for value in disk_fields):
                raise ValueError("boundary spectrum may not carry disk-hit diagnostics")
            expected_kind = f"{self.ray_boundary_outcome}-boundary"
            if self.source_kind != expected_kind:
                raise ValueError("boundary source kind disagrees with ray outcome")
            if self.source_kind == "captured-boundary" and any(intensities):
                raise ValueError("captured boundary must be exactly black")
        else:
            if not isinstance(self.crossing, RecordedSurfaceCrossing):
                raise ValueError("disk spectrum requires a recorded surface crossing")
            if not isinstance(self.emitter, KerrDiskEmitter):
                raise ValueError("disk spectrum requires a Kerr disk emitter")
            if expected_opaque_index is None:
                raise ValueError("disk spectrum requires an opaque crossing signature")
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
            if (
                self.crossing.orientation
                != signature[expected_opaque_index].orientation
            ):
                raise ValueError("visible disk crossing disagrees with its signature")
            if self.crossing.bracket_affine_width != bracket_widths[
                expected_opaque_index
            ]:
                raise ValueError(
                    "visible disk crossing disagrees with its bracket diagnostic"
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
            if self.emitted_frequencies_hz is None:
                raise ValueError("disk spectrum requires emitted frequencies")
            if self.isotropic_emitted_specific_intensities_nu is None:
                raise ValueError("disk spectrum requires isotropic emitted intensities")
            if self.emitted_specific_intensities_nu is None:
                raise ValueError("disk spectrum requires emitted intensities")
            emitted_frequencies = tuple(
                _finite_number(value, f"emitted_frequencies_hz[{index}]")
                for index, value in enumerate(self.emitted_frequencies_hz)
            )
            emitted_intensities = tuple(
                _finite_number(
                    value,
                    f"emitted_specific_intensities_nu[{index}]",
                )
                for index, value in enumerate(
                    self.emitted_specific_intensities_nu
                )
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
            ):
                raise ValueError("emitted and observed spectra must have equal length")
            if any(value <= 0.0 for value in emitted_frequencies):
                raise ValueError("emitted frequencies must be positive")
            if any(
                value < 0.0
                for value in (*isotropic_intensities, *emitted_intensities)
            ):
                raise ValueError("emitted intensities must be non-negative")
            if (self.emission_angle_cosine is None) != (
                self.angular_emission_multiplier is None
            ):
                raise ValueError(
                    "disk angular-emission diagnostics must be both present or absent"
                )
            if self.emission_angle_cosine is not None:
                emission_angle = _finite_number(
                    self.emission_angle_cosine,
                    "emission_angle_cosine",
                )
                angular_multiplier = _finite_number(
                    self.angular_emission_multiplier,
                    "angular_emission_multiplier",
                )
                if not 0.0 <= emission_angle <= 1.0:
                    raise ValueError("emission angle cosine must lie in [0, 1]")
                if angular_multiplier <= 0.0:
                    raise ValueError("angular emission multiplier must be positive")
                object.__setattr__(
                    self,
                    "emission_angle_cosine",
                    emission_angle,
                )
                object.__setattr__(
                    self,
                    "angular_emission_multiplier",
                    angular_multiplier,
                )
            effective_multiplier = (
                1.0
                if self.angular_emission_multiplier is None
                else self.angular_emission_multiplier
            )
            for isotropic, emitted in zip(
                isotropic_intensities,
                emitted_intensities,
            ):
                expected_emitted = isotropic * effective_multiplier
                if not math.isfinite(expected_emitted) or not math.isclose(
                    emitted,
                    expected_emitted,
                    rel_tol=8.0e-14,
                    abs_tol=0.0,
                ):
                    raise ValueError(
                        "emitted intensity is inconsistent with angular emission"
                    )
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
            shift_cubed = shift * shift * shift
            for observed, emitted in zip(intensities, emitted_intensities):
                expected = shift_cubed * emitted
                if not math.isfinite(expected) or not math.isclose(
                    observed,
                    expected,
                    rel_tol=8.0e-14,
                    abs_tol=0.0,
                ):
                    raise ValueError("observed intensity is inconsistent with g^3")

            object.__setattr__(self, "disk_radius_m", radius)
            object.__setattr__(self, "frequency_shift_g", shift)
            object.__setattr__(
                self,
                "emitted_frequencies_hz",
                emitted_frequencies,
            )
            object.__setattr__(
                self,
                "emitted_specific_intensities_nu",
                emitted_intensities,
            )
            object.__setattr__(
                self,
                "isotropic_emitted_specific_intensities_nu",
                isotropic_intensities,
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

        object.__setattr__(self, "observer_frequencies_hz", frequencies)
        object.__setattr__(self, "observed_specific_intensities_nu", intensities)
        object.__setattr__(self, "crossing_signature", signature)
        object.__setattr__(
            self,
            "crossing_bracket_affine_widths",
            bracket_widths,
        )


def _validated_recorded_ray(
    ray: RayTraceResult,
    surface_options: SurfaceEventOptions,
) -> tuple[RayPathSegment, ...]:
    if not isinstance(ray, RayTraceResult):
        raise TypeError("ray must be a RayTraceResult")
    if ray.outcome not in ("captured", "escaped"):
        raise ValueError(
            "Kerr disk transfer requires a successful captured or escaped ray"
        )
    if ray.failure_reason is not None:
        raise ValueError("failed ray may not enter Kerr disk transfer")
    if (
        not isinstance(ray.terminal_target_id, str)
        or not ray.terminal_target_id.strip()
    ):
        raise ValueError("terminal ray outcome requires a boundary target id")
    if (
        type(ray.accepted_steps) is not int
        or ray.accepted_steps < 1
        or type(ray.rejected_steps) is not int
        or ray.rejected_steps < 0
    ):
        raise ValueError("ray step counts are invalid")
    affine_length = _finite_number(ray.affine_length, "ray.affine_length")
    maximum_null = _finite_number(
        ray.maximum_null_residual,
        "ray.maximum_null_residual",
    )
    maximum_metric_error = _finite_number(
        ray.maximum_metric_interpolation_error,
        "ray.maximum_metric_interpolation_error",
    )
    if affine_length <= 0.0:
        raise ValueError("successful recorded ray must have positive affine length")
    if maximum_null < 0.0 or maximum_metric_error < 0.0:
        raise ValueError("ray error diagnostics must be non-negative")
    if maximum_null > surface_options.null_residual_limit:
        raise ValueError("ray null residual exceeds the surface-event limit")
    if maximum_metric_error > surface_options.metric_interpolation_error_limit:
        raise ValueError(
            "ray metric interpolation error exceeds the surface-event limit"
        )

    if not isinstance(ray.segments, tuple) or not ray.segments:
        raise ValueError("ray path was not completely recorded")
    segments = ray.segments
    if len(segments) != ray.accepted_steps:
        raise ValueError("recorded ray segment count does not match accepted steps")
    if any(not isinstance(segment, RayPathSegment) for segment in segments):
        raise ValueError("recorded ray contains an invalid path segment")
    if not isinstance(ray.terminal_state, HamiltonianState) or any(
        not isinstance(state, HamiltonianState)
        for segment in segments
        for state in (segment.start, segment.midpoint, segment.end)
    ):
        raise ValueError("recorded ray contains an invalid Hamiltonian state")
    if any(
        not math.isfinite(segment.midpoint_null_residual)
        or segment.midpoint_null_residual < 0.0
        for segment in segments
    ):
        raise ValueError("recorded ray contains an invalid midpoint residual")
    if any(
        not math.isfinite(segment.affine_length) or segment.affine_length <= 0.0
        for segment in segments
    ):
        raise ValueError("recorded ray contains a non-positive segment")
    if any(
        previous.end != current.start
        for previous, current in zip(segments, segments[1:])
    ):
        raise ValueError("recorded ray segments are not contiguous")
    if segments[-1].end != ray.terminal_state:
        raise ValueError("recorded ray does not end at its terminal state")
    recorded_length = math.fsum(segment.affine_length for segment in segments)
    if not math.isclose(
        recorded_length,
        affine_length,
        rel_tol=2.0e-13,
        abs_tol=2.0e-13,
    ):
        raise ValueError("recorded ray segments do not cover its affine length")
    return segments


def _checked_intensity(value: Any, label: str) -> float:
    intensity = _finite_number(value, label)
    if intensity < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return intensity


def transfer_kerr_disk_spectrum(
    disk: StationaryNovikovThorneDisk,
    ray: RayTraceResult,
    observer_four_velocity: Sequence[float],
    observer_frequencies_hz: Sequence[float],
    *,
    outer_radius_m: float,
    escaped_observer_specific_intensity_nu: EscapedObserverSpecificIntensity,
    surface_options: SurfaceEventOptions = SurfaceEventOptions(),
    frequency_null_residual_limit: float = 1.0e-7,
    conserved_quantity_tolerance: float = 1.0e-7,
    emitter_event_tolerance_m: float | None = None,
    angular_emission_law: AngularEmissionLaw | None = None,
) -> KerrDiskSpectrumResult:
    """Transfer one recorded Kerr ray into an observer-frame scalar spectrum.

    Surface crossings are consumed in observer-to-source order.  The first one
    in ``ISCO <= r <= outer_radius_m`` is an opaque disk hit; crossings inside
    the ISCO or outside the declared disk are transparent.  A disk hit hides
    the farther capture/escape boundary.  With no valid hit, capture is exactly
    black and escape is supplied by ``escaped_observer_specific_intensity_nu``.
    That callback returns an already-observer-frame intensity: it owns any
    finite escape-boundary tetrad and redshift, which are not inferred here.

    For every disk bin this applies Liouville's invariant exactly as
    ``I_nu_obs = g^3 I_nu_emit(nu_obs / g)``.  The result contains scalar
    intensity only and makes no polarization or returning-radiation claim.
    When ``angular_emission_law`` is supplied, the local emission angle is
    evaluated covariantly and the explicitly declared flux-normalized law is
    applied before the same ``g^3`` transfer.  ``None`` preserves isotropic
    surface emission without evaluating an otherwise unused angle.
    """

    if not isinstance(disk, StationaryNovikovThorneDisk):
        raise TypeError("disk must be a StationaryNovikovThorneDisk")
    if not isinstance(surface_options, SurfaceEventOptions):
        raise TypeError("surface_options must be a SurfaceEventOptions")
    if not callable(escaped_observer_specific_intensity_nu):
        raise TypeError("escaped_observer_specific_intensity_nu must be callable")
    if angular_emission_law is not None and not isinstance(
        angular_emission_law,
        AngularEmissionLaw,
    ):
        raise TypeError("angular_emission_law must implement AngularEmissionLaw")
    frequencies = _positive_frequencies(observer_frequencies_hz)
    observer_velocity = _finite_four_velocity(observer_four_velocity)
    outer_radius = _finite_number(outer_radius_m, "outer_radius_m")
    null_limit = _finite_number(
        frequency_null_residual_limit,
        "frequency_null_residual_limit",
    )
    constant_tolerance = _finite_number(
        conserved_quantity_tolerance,
        "conserved_quantity_tolerance",
    )
    if outer_radius < disk.isco_radius_m:
        raise ValueError("outer_radius_m must be at or outside the disk ISCO")
    if null_limit <= 0.0 or constant_tolerance <= 0.0:
        raise ValueError("frequency-shift tolerances must be positive")
    event_tolerance = (
        1.0e-8 * disk.metric.mass_m
        if emitter_event_tolerance_m is None
        else _finite_number(
            emitter_event_tolerance_m,
            "emitter_event_tolerance_m",
        )
    )
    if event_tolerance < 0.0:
        raise ValueError("emitter_event_tolerance_m must be non-negative")

    segments = _validated_recorded_ray(ray, surface_options)
    crossings = tuple(
        locate_recorded_surface_crossings(
            disk.metric,
            segments,
            lambda state: state.event[3],
            options=surface_options,
            ignore_unbracketed_path_endpoints=True,
        )
    )
    segment_start_affines: list[float] = []
    cumulative_affine = 0.0
    for segment in segments:
        segment_start_affines.append(cumulative_affine)
        cumulative_affine += segment.affine_length
    previous_affine = -math.inf
    for crossing in crossings:
        if not isinstance(crossing, RecordedSurfaceCrossing):
            raise KerrDiskTransferError("surface locator returned an invalid crossing")
        if (
            not isinstance(crossing.state, HamiltonianState)
            or crossing.ray_affine_length <= 0.0
            or crossing.ray_affine_length <= previous_affine
            or crossing.ray_affine_length > ray.affine_length
            or crossing.segment_index < 0
            or crossing.segment_index >= len(segments)
            or crossing.segment_affine_length
            > segments[crossing.segment_index].affine_length
        ):
            raise KerrDiskTransferError(
                "surface crossings are not ordered within the recorded ray"
            )
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
            raise KerrDiskTransferError(
                "surface crossing affine diagnostics do not match its segment"
            )
        previous_affine = crossing.ray_affine_length

    disk_crossing: RecordedSurfaceCrossing | None = None
    disk_radius: float | None = None
    disk_oblate_event: KerrOblateEvent | None = None
    crossing_signature: list[KerrDiskCrossingSignatureEntry] = []
    first_opaque_crossing_index: int | None = None
    for crossing_index, crossing in enumerate(crossings):
        oblate_event = kerr_ks_event_to_oblate(disk.metric, crossing.state.event)
        radius = oblate_event.radius_m
        if radius < disk.isco_radius_m:
            radial_region: KerrDiskCrossingRegion = "inside-isco"
        elif radius > outer_radius:
            radial_region = "outside-outer-radius"
        else:
            radial_region = "opaque-annulus"
        crossing_signature.append(
            KerrDiskCrossingSignatureEntry(
                orientation=crossing.orientation,
                radial_region=radial_region,
            )
        )
        if radial_region == "opaque-annulus" and disk_crossing is None:
            disk_crossing = crossing
            disk_radius = radius
            disk_oblate_event = oblate_event
            first_opaque_crossing_index = crossing_index

    boundary_outcome = cast(Literal["captured", "escaped"], ray.outcome)
    boundary_target = cast(str, ray.terminal_target_id)
    if disk_crossing is None:
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
        return KerrDiskSpectrumResult(
            observer_frequencies_hz=frequencies,
            observed_specific_intensities_nu=intensities,
            source_kind=source_kind,
            ray_boundary_outcome=boundary_outcome,
            ray_boundary_target_id=boundary_target,
            crossing_signature=tuple(crossing_signature),
            crossing_bracket_affine_widths=tuple(
                crossing.bracket_affine_width for crossing in crossings
            ),
            first_opaque_crossing_index=first_opaque_crossing_index,
        )

    if disk_radius is None or disk_oblate_event is None:
        raise AssertionError("disk crossing diagnostics are incomplete")
    emitter = disk.emitter(
        disk_radius,
        phi_ks_rad=disk_oblate_event.phi_ks_rad,
        coordinate_time_m=disk_oblate_event.coordinate_time_m,
    )
    shift = _finite_number(
        observer_to_emitter_frequency_shift_g(
            disk.metric,
            segments[0].start,
            observer_velocity,
            disk_crossing.state,
            emitter,
            null_residual_limit=null_limit,
            conserved_quantity_tolerance=constant_tolerance,
            emitter_event_tolerance_m=event_tolerance,
        ),
        "frequency_shift_g",
    )
    if shift <= 0.0:
        raise KerrDiskTransferError("frequency shift g must be positive")
    emitted_frequencies = tuple(frequency / shift for frequency in frequencies)
    if any(
        not math.isfinite(frequency) or frequency <= 0.0
        for frequency in emitted_frequencies
    ):
        raise KerrDiskTransferError("emitter-frame frequency is invalid")
    isotropic_emitted_intensities = tuple(
        _checked_intensity(
            disk.emitted_specific_intensity_nu(disk_radius, frequency),
            f"emitted specific intensity at bin {index}",
        )
        for index, frequency in enumerate(emitted_frequencies)
    )
    emission_angle: float | None = None
    angular_multiplier: float | None = None
    if angular_emission_law is None:
        emitted_intensities = isotropic_emitted_intensities
    else:
        emission_angle = _finite_number(
            equatorial_emission_angle_cosine(
                disk.metric,
                disk_crossing.state,
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
            raise KerrDiskTransferError(
                "angular emission multiplier must be positive"
            )
        emitted_intensities = tuple(
            _checked_intensity(
                intensity * angular_multiplier,
                f"angular emitted specific intensity at bin {index}",
            )
            for index, intensity in enumerate(isotropic_emitted_intensities)
        )
    shift_cubed = shift * shift * shift
    if not math.isfinite(shift_cubed) or shift_cubed <= 0.0:
        raise KerrDiskTransferError("g^3 transfer factor is invalid")
    observed_intensities = tuple(
        shift_cubed * intensity for intensity in emitted_intensities
    )
    if any(
        not math.isfinite(intensity) or intensity < 0.0
        for intensity in observed_intensities
    ):
        raise KerrDiskTransferError("observer-frame specific intensity is invalid")
    return KerrDiskSpectrumResult(
        observer_frequencies_hz=frequencies,
        observed_specific_intensities_nu=observed_intensities,
        source_kind="disk",
        ray_boundary_outcome=boundary_outcome,
        ray_boundary_target_id=boundary_target,
        crossing=disk_crossing,
        disk_radius_m=disk_radius,
        emitter=emitter,
        frequency_shift_g=shift,
        emitted_frequencies_hz=emitted_frequencies,
        isotropic_emitted_specific_intensities_nu=(
            isotropic_emitted_intensities
        ),
        emitted_specific_intensities_nu=emitted_intensities,
        emission_angle_cosine=emission_angle,
        angular_emission_multiplier=angular_multiplier,
        emitter_event_tolerance_m=event_tolerance,
        crossing_signature=tuple(crossing_signature),
        crossing_bracket_affine_widths=tuple(
            crossing.bracket_affine_width for crossing in crossings
        ),
        first_opaque_crossing_index=first_opaque_crossing_index,
    )


__all__ = (
    "EscapedObserverSpecificIntensity",
    "KERR_DISK_TRANSFER_SCIENTIFIC_STATUS",
    "KerrDiskCrossingRegion",
    "KerrDiskCrossingSignatureEntry",
    "KerrDiskEmissionPayload",
    "KerrDiskSourceKind",
    "KerrDiskSpectrumResult",
    "KerrDiskTransferError",
    "transfer_kerr_disk_spectrum",
)
