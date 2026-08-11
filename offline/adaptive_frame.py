"""Deterministic adaptive integration of observer-frame spectral pixels.

This module is deliberately independent of any spacetime or emission model.
It integrates already-converged scalar ray samples over the solid angle of one
pinhole-camera pixel.  A finite stencil cannot prove that an arbitrarily thin
caustic lies between all probes; the returned diagnostics therefore describe
sampling convergence at a declared dyadic depth and never claim a Jacobi ray
bundle or caustic-complete image.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Callable, Final, Mapping, Protocol, Sequence, runtime_checkable


STENCIL_VERSION: Final = "dyadic-center-quarter-guard13-v1"
SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": "deterministic adaptive independent-ray spectral integral",
        "solidAngleProjection": "pinhole Jacobian integrated in screen coordinates",
        "errorSemantics": "finite-stencil estimated absolute sampling error",
        "isJacobiRayBundle": False,
        "isCausticComplete": False,
        "isRigorousErrorBound": False,
        "prohibitedClaim": (
            "Do not describe finite independent-ray stencils as a Jacobi/Sachs "
            "bundle, a proof that no sub-stencil image exists, or caustic-complete."
        ),
    }
)


class AdaptiveSamplingError(RuntimeError):
    """Raised when a pixel cannot remain inside its declared sampling contract."""


@dataclass(frozen=True, slots=True)
class RayConvergenceAudit:
    """Per-sample numerical evidence aggregated into scientific frame records."""

    maximum_null_residual: float = 0.0
    maximum_metric_interpolation_error: float = 0.0
    terminal_event_difference_m: float = 0.0
    terminal_covector_relative_difference: float = 0.0
    disk_radius_difference_m: float = 0.0
    relative_g_difference: float = 0.0
    surface_bracket_affine_width: float = 0.0
    accepted_steps: int = 0
    rejected_steps: int = 0
    ray_gate_passed: bool = False
    source_gate_passed: bool = False
    transfer_gate_passed: bool = False

    def __post_init__(self) -> None:
        scalar_fields = (
            "maximum_null_residual",
            "maximum_metric_interpolation_error",
            "terminal_event_difference_m",
            "terminal_covector_relative_difference",
            "disk_radius_difference_m",
            "relative_g_difference",
            "surface_bracket_affine_width",
        )
        for name in scalar_fields:
            value = _finite_number(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in ("accepted_steps", "rejected_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "ray_gate_passed",
            "source_gate_passed",
            "transfer_gate_passed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if self.ray_gate_passed and self.accepted_steps < 1:
            raise ValueError("a passed ray gate requires at least one accepted step")


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _finite_tuple(values: Sequence[float], label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    try:
        return tuple(
            _finite_number(value, f"{label}[{index}]")
            for index, value in enumerate(values)
        )
    except TypeError as error:
        raise ValueError(f"{label} must be a sequence") from error


def _solid_angle_roundoff(value: float) -> float:
    """Return a scale-local roundoff allowance without hiding tiny pixels."""

    return 16.0 * math.ulp(value)


def _unit_direction(
    value: Sequence[float] | None,
    label: str,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    entries = _finite_tuple(value, label)
    if len(entries) != 3:
        raise ValueError(f"{label} must contain three finite numbers")
    norm = math.sqrt(math.fsum(component * component for component in entries))
    if not math.isfinite(norm) or abs(norm - 1.0) > 2.0e-10:
        raise ValueError(f"{label} must be a unit direction")
    return entries  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class SpectralRaySample:
    """One fully converged ray sample before angular pixel integration."""

    specific_intensities_nu: tuple[float, ...]
    absolute_errors_nu: tuple[float, ...]
    visible_source: str
    topology_signature: str
    frequency_shift_g: float | None = None
    escape_direction: tuple[float, float, float] | None = None
    ray_converged: bool = False
    convergence_audit: RayConvergenceAudit = RayConvergenceAudit()

    def __post_init__(self) -> None:
        intensities = _finite_tuple(
            self.specific_intensities_nu,
            "specific_intensities_nu",
        )
        errors = _finite_tuple(self.absolute_errors_nu, "absolute_errors_nu")
        if not intensities or len(errors) != len(intensities):
            raise ValueError("ray intensities and errors need equal non-zero length")
        if any(value < 0.0 for value in (*intensities, *errors)):
            raise ValueError("ray intensities and errors must be non-negative")
        if not isinstance(self.visible_source, str) or not self.visible_source.strip():
            raise ValueError("visible_source must be a non-empty string")
        if (
            not isinstance(self.topology_signature, str)
            or not self.topology_signature.strip()
        ):
            raise ValueError("topology_signature must be a non-empty string")
        if type(self.ray_converged) is not bool:
            raise TypeError("ray_converged must be a bool")
        if not isinstance(self.convergence_audit, RayConvergenceAudit):
            raise TypeError("convergence_audit must be a RayConvergenceAudit")
        if self.ray_converged and not (
            self.convergence_audit.ray_gate_passed
            and self.convergence_audit.source_gate_passed
            and self.convergence_audit.transfer_gate_passed
        ):
            raise ValueError("converged ray sample has a failed audit gate")
        if self.visible_source == "captured-boundary" and any(
            value != 0.0 or math.copysign(1.0, value) < 0.0
            for value in intensities
        ):
            raise ValueError("captured-boundary ray intensity must be positive zero")
        shift = self.frequency_shift_g
        if shift is not None:
            shift = _finite_number(shift, "frequency_shift_g")
            if shift <= 0.0:
                raise ValueError("frequency_shift_g must be positive")
        direction = _unit_direction(self.escape_direction, "escape_direction")
        if self.visible_source == "disk" and (
            shift is None or direction is not None
        ):
            raise ValueError("disk rays require g and may not carry escape direction")
        if self.visible_source == "captured-boundary" and (
            shift is not None or direction is not None
        ):
            raise ValueError("captured-boundary rays may not carry g or direction")
        if self.visible_source == "escaped-boundary" and (
            shift is not None or direction is None
        ):
            raise ValueError("escaped-boundary rays require direction and no g")
        object.__setattr__(self, "specific_intensities_nu", intensities)
        object.__setattr__(self, "absolute_errors_nu", errors)
        object.__setattr__(self, "frequency_shift_g", shift)
        object.__setattr__(self, "escape_direction", direction)


@runtime_checkable
class SpectralRaySampler(Protocol):
    """Screen-space sampler consumed by the adaptive integrator."""

    def sample(
        self,
        screen_x: float,
        screen_y: float,
        observer_frequencies_hz: tuple[float, ...],
    ) -> SpectralRaySample:
        """Return one converged scalar spectral ray sample."""


@dataclass(frozen=True, slots=True)
class AdaptivePixelOptions:
    minimum_depth: int = 0
    maximum_depth: int = 3
    maximum_ray_evaluations: int = 2_000
    radiance_absolute_tolerances: tuple[float, ...] = (0.0,)
    radiance_relative_tolerance: float = 1.0e-3
    unresolved_solid_angle_fraction_tolerance: float = 0.0
    weighted_log_g_tolerance: float = 1.0e-3
    weighted_direction_tolerance_rad: float = 1.0e-4
    radiance_guard_ceilings: tuple[float, ...] = (1.0,)
    stencil_version: str = STENCIL_VERSION

    def __post_init__(self) -> None:
        integer_values = (
            self.minimum_depth,
            self.maximum_depth,
            self.maximum_ray_evaluations,
        )
        if any(type(value) is not int for value in integer_values):
            raise TypeError("adaptive depths and ray budget must be integers")
        if self.minimum_depth < 0 or self.maximum_depth < self.minimum_depth:
            raise ValueError("adaptive depths must satisfy 0 <= minimum <= maximum")
        if self.maximum_ray_evaluations < 1:
            raise ValueError("maximum_ray_evaluations must be positive")
        absolute = _finite_tuple(
            self.radiance_absolute_tolerances,
            "radiance_absolute_tolerances",
        )
        ceilings = _finite_tuple(
            self.radiance_guard_ceilings,
            "radiance_guard_ceilings",
        )
        if not absolute or len(ceilings) != len(absolute):
            raise ValueError("radiance tolerances and ceilings need equal non-zero length")
        if any(value < 0.0 for value in absolute):
            raise ValueError("radiance absolute tolerances must be non-negative")
        if any(value <= 0.0 for value in ceilings):
            raise ValueError("radiance guard ceilings must be positive")
        scalar_limits = (
            self.radiance_relative_tolerance,
            self.unresolved_solid_angle_fraction_tolerance,
            self.weighted_log_g_tolerance,
            self.weighted_direction_tolerance_rad,
        )
        normalized_limits = tuple(
            _finite_number(value, f"adaptive scalar limit {index}")
            for index, value in enumerate(scalar_limits)
        )
        if any(value < 0.0 for value in normalized_limits):
            raise ValueError("adaptive scalar limits must be non-negative")
        if normalized_limits[1] > 1.0:
            raise ValueError("unresolved solid-angle tolerance cannot exceed one")
        if self.stencil_version != STENCIL_VERSION:
            raise ValueError(f"unsupported adaptive stencil {self.stencil_version!r}")
        object.__setattr__(self, "radiance_absolute_tolerances", absolute)
        object.__setattr__(self, "radiance_guard_ceilings", ceilings)
        object.__setattr__(self, "radiance_relative_tolerance", normalized_limits[0])
        object.__setattr__(
            self,
            "unresolved_solid_angle_fraction_tolerance",
            normalized_limits[1],
        )
        object.__setattr__(self, "weighted_log_g_tolerance", normalized_limits[2])
        object.__setattr__(
            self,
            "weighted_direction_tolerance_rad",
            normalized_limits[3],
        )


@dataclass(frozen=True, slots=True)
class AdaptivePixelResult:
    observer_frequencies_hz: tuple[float, ...]
    integrated_specific_intensity_nu_sr: tuple[float, ...]
    mean_specific_intensities_nu: tuple[float, ...]
    estimated_absolute_errors_nu_sr: tuple[float, ...]
    pixel_solid_angle_sr: float
    sample_count: int
    maximum_depth_reached: int
    unresolved_solid_angle_sr: float
    frequency_shift_solid_angle_sr: float
    escape_direction_solid_angle_sr: float
    source_solid_angles_sr: tuple[tuple[str, float], ...]
    minimum_frequency_shift_g: float | None
    maximum_frequency_shift_g: float | None
    maximum_escape_direction_span_rad: float
    weighted_log_g_variation: float
    weighted_escape_direction_variation_rad: float
    maximum_null_residual: float
    maximum_metric_interpolation_error: float
    maximum_terminal_event_difference_m: float
    maximum_terminal_covector_relative_difference: float
    maximum_disk_radius_difference_m: float
    maximum_relative_g_difference: float
    maximum_surface_bracket_affine_width: float
    maximum_accepted_steps: int
    maximum_rejected_steps: int
    all_ray_gates_passed: bool
    all_source_gates_passed: bool
    all_transfer_gates_passed: bool
    converged: bool

    def __post_init__(self) -> None:
        frequencies = _finite_tuple(
            self.observer_frequencies_hz,
            "observer_frequencies_hz",
        )
        flux = _finite_tuple(
            self.integrated_specific_intensity_nu_sr,
            "integrated_specific_intensity_nu_sr",
        )
        means = _finite_tuple(
            self.mean_specific_intensities_nu,
            "mean_specific_intensities_nu",
        )
        errors = _finite_tuple(
            self.estimated_absolute_errors_nu_sr,
            "estimated_absolute_errors_nu_sr",
        )
        if (
            not frequencies
            or len(flux) != len(frequencies)
            or len(means) != len(frequencies)
            or len(errors) != len(frequencies)
        ):
            raise ValueError("adaptive pixel spectral tuples need equal non-zero length")
        if any(value <= 0.0 for value in frequencies) or any(
            right <= left for left, right in zip(frequencies, frequencies[1:])
        ):
            raise ValueError("adaptive pixel frequencies must increase strictly")
        if any(value < 0.0 for value in (*flux, *means, *errors)):
            raise ValueError("adaptive pixel radiance and errors must be non-negative")
        solid_angle = _finite_number(
            self.pixel_solid_angle_sr,
            "pixel_solid_angle_sr",
        )
        unresolved = _finite_number(
            self.unresolved_solid_angle_sr,
            "unresolved_solid_angle_sr",
        )
        frequency_shift_area = _finite_number(
            self.frequency_shift_solid_angle_sr,
            "frequency_shift_solid_angle_sr",
        )
        escape_direction_area = _finite_number(
            self.escape_direction_solid_angle_sr,
            "escape_direction_solid_angle_sr",
        )
        if solid_angle <= 0.0 or any(
            value < 0.0 or value > solid_angle
            for value in (
                unresolved,
                frequency_shift_area,
                escape_direction_area,
            )
        ):
            raise ValueError("adaptive pixel solid-angle diagnostics are invalid")
        for integrated, mean in zip(flux, means):
            if not math.isclose(
                mean,
                integrated / solid_angle,
                rel_tol=2.0e-13,
                abs_tol=0.0,
            ):
                raise ValueError("adaptive pixel mean and integrated radiance disagree")
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise ValueError("adaptive pixel sample_count must be a positive integer")
        if (
            type(self.maximum_depth_reached) is not int
            or self.maximum_depth_reached < 0
        ):
            raise ValueError("adaptive pixel maximum depth must be non-negative")
        sources = tuple(self.source_solid_angles_sr)
        if any(
            not isinstance(source, str)
            or not source
            or not math.isfinite(area)
            or area < 0.0
            for source, area in sources
        ):
            raise ValueError("adaptive pixel source solid angles are invalid")
        source_names = tuple(source for source, _area in sources)
        if source_names != tuple(sorted(set(source_names))):
            raise ValueError("adaptive pixel source names must be unique and sorted")
        if not math.isclose(
            math.fsum(area for _source, area in sources),
            solid_angle,
            rel_tol=2.0e-13,
            abs_tol=_solid_angle_roundoff(solid_angle),
        ):
            raise ValueError("adaptive pixel source solid angles do not cover the pixel")
        minimum_g = self.minimum_frequency_shift_g
        maximum_g = self.maximum_frequency_shift_g
        if (minimum_g is None) != (maximum_g is None):
            raise ValueError("adaptive pixel g bounds must both be present or absent")
        if minimum_g is not None and maximum_g is not None:
            minimum_g = _finite_number(minimum_g, "minimum_frequency_shift_g")
            maximum_g = _finite_number(maximum_g, "maximum_frequency_shift_g")
            if minimum_g <= 0.0 or maximum_g < minimum_g:
                raise ValueError("adaptive pixel g bounds are invalid")
            if frequency_shift_area <= 0.0:
                raise ValueError("adaptive pixel g bounds require positive coverage")
        elif frequency_shift_area != 0.0:
            raise ValueError("adaptive pixel g coverage requires frequency-shift bounds")
        scalar_diagnostics = (
            self.maximum_escape_direction_span_rad,
            self.weighted_log_g_variation,
            self.weighted_escape_direction_variation_rad,
            self.maximum_null_residual,
            self.maximum_metric_interpolation_error,
            self.maximum_terminal_event_difference_m,
            self.maximum_terminal_covector_relative_difference,
            self.maximum_disk_radius_difference_m,
            self.maximum_relative_g_difference,
            self.maximum_surface_bracket_affine_width,
        )
        if any(
            _finite_number(value, f"adaptive diagnostic {index}") < 0.0
            for index, value in enumerate(scalar_diagnostics)
        ):
            raise ValueError("adaptive pixel variation diagnostics are invalid")
        if (
            self.maximum_escape_direction_span_rad > 0.0
            or self.weighted_escape_direction_variation_rad > 0.0
        ) and escape_direction_area <= 0.0:
            raise ValueError("adaptive direction variation requires positive coverage")
        if self.weighted_log_g_variation > 0.0 and frequency_shift_area <= 0.0:
            raise ValueError("adaptive g variation requires positive coverage")
        for name in ("maximum_accepted_steps", "maximum_rejected_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"adaptive pixel {name} must be non-negative")
        for name in (
            "all_ray_gates_passed",
            "all_source_gates_passed",
            "all_transfer_gates_passed",
            "converged",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"adaptive pixel {name} must be a bool")
        object.__setattr__(self, "observer_frequencies_hz", frequencies)
        object.__setattr__(self, "integrated_specific_intensity_nu_sr", flux)
        object.__setattr__(self, "mean_specific_intensities_nu", means)
        object.__setattr__(self, "estimated_absolute_errors_nu_sr", errors)
        object.__setattr__(self, "pixel_solid_angle_sr", solid_angle)
        object.__setattr__(self, "unresolved_solid_angle_sr", unresolved)
        object.__setattr__(
            self,
            "frequency_shift_solid_angle_sr",
            frequency_shift_area,
        )
        object.__setattr__(
            self,
            "escape_direction_solid_angle_sr",
            escape_direction_area,
        )
        object.__setattr__(self, "source_solid_angles_sr", sources)
        object.__setattr__(self, "minimum_frequency_shift_g", minimum_g)
        object.__setattr__(self, "maximum_frequency_shift_g", maximum_g)


@dataclass(frozen=True, slots=True)
class _Leaf:
    flux: tuple[float, ...]
    error: tuple[float, ...]
    solid_angle: float
    unresolved_solid_angle: float
    source_solid_angles: tuple[tuple[str, float], ...]
    topology_tokens: frozenset[tuple[str, str]]
    minimum_intensities: tuple[float, ...]
    maximum_intensities: tuple[float, ...]
    minimum_g: float | None
    maximum_g: float | None
    g_solid_angle: float
    direction_solid_angle: float
    directions: tuple[tuple[float, float, float], ...]
    direction_span: float
    weighted_log_g_variation: float
    weighted_direction_variation: float
    maximum_depth: int
    converged: bool


def pinhole_solid_angle(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> float:
    """Return the exact solid angle of a rectangle on the plane ``z=1``."""

    values = tuple(
        _finite_number(value, f"pinhole bound {index}")
        for index, value in enumerate((x_min, x_max, y_min, y_max))
    )
    x0, x1, y0, y1 = values
    if x1 <= x0 or y1 <= y0:
        raise ValueError("pinhole rectangle bounds must be strictly increasing")

    corners = (
        (x0, y0, 1.0),
        (x1, y0, 1.0),
        (x1, y1, 1.0),
        (x0, y1, 1.0),
    )
    norms = tuple(math.hypot(*corner) for corner in corners)

    def dot(first: int, second: int) -> float:
        return math.fsum(
            corners[first][axis] * corners[second][axis]
            for axis in range(3)
        ) / (norms[first] * norms[second])

    determinant = (x1 - x0) * (y1 - y0)

    def triangle(first: int, second: int, third: int) -> float:
        normalized_determinant = determinant / (
            norms[first] * norms[second] * norms[third]
        )
        denominator = 1.0 + math.fsum(
            (
                dot(first, second),
                dot(second, third),
                dot(third, first),
            )
        )
        return 2.0 * math.atan2(abs(normalized_determinant), denominator)

    result = math.fsum((triangle(0, 1, 2), triangle(0, 2, 3)))
    if not math.isfinite(result) or result <= 0.0:
        raise AdaptiveSamplingError("pinhole rectangle solid angle is invalid")
    return result


def _maximum_direction_span_vectors(
    directions: Sequence[tuple[float, float, float]],
) -> float:
    directions = tuple(directions)
    if len(directions) < 2:
        return 0.0
    maximum = 0.0
    for first_index, first in enumerate(directions):
        for second in directions[first_index + 1 :]:
            cross = (
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            )
            maximum = max(
                maximum,
                math.atan2(
                    math.sqrt(math.fsum(value * value for value in cross)),
                    math.fsum(first[index] * second[index] for index in range(3)),
                ),
            )
    return maximum


def _maximum_direction_span(samples: Sequence[SpectralRaySample]) -> float:
    directions = tuple(
        sample.escape_direction
        for sample in samples
        if sample.escape_direction is not None
    )
    return _maximum_direction_span_vectors(directions)


def integrate_spectral_pixel(
    sampler: SpectralRaySampler | Callable[
        [float, float, tuple[float, ...]], SpectralRaySample
    ],
    observer_frequencies_hz: Sequence[float],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    options: AdaptivePixelOptions,
) -> AdaptivePixelResult:
    """Integrate one pinhole pixel with a deterministic dyadic quadtree.

    Each cell compares its center estimate with four child-center estimates and
    probes eight inset guards.  Mixed topology is never averaged away silently:
    it is refined to the declared maximum depth, then contributes a conservative
    guard-ceiling error and unresolved solid-angle diagnostic.
    """

    if not isinstance(options, AdaptivePixelOptions):
        raise TypeError("options must be AdaptivePixelOptions")
    frequencies = _finite_tuple(observer_frequencies_hz, "observer_frequencies_hz")
    if not frequencies or any(value <= 0.0 for value in frequencies):
        raise ValueError("observer frequencies must be finite and positive")
    if any(right <= left for left, right in zip(frequencies, frequencies[1:])):
        raise ValueError("observer frequencies must be strictly increasing")
    if len(frequencies) != len(options.radiance_absolute_tolerances):
        raise ValueError("adaptive spectral options do not match frequency bins")
    bounds = tuple(
        _finite_number(value, f"pixel bound {index}")
        for index, value in enumerate((x_min, x_max, y_min, y_max))
    )
    x0, x1, y0, y1 = bounds
    if x1 <= x0 or y1 <= y0:
        raise ValueError("pixel bounds must be strictly increasing")
    if isinstance(sampler, SpectralRaySampler):
        sample_function = sampler.sample
    elif callable(sampler):
        sample_function = sampler
    else:
        raise TypeError("sampler must implement SpectralRaySampler or be callable")

    pixel_solid_angle = pinhole_solid_angle(x0, x1, y0, y1)
    screen_width = x1 - x0
    screen_height = y1 - y0
    cache: dict[tuple[Fraction, Fraction], SpectralRaySample] = {}
    deepest = 0

    def evaluate(local_x: Fraction, local_y: Fraction) -> SpectralRaySample:
        key = (local_x, local_y)
        cached = cache.get(key)
        if cached is not None:
            return cached
        if len(cache) >= options.maximum_ray_evaluations:
            raise AdaptiveSamplingError("adaptive ray-evaluation budget exhausted")
        screen_x = x0 + screen_width * float(local_x)
        screen_y = y0 + screen_height * float(local_y)
        sample = sample_function(screen_x, screen_y, frequencies)
        if not isinstance(sample, SpectralRaySample):
            raise TypeError("spectral sampler returned an invalid sample")
        if len(sample.specific_intensities_nu) != len(frequencies):
            raise ValueError("spectral sampler returned the wrong bin count")
        if not sample.ray_converged:
            raise AdaptiveSamplingError("spectral ray did not pass convergence gates")
        for value, error, ceiling in zip(
            sample.specific_intensities_nu,
            sample.absolute_errors_nu,
            options.radiance_guard_ceilings,
        ):
            if value + error > ceiling:
                raise AdaptiveSamplingError(
                    "sample exceeded its declared radiance guard ceiling"
                )
        cache[key] = sample
        return sample

    child_order = ((0, 0), (1, 0), (0, 1), (1, 1))
    guard_offsets = (
        (1, 1),
        (4, 1),
        (7, 1),
        (1, 4),
        (7, 4),
        (1, 7),
        (4, 7),
        (7, 7),
    )

    def point(depth: int, cell_x: int, cell_y: int, nx: int, ny: int) -> tuple[Fraction, Fraction]:
        denominator = 8 * (1 << depth)
        return (
            Fraction(8 * cell_x + nx, denominator),
            Fraction(8 * cell_y + ny, denominator),
        )

    def merge(leaves: Sequence[_Leaf]) -> _Leaf:
        source_totals: dict[str, list[float]] = {}
        for leaf in leaves:
            for source, area in leaf.source_solid_angles:
                source_totals.setdefault(source, []).append(area)
        minimum_values = tuple(
            leaf.minimum_g for leaf in leaves if leaf.minimum_g is not None
        )
        maximum_values = tuple(
            leaf.maximum_g for leaf in leaves if leaf.maximum_g is not None
        )
        merged_directions = tuple(
            sorted(
                {
                    direction
                    for leaf in leaves
                    for direction in leaf.directions
                }
            )
        )
        return _Leaf(
            flux=tuple(
                math.fsum(leaf.flux[index] for leaf in leaves)
                for index in range(len(frequencies))
            ),
            error=tuple(
                math.fsum(leaf.error[index] for leaf in leaves)
                for index in range(len(frequencies))
            ),
            solid_angle=math.fsum(leaf.solid_angle for leaf in leaves),
            unresolved_solid_angle=math.fsum(
                leaf.unresolved_solid_angle for leaf in leaves
            ),
            source_solid_angles=tuple(
                (source, math.fsum(source_totals[source]))
                for source in sorted(source_totals)
            ),
            topology_tokens=frozenset().union(
                *(leaf.topology_tokens for leaf in leaves)
            ),
            minimum_intensities=tuple(
                min(leaf.minimum_intensities[index] for leaf in leaves)
                for index in range(len(frequencies))
            ),
            maximum_intensities=tuple(
                max(leaf.maximum_intensities[index] for leaf in leaves)
                for index in range(len(frequencies))
            ),
            minimum_g=min(minimum_values) if minimum_values else None,
            maximum_g=max(maximum_values) if maximum_values else None,
            g_solid_angle=math.fsum(leaf.g_solid_angle for leaf in leaves),
            direction_solid_angle=math.fsum(
                leaf.direction_solid_angle for leaf in leaves
            ),
            directions=merged_directions,
            direction_span=_maximum_direction_span_vectors(merged_directions),
            weighted_log_g_variation=math.fsum(
                leaf.weighted_log_g_variation for leaf in leaves
            ),
            weighted_direction_variation=math.fsum(
                leaf.weighted_direction_variation for leaf in leaves
            ),
            maximum_depth=max(leaf.maximum_depth for leaf in leaves),
            converged=all(leaf.converged for leaf in leaves),
        )

    def visit(depth: int, cell_x: int, cell_y: int) -> _Leaf:
        nonlocal deepest
        deepest = max(deepest, depth)
        center_position = point(depth, cell_x, cell_y, 4, 4)
        center = evaluate(*center_position)
        child_positions = tuple(
            point(
                depth,
                cell_x,
                cell_y,
                2 + 4 * offset_x,
                2 + 4 * offset_y,
            )
            for offset_x, offset_y in child_order
        )
        children = tuple(evaluate(*position) for position in child_positions)
        guards = tuple(
            evaluate(*point(depth, cell_x, cell_y, offset_x, offset_y))
            for offset_x, offset_y in guard_offsets
        )
        probes = (center, *children, *guards)

        local_x0 = x0 + screen_width * cell_x / (1 << depth)
        local_x1 = x0 + screen_width * (cell_x + 1) / (1 << depth)
        local_y0 = y0 + screen_height * cell_y / (1 << depth)
        local_y1 = y0 + screen_height * (cell_y + 1) / (1 << depth)
        solid_angle = pinhole_solid_angle(local_x0, local_x1, local_y0, local_y1)
        if solid_angle < sys.float_info.min:
            raise AdaptiveSamplingError(
                "adaptive cell solid angle is subnormal and cannot be integrated "
                "with reliable relative coverage"
            )
        child_solid_angles = tuple(
            pinhole_solid_angle(
                x0
                + screen_width * (2 * cell_x + offset_x) / (1 << (depth + 1)),
                x0
                + screen_width
                * (2 * cell_x + offset_x + 1)
                / (1 << (depth + 1)),
                y0
                + screen_height * (2 * cell_y + offset_y) / (1 << (depth + 1)),
                y0
                + screen_height
                * (2 * cell_y + offset_y + 1)
                / (1 << (depth + 1)),
            )
            for offset_x, offset_y in child_order
        )
        coarse_flux = tuple(
            solid_angle * center.specific_intensities_nu[index]
            for index in range(len(frequencies))
        )
        fine_flux = tuple(
            math.fsum(
                child.specific_intensities_nu[index]
                * child_solid_angles[child_index]
                for child_index, child in enumerate(children)
            )
            for index in range(len(frequencies))
        )
        error = []
        for index in range(len(frequencies)):
            intensity_values = (
                center.specific_intensities_nu[index],
                *(child.specific_intensities_nu[index] for child in children),
                *(guard.specific_intensities_nu[index] for guard in guards),
            )
            ray_error = solid_angle * max(
                probe.absolute_errors_nu[index]
                for probe in probes
            )
            spatial_error = max(
                abs(fine_flux[index] - coarse_flux[index]),
                solid_angle * (max(intensity_values) - min(intensity_values)),
            )
            error.append(
                math.fsum((spatial_error, ray_error))
            )

        topology_tokens = frozenset(
            (sample.visible_source, sample.topology_signature)
            for sample in probes
        )
        topology_mixed = len(topology_tokens) > 1
        probe_minimum_intensities = tuple(
            min(sample.specific_intensities_nu[index] for sample in probes)
            for index in range(len(frequencies))
        )
        probe_maximum_intensities = tuple(
            max(sample.specific_intensities_nu[index] for sample in probes)
            for index in range(len(frequencies))
        )
        shifts = tuple(
            sample.frequency_shift_g
            for sample in probes
            if sample.frequency_shift_g is not None
        )
        log_g_span = (
            math.log(max(shifts)) - math.log(min(shifts))
            if len(shifts) >= 2
            else 0.0
        )
        direction_span = _maximum_direction_span(probes)
        area_fraction = solid_angle / pixel_solid_angle
        radiance_needs_refinement = any(
            error[index]
            > options.radiance_absolute_tolerances[index] * solid_angle
            + options.radiance_relative_tolerance * abs(fine_flux[index])
            for index in range(len(frequencies))
        )
        weighted_log_g = area_fraction * log_g_span
        weighted_direction = area_fraction * direction_span
        log_g_needs_refinement = (
            weighted_log_g > options.weighted_log_g_tolerance
        )
        direction_needs_refinement = (
            weighted_direction > options.weighted_direction_tolerance_rad
        )
        must_refine = depth < options.minimum_depth
        wants_refinement = (
            must_refine
            or radiance_needs_refinement
            or topology_mixed
            or log_g_needs_refinement
            or direction_needs_refinement
        )
        if wants_refinement and depth < options.maximum_depth:
            refined = merge(
                tuple(
                    visit(
                        depth + 1,
                        2 * cell_x + offset_x,
                        2 * cell_y + offset_y,
                    )
                    for offset_x, offset_y in child_order
                )
            )
            probe_minimum_g = min(shifts) if shifts else None
            probe_maximum_g = max(shifts) if shifts else None
            has_probe_direction = any(
                sample.escape_direction is not None for sample in probes
            )
            coverage_tolerance = _solid_angle_roundoff(solid_angle)
            evidence_was_lost = (
                not topology_tokens.issubset(refined.topology_tokens)
                or (
                    probe_minimum_g is not None
                    and (
                        refined.minimum_g is None
                        or refined.maximum_g is None
                        or probe_minimum_g < refined.minimum_g
                        or probe_maximum_g > refined.maximum_g
                    )
                )
                or (
                    not topology_mixed
                    and bool(shifts)
                    and refined.g_solid_angle
                    < solid_angle - coverage_tolerance
                )
                or (
                    direction_span > refined.direction_span
                )
                or (
                    not topology_mixed
                    and has_probe_direction
                    and refined.direction_solid_angle
                    < solid_angle - coverage_tolerance
                )
                or any(
                    probe_minimum_intensities[index]
                    < refined.minimum_intensities[index]
                    or probe_maximum_intensities[index]
                    > refined.maximum_intensities[index]
                    for index in range(len(frequencies))
                )
            )
            if evidence_was_lost:
                return replace(
                    refined,
                    error=tuple(
                        math.fsum(
                            (
                                refined.error[index],
                                solid_angle
                                * options.radiance_guard_ceilings[index],
                            )
                        )
                        for index in range(len(frequencies))
                    ),
                    unresolved_solid_angle=max(
                        refined.unresolved_solid_angle,
                        solid_angle,
                    ),
                    topology_tokens=refined.topology_tokens | topology_tokens,
                    minimum_intensities=tuple(
                        min(
                            refined.minimum_intensities[index],
                            probe_minimum_intensities[index],
                        )
                        for index in range(len(frequencies))
                    ),
                    maximum_intensities=tuple(
                        max(
                            refined.maximum_intensities[index],
                            probe_maximum_intensities[index],
                        )
                        for index in range(len(frequencies))
                    ),
                    g_solid_angle=(
                        solid_angle
                        if shifts and not topology_mixed
                        else refined.g_solid_angle
                    ),
                    direction_solid_angle=(
                        solid_angle
                        if has_probe_direction and not topology_mixed
                        else refined.direction_solid_angle
                    ),
                    minimum_g=(
                        min(
                            value
                            for value in (refined.minimum_g, probe_minimum_g)
                            if value is not None
                        )
                        if refined.minimum_g is not None
                        or probe_minimum_g is not None
                        else None
                    ),
                    maximum_g=(
                        max(
                            value
                            for value in (refined.maximum_g, probe_maximum_g)
                            if value is not None
                        )
                        if refined.maximum_g is not None
                        or probe_maximum_g is not None
                        else None
                    ),
                    direction_span=max(refined.direction_span, direction_span),
                    directions=tuple(
                        sorted(
                            set(refined.directions)
                            | {
                                sample.escape_direction
                                for sample in probes
                                if sample.escape_direction is not None
                            }
                        )
                    ),
                )
            return refined

        unresolved = solid_angle if topology_mixed else 0.0
        if topology_mixed:
            for index, ceiling in enumerate(options.radiance_guard_ceilings):
                error[index] += solid_angle * ceiling
        local_converged = not (
            radiance_needs_refinement
            or log_g_needs_refinement
            or direction_needs_refinement
        )
        source_totals: dict[str, list[float]] = {}
        for child_index, sample in enumerate(children):
            source_totals.setdefault(sample.visible_source, []).append(
                child_solid_angles[child_index]
            )
        g_solid_angle = math.fsum(
            child_solid_angles[child_index]
            for child_index, sample in enumerate(children)
            if sample.frequency_shift_g is not None
        )
        direction_solid_angle = math.fsum(
            child_solid_angles[child_index]
            for child_index, sample in enumerate(children)
            if sample.escape_direction is not None
        )
        probe_directions = tuple(
            sorted(
                {
                    sample.escape_direction
                    for sample in probes
                    if sample.escape_direction is not None
                }
            )
        )
        return _Leaf(
            flux=fine_flux,
            error=tuple(error),
            solid_angle=solid_angle,
            unresolved_solid_angle=unresolved,
            source_solid_angles=tuple(
                (source, math.fsum(source_totals[source]))
                for source in sorted(source_totals)
            ),
            topology_tokens=topology_tokens,
            minimum_intensities=probe_minimum_intensities,
            maximum_intensities=probe_maximum_intensities,
            minimum_g=min(shifts) if shifts else None,
            maximum_g=max(shifts) if shifts else None,
            g_solid_angle=g_solid_angle,
            direction_solid_angle=direction_solid_angle,
            directions=probe_directions,
            direction_span=direction_span,
            weighted_log_g_variation=weighted_log_g,
            weighted_direction_variation=weighted_direction,
            maximum_depth=depth,
            converged=local_converged,
        )

    root = visit(0, 0, 0)
    if not math.isclose(
        root.solid_angle,
        pixel_solid_angle,
        rel_tol=2.0e-13,
        abs_tol=_solid_angle_roundoff(pixel_solid_angle),
    ):
        raise AdaptiveSamplingError("adaptive leaves do not cover the pixel")
    unresolved_fraction = root.unresolved_solid_angle / pixel_solid_angle
    all_samples = tuple(cache.values())
    all_shifts = tuple(
        sample.frequency_shift_g
        for sample in all_samples
        if sample.frequency_shift_g is not None
    )
    # Guard probes are evidence that a cell may contain another source, but
    # they are not quadrature area.  At maximum depth it is therefore possible
    # (and legitimate) for a guard to carry g or an escape direction while all
    # four child centres carry neither.  Preserve that situation through the
    # unresolved/error diagnostics without publishing a finite bound whose
    # declared coverage is zero.
    has_frequency_shift_coverage = root.g_solid_angle > 0.0
    global_minimum_g = (
        min(all_shifts) if all_shifts and has_frequency_shift_coverage else None
    )
    global_maximum_g = (
        max(all_shifts) if all_shifts and has_frequency_shift_coverage else None
    )
    global_log_g_span = (
        math.log(global_maximum_g) - math.log(global_minimum_g)
        if global_minimum_g is not None
        and global_maximum_g is not None
        and len(all_shifts) >= 2
        else 0.0
    )
    has_escape_direction_coverage = root.direction_solid_angle > 0.0
    global_direction_span = (
        _maximum_direction_span(all_samples)
        if has_escape_direction_coverage
        else 0.0
    )
    weighted_log_g_variation = (
        min(root.g_solid_angle, pixel_solid_angle)
        / pixel_solid_angle
        * global_log_g_span
    )
    weighted_direction_variation = (
        min(root.direction_solid_angle, pixel_solid_angle)
        / pixel_solid_angle
        * global_direction_span
    )
    audits = tuple(sample.convergence_audit for sample in all_samples)

    def audit_maximum(name: str) -> float:
        return max(float(getattr(audit, name)) for audit in audits)

    all_ray_gates_passed = all(audit.ray_gate_passed for audit in audits)
    all_source_gates_passed = all(audit.source_gate_passed for audit in audits)
    all_transfer_gates_passed = all(audit.transfer_gate_passed for audit in audits)
    spectral_error_converged = all(
        root.error[index]
        <= options.radiance_absolute_tolerances[index] * pixel_solid_angle
        + options.radiance_relative_tolerance * abs(root.flux[index])
        for index in range(len(frequencies))
    )
    converged = (
        root.converged
        and spectral_error_converged
        and unresolved_fraction
        <= options.unresolved_solid_angle_fraction_tolerance
        and weighted_log_g_variation
        <= options.weighted_log_g_tolerance
        and weighted_direction_variation
        <= options.weighted_direction_tolerance_rad
        and all_ray_gates_passed
        and all_source_gates_passed
        and all_transfer_gates_passed
    )
    means = tuple(value / pixel_solid_angle for value in root.flux)
    if any(not math.isfinite(value) or value < 0.0 for value in (*means, *root.flux)):
        raise AdaptiveSamplingError("adaptive pixel produced invalid radiance")
    return AdaptivePixelResult(
        observer_frequencies_hz=frequencies,
        integrated_specific_intensity_nu_sr=root.flux,
        mean_specific_intensities_nu=means,
        estimated_absolute_errors_nu_sr=root.error,
        pixel_solid_angle_sr=pixel_solid_angle,
        sample_count=len(cache),
        maximum_depth_reached=deepest,
        unresolved_solid_angle_sr=root.unresolved_solid_angle,
        frequency_shift_solid_angle_sr=min(
            root.g_solid_angle,
            pixel_solid_angle,
        ),
        escape_direction_solid_angle_sr=min(
            root.direction_solid_angle,
            pixel_solid_angle,
        ),
        source_solid_angles_sr=root.source_solid_angles,
        minimum_frequency_shift_g=global_minimum_g,
        maximum_frequency_shift_g=global_maximum_g,
        maximum_escape_direction_span_rad=global_direction_span,
        weighted_log_g_variation=weighted_log_g_variation,
        weighted_escape_direction_variation_rad=(
            weighted_direction_variation
        ),
        maximum_null_residual=audit_maximum("maximum_null_residual"),
        maximum_metric_interpolation_error=audit_maximum(
            "maximum_metric_interpolation_error"
        ),
        maximum_terminal_event_difference_m=audit_maximum(
            "terminal_event_difference_m"
        ),
        maximum_terminal_covector_relative_difference=audit_maximum(
            "terminal_covector_relative_difference"
        ),
        maximum_disk_radius_difference_m=audit_maximum(
            "disk_radius_difference_m"
        ),
        maximum_relative_g_difference=audit_maximum("relative_g_difference"),
        maximum_surface_bracket_affine_width=audit_maximum(
            "surface_bracket_affine_width"
        ),
        maximum_accepted_steps=max(audit.accepted_steps for audit in audits),
        maximum_rejected_steps=max(audit.rejected_steps for audit in audits),
        all_ray_gates_passed=all_ray_gates_passed,
        all_source_gates_passed=all_source_gates_passed,
        all_transfer_gates_passed=all_transfer_gates_passed,
        converged=converged,
    )


__all__ = (
    "AdaptivePixelOptions",
    "AdaptivePixelResult",
    "AdaptiveSamplingError",
    "RayConvergenceAudit",
    "SCIENTIFIC_STATUS",
    "STENCIL_VERSION",
    "SpectralRaySample",
    "SpectralRaySampler",
    "integrate_spectral_pixel",
    "pinhole_solid_angle",
)
