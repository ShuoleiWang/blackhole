"""Fixed binary pixel records for linear scientific spectral frames.

This module owns only the per-pixel little-endian ABI.  Tile receipts, frame
manifests, independent verification, and display transforms are separate
layers.  The stored adaptive error is explicitly a finite-stencil estimate,
not a rigorous bound against arbitrarily narrow unsampled caustics.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

from offline.adaptive_frame import AdaptivePixelOptions, AdaptivePixelResult


PIXEL_LAYOUT_ID: Final = "blackhole.scientific-spectral-pixel/le-f64-v1"
FIXED_RECORD_BYTES: Final = 160

CONVERGED_OVERALL: Final = 1 << 0
CONVERGED_SPECTRAL_ESTIMATE: Final = 1 << 1
CONVERGED_UNRESOLVED_COVERAGE: Final = 1 << 2
CONVERGED_WEIGHTED_G: Final = 1 << 3
CONVERGED_ESCAPE_DIRECTION: Final = 1 << 4
CONVERGED_ALL_RAYS: Final = 1 << 5
CONVERGED_ALL_SOURCES: Final = 1 << 6
CONVERGED_ALL_TRANSFERS: Final = 1 << 7
HAS_FREQUENCY_SHIFT: Final = 1 << 8
HAS_ESCAPE_DIRECTION: Final = 1 << 9
REQUIRED_CONVERGENCE_MASK: Final = (1 << 8) - 1
ALLOWED_CONVERGENCE_MASK: Final = (1 << 10) - 1

SOURCE_DISK: Final = 1 << 0
SOURCE_CAPTURED_BOUNDARY: Final = 1 << 1
SOURCE_ESCAPED_BOUNDARY: Final = 1 << 2
ALLOWED_SOURCE_MASK: Final = (1 << 3) - 1

SCIENTIFIC_STATUS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "classification": "linear observer-frame scalar spectral pixel record",
        "quantity": "spectral-specific-intensity-I_nu",
        "units": "W m^-2 sr^-1 Hz^-1",
        "errorSemantics": "finite-stencil estimated absolute error",
        "isDisplayImage": False,
        "isRigorousCausticErrorBound": False,
        "isPolarized": False,
    }
)


class SpectralFrameError(RuntimeError):
    """Raised when a spectral pixel violates its versioned binary contract."""


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


def _is_positive_zero(value: float) -> bool:
    return value == 0.0 and math.copysign(1.0, value) > 0.0


def _solid_angle_roundoff(value: float) -> float:
    return 16.0 * math.ulp(value)


@dataclass(frozen=True, slots=True)
class SpectralPixelLayout:
    """One job-fixed frequency layout and its exact binary64 record struct."""

    observer_frequencies_hz: tuple[float, ...]

    def __post_init__(self) -> None:
        frequencies = _finite_tuple(
            self.observer_frequencies_hz,
            "observer_frequencies_hz",
        )
        if not frequencies or any(value <= 0.0 for value in frequencies):
            raise ValueError("observer frequencies must be non-empty and positive")
        if any(right <= left for left, right in zip(frequencies, frequencies[1:])):
            raise ValueError("observer frequencies must increase strictly")
        object.__setattr__(self, "observer_frequencies_hz", frequencies)

    @property
    def frequency_count(self) -> int:
        return len(self.observer_frequencies_hz)

    @property
    def record_bytes(self) -> int:
        return 16 * self.frequency_count + FIXED_RECORD_BYTES

    @property
    def record_struct(self) -> struct.Struct:
        return struct.Struct(
            "<" + f"{2 * self.frequency_count + 17}d" + "IIIIHHI"
        )

    def descriptor(self) -> Mapping[str, Any]:
        base = 16 * self.frequency_count
        return {
            "endianness": "little",
            "errorSemantics": "finite-stencil-estimate-not-rigorous-bound",
            "floatEncoding": "IEEE-754-binary64",
            "frequencyCount": self.frequency_count,
            "id": PIXEL_LAYOUT_ID,
            "meanErrorOffsetBytes": 8 * self.frequency_count,
            "meanIntensityOffsetBytes": 0,
            "pixelSolidAngleOffsetBytes": base,
            "recordBytes": self.record_bytes,
            "tailBytes": FIXED_RECORD_BYTES,
        }


@dataclass(frozen=True, slots=True)
class ScientificSpectralPixelRecord:
    mean_specific_intensities_nu: tuple[float, ...]
    mean_estimated_absolute_errors_nu: tuple[float, ...]
    pixel_solid_angle_sr: float
    disk_coverage_fraction: float
    captured_boundary_coverage_fraction: float
    escaped_boundary_coverage_fraction: float
    unresolved_solid_angle_fraction: float
    minimum_frequency_shift_g: float
    maximum_frequency_shift_g: float
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
    sample_count: int
    maximum_accepted_steps: int
    maximum_rejected_steps: int
    convergence_mask: int
    maximum_depth_reached: int
    source_mask: int
    reserved: int = 0

    def __post_init__(self) -> None:
        means = _finite_tuple(
            self.mean_specific_intensities_nu,
            "mean_specific_intensities_nu",
        )
        errors = _finite_tuple(
            self.mean_estimated_absolute_errors_nu,
            "mean_estimated_absolute_errors_nu",
        )
        if not means or len(errors) != len(means):
            raise ValueError("spectral pixel means and errors need equal length")
        if any(value < 0.0 for value in (*means, *errors)):
            raise ValueError("spectral pixel radiance and errors must be non-negative")
        scalar_names = (
            "pixel_solid_angle_sr",
            "disk_coverage_fraction",
            "captured_boundary_coverage_fraction",
            "escaped_boundary_coverage_fraction",
            "unresolved_solid_angle_fraction",
            "minimum_frequency_shift_g",
            "maximum_frequency_shift_g",
            "maximum_escape_direction_span_rad",
            "weighted_log_g_variation",
            "weighted_escape_direction_variation_rad",
            "maximum_null_residual",
            "maximum_metric_interpolation_error",
            "maximum_terminal_event_difference_m",
            "maximum_terminal_covector_relative_difference",
            "maximum_disk_radius_difference_m",
            "maximum_relative_g_difference",
            "maximum_surface_bracket_affine_width",
        )
        values = {
            name: _finite_number(getattr(self, name), name)
            for name in scalar_names
        }
        if values["pixel_solid_angle_sr"] <= 0.0:
            raise ValueError("pixel solid angle must be positive")
        coverage_names = (
            "disk_coverage_fraction",
            "captured_boundary_coverage_fraction",
            "escaped_boundary_coverage_fraction",
            "unresolved_solid_angle_fraction",
        )
        if any(values[name] < 0.0 or values[name] > 1.0 for name in coverage_names):
            raise ValueError("spectral pixel coverage fractions must lie in [0, 1]")
        if not math.isclose(
            math.fsum(values[name] for name in coverage_names[:3]),
            1.0,
            rel_tol=0.0,
            abs_tol=3.0e-13,
        ):
            raise ValueError("spectral pixel source coverage must sum to one")
        non_negative_names = scalar_names[7:]
        if any(values[name] < 0.0 for name in non_negative_names):
            raise ValueError("spectral pixel diagnostics must be non-negative")
        integer_limits = {
            "sample_count": 0xFFFFFFFF,
            "maximum_accepted_steps": 0xFFFFFFFF,
            "maximum_rejected_steps": 0xFFFFFFFF,
            "convergence_mask": 0xFFFFFFFF,
            "maximum_depth_reached": 0xFFFF,
            "source_mask": 0xFFFF,
            "reserved": 0xFFFFFFFF,
        }
        for name, limit in integer_limits.items():
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value > limit:
                raise ValueError(f"{name} does not fit its unsigned ABI field")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if self.maximum_accepted_steps < 1:
            raise ValueError("published spectral pixels require an accepted ray step")
        if self.convergence_mask & ~ALLOWED_CONVERGENCE_MASK:
            raise ValueError("convergence mask contains reserved bits")
        if self.convergence_mask & REQUIRED_CONVERGENCE_MASK != REQUIRED_CONVERGENCE_MASK:
            raise ValueError("spectral pixel is missing a required convergence gate")
        if self.source_mask & ~ALLOWED_SOURCE_MASK:
            raise ValueError("source mask contains reserved bits")
        expected_source_mask = 0
        for fraction, bit in (
            (values["disk_coverage_fraction"], SOURCE_DISK),
            (values["captured_boundary_coverage_fraction"], SOURCE_CAPTURED_BOUNDARY),
            (values["escaped_boundary_coverage_fraction"], SOURCE_ESCAPED_BOUNDARY),
        ):
            if fraction > 0.0:
                expected_source_mask |= bit
        if self.source_mask != expected_source_mask:
            raise ValueError("source mask disagrees with coverage fractions")
        if (
            self.source_mask == SOURCE_CAPTURED_BOUNDARY
            and any(not _is_positive_zero(value) for value in means)
        ):
            raise ValueError("pure captured-boundary pixels must be exactly black")
        has_g = bool(self.convergence_mask & HAS_FREQUENCY_SHIFT)
        if has_g:
            if (
                values["minimum_frequency_shift_g"] <= 0.0
                or values["maximum_frequency_shift_g"]
                < values["minimum_frequency_shift_g"]
            ):
                raise ValueError("frequency-shift bounds are invalid")
        elif not (
            _is_positive_zero(values["minimum_frequency_shift_g"])
            and _is_positive_zero(values["maximum_frequency_shift_g"])
        ):
            raise ValueError("absent frequency shift must use positive-zero sentinels")
        has_direction = bool(self.convergence_mask & HAS_ESCAPE_DIRECTION)
        if bool(self.source_mask & SOURCE_DISK) != has_g:
            raise ValueError("disk coverage and frequency-shift presence disagree")
        if bool(self.source_mask & SOURCE_ESCAPED_BOUNDARY) != has_direction:
            raise ValueError("escape coverage and direction presence disagree")
        if not has_g and not (
            _is_positive_zero(values["weighted_log_g_variation"])
            and _is_positive_zero(values["maximum_disk_radius_difference_m"])
            and _is_positive_zero(values["maximum_relative_g_difference"])
        ):
            raise ValueError("absent frequency shift requires zero g diagnostics")
        if not has_direction and not (
            _is_positive_zero(values["maximum_escape_direction_span_rad"])
            and _is_positive_zero(
                values["weighted_escape_direction_variation_rad"]
            )
        ):
            raise ValueError("absent escape direction requires zero diagnostics")
        if has_g:
            expected_weighted_log_g = values["disk_coverage_fraction"] * (
                math.log(values["maximum_frequency_shift_g"])
                - math.log(values["minimum_frequency_shift_g"])
            )
            if not math.isclose(
                values["weighted_log_g_variation"],
                expected_weighted_log_g,
                rel_tol=2.0e-12,
                abs_tol=16.0 * math.ulp(expected_weighted_log_g),
            ):
                raise ValueError("weighted g variation disagrees with its bounds")
        if values["maximum_escape_direction_span_rad"] > math.pi:
            raise ValueError("escape direction span may not exceed pi")
        if has_direction:
            expected_weighted_direction = (
                values["escaped_boundary_coverage_fraction"]
                * values["maximum_escape_direction_span_rad"]
            )
            if not math.isclose(
                values["weighted_escape_direction_variation_rad"],
                expected_weighted_direction,
                rel_tol=2.0e-12,
                abs_tol=16.0 * math.ulp(expected_weighted_direction),
            ):
                raise ValueError(
                    "weighted escape-direction variation disagrees with its span"
                )
        if self.reserved != 0:
            raise ValueError("spectral pixel reserved field must be zero")
        object.__setattr__(self, "mean_specific_intensities_nu", means)
        object.__setattr__(self, "mean_estimated_absolute_errors_nu", errors)
        for name, value in values.items():
            object.__setattr__(self, name, value)


def _source_coverage(result: AdaptivePixelResult) -> dict[str, float]:
    allowed = {"disk", "captured-boundary", "escaped-boundary"}
    source_areas = dict(result.source_solid_angles_sr)
    unknown = set(source_areas) - allowed
    if unknown:
        raise ValueError(f"unsupported scientific frame source classes: {sorted(unknown)!r}")
    total_area = math.fsum(source_areas.values())
    if not math.isclose(
        total_area,
        result.pixel_solid_angle_sr,
        rel_tol=2.0e-13,
        abs_tol=_solid_angle_roundoff(result.pixel_solid_angle_sr),
    ):
        raise ValueError("adaptive source areas do not cover the pixel")
    return {
        source: source_areas.get(source, 0.0) / total_area
        for source in sorted(allowed)
    }


def pack_adaptive_pixel(
    layout: SpectralPixelLayout,
    result: AdaptivePixelResult,
    options: AdaptivePixelOptions,
) -> bytes:
    """Pack one fully converged adaptive pixel into the v1 binary ABI."""

    if not isinstance(layout, SpectralPixelLayout):
        raise TypeError("layout must be a SpectralPixelLayout")
    if not isinstance(result, AdaptivePixelResult):
        raise TypeError("result must be an AdaptivePixelResult")
    if not isinstance(options, AdaptivePixelOptions):
        raise TypeError("options must be an AdaptivePixelOptions")
    if result.observer_frequencies_hz != layout.observer_frequencies_hz:
        raise ValueError("adaptive pixel frequencies disagree with the layout")
    if len(options.radiance_absolute_tolerances) != layout.frequency_count:
        raise ValueError("adaptive options disagree with the layout frequency count")
    coverage = _source_coverage(result)
    solid_angle = result.pixel_solid_angle_sr
    mean_errors = tuple(
        value / solid_angle for value in result.estimated_absolute_errors_nu_sr
    )
    unresolved_fraction = result.unresolved_solid_angle_sr / solid_angle
    frequency_shift_fraction = result.frequency_shift_solid_angle_sr / solid_angle
    escape_direction_fraction = result.escape_direction_solid_angle_sr / solid_angle

    spectral_pass = all(
        error
        <= options.radiance_absolute_tolerances[index]
        + options.radiance_relative_tolerance
        * result.mean_specific_intensities_nu[index]
        for index, error in enumerate(mean_errors)
    )
    unresolved_pass = (
        unresolved_fraction
        <= options.unresolved_solid_angle_fraction_tolerance
    )
    weighted_g_pass = (
        result.weighted_log_g_variation <= options.weighted_log_g_tolerance
    )
    direction_pass = (
        result.weighted_escape_direction_variation_rad
        <= options.weighted_direction_tolerance_rad
    )
    convergence_mask = 0
    for passed, bit in (
        (result.converged, CONVERGED_OVERALL),
        (spectral_pass, CONVERGED_SPECTRAL_ESTIMATE),
        (unresolved_pass, CONVERGED_UNRESOLVED_COVERAGE),
        (weighted_g_pass, CONVERGED_WEIGHTED_G),
        (direction_pass, CONVERGED_ESCAPE_DIRECTION),
        (result.all_ray_gates_passed, CONVERGED_ALL_RAYS),
        (result.all_source_gates_passed, CONVERGED_ALL_SOURCES),
        (result.all_transfer_gates_passed, CONVERGED_ALL_TRANSFERS),
    ):
        if passed:
            convergence_mask |= bit
    if convergence_mask & REQUIRED_CONVERGENCE_MASK != REQUIRED_CONVERGENCE_MASK:
        raise SpectralFrameError("adaptive pixel has not passed every publication gate")

    disk_fraction = coverage["disk"]
    escaped_fraction = coverage["escaped-boundary"]
    coverage_tolerance = 3.0e-13
    if not math.isclose(
        disk_fraction,
        frequency_shift_fraction,
        rel_tol=0.0,
        abs_tol=coverage_tolerance,
    ):
        raise SpectralFrameError("disk and frequency-shift coverage disagree")
    if not math.isclose(
        escaped_fraction,
        escape_direction_fraction,
        rel_tol=0.0,
        abs_tol=coverage_tolerance,
    ):
        raise SpectralFrameError("escape and direction coverage disagree")

    minimum_g = result.minimum_frequency_shift_g
    maximum_g = result.maximum_frequency_shift_g
    if disk_fraction > 0.0:
        if minimum_g is None or maximum_g is None:
            raise SpectralFrameError("disk coverage requires frequency-shift bounds")
        convergence_mask |= HAS_FREQUENCY_SHIFT
    else:
        if minimum_g is not None or maximum_g is not None:
            raise SpectralFrameError("non-disk pixel may not carry frequency-shift bounds")
        minimum_g = 0.0
        maximum_g = 0.0
    if escaped_fraction > 0.0:
        convergence_mask |= HAS_ESCAPE_DIRECTION

    source_mask = 0
    for fraction, bit in (
        (disk_fraction, SOURCE_DISK),
        (coverage["captured-boundary"], SOURCE_CAPTURED_BOUNDARY),
        (escaped_fraction, SOURCE_ESCAPED_BOUNDARY),
    ):
        if fraction > 0.0:
            source_mask |= bit

    record = ScientificSpectralPixelRecord(
        mean_specific_intensities_nu=result.mean_specific_intensities_nu,
        mean_estimated_absolute_errors_nu=mean_errors,
        pixel_solid_angle_sr=solid_angle,
        disk_coverage_fraction=disk_fraction,
        captured_boundary_coverage_fraction=coverage["captured-boundary"],
        escaped_boundary_coverage_fraction=escaped_fraction,
        unresolved_solid_angle_fraction=unresolved_fraction,
        minimum_frequency_shift_g=minimum_g,
        maximum_frequency_shift_g=maximum_g,
        maximum_escape_direction_span_rad=(
            result.maximum_escape_direction_span_rad
        ),
        weighted_log_g_variation=result.weighted_log_g_variation,
        weighted_escape_direction_variation_rad=(
            result.weighted_escape_direction_variation_rad
        ),
        maximum_null_residual=result.maximum_null_residual,
        maximum_metric_interpolation_error=(
            result.maximum_metric_interpolation_error
        ),
        maximum_terminal_event_difference_m=(
            result.maximum_terminal_event_difference_m
        ),
        maximum_terminal_covector_relative_difference=(
            result.maximum_terminal_covector_relative_difference
        ),
        maximum_disk_radius_difference_m=(
            result.maximum_disk_radius_difference_m
        ),
        maximum_relative_g_difference=result.maximum_relative_g_difference,
        maximum_surface_bracket_affine_width=(
            result.maximum_surface_bracket_affine_width
        ),
        sample_count=result.sample_count,
        maximum_accepted_steps=result.maximum_accepted_steps,
        maximum_rejected_steps=result.maximum_rejected_steps,
        convergence_mask=convergence_mask,
        maximum_depth_reached=result.maximum_depth_reached,
        source_mask=source_mask,
    )
    return pack_spectral_pixel(layout, record)


def pack_spectral_pixel(
    layout: SpectralPixelLayout,
    record: ScientificSpectralPixelRecord,
) -> bytes:
    if not isinstance(layout, SpectralPixelLayout):
        raise TypeError("layout must be a SpectralPixelLayout")
    if not isinstance(record, ScientificSpectralPixelRecord):
        raise TypeError("record must be a ScientificSpectralPixelRecord")
    if len(record.mean_specific_intensities_nu) != layout.frequency_count:
        raise ValueError("record intensity count disagrees with layout")
    values = (
        *record.mean_specific_intensities_nu,
        *record.mean_estimated_absolute_errors_nu,
        record.pixel_solid_angle_sr,
        record.disk_coverage_fraction,
        record.captured_boundary_coverage_fraction,
        record.escaped_boundary_coverage_fraction,
        record.unresolved_solid_angle_fraction,
        record.minimum_frequency_shift_g,
        record.maximum_frequency_shift_g,
        record.maximum_escape_direction_span_rad,
        record.weighted_log_g_variation,
        record.weighted_escape_direction_variation_rad,
        record.maximum_null_residual,
        record.maximum_metric_interpolation_error,
        record.maximum_terminal_event_difference_m,
        record.maximum_terminal_covector_relative_difference,
        record.maximum_disk_radius_difference_m,
        record.maximum_relative_g_difference,
        record.maximum_surface_bracket_affine_width,
        record.sample_count,
        record.maximum_accepted_steps,
        record.maximum_rejected_steps,
        record.convergence_mask,
        record.maximum_depth_reached,
        record.source_mask,
        record.reserved,
    )
    payload = layout.record_struct.pack(*values)
    if len(payload) != layout.record_bytes:
        raise AssertionError("packed spectral pixel has the wrong byte length")
    return payload


def unpack_spectral_pixel(
    layout: SpectralPixelLayout,
    payload: bytes | bytearray | memoryview,
) -> ScientificSpectralPixelRecord:
    if not isinstance(layout, SpectralPixelLayout):
        raise TypeError("layout must be a SpectralPixelLayout")
    raw = bytes(payload)
    if len(raw) != layout.record_bytes:
        raise SpectralFrameError("spectral pixel payload has the wrong byte length")
    values = layout.record_struct.unpack(raw)
    count = layout.frequency_count
    fixed = values[2 * count : 2 * count + 17]
    integers = values[2 * count + 17 :]
    try:
        return ScientificSpectralPixelRecord(
            mean_specific_intensities_nu=tuple(values[:count]),
            mean_estimated_absolute_errors_nu=tuple(values[count : 2 * count]),
            pixel_solid_angle_sr=fixed[0],
            disk_coverage_fraction=fixed[1],
            captured_boundary_coverage_fraction=fixed[2],
            escaped_boundary_coverage_fraction=fixed[3],
            unresolved_solid_angle_fraction=fixed[4],
            minimum_frequency_shift_g=fixed[5],
            maximum_frequency_shift_g=fixed[6],
            maximum_escape_direction_span_rad=fixed[7],
            weighted_log_g_variation=fixed[8],
            weighted_escape_direction_variation_rad=fixed[9],
            maximum_null_residual=fixed[10],
            maximum_metric_interpolation_error=fixed[11],
            maximum_terminal_event_difference_m=fixed[12],
            maximum_terminal_covector_relative_difference=fixed[13],
            maximum_disk_radius_difference_m=fixed[14],
            maximum_relative_g_difference=fixed[15],
            maximum_surface_bracket_affine_width=fixed[16],
            sample_count=integers[0],
            maximum_accepted_steps=integers[1],
            maximum_rejected_steps=integers[2],
            convergence_mask=integers[3],
            maximum_depth_reached=integers[4],
            source_mask=integers[5],
            reserved=integers[6],
        )
    except (TypeError, ValueError) as error:
        raise SpectralFrameError(f"spectral pixel payload is invalid: {error}") from error


__all__ = (
    "ALLOWED_CONVERGENCE_MASK",
    "ALLOWED_SOURCE_MASK",
    "FIXED_RECORD_BYTES",
    "HAS_ESCAPE_DIRECTION",
    "HAS_FREQUENCY_SHIFT",
    "PIXEL_LAYOUT_ID",
    "REQUIRED_CONVERGENCE_MASK",
    "SCIENTIFIC_STATUS",
    "ScientificSpectralPixelRecord",
    "SpectralFrameError",
    "SpectralPixelLayout",
    "pack_adaptive_pixel",
    "pack_spectral_pixel",
    "unpack_spectral_pixel",
)
