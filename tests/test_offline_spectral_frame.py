from __future__ import annotations

import math
import struct
import unittest

from offline.adaptive_frame import (
    AdaptivePixelOptions,
    RayConvergenceAudit,
    SpectralRaySample,
    integrate_spectral_pixel,
)
from offline.spectral_frame import (
    HAS_ESCAPE_DIRECTION,
    HAS_FREQUENCY_SHIFT,
    REQUIRED_CONVERGENCE_MASK,
    SOURCE_DISK,
    SOURCE_ESCAPED_BOUNDARY,
    SpectralFrameError,
    SpectralPixelLayout,
    pack_adaptive_pixel,
    unpack_spectral_pixel,
)


def _options(count: int) -> AdaptivePixelOptions:
    return AdaptivePixelOptions(
        maximum_depth=0,
        radiance_absolute_tolerances=(1.0e-12,) * count,
        radiance_relative_tolerance=1.0e-8,
        radiance_guard_ceilings=(100.0,) * count,
        weighted_log_g_tolerance=1.0,
        weighted_direction_tolerance_rad=1.0,
    )


def _pixel(
    frequencies: tuple[float, ...],
    *,
    source: str,
):
    values = (
        (0.0,) * len(frequencies)
        if source == "captured-boundary"
        else tuple(1.0 + index for index in range(len(frequencies)))
    )
    shift = 1.2 if source == "disk" else None
    direction = (0.0, 0.0, 1.0) if source == "escaped-boundary" else None
    return integrate_spectral_pixel(
        lambda _x, _y, _bins: SpectralRaySample(
            specific_intensities_nu=values,
            absolute_errors_nu=(0.0,) * len(frequencies),
            visible_source=source,
            topology_signature=source,
            frequency_shift_g=shift,
            escape_direction=direction,
            ray_converged=True,
            convergence_audit=RayConvergenceAudit(
                accepted_steps=1,
                ray_gate_passed=True,
                source_gate_passed=True,
                transfer_gate_passed=True,
            ),
        ),
        frequencies,
        x_min=-0.1,
        x_max=0.1,
        y_min=-0.1,
        y_max=0.1,
        options=_options(len(frequencies)),
    )


class OfflineSpectralFrameTests(unittest.TestCase):
    def test_layout_sizes_and_round_trip_for_multiple_frequency_counts(self) -> None:
        for count in (1, 3, 17):
            frequencies = tuple(1.0e14 * (index + 1) for index in range(count))
            layout = SpectralPixelLayout(frequencies)
            self.assertEqual(layout.record_bytes, 16 * count + 160)
            self.assertEqual(layout.record_struct.size, layout.record_bytes)
            result = _pixel(frequencies, source="disk")
            payload = pack_adaptive_pixel(layout, result, _options(count))
            record = unpack_spectral_pixel(layout, payload)
            self.assertEqual(record.mean_specific_intensities_nu, result.mean_specific_intensities_nu)
            self.assertEqual(record.source_mask, SOURCE_DISK)
            self.assertEqual(
                record.convergence_mask & REQUIRED_CONVERGENCE_MASK,
                REQUIRED_CONVERGENCE_MASK,
            )
            self.assertTrue(record.convergence_mask & HAS_FREQUENCY_SHIFT)

    def test_escape_direction_flag_and_positive_zero_g_sentinel(self) -> None:
        frequencies = (4.0e14,)
        layout = SpectralPixelLayout(frequencies)
        payload = pack_adaptive_pixel(
            layout,
            _pixel(frequencies, source="escaped-boundary"),
            _options(1),
        )
        record = unpack_spectral_pixel(layout, payload)
        self.assertEqual(record.source_mask, SOURCE_ESCAPED_BOUNDARY)
        self.assertTrue(record.convergence_mask & HAS_ESCAPE_DIRECTION)
        self.assertFalse(record.convergence_mask & HAS_FREQUENCY_SHIFT)
        self.assertEqual(math.copysign(1.0, record.minimum_frequency_shift_g), 1.0)
        self.assertEqual(math.copysign(1.0, record.maximum_frequency_shift_g), 1.0)

    def test_fixed_offsets_match_the_declared_abi(self) -> None:
        frequencies = (1.0, 2.0, 3.0)
        layout = SpectralPixelLayout(frequencies)
        payload = pack_adaptive_pixel(
            layout,
            _pixel(frequencies, source="disk"),
            _options(3),
        )
        base = 16 * len(frequencies)
        solid_angle = struct.unpack_from("<d", payload, base)[0]
        convergence_mask = struct.unpack_from("<I", payload, base + 148)[0]
        reserved = struct.unpack_from("<I", payload, base + 156)[0]
        self.assertGreater(solid_angle, 0.0)
        self.assertEqual(convergence_mask & REQUIRED_CONVERGENCE_MASK, REQUIRED_CONVERGENCE_MASK)
        self.assertEqual(reserved, 0)

    def test_wrong_layout_and_unconverged_pixel_are_rejected(self) -> None:
        frequencies = (1.0,)
        result = _pixel(frequencies, source="disk")
        with self.assertRaises(ValueError):
            pack_adaptive_pixel(
                SpectralPixelLayout((1.0, 2.0)),
                result,
                _options(2),
            )

        edge = integrate_spectral_pixel(
            lambda x_value, _y, _bins: SpectralRaySample(
                specific_intensities_nu=(
                    (1.0,) if x_value < 0.07 else (0.0,)
                ),
                absolute_errors_nu=(0.0,),
                visible_source="disk" if x_value < 0.07 else "captured-boundary",
                topology_signature="left" if x_value < 0.07 else "right",
                frequency_shift_g=1.0 if x_value < 0.07 else None,
                ray_converged=True,
                convergence_audit=RayConvergenceAudit(
                    accepted_steps=1,
                    ray_gate_passed=True,
                    source_gate_passed=True,
                    transfer_gate_passed=True,
                ),
            ),
            frequencies,
            x_min=-0.1,
            x_max=0.1,
            y_min=-0.1,
            y_max=0.1,
            options=_options(1),
        )
        self.assertFalse(edge.converged)
        with self.assertRaises(SpectralFrameError):
            pack_adaptive_pixel(
                SpectralPixelLayout(frequencies),
                edge,
                _options(1),
            )

    def test_binary_tampering_is_rejected(self) -> None:
        frequencies = (1.0, 2.0, 3.0)
        layout = SpectralPixelLayout(frequencies)
        payload = bytearray(
            pack_adaptive_pixel(
                layout,
                _pixel(frequencies, source="disk"),
                _options(3),
            )
        )
        base = 16 * len(frequencies)

        mutations = []
        non_finite = bytearray(payload)
        struct.pack_into("<d", non_finite, 0, math.nan)
        mutations.append(non_finite)
        missing_gate = bytearray(payload)
        mask = struct.unpack_from("<I", missing_gate, base + 148)[0]
        struct.pack_into("<I", missing_gate, base + 148, mask & ~1)
        mutations.append(missing_gate)
        reserved = bytearray(payload)
        struct.pack_into("<I", reserved, base + 156, 1)
        mutations.append(reserved)
        source_mismatch = bytearray(payload)
        struct.pack_into("<H", source_mismatch, base + 154, 0)
        mutations.append(source_mismatch)
        zero_ray_steps = bytearray(payload)
        struct.pack_into("<I", zero_ray_steps, base + 140, 0)
        mutations.append(zero_ray_steps)

        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(SpectralFrameError):
                    unpack_spectral_pixel(layout, mutation)
        with self.assertRaises(SpectralFrameError):
            unpack_spectral_pixel(layout, payload[:-1])

    def test_absent_g_and_direction_flags_require_zero_diagnostics(self) -> None:
        frequencies = (1.0,)
        layout = SpectralPixelLayout(frequencies)
        payload = bytearray(
            pack_adaptive_pixel(
                layout,
                _pixel(frequencies, source="captured-boundary"),
                _options(1),
            )
        )
        base = 16 * len(frequencies)
        for offset, value in ((56, 1.25), (64, 2.5), (72, 0.75)):
            mutation = bytearray(payload)
            struct.pack_into("<d", mutation, base + offset, value)
            with self.subTest(offset=offset):
                with self.assertRaises(SpectralFrameError):
                    unpack_spectral_pixel(layout, mutation)

    def test_present_g_and_direction_diagnostics_are_self_consistent(self) -> None:
        frequencies = (1.0,)
        layout = SpectralPixelLayout(frequencies)
        disk_payload = bytearray(
            pack_adaptive_pixel(
                layout,
                _pixel(frequencies, source="disk"),
                _options(1),
            )
        )
        base = 16
        bad_weighted_g = bytearray(disk_payload)
        struct.pack_into("<d", bad_weighted_g, base + 64, 1.0)
        with self.assertRaisesRegex(SpectralFrameError, "weighted g"):
            unpack_spectral_pixel(layout, bad_weighted_g)

        escaped_payload = bytearray(
            pack_adaptive_pixel(
                layout,
                _pixel(frequencies, source="escaped-boundary"),
                _options(1),
            )
        )
        for offset, value, message in (
            (56, 4.0, "may not exceed pi"),
            (72, 1.0, "weighted escape-direction"),
        ):
            mutation = bytearray(escaped_payload)
            struct.pack_into("<d", mutation, base + offset, value)
            with self.subTest(offset=offset):
                with self.assertRaisesRegex(SpectralFrameError, message):
                    unpack_spectral_pixel(layout, mutation)

    def test_strict_layout_validation(self) -> None:
        self.assertRaises(ValueError, SpectralPixelLayout, ())
        self.assertRaises(ValueError, SpectralPixelLayout, (2.0, 1.0))
        self.assertRaises(ValueError, SpectralPixelLayout, (math.inf,))

    def test_zero_evidence_and_luminous_capture_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "failed audit gate"):
            SpectralRaySample(
                specific_intensities_nu=(0.0,),
                absolute_errors_nu=(0.0,),
                visible_source="captured-boundary",
                topology_signature="capture",
                ray_converged=True,
            )
        with self.assertRaisesRegex(ValueError, "positive zero"):
            SpectralRaySample(
                specific_intensities_nu=(1.0,),
                absolute_errors_nu=(0.0,),
                visible_source="captured-boundary",
                topology_signature="capture",
            )
        with self.assertRaisesRegex(ValueError, "disk rays require g"):
            SpectralRaySample(
                specific_intensities_nu=(1.0,),
                absolute_errors_nu=(0.0,),
                visible_source="disk",
                topology_signature="disk",
            )
        with self.assertRaisesRegex(ValueError, "captured-boundary rays"):
            SpectralRaySample(
                specific_intensities_nu=(0.0,),
                absolute_errors_nu=(0.0,),
                visible_source="captured-boundary",
                topology_signature="capture",
                frequency_shift_g=1.2,
            )
        with self.assertRaisesRegex(ValueError, "escaped-boundary rays"):
            SpectralRaySample(
                specific_intensities_nu=(1.0,),
                absolute_errors_nu=(0.0,),
                visible_source="escaped-boundary",
                topology_signature="escape",
            )


if __name__ == "__main__":
    unittest.main()
