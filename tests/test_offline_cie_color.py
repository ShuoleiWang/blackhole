from __future__ import annotations

from decimal import Decimal, localcontext
import math
from pathlib import Path
import shutil
import tempfile
import unittest

from offline.cie_color import (
    CIE_COLUMN_SUMS,
    CIE_CSV_MD5,
    CIE_CSV_SHA256,
    CIE_DATASET_DOI,
    CIE_LICENSE_ID,
    CIE_METADATA_SHA256,
    CIE_PARSED_BINARY64_SHA256,
    CIE_ROW_COUNT,
    Cie1931Table,
    Cie1931Xyz,
    CieColorError,
    DEFAULT_CIE_CSV,
    DEFAULT_CIE_METADATA,
    LIGHT_SPEED_M_S,
    LinearSrgb,
    cie_1931_frequency_grid_hz,
    cie_xyz_to_unclamped_linear_srgb,
    derive_display_srgb,
    load_authenticated_cie_1931_2deg,
    spectral_i_nu_to_cie_xyz,
)


PLANCK_GOLDEN = {
    3000: {
        "xy": (
            0.436931071812130422850004793970298789869831011221045,
            0.404074013067330688078044690284591345281634857290771,
        ),
        "xyz_over_y": (
            1.08131445646648099705283380261483872496884073589672,
            1.0,
            0.393479684361799409206596813504908002607461506792260,
        ),
    },
    6500: {
        "xy": (
            0.313526034676370386755345714970489506082077409918741,
            0.323628662059832201353247562928908260206825920969446,
        ),
        "xyz_over_y": (
            0.968783273647146713039994273860845606347593106162874,
            1.0,
            1.12117789862726951578821825135794483956637011029579,
        ),
    },
    10000: {
        "xy": (
            0.280633784589625225747423470849738577696914187932896,
            0.288288414018236768801075162603974859707215508424901,
        ),
        "xyz_over_y": (
            0.973448015749507988705832347610438138208537422664808,
            1.0,
            1.49530047143992597796771930712545799907221003277105,
        ),
    },
    22500: {
        "xy": (
            0.254231917434593205454500672999285850543430119526794,
            0.254587761803429792238556708933137423714220076019336,
        ),
        "xyz_over_y": (
            0.998602272291818416589689744141993933436736779543130,
            1.0,
            1.92931630838258089516198873372246578135339861025460,
        ),
    },
}


class CieOfflineColourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.table = load_authenticated_cie_1931_2deg()
        cls.frequencies = cie_1931_frequency_grid_hz(cls.table)
        cls.decimal_rows = tuple(
            (
                int(columns[0]),
                Decimal(columns[1]),
                Decimal(columns[2]),
                Decimal(columns[3]),
            )
            for columns in (
                line.split(",") for line in DEFAULT_CIE_CSV.read_text().splitlines()
            )
        )

    @staticmethod
    def assert_relative_close(
        actual: float,
        expected: float,
        relative_tolerance: float = 5.0e-14,
    ) -> None:
        difference = abs(actual - expected)
        limit = relative_tolerance * max(abs(actual), abs(expected), 1.0e-300)
        if difference > limit:
            raise AssertionError(
                f"{actual:.17g} != {expected:.17g}; relative error "
                f"{difference / max(abs(actual), abs(expected)):.3e}"
            )

    @staticmethod
    def srgb_decode(encoded: float) -> float:
        if encoded <= 0.04045:
            return encoded / 12.92
        return ((encoded + 0.055) / 1.055) ** 2.4

    @classmethod
    def decimal_planck_reference(
        cls,
        temperature_k: int,
    ) -> tuple[tuple[float, ...], tuple[Decimal, Decimal, Decimal]]:
        """Independent Decimal B_nu inputs and B_lambda XYZ integration."""

        with localcontext() as context:
            context.prec = 70
            light_speed = Decimal("299792458")
            planck = Decimal("6.62607015e-34")
            boltzmann = Decimal("1.380649e-23")
            temperature = Decimal(temperature_k)
            metre_per_nm = Decimal("1e-9")
            intensities_by_wavelength: list[Decimal] = []
            xyz = [Decimal(0), Decimal(0), Decimal(0)]
            for index, row in enumerate(cls.decimal_rows):
                wavelength_m = Decimal(row[0]) * metre_per_nm
                frequency = light_speed / wavelength_m
                exponent = planck * frequency / (boltzmann * temperature)
                denominator = exponent.exp() - Decimal(1)
                b_nu = (
                    Decimal(2)
                    * planck
                    * frequency**3
                    / (light_speed**2 * denominator)
                )
                b_lambda = (
                    Decimal(2)
                    * planck
                    * light_speed**2
                    / (wavelength_m**5 * denominator)
                )
                weight = (
                    Decimal("0.5")
                    if index in (0, CIE_ROW_COUNT - 1)
                    else Decimal(1)
                )
                intensities_by_wavelength.append(b_nu)
                for channel in range(3):
                    xyz[channel] += (
                        weight * b_lambda * row[channel + 1] * metre_per_nm
                    )
            return (
                tuple(float(value) for value in reversed(intensities_by_wavelength)),
                (xyz[0], xyz[1], xyz[2]),
            )

    def test_official_assets_and_all_declared_table_validations_are_authenticated(
        self,
    ) -> None:
        self.assertEqual(len(self.table.wavelengths_nm), CIE_ROW_COUNT)
        self.assertEqual(self.table.wavelengths_nm, tuple(range(360, 831)))
        self.assertEqual(self.table.csv_sha256, CIE_CSV_SHA256)
        self.assertEqual(self.table.csv_md5, CIE_CSV_MD5)
        self.assertEqual(self.table.metadata_sha256, CIE_METADATA_SHA256)
        descriptor = self.table.descriptor()
        self.assertEqual(descriptor["datasetDoi"], CIE_DATASET_DOI)
        self.assertEqual(descriptor["license"]["id"], CIE_LICENSE_ID)
        self.assertEqual(descriptor["columnSums"], list(CIE_COLUMN_SUMS))
        self.assertEqual(
            descriptor["parsedBinary64Sha256"],
            CIE_PARSED_BINARY64_SHA256,
        )
        self.assertEqual(descriptor["sampleRow"]["oneBasedIndex"], 120)
        self.assertEqual(
            descriptor["sampleRow"]["value"],
            "479,0.104297900000,0.1334528000000,0.856619300000",
        )

    def test_csv_and_metadata_tampering_fail_before_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / DEFAULT_CIE_CSV.name
            metadata_path = root / DEFAULT_CIE_METADATA.name
            shutil.copyfile(DEFAULT_CIE_CSV, csv_path)
            shutil.copyfile(DEFAULT_CIE_METADATA, metadata_path)

            csv_payload = bytearray(csv_path.read_bytes())
            csv_payload[0] = ord("4")
            csv_path.write_bytes(csv_payload)
            with self.assertRaisesRegex(CieColorError, "CSV: SHA-256 mismatch"):
                load_authenticated_cie_1931_2deg(csv_path, metadata_path)

            shutil.copyfile(DEFAULT_CIE_CSV, csv_path)
            metadata_path.write_bytes(metadata_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(CieColorError, "metadata: SHA-256 mismatch"):
                load_authenticated_cie_1931_2deg(csv_path, metadata_path)

        forged_x = list(self.table.x_bar)
        forged_x[0] = math.nextafter(forged_x[0], math.inf)
        with self.assertRaisesRegex(ValueError, "not the authenticated dataset"):
            Cie1931Table(
                self.table.wavelengths_nm,
                tuple(forged_x),
                self.table.y_bar,
                self.table.z_bar,
            )

    def test_exact_frequency_grid_is_increasing_and_excludes_out_of_band_bins(
        self,
    ) -> None:
        self.assertEqual(len(self.frequencies), CIE_ROW_COUNT)
        self.assertTrue(
            all(
                right > left
                for left, right in zip(self.frequencies, self.frequencies[1:])
            )
        )
        self.assertEqual(self.frequencies[0], LIGHT_SPEED_M_S / (830 * 1.0e-9))
        self.assertEqual(self.frequencies[-1], LIGHT_SPEED_M_S / (360 * 1.0e-9))
        zeros = (0.0,) * CIE_ROW_COUNT

        with self.assertRaisesRegex(ValueError, "exactly the 471 visible CIE bins"):
            spectral_i_nu_to_cie_xyz(
                self.frequencies[:-1],
                zeros[:-1],
                table=self.table,
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            spectral_i_nu_to_cie_xyz(
                tuple(reversed(self.frequencies)),
                zeros,
                table=self.table,
            )
        rounded = list(self.frequencies)
        rounded[100] = math.nextafter(rounded[100], math.inf)
        with self.assertRaisesRegex(ValueError, "does not equal the exact CIE grid"):
            spectral_i_nu_to_cie_xyz(rounded, zeros, table=self.table)
        ultraviolet_extra = (*self.frequencies, 1.0e16)
        with self.assertRaisesRegex(ValueError, "exactly the 471 visible CIE bins"):
            spectral_i_nu_to_cie_xyz(
                ultraviolet_extra,
                (*zeros, 1.0),
                table=self.table,
            )

    def test_nan_negative_and_wrong_length_spectra_fail_closed(self) -> None:
        values = [0.0] * CIE_ROW_COUNT
        values[7] = math.nan
        with self.assertRaisesRegex(ValueError, r"specific_intensities_nu\[7\]"):
            spectral_i_nu_to_cie_xyz(
                self.frequencies,
                values,
                table=self.table,
            )
        values[7] = -1.0
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            spectral_i_nu_to_cie_xyz(
                self.frequencies,
                values,
                table=self.table,
            )
        values[7] = 0.0
        with self.assertRaisesRegex(ValueError, "exactly 471 values"):
            spectral_i_nu_to_cie_xyz(
                self.frequencies,
                values[:-1],
                table=self.table,
            )

    def test_high_precision_planck_goldens_at_four_temperatures(self) -> None:
        for temperature, golden in PLANCK_GOLDEN.items():
            with self.subTest(temperature=temperature):
                intensities, decimal_xyz = self.decimal_planck_reference(temperature)
                decimal_y = decimal_xyz[1]
                decimal_total = sum(decimal_xyz)
                decimal_xy = (
                    float(decimal_xyz[0] / decimal_total),
                    float(decimal_xyz[1] / decimal_total),
                )
                decimal_normalized = tuple(
                    float(value / decimal_y) for value in decimal_xyz
                )
                for actual, expected in zip(decimal_xy, golden["xy"]):
                    self.assert_relative_close(actual, expected, 2.0e-15)
                for actual, expected in zip(
                    decimal_normalized,
                    golden["xyz_over_y"],
                ):
                    self.assert_relative_close(actual, expected, 2.0e-15)

                actual_xyz = spectral_i_nu_to_cie_xyz(
                    self.frequencies,
                    intensities,
                    table=self.table,
                )
                for actual, expected in zip(
                    (actual_xyz.x, actual_xyz.y, actual_xyz.z),
                    (float(value) for value in decimal_xyz),
                ):
                    self.assert_relative_close(actual, expected)
                for actual, expected in zip(actual_xyz.chromaticity_xy, golden["xy"]):
                    self.assert_relative_close(actual, expected)

    def test_monochromatic_equal_energy_and_scale_linearity(self) -> None:
        wavelength_nm = 555
        wavelength_index = wavelength_nm - 360
        frequency_index = 830 - wavelength_nm
        intensity_nu = 1.0e-20
        monochromatic = [0.0] * CIE_ROW_COUNT
        monochromatic[frequency_index] = intensity_nu
        xyz = spectral_i_nu_to_cie_xyz(
            self.frequencies,
            monochromatic,
            table=self.table,
        )
        wavelength_m = wavelength_nm * 1.0e-9
        factor = (
            intensity_nu
            * LIGHT_SPEED_M_S
            / (wavelength_m * wavelength_m)
            * 1.0e-9
        )
        expected = (
            factor * self.table.x_bar[wavelength_index],
            factor * self.table.y_bar[wavelength_index],
            factor * self.table.z_bar[wavelength_index],
        )
        for actual, reference in zip((xyz.x, xyz.y, xyz.z), expected):
            self.assert_relative_close(actual, reference, 3.0e-15)

        # Equal-energy means constant I_lambda, not constant I_nu.  Construct
        # the exact corresponding I_nu(lambda)=I_lambda*lambda^2/c samples.
        equal_energy_by_wavelength = tuple(
            (wavelength_nm * 1.0e-9) ** 2 / LIGHT_SPEED_M_S
            for wavelength_nm in self.table.wavelengths_nm
        )
        equal_energy = tuple(reversed(equal_energy_by_wavelength))
        equal_xyz = spectral_i_nu_to_cie_xyz(
            self.frequencies,
            equal_energy,
            table=self.table,
        )
        equal_xy = equal_xyz.chromaticity_xy
        self.assert_relative_close(equal_xy[0], 0.33331456174474133, 3.0e-15)
        self.assert_relative_close(equal_xy[1], 0.3332880844537104, 3.0e-15)

        scale = 7.25
        scaled_xyz = spectral_i_nu_to_cie_xyz(
            self.frequencies,
            tuple(scale * value for value in equal_energy),
            table=self.table,
        )
        for actual, reference in zip(
            (scaled_xyz.x, scaled_xyz.y, scaled_xyz.z),
            (equal_xyz.x, equal_xyz.y, equal_xyz.z),
        ):
            self.assert_relative_close(actual, scale * reference, 3.0e-15)

    def test_blackbody_colour_temperature_order_is_physical(self) -> None:
        chromaticities = []
        blue_to_red = []
        for temperature in PLANCK_GOLDEN:
            intensities, _decimal_xyz = self.decimal_planck_reference(temperature)
            xyz = spectral_i_nu_to_cie_xyz(
                self.frequencies,
                intensities,
                table=self.table,
            )
            linear = cie_xyz_to_unclamped_linear_srgb(xyz)
            chromaticities.append(xyz.chromaticity_xy)
            blue_to_red.append(linear.b / linear.r)
        self.assertTrue(
            all(
                right[0] < left[0]
                for left, right in zip(chromaticities, chromaticities[1:])
            )
        )
        self.assertTrue(
            all(right > left for left, right in zip(blue_to_red, blue_to_red[1:]))
        )

    def test_linear_srgb_is_unclamped_and_display_mapping_is_explicit(self) -> None:
        science_xyz = Cie1931Xyz(1.0, 0.0, 0.0)
        linear = cie_xyz_to_unclamped_linear_srgb(science_xyz)
        self.assertGreater(linear.r, 1.0)
        self.assertLess(linear.g, 0.0)
        self.assertFalse(linear.descriptor()["clamped"])
        self.assertFalse(linear.descriptor()["toneMappingApplied"])

        display = derive_display_srgb(linear, exposure=0.25)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in (display.r, display.g, display.b)))
        self.assertEqual(display.g, 0.0)
        self.assertTrue(display.descriptor()["derivedDisplayOutput"])
        self.assertEqual(
            display.descriptor()["negativeLinearPolicy"],
            "clip-to-zero-at-display-boundary",
        )
        self.assertEqual(
            display.descriptor()["toneMappingDomain"],
            "Rec.709-linear-luminance",
        )
        self.assertEqual(
            display.descriptor()["gamutPolicy"],
            "uniform-max-channel-scale-if-needed",
        )
        self.assertEqual(science_xyz, Cie1931Xyz(1.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "exposure must be finite and positive"):
            derive_display_srgb(linear, exposure=0.0)

    def test_luminance_tone_and_gamut_scales_preserve_linear_rgb_ratios(self) -> None:
        source = (0.25, 1.0, 4.0)
        display = derive_display_srgb(LinearSrgb(*source), exposure=3.0)
        decoded = tuple(
            self.srgb_decode(value) for value in (display.r, display.g, display.b)
        )
        source_normalized = tuple(value / max(source) for value in source)
        decoded_normalized = tuple(value / max(decoded) for value in decoded)
        for actual, expected in zip(decoded_normalized, source_normalized):
            self.assert_relative_close(actual, expected, 2.0e-14)
        self.assert_relative_close(decoded[0] / decoded[1], source[0] / source[1])
        self.assert_relative_close(decoded[1] / decoded[2], source[1] / source[2])
        self.assertTrue(
            display.descriptor()[
                "nonNegativeLinearRgbRatioPreservedBeforeSrgbEncoding"
            ]
        )

    def test_display_white_is_neutral_monotonic_and_bounded(self) -> None:
        neutral = derive_display_srgb(LinearSrgb(8.0, 8.0, 8.0), exposure=1.0)
        self.assertEqual(neutral.r, neutral.g)
        self.assertEqual(neutral.g, neutral.b)

        base = (0.1, 0.3, 0.8)
        previous = (0.0, 0.0, 0.0)
        for amplitude in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0):
            display = derive_display_srgb(
                LinearSrgb(*(amplitude * value for value in base)),
                exposure=1.0,
            )
            current = (display.r, display.g, display.b)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in current))
            self.assertTrue(
                all(right >= left for left, right in zip(previous, current))
            )
            previous = current

    def test_strong_red_and_blue_do_not_collapse_to_white_and_zero_y_fails(self) -> None:
        strong_red = derive_display_srgb(
            LinearSrgb(1000.0, 0.1, 0.05),
            exposure=1.0,
        )
        strong_blue = derive_display_srgb(
            LinearSrgb(0.05, 0.1, 1000.0),
            exposure=1.0,
        )
        self.assertGreater(strong_red.r, 0.99)
        self.assertLess(strong_red.g, 0.2)
        self.assertLess(strong_red.b, 0.2)
        self.assertGreater(strong_blue.b, 0.99)
        self.assertLess(strong_blue.r, 0.2)
        self.assertLess(strong_blue.g, 0.2)
        self.assertGreater(
            max(strong_red.r, strong_red.g, strong_red.b)
            - min(strong_red.r, strong_red.g, strong_red.b),
            0.8,
        )
        self.assertGreater(
            max(strong_blue.r, strong_blue.g, strong_blue.b)
            - min(strong_blue.r, strong_blue.g, strong_blue.b),
            0.8,
        )

        positive_subnormal = math.ulp(0.0)
        with self.assertRaisesRegex(
            CieColorError,
            "positive display RGB has zero representable Rec.709 luminance",
        ):
            derive_display_srgb(
                LinearSrgb(positive_subnormal, 0.0, 0.0),
                exposure=1.0,
            )
        black = derive_display_srgb(LinearSrgb(0.0, 0.0, 0.0), exposure=1.0)
        self.assertEqual((black.r, black.g, black.b), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
