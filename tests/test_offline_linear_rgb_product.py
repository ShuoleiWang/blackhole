from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import offline.cie_product as cie_product_module
import offline.linear_rgb_product as linear_product_module
import scripts.verify_offline_linear_srgb as linear_verifier_module
from offline.linear_rgb_product import (
    CONVERTER_SOURCE_FILES,
    LinearRgbProductError,
    LinearSrgbPixelRecord,
    RECORD_BYTES,
    SCIENTIFIC_STATUS,
    XYZ_TO_LINEAR_SRGB_D65,
    apply_declared_xyz_to_linear_srgb,
    convert_cie_xyz_product_to_linear_srgb,
    pack_linear_srgb_pixel,
    propagated_linear_srgb_absolute_error,
    unpack_linear_srgb_pixel,
)
from tests.test_offline_cie_product import create_exact_cie_spectral_fixture


ROOT = Path(__file__).resolve().parents[1]
TEST_BACKEND = {
    "implementationId": "tests.linear-srgb-backend/v1",
}


def create_linear_srgb_fixture(root: Path):
    cie_fixture = create_exact_cie_spectral_fixture(root)
    output = root / "linear-srgb-product"
    publication = convert_cie_xyz_product_to_linear_srgb(
        cie_fixture.output_manifest,
        cie_fixture.input_manifest,
        output,
        numeric_backend=TEST_BACKEND,
    )
    return cie_fixture, output, publication


class OfflineLinearRgbProductTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_product_preserves_unclamped_linear_values_and_record_binding(self) -> None:
        cie_fixture, output, publication = create_linear_srgb_fixture(self.root)
        manifest = json.loads(publication.manifest_path.read_bytes())
        xyz_manifest = json.loads(cie_fixture.output_manifest.read_bytes())
        self.assertEqual(publication.record_count, 2)
        self.assertEqual(publication.tile_count, 2)
        self.assertEqual(manifest["scientificStatus"], dict(SCIENTIFIC_STATUS))
        self.assertFalse(manifest["scientificStatus"]["clamped"])
        self.assertFalse(manifest["scientificStatus"]["toneMappingApplied"])
        self.assertFalse(manifest["scientificStatus"]["srgbTransferCurveApplied"])
        entry = manifest["tiles"][0]
        xyz_entry = xyz_manifest["tiles"][0]
        raw = (output / entry["outputPayload"]["uri"]).read_bytes()
        xyz_raw = (
            cie_fixture.output_product / xyz_entry["outputPayload"]["uri"]
        ).read_bytes()
        record = unpack_linear_srgb_pixel(raw)
        xyz_values = struct.unpack("<6d32sII", xyz_raw)
        expected = apply_declared_xyz_to_linear_srgb(
            tuple(xyz_values[:3])
        )
        self.assertEqual(record.linear_srgb, expected)
        self.assertEqual(
            record.estimated_absolute_error_linear_srgb,
            propagated_linear_srgb_absolute_error(tuple(xyz_values[3:6])),
        )
        self.assertEqual(
            record.input_cie_xyz_record_sha256,
            hashlib.sha256(xyz_raw).digest(),
        )
        self.assertEqual(record.input_spectral_record_sha256, xyz_values[6])
        self.assertEqual(record.source_mask, xyz_values[7])
        self.assertEqual(record.convergence_mask, xyz_values[8])

    def test_negative_out_of_gamut_channels_and_d65_white_are_not_clamped(self) -> None:
        green_axis = apply_declared_xyz_to_linear_srgb((0.0, 1.0, 0.0))
        self.assertLess(green_axis[0], 0.0)
        self.assertGreater(green_axis[1], 1.0)
        self.assertLess(green_axis[2], 0.0)
        record = LinearSrgbPixelRecord(
            linear_srgb=green_axis,
            estimated_absolute_error_linear_srgb=(0.0, 0.0, 0.0),
            input_cie_xyz_record_sha256=b"x" * 32,
            input_spectral_record_sha256=b"y" * 32,
            source_mask=1,
            convergence_mask=255,
        )
        self.assertEqual(
            unpack_linear_srgb_pixel(pack_linear_srgb_pixel(record)),
            record,
        )
        white = apply_declared_xyz_to_linear_srgb(
            (0.9504559270516718, 1.0, 1.0890577507598784)
        )
        for channel in white:
            self.assertAlmostEqual(channel, 1.0, places=14)

    def test_product_transform_is_the_manifest_matrix_authority(self) -> None:
        xyz = (0.31, 0.47, 0.19)
        expected = tuple(
            math.fsum(
                XYZ_TO_LINEAR_SRGB_D65[row][column] * xyz[column]
                for column in range(3)
            )
            for row in range(3)
        )
        self.assertEqual(apply_declared_xyz_to_linear_srgb(xyz), expected)

    def test_absolute_matrix_error_propagation_is_exact(self) -> None:
        errors = (0.1, 0.2, 0.3)
        expected = tuple(
            math.fsum(
                abs(XYZ_TO_LINEAR_SRGB_D65[row][column]) * errors[column]
                for column in range(3)
            )
            for row in range(3)
        )
        self.assertEqual(propagated_linear_srgb_absolute_error(errors), expected)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            propagated_linear_srgb_absolute_error((0.1, -0.2, 0.3))

    def test_numeric_and_source_identity_drift_changes_product_hash(self) -> None:
        cie_fixture, _output, baseline = create_linear_srgb_fixture(self.root)
        manifest = json.loads(baseline.manifest_path.read_bytes())
        self.assertEqual(
            tuple(
                entry["moduleUri"]
                for entry in manifest["converter"]["descriptor"]["sourceFiles"]
            ),
            CONVERTER_SOURCE_FILES,
        )
        self.assertEqual(
            tuple(
                entry["moduleUri"]
                for entry in linear_verifier_module._source_descriptor()
            ),
            CONVERTER_SOURCE_FILES,
        )
        schema = json.loads(
            (ROOT / "schemas" / "offline-linear-srgb-frame-v1.schema.json")
            .read_bytes()
        )
        source_files_schema = schema["$defs"]["converterDescriptor"][
            "properties"
        ]["sourceFiles"]
        self.assertEqual(source_files_schema["minItems"], len(CONVERTER_SOURCE_FILES))
        self.assertEqual(source_files_schema["maxItems"], len(CONVERTER_SOURCE_FILES))
        self.assertEqual(
            tuple(schema["$defs"]["sourceFile"]["properties"]["moduleUri"]["enum"]),
            CONVERTER_SOURCE_FILES,
        )
        backend_variant = convert_cie_xyz_product_to_linear_srgb(
            cie_fixture.output_manifest,
            cie_fixture.input_manifest,
            self.root / "backend-variant",
            numeric_backend={"implementationId": "tests.different-backend/v1"},
        )
        self.assertNotEqual(backend_variant.product_sha256, baseline.product_sha256)
        real_descriptor = linear_product_module._source_file_descriptor

        def changed_descriptor(path: Path, module_uri: str) -> dict[str, object]:
            result = real_descriptor(path, module_uri)
            if module_uri == "offline/job.py":
                result["sha256"] = "0" * 64
            return result

        with patch.object(
            linear_product_module,
            "_source_file_descriptor",
            side_effect=changed_descriptor,
        ):
            source_variant = convert_cie_xyz_product_to_linear_srgb(
                cie_fixture.output_manifest,
                cie_fixture.input_manifest,
                self.root / "source-variant",
                numeric_backend=TEST_BACKEND,
            )
        self.assertNotEqual(source_variant.product_sha256, baseline.product_sha256)

    def test_transaction_failures_clean_staging_and_retry_same_path(self) -> None:
        cie_fixture = create_exact_cie_spectral_fixture(self.root)
        real_write = cie_product_module._atomic_write_no_replace
        for failure_call, label in (
            (2, "second-tile"),
            (3, "sidecar"),
            (4, "manifest"),
        ):
            with self.subTest(label=label):
                output = self.root / f"transaction-{label}"
                calls = 0

                def injected(path: Path, payload: bytes) -> None:
                    nonlocal calls
                    calls += 1
                    if calls == failure_call:
                        raise LinearRgbProductError(f"injected {label} failure")
                    real_write(path, payload)

                with patch.object(
                    cie_product_module,
                    "_atomic_write_no_replace",
                    side_effect=injected,
                ):
                    with self.assertRaisesRegex(LinearRgbProductError, label):
                        convert_cie_xyz_product_to_linear_srgb(
                            cie_fixture.output_manifest,
                            cie_fixture.input_manifest,
                            output,
                            numeric_backend=TEST_BACKEND,
                        )
                self.assertFalse(output.exists())
                self.assertEqual(tuple(self.root.glob(f".{output.name}.staging-*")), ())
                retry = convert_cie_xyz_product_to_linear_srgb(
                    cie_fixture.output_manifest,
                    cie_fixture.input_manifest,
                    output,
                    numeric_backend=TEST_BACKEND,
                )
                self.assertTrue(retry.manifest_path.is_file())

    def test_no_overwrite_and_cli(self) -> None:
        cie_fixture, output, _publication = create_linear_srgb_fixture(self.root)
        with self.assertRaisesRegex(LinearRgbProductError, "refusing to overwrite"):
            convert_cie_xyz_product_to_linear_srgb(
                cie_fixture.output_manifest,
                cie_fixture.input_manifest,
                output,
                numeric_backend=TEST_BACKEND,
            )
        cli_output = self.root / "cli-product"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "convert_offline_cie_xyz_to_linear_srgb.py"),
                str(cie_fixture.output_manifest),
                str(cie_fixture.input_manifest),
                str(cli_output),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["recordCount"], 2)
        self.assertTrue((cli_output / "manifest.json").is_file())

    def test_record_abi_allows_signed_rgb_and_rejects_invalid_values(
        self,
    ) -> None:
        self.assertEqual(RECORD_BYTES, 120)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            LinearSrgbPixelRecord(
                linear_srgb=(math.nan, -1.0, 2.0),
                estimated_absolute_error_linear_srgb=(0.0, 0.0, 0.0),
                input_cie_xyz_record_sha256=b"x" * 32,
                input_spectral_record_sha256=b"y" * 32,
                source_mask=1,
                convergence_mask=255,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            LinearSrgbPixelRecord(
                linear_srgb=(-1.0, 2.0, 3.0),
                estimated_absolute_error_linear_srgb=(-0.1, 0.0, 0.0),
                input_cie_xyz_record_sha256=b"x" * 32,
                input_spectral_record_sha256=b"y" * 32,
                source_mask=1,
                convergence_mask=255,
            )


if __name__ == "__main__":
    unittest.main()
