from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import offline.cie_product as cie_product_module
from offline.cie_color import LinearSrgb, derive_display_srgb
import offline.display_product as display_product_module
from offline.display_product import (
    DISPLAY_SOURCE_FILES,
    DISPLAY_STATUS,
    DisplayProductError,
    convert_linear_srgb_to_sdr_display,
    derive_display_rgb16,
    encode_ppm16,
)
from offline.job import canonical_json_bytes
from offline.linear_rgb_product import RECORD_BYTES, unpack_linear_srgb_pixel
from scripts.verify_nr_contract import ContractError
import scripts.verify_offline_sdr_display as display_verifier_module
from scripts.verify_offline_sdr_display import validate_sdr_display_quicklook
from tests.test_offline_linear_rgb_product import create_linear_srgb_fixture


ROOT = Path(__file__).resolve().parents[1]
EXPOSURE = 2.5e25


def create_display_fixture(root: Path, *, exposure: float = EXPOSURE):
    cie_fixture, linear_output, linear_publication = create_linear_srgb_fixture(root)
    output = root / "sdr-display-product"
    publication = convert_linear_srgb_to_sdr_display(
        linear_publication.manifest_path,
        cie_fixture.output_manifest,
        cie_fixture.input_manifest,
        output,
        exposure=exposure,
    )
    return cie_fixture, linear_output, linear_publication, output, publication


class OfflineDisplayProductTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_product_is_separate_explicit_sdr_ppm16_derivative(self) -> None:
        fixture, linear_root, linear_publication, output, publication = (
            create_display_fixture(self.root)
        )
        manifest = json.loads(publication.manifest_path.read_bytes())
        linear_manifest = json.loads(linear_publication.manifest_path.read_bytes())
        self.assertEqual(manifest["displayStatus"], dict(DISPLAY_STATUS))
        self.assertTrue(manifest["displayStatus"]["isDisplayImage"])
        self.assertFalse(manifest["displayStatus"]["isHdr"])
        self.assertFalse(manifest["displayStatus"]["inputPhysicsVerified"])
        self.assertFalse(
            manifest["displayStatus"]["scientificLinearSrgbModified"]
        )
        self.assertEqual(
            manifest["displayTransform"]["descriptor"]["exposure"], EXPOSURE
        )
        self.assertEqual(
            manifest["displayTransform"]["descriptor"]["toneMapper"],
            "reinhard-rec709-luminance-uniform-gamut/v2",
        )
        entry = manifest["tiles"][0]
        image_entry = manifest["images"][0]
        linear_entry = linear_manifest["tiles"][0]
        self.assertEqual(
            entry["inputLinearSrgbPayload"], linear_entry["outputPayload"]
        )
        ppm = (output / image_entry["outputPayload"]["uri"]).read_bytes()
        width = image_entry["widthPixels"]
        height = image_entry["heightPixels"]
        header = f"P6\n{width} {height}\n65535\n".encode("ascii")
        self.assertTrue(ppm.startswith(header))
        self.assertEqual(len(ppm), len(header) + 6 * image_entry["pixelCount"])
        raw_linear = (
            linear_root / linear_entry["outputPayload"]["uri"]
        ).read_bytes()[:RECORD_BYTES]
        record = unpack_linear_srgb_pixel(raw_linear)
        expected = derive_display_rgb16(record.linear_srgb, exposure=EXPOSURE)
        actual = tuple(
            int.from_bytes(ppm[len(header) + 2 * i : len(header) + 2 * i + 2], "big")
            for i in range(3)
        )
        self.assertEqual(actual, expected)
        self.assertTrue(fixture.input_manifest.is_file())

    def test_transform_matches_cie_colour_api_and_quantization_rule(self) -> None:
        for linear in (
            (-1.0, 0.25, 3.0),
            (0.0, 0.0, 0.0),
            (2.0, 1.0, 0.5),
            (1.0e20, 2.0e20, 3.0e20),
        ):
            with self.subTest(linear=linear):
                expected_display = derive_display_srgb(
                    LinearSrgb(*linear), exposure=0.75
                )
                expected = tuple(
                    math.floor(channel * 65535.0 + 0.5)
                    for channel in (
                        expected_display.r,
                        expected_display.g,
                        expected_display.b,
                    )
                )
                self.assertEqual(
                    derive_display_rgb16(linear, exposure=0.75), expected
                )
        self.assertEqual(
            derive_display_rgb16((-1.0, -2.0, -3.0), exposure=1.0),
            (0, 0, 0),
        )

    def test_ppm_encoder_flips_lower_left_rows_and_uses_big_endian(self) -> None:
        pixels = [
            (0x0001, 0x0203, 0x0405),
            (0x0607, 0x0809, 0x0A0B),
            (0x1011, 0x1213, 0x1415),
            (0x1617, 0x1819, 0x1A1B),
        ]
        ppm = encode_ppm16(2, 2, pixels)
        expected_samples = b"".join(
            channel.to_bytes(2, "big")
            for pixel in (pixels[2], pixels[3], pixels[0], pixels[1])
            for channel in pixel
        )
        self.assertEqual(ppm, b"P6\n2 2\n65535\n" + expected_samples)

    def test_independent_verifier_recomputes_every_display_pixel(self) -> None:
        fixture, _linear_root, linear_publication, output, publication = (
            create_display_fixture(self.root)
        )
        report = validate_sdr_display_quicklook(
            publication.manifest_path,
            linear_publication.manifest_path,
            fixture.output_manifest,
            fixture.input_manifest,
        )
        self.assertTrue(report["displayReplayVerified"])
        self.assertTrue(report["ppm16EncodingVerified"])
        self.assertFalse(report["hdrVerified"])
        self.assertFalse(report["inputPhysicsVerified"])

        manifest = json.loads(publication.manifest_path.read_bytes())
        entry = manifest["images"][0]
        ppm_path = output / entry["outputPayload"]["uri"]
        payload = bytearray(ppm_path.read_bytes())
        payload[-1] ^= 1
        ppm_path.write_bytes(payload)
        entry["outputPayload"]["sha256"] = hashlib.sha256(payload).hexdigest()
        identity = {
            "configurationSha256": manifest["integrity"]["configurationSha256"],
            "images": manifest["images"],
            "schema": manifest["schema"],
            "summary": manifest["summary"],
            "tiles": manifest["tiles"],
        }
        product_hash = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        manifest["integrity"]["productSha256"] = product_hash
        manifest["id"] = f"sdr-display-quicklook-{product_hash[:24]}"
        manifest_payload = canonical_json_bytes(manifest)
        publication.manifest_path.write_bytes(manifest_payload)
        (output / "manifest.sha256").write_bytes(
            f"{hashlib.sha256(manifest_payload).hexdigest()}  manifest.json\n".encode()
        )
        with self.assertRaisesRegex(ContractError, "independent PPM replay"):
            validate_sdr_display_quicklook(
                publication.manifest_path,
                linear_publication.manifest_path,
                fixture.output_manifest,
                fixture.input_manifest,
            )

    def test_transaction_no_overwrite_and_strict_exposure(self) -> None:
        fixture, output, linear_publication = create_linear_srgb_fixture(self.root)
        destination = self.root / "transaction-failure"
        real_write = cie_product_module._atomic_write_no_replace
        calls = 0

        def injected(path: Path, payload: bytes) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise DisplayProductError("injected display failure")
            real_write(path, payload)

        with patch.object(
            cie_product_module,
            "_atomic_write_no_replace",
            side_effect=injected,
        ):
            with self.assertRaisesRegex(DisplayProductError, "injected"):
                convert_linear_srgb_to_sdr_display(
                    linear_publication.manifest_path,
                    fixture.output_manifest,
                    fixture.input_manifest,
                    destination,
                    exposure=EXPOSURE,
                )
        self.assertFalse(destination.exists())
        self.assertEqual(tuple(self.root.glob(f".{destination.name}.staging-*")), ())
        publication = convert_linear_srgb_to_sdr_display(
            linear_publication.manifest_path,
            fixture.output_manifest,
            fixture.input_manifest,
            destination,
            exposure=EXPOSURE,
        )
        with self.assertRaisesRegex(DisplayProductError, "refusing to overwrite"):
            convert_linear_srgb_to_sdr_display(
                linear_publication.manifest_path,
                fixture.output_manifest,
                fixture.input_manifest,
                destination,
                exposure=EXPOSURE,
            )
        self.assertTrue(publication.manifest_path.is_file())
        for invalid in (True, 0.0, -1.0, math.inf, math.nan):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    display_product_module._strict_exposure(invalid)
        self.assertTrue(output.is_dir())

    def test_cli_requires_manual_exposure_and_emits_publication(self) -> None:
        fixture, _linear_output, linear_publication = create_linear_srgb_fixture(
            self.root
        )
        output = self.root / "cli-display"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "convert_offline_linear_srgb_to_sdr_display.py"),
                str(linear_publication.manifest_path),
                str(fixture.output_manifest),
                str(fixture.input_manifest),
                str(output),
                "--exposure",
                str(EXPOSURE),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["pixelCount"], 2)
        self.assertEqual(report["imageCount"], 1)
        self.assertEqual(report["tileCount"], 2)
        verified = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "verify_offline_sdr_display.py"),
                str(output / "manifest.json"),
                str(linear_publication.manifest_path),
                str(fixture.output_manifest),
                str(fixture.input_manifest),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(verified.returncode, 0, verified.stderr)
        self.assertTrue(json.loads(verified.stdout)["displayReplayVerified"])

    def test_negative_exposure_overflow_is_rejected_before_display_clip(self) -> None:
        with self.assertRaisesRegex(
            ContractError, "overflowed before display clipping"
        ):
            display_verifier_module._derive_rgb16(
                (-1.0e308, 0.0, 0.0),
                1.0e308,
                "$.negative-overflow",
            )
        with self.assertRaisesRegex(DisplayProductError, "exposure overflowed"):
            derive_display_rgb16(
                (-1.0e308, 0.0, 0.0),
                exposure=1.0e308,
            )

    def test_final_self_snapshot_detects_every_post_source_mutation(self) -> None:
        expected_errors = {
            "manifest": "final payload hash changed",
            "sidecar": "final payload hash changed",
            "ppm": "final payload hash changed",
            "extra": "final product root entries differ",
        }
        for target_name, expected_error in expected_errors.items():
            with self.subTest(target=target_name):
                case_root = self.root / target_name
                case_root.mkdir()
                fixture, _linear_root, linear_publication, output, publication = (
                    create_display_fixture(case_root)
                )
                manifest = json.loads(publication.manifest_path.read_bytes())
                ppm_path = output / manifest["images"][0]["outputPayload"]["uri"]
                real_sources = display_verifier_module._source_descriptor
                calls = 0

                def mutate_after_final_source_check():
                    nonlocal calls
                    result = real_sources()
                    calls += 1
                    if calls == 2:
                        if target_name == "manifest":
                            payload = bytearray(publication.manifest_path.read_bytes())
                            payload[-1] ^= 1
                            publication.manifest_path.write_bytes(payload)
                        elif target_name == "sidecar":
                            sidecar_path = output / "manifest.sha256"
                            payload = bytearray(sidecar_path.read_bytes())
                            payload[0] = ord("0") if payload[0] != ord("0") else ord("1")
                            sidecar_path.write_bytes(payload)
                        elif target_name == "ppm":
                            payload = bytearray(ppm_path.read_bytes())
                            payload[-1] ^= 1
                            ppm_path.write_bytes(payload)
                        else:
                            (output / "undeclared.bin").write_bytes(b"late file")
                    return result

                with patch.object(
                    display_verifier_module,
                    "_source_descriptor",
                    side_effect=mutate_after_final_source_check,
                ):
                    with self.assertRaisesRegex(ContractError, expected_error):
                        validate_sdr_display_quicklook(
                            publication.manifest_path,
                            linear_publication.manifest_path,
                            fixture.output_manifest,
                            fixture.input_manifest,
                        )
                self.assertEqual(calls, 2)

    def test_final_no_extra_tail_mutation_is_inside_anchored_closure(self) -> None:
        fixture, _linear_root, linear_publication, output, publication = (
            create_display_fixture(self.root)
        )
        manifest = json.loads(publication.manifest_path.read_bytes())
        ppm_path = output / manifest["images"][0]["outputPayload"]["uri"]
        real_no_extra = display_verifier_module._validate_no_extra_files
        calls = 0

        def flip_ppm_after_completed_scan(root: Path, allowed: set[str]) -> None:
            nonlocal calls
            real_no_extra(root, allowed)
            calls += 1
            if calls == 2:
                payload = bytearray(ppm_path.read_bytes())
                payload[-1] ^= 1
                ppm_path.write_bytes(payload)

        with patch.object(
            display_verifier_module,
            "_validate_no_extra_files",
            side_effect=flip_ppm_after_completed_scan,
        ):
            with self.assertRaisesRegex(
                ContractError,
                "final payload hash changed|identity changed",
            ):
                validate_sdr_display_quicklook(
                    publication.manifest_path,
                    linear_publication.manifest_path,
                    fixture.output_manifest,
                    fixture.input_manifest,
                )
        self.assertEqual(calls, 2)

    def test_numeric_backend_is_exact_current_and_rechecked_at_end(self) -> None:
        current = display_product_module.default_numeric_backend_descriptor()
        variant = json.loads(canonical_json_bytes(current))
        variant["processor"] = variant["processor"] + "-drift"
        with self.assertRaisesRegex(
            DisplayProductError, "authenticated current default v2"
        ):
            display_product_module.display_transform_descriptor(1.0, variant)

        schema_root = self.root / "schema-extra"
        schema_root.mkdir()
        fixture, _linear_root, linear_publication, _output, publication = (
            create_display_fixture(schema_root)
        )
        manifest = json.loads(publication.manifest_path.read_bytes())
        manifest["displayTransform"]["descriptor"]["numericBackend"][
            "unexpected"
        ] = 1
        publication.manifest_path.write_bytes(canonical_json_bytes(manifest))
        with self.assertRaisesRegex(ContractError, "unknown property"):
            validate_sdr_display_quicklook(
                publication.manifest_path,
                linear_publication.manifest_path,
                fixture.output_manifest,
                fixture.input_manifest,
            )

        drift_root = self.root / "runtime-drift"
        drift_root.mkdir()
        fixture, _linear_root, linear_publication, _output, publication = (
            create_display_fixture(drift_root)
        )
        real_backend = display_verifier_module.default_numeric_backend_descriptor
        backend_calls = 0

        def drift_after_initial_backend_check():
            nonlocal backend_calls
            result = real_backend()
            backend_calls += 1
            if backend_calls == 2:
                result = json.loads(canonical_json_bytes(result))
                result["processor"] = result["processor"] + "-late-drift"
            return result

        with patch.object(
            display_verifier_module,
            "default_numeric_backend_descriptor",
            side_effect=drift_after_initial_backend_check,
        ):
            with self.assertRaisesRegex(ContractError, "changed during verification"):
                validate_sdr_display_quicklook(
                    publication.manifest_path,
                    linear_publication.manifest_path,
                    fixture.output_manifest,
                    fixture.input_manifest,
                )
        self.assertEqual(backend_calls, 2)

    def test_source_closure_and_compact_buffer_limits_are_bound(self) -> None:
        fixture, _linear_root, _linear_publication, _output, publication = (
            create_display_fixture(self.root)
        )
        manifest = json.loads(publication.manifest_path.read_bytes())
        descriptor = manifest["displayTransform"]["descriptor"]
        source_sequence = tuple(
            entry["moduleUri"] for entry in descriptor["sourceFiles"]
        )
        self.assertEqual(source_sequence, DISPLAY_SOURCE_FILES)
        self.assertEqual(
            tuple(
                entry["moduleUri"]
                for entry in display_verifier_module._source_descriptor()
            ),
            DISPLAY_SOURCE_FILES,
        )
        schema = json.loads(
            (ROOT / "schemas" / "offline-sdr-display-quicklook-v1.schema.json")
            .read_bytes()
        )
        source_files_schema = schema["$defs"]["transformDescriptor"][
            "properties"
        ]["sourceFiles"]
        self.assertEqual(source_files_schema["minItems"], len(DISPLAY_SOURCE_FILES))
        self.assertEqual(source_files_schema["maxItems"], len(DISPLAY_SOURCE_FILES))
        self.assertEqual(
            tuple(schema["$defs"]["sourceFile"]["properties"]["moduleUri"]["enum"]),
            DISPLAY_SOURCE_FILES,
        )
        limits = descriptor["resourceLimits"]
        self.assertEqual(
            limits,
            {
                "bulkBufferAccounting": (
                    "counts only RGB16 frame buffers, coverage buffers, one "
                    "authenticated tile or up to two PPM payloads, and "
                    "producer record scratch; excludes raw manifest/schema "
                    "bytes, parsed JSON objects, and upstream verifier runtime "
                    "memory"
                ),
                "coverageBytesPerPixel": 1,
                "maxFramePixels": 1 << 23,
                "maxInputTileBytes": 1 << 26,
                "maxInputTilePayloadBytes": 67108800,
                "maxSampleCount": 64,
                "maxTotalPixels": 1 << 24,
                "ppmHeaderUpperBoundBytes": 19,
                "producerBulkBufferUpperBoundBytes": 184549432,
                "rgb16BytesPerPixel": 6,
                "verifierBulkBufferUpperBoundBytes": 218103846,
            },
        )
        compact = bytearray.fromhex("000102030405")
        encoded = display_product_module._encode_ppm16_buffer(1, 1, compact)
        self.assertIsInstance(encoded, bytearray)
        self.assertEqual(encoded, b"P6\n1 1\n65535\n" + compact)
        self.assertEqual(
            display_product_module._checked_frame_topology(1 << 23, 1, (0,)),
            1 << 23,
        )
        with self.assertRaisesRegex(DisplayProductError, r"2\^23"):
            display_product_module._checked_frame_topology(
                (1 << 23) + 1, 1, (0,)
            )
        with self.assertRaisesRegex(DisplayProductError, "sample count"):
            display_product_module._checked_frame_topology(
                1, 1, tuple(range(65))
            )
        with self.assertRaisesRegex(DisplayProductError, r"2\^24"):
            display_product_module._checked_frame_topology(
                1 << 22, 1, (0, 1, 2, 3, 4)
            )
        maximum_tile_records = (1 << 26) // RECORD_BYTES
        self.assertEqual(
            display_product_module._checked_input_tile_bytes(
                maximum_tile_records
            ),
            maximum_tile_records * RECORD_BYTES,
        )
        with self.assertRaisesRegex(DisplayProductError, r"2\^26"):
            display_product_module._checked_input_tile_bytes(
                maximum_tile_records + 1
            )
        self.assertTrue(fixture.input_manifest.is_file())

    def test_huge_json_integer_exposure_fails_as_contract_error(self) -> None:
        fixture, _linear_root, linear_publication, _output, publication = (
            create_display_fixture(self.root)
        )
        manifest = json.loads(publication.manifest_path.read_bytes())
        manifest["displayTransform"]["descriptor"]["exposure"] = 10**3999
        publication.manifest_path.write_bytes(canonical_json_bytes(manifest))
        with self.assertRaisesRegex(
            ContractError, "JSON number exceeds verifier binary64 range"
        ):
            validate_sdr_display_quicklook(
                publication.manifest_path,
                linear_publication.manifest_path,
                fixture.output_manifest,
                fixture.input_manifest,
            )

    def test_nondefault_schema_bytes_are_rejected_even_with_the_same_id(self) -> None:
        fixture, _linear_root, linear_publication, _output, publication = (
            create_display_fixture(self.root)
        )
        schema = json.loads(display_verifier_module.DEFAULT_SCHEMA.read_bytes())
        schema["$comment"] = "not the authenticated repository schema bytes"
        alternate = self.root / "alternate-display-schema.json"
        alternate.write_bytes(canonical_json_bytes(schema))
        with self.assertRaisesRegex(ContractError, "schema must byte-match"):
            validate_sdr_display_quicklook(
                publication.manifest_path,
                linear_publication.manifest_path,
                fixture.output_manifest,
                fixture.input_manifest,
                schema_path=alternate,
            )


if __name__ == "__main__":
    unittest.main()
