from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from offline.job import canonical_json_bytes
from scripts.verify_nr_contract import ContractError
import scripts.verify_offline_linear_srgb as linear_verifier_module
from scripts.verify_offline_linear_srgb import (
    validate_scientific_linear_srgb_frame,
)
from tests.test_offline_linear_rgb_product import create_linear_srgb_fixture


ROOT = Path(__file__).resolve().parents[1]


class OfflineLinearSrgbVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cie_fixture, self.output, self.publication = (
            create_linear_srgb_fixture(self.root)
        )
        self.manifest_path = self.publication.manifest_path
        self.xyz_manifest_path = self.cie_fixture.output_manifest
        self.spectral_manifest_path = self.cie_fixture.input_manifest

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def read_manifest(self) -> dict[str, object]:
        return json.loads(self.manifest_path.read_bytes())

    def write_manifest(self, manifest: dict[str, object]) -> None:
        payload = canonical_json_bytes(manifest)
        self.manifest_path.write_bytes(payload)
        (self.output / "manifest.sha256").write_bytes(
            f"{hashlib.sha256(payload).hexdigest()}  manifest.json\n".encode(
                "ascii"
            )
        )

    def reseal(self, manifest: dict[str, object]) -> None:
        descriptor = manifest["converter"]["descriptor"]
        manifest["converter"]["descriptorSha256"] = hashlib.sha256(
            canonical_json_bytes(descriptor)
        ).hexdigest()
        configuration = {
            "cieDataset": manifest["cieDataset"],
            "converter": descriptor,
            "frame": manifest["frame"],
            "inputCieXyzProduct": manifest["inputCieXyzProduct"],
            "inputSpectralProduct": manifest["inputSpectralProduct"],
            "pixelLayout": manifest["pixelLayout"],
            "schema": manifest["schema"],
        }
        configuration_hash = hashlib.sha256(
            canonical_json_bytes(configuration)
        ).hexdigest()
        manifest["integrity"]["configurationSha256"] = configuration_hash
        identity = {
            "configurationSha256": configuration_hash,
            "schema": manifest["schema"],
            "summary": manifest["summary"],
            "tiles": manifest["tiles"],
        }
        product_hash = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        manifest["integrity"]["productSha256"] = product_hash
        manifest["id"] = f"scientific-linear-srgb-frame-{product_hash[:24]}"
        self.write_manifest(manifest)

    def rewrite_tile(
        self,
        manifest: dict[str, object],
        tile_index: int,
        payload: bytes,
    ) -> None:
        artifact = manifest["tiles"][tile_index]["outputPayload"]
        (self.output / artifact["uri"]).write_bytes(payload)
        artifact["byteLength"] = len(payload)
        artifact["sha256"] = hashlib.sha256(payload).hexdigest()

    def assert_contract_error(self, fragment: str) -> None:
        with self.assertRaisesRegex(ContractError, fragment):
            validate_scientific_linear_srgb_frame(
                self.manifest_path,
                self.xyz_manifest_path,
                self.spectral_manifest_path,
            )

    def test_valid_report_and_cli(self) -> None:
        report = validate_scientific_linear_srgb_frame(
            self.manifest_path,
            self.xyz_manifest_path,
            self.spectral_manifest_path,
        )
        self.assertTrue(report["matrixTransformVerified"])
        self.assertFalse(report["colourAlgorithmOracleIndependent"])
        self.assertFalse(report["inputPhysicsVerified"])
        self.assertEqual(report["maximumUlpDifference"], 0)
        self.assertEqual(report["recordCount"], 2)
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "verify_offline_linear_srgb.py"),
                str(self.manifest_path),
                str(self.xyz_manifest_path),
                str(self.spectral_manifest_path),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), report)

    def test_one_ulp_is_accepted_and_two_ulps_fail(self) -> None:
        manifest = self.read_manifest()
        artifact = manifest["tiles"][0]["outputPayload"]
        payload = bytearray((self.output / artifact["uri"]).read_bytes())
        original = struct.unpack_from("<d", payload, 0)[0]
        one_ulp = math.nextafter(original, math.inf)
        struct.pack_into("<d", payload, 0, one_ulp)
        self.rewrite_tile(manifest, 0, bytes(payload))
        self.reseal(manifest)
        report = validate_scientific_linear_srgb_frame(
            self.manifest_path,
            self.xyz_manifest_path,
            self.spectral_manifest_path,
        )
        self.assertEqual(report["maximumUlpDifference"], 1)
        two_ulp = math.nextafter(one_ulp, math.inf)
        struct.pack_into("<d", payload, 0, two_ulp)
        manifest = self.read_manifest()
        self.rewrite_tile(manifest, 0, bytes(payload))
        self.reseal(manifest)
        self.assert_contract_error("differs by 2 ULP")

    def test_resealed_value_and_both_record_hashes_fail(self) -> None:
        manifest = self.read_manifest()
        artifact = manifest["tiles"][0]["outputPayload"]
        payload = bytearray((self.output / artifact["uri"]).read_bytes())
        value = struct.unpack_from("<d", payload, 8)[0]
        struct.pack_into("<d", payload, 8, value * 1.01)
        self.rewrite_tile(manifest, 0, bytes(payload))
        self.reseal(manifest)
        self.assert_contract_error("matrix replay differs")

        for offset, label in ((48, "XYZ"), (80, "spectral")):
            second_root = self.root / f"digest-{label}"
            second_root.mkdir()
            fixture, output, publication = create_linear_srgb_fixture(second_root)
            self.cie_fixture = fixture
            self.output = output
            self.publication = publication
            self.manifest_path = publication.manifest_path
            self.xyz_manifest_path = fixture.output_manifest
            self.spectral_manifest_path = fixture.input_manifest
            manifest = self.read_manifest()
            artifact = manifest["tiles"][0]["outputPayload"]
            payload = bytearray((self.output / artifact["uri"]).read_bytes())
            payload[offset] ^= 1
            self.rewrite_tile(manifest, 0, bytes(payload))
            self.reseal(manifest)
            self.assert_contract_error("input record SHA-256 binding mismatch")

    def test_identity_source_schema_extra_and_symlink_tamper_fail(self) -> None:
        manifest = self.read_manifest()
        manifest["inputSpectralProduct"]["productSha256"] = "0" * 64
        self.reseal(manifest)
        self.assert_contract_error("original spectral identity mismatch")

        cases = ("source", "schema", "extra", "symlink")
        for label in cases:
            second_root = self.root / label
            second_root.mkdir()
            fixture, output, publication = create_linear_srgb_fixture(second_root)
            self.output = output
            self.manifest_path = publication.manifest_path
            self.xyz_manifest_path = fixture.output_manifest
            self.spectral_manifest_path = fixture.input_manifest
            manifest = self.read_manifest()
            if label == "source":
                manifest["converter"]["descriptor"]["sourceFiles"][0][
                    "sha256"
                ] = "0" * 64
                self.reseal(manifest)
                self.assert_contract_error("producer source hash mismatch")
            elif label == "schema":
                manifest["eightBitPreview"] = True
                self.write_manifest(manifest)
                self.assert_contract_error("unknown property")
            elif label == "extra":
                (output / "extra.txt").write_text("x", encoding="utf-8")
                self.assert_contract_error("undeclared output file")
            else:
                artifact = manifest["tiles"][0]["outputPayload"]
                tile = output / artifact["uri"]
                outside = self.root / "outside.lsrgb"
                outside.write_bytes(tile.read_bytes())
                tile.unlink()
                tile.symlink_to(outside)
                self.assert_contract_error("symlink|traversal-safe")

    def test_nondefault_schema_bytes_are_rejected_even_with_the_same_id(self) -> None:
        schema = json.loads(linear_verifier_module.DEFAULT_SCHEMA.read_bytes())
        schema["$comment"] = "not the authenticated repository schema bytes"
        alternate = self.root / "alternate-linear-schema.json"
        alternate.write_bytes(canonical_json_bytes(schema))
        with self.assertRaisesRegex(ContractError, "schema must byte-match"):
            validate_scientific_linear_srgb_frame(
                self.manifest_path,
                self.xyz_manifest_path,
                self.spectral_manifest_path,
                schema_path=alternate,
            )


if __name__ == "__main__":
    unittest.main()
