from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from offline.cie_color import DEFAULT_CIE_CSV, DEFAULT_CIE_METADATA
from offline.job import canonical_json_bytes
from scripts.verify_nr_contract import ContractError
import scripts.verify_offline_cie_xyz as cie_verifier_module
from scripts.verify_offline_cie_xyz import (
    validate_scientific_cie_xyz_frame,
)
from tests.test_offline_cie_product import create_exact_cie_spectral_fixture


ROOT = Path(__file__).resolve().parents[1]


class OfflineCieXyzVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = create_exact_cie_spectral_fixture(self.root)
        self.output = self.fixture.output_product
        self.manifest_path = self.fixture.output_manifest
        self.input_manifest_path = self.fixture.input_manifest

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

    def reseal_product(self, manifest: dict[str, object]) -> None:
        descriptor = manifest["converter"]["descriptor"]
        manifest["converter"]["descriptorSha256"] = hashlib.sha256(
            canonical_json_bytes(descriptor)
        ).hexdigest()
        configuration = {
            "cieDataset": manifest["cieDataset"],
            "converter": descriptor,
            "frame": manifest["frame"],
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
        manifest["id"] = f"scientific-cie-xyz-frame-{product_hash[:24]}"
        self.write_manifest(manifest)

    def rewrite_output_tile(
        self,
        manifest: dict[str, object],
        tile_index: int,
        payload: bytes,
    ) -> None:
        artifact = manifest["tiles"][tile_index]["outputPayload"]
        (self.output / artifact["uri"]).write_bytes(payload)
        artifact["byteLength"] = len(payload)
        artifact["sha256"] = hashlib.sha256(payload).hexdigest()

    def assert_contract_error(self, fragment: str, **kwargs: object) -> None:
        with self.assertRaisesRegex(ContractError, fragment):
            validate_scientific_cie_xyz_frame(
                self.manifest_path,
                self.input_manifest_path,
                **kwargs,
            )

    def test_valid_report_and_cli_recompute_every_record(self) -> None:
        first = validate_scientific_cie_xyz_frame(
            self.manifest_path,
            self.input_manifest_path,
        )
        second = validate_scientific_cie_xyz_frame(
            self.manifest_path,
            self.input_manifest_path,
        )
        self.assertEqual(first, second)
        self.assertTrue(first["cieIntegrationVerified"])
        self.assertFalse(first["inputPhysicsVerified"])
        self.assertEqual(first["maximumUlpDifference"], 0)
        self.assertEqual(first["recordCount"], 2)
        self.assertEqual(first["tileCount"], 2)
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "verify_offline_cie_xyz.py"),
                str(self.manifest_path),
                str(self.input_manifest_path),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), first)

    def test_one_ulp_is_accepted_but_two_ulps_fail(self) -> None:
        manifest = self.read_manifest()
        artifact = manifest["tiles"][0]["outputPayload"]
        path = self.output / artifact["uri"]
        payload = bytearray(path.read_bytes())
        original = struct.unpack_from("<d", payload, 0)[0]
        one_ulp = math.nextafter(original, math.inf)
        struct.pack_into("<d", payload, 0, one_ulp)
        self.rewrite_output_tile(manifest, 0, bytes(payload))
        self.reseal_product(manifest)
        report = validate_scientific_cie_xyz_frame(
            self.manifest_path,
            self.input_manifest_path,
        )
        self.assertEqual(report["maximumUlpDifference"], 1)

        two_ulp = math.nextafter(one_ulp, math.inf)
        struct.pack_into("<d", payload, 0, two_ulp)
        manifest = self.read_manifest()
        self.rewrite_output_tile(manifest, 0, bytes(payload))
        self.reseal_product(manifest)
        self.assert_contract_error("differs by 2 ULP")

    def test_resealed_xyz_and_record_binding_tampering_fail(self) -> None:
        manifest = self.read_manifest()
        artifact = manifest["tiles"][0]["outputPayload"]
        path = self.output / artifact["uri"]
        payload = bytearray(path.read_bytes())
        value = struct.unpack_from("<d", payload, 8)[0]
        struct.pack_into("<d", payload, 8, value * 1.01)
        self.rewrite_output_tile(manifest, 0, bytes(payload))
        self.reseal_product(manifest)
        self.assert_contract_error("recomputed value differs")

        second_root = self.root / "digest-case"
        second_root.mkdir()
        second = create_exact_cie_spectral_fixture(second_root)
        self.output = second.output_product
        self.manifest_path = second.output_manifest
        self.input_manifest_path = second.input_manifest
        manifest = self.read_manifest()
        artifact = manifest["tiles"][0]["outputPayload"]
        payload = bytearray((self.output / artifact["uri"]).read_bytes())
        payload[48] ^= 1
        self.rewrite_output_tile(manifest, 0, bytes(payload))
        self.reseal_product(manifest)
        self.assert_contract_error("input record SHA-256 binding mismatch")

    def test_converter_source_hash_and_tile_topology_are_recomputed(self) -> None:
        manifest = self.read_manifest()
        manifest["converter"]["descriptor"]["sourceFiles"][0]["sha256"] = "0" * 64
        self.reseal_product(manifest)
        self.assert_contract_error("converter source hash mismatch")

        second_root = self.root / "topology-case"
        second_root.mkdir()
        second = create_exact_cie_spectral_fixture(second_root)
        self.output = second.output_product
        self.manifest_path = second.output_manifest
        self.input_manifest_path = second.input_manifest
        manifest = self.read_manifest()
        manifest["tiles"][0]["tile"]["x"] = 1
        self.reseal_product(manifest)
        self.assert_contract_error("tile topology differs")

    def test_every_package_initializer_dependency_hash_drift_is_rejected(self) -> None:
        original_read = cie_verifier_module._read_stable_file
        package_dependencies = (
            "offline/__init__.py",
            "offline/geodesic.py",
            "offline/kerr.py",
            "offline/radiative_transfer.py",
            "offline/spacetime.py",
        )
        for module_uri in package_dependencies:
            selected = ROOT / module_uri

            def drift_selected_source(
                path: Path,
                label: str,
                maximum_bytes: int,
                *,
                selected: Path = selected,
            ) -> bytes:
                payload = original_read(path, label, maximum_bytes)
                if Path(path).absolute() == selected.absolute():
                    return payload + b"\n# simulated source drift\n"
                return payload

            with self.subTest(module_uri=module_uri), patch.object(
                cie_verifier_module,
                "_read_stable_file",
                side_effect=drift_selected_source,
            ):
                self.assert_contract_error("converter source hash mismatch")

    def test_schema_extra_files_and_symlinks_fail_closed(self) -> None:
        manifest = self.read_manifest()
        manifest["displaySrgb"] = [1.0, 1.0, 1.0]
        self.write_manifest(manifest)
        self.assert_contract_error("unknown property")

        second_root = self.root / "extra-case"
        second_root.mkdir()
        second = create_exact_cie_spectral_fixture(second_root)
        self.output = second.output_product
        self.manifest_path = second.output_manifest
        self.input_manifest_path = second.input_manifest
        (self.output / "extra.txt").write_text("forbidden", encoding="utf-8")
        self.assert_contract_error("undeclared output file")

        (self.output / "extra.txt").unlink()
        manifest = self.read_manifest()
        artifact = manifest["tiles"][0]["outputPayload"]
        tile = self.output / artifact["uri"]
        outside = self.root / "outside.cxyz"
        outside.write_bytes(tile.read_bytes())
        tile.unlink()
        tile.symlink_to(outside)
        self.assert_contract_error("symlink|traversal-safe")

    def test_nondefault_schema_bytes_are_rejected_even_with_the_same_id(self) -> None:
        schema = json.loads(cie_verifier_module.DEFAULT_SCHEMA.read_bytes())
        schema["$comment"] = "not the authenticated repository schema bytes"
        alternate = self.root / "alternate-cie-schema.json"
        alternate.write_bytes(canonical_json_bytes(schema))
        self.assert_contract_error(
            "schema must byte-match",
            schema_path=alternate,
        )

    def test_input_and_cie_assets_are_reauthenticated(self) -> None:
        (self.fixture.input_product / "undeclared.txt").write_text(
            "x",
            encoding="utf-8",
        )
        self.assert_contract_error("input verification failed")
        (self.fixture.input_product / "undeclared.txt").unlink()

        csv_path = self.root / DEFAULT_CIE_CSV.name
        metadata_path = self.root / DEFAULT_CIE_METADATA.name
        shutil.copyfile(DEFAULT_CIE_CSV, csv_path)
        shutil.copyfile(DEFAULT_CIE_METADATA, metadata_path)
        payload = bytearray(csv_path.read_bytes())
        payload[10] ^= 1
        csv_path.write_bytes(payload)
        self.assert_contract_error(
            "CIE authentication failed",
            cie_csv_path=csv_path,
            cie_metadata_path=metadata_path,
        )


if __name__ == "__main__":
    unittest.main()
