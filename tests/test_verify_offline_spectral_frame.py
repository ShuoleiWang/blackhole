from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from offline.job import InputArtifact, JobSpec, TaskKey, canonical_json_bytes
from scripts.verify_offline_spectral_frame import (
    validate_scientific_spectral_frame,
)
from scripts.verify_nr_contract import ContractError
from tests.test_offline_spectral_product import create_product_fixture


ROOT = Path(__file__).resolve().parents[1]


class OfflineSpectralFrameVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = create_product_fixture(self.root)
        self.output = self.fixture.output
        self.manifest_path = self.fixture.publication.manifest_path

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

    def job_spec_from_manifest(self, manifest: dict[str, object]) -> JobSpec:
        raw = manifest["producer"]["jobSpec"]
        return JobSpec(
            producer=raw["producer"],
            algorithm_version=raw["algorithmVersion"],
            tasks=tuple(
                TaskKey(
                    task["sampleIndex"],
                    task["y"],
                    task["x"],
                    task["width"],
                    task["height"],
                )
                for task in raw["tasks"]
            ),
            parameters=raw["parameters"],
            inputs=tuple(
                InputArtifact(
                    item["uri"],
                    item["byteLength"],
                    item["sha256"],
                )
                for item in raw["inputs"]
            ),
            producer_source_hashes=tuple(raw["producerSourceHashes"]),
            record_bytes=raw["recordBytes"],
        )

    def reseal_identity(
        self,
        manifest: dict[str, object],
        *,
        bind_job_tasks_to_tiles: bool = False,
    ) -> None:
        producer = manifest["producer"]
        raw = producer["jobSpec"]
        parameters = {
            "adaptivePixelOptions": manifest["adaptivePixelOptions"],
            "frame": manifest["frame"],
            "numericBackend": manifest["runtimeNumericBackend"]["descriptor"],
            "observerFrequencyBinsHz": manifest["observerFrequencyBinsHz"],
            "pixelLayout": manifest["pixelLayout"],
            "samplerDescriptor": manifest["sampler"]["descriptor"],
            "schema": manifest["schema"],
        }
        raw["parameters"] = parameters
        if bind_job_tasks_to_tiles:
            raw["tasks"] = [
                {
                    "height": entry["tile"]["height"],
                    "sampleIndex": entry["sampleIndex"],
                    "width": entry["tile"]["width"],
                    "x": entry["tile"]["x"],
                    "y": entry["tile"]["y"],
                }
                for entry in manifest["tiles"]
            ]
        spec = self.job_spec_from_manifest(manifest)
        producer["jobSpec"] = spec.as_dict()
        producer["id"] = spec.producer
        producer["algorithmVersion"] = spec.algorithm_version
        producer["jobKey"] = spec.job_key
        manifest["sampler"]["descriptorSha256"] = hashlib.sha256(
            canonical_json_bytes(manifest["sampler"]["descriptor"])
        ).hexdigest()
        manifest["runtimeNumericBackend"]["descriptorSha256"] = hashlib.sha256(
            canonical_json_bytes(
                manifest["runtimeNumericBackend"]["descriptor"]
            )
        ).hexdigest()
        configuration = {
            **parameters,
            "jobKey": spec.job_key,
            "jobSpec": spec.as_dict(),
        }
        configuration_hash = hashlib.sha256(
            canonical_json_bytes(configuration)
        ).hexdigest()
        manifest["integrity"]["configurationSha256"] = configuration_hash
        product_identity = {
            "configurationSha256": configuration_hash,
            "schema": manifest["schema"],
            "summary": manifest["summary"],
            "tiles": manifest["tiles"],
        }
        product_hash = hashlib.sha256(
            canonical_json_bytes(product_identity)
        ).hexdigest()
        manifest["integrity"]["productSha256"] = product_hash
        manifest["id"] = f"scientific-spectral-frame-{product_hash[:24]}"
        self.write_manifest(manifest)

    def rewrite_tile(
        self,
        manifest: dict[str, object],
        tile_index: int,
        payload: bytes,
    ) -> None:
        artifact = manifest["tiles"][tile_index]["payload"]
        (self.output / artifact["uri"]).write_bytes(payload)
        artifact["byteLength"] = len(payload)
        artifact["sha256"] = hashlib.sha256(payload).hexdigest()

    def assert_contract_error(self, fragment: str) -> None:
        with self.assertRaisesRegex(
            ContractError,
            fragment,
        ):
            validate_scientific_spectral_frame(self.manifest_path)

    def test_valid_report_is_deterministic_and_cli_conformant(self) -> None:
        first = validate_scientific_spectral_frame(self.manifest_path)
        second = validate_scientific_spectral_frame(self.manifest_path)
        self.assertEqual(first, second)
        self.assertEqual(first["recordCount"], 8)
        self.assertEqual(first["tileCount"], 4)
        self.assertFalse(first["physicsVerified"])
        self.assertEqual(
            first["provenanceScope"],
            "unknown-producer-or-sampler-structural-only",
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "verify_offline_spectral_frame.py"),
                str(self.manifest_path),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), first)

    def test_schema_and_parser_reject_unknown_and_duplicate_fields(self) -> None:
        manifest = self.read_manifest()
        manifest["inventedPhysicsClaim"] = True
        self.write_manifest(manifest)
        self.assert_contract_error("unknown property")

        original = self.fixture.publication.manifest_path.read_bytes()
        duplicate = (
            b'{"schema":"blackhole.scientific-spectral-frame/v1",'
            + original[1:]
        )
        self.manifest_path.write_bytes(duplicate)
        (self.output / "manifest.sha256").write_bytes(
            f"{hashlib.sha256(duplicate).hexdigest()}  manifest.json\n".encode(
                "ascii"
            )
        )
        self.assert_contract_error("duplicate JSON object key")

    def test_manifest_sidecar_and_tile_hash_are_authenticated(self) -> None:
        (self.output / "manifest.sha256").write_text(
            f"{'0' * 64}  manifest.json\n",
            encoding="ascii",
        )
        self.assert_contract_error("sidecar must exactly")

        self.write_manifest(self.read_manifest())
        manifest = self.read_manifest()
        tile_path = self.output / manifest["tiles"][0]["payload"]["uri"]
        payload = bytearray(tile_path.read_bytes())
        payload[0] ^= 1
        tile_path.write_bytes(payload)
        self.assert_contract_error("tile hash mismatch")

    def test_paths_reject_symlinks_traversal_and_undeclared_files(self) -> None:
        manifest = self.read_manifest()
        tile_path = self.output / manifest["tiles"][0]["payload"]["uri"]
        outside = self.root / "outside.spx"
        outside.write_bytes(tile_path.read_bytes())
        tile_path.unlink()
        tile_path.symlink_to(outside)
        self.assert_contract_error("symlinked")

        tile_path.unlink()
        tile_path.write_bytes(outside.read_bytes())
        (self.output / "undeclared.txt").write_text("x", encoding="ascii")
        self.assert_contract_error("undeclared output file")

        (self.output / "undeclared.txt").unlink()
        manifest["tiles"][0]["payload"]["uri"] = "../outside.spx"
        self.write_manifest(manifest)
        self.assert_contract_error("pattern|normalized|traversal")

    def test_resealed_overlap_and_gap_fail_topology_validation(self) -> None:
        manifest = self.read_manifest()
        entry = manifest["tiles"][1]
        old_path = self.output / entry["payload"]["uri"]
        payload = old_path.read_bytes()
        entry["tile"]["x"] = 1
        entry["payload"]["uri"] = (
            "tiles/t000000-y000000-x000001-w000002-h000001.spx"
        )
        new_path = self.output / entry["payload"]["uri"]
        new_path.write_bytes(payload)
        old_path.unlink()
        self.reseal_identity(manifest, bind_job_tasks_to_tiles=True)
        self.assert_contract_error("overlap|coverage gap")

    def test_resealed_binary_gate_and_direction_sentinel_tampering_fails(self) -> None:
        manifest = self.read_manifest()
        tile = manifest["tiles"][0]
        path = self.output / tile["payload"]["uri"]
        payload = bytearray(path.read_bytes())
        base = 16 * len(manifest["observerFrequencyBinsHz"])
        mask = struct.unpack_from("<I", payload, base + 148)[0]
        struct.pack_into("<I", payload, base + 148, mask & ~1)
        self.rewrite_tile(manifest, 0, bytes(payload))
        self.reseal_identity(manifest)
        self.assert_contract_error("convergence gate")

        second_root = self.root / "direction-case"
        second_root.mkdir()
        second = create_product_fixture(second_root)
        self.output = second.output
        self.manifest_path = second.publication.manifest_path
        manifest = self.read_manifest()
        tile = manifest["tiles"][0]
        path = self.output / tile["payload"]["uri"]
        payload = bytearray(path.read_bytes())
        base = 16 * len(manifest["observerFrequencyBinsHz"])
        struct.pack_into("<d", payload, base + 56, 0.25)
        self.rewrite_tile(manifest, 0, bytes(payload))
        self.reseal_identity(manifest)
        self.assert_contract_error("absent.*direction")

    def test_resealed_summary_tamper_is_independently_recomputed(self) -> None:
        manifest = self.read_manifest()
        manifest["summary"]["totalRaySamples"] += 1
        self.reseal_identity(manifest)
        self.assert_contract_error("summary does not match decoded records")

    def test_recognized_ids_never_claim_physics_verification(self) -> None:
        manifest = self.read_manifest()
        manifest["sampler"]["descriptor"]["implementationId"] = (
            "exact-kerr-nt-spectral-ray-sampler/v2"
        )
        self.reseal_identity(manifest)
        report = validate_scientific_spectral_frame(self.manifest_path)
        self.assertEqual(
            report["provenanceScope"],
            "recognized-kerr-identifiers-structural-only",
        )
        self.assertFalse(report["physicsVerified"])


if __name__ == "__main__":
    unittest.main()
