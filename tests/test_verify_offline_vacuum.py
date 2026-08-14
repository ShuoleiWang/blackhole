from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from offline.vacuum import PlanckBlackbodyEnvironment, render_offline_vacuum
from scripts.generate_schwarzschild_transfer_map import generate_dataset
from scripts.verify_nr_contract import ContractError
from scripts.verify_offline_vacuum import validate_offline_vacuum_product


ROOT = Path(__file__).resolve().parents[1]
TRANSFER_RECORD = struct.Struct("<7fBBH")
UINT32 = struct.Struct("<I")


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def write_transfer_manifest(dataset: Path, manifest: dict[str, object]) -> None:
    payload = canonical_json_bytes(manifest)
    (dataset / "manifest.json").write_bytes(payload)
    (dataset / "manifest.sha256").write_bytes(
        f"{hashlib.sha256(payload).hexdigest()}  manifest.json\n".encode("ascii")
    )


def make_first_record_unresolved(dataset: Path) -> None:
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    chunk_path = dataset / manifest["chunks"][0]["uri"]
    payload = bytearray(chunk_path.read_bytes())
    payload[: TRANSFER_RECORD.size] = TRANSFER_RECORD.pack(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2,
        255,
        (1 << 3) | (1 << 4),
    )
    chunk_path.write_bytes(payload)
    manifest["chunks"][0]["sha256"] = hashlib.sha256(payload).hexdigest()

    outcome_names = {
        int(code): name
        for name, code in manifest["recordLayout"]["rayOutcomes"].items()
    }
    counts = {name: 0 for name in outcome_names.values()}
    total = 0
    for chunk in manifest["chunks"]:
        chunk_payload = (dataset / chunk["uri"]).read_bytes()
        for offset in range(0, len(chunk_payload), TRANSFER_RECORD.size):
            outcome = TRANSFER_RECORD.unpack_from(chunk_payload, offset)[7]
            counts[outcome_names[outcome]] += 1
            total += 1
    fractions = {name: count / total for name, count in counts.items()}
    fractions["unusable"] = sum(
        counts[name]
        for name in (
            "unresolved",
            "outside-domain",
            "integrator-failure",
            "missing",
        )
    ) / total
    manifest["accuracy"]["outcomeFractions"] = fractions
    manifest["accuracy"]["unresolvedFraction"] = counts["unresolved"] / total
    write_transfer_manifest(dataset, manifest)


class OfflineVacuumVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root / "input"
        with redirect_stdout(io.StringIO()):
            generate_dataset(self.input, width=8, height=8, tile_height=4)
        self.output = self.root / "output"
        self.frequencies = (4.0e14, 5.0e14)
        render_offline_vacuum(
            self.input / "manifest.json",
            self.output,
            self.frequencies,
            PlanckBlackbodyEnvironment(6500.0),
        )
        self.manifest_path = self.output / "manifest.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def read_manifest(self) -> dict[str, object]:
        return json.loads(self.manifest_path.read_bytes())

    def write_manifest(
        self,
        manifest: dict[str, object],
        *,
        reseal_identity: bool = False,
    ) -> None:
        if reseal_identity:
            configuration = {
                "compositor": manifest["compositor"],
                "environment": manifest["environment"],
                "inputManifestSha256": manifest["inputTransferMap"][
                    "manifestSha256"
                ],
                "observerFrequencyBinsHz": manifest["observerFrequencyBinsHz"],
                "schema": manifest["schema"],
            }
            configuration_digest = hashlib.sha256(
                canonical_json_bytes(configuration)
            ).hexdigest()
            manifest["integrity"]["configurationSha256"] = configuration_digest
            product_identity = {
                "configurationSha256": configuration_digest,
                "outputChunks": manifest["chunks"],
                "recordLayout": manifest["recordLayout"],
            }
            product_digest = hashlib.sha256(
                canonical_json_bytes(product_identity)
            ).hexdigest()
            manifest["integrity"]["productSha256"] = product_digest
            manifest["id"] = f"offline-vacuum-{product_digest[:20]}"
        payload = canonical_json_bytes(manifest)
        self.manifest_path.write_bytes(payload)
        (self.output / "manifest.sha256").write_bytes(
            f"{hashlib.sha256(payload).hexdigest()}  manifest.json\n".encode(
                "ascii"
            )
        )

    def write_artifact(
        self,
        manifest: dict[str, object],
        chunk_index: int,
        artifact_name: str,
        payload: bytes,
    ) -> None:
        artifact = manifest["chunks"][chunk_index][artifact_name]
        (self.output / artifact["uri"]).write_bytes(payload)
        artifact["byteLength"] = len(payload)
        artifact["sha256"] = hashlib.sha256(payload).hexdigest()

    def find_outcome(self, outcome: int) -> tuple[int, int]:
        manifest = self.read_manifest()
        for chunk_index, chunk in enumerate(manifest["chunks"]):
            state = (self.output / chunk["state"]["uri"]).read_bytes()
            for record_index in range(chunk["recordCount"]):
                values = TRANSFER_RECORD.unpack_from(
                    state,
                    record_index * TRANSFER_RECORD.size,
                )
                if values[7] == outcome:
                    return chunk_index, record_index
        self.fail(f"generated fixture has no ray outcome {outcome}")

    def assert_verification_error(self, fragment: str) -> None:
        with self.assertRaisesRegex(ContractError, fragment):
            validate_offline_vacuum_product(
                self.manifest_path,
                self.input / "manifest.json",
            )

    def test_valid_product_has_deterministic_independent_report(self) -> None:
        first = validate_offline_vacuum_product(
            self.manifest_path,
            self.input / "manifest.json",
        )
        second = validate_offline_vacuum_product(
            self.manifest_path,
            self.input / "manifest.json",
        )
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "offline-vacuum-contract-conformant")
        self.assertEqual(first["records"], 64)
        self.assertEqual(first["chunks"], 2)
        self.assertEqual(first["maxPlanckFloat32Ulp"], 0)

    def test_schema_rejects_unknown_fields_and_strict_json_duplicates(self) -> None:
        manifest = self.read_manifest()
        manifest["unknownClaim"] = True
        self.write_manifest(manifest)
        self.assert_verification_error("unknown property")

        original = self.manifest_path.read_bytes()
        duplicate = b'{"schema":"blackhole.offline-vacuum-spectral/v1",' + original[1:]
        self.manifest_path.write_bytes(duplicate)
        (self.output / "manifest.sha256").write_bytes(
            f"{hashlib.sha256(duplicate).hexdigest()}  manifest.json\n".encode("ascii")
        )
        self.assert_verification_error("duplicate JSON object key")

    def test_manifest_and_artifact_hashes_are_authenticated(self) -> None:
        (self.output / "manifest.sha256").write_text(
            f"{'0' * 64}  manifest.json\n",
            encoding="ascii",
        )
        self.assert_verification_error("sidecar must exactly")

        manifest = self.read_manifest()
        spectral_path = self.output / manifest["chunks"][0]["spectral"]["uri"]
        payload = bytearray(spectral_path.read_bytes())
        payload[0] ^= 1
        spectral_path.write_bytes(payload)
        self.write_manifest(manifest)
        self.assert_verification_error("hash mismatch")

    def test_paths_reject_symlinks_and_undeclared_files(self) -> None:
        manifest = self.read_manifest()
        spectral_path = self.output / manifest["chunks"][0]["spectral"]["uri"]
        target = self.root / "outside.f32"
        target.write_bytes(spectral_path.read_bytes())
        spectral_path.unlink()
        spectral_path.symlink_to(target)
        self.assert_verification_error("symlinked")

        spectral_path.unlink()
        spectral_path.write_bytes(target.read_bytes())
        (self.output / "unhashed.txt").write_text("not authenticated", encoding="utf-8")
        self.assert_verification_error("undeclared output file")

    def test_input_manifest_and_tile_topology_are_exactly_bound(self) -> None:
        manifest = self.read_manifest()
        manifest["inputTransferMap"]["id"] = "different-valid-id"
        self.write_manifest(manifest, reseal_identity=True)
        self.assert_verification_error("does not exactly match")

        manifest = self.read_manifest()
        manifest["inputTransferMap"]["id"] = json.loads(
            (self.input / "manifest.json").read_bytes()
        )["id"]
        manifest["chunks"][0]["sampleIndex"] = 1
        self.write_manifest(manifest, reseal_identity=True)
        self.assert_verification_error("does not match the validated input chunk")

    def test_state_must_be_byte_exact_validated_input_abi(self) -> None:
        manifest = self.read_manifest()
        state_path = self.output / manifest["chunks"][0]["state"]["uri"]
        state = bytearray(state_path.read_bytes())
        state[0] ^= 1
        self.write_artifact(manifest, 0, "state", bytes(state))
        self.write_manifest(manifest, reseal_identity=True)
        self.assert_verification_error("byte-exact input record copy")

    def test_captured_policy_requires_positive_zero_bits(self) -> None:
        chunk_index, record_index = self.find_outcome(1)
        manifest = self.read_manifest()
        chunk = manifest["chunks"][chunk_index]
        spectral_path = self.output / chunk["spectral"]["uri"]
        spectral = bytearray(spectral_path.read_bytes())
        offset = record_index * len(self.frequencies) * 4
        spectral[offset : offset + 4] = b"\x00\x00\x00\x80"  # -0.0
        self.write_artifact(manifest, chunk_index, "spectral", bytes(spectral))
        self.write_manifest(manifest, reseal_identity=True)
        self.assert_verification_error("positive float32 zero")

    def test_planck_liouville_values_are_recomputed_to_one_ulp(self) -> None:
        chunk_index, record_index = self.find_outcome(0)
        manifest = self.read_manifest()
        chunk = manifest["chunks"][chunk_index]
        spectral_path = self.output / chunk["spectral"]["uri"]
        spectral = bytearray(spectral_path.read_bytes())
        offset = record_index * len(self.frequencies) * 4
        bits = UINT32.unpack_from(spectral, offset)[0]
        self.assertLess(bits, 0x7F7FFFFD)
        UINT32.pack_into(spectral, offset, bits + 2)
        self.write_artifact(manifest, chunk_index, "spectral", bytes(spectral))
        self.write_manifest(manifest, reseal_identity=True)
        self.assert_verification_error("Planck/Liouville mismatch by 2 float32 ULP")

    def test_unusable_policy_requires_canonical_nan_bits(self) -> None:
        make_first_record_unresolved(self.input)
        unresolved_output = self.root / "unresolved-output"
        render_offline_vacuum(
            self.input / "manifest.json",
            unresolved_output,
            self.frequencies,
            PlanckBlackbodyEnvironment(6500.0),
        )
        self.output = unresolved_output
        self.manifest_path = unresolved_output / "manifest.json"
        manifest = self.read_manifest()
        spectral_path = self.output / manifest["chunks"][0]["spectral"]["uri"]
        spectral = bytearray(spectral_path.read_bytes())
        self.assertTrue(math.isnan(struct.unpack_from("<f", spectral, 0)[0]))
        spectral[:4] = b"\x01\x00\xc0\x7f"  # Quiet NaN, but not canonical.
        self.write_artifact(manifest, 0, "spectral", bytes(spectral))
        self.write_manifest(manifest, reseal_identity=True)
        self.assert_verification_error("canonical float32 quiet NaN")

    def test_configuration_and_product_identity_are_recomputed(self) -> None:
        manifest = self.read_manifest()
        manifest["environment"]["normalization"] = 2.0
        self.write_manifest(manifest)
        self.assert_verification_error("configuration digest mismatch")

        manifest = self.read_manifest()
        manifest["integrity"]["configurationSha256"] = hashlib.sha256(
            canonical_json_bytes(
                {
                    "compositor": manifest["compositor"],
                    "environment": manifest["environment"],
                    "inputManifestSha256": manifest["inputTransferMap"][
                        "manifestSha256"
                    ],
                    "observerFrequencyBinsHz": manifest[
                        "observerFrequencyBinsHz"
                    ],
                    "schema": manifest["schema"],
                }
            )
        ).hexdigest()
        self.write_manifest(manifest)
        self.assert_verification_error("product identity digest mismatch")

    def test_cli_is_executable_and_fail_closed(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/verify_offline_vacuum.py",
                str(self.manifest_path),
                "--input-manifest",
                str(self.input / "manifest.json"),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Offline vacuum product checks passed", completed.stdout)
        self.assertIn("offline-vacuum-contract-conformant", completed.stdout)


if __name__ == "__main__":
    unittest.main()
