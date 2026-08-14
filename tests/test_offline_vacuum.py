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
from unittest.mock import patch

import offline.vacuum as vacuum_module
from offline.vacuum import (
    CANONICAL_NAN_FLOAT32,
    OUTCOME_CAPTURED,
    OUTCOME_ESCAPED,
    OUTCOME_UNRESOLVED,
    OUTPUT_SCHEMA,
    PlanckBlackbodyEnvironment,
    VALID_DIRECTION,
    VALID_FREQUENCY_SHIFT,
    compose_vacuum_spectrum,
    decode_transfer_record,
    render_offline_vacuum,
)
from scripts.generate_schwarzschild_transfer_map import generate_dataset
from scripts.verify_nr_contract import ContractError, validate_contract


ROOT = Path(__file__).resolve().parents[1]
TRANSFER_RECORD = struct.Struct("<7fBBH")


class CountingEnvironment:
    def __init__(self) -> None:
        self.calls: list[tuple[float, tuple[float, float, float]]] = []

    def specific_intensity_nu(
        self,
        emitted_frequency_hz: float,
        direction_icrs: tuple[float, float, float],
    ) -> float:
        self.calls.append((emitted_frequency_hz, direction_icrs))
        return emitted_frequency_hz

    def descriptor(self) -> dict[str, object]:
        return {
            "directionFrame": "ICRS-stored-escape-direction",
            "frequencyFrame": "transfer-map-escape-boundary-reference-observer",
            "implementationId": "counting-linear-frequency-oracle/v1",
            "kind": "counting-linear-frequency-oracle",
            "units": "test-only",
        }


class CountingPlanckEnvironment(PlanckBlackbodyEnvironment):
    def __init__(self, temperature_k: float = 6500.0) -> None:
        super().__init__(temperature_k)
        object.__setattr__(self, "calls", [])

    def specific_intensity_nu(
        self,
        emitted_frequency_hz: float,
        direction_icrs: tuple[float, float, float],
    ) -> float:
        self.calls.append((emitted_frequency_hz, direction_icrs))
        return super().specific_intensity_nu(
            emitted_frequency_hz,
            direction_icrs,
        )


def packed_record(
    *,
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    frequency_shift: float = 1.0,
    outcome: int = OUTCOME_ESCAPED,
    capture_target: int = 255,
    validity_mask: int = 0x1F,
) -> bytes:
    return TRANSFER_RECORD.pack(
        *direction,
        frequency_shift,
        1.0,
        0.0,
        0.0,
        outcome,
        capture_target,
        validity_mask,
    )


def generate_small_reference(path: Path) -> None:
    with redirect_stdout(io.StringIO()):
        generate_dataset(path, width=8, height=8, tile_height=4)


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    (path / "manifest.json").write_bytes(payload)
    (path / "manifest.sha256").write_bytes(
        f"{hashlib.sha256(payload).hexdigest()}  manifest.json\n".encode("ascii")
    )


def make_first_record_unresolved(dataset: Path) -> bytes:
    manifest = json.loads((dataset / "manifest.json").read_bytes())
    first_chunk_path = dataset / manifest["chunks"][0]["uri"]
    payload = bytearray(first_chunk_path.read_bytes())
    unresolved = TRANSFER_RECORD.pack(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        OUTCOME_UNRESOLVED,
        255,
        (1 << 3) | (1 << 4),
    )
    payload[: TRANSFER_RECORD.size] = unresolved
    first_chunk_path.write_bytes(payload)
    manifest["chunks"][0]["sha256"] = hashlib.sha256(payload).hexdigest()

    outcome_names = {
        code: name for name, code in manifest["recordLayout"]["rayOutcomes"].items()
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
        for name in ("unresolved", "outside-domain", "integrator-failure", "missing")
    ) / total
    manifest["accuracy"]["outcomeFractions"] = fractions
    manifest["accuracy"]["unresolvedFraction"] = counts["unresolved"] / total
    write_manifest(dataset, manifest)
    return unresolved


class OfflineVacuumTests(unittest.TestCase):
    def test_planck_environment_matches_direct_b_nu(self) -> None:
        environment = PlanckBlackbodyEnvironment(6500.0, 0.75)
        frequency = 5.0e14
        actual = environment.specific_intensity_nu(
            frequency,
            (0.0, 0.0, 1.0),
        )
        planck = 6.62607015e-34
        light_speed = 299_792_458.0
        boltzmann = 1.380649e-23
        expected = (
            0.75
            * 2.0
            * planck
            * frequency**3
            / light_speed**2
            / math.expm1(planck * frequency / (boltzmann * 6500.0))
        )
        self.assertTrue(math.isclose(actual, expected, rel_tol=2.0e-14))

    def test_escaped_transport_uses_emitted_frequency_and_g_cubed(self) -> None:
        environment = CountingEnvironment()
        record = decode_transfer_record(
            packed_record(frequency_shift=2.0),
        )
        sample = compose_vacuum_spectrum(record, (4.0, 8.0), environment)

        self.assertEqual(
            environment.calls,
            [
                (2.0, (1.0, 0.0, 0.0)),
                (4.0, (1.0, 0.0, 0.0)),
            ],
        )
        self.assertEqual(sample.specific_intensity_nu, (16.0, 32.0))
        self.assertEqual(sample.ray_outcome, OUTCOME_ESCAPED)
        self.assertEqual(sample.validity_mask, 0x1F)

    def test_captured_and_unusable_states_never_sample_environment(self) -> None:
        environment = CountingEnvironment()
        captured = decode_transfer_record(
            packed_record(
                direction=(0.0, 0.0, 0.0),
                frequency_shift=0.0,
                outcome=OUTCOME_CAPTURED,
                capture_target=0,
                validity_mask=(1 << 2) | (1 << 3) | (1 << 4),
            )
        )
        unresolved = decode_transfer_record(
            packed_record(
                direction=(0.0, 0.0, 0.0),
                frequency_shift=0.0,
                outcome=OUTCOME_UNRESOLVED,
                validity_mask=(1 << 3) | (1 << 4),
            )
        )
        invalid_escape = decode_transfer_record(
            packed_record(
                frequency_shift=2.0,
                outcome=OUTCOME_ESCAPED,
                validity_mask=VALID_DIRECTION,
            )
        )
        invalid_captured = decode_transfer_record(
            packed_record(
                direction=(0.0, 0.0, 0.0),
                frequency_shift=0.0,
                outcome=OUTCOME_CAPTURED,
                capture_target=255,
                validity_mask=(1 << 2) | (1 << 3) | (1 << 4),
            )
        )

        captured_sample = compose_vacuum_spectrum(
            captured,
            (4.0, 8.0),
            environment,
        )
        validated_captured_sample = compose_vacuum_spectrum(
            captured,
            (4.0, 8.0),
            environment,
            valid_capture_targets={0},
        )
        unresolved_sample = compose_vacuum_spectrum(
            unresolved,
            (4.0, 8.0),
            environment,
        )
        invalid_sample = compose_vacuum_spectrum(
            invalid_escape,
            (4.0, 8.0),
            environment,
        )
        invalid_captured_sample = compose_vacuum_spectrum(
            invalid_captured,
            (4.0, 8.0),
            environment,
        )

        self.assertTrue(
            all(
                math.isnan(value)
                for value in captured_sample.specific_intensity_nu
            )
        )
        self.assertEqual(
            validated_captured_sample.specific_intensity_nu,
            (0.0, 0.0),
        )
        self.assertEqual(captured_sample.capture_target, 0)
        self.assertTrue(
            all(math.isnan(value) for value in unresolved_sample.specific_intensity_nu)
        )
        self.assertEqual(unresolved_sample.ray_outcome, OUTCOME_UNRESOLVED)
        self.assertTrue(
            all(math.isnan(value) for value in invalid_sample.specific_intensity_nu)
        )
        self.assertEqual(invalid_sample.validity_mask, VALID_DIRECTION)
        self.assertTrue(
            all(
                math.isnan(value)
                for value in invalid_captured_sample.specific_intensity_nu
            )
        )
        self.assertEqual(environment.calls, [])

    def test_streamed_product_is_byte_deterministic_and_fully_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            generate_small_reference(dataset)
            environment = PlanckBlackbodyEnvironment(6500.0)
            first = render_offline_vacuum(
                dataset / "manifest.json",
                root / "render-a",
                (4.0e14, 5.0e14),
                environment,
            )
            second = render_offline_vacuum(
                dataset / "manifest.json",
                root / "render-b",
                (4.0e14, 5.0e14),
                environment,
            )

            first_files = {
                path.relative_to(first.output_directory): path.read_bytes()
                for path in first.output_directory.rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second.output_directory): path.read_bytes()
                for path in second.output_directory.rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)
            self.assertEqual(first.manifest_sha256, second.manifest_sha256)

            manifest_bytes = first.manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
            self.assertEqual(manifest["schema"], OUTPUT_SCHEMA)
            self.assertFalse(
                manifest["scientificStatus"]["isNumericalRelativitySolver"]
            )
            self.assertFalse(
                manifest["scientificStatus"]
                ["isGeneralRelativisticRadiativeTransfer"]
            )
            self.assertFalse(
                manifest["scientificStatus"]["isOpenExrScientificMaster"]
            )
            self.assertEqual(
                (first.output_directory / "manifest.sha256").read_text("ascii"),
                f"{manifest_hash}  manifest.json\n",
            )
            self.assertEqual(manifest_hash, first.manifest_sha256)
            self.assertEqual(manifest["outcomes"]["recordCount"], 64)
            self.assertEqual(manifest["outcomes"]["unusableRecordCount"], 0)
            self.assertGreater(manifest["outcomes"]["counts"]["captured"], 0)
            self.assertGreater(manifest["outcomes"]["counts"]["escaped"], 0)

            configuration = {
                "compositor": manifest["compositor"],
                "environment": manifest["environment"],
                "inputManifestSha256": manifest["inputTransferMap"][
                    "manifestSha256"
                ],
                "observerFrequencyBinsHz": manifest["observerFrequencyBinsHz"],
                "schema": manifest["schema"],
            }
            canonical = lambda value: (
                json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            configuration_digest = hashlib.sha256(canonical(configuration)).hexdigest()
            product_identity = {
                "configurationSha256": configuration_digest,
                "outputChunks": manifest["chunks"],
                "recordLayout": manifest["recordLayout"],
            }
            product_digest = hashlib.sha256(canonical(product_identity)).hexdigest()
            self.assertEqual(
                manifest["integrity"]["configurationSha256"],
                configuration_digest,
            )
            self.assertEqual(manifest["integrity"]["productSha256"], product_digest)
            changed_sample = json.loads(json.dumps(product_identity))
            changed_sample["outputChunks"][0]["sampleIndex"] += 1
            changed_count = json.loads(json.dumps(product_identity))
            changed_count["outputChunks"][0]["recordCount"] += 1
            changed_layout = json.loads(json.dumps(product_identity))
            changed_layout["recordLayout"]["state"]["recordBytes"] += 1
            for changed in (changed_sample, changed_count, changed_layout):
                self.assertNotEqual(
                    hashlib.sha256(canonical(changed)).hexdigest(),
                    product_digest,
                )

            for chunk in manifest["chunks"]:
                for name in ("spectral", "state"):
                    artifact = chunk[name]
                    path = first.output_directory / artifact["uri"]
                    payload = path.read_bytes()
                    self.assertEqual(len(payload), artifact["byteLength"])
                    self.assertEqual(
                        hashlib.sha256(payload).hexdigest(),
                        artifact["sha256"],
                    )

                spectra = (
                    first.output_directory / chunk["spectral"]["uri"]
                ).read_bytes()
                states = (
                    first.output_directory / chunk["state"]["uri"]
                ).read_bytes()
                for index in range(chunk["recordCount"]):
                    record_values = TRANSFER_RECORD.unpack_from(
                        states,
                        index * TRANSFER_RECORD.size,
                    )
                    outcome = record_values[7]
                    values = struct.unpack_from("<2f", spectra, index * 8)
                    if outcome == OUTCOME_CAPTURED:
                        self.assertEqual(values, (0.0, 0.0))
                    elif outcome == OUTCOME_ESCAPED:
                        self.assertTrue(
                            all(math.isfinite(value) and value > 0.0 for value in values)
                        )
                    else:  # The small stationary reference should have no unusable rays.
                        self.fail(f"unexpected ray outcome {outcome}")

    def test_tampered_input_is_rejected_before_environment_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            generate_small_reference(dataset)
            manifest = json.loads((dataset / "manifest.json").read_bytes())
            chunk_path = dataset / manifest["chunks"][0]["uri"]
            tampered = bytearray(chunk_path.read_bytes())
            tampered[0] ^= 0x01
            chunk_path.write_bytes(tampered)
            environment = CountingPlanckEnvironment()
            output = root / "must-not-exist"

            with self.assertRaises(ContractError):
                render_offline_vacuum(
                    dataset / "manifest.json",
                    output,
                    (4.0e14,),
                    environment,
                )
            self.assertEqual(environment.calls, [])
            self.assertFalse(output.exists())

    def test_unresolved_record_writes_canonical_nan_and_exact_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            generate_small_reference(dataset)
            unresolved = make_first_record_unresolved(dataset)
            environment = CountingPlanckEnvironment()
            result = render_offline_vacuum(
                dataset / "manifest.json",
                root / "render",
                (4.0e14,),
                environment,
            )

            manifest = json.loads(result.manifest_path.read_bytes())
            first_chunk = manifest["chunks"][0]
            spectral = (
                result.output_directory / first_chunk["spectral"]["uri"]
            ).read_bytes()
            state = (
                result.output_directory / first_chunk["state"]["uri"]
            ).read_bytes()
            self.assertEqual(spectral[:4], CANONICAL_NAN_FLOAT32)
            self.assertEqual(state[: TRANSFER_RECORD.size], unresolved)
            self.assertEqual(manifest["outcomes"]["unusableRecordCount"], 1)
            escaped = manifest["outcomes"]["counts"]["escaped"]
            self.assertEqual(len(environment.calls), escaped)

    def test_chunk_mutation_after_first_validation_never_reaches_environment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            generate_small_reference(dataset)
            manifest = json.loads((dataset / "manifest.json").read_bytes())
            chunk_path = dataset / manifest["chunks"][0]["uri"]
            environment = CountingPlanckEnvironment()
            validation_calls = 0

            def validate_then_mutate(path: Path) -> dict[str, object]:
                nonlocal validation_calls
                report = validate_contract(path)
                validation_calls += 1
                if validation_calls == 1:
                    payload = bytearray(chunk_path.read_bytes())
                    payload[0] ^= 0x01
                    chunk_path.write_bytes(payload)
                return report

            output = root / "must-not-exist"
            with patch.object(
                vacuum_module,
                "validate_contract",
                side_effect=validate_then_mutate,
            ):
                with self.assertRaises(ContractError):
                    render_offline_vacuum(
                        dataset / "manifest.json",
                        output,
                        (4.0e14,),
                        environment,
                    )
            self.assertEqual(environment.calls, [])
            self.assertFalse(output.exists())

    def test_manifest_uri_race_is_rejected_before_snapshot_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            generate_small_reference(dataset)
            environment = CountingPlanckEnvironment()
            validation_calls = 0

            def validate_then_replace_manifest(path: Path) -> dict[str, object]:
                nonlocal validation_calls
                report = validate_contract(path)
                validation_calls += 1
                if validation_calls == 1:
                    manifest = json.loads((dataset / "manifest.json").read_bytes())
                    manifest["chunks"][0]["uri"] = "../outside.bin"
                    write_manifest(dataset, manifest)
                return report

            output = root / "must-not-exist"
            with patch.object(
                vacuum_module,
                "validate_contract",
                side_effect=validate_then_replace_manifest,
            ):
                with self.assertRaises(vacuum_module.VacuumRenderError):
                    render_offline_vacuum(
                        dataset / "manifest.json",
                        output,
                        (4.0e14,),
                        environment,
                    )
            self.assertEqual(environment.calls, [])
            self.assertFalse(output.exists())

    def test_cli_is_executable(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/render_offline_vacuum.py", "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--frequency-hz", completed.stdout)


if __name__ == "__main__":
    unittest.main()
