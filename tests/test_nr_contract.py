from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.verify_nr_contract import (
    ContractError,
    DEFAULT_SCHEMA,
    RECORD_STRUCT,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "assets" / "transfer-maps" / "contract-fixture-v1"


class NrContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.dataset = Path(self.temporary.name) / "contract-fixture-v1"
        shutil.copytree(FIXTURE, self.dataset)
        self.manifest_path = self.dataset / "manifest.json"
        self.original_manifest_bytes = self.manifest_path.read_bytes()
        self.original = json.loads(self.original_manifest_bytes)
        self.chunk_path = self.dataset / self.original["chunks"][0]["uri"]
        self.original_chunk = self.chunk_path.read_bytes()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_manifest(self, data: dict[str, object]) -> None:
        payload = (
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self.manifest_path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        (self.dataset / "manifest.sha256").write_text(
            f"{digest}  manifest.json\n",
            encoding="ascii",
        )

    def write_chunk(
        self,
        data: dict[str, object],
        payload: bytes,
        chunk_index: int = 0,
    ) -> None:
        chunk = data["chunks"][chunk_index]
        path = self.dataset / chunk["uri"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        chunk["byteLength"] = len(payload)
        chunk["sha256"] = hashlib.sha256(payload).hexdigest()

    def assert_contract_error(self, fragment: str) -> None:
        with self.assertRaisesRegex(ContractError, fragment):
            validate_contract(self.manifest_path, DEFAULT_SCHEMA)

    def mutate_record(
        self,
        record_index: int,
        value_index: int,
        value: float | int,
    ) -> None:
        data = copy.deepcopy(self.original)
        payload = bytearray(self.original_chunk)
        values = list(
            RECORD_STRUCT.unpack_from(payload, record_index * RECORD_STRUCT.size)
        )
        values[value_index] = value
        RECORD_STRUCT.pack_into(
            payload,
            record_index * RECORD_STRUCT.size,
            *values,
        )
        self.write_chunk(data, bytes(payload))
        self.write_manifest(data)

    @staticmethod
    def mark_accuracy_measured(data: dict[str, object]) -> None:
        accuracy = data["accuracy"]
        accuracy["status"] = "measured"
        accuracy["notMeasuredReason"] = None
        accuracy["fixtureAssertions"] = None
        for name in (
            "nrConvergence",
            "constraintNorms",
            "geodesicNullResidual",
            "interpolationError",
        ):
            section = accuracy[name]
            section["status"] = "measured"
            section["method"] = "deterministic test measurement"
            section["quantity"] = f"test {name}"
            section["value"] = 0.0

    def make_stationary_renderable(self) -> dict[str, object]:
        data = copy.deepcopy(self.original)
        data["datasetKind"] = "stationary-reference-transfer-map"
        data["renderable"] = True
        data["physicalSystem"]["kind"] = "stationary-black-hole"
        data["physicalSystem"]["vacuum"] = True
        data["coordinates"]["nrChart"]["status"] = "declared"
        data["rayIntegration"]["spacetimeMode"] = "stationary"
        data["provenance"]["sourceSimulation"]["kind"] = "stationary-reference"
        data["provenance"]["sourceSimulation"]["notApplicableReason"] = None
        self.mark_accuracy_measured(data)
        for name in ("nrConvergence", "constraintNorms"):
            data["accuracy"][name]["status"] = "not-applicable"
            data["accuracy"][name]["method"] = None
            data["accuracy"][name]["value"] = None
        data["accuracy"]["unresolvedFraction"] = 1.0 / 8.0
        data["accuracy"]["outcomeFractions"] = {
            "escaped": 2.0 / 8.0,
            "captured": 2.0 / 8.0,
            "unresolved": 1.0 / 8.0,
            "outside-domain": 1.0 / 8.0,
            "integrator-failure": 1.0 / 8.0,
            "missing": 1.0 / 8.0,
            "unusable": 4.0 / 8.0,
        }
        return data

    def make_pseudo_nr(self) -> dict[str, object]:
        data = copy.deepcopy(self.original)
        data["datasetKind"] = "nr-slow-light-transfer-map"
        for flag in (
            "sourceIsNumericalRelativity",
            "derivedFromNearZoneSpacetime",
            "derivedWithSlowLightGeodesics",
        ):
            data["scientificStatus"][flag] = True
        physical = data["physicalSystem"]
        physical.update(
            {
                "kind": "binary-black-hole",
                "vacuum": True,
                "parameterEpochProtocolM": 0.0,
                "massRatioQ": 1.0,
                "dimensionlessSpins": [
                    {"componentId": "A", "vector": [0.0, 0.0, 0.0]},
                    {"componentId": "B", "vector": [0.0, 0.0, 0.0]},
                ],
                "eccentricity": 0.0,
                "referenceOrbitalPhaseRad": 0.0,
                "remnant": {
                    "massFraction": 0.95,
                    "dimensionlessSpin": [0.0, 0.0, 0.68],
                },
                "notApplicableReason": None,
            }
        )
        data["coordinates"]["nrChart"]["status"] = "declared"
        simulation = data["provenance"]["sourceSimulation"]
        simulation.update(
            {
                "kind": "catalog-simulation",
                "catalog": "SXS",
                "doi": "10.1234/test",
                "evolutionCode": {
                    "name": "TestNR",
                    "release": "1.0",
                    "commit": "0123456789abcdef",
                    "commitNotAvailableReason": None,
                },
                "notApplicableReason": None,
            }
        )
        data["rayIntegration"]["spacetimeMode"] = "time-dependent"
        data["rayIntegration"]["tolerances"] = {
            "absolute": 1.0e-10,
            "relative": 1.0e-10,
            "nullConstraint": 1.0e-8,
        }
        data["units"]["massNormalization"]["referenceEpochSourceM"] = 0.0
        for target in data["captureTargets"]:
            target["sourceArtifactRole"] = "horizon-data"
        self.mark_accuracy_measured(data)
        return data

    def test_fixture_passes_with_deterministic_report(self) -> None:
        first = validate_contract(self.manifest_path, DEFAULT_SCHEMA)
        second = validate_contract(self.manifest_path, DEFAULT_SCHEMA)
        self.assertEqual(first, second)
        self.assertEqual(first["records"], 8)
        self.assertEqual(first["maxFrameError"], "0.000e+00")
        self.assertIn("escaped:2", first["outcomes"])

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers(self) -> None:
        cases = (
            (
                "duplicate JSON object key",
                b'{"schema":"blackhole.nr-transfer-map/v1",'
                + self.original_manifest_bytes[1:],
            ),
            (
                "non-finite JSON number",
                self.original_manifest_bytes.replace(
                    b'"renderable": false',
                    b'"renderable": NaN',
                    1,
                ),
            ),
            (
                "non-finite JSON number",
                self.original_manifest_bytes.replace(
                    b'"renderable": false',
                    b'"renderable": Infinity',
                    1,
                ),
            ),
        )
        for expected, payload in cases:
            with self.subTest(expected=expected):
                self.manifest_path.write_bytes(payload)
                self.assert_contract_error(expected)
                self.manifest_path.write_bytes(self.original_manifest_bytes)

    def test_schema_is_fail_closed(self) -> None:
        mutations = (
            ("unknown property", lambda data: data.__setitem__("unexpected", 1)),
            ("missing required property", lambda data: data.pop("id")),
            (
                "expected type integer",
                lambda data: data["projection"].__setitem__("widthPixels", True),
            ),
            (
                "expected constant",
                lambda data: data.__setitem__(
                    "schema", "blackhole.nr-transfer-map/v2"
                ),
            ),
        )
        for expected, mutation in mutations:
            with self.subTest(expected=expected):
                data = copy.deepcopy(self.original)
                mutation(data)
                self.write_manifest(data)
                self.assert_contract_error(expected)

    def test_schema_dialect_rejects_unknown_validation_keywords(self) -> None:
        schema = json.loads(DEFAULT_SCHEMA.read_bytes())
        schema["properties"]["id"]["unsupportedValidationKeyword"] = True
        schema_path = Path(self.temporary.name) / "schema.json"
        schema_path.write_text(
            json.dumps(schema, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ContractError, "unsupported JSON Schema keyword"):
            validate_contract(self.manifest_path, schema_path)

    def test_schema_provenance_matches_content_not_absolute_path(self) -> None:
        schema_copy = Path(self.temporary.name) / "schema-copy.json"
        shutil.copyfile(DEFAULT_SCHEMA, schema_copy)
        report = validate_contract(self.manifest_path, schema_copy)
        self.assertEqual(report["status"], "protocol-conformant")

    def test_portable_manifest_directory_and_external_artifacts(self) -> None:
        data = copy.deepcopy(self.original)
        portable = self.dataset / "provenance"
        portable.mkdir()
        generator_copy = portable / "generator.py"
        schema_copy = portable / "schema.json"
        shutil.copyfile(ROOT / "scripts" / "generate_nr_contract_fixture.py", generator_copy)
        shutil.copyfile(DEFAULT_SCHEMA, schema_copy)
        data["provenance"]["artifactUriBase"] = "manifest-directory"
        data["provenance"]["sourceArtifacts"][0]["uri"] = "provenance/generator.py"
        data["provenance"]["sourceArtifacts"][1]["uri"] = "provenance/schema.json"
        data["provenance"]["generator"]["uri"] = "provenance/generator.py"
        self.write_manifest(data)
        report = validate_contract(self.manifest_path, schema_copy)
        self.assertEqual(report["status"], "protocol-conformant")

        data = copy.deepcopy(self.original)
        data["provenance"]["sourceArtifacts"][0].update(
            {
                "storage": "external-reference",
                "uri": "https://example.invalid/generator.py",
            }
        )
        data["provenance"]["sourceArtifacts"][1].update(
            {
                "storage": "external-reference",
                "uri": "doi:10.1234/schema-fixture",
            }
        )
        data["provenance"]["generator"]["uri"] = (
            "https://example.invalid/generator.py"
        )
        self.write_manifest(data)
        report = validate_contract(self.manifest_path, DEFAULT_SCHEMA)
        self.assertEqual(report["status"], "protocol-conformant")

        data["provenance"]["sourceArtifacts"][0]["uri"] = (
            "http://example.invalid/generator.py"
        )
        data["provenance"]["generator"]["uri"] = (
            "http://example.invalid/generator.py"
        )
        self.write_manifest(data)
        self.assert_contract_error("HTTPS URL or doi")

    def test_provenance_rejects_traversal_and_hash_drift(self) -> None:
        data = copy.deepcopy(self.original)
        data["provenance"]["sourceArtifacts"][0]["uri"] = "../escape.py"
        self.write_manifest(data)
        self.assert_contract_error("traversal-free")

        data = copy.deepcopy(self.original)
        data["provenance"]["sourceArtifacts"][0]["sha256"] = "0" * 64
        self.write_manifest(data)
        self.assert_contract_error("hash mismatch")

    def test_generator_uri_must_bind_generator_source_role(self) -> None:
        data = copy.deepcopy(self.original)
        data["provenance"]["generator"]["uri"] = data["provenance"][
            "sourceArtifacts"
        ][1]["uri"]
        self.write_manifest(data)
        self.assert_contract_error("generator-source artifact")

        data = copy.deepcopy(self.original)
        data["provenance"]["generator"]["codeRevision"] = "sha256:" + "0" * 64
        self.write_manifest(data)
        self.assert_contract_error("generator-source SHA-256")

    def test_chunk_rejects_traversal_and_symlink_escape(self) -> None:
        data = copy.deepcopy(self.original)
        data["chunks"][0]["uri"] = "../outside.bin"
        self.write_manifest(data)
        self.assert_contract_error("pattern")

        self.write_manifest(copy.deepcopy(self.original))
        outside = Path(self.temporary.name) / "outside.bin"
        outside.write_bytes(self.original_chunk)
        self.chunk_path.unlink()
        self.chunk_path.symlink_to(outside)
        self.assert_contract_error("symlinked artifacts")

    def test_size_hash_and_sidecar_are_enforced(self) -> None:
        data = copy.deepcopy(self.original)
        data["chunks"][0]["byteLength"] = 288
        self.write_manifest(data)
        self.assert_contract_error("recordCount \\* recordBytes")

        self.chunk_path.write_bytes(self.original_chunk[:-1] + b"\xff")
        self.write_manifest(copy.deepcopy(self.original))
        self.assert_contract_error("hash mismatch")

        self.chunk_path.write_bytes(self.original_chunk)
        self.write_manifest(copy.deepcopy(self.original))
        digest = hashlib.sha256(self.original_manifest_bytes).hexdigest()
        (self.dataset / "manifest.sha256").write_text(
            f"{digest} manifest.json\n",
            encoding="ascii",
        )
        self.assert_contract_error("sidecar must exactly match")

    def test_chunk_gap_overlap_and_order_are_rejected(self) -> None:
        data = copy.deepcopy(self.original)
        data["chunks"][0]["tile"]["width"] = 3
        data["chunks"][0]["recordCount"] = 6
        self.write_chunk(data, self.original_chunk[: 6 * RECORD_STRUCT.size])
        self.write_manifest(data)
        self.assert_contract_error("tile gap")

        data = copy.deepcopy(self.original)
        first = copy.deepcopy(data["chunks"][0])
        first["tile"]["width"] = 3
        first["recordCount"] = 6
        first["uri"] = "chunks/overlap-a.bin"
        second = copy.deepcopy(data["chunks"][0])
        second["tile"]["x"] = 2
        second["tile"]["width"] = 2
        second["recordCount"] = 4
        second["uri"] = "chunks/overlap-b.bin"
        data["chunks"] = [first, second]
        self.write_chunk(data, self.original_chunk[: 6 * RECORD_STRUCT.size], 0)
        self.write_chunk(data, self.original_chunk[: 4 * RECORD_STRUCT.size], 1)
        self.write_manifest(data)
        self.assert_contract_error("overlap")

        data = copy.deepcopy(self.original)
        left = copy.deepcopy(data["chunks"][0])
        left["tile"]["width"] = 2
        left["recordCount"] = 4
        left["uri"] = "chunks/left.bin"
        right = copy.deepcopy(left)
        right["tile"]["x"] = 2
        right["uri"] = "chunks/right.bin"
        data["chunks"] = [right, left]
        self.write_chunk(data, self.original_chunk[: 4 * RECORD_STRUCT.size], 0)
        self.write_chunk(data, self.original_chunk[: 4 * RECORD_STRUCT.size], 1)
        self.write_manifest(data)
        self.assert_contract_error("canonical")

    def test_observation_times_are_strictly_monotonic(self) -> None:
        data = copy.deepcopy(self.original)
        data["sampling"]["observationTimesM"] = [0.0, -1.0]
        self.write_manifest(data)
        self.assert_contract_error("strictly increasing")

    def test_observer_time_mapping_proper_time_and_fixed_camera(self) -> None:
        data = copy.deepcopy(self.original)
        data["observer"]["samples"][0]["eventNr"][0] = 1.0
        self.write_manifest(data)
        self.assert_contract_error("does not map to protocolTimeM")

        data = copy.deepcopy(self.original)
        second = copy.deepcopy(data["observer"]["samples"][0])
        second.update(
            {
                "sampleIndex": 1,
                "protocolTimeM": 1.0,
                "properTimeM": 0.0,
                "eventNr": [1.0, 4.0, 3.0, 12.0],
            }
        )
        data["sampling"]["observationTimesM"] = [0.0, 1.0]
        data["observer"]["samples"].append(second)
        self.write_manifest(data)
        self.assert_contract_error("properTimeM must be strictly increasing")

        second["properTimeM"] = 1.0
        second["eventNr"][1] = 5.0
        self.write_manifest(data)
        self.assert_contract_error("fixed camera is not anchored")

    def test_matrix_inverse_and_round_trip_are_enforced(self) -> None:
        data = copy.deepcopy(self.original)
        data["camera"]["worldToCamera"][3] = -5.0
        self.write_manifest(data)
        self.assert_contract_error("not mutual inverses")

        data = copy.deepcopy(self.original)
        data["camera"]["worldToCamera"][3] = -4.0 + 5.0e-11
        self.write_manifest(data)
        self.assert_contract_error("point round-trip error")

    def test_affine_spatial_block_must_be_a_proper_rotation(self) -> None:
        data = copy.deepcopy(self.original)
        data["camera"]["cameraToWorld"] = [
            -1.0, 0.0, 0.0, 4.0,
            0.0, 1.0, 0.0, 3.0,
            0.0, 0.0, 1.0, 12.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        data["camera"]["worldToCamera"] = [
            -1.0, 0.0, 0.0, 4.0,
            0.0, 1.0, 0.0, -3.0,
            0.0, 0.0, 1.0, -12.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        self.write_manifest(data)
        self.assert_contract_error("proper rotation")

        data = copy.deepcopy(self.original)
        reflection = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        data["coordinates"]["sky"]["worldToIcrs"] = reflection
        data["coordinates"]["sky"]["icrsToWorld"] = reflection
        self.write_manifest(data)
        self.assert_contract_error("proper rotation")

    def test_tetrad_orthonormality_and_e0_are_enforced(self) -> None:
        data = copy.deepcopy(self.original)
        data["observer"]["samples"][0]["tetradContravariantNr"][1][1] = 2.0
        self.write_manifest(data)
        self.assert_contract_error("not orthonormal")

        data = copy.deepcopy(self.original)
        data["observer"]["samples"][0]["fourVelocityContravariantNr"][1] = 0.01
        self.write_manifest(data)
        self.assert_contract_error("time leg must equal")

        data = copy.deepcopy(self.original)
        data["observer"]["samples"][0]["metricContravariantNr"][5] = 2.0
        self.write_manifest(data)
        self.assert_contract_error("inverse metric mismatch")

    def test_capture_metadata_and_sample_validity_are_enforced(self) -> None:
        data = copy.deepcopy(self.original)
        data["captureTargets"][1]["classificationPriority"] = 0
        self.write_manifest(data)
        self.assert_contract_error("duplicate classification priority")

        data = copy.deepcopy(self.original)
        data["captureTargets"][0]["sourceArtifactRole"] = "horizon-data"
        self.write_manifest(data)
        self.assert_contract_error("is not present in provenance")

        data = copy.deepcopy(self.original)
        data["captureTargets"][0]["validityIntervalProtocolM"] = [1.0, 2.0]
        self.write_manifest(data)
        self.assert_contract_error("is not valid at observation time")

    def test_dataset_claim_gates_reject_pseudo_nr_and_bad_binary_physics(self) -> None:
        data = self.make_pseudo_nr()
        self.write_manifest(data)
        self.assert_contract_error("requires near-zone-metric and horizon-data")

        data = self.make_pseudo_nr()
        data["physicalSystem"]["dimensionlessSpins"][1]["componentId"] = "A"
        self.write_manifest(data)
        self.assert_contract_error("cover each physical component exactly once")

        data = self.make_pseudo_nr()
        data["physicalSystem"]["dimensionlessSpins"][0]["vector"] = [
            1.0,
            1.0,
            0.0,
        ]
        self.write_manifest(data)
        self.assert_contract_error("spin magnitude exceeds one")

        data = self.make_pseudo_nr()
        data["provenance"]["sourceSimulation"]["evolutionCode"]["commit"] = None
        self.write_manifest(data)
        self.assert_contract_error("missing evolution-code commit needs")

    def test_renderable_outcome_fractions_and_resolved_ray_gate(self) -> None:
        data = self.make_stationary_renderable()
        self.write_manifest(data)
        report = validate_contract(self.manifest_path, DEFAULT_SCHEMA)
        self.assertEqual(report["status"], "protocol-conformant")

        data = self.make_stationary_renderable()
        data["accuracy"]["outcomeFractions"]["escaped"] = 0.0
        self.write_manifest(data)
        self.assert_contract_error("declares 0.0, decoded 0.25")

        data = self.make_stationary_renderable()
        missing = RECORD_STRUCT.pack(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            data["recordLayout"]["rayOutcomes"]["missing"],
            data["recordLayout"]["captureTargetNone"],
            0,
        )
        payload = missing * 8
        self.write_chunk(data, payload)
        data["accuracy"]["unresolvedFraction"] = 0.0
        data["accuracy"]["outcomeFractions"] = {
            "escaped": 0.0,
            "captured": 0.0,
            "unresolved": 0.0,
            "outside-domain": 0.0,
            "integrator-failure": 0.0,
            "missing": 1.0,
            "unusable": 1.0,
        }
        self.write_manifest(data)
        self.assert_contract_error("no resolved escaped or captured rays")

    def test_binary_record_state_machine_is_fail_closed(self) -> None:
        cases = (
            (0, 7, 5, "unknown outcome code"),
            (0, 9, 0, "requires mask"),
            (0, 9, 63, "unknown validity bits"),
            (0, 8, 0, "no-target sentinel"),
            (2, 8, 2, "unknown target"),
            (2, 0, 0.25, "canonical positive float32 zero"),
            (2, 0, -0.0, "canonical positive float32 zero"),
            (0, 3, math.nan, "NaN or Infinity"),
            (0, 0, 0.25, "not unit length"),
            (0, 3, 0.0, "strictly positive"),
            (0, 4, -1.0, "lookback time must be non-negative"),
        )
        for record, value_index, value, expected in cases:
            with self.subTest(record=record, value_index=value_index, expected=expected):
                self.chunk_path.write_bytes(self.original_chunk)
                self.mutate_record(record, value_index, value)
                self.assert_contract_error(expected)


if __name__ == "__main__":
    unittest.main()
