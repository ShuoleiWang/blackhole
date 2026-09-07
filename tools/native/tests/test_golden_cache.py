from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from tools.native import golden_cache as gc


def _transport(coordinate: dict[str, object], ordinal: int) -> dict[str, object]:
    fates = (
        "escaped",
        "return-upper",
        "plunge-sink",
        "captured",
        "return-lower",
    )
    fate = fates[ordinal % len(fates)]
    returning = fate.startswith("return-")
    receiver_face = fate.removeprefix("return-") if returning else None
    return {
        "formulation": "forward",
        "sample": {
            "emissionAngleCosine": 0.125 + ordinal * 0.001,
            "muIndex": coordinate["muIndex"],
            "normalizedEmittedFluxWeight": 0.25,
            "passIndex": coordinate["passIndex"],
            "passName": coordinate["passName"],
            "psiIndex": coordinate["psiIndex"],
            "rhoIndex": coordinate["rhoIndex"],
            "sourceAnnulusIndex": coordinate["annulusIndex"],
            "sourceFace": coordinate["face"],
            "sourceRadiusOverMass": 10.0 + ordinal,
            "tangentAzimuthRad": 0.5 + ordinal * 0.01,
        },
        "schema": gc.FORWARD_TRANSPORT_SCHEMA,
        "transport": {
            "coarseReceiverAnnulusIndex": 0 if returning else None,
            "coarseReceiverFace": receiver_face,
            "coarseReceiverRadiusOverMass": 12.0 + ordinal if returning else None,
            "fate": fate,
            "frequencyRatio": 1.1 if returning else None,
            "g2": 1.1 * 1.1 if returning else 0.0,
            "primitiveDescriptorSha256": hashlib.sha256(
                f"primitive-{ordinal}".encode("ascii")
            ).hexdigest(),
            "receiverAnnulusIndex": 0 if returning else None,
            "receiverFace": receiver_face,
            "receiverRadiusOverMass": 12.25 + ordinal if returning else None,
        },
    }


def _make_fixture(parent: Path) -> tuple[Path, gc.CacheTrust]:
    passes = [
        {
            "muOrder": 1,
            "name": name,
            "phaseCells": 0.5 if name == "phase-shifted" else 0.0,
            "psiCount": 4 if name == "full" else 2,
            "rhoOrder": 1,
        }
        for name in gc._PASS_NAMES
    ]
    direction_count = 2 * sum(
        item["rhoOrder"] * item["muOrder"] * item["psiCount"]
        for item in passes
    )
    plan = {
        "annulusCount": 1,
        "canonicalOrder": "pass/face/annulus/rho/mu/psi",
        "directionCount": direction_count,
        "faces": ["upper", "lower"],
        "formulation": "forward",
        "passes": passes,
        "schema": gc.TASK_PLAN_SCHEMA,
    }
    scientific_document = {
        "algorithmVersion": "1.1.0",
        "inputs": [],
        "plan": plan,
        "producer": "test",
        "producerSourceHashes": [],
        "schema": "test-scientific/v1",
        "scientificIdentity": {"annulusEdgesOverMass": [3.0, 30.0]},
        "scientificStatus": {},
    }
    scientific_job_key = hashlib.sha256(
        gc.canonical_json_bytes(scientific_document)
    ).hexdigest()
    plan_sha256 = hashlib.sha256(gc.canonical_json_bytes(plan)).hexdigest()
    directions_per_task = 2
    expected_tasks = gc._expected_tasks(plan, passes, directions_per_task)
    parameters = {
        "cacheLayout": {
            "chunkBoundary": "never crosses pass/face/annulus/rho",
            "directionsPerTaskMaximum": directions_per_task,
            "taskCount": len(expected_tasks),
        },
        "payloadBudget": {},
        "scientificDocument": scientific_document,
        "scientificJobKey": scientific_job_key,
        "scientificPlanSha256": plan_sha256,
        "scientificStatus": {},
    }
    spec = {
        "algorithmVersion": "1.1.0",
        "inputs": [],
        "parameters": parameters,
        "producer": "offline.kerr-returning-radiation-direction-cache",
        "producerSourceHashes": [],
        "recordBytes": 1,
        "schema": "blackhole.offline-job/v1",
        "tasks": list(expected_tasks),
    }
    cache_job_key = hashlib.sha256(gc.canonical_json_bytes(spec)).hexdigest()
    job_document = {"jobKey": cache_job_key, "spec": spec}
    job_payload = gc.canonical_json_bytes(job_document)
    trust = gc.CacheTrust(
        cache_job_key,
        scientific_job_key,
        plan_sha256,
        hashlib.sha256(job_payload).hexdigest(),
    )
    job_directory = parent / cache_job_key
    task_directory = job_directory / "tasks"
    task_directory.mkdir(parents=True)
    (job_directory / "job.json").write_bytes(job_payload)
    definition = gc._validated_job_document(job_payload, trust)

    # Freeze only the first six task pairs.  The fixture is deliberately
    # incomplete and therefore exercises the read-only evidence path.
    for task in expected_tasks[:6]:
        stem = gc._task_stem(task)
        records = [
            {"coordinate": coordinate, "transport": _transport(coordinate, coordinate["ordinal"])}
            for coordinate in gc._expected_coordinates(definition, task)
        ]
        payload_document = {
            "cacheJobKey": cache_job_key,
            "records": records,
            "schema": gc.TASK_PAYLOAD_SCHEMA,
            "scientificJobKey": scientific_job_key,
            "scientificPlanSha256": plan_sha256,
            "task": task,
        }
        payload = gc.canonical_json_bytes(payload_document)
        payload_name = f"{stem}.bin"
        receipt_name = f"{stem}.receipt.json"
        receipt = gc.canonical_json_bytes(
            {
                "byteLength": len(payload),
                "jobKey": cache_job_key,
                "payload": payload_name,
                "recordCount": len(payload),
                "schema": gc.RECEIPT_SCHEMA,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "task": task,
            }
        )
        (task_directory / payload_name).write_bytes(payload)
        (task_directory / receipt_name).write_bytes(receipt)
        (task_directory / f"{stem}.lock").write_bytes(b"")
    return job_directory, trust


def _refresh_receipt(task_directory: Path, payload_path: Path) -> None:
    receipt_path = payload_path.with_name(
        f"{payload_path.name.removesuffix('.bin')}.receipt.json"
    )
    receipt = json.loads(receipt_path.read_bytes())
    payload = payload_path.read_bytes()
    receipt["byteLength"] = len(payload)
    receipt["recordCount"] = len(payload)
    receipt["sha256"] = hashlib.sha256(payload).hexdigest()
    receipt_path.write_bytes(gc.canonical_json_bytes(receipt))


class GoldenCacheExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        # macOS exposes /var as a symlink.  The reader intentionally rejects
        # symlinked ancestors, so fixtures live below the canonical /private
        # path just like the frozen cache.
        self.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temporary.name)
        self.job_directory, self.trust = _make_fixture(self.root)
        self.tasks = self.job_directory / "tasks"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def extract(self) -> dict[str, object]:
        return gc.extract_golden_corpus(
            self.job_directory,
            selected_ordinals=(0, 3, 8, 11),
            trust=self.trust,
        )

    def test_extracts_authenticated_partial_prefix_in_canonical_order(self) -> None:
        corpus = self.extract()
        evidence = corpus["evidence"]
        self.assertEqual(evidence["authenticatedTaskCount"], 6)
        self.assertEqual(evidence["authenticatedDirectionCount"], 12)
        self.assertEqual(evidence["continuousOrdinalPrefixCount"], 12)
        self.assertTrue(evidence["isContinuousOrdinalPrefix"])
        self.assertFalse(evidence["isCompleteCache"])
        self.assertEqual(
            [item["coordinate"]["ordinal"] for item in corpus["records"]],
            [0, 3, 8, 11],
        )
        self.assertEqual(sum(evidence["fateDistribution"].values()), 12)
        self.assertEqual(
            hashlib.sha256(gc.canonical_json_bytes(corpus["records"])).hexdigest(),
            evidence["selectedRecordSetSha256"],
        )

    def test_receipt_sha_tamper_fails_closed(self) -> None:
        receipt_path = sorted(self.tasks.glob("*.receipt.json"))[0]
        receipt = json.loads(receipt_path.read_bytes())
        receipt["sha256"] = "0" * 64
        receipt_path.write_bytes(gc.canonical_json_bytes(receipt))
        with self.assertRaisesRegex(gc.GoldenCacheError, "receipt does not authenticate"):
            self.extract()

    def test_receipt_bool_task_field_fails_exact_type_gate(self) -> None:
        receipt_path = sorted(self.tasks.glob("*.receipt.json"))[0]
        receipt = json.loads(receipt_path.read_bytes())
        receipt["task"]["height"] = True
        receipt_path.write_bytes(gc.canonical_json_bytes(receipt))
        with self.assertRaisesRegex(gc.GoldenCacheError, "exact int"):
            self.extract()

    def test_payload_tamper_without_receipt_update_fails_closed(self) -> None:
        payload_path = sorted(self.tasks.glob("*.bin"))[0]
        document = json.loads(payload_path.read_bytes())
        document["records"][0]["transport"]["sample"]["sourceRadiusOverMass"] = 99.0
        payload_path.write_bytes(gc.canonical_json_bytes(document))
        with self.assertRaisesRegex(gc.GoldenCacheError, "receipt does not authenticate"):
            self.extract()

    def test_payload_schema_tamper_with_fresh_receipt_fails_closed(self) -> None:
        payload_path = sorted(self.tasks.glob("*.bin"))[0]
        document = json.loads(payload_path.read_bytes())
        document["schema"] = "blackhole.returning-radiation-kernel-direction-task/v0"
        payload_path.write_bytes(gc.canonical_json_bytes(document))
        _refresh_receipt(self.tasks, payload_path)
        with self.assertRaisesRegex(gc.GoldenCacheError, "stale scientific identity"):
            self.extract()

    def test_paired_numeric_tamper_fails_frozen_snapshot_anchor(self) -> None:
        baseline = self.extract()
        anchored = gc.CacheTrust(
            self.trust.cache_job_key,
            self.trust.scientific_job_key,
            self.trust.scientific_plan_sha256,
            self.trust.job_document_sha256,
            authenticated_task_count=6,
            authenticated_direction_count=12,
            authenticated_payload_set_sha256=baseline["evidence"][
                "authenticatedPayloadSetSha256"
            ],
            authenticated_direction_stream_sha256=baseline["evidence"][
                "authenticatedDirectionStreamSha256"
            ],
        )
        payload_path = sorted(self.tasks.glob("*.bin"))[0]
        document = json.loads(payload_path.read_bytes())
        document["records"][0]["transport"]["sample"]["sourceRadiusOverMass"] = 99.0
        payload_path.write_bytes(gc.canonical_json_bytes(document))
        _refresh_receipt(self.tasks, payload_path)
        with self.assertRaisesRegex(gc.GoldenCacheError, "trust anchor"):
            gc.extract_golden_corpus(
                self.job_directory,
                selected_ordinals=(0, 3, 8, 11),
                trust=anchored,
            )

    def test_record_reordering_with_fresh_receipt_fails_closed(self) -> None:
        payload_path = sorted(self.tasks.glob("*.bin"))[0]
        document = json.loads(payload_path.read_bytes())
        document["records"][0], document["records"][1] = (
            document["records"][1],
            document["records"][0],
        )
        payload_path.write_bytes(gc.canonical_json_bytes(document))
        _refresh_receipt(self.tasks, payload_path)
        with self.assertRaisesRegex(gc.GoldenCacheError, "out of canonical order"):
            self.extract()

    def test_g2_invariant_tamper_with_fresh_receipt_fails_closed(self) -> None:
        payload_path = sorted(self.tasks.glob("*.bin"))[0]
        document = json.loads(payload_path.read_bytes())
        returning = document["records"][1]["transport"]["transport"]
        self.assertEqual(returning["fate"], "return-upper")
        returning["g2"] *= 1.01
        payload_path.write_bytes(gc.canonical_json_bytes(document))
        _refresh_receipt(self.tasks, payload_path)
        with self.assertRaisesRegex(gc.GoldenCacheError, "squared frequency ratio"):
            self.extract()

    def test_receiver_bin_tamper_with_fresh_receipt_fails_closed(self) -> None:
        payload_path = sorted(self.tasks.glob("*.bin"))[0]
        document = json.loads(payload_path.read_bytes())
        returning = document["records"][1]["transport"]["transport"]
        returning["receiverAnnulusIndex"] = 1
        returning["coarseReceiverAnnulusIndex"] = 1
        payload_path.write_bytes(gc.canonical_json_bytes(document))
        _refresh_receipt(self.tasks, payload_path)
        with self.assertRaisesRegex(gc.GoldenCacheError, "exact annulus edges"):
            self.extract()

    def test_noncanonical_receipt_fails_closed(self) -> None:
        receipt_path = sorted(self.tasks.glob("*.receipt.json"))[0]
        receipt = json.loads(receipt_path.read_bytes())
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        with self.assertRaisesRegex(gc.GoldenCacheError, "not canonical JSON"):
            self.extract()

    def test_one_sided_pair_fails_closed(self) -> None:
        sorted(self.tasks.glob("*.receipt.json"))[0].unlink()
        with self.assertRaisesRegex(gc.GoldenCacheError, "one-sided"):
            self.extract()

    def test_unknown_task_entry_fails_closed(self) -> None:
        (self.tasks / "surprise.bin").write_bytes(b"")
        with self.assertRaisesRegex(gc.GoldenCacheError, "unexpected entry"):
            self.extract()

    def test_payload_symlink_fails_closed(self) -> None:
        payload_path = sorted(self.tasks.glob("*.bin"))[0]
        copied = self.root / "outside.bin"
        shutil.copyfile(payload_path, copied)
        payload_path.unlink()
        payload_path.symlink_to(copied)
        with self.assertRaisesRegex(gc.GoldenCacheError, "regular non-symlink"):
            self.extract()

    def test_job_document_tamper_fails_external_hash_anchor(self) -> None:
        job_path = self.job_directory / "job.json"
        document = json.loads(job_path.read_bytes())
        document["spec"]["algorithmVersion"] = "1.1.1"
        job_path.write_bytes(gc.canonical_json_bytes(document))
        with self.assertRaisesRegex(gc.GoldenCacheError, "frozen SHA-256"):
            self.extract()


class GoldenComparatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temporary.name)
        job_directory, trust = _make_fixture(self.root)
        self.golden = gc.extract_golden_corpus(
            job_directory,
            selected_ordinals=(0, 3, 8, 11),
            trust=trust,
        )
        self.candidate = gc.candidate_document_from_corpus(copy.deepcopy(self.golden))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_candidate_is_byte_exact_and_qualified(self) -> None:
        report = gc.compare_corpus_documents(
            self.golden,
            self.candidate,
            require_byte_exact=True,
        )
        self.assertTrue(report["qualified"])
        self.assertTrue(report["byteExact"])
        self.assertEqual(report["categoricalDriftCount"], 0)
        self.assertEqual(report["numericDriftCount"], 0)
        self.assertFalse(report["scope"]["provesFullAuthenticatedCorpusParity"])
        self.assertFalse(report["scope"]["provesNativeExecutionProvenance"])

    def test_categorical_fate_drift_is_never_tolerated(self) -> None:
        self.candidate["records"][0]["transport"]["transport"]["fate"] = "captured"
        report = gc.compare_corpus_documents(
            self.golden,
            self.candidate,
            absolute_tolerance=1.0,
            relative_tolerance=1.0,
            maximum_ulp=2**63,
        )
        self.assertFalse(report["qualified"])
        self.assertGreater(report["categoricalDriftCount"], 0)

    def test_numeric_drift_requires_explicit_threshold(self) -> None:
        sample = self.candidate["records"][0]["transport"]["sample"]
        sample["sourceRadiusOverMass"] += 1.0e-12
        strict = gc.compare_corpus_documents(self.golden, self.candidate)
        tolerant = gc.compare_corpus_documents(
            self.golden,
            self.candidate,
            absolute_tolerance=2.0e-12,
        )
        self.assertFalse(strict["qualified"])
        self.assertEqual(strict["numericLimitViolationCount"], 1)
        self.assertTrue(tolerant["qualified"])
        self.assertFalse(tolerant["byteExact"])

    def test_signed_zero_g2_drift_fails_semantic_gate(self) -> None:
        non_return = self.candidate["records"][0]["transport"]["transport"]
        self.assertEqual(non_return["g2"].hex(), 0.0.hex())
        non_return["g2"] = -0.0
        with self.assertRaisesRegex(gc.GoldenCacheError, r"must be \+0.0"):
            gc.compare_corpus_documents(self.golden, self.candidate)

    def test_tolerated_g2_drift_cannot_break_frequency_square_invariant(self) -> None:
        returning = next(
            record["transport"]["transport"]
            for record in self.candidate["records"]
            if record["transport"]["transport"]["fate"].startswith("return-")
        )
        returning["g2"] += 0.01
        with self.assertRaisesRegex(gc.GoldenCacheError, "squared frequency ratio"):
            gc.compare_corpus_documents(
                self.golden,
                self.candidate,
                absolute_tolerance=0.02,
            )

    def test_tolerated_radius_drift_cannot_escape_retained_annulus_bin(self) -> None:
        returning = next(
            record["transport"]["transport"]
            for record in self.candidate["records"]
            if record["transport"]["transport"]["fate"].startswith("return-")
        )
        returning["receiverRadiusOverMass"] = 31.0
        with self.assertRaisesRegex(gc.GoldenCacheError, "outside annulus edges"):
            gc.compare_corpus_documents(
                self.golden,
                self.candidate,
                absolute_tolerance=100.0,
            )

    def test_candidate_record_reordering_fails_closed(self) -> None:
        records = self.candidate["records"]
        records[0], records[1] = records[1], records[0]
        with self.assertRaisesRegex(gc.GoldenCacheError, "reordered"):
            gc.compare_corpus_documents(self.golden, self.candidate)

    def test_expanded_leaf_is_optional_and_ignored_by_narrow_comparison(self) -> None:
        self.candidate["expanded"] = {"backend": "strict-fp64-test"}
        report = gc.compare_corpus_documents(self.golden, self.candidate)
        self.assertTrue(report["qualified"])

    def test_external_corpus_sha_rejects_self_rehashed_golden_tamper(self) -> None:
        golden_path = self.root / "golden.json"
        candidate_path = self.root / "candidate.json"
        original = gc.canonical_json_bytes(self.golden)
        expected_sha256 = hashlib.sha256(original).hexdigest()
        golden_path.write_bytes(original)
        candidate_path.write_bytes(gc.canonical_json_bytes(self.candidate))
        tampered = copy.deepcopy(self.golden)
        tampered["records"][0]["transport"]["sample"]["sourceRadiusOverMass"] = 99.0
        tampered["evidence"]["selectedRecordSetSha256"] = hashlib.sha256(
            gc.canonical_json_bytes(tampered["records"])
        ).hexdigest()
        golden_path.write_bytes(gc.canonical_json_bytes(tampered))
        with self.assertRaisesRegex(gc.GoldenCacheError, "external SHA-256"):
            gc.compare_corpus_files(
                golden_path,
                candidate_path,
                expected_golden_sha256=expected_sha256,
                require_byte_exact=True,
            )

    def test_candidate_expanded_leaf_must_be_an_object(self) -> None:
        candidate_path = self.root / "candidate.json"
        candidate = copy.deepcopy(self.candidate)
        candidate["expanded"] = []
        candidate_path.write_bytes(gc.canonical_json_bytes(candidate))
        with self.assertRaisesRegex(gc.GoldenCacheError, "expanded leaf"):
            gc.load_candidate_document(candidate_path)

    def test_golden_evidence_bool_count_fails_exact_type_gate(self) -> None:
        golden_path = self.root / "golden.json"
        malformed = copy.deepcopy(self.golden)
        malformed["evidence"]["authenticatedTaskCount"] = True
        payload = gc.canonical_json_bytes(malformed)
        golden_path.write_bytes(payload)
        with self.assertRaisesRegex(gc.GoldenCacheError, "exact int"):
            gc.load_golden_corpus(
                golden_path,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
            )

    def test_output_parent_symlink_is_rejected(self) -> None:
        real_parent = self.root / "real-output"
        real_parent.mkdir()
        linked_parent = self.root / "linked-output"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaisesRegex(gc.GoldenCacheError, "symlink"):
            gc._write_canonical_output(linked_parent / "report.json", b"{}\n")
        self.assertFalse((real_parent / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
