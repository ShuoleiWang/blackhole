from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from tools.native import phase_space_golden as pg


class PhaseSpaceGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.golden_path = (
            Path(pg.__file__).absolute().parent / pg.REFERENCE_FILENAME
        )
        cls.golden_bytes = cls.golden_path.read_bytes()
        cls.golden = pg.load_reference_document(
            cls.golden_path,
            expected_sha256=pg.FROZEN_REFERENCE_SHA256,
        )

    def candidate(self) -> dict[str, object]:
        return pg.candidate_document_from_reference(
            self.golden,
            golden_sha256=pg.FROZEN_REFERENCE_SHA256,
            backend={
                "artifacts": [],
                "implementationId": "unit-test-reference-copy",
            },
        )

    def test_checked_in_reference_is_externally_and_internally_anchored(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.golden_bytes).hexdigest(),
            pg.FROZEN_REFERENCE_SHA256,
        )
        self.assertEqual(
            self.golden["selectedOrdinals"],
            list(pg.REFERENCE_ORDINALS),
        )
        self.assertEqual(len(self.golden["manifest"]["recordSha256"]), 8)
        for record, ordinal in zip(
            self.golden["records"], pg.REFERENCE_ORDINALS
        ):
            self.assertEqual(record["coordinate"]["ordinal"], ordinal)
            self.assertGreater(len(record["phaseSpace"]["fine"]["segments"]), 0)
            self.assertGreater(len(record["phaseSpace"]["coarse"]["segments"]), 0)

    def test_exact_reference_candidate_is_byte_exact_and_qualified(self) -> None:
        report = pg.compare_documents(
            self.golden,
            self.candidate(),
            golden_sha256=pg.FROZEN_REFERENCE_SHA256,
            require_byte_exact=True,
        )
        self.assertTrue(report["qualified"])
        self.assertTrue(report["byteExact"])
        self.assertEqual(report["mismatchCount"], 0)
        self.assertGreater(report["diagnostics"]["comparedFloatCount"], 1_000)

    def test_numeric_drift_needs_an_explicit_ulp_budget(self) -> None:
        candidate = self.candidate()
        convergence = candidate["records"][0]["phaseSpace"]["convergence"]
        original = convergence["terminal_event_difference_m"]
        convergence["terminal_event_difference_m"] = math.nextafter(
            original, math.inf
        )
        strict = pg.compare_documents(
            self.golden,
            candidate,
            golden_sha256=pg.FROZEN_REFERENCE_SHA256,
        )
        self.assertFalse(strict["qualified"])
        tolerant = pg.compare_documents(
            self.golden,
            candidate,
            golden_sha256=pg.FROZEN_REFERENCE_SHA256,
            maximum_ulp=1,
        )
        self.assertTrue(tolerant["qualified"])
        self.assertFalse(tolerant["byteExact"])

    def test_signed_zero_is_never_hidden_by_numeric_tolerance(self) -> None:
        candidate = self.candidate()
        candidate["records"][0]["phaseSpace"]["coarse"][
            "maximumMetricInterpolationError"
        ] = -0.0
        report = pg.compare_documents(
            self.golden,
            candidate,
            golden_sha256=pg.FROZEN_REFERENCE_SHA256,
            absolute_tolerance=1.0,
            relative_tolerance=1.0,
            maximum_ulp=1_000_000,
        )
        self.assertFalse(report["qualified"])
        self.assertEqual(report["mismatches"][0]["reason"], "signed-zero")

    def test_crossing_topology_and_classification_are_always_exact(self) -> None:
        candidate = self.candidate()
        returned = next(
            record
            for record in candidate["records"]
            if record["phaseSpace"]["fate"].startswith("return-")
        )
        crossing = returned["phaseSpace"]["fine"]["surfaceTrace"]["crossings"][0]
        crossing["decision"]["classification"] = "forged-transparent-classification"
        report = pg.compare_documents(
            self.golden,
            candidate,
            golden_sha256=pg.FROZEN_REFERENCE_SHA256,
            absolute_tolerance=1.0,
            relative_tolerance=1.0,
            maximum_ulp=1_000_000,
        )
        self.assertFalse(report["qualified"])
        self.assertTrue(
            any(item["reason"] == "categorical" for item in report["mismatches"])
        )

    def test_candidate_record_reordering_fails_closed_before_comparison(self) -> None:
        candidate = self.candidate()
        candidate["records"][0], candidate["records"][1] = (
            candidate["records"][1],
            candidate["records"][0],
        )
        with self.assertRaisesRegex(pg.PhaseSpaceGoldenError, "coordinate/order"):
            pg.compare_documents(
                self.golden,
                candidate,
                golden_sha256=pg.FROZEN_REFERENCE_SHA256,
            )

    def test_segment_state_tamper_fails_even_after_manifest_rehash(self) -> None:
        forged = deepcopy(self.golden)
        without_manifest = {key: value for key, value in forged.items() if key != "manifest"}
        second = without_manifest["records"][0]["phaseSpace"]["fine"]["segments"][1]
        second["start"]["event"][0] = math.nextafter(
            second["start"]["event"][0], math.inf
        )
        forged = pg._attach_manifest(without_manifest)
        with self.assertRaisesRegex(pg.PhaseSpaceGoldenError, "not contiguous"):
            pg.validate_reference_document(forged)

    def test_cached_primitive_sha_tamper_fails_after_manifest_rehash(self) -> None:
        forged = deepcopy(self.golden)
        without_manifest = {key: value for key, value in forged.items() if key != "manifest"}
        without_manifest["records"][0]["cachePrimitiveDescriptorSha256"] = "0" * 64
        forged = pg._attach_manifest(without_manifest)
        with self.assertRaisesRegex(pg.PhaseSpaceGoldenError, "cached primitive SHA"):
            pg.validate_reference_document(forged)

    def test_source_runtime_binding_tamper_fails_closed(self) -> None:
        forged = deepcopy(self.golden)
        forged["reference"]["sourceRuntimeClosure"][0]["bindingSha256"] = "0" * 64
        with self.assertRaisesRegex(pg.PhaseSpaceGoldenError, "binding is stale"):
            pg.validate_reference_document(forged)

    def test_manifest_hash_tamper_fails_closed(self) -> None:
        forged = deepcopy(self.golden)
        forged["manifest"]["recordsSha256"] = "0" * 64
        with self.assertRaisesRegex(pg.PhaseSpaceGoldenError, "manifest hashes"):
            pg.validate_reference_document(forged)

    def test_self_rehashed_tamper_still_fails_external_file_anchor(self) -> None:
        forged = deepcopy(self.golden)
        without_manifest = {key: value for key, value in forged.items() if key != "manifest"}
        sample = without_manifest["records"][0]["sample"]
        sample["normalizedEmittedFluxWeight"] = math.nextafter(
            sample["normalizedEmittedFluxWeight"], math.inf
        )
        forged = pg._attach_manifest(without_manifest)
        payload = pg._canonical_bytes(forged)
        # The attacker can refresh every self-contained manifest hash; the
        # frozen source-code anchor must still reject the changed file.
        pg.validate_reference_document(forged)
        with tempfile.TemporaryDirectory(dir="/private/tmp") as root:
            path = Path(root) / "forged.json"
            path.write_bytes(payload)
            with self.assertRaisesRegex(
                pg.PhaseSpaceGoldenError, "external SHA-256 anchor"
            ):
                pg.load_reference_document(
                    path,
                    expected_sha256=pg.FROZEN_REFERENCE_SHA256,
                )

    def test_candidate_cannot_rebind_to_a_different_golden(self) -> None:
        candidate = self.candidate()
        candidate["goldenSha256"] = "0" * 64
        with self.assertRaisesRegex(pg.PhaseSpaceGoldenError, "different golden"):
            pg.compare_documents(
                self.golden,
                candidate,
                golden_sha256=pg.FROZEN_REFERENCE_SHA256,
            )

    def test_generation_output_inside_golden_cache_is_rejected(self) -> None:
        cache = Path("/private/tmp/frozen-cache-job")
        with self.assertRaisesRegex(pg.PhaseSpaceGoldenError, "inside the golden cache"):
            pg._require_output_outside_cache(cache / "forged.json", cache)
        self.assertEqual(
            pg._require_output_outside_cache(
                Path("/private/tmp/separate-reference.json"), cache
            ),
            Path("/private/tmp/separate-reference.json"),
        )


if __name__ == "__main__":
    unittest.main()
