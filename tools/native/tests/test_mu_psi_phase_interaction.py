from __future__ import annotations

from copy import deepcopy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from offline.job import canonical_json_bytes
from offline.kerr_returning_radiation_convergence_v2 import (
    KerrReturningRadiationConvergenceV2Policy,
    KerrReturningRadiationGridSummaryV2,
)
from offline.kerr_returning_radiation_refinement_checkpoint import (
    NATIVE_CPU_EVALUATOR_MODE,
    NATIVE_MANIFEST_SCHEMA,
)
from tools.native import mu_psi_phase_interaction as analyzer


_PASS_NAMES = (
    "full",
    "half-rho",
    "half-mu",
    "half-psi",
    "phase-shifted",
)


def _summary(grid_id: str, returned_power: float) -> KerrReturningRadiationGridSummaryV2:
    return KerrReturningRadiationGridSummaryV2(
        grid_id,
        (1.0, 1.0),
        (1.0, 1.0),
        ((returned_power, 0.0), (0.0, returned_power)),
        (returned_power, returned_power),
        (
            (returned_power, 0.0, 1.0 - returned_power, 0.0, 0.0),
            (0.0, returned_power, 1.0 - returned_power, 0.0, 0.0),
        ),
        0.0004,
    )


def _passes(mu_order: int) -> list[dict[str, object]]:
    return [
        {
            "muOrder": mu_order,
            "name": "full",
            "phaseCells": 0.0,
            "psiCount": 64,
            "rhoOrder": 16,
        },
        {
            "muOrder": mu_order,
            "name": "half-rho",
            "phaseCells": 0.0,
            "psiCount": 64,
            "rhoOrder": 8,
        },
        {
            "muOrder": mu_order // 2,
            "name": "half-mu",
            "phaseCells": 0.0,
            "psiCount": 64,
            "rhoOrder": 16,
        },
        {
            "muOrder": mu_order,
            "name": "half-psi",
            "phaseCells": 0.0,
            "psiCount": 32,
            "rhoOrder": 16,
        },
        {
            "muOrder": mu_order,
            "name": "phase-shifted",
            "phaseCells": 0.5,
            "psiCount": 64,
            "rhoOrder": 16,
        },
    ]


def _evidence(
    pass_index: int,
    plan: dict[str, object],
    summary: KerrReturningRadiationGridSummaryV2,
    audit_sha256: str,
) -> dict[str, object]:
    mu_order = int(plan["muOrder"])
    psi_count = int(plan["psiCount"])
    rho_order = int(plan["rhoOrder"])
    name = str(plan["name"])
    return {
        "directionEvaluations": 2 * rho_order * mu_order * psi_count,
        "lowerEmitterG2ColumnClosureResiduals": [0.0],
        "maximumNormalizedSampleWeight": summary.maximum_normalized_sample_weight,
        "maximumNormalizedSampleWeightWitness": {
            "emissionAngleCosine": 0.5,
            "muIndex": 0,
            "normalizedEmittedFluxDirectionWeight": 0.01,
            "normalizedSampleWeight": summary.maximum_normalized_sample_weight,
            "passIndex": pass_index,
            "passName": name,
            "psiIndex": 0,
            "rhoAreaOverMassSquared": 0.04,
            "rhoIndex": 0,
            "sourceAnnulusIndex": 0,
            "sourceFace": "upper",
            "sourceRadiusOverMass": 7.0,
            "tangentAzimuthRad": 0.125,
        },
        "muOrder": mu_order,
        "passIndex": pass_index,
        "passName": name,
        "phaseCells": float(plan["phaseCells"]),
        "psiCount": psi_count,
        "rhoOrder": rho_order,
        "sampleAuditSha256": audit_sha256,
        "upperEmitterG2ColumnClosureResiduals": [0.0],
    }


def _library() -> dict[str, object]:
    return {
        "artifactName": "libblackhole_cpu.dylib",
        "byteLength": 1234,
        "logicalPath": "runtime/libblackhole_cpu.dylib",
        "sha256": "a" * 64,
    }


def _checkpoint(
    *,
    mu_order: int,
    values: tuple[float, float, float, float, float],
    qualified: bool,
    raw_audits: tuple[str, str, str, str, str],
) -> dict[str, object]:
    plans = _passes(mu_order)
    summaries = tuple(
        _summary(f"cached-{name}", value)
        for name, value in zip(_PASS_NAMES, values)
    )
    evidence = [
        _evidence(index, plan, summary, audit)
        for index, (plan, summary, audit) in enumerate(
            zip(plans, summaries, raw_audits)
        )
    ]
    policy = KerrReturningRadiationConvergenceV2Policy()
    runtime = {
        "descriptor": {
            "nativeWholeRay": {"artifacts": {"library": _library()}},
        },
        "descriptorSha256": "b" * 64,
    }
    source_freeze = {
        "afterAuthentication": {"manifestSha256": "c" * 64},
        "beforeExecution": {"manifestSha256": "c" * 64},
        "stable": True,
    }
    return {
        "authenticatedConvergenceV2": {
            "descriptor": {"passEvidence": evidence},
            "policy": {
                "descriptor": dict(policy.descriptor()),
                "descriptorSha256": policy.model_descriptor_sha256,
            },
            "summaries": [
                {
                    "descriptor": dict(summary.descriptor()),
                    "descriptorSha256": summary.model_descriptor_sha256,
                }
                for summary in summaries
            ],
        },
        "evaluatorMode": NATIVE_CPU_EVALUATOR_MODE,
        "id": f"synthetic-{mu_order}",
        "producer": {
            "cacheDefinition": {
                "scientificIdentity": {
                    "annulusEdgesOverMass": [6.0, 7.0],
                    "fixedPhysicsIdentity": "synthetic-no-ray",
                },
                "scientificPlan": {
                    "annulusCount": 1,
                    "canonicalOrder": "pass/face/annulus/rho/mu/psi",
                    "directionCount": sum(
                        2
                        * int(item["rhoOrder"])
                        * int(item["muOrder"])
                        * int(item["psiCount"])
                        for item in plans
                    ),
                    "faces": ["upper", "lower"],
                    "formulation": "forward",
                    "passes": plans,
                    "schema": "blackhole.returning-radiation-kernel-task-plan/v1",
                },
            }
        },
        "qualified": qualified,
        "runtimeNumericBackend": runtime,
        "schema": NATIVE_MANIFEST_SCHEMA,
        "sourceFreeze": source_freeze,
    }


def _fixture() -> tuple[dict[str, object], dict[str, object]]:
    parent = _checkpoint(
        mu_order=32,
        values=(0.20, 0.19, 0.25, 0.18, 0.22),
        qualified=False,
        raw_audits=tuple(character * 64 for character in "12345"),
    )
    target = _checkpoint(
        mu_order=16,
        values=(0.25, 0.24, 0.26, 0.24, 0.29),
        qualified=False,
        # The physical sample audit excludes pass identity, so target full
        # reproduces the parent half-mu raw digest exactly.
        raw_audits=("3" * 64, "7" * 64, "8" * 64, "9" * 64, "a" * 64),
    )
    return parent, target


class MuPsiPhaseInteractionTests(unittest.TestCase):
    def _body(
        self,
        parent: dict[str, object],
        target: dict[str, object],
    ) -> dict[str, object]:
        return analyzer._analysis_body(
            parent,
            target,
            parent_manifest_sha256="d" * 64,
            target_manifest_sha256="e" * 64,
            dylib_artifact=_library(),
            source_snapshot={
                "artifactCount": 1,
                "artifacts": [
                    {
                        "byteLength": 1,
                        "logicalPath": "synthetic.py",
                        "sha256": "f" * 64,
                    }
                ],
                "manifestSha256": "0" * 64,
            },
        )

    def test_six_cells_edges_and_two_identified_interactions_are_deterministic(self) -> None:
        parent, target = _fixture()
        body = self._body(parent, target)

        self.assertEqual(tuple(body["cells"]), ("A", "B", "C", "D", "E", "F"))
        self.assertEqual(body["cells"]["B"]["phaseCells"], 0.5)
        self.assertEqual(body["cells"]["D"]["muOrder"], 16)
        self.assertEqual(body["cells"]["E"]["psiCount"], 32)
        self.assertEqual(body["cells"]["F"]["psiCount"], 32)
        self.assertEqual(len(body["edgeDiagnostics"]), 7)

        mu_phase = body["interactionDiagnostics"]["muByPhase"]
        mu_psi = body["interactionDiagnostics"]["muByPsi"]
        self.assertAlmostEqual(mu_phase["maximumAbsoluteInteraction"], 0.02)
        self.assertAlmostEqual(mu_psi["maximumAbsoluteInteraction"], 0.01)
        self.assertEqual(mu_phase["definition"], "D-C-B+A")
        self.assertAlmostEqual(mu_phase["signedG2Interaction"][0], 0.02)
        self.assertEqual(mu_psi["definition"], "F-C-E+A")
        self.assertAlmostEqual(mu_psi["signedG2Interaction"][0], 0.01)
        self.assertGreater(mu_phase["maximumInteractionToleranceUnits"], 0.0)
        self.assertFalse(body["coverage"]["psiByPhaseIdentified"])
        self.assertFalse(body["coverage"]["muByPsiByPhaseIdentified"])
        self.assertEqual(
            canonical_json_bytes(body),
            canonical_json_bytes(self._body(parent, target)),
        )

    def test_c_reproduction_excludes_pass_identity_but_requires_raw_audit(self) -> None:
        parent, target = _fixture()
        body = self._body(parent, target)
        reproduction = body["reproductionGate"]

        self.assertTrue(reproduction["exactPhysicalSummary"])
        self.assertTrue(reproduction["normalizedPassEvidenceExact"])
        self.assertTrue(reproduction["rawSampleAuditDigestEqualityRequired"])
        self.assertTrue(reproduction["rawSampleAuditSha256"]["equal"])
        self.assertIn("summary.gridId", reproduction["excludedPassIdentityFields"])

    def test_c_raw_physical_sample_audit_drift_fails_closed(self) -> None:
        parent, target = _fixture()
        target["authenticatedConvergenceV2"]["descriptor"]["passEvidence"][0][
            "sampleAuditSha256"
        ] = "f" * 64
        with self.assertRaisesRegex(
            analyzer.MuPsiPhaseInteractionError,
            "physical sample audit",
        ):
            self._body(parent, target)

    def test_phase64_even_reclosure_and_record_substitution_reproduce_e(self) -> None:
        phase64, even, odd, evidence = analyzer._phase_half_grid_rules()
        self.assertEqual(len(phase64), 32 * 64)
        self.assertEqual(len(even), 32 * 32)
        self.assertEqual(len(odd), 32 * 32)
        self.assertGreater(evidence["even"]["tailClosureIterations"], 0)
        self.assertGreater(evidence["odd"]["tailClosureIterations"], 0)
        # The final even node demonstrates why selection*2 without canonical
        # reclosure is not an exact E reproduction.
        self.assertNotEqual(
            (2.0 * phase64[-2].normalized_emitted_flux_weight).hex(),
            even[-1].normalized_emitted_flux_weight.hex(),
        )

        def record(pass_index: int, pass_name: str, psi_index: int, node):
            return {
                "coordinate": {
                    "annulusIndex": 0,
                    "face": "upper",
                    "faceIndex": 0,
                    "muIndex": 0,
                    "ordinal": 100 + pass_index,
                    "passIndex": pass_index,
                    "passName": pass_name,
                    "psiIndex": psi_index,
                    "rhoIndex": 0,
                },
                "transport": {
                    "formulation": "forward",
                    "sample": {
                        "emissionAngleCosine": node.emission_angle_cosine,
                        "muIndex": 0,
                        "normalizedEmittedFluxWeight": (
                            node.normalized_emitted_flux_weight
                        ),
                        "passIndex": pass_index,
                        "passName": pass_name,
                        "psiIndex": psi_index,
                        "rhoIndex": 0,
                        "sourceAnnulusIndex": 0,
                        "sourceFace": "upper",
                        "sourceRadiusOverMass": 6.5,
                        "tangentAzimuthRad": node.tangent_azimuth_rad,
                    },
                    "schema": (
                        "blackhole.kerr-forward-returning-direction-transport/v1"
                    ),
                    "transport": {
                        "fate": "captured",
                        "g2": 0.0,
                        "primitiveDescriptorSha256": "1" * 64,
                    },
                },
            }

        b = record(4, "phase-shifted", 0, phase64[0])
        e = record(3, "half-psi", 0, even[0])
        b_key, b_normalized = analyzer._normalized_phase_cache_record(
            b,
            source_pass="B",
            mapped_psi_index=0,
            source_node=phase64[0],
            derived_node=even[0],
        )
        e_key, e_normalized = analyzer._normalized_phase_cache_record(
            e,
            source_pass="E",
            mapped_psi_index=0,
            source_node=even[0],
            derived_node=even[0],
        )
        self.assertEqual(b_key, e_key)
        self.assertEqual(b_normalized, e_normalized)

    def test_c_physical_drift_fails_closed(self) -> None:
        parent, target = _fixture()
        changed = _summary("cached-full", 0.251)
        target["authenticatedConvergenceV2"]["summaries"][0] = {
            "descriptor": dict(changed.descriptor()),
            "descriptorSha256": changed.model_descriptor_sha256,
        }
        with self.assertRaisesRegex(
            analyzer.MuPsiPhaseInteractionError,
            "does not exactly reproduce parent C physical summary",
        ):
            self._body(parent, target)

    def test_nonqualified_boundary_is_permanent_even_when_edges_converge(self) -> None:
        parent = _checkpoint(
            mu_order=32,
            values=(0.20, 0.20, 0.20, 0.20, 0.20),
            qualified=False,
            raw_audits=tuple(character * 64 for character in "12345"),
        )
        target = _checkpoint(
            mu_order=16,
            values=(0.20, 0.20, 0.20, 0.20, 0.20),
            qualified=False,
            raw_audits=("3" * 64, "7" * 64, "8" * 64, "9" * 64, "a" * 64),
        )
        body = self._body(parent, target)
        self.assertTrue(
            all(
                edge["withinV2Tolerance"]
                for edge in body["edgeDiagnostics"].values()
            )
        )
        report = analyzer._complete_report(body)

        self.assertFalse(report["qualified"])
        self.assertFalse(report["productionQualified"])
        self.assertFalse(report["productEligible"])
        self.assertFalse(report["automaticEscalation"])
        self.assertFalse(report["productionFivePassContractModified"])
        self.assertFalse(report["scientificBoundary"]["mayAuthorizeProduct"])
        self.assertFalse(report["scientificBoundary"]["mayEscalateAutomatically"])
        integrity_body = {
            key: value
            for key, value in report.items()
            if key not in ("id", "integrity")
        }
        self.assertEqual(
            report["integrity"]["analysisBodySha256"],
            analyzer._canonical_sha256(integrity_body),
        )

    def test_parent_must_be_the_nonqualified_checkpoint(self) -> None:
        parent, target = _fixture()
        parent["qualified"] = True
        with self.assertRaisesRegex(
            analyzer.MuPsiPhaseInteractionError,
            "parent checkpoint must be the retained non-qualified",
        ):
            self._body(parent, target)

    def test_atomic_new_report_refuses_overwrite(self) -> None:
        payload = canonical_json_bytes({"permanentlyQualified": False})
        # macOS exposes /var as a symlink to /private/var; use the canonical
        # path because the publisher deliberately rejects symlink ancestors.
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            output = Path(temporary) / "analysis.json"
            analyzer._publish_new_report(output, payload)
            self.assertEqual(output.read_bytes(), payload)
            with self.assertRaises(FileExistsError):
                analyzer._publish_new_report(output, b"replacement\n")
            self.assertEqual(output.read_bytes(), payload)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"permanentlyQualified": False},
            )

    def test_public_verifier_call_alias_cannot_be_replaced(self) -> None:
        with (
            patch.object(analyzer, "_VERIFY_CALL_ENTRY", lambda *_args, **_kwargs: None),
            self.assertRaisesRegex(
                analyzer.MuPsiPhaseInteractionError,
                "binding changed",
            ),
        ):
            analyzer.analyze_mu_psi_phase_interaction(
                Path("/private/tmp/parent/manifest.json"),
                "a" * 64,
                Path("/private/tmp/target/manifest.json"),
                "b" * 64,
                Path("/private/tmp/libblackhole_cpu.dylib"),
                Path("/private/tmp/report.json"),
            )

    def test_transitive_science_helper_cannot_be_replaced(self) -> None:
        with (
            patch.object(analyzer, "_c_reproduction", lambda *_args: {}),
            self.assertRaisesRegex(
                analyzer.MuPsiPhaseInteractionError,
                "helper binding changed",
            ),
        ):
            analyzer.analyze_mu_psi_phase_interaction(
                Path("/private/tmp/parent/manifest.json"),
                "a" * 64,
                Path("/private/tmp/target/manifest.json"),
                "b" * 64,
                Path("/private/tmp/libblackhole_cpu.dylib"),
                Path("/private/tmp/report.json"),
            )

    def test_output_inode_ancestry_rejects_input_directory(self) -> None:
        payload = canonical_json_bytes({"qualified": False})
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            directory = Path(temporary)
            identity = analyzer._secure_directory_identity(
                directory,
                "synthetic checkpoint",
            )
            output = directory / "analysis.json"
            with self.assertRaisesRegex(
                analyzer.MuPsiPhaseInteractionError,
                "input checkpoint/cache directory",
            ):
                analyzer._publish_new_report(
                    output,
                    payload,
                    forbidden_directory_identities=frozenset((identity,)),
                )
            self.assertFalse(output.exists())

    def test_loaded_callable_snapshot_matches_frozen_source(self) -> None:
        self.assertEqual(
            analyzer._callable_snapshot(),
            analyzer._FROZEN_CALLABLE_SNAPSHOT,
        )
        self.assertEqual(
            analyzer._source_snapshot(),
            analyzer._FROZEN_SOURCE_SNAPSHOT,
        )

    def test_direct_cli_help_is_source_compilation_stable(self) -> None:
        root = Path(__file__).resolve().parents[3]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "tools/native/mu_psi_phase_interaction.py"),
                "--help",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("mu x psi x phase analysis", completed.stdout)

    def test_cache_trust_retains_both_external_snapshot_anchors(self) -> None:
        document = {
            "operationalExecution": {"taskCount": 1},
            "producer": {
                "cacheDefinition": {
                    "cacheJobKey": "1" * 64,
                    "jobSpec": {},
                    "scientificJobKey": "2" * 64,
                    "scientificPlan": {"directionCount": 1},
                    "scientificPlanSha256": "3" * 64,
                }
            },
        }
        trust, _job = analyzer._cache_trust_from_parent_manifest(
            document,
            expected_payload_set_sha256="4" * 64,
            expected_direction_stream_sha256="5" * 64,
        )
        self.assertEqual(trust.authenticated_payload_set_sha256, "4" * 64)
        self.assertEqual(trust.authenticated_direction_stream_sha256, "5" * 64)


if __name__ == "__main__":
    unittest.main()
