from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from offline.authenticated_artifact import AuthenticatedArtifactDigest
import offline.kerr_returning_radiation_live_replay_attestation as attestation
from offline.kerr_returning_radiation_live_replay_attestation import (
    ReturningRadiationLiveReplayAttestationError,
    default_live_replay_attestation_directory,
    publish_returning_radiation_live_replay_attestation,
    read_stable_bounded_file,
    validated_live_replay_report,
)
from offline.job import InputArtifact, canonical_json_bytes
from offline.kerr_returning_radiation_finite_thickness_frame import (
    KerrReturningRadiationFiniteThicknessRaySampler,
)
from offline.spectral_product import SpectralProductPublication
from scripts import (
    verify_offline_kerr_returning_radiation_live_replay_attestation as verifier,
)


def _publication(root: Path) -> SpectralProductPublication:
    return SpectralProductPublication(
        output_directory=root / "product",
        manifest_path=root / "product" / "manifest.json",
        manifest_sha256="a" * 64,
        product_id="scientific-spectral-frame-" + "b" * 24,
        product_sha256="c" * 64,
        tile_count=2,
        record_count=3,
    )


def _report(publication: SpectralProductPublication) -> dict[str, object]:
    return {
        "id": publication.product_id,
        "status": "returning-radiation-live-sampler-byte-replay-conformant",
        "structuralContractVerified": True,
        "producerIdentityCurrentMatch": True,
        "jobSpecVerified": True,
        "samplerDescriptorLiveMatch": True,
        "numericBackendCurrentMatch": True,
        "sourceArtifactsCurrentMatch": True,
        "frameGeodesicsReplayed": True,
        "frozenThermalSnapshotBound": True,
        "thermalFixedPointReplayed": False,
        "directionCacheRecordsReplayed": False,
        "directionRaysRetraced": False,
        "independentPhysicsOracle": False,
        "pixelBytesExact": True,
        "recordCount": publication.record_count,
        "tileCount": publication.tile_count,
        "totalRaySamples": 5,
        "totalFrameGeodesicsReplayed": 10,
        "replayScope": "same-code production whole-pixel byte replay",
        "scientificScope": "finite-grid returning-thermal frame geodesics",
        "sourceHashScope": "exact current producer source bytes",
    }


class ReturningRadiationLiveReplayAttestationUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve(strict=True)
        self.publication = _publication(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _mock_publish(
        self,
        output: Path,
        *,
        promotion: object | None = None,
        live_identities: object | None = None,
        structural_validation: object | None = None,
    ) -> tuple[object, Mock, Path]:
        product = self.root / "mock-product"
        product.mkdir(exist_ok=True)
        subject = product / "manifest.json"
        subject_payload = canonical_json_bytes({})
        subject.write_bytes(subject_payload)
        publication = SpectralProductPublication(
            output_directory=product,
            manifest_path=subject,
            manifest_sha256=hashlib.sha256(subject_payload).hexdigest(),
            product_id="scientific-spectral-frame-" + "b" * 24,
            product_sha256="c" * 64,
            tile_count=2,
            record_count=3,
        )
        sampler = object.__new__(
            KerrReturningRadiationFiniteThicknessRaySampler
        )
        entry = Mock(return_value=_report(publication))
        snapshot = object()
        live_identity = Mock(
            return_value=(b"canonical-sampler-descriptor", snapshot)
        )
        if live_identities is not None:
            live_identity.side_effect = live_identities
        artifact = {
            "id": "returning-radiation-live-replay-attestation-" + "d" * 24,
        }
        structural_validator = Mock(return_value=b"structural-report")
        if structural_validation is not None:
            structural_validator.side_effect = structural_validation
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    attestation,
                    "_canonical_live_replay_entry",
                    return_value=entry,
                )
            )
            stack.enter_context(
                patch.object(
                    attestation,
                    "_canonical_structural_verifier_entry",
                    return_value=structural_validator,
                )
            )
            stack.enter_context(
                patch.object(
                    attestation,
                    "_validated_sampler_live_identity",
                    live_identity,
                )
            )
            stack.enter_context(
                patch.object(
                    attestation,
                    "_manifest_claims",
                    return_value=artifact,
                )
            )
            stack.enter_context(
                patch.object(
                    attestation,
                    "_validated_structural_subject",
                    structural_validator,
                )
            )
            if promotion is not None:
                stack.enter_context(
                    patch.object(
                        attestation,
                        "_promote_directory_no_replace_at",
                        side_effect=promotion,
                    )
                )
            result = publish_returning_radiation_live_replay_attestation(
                output,
                spectral_publication=publication,
                sampler=sampler,
                spectral_schema_path=self.root / "unused-schema.json",
            )
        return result, live_identity, subject

    def test_publisher_owns_report_and_default_path_is_outer_sibling(self) -> None:
        parameters = inspect.signature(
            publish_returning_radiation_live_replay_attestation
        ).parameters
        self.assertNotIn("report", parameters)
        self.assertNotIn("live_replay_report", parameters)
        self.assertEqual(
            default_live_replay_attestation_directory(
                self.publication.output_directory
            ),
            self.publication.output_directory.with_name(
                "product.live-replay-attestation-v1"
            ),
        )
        with self.assertRaises(TypeError):
            publish_returning_radiation_live_replay_attestation(
                self.root / "attestation",
                spectral_publication=self.publication,
                sampler=object(),
                spectral_schema_path=self.root / "schema.json",
                live_replay_report=_report(self.publication),  # type: ignore[call-arg]
            )

    def test_publication_fields_and_subject_paths_are_exact_at_front_door(
        self,
    ) -> None:
        output = self.root / "product"
        valid = SpectralProductPublication(
            output_directory=output,
            manifest_path=output / "manifest.json",
            manifest_sha256="a" * 64,
            product_id="scientific-spectral-frame-" + "b" * 24,
            product_sha256="c" * 64,
            tile_count=2,
            record_count=3,
        )
        attestation._validate_spectral_publication(valid)
        cases = (
            replace(valid, output_directory=Path("relative-product")),
            replace(valid, manifest_path=self.root / "foreign" / "manifest.json"),
            replace(valid, manifest_path=output / "foreign.json"),
            replace(valid, manifest_sha256="A" * 64),
            replace(valid, product_id="foreign-product"),
            replace(valid, product_sha256="z" * 64),
            replace(valid, tile_count=True),
            replace(valid, record_count=0),
        )
        for publication in cases:
            with self.subTest(publication=publication), self.assertRaises(TypeError):
                publish_returning_radiation_live_replay_attestation(
                    self.root / "attestation.live-replay-attestation-v1",
                    spectral_publication=publication,
                    sampler=object(),
                    spectral_schema_path=self.root / "schema.json",
                )

        sampler = object.__new__(
            KerrReturningRadiationFiniteThicknessRaySampler
        )
        with self.assertRaisesRegex(ValueError, "non-nested"):
            publish_returning_radiation_live_replay_attestation(
                output / "inside.live-replay-attestation-v1",
                spectral_publication=valid,
                sampler=sampler,
                spectral_schema_path=self.root / "schema.json",
            )

    def test_report_exact_schema_counters_and_false_tiers(self) -> None:
        expected = _report(self.publication)
        self.assertEqual(
            validated_live_replay_report(expected, self.publication),
            expected,
        )
        cases = (
            ({**expected, "foreign": False}, "non-exact"),
            ({**expected, "physicsVerified": False}, "non-exact"),
            ({**expected, "thermalFixedPointReplayed": True}, "overclaims"),
            ({**expected, "independentPhysicsOracle": True}, "overclaims"),
            ({**expected, "recordCount": True}, "invalid counters"),
            ({**expected, "totalFrameGeodesicsReplayed": 9}, "do not close"),
            ({**expected, "replayScope": ""}, "scope"),
        )
        for candidate, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                (ReturningRadiationLiveReplayAttestationError, TypeError),
                message,
            ):
                validated_live_replay_report(candidate, self.publication)

    def test_bounded_reader_rejects_oversize_symlink_and_toctou(self) -> None:
        path = self.root / "artifact.json"
        path.write_bytes(b"12345")
        with self.assertRaisesRegex(RuntimeError, "byte limit"):
            read_stable_bounded_file(path, 4, "artifact")
        link = self.root / "link.json"
        link.symlink_to(path)
        with self.assertRaisesRegex(RuntimeError, "without following symbolic links"):
            read_stable_bounded_file(link, 16, "artifact link")

        before = AuthenticatedArtifactDigest(path, 5, "1" * 64)
        after = replace(before, sha256="2" * 64)
        with patch.object(
            attestation,
            "authenticate_stable_artifact",
            side_effect=(before, after),
        ), self.assertRaisesRegex(RuntimeError, "changed across"):
            read_stable_bounded_file(path, 16, "changing artifact")

    def test_strict_json_resource_failures_are_contract_errors(self) -> None:
        huge_integer = b'{"value":' + (b"9" * 10_000) + b"}"
        with self.assertRaisesRegex(RuntimeError, "strict UTF-8 JSON"):
            attestation._strict_canonical_json(huge_integer, "manifest")
        with self.assertRaisesRegex(
            verifier.ReturningRadiationLiveReplayAttestationContractError,
            "invalid UTF-8 JSON",
        ):
            verifier._strict_json(huge_integer, "$")

        deeply_nested = (b'{"value":' * 2_000) + b"null" + (b"}" * 2_000)
        with self.assertRaisesRegex(RuntimeError, "nesting limit"):
            attestation._strict_canonical_json(deeply_nested, "manifest")
        with self.assertRaisesRegex(
            verifier.ReturningRadiationLiveReplayAttestationContractError,
            "nesting limit",
        ):
            verifier._strict_json(deeply_nested, "$")

    def test_current_source_binding_rejects_extra_repo_source_entries(self) -> None:
        current = InputArtifact("repo-source://offline/current.py", 1, "a" * 64)
        job_spec = {
            "inputs": [
                current.as_dict(),
                InputArtifact("science-data://table", 2, "b" * 64).as_dict(),
            ],
            "producerSourceHashes": ["a" * 64],
        }
        manifest = {"producer": {"jobSpec": job_spec}}
        verifier._source_identity_matches_current(manifest, (current,))

        job_spec["inputs"].append(
            InputArtifact("repo-source://offline/old.py", 1, "a" * 64).as_dict()
        )
        with self.assertRaisesRegex(
            verifier.ReturningRadiationLiveReplayAttestationContractError,
            "exact current source artifact set",
        ):
            verifier._source_identity_matches_current(manifest, (current,))

    def test_publisher_uses_anchored_promotion_and_final_stable_closure(self) -> None:
        success_parent = self.root / "success-parent"
        success_parent.mkdir()
        success = success_parent / "frame.live-replay-attestation-v1"
        publication, live_identity, _subject = self._mock_publish(success)
        self.assertEqual(publication.output_directory, success)
        self.assertEqual(
            set(path.name for path in success.iterdir()),
            {"manifest.json", "manifest.sha256"},
        )
        self.assertGreaterEqual(live_identity.call_count, 4)
        with self.assertRaises(FileExistsError):
            self._mock_publish(success)

        original_promotion = attestation._promote_directory_no_replace_at

        late_failure_parent = self.root / "late-failure-parent"
        late_failure_parent.mkdir()
        late_failure_output = (
            late_failure_parent / "frame.live-replay-attestation-v1"
        )

        def fail_after_promotion(*arguments: object) -> None:
            original_promotion(*arguments)
            raise OSError("simulated post-rename durability failure")

        with self.assertRaisesRegex(OSError, "durability failure"):
            self._mock_publish(
                late_failure_output,
                promotion=fail_after_promotion,
            )
        self.assertEqual(
            set(path.name for path in late_failure_output.iterdir()),
            {"manifest.json", "manifest.sha256"},
        )

        subject_parent = self.root / "subject-parent"
        subject_parent.mkdir()
        subject_output = subject_parent / "frame.live-replay-attestation-v1"

        def mutate_subject(*arguments: object) -> None:
            original_promotion(*arguments)
            (self.root / "mock-product" / "manifest.json").write_bytes(
                canonical_json_bytes({"changed": True})
            )

        with self.assertRaisesRegex(RuntimeError, "final subject"):
            self._mock_publish(subject_output, promotion=mutate_subject)

        bytes_parent = self.root / "bytes-parent"
        bytes_parent.mkdir()
        bytes_output = bytes_parent / "frame.live-replay-attestation-v1"

        def mutate_published_bytes(*arguments: object) -> None:
            original_promotion(*arguments)
            (bytes_output / "manifest.json").write_bytes(
                canonical_json_bytes({"changed": True})
            )

        with self.assertRaisesRegex(RuntimeError, "published bytes"):
            self._mock_publish(bytes_output, promotion=mutate_published_bytes)

        extra_parent = self.root / "extra-parent"
        extra_parent.mkdir()
        extra_output = extra_parent / "frame.live-replay-attestation-v1"

        def add_late_subject_entry(*arguments: object) -> None:
            original_promotion(*arguments)
            extra = self.root / "mock-product" / "late-extra"
            extra.mkdir()

        def require_closed_subject(_publication: object, _schema: object) -> bytes:
            if (self.root / "mock-product" / "late-extra").exists():
                raise RuntimeError("subject closed tree changed after promotion")
            return b"structural-report"

        with self.assertRaisesRegex(RuntimeError, "closed tree changed"):
            self._mock_publish(
                extra_output,
                promotion=add_late_subject_entry,
                structural_validation=require_closed_subject,
            )

        identity_parent = self.root / "identity-parent"
        identity_parent.mkdir()
        identity_output = identity_parent / "frame.live-replay-attestation-v1"
        moved_parent = self.root / "moved-identity-parent"

        def replace_parent_identity(*arguments: object) -> None:
            original_promotion(*arguments)
            identity_parent.rename(moved_parent)
            identity_parent.mkdir()

        with self.assertRaisesRegex(RuntimeError, "parent.*identity changed"):
            self._mock_publish(identity_output, promotion=replace_parent_identity)

        sampler_parent = self.root / "sampler-parent"
        sampler_parent.mkdir()
        sampler_output = sampler_parent / "frame.live-replay-attestation-v1"
        stable_snapshot = object()
        with self.assertRaisesRegex(RuntimeError, "final subject, sampler"):
            self._mock_publish(
                sampler_output,
                live_identities=(
                    (b"descriptor", stable_snapshot),
                    (b"descriptor", stable_snapshot),
                    (b"changed", stable_snapshot),
                ),
            )

    def test_public_publisher_rejects_symlinked_parent_before_replay(self) -> None:
        real_parent = self.root / "real-parent"
        real_parent.mkdir()
        alias = self.root / "alias-parent"
        alias.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "without following symlinks"):
            self._mock_publish(
                alias / "frame.live-replay-attestation-v1",
            )
        self.assertEqual(list(real_parent.iterdir()), [])

    def test_verifier_rereads_bytes_after_its_final_closed_tree_scan(self) -> None:
        root = self.root / "history.live-replay-attestation-v1"
        root.mkdir()
        manifest = root / "manifest.json"
        manifest.write_bytes(canonical_json_bytes({"stable": True}))
        (root / "manifest.sha256").write_bytes(b"sidecar")
        original_scan = verifier._closed_artifact_root
        calls = 0

        def mutate_after_final_scan(path: Path):
            nonlocal calls
            identity = original_scan(path)
            calls += 1
            if calls == 2:
                manifest.write_bytes(canonical_json_bytes({"stable": False}))
            return identity

        with patch.object(
            verifier,
            "_closed_artifact_root",
            side_effect=mutate_after_final_scan,
        ), self.assertRaisesRegex(
            verifier.ReturningRadiationLiveReplayAttestationContractError,
            "final self-closure",
        ):
            verifier._closed_artifact_snapshot(manifest, "final")

    def test_clean_process_verifier_loads_lazy_replay_before_origin_gate(
        self,
    ) -> None:
        probe = """
from scripts.verify_offline_kerr_returning_radiation_live_replay_attestation import _current_source_artifacts
artifacts = _current_source_artifacts()
print(len(artifacts))
"""
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=attestation.ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertGreater(int(completed.stdout.strip()), 40)


if __name__ == "__main__":
    unittest.main()
