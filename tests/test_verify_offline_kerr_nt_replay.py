from __future__ import annotations

import hashlib
import io
import json
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from offline.job import InputArtifact, JobSpec, TaskKey, canonical_json_bytes
from offline.kerr_nt_replay import (
    DEFAULT_REPLAY_LIMITS,
    KerrNtReplayError,
    MAXIMUM_OFFICIAL_CIE_FREQUENCY_BINS,
    ReplayResourceLimits,
    validate_kerr_nt_replay,
)
from offline.spectral_frame import (
    SpectralPixelLayout,
    pack_spectral_pixel,
    unpack_spectral_pixel,
)
import scripts.render_offline_kerr_nt_frame as renderer
import scripts.verify_offline_kerr_nt_replay as replay_cli
from scripts.verify_offline_spectral_frame import (
    validate_scientific_spectral_frame,
)


ROOT = Path(__file__).resolve().parents[1]


class KerrNtReplayVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.baseline = cls.root / "baseline"
        arguments = renderer.parse_args(
            [
                str(cls.baseline),
                "--cache",
                str(cls.root / "cache"),
                "--spin",
                "0.7",
                "--black-hole-mass-solar",
                "1e8",
                "--accretion-rate-kg-s",
                "1e22",
                "--inclination-deg",
                "63.0253574644",
                "--frequency-hz",
                "5e14",
                "--width",
                "1",
                "--height",
                "1",
                "--tile-width",
                "1",
                "--tile-height",
                "1",
                "--screen-x-min",
                "0.49999",
                "--screen-x-max",
                "0.50001",
                "--screen-y-min",
                "-0.50001",
                "--screen-y-max",
                "-0.49999",
                "--minimum-depth",
                "0",
                "--maximum-depth",
                "0",
                "--maximum-ray-evaluations",
                "64",
                "--radiance-guard-ceiling",
                "100",
                "--ray-absolute-tolerance",
                "5e-10",
                "--ray-relative-tolerance",
                "5e-10",
                "--ray-maximum-step",
                "0.25",
                "--ray-maximum-affine-length",
                "300",
                "--surface-absolute-tolerance",
                "5e-10",
                "--surface-relative-tolerance",
                "5e-10",
                "--surface-null-residual-limit",
                "2e-7",
                "--frequency-null-residual-limit",
                "2e-7",
            ]
        )
        plan = renderer.build_render_plan(arguments)
        renderer.execute_render_plan(plan)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def setUp(self) -> None:
        self.case_root = Path(tempfile.mkdtemp(dir=self.root))
        self.output = self.case_root / "product"
        shutil.copytree(self.baseline, self.output)
        self.manifest_path = self.output / "manifest.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.case_root)

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

    @staticmethod
    def job_spec(manifest: dict[str, object]) -> JobSpec:
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

    def reseal(self, manifest: dict[str, object]) -> None:
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
        spec = self.job_spec(manifest)
        producer["jobSpec"] = spec.as_dict()
        producer["id"] = spec.producer
        producer["algorithmVersion"] = spec.algorithm_version
        producer["jobKey"] = spec.job_key
        for name in ("sampler", "runtimeNumericBackend"):
            described = manifest[name]
            described["descriptorSha256"] = hashlib.sha256(
                canonical_json_bytes(described["descriptor"])
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

    def test_valid_real_ray_is_byte_exact_and_honestly_scoped(self) -> None:
        report = validate_kerr_nt_replay(self.manifest_path)
        self.assertTrue(report["physicsReplayVerified"])
        self.assertTrue(report["structuralContractVerified"])
        self.assertFalse(report["independentPhysicsOracle"])
        self.assertFalse(report["isNumericalRelativitySolver"])
        self.assertFalse(report["isGeneralRelativisticMagnetohydrodynamics"])
        self.assertTrue(report["numericBackendCurrentMatch"])
        self.assertTrue(report["productBoundSourceHashesCurrentMatch"])
        self.assertGreater(report["sourceArtifactCount"], 0)
        self.assertEqual(report["recordCount"], 1)
        self.assertEqual(report["tileCount"], 1)
        self.assertEqual(report["totalRaySamples"], 13)
        self.assertIn("same-code-family", report["replayScope"])

    def test_resealed_payload_physics_tamper_fails_replay(self) -> None:
        manifest = self.read_manifest()
        layout = SpectralPixelLayout(tuple(manifest["observerFrequencyBinsHz"]))
        tile = manifest["tiles"][0]
        path = self.output / tile["payload"]["uri"]
        record = unpack_spectral_pixel(layout, path.read_bytes())
        modified = replace(
            record,
            maximum_null_residual=record.maximum_null_residual * 1.01,
        )
        payload = pack_spectral_pixel(layout, modified)
        path.write_bytes(payload)
        tile["payload"]["sha256"] = hashlib.sha256(payload).hexdigest()
        manifest["summary"]["maximumNullResidual"] = (
            modified.maximum_null_residual
        )
        self.reseal(manifest)
        validate_scientific_spectral_frame(self.manifest_path)
        with self.assertRaisesRegex(
            KerrNtReplayError,
            "replay mismatch.*maximum_null_residual",
        ):
            validate_kerr_nt_replay(self.manifest_path)

    def test_resealed_manifest_physics_tamper_fails_replay(self) -> None:
        manifest = self.read_manifest()
        disk = manifest["sampler"]["descriptor"]["disk"]
        disk["blackHoleMassKg"] *= 2.0
        self.reseal(manifest)
        validate_scientific_spectral_frame(self.manifest_path)
        with self.assertRaisesRegex(KerrNtReplayError, "replay mismatch"):
            validate_kerr_nt_replay(self.manifest_path)

    def test_source_hash_and_closed_implementation_fail_before_rays(self) -> None:
        manifest = self.read_manifest()
        raw = manifest["producer"]["jobSpec"]
        raw["inputs"][0]["sha256"] = "0" * 64
        self.reseal(manifest)
        validate_scientific_spectral_frame(self.manifest_path)
        with self.assertRaisesRegex(
            KerrNtReplayError,
            "producer sources do not exactly match",
        ):
            validate_kerr_nt_replay(self.manifest_path)

        shutil.rmtree(self.output)
        shutil.copytree(self.baseline, self.output)
        manifest = self.read_manifest()
        angular = manifest["sampler"]["descriptor"]["angularEmission"]
        angular["descriptor"]["implementationId"] = "invented-atmosphere/v1"
        self.reseal(manifest)
        with self.assertRaisesRegex(
            KerrNtReplayError,
            "unsupported closed angular law",
        ):
            validate_kerr_nt_replay(self.manifest_path)

    def test_resource_preflight_and_cli_help_are_bounded(self) -> None:
        self.assertEqual(MAXIMUM_OFFICIAL_CIE_FREQUENCY_BINS, 471)
        self.assertEqual(DEFAULT_REPLAY_LIMITS.maximum_frequency_bins, 471)
        for frequency_bins in (65, 471):
            self.assertEqual(
                ReplayResourceLimits(
                    maximum_frequency_bins=frequency_bins
                ).maximum_frequency_bins,
                frequency_bins,
            )
            arguments = replay_cli.build_parser().parse_args(
                [
                    str(self.manifest_path),
                    "--maximum-frequency-bins",
                    str(frequency_bins),
                ]
            )
            self.assertEqual(arguments.max_frequency_bins, frequency_bins)
        with self.assertRaises(ValueError):
            ReplayResourceLimits(maximum_frequency_bins=472)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            replay_cli.build_parser().parse_args(
                [
                    str(self.manifest_path),
                    "--maximum-frequency-bins",
                    "472",
                ]
            )

        limits = replace(
            DEFAULT_REPLAY_LIMITS,
            maximum_total_ray_evaluations=12,
        )
        with self.assertRaisesRegex(KerrNtReplayError, "ray budget exceeds"):
            validate_kerr_nt_replay(self.manifest_path, limits=limits)

        manifest = self.read_manifest()
        manifest["observerFrequencyBinsHz"] = [
            float(index + 1) for index in range(472)
        ]
        self.write_manifest(manifest)
        with self.assertRaisesRegex(KerrNtReplayError, "frequency count exceeds"):
            validate_kerr_nt_replay(self.manifest_path)

        completed = subprocess.run(
            [
                sys.executable,
                "scripts/verify_offline_kerr_nt_replay.py",
                "--help",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--max-total-ray-evaluations", completed.stdout)
        self.assertIn("--maximum-frequency-bins", completed.stdout)
        self.assertIn("independent analytic physics oracle", completed.stdout)


if __name__ == "__main__":
    unittest.main()
