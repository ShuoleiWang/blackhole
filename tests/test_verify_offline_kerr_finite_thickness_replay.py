from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from offline.job import InputArtifact, JobSpec, TaskKey, canonical_json_bytes
from offline.kerr_finite_thickness_replay import (
    DEFAULT_REPLAY_LIMITS,
    KerrFiniteThicknessReplayError,
    ReplayResourceLimits,
    validate_kerr_finite_thickness_replay,
)
from offline.spectral_frame import (
    SpectralPixelLayout,
    pack_spectral_pixel,
    unpack_spectral_pixel,
)
import scripts.render_offline_kerr_finite_thickness_frame as renderer
from scripts.verify_offline_spectral_frame import (
    validate_scientific_spectral_frame,
)


ROOT = Path(__file__).resolve().parents[1]


class KerrFiniteThicknessReplayVerifierTests(unittest.TestCase):
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
                "--width",
                "1",
                "--height",
                "1",
                "--tile-width",
                "1",
                "--tile-height",
                "1",
                "--minimum-depth",
                "0",
                "--maximum-depth",
                "0",
                "--maximum-ray-evaluations",
                "32",
                "--ray-absolute-tolerance",
                "1e-8",
                "--ray-relative-tolerance",
                "1e-8",
                "--ray-initial-step-over-mass",
                "0.2",
                "--ray-maximum-step-over-mass",
                "0.5",
                "--ray-maximum-affine-length-over-mass",
                "100",
                "--surface-absolute-tolerance",
                "1e-8",
                "--surface-relative-tolerance",
                "1e-8",
                "--surface-subdivisions-per-segment",
                "2",
                "--thinness-gate-maximum-h-over-rho",
                "0.1",
                "--coarse-tolerance-multiplier",
                "8",
            ]
        )
        renderer.execute_render_plan(renderer.build_render_plan(arguments))

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
                InputArtifact(item["uri"], item["byteLength"], item["sha256"])
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

    def test_real_cie471_product_is_byte_exact_and_honestly_scoped(self) -> None:
        report = validate_kerr_finite_thickness_replay(self.manifest_path)
        self.assertTrue(report["physicsReplayVerified"])
        self.assertTrue(report["structuralContractVerified"])
        self.assertTrue(report["officialCie471CurrentMatch"])
        self.assertFalse(report["independentPhysicsOracle"])
        self.assertFalse(report["includesReturningRadiation"])
        self.assertFalse(report["isGeneralRelativisticMagnetohydrodynamics"])
        self.assertFalse(report["isNumericalRelativitySolver"])
        self.assertEqual(report["recordCount"], 1)
        self.assertEqual(report["tileCount"], 1)
        self.assertEqual(report["totalRaySamples"], 13)
        self.assertEqual(report["totalGeodesicTraces"], 26)
        self.assertIn("same-code-family", report["replayScope"])
        manifest = self.read_manifest()
        self.assertEqual(
            manifest["sampler"]["descriptor"]["finiteThicknessSurface"]
            ["thinnessGateMaximumHOverRho"],
            0.1,
        )

    def test_resealed_tile_diagnostic_tamper_fails_replay(self) -> None:
        manifest = self.read_manifest()
        layout = SpectralPixelLayout(tuple(manifest["observerFrequencyBinsHz"]))
        tile = manifest["tiles"][0]
        path = self.output / tile["payload"]["uri"]
        record = unpack_spectral_pixel(layout, path.read_bytes())
        modified_value = record.maximum_null_residual * 1.01
        if modified_value == record.maximum_null_residual:
            modified_value = math.nextafter(modified_value, math.inf)
        modified = replace(record, maximum_null_residual=modified_value)
        payload = pack_spectral_pixel(layout, modified)
        path.write_bytes(payload)
        tile["payload"]["sha256"] = hashlib.sha256(payload).hexdigest()
        manifest["summary"]["maximumNullResidual"] = modified_value
        self.reseal(manifest)
        validate_scientific_spectral_frame(self.manifest_path)
        with self.assertRaisesRegex(
            KerrFiniteThicknessReplayError,
            "replay mismatch.*maximum_null_residual",
        ):
            validate_kerr_finite_thickness_replay(self.manifest_path)

    def test_resealed_config_tamper_fails_physics_replay(self) -> None:
        manifest = self.read_manifest()
        disk = manifest["sampler"]["descriptor"]["diskThermalProxy"]
        disk["blackHoleMassKg"] *= 2.0
        self.reseal(manifest)
        validate_scientific_spectral_frame(self.manifest_path)
        with self.assertRaisesRegex(
            KerrFiniteThicknessReplayError,
            "replay mismatch",
        ):
            validate_kerr_finite_thickness_replay(self.manifest_path)

    def test_resealed_source_tamper_fails_before_rays(self) -> None:
        manifest = self.read_manifest()
        raw = manifest["producer"]["jobSpec"]
        source = next(
            item for item in raw["inputs"] if item["uri"].startswith("repo-source://")
        )
        previous = source["sha256"]
        replacement = "0" * 64 if previous != "0" * 64 else "1" * 64
        source["sha256"] = replacement
        raw["producerSourceHashes"] = sorted(
            replacement if item == previous else item
            for item in raw["producerSourceHashes"]
        )
        self.reseal(manifest)
        validate_scientific_spectral_frame(self.manifest_path)
        with patch(
            "offline.kerr_finite_thickness_replay.integrate_spectral_pixel",
            side_effect=AssertionError("source mismatch must fail before rays"),
        ):
            with self.assertRaisesRegex(
                KerrFiniteThicknessReplayError,
                "producer/CIE inputs do not exactly match",
            ):
                validate_kerr_finite_thickness_replay(self.manifest_path)

    def test_resealed_backend_tamper_fails_before_rays(self) -> None:
        manifest = self.read_manifest()
        backend = manifest["runtimeNumericBackend"]["descriptor"]
        backend["processor"] = backend["processor"] + "-tampered"
        self.reseal(manifest)
        validate_scientific_spectral_frame(self.manifest_path)
        with patch(
            "offline.kerr_finite_thickness_replay.integrate_spectral_pixel",
            side_effect=AssertionError("backend mismatch must fail before rays"),
        ):
            with self.assertRaisesRegex(
                KerrFiniteThicknessReplayError,
                "current CPython/binary64 backend",
            ):
                validate_kerr_finite_thickness_replay(self.manifest_path)

    def test_unknown_sampler_fails_closed_before_rays(self) -> None:
        manifest = self.read_manifest()
        manifest["sampler"]["descriptor"]["implementationId"] = "unknown/v1"
        self.reseal(manifest)
        validate_scientific_spectral_frame(self.manifest_path)
        with patch(
            "offline.kerr_finite_thickness_replay.integrate_spectral_pixel",
            side_effect=AssertionError("unknown sampler must fail before rays"),
        ):
            with self.assertRaisesRegex(
                KerrFiniteThicknessReplayError,
                "only 'kerr-finite-thickness-spectral-ray-sampler/v1'",
            ):
                validate_kerr_finite_thickness_replay(self.manifest_path)

    def test_event_horizon_name_requires_exact_horizon_radius_before_rays(
        self,
    ) -> None:
        manifest = self.read_manifest()
        termination = manifest["sampler"]["descriptor"]["termination"]
        self.assertEqual(
            termination["captureTargetId"],
            "analytic-kerr-stretched-horizon",
        )
        termination["captureTargetId"] = "analytic-kerr-event-horizon"
        self.reseal(manifest)
        validate_scientific_spectral_frame(self.manifest_path)
        with patch(
            "offline.kerr_finite_thickness_replay.integrate_spectral_pixel",
            side_effect=AssertionError("target/radius mismatch must fail first"),
        ):
            with self.assertRaisesRegex(
                KerrFiniteThicknessReplayError,
                "event-horizon target requires the exact Kerr outer-horizon",
            ):
                validate_kerr_finite_thickness_replay(self.manifest_path)

    def test_resource_limits_cover_cie_rays_steps_events_and_surfaces(self) -> None:
        cases = (
            (
                replace(DEFAULT_REPLAY_LIMITS, maximum_frequency_bins=470),
                "frequency count exceeds",
            ),
            (
                replace(DEFAULT_REPLAY_LIMITS, maximum_total_ray_evaluations=63),
                "fine-plus-coarse geodesic budget exceeds",
            ),
            (
                replace(DEFAULT_REPLAY_LIMITS, maximum_ray_accepted_steps=99_999),
                "maximumAcceptedSteps.*exceeds replay limit",
            ),
            (
                replace(DEFAULT_REPLAY_LIMITS, maximum_ray_event_iterations=63),
                "eventMaximumIterations.*exceeds replay limit",
            ),
            (
                replace(
                    DEFAULT_REPLAY_LIMITS,
                    maximum_surface_subdivisions_per_segment=1,
                ),
                "subdivisionsPerSegment.*exceeds replay limit",
            ),
        )
        for limits, message in cases:
            with self.subTest(message=message):
                with patch(
                    "offline.kerr_finite_thickness_replay.integrate_spectral_pixel",
                    side_effect=AssertionError("resource gate must fail before rays"),
                ):
                    with self.assertRaisesRegex(
                        KerrFiniteThicknessReplayError,
                        message,
                    ):
                        validate_kerr_finite_thickness_replay(
                            self.manifest_path,
                            limits=limits,
                        )

    def test_resource_preflight_bounds_tiles_records_and_depth(self) -> None:
        baseline = self.read_manifest()
        cases = (
            (
                lambda manifest: manifest["tiles"].append(
                    deepcopy(manifest["tiles"][0])
                ),
                replace(DEFAULT_REPLAY_LIMITS, maximum_tiles=1),
                "tile count exceeds replay limit 1",
            ),
            (
                lambda manifest: manifest["tiles"][0].__setitem__(
                    "recordCount", 2
                ),
                replace(DEFAULT_REPLAY_LIMITS, maximum_records=1),
                "record count exceeds replay limit 1",
            ),
            (
                lambda manifest: manifest["adaptivePixelOptions"].__setitem__(
                    "maximumDepth", 1
                ),
                replace(DEFAULT_REPLAY_LIMITS, maximum_adaptive_depth=0),
                "adaptive depth exceeds replay limit 0",
            ),
        )
        for mutate, limits, message in cases:
            with self.subTest(message=message):
                manifest = deepcopy(baseline)
                mutate(manifest)
                self.write_manifest(manifest)
                with patch(
                    "offline.kerr_finite_thickness_replay.integrate_spectral_pixel",
                    side_effect=AssertionError("preflight must fail before rays"),
                ):
                    with self.assertRaisesRegex(
                        KerrFiniteThicknessReplayError,
                        message,
                    ):
                        validate_kerr_finite_thickness_replay(
                            self.manifest_path,
                            limits=limits,
                        )

    def test_limit_type_and_cie_ceiling_are_fail_closed(self) -> None:
        with self.assertRaisesRegex(TypeError, "limits must be ReplayResourceLimits"):
            validate_kerr_finite_thickness_replay(
                self.manifest_path,
                limits=object(),
            )
        with self.assertRaisesRegex(ValueError, "official CIE 471-bin grid"):
            ReplayResourceLimits(maximum_frequency_bins=472)

    def test_cli_help_and_limit_failure(self) -> None:
        help_result = subprocess.run(
            [
                sys.executable,
                "scripts/verify_offline_kerr_finite_thickness_replay.py",
                "--help",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("finite-thickness exact-Kerr", help_result.stdout)
        self.assertIn("--max-total-ray-evaluations", help_result.stdout)
        self.assertIn("--max-ray-event-iterations", help_result.stdout)
        self.assertIn("--max-surface-reintegrations", help_result.stdout)

        limited = subprocess.run(
            [
                sys.executable,
                "scripts/verify_offline_kerr_finite_thickness_replay.py",
                str(self.manifest_path),
                "--max-frequency-bins",
                "470",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(limited.returncode, 2)
        self.assertIn("frequency count exceeds replay limit 470", limited.stderr)


if __name__ == "__main__":
    unittest.main()
