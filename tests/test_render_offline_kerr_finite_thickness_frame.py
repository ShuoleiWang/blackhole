from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from offline.job import JobRun
from offline.kerr_finite_thickness_frame import KerrFiniteThicknessRaySampler
from offline.spectral_product import (
    SpectralProductError,
    SpectralProductPublication,
)
import scripts.render_offline_kerr_finite_thickness_frame as renderer


ROOT = Path(__file__).resolve().parents[1]
STRUCTURAL_STATUS = (
    "scientific-spectral-frame-structural-contract-conformant"
)


def minimal_arguments(
    output: Path,
    cache: Path,
    *extra: str,
) -> list[str]:
    return [
        str(output),
        "--cache",
        str(cache),
        "--width",
        "2",
        "--height",
        "1",
        "--tile-width",
        "1",
        "--tile-height",
        "1",
        "--maximum-depth",
        "0",
        *extra,
    ]


class RenderOfflineKerrFiniteThicknessFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "product"
        self.cache = self.root / "cache"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build_plan(
        self,
        *extra: str,
    ) -> renderer.KerrFiniteThicknessFramePlan:
        arguments = renderer.parse_args(
            minimal_arguments(self.output, self.cache, *extra)
        )
        return renderer.build_render_plan(arguments)

    def test_help_exposes_bound_physics_and_cie_grid_without_rendering(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/render_offline_kerr_finite_thickness_frame.py",
                "--help",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("authenticated CIE 471-bin", completed.stdout)
        self.assertIn("--metric-mass-m", completed.stdout)
        self.assertIn("--height-accretion-rate-eddington", completed.stdout)
        self.assertIn("--outer-radius-over-mass", completed.stdout)
        self.assertIn("--surface-subdivisions-per-segment", completed.stdout)
        self.assertIn("--coarse-tolerance-multiplier", completed.stdout)
        self.assertIn("--maximum-ray-evaluations", completed.stdout)

    def test_declared_sources_cover_transitive_offline_imports(self) -> None:
        declared = set(renderer.PRODUCER_SOURCE_FILES)
        imported: set[Path] = set()
        for relative in declared:
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            module_names: list[str] = []
            module_parts = list(relative.with_suffix("").parts)
            package_parts = module_parts[:-1]
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.level:
                        retained = len(package_parts) - (node.level - 1)
                        if retained < 0:
                            continue
                        imported_parts = package_parts[:retained]
                        if node.module is not None:
                            imported_parts.extend(node.module.split("."))
                        module_names.append(".".join(imported_parts))
                    elif node.module is not None:
                        module_names.append(node.module)
                elif isinstance(node, ast.Import):
                    module_names.extend(alias.name for alias in node.names)
            for module_name in module_names:
                if not module_name.startswith("offline."):
                    continue
                candidate = Path(*module_name.split(".")).with_suffix(".py")
                if (ROOT / candidate).is_file():
                    imported.add(candidate)
        radiative_transfer = Path("offline/radiative_transfer.py")
        self.assertIn(radiative_transfer, imported)
        self.assertIn(radiative_transfer, declared)
        self.assertEqual(imported - declared, set())

    def test_defaults_are_cie471_and_intentionally_resource_bounded(self) -> None:
        arguments = renderer.parse_args(
            [str(self.output), "--cache", str(self.cache)]
        )
        plan = renderer.build_render_plan(arguments)
        self.assertEqual(plan.layout.frequency_count, 471)
        self.assertEqual((plan.grid.width_pixels, plan.grid.height_pixels), (1, 1))
        self.assertEqual(
            (
                plan.grid.screen_x_min,
                plan.grid.screen_x_max,
                plan.grid.screen_y_min,
                plan.grid.screen_y_max,
            ),
            (0.49999, 0.50001, -0.50001, -0.49999),
        )
        self.assertEqual(len(plan.job_spec.tasks), 1)
        self.assertEqual(plan.adaptive_options.minimum_depth, 0)
        self.assertEqual(plan.adaptive_options.maximum_depth, 0)
        self.assertEqual(plan.adaptive_options.maximum_ray_evaluations, 32)
        self.assertEqual((plan.jobs, plan.max_in_flight), (1, 1))

    def test_plan_binds_finite_photosphere_cie471_and_source_identity(self) -> None:
        with patch.object(
            KerrFiniteThicknessRaySampler,
            "sample",
            side_effect=AssertionError("configuration must not trace rays"),
        ):
            plan = self.build_plan(
                "--metric-mass-m",
                "2",
                "--spin",
                "0.82",
                "--black-hole-mass-solar",
                "4000000",
                "--thermal-accretion-rate-kg-s",
                "2.5e20",
                "--dotm",
                "0.04",
                "--thinness-gate-maximum-h-over-rho",
                "0.1",
                "--rout-over-mass",
                "20",
                "--inclination-deg",
                "55",
                "--observer-phi-deg",
                "12",
                "--observer-coordinate-time-over-mass",
                "3",
                "--ray-relative-tolerance",
                "4e-10",
            )

        self.assertEqual(plan.layout.frequency_count, 471)
        self.assertTrue(
            all(
                right > left
                for left, right in zip(
                    plan.layout.observer_frequencies_hz,
                    plan.layout.observer_frequencies_hz[1:],
                )
            )
        )
        self.assertEqual((plan.grid.width_pixels, plan.grid.height_pixels), (2, 1))
        self.assertEqual(len(plan.job_spec.tasks), 2)
        self.assertEqual(plan.sampler.metric.mass_m, 2.0)
        self.assertEqual(plan.sampler.metric.dimensionless_spin, 0.82)
        self.assertEqual(
            plan.sampler.surface.calibration.eddington_scaled_mass_accretion_rate,
            0.04,
        )
        self.assertEqual(
            plan.sampler.surface.calibration.outer_radius_over_mass,
            20.0,
        )
        self.assertEqual(plan.sampler.observer_radius_m, 60.0)
        self.assertEqual(plan.sampler.observer_theta_rad, math.radians(55.0))
        self.assertEqual(plan.sampler.observer_coordinate_time_m, 6.0)
        self.assertEqual(plan.sampler.termination.escape_radius_m, 100.0)
        self.assertEqual(plan.sampler.fine_options.relative_tolerance, 4.0e-10)
        self.assertTrue(plan.sampler.fine_options.record_path)
        self.assertIs(plan.producer.inner.sampler, plan.sampler)

        descriptor = plan.sampler.descriptor()
        self.assertEqual(
            descriptor["implementationId"],
            "kerr-finite-thickness-spectral-ray-sampler/v1",
        )
        status = descriptor["scientificStatus"]
        self.assertFalse(status["includesReturningRadiation"])
        self.assertFalse(status["isGeneralRelativisticMagnetohydrodynamics"])
        self.assertFalse(status["isCompleteGeneralRelativisticRadiativeTransfer"])
        self.assertEqual(
            descriptor["finiteThicknessSurface"]["type"],
            "Zhou-prescribed-stationary-photosphere",
        )
        self.assertTrue(
            descriptor["finiteThicknessSurface"]
            ["heightRateIsIndependentOfThermalRate"]
        )
        self.assertEqual(
            descriptor["finiteThicknessSurface"]
            ["thinnessGateMaximumHOverRho"],
            0.1,
        )

        default_gate = self.build_plan(
            "--dotm",
            "0.04",
            "--thinness-gate-maximum-h-over-rho",
            "0.25",
        )
        self.assertNotEqual(
            plan.sampler.descriptor(),
            default_gate.sampler.descriptor(),
        )
        self.assertNotEqual(plan.job_spec.job_key, default_gate.job_spec.job_key)

        science_uris = {artifact.uri for artifact in plan.science_artifacts}
        self.assertEqual(
            science_uris,
            {renderer.CIE_CSV_INPUT_URI, renderer.CIE_METADATA_INPUT_URI},
        )
        expected_inputs = tuple(
            sorted((*plan.source_artifacts, *plan.science_artifacts))
        )
        self.assertEqual(tuple(plan.job_spec.inputs), expected_inputs)
        self.assertEqual(
            tuple(plan.job_spec.producer_source_hashes),
            tuple(
                sorted({artifact.sha256 for artifact in plan.source_artifacts})
            ),
        )
        parameters = plan.job_spec.as_dict()["parameters"]
        self.assertEqual(len(parameters["observerFrequencyBinsHz"]), 471)
        self.assertEqual(parameters["samplerDescriptor"], descriptor)
        json.dumps(parameters, allow_nan=False, sort_keys=True)

    def test_scientific_changes_alter_job_identity_but_scheduling_does_not(self) -> None:
        baseline = self.build_plan()
        changed_dotm = self.build_plan("--dotm", "0.045")
        changed_outer = self.build_plan("--rout-over-mass", "24")
        changed_observer = self.build_plan("--inclination-deg", "57")
        changed_termination = self.build_plan(
            "--escape-radius-over-mass",
            "55",
        )
        changed_tolerance = self.build_plan(
            "--specific-intensity-relative-tolerance",
            "3e-4",
        )
        changed_jobs = self.build_plan(
            "--jobs",
            "2",
            "--max-in-flight",
            "2",
        )
        for changed in (
            changed_dotm,
            changed_outer,
            changed_observer,
            changed_termination,
            changed_tolerance,
        ):
            self.assertNotEqual(baseline.job_spec.job_key, changed.job_spec.job_key)
        self.assertEqual(baseline.job_spec.job_key, changed_jobs.job_spec.job_key)

    def test_descriptor_and_jobspec_tampering_fail_before_ray_sampling(self) -> None:
        plan = self.build_plan()
        parameters = plan.job_spec.as_dict()["parameters"]
        tampered_spec = replace(
            plan.job_spec,
            parameters={**parameters, "tampered": True},
        )
        with (
            patch.object(
                KerrFiniteThicknessRaySampler,
                "sample",
                side_effect=AssertionError("tamper must fail before tracing"),
            ),
            self.assertRaisesRegex(
                SpectralProductError,
                "does not bind the producer configuration",
            ),
        ):
            plan.producer.inner(tampered_spec, tampered_spec.tasks[0])

        object.__setattr__(
            plan.sampler,
            "specific_intensity_relative_tolerance",
            1.5e-4,
        )
        with (
            patch.object(
                KerrFiniteThicknessRaySampler,
                "sample",
                side_effect=AssertionError("tamper must fail before tracing"),
            ),
            self.assertRaisesRegex(
                SpectralProductError,
                "sampler descriptor changed",
            ),
        ):
            plan.producer.inner(plan.job_spec, plan.job_spec.tasks[0])

    def test_cie_tampering_fails_authentication_before_sampler_construction(self) -> None:
        csv_path = self.root / "CIE.csv"
        metadata_path = self.root / "CIE.metadata.json"
        shutil.copyfile(renderer.DEFAULT_CIE_CSV, csv_path)
        shutil.copyfile(renderer.DEFAULT_CIE_METADATA, metadata_path)
        payload = bytearray(csv_path.read_bytes())
        payload[-2] = ord("1") if payload[-2] != ord("1") else ord("2")
        csv_path.write_bytes(payload)

        with patch.object(
            KerrFiniteThicknessRaySampler,
            "sample",
            side_effect=AssertionError("bad CIE bytes must not trace rays"),
        ):
            with self.assertRaisesRegex(Exception, "SHA-256 mismatch"):
                self.build_plan(
                    "--cie-csv",
                    str(csv_path),
                    "--cie-metadata",
                    str(metadata_path),
                )

    def test_tile_cache_write_is_guarded_by_code_and_cie_snapshot_checks(self) -> None:
        plan = self.build_plan()
        with (
            patch.object(renderer, "assert_bound_inputs_stable") as stable,
            patch.object(
                renderer.AdaptiveSpectralTileProducer,
                "__call__",
                return_value=b"authenticated-finite-tile",
            ) as inner,
        ):
            payload = plan.producer(plan.job_spec, plan.job_spec.tasks[0])
        self.assertEqual(payload, b"authenticated-finite-tile")
        self.assertEqual(stable.call_count, 2)
        inner.assert_called_once_with(plan.job_spec, plan.job_spec.tasks[0])

    def test_main_wires_resume_transactional_publish_and_strict_verifier(self) -> None:
        publication = SpectralProductPublication(
            output_directory=self.output.absolute(),
            manifest_path=self.output.absolute() / "manifest.json",
            manifest_sha256="a" * 64,
            product_id="scientific-spectral-frame-finite-test",
            product_sha256="b" * 64,
            tile_count=2,
            record_count=2,
        )
        verification = {
            "status": STRUCTURAL_STATUS,
            "physicsVerified": False,
        }
        observed: dict[str, object] = {}

        def fake_run_job(spec, producer, cache_root, **keywords):
            observed["spec"] = spec
            observed["producer"] = producer
            observed["cache"] = cache_root
            observed["run_keywords"] = keywords
            return JobRun(
                job_key=spec.job_key,
                results=(),
                reused_tasks=2,
                executed_tasks=0,
                max_in_flight_observed=0,
            )

        def fake_publish(output, **keywords):
            observed["output"] = output
            observed["publish_keywords"] = keywords
            return publication

        with (
            patch.object(renderer, "assert_bound_inputs_stable"),
            patch.object(renderer, "run_job", side_effect=fake_run_job) as run,
            patch.object(
                renderer,
                "publish_spectral_product",
                side_effect=fake_publish,
            ) as publish,
            patch.object(
                renderer,
                "validate_scientific_spectral_frame",
                return_value=verification,
            ) as verify,
            patch.object(
                KerrFiniteThicknessRaySampler,
                "sample",
                side_effect=AssertionError("wiring test must not trace rays"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            result = renderer.main(
                minimal_arguments(
                    self.output,
                    self.cache,
                    "--jobs",
                    "2",
                    "--max-in-flight",
                    "2",
                )
            )

        self.assertEqual(result, 0)
        run.assert_called_once()
        publish.assert_called_once()
        verify.assert_called_once_with(
            publication.manifest_path,
            renderer.DEFAULT_SPECTRAL_SCHEMA.absolute(),
        )
        self.assertEqual(observed["cache"], self.cache.absolute())
        self.assertEqual(
            observed["run_keywords"],
            {"jobs": 2, "max_in_flight": 2},
        )
        self.assertEqual(observed["output"], self.output.absolute())
        publish_keywords = observed["publish_keywords"]
        self.assertIs(publish_keywords["job_spec"], observed["spec"])
        self.assertEqual(
            publish_keywords["sampler_descriptor"],
            observed["spec"].as_dict()["parameters"]["samplerDescriptor"],
        )

    def test_existing_output_is_rejected_before_job_execution(self) -> None:
        self.output.mkdir()
        stderr = io.StringIO()
        with (
            patch.object(renderer, "run_job") as run,
            redirect_stderr(stderr),
        ):
            result = renderer.main(minimal_arguments(self.output, self.cache))
        self.assertEqual(result, 1)
        self.assertIn("refusing to overwrite", stderr.getvalue())
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
