from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from offline.disk_atmosphere import FluxConservingLinearLimbDarkening
from offline.job import JobRun
from offline.kerr_disk_frame import KerrDiskRaySampler
from offline.spectral_product import SpectralProductPublication
import scripts.render_offline_kerr_nt_frame as renderer


ROOT = Path(__file__).resolve().parents[1]


def minimal_arguments(
    output: Path,
    cache: Path,
    *extra: str,
) -> list[str]:
    return [
        str(output),
        "--cache",
        str(cache),
        "--frequency-hz",
        "5e14",
        "--width",
        "2",
        "--height",
        "1",
        "--tile-width",
        "1",
        "--tile-height",
        "1",
        *extra,
    ]


class RenderOfflineKerrNtFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "product"
        self.cache = self.root / "cache"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build_plan(self, *extra: str) -> renderer.KerrNtFramePlan:
        arguments = renderer.parse_args(
            minimal_arguments(self.output, self.cache, *extra)
        )
        return renderer.build_render_plan(arguments)

    def test_help_is_executable_without_starting_a_render(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/render_offline_kerr_nt_frame.py",
                "--help",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--frequency-hz", completed.stdout)
        self.assertIn("--inclination-deg", completed.stdout)
        self.assertIn("--surface-subdivisions-per-segment", completed.stdout)
        self.assertIn("--maximum-depth", completed.stdout)

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

    def test_plan_binds_exact_kerr_nt_d20_and_complete_source_identity(
        self,
    ) -> None:
        with patch.object(
            KerrDiskRaySampler,
            "sample",
            side_effect=AssertionError("configuration must not trace rays"),
        ):
            plan = self.build_plan(
                "--spin",
                "0.82",
                "--black-hole-mass-solar",
                "4000000",
                "--accretion-rate-kg-s",
                "2.5e20",
                "--inclination-deg",
                "55",
                "--ray-relative-tolerance",
                "4e-10",
                "--maximum-depth",
                "0",
            )

        self.assertEqual(plan.layout.observer_frequencies_hz, (5.0e14,))
        self.assertEqual((plan.grid.width_pixels, plan.grid.height_pixels), (2, 1))
        self.assertEqual(len(plan.job_spec.tasks), 2)
        self.assertEqual(plan.sampler.metric.dimensionless_spin, 0.82)
        self.assertEqual(plan.sampler.observer_theta_rad, math.radians(55.0))
        self.assertNotAlmostEqual(
            plan.sampler.observer_theta_rad,
            0.5 * math.pi,
        )
        self.assertIsInstance(
            plan.sampler.angular_emission_law,
            FluxConservingLinearLimbDarkening,
        )
        self.assertEqual(plan.sampler.angular_emission_law.coefficient, 1.5)
        self.assertEqual(plan.sampler.fine_options.relative_tolerance, 4.0e-10)
        self.assertTrue(plan.sampler.fine_options.record_path)
        self.assertIs(plan.producer.inner.sampler, plan.sampler)
        self.assertEqual(plan.producer.source_artifacts, plan.source_artifacts)

        expected_uris = {
            f"repo-source://{path.as_posix()}"
            for path in renderer.PRODUCER_SOURCE_FILES
        }
        self.assertEqual(
            {artifact.uri for artifact in plan.source_artifacts},
            expected_uris,
        )
        self.assertIn(
            "repo-source://offline/kerr_disk_early_stop.py",
            expected_uris,
        )
        self.assertEqual(
            tuple(plan.job_spec.inputs),
            tuple(sorted(plan.source_artifacts)),
        )
        self.assertEqual(
            tuple(plan.job_spec.producer_source_hashes),
            tuple(
                sorted(
                    {artifact.sha256 for artifact in plan.source_artifacts}
                )
            ),
        )
        sampler_descriptor = plan.job_spec.as_dict()["parameters"][
            "samplerDescriptor"
        ]
        self.assertRegex(
            sampler_descriptor["implementationId"],
            r"^exact-kerr-nt-spectral-ray-sampler/v[1-9][0-9]*$",
        )
        self.assertEqual(
            sampler_descriptor["angularEmission"]["descriptor"]["kind"],
            "linear-electron-scattering-proxy",
        )

    def test_scientific_parameters_change_identity_but_jobs_do_not(self) -> None:
        baseline = self.build_plan("--maximum-depth", "0")
        self.assertEqual(
            baseline.sampler.observer_theta_rad,
            math.radians(60.0),
        )
        changed_spin = self.build_plan(
            "--maximum-depth",
            "0",
            "--spin",
            "0.5",
        )
        changed_tolerance = self.build_plan(
            "--maximum-depth",
            "0",
            "--radiance-relative-tolerance",
            "2e-3",
        )
        changed_jobs = self.build_plan(
            "--maximum-depth",
            "0",
            "--jobs",
            "2",
            "--max-in-flight",
            "3",
        )

        self.assertNotEqual(baseline.job_spec.job_key, changed_spin.job_spec.job_key)
        self.assertNotEqual(
            baseline.job_spec.job_key,
            changed_tolerance.job_spec.job_key,
        )
        self.assertEqual(baseline.job_spec.job_key, changed_jobs.job_spec.job_key)

    def test_tile_cache_write_is_guarded_by_source_checks(self) -> None:
        plan = self.build_plan("--maximum-depth", "0")
        with (
            patch.object(renderer, "assert_source_snapshot_stable") as stable,
            patch.object(
                renderer.AdaptiveSpectralTileProducer,
                "__call__",
                return_value=b"authenticated-tile",
            ) as inner,
        ):
            payload = plan.producer(plan.job_spec, plan.job_spec.tasks[0])

        self.assertEqual(payload, b"authenticated-tile")
        self.assertEqual(stable.call_count, 2)
        inner.assert_called_once_with(plan.job_spec, plan.job_spec.tasks[0])

    def test_main_wires_resume_and_publication_without_tracing(self) -> None:
        publication = SpectralProductPublication(
            output_directory=self.output.absolute(),
            manifest_path=self.output.absolute() / "manifest.json",
            manifest_sha256="a" * 64,
            product_id="scientific-spectral-frame-test",
            product_sha256="b" * 64,
            tile_count=2,
            record_count=2,
        )
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
            patch.object(
                KerrDiskRaySampler,
                "sample",
                side_effect=AssertionError("main wiring must not trace in this test"),
            ),
            patch.object(renderer, "run_job", side_effect=fake_run_job) as run,
            patch.object(
                renderer,
                "publish_spectral_product",
                side_effect=fake_publish,
            ) as publish,
            redirect_stdout(io.StringIO()),
        ):
            result = renderer.main(
                minimal_arguments(
                    self.output,
                    self.cache,
                    "--jobs",
                    "2",
                    "--max-in-flight",
                    "3",
                    "--maximum-depth",
                    "0",
                )
            )

        self.assertEqual(result, 0)
        run.assert_called_once()
        publish.assert_called_once()
        self.assertEqual(observed["cache"], self.cache.absolute())
        self.assertEqual(
            observed["run_keywords"],
            {"jobs": 2, "max_in_flight": 3},
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
            result = renderer.main(
                minimal_arguments(
                    self.output,
                    self.cache,
                    "--maximum-depth",
                    "0",
                )
            )
        self.assertEqual(result, 1)
        self.assertIn("refusing to overwrite", stderr.getvalue())
        run.assert_not_called()

    def test_exact_edge_on_configuration_fails_closed_without_tracing(self) -> None:
        with patch.object(
            KerrDiskRaySampler,
            "sample",
            side_effect=AssertionError("invalid configuration must not trace"),
        ):
            with self.assertRaisesRegex(ValueError, "exactly edge-on"):
                self.build_plan("--inclination-deg", "90")


if __name__ == "__main__":
    unittest.main()
