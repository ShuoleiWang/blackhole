from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import offline.kerr_returning_radiation_kernel as forward_module
from offline.kerr_returning_radiation_refinement_checkpoint import (
    KerrReturningRadiationRefinementPublication,
)
import scripts.run_offline_kerr_returning_radiation_refinement as refinement_cli

from tests.test_offline_kerr_returning_radiation_refinement_checkpoint import (
    production_forward_classifier,
)


ROOT = Path(__file__).resolve().parents[1]


def minimal_cli_arguments(output: Path, cache: Path, *extra: str) -> list[str]:
    return [
        str(output),
        "--kernel-cache",
        str(cache),
        "--outer-radius-over-mass",
        "8",
        "--height-accretion-rate-eddington",
        "0.08",
        "--escape-radius-over-mass",
        "20",
        "--kernel-rho-order",
        "4",
        "--kernel-mu-order",
        "4",
        "--kernel-psi-count",
        "4",
        "--kernel-absolute-tolerance",
        "0.25",
        "--kernel-relative-tolerance",
        "0.25",
        "--kernel-symmetry-absolute-tolerance",
        "0.25",
        "--kernel-symmetry-relative-tolerance",
        "0.25",
        "--kernel-maximum-direction-evaluations",
        "10000",
        "--kernel-maximum-whole-ray-traces",
        "40000",
        "--kernel-directions-per-task",
        "16",
        "--area-relative-tolerance",
        "1e-8",
        "--area-absolute-tolerance-over-mass-squared",
        "1e-8",
        *extra,
    ]


class RunOfflineKerrReturningRadiationRefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve(strict=True)
        self.output = self.root / "checkpoint"
        self.cache = self.root / "cache"

    def test_help_exposes_absolute_separate_paths_and_kernel_only_scope(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_offline_kerr_returning_radiation_refinement.py",
                "--help",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        normalized_help = " ".join(completed.stdout.split())
        for text in (
            "new absolute checkpoint directory",
            "separate absolute resumable production-forward cache root",
            "--kernel-cache",
            "--annulus-count",
            "--annulus-edges-over-mass",
            "--kernel-rho-order",
            "--kernel-mu-order",
            "--kernel-psi-count",
            "--kernel-directions-per-task",
        ):
            self.assertIn(text, normalized_help)
        self.assertIn("kernel-only", normalized_help)

    def test_build_plan_is_trace_and_write_free_with_expected_defaults(self) -> None:
        arguments = refinement_cli.parse_args(
            [str(self.output), "--kernel-cache", str(self.cache)]
        )
        with (
            patch.object(
                forward_module,
                "_trace_direction",
                side_effect=AssertionError("planning traced a direction"),
            ) as tracer,
            patch.object(
                refinement_cli,
                "_EXECUTE_REFINEMENT_CHECKPOINT_CALL_ENTRY",
                side_effect=AssertionError("planning executed the checkpoint"),
            ) as execute,
        ):
            plan = refinement_cli.build_refinement_plan(arguments)

        tracer.assert_not_called()
        execute.assert_not_called()
        self.assertEqual(plan.output_directory, self.output)
        self.assertEqual(plan.cache_root, self.cache)
        self.assertEqual(len(plan.annulus_edges_over_mass), 2)
        self.assertEqual(
            (
                plan.kernel_policy.rho_order,
                plan.kernel_policy.mu_order,
                plan.kernel_policy.psi_count,
                plan.directions_per_task,
                plan.jobs,
                plan.max_in_flight,
            ),
            (8, 16, 32, 64, 1, None),
        )
        self.assertEqual(
            plan.convergence_policy.maximum_normalized_sample_weight,
            1.0e-3,
        )
        self.assertFalse(self.output.exists())
        self.assertFalse(self.cache.exists())

    def test_main_returns_two_after_real_nonqualified_publish(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(
                forward_module,
                "_trace_direction",
                side_effect=production_forward_classifier,
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            result = refinement_cli.main(
                minimal_cli_arguments(self.output, self.cache)
            )

        self.assertEqual(result, 2, stderr.getvalue())
        self.assertEqual(stderr.getvalue(), "")
        self.assertIn("checkpoint published", stdout.getvalue())
        self.assertIn("v2 qualified = False", stdout.getvalue())
        self.assertTrue((self.output / "manifest.json").is_file())
        self.assertTrue((self.output / "manifest.sha256").is_file())

    def test_main_maps_qualified_publication_to_zero_and_failure_to_one(self) -> None:
        publication = KerrReturningRadiationRefinementPublication(
            output_directory=self.output,
            manifest_path=self.output / "manifest.json",
            manifest_sha256="a" * 64,
            checkpoint_id="kerr-returning-radiation-refinement-test",
            qualified=True,
        )
        stdout = io.StringIO()
        with (
            patch.object(refinement_cli, "parse_args", return_value=object()),
            patch.object(refinement_cli, "build_refinement_plan", return_value=object()),
            patch.object(
                refinement_cli,
                "_EXECUTE_REFINEMENT_CHECKPOINT_CALL_ENTRY",
                return_value=publication,
            ) as execute,
            redirect_stdout(stdout),
        ):
            self.assertEqual(refinement_cli.main([]), 0)
        execute.assert_called_once()
        self.assertIn("v2 qualified = True", stdout.getvalue())

        stderr = io.StringIO()
        with (
            patch.object(
                refinement_cli,
                "parse_args",
                side_effect=RuntimeError("injected execution failure"),
            ),
            redirect_stderr(stderr),
        ):
            self.assertEqual(refinement_cli.main([]), 1)
        self.assertIn("injected execution failure", stderr.getvalue())

    def test_public_execute_rebindings_fail_before_parse_build_or_cache(self) -> None:
        for label, owner in (
            ("cli-alias", refinement_cli),
            ("checkpoint-public", refinement_cli._checkpoint),
        ):
            with (
                self.subTest(label=label),
                patch.object(
                    owner,
                    "execute_kerr_returning_radiation_refinement_checkpoint",
                    return_value=object(),
                ) as replacement,
                patch.object(refinement_cli, "parse_args") as parse,
                patch.object(refinement_cli, "build_refinement_plan") as build,
                patch.object(
                    refinement_cli._checkpoint,
                    "_CACHED_INTEGRATOR_CALL_ENTRY",
                ) as cache,
                redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(refinement_cli.main([]), 1)
            replacement.assert_not_called()
            parse.assert_not_called()
            build.assert_not_called()
            cache.assert_not_called()
            self.assertRegex(stderr.getvalue(), "binding changed|foreign callable")

    def test_cli_rejects_persistent_preimport_checkpoint_execute_poison(self) -> None:
        probe = """
import json
import pathlib
import sys

root = pathlib.Path.cwd()
sys.path.insert(0, str(root))
import offline.kerr_returning_radiation_refinement_checkpoint as checkpoint

calls = 0
def poisoned(*args, **kwargs):
    global calls
    calls += 1
    raise AssertionError("poisoned checkpoint execute was called")

canonical = checkpoint._EXECUTE_REFINEMENT_CHECKPOINT_CANONICAL_ENTRY
checkpoint.execute_kerr_returning_radiation_refinement_checkpoint = poisoned
import scripts.run_offline_kerr_returning_radiation_refinement as cli
status = cli.main([])
print(json.dumps(dict(
    calls=calls,
    canonicalIsPoisoned=(canonical is poisoned),
    checkpointPublicIsPoisoned=(
        checkpoint.execute_kerr_returning_radiation_refinement_checkpoint
        is poisoned
    ),
    cliAliasIsPoisoned=(
        cli.execute_kerr_returning_radiation_refinement_checkpoint is poisoned
    ),
    status=status,
)))
"""
        completed = subprocess.run(
            (sys.executable, "-I", "-B", "-c", probe),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], 1)
        self.assertEqual(result["calls"], 0)
        self.assertFalse(result["canonicalIsPoisoned"])
        self.assertTrue(result["checkpointPublicIsPoisoned"])
        self.assertTrue(result["cliAliasIsPoisoned"])
        self.assertRegex(
            completed.stderr,
            "binding changed|not executed from the trusted source path",
        )

    def test_cli_origin_gate_rejects_foreign_checkpoint_before_build_or_execute(
        self,
    ) -> None:
        with (
            patch.object(
                refinement_cli._checkpoint,
                "__file__",
                str(self.root / "shadow" / "refinement_checkpoint.py"),
            ),
            patch.object(refinement_cli, "parse_args") as parse,
            patch.object(refinement_cli, "build_refinement_plan") as build,
            patch.object(
                refinement_cli,
                "execute_kerr_returning_radiation_refinement_checkpoint",
            ) as execute,
            redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(refinement_cli.main([]), 1)
        parse.assert_not_called()
        build.assert_not_called()
        execute.assert_not_called()
        self.assertIn("exact renderer ROOT", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
