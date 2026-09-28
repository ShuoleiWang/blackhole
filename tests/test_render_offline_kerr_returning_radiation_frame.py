from __future__ import annotations

import ast
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import ModuleType, SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import offline.authenticated_artifact as artifact_module
from offline.adaptive_frame import (
    RayConvergenceAudit,
    SpectralRaySample,
    integrate_spectral_pixel,
)
from offline.job import InputArtifact, JobRun, TaskResult
from offline.kerr_finite_thickness_frame import KerrFiniteThicknessRaySampler
from offline.kerr_returning_radiation_finite_thickness_frame import (
    IMPLEMENTATION_ID as SAMPLER_IMPLEMENTATION_ID,
    KerrReturningRadiationFiniteThicknessRaySampler,
)
import offline.kerr_returning_radiation_finite_thickness_frame as returning_frame_module
from offline.kerr_returning_radiation_kernel_cached import (
    KerrCachedKernelExecutionAudit,
    KerrCachedReturningRadiationKernelExecution,
)
from offline.kerr_returning_radiation_live_replay_attestation import (
    ReturningRadiationLiveReplayAttestationPublication,
)
from offline.kerr_returning_radiation_refinement_checkpoint import (
    KerrReturningRadiationVerifiedRefinementCheckpoint,
)
import offline.kerr_returning_radiation_spectral_product as product_module
from offline.kerr_returning_radiation_spectral_product import (
    ReturningRadiationAdaptiveSpectralTileProducer,
    build_returning_radiation_spectral_job_spec,
)
from offline.spectral_product import SpectralProductPublication
import scripts.render_offline_kerr_returning_radiation_frame as renderer


ROOT = Path(__file__).resolve().parents[1]
STRUCTURAL_STATUS = (
    "scientific-spectral-frame-structural-contract-conformant"
)


def minimal_arguments(
    output: Path,
    tile_cache: Path,
    kernel_cache: Path,
    *extra: str,
) -> list[str]:
    return [
        str(output),
        "--cache",
        str(tile_cache),
        "--kernel-cache",
        str(kernel_cache),
        "--required-v2-checkpoint",
        str(output.parent / "qualified-checkpoint" / "manifest.json"),
        "--required-v2-checkpoint-sha256",
        "a" * 64,
        "--width",
        "1",
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


def returning_descriptor() -> dict[str, object]:
    return {
        "implementationId": SAMPLER_IMPLEMENTATION_ID,
        "authority": {
            "sourceAuthentication": {
                "productionForwardOnly": True,
                "scientificBindingSha256": "a" * 64,
                "excludedOperationalIdentity": {
                    "cacheAbsolutePaths": True,
                    "executedOrReusedCounters": True,
                    "reuseHistory": True,
                    "workerScheduling": True,
                },
            }
        },
        "scientificStatus": {
            "classification": "same-code finite-grid piecewise-annulus test",
            "hasIndependentPhysicsOracle": False,
            "includesReturningRadiationStressWorkFS": False,
            "includesScatteringOrSolvedAtmosphere": False,
            "isGeneralRelativisticMagnetohydrodynamics": False,
        },
    }


def live_replay_report(
    publication: SpectralProductPublication,
    *,
    ray_samples: int = 13,
) -> dict[str, object]:
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
        "totalRaySamples": ray_samples,
        "totalFrameGeodesicsReplayed": 2 * ray_samples,
        "replayScope": "same-code production whole-pixel byte replay",
        "scientificScope": "finite-grid returning-thermal frame geodesics",
        "sourceHashScope": "exact declared returning producer closure",
    }


def live_replay_attestation(
    plan: object,
    publication: SpectralProductPublication,
) -> ReturningRadiationLiveReplayAttestationPublication:
    directory = plan.live_replay_attestation_directory
    return ReturningRadiationLiveReplayAttestationPublication(
        output_directory=directory,
        manifest_path=directory / "manifest.json",
        manifest_sha256="f" * 64,
        attestation_id="returning-radiation-live-replay-attestation-" + "1" * 24,
        spectral_manifest_sha256=publication.manifest_sha256,
        live_replay_report=live_replay_report(publication),
    )


class _ConstantDiskSampler:
    def sample(self, screen_x, screen_y, observer_frequencies_hz):
        del screen_x, screen_y
        return SpectralRaySample(
            specific_intensities_nu=(1.0,) * len(observer_frequencies_hz),
            absolute_errors_nu=(0.0,) * len(observer_frequencies_hz),
            visible_source="disk",
            topology_signature="returning-test",
            frequency_shift_g=1.0,
            ray_converged=True,
            convergence_audit=RayConvergenceAudit(
                accepted_steps=1,
                ray_gate_passed=True,
                source_gate_passed=True,
                transfer_gate_passed=True,
            ),
        )


class _Authority:
    def __init__(self) -> None:
        self.snapshot = object()
        self.live_calls = 0

    def require_live(self):
        self.live_calls += 1
        return self.snapshot


class RenderOfflineKerrReturningRadiationFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve(strict=True)
        self.output = self.root / "product"
        self.tile_cache = self.root / "tile-cache"
        self.kernel_cache = self.root / "kernel-cache"
        self.checkpoint_manifest = (
            self.root / "qualified-checkpoint" / "manifest.json"
        )
        self.checkpoint_sha256 = "a" * 64
        self.real_checkpoint_binder = (
            renderer._bind_qualified_v2_checkpoint_to_kernel
        )

        def verify_checkpoint(
            manifest_path,
            *,
            expected_manifest_sha256,
        ):
            return KerrReturningRadiationVerifiedRefinementCheckpoint(
                manifest_path.parent,
                manifest_path,
                expected_manifest_sha256,
                "kerr-returning-radiation-refinement-" + "1" * 24,
                True,
                {"qualified": True},
            )

        def bind_checkpoint(verified, definition, reduction_configuration):
            return renderer._QualifiedV2CheckpointBinding(
                verified.checkpoint_id,
                InputArtifact(
                    (
                        "urn:blackhole:kerr-returning-radiation-refinement-"
                        f"checkpoint:v1:{verified.checkpoint_id}"
                    ),
                    1,
                    verified.manifest_sha256,
                ),
                "d" * 64,
                definition.scientific_job_key,
                definition.job_spec.job_key,
                renderer.canonical_json_bytes(reduction_configuration.as_dict()),
            )

        checkpoint_patch = patch.object(
            renderer,
            "_REFINEMENT_CHECKPOINT_VERIFIER_CALL_ENTRY",
            side_effect=verify_checkpoint,
        )
        self.checkpoint_verifier = checkpoint_patch.start()
        self.addCleanup(checkpoint_patch.stop)
        checkpoint_binding_patch = patch.object(
            renderer,
            "_bind_qualified_v2_checkpoint_to_kernel",
            side_effect=bind_checkpoint,
        )
        self.checkpoint_binder = checkpoint_binding_patch.start()
        self.addCleanup(checkpoint_binding_patch.stop)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build_plan(self, *extra: str) -> renderer.KerrReturningRadiationFramePlan:
        arguments = renderer.parse_args(
            minimal_arguments(
                self.output,
                self.tile_cache,
                self.kernel_cache,
                *extra,
            )
        )
        return renderer.build_render_plan(arguments)

    def checkpoint_document_for(
        self,
        definition,
        reduction_configuration,
        *,
        scientific_job_key: str | None = None,
        cache_job_key: str | None = None,
        reduction_descriptor: object | None = None,
    ) -> dict[str, object]:
        return {
            "producer": {
                "cacheDefinition": {
                    "scientificJobKey": (
                        definition.scientific_job_key
                        if scientific_job_key is None
                        else scientific_job_key
                    ),
                    "cacheJobKey": (
                        definition.job_spec.job_key
                        if cache_job_key is None
                        else cache_job_key
                    ),
                },
                "reductionConfiguration": {
                    "descriptor": (
                        reduction_configuration.as_dict()
                        if reduction_descriptor is None
                        else reduction_descriptor
                    ),
                },
            },
            "kernel": {"descriptorSha256": "d" * 64},
        }

    def verified_checkpoint_for(
        self,
        document: dict[str, object],
        *,
        qualified: bool = True,
        checkpoint_id: str = (
            "kerr-returning-radiation-refinement-" + "1" * 24
        ),
    ) -> KerrReturningRadiationVerifiedRefinementCheckpoint:
        manifest_sha256 = hashlib.sha256(
            renderer.canonical_json_bytes(document)
        ).hexdigest()
        return KerrReturningRadiationVerifiedRefinementCheckpoint(
            self.checkpoint_manifest.parent,
            self.checkpoint_manifest,
            manifest_sha256,
            checkpoint_id,
            qualified,
            document,
        )

    def test_help_exposes_separate_caches_and_all_returning_policies(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/render_offline_kerr_returning_radiation_frame.py",
                "--help",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for text in (
            "--kernel-cache",
            "--required-v2-checkpoint",
            "--required-v2-checkpoint-sha256",
            "--annulus-count",
            "--annulus-edges-over-mass",
            "--kernel-rho-order",
            "--kernel-mu-order",
            "--kernel-psi-count",
            "--area-gauss-legendre-order",
            "--fixed-point-maximum-iterations",
            "--annulus-edge-clearance-multiplier",
        ):
            self.assertIn(text, completed.stdout)
        self.assertRegex(completed.stdout, r"same-\s*code")
        self.assertIn("piecewise-constant-annulus", completed.stdout)

    def test_required_checkpoint_forwards_external_digest_exactly(self) -> None:
        verified = KerrReturningRadiationVerifiedRefinementCheckpoint(
            self.checkpoint_manifest.parent,
            self.checkpoint_manifest,
            self.checkpoint_sha256,
            "kerr-returning-radiation-refinement-" + "1" * 24,
            True,
            {},
        )
        with patch.object(
            renderer,
            "_REFINEMENT_CHECKPOINT_VERIFIER_CALL_ENTRY",
            return_value=verified,
        ) as verifier:
            actual = renderer._require_qualified_v2_checkpoint(
                self.checkpoint_manifest,
                self.checkpoint_sha256,
            )
        self.assertIs(actual, verified)
        verifier.assert_called_once_with(
            self.checkpoint_manifest,
            expected_manifest_sha256=self.checkpoint_sha256,
        )

    def test_bad_required_checkpoint_fails_build_before_product_work(self) -> None:
        exact = KerrReturningRadiationVerifiedRefinementCheckpoint(
            self.checkpoint_manifest.parent,
            self.checkpoint_manifest,
            self.checkpoint_sha256,
            "kerr-returning-radiation-refinement-" + "1" * 24,
            True,
            {},
        )
        cases = (
            (
                "non-qualified",
                replace(exact, qualified=False),
                RuntimeError,
                "non-qualified",
            ),
            (
                "foreign-result",
                SimpleNamespace(
                    output_directory=exact.output_directory,
                    manifest_path=exact.manifest_path,
                    manifest_sha256=exact.manifest_sha256,
                    checkpoint_id=exact.checkpoint_id,
                    qualified=True,
                    document={},
                ),
                TypeError,
                "foreign type",
            ),
            (
                "mismatched-identity",
                replace(exact, manifest_sha256="b" * 64),
                RuntimeError,
                "mismatched identity",
            ),
        )
        guarded = (
            "_current_numeric_backend_bytes",
            "_preflight_product_paths",
            "collect_source_artifacts",
            "collect_science_artifacts",
            "_CACHED_KERNEL_INTEGRATOR_CALL_ENTRY",
        )
        for label, result, error_type, message in cases:
            with self.subTest(case=label):
                arguments = renderer.parse_args(
                    minimal_arguments(
                        self.output,
                        self.tile_cache,
                        self.kernel_cache,
                    )
                )
                with ExitStack() as stack:
                    verifier = stack.enter_context(
                        patch.object(
                            renderer,
                            "_REFINEMENT_CHECKPOINT_VERIFIER_CALL_ENTRY",
                            return_value=result,
                        )
                    )
                    work = [
                        stack.enter_context(
                            patch.object(
                                renderer,
                                name,
                                side_effect=AssertionError(
                                    f"{name} ran before checkpoint rejection"
                                ),
                            )
                        )
                        for name in guarded
                    ]
                    base = stack.enter_context(
                        patch.object(
                            renderer.base_renderer,
                            "build_render_plan",
                            side_effect=AssertionError(
                                "base planning ran before checkpoint rejection"
                            ),
                        )
                    )
                    ray = stack.enter_context(
                        patch.object(
                            KerrFiniteThicknessRaySampler,
                            "sample",
                            side_effect=AssertionError(
                                "camera ray ran before checkpoint rejection"
                            ),
                        )
                    )
                    with self.assertRaisesRegex(error_type, message):
                        renderer.build_render_plan(arguments)
                verifier.assert_called_once_with(
                    self.checkpoint_manifest,
                    expected_manifest_sha256=self.checkpoint_sha256,
                )
                for mocked in (*work, base, ray):
                    mocked.assert_not_called()

    def test_real_checkpoint_binding_matches_only_exact_kernel_identity(self) -> None:
        plan = self.build_plan()
        definition = plan.kernel_cache_definition
        reduction = plan.kernel_reduction_configuration
        document = self.checkpoint_document_for(definition, reduction)
        verified = self.verified_checkpoint_for(document)

        binding = self.real_checkpoint_binder(
            verified,
            definition,
            reduction,
        )
        self.assertEqual(binding.checkpoint_id, verified.checkpoint_id)
        self.assertEqual(binding.input_artifact.sha256, verified.manifest_sha256)
        self.assertTrue(binding.input_artifact.uri.endswith(verified.checkpoint_id))
        self.assertEqual(
            binding.scientific_job_key,
            definition.scientific_job_key,
        )
        self.assertEqual(binding.cache_job_key, definition.job_spec.job_key)
        self.assertEqual(
            binding.reduction_configuration_bytes,
            renderer.canonical_json_bytes(reduction.as_dict()),
        )

        mismatches = (
            (
                "scientific job",
                self.checkpoint_document_for(
                    definition,
                    reduction,
                    scientific_job_key="e" * 64,
                ),
            ),
            (
                "cache job",
                self.checkpoint_document_for(
                    definition,
                    reduction,
                    cache_job_key="e" * 64,
                ),
            ),
            (
                "reduction",
                self.checkpoint_document_for(
                    definition,
                    reduction,
                    reduction_descriptor={"schema": "changed"},
                ),
            ),
        )
        for label, changed_document in mismatches:
            with self.subTest(binding=label), self.assertRaisesRegex(
                RuntimeError,
                "does not bind the product kernel definition",
            ):
                self.real_checkpoint_binder(
                    self.verified_checkpoint_for(changed_document),
                    definition,
                    reduction,
                )

    def test_declared_sources_cover_transitive_repo_imports(self) -> None:
        declared = set(renderer.PRODUCER_SOURCE_FILES)
        imported: set[Path] = set()
        for relative in declared:
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            module_names: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module is not None:
                    module_names.append(node.module)
                elif isinstance(node, ast.Import):
                    module_names.extend(alias.name for alias in node.names)
            for module_name in module_names:
                if not (
                    module_name.startswith("offline.")
                    or module_name.startswith("scripts.")
                ):
                    continue
                candidate = Path(*module_name.split(".")).with_suffix(".py")
                if (ROOT / candidate).is_file():
                    imported.add(candidate)
        self.assertEqual(imported - declared, set())
        for required in (
            Path("offline/kerr_returning_radiation_convergence_v2.py"),
            Path("offline/kerr_returning_radiation_kernel_cached.py"),
            Path("offline/kerr_returning_radiation_refinement_checkpoint.py"),
            Path("offline/kerr_returning_radiation_frame_context.py"),
            Path("offline/kerr_returning_radiation_finite_thickness_frame.py"),
            Path("offline/kerr_returning_radiation_spectral_product.py"),
            Path("offline/kerr_returning_radiation_spectral_replay.py"),
            Path("scripts/render_offline_kerr_returning_radiation_frame.py"),
        ):
            self.assertIn(required, declared)

    def test_declared_python_sources_map_to_48_exact_loaded_origins(self) -> None:
        renderer._frozen_live_replay_entry()
        self.assertEqual(len(renderer.PRODUCER_SOURCE_MODULES), 48)
        self.assertEqual(
            renderer.PRODUCER_SOURCE_MODULES[0],
            ("offline", Path("offline/__init__.py")),
        )
        self.assertIn(
            (
                "scripts.render_offline_kerr_returning_radiation_frame",
                Path("scripts/render_offline_kerr_returning_radiation_frame.py"),
            ),
            renderer.PRODUCER_SOURCE_MODULES,
        )
        renderer._assert_source_provenance(ROOT)
        for module_name, relative in renderer.PRODUCER_SOURCE_MODULES:
            module = sys.modules[module_name]
            self.assertIs(type(module), ModuleType)
            self.assertIs(type(module.__file__), str)
            self.assertEqual(Path(module.__file__), ROOT / relative)

    def test_clean_renderer_import_has_no_undeclared_loaded_offline_module(self) -> None:
        probe = r'''
from pathlib import Path
import sys
import scripts.render_offline_kerr_returning_radiation_frame as renderer

renderer._frozen_live_replay_entry()
root = renderer.ROOT
loaded = set()
for name, module in tuple(sys.modules.items()):
    if name != "offline" and not name.startswith("offline."):
        continue
    origin = getattr(module, "__file__", None)
    if type(origin) is not str:
        continue
    path = Path(origin)
    if path.is_relative_to(root / "offline"):
        loaded.add(path.relative_to(root))
declared = {
    relative
    for _name, relative in renderer.PRODUCER_SOURCE_MODULES
    if relative.parts[0] == "offline"
}
missing = loaded - declared
assert not missing, sorted(path.as_posix() for path in missing)
assert Path("offline/radiative_transfer.py") in declared
'''
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_executing_main_module_origin_is_bound_separately(self) -> None:
        renderer._frozen_live_replay_entry()
        direct = ModuleType("__main__")
        direct.__file__ = os.fspath(
            ROOT / "scripts/render_offline_kerr_returning_radiation_frame.py"
        )
        with patch.object(renderer, "__name__", "__main__"), patch.dict(
            sys.modules,
            {"__main__": direct},
        ):
            renderer._assert_producer_source_module_origins()
            direct.__file__ = os.fspath(
                ROOT / "scripts/render_offline_kerr_finite_thickness_frame.py"
            )
            with self.assertRaisesRegex(RuntimeError, "__main__.*exact renderer ROOT"):
                renderer._assert_producer_source_module_origins()

    def _assert_build_origin_failure_before_effectful_front_door(
        self,
        operation,
        message: str,
    ) -> None:
        arguments = renderer.parse_args(
            minimal_arguments(self.output, self.tile_cache, self.kernel_cache)
        )
        guarded = (
            "_preflight_product_paths",
            "_current_numeric_backend_bytes",
            "collect_source_artifacts",
            "collect_science_artifacts",
            "_CACHED_KERNEL_INTEGRATOR_CALL_ENTRY",
            "_THERMAL_PROFILE_SOLVER_CALL_ENTRY",
            "_THERMAL_AUTHENTICATOR_CALL_ENTRY",
        )
        with ExitStack() as stack:
            mocks = [
                stack.enter_context(
                    patch.object(
                        renderer,
                        name,
                        side_effect=AssertionError(
                            f"{name} must not run before source-origin rejection"
                        ),
                    )
                )
                for name in guarded
            ]
            base = stack.enter_context(
                patch.object(
                    renderer.base_renderer,
                    "build_render_plan",
                    side_effect=AssertionError(
                        "base planning must not run before source-origin rejection"
                    ),
                )
            )
            with self.assertRaisesRegex((TypeError, ValueError, RuntimeError), message):
                operation(arguments)
        for mocked in (*mocks, base):
            mocked.assert_not_called()

    def test_programmatic_shadow_root_and_symlink_alias_fail_at_front_door(
        self,
    ) -> None:
        shadow = self.root / "shadow"
        shadow.mkdir()
        alias = self.root / "source-alias"
        alias.symlink_to(ROOT, target_is_directory=True)
        for source_root in (shadow, alias):
            with self.subTest(source_root=source_root):
                self._assert_build_origin_failure_before_effectful_front_door(
                    lambda arguments, source_root=source_root: (
                        renderer.build_render_plan(
                            arguments,
                            source_root=source_root,
                        )
                    ),
                    "exact lexical renderer ROOT",
                )

    def test_missing_foreign_and_wrong_script_module_origins_fail_at_front_door(
        self,
    ) -> None:
        renderer._frozen_live_replay_entry()
        non_string_origin = ModuleType("offline.kerr")
        non_string_origin.__file__ = ROOT / "offline/kerr.py"
        cases = (
            (
                "missing",
                "offline.kerr",
                None,
                "missing or has a foreign type",
            ),
            (
                "foreign-type",
                "offline.kerr",
                SimpleNamespace(__file__=os.fspath(ROOT / "offline/kerr.py")),
                "missing or has a foreign type",
            ),
            (
                "non-exact-origin-type",
                "offline.kerr",
                non_string_origin,
                "non-exact file origin",
            ),
        )
        for label, module_name, replacement, message in cases:
            with self.subTest(case=label), patch.dict(
                sys.modules,
                {module_name: replacement},
            ), patch.object(
                renderer,
                "_assert_returning_renderer_runtime_bindings",
                return_value=None,
            ):
                self._assert_build_origin_failure_before_effectful_front_door(
                    renderer.build_render_plan,
                    message,
                )

        script_name = "scripts.verify_nr_contract"
        script_module = sys.modules[script_name]
        with patch.object(
            script_module,
            "__file__",
            os.fspath(ROOT / "scripts/verify_offline_spectral_frame.py"),
        ), patch.object(
            renderer,
            "_assert_returning_renderer_runtime_bindings",
            return_value=None,
        ):
            self._assert_build_origin_failure_before_effectful_front_door(
                renderer.build_render_plan,
                "scripts.verify_nr_contract.*exact renderer ROOT",
            )

    def test_preloaded_pythonpath_shadow_fails_before_effectful_front_door(
        self,
    ) -> None:
        shadow = self.root / "shadow"
        shutil.copytree(
            ROOT / "offline",
            shadow / "offline",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        probe = r"""
from pathlib import Path
from unittest.mock import patch
import offline.kerr
import scripts.render_offline_kerr_returning_radiation_frame as renderer
from tests.test_render_offline_kerr_returning_radiation_frame import minimal_arguments

root = Path.cwd()
arguments = renderer.parse_args(
    minimal_arguments(root / "out", root / "tiles", root / "kernel")
)
guarded = (
    "_preflight_product_paths",
    "_current_numeric_backend_bytes",
    "collect_source_artifacts",
    "collect_science_artifacts",
    "_CACHED_KERNEL_INTEGRATOR_CALL_ENTRY",
    "_THERMAL_PROFILE_SOLVER_CALL_ENTRY",
    "_THERMAL_AUTHENTICATOR_CALL_ENTRY",
)
mocks = []
patchers = []
try:
    for name in guarded:
        patcher = patch.object(
            renderer,
            name,
            side_effect=AssertionError(name + " ran before origin rejection"),
        )
        patchers.append(patcher)
        mocks.append(patcher.start())
    base_patcher = patch.object(
        renderer.base_renderer,
        "build_render_plan",
        side_effect=AssertionError("base planning ran before origin rejection"),
    )
    patchers.append(base_patcher)
    mocks.append(base_patcher.start())
    try:
        renderer.build_render_plan(arguments)
    except RuntimeError as error:
        assert "exact renderer ROOT" in str(error), error
    else:
        raise AssertionError("preloaded PYTHONPATH shadow was accepted")
    assert all(mock.call_count == 0 for mock in mocks), mocks
finally:
    for patcher in reversed(patchers):
        patcher.stop()
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (os.fspath(shadow), os.fspath(ROOT))
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=self.root,
            env=environment,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_runtime_binding_scope_keeps_private_call_entries_trusted(self) -> None:
        self.assertEqual(
            renderer.RUNTIME_BINDING_SCOPE,
            (
                "public-and-owning-module-callable-rebinding-is-identity-gated",
                "private-call-entry-globals-are-trusted-in-process-test-seams",
                (
                    "no-resistance-claim-for-malicious-same-process-private-"
                    "global-rewrite"
                ),
            ),
        )

    def test_build_is_trace_free_write_free_cie471_and_production_forward_only(
        self,
    ) -> None:
        with (
            patch.object(
                KerrFiniteThicknessRaySampler,
                "sample",
                side_effect=AssertionError("planning must not sample a camera ray"),
            ),
            patch.object(
                renderer,
                "_CACHED_KERNEL_INTEGRATOR_CALL_ENTRY",
                side_effect=AssertionError("planning must not run the kernel"),
            ),
            patch.object(
                renderer,
                "_ATTESTATION_PUBLISH_CALL_ENTRY",
                side_effect=AssertionError("planning must not replay or attest pixels"),
            ),
        ):
            plan = self.build_plan(
                "--annulus-count",
                "2",
                "--kernel-rho-order",
                "4",
                "--kernel-mu-order",
                "4",
                "--kernel-psi-count",
                "4",
            )
        self.assertEqual(plan.layout.frequency_count, 471)
        self.assertEqual(len(plan.annulus_edges_over_mass), 3)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.tile_cache.exists())
        self.assertFalse(self.kernel_cache.exists())
        self.assertEqual(
            plan.live_replay_attestation_directory,
            self.output.with_name(
                self.output.name + ".live-replay-attestation-v1"
            ),
        )
        self.assertFalse(plan.live_replay_attestation_directory.exists())
        evaluator = plan.kernel_cache_definition.scientific_context.inputs
        self.assertEqual(len(evaluator), 1)
        self.assertIn("_evaluate_forward_direction/v1", evaluator[0].uri)
        self.assertNotIn("synthetic", evaluator[0].uri)
        self.assertEqual(
            plan.kernel_cache_definition.scientific_context.plan.formulation,
            "forward",
        )

    def test_explicit_annulus_edges_and_scientific_policy_change_identity(self) -> None:
        baseline = self.build_plan()
        inner, outer = baseline.annulus_edges_over_mass
        midpoint = 0.5 * (inner + outer)
        explicit = self.build_plan(
            "--annulus-edges-over-mass",
            repr(inner),
            repr(midpoint),
            repr(outer),
        )
        changed_policy = self.build_plan(
            "--kernel-rho-order",
            "4",
            "--kernel-mu-order",
            "4",
            "--kernel-psi-count",
            "4",
        )
        changed_workers = self.build_plan(
            "--width",
            "2",
            "--kernel-jobs",
            "2",
            "--kernel-max-in-flight",
            "3",
            "--jobs",
            "2",
            "--max-in-flight",
            "2",
        )
        self.assertEqual(
            explicit.annulus_edges_over_mass,
            (inner, midpoint, outer),
        )
        self.assertNotEqual(
            baseline.kernel_cache_definition.scientific_job_key,
            explicit.kernel_cache_definition.scientific_job_key,
        )
        self.assertNotEqual(
            baseline.kernel_cache_definition.scientific_job_key,
            changed_policy.kernel_cache_definition.scientific_job_key,
        )
        self.assertEqual(
            baseline.kernel_cache_definition.scientific_job_key,
            changed_workers.kernel_cache_definition.scientific_job_key,
        )

    def test_tile_and_kernel_cache_must_be_separate(self) -> None:
        arguments = renderer.parse_args(
            minimal_arguments(
                self.output,
                self.tile_cache,
                self.tile_cache,
            )
        )
        with self.assertRaisesRegex(ValueError, "must be separate"):
            renderer.build_render_plan(arguments)

    def test_attestation_path_is_absent_and_non_nested_at_planning_front_door(
        self,
    ) -> None:
        attestation = renderer.default_live_replay_attestation_directory(self.output)
        attestation.mkdir()
        arguments = renderer.parse_args(
            minimal_arguments(self.output, self.tile_cache, self.kernel_cache)
        )
        with patch.object(renderer.base_renderer, "build_render_plan") as base, \
             self.assertRaisesRegex(FileExistsError, "live replay attestation"):
            renderer.build_render_plan(arguments)
        base.assert_not_called()
        with self.assertRaisesRegex(ValueError, "must be separate"):
            renderer._preflight_product_paths(
                self.output,
                self.tile_cache,
                self.kernel_cache,
                self.output / "attestation",
            )

    def test_output_and_both_caches_must_be_pairwise_non_nested(self) -> None:
        cases = (
            (
                self.output,
                self.tile_cache,
                self.tile_cache / "kernel",
            ),
            (
                self.output,
                self.kernel_cache / "tile",
                self.kernel_cache,
            ),
            (
                self.root / "tree",
                self.root / "tree" / "tile",
                self.kernel_cache,
            ),
            (
                self.root / "tree" / "output",
                self.tile_cache,
                self.root / "tree",
            ),
        )
        for output, tile_cache, kernel_cache in cases:
            with self.subTest(
                output=output,
                tile_cache=tile_cache,
                kernel_cache=kernel_cache,
            ):
                arguments = renderer.parse_args(
                    minimal_arguments(output, tile_cache, kernel_cache)
                )
                with (
                    patch.object(renderer.base_renderer, "build_render_plan") as base,
                    self.assertRaisesRegex(ValueError, "non-nested"),
                ):
                    renderer.build_render_plan(arguments)
                base.assert_not_called()

    def test_existing_symlink_in_product_path_rejected_before_base_build(self) -> None:
        real = self.root / "real-cache-parent"
        real.mkdir()
        alias = self.root / "cache-alias"
        alias.symlink_to(real, target_is_directory=True)
        arguments = renderer.parse_args(
            minimal_arguments(
                self.output,
                alias / "tiles",
                self.kernel_cache,
            )
        )
        with (
            patch.object(renderer.base_renderer, "build_render_plan") as base,
            self.assertRaisesRegex(ValueError, "symbolic-link component"),
        ):
            renderer.build_render_plan(arguments)
        base.assert_not_called()

    def test_resource_and_tile_controls_fail_before_path_or_base_io(self) -> None:
        payload_width = (
            renderer.MAXIMUM_RETURNING_TILE_PAYLOAD_BYTES
            // renderer.RETURNING_CIE_RECORD_BYTES
            + 1
        )
        cases = (
            (
                ("--width", str(renderer.MAXIMUM_RETURNING_FRAME_PIXELS + 1)),
                None,
                "frame pixel count",
            ),
            (
                ("--width", str(renderer.MAXIMUM_RETURNING_TILE_TASKS + 1)),
                None,
                "tile task count",
            ),
            (
                ("--width", str(payload_width), "--tile-width", str(payload_width)),
                None,
                "tile payload",
            ),
            (
                ("--jobs", "2"),
                None,
                r"jobs must not exceed min\(64, task_count\)",
            ),
            (
                (
                    "--width",
                    "65",
                    "--jobs",
                    "65",
                    "--max-in-flight",
                    "65",
                ),
                None,
                r"jobs must not exceed min\(64, task_count\)",
            ),
            (
                (
                    "--width",
                    "257",
                    "--max-in-flight",
                    "257",
                ),
                None,
                r"max_in_flight must not exceed min\(256, task_count\)",
            ),
            ((), ("jobs", True), "exact positive integer"),
            ((), ("max_in_flight", True), "exact positive integer"),
        )
        for extra, mutation, message in cases:
            with self.subTest(extra=extra, mutation=mutation):
                arguments = renderer.parse_args(
                    minimal_arguments(
                        self.output,
                        self.tile_cache,
                        self.kernel_cache,
                        *extra,
                    )
                )
                if mutation is not None:
                    setattr(arguments, mutation[0], mutation[1])
                with (
                    patch.object(renderer, "_preflight_product_paths") as path_io,
                    patch.object(renderer.base_renderer, "build_render_plan") as base,
                    self.assertRaisesRegex(ValueError, message),
                ):
                    renderer.build_render_plan(arguments)
                path_io.assert_not_called()
                base.assert_not_called()

    def test_cie_and_source_artifacts_bracket_base_build_exactly(self) -> None:
        events: list[str] = []
        collect_source = renderer.collect_source_artifacts
        collect_science = renderer.collect_science_artifacts
        build_base = renderer.base_renderer.build_render_plan

        def recorded_source(*args, **kwargs):
            events.append("source")
            return collect_source(*args, **kwargs)

        def recorded_science(*args, **kwargs):
            events.append("science")
            return collect_science(*args, **kwargs)

        def recorded_base(*args, **kwargs):
            events.append("base")
            return build_base(*args, **kwargs)

        with (
            patch.object(
                renderer,
                "collect_source_artifacts",
                side_effect=recorded_source,
            ),
            patch.object(
                renderer,
                "collect_science_artifacts",
                side_effect=recorded_science,
            ),
            patch.object(
                renderer.base_renderer,
                "build_render_plan",
                side_effect=recorded_base,
            ),
        ):
            self.build_plan()
        self.assertEqual(
            events,
            ["source", "science", "base", "source", "science"],
        )

    def test_source_and_cie_artifact_digests_match_current_files(self) -> None:
        self.assertIn(
            Path("offline/authenticated_artifact.py"),
            renderer.PRODUCER_SOURCE_FILES,
        )
        self.assertIn(
            Path("offline/kerr_returning_radiation_spectral_replay.py"),
            renderer.PRODUCER_SOURCE_FILES,
        )
        sources = renderer.collect_source_artifacts(ROOT)
        self.assertEqual(len(sources), len(renderer.PRODUCER_SOURCE_FILES))
        for artifact, relative in zip(sources, renderer.PRODUCER_SOURCE_FILES):
            payload = (ROOT / relative).read_bytes()
            self.assertEqual(artifact.uri, f"repo-source://{relative.as_posix()}")
            self.assertEqual(artifact.byte_length, len(payload))
            self.assertEqual(artifact.sha256, hashlib.sha256(payload).hexdigest())

        science = renderer.collect_science_artifacts(
            renderer.DEFAULT_CIE_CSV.absolute(),
            renderer.DEFAULT_CIE_METADATA.absolute(),
        )
        expected = {
            renderer.CIE_CSV_INPUT_URI: renderer.DEFAULT_CIE_CSV.absolute(),
            renderer.CIE_METADATA_INPUT_URI: (
                renderer.DEFAULT_CIE_METADATA.absolute()
            ),
        }
        self.assertEqual({artifact.uri for artifact in science}, set(expected))
        for artifact in science:
            payload = expected[artifact.uri].read_bytes()
            self.assertEqual(artifact.byte_length, len(payload))
            self.assertEqual(artifact.sha256, hashlib.sha256(payload).hexdigest())

    def test_source_and_cie_symlink_inputs_fail_closed(self) -> None:
        source_alias = self.root / "source-alias"
        source_alias.symlink_to(ROOT, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "exact lexical renderer ROOT"):
            renderer.collect_source_artifacts(source_alias)

        cie_alias = self.root / "cie-alias.csv"
        cie_alias.symlink_to(renderer.DEFAULT_CIE_CSV.absolute())
        with self.assertRaisesRegex(RuntimeError, "symbolic links"):
            renderer.collect_science_artifacts(
                cie_alias,
                renderer.DEFAULT_CIE_METADATA.absolute(),
            )

    def test_artifact_caps_precede_source_and_cie_data_reads(self) -> None:
        cases = (
            (
                "MAXIMUM_PRODUCER_SOURCE_ARTIFACT_BYTE_LENGTH",
                lambda: renderer.collect_source_artifacts(ROOT),
            ),
            (
                "MAXIMUM_CIE_ARTIFACT_TOTAL_BYTE_LENGTH",
                lambda: renderer.collect_science_artifacts(
                    renderer.DEFAULT_CIE_CSV.absolute(),
                    renderer.DEFAULT_CIE_METADATA.absolute(),
                ),
            ),
        )
        for limit_name, operation in cases:
            with self.subTest(limit=limit_name), patch.object(
                renderer,
                limit_name,
                0,
            ), patch.object(
                artifact_module.os,
                "read",
                side_effect=AssertionError("artifact cap must precede data reads"),
            ) as read, self.assertRaises(RuntimeError):
                operation()
            read.assert_not_called()

    def test_source_artifact_change_during_base_build_fails_closed(self) -> None:
        arguments = renderer.parse_args(
            minimal_arguments(self.output, self.tile_cache, self.kernel_cache)
        )
        current = renderer.collect_source_artifacts()
        changed_first = InputArtifact(
            current[0].uri,
            current[0].byte_length,
            "f" * 64,
        )
        changed = (changed_first, *current[1:])
        with (
            patch.object(
                renderer,
                "collect_source_artifacts",
                side_effect=(current, changed),
            ),
            self.assertRaisesRegex(RuntimeError, "source files changed during"),
        ):
            renderer.build_render_plan(arguments)

    def test_cie_artifact_change_during_base_build_fails_closed(self) -> None:
        arguments = renderer.parse_args(
            minimal_arguments(self.output, self.tile_cache, self.kernel_cache)
        )
        current = renderer.collect_science_artifacts(
            Path(arguments.cie_csv).absolute(),
            Path(arguments.cie_metadata).absolute(),
        )
        changed_first = InputArtifact(
            current[0].uri,
            current[0].byte_length,
            "f" * 64,
        )
        changed = tuple(sorted((changed_first, current[1])))
        with (
            patch.object(
                renderer,
                "collect_science_artifacts",
                side_effect=(current, changed),
            ),
            self.assertRaisesRegex(RuntimeError, "CIE inputs changed during"),
        ):
            renderer.build_render_plan(arguments)

    def test_existing_output_fails_before_kernel_cache_execution(self) -> None:
        plan = self.build_plan()
        self.output.mkdir()
        with (
            patch.object(
                renderer,
                "_CACHED_KERNEL_INTEGRATOR_CALL_ENTRY",
            ) as kernel,
            self.assertRaisesRegex(FileExistsError, "refusing to overwrite"),
        ):
            renderer.execute_render_plan(plan)
        kernel.assert_not_called()

    def test_tampered_plan_tile_controls_fail_before_path_or_cache_io(self) -> None:
        plan = replace(self.build_plan(), tile_jobs=2)
        with (
            patch.object(renderer, "_preflight_product_paths") as path_io,
            patch.object(
                renderer,
                "_CACHED_KERNEL_INTEGRATOR_CALL_ENTRY",
            ) as kernel,
            self.assertRaisesRegex(ValueError, "jobs must not exceed"),
        ):
            renderer.execute_render_plan(plan)
        path_io.assert_not_called()
        kernel.assert_not_called()

    def test_kernel_budget_preflight_rejects_before_source_or_base_io(self) -> None:
        cases = (
            ("--kernel-maximum-direction-evaluations", "1", "direction budget"),
            ("--kernel-maximum-whole-ray-traces", "1", "whole-ray budget"),
        )
        for option, value, message in cases:
            with self.subTest(option=option):
                arguments = renderer.parse_args(
                    minimal_arguments(
                        self.output,
                        self.tile_cache,
                        self.kernel_cache,
                        option,
                        value,
                    )
                )
                with (
                    patch.object(renderer, "collect_source_artifacts") as source,
                    patch.object(renderer, "collect_science_artifacts") as science,
                    patch.object(renderer.base_renderer, "build_render_plan") as base,
                    self.assertRaisesRegex(ValueError, message),
                ):
                    renderer.build_render_plan(arguments)
                source.assert_not_called()
                science.assert_not_called()
                base.assert_not_called()

    def test_reduction_plan_mutation_fails_before_source_or_cache_io(self) -> None:
        cases = (
            ("kernel tolerance", "kernel", "absolute_tolerance", 0.24),
            ("area order", "area", "gauss_legendre_order", 12),
            ("area budget", "area", "maximum_point_evaluations", 383),
        )
        for name, owner, field_name, changed in cases:
            with self.subTest(name=name):
                plan = self.build_plan()
                target = (
                    plan.kernel_policy if owner == "kernel" else plan.area_policy
                )
                object.__setattr__(target, field_name, changed)
                # The transport definition intentionally ignores reduction-only
                # controls; the separate frozen configuration must catch drift.
                self.assertEqual(
                    renderer._kernel_cache_definition(plan),
                    plan.kernel_cache_definition,
                )
                with (
                    patch.object(renderer, "_assert_source_provenance") as source,
                    patch.object(
                        renderer,
                        "_CACHED_KERNEL_INTEGRATOR_CALL_ENTRY",
                    ) as kernel,
                    self.assertRaisesRegex(RuntimeError, "reduction configuration"),
                ):
                    renderer.execute_render_plan(plan)
                source.assert_not_called()
                kernel.assert_not_called()

    def test_stale_or_changed_numeric_backend_fails_before_cache_io(self) -> None:
        plan = self.build_plan()
        changed_backend = json.loads(
            renderer.canonical_json_bytes(plan.numeric_backend)
        )
        changed_backend["implementationId"] = "tests.stale-runtime/v1"
        cases = (
            (replace(plan, numeric_backend=changed_backend), None),
            (plan, changed_backend),
        )
        for candidate, current_override in cases:
            with self.subTest(current_override=current_override is not None):
                with ExitStack() as stack:
                    path_io = stack.enter_context(
                        patch.object(renderer, "_preflight_product_paths")
                    )
                    kernel = stack.enter_context(
                        patch.object(
                            renderer,
                            "_CACHED_KERNEL_INTEGRATOR_CALL_ENTRY",
                        )
                    )
                    if current_override is not None:
                        stack.enter_context(
                            patch.object(
                                renderer,
                                "_DEFAULT_NUMERIC_BACKEND_CALL_ENTRY",
                                return_value=current_override,
                            )
                        )
                    with self.assertRaisesRegex(RuntimeError, "numeric backend"):
                        renderer.execute_render_plan(candidate)
                path_io.assert_not_called()
                kernel.assert_not_called()

    def test_public_runtime_rebindings_fail_before_alternative_call(self) -> None:
        alternatives = (
            (renderer, "run_job"),
            (renderer, "publish_spectral_product"),
            (renderer, "validate_scientific_spectral_frame"),
            (renderer, "_returning_tile_executor"),
            (renderer, "build_returning_radiation_spectral_job_spec"),
            (renderer, "generic_build_spectral_job_spec"),
            (renderer.SpectralFrameGrid, "tasks"),
        )
        for owner, name in alternatives:
            replacement = Mock(return_value=None)
            with (
                self.subTest(entry=name),
                patch.object(owner, name, replacement),
                self.assertRaisesRegex(RuntimeError, "identity|binding"),
            ):
                renderer._assert_returning_tile_runtime_bindings()
            replacement.assert_not_called()

    def test_scientific_runtime_rebindings_fail_before_alternative_call(
        self,
    ) -> None:
        plan = self.build_plan()
        alternatives = (
            (
                renderer,
                "verify_kerr_returning_radiation_refinement_checkpoint",
            ),
            (
                renderer.refinement_checkpoint_module,
                "verify_kerr_returning_radiation_refinement_checkpoint",
            ),
            (
                renderer,
                "integrate_existing_cached_kerr_returning_radiation_energy_kernel",
            ),
            (
                renderer.cached_kernel_module,
                "integrate_existing_cached_kerr_returning_radiation_energy_kernel",
            ),
            (
                renderer,
                "solve_certified_kerr_returning_radiation_thermal_profile",
            ),
            (
                renderer.thermal_profile_module,
                "solve_certified_kerr_returning_radiation_thermal_profile",
            ),
            (
                renderer,
                "build_certified_returning_radiation_thermal_spectrum_provider",
            ),
            (
                renderer.thermal_spectrum_module,
                "build_certified_returning_radiation_thermal_spectrum_provider",
            ),
            (renderer, "authenticate_returning_thermal_emission"),
            (
                renderer.frame_context_module,
                "authenticate_returning_thermal_emission",
            ),
            (renderer, "KerrReturningRadiationFiniteThicknessRaySampler"),
            (
                renderer.returning_frame_module,
                "KerrReturningRadiationFiniteThicknessRaySampler",
            ),
            (KerrReturningRadiationFiniteThicknessRaySampler, "__init__"),
            (KerrReturningRadiationFiniteThicknessRaySampler, "__post_init__"),
        )
        for owner, name in alternatives:
            replacement = Mock(return_value=None)
            with (
                self.subTest(owner=owner, entry=name),
                patch.object(owner, name, replacement),
                self.assertRaisesRegex(
                    RuntimeError,
                    "scientific runtime binding changed",
                ),
            ):
                renderer.execute_render_plan(plan)
            replacement.assert_not_called()

        # Patching an inherited ``__new__`` mutates CPython's type slot even
        # after mock restoration, so isolate that adversarial check in a child.
        probe = """
from unittest.mock import Mock, patch
from offline.kerr_returning_radiation_finite_thickness_frame import (
    KerrReturningRadiationFiniteThicknessRaySampler,
)
import scripts.render_offline_kerr_returning_radiation_frame as renderer

replacement = Mock(return_value=None)
with patch.object(
    KerrReturningRadiationFiniteThicknessRaySampler,
    "__new__",
    replacement,
):
    try:
        renderer._assert_returning_scientific_runtime_bindings()
    except RuntimeError as error:
        assert "scientific runtime binding changed" in str(error), error
    else:
        raise AssertionError("sampler __new__ replacement was accepted")
replacement.assert_not_called()
"""
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_live_replay_rebindings_fail_before_alternative_call(self) -> None:
        from offline import kerr_returning_radiation_spectral_replay as replay_module

        plan = self.build_plan()
        alternatives = (
            (renderer, "_run_returning_radiation_spectral_live_replay"),
            (
                renderer,
                "publish_returning_radiation_live_replay_attestation",
            ),
            (
                renderer.attestation_module,
                "publish_returning_radiation_live_replay_attestation",
            ),
            (
                replay_module,
                "validate_returning_radiation_spectral_live_replay",
            ),
        )
        for owner, name in alternatives:
            replacement = Mock(return_value=None)
            with (
                self.subTest(entry=name),
                patch.object(owner, name, replacement),
                self.assertRaisesRegex(RuntimeError, "live-replay.*identity"),
            ):
                renderer.execute_render_plan(plan)
            replacement.assert_not_called()

    def test_replay_first_public_rebind_cannot_poison_first_lazy_freeze(
        self,
    ) -> None:
        probe = """
from unittest.mock import Mock
from offline import kerr_returning_radiation_spectral_replay as replay
import scripts.render_offline_kerr_returning_radiation_frame as renderer

replacement = Mock(return_value=None)
replay.validate_returning_radiation_spectral_live_replay = replacement
try:
    renderer._assert_returning_live_replay_runtime_binding()
except RuntimeError as error:
    assert "live-replay callable identity changed" in str(error), error
else:
    raise AssertionError("pre-first-use live replay replacement was accepted")
replacement.assert_not_called()
"""
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_source_tamper_fails_before_kernel_execution(self) -> None:
        plan = self.build_plan()
        with (
            patch.object(
                renderer,
                "assert_bound_inputs_stable",
                side_effect=RuntimeError("producer source files changed"),
            ),
            patch.object(
                renderer,
                "_CACHED_KERNEL_INTEGRATOR_CALL_ENTRY",
            ) as kernel,
            self.assertRaisesRegex(RuntimeError, "source files changed"),
        ):
            renderer.execute_render_plan(plan)
        kernel.assert_not_called()

    def test_base_sampler_tamper_fails_before_kernel_execution(self) -> None:
        plan = self.build_plan()
        object.__setattr__(
            plan.base_sampler,
            "observer_theta_rad",
            plan.base_sampler.observer_theta_rad + 0.01,
        )
        with (
            patch.object(
                renderer,
                "_CACHED_KERNEL_INTEGRATOR_CALL_ENTRY",
            ) as kernel,
            self.assertRaisesRegex(RuntimeError, "sampler changed"),
        ):
            renderer.execute_render_plan(plan)
        kernel.assert_not_called()

    def test_tile_payload_is_guarded_by_pre_and_post_source_cie_checks(self) -> None:
        plan = self.build_plan()
        authority = _Authority()
        sampler = object.__new__(
            KerrReturningRadiationFiniteThicknessRaySampler
        )
        object.__setattr__(sampler, "authority", authority)
        object.__setattr__(sampler, "_authenticated_snapshot", authority.snapshot)
        source_hashes = tuple(
            sorted({artifact.sha256 for artifact in plan.source_artifacts})
        )
        inputs = tuple(sorted((*plan.source_artifacts, *plan.science_artifacts)))
        with patch.object(
            KerrReturningRadiationFiniteThicknessRaySampler,
            "descriptor",
            return_value=returning_descriptor(),
        ):
            specification = build_returning_radiation_spectral_job_spec(
                sampler,
                plan.layout,
                plan.grid,
                plan.adaptive_options,
                tile_width=plan.tile_width,
                tile_height=plan.tile_height,
                numeric_backend=plan.numeric_backend,
                inputs=inputs,
                producer_source_hashes=source_hashes,
            )
            inner = ReturningRadiationAdaptiveSpectralTileProducer(
                sampler,
                plan.layout,
                plan.grid,
                plan.adaptive_options,
                plan.numeric_backend,
                specification,
            )
            producer = renderer.BoundInputStableReturningSpectralTileProducer(
                inner,
                plan.source_artifacts,
                plan.science_artifacts,
                plan.source_root,
                plan.cie_csv_path,
                plan.cie_metadata_path,
            )
            def fake_adaptive_pixel(
                sampler,
                observer_frequencies_hz,
                *,
                x_min,
                x_max,
                y_min,
                y_max,
                options,
            ):
                del sampler
                return integrate_spectral_pixel(
                    _ConstantDiskSampler(),
                    observer_frequencies_hz,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    options=options,
                )

            with (
                patch.object(renderer, "assert_bound_inputs_stable") as stable,
                patch.object(
                    renderer,
                    "_assert_source_provenance",
                    wraps=renderer._assert_source_provenance,
                ) as origin,
                patch.object(
                    returning_frame_module,
                    "integrate_spectral_pixel",
                    side_effect=fake_adaptive_pixel,
                ) as pixel,
            ):
                payload = producer(specification, specification.tasks[0])
        self.assertEqual(len(payload), plan.layout.record_bytes)
        self.assertEqual(stable.call_count, 2)
        self.assertEqual(origin.call_count, 2)
        pixel.assert_called_once()

    def test_bound_input_gate_rechecks_module_origins(self) -> None:
        plan = self.build_plan()
        with patch.object(
            renderer,
            "_assert_source_provenance",
            wraps=renderer._assert_source_provenance,
        ) as origin, patch.object(
            renderer,
            "collect_source_artifacts",
            return_value=plan.source_artifacts,
        ), patch.object(
            renderer,
            "collect_science_artifacts",
            return_value=plan.science_artifacts,
        ):
            renderer.assert_bound_inputs_stable(
                plan.source_artifacts,
                plan.science_artifacts,
                source_root=plan.source_root,
                cie_csv_path=plan.cie_csv_path,
                cie_metadata_path=plan.cie_metadata_path,
            )
        origin.assert_called_once_with(plan.source_root)

    def test_execute_origin_drift_fails_before_path_or_kernel_io(self) -> None:
        plan = self.build_plan()
        with patch.object(
            renderer,
            "_assert_producer_source_module_origins",
            side_effect=RuntimeError("producer source module origin drifted"),
        ), patch.object(renderer, "_preflight_product_paths") as path_io, patch.object(
            renderer,
            "_CACHED_KERNEL_INTEGRATOR_CALL_ENTRY",
        ) as kernel, self.assertRaisesRegex(RuntimeError, "origin drifted"):
            renderer.execute_render_plan(plan)
        path_io.assert_not_called()
        kernel.assert_not_called()

    def test_execute_reverifies_checkpoint_before_paths_cache_or_thermal(
        self,
    ) -> None:
        plan = self.build_plan()
        nonqualified = KerrReturningRadiationVerifiedRefinementCheckpoint(
            plan.required_v2_checkpoint_manifest.parent,
            plan.required_v2_checkpoint_manifest,
            plan.required_v2_checkpoint_manifest_sha256,
            "kerr-returning-radiation-refinement-" + "1" * 24,
            False,
            {},
        )
        with (
            patch.object(
                renderer,
                "_REFINEMENT_CHECKPOINT_VERIFIER_CALL_ENTRY",
                return_value=nonqualified,
            ) as verifier,
            patch.object(renderer, "_preflight_product_paths") as paths,
            patch.object(
                renderer,
                "_CACHED_KERNEL_INTEGRATOR_CALL_ENTRY",
            ) as cache_only,
            patch.object(
                renderer,
                "_THERMAL_PROFILE_SOLVER_CALL_ENTRY",
            ) as thermal,
            self.assertRaisesRegex(RuntimeError, "non-qualified"),
        ):
            renderer.execute_render_plan(plan)
        verifier.assert_called_once_with(
            plan.required_v2_checkpoint_manifest,
            expected_manifest_sha256=(
                plan.required_v2_checkpoint_manifest_sha256
            ),
        )
        paths.assert_not_called()
        cache_only.assert_not_called()
        thermal.assert_not_called()

    def test_execute_rejects_checkpoint_plan_binding_drift_before_cache(
        self,
    ) -> None:
        plan = self.build_plan()
        changed_artifact = InputArtifact(
            plan.required_v2_checkpoint_binding.input_artifact.uri,
            plan.required_v2_checkpoint_binding.input_artifact.byte_length,
            "e" * 64,
        )
        changed_binding = replace(
            plan.required_v2_checkpoint_binding,
            input_artifact=changed_artifact,
        )
        changed_plan = replace(
            plan,
            required_v2_checkpoint_binding=changed_binding,
        )
        with (
            patch.object(
                renderer,
                "_CACHED_KERNEL_INTEGRATOR_CALL_ENTRY",
            ) as cache_only,
            patch.object(
                renderer,
                "_THERMAL_PROFILE_SOLVER_CALL_ENTRY",
            ) as thermal,
            self.assertRaisesRegex(
                RuntimeError,
                "checkpoint binding changed after renderer planning",
            ),
        ):
            renderer.execute_render_plan(changed_plan)
        cache_only.assert_not_called()
        thermal.assert_not_called()

    def test_deleted_kernel_cache_never_traces_or_starts_product_pipeline(
        self,
    ) -> None:
        self.kernel_cache.mkdir()
        plan = self.build_plan()
        self.kernel_cache.rmdir()
        with (
            patch.object(
                renderer.cached_kernel_module._forward,
                "_trace_direction",
                side_effect=AssertionError(
                    "cache-only product gate must never trace a direction"
                ),
            ) as trace,
            patch.object(
                renderer,
                "_THERMAL_PROFILE_SOLVER_CALL_ENTRY",
            ) as thermal,
            patch.object(renderer, "_PUBLISH_CALL_ENTRY") as publish,
            self.assertRaisesRegex(RuntimeError, "cache_root|cache root"),
        ):
            renderer.execute_render_plan(plan)
        trace.assert_not_called()
        thermal.assert_not_called()
        publish.assert_not_called()

    def _mock_execution_dependencies(self, plan, *, run_job_side_effect):
        results = tuple(
            TaskResult(
                key,
                self.kernel_cache / "payload.bin",
                self.kernel_cache / "receipt.json",
                key.width,
                key.width,
                "e" * 64,
                True,
            )
            for key in plan.kernel_cache_definition.job_spec.tasks
        )
        full_hit_run = JobRun(
            plan.kernel_cache_definition.job_spec.job_key,
            results,
            len(results),
            0,
            0,
        )
        full_hit_audit = KerrCachedKernelExecutionAudit(
            "forward",
            renderer.cached_kernel_module._FORWARD_EVALUATOR_ID,
            plan.kernel_cache_definition.scientific_job_key,
            plan.kernel_cache_definition.job_spec.job_key,
            plan.kernel_cache_definition.plan.direction_count,
            0,
            plan.kernel_cache_definition.plan.direction_count,
            0,
            plan.kernel_cache_definition.plan.task_count,
            "f" * 64,
            True,
            True,
            False,
            False,
        )
        fake_kernel = SimpleNamespace(model_descriptor_sha256="d" * 64)
        fake_cached = object.__new__(KerrCachedReturningRadiationKernelExecution)
        for name, value in (
            ("formulation", "forward"),
            ("kernel", fake_kernel),
            ("cache_definition", plan.kernel_cache_definition),
            (
                "reduction_configuration",
                plan.kernel_reduction_configuration,
            ),
            ("job_run", full_hit_run),
            ("source_closure", ()),
            (
                "execution_audit",
                full_hit_audit,
            ),
        ):
            object.__setattr__(fake_cached, name, value)
        fake_profile = object()
        fake_provider = object()
        authority = _Authority()
        sampler = object.__new__(
            KerrReturningRadiationFiniteThicknessRaySampler
        )
        object.__setattr__(sampler, "authority", authority)
        object.__setattr__(sampler, "_authenticated_snapshot", authority.snapshot)
        publication = SpectralProductPublication(
            output_directory=plan.output_directory,
            manifest_path=plan.output_directory / "manifest.json",
            manifest_sha256="b" * 64,
            product_id="scientific-spectral-frame-returning-test",
            product_sha256="c" * 64,
            tile_count=1,
            record_count=1,
        )
        verification = {"status": STRUCTURAL_STATUS, "physicsVerified": False}
        patches = (
            patch.object(renderer, "assert_bound_inputs_stable"),
            patch.object(
                renderer,
                "_CACHED_KERNEL_INTEGRATOR_CALL_ENTRY",
                return_value=fake_cached,
            ),
            patch.object(
                renderer,
                "_THERMAL_PROFILE_SOLVER_CALL_ENTRY",
                return_value=fake_profile,
            ),
            patch.object(
                renderer,
                "_THERMAL_PROVIDER_BUILDER_CALL_ENTRY",
                return_value=fake_provider,
            ),
            patch.object(
                renderer,
                "_THERMAL_AUTHENTICATOR_CALL_ENTRY",
                return_value=authority,
            ),
            patch.object(
                renderer,
                "_RETURNING_SAMPLER_CONSTRUCTOR_CALL_ENTRY",
                return_value=sampler,
            ),
            patch.object(
                KerrReturningRadiationFiniteThicknessRaySampler,
                "descriptor",
                return_value=returning_descriptor(),
            ),
            patch.object(
                renderer,
                "_RUN_JOB_CALL_ENTRY",
                side_effect=run_job_side_effect,
            ),
            patch.object(
                renderer,
                "_PUBLISH_CALL_ENTRY",
                return_value=publication,
            ),
            patch.object(
                renderer,
                "_VERIFIER_CALL_ENTRY",
                return_value=verification,
            ),
            patch.object(
                renderer,
                "_ATTESTATION_PUBLISH_CALL_ENTRY",
                return_value=live_replay_attestation(plan, publication),
            ),
        )
        return (
            fake_cached,
            fake_profile,
            fake_provider,
            authority,
            sampler,
            publication,
            verification,
            patches,
        )

    def test_execute_wires_kernel_profile_authority_tiles_publish_and_verifier(
        self,
    ) -> None:
        plan = self.build_plan(
            "--width",
            "2",
            "--kernel-jobs",
            "2",
            "--kernel-max-in-flight",
            "3",
            "--jobs",
            "2",
            "--max-in-flight",
            "2",
        )

        def fake_run(spec, producer, cache_root, **keywords):
            bound = producer.args[0]
            self.assertIsInstance(
                bound.inner,
                ReturningRadiationAdaptiveSpectralTileProducer,
            )
            self.assertEqual(cache_root, plan.tile_cache_root)
            self.assertEqual(keywords["jobs"], 2)
            self.assertEqual(keywords["max_in_flight"], 2)
            self.assertIs(
                keywords["executor_factory"],
                renderer._returning_tile_executor,
            )
            return JobRun(spec.job_key, (), 0, 0, 0)

        (
            fake_cached,
            fake_profile,
            fake_provider,
            authority,
            sampler,
            publication,
            verification,
            patches,
        ) = self._mock_execution_dependencies(plan, run_job_side_effect=fake_run)
        with patches[0] as stable, patches[1] as kernel, patches[2] as solve, \
             patches[3] as provider, patches[4] as authenticate, \
             patches[5], patches[6], patches[7] as run, patches[8] as publish, \
             patches[9] as verify, patches[10] as replay:
            execution = renderer.execute_render_plan(plan)

        self.assertIs(execution.cached_kernel, fake_cached)
        self.assertIs(execution.thermal_profile, fake_profile)
        self.assertIs(execution.thermal_provider, fake_provider)
        self.assertIs(execution.authority, authority)
        self.assertIs(execution.sampler, sampler)
        self.assertIs(execution.publication, publication)
        self.assertIs(execution.verification, verification)
        self.assertEqual(
            execution.live_replay_attestation.manifest_sha256,
            "f" * 64,
        )
        self.assertEqual(
            execution.live_replay_report,
            live_replay_report(publication),
        )
        self.assertIs(execution.verification["physicsVerified"], False)
        self.assertNotIn("physicsVerified", execution.live_replay_report)
        self.assertIs(
            execution.live_replay_report["frameGeodesicsReplayed"],
            True,
        )
        for name in (
            "thermalFixedPointReplayed",
            "directionCacheRecordsReplayed",
            "directionRaysRetraced",
            "independentPhysicsOracle",
        ):
            self.assertIs(execution.live_replay_report[name], False)
        self.assertGreaterEqual(stable.call_count, 4)
        self.assertEqual(authority.live_calls, 2)
        kernel.assert_called_once()
        kernel_keywords = kernel.call_args.kwargs
        self.assertEqual(kernel_keywords["cache_root"], plan.kernel_cache_root)
        self.assertNotIn("jobs", kernel_keywords)
        self.assertNotIn("max_in_flight", kernel_keywords)
        solve.assert_called_once_with(
            fake_cached,
            disk=plan.base_sampler.disk,
            policy=plan.fixed_point_policy,
        )
        provider.assert_called_once_with(fake_profile)
        authenticate.assert_called_once_with(plan.base_sampler.surface, fake_provider)
        run.assert_called_once()
        publish.assert_called_once()
        verify.assert_called_once_with(
            publication.manifest_path,
            plan.verification_schema,
        )
        replay.assert_called_once_with(
            plan.live_replay_attestation_directory,
            spectral_publication=publication,
            sampler=sampler,
            spectral_schema_path=plan.verification_schema,
        )
        descriptor_json = json.dumps(
            execution.job_spec.as_dict()["parameters"]["samplerDescriptor"],
            allow_nan=False,
            sort_keys=True,
        )
        self.assertNotIn(str(plan.kernel_cache_root), descriptor_json)
        self.assertNotIn(str(plan.tile_cache_root), descriptor_json)
        self.assertNotIn("reusedTasks", descriptor_json)
        self.assertNotIn("workerCount", descriptor_json)

    def test_product_job_inputs_bind_checkpoint_artifact_digest_and_id(
        self,
    ) -> None:
        plan = self.build_plan()
        dependencies = self._mock_execution_dependencies(
            plan,
            run_job_side_effect=lambda spec, *_args, **_keywords: JobRun(
                spec.job_key,
                (),
                0,
                0,
                0,
            ),
        )
        patches = dependencies[-1]
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            execution = renderer.execute_render_plan(plan)

        expected = plan.required_v2_checkpoint_binding.input_artifact
        matches = tuple(
            artifact
            for artifact in execution.job_spec.inputs
            if artifact.uri.startswith(
                "urn:blackhole:kerr-returning-radiation-refinement-"
                "checkpoint:v1:"
            )
        )
        self.assertEqual(matches, (expected,))
        self.assertTrue(
            matches[0].uri.endswith(
                plan.required_v2_checkpoint_binding.checkpoint_id
            )
        )
        self.assertEqual(
            matches[0].sha256,
            plan.required_v2_checkpoint_manifest_sha256,
        )

    def test_non_full_kernel_audit_and_checkpoint_sha_mismatch_stop_thermal(
        self,
    ) -> None:
        plan = self.build_plan()
        for case in ("non-full-audit", "kernel-sha"):
            with self.subTest(case=case):
                dependencies = self._mock_execution_dependencies(
                    plan,
                    run_job_side_effect=AssertionError(
                        "tile job must not start"
                    ),
                )
                fake_cached = dependencies[0]
                patches = dependencies[-1]
                if case == "non-full-audit":
                    audit = fake_cached.execution_audit
                    object.__setattr__(
                        fake_cached,
                        "execution_audit",
                        replace(
                            audit,
                            executed_direction_records=1,
                            reused_direction_records=(
                                audit.authenticated_direction_records - 1
                            ),
                            executed_tasks=1,
                            reused_tasks=audit.reused_tasks - 1,
                        ),
                    )
                    expected = "complete pre-existing kernel cache hit"
                else:
                    fake_cached.kernel.model_descriptor_sha256 = "e" * 64
                    expected = "differs from the qualified v2 checkpoint"
                with ExitStack() as stack:
                    mocks = [stack.enter_context(item) for item in patches]
                    with self.assertRaisesRegex(RuntimeError, expected):
                        renderer.execute_render_plan(plan)
                mocks[1].assert_called_once()
                mocks[2].assert_not_called()
                mocks[7].assert_not_called()
                mocks[8].assert_not_called()

    def test_mid_pipeline_scientific_rebind_fails_before_replacement_call(
        self,
    ) -> None:
        plan = self.build_plan()
        dependencies = self._mock_execution_dependencies(
            plan,
            run_job_side_effect=AssertionError("tile job must not start"),
        )
        fake_cached = dependencies[0]
        patches = dependencies[-1]
        replacement = Mock(return_value=object())
        rebind = patch.object(
            renderer,
            "solve_certified_kerr_returning_radiation_thermal_profile",
            replacement,
        )

        def finish_kernel_then_rebind(*args, **keywords):
            del args, keywords
            rebind.start()
            self.addCleanup(rebind.stop)
            return fake_cached

        with ExitStack() as stack:
            mocks = [stack.enter_context(item) for item in patches]
            mocks[1].side_effect = finish_kernel_then_rebind
            with self.assertRaisesRegex(
                RuntimeError,
                "scientific runtime binding changed",
            ):
                renderer.execute_render_plan(plan)
        replacement.assert_not_called()
        mocks[2].assert_not_called()
        mocks[7].assert_not_called()
        mocks[10].assert_not_called()

    def test_live_replay_attestation_is_mandatory_for_success(self) -> None:
        plan = self.build_plan()
        cases = (
            (RuntimeError("live sampler unavailable"), "unavailable"),
            (RuntimeError("live replay report overclaims"), "overclaims"),
        )
        for live_result, message in cases:
            with self.subTest(message=message):
                dependencies = self._mock_execution_dependencies(
                    plan,
                    run_job_side_effect=lambda spec, *_args, **_keywords: JobRun(
                        spec.job_key,
                        (),
                        0,
                        0,
                        0,
                    ),
                )
                patches = dependencies[-1]
                with ExitStack() as stack:
                    mocks = [stack.enter_context(item) for item in patches]
                    mocks[10].side_effect = live_result
                    with self.assertRaisesRegex(RuntimeError, message):
                        renderer.execute_render_plan(plan)
                mocks[10].assert_called_once()

    def test_foreign_sampler_constructor_result_never_degrades(self) -> None:
        plan = self.build_plan()
        dependencies = self._mock_execution_dependencies(
            plan,
            run_job_side_effect=AssertionError("tile job must not start"),
        )
        patches = dependencies[-1]
        with ExitStack() as stack:
            mocks = [stack.enter_context(item) for item in patches]
            mocks[5].return_value = object()
            with self.assertRaisesRegex(TypeError, "foreign type"):
                renderer.execute_render_plan(plan)
        mocks[7].assert_not_called()
        mocks[8].assert_not_called()
        mocks[10].assert_not_called()

    def test_full_tile_cache_hit_still_postchecks_runtime_bindings(self) -> None:
        plan = self.build_plan()
        replacement = Mock(return_value=None)
        replacement_patch = patch.object(
            returning_frame_module,
            "integrate_returning_thermal_spectral_pixel",
            replacement,
        )

        def fake_full_hit(spec, producer, cache_root, **keywords):
            del producer, cache_root, keywords
            replacement_patch.start()
            self.addCleanup(replacement_patch.stop)
            return JobRun(spec.job_key, (), 1, 0, 0)

        dependencies = self._mock_execution_dependencies(
            plan,
            run_job_side_effect=fake_full_hit,
        )
        patches = dependencies[-1]
        with ExitStack() as stack:
            mocks = [stack.enter_context(item) for item in patches]
            with self.assertRaisesRegex(RuntimeError, "integrator identity changed"):
                renderer.execute_render_plan(plan)
        replacement.assert_not_called()
        mocks[8].assert_not_called()

    def test_full_tile_cache_hit_still_postchecks_source_origins(self) -> None:
        plan = self.build_plan()

        def fake_full_hit(spec, producer, cache_root, **keywords):
            del producer, cache_root, keywords
            return JobRun(spec.job_key, (), 1, 0, 0)

        dependencies = self._mock_execution_dependencies(
            plan,
            run_job_side_effect=fake_full_hit,
        )
        patches = dependencies[-1]
        with ExitStack() as stack:
            mocks = [stack.enter_context(item) for item in patches]
            origin = stack.enter_context(
                patch.object(
                    renderer,
                    "_assert_source_provenance",
                    side_effect=(
                        None,
                        RuntimeError("source origin changed on full hit"),
                    ),
                )
            )
            with self.assertRaisesRegex(RuntimeError, "origin changed on full hit"):
                renderer.execute_render_plan(plan)
        self.assertEqual(origin.call_count, 2)
        mocks[7].assert_called_once()
        mocks[8].assert_not_called()

    def test_runtime_rebind_fails_before_run_job_or_alternative_call(self) -> None:
        plan = self.build_plan()
        dependencies = self._mock_execution_dependencies(
            plan,
            run_job_side_effect=AssertionError("run_job must not start"),
        )
        patches = dependencies[-1]
        replacement = Mock(return_value=b"foreign")
        with ExitStack() as stack:
            mocks = [stack.enter_context(item) for item in patches]
            stack.enter_context(
                patch.object(
                    product_module,
                    "invoke_returning_radiation_spectral_tile_producer",
                    replacement,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "runtime binding changed"):
                renderer.execute_render_plan(plan)
        replacement.assert_not_called()
        mocks[7].assert_not_called()
        mocks[8].assert_not_called()

    def test_numeric_backend_change_on_full_hit_fails_before_publication(self) -> None:
        plan = self.build_plan()
        changed_backend = json.loads(
            renderer.canonical_json_bytes(plan.numeric_backend)
        )
        changed_backend["implementationId"] = "tests.changed-during-run/v1"
        backend_patch = patch.object(
            renderer,
            "_DEFAULT_NUMERIC_BACKEND_CALL_ENTRY",
            return_value=changed_backend,
        )

        def fake_full_hit(spec, producer, cache_root, **keywords):
            del producer, cache_root, keywords
            backend_patch.start()
            self.addCleanup(backend_patch.stop)
            return JobRun(spec.job_key, (), 1, 0, 0)

        dependencies = self._mock_execution_dependencies(
            plan,
            run_job_side_effect=fake_full_hit,
        )
        patches = dependencies[-1]
        with ExitStack() as stack:
            mocks = [stack.enter_context(item) for item in patches]
            with self.assertRaisesRegex(RuntimeError, "numeric backend changed"):
                renderer.execute_render_plan(plan)
        mocks[8].assert_not_called()

    def test_mid_tile_backend_drift_writes_no_receipt_and_recomputes(self) -> None:
        plan = self.build_plan()
        original_run_job = renderer.run_job
        changed_backend = json.loads(
            renderer.canonical_json_bytes(plan.numeric_backend)
        )
        changed_backend["implementationId"] = "tests.mid-tile-drift/v1"
        backend_patch = patch.object(
            renderer,
            "_DEFAULT_NUMERIC_BACKEND_CALL_ENTRY",
            return_value=changed_backend,
        )
        drift_once = True

        def fake_pixel(
            sampler,
            observer_frequencies_hz,
            *,
            x_min,
            x_max,
            y_min,
            y_max,
            options,
        ):
            nonlocal drift_once
            del sampler
            if drift_once:
                drift_once = False
                backend_patch.start()
            return integrate_spectral_pixel(
                _ConstantDiskSampler(),
                observer_frequencies_hz,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                options=options,
            )

        dependencies = self._mock_execution_dependencies(
            plan,
            run_job_side_effect=original_run_job,
        )
        patches = dependencies[-1]
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            pixel = stack.enter_context(
                patch.object(
                    returning_frame_module,
                    "integrate_spectral_pixel",
                    side_effect=fake_pixel,
                )
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "numeric backend"):
                    renderer.execute_render_plan(plan)
            finally:
                backend_patch.stop()
            self.assertEqual(
                tuple(plan.tile_cache_root.rglob("*.receipt.json")),
                (),
            )
            second = renderer.execute_render_plan(plan)
        self.assertEqual(
            (second.job_run.executed_tasks, second.job_run.reused_tasks),
            (1, 0),
        )
        self.assertEqual(pixel.call_count, 2)

    def test_tile_cache_reuse_with_mocked_production_kernel_pipeline(self) -> None:
        plan = self.build_plan(
            "--width",
            "2",
            "--jobs",
            "2",
            "--max-in-flight",
            "2",
        )
        original_run_job = renderer.run_job

        def fake_pixel(
            sampler,
            observer_frequencies_hz,
            *,
            x_min,
            x_max,
            y_min,
            y_max,
            options,
        ):
            del sampler
            return integrate_spectral_pixel(
                _ConstantDiskSampler(),
                observer_frequencies_hz,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                options=options,
            )

        dependencies = self._mock_execution_dependencies(
            plan,
            run_job_side_effect=original_run_job,
        )
        patches = dependencies[-1]
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9], \
             patches[10], \
             patch.object(
                 returning_frame_module,
                 "integrate_spectral_pixel",
                 side_effect=fake_pixel,
             ) as pixel:
            first = renderer.execute_render_plan(plan)
            second = renderer.execute_render_plan(plan)

        self.assertEqual(
            (first.job_run.executed_tasks, first.job_run.reused_tasks),
            (2, 0),
        )
        self.assertEqual(
            (second.job_run.executed_tasks, second.job_run.reused_tasks),
            (0, 2),
        )
        self.assertEqual(pixel.call_count, 2)

    def test_main_prints_structural_and_live_replay_as_separate_layers(
        self,
    ) -> None:
        publication = SpectralProductPublication(
            output_directory=self.output,
            manifest_path=self.output / "manifest.json",
            manifest_sha256="b" * 64,
            product_id="scientific-spectral-frame-returning-test",
            product_sha256="c" * 64,
            tile_count=1,
            record_count=1,
        )
        plan = SimpleNamespace(
            layout=SimpleNamespace(frequency_count=471),
            annulus_edges_over_mass=(1.0, 2.0),
        )
        execution = SimpleNamespace(
            publication=publication,
            job_run=SimpleNamespace(
                job_key="d" * 64,
                reused_tasks=0,
                executed_tasks=1,
            ),
            cached_kernel=SimpleNamespace(
                cache_definition=SimpleNamespace(scientific_job_key="e" * 64),
                execution_audit=SimpleNamespace(
                    reused_direction_records=0,
                    executed_direction_records=7,
                ),
            ),
            verification={
                "status": STRUCTURAL_STATUS,
                "physicsVerified": False,
            },
            live_replay_attestation=SimpleNamespace(
                manifest_path=(
                    self.output.with_name(
                        self.output.name + ".live-replay-attestation-v1"
                    )
                    / "manifest.json"
                ),
                manifest_sha256="f" * 64,
            ),
            live_replay_report=live_replay_report(publication),
        )
        with (
            patch.object(renderer, "parse_args", return_value=object()),
            patch.object(renderer, "build_render_plan", return_value=plan),
            patch.object(renderer, "execute_render_plan", return_value=execution),
            patch("builtins.print") as printer,
        ):
            self.assertEqual(renderer.main([]), 0)
        output = "\n".join(
            " ".join(str(item) for item in call.args)
            for call in printer.call_args_list
        )
        self.assertIn("strict structural verifier", output)
        self.assertIn("live replay attestation sha256", output)
        self.assertIn("live sampler replay", output)
        self.assertIn("pixelBytesExact=True", output)
        self.assertIn("thermalFixedPointReplayed=False", output)
        self.assertIn("directionRaysRetraced=False", output)
        self.assertIn("independentPhysicsOracle=False", output)


if __name__ == "__main__":
    unittest.main()
