from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.job import canonical_json_bytes
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_finite_thickness import (
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
from offline.kerr_returning_radiation_convergence_v2 import (
    KerrReturningRadiationConvergenceV2Policy,
)
import offline.kerr_returning_radiation_kernel as forward_module
from offline.kerr_returning_radiation_kernel import (
    KerrReturningRadiationKernelPolicy,
)
import offline.kerr_returning_radiation_kernel_cached as cached_module
from offline.kerr_returning_radiation_kernel_cached import (
    KerrCachedReturningRadiationKernelExecution,
)
import offline.kerr_returning_radiation_kernel_jobs as jobs_module
from offline.kerr_returning_radiation_kernel_jobs import FORWARD, RECEIVER
import offline.kerr_returning_radiation_refinement_checkpoint as checkpoint_module
from offline.kerr_returning_radiation_refinement_checkpoint import (
    MANIFEST_NAME,
    SIDECAR_NAME,
    KerrReturningRadiationRefinementCheckpointError,
    KerrReturningRadiationRefinementPlan,
    execute_kerr_returning_radiation_refinement_checkpoint,
    verify_kerr_returning_radiation_refinement_checkpoint,
)


def production_forward_classifier(*arguments):
    """Cheap deterministic stand-in at the production forward trace seam."""

    source_radius = arguments[7]
    tangent_azimuth = arguments[9]
    receiver_face = "upper" if tangent_azimuth < math.pi else "lower"
    ratio = 2.0
    identity = hashlib.sha256(repr(arguments[6:]).encode("utf-8")).hexdigest()
    return forward_module._DirectionTransport(
        f"return-{receiver_face}",
        receiver_face,
        source_radius,
        ratio,
        ratio * ratio,
        identity,
        receiver_face,
        source_radius,
    )


def noncanonical_convergence_policy() -> KerrReturningRadiationConvergenceV2Policy:
    """A deliberately relaxed policy that the checkpoint front door must reject."""

    return KerrReturningRadiationConvergenceV2Policy(
        g2_column_relative_tolerance=1.0,
        g2_column_absolute_tolerance=1.0,
        g2_column_relative_floor=1.0,
        column_normalized_l1_tolerance=1.0,
        significant_cell_fraction_of_column=1.0,
        significant_cell_symmetric_relative_tolerance=1.0,
        insignificant_tail_normalized_tolerance=1.0,
        support_flip_absolute_tolerance=1.0,
        fate_total_variation_tolerance=1.0,
        fate_component_absolute_tolerance=1.0,
        fate_component_relative_tolerance=1.0,
        fate_component_relative_floor=1.0,
        maximum_normalized_sample_weight=1.0,
    )


def refinement_plan(
    output: Path,
    cache: Path,
    *,
    convergence_policy: KerrReturningRadiationConvergenceV2Policy | None = None,
) -> KerrReturningRadiationRefinementPlan:
    metric = KerrKerrSchildMetric(spin_a_m=0.7)
    calibration = StationaryKerrFiniteThicknessCalibration(
        dimensionless_spin=0.7,
        eddington_scaled_mass_accretion_rate=0.08,
        outer_radius_over_mass=8.0,
    )
    surface = KerrFiniteThicknessMultiSurface(metric, calibration)
    termination = KerrOblateTermination.horizon_worldtube(
        metric,
        escape_radius_m=20.0,
        offset_m=0.02,
    )
    return KerrReturningRadiationRefinementPlan(
        output_directory=output,
        cache_root=cache,
        surface=surface,
        termination=termination,
        annulus_edges_over_mass=(
            float(calibration.isco_radius_over_mass),
            float(calibration.outer_radius_over_mass),
        ),
        ray_options=RayTraceOptions(
            absolute_tolerance=5.0e-9,
            relative_tolerance=5.0e-9,
            initial_step=0.025,
            maximum_step=0.3,
            maximum_affine_length=100.0,
        ),
        surface_options=SurfaceEventOptions(
            absolute_tolerance=5.0e-9,
            relative_tolerance=5.0e-9,
            subdivisions_per_segment=4,
        ),
        kernel_policy=KerrReturningRadiationKernelPolicy(
            rho_order=4,
            mu_order=4,
            psi_count=4,
            absolute_tolerance=0.25,
            relative_tolerance=0.25,
            symmetry_absolute_tolerance=0.25,
            symmetry_relative_tolerance=0.25,
            maximum_direction_evaluations=10_000,
            maximum_whole_ray_traces=40_000,
        ),
        area_policy=KerrFiniteThicknessAreaQuadraturePolicy(
            gauss_legendre_order=24,
            relative_tolerance=1.0e-8,
            absolute_tolerance_over_mass_squared=1.0e-8,
            maximum_point_evaluations=384,
        ),
        convergence_policy=(
            KerrReturningRadiationConvergenceV2Policy()
            if convergence_policy is None
            else convergence_policy
        ),
        directions_per_task=16,
        jobs=1,
    )


def _file_identity(path: Path) -> tuple[int, int, int, int, int, int]:
    status = path.stat()
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


class OfflineKerrReturningRadiationRefinementCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Secure checkpoint/cache paths reject symlinked ancestors.  On macOS,
        # /tmp and /var are symlinks, so all fixtures are anchored in /private/tmp.
        cls.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name).resolve(strict=True)
        cls.cache = cls.root / "production-forward-cache"
        cls.nonqualified_output = cls.root / "nonqualified-checkpoint"
        with patch.object(
            forward_module,
            "_trace_direction",
            side_effect=production_forward_classifier,
        ):
            cls.nonqualified_publication = (
                execute_kerr_returning_radiation_refinement_checkpoint(
                    refinement_plan(cls.nonqualified_output, cls.cache)
                )
            )

    def _case_path(self, name: str) -> Path:
        return self.root / f"{self._testMethodName}-{name}"

    def _checkpoint_copy(self, name: str) -> Path:
        output = self._case_path(name)
        shutil.copytree(self.nonqualified_output, output)
        return output

    def _synthetic_cached_source(
        self,
        formulation: str,
        name: str,
    ) -> KerrCachedReturningRadiationKernelExecution:
        plan = refinement_plan(
            self._case_path(f"{name}-unused-output"),
            self._case_path(f"{name}-unused-plan-cache"),
        )
        evaluator_id = (
            cached_module._SYNTHETIC_FORWARD_EVALUATOR_ID
            if formulation == FORWARD
            else cached_module._SYNTHETIC_RECEIVER_EVALUATOR_ID
        )
        evaluator = (
            cached_module._synthetic_forward_direction_evaluator
            if formulation == FORWARD
            else cached_module._synthetic_receiver_direction_evaluator
        )
        definition, closure, reduction = cached_module._build_cache_definition(
            formulation,
            surface=plan.surface,
            termination=plan.termination,
            annulus_edges_over_mass=plan.annulus_edges_over_mass,
            ray_options=plan.ray_options,
            surface_options=plan.surface_options,
            coarse_ray_options=None,
            coarse_surface_options=None,
            policy=plan.kernel_policy,
            area_policy=plan.area_policy,
            directions_per_task=plan.directions_per_task,
            evaluator_id=evaluator_id,
        )
        run = jobs_module.run_kernel_direction_cache(
            definition,
            evaluator,
            self._case_path(f"{name}-source-cache"),
            jobs=1,
        )
        return cached_module._reduce_cached_execution(
            definition,
            run,
            closure,
            reduction,
        )

    @staticmethod
    def _write_manifest_and_sidecar(output: Path, payload: bytes) -> None:
        (output / MANIFEST_NAME).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        (output / SIDECAR_NAME).write_bytes(
            f"{digest}  {MANIFEST_NAME}\n".encode("ascii")
        )

    @staticmethod
    def _json_copy(value):
        return json.loads(canonical_json_bytes(value))

    @classmethod
    def _fully_reseal_internal_bindings(
        cls,
        document: dict,
    ) -> tuple[bytes, str]:
        """Reseal every mutable internal digest after an adversarial rewrite."""

        producer = document["producer"]
        closure = producer["sourceRuntimeClosure"]
        entries = closure["entries"]
        for entry in entries:
            entry["bindingSha256"] = hashlib.sha256(
                (
                    f"{entry['logicalPath']}\0{entry['byteLength']}\0"
                    f"{entry['sha256']}"
                ).encode("utf-8")
            ).hexdigest()
        closure_manifest = hashlib.sha256(
            "\n".join(
                f"{entry['logicalPath']}\0{entry['byteLength']}\0{entry['sha256']}"
                for entry in entries
            ).encode("utf-8")
        ).hexdigest()
        closure["manifestSha256"] = closure_manifest
        source_bindings = sorted(entry["bindingSha256"] for entry in entries)

        definition = producer["cacheDefinition"]
        scientific_identity = definition["scientificIdentity"]
        scientific_identity["sourceClosureSha256"] = source_bindings
        definition["scientificIdentitySha256"] = checkpoint_module._canonical_sha256(
            scientific_identity
        )
        scientific_document = definition["scientificDocument"]
        scientific_document["scientificIdentity"] = cls._json_copy(
            scientific_identity
        )
        scientific_document["producerSourceHashes"] = list(source_bindings)
        scientific_job_key = checkpoint_module._canonical_sha256(scientific_document)
        definition["scientificJobKey"] = scientific_job_key

        job_spec = definition["jobSpec"]
        job_spec["parameters"]["scientificDocument"] = cls._json_copy(
            scientific_document
        )
        job_spec["parameters"]["scientificJobKey"] = scientific_job_key
        job_spec["producerSourceHashes"] = list(source_bindings)
        cache_job_key = checkpoint_module._canonical_sha256(job_spec)
        definition["jobSpecSha256"] = cache_job_key
        definition["cacheJobKey"] = cache_job_key

        operational = document["operationalExecution"]
        audit = producer["executionAudit"]
        operational["cacheJobKey"] = cache_job_key
        audit["cache_job_key"] = cache_job_key
        audit["scientific_job_key"] = scientific_job_key
        audit["source_closure_manifest_sha256"] = closure_manifest

        binding = producer["scientificBinding"]
        binding_descriptor = binding["descriptor"]
        binding_descriptor["scientificJobKey"] = scientific_job_key
        binding_descriptor["transportScientificJobKey"] = scientific_job_key
        binding_descriptor["sourceRuntimeClosure"] = cls._json_copy(closure)
        binding_sha = hashlib.sha256(
            canonical_json_bytes(binding_descriptor)[:-1]
        ).hexdigest()
        binding["descriptorSha256"] = binding_sha

        authentication = document["authenticatedConvergenceV2"]
        provenance = authentication["descriptor"]["provenance"]
        provenance["scientificBindingSha256"] = binding_sha
        provenance["scientificBindingJsonSha256"] = binding_sha
        provenance["transportScientificJobKey"] = scientific_job_key
        provenance["sourceClosureManifestSha256"] = closure_manifest
        authentication["descriptorSha256"] = hashlib.sha256(
            checkpoint_module._descriptor_bytes(authentication["descriptor"])
        ).hexdigest()

        body = {
            key: value
            for key, value in document.items()
            if key not in ("id", "integrity")
        }
        checkpoint_sha = checkpoint_module._canonical_sha256(body)
        document["integrity"]["checkpointSha256"] = checkpoint_sha
        document["id"] = (
            f"kerr-returning-radiation-refinement-{checkpoint_sha[:24]}"
        )
        payload = canonical_json_bytes(document)
        return payload, hashlib.sha256(payload).hexdigest()

    @classmethod
    def _write_fully_resealed_checkpoint(
        cls,
        output: Path,
        document: dict,
    ) -> str:
        payload, digest = cls._fully_reseal_internal_bindings(document)
        cls._write_manifest_and_sidecar(output, payload)
        return digest

    def test_real_production_forward_nonqualified_publish_and_secure_verify(
        self,
    ) -> None:
        publication = self.nonqualified_publication
        self.assertFalse(publication.qualified)
        self.assertEqual(
            {path.name for path in publication.output_directory.iterdir()},
            {MANIFEST_NAME, SIDECAR_NAME},
        )

        verified = verify_kerr_returning_radiation_refinement_checkpoint(
            publication.manifest_path
        )
        self.assertFalse(verified.qualified)
        self.assertEqual(verified.manifest_sha256, publication.manifest_sha256)
        self.assertEqual(verified.checkpoint_id, publication.checkpoint_id)
        authentication = verified.document["authenticatedConvergenceV2"]
        self.assertEqual(len(authentication["summaries"]), 5)
        self.assertEqual(len(authentication["comparisonReports"]), 4)
        self.assertFalse(authentication["qualified"])
        self.assertEqual(
            verified.document["classification"],
            "source-current-v2-non-qualified-finite-grid-checkpoint",
        )
        self.assertEqual(
            verified.document["operationalExecution"]["executedDirectionRecords"],
            448,
        )

    def test_verifier_external_expected_digest_is_a_pre_semantic_trust_anchor(
        self,
    ) -> None:
        publication = self.nonqualified_publication
        verified = verify_kerr_returning_radiation_refinement_checkpoint(
            publication.manifest_path,
            expected_manifest_sha256=publication.manifest_sha256,
        )
        self.assertEqual(verified.manifest_sha256, publication.manifest_sha256)

        with (
            patch.object(checkpoint_module, "_strict_canonical_manifest") as parser,
            patch.object(checkpoint_module, "_source_snapshot") as source,
            self.assertRaisesRegex(
                KerrReturningRadiationRefinementCheckpointError,
                "external expected SHA-256",
            ),
        ):
            verify_kerr_returning_radiation_refinement_checkpoint(
                publication.manifest_path,
                expected_manifest_sha256="0" * 64,
            )
        parser.assert_not_called()
        source.assert_not_called()

    def test_qualified_acceptance_requires_external_expected_digest(self) -> None:
        with patch.object(
            checkpoint_module,
            "_verify_checkpoint_document",
            return_value=True,
        ):
            with self.assertRaisesRegex(
                KerrReturningRadiationRefinementCheckpointError,
                "qualified checkpoint acceptance requires an external expected",
            ):
                verify_kerr_returning_radiation_refinement_checkpoint(
                    self.nonqualified_publication.manifest_path
                )

            verified = verify_kerr_returning_radiation_refinement_checkpoint(
                self.nonqualified_publication.manifest_path,
                expected_manifest_sha256=(
                    self.nonqualified_publication.manifest_sha256
                ),
            )
        self.assertTrue(verified.qualified)

    def test_manifest_cross_bindings_reject_resealed_inconsistent_claims(self) -> None:
        mutations = (
            (
                "source-closure",
                lambda document: document["producer"]["sourceRuntimeClosure"][
                    "entries"
                ][0].__setitem__("byteLength", 0),
            ),
            (
                "reduction-copy",
                lambda document: document["producer"]["scientificBinding"][
                    "descriptor"
                ]["reductionConfiguration"]["kernelPolicy"].__setitem__(
                    "rho_order", 6
                ),
            ),
            (
                "negative-counter",
                lambda document: document["operationalExecution"].__setitem__(
                    "executedTasks", -1
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                document = json.loads(
                    self.nonqualified_publication.manifest_path.read_bytes()
                )
                mutate(document)
                with self.assertRaisesRegex(
                    KerrReturningRadiationRefinementCheckpointError,
                    "internally inconsistent",
                ):
                    checkpoint_module._verify_checkpoint_document(document)

    def test_post_publication_verifier_is_the_final_effect(self) -> None:
        output = self._case_path("final-verifier-output")
        original = checkpoint_module._assert_snapshots_stable
        labels: list[str] = []

        def check_then_mutate(
            source,
            runtime,
            label,
            *,
            evaluator_mode,
            definition,
        ):
            original(
                source,
                runtime,
                label,
                evaluator_mode=evaluator_mode,
                definition=definition,
            )
            labels.append(label)
            if label == "after verification":
                (output / MANIFEST_NAME).write_bytes(b"late mutation\n")

        with (
            patch.object(
                checkpoint_module,
                "_assert_snapshots_stable",
                side_effect=check_then_mutate,
            ),
            self.assertRaisesRegex(
                KerrReturningRadiationRefinementCheckpointError,
                "external expected SHA-256|sidecar differs|canonical JSON",
            ),
        ):
            execute_kerr_returning_radiation_refinement_checkpoint(
                refinement_plan(output, self.cache)
            )
        self.assertIn("after verification", labels)

    def test_same_cache_strict_policy_reuses_all_tasks_with_zero_rays_or_evaluators(
        self,
    ) -> None:
        output = self._case_path("cache-hit-checkpoint")
        plan = refinement_plan(output, self.cache)
        with (
            patch.object(
                forward_module,
                "_trace_direction",
                side_effect=AssertionError("full cache hit retraced a ray"),
            ) as tracer,
            patch.object(
                jobs_module._KernelDirectionTaskProducer,
                "__call__",
                side_effect=AssertionError("full cache hit ran an evaluator task"),
            ) as evaluator,
        ):
            publication = execute_kerr_returning_radiation_refinement_checkpoint(plan)

        tracer.assert_not_called()
        evaluator.assert_not_called()
        self.assertFalse(publication.qualified)
        verified = verify_kerr_returning_radiation_refinement_checkpoint(
            publication.manifest_path
        )
        self.assertFalse(verified.qualified)
        reports = verified.document["authenticatedConvergenceV2"][
            "comparisonReports"
        ]
        self.assertTrue(all(item["descriptor"]["converged"] is False for item in reports))
        operational = verified.document["operationalExecution"]
        self.assertEqual(
            (
                operational["executedTasks"],
                operational["reusedTasks"],
                operational["executedDirectionRecords"],
                operational["reusedDirectionRecords"],
            ),
            (0, operational["taskCount"], 0, 448),
        )

    def test_call17_failure_leaves_one_valid_task_then_retry_resumes(self) -> None:
        output = self._case_path("call17-output")
        cache = self._case_path("call17-cache")
        plan = refinement_plan(output, cache)
        failed_calls = 0

        def fail_on_call17(*arguments):
            nonlocal failed_calls
            failed_calls += 1
            if failed_calls == 17:
                raise RuntimeError("injected classifier call 17 failure")
            return production_forward_classifier(*arguments)

        with (
            patch.object(
                forward_module,
                "_trace_direction",
                side_effect=fail_on_call17,
            ),
            self.assertRaisesRegex(RuntimeError, "classifier call 17"),
        ):
            execute_kerr_returning_radiation_refinement_checkpoint(plan)

        self.assertEqual(failed_calls, 17)
        self.assertFalse(output.exists())
        self.assertEqual(tuple(output.parent.glob(f".{output.name}.staging-*")), ())
        self.assertEqual(len(tuple(cache.rglob("*.bin"))), 1)
        self.assertEqual(len(tuple(cache.rglob("*.receipt.json"))), 1)

        resumed_calls = 0

        def count_resumed(*arguments):
            nonlocal resumed_calls
            resumed_calls += 1
            return production_forward_classifier(*arguments)

        with patch.object(
            forward_module,
            "_trace_direction",
            side_effect=count_resumed,
        ):
            publication = execute_kerr_returning_radiation_refinement_checkpoint(plan)

        verified = verify_kerr_returning_radiation_refinement_checkpoint(
            publication.manifest_path
        )
        operational = verified.document["operationalExecution"]
        self.assertEqual(resumed_calls, 432)
        self.assertEqual(
            (
                operational["executedTasks"],
                operational["reusedTasks"],
                operational["executedDirectionRecords"],
                operational["reusedDirectionRecords"],
            ),
            (operational["taskCount"] - 1, 1, 432, 16),
        )

    def test_authentication_failure_leaves_full_cache_for_zero_work_retry(
        self,
    ) -> None:
        output = self._case_path("authentication-output")
        cache = self._case_path("authentication-cache")
        plan = refinement_plan(output, cache)
        with (
            patch.object(
                forward_module,
                "_trace_direction",
                side_effect=production_forward_classifier,
            ) as first_trace,
            patch.object(
                checkpoint_module,
                "_AUTHENTICATE_CALL_ENTRY",
                side_effect=RuntimeError("injected authentication failure"),
            ) as authentication,
            self.assertRaisesRegex(RuntimeError, "authentication failure"),
        ):
            execute_kerr_returning_radiation_refinement_checkpoint(plan)

        self.assertEqual(first_trace.call_count, 448)
        self.assertEqual(authentication.call_count, 1)
        self.assertFalse(output.exists())
        self.assertEqual(tuple(output.parent.glob(f".{output.name}.staging-*")), ())
        self.assertEqual(len(tuple(cache.rglob("*.bin"))), 36)
        self.assertEqual(len(tuple(cache.rglob("*.receipt.json"))), 36)

        with (
            patch.object(
                forward_module,
                "_trace_direction",
                side_effect=AssertionError("authentication retry retraced a ray"),
            ) as tracer,
            patch.object(
                jobs_module._KernelDirectionTaskProducer,
                "__call__",
                side_effect=AssertionError(
                    "authentication retry ran an evaluator task"
                ),
            ) as evaluator,
        ):
            publication = execute_kerr_returning_radiation_refinement_checkpoint(plan)

        tracer.assert_not_called()
        evaluator.assert_not_called()
        operational = verify_kerr_returning_radiation_refinement_checkpoint(
            publication.manifest_path
        ).document["operationalExecution"]
        self.assertEqual(
            (
                operational["executedTasks"],
                operational["reusedTasks"],
                operational["executedDirectionRecords"],
                operational["reusedDirectionRecords"],
            ),
            (0, operational["taskCount"], 0, 448),
        )

    def test_execute_rejects_nonproduction_cached_sources_before_auth_or_publish(
        self,
    ) -> None:
        synthetic = self._synthetic_cached_source(FORWARD, "synthetic")
        receiver = self._synthetic_cached_source(RECEIVER, "receiver")

        class CachedExecutionSubclass(KerrCachedReturningRadiationKernelExecution):
            pass

        subclass = CachedExecutionSubclass(
            synthetic.formulation,
            synthetic.kernel,
            synthetic.cache_definition,
            synthetic.reduction_configuration,
            synthetic.job_run,
            synthetic.source_closure,
            synthetic.execution_audit,
        )
        cases = (
            ("synthetic", synthetic, "non-production evaluator"),
            ("receiver", receiver, "non-production evaluator"),
            ("foreign", object(), "foreign execution type"),
            ("subclass", subclass, "foreign execution type"),
        )
        self.assertEqual(synthetic.formulation, FORWARD)
        self.assertIn("synthetic", synthetic.execution_audit.evaluator_id)
        self.assertEqual(receiver.formulation, RECEIVER)
        self.assertIs(type(subclass), CachedExecutionSubclass)

        for label, source, message in cases:
            output = self._case_path(f"{label}-rejected-output")
            plan_cache = self._case_path(f"{label}-unused-execution-cache")
            plan = refinement_plan(output, plan_cache)
            with (
                self.subTest(label=label),
                patch.object(
                    checkpoint_module,
                    "_CACHED_INTEGRATOR_CALL_ENTRY",
                    return_value=source,
                ),
                patch.object(
                    checkpoint_module,
                    "_AUTHENTICATE_CALL_ENTRY",
                    side_effect=AssertionError("rejected source reached auth"),
                ) as authentication,
                patch.object(
                    checkpoint_module,
                    "_publish_document",
                    side_effect=AssertionError("rejected source reached publication"),
                ) as publisher,
                self.assertRaisesRegex((TypeError, RuntimeError), message),
            ):
                execute_kerr_returning_radiation_refinement_checkpoint(plan)
            authentication.assert_not_called()
            publisher.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(plan_cache.exists())

    def test_clean_subprocess_checkpoint_stays_before_every_product_stage(
        self,
    ) -> None:
        source_root = Path(__file__).resolve().parents[1]
        output = self._case_path("clean-subprocess-output")
        cache = self._case_path("clean-subprocess-cache")
        probe = f"""
import hashlib
import json
import math
import pathlib
import sys
from unittest.mock import patch

root = pathlib.Path({str(source_root)!r})
sys.path.insert(0, str(root))
from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_finite_thickness import StationaryKerrFiniteThicknessCalibration
from offline.kerr_finite_thickness_area import KerrFiniteThicknessAreaQuadraturePolicy
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
import offline.kerr_returning_radiation_kernel as forward
from offline.kerr_returning_radiation_kernel import KerrReturningRadiationKernelPolicy
from offline.kerr_returning_radiation_convergence_v2 import KerrReturningRadiationConvergenceV2Policy
from offline.kerr_returning_radiation_refinement_checkpoint import (
    KerrReturningRadiationRefinementPlan,
    execute_kerr_returning_radiation_refinement_checkpoint,
)

def classifier(*arguments):
    source_radius = arguments[7]
    tangent = arguments[9]
    receiver_face = "upper" if tangent < math.pi else "lower"
    identity = hashlib.sha256(repr(arguments[6:]).encode("utf-8")).hexdigest()
    return forward._DirectionTransport(
        f"return-{{receiver_face}}", receiver_face, source_radius,
        2.0, 4.0, identity, receiver_face, source_radius,
    )

metric = KerrKerrSchildMetric(spin_a_m=0.7)
calibration = StationaryKerrFiniteThicknessCalibration(
    dimensionless_spin=0.7,
    eddington_scaled_mass_accretion_rate=0.08,
    outer_radius_over_mass=8.0,
)
surface = KerrFiniteThicknessMultiSurface(metric, calibration)
plan = KerrReturningRadiationRefinementPlan(
    output_directory=pathlib.Path({str(output)!r}),
    cache_root=pathlib.Path({str(cache)!r}),
    surface=surface,
    termination=KerrOblateTermination.horizon_worldtube(
        metric, escape_radius_m=20.0, offset_m=0.02,
    ),
    annulus_edges_over_mass=(
        float(calibration.isco_radius_over_mass),
        float(calibration.outer_radius_over_mass),
    ),
    ray_options=RayTraceOptions(
        absolute_tolerance=5e-9, relative_tolerance=5e-9,
        initial_step=0.025, maximum_step=0.3, maximum_affine_length=100.0,
    ),
    surface_options=SurfaceEventOptions(
        absolute_tolerance=5e-9, relative_tolerance=5e-9,
        subdivisions_per_segment=4,
    ),
    kernel_policy=KerrReturningRadiationKernelPolicy(
        rho_order=4, mu_order=4, psi_count=4,
        absolute_tolerance=0.25, relative_tolerance=0.25,
        symmetry_absolute_tolerance=0.25, symmetry_relative_tolerance=0.25,
        maximum_direction_evaluations=10000,
        maximum_whole_ray_traces=40000,
    ),
    area_policy=KerrFiniteThicknessAreaQuadraturePolicy(
        gauss_legendre_order=24, relative_tolerance=1e-8,
        absolute_tolerance_over_mass_squared=1e-8,
        maximum_point_evaluations=384,
    ),
    convergence_policy=KerrReturningRadiationConvergenceV2Policy(),
    directions_per_task=16,
    jobs=1,
)
with patch.object(forward, "_trace_direction", side_effect=classifier):
    publication = execute_kerr_returning_radiation_refinement_checkpoint(plan)
document = json.loads(publication.manifest_path.read_bytes())
tokens = ("cie", "colour", "observer", "screen", "thermal", "spectral", "tile", "frame", "product")
forbidden = sorted(
    name for name in sys.modules
    if name.startswith("offline.") and any(token in name.lower() for token in tokens)
)
print(json.dumps({{
    "boundary": document["scientificBoundary"],
    "forbidden": forbidden,
    "manifest": publication.manifest_path.is_file(),
    "qualified": publication.qualified,
}}))
"""
        completed = subprocess.run(
            (sys.executable, "-I", "-B", "-c", probe),
            cwd=source_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertTrue(result["manifest"])
        self.assertFalse(result["qualified"])
        self.assertEqual(result["forbidden"], [])
        boundary = result["boundary"]
        self.assertIs(boundary["containsCieOrColourInputs"], False)
        self.assertIs(boundary["containsFrameOrTileInputs"], False)
        self.assertIs(boundary["containsThermalOrSpectralInputs"], False)

    def test_noncanonical_convergence_policy_fails_before_cache_or_rays(self) -> None:
        output = self._case_path("relaxed-policy-output")
        cache = self._case_path("relaxed-policy-cache")
        with (
            patch.object(
                checkpoint_module,
                "_CACHED_INTEGRATOR_CALL_ENTRY",
                side_effect=AssertionError("relaxed policy reached cache execution"),
            ) as integrator,
            patch.object(
                forward_module,
                "_trace_direction",
                side_effect=AssertionError("relaxed policy traced a ray"),
            ) as tracer,
            self.assertRaisesRegex(
                (TypeError, ValueError, KerrReturningRadiationRefinementCheckpointError),
                "canonical|convergence.*policy|policy.*default|strict",
            ),
        ):
            plan = refinement_plan(
                output,
                cache,
                convergence_policy=noncanonical_convergence_policy(),
            )
            execute_kerr_returning_radiation_refinement_checkpoint(plan)
        integrator.assert_not_called()
        tracer.assert_not_called()
        self.assertFalse(output.exists())
        self.assertFalse(cache.exists())

    def test_pre_promotion_failure_preserves_parent_sentinels_and_cleans_staging(
        self,
    ) -> None:
        parent = self._case_path("sentinel-parent")
        parent.mkdir()
        output = parent / "checkpoint"
        sentinels = {
            parent / MANIFEST_NAME: b"parent manifest sentinel",
            parent / SIDECAR_NAME: b"parent sidecar sentinel",
        }
        for path, payload in sentinels.items():
            path.write_bytes(payload)
        before = {
            path: (path.read_bytes(), _file_identity(path)) for path in sentinels
        }

        with (
            patch.object(
                checkpoint_module,
                "_promote_no_replace_at",
                side_effect=RuntimeError("injected pre-promotion failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "pre-promotion failure"),
        ):
            checkpoint_module._publish_document(
                output,
                {"id": "test-only-refinement-checkpoint"},
                False,
            )

        self.assertFalse(output.exists())
        self.assertEqual(tuple(parent.glob(f".{output.name}.staging-*")), ())
        for path, expected in before.items():
            self.assertEqual((path.read_bytes(), _file_identity(path)), expected)

    def test_failure_after_staging_mkdir_before_open_cleans_staging(self) -> None:
        parent = self._case_path("pre-open-parent")
        parent.mkdir()
        output = parent / "checkpoint"
        original_stat = checkpoint_module.os.stat

        def fail_staging_stat(path, *arguments, **keywords):
            if (
                type(path) is str
                and path.startswith(f".{output.name}.staging-")
                and keywords.get("follow_symlinks") is False
            ):
                raise OSError("injected staging stat failure")
            return original_stat(path, *arguments, **keywords)

        with (
            patch.object(
                checkpoint_module.os,
                "stat",
                side_effect=fail_staging_stat,
            ),
            self.assertRaisesRegex(OSError, "staging stat failure"),
        ):
            checkpoint_module._publish_document(
                output,
                {"id": "test-only-refinement-checkpoint"},
                False,
            )

        self.assertFalse(output.exists())
        self.assertEqual(tuple(parent.glob(f".{output.name}.staging-*")), ())

    def test_verifier_rejects_noncanonical_sidecar_closed_tree_and_internal_tamper(
        self,
    ) -> None:
        noncanonical = self._checkpoint_copy("noncanonical")
        document = json.loads((noncanonical / MANIFEST_NAME).read_bytes())
        pretty = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        self._write_manifest_and_sidecar(noncanonical, pretty)
        with self.assertRaisesRegex(
            KerrReturningRadiationRefinementCheckpointError,
            "not canonical JSON",
        ):
            verify_kerr_returning_radiation_refinement_checkpoint(
                noncanonical / MANIFEST_NAME
            )

        wrong_sidecar = self._checkpoint_copy("wrong-sidecar")
        (wrong_sidecar / SIDECAR_NAME).write_bytes(
            f"{'0' * 64}  {MANIFEST_NAME}\n".encode("ascii")
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationRefinementCheckpointError,
            "sidecar differs",
        ):
            verify_kerr_returning_radiation_refinement_checkpoint(
                wrong_sidecar / MANIFEST_NAME
            )

        open_tree = self._checkpoint_copy("open-tree")
        (open_tree / "undeclared.bin").write_bytes(b"undeclared")
        with self.assertRaisesRegex(
            KerrReturningRadiationRefinementCheckpointError,
            "closed two-file directory",
        ):
            verify_kerr_returning_radiation_refinement_checkpoint(
                open_tree / MANIFEST_NAME
            )

        internally_changed = self._checkpoint_copy("internally-changed")
        changed = json.loads((internally_changed / MANIFEST_NAME).read_bytes())
        changed["qualified"] = True
        self._write_manifest_and_sidecar(
            internally_changed,
            canonical_json_bytes(changed),
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationRefinementCheckpointError,
            "internally inconsistent",
        ):
            verify_kerr_returning_radiation_refinement_checkpoint(
                internally_changed / MANIFEST_NAME
            )

    def test_verifier_rejects_policy_widening_even_after_rehashed_tamper(
        self,
    ) -> None:
        output = self._checkpoint_copy("policy-widening")
        document = json.loads((output / MANIFEST_NAME).read_bytes())
        policy = document["authenticatedConvergenceV2"]["policy"]
        policy["descriptor"]["maximumNormalizedSampleWeight"] = 1.0
        policy["descriptorSha256"] = hashlib.sha256(
            checkpoint_module._descriptor_bytes(policy["descriptor"])
        ).hexdigest()
        document["authenticatedConvergenceV2"]["descriptor"][
            "policySha256"
        ] = policy["descriptorSha256"]
        descriptor = document["authenticatedConvergenceV2"]["descriptor"]
        document["authenticatedConvergenceV2"]["descriptorSha256"] = (
            hashlib.sha256(checkpoint_module._descriptor_bytes(descriptor)).hexdigest()
        )
        self._write_manifest_and_sidecar(output, canonical_json_bytes(document))

        with self.assertRaisesRegex(
            KerrReturningRadiationRefinementCheckpointError,
            "canonical strict v2 convergence policy|internally inconsistent",
        ):
            verify_kerr_returning_radiation_refinement_checkpoint(
                output / MANIFEST_NAME
            )

    def test_verifier_rejects_fully_resealed_noncanonical_source_closures(
        self,
    ) -> None:
        original = json.loads(self.nonqualified_publication.manifest_path.read_bytes())
        original_entries = original["producer"]["sourceRuntimeClosure"]["entries"]
        runtime_path = original["producer"]["sourceRuntimeClosure"][
            "numericBackendLogicalPath"
        ]
        runtime_entry = next(
            item for item in original_entries if item["logicalPath"] == runtime_path
        )
        fixed_source = original_entries[0]
        foreign_frozen = next(
            item
            for item in original["sourceFreeze"]["beforeExecution"]["artifacts"]
            if item["logicalPath"]
            not in {entry["logicalPath"] for entry in original_entries}
        )

        def remove_fixed(entries, _document):
            entries.remove(next(
                item
                for item in entries
                if item["logicalPath"] == fixed_source["logicalPath"]
            ))

        def duplicate_source(entries, _document):
            entries.insert(1, self._json_copy(fixed_source))

        def duplicate_runtime(entries, _document):
            entries.append(self._json_copy(runtime_entry))

        def replace_with_nonfixed_frozen(entries, _document):
            replacement = entries[0]
            replacement["logicalPath"] = foreign_frozen["logicalPath"]
            replacement["byteLength"] = foreign_frozen["byteLength"]
            replacement["sha256"] = foreign_frozen["sha256"]

        cases = (
            ("missing-fixed", remove_fixed),
            ("duplicate-source", duplicate_source),
            ("duplicate-runtime", duplicate_runtime),
            ("foreign-frozen", replace_with_nonfixed_frozen),
        )
        for label, mutate in cases:
            output = self._checkpoint_copy(f"resealed-closure-{label}")
            document = self._json_copy(original)
            entries = document["producer"]["sourceRuntimeClosure"]["entries"]
            mutate(entries, document)
            expected_digest = self._write_fully_resealed_checkpoint(output, document)
            with self.subTest(label=label), self.assertRaisesRegex(
                KerrReturningRadiationRefinementCheckpointError,
                "internally inconsistent",
            ):
                verify_kerr_returning_radiation_refinement_checkpoint(
                    output / MANIFEST_NAME,
                    expected_manifest_sha256=expected_digest,
                )

    def test_verifier_rejects_fully_resealed_impossible_execution_counters(
        self,
    ) -> None:
        original = json.loads(self.nonqualified_publication.manifest_path.read_bytes())

        def set_counters(document, **values):
            operational = document["operationalExecution"]
            audit = document["producer"]["executionAudit"]
            names = {
                "executedTasks": "executed_tasks",
                "reusedTasks": "reused_tasks",
                "executedDirectionRecords": "executed_direction_records",
                "reusedDirectionRecords": "reused_direction_records",
            }
            for name, value in values.items():
                operational[name] = value
                if name in names:
                    audit[names[name]] = value
            audit["authenticated_direction_records"] = (
                operational["executedDirectionRecords"]
                + operational["reusedDirectionRecords"]
            )

        cases = (
            (
                "zero-executed-tasks-with-work",
                {
                    "executedTasks": 0,
                    "reusedTasks": 36,
                    "executedDirectionRecords": 16,
                    "reusedDirectionRecords": 432,
                    "maximumInFlightObserved": 1,
                },
            ),
            (
                "zero-reused-tasks-with-records",
                {
                    "executedTasks": 36,
                    "reusedTasks": 0,
                    "executedDirectionRecords": 432,
                    "reusedDirectionRecords": 16,
                    "maximumInFlightObserved": 1,
                },
            ),
            (
                "in-flight-exceeds-executed-tasks",
                {
                    "executedTasks": 1,
                    "reusedTasks": 35,
                    "executedDirectionRecords": 16,
                    "reusedDirectionRecords": 432,
                    "maximumInFlightObserved": 2,
                },
            ),
            (
                "records-not-any-task-width-subset",
                {
                    "executedTasks": 1,
                    "reusedTasks": 35,
                    "executedDirectionRecords": 9,
                    "reusedDirectionRecords": 439,
                    "maximumInFlightObserved": 1,
                },
            ),
        )
        for label, values in cases:
            output = self._checkpoint_copy(f"resealed-counters-{label}")
            document = self._json_copy(original)
            set_counters(document, **values)
            expected_digest = self._write_fully_resealed_checkpoint(output, document)
            with self.subTest(label=label), self.assertRaisesRegex(
                KerrReturningRadiationRefinementCheckpointError,
                "internally inconsistent.*operational counters",
            ):
                verify_kerr_returning_radiation_refinement_checkpoint(
                    output / MANIFEST_NAME,
                    expected_manifest_sha256=expected_digest,
                )

    def test_verifier_rejects_file_added_during_verification(self) -> None:
        output = self._checkpoint_copy("late-extra-file")
        original_reader = checkpoint_module._read_published_at
        calls = 0

        def add_file_after_closing_read(directory_descriptor, name, maximum_bytes):
            nonlocal calls
            payload = original_reader(directory_descriptor, name, maximum_bytes)
            calls += 1
            if calls == 4:
                (output / "late-extra.bin").write_bytes(b"late")
            return payload

        with (
            patch.object(
                checkpoint_module,
                "_read_published_at",
                side_effect=add_file_after_closing_read,
            ),
            self.assertRaisesRegex(
                KerrReturningRadiationRefinementCheckpointError,
                "closed two-file",
            ),
        ):
            verify_kerr_returning_radiation_refinement_checkpoint(
                output / MANIFEST_NAME
            )
        self.assertEqual(calls, 4)

    def test_verifier_rejects_deep_json_before_semantic_or_source_runtime_work(
        self,
    ) -> None:
        output = self._case_path("deep-json")
        output.mkdir()
        depth = checkpoint_module.MAXIMUM_JSON_NESTING_DEPTH + 1
        payload = (b'{"value":' * depth) + b"null" + (b"}" * depth) + b"\n"
        self._write_manifest_and_sidecar(output, payload)

        with (
            patch.object(checkpoint_module, "_verify_checkpoint_document") as semantic,
            patch.object(checkpoint_module, "_source_snapshot") as source,
            patch.object(checkpoint_module, "_runtime_snapshot") as runtime,
            self.assertRaisesRegex(
                KerrReturningRadiationRefinementCheckpointError,
                "JSON nesting limit",
            ),
        ):
            verify_kerr_returning_radiation_refinement_checkpoint(
                output / MANIFEST_NAME
            )
        semantic.assert_not_called()
        source.assert_not_called()
        runtime.assert_not_called()

    def test_plan_and_verifier_require_exact_absolute_non_nested_paths(self) -> None:
        class ForeignPath(type(Path())):
            pass

        cache = self._case_path("exact-cache")
        output = self._case_path("exact-output")
        cases = (
            (Path("relative-output"), cache, TypeError, "absolute"),
            (str(output), cache, TypeError, "exact platform Path"),
            (ForeignPath(output), cache, TypeError, "exact platform Path"),
            (cache / "checkpoint", cache, ValueError, "non-nested"),
        )
        for selected_output, selected_cache, error_type, message in cases:
            with self.subTest(output=selected_output), self.assertRaisesRegex(
                error_type, message
            ):
                refinement_plan(selected_output, selected_cache)  # type: ignore[arg-type]

        with self.assertRaisesRegex(TypeError, "exact platform Path"):
            verify_kerr_returning_radiation_refinement_checkpoint(
                str(self.nonqualified_publication.manifest_path)  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(TypeError, "absolute"):
            verify_kerr_returning_radiation_refinement_checkpoint(
                Path(MANIFEST_NAME)
            )
        with self.assertRaisesRegex(ValueError, "must be named"):
            verify_kerr_returning_radiation_refinement_checkpoint(
                self.root / "wrong-name.json"
            )
        self.assertFalse(output.exists())
        self.assertFalse(cache.exists())

    def test_paths_reject_ancestor_and_final_symlinks_irregular_cache_and_overwrite(
        self,
    ) -> None:
        cache = self._case_path("path-cache")
        real_parent = self._case_path("real-parent")
        real_parent.mkdir()
        linked_parent = self._case_path("linked-parent")
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            refinement_plan(linked_parent / "checkpoint", cache)

        final_target = self._case_path("final-target")
        final_target.mkdir()
        final_link = self._case_path("final-link")
        final_link.symlink_to(final_target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            refinement_plan(final_link, cache)

        irregular_cache = self._case_path("irregular-cache")
        irregular_cache.write_bytes(b"not a cache directory")
        with self.assertRaisesRegex(NotADirectoryError, "must be a directory"):
            refinement_plan(self._case_path("irregular-output"), irregular_cache)

        existing_output = self._case_path("existing-output")
        existing_output.mkdir()
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            refinement_plan(existing_output, cache)

    def test_verifier_rejects_oversized_manifest_before_semantic_or_source_work(
        self,
    ) -> None:
        output = self._checkpoint_copy("oversized")
        manifest = output / MANIFEST_NAME
        with (
            patch.object(
                checkpoint_module,
                "MAXIMUM_MANIFEST_BYTES",
                manifest.stat().st_size - 1,
            ),
            patch.object(checkpoint_module, "_verify_checkpoint_document") as semantic,
            patch.object(checkpoint_module, "_source_snapshot") as source,
            self.assertRaisesRegex(
                KerrReturningRadiationRefinementCheckpointError,
                "invalid type or size|hard byte limit",
            ),
        ):
            verify_kerr_returning_radiation_refinement_checkpoint(manifest)
        semantic.assert_not_called()
        source.assert_not_called()

    def test_verifier_rejects_output_path_inode_swap(self) -> None:
        output = self._checkpoint_copy("path-swap")
        displaced = self._case_path("path-swap-displaced")
        original_reader = checkpoint_module._read_published_at
        calls = 0

        def swap_after_closing_read(directory_descriptor, name, maximum_bytes):
            nonlocal calls
            payload = original_reader(directory_descriptor, name, maximum_bytes)
            calls += 1
            if calls == 4:
                output.rename(displaced)
                shutil.copytree(displaced, output)
            return payload

        with (
            patch.object(
                checkpoint_module,
                "_read_published_at",
                side_effect=swap_after_closing_read,
            ),
            self.assertRaisesRegex(
                KerrReturningRadiationRefinementCheckpointError,
                "path identity changed",
            ),
        ):
            verify_kerr_returning_radiation_refinement_checkpoint(
                output / MANIFEST_NAME
            )
        self.assertEqual(calls, 4)

    def test_source_and_runtime_planning_drift_fail_before_cache_or_rays(
        self,
    ) -> None:
        source = checkpoint_module._source_snapshot()
        changed_source = dict(source)
        changed_source["manifestSha256"] = "0" * 64
        runtime = checkpoint_module._runtime_snapshot()
        changed_runtime = dict(runtime)
        changed_runtime["descriptorSha256"] = "0" * 64
        cases = (
            ("source", "_source_snapshot", (source, changed_source)),
            ("runtime", "_runtime_snapshot", (runtime, changed_runtime)),
        )
        for label, snapshot_name, snapshots in cases:
            output = self._case_path(f"{label}-drift-output")
            cache = self._case_path(f"{label}-drift-cache")
            plan = refinement_plan(output, cache)
            with (
                self.subTest(label=label),
                patch.object(
                    checkpoint_module,
                    snapshot_name,
                    side_effect=snapshots,
                ),
                patch.object(
                    checkpoint_module,
                    "_CACHED_INTEGRATOR_CALL_ENTRY",
                    side_effect=AssertionError("planning drift reached cache execution"),
                ) as integrator,
                patch.object(
                    forward_module,
                    "_trace_direction",
                    side_effect=AssertionError("planning drift traced a ray"),
                ) as tracer,
                self.assertRaisesRegex(
                    KerrReturningRadiationRefinementCheckpointError,
                    (
                        "source.* changed during planning"
                        if label == "source"
                        else "numeric backend changed during planning"
                    ),
                ),
            ):
                execute_kerr_returning_radiation_refinement_checkpoint(plan)
            integrator.assert_not_called()
            tracer.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(cache.exists())

    def test_hard_execution_limits_fail_before_cache_or_rays(self) -> None:
        base = refinement_plan(
            self._case_path("limit-output"),
            self._case_path("limit-cache"),
        )
        with (
            patch.object(
                checkpoint_module,
                "_CACHED_INTEGRATOR_CALL_ENTRY",
                side_effect=AssertionError("hard limit reached cache execution"),
            ) as integrator,
            patch.object(
                forward_module,
                "_trace_direction",
                side_effect=AssertionError("hard limit traced a ray"),
            ) as tracer,
        ):
            for field, value, message in (
                (
                    "directions_per_task",
                    checkpoint_module.MAXIMUM_DIRECTIONS_PER_TASK + 1,
                    "directions_per_task exceeds",
                ),
                ("jobs", jobs_module.MAXIMUM_WORKERS + 1, "jobs exceeds"),
            ):
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, message
                ):
                    replace(base, **{field: value})
        integrator.assert_not_called()
        tracer.assert_not_called()
        self.assertFalse(base.output_directory.exists())
        self.assertFalse(base.cache_root.exists())

    def test_foreign_adapter_origin_fails_before_cache_or_ray_work(self) -> None:
        output = self._case_path("origin-output")
        cache = self._case_path("origin-cache")
        shadow = (
            self.root
            / "shadow"
            / "offline"
            / "kerr_returning_radiation_convergence_v2.py"
        )
        with (
            patch.object(
                checkpoint_module._convergence,
                "__file__",
                str(shadow),
            ),
            patch.object(
                checkpoint_module,
                "_CACHED_INTEGRATOR_CALL_ENTRY",
                side_effect=AssertionError("origin failure reached cache execution"),
            ) as integrator,
            patch.object(
                forward_module,
                "_trace_direction",
                side_effect=AssertionError("origin failure traced a ray"),
            ) as tracer,
            self.assertRaisesRegex(
                KerrReturningRadiationRefinementCheckpointError,
                "another tree",
            ),
        ):
            refinement_plan(output, cache)
        integrator.assert_not_called()
        tracer.assert_not_called()
        self.assertFalse(output.exists())
        self.assertFalse(cache.exists())


if __name__ == "__main__":
    unittest.main()
