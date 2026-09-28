from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import fields, replace
import hashlib
import inspect
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
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_finite_thickness import (
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
from offline.job import canonical_json_bytes
import offline.kerr_returning_radiation_kernel as forward_module
from offline.kerr_returning_radiation_kernel import (
    KerrReturningRadiationKernelPolicy,
)
import offline.kerr_returning_radiation_kernel_cached as cached_module
from offline.kerr_returning_radiation_convergence_v2 import (
    authenticate_cached_kerr_returning_radiation_convergence_v2,
)
from offline.kerr_returning_radiation_kernel_cached import (
    KerrCachedKernelExecutionAudit,
    KerrCachedReturningRadiationKernelExecution,
    KerrKernelSourceClosureEntry,
    KerrReturningRadiationKernelCacheError,
    KerrReturningRadiationKernelCacheVerificationError,
    verify_cached_kerr_returning_radiation_kernel_execution,
)
import offline.kerr_returning_radiation_kernel_jobs as jobs_module
from offline.kerr_returning_radiation_kernel_jobs import (
    FORWARD,
    RECEIVER,
    KerrCachedKernelDirectionRecord,
    KerrKernelDirectionTaskPlan,
    KerrReturningRadiationKernelJobError,
    build_forward_kernel_scientific_identity,
    iter_cached_kernel_direction_records,
    make_kernel_direction_evaluator_input,
    make_kernel_direction_cache_definition,
    run_kernel_direction_cache,
)
import offline.kerr_returning_radiation_receiver_kernel as receiver_module


def _production_forward_classifier(*arguments):
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


def thread_executor(max_workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=max_workers)


def schema_valid_wrong_forward_evaluator(context, coordinate) -> dict:
    node = cached_module._forward_coordinate_node(context, coordinate)
    transport = cached_module._synthetic_forward_transport(node)
    return cached_module._forward_transport_document(
        node,
        transport,
        context.identity.annulus_edges_over_mass,
    )


class AlwaysEqualStr(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class AlwaysEqualInt(int):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class OfflineKerrReturningRadiationKernelCachedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temporary.name)
        self.metric = KerrKerrSchildMetric(spin_a_m=0.7)
        self.calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=0.08,
            outer_radius_over_mass=8.0,
        )
        self.surface = KerrFiniteThicknessMultiSurface(
            self.metric,
            self.calibration,
        )
        self.termination = KerrOblateTermination.horizon_worldtube(
            self.metric,
            escape_radius_m=20.0,
            offset_m=0.02,
        )
        self.ray_options = RayTraceOptions(
            absolute_tolerance=5.0e-9,
            relative_tolerance=5.0e-9,
            initial_step=0.025,
            maximum_step=0.3,
            maximum_affine_length=100.0,
        )
        self.surface_options = SurfaceEventOptions(
            absolute_tolerance=5.0e-9,
            relative_tolerance=5.0e-9,
            subdivisions_per_segment=4,
        )
        self.policy = KerrReturningRadiationKernelPolicy(
            rho_order=4,
            mu_order=4,
            psi_count=4,
            absolute_tolerance=0.25,
            relative_tolerance=0.25,
            symmetry_absolute_tolerance=0.25,
            symmetry_relative_tolerance=0.25,
            maximum_direction_evaluations=10_000,
            maximum_whole_ray_traces=40_000,
        )
        self.area_policy = KerrFiniteThicknessAreaQuadraturePolicy(
            gauss_legendre_order=24,
            relative_tolerance=1.0e-8,
            absolute_tolerance_over_mass_squared=1.0e-8,
            maximum_point_evaluations=384,
        )
        self.edges = (
            float(self.calibration.isco_radius_over_mass),
            float(self.calibration.outer_radius_over_mass),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def definition(
        self,
        formulation: str,
        directions_per_task: int = 16,
        *,
        surface=None,
        annulus_edges_over_mass=None,
        ray_options=None,
        policy=None,
        area_policy=None,
    ):
        evaluator_id = (
            cached_module._SYNTHETIC_FORWARD_EVALUATOR_ID
            if formulation == FORWARD
            else cached_module._SYNTHETIC_RECEIVER_EVALUATOR_ID
        )
        return cached_module._build_cache_definition(
            formulation,
            surface=self.surface if surface is None else surface,
            termination=self.termination,
            annulus_edges_over_mass=(
                self.edges
                if annulus_edges_over_mass is None
                else annulus_edges_over_mass
            ),
            ray_options=self.ray_options if ray_options is None else ray_options,
            surface_options=self.surface_options,
            coarse_ray_options=None,
            coarse_surface_options=None,
            policy=self.policy if policy is None else policy,
            area_policy=self.area_policy if area_policy is None else area_policy,
            directions_per_task=directions_per_task,
            evaluator_id=evaluator_id,
        )

    @staticmethod
    def evaluator(formulation: str):
        return (
            cached_module._synthetic_forward_direction_evaluator
            if formulation == FORWARD
            else cached_module._synthetic_receiver_direction_evaluator
        )

    @staticmethod
    def provider(formulation: str):
        return (
            cached_module._synthetic_forward_transport_for_sample
            if formulation == FORWARD
            else cached_module._synthetic_receiver_transport_for_sample
        )

    def direct(
        self,
        formulation: str,
        *,
        policy=None,
        area_policy=None,
    ):
        common = {
            "surface": self.surface,
            "termination": self.termination,
            "annulus_edges_over_mass": self.edges,
            "ray_options": self.ray_options,
            "surface_options": self.surface_options,
            "coarse_ray_options": None,
            "coarse_surface_options": None,
            "policy": self.policy if policy is None else policy,
            "area_policy": self.area_policy if area_policy is None else area_policy,
            "direction_transport_provider": self.provider(formulation),
        }
        if formulation == FORWARD:
            return (
                forward_module
                ._integrate_kerr_returning_radiation_energy_kernel_with_transport_provider(
                    **common
                )
            )
        return (
            receiver_module
            ._integrate_kerr_returning_radiation_receiver_energy_kernel_with_transport_provider(
                **common
            )
        )

    def cached(self, formulation: str, name: str, directions_per_task: int = 16):
        definition, closure, reduction_configuration = self.definition(
            formulation,
            directions_per_task,
        )
        run = run_kernel_direction_cache(
            definition,
            self.evaluator(formulation),
            self.root / name,
            jobs=1,
        )
        return cached_module._reduce_cached_execution(
            definition,
            run,
            closure,
            reduction_configuration,
        )

    def production_cached(
        self,
        name: str,
        directions_per_task: int = 16,
        *,
        policy=None,
        area_policy=None,
    ):
        return cached_module.integrate_cached_kerr_returning_radiation_energy_kernel(
            self.surface,
            termination=self.termination,
            annulus_edges_over_mass=self.edges,
            cache_root=self.root / name,
            ray_options=self.ray_options,
            surface_options=self.surface_options,
            policy=self.policy if policy is None else policy,
            area_policy=self.area_policy if area_policy is None else area_policy,
            directions_per_task=directions_per_task,
            jobs=1,
        )

    def existing_production_cached(
        self,
        name: str,
        directions_per_task: int = 16,
        *,
        policy=None,
        area_policy=None,
    ):
        return (
            cached_module
            .integrate_existing_cached_kerr_returning_radiation_energy_kernel(
                self.surface,
                termination=self.termination,
                annulus_edges_over_mass=self.edges,
                cache_root=self.root / name,
                ray_options=self.ray_options,
                surface_options=self.surface_options,
                policy=self.policy if policy is None else policy,
                area_policy=self.area_policy if area_policy is None else area_policy,
                directions_per_task=directions_per_task,
            )
        )

    @staticmethod
    def cache_tree_snapshot(
        cache_root: Path,
    ) -> tuple[tuple[str, str, int, str], ...]:
        if not cache_root.exists() and not cache_root.is_symlink():
            return ()
        snapshot: list[tuple[str, str, int, str]] = []
        for path in sorted(cache_root.rglob("*")):
            relative = path.relative_to(cache_root).as_posix()
            stat_result = path.lstat()
            if path.is_symlink():
                snapshot.append(
                    (relative, "symlink", stat_result.st_mode, os.readlink(path))
                )
            elif path.is_file():
                snapshot.append(
                    (
                        relative,
                        "file",
                        stat_result.st_mode,
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
            else:
                snapshot.append((relative, "directory", stat_result.st_mode, ""))
        return tuple(snapshot)

    @staticmethod
    def forge(original, **changes):
        forged = object.__new__(type(original))
        for item in fields(original):
            object.__setattr__(
                forged,
                item.name,
                changes.get(item.name, object.__getattribute__(original, item.name)),
            )
        return forged

    def scientific_key_for_closure(
        self,
        closure: tuple[KerrKernelSourceClosureEntry, ...],
    ) -> str:
        identity = build_forward_kernel_scientific_identity(
            surface=self.surface,
            termination=self.termination,
            annulus_edges_over_mass=self.edges,
            fine_ray_options=self.ray_options,
            fine_surface_options=self.surface_options,
            coarse_ray_options=None,
            coarse_surface_options=None,
            source_closure_sha256=tuple(
                item.binding_sha256 for item in closure
            ),
        )
        plan = KerrKernelDirectionTaskPlan(
            FORWARD,
            1,
            4,
            4,
            4,
            16,
        )
        return make_kernel_direction_cache_definition(
            plan,
            identity=identity,
            inputs=(
                cached_module._evaluator_input(
                    cached_module._SYNTHETIC_FORWARD_EVALUATOR_ID
                ),
            ),
        ).scientific_job_key

    def test_forward_and_receiver_cached_reducers_are_exact_direct_parity(self) -> None:
        for formulation in (FORWARD, RECEIVER):
            with self.subTest(formulation=formulation):
                direct = self.direct(formulation)
                execution = self.cached(formulation, f"parity-{formulation}")
                self.assertEqual(
                    execution.kernel.model_descriptor_sha256,
                    direct.model_descriptor_sha256,
                )
                self.assertEqual(
                    object.__getattribute__(execution.kernel, "_descriptor_json"),
                    object.__getattribute__(direct, "_descriptor_json"),
                )
                self.assertTrue(
                    execution.execution_audit.is_same_code_cache_evidence
                )
                self.assertFalse(
                    execution.execution_audit.is_independent_physics_or_geodesic_oracle
                )
                self.assertFalse(
                    execution.execution_audit.
                    is_task_reuse_history_cryptographically_authenticated
                )
                execution.revalidate_cached()

    def test_production_cached_forward_reduces_once_without_any_ray_replay(self) -> None:
        common = {
            "surface": self.surface,
            "termination": self.termination,
            "annulus_edges_over_mass": self.edges,
            "ray_options": self.ray_options,
            "surface_options": self.surface_options,
            "policy": self.policy,
            "area_policy": self.area_policy,
        }
        with patch.object(
            forward_module,
            "_trace_direction",
            side_effect=_production_forward_classifier,
        ):
            direct = forward_module.integrate_kerr_returning_radiation_energy_kernel(
                **common
            )
            direct_axisymmetric = (
                forward_module.verify_and_reduce_kerr_returning_radiation_energy_kernel(
                    direct
                )
            )
            execution = self.production_cached("production-once")

        with (
            patch.object(
                forward_module,
                "_trace_direction",
                side_effect=AssertionError("cached validation retraced a ray"),
            ) as tracer,
            patch.object(
                cached_module,
                "_rebuild_verified_cached_execution",
                wraps=cached_module._rebuild_verified_cached_execution,
            ) as verifier,
        ):
            validated = (
                cached_module
                .validate_and_reduce_cached_kerr_returning_radiation_energy_kernel(
                    execution
                )
            )
        self.assertEqual(tracer.call_count, 0)
        self.assertEqual(verifier.call_count, 1)
        self.assertEqual(
            validated.forward_kernel.model_descriptor_sha256,
            direct.model_descriptor_sha256,
        )
        self.assertEqual(
            validated.axisymmetric_kernel.canonical_descriptor_sha256,
            direct_axisymmetric.canonical_descriptor_sha256,
        )
        binding = validated.scientific_binding()
        self.assertEqual(
            binding["scientificJobKey"],
            execution.cache_definition.scientific_job_key,
        )
        self.assertEqual(
            binding["transportScientificJobKey"],
            execution.cache_definition.scientific_job_key,
        )
        self.assertEqual(
            binding["reductionConfiguration"],
            execution.reduction_configuration.as_dict(),
        )
        self.assertEqual(
            binding["reductionConfigurationSha256"],
            execution.reduction_configuration_sha256,
        )
        self.assertEqual(
            binding["forwardKernelDescriptorSha256"],
            direct.model_descriptor_sha256,
        )
        self.assertTrue(binding["evaluator"]["productionForwardOnly"])
        self.assertFalse(binding["authentication"]["cachedReplayRetracesRays"])
        serialized = validated.scientific_binding_json
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn(execution.execution_audit.cache_job_key, serialized)

    def test_existing_production_cache_integrator_is_exact_full_reuse_and_read_only(
        self,
    ) -> None:
        cache_root = self.root / "existing-production"
        with patch.object(
            forward_module,
            "_trace_direction",
            side_effect=_production_forward_classifier,
        ):
            created = self.production_cached("existing-production")
        before = self.cache_tree_snapshot(cache_root)
        with (
            patch.object(
                forward_module,
                "_trace_direction",
                side_effect=AssertionError("existing-cache integration traced a ray"),
            ) as tracer,
            patch.object(
                cached_module,
                "run_kernel_direction_cache",
                side_effect=AssertionError("existing-cache integration ran jobs"),
            ) as runner,
            patch.object(
                jobs_module,
                "_KernelDirectionTaskProducer",
                side_effect=AssertionError(
                    "existing-cache integration constructed an evaluator"
                ),
            ) as producer,
            patch.object(
                jobs_module,
                "_atomic_write_at",
                side_effect=AssertionError("existing-cache integration wrote cache"),
            ) as writer,
        ):
            reused = self.existing_production_cached("existing-production")
        self.assertEqual(tracer.call_count, 0)
        self.assertEqual(runner.call_count, 0)
        self.assertEqual(producer.call_count, 0)
        self.assertEqual(writer.call_count, 0)
        self.assertIs(type(reused), KerrCachedReturningRadiationKernelExecution)
        self.assertEqual(
            reused.cache_definition,
            created.cache_definition,
        )
        self.assertEqual(
            reused.kernel.model_descriptor_sha256,
            created.kernel.model_descriptor_sha256,
        )
        self.assertEqual(reused.job_run.executed_tasks, 0)
        self.assertEqual(
            reused.job_run.reused_tasks,
            reused.cache_definition.plan.task_count,
        )
        self.assertEqual(reused.job_run.max_in_flight_observed, 0)
        self.assertTrue(all(item.reused is True for item in reused.job_run.results))
        self.assertEqual(reused.execution_audit.executed_direction_records, 0)
        self.assertEqual(reused.execution_audit.executed_tasks, 0)
        self.assertEqual(
            reused.execution_audit.reused_direction_records,
            reused.cache_definition.plan.direction_count,
        )
        self.assertEqual(self.cache_tree_snapshot(cache_root), before)

    def test_existing_production_cache_integrator_fails_without_repair_or_rays(
        self,
    ) -> None:
        cache_root = self.root / "existing-production-corrupt"
        with patch.object(
            forward_module,
            "_trace_direction",
            side_effect=_production_forward_classifier,
        ):
            created = self.production_cached("existing-production-corrupt")
        created.job_run.results[0].receipt_path.write_bytes(b"{}\n")
        before = self.cache_tree_snapshot(cache_root)
        with (
            patch.object(
                forward_module,
                "_trace_direction",
                side_effect=AssertionError("invalid existing cache traced a ray"),
            ) as tracer,
            patch.object(
                cached_module,
                "run_kernel_direction_cache",
                side_effect=AssertionError("invalid existing cache ran jobs"),
            ) as runner,
            patch.object(
                jobs_module,
                "_KernelDirectionTaskProducer",
                side_effect=AssertionError(
                    "invalid existing cache constructed an evaluator"
                ),
            ) as producer,
            patch.object(
                jobs_module,
                "_atomic_write_at",
                side_effect=AssertionError("invalid existing cache wrote cache"),
            ) as writer,
            self.assertRaisesRegex(
                KerrReturningRadiationKernelJobError,
                "every kernel cache task",
            ),
        ):
            self.existing_production_cached("existing-production-corrupt")
        self.assertEqual(tracer.call_count, 0)
        self.assertEqual(runner.call_count, 0)
        self.assertEqual(producer.call_count, 0)
        self.assertEqual(writer.call_count, 0)
        self.assertEqual(self.cache_tree_snapshot(cache_root), before)

        missing_root = self.root / "existing-production-missing"
        with (
            patch.object(
                forward_module,
                "_trace_direction",
                side_effect=AssertionError("missing existing cache traced a ray"),
            ) as tracer,
            patch.object(
                jobs_module,
                "_atomic_write_at",
                side_effect=AssertionError("missing existing cache wrote cache"),
            ) as writer,
            self.assertRaises(KerrReturningRadiationKernelJobError),
        ):
            self.existing_production_cached("existing-production-missing")
        self.assertEqual(tracer.call_count, 0)
        self.assertEqual(writer.call_count, 0)
        self.assertFalse(missing_root.exists())

    def test_existing_cache_authenticator_rebinding_fails_before_definition_or_cache(
        self,
    ) -> None:
        cache_root = self.root / "existing-rebound"
        sentinel = object()
        for name, target, attribute in (
            (
                "local-alias",
                cached_module,
                "require_complete_existing_kernel_direction_cache",
            ),
            (
                "jobs-public",
                jobs_module,
                "require_complete_existing_kernel_direction_cache",
            ),
        ):
            with (
                self.subTest(name=name),
                patch.object(target, attribute, return_value=sentinel) as rebound,
                patch.object(
                    cached_module,
                    "_build_cache_definition",
                    side_effect=AssertionError(
                        "authenticator rebinding reached definition construction"
                    ),
                ) as builder,
                patch.object(
                    jobs_module,
                    "_open_absolute_cache_root",
                    side_effect=AssertionError(
                        "authenticator rebinding reached cache access"
                    ),
                ) as cache_opener,
                self.assertRaisesRegex(
                    KerrReturningRadiationKernelCacheError,
                    "runtime binding changed",
                ),
            ):
                self.existing_production_cached("existing-rebound")
            self.assertEqual(rebound.call_count, 0)
            self.assertEqual(builder.call_count, 0)
            self.assertEqual(cache_opener.call_count, 0)
            self.assertFalse(cache_root.exists())

    def test_existing_cache_authenticator_rejects_persistent_preimport_public_poison(
        self,
    ) -> None:
        source_root = Path(cached_module.__file__).resolve().parents[1]
        probe = inspect.cleandoc(
            f"""
            import json
            import pathlib
            import sys

            root = pathlib.Path({str(source_root)!r})
            sys.path.insert(0, str(root))
            import offline.kerr_returning_radiation_kernel_jobs as jobs

            calls = 0
            def poisoned(*args, **kwargs):
                global calls
                calls += 1
                raise AssertionError("poisoned authenticator was called")

            canonical = jobs._REQUIRE_COMPLETE_EXISTING_CACHE_CANONICAL_ENTRY
            jobs.require_complete_existing_kernel_direction_cache = poisoned
            import offline.kerr_returning_radiation_kernel_cached as cached

            try:
                cached._require_cache_only_runtime_binding()
            except cached.KerrReturningRadiationKernelCacheError as error:
                status = "rejected"
                message = str(error)
            else:
                status = "accepted"
                message = ""
            print(json.dumps(dict(
                calls=calls,
                canonicalIsPoisoned=canonical is poisoned,
                jobsPublicIsPoisoned=(
                    jobs.require_complete_existing_kernel_direction_cache
                    is poisoned
                ),
                localAliasIsPoisoned=(
                    cached.require_complete_existing_kernel_direction_cache
                    is poisoned
                ),
                message=message,
                status=status,
            )))
            """
        )
        completed = subprocess.run(
            (sys.executable, "-I", "-B", "-c", probe),
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("runtime binding changed", result["message"])
        self.assertEqual(result["calls"], 0)
        self.assertFalse(result["canonicalIsPoisoned"])
        self.assertTrue(result["jobsPublicIsPoisoned"])
        self.assertTrue(result["localAliasIsPoisoned"])

    def test_authenticated_cached_forward_replays_records_with_zero_rays(self) -> None:
        with patch.object(
            forward_module,
            "_trace_direction",
            side_effect=_production_forward_classifier,
        ):
            execution = self.production_cached("authenticated-five-pass")
        with (
            patch.object(
                forward_module,
                "_trace_direction",
                side_effect=AssertionError("cached convergence retraced a ray"),
            ) as tracer,
            patch.object(
                cached_module,
                "run_kernel_direction_cache",
                side_effect=AssertionError("cached convergence ran evaluator jobs"),
            ) as runner,
        ):
            result = authenticate_cached_kerr_returning_radiation_convergence_v2(
                execution
            )
            result.revalidate()
        self.assertEqual(tracer.call_count, 0)
        self.assertEqual(runner.call_count, 0)
        self.assertEqual(len(result.summaries), 5)
        self.assertEqual(len(result.reports), 4)
        self.assertEqual(result.source_kind, "cached")
        with patch.object(
            forward_module,
            "_trace_direction",
            side_effect=AssertionError("cached descriptor retraced a ray"),
        ):
            provenance = result.descriptor()["provenance"]
        self.assertEqual(
            provenance["forwardKernelDescriptorSha256"],
            execution.kernel.model_descriptor_sha256,
        )
        self.assertEqual(
            provenance["reductionConfigurationSha256"],
            execution.reduction_configuration_sha256,
        )
        self.assertEqual(
            provenance["transportScientificJobKey"],
            execution.cache_definition.scientific_job_key,
        )

        synthetic = self.cached(FORWARD, "authenticated-synthetic-reject")
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelCacheVerificationError,
            "production evaluator",
        ):
            authenticate_cached_kerr_returning_radiation_convergence_v2(synthetic)
        receiver = self.cached(RECEIVER, "authenticated-receiver-reject")
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelCacheVerificationError,
            "forward cached execution",
        ):
            authenticate_cached_kerr_returning_radiation_convergence_v2(receiver)

    def test_reduction_only_changes_reuse_exact_transport_and_rebuild_direct(self) -> None:
        changed_policy = replace(
            self.policy,
            absolute_tolerance=0.24,
            relative_tolerance=0.24,
            symmetry_absolute_tolerance=0.24,
            symmetry_relative_tolerance=0.24,
            maximum_direction_evaluations=9_999,
            maximum_whole_ray_traces=39_999,
        )
        changed_area = replace(
            self.area_policy,
            gauss_legendre_order=12,
            relative_tolerance=2.0e-8,
            absolute_tolerance_over_mass_squared=2.0e-8,
            maximum_point_evaluations=383,
        )
        first_definition, first_closure, first_reduction = self.definition(FORWARD)
        changed_definition, changed_closure, changed_reduction = self.definition(
            FORWARD,
            policy=changed_policy,
            area_policy=changed_area,
        )
        self.assertEqual(
            first_definition.scientific_job_key,
            changed_definition.scientific_job_key,
        )
        self.assertEqual(
            first_definition.job_spec.job_key,
            changed_definition.job_spec.job_key,
        )
        self.assertEqual(first_closure, changed_closure)
        self.assertNotEqual(
            first_reduction.descriptor_sha256,
            changed_reduction.descriptor_sha256,
        )

        cache_root = self.root / "reduction-only-reuse"
        first_run = run_kernel_direction_cache(
            first_definition,
            cached_module._synthetic_forward_direction_evaluator,
            cache_root,
            jobs=1,
        )
        with patch.object(
            cached_module,
            "_synthetic_forward_transport",
            side_effect=AssertionError("full cache hit evaluated a direction"),
        ) as evaluator_body:
            changed_run = run_kernel_direction_cache(
                changed_definition,
                cached_module._synthetic_forward_direction_evaluator,
                cache_root,
                jobs=1,
            )
        self.assertEqual(evaluator_body.call_count, 0)
        self.assertEqual(changed_run.executed_tasks, 0)
        self.assertEqual(
            changed_run.reused_tasks,
            changed_definition.plan.task_count,
        )

        first_execution = cached_module._reduce_cached_execution(
            first_definition,
            first_run,
            first_closure,
            first_reduction,
        )
        changed_execution = cached_module._reduce_cached_execution(
            changed_definition,
            changed_run,
            changed_closure,
            changed_reduction,
        )
        changed_direct = self.direct(
            FORWARD,
            policy=changed_policy,
            area_policy=changed_area,
        )
        self.assertEqual(
            changed_execution.kernel.model_descriptor_sha256,
            changed_direct.model_descriptor_sha256,
        )
        self.assertEqual(
            object.__getattribute__(changed_execution.kernel, "_descriptor_json"),
            object.__getattribute__(changed_direct, "_descriptor_json"),
        )
        self.assertNotEqual(
            first_execution.kernel.model_descriptor_sha256,
            changed_execution.kernel.model_descriptor_sha256,
        )

    def test_transport_inputs_and_plan_orders_change_both_cache_keys(self) -> None:
        original, _closure, _reduction = self.definition(FORWARD)
        changed_policy = replace(self.policy, psi_count=8)
        changed_ray = replace(self.ray_options, initial_step=0.026)
        changed_calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=0.09,
            outer_radius_over_mass=8.0,
        )
        changed_surface = KerrFiniteThicknessMultiSurface(
            self.metric,
            changed_calibration,
        )
        variants = {
            "R/M/P plan": self.definition(FORWARD, policy=changed_policy)[0],
            "ray options": self.definition(FORWARD, ray_options=changed_ray)[0],
            "surface": self.definition(FORWARD, surface=changed_surface)[0],
            "annulus edges": self.definition(
                FORWARD,
                annulus_edges_over_mass=(self.edges[0], 7.5, self.edges[1]),
            )[0],
        }
        for name, changed in variants.items():
            with self.subTest(name=name):
                self.assertNotEqual(
                    original.scientific_job_key,
                    changed.scientific_job_key,
                )
                self.assertNotEqual(
                    original.job_spec.job_key,
                    changed.job_spec.job_key,
                )

    def test_reduction_configuration_and_budget_fail_before_cache_io(self) -> None:
        direction_count = KerrKernelDirectionTaskPlan(
            FORWARD,
            1,
            self.policy.rho_order,
            self.policy.mu_order,
            self.policy.psi_count,
            16,
        ).direction_count
        low_direction_budget = replace(
            self.policy,
            maximum_direction_evaluations=direction_count - 1,
        )
        low_whole_ray_budget = replace(
            self.policy,
            maximum_whole_ray_traces=2 * direction_count - 1,
        )
        forged_policy = self.forge(
            self.policy,
            maximum_direction_evaluations=AlwaysEqualInt(
                self.policy.maximum_direction_evaluations
            ),
        )
        forged_area = self.forge(
            self.area_policy,
            maximum_point_evaluations=AlwaysEqualInt(
                self.area_policy.maximum_point_evaluations
            ),
        )
        cases = (
            ("direction", low_direction_budget, self.area_policy, "direction budget"),
            ("whole-ray", low_whole_ray_budget, self.area_policy, "whole-ray budget"),
            ("policy-type", forged_policy, self.area_policy, "exact int"),
            ("area-type", self.policy, forged_area, "exact int"),
        )
        for name, policy, area_policy, message in cases:
            with self.subTest(name=name):
                cache_root = self.root / f"pre-io-{name}"
                with (
                    patch.object(
                        cached_module,
                        "_source_closure_manifest",
                        side_effect=AssertionError("source I/O preceded validation"),
                    ) as source_manifest,
                    patch.object(
                        cached_module,
                        "make_kernel_direction_cache_definition",
                        side_effect=AssertionError("transport definition preceded validation"),
                    ) as definition_builder,
                    patch.object(
                        cached_module,
                        "run_kernel_direction_cache",
                        side_effect=AssertionError("cache I/O preceded validation"),
                    ) as cache_runner,
                ):
                    with self.assertRaisesRegex((TypeError, ValueError), message):
                        cached_module.integrate_cached_kerr_returning_radiation_energy_kernel(
                            self.surface,
                            termination=self.termination,
                            annulus_edges_over_mass=self.edges,
                            cache_root=cache_root,
                            ray_options=self.ray_options,
                            surface_options=self.surface_options,
                            policy=policy,
                            area_policy=area_policy,
                            directions_per_task=16,
                            jobs=1,
                        )
                self.assertEqual(source_manifest.call_count, 0)
                self.assertEqual(definition_builder.call_count, 0)
                self.assertEqual(cache_runner.call_count, 0)
                self.assertFalse(cache_root.exists())

    def test_verifier_rejects_reduction_budget_before_source_io(self) -> None:
        execution = self.cached(FORWARD, "verify-budget-before-source")
        low_policy = replace(
            execution.reduction_configuration.policy,
            maximum_direction_evaluations=(
                execution.cache_definition.plan.direction_count - 1
            ),
        )
        low_configuration = cached_module.KerrKernelReductionConfiguration(
            low_policy,
            execution.reduction_configuration.area_policy,
        )
        forged = self.forge(
            execution,
            reduction_configuration=low_configuration,
        )
        with patch.object(
            cached_module,
            "_source_closure_manifest",
            side_effect=AssertionError("source I/O preceded verifier budget gate"),
        ) as source_manifest, self.assertRaisesRegex(
            KerrReturningRadiationKernelCacheVerificationError,
            "reduction configuration is invalid",
        ):
            verify_cached_kerr_returning_radiation_kernel_execution(forged)
        self.assertEqual(source_manifest.call_count, 0)
    def test_certified_cached_source_rejects_test_evaluator_and_receiver(self) -> None:
        synthetic = self.cached(FORWARD, "synthetic-certification")
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelCacheVerificationError,
            "production evaluator",
        ):
            cached_module.validate_and_reduce_cached_kerr_returning_radiation_energy_kernel(
                synthetic
            )
        receiver = self.cached(RECEIVER, "receiver-certification")
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelCacheVerificationError,
            "forward cached execution",
        ):
            cached_module.validate_and_reduce_cached_kerr_returning_radiation_energy_kernel(
                receiver
            )
        projection = forward_module._coarsen_verified_kernel(
            synthetic.kernel,
            self.edges,
        )
        with self.assertRaisesRegex(TypeError, "exact cached wrapper"):
            cached_module.validate_and_reduce_cached_kerr_returning_radiation_energy_kernel(
                projection
            )

    def test_layout_root_and_reuse_do_not_change_cached_scientific_binding(self) -> None:
        with patch.object(
            forward_module,
            "_trace_direction",
            side_effect=_production_forward_classifier,
        ):
            fine = self.production_cached("binding-fine", directions_per_task=3)
            chunky = self.production_cached("binding-chunky", directions_per_task=16)
        with patch.object(
            forward_module,
            "_trace_direction",
            side_effect=AssertionError("cache hit evaluated a ray"),
        ):
            reused = self.production_cached("binding-fine", directions_per_task=3)
        bindings = tuple(
            cached_module
            .validate_and_reduce_cached_kerr_returning_radiation_energy_kernel(item)
            .scientific_binding_json
            for item in (fine, chunky, reused)
        )
        self.assertEqual(bindings[0], bindings[1])
        self.assertEqual(bindings[0], bindings[2])
        self.assertNotEqual(
            fine.cache_definition.job_spec.job_key,
            chunky.cache_definition.job_spec.job_key,
        )
        self.assertGreater(reused.execution_audit.reused_tasks, 0)

    def test_cache_hit_calls_evaluator_zero_times_and_repairs_one_chunk(self) -> None:
        definition, closure, reduction_configuration = self.definition(FORWARD)
        evaluator = cached_module._synthetic_forward_direction_evaluator

        first = run_kernel_direction_cache(
            definition,
            evaluator,
            self.root / "reuse",
            jobs=1,
        )
        resumed = run_kernel_direction_cache(
            definition,
            evaluator,
            self.root / "reuse",
            jobs=1,
        )
        self.assertEqual(resumed.executed_tasks, 0)
        self.assertEqual(resumed.reused_tasks, definition.plan.task_count)
        cached_module._reduce_cached_execution(
            definition,
            resumed,
            closure,
            reduction_configuration,
        )

        damaged = first.results[len(first.results) // 2]
        damaged.payload_path.write_bytes(b"corrupt")
        repaired = run_kernel_direction_cache(
            definition,
            evaluator,
            self.root / "reuse",
            jobs=1,
        )
        self.assertEqual(repaired.executed_tasks, 1)
        cached_module._reduce_cached_execution(
            definition,
            repaired,
            closure,
            reduction_configuration,
        )

    def test_chunk_and_worker_layout_preserve_transport_order_and_kernel(self) -> None:
        fine_definition, fine_closure, fine_reduction = self.definition(FORWARD, 3)
        chunky_definition, chunky_closure, chunky_reduction = self.definition(
            FORWARD,
            16,
        )
        self.assertEqual(
            fine_definition.scientific_job_key,
            chunky_definition.scientific_job_key,
        )
        self.assertNotEqual(
            fine_definition.job_spec.job_key,
            chunky_definition.job_spec.job_key,
        )

        evaluator = cached_module._synthetic_forward_direction_evaluator

        serial = run_kernel_direction_cache(
            fine_definition,
            evaluator,
            self.root / "fine",
            jobs=1,
        )
        parallel = run_kernel_direction_cache(
            chunky_definition,
            evaluator,
            self.root / "chunky",
            jobs=4,
            max_in_flight=3,
            executor_factory=thread_executor,
        )
        serial_records = tuple(
            iter_cached_kernel_direction_records(fine_definition, serial)
        )
        parallel_records = tuple(
            iter_cached_kernel_direction_records(chunky_definition, parallel)
        )
        self.assertEqual(
            [item.transport_sha256 for item in serial_records],
            [item.transport_sha256 for item in parallel_records],
        )
        left = cached_module._reduce_cached_execution(
            fine_definition,
            serial,
            fine_closure,
            fine_reduction,
        )
        right = cached_module._reduce_cached_execution(
            chunky_definition,
            parallel,
            chunky_closure,
            chunky_reduction,
        )
        self.assertEqual(
            left.kernel.model_descriptor_sha256,
            right.kernel.model_descriptor_sha256,
        )

    def test_public_execution_has_no_arbitrary_executor_and_evaluator_is_bound(self) -> None:
        self.assertTrue(
            cached_module.SCIENTIFIC_STATUS[
                "fixedEvaluatorCallableDescriptorCheckedBeforeCacheAccess"
            ]
        )
        self.assertFalse(
            cached_module.SCIENTIFIC_STATUS[
                "claimsProtectionFromMaliciousSameProcessCodeExecution"
            ]
        )
        self.assertNotIn(
            "direction_transport_provider",
            inspect.signature(
                forward_module.integrate_kerr_returning_radiation_energy_kernel
            ).parameters,
        )
        self.assertNotIn(
            "direction_transport_provider",
            inspect.signature(
                receiver_module.integrate_kerr_returning_radiation_receiver_energy_kernel
            ).parameters,
        )
        for function in (
            cached_module.integrate_cached_kerr_returning_radiation_energy_kernel,
            cached_module.integrate_cached_kerr_returning_radiation_receiver_energy_kernel,
        ):
            self.assertNotIn("executor_factory", inspect.signature(function).parameters)
        production = (
            cached_module.build_forward_kerr_returning_radiation_kernel_cache_definition(
                self.surface,
                termination=self.termination,
                annulus_edges_over_mass=self.edges,
                ray_options=self.ray_options,
                surface_options=self.surface_options,
                policy=self.policy,
                area_policy=self.area_policy,
                directions_per_task=16,
            )
        )
        self.assertEqual(
            cached_module._evaluator_id_from_definition(production),
            cached_module._FORWARD_EVALUATOR_ID,
        )
        wrong_cache_root = self.root / "wrong-production-evaluator"
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelJobError,
            "callable descriptor differs",
        ):
            run_kernel_direction_cache(
                production,
                schema_valid_wrong_forward_evaluator,
                wrong_cache_root,
                jobs=1,
            )
        self.assertFalse(wrong_cache_root.exists())
        cached_module._validate_definition_execution_identity(
            production,
            cached_module._source_closure_manifest(),
        )
        forward_definition, _closure, _reduction = self.definition(FORWARD)
        coordinate = forward_definition.plan.coordinates_for_task(
            forward_definition.job_spec.tasks[0]
        )[0]
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelCacheError,
            "identity is not bound",
        ):
            cached_module._evaluate_receiver_direction(
                forward_definition.scientific_context,
                coordinate,
            )

    def test_malformed_transport_schema_and_exact_types_fail_closed(self) -> None:
        definition, closure, _reduction = self.definition(FORWARD)
        run = run_kernel_direction_cache(
            definition,
            cached_module._synthetic_forward_direction_evaluator,
            self.root / "malformed-forward-source",
            jobs=1,
        )
        record = next(iter_cached_kernel_direction_records(definition, run))
        node = cached_module._forward_coordinate_node(
            definition.scientific_context,
            record.coordinate,
        )
        sample = forward_module._ForwardDirectionSample(
            node.pass_index,
            node.pass_name,
            node.source_face,
            node.source_annulus_index,
            node.rho_index,
            node.mu_index,
            node.psi_index,
            node.source_radius_over_mass,
            1.0,
            node.emission_angle_cosine,
            node.tangent_azimuth_rad,
            node.normalized_emitted_flux_weight,
        )

        def forged_forward(document):
            payload = canonical_json_bytes(document)
            return KerrCachedKernelDirectionRecord(
                record.coordinate,
                payload,
                hashlib.sha256(payload).hexdigest(),
            )

        unknown_key = json.loads(record.transport_json)
        unknown_key["transport"]["unknown"] = 1
        with self.assertRaises(KerrReturningRadiationKernelCacheError):
            cached_module._forward_transport_from_record(
                forged_forward(unknown_key),
                sample,
                self.edges,
            )

        wrong_bin = json.loads(record.transport_json)
        wrong_bin["transport"]["receiverAnnulusIndex"] = 999
        with self.assertRaises(KerrReturningRadiationKernelCacheError):
            cached_module._forward_transport_from_record(
                forged_forward(wrong_bin),
                sample,
                self.edges,
            )

        non_finite = json.loads(record.transport_json)
        non_finite["transport"]["g2"] = float("nan")
        non_finite_payload = (
            json.dumps(
                non_finite,
                allow_nan=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        with self.assertRaises(KerrReturningRadiationKernelJobError):
            KerrCachedKernelDirectionRecord(
                record.coordinate,
                non_finite_payload,
                hashlib.sha256(non_finite_payload).hexdigest(),
            )

        bad_identity = json.loads(record.transport_json)
        bad_identity["transport"]["primitiveDescriptorSha256"] = "0" * 63
        with self.assertRaises(KerrReturningRadiationKernelCacheError):
            cached_module._forward_transport_from_record(
                forged_forward(bad_identity),
                sample,
                self.edges,
            )

        receiver_definition, receiver_closure, _receiver_reduction = self.definition(
            RECEIVER
        )
        receiver_run = run_kernel_direction_cache(
            receiver_definition,
            cached_module._synthetic_receiver_direction_evaluator,
            self.root / "malformed-receiver-source",
            jobs=1,
        )
        receiver_record = next(
            iter_cached_kernel_direction_records(
                receiver_definition,
                receiver_run,
            )
        )
        receiver_node = cached_module._receiver_coordinate_node(
            receiver_definition.scientific_context,
            receiver_record.coordinate,
        )
        receiver_sample = receiver_module._ReceiverDirectionSample(
            receiver_node.pass_index,
            receiver_node.pass_name,
            receiver_node.receiver_face,
            receiver_node.receiver_annulus_index,
            receiver_node.rho_index,
            receiver_node.mu_index,
            receiver_node.psi_index,
            receiver_node.receiver_radius_over_mass,
            1.0,
            receiver_node.incidence_cosine,
            receiver_node.mu_weight,
            receiver_node.tangent_azimuth_rad,
            receiver_node.phase_cells,
        )
        integer_integrand = json.loads(receiver_record.transport_json)
        integer_integrand["transport"]["receiverIntegrand"] = 2
        integer_payload = canonical_json_bytes(integer_integrand)
        integer_record = KerrCachedKernelDirectionRecord(
            receiver_record.coordinate,
            integer_payload,
            hashlib.sha256(integer_payload).hexdigest(),
        )
        with self.assertRaises(KerrReturningRadiationKernelCacheError):
            cached_module._receiver_transport_from_record(
                integer_record,
                receiver_sample,
                self.edges,
            )

    def test_cached_verifier_rejects_forged_subclasses_and_audit(self) -> None:
        execution = self.cached(FORWARD, "verify")
        verify_cached_kerr_returning_radiation_kernel_execution(execution)

        bad_audit = self.forge(
            execution.execution_audit,
            evaluator_id=AlwaysEqualStr(
                execution.execution_audit.evaluator_id
            ),
        )
        forged_execution = self.forge(execution, execution_audit=bad_audit)
        with self.assertRaises(KerrReturningRadiationKernelCacheVerificationError):
            verify_cached_kerr_returning_radiation_kernel_execution(
                forged_execution
            )

        bad_counter = self.forge(
            execution.execution_audit,
            executed_tasks=AlwaysEqualInt(
                execution.execution_audit.executed_tasks
            ),
        )
        with self.assertRaises(KerrReturningRadiationKernelCacheVerificationError):
            verify_cached_kerr_returning_radiation_kernel_execution(
                self.forge(execution, execution_audit=bad_counter)
            )

        first = execution.source_closure[0]
        bad_entry = self.forge(
            first,
            logical_path=AlwaysEqualStr(first.logical_path),
        )
        bad_closure = (bad_entry, *execution.source_closure[1:])
        with self.assertRaises(KerrReturningRadiationKernelCacheVerificationError):
            verify_cached_kerr_returning_radiation_kernel_execution(
                self.forge(execution, source_closure=bad_closure)
            )

        bad_scientific_key = self.forge(
            execution.execution_audit,
            scientific_job_key=AlwaysEqualStr(
                execution.execution_audit.scientific_job_key
            ),
        )
        with self.assertRaises(KerrReturningRadiationKernelCacheVerificationError):
            verify_cached_kerr_returning_radiation_kernel_execution(
                self.forge(execution, execution_audit=bad_scientific_key)
            )

        forged_run = self.forge(
            execution.job_run,
            job_key=AlwaysEqualStr(execution.job_run.job_key),
        )
        with self.assertRaises(KerrReturningRadiationKernelCacheVerificationError):
            verify_cached_kerr_returning_radiation_kernel_execution(
                self.forge(execution, job_run=forged_run)
            )

        forged_run_counter = self.forge(
            execution.job_run,
            reused_tasks=AlwaysEqualInt(execution.job_run.reused_tasks),
        )
        with self.assertRaises(KerrReturningRadiationKernelCacheVerificationError):
            verify_cached_kerr_returning_radiation_kernel_execution(
                self.forge(execution, job_run=forged_run_counter)
            )

        forged_kernel = self.forge(
            execution.kernel,
            full_grid_sample_audit_sha256="0" * 64,
        )
        with self.assertRaises(KerrReturningRadiationKernelCacheVerificationError):
            verify_cached_kerr_returning_radiation_kernel_execution(
                self.forge(execution, kernel=forged_kernel)
            )

        bad_policy = self.forge(
            execution.reduction_configuration.policy,
            maximum_direction_evaluations=AlwaysEqualInt(
                execution.reduction_configuration.policy.
                maximum_direction_evaluations
            ),
        )
        bad_reduction = self.forge(
            execution.reduction_configuration,
            policy=bad_policy,
        )
        with self.assertRaises(KerrReturningRadiationKernelCacheVerificationError):
            verify_cached_kerr_returning_radiation_kernel_execution(
                self.forge(
                    execution,
                    reduction_configuration=bad_reduction,
                )
            )

        changed_reduction = cached_module.KerrKernelReductionConfiguration(
            replace(
                execution.reduction_configuration.policy,
                absolute_tolerance=0.24,
            ),
            execution.reduction_configuration.area_policy,
        )
        with self.assertRaises(
            KerrReturningRadiationKernelCacheVerificationError
        ):
            replace(
                execution,
                reduction_configuration=changed_reduction,
            )
        with self.assertRaises(KerrReturningRadiationKernelCacheVerificationError):
            verify_cached_kerr_returning_radiation_kernel_execution(
                self.forge(
                    execution,
                    reduction_configuration=changed_reduction,
                )
            )

    def test_source_closure_binds_path_runtime_and_pre_post_gate(self) -> None:
        closure = cached_module._source_closure_manifest()
        source_root = Path(cached_module.__file__).resolve().parents[1]
        declared = set(cached_module._SOURCE_CLOSURE_PATHS)
        probe = inspect.cleandoc(
            f"""
            import json
            import os
            import pathlib
            import sys

            root = pathlib.Path(os.path.abspath({str(source_root)!r}))
            sys.path.insert(0, str(root))
            from offline.geodesic import RayTraceOptions, SurfaceEventOptions
            from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
            from offline.kerr_finite_thickness import (
                StationaryKerrFiniteThicknessCalibration,
            )
            from offline.kerr_finite_thickness_area import (
                KerrFiniteThicknessAreaQuadraturePolicy,
            )
            from offline.kerr_finite_thickness_surface import (
                KerrFiniteThicknessMultiSurface,
            )
            from offline.kerr_returning_radiation_kernel import (
                KerrReturningRadiationKernelPolicy,
            )
            import offline.kerr_returning_radiation_kernel_cached as cached
            from offline.kerr_returning_radiation_kernel_jobs import FORWARD

            def loaded_project_modules():
                loaded = set()
                for module in tuple(sys.modules.values()):
                    value = getattr(module, "__file__", None)
                    if type(value) is not str:
                        continue
                    path = pathlib.Path(os.path.abspath(value))
                    try:
                        relative = path.relative_to(root)
                    except ValueError:
                        continue
                    if relative.suffix == ".py":
                        loaded.add(relative.as_posix())
                return sorted(loaded)

            imported = loaded_project_modules()
            metric = KerrKerrSchildMetric(spin_a_m=0.7)
            calibration = StationaryKerrFiniteThicknessCalibration(
                dimensionless_spin=0.7,
                eddington_scaled_mass_accretion_rate=0.08,
                outer_radius_over_mass=30.0,
            )
            surface = KerrFiniteThicknessMultiSurface(metric, calibration)
            termination = KerrOblateTermination.horizon_worldtube(
                metric,
                escape_radius_m=50.0,
                offset_m=0.02,
            )
            ray_options = RayTraceOptions(
                absolute_tolerance=5.0e-9,
                relative_tolerance=5.0e-9,
                initial_step=0.025,
                maximum_step=0.3,
                maximum_affine_length=100.0,
            )
            surface_options = SurfaceEventOptions(
                absolute_tolerance=5.0e-9,
                relative_tolerance=5.0e-9,
                subdivisions_per_segment=4,
            )
            policy = KerrReturningRadiationKernelPolicy(
                rho_order=4,
                mu_order=4,
                psi_count=4,
                absolute_tolerance=0.25,
                relative_tolerance=0.25,
                symmetry_absolute_tolerance=0.25,
                symmetry_relative_tolerance=0.25,
                maximum_direction_evaluations=10_000,
                maximum_whole_ray_traces=40_000,
            )
            area_policy = KerrFiniteThicknessAreaQuadraturePolicy(
                gauss_legendre_order=24,
                relative_tolerance=1.0e-8,
                absolute_tolerance_over_mass_squared=1.0e-8,
                maximum_point_evaluations=384,
            )
            definition, _closure, _reduction = cached._build_cache_definition(
                FORWARD,
                surface=surface,
                termination=termination,
                annulus_edges_over_mass=(
                    float(calibration.isco_radius_over_mass),
                    30.0,
                ),
                ray_options=ray_options,
                surface_options=surface_options,
                coarse_ray_options=None,
                coarse_surface_options=None,
                policy=policy,
                area_policy=area_policy,
                directions_per_task=16,
                evaluator_id=cached._FORWARD_EVALUATOR_ID,
            )
            coordinate = next(
                coordinate
                for task in definition.plan.task_keys()
                for coordinate in definition.plan.coordinates_for_task(task)
                if coordinate.ordinal == 17
            )
            document = cached._evaluate_forward_direction(
                definition.scientific_context,
                coordinate,
            )
            print(
                json.dumps(
                    dict(
                        declared=sorted(cached._SOURCE_CLOSURE_PATHS),
                        fate=document["transport"]["fate"],
                        imported=imported,
                        afterCall=loaded_project_modules(),
                    )
                )
            )
            """
        )
        completed = subprocess.run(
            (sys.executable, "-I", "-B", "-c", probe),
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
        clean_probe = json.loads(completed.stdout)
        self.assertEqual(set(clean_probe["declared"]), declared)
        self.assertEqual(set(clean_probe["imported"]), declared)
        self.assertEqual(set(clean_probe["afterCall"]), declared)
        self.assertEqual(clean_probe["fate"], "return-upper")
        with patch.object(
            forward_module,
            "__file__",
            str(
                source_root
                / "shadow"
                / "offline"
                / "kerr_returning_radiation_kernel.py"
            ),
        ):
            with self.assertRaisesRegex(
                KerrReturningRadiationKernelCacheError,
                "loaded from another tree",
            ):
                cached_module._source_closure_manifest()
        for logical_path in cached_module._SOURCE_CLOSURE_PATHS:
            tree = ast.parse((source_root / logical_path).read_text("utf-8"))
            imported_modules: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module is not None:
                    imported_modules.append(node.module)
                elif isinstance(node, ast.Import):
                    imported_modules.extend(alias.name for alias in node.names)
            for module in imported_modules:
                if module.startswith("offline."):
                    dependency = f"{module.replace('.', '/')}.py"
                    if (source_root / dependency).is_file():
                        self.assertIn(
                            dependency,
                            declared,
                            f"missing transitive source dependency from {logical_path}",
                        )
        first_path = cached_module._SOURCE_CLOSURE_PATHS[0]
        second_path = cached_module._SOURCE_CLOSURE_PATHS[1]
        first_bytes = (source_root / first_path).read_bytes()
        second_bytes = (source_root / second_path).read_bytes()
        swapped = cached_module._source_closure_manifest(
            {
                first_path: second_bytes,
                second_path: first_bytes,
            }
        )
        self.assertNotEqual(
            self.scientific_key_for_closure(closure),
            self.scientific_key_for_closure(swapped),
        )

        backend = cached_module._numeric_backend_descriptor()
        changed_backend = dict(backend)
        changed_backend["machine"] = f"{backend['machine']}-different"
        changed_runtime = cached_module._source_closure_manifest(
            numeric_backend_override=changed_backend
        )
        self.assertNotEqual(
            self.scientific_key_for_closure(closure),
            self.scientific_key_for_closure(changed_runtime),
        )
        self.assertEqual(
            closure[-1].logical_path,
            "runtime/numeric-backend.json",
        )

    def test_source_origin_root_is_frozen_against_equal_tree_rebase(self) -> None:
        original_root = cached_module._FROZEN_SOURCE_ROOT
        shadow_root = self.root / "equal-source-copy"
        for logical_path in cached_module._SOURCE_CLOSURE_PATHS:
            destination = shadow_root / logical_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original_root / logical_path, destination)
        forged_owners = tuple(
            (module_name, module, shadow_root / logical_path)
            for logical_path, (module_name, module, _expected_file) in zip(
                cached_module._SOURCE_CLOSURE_PATHS,
                cached_module._SOURCE_CLOSURE_MODULE_OWNERS,
            )
        )
        with ExitStack() as stack:
            for logical_path, (_module_name, module, _expected_file) in zip(
                cached_module._SOURCE_CLOSURE_PATHS,
                cached_module._SOURCE_CLOSURE_MODULE_OWNERS,
            ):
                stack.enter_context(
                    patch.object(module, "__file__", str(shadow_root / logical_path))
                )
            stack.enter_context(
                patch.object(cached_module, "_FROZEN_SOURCE_ROOT", shadow_root)
            )
            stack.enter_context(
                patch.object(
                    cached_module,
                    "_FROZEN_SOURCE_MODULE_FILE",
                    shadow_root
                    / "offline"
                    / "kerr_returning_radiation_kernel_cached.py",
                )
            )
            stack.enter_context(
                patch.object(
                    cached_module,
                    "_SOURCE_CLOSURE_MODULE_OWNERS",
                    forged_owners,
                )
            )
            with self.assertRaisesRegex(
                KerrReturningRadiationKernelCacheError,
                "frozen source identity",
            ):
                cached_module._source_closure_manifest()

        definition, expected_closure, reduction_configuration = self.definition(FORWARD)
        run = run_kernel_direction_cache(
            definition,
            self.evaluator(FORWARD),
            self.root / "toctou",
            jobs=1,
        )
        changed_entry = self.forge(
            expected_closure[0],
            sha256="0" * 64,
        )
        changed = (changed_entry, *expected_closure[1:])
        with patch.object(
            cached_module,
            "_source_closure_manifest",
            side_effect=(expected_closure, changed),
        ):
            with self.assertRaisesRegex(
                KerrReturningRadiationKernelCacheError,
                "changed while cached reduction",
            ):
                cached_module._reduce_cached_execution(
                    definition,
                    run,
                    expected_closure,
                    reduction_configuration,
                )

    def test_source_and_evaluator_streaming_preserve_exact_identity_without_read_bytes(
        self,
    ) -> None:
        source_root = Path(cached_module.__file__).resolve().parents[1]
        backend = cached_module._numeric_backend_descriptor()
        expected_entries = tuple(
            KerrKernelSourceClosureEntry(
                logical_path,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            )
            for logical_path in cached_module._SOURCE_CLOSURE_PATHS
            for payload in ((source_root / logical_path).read_bytes(),)
        )
        backend_payload = canonical_json_bytes(backend)
        expected_closure = (
            *expected_entries,
            KerrKernelSourceClosureEntry(
                cached_module._NUMERIC_BACKEND_LOGICAL_PATH,
                len(backend_payload),
                hashlib.sha256(backend_payload).hexdigest(),
            ),
        )
        evaluators = (
            (
                cached_module._FORWARD_EVALUATOR_ID,
                cached_module._evaluate_forward_direction,
            ),
            (
                cached_module._RECEIVER_EVALUATOR_ID,
                cached_module._evaluate_receiver_direction,
            ),
            (
                cached_module._SYNTHETIC_FORWARD_EVALUATOR_ID,
                cached_module._synthetic_forward_direction_evaluator,
            ),
            (
                cached_module._SYNTHETIC_RECEIVER_EVALUATOR_ID,
                cached_module._synthetic_receiver_direction_evaluator,
            ),
        )
        expected_evaluator_inputs = tuple(
            make_kernel_direction_evaluator_input(
                evaluator,
                implementation_id=evaluator_id,
            )
            for evaluator_id, evaluator in evaluators
        )

        with patch.object(
            Path,
            "read_bytes",
            side_effect=AssertionError("whole-file Path.read_bytes is forbidden"),
        ):
            actual_closure = cached_module._source_closure_manifest(
                numeric_backend_override=backend
            )
            actual_evaluator_inputs = tuple(
                cached_module._evaluator_input(evaluator_id)
                for evaluator_id, _evaluator in evaluators
            )

        self.assertEqual(actual_closure, expected_closure)
        self.assertEqual(actual_evaluator_inputs, expected_evaluator_inputs)
        self.assertEqual(
            cached_module._source_closure_manifest_sha256(actual_closure),
            cached_module._source_closure_manifest_sha256(expected_closure),
        )
        self.assertEqual(
            self.scientific_key_for_closure(actual_closure),
            self.scientific_key_for_closure(expected_closure),
        )

    def test_source_authentication_is_bounded_and_not_cached_across_gates(
        self,
    ) -> None:
        original = cached_module._authenticated_regular_file_sha256
        with patch.object(
            cached_module,
            "_authenticated_regular_file_sha256",
            wraps=original,
        ) as authenticate:
            closure = cached_module._source_closure_manifest(
                numeric_backend_override={"backend": "bounded-test"}
            )
            closure_calls = authenticate.call_count
            self.assertEqual(
                closure_calls,
                len(cached_module._SOURCE_CLOSURE_PATHS),
            )
            first = cached_module._evaluator_input(
                cached_module._FORWARD_EVALUATOR_ID
            )
            second = cached_module._evaluator_input(
                cached_module._FORWARD_EVALUATOR_ID
            )
            self.assertEqual(authenticate.call_count, closure_calls + 2)
        self.assertEqual(first, second)
        self.assertEqual(
            tuple(item.logical_path for item in closure[:-1]),
            cached_module._SOURCE_CLOSURE_PATHS,
        )

        definition, _closure, _reduction = self.definition(FORWARD)
        first_task = definition.plan.task_keys()[0]
        coordinate = definition.plan.coordinates_for_task(first_task)[0]
        with patch.object(
            cached_module,
            "_authenticated_regular_file_sha256",
            side_effect=AssertionError(
                "per-direction evaluator hot paths must not replay source I/O"
            ),
        ):
            document = cached_module._synthetic_forward_direction_evaluator(
                definition.scientific_context,
                coordinate,
            )
        self.assertEqual(document["formulation"], FORWARD)

        chunk_source = self.root / "chunk-source.py"
        chunk_source.write_bytes(
            b"x" * (cached_module._SOURCE_AUTHENTICATION_READ_CHUNK_BYTES + 3)
        )
        requested: list[int] = []
        original_read = os.read

        def counted_read(descriptor: int, byte_count: int) -> bytes:
            requested.append(byte_count)
            return original_read(descriptor, byte_count)

        with patch.object(cached_module.os, "read", side_effect=counted_read):
            byte_length, _digest = cached_module._authenticated_regular_file_sha256(
                chunk_source,
                label="chunk source",
                maximum_bytes=2
                * cached_module._SOURCE_AUTHENTICATION_READ_CHUNK_BYTES,
            )
        self.assertEqual(byte_length, chunk_source.stat().st_size)
        self.assertGreaterEqual(len(requested), 3)
        self.assertLessEqual(
            max(requested),
            cached_module._SOURCE_AUTHENTICATION_READ_CHUNK_BYTES,
        )

    def test_source_closure_rejects_per_file_and_total_oversize(self) -> None:
        fixture_root = self.root / "bounded-closure"
        fixture_root.mkdir(parents=True)
        (fixture_root / "large.py").write_bytes(b"123456789")
        with (
            patch.object(
                cached_module,
                "_require_exact_source_module_origins",
                return_value=fixture_root,
            ),
            patch.object(cached_module, "_SOURCE_CLOSURE_PATHS", ("large.py",)),
            patch.object(cached_module, "_SOURCE_CLOSURE_MAXIMUM_FILE_BYTES", 8),
            patch.object(cached_module, "_SOURCE_CLOSURE_MAXIMUM_TOTAL_BYTES", 64),
        ):
            with self.assertRaisesRegex(
                KerrReturningRadiationKernelCacheError,
                "fixed byte limit",
            ):
                cached_module._source_closure_manifest(
                    numeric_backend_override={}
                )

        (fixture_root / "first.py").write_bytes(b"1234")
        (fixture_root / "second.py").write_bytes(b"5678")
        with (
            patch.object(
                cached_module,
                "_require_exact_source_module_origins",
                return_value=fixture_root,
            ),
            patch.object(
                cached_module,
                "_SOURCE_CLOSURE_PATHS",
                ("first.py", "second.py"),
            ),
            patch.object(cached_module, "_SOURCE_CLOSURE_MAXIMUM_FILE_BYTES", 8),
            patch.object(cached_module, "_SOURCE_CLOSURE_MAXIMUM_TOTAL_BYTES", 9),
        ):
            with self.assertRaisesRegex(
                KerrReturningRadiationKernelCacheError,
                "fixed byte limits",
            ):
                cached_module._source_closure_manifest(
                    numeric_backend_override={}
                )

    def test_source_authentication_rejects_symlink_toctou_and_growth(self) -> None:
        target = self.root / "source-target.py"
        target.write_bytes(b"safe source")
        symlink = self.root / "source-link.py"
        symlink.symlink_to(target)
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelCacheError,
            "regular non-symlink",
        ):
            cached_module._authenticated_regular_file_sha256(
                symlink,
                label="symlink source",
                maximum_bytes=1024,
            )

        toctou = self.root / "source-toctou.py"
        toctou.write_bytes(b"a" * 32)
        original_read = os.read
        mutated = False

        def mutate_after_read(descriptor: int, byte_count: int) -> bytes:
            nonlocal mutated
            block = original_read(descriptor, byte_count)
            if block and not mutated:
                mutated = True
                with toctou.open("r+b") as handle:
                    handle.write(b"b")
                    handle.flush()
                    os.fsync(handle.fileno())
            return block

        with patch.object(
            cached_module.os,
            "read",
            side_effect=mutate_after_read,
        ):
            with self.assertRaisesRegex(
                KerrReturningRadiationKernelCacheError,
                "changed while it was authenticated",
            ):
                cached_module._authenticated_regular_file_sha256(
                    toctou,
                    label="TOCTOU source",
                    maximum_bytes=1024,
                )

        growing = self.root / "source-growth.py"
        growing.write_bytes(b"12345678")
        grew = False

        def grow_after_read(descriptor: int, byte_count: int) -> bytes:
            nonlocal grew
            block = original_read(descriptor, byte_count)
            if block and not grew:
                grew = True
                with growing.open("ab") as handle:
                    handle.write(b"9")
                    handle.flush()
                    os.fsync(handle.fileno())
            return block

        with patch.object(cached_module.os, "read", side_effect=grow_after_read):
            with self.assertRaisesRegex(
                KerrReturningRadiationKernelCacheError,
                "grew beyond its fixed byte limit",
            ):
                cached_module._authenticated_regular_file_sha256(
                    growing,
                    label="growing source",
                    maximum_bytes=8,
                )

    def test_one_real_forward_job_evaluator_traces_and_narrows_direction(self) -> None:
        calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=0.08,
            outer_radius_over_mass=30.0,
        )
        surface = KerrFiniteThicknessMultiSurface(self.metric, calibration)
        termination = KerrOblateTermination.horizon_worldtube(
            self.metric,
            escape_radius_m=50.0,
            offset_m=0.02,
        )
        definition, _closure, _reduction = cached_module._build_cache_definition(
            FORWARD,
            surface=surface,
            termination=termination,
            annulus_edges_over_mass=(
                float(calibration.isco_radius_over_mass),
                30.0,
            ),
            ray_options=self.ray_options,
            surface_options=self.surface_options,
            coarse_ray_options=None,
            coarse_surface_options=None,
            policy=self.policy,
            area_policy=self.area_policy,
            directions_per_task=16,
            evaluator_id=cached_module._FORWARD_EVALUATOR_ID,
        )
        coordinates = tuple(
            coordinate
            for key in definition.plan.task_keys()
            for coordinate in definition.plan.coordinates_for_task(key)
        )
        coordinate = next(item for item in coordinates if item.ordinal == 17)
        document = cached_module._evaluate_forward_direction(
            definition.scientific_context,
            coordinate,
        )
        self.assertEqual(
            document["schema"],
            cached_module.FORWARD_TRANSPORT_SCHEMA,
        )
        self.assertEqual(document["transport"]["fate"], "return-upper")
        self.assertEqual(
            len(document["transport"]["primitiveDescriptorSha256"]),
            64,
        )


if __name__ == "__main__":
    unittest.main()
