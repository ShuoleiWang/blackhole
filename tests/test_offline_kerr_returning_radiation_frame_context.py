from __future__ import annotations

import copy
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import offline.authenticated_artifact as artifact_module
from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_disk import (
    STEFAN_BOLTZMANN_W_M2_K4,
    StationaryNovikovThorneDisk,
)
from offline.kerr_finite_thickness import (
    RETROGRADE,
    UPPER,
    LOWER,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_finite_thickness_surface import (
    FINITE_THICKNESS_SURFACE_IDS,
    KerrFiniteThicknessMultiSurface,
)
import offline.kerr_returning_radiation_frame_context as context_module
from offline.kerr_returning_radiation_frame_context import (
    IMPLEMENTATION_ID,
    MAXIMUM_SIGMA_T4_RELATIVE_RESIDUAL,
    SCIENTIFIC_STATUS,
    KerrReturningRadiationFrameContextError,
    KerrReturningRadiationFrameContextVerificationError,
    ReturningThermalEmissionSnapshotV1,
    ValidatedReturningThermalAuthority,
    authenticate_returning_thermal_emission,
    require_authority,
)
import offline.kerr_returning_radiation_kernel as kernel_module
from offline.kerr_returning_radiation_kernel import (
    KerrReturningRadiationKernelPolicy,
    integrate_kerr_returning_radiation_energy_kernel,
)
import offline.kerr_returning_radiation_kernel_cached as cached_kernel_module
from offline.kerr_returning_radiation_kernel_cached import (
    KerrCachedReturningRadiationKernelExecution,
    integrate_cached_kerr_returning_radiation_energy_kernel,
)
from offline.kerr_returning_radiation_thermal_profile import (
    solve_axisymmetric_returning_radiation_thermal_profile,
    solve_certified_kerr_returning_radiation_thermal_profile,
)
from offline.kerr_returning_radiation_thermal_spectrum import (
    build_certified_returning_radiation_thermal_spectrum_provider,
)
from offline.novikov_thorne import PROGRADE, kerr_isco_radius_m
from offline.returning_radiation import AxisymmetricReturningRadiationKernel


class AlwaysEqualFloat(float):
    def __eq__(self, other):
        return True


class AlwaysEqualStr(str):
    def __eq__(self, other):
        return True


def _identity_for(*values: object) -> str:
    return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()


def _low_return_classifier(*arguments):
    source_radius = arguments[7]
    tangent_azimuth = arguments[9]
    receiver_face = UPPER if tangent_azimuth < math.pi else LOWER
    ratio = 0.2
    return kernel_module._DirectionTransport(
        "return-upper" if receiver_face == UPPER else "return-lower",
        receiver_face,
        source_radius,
        ratio,
        ratio * ratio,
        _identity_for(*arguments[6:]),
        receiver_face,
        source_radius,
    )


class ReturningRadiationFrameContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = KerrKerrSchildMetric(mass_m=1.0, spin_a_m=0.5)
        cls.calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.5,
            eddington_scaled_mass_accretion_rate=0.08,
            outer_radius_over_mass=12.0,
        )
        cls.surface = KerrFiniteThicknessMultiSurface(
            cls.metric,
            cls.calibration,
        )
        termination = KerrOblateTermination.horizon_worldtube(
            cls.metric,
            escape_radius_m=20.0,
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
        kernel_policy = KerrReturningRadiationKernelPolicy(
            rho_order=4,
            mu_order=4,
            psi_count=4,
            absolute_tolerance=1.0e-10,
            relative_tolerance=1.0e-10,
            symmetry_absolute_tolerance=1.0e-10,
            symmetry_relative_tolerance=1.0e-10,
            maximum_direction_evaluations=10_000,
            maximum_whole_ray_traces=40_000,
        )
        area_policy = KerrFiniteThicknessAreaQuadraturePolicy(
            gauss_legendre_order=8,
            relative_tolerance=1.0e-8,
            absolute_tolerance_over_mass_squared=1.0e-8,
            maximum_point_evaluations=128,
        )
        edges = (
            float(cls.calibration.isco_radius_over_mass),
            float(cls.calibration.outer_radius_over_mass),
        )
        cls._ray_patcher = patch.object(
            kernel_module,
            "_trace_direction",
            side_effect=_low_return_classifier,
        )
        cls._ray_patcher.start()
        cls.source = integrate_kerr_returning_radiation_energy_kernel(
            cls.surface,
            termination=termination,
            annulus_edges_over_mass=edges,
            ray_options=ray_options,
            surface_options=surface_options,
            policy=kernel_policy,
            area_policy=area_policy,
        )
        cls._temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        cls.cache_root = Path(cls._temporary.name)
        cls.cached_source = (
            integrate_cached_kerr_returning_radiation_energy_kernel(
                cls.surface,
                termination=termination,
                annulus_edges_over_mass=edges,
                cache_root=cls.cache_root,
                ray_options=ray_options,
                surface_options=surface_options,
                policy=kernel_policy,
                area_policy=area_policy,
                directions_per_task=16,
                jobs=1,
            )
        )
        cls.disk = StationaryNovikovThorneDisk(
            metric=cls.metric,
            black_hole_mass_kg=1.0e31,
            mass_accretion_rate_kg_s=1.0e18,
            orientation=PROGRADE,
            colour_correction=1.7,
        )
        cls.profile = solve_certified_kerr_returning_radiation_thermal_profile(
            cls.source,
            disk=cls.disk,
        )
        cls.cached_profile = (
            solve_certified_kerr_returning_radiation_thermal_profile(
                cls.cached_source,
                disk=cls.disk,
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._ray_patcher.stop()
        cls._temporary.cleanup()

    @staticmethod
    def _forge(value, **changes):
        forged = object.__new__(type(value))
        for name in type(value).__dataclass_fields__:
            object.__setattr__(
                forged,
                name,
                changes.get(name, object.__getattribute__(value, name)),
            )
        return forged

    def provider(self):
        return build_certified_returning_radiation_thermal_spectrum_provider(
            self.profile
        )

    def cached_provider(self):
        return build_certified_returning_radiation_thermal_spectrum_provider(
            self.cached_profile
        )

    def foreign_surface(self):
        metric = KerrKerrSchildMetric(
            mass_m=self.metric.mass_m,
            spin_a_m=self.metric.spin_a_m,
            singularity_guard_m=self.metric.singularity_guard_m,
            source_id=self.metric.source_id,
            time_dependent=self.metric.time_dependent,
        )
        calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=self.calibration.dimensionless_spin,
            eddington_scaled_mass_accretion_rate=(
                self.calibration.eddington_scaled_mass_accretion_rate
            ),
            orientation=self.calibration.orientation,
            outer_radius_over_mass=self.calibration.outer_radius_over_mass,
            thinness_gate_maximum_h_over_rho=(
                self.calibration.thinness_gate_maximum_h_over_rho
            ),
        )
        return KerrFiniteThicknessMultiSurface(metric, calibration)

    def test_scope_and_equal_reconstructed_surface_authenticate_once(self) -> None:
        self.assertEqual(SCIENTIFIC_STATUS["implementationId"], IMPLEMENTATION_ID)
        self.assertTrue(SCIENTIFIC_STATUS["oneFaceEmission"])
        self.assertFalse(SCIENTIFIC_STATUS["implicitFactorOfTwo"])
        self.assertFalse(SCIENTIFIC_STATUS["requiresSamePythonSurfaceObject"])
        self.assertTrue(
            SCIENTIFIC_STATUS["bindsOriginalForwardKernelSurfaceByExactTree"]
        )
        for key in (
            "hasIndependentPhysicsOracle",
            "isCompleteKerrbb",
            "includesReturningRadiationStressWorkFS",
            "includesSpectralRedistribution",
            "includesScatteringOrSolvedAtmosphere",
            "includesPolarization",
            "isGeneralRelativisticMagnetohydrodynamics",
        ):
            self.assertIs(SCIENTIFIC_STATUS[key], False)

        provider = self.provider()
        foreign = self.foreign_surface()
        original_verify = (
            context_module.verify_certified_returning_radiation_thermal_spectrum_provider
        )
        with patch.object(
            context_module,
            "verify_certified_returning_radiation_thermal_spectrum_provider",
            wraps=original_verify,
        ) as replay, patch.object(
            kernel_module,
            "verify_kerr_returning_radiation_energy_kernel",
            wraps=kernel_module.verify_kerr_returning_radiation_energy_kernel,
        ) as full_kernel_replay:
            authority = authenticate_returning_thermal_emission(foreign, provider)
            self.assertEqual(replay.call_count, 1)
            self.assertEqual(full_kernel_replay.call_count, 1)
            snapshot = require_authority(authority, foreign, provider)
            self.assertIs(snapshot, authority.require_live())
            with patch.object(
                type(provider),
                "emitted_specific_intensity_nu",
                side_effect=AssertionError("authority batch cannot use scalar API"),
            ), patch.object(
                type(provider),
                "emitted_specific_intensity_nu_batch",
                side_effect=AssertionError("authority batch cannot read live provider"),
            ), patch.object(
                type(provider),
                "sample",
                side_effect=AssertionError("authority batch cannot build samples"),
            ), patch.object(
                context_module,
                "_canonical_json",
                side_effect=AssertionError("authority batch cannot build JSON"),
            ), patch.object(
                context_module,
                "_sha256_text",
                side_effect=AssertionError("authority batch cannot hash evidence"),
            ):
                values = authority.emitted_specific_intensity_nu_batch(
                    self.profile.annulus_representative_radii_over_mass[0],
                    (1.0e14, 1.0e15, 1.0e16),
                )
            self.assertEqual(replay.call_count, 1)
        self.assertEqual(len(values), 3)
        self.assertTrue(all(math.isfinite(value) and value >= 0.0 for value in values))

    def test_cached_authentication_replays_cache_once_then_live_gate_is_io_free(
        self,
    ) -> None:
        provider = self.cached_provider()
        surface = self.foreign_surface()
        original_rebuild = (
            cached_kernel_module._rebuild_verified_cached_execution
        )
        with patch.object(
            cached_kernel_module,
            "_rebuild_verified_cached_execution",
            wraps=original_rebuild,
        ) as cache_replay, patch.object(
            kernel_module,
            "_trace_direction",
            side_effect=AssertionError(
                "cached authentication must not trace a camera or kernel ray"
            ),
        ) as ray_trace:
            authority = authenticate_returning_thermal_emission(
                surface,
                provider,
            )
        self.assertEqual(cache_replay.call_count, 1)
        ray_trace.assert_not_called()
        snapshot = authority.snapshot
        self.assertEqual(
            snapshot.source_kernel_kind,
            "KerrCachedReturningRadiationKernelExecution",
        )
        self.assertEqual(
            snapshot.source_evidence_descriptor_sha256,
            self.cached_profile.source_authentication_descriptor_sha256,
        )
        self.assertEqual(
            snapshot.source_kernel_descriptor_sha256,
            self.cached_source.kernel.model_descriptor_sha256,
        )
        self.assertEqual(
            snapshot.underlying_kernel_descriptor_sha256,
            self.cached_source.kernel.model_descriptor_sha256,
        )

        with patch.object(
            cached_kernel_module,
            "_rebuild_verified_cached_execution",
            side_effect=AssertionError("require_live must not replay cache records"),
        ), patch.object(
            cached_kernel_module,
            "iter_cached_kernel_direction_records",
            side_effect=AssertionError("require_live must not scan cache records"),
        ), patch.object(
            kernel_module,
            "_trace_direction",
            side_effect=AssertionError("require_live must not trace rays"),
        ):
            self.assertIs(
                require_authority(authority, surface, provider),
                snapshot,
            )
            self.assertIs(authority.require_live(), snapshot)
            values = authority.emitted_specific_intensity_nu_batch(
                self.cached_profile.annulus_representative_radii_over_mass[0],
                (1.0e14, 1.0e15),
            )
        self.assertEqual(len(values), 2)

    def test_context_independently_rejects_stale_live_axis_for_all_sources(
        self,
    ) -> None:
        projection = self.source.coarsen_annuli(self.source.annulus_edges_over_mass)
        profiles = (
            solve_certified_kerr_returning_radiation_thermal_profile(
                self.source,
                disk=self.disk,
            ),
            solve_certified_kerr_returning_radiation_thermal_profile(
                projection,
                disk=self.disk,
            ),
            solve_certified_kerr_returning_radiation_thermal_profile(
                self.cached_source,
                disk=self.disk,
            ),
        )
        for profile in profiles:
            provider = (
                build_certified_returning_radiation_thermal_spectrum_provider(
                    profile
                )
            )
            kernel = object.__getattribute__(profile, "_kernel")
            rows = object.__getattribute__(
                kernel,
                "receiver_emitter_coefficients",
            )
            self.assertTrue(any(value != 0.0 for row in rows for value in row))
            count = len(rows)
            object.__setattr__(
                kernel,
                "receiver_emitter_coefficients",
                tuple(tuple(0.0 for _ in range(count)) for _ in range(count)),
            )
            with (
                self.subTest(classification=profile.provenance_classification),
                patch.object(
                    context_module,
                    "verify_certified_returning_radiation_thermal_spectrum_provider",
                    return_value=None,
                ) as provider_replay,
                patch.object(
                    cached_kernel_module,
                    "_rebuild_verified_cached_execution",
                    side_effect=AssertionError(
                        "context live-owner gate must not scan cached records"
                    ),
                ) as cache_replay,
                patch.object(
                    kernel_module,
                    "_trace_direction",
                    side_effect=AssertionError(
                        "context live-owner gate must not trace rays"
                    ),
                ) as ray_trace,
                self.assertRaises(
                    KerrReturningRadiationFrameContextVerificationError
                ),
            ):
                authenticate_returning_thermal_emission(
                    self.foreign_surface(),
                    provider,
                )
            self.assertEqual(provider_replay.call_count, 1)
            cache_replay.assert_not_called()
            ray_trace.assert_not_called()

    def test_cached_snapshot_contains_no_execution_or_absolute_cache_path(
        self,
    ) -> None:
        snapshot = authenticate_returning_thermal_emission(
            self.foreign_surface(),
            self.cached_provider(),
        ).snapshot

        def contains_live_cache_value(value) -> bool:
            if isinstance(value, (Path, KerrCachedReturningRadiationKernelExecution)):
                return True
            if type(value) is tuple:
                return any(contains_live_cache_value(item) for item in value)
            if is_dataclass(value) and not isinstance(value, type):
                return any(
                    contains_live_cache_value(
                        object.__getattribute__(value, item.name)
                    )
                    for item in fields(value)
                )
            return False

        self.assertFalse(contains_live_cache_value(snapshot))
        payload = pickle.dumps(snapshot)
        self.assertNotIn(self.cache_root.as_posix().encode("utf-8"), payload)
        self.assertNotIn(
            b"offline.kerr_returning_radiation_kernel_cached\x94",
            payload,
        )
        restored = pickle.loads(payload)
        restored.revalidate()
        self.assertEqual(
            restored.model_descriptor_sha256,
            snapshot.model_descriptor_sha256,
        )

    def test_live_identity_accepts_only_exact_absolute_platform_path(self) -> None:
        absolute = self.cache_root.resolve()
        with patch.object(
            context_module.Path,
            "read_bytes",
            side_effect=AssertionError("live Path identity must not perform I/O"),
        ):
            tree = context_module._canonical_identity_tree(
                absolute,
                "cache_path",
            )
        self.assertEqual(tree["absolutePath"], absolute.as_posix())
        self.assertEqual(
            tree["exactType"],
            f"{type(absolute).__module__}.{type(absolute).__qualname__}",
        )
        with self.assertRaises(
            KerrReturningRadiationFrameContextVerificationError
        ):
            context_module._canonical_identity_tree(Path("relative"), "cache_path")

        class PathSubclass(type(absolute)):
            pass

        with self.assertRaises(
            KerrReturningRadiationFrameContextVerificationError
        ):
            context_module._canonical_identity_tree(
                PathSubclass(absolute),
                "cache_path",
            )

    def test_snapshot_descriptor_d20_source_closure_and_pickle_are_self_owned(
        self,
    ) -> None:
        provider = self.provider()
        authority = authenticate_returning_thermal_emission(
            self.foreign_surface(),
            provider,
        )
        snapshot = authority.snapshot
        self.assertIs(type(snapshot), ReturningThermalEmissionSnapshotV1)
        descriptor = snapshot.model_descriptor()
        self.assertTrue(
            descriptor["scientificBoundary"]["isSameCodeAuthenticatedInput"]
        )
        self.assertFalse(descriptor["scientificBoundary"]["isArtifactLoader"])
        self.assertTrue(descriptor["emission"]["oneFace"])
        certificate = snapshot.d20_certificate
        self.assertEqual(certificate.coefficient.hex(), (1.5).hex())
        self.assertEqual(certificate.normalization.hex(), (0.5).hex())
        self.assertEqual(certificate.hemisphere_flux_normalization.hex(), (1.0).hex())
        self.assertLessEqual(
            certificate.maximum_sigma_t4_relative_residual,
            MAXIMUM_SIGMA_T4_RELATIVE_RESIDUAL,
        )
        paths = tuple(entry.logical_path for entry in snapshot.source_closure)
        self.assertIn("offline/kerr_returning_radiation_frame_context.py", paths)
        self.assertIn("offline/kerr_returning_radiation_thermal_spectrum.py", paths)
        self.assertIn("offline/job.py", paths)
        self.assertIn("offline/kerr_returning_radiation_kernel_cached.py", paths)
        self.assertIn("offline/kerr_returning_radiation_kernel_jobs.py", paths)
        self.assertIn("offline/kerr_returning_radiation_receiver_kernel.py", paths)
        self.assertIn("offline/kerr_returning_radiation_receiver_rays.py", paths)
        self.assertIn("runtime/numeric-backend.json", paths)
        with patch.object(
            context_module,
            "verify_certified_returning_radiation_thermal_spectrum_provider",
            side_effect=AssertionError("snapshot revalidation must not replay"),
        ):
            restored = pickle.loads(pickle.dumps(snapshot))
            restored.revalidate()
            self.assertEqual(
                restored.model_descriptor_sha256,
                snapshot.model_descriptor_sha256,
            )

    def test_projection_source_is_bound_to_its_underlying_full_surface(self) -> None:
        projection = self.source.coarsen_annuli(self.source.annulus_edges_over_mass)
        profile = solve_certified_kerr_returning_radiation_thermal_profile(
            projection,
            disk=self.disk,
        )
        provider = build_certified_returning_radiation_thermal_spectrum_provider(
            profile
        )
        with patch.object(
            kernel_module,
            "verify_kerr_returning_radiation_kernel_projection",
            wraps=kernel_module.verify_kerr_returning_radiation_kernel_projection,
        ) as projection_replay, patch.object(
            kernel_module,
            "verify_kerr_returning_radiation_energy_kernel",
            wraps=kernel_module.verify_kerr_returning_radiation_energy_kernel,
        ) as full_kernel_replay:
            authority = authenticate_returning_thermal_emission(
                self.foreign_surface(),
                provider,
            )
        self.assertEqual(projection_replay.call_count, 1)
        self.assertEqual(full_kernel_replay.call_count, 1)
        self.assertEqual(
            authority.snapshot.source_kernel_kind,
            "KerrForwardReturningRadiationKernelProjection",
        )
        self.assertEqual(
            authority.snapshot.underlying_kernel_descriptor_sha256,
            self.source.model_descriptor_sha256,
        )

    def test_foreign_surface_one_ulp_and_face_policy_differences_fail(self) -> None:
        clean = self.foreign_surface()
        metric = clean.metric
        calibration = clean.calibration
        cases = (
            self._forge(
                clean,
                metric=self._forge(
                    metric,
                    mass_m=math.nextafter(metric.mass_m, math.inf),
                ),
            ),
            self._forge(
                clean,
                metric=self._forge(
                    metric,
                    spin_a_m=math.nextafter(metric.spin_a_m, math.inf),
                ),
            ),
            self._forge(
                clean,
                metric=self._forge(
                    metric,
                    singularity_guard_m=math.nextafter(
                        metric.singularity_guard_m,
                        math.inf,
                    ),
                ),
            ),
            self._forge(
                clean,
                calibration=self._forge(
                    calibration,
                    eddington_scaled_mass_accretion_rate=math.nextafter(
                        calibration.eddington_scaled_mass_accretion_rate,
                        math.inf,
                    ),
                ),
            ),
            self._forge(
                clean,
                calibration=self._forge(calibration, orientation=RETROGRADE),
            ),
            self._forge(
                clean,
                calibration=self._forge(
                    calibration,
                    dimensionless_spin=math.nextafter(
                        calibration.dimensionless_spin,
                        math.inf,
                    ),
                ),
            ),
            self._forge(
                clean,
                calibration=self._forge(
                    calibration,
                    outer_radius_over_mass=math.nextafter(
                        calibration.outer_radius_over_mass,
                        math.inf,
                    ),
                ),
            ),
            self._forge(
                clean,
                surface_ids=(FINITE_THICKNESS_SURFACE_IDS[0], "changed-face-id"),
            ),
        )
        for index, surface in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(
                (TypeError, ValueError, RuntimeError)
            ):
                authenticate_returning_thermal_emission(surface, self.provider())

    def test_authority_is_opaque_and_bound_to_specific_live_objects(self) -> None:
        provider = self.provider()
        surface = self.foreign_surface()
        authority = authenticate_returning_thermal_emission(surface, provider)
        equal_provider = self.provider()
        equal_surface = self.foreign_surface()
        with self.assertRaisesRegex(
            KerrReturningRadiationFrameContextVerificationError,
            "different live",
        ):
            require_authority(authority, surface, equal_provider)
        with self.assertRaisesRegex(
            KerrReturningRadiationFrameContextVerificationError,
            "different live",
        ):
            require_authority(authority, equal_surface, provider)
        fake = object.__new__(ValidatedReturningThermalAuthority)
        with self.assertRaises(KerrReturningRadiationFrameContextVerificationError):
            fake.require_live()
        with self.assertRaises(TypeError):
            ValidatedReturningThermalAuthority()
        with self.assertRaises(TypeError):
            copy.copy(authority)
        with self.assertRaises(TypeError):
            copy.deepcopy(authority)
        with self.assertRaises(TypeError):
            pickle.dumps(authority)
        with self.assertRaises(TypeError):
            class ForbiddenAuthoritySubclass(ValidatedReturningThermalAuthority):
                pass

    def test_post_auth_live_provider_profile_source_and_surfaces_fail_closed(self) -> None:
        cases = (
            ("provider", "_descriptor_sha256"),
            ("profile", "_descriptor_sha256"),
            ("source", "_descriptor_sha256"),
            ("render_surface_metric", "mass_m"),
            ("kernel_surface_metric", "mass_m"),
        )
        for target_name, field_name in cases:
            with self.subTest(target=target_name):
                provider = self.provider()
                render_surface = self.foreign_surface()
                authority = authenticate_returning_thermal_emission(
                    render_surface,
                    provider,
                )
                profile = object.__getattribute__(provider, "_profile")
                source = object.__getattribute__(profile, "_certified_source")
                targets = {
                    "provider": provider,
                    "profile": profile,
                    "source": source,
                    "render_surface_metric": render_surface.metric,
                    "kernel_surface_metric": source.surface.metric,
                }
                target = targets[target_name]
                original = object.__getattribute__(target, field_name)
                changed = (
                    "0" * 64
                    if type(original) is str
                    else math.nextafter(original, math.inf)
                )
                object.__setattr__(target, field_name, changed)
                try:
                    with self.assertRaises(
                        KerrReturningRadiationFrameContextVerificationError
                    ):
                        authority.require_live()
                finally:
                    object.__setattr__(target, field_name, original)

    def test_batch_isolated_from_live_mutation_and_frame_gate_catches_it(self) -> None:
        provider = self.provider()
        surface = self.foreign_surface()
        authority = authenticate_returning_thermal_emission(surface, provider)
        original_sha = object.__getattribute__(provider, "_descriptor_sha256")
        radius = self.profile.annulus_representative_radii_over_mass[0]
        baseline = authority.emitted_specific_intensity_nu_batch(
            radius,
            (1.0e15,),
        )
        try:
            object.__setattr__(provider, "_descriptor_sha256", "0" * 64)
            # The render hot path consumes only its closure-private frozen
            # table, so low-level mutation of the live provider cannot alter
            # the frame.  The explicit frame-boundary gate still rejects it.
            self.assertEqual(
                authority.emitted_specific_intensity_nu_batch(
                    radius,
                    (1.0e15,),
                ),
                baseline,
            )
            with self.assertRaises(KerrReturningRadiationFrameContextVerificationError):
                authority.require_live()
        finally:
            object.__setattr__(provider, "_descriptor_sha256", original_sha)

    def test_require_live_rejects_equal_foreign_profile_kernel_and_disk(self) -> None:
        provider = self.provider()
        surface = self.foreign_surface()
        authority = authenticate_returning_thermal_emission(surface, provider)
        profile = object.__getattribute__(provider, "_profile")
        for field_name, message in (
            ("_kernel", "axisymmetric-kernel identity changed"),
            ("_disk", "disk identity changed"),
        ):
            with self.subTest(field=field_name):
                original = object.__getattribute__(profile, field_name)
                equal_foreign = self._forge(original)
                self.assertIsNot(equal_foreign, original)
                object.__setattr__(profile, field_name, equal_foreign)
                try:
                    with self.assertRaisesRegex(
                        KerrReturningRadiationFrameContextVerificationError,
                        message,
                    ):
                        authority.require_live()
                finally:
                    object.__setattr__(profile, field_name, original)
                self.assertIs(authority.require_live(), authority.snapshot)

    def test_d20_sigma_t4_zero_extremes_and_non_relaxable_cap(self) -> None:
        self.assertEqual(
            context_module._sigma_t4_relative_residual(0.0, 0.0).hex(),
            (0.0).hex(),
        )
        self.assertEqual(
            context_module._sigma_t4_relative_residual(
                STEFAN_BOLTZMANN_W_M2_K4,
                1.0,
            ).hex(),
            (0.0).hex(),
        )
        accepted = context_module._sigma_t4_relative_residual(
            math.nextafter(STEFAN_BOLTZMANN_W_M2_K4, math.inf),
            1.0,
        )
        self.assertLessEqual(accepted, MAXIMUM_SIGMA_T4_RELATIVE_RESIDUAL)
        for flux, temperature in (
            (-0.0, 0.0),
            (0.0, -0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (STEFAN_BOLTZMANN_W_M2_K4 * (1.0 + 1.0e-9), 1.0),
            (1.0, 1.0e-100),
            (1.0, 1.0e100),
        ):
            with self.subTest(flux=flux, temperature=temperature), self.assertRaises(
                (TypeError, ValueError, RuntimeError)
            ):
                context_module._sigma_t4_relative_residual(flux, temperature)

    def test_source_closure_pre_post_gate_and_runtime_identity(self) -> None:
        with patch.object(
            context_module.Path,
            "read_bytes",
            side_effect=AssertionError("source hashing must be streaming"),
        ), patch.object(
            artifact_module.os,
            "fstat",
            wraps=artifact_module.os.fstat,
        ) as fstat, patch.object(
            artifact_module.os,
            "read",
            wraps=artifact_module.os.read,
        ) as read:
            closure = context_module._source_closure_manifest()
        self.assertIn(
            "offline/authenticated_artifact.py",
            {entry.logical_path for entry in closure},
        )
        self.assertIn(
            "offline/__init__.py",
            {entry.logical_path for entry in closure},
        )
        self.assertIn(
            "offline/radiative_transfer.py",
            {entry.logical_path for entry in closure},
        )
        source_root = Path(context_module.__file__).resolve().parents[1]
        probe = (
            "import json,pathlib,sys;"
            f"root=pathlib.Path({str(source_root)!r}).resolve();"
            "sys.path.insert(0,str(root));"
            "import offline.kerr_returning_radiation_frame_context as m;"
            "loaded=set();"
            "[(loaded.add(pathlib.Path(p).resolve().relative_to(root).as_posix()) "
            "if pathlib.Path(p).resolve().is_relative_to(root) and "
            "pathlib.Path(p).suffix == '.py' else None) "
            "for item in tuple(sys.modules.values()) "
            "for p in (getattr(item,'__file__',None),) if type(p) is str];"
            "print(json.dumps(sorted(loaded)))"
        )
        completed = subprocess.run(
            (sys.executable, "-I", "-B", "-c", probe),
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            set(json.loads(completed.stdout)),
            set(context_module._SOURCE_CLOSURE_PATHS),
        )
        with patch.object(
            kernel_module,
            "__file__",
            str(
                source_root
                / "shadow"
                / "offline"
                / "kerr_returning_radiation_kernel.py"
            ),
        ):
            with self.assertRaisesRegex(
                KerrReturningRadiationFrameContextError,
                "loaded from another tree",
            ):
                context_module._source_closure_manifest()
        self.assertGreaterEqual(
            fstat.call_count,
            4 * (len(context_module._SOURCE_CLOSURE_PATHS) + 2),
        )
        self.assertGreaterEqual(
            read.call_count,
            len(context_module._SOURCE_CLOSURE_PATHS) + 2,
        )
        first = closure[0]
        changed = (
            self._forge(first, sha256="0" * 64),
            *closure[1:],
        )
        with patch.object(
            context_module,
            "_source_closure_manifest",
            side_effect=(closure, changed),
        ), self.assertRaisesRegex(
            KerrReturningRadiationFrameContextVerificationError,
            "changed while authentication",
        ):
            authenticate_returning_thermal_emission(
                self.foreign_surface(),
                self.provider(),
            )
        backend = context_module._numeric_backend_descriptor()
        for logical_path in context_module._SOURCE_CLOSURE_PATHS:
            entry = next(
                item for item in closure if item.logical_path == logical_path
            )
            payload = (
                Path(context_module.__file__).resolve().parents[1] / logical_path
            ).read_bytes()
            self.assertEqual(entry.byte_length, len(payload))
            self.assertEqual(entry.sha256, hashlib.sha256(payload).hexdigest())
        for descriptor, path in (
            (backend["mathExtension"], Path(math.__file__).resolve(strict=True)),
            (
                backend["pythonExecutable"],
                Path(context_module.sys.executable).resolve(strict=True),
            ),
        ):
            payload = path.read_bytes()
            self.assertEqual(descriptor["artifactName"], path.name)
            self.assertEqual(descriptor["byteLength"], len(payload))
            self.assertEqual(descriptor["sha256"], hashlib.sha256(payload).hexdigest())
        changed_backend = dict(backend)
        changed_backend["machine"] = f"{backend['machine']}-different"
        self.assertNotEqual(
            context_module._source_closure_manifest_sha256(closure),
            context_module._source_closure_manifest_sha256(
                context_module._source_closure_manifest(
                    numeric_backend_override=changed_backend
                )
            ),
        )

    def test_source_origin_root_is_frozen_against_equal_tree_rebase(self) -> None:
        original_root = context_module._FROZEN_SOURCE_ROOT
        shadow_root = Path(self._temporary.name) / "equal-context-source-copy"
        for logical_path in context_module._SOURCE_CLOSURE_PATHS:
            destination = shadow_root / logical_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original_root / logical_path, destination)
        forged_owners = tuple(
            (module_name, module, shadow_root / logical_path)
            for logical_path, (module_name, module, _expected_file) in zip(
                context_module._SOURCE_CLOSURE_PATHS,
                context_module._SOURCE_CLOSURE_MODULE_OWNERS,
            )
        )
        with ExitStack() as stack:
            for logical_path, (_module_name, module, _expected_file) in zip(
                context_module._SOURCE_CLOSURE_PATHS,
                context_module._SOURCE_CLOSURE_MODULE_OWNERS,
            ):
                stack.enter_context(
                    patch.object(
                        module,
                        "__file__",
                        str(shadow_root / logical_path),
                    )
                )
            stack.enter_context(
                patch.object(context_module, "_FROZEN_SOURCE_ROOT", shadow_root)
            )
            stack.enter_context(
                patch.object(
                    context_module,
                    "_FROZEN_SOURCE_MODULE_FILE",
                    shadow_root
                    / "offline"
                    / "kerr_returning_radiation_frame_context.py",
                )
            )
            stack.enter_context(
                patch.object(
                    context_module,
                    "_SOURCE_CLOSURE_MODULE_OWNERS",
                    forged_owners,
                )
            )
            with self.assertRaisesRegex(
                KerrReturningRadiationFrameContextError,
                "frozen source identity",
            ):
                context_module._source_closure_manifest()

    def test_runtime_artifact_cap_precedes_data_read(self) -> None:
        with patch.object(
            context_module,
            "MAXIMUM_RUNTIME_ARTIFACT_BYTE_LENGTH",
            0,
        ), patch.object(
            artifact_module.os,
            "read",
            side_effect=AssertionError("runtime byte cap must precede data reads"),
        ) as read, self.assertRaises(
            context_module.KerrReturningRadiationFrameContextError
        ):
            context_module._numeric_backend_descriptor()
        read.assert_not_called()

    def test_runtime_locator_symlink_swap_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(strict=True)
            first = root / "runtime-first.bin"
            second = root / "runtime-second.bin"
            locator = root / "runtime-current.bin"
            first.write_bytes(b"first-runtime")
            second.write_bytes(b"second-runtime")
            locator.symlink_to(first)
            authenticate = context_module.authenticate_stable_artifact
            swapped = False

            def authenticate_then_swap(*args, **kwargs):
                nonlocal swapped
                result = authenticate(*args, **kwargs)
                if not swapped:
                    swapped = True
                    locator.unlink()
                    locator.symlink_to(second)
                return result

            with patch.object(
                context_module,
                "authenticate_stable_artifact",
                side_effect=authenticate_then_swap,
            ), self.assertRaisesRegex(
                context_module.KerrReturningRadiationFrameContextError,
                "cannot authenticate",
            ):
                context_module._runtime_artifact_descriptor(
                    os.fspath(locator),
                    "test runtime",
                )

    def test_snapshot_source_closure_exact_fields_and_resource_caps(self) -> None:
        snapshot = authenticate_returning_thermal_emission(
            self.foreign_surface(),
            self.provider(),
        ).snapshot
        first = snapshot.source_closure[0]
        forged_string_entry = self._forge(
            first,
            logical_path=AlwaysEqualStr(first.logical_path),
        )
        with self.assertRaises(TypeError):
            self._forge(
                snapshot,
                source_closure=(
                    forged_string_entry,
                    *snapshot.source_closure[1:],
                ),
            ).revalidate()

        too_many = tuple(
            first
            for _ in range(
                context_module.MAXIMUM_SOURCE_CLOSURE_ENTRY_COUNT + 1
            )
        )
        with self.assertRaises(
            KerrReturningRadiationFrameContextVerificationError
        ):
            self._forge(snapshot, source_closure=too_many).revalidate()

        oversized = self._forge(
            first,
            byte_length=(
                context_module.MAXIMUM_SOURCE_CLOSURE_ENTRY_BYTE_LENGTH + 1
            ),
        )
        with self.assertRaises(
            KerrReturningRadiationFrameContextVerificationError
        ):
            self._forge(
                snapshot,
                source_closure=(oversized, *snapshot.source_closure[1:]),
            ).revalidate()

        per_entry_total_attack = (
            context_module.MAXIMUM_SOURCE_CLOSURE_TOTAL_BYTE_LENGTH
            // len(snapshot.source_closure)
            + 1
        )
        total_oversized = tuple(
            self._forge(entry, byte_length=per_entry_total_attack)
            for entry in snapshot.source_closure
        )
        self.assertLessEqual(
            per_entry_total_attack,
            context_module.MAXIMUM_SOURCE_CLOSURE_ENTRY_BYTE_LENGTH,
        )
        with self.assertRaises(
            KerrReturningRadiationFrameContextVerificationError
        ):
            self._forge(snapshot, source_closure=total_oversized).revalidate()

        excessive_fluxes = tuple(
            0.0
            for _ in range(
                context_module.MAXIMUM_AUTHENTICATED_ANNULUS_COUNT + 1
            )
        )
        with self.assertRaises(
            KerrReturningRadiationFrameContextVerificationError
        ):
            self._forge(
                snapshot,
                outgoing_flux_w_m2=excessive_fluxes,
            ).revalidate()

    def test_annulus_resource_gate_runs_before_source_io_or_physics_replay(
        self,
    ) -> None:
        provider = self.provider()
        count = context_module.MAXIMUM_AUTHENTICATED_ANNULUS_COUNT + 1
        forged_provider = self._forge(
            provider,
            annulus_edges_over_mass=tuple(
                float(index + 1) for index in range(count + 1)
            ),
            outgoing_effective_temperature_k=tuple(0.0 for _ in range(count)),
        )
        with patch.object(
            context_module,
            "_source_closure_manifest",
            side_effect=AssertionError("source closure must be gated"),
        ) as closure_read, patch.object(
            context_module,
            "verify_certified_returning_radiation_thermal_spectrum_provider",
            side_effect=AssertionError("physics replay must be gated"),
        ) as replay:
            with self.assertRaises(
                KerrReturningRadiationFrameContextVerificationError
            ):
                authenticate_returning_thermal_emission(
                    self.foreign_surface(),
                    forged_provider,
                )
        closure_read.assert_not_called()
        replay.assert_not_called()

    def test_external_profile_snapshot_and_exact_type_attacks_fail(self) -> None:
        provider = self.provider()
        edges = self.source.annulus_edges_over_mass
        external_kernel = AxisymmetricReturningRadiationKernel(
            annulus_radii_over_mass=(0.5 * math.fsum(edges),),
            receiver_emitter_coefficients=((0.0,),),
            ray_kernel_producer_id="external-test/v1",
        )
        external_profile = solve_axisymmetric_returning_radiation_thermal_profile(
            external_kernel,
            annulus_edges_over_mass=edges,
            annulus_proper_areas_over_mass_squared=(100.0,),
            disk=self.disk,
        )
        forged_provider = self._forge(provider, _profile=external_profile)
        with self.assertRaises(KerrReturningRadiationFrameContextVerificationError):
            authenticate_returning_thermal_emission(
                self.foreign_surface(),
                forged_provider,
            )
        authority = authenticate_returning_thermal_emission(
            self.foreign_surface(),
            provider,
        )
        snapshot = authority.snapshot
        bad_edges = self._forge(
            snapshot,
            annulus_edges_over_mass=[*snapshot.annulus_edges_over_mass],
        )
        with self.assertRaises(TypeError):
            bad_edges.revalidate()
        bad_float = self._forge(
            snapshot,
            colour_correction=AlwaysEqualFloat(snapshot.colour_correction),
        )
        with self.assertRaises(TypeError):
            bad_float.revalidate()
        bad_sha = self._forge(
            snapshot,
            provider_descriptor_sha256=AlwaysEqualStr(
                snapshot.provider_descriptor_sha256
            ),
        )
        with self.assertRaises(TypeError):
            bad_sha.revalidate()
        radius = self.profile.annulus_representative_radii_over_mass[0]
        with self.assertRaises(TypeError):
            authority.emitted_specific_intensity_nu_batch(radius, [1.0e15])
        with self.assertRaises(TypeError):
            authority.emitted_specific_intensity_nu_batch(
                AlwaysEqualFloat(radius),
                (1.0e15,),
            )
        with self.assertRaises(TypeError):
            authority.emitted_specific_intensity_nu_batch(
                radius,
                (AlwaysEqualFloat(1.0e15),),
            )


if __name__ == "__main__":
    unittest.main()
