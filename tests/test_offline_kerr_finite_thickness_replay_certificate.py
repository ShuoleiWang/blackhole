from __future__ import annotations

import copy
from dataclasses import replace
import pickle
import unittest
from unittest import mock

from offline.geodesic import RayTraceOptions, SurfaceEventOptions, trace_null_geodesic
from offline.kerr import (
    KerrKerrSchildMetric,
    KerrOblateTermination,
    kerr_zamo_camera_ray,
)
from offline.kerr_finite_thickness import (
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_surface import (
    KerrFiniteThicknessMultiSurface,
)
import offline.kerr_finite_thickness_replay_certificate as replay_module


class OfflineKerrFiniteThicknessReplayCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = KerrKerrSchildMetric(spin_a_m=0.7)
        cls.calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=0.05,
            outer_radius_over_mass=25.0,
        )
        cls.surface = KerrFiniteThicknessMultiSurface(
            cls.metric,
            cls.calibration,
        )
        cls.termination = KerrOblateTermination.horizon_worldtube(
            cls.metric,
            escape_radius_m=50.0,
            offset_m=0.02,
        )
        cls.ray_options = RayTraceOptions(
            absolute_tolerance=5.0e-10,
            relative_tolerance=5.0e-10,
            initial_step=0.05,
            maximum_step=0.25,
            maximum_affine_length=300.0,
            null_residual_limit=2.0e-7,
            record_path=True,
        )
        cls.surface_options = SurfaceEventOptions(
            absolute_tolerance=5.0e-10,
            relative_tolerance=5.0e-10,
            null_residual_limit=2.0e-7,
            subdivisions_per_segment=4,
        )
        cls.initial = kerr_zamo_camera_ray(
            cls.metric,
            observer_radius_m=30.0,
            theta_rad=1.1,
            screen_x=0.5,
            screen_y=-0.5,
        )
        cls.ray = trace_null_geodesic(
            cls.metric,
            cls.initial,
            termination=cls.termination,
            multi_interior_surface=cls.surface,
            surface_options=cls.surface_options,
            options=cls.ray_options,
        )
        cls.certificate = replay_module._issue_replay_certificate(
            cls.surface,
            cls.termination,
            cls.initial,
            cls.ray,
            cls.ray_options,
            cls.surface_options,
        )

    def require(self, certificate=None, **changes) -> None:
        replay_module._require_replay_certificate(
            self.certificate if certificate is None else certificate,
            changes.get("surface", self.surface),
            changes.get("termination", self.termination),
            changes.get("observer_initial_state", self.initial),
            changes.get("ray", self.ray),
            changes.get("ray_options", self.ray_options),
            changes.get("surface_options", self.surface_options),
        )

    def test_issue_once_and_repeated_require_never_reintegrates(self) -> None:
        with mock.patch.object(
            replay_module,
            "trace_null_geodesic",
            wraps=replay_module.trace_null_geodesic,
        ) as replay:
            certificate = replay_module._issue_replay_certificate(
                self.surface,
                self.termination,
                self.initial,
                self.ray,
                self.ray_options,
                self.surface_options,
            )
            self.assertEqual(replay.call_count, 1)
            for _ in range(3):
                self.require(certificate)
            self.assertEqual(replay.call_count, 1)

    def test_token_cannot_be_constructed_subclassed_copied_or_pickled(self) -> None:
        with self.assertRaisesRegex(TypeError, "only be issued"):
            replay_module._ReplayCertificate()
        with self.assertRaisesRegex(TypeError, "subclassed"):

            class ForgedSubclass(replay_module._ReplayCertificate):
                pass

        for operation in (
            lambda: copy.copy(self.certificate),
            lambda: copy.deepcopy(self.certificate),
            lambda: pickle.dumps(self.certificate),
        ):
            with self.subTest(operation=operation), self.assertRaises(TypeError):
                operation()

        fake = object.__new__(replay_module._ReplayCertificate)
        with self.assertRaisesRegex(
            replay_module.ReplayCertificateError,
            "not issued here",
        ):
            self.require(fake)

    def test_equal_but_foreign_inputs_and_wrong_bindings_are_rejected(self) -> None:
        foreign_surface = replace(self.surface)
        foreign_termination = replace(self.termination)
        foreign_initial = replace(self.initial)
        foreign_ray = replace(self.ray)
        foreign_ray_options = replace(self.ray_options)
        foreign_surface_options = replace(self.surface_options)
        for changes in (
            {"surface": foreign_surface},
            {"termination": foreign_termination},
            {"observer_initial_state": foreign_initial},
            {"ray": foreign_ray},
            {"ray_options": foreign_ray_options},
            {"surface_options": foreign_surface_options},
        ):
            with self.subTest(changes=tuple(changes)), self.assertRaisesRegex(
                replay_module.ReplayCertificateError,
                "bound to other inputs",
            ):
                self.require(**changes)

    def test_low_level_context_and_ray_mutation_invalidates_until_restored(
        self,
    ) -> None:
        old_step = self.ray_options.maximum_step
        try:
            object.__setattr__(self.ray_options, "maximum_step", 0.9 * old_step)
            with self.assertRaisesRegex(
                replay_module.ReplayCertificateError,
                "context is stale",
            ):
                self.require()
        finally:
            object.__setattr__(self.ray_options, "maximum_step", old_step)
        self.require()

        old_residual = self.ray.maximum_null_residual
        try:
            object.__setattr__(
                self.ray,
                "maximum_null_residual",
                old_residual + 1.0e-12,
            )
            with self.assertRaisesRegex(
                replay_module.ReplayCertificateError,
                "ray is stale",
            ):
                self.require()
        finally:
            object.__setattr__(
                self.ray,
                "maximum_null_residual",
                old_residual,
            )
        self.require()


if __name__ == "__main__":
    unittest.main()
