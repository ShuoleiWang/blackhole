from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from unittest.mock import patch
import unittest

from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_finite_thickness import (
    LOWER,
    UPPER,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_area import (
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_finite_thickness_emitter import KerrFiniteThicknessFaceEmitter
from offline.kerr_finite_thickness_launch import (
    KerrFiniteThicknessEmissionLaunch,
    KerrFiniteThicknessSurfaceFrame,
)
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
import offline.kerr_returning_radiation_kernel as forward_module
from offline.kerr_returning_radiation_kernel import (
    KerrReturningRadiationKernelPolicy,
    integrate_kerr_returning_radiation_energy_kernel,
)
from offline.kerr_returning_radiation_rays import (
    trace_kerr_returning_radiation_direction,
)
from offline.kerr_returning_radiation_receiver_rays import (
    trace_kerr_returning_radiation_receiver_direction,
    verify_kerr_returning_radiation_receiver_direction,
)
import offline.kerr_returning_radiation_receiver_kernel as receiver_module
from offline.kerr_returning_radiation_receiver_kernel import (
    IMPLEMENTATION_ID,
    SCIENTIFIC_STATUS,
    KerrForwardReceiverKernelComparison,
    KerrReceiverReturningRadiationKernel,
    KerrReturningRadiationReceiverKernelConvergenceError,
    KerrReturningRadiationReceiverKernelError,
    KerrReturningRadiationReceiverKernelVerificationError,
    compare_kerr_returning_radiation_kernels,
    integrate_kerr_returning_radiation_receiver_energy_kernel,
    verify_kerr_returning_radiation_kernel_comparison,
    verify_kerr_returning_radiation_receiver_energy_kernel,
)
from offline.returning_radiation import AxisymmetricReturningRadiationKernel


_REAL_TRACE_DIRECTION = receiver_module._trace_direction


class AlwaysEqualFloat(float):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


def _identity_for(*values: object) -> str:
    return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()


def _symmetric_receiver_classifier(
    surface,
    termination,
    ray_options,
    surface_options,
    coarse_ray_options,
    coarse_surface_options,
    receiver_face,
    receiver_radius_over_mass,
    incidence_cosine,
    tangent_azimuth_rad,
):
    del (
        surface,
        termination,
        ray_options,
        surface_options,
        coarse_ray_options,
        coarse_surface_options,
        incidence_cosine,
    )
    source_face = UPPER if tangent_azimuth_rad < math.pi else LOWER
    outcome = "source-upper" if source_face == UPPER else "source-lower"
    return receiver_module._ReceiverDirectionTransport(
        outcome,
        source_face,
        receiver_radius_over_mass,
        receiver_radius_over_mass,
        2.0,
        _identity_for(
            receiver_face,
            receiver_radius_over_mass,
            tangent_azimuth_rad,
        ),
    )


def _symmetric_forward_classifier(
    surface,
    termination,
    ray_options,
    surface_options,
    coarse_ray_options,
    coarse_surface_options,
    source_face,
    source_radius_over_mass,
    emission_angle_cosine,
    tangent_azimuth_rad,
):
    del (
        surface,
        termination,
        ray_options,
        surface_options,
        coarse_ray_options,
        coarse_surface_options,
    )
    receiver_face = UPPER if tangent_azimuth_rad < math.pi else LOWER
    fate = "return-upper" if receiver_face == UPPER else "return-lower"
    ratio = 2.0
    return forward_module._DirectionTransport(
        fate,
        receiver_face,
        source_radius_over_mass,
        ratio,
        ratio * ratio,
        _identity_for(
            source_face,
            source_radius_over_mass,
            emission_angle_cosine,
            tangent_azimuth_rad,
        ),
        receiver_face,
        source_radius_over_mass,
    )


class OfflineKerrReturningRadiationReceiverKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = KerrKerrSchildMetric(spin_a_m=0.7)
        cls.calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=0.08,
            outer_radius_over_mass=8.0,
        )
        cls.surface = KerrFiniteThicknessMultiSurface(cls.metric, cls.calibration)
        cls.termination = KerrOblateTermination.horizon_worldtube(
            cls.metric,
            escape_radius_m=20.0,
            offset_m=0.02,
        )
        cls.ray_options = RayTraceOptions(
            absolute_tolerance=5.0e-9,
            relative_tolerance=5.0e-9,
            initial_step=0.025,
            maximum_step=0.3,
            maximum_affine_length=100.0,
        )
        cls.surface_options = SurfaceEventOptions(
            absolute_tolerance=5.0e-9,
            relative_tolerance=5.0e-9,
            subdivisions_per_segment=4,
        )
        cls.policy = KerrReturningRadiationKernelPolicy(
            rho_order=4,
            mu_order=4,
            psi_count=4,
            absolute_tolerance=1.0e-8,
            relative_tolerance=1.0e-8,
            symmetry_absolute_tolerance=1.0e-8,
            symmetry_relative_tolerance=1.0e-8,
            maximum_direction_evaluations=10_000,
            maximum_whole_ray_traces=40_000,
        )
        cls.area_policy = KerrFiniteThicknessAreaQuadraturePolicy(
            gauss_legendre_order=24,
            relative_tolerance=1.0e-8,
            absolute_tolerance_over_mass_squared=1.0e-8,
            maximum_point_evaluations=384,
        )
        cls.edges = (
            float(cls.calibration.isco_radius_over_mass),
            float(cls.calibration.outer_radius_over_mass),
        )
        cls._patcher = patch.object(
            receiver_module,
            "_trace_direction",
            side_effect=_symmetric_receiver_classifier,
        )
        cls._patcher.start()
        cls.result = cls.build()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._patcher.stop()

    @classmethod
    def build(cls, **overrides):
        arguments = {
            "surface": cls.surface,
            "termination": cls.termination,
            "annulus_edges_over_mass": cls.edges,
            "ray_options": cls.ray_options,
            "surface_options": cls.surface_options,
            "policy": cls.policy,
            "area_policy": cls.area_policy,
        }
        arguments.update(overrides)
        return integrate_kerr_returning_radiation_receiver_energy_kernel(**arguments)

    @staticmethod
    def forge(original, **changes):
        forged = object.__new__(type(original))
        for name in type(original).__dataclass_fields__:
            object.__setattr__(
                forged,
                name,
                changes.get(name, getattr(original, name)),
            )
        return forged

    def test_receiver_gauss_rule_covers_every_public_kernel_order(self) -> None:
        for order in range(1, 65):
            with self.subTest(order=order):
                receiver_rule = receiver_module._gauss_legendre_unit_interval(order)
                forward_rule = forward_module._gauss_legendre_unit_interval(order)
                self.assertEqual(
                    tuple((node.hex(), weight.hex()) for node, weight in receiver_rule),
                    tuple((node.hex(), weight.hex()) for node, weight in forward_rule),
                )

    def test_scientific_scope_and_shared_code_boundary_are_explicit(self) -> None:
        self.assertEqual(SCIENTIFIC_STATUS["implementationId"], IMPLEMENTATION_ID)
        self.assertTrue(SCIENTIFIC_STATUS["isFiniteGridEnergyOnlyKernel"])
        self.assertTrue(
            SCIENTIFIC_STATUS["sharesExactKerrGeodesicCodeFamilyWithForwardKernel"]
        )
        self.assertFalse(SCIENTIFIC_STATUS["isIndependentGeodesicOracle"])
        self.assertFalse(SCIENTIFIC_STATUS["hasIndependentPhysicsOracle"])
        self.assertIn("g^4", SCIENTIFIC_STATUS["coefficientEquation"])
        self.assertIn("independent geodesic oracle", SCIENTIFIC_STATUS["prohibitedClaim"])
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["isCompleteKerrbb"] = True

    def test_one_over_pi_sky_weight_four_blocks_and_row_closure(self) -> None:
        result = self.result
        # Integrand=2 on each half-azimuth source face.  The receiver rule is
        # 2*w_mu/Npsi, so each half sky contributes exactly 2 and the row is 4.
        for block in (
            result.upper_receiver_upper_source_coefficients,
            result.upper_receiver_lower_source_coefficients,
            result.lower_receiver_upper_source_coefficients,
            result.lower_receiver_lower_source_coefficients,
        ):
            self.assertAlmostEqual(block[0][0], 2.0, places=14)
        self.assertAlmostEqual(
            result.upper_receiver_returning_row_totals[0],
            4.0,
            places=14,
        )
        self.assertAlmostEqual(
            result.lower_receiver_returning_row_totals[0],
            4.0,
            places=14,
        )
        self.assertLessEqual(result.upper_receiver_row_closure_residuals[0], 1.0e-14)
        self.assertLessEqual(result.lower_receiver_row_closure_residuals[0], 1.0e-14)
        for fractions in (
            result.upper_receiver_sky_source_fractions[0],
            result.lower_receiver_sky_source_fractions[0],
        ):
            self.assertEqual(fractions.total, 1.0)
            self.assertAlmostEqual(fractions.source_upper, 0.5, places=14)
            self.assertAlmostEqual(fractions.source_lower, 0.5, places=14)
            self.assertEqual(fractions.past_worldtube_no_disk_source, 0.0)

    def test_four_block_labels_are_independently_distinguishable(self) -> None:
        expected = {
            (UPPER, UPPER): 2.0,
            (UPPER, LOWER): 3.0,
            (LOWER, UPPER): 5.0,
            (LOWER, LOWER): 7.0,
        }

        def classified(*args):
            receiver_face = args[6]
            rho = args[7]
            psi = args[9]
            source_face = UPPER if psi < math.pi else LOWER
            outcome = "source-upper" if source_face == UPPER else "source-lower"
            return receiver_module._ReceiverDirectionTransport(
                outcome,
                source_face,
                rho,
                rho,
                expected[(receiver_face, source_face)],
                _identity_for(*args[6:]),
            )

        with patch.object(receiver_module, "_trace_direction", side_effect=classified):
            result = self.build()
        self.assertAlmostEqual(
            result.upper_receiver_upper_source_coefficients[0][0], 2.0, places=14
        )
        self.assertAlmostEqual(
            result.upper_receiver_lower_source_coefficients[0][0], 3.0, places=14
        )
        self.assertAlmostEqual(
            result.lower_receiver_upper_source_coefficients[0][0], 5.0, places=14
        )
        self.assertAlmostEqual(
            result.lower_receiver_lower_source_coefficients[0][0], 7.0, places=14
        )

    def test_two_annulus_receiver_proper_area_average_and_source_columns(self) -> None:
        middle = 0.5 * math.fsum(self.edges)
        edges = (self.edges[0], middle, self.edges[-1])

        def radial_integrand(*args):
            rho = args[7]
            psi = args[9]
            source_face = UPPER if psi < math.pi else LOWER
            outcome = "source-upper" if source_face == UPPER else "source-lower"
            return receiver_module._ReceiverDirectionTransport(
                outcome,
                source_face,
                rho,
                rho,
                rho,
                _identity_for(*args[6:]),
            )

        relaxed = replace(
            self.policy,
            absolute_tolerance=2.0e-3,
            relative_tolerance=2.0e-3,
        )
        with patch.object(
            receiver_module,
            "_trace_direction",
            side_effect=radial_integrand,
        ):
            result = self.build(annulus_edges_over_mass=edges, policy=relaxed)

        for receiver_face, areas, same_face, cross_face, rows in (
            (
                UPPER,
                result.upper_annulus_areas_over_mass_squared,
                result.upper_receiver_upper_source_coefficients,
                result.upper_receiver_lower_source_coefficients,
                result.upper_receiver_returning_row_totals,
            ),
            (
                LOWER,
                result.lower_annulus_areas_over_mass_squared,
                result.lower_receiver_lower_source_coefficients,
                result.lower_receiver_upper_source_coefficients,
                result.lower_receiver_returning_row_totals,
            ),
        ):
            for index, (inner, outer) in enumerate(zip(edges, edges[1:])):
                nodes = receiver_module._receiver_rho_area_nodes(
                    self.surface,
                    inner,
                    outer,
                    receiver_face,
                    self.policy.rho_order,
                    areas[index],
                )
                expected_average = math.fsum(rho * area for rho, area in nodes) / areas[
                    index
                ]
                self.assertAlmostEqual(
                    same_face[index][index], expected_average, places=13
                )
                self.assertAlmostEqual(
                    cross_face[index][index], expected_average, places=13
                )
                self.assertAlmostEqual(rows[index], 2.0 * expected_average, places=13)
                other = 1 - index
                self.assertEqual(same_face[index][other], 0.0)
                self.assertEqual(cross_face[index][other], 0.0)

    def test_past_worldtube_is_zero_and_not_future_fate(self) -> None:
        def no_source(*args):
            return receiver_module._ReceiverDirectionTransport(
                receiver_module.PAST_WORLDTUBE_NO_SOURCE,
                None,
                None,
                None,
                0.0,
                _identity_for(*args[6:]),
            )

        with patch.object(receiver_module, "_trace_direction", side_effect=no_source):
            result = self.build()
        for block in (
            result.upper_receiver_upper_source_coefficients,
            result.upper_receiver_lower_source_coefficients,
            result.lower_receiver_upper_source_coefficients,
            result.lower_receiver_lower_source_coefficients,
        ):
            self.assertEqual(block, ((0.0,),))
        self.assertEqual(result.upper_receiver_returning_row_totals, (0.0,))
        self.assertEqual(result.lower_receiver_returning_row_totals, (0.0,))
        for fraction in (
            result.upper_receiver_sky_source_fractions[0],
            result.lower_receiver_sky_source_fractions[0],
        ):
            self.assertEqual(fraction.past_worldtube_no_disk_source, 1.0)

    def test_five_grid_gates_and_whole_ray_budget_are_exact(self) -> None:
        convergence = self.result.convergence
        self.assertTrue(convergence.converged)
        self.assertTrue(self.result.fine_coarse_source_bin_topology_verified)
        for item in (
            convergence.half_receiver_rho,
            convergence.half_mu,
            convergence.half_psi,
            convergence.phase_shifted,
        ):
            self.assertTrue(item.converged)
            self.assertLessEqual(item.matrix_maximum_scaled_difference, 1.0)
            self.assertLessEqual(item.row_total_maximum_scaled_difference, 1.0)
            self.assertLessEqual(item.sky_fraction_maximum_scaled_difference, 1.0)
        self.assertEqual(self.result.direction_evaluations_consumed, 448)
        self.assertEqual(self.result.whole_ray_traces_consumed, 1792)
        descriptor = self.result.model_descriptor()
        self.assertEqual(descriptor["coefficient"]["directionWeight"], "2*w_mu/N_psi")
        self.assertIn(
            "public revalidated exact-tree-bound coarse_ray",
            descriptor["coefficient"]["coarseSourceRadiusProvenance"],
        )

    def test_source_bin_fine_coarse_split_fails_inside_integrator(self) -> None:
        middle = 0.5 * math.fsum(self.edges)

        def split(*args):
            return receiver_module._ReceiverDirectionTransport(
                "source-upper",
                UPPER,
                math.nextafter(middle, math.inf),
                math.nextafter(middle, -math.inf),
                1.0,
                _identity_for(*args[6:]),
            )

        with patch.object(receiver_module, "_trace_direction", side_effect=split):
            with self.assertRaisesRegex(
                KerrReturningRadiationReceiverKernelConvergenceError,
                "different source annuli",
            ):
                self.build(
                    annulus_edges_over_mass=(self.edges[0], middle, self.edges[-1])
                )

    def test_source_outside_grid_and_exact_edge_ownership_fail_closed(self) -> None:
        middle = 0.5 * math.fsum(self.edges)
        edges = (self.edges[0], middle, self.edges[-1])
        self.assertEqual(receiver_module._bin_index(self.edges[0], edges, "rho"), 0)
        self.assertEqual(receiver_module._bin_index(middle, edges, "rho"), 1)
        self.assertEqual(receiver_module._bin_index(self.edges[-1], edges, "rho"), 1)

        def outside(*args):
            rho = math.nextafter(self.edges[-1], math.inf)
            return receiver_module._ReceiverDirectionTransport(
                "source-upper", UPPER, rho, rho, 1.0, _identity_for(*args[6:])
            )

        with patch.object(receiver_module, "_trace_direction", side_effect=outside):
            with self.assertRaisesRegex(
                KerrReturningRadiationReceiverKernelError,
                "outside",
            ):
                self.build()

    def test_options_policy_edges_and_budget_use_exact_schema(self) -> None:
        with self.assertRaises(KerrReturningRadiationReceiverKernelVerificationError):
            self.build(ray_options=RayTraceOptions(maximum_step=True))
        forged_surface = replace(self.surface_options)
        object.__setattr__(forged_surface, "absolute_tolerance", 1)
        with self.assertRaises(KerrReturningRadiationReceiverKernelVerificationError):
            self.build(surface_options=forged_surface)
        forged_policy = replace(self.policy)
        object.__setattr__(forged_policy, "maximum_whole_ray_traces", True)
        with self.assertRaises(KerrReturningRadiationReceiverKernelVerificationError):
            self.build(policy=forged_policy)
        with self.assertRaises(ValueError):
            self.build(
                annulus_edges_over_mass=(AlwaysEqualFloat(self.edges[0]), self.edges[1])
            )
        tiny = replace(self.policy, maximum_direction_evaluations=447)
        with self.assertRaisesRegex(ValueError, "448 direction evaluations"):
            self.build(policy=tiny)

    def test_periodic_phase_disagreement_fails_closed(self) -> None:
        def narrow_sector(*args):
            rho = args[7]
            psi = args[9]
            if psi < 0.4:
                return receiver_module._ReceiverDirectionTransport(
                    "source-upper",
                    UPPER,
                    rho,
                    rho,
                    1.0,
                    _identity_for(*args[6:]),
                )
            return receiver_module._ReceiverDirectionTransport(
                receiver_module.PAST_WORLDTUBE_NO_SOURCE,
                None,
                None,
                None,
                0.0,
                _identity_for(*args[6:]),
            )

        strict = replace(
            self.policy,
            absolute_tolerance=1.0e-6,
            relative_tolerance=1.0e-6,
        )
        with patch.object(receiver_module, "_trace_direction", side_effect=narrow_sector):
            with self.assertRaises(
                KerrReturningRadiationReceiverKernelConvergenceError
            ):
                self.build(policy=strict)

    def test_result_replay_axisym_and_live_tamper(self) -> None:
        verify_kerr_returning_radiation_receiver_energy_kernel(self.result)
        self.result.revalidate()
        reduced = self.result.to_axisymmetric_energy_kernel()
        self.assertIs(type(reduced), AxisymmetricReturningRadiationKernel)
        self.assertAlmostEqual(reduced.receiver_emitter_coefficients[0][0], 4.0)
        with self.assertRaises(ValueError):
            self.result.to_axisymmetric_energy_kernel(False)
        forged = self.forge(
            self.result,
            upper_receiver_upper_source_coefficients=((99.0,),),
        )
        with self.assertRaises(KerrReturningRadiationReceiverKernelVerificationError):
            verify_kerr_returning_radiation_receiver_energy_kernel(forged)

    def test_forward_receiver_comparison_blocks_columns_and_replay(self) -> None:
        with patch.object(
            forward_module,
            "_trace_direction",
            side_effect=_symmetric_forward_classifier,
        ):
            forward = integrate_kerr_returning_radiation_energy_kernel(
                self.surface,
                termination=self.termination,
                annulus_edges_over_mass=self.edges,
                ray_options=self.ray_options,
                surface_options=self.surface_options,
                policy=self.policy,
                area_policy=self.area_policy,
            )
            comparison = compare_kerr_returning_radiation_kernels(
                forward,
                self.result,
                absolute_tolerance=1.0e-8,
                relative_tolerance=1.0e-8,
            )
            self.assertIs(type(comparison), KerrForwardReceiverKernelComparison)
            self.assertTrue(comparison.converged)
            self.assertLessEqual(comparison.maximum_absolute_difference, 1.0e-14)
            self.assertAlmostEqual(
                comparison.receiver_reconstructed_upper_source_g2_columns[0],
                4.0,
                places=14,
            )
            self.assertAlmostEqual(
                comparison.receiver_reconstructed_lower_source_g2_columns[0],
                4.0,
                places=14,
            )
            self.assertTrue(comparison.shares_exact_kerr_geodesic_code_family)
            self.assertFalse(comparison.is_independent_geodesic_oracle)
            comparison.revalidate()
            forged = self.forge(comparison, converged=False)
            with self.assertRaises(
                KerrReturningRadiationReceiverKernelVerificationError
            ):
                verify_kerr_returning_radiation_kernel_comparison(forged)

    def test_real_reciprocal_direction_and_authenticated_coarse_source_seam(self) -> None:
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
        source = KerrFiniteThicknessFaceEmitter(
            metric=metric,
            calibration=calibration,
            pseudo_cylindrical_radius_over_mass=8.0,
            face=UPPER,
        )
        launch = KerrFiniteThicknessEmissionLaunch(
            KerrFiniteThicknessSurfaceFrame(source),
            0.02,
            0.5 * math.pi,
        )
        forward = trace_kerr_returning_radiation_direction(
            launch,
            surface,
            termination=termination,
            ray_options=self.ray_options,
            surface_options=self.surface_options,
        )
        self.assertIsNotNone(forward.receiver)
        frame = KerrFiniteThicknessSurfaceFrame(forward.receiver)
        covector = forward.ray.terminal_state.covector
        receiver = frame.emitter
        frequency = -math.fsum(
            receiver.four_velocity[index] * covector[index] for index in range(4)
        )
        mu_i = -math.fsum(
            receiver.outward_unit_normal[index] * covector[index]
            for index in range(4)
        ) / frequency
        meridional = math.fsum(
            frame.meridional_tangent[index] * covector[index]
            for index in range(4)
        ) / frequency
        azimuthal = math.fsum(
            frame.azimuthal_tangent[index] * covector[index]
            for index in range(4)
        ) / frequency
        psi_i = math.atan2(azimuthal, meridional) % (2.0 * math.pi)
        backward = trace_kerr_returning_radiation_receiver_direction(
            frame,
            surface,
            float(mu_i),
            float(psi_i),
            termination=termination,
            ray_options=self.ray_options,
            surface_options=self.surface_options,
        )
        verify_kerr_returning_radiation_receiver_direction(backward)
        self.assertTrue(
            math.isclose(
                backward.source_to_receiver_frequency_ratio,
                forward.emitter_to_receiver_frequency_ratio,
                rel_tol=5.0e-9,
            )
        )
        self.assertAlmostEqual(
            backward.source_emission_cosine,
            0.02,
            delta=2.0e-8,
        )
        transport = _REAL_TRACE_DIRECTION(
            surface,
            termination,
            self.ray_options,
            self.surface_options,
            None,
            None,
            forward.receiver_face,
            forward.receiver_radius_over_mass,
            float(mu_i),
            float(psi_i),
        )
        self.assertEqual(transport.outcome, "source-upper")
        self.assertEqual(transport.source_face, UPPER)
        self.assertAlmostEqual(transport.source_radius_over_mass, 8.0, delta=2.0e-8)
        self.assertGreater(transport.receiver_integrand, 0.0)
        self.assertTrue(
            math.isclose(
                transport.receiver_integrand,
                backward.receiver_directional_integrand,
                rel_tol=5.0e-9,
            )
        )
        self.assertGreater(transport.coarse_source_radius_over_mass, 0.0)
        seam = 0.5 * math.fsum(
            (
                transport.source_radius_over_mass,
                transport.coarse_source_radius_over_mass,
            )
        )
        seam_edges = (
            float(calibration.isco_radius_over_mass),
            seam,
            float(calibration.outer_radius_over_mass),
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationReceiverKernelConvergenceError,
            "different source annuli",
        ):
            receiver_module._validated_source_bin(transport, seam_edges)


if __name__ == "__main__":
    unittest.main()
