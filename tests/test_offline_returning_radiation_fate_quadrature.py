from __future__ import annotations

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from offline.disk_atmosphere import (
    FluxConservingLinearLimbDarkening,
    IsotropicAngularEmission,
)
from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_finite_thickness import (
    UPPER,
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_emitter import (
    KerrFiniteThicknessFaceEmitter,
)
from offline.kerr_finite_thickness_launch import (
    KerrFiniteThicknessSurfaceFrame,
)
from offline.kerr_finite_thickness_surface import (
    KerrFiniteThicknessMultiSurface,
)
from offline.returning_radiation_fate_quadrature import (
    D20_EMITTED_FLUX_QUADRATURE_DESCRIPTOR,
    D20_QUADRATURE_IMPLEMENTATION_ID,
    IMPLEMENTATION_ID,
    SCIENTIFIC_STATUS,
    EmittedFluxFateFractions,
    KerrReturningRadiationFateConvergenceError,
    KerrReturningRadiationFateQuadrature,
    KerrReturningRadiationFateVerificationError,
    _DirectionClassification,
    _emitted_flux_nodes,
    _gauss_legendre_unit_interval,
    _integrate_classification_grid,
    integrate_kerr_returning_radiation_fates,
    kerrbb_d20_emitted_flux_direction_nodes,
    verify_kerr_returning_radiation_fate_quadrature,
)


class AlwaysEqualStr(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class RayOptionsSubclass(RayTraceOptions):
    pass


class OfflineReturningRadiationFateQuadratureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metric = KerrKerrSchildMetric(spin_a_m=0.7)
        cls.calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=0.7,
            eddington_scaled_mass_accretion_rate=0.08,
            outer_radius_over_mass=30.0,
        )
        cls.surface = KerrFiniteThicknessMultiSurface(
            cls.metric,
            cls.calibration,
        )
        cls.frame = KerrFiniteThicknessSurfaceFrame(
            KerrFiniteThicknessFaceEmitter(
                metric=cls.metric,
                calibration=cls.calibration,
                pseudo_cylindrical_radius_over_mass=20.0,
                face=UPPER,
            )
        )
        cls.termination = KerrOblateTermination.horizon_worldtube(
            cls.metric,
            escape_radius_m=50.0,
            offset_m=0.02,
        )
        cls.ray_options = RayTraceOptions(
            absolute_tolerance=5.0e-9,
            relative_tolerance=5.0e-9,
            initial_step=0.025,
            maximum_step=0.3,
            maximum_affine_length=200.0,
        )
        cls.surface_options = SurfaceEventOptions(
            absolute_tolerance=5.0e-9,
            relative_tolerance=5.0e-9,
            subdivisions_per_segment=4,
        )

    @staticmethod
    def _sha(character: str) -> str:
        return character * 64

    @staticmethod
    def _forge(original, **changes):
        forged = object.__new__(type(original))
        for name in type(original).__dataclass_fields__:
            object.__setattr__(
                forged,
                name,
                changes.get(name, getattr(original, name)),
            )
        return forged

    def _mock_product(self) -> KerrReturningRadiationFateQuadrature:
        with patch(
            "offline.returning_radiation_fate_quadrature._trace_classification",
            return_value=_DirectionClassification(
                "escaped",
                None,
                self._sha("a"),
            ),
        ):
            return integrate_kerr_returning_radiation_fates(
                self.frame,
                self.surface,
                termination=self.termination,
                ray_options=self.ray_options,
                surface_options=self.surface_options,
                mu_order=4,
                psi_count=4,
                convergence_absolute_tolerance=0.01,
                maximum_whole_ray_traces=192,
            )

    def test_scientific_boundary_excludes_kernel_photon_number_g_mu_and_fs(self) -> None:
        self.assertEqual(SCIENTIFIC_STATUS["implementationId"], IMPLEMENTATION_ID)
        self.assertIn("emitter-local", SCIENTIFIC_STATUS["classification"])
        self.assertIn("2 mu f(mu)", SCIENTIFIC_STATUS["measure"])
        for key in (
            "isPhotonNumberProbability",
            "isReceiverCentredIncidentFlux",
            "isIndependentRayKernel",
            "outputsReturningRadiationKernelK",
            "usesFrequencyShiftAsJacobian",
            "usesReceiverIncidenceCosineAsJacobian",
            "includesReturningRadiationStressWorkFS",
            "isCompleteKerrbb",
        ):
            self.assertIs(SCIENTIFIC_STATUS[key], False)
        self.assertIn("K coefficient", SCIENTIFIC_STATUS["prohibitedClaim"])
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["outputsReturningRadiationKernelK"] = True

    def test_d20_gauss_legendre_rule_matches_analytic_flux_moment(self) -> None:
        nodes = _emitted_flux_nodes(
            3,
            8,
            phase_cells=0.0,
            angular_law=FluxConservingLinearLimbDarkening(),
        )
        self.assertEqual(len(nodes), 24)
        self.assertEqual(
            math.fsum(node.normalized_emitted_flux_weight for node in nodes),
            1.0,
        )
        # With p(mu)=2 mu(1/2+3mu/4)=mu+3mu^2/2,
        # E[mu^2]=integral(mu^3+3mu^4/2)=1/4+3/10=11/20.
        second_moment = math.fsum(
            node.normalized_emitted_flux_weight
            * node.emission_angle_cosine**2
            for node in nodes
        )
        self.assertTrue(math.isclose(second_moment, 11.0 / 20.0, rel_tol=3e-15))

    def test_every_supported_mu_order_has_a_resolved_float64_rule(self) -> None:
        for order in range(1, 65):
            with self.subTest(order=order):
                rule = _gauss_legendre_unit_interval(order)
                self.assertEqual(len(rule), order)
                self.assertLessEqual(
                    abs(math.fsum(weight for _node, weight in rule) - 1.0),
                    64.0 * math.ulp(1.0),
                )
                power = min(3, 2 * order - 1)
                moment = math.fsum(
                    weight * node**power for node, weight in rule
                )
                self.assertTrue(
                    math.isclose(moment, 1.0 / (power + 1), abs_tol=2.0e-15)
                )

    def test_public_d20_rule_locks_weights_phase_and_raw_normalization_gate(self) -> None:
        self.assertEqual(
            D20_EMITTED_FLUX_QUADRATURE_DESCRIPTOR["implementationId"],
            D20_QUADRATURE_IMPLEMENTATION_ID,
        )
        unshifted = kerrbb_d20_emitted_flux_direction_nodes(
            4,
            4,
            phase_cells=0.0,
        )
        shifted = kerrbb_d20_emitted_flux_direction_nodes(
            4,
            4,
            phase_cells=0.5,
        )
        self.assertEqual(len(unshifted), 16)
        self.assertEqual(
            unshifted[0].emission_angle_cosine,
            0.06943184420297371,
        )
        self.assertEqual(
            unshifted[0].normalized_emitted_flux_weight,
            0.0033334501812013486,
        )
        self.assertEqual(unshifted[0].tangent_azimuth_rad, math.pi / 4.0)
        self.assertEqual(shifted[0].tangent_azimuth_rad, math.pi / 2.0)
        self.assertEqual(
            tuple(node.normalized_emitted_flux_weight for node in unshifted),
            tuple(node.normalized_emitted_flux_weight for node in shifted),
        )
        self.assertEqual(
            math.fsum(
                node.normalized_emitted_flux_weight for node in unshifted
            ),
            1.0,
        )
        # One D20 mu node has raw total 0.875.  It must fail rather than be
        # silently divided by 0.875 and advertised as a normalized rule.
        with self.assertRaisesRegex(
            RuntimeError,
            "does not reproduce analytic unit emitted flux",
        ):
            kerrbb_d20_emitted_flux_direction_nodes(1, 8)
        with self.assertRaises(TypeError):
            D20_EMITTED_FLUX_QUADRATURE_DESCRIPTOR["muRule"] = "spoofed"

    def test_constant_isotropic_law_is_normalized_and_phase_invariant(self) -> None:
        unshifted = _emitted_flux_nodes(
            2,
            6,
            phase_cells=0.0,
            angular_law=IsotropicAngularEmission(),
        )
        shifted = _emitted_flux_nodes(
            2,
            6,
            phase_cells=0.5,
            angular_law=IsotropicAngularEmission(),
        )
        for nodes in (unshifted, shifted):
            self.assertEqual(
                math.fsum(
                    node.normalized_emitted_flux_weight for node in nodes
                ),
                1.0,
            )
            # Isotropic local emitted-flux density is p(mu)=2mu.
            self.assertTrue(
                math.isclose(
                    math.fsum(
                        node.normalized_emitted_flux_weight
                        * node.emission_angle_cosine**2
                        for node in nodes
                    ),
                    0.5,
                    rel_tol=2e-15,
                )
            )

    def test_one_hot_weights_close_exactly_and_bins_publish_roundoff_residual(self) -> None:
        nodes = _emitted_flux_nodes(
            2,
            8,
            phase_cells=0.0,
            angular_law=FluxConservingLinearLimbDarkening(),
        )
        all_escaped = _integrate_classification_grid(
            nodes,
            (2.0, 5.0, 10.0),
            lambda _mu, _psi: _DirectionClassification(
                "escaped",
                None,
                self._sha("e"),
            ),
        )
        self.assertEqual(
            all_escaped.fractions,
            EmittedFluxFateFractions(0.0, 0.0, 0.0, 1.0, 0.0),
        )
        self.assertEqual(all_escaped.fractions.total, 1.0)
        self.assertEqual(all_escaped.return_upper_by_receiver_bin, (0.0, 0.0))

        fates = (
            "return-upper",
            "return-lower",
            "captured",
            "escaped",
            "plunge-sink",
        )
        cursor = iter(fates * 4)

        def classify(_mu, _psi):
            fate = next(cursor)
            radius = 4.0 if fate == "return-upper" else (
                7.0 if fate == "return-lower" else None
            )
            return _DirectionClassification(fate, radius, self._sha("f"))

        mixed = _integrate_classification_grid(
            nodes,
            (2.0, 5.0, 10.0),
            classify,
        )
        self.assertEqual(mixed.fractions.total, 1.0)
        self.assertTrue(
            math.isclose(
                math.fsum(mixed.return_upper_by_receiver_bin),
                mixed.fractions.return_upper,
                rel_tol=0.0,
                abs_tol=8.0 * math.ulp(1.0),
            )
        )
        self.assertLessEqual(
            mixed.return_upper_bin_closure_residual,
            16.0 * math.ulp(1.0),
        )
        self.assertLessEqual(
            mixed.return_lower_bin_closure_residual,
            16.0 * math.ulp(1.0),
        )
        self.assertTrue(
            math.isclose(
                math.fsum(mixed.return_lower_by_receiver_bin),
                mixed.fractions.return_lower,
                rel_tol=0.0,
                abs_tol=8.0 * math.ulp(1.0),
            )
        )

    def test_periodic_exchange_symmetry_swaps_upper_and_lower_diagnostics(self) -> None:
        nodes = _emitted_flux_nodes(
            4,
            8,
            phase_cells=0.0,
            angular_law=FluxConservingLinearLimbDarkening(),
        )

        def classify(mu, psi, *, swapped):
            first_half = psi < math.pi
            upper = first_half is not swapped
            return _DirectionClassification(
                "return-upper" if upper else "return-lower",
                4.0 if mu < 0.5 else 7.0,
                self._sha("b"),
            )

        original = _integrate_classification_grid(
            nodes,
            (2.0, 5.0, 10.0),
            lambda mu, psi: classify(mu, psi, swapped=False),
        )
        swapped = _integrate_classification_grid(
            nodes,
            (2.0, 5.0, 10.0),
            lambda mu, psi: classify(mu, psi, swapped=True),
        )
        self.assertTrue(
            math.isclose(original.fractions.return_upper, 0.5, rel_tol=2e-15)
        )
        self.assertTrue(
            math.isclose(original.fractions.return_lower, 0.5, rel_tol=2e-15)
        )
        self.assertEqual(
            original.return_upper_by_receiver_bin,
            swapped.return_lower_by_receiver_bin,
        )
        self.assertEqual(
            original.return_lower_by_receiver_bin,
            swapped.return_upper_by_receiver_bin,
        )

    def test_resolution_phase_gate_and_hard_budget_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "even integer at least 4"):
            integrate_kerr_returning_radiation_fates(
                self.frame,
                self.surface,
                termination=self.termination,
                ray_options=self.ray_options,
                surface_options=self.surface_options,
                mu_order=2,
                psi_count=4,
            )
        with self.assertRaisesRegex(ValueError, "requires 192 whole rays"):
            integrate_kerr_returning_radiation_fates(
                self.frame,
                self.surface,
                termination=self.termination,
                ray_options=self.ray_options,
                surface_options=self.surface_options,
                mu_order=4,
                psi_count=4,
                maximum_whole_ray_traces=191,
            )
        with self.assertRaisesRegex(TypeError, "exact RayTraceOptions"):
            integrate_kerr_returning_radiation_fates(
                self.frame,
                self.surface,
                termination=self.termination,
                ray_options=RayOptionsSubclass(),
                surface_options=self.surface_options,
                mu_order=4,
                psi_count=4,
            )

        def phase_sensitive(
            _frame,
            _surface,
            _termination,
            _ray_options,
            _surface_options,
            _coarse_ray_options,
            _coarse_surface_options,
            _mu,
            psi,
        ):
            fate = "captured" if psi < 0.1 else "escaped"
            return _DirectionClassification(fate, None, self._sha("c"))

        with patch(
            "offline.returning_radiation_fate_quadrature._trace_classification",
            side_effect=phase_sensitive,
        ):
            with self.assertRaisesRegex(
                KerrReturningRadiationFateConvergenceError,
                "periodic phase gate",
            ):
                integrate_kerr_returning_radiation_fates(
                    self.frame,
                    self.surface,
                    termination=self.termination,
                    ray_options=self.ray_options,
                    surface_options=self.surface_options,
                    mu_order=4,
                    psi_count=4,
                    convergence_absolute_tolerance=0.01,
                    maximum_whole_ray_traces=192,
                )

    def test_mu_band_counterexample_is_detected_instead_of_renormalized_green(self) -> None:
        def mu_band_sensitive(
            _frame,
            _surface,
            _termination,
            _ray_options,
            _surface_options,
            _coarse_ray_options,
            _coarse_surface_options,
            mu,
            _psi,
        ):
            fate = "captured" if 0.25 < mu < 0.45 else "escaped"
            return _DirectionClassification(fate, None, self._sha("d"))

        with patch(
            "offline.returning_radiation_fate_quadrature._trace_classification",
            side_effect=mu_band_sensitive,
        ):
            with self.assertRaisesRegex(
                KerrReturningRadiationFateConvergenceError,
                "mu N/2N",
            ):
                integrate_kerr_returning_radiation_fates(
                    self.frame,
                    self.surface,
                    termination=self.termination,
                    ray_options=self.ray_options,
                    surface_options=self.surface_options,
                    mu_order=4,
                    psi_count=4,
                    convergence_absolute_tolerance=0.01,
                    maximum_whole_ray_traces=192,
                )

    def test_mock_product_self_replay_rejects_live_fraction_and_audit_tampering(self) -> None:
        product = self._mock_product()
        self.assertEqual(product.whole_ray_traces_consumed, 192)
        self.assertEqual(product.fractions.total, 1.0)
        descriptor = product.model_descriptor()
        self.assertIs(descriptor["measure"]["usesFrequencyShiftG"], False)
        self.assertIs(
            descriptor["measure"]["usesReceiverIncidenceCosine"],
            False,
        )
        self.assertEqual(
            descriptor["result"]["estimateKind"],
            "finite-grid-point-estimate",
        )
        self.assertIs(
            descriptor["result"]
            ["publishedFiniteGridUncertaintyDiagnostics"]
            ["rigorousErrorBound"],
            False,
        )
        with patch(
            "offline.returning_radiation_fate_quadrature._trace_classification",
            return_value=_DirectionClassification(
                "escaped",
                None,
                self._sha("a"),
            ),
        ):
            verify_kerr_returning_radiation_fate_quadrature(product)
            forged_fractions = self._forge(
                product,
                fractions=EmittedFluxFateFractions(0.0, 0.0, 1.0, 0.0, 0.0),
            )
            with self.assertRaisesRegex(
                KerrReturningRadiationFateVerificationError,
                "result.fractions",
            ):
                verify_kerr_returning_radiation_fate_quadrature(
                    forged_fractions
                )

            forged_audit_entry = replace(
                product.full_grid_direction_audit[0],
                primitive_descriptor_sha256=AlwaysEqualStr(
                    product.full_grid_direction_audit[0].primitive_descriptor_sha256
                ),
            )
            forged_audit = self._forge(
                product,
                full_grid_direction_audit=(
                    forged_audit_entry,
                    *product.full_grid_direction_audit[1:],
                ),
            )
            with self.assertRaisesRegex(
                KerrReturningRadiationFateVerificationError,
                "non-exact type",
            ):
                verify_kerr_returning_radiation_fate_quadrature(forged_audit)

    def test_z_real_spin_point_seven_4_by_4_closes_once(self) -> None:
        fast_termination = KerrOblateTermination.horizon_worldtube(
            self.metric,
            escape_radius_m=35.0,
            offset_m=0.02,
        )
        fast_ray_options = replace(
            self.ray_options,
            initial_step=0.05,
            maximum_step=1.0,
            maximum_affine_length=100.0,
        )
        product = integrate_kerr_returning_radiation_fates(
            self.frame,
            self.surface,
            termination=fast_termination,
            ray_options=fast_ray_options,
            surface_options=self.surface_options,
            mu_order=4,
            psi_count=4,
            convergence_absolute_tolerance=0.05,
            maximum_whole_ray_traces=192,
            receiver_radius_bin_edges_over_mass=(
                self.calibration.isco_radius_over_mass,
                6.0,
                10.0,
                20.0,
                30.0,
            ),
        )
        self.assertEqual(product.fractions.total, 1.0)
        self.assertTrue(
            math.isclose(
                product.fractions.return_upper,
                0.0033334501812013486,
                rel_tol=0.0,
                abs_tol=2.0 * math.ulp(1.0),
            )
        )
        self.assertEqual(product.fractions.return_lower, 0.0)
        self.assertEqual(product.fractions.captured, 0.0)
        self.assertEqual(product.fractions.plunge_sink, 0.0)
        self.assertTrue(product.convergence.converged)
        self.assertLessEqual(
            product.convergence.resolution_maximum_absolute_difference,
            0.05,
        )
        self.assertLessEqual(
            product.convergence.periodic_phase_maximum_absolute_difference,
            0.05,
        )
        self.assertEqual(product.whole_ray_traces_consumed, 192)
        self.assertEqual(len(product.full_grid_direction_audit), 16)
        self.assertEqual(len(product.half_mu_grid_direction_audit), 8)
        self.assertEqual(len(product.half_psi_grid_direction_audit), 8)
        self.assertEqual(len(product.phase_shifted_direction_audit), 16)
        # The 48 direction evaluations already include each primitive's two
        # whole rays plus its mandatory two-ray public replay.  Mock coverage
        # above tests whole-product replay without doubling this real fixture.


if __name__ == "__main__":
    unittest.main()
