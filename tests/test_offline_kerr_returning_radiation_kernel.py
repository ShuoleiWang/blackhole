from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
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
from offline.kerr_finite_thickness_surface import KerrFiniteThicknessMultiSurface
import offline.kerr_returning_radiation_kernel as kernel_module
from offline.kerr_returning_radiation_kernel import (
    FiniteVolumeFateFractions,
    IMPLEMENTATION_ID,
    SCIENTIFIC_STATUS,
    KerrForwardReturningRadiationKernel,
    KerrForwardReturningRadiationKernelProjection,
    KerrReturningRadiationKernelConvergenceError,
    KerrReturningRadiationKernelError,
    KerrReturningRadiationKernelPolicy,
    KerrReturningRadiationKernelVerificationError,
    integrate_kerr_returning_radiation_energy_kernel,
    verify_and_reduce_kerr_returning_radiation_energy_kernel,
    verify_and_reduce_kerr_returning_radiation_kernel_projection,
    verify_kerr_returning_radiation_energy_kernel,
    verify_kerr_returning_radiation_kernel_projection,
)
from offline.returning_radiation import AxisymmetricReturningRadiationKernel


_REAL_TRACE_DIRECTION = kernel_module._trace_direction


class AlwaysEqualFloat(float):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


def _identity_for(*values: object) -> str:
    return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()


def _symmetric_g2_four_block_classifier(
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
    return kernel_module._DirectionTransport(
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


class OfflineKerrReturningRadiationKernelTests(unittest.TestCase):
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
            kernel_module,
            "_trace_direction",
            side_effect=_symmetric_g2_four_block_classifier,
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
        return integrate_kerr_returning_radiation_energy_kernel(**arguments)

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

    def test_every_supported_kernel_order_has_a_resolved_float64_rule(self) -> None:
        legacy_noncycling = hashlib.sha256()
        for order in range(1, 65):
            with self.subTest(order=order):
                rule = kernel_module._gauss_legendre_unit_interval(order)
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
                if order != 62:
                    for node, weight in rule:
                        legacy_noncycling.update(
                            f"{order}\0{node.hex()}\0{weight.hex()}\n".encode(
                                "ascii"
                            )
                        )
        self.assertEqual(
            legacy_noncycling.hexdigest(),
            "c021a5dd321d8f9e28813e2976248f743a28a060612fa662852c13f7f462244c",
        )

    def test_scientific_scope_and_forward_coefficient_are_explicit(self) -> None:
        self.assertEqual(SCIENTIFIC_STATUS["implementationId"], IMPLEMENTATION_ID)
        self.assertTrue(SCIENTIFIC_STATUS["isFiniteGridEnergyOnlyKernel"])
        self.assertTrue(SCIENTIFIC_STATUS["isSameCodePhysicsReplayVerified"])
        self.assertFalse(SCIENTIFIC_STATUS["hasIndependentReceiverReverseRayOracle"])
        for key in (
            "includesReturningRadiationStressWorkFS",
            "includesSpectralRedistribution",
            "includesSolvedAtmosphere",
            "isGeneralRelativisticMagnetohydrodynamics",
            "isCompleteKerrbb",
        ):
            self.assertIs(SCIENTIFIC_STATUS[key], False)
        self.assertIn("g^2", SCIENTIFIC_STATUS["coefficientEquation"])
        self.assertIn(
            "process-local fresh-issued",
            SCIENTIFIC_STATUS["primitiveConsumptionBoundary"],
        )
        self.assertIn("F_S", SCIENTIFIC_STATUS["prohibitedClaim"])
        self.assertEqual(
            SCIENTIFIC_STATUS["fateFractionClosureMaximumUlpsAtUnity"],
            8,
        )
        self.assertEqual(
            SCIENTIFIC_STATUS["fateFractionClosureMaximumNextafterSteps"],
            8,
        )
        self.assertEqual(
            SCIENTIFIC_STATUS[
                "fateFractionClosureMaximumCorrectionUlpsAtUnity"
            ],
            9,
        )
        self.assertIs(
            SCIENTIFIC_STATUS["fateFractionClosureIsMissingFateAllowance"],
            False,
        )
        self.assertIn(
            "largest positive fate",
            SCIENTIFIC_STATUS["fateFractionBinary64ClosureMethod"],
        )
        self.assertTrue(
            SCIENTIFIC_STATUS[
                "fateFractionStructuralExactOneFatePerDirection"
            ]
        )
        self.assertFalse(
            SCIENTIFIC_STATUS["fateFractionResidualCauseNumericallyIdentified"]
        )
        self.assertIn(
            "representation-only",
            SCIENTIFIC_STATUS["fateFractionMergeAccounting"],
        )
        self.assertEqual(
            SCIENTIFIC_STATUS[
                "fateFractionSingleFsumMaximumCorrectRoundingErrorAtUnity"
            ].hex(),
            (0.5 * math.ulp(1.0)).hex(),
        )
        self.assertEqual(
            SCIENTIFIC_STATUS[
                "fateFractionPrePostStructuralUndetectableMass"
            ].hex(),
            0.0.hex(),
        )
        capabilities = self.result.model_descriptor()["capabilities"]
        self.assertEqual(
            capabilities["fateFractionClosureMaximumUlpsAtUnity"],
            8,
        )
        self.assertFalse(
            capabilities["fateFractionClosureIsMissingFateAllowance"]
        )
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["isCompleteKerrbb"] = True

    def test_binary64_fate_closure_repairs_a_single_residual_add_counterexample(
        self,
    ) -> None:
        raw = (
            float.fromhex("0x1.0000000000000p-2"),
            0.0,
            0.0,
            float.fromhex("0x1.8000000000003p-1"),
            0.0,
        )
        residual = 1.0 - math.fsum(raw)
        one_shot = list(raw)
        one_shot[3] = math.fsum((one_shot[3], residual))
        self.assertEqual(math.fsum(raw).hex(), "0x1.0000000000002p+0")
        self.assertEqual(math.fsum(one_shot).hex(), "0x1.fffffffffffffp-1")

        first = kernel_module._FateContribution(0, raw[0])
        second = kernel_module._FateContribution(1, raw[3])
        closed = kernel_module._audited_fate_fraction_partition(
            [first, second],
            [[first], [], [], [second], []],
            expected_direction_count=2,
        )

        self.assertEqual(math.fsum(closed).hex(), 1.0.hex())
        self.assertEqual(closed[0].hex(), raw[0].hex())
        self.assertEqual(closed[3].hex(), "0x1.8000000000000p-1")

    def test_five_pass_evidence_records_exhaustive_maximum_weight_witness(self) -> None:
        rebuilt, evidence = (
            kernel_module._rebuild_verified_kerr_returning_radiation_energy_kernel(
                self.result
            )
        )
        self.assertEqual(
            rebuilt.model_descriptor_sha256,
            self.result.model_descriptor_sha256,
        )
        self.assertEqual(
            tuple((item.pass_index, item.pass_name) for item in evidence.passes),
            (
                (0, "full"),
                (1, "half-rho"),
                (2, "half-mu"),
                (3, "half-psi"),
                (4, "phase-shifted"),
            ),
        )
        self.assertEqual(
            sum(item.direction_evaluations for item in evidence.passes),
            self.result.direction_evaluations_consumed,
        )
        for item in evidence.passes:
            witness = item.maximum_normalized_sample_weight_witness
            areas = (
                evidence.upper_annulus_areas_over_mass_squared
                if witness.source_face == UPPER
                else evidence.lower_annulus_areas_over_mass_squared
            )
            exhaustive = []
            directions = kernel_module.kerrbb_d20_emitted_flux_direction_nodes(
                item.mu_order,
                item.psi_count,
                phase_cells=item.phase_cells,
            )
            for face, face_areas in (
                (UPPER, evidence.upper_annulus_areas_over_mass_squared),
                (LOWER, evidence.lower_annulus_areas_over_mass_squared),
            ):
                for source_index, (inner, outer) in enumerate(
                    zip(
                        evidence.annulus_edges_over_mass,
                        evidence.annulus_edges_over_mass[1:],
                    )
                ):
                    for _rho, rho_area in kernel_module._rho_area_nodes(
                        self.surface,
                        inner,
                        outer,
                        face,
                        item.rho_order,
                        face_areas[source_index],
                    ):
                        exhaustive.extend(
                            rho_area * node.normalized_emitted_flux_weight
                            / face_areas[source_index]
                            for node in directions
                        )
            self.assertEqual(
                item.maximum_normalized_sample_weight.hex(),
                max(exhaustive).hex(),
            )
            reconstructed = (
                witness.rho_area_over_mass_squared
                * witness.normalized_emitted_flux_direction_weight
                / areas[witness.source_annulus_index]
            )
            self.assertEqual(
                reconstructed.hex(), item.maximum_normalized_sample_weight.hex()
            )
        with self.assertRaisesRegex(TypeError, "built only"):
            kernel_module.KerrForwardReturningRadiationConvergenceEvidence()

    def test_binary64_fate_closure_replays_real_half_psi_cache_vector(self) -> None:
        raw = (
            float.fromhex("0x1.14611aa5fa09ep-11"),
            0.0,
            0.0,
            float.fromhex("0x1.ffbae7b956819p-1"),
            0.0,
        )
        residual = 1.0 - math.fsum(raw)
        one_shot = list(raw)
        one_shot[3] = math.fsum((one_shot[3], residual))
        self.assertEqual(math.fsum(raw).hex(), "0x1.0000000000001p+0")
        self.assertEqual(math.fsum(one_shot).hex(), "0x1.fffffffffffffp-1")

        closed = kernel_module._close_fate_fraction_binary64_roundoff(
            raw,
            independently_accumulated_total=math.fsum(raw),
        )

        self.assertEqual(math.fsum(closed).hex(), 1.0.hex())
        self.assertEqual(closed[0].hex(), raw[0].hex())
        self.assertEqual(closed[3].hex(), "0x1.ffbae7b956818p-1")

    def test_binary64_fate_closure_rejects_non_roundoff_and_invalid_inputs(
        self,
    ) -> None:
        too_large = (
            1.0,
            9.0 * math.ulp(1.0),
            0.0,
            0.0,
            0.0,
        )
        invalid = (
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, -math.ulp(1.0), 0.0, 0.0, 0.0),
            (1.0, -0.0, 0.0, 0.0, 0.0),
            (1.0, math.nan, 0.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelError,
            "strict binary64 representation gate",
        ):
            kernel_module._close_fate_fraction_binary64_roundoff(
                too_large,
                independently_accumulated_total=math.fsum(too_large),
            )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(
                KerrReturningRadiationKernelError
            ):
                kernel_module._close_fate_fraction_binary64_roundoff(
                    raw,
                    independently_accumulated_total=math.fsum(raw),
                )
        with self.assertRaises(TypeError):
            kernel_module._close_fate_fraction_binary64_roundoff(
                (1, 0.0, 0.0, 0.0, 0.0),
                independently_accumulated_total=1.0,
            )
        with self.assertRaisesRegex(ValueError, "canonical positive zero"):
            FiniteVolumeFateFractions(1.0, -0.0, 0.0, 0.0, 0.0)

    def test_binary64_fate_closure_enforces_ulp_step_and_correction_bounds(
        self,
    ) -> None:
        def stepped(value: float, direction: float, count: int) -> float:
            for _step in range(count):
                value = math.nextafter(value, direction)
            return value

        at_eight_ulps = stepped(1.0, math.inf, 8)
        closed = kernel_module._close_fate_fraction_binary64_roundoff(
            (at_eight_ulps, 0.0, 0.0, 0.0, 0.0),
            independently_accumulated_total=at_eight_ulps,
        )
        self.assertEqual(math.fsum(closed).hex(), 1.0.hex())

        for count, label in ((17, "8.5 ULP"), (18, "9 ULP")):
            outside = stepped(1.0, 0.0, count)
            with self.subTest(boundary=label), self.assertRaisesRegex(
                KerrReturningRadiationKernelError,
                "strict binary64 representation gate",
            ):
                kernel_module._close_fate_fraction_binary64_roundoff(
                    (outside, 0.0, 0.0, 0.0, 0.0),
                    independently_accumulated_total=outside,
                )

        needs_neighbour_step = (
            float.fromhex("0x1.0000000000000p-2"),
            0.0,
            0.0,
            float.fromhex("0x1.8000000000003p-1"),
            0.0,
        )
        with patch.object(
            kernel_module,
            "_FATE_CLOSURE_MAXIMUM_NEXTAFTER_STEPS",
            0,
        ), self.assertRaisesRegex(
            KerrReturningRadiationKernelError,
            "step bound",
        ):
            kernel_module._close_fate_fraction_binary64_roundoff(
                needs_neighbour_step,
                independently_accumulated_total=math.fsum(
                    needs_neighbour_step
                ),
            )
        with patch.object(
            kernel_module,
            "_FATE_CLOSURE_MAXIMUM_CORRECTION_ULPS_AT_UNITY",
            0,
        ), self.assertRaisesRegex(
            KerrReturningRadiationKernelError,
            "correction exceeds",
        ):
            kernel_module._close_fate_fraction_binary64_roundoff(
                needs_neighbour_step,
                independently_accumulated_total=math.fsum(
                    needs_neighbour_step
                ),
            )

    def test_binary64_fate_closure_has_stable_ties_and_exact_schema(self) -> None:
        half_up = math.nextafter(0.5, math.inf)
        tied = (half_up, half_up, 0.0, 0.0, 0.0)
        closed = kernel_module._close_fate_fraction_binary64_roundoff(
            tied,
            independently_accumulated_total=math.fsum(tied),
        )
        self.assertNotEqual(closed[0].hex(), tied[0].hex())
        self.assertEqual(closed[1].hex(), tied[1].hex())
        self.assertEqual(math.fsum(closed).hex(), 1.0.hex())

        class TupleSubclass(tuple):
            pass

        invalid_raw_values = (
            TupleSubclass((1.0, 0.0, 0.0, 0.0, 0.0)),
            (AlwaysEqualFloat(1.0), 0.0, 0.0, 0.0, 0.0),
            (1.0, math.inf, 0.0, 0.0, 0.0),
        )
        for raw in invalid_raw_values:
            with self.subTest(raw=raw), self.assertRaises(
                (TypeError, KerrReturningRadiationKernelError)
            ):
                kernel_module._close_fate_fraction_binary64_roundoff(
                    raw,
                    independently_accumulated_total=1.0,
                )
        with self.assertRaises(TypeError):
            kernel_module._close_fate_fraction_binary64_roundoff(
                (1.0, 0.0, 0.0, 0.0, 0.0),
                independently_accumulated_total=AlwaysEqualFloat(1.0),
            )

    def test_binary64_exact_one_is_an_honest_correctly_rounded_contract(
        self,
    ) -> None:
        just_below_half = math.nextafter(0.5, 0.0)
        raw = (0.5, just_below_half, 0.0, 0.0, 0.0)
        exact_sum = sum(
            (Fraction.from_float(value) for value in raw),
            start=Fraction(0, 1),
        )
        self.assertEqual(exact_sum, Fraction(1, 1) - Fraction(1, 2**54))
        self.assertEqual(math.fsum(raw).hex(), 1.0.hex())

        closed = kernel_module._close_fate_fraction_binary64_roundoff(
            raw,
            independently_accumulated_total=math.fsum(raw),
        )

        self.assertEqual(closed, raw)
        self.assertFalse(
            SCIENTIFIC_STATUS["fateFractionResidualCauseNumericallyIdentified"]
        )

    def test_structural_fate_audit_rejects_missing_or_duplicated_entries(
        self,
    ) -> None:
        first = kernel_module._FateContribution(0, 0.25)
        second = kernel_module._FateContribution(1, 0.75)
        unclassified = [first, second]
        classified = [[first], [], [], [second], []]
        self.assertEqual(
            kernel_module._audited_fate_fraction_partition(
                unclassified,
                classified,
                expected_direction_count=2,
            ),
            (0.25, 0.0, 0.0, 0.75, 0.0),
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelError,
            "count differs",
        ):
            kernel_module._audited_fate_fraction_partition(
                unclassified,
                [[first], [], [], [], []],
                expected_direction_count=2,
            )
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelError,
            "ordinals are duplicated",
        ):
            kernel_module._audited_fate_fraction_partition(
                unclassified,
                [[first, first], [], [], [], []],
                expected_direction_count=2,
            )

    def test_structural_fate_audit_rejects_same_fsum_omit_duplicate_attack(
        self,
    ) -> None:
        small = 0.25
        large = 0.25
        for _step in range(4):
            large = math.nextafter(large, math.inf)
        rest = math.nextafter(0.5, math.inf)
        pre = [
            kernel_module._FateContribution(0, small),
            kernel_module._FateContribution(1, large),
            kernel_module._FateContribution(2, rest),
        ]
        attacked_post = [[pre[1], pre[1], pre[2]], [], [], [], []]
        self.assertEqual(
            math.fsum(item.normalized_weight for item in pre).hex(),
            "0x1.0000000000002p+0",
        )
        self.assertEqual(
            math.fsum(
                item.normalized_weight for item in attacked_post[0]
            ).hex(),
            "0x1.0000000000002p+0",
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelError,
            "ordinals are duplicated",
        ):
            kernel_module._audited_fate_fraction_partition(
                pre,
                attacked_post,
                expected_direction_count=3,
            )

        same_weight_pre = [
            kernel_module._FateContribution(0, 0.5),
            kernel_module._FateContribution(1, 0.5),
        ]
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelError,
            "ordinals are duplicated",
        ):
            kernel_module._audited_fate_fraction_partition(
                same_weight_pre,
                [[same_weight_pre[0], same_weight_pre[0]], [], [], [], []],
                expected_direction_count=2,
            )
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelError,
            "differs from its exact pre-entry",
        ):
            kernel_module._audited_fate_fraction_partition(
                same_weight_pre,
                [
                    [
                        kernel_module._FateContribution(
                            0,
                            math.nextafter(0.5, math.inf),
                        ),
                        same_weight_pre[1],
                    ],
                    [],
                    [],
                    [],
                    [],
                ],
                expected_direction_count=2,
            )
        with self.assertRaises((AttributeError, TypeError)):
            same_weight_pre[0].normalized_weight = 0.25

    def test_merged_fates_use_the_same_exact_binary64_closure(self) -> None:
        upper_only = FiniteVolumeFateFractions(1.0, 0.0, 0.0, 0.0, 0.0)
        lower_only = FiniteVolumeFateFractions(0.0, 1.0, 0.0, 0.0, 0.0)
        raw = (
            math.fsum((0.1,)) / math.fsum((0.1, 0.3)),
            math.fsum((0.3,)) / math.fsum((0.1, 0.3)),
            0.0,
            0.0,
            0.0,
        )
        self.assertEqual(math.fsum(raw).hex(), "0x1.fffffffffffffp-1")

        merged = kernel_module._merge_fates(
            (upper_only, lower_only),
            (0.1, 0.3),
            ((0, 1),),
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].total.hex(), 1.0.hex())
        self.assertEqual(merged[0].return_upper.hex(), "0x1.0000000000000p-2")
        self.assertEqual(merged[0].return_lower.hex(), "0x1.8000000000000p-1")
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            kernel_module._merge_fates(
                (upper_only, lower_only),
                (0.1, 0.3),
                ((1, 0),),
            )

    def test_g_equals_two_uses_g2_area_ratio_and_populates_four_blocks(self) -> None:
        result = self.result
        upper_area = result.upper_annulus_areas_over_mass_squared[0]
        lower_area = result.lower_annulus_areas_over_mass_squared[0]
        expected_uu = 2.0
        expected_lu = 2.0 * upper_area / lower_area
        expected_ul = 2.0 * lower_area / upper_area
        expected_ll = 2.0
        self.assertAlmostEqual(
            result.upper_receiver_upper_emitter_coefficients[0][0],
            expected_uu,
            places=13,
        )
        self.assertAlmostEqual(
            result.lower_receiver_upper_emitter_coefficients[0][0],
            expected_lu,
            places=13,
        )
        self.assertAlmostEqual(
            result.upper_receiver_lower_emitter_coefficients[0][0],
            expected_ul,
            places=13,
        )
        self.assertAlmostEqual(
            result.lower_receiver_lower_emitter_coefficients[0][0],
            expected_ll,
            places=13,
        )
        self.assertAlmostEqual(
            result.upper_emitter_g2_returned_power_columns[0], 4.0, places=13
        )
        self.assertAlmostEqual(
            result.lower_emitter_g2_returned_power_columns[0], 4.0, places=13
        )

    def test_fates_are_independent_of_g2_and_close_exactly(self) -> None:
        for fate in (
            self.result.upper_emitter_fate_fractions[0],
            self.result.lower_emitter_fate_fractions[0],
        ):
            self.assertEqual(fate.total, 1.0)
            self.assertAlmostEqual(fate.return_upper, 0.5, places=14)
            self.assertAlmostEqual(fate.return_lower, 0.5, places=14)
            self.assertEqual(fate.captured, 0.0)
            self.assertEqual(fate.escaped, 0.0)
            self.assertEqual(fate.plunge_sink, 0.0)

    def test_each_column_obeys_proper_area_energy_closure(self) -> None:
        result = self.result
        for source_face in (UPPER, LOWER):
            source_area = (
                result.upper_annulus_areas_over_mass_squared[0]
                if source_face == UPPER
                else result.lower_annulus_areas_over_mass_squared[0]
            )
            upper_coefficient = (
                result.upper_receiver_upper_emitter_coefficients[0][0]
                if source_face == UPPER
                else result.upper_receiver_lower_emitter_coefficients[0][0]
            )
            lower_coefficient = (
                result.lower_receiver_upper_emitter_coefficients[0][0]
                if source_face == UPPER
                else result.lower_receiver_lower_emitter_coefficients[0][0]
            )
            reconstructed = math.fsum(
                (
                    result.upper_annulus_areas_over_mass_squared[0]
                    * upper_coefficient
                    / source_area,
                    result.lower_annulus_areas_over_mass_squared[0]
                    * lower_coefficient
                    / source_area,
                )
            )
            direct = (
                result.upper_emitter_g2_returned_power_columns[0]
                if source_face == UPPER
                else result.lower_emitter_g2_returned_power_columns[0]
            )
            self.assertLessEqual(abs(reconstructed - direct), 2.0e-14)

    def test_full_half_rho_mu_psi_and_phase_gates_are_published(self) -> None:
        convergence = self.result.convergence
        self.assertTrue(convergence.converged)
        self.assertTrue(self.result.fine_coarse_receiver_bin_topology_verified)
        for item in (
            convergence.half_rho,
            convergence.half_mu,
            convergence.half_psi,
            convergence.phase_shifted,
        ):
            self.assertTrue(item.converged)
            self.assertLessEqual(item.matrix_maximum_scaled_difference, 1.0)
            self.assertLessEqual(item.g2_column_maximum_scaled_difference, 1.0)
            self.assertLessEqual(item.fate_maximum_scaled_difference, 1.0)
        descriptor = self.result.model_descriptor()
        self.assertTrue(
            descriptor["convergence"]["eachMatrixCoefficientAndG2ColumnGated"]
        )
        self.assertTrue(
            descriptor["convergence"]["fineCoarseReceiverBinTopologyVerified"]
        )
        self.assertIn("g^4", descriptor["coefficient"]["forbiddenSingleRayFactors"])
        self.assertIn(
            "same user-declared annulus",
            descriptor["coefficient"]["fineCoarseReceiverTopologyGate"],
        )
        self.assertEqual(descriptor["coefficient"]["unconditionalFaceMultiplicity"], 1)

    def test_budget_counts_two_issued_rays_and_preserves_public_replay(self) -> None:
        self.assertEqual(self.result.direction_evaluations_consumed, 448)
        self.assertEqual(self.result.whole_ray_traces_consumed, 896)
        descriptor = self.result.model_descriptor()["workBudget"]
        self.assertEqual(descriptor["primitiveWholeRaysPerDirection"], 2)
        self.assertEqual(
            descriptor["productionIssuedPrimitiveReplayWholeRaysPerDirection"],
            0,
        )
        self.assertEqual(descriptor["publicPrimitiveReplayWholeRaysPerDirection"], 2)

    def test_work_only_change_preserves_scientific_float_and_sample_audits(self) -> None:
        self.assertEqual(
            self.result.full_grid_sample_audit_sha256,
            "ebef96b0a7399fd811f93032a65220a00522ae7d3b60b8929db107d2b1378a1c",
        )
        self.assertEqual(
            self.result.half_rho_sample_audit_sha256,
            "05d17dd36657bd9029960a5162df650cba915d707395fcf5692ba77c964c2c20",
        )
        self.assertEqual(
            self.result.half_mu_sample_audit_sha256,
            "b8d9e72979a0ad4c137dc7e1b6a7780275e6e8bb858f3d4a333080157a75709f",
        )
        self.assertEqual(
            self.result.half_psi_sample_audit_sha256,
            "eb843b9f0f88fa96e9d18458c88a3f05810b047e152a40e079bdca2a5c06225b",
        )
        self.assertEqual(
            self.result.phase_shifted_sample_audit_sha256,
            "56963655a9ffc20919da8ddc6325d5a5a3deaf64fb316e559fcda7d072807e5e",
        )
        self.assertEqual(
            self.result.upper_receiver_upper_emitter_coefficients[0][0].hex(),
            float.fromhex("0x1.0000000000001p+1").hex(),
        )
        self.assertEqual(
            self.result.upper_receiver_lower_emitter_coefficients[0][0].hex(),
            float.fromhex("0x1.0000000000001p+1").hex(),
        )
        self.assertEqual(
            self.result.lower_receiver_upper_emitter_coefficients[0][0].hex(),
            2.0.hex(),
        )
        self.assertEqual(
            self.result.lower_receiver_lower_emitter_coefficients[0][0].hex(),
            2.0.hex(),
        )
        self.assertEqual(
            tuple(value.hex() for value in self.result.upper_emitter_fate_fractions[0].as_tuple()),
            (
                float.fromhex("0x1.0000000000001p-1").hex(),
                0.5.hex(),
                0.0.hex(),
                0.0.hex(),
                0.0.hex(),
            ),
        )
        self.assertEqual(
            self.result.upper_emitter_g2_returned_power_columns[0].hex(),
            4.0.hex(),
        )

    def test_strict_self_replay_and_live_matrix_tamper_rejection(self) -> None:
        verify_kerr_returning_radiation_energy_kernel(self.result)
        forged_rows = (
            (
                self.result.upper_receiver_upper_emitter_coefficients[0][0] + 1.0,
            ),
        )
        forged = self.forge(
            self.result,
            upper_receiver_upper_emitter_coefficients=forged_rows,
        )
        with self.assertRaises(KerrReturningRadiationKernelVerificationError):
            verify_kerr_returning_radiation_energy_kernel(forged)

    def test_axisymmetric_reduction_requires_symmetry_and_uses_all_four_blocks(self) -> None:
        with (
            patch.object(
                kernel_module,
                "verify_kerr_returning_radiation_energy_kernel",
                wraps=verify_kerr_returning_radiation_energy_kernel,
            ) as verifier,
            patch.object(
                kernel_module,
                "verify_and_reduce_kerr_returning_radiation_energy_kernel",
                wraps=verify_and_reduce_kerr_returning_radiation_energy_kernel,
            ) as combiner,
        ):
            reduced = self.result.to_axisymmetric_energy_kernel()
        self.assertEqual(combiner.call_count, 1)
        self.assertEqual(verifier.call_count, 1)
        self.assertIs(type(reduced), AxisymmetricReturningRadiationKernel)
        expected = 0.5 * math.fsum(
            (
                self.result.upper_receiver_upper_emitter_coefficients[0][0],
                self.result.upper_receiver_lower_emitter_coefficients[0][0],
                self.result.lower_receiver_upper_emitter_coefficients[0][0],
                self.result.lower_receiver_lower_emitter_coefficients[0][0],
            )
        )
        self.assertEqual(reduced.receiver_emitter_coefficients[0][0], expected)
        expected_result = AxisymmetricReturningRadiationKernel(
            annulus_radii_over_mass=(
                self.result.annulus_representative_radii_over_mass[0],
            ),
            receiver_emitter_coefficients=((expected,),),
            ray_kernel_producer_id=(
                f"{IMPLEMENTATION_ID}:{self.result.model_descriptor_sha256}"
            ),
        )
        self.assertEqual(
            reduced.canonical_descriptor_json,
            expected_result.canonical_descriptor_json,
        )
        with self.assertRaises(ValueError):
            self.result.to_axisymmetric_energy_kernel(False)

    def test_asymmetric_physical_blocks_fail_axisymmetric_reduction(self) -> None:
        def asymmetric(*args):
            transport = _symmetric_g2_four_block_classifier(*args)
            source_face = args[6]
            ratio = 2.0 if source_face == UPPER else 1.0
            return kernel_module._DirectionTransport(
                transport.fate,
                transport.receiver_face,
                transport.receiver_radius_over_mass,
                ratio,
                ratio * ratio,
                transport.primitive_descriptor_sha256,
                transport.coarse_receiver_face,
                transport.coarse_receiver_radius_over_mass,
            )

        with patch.object(kernel_module, "_trace_direction", side_effect=asymmetric):
            result = self.build()
            with self.assertRaises(KerrReturningRadiationKernelConvergenceError):
                result.to_axisymmetric_energy_kernel()

    def test_bin_edges_complete_coverage_exact_type_and_budget_fail_closed(self) -> None:
        inner, outer = self.edges
        with self.assertRaisesRegex(ValueError, "exactly and completely cover"):
            self.build(annulus_edges_over_mass=(inner, math.nextafter(outer, 0.0)))
        with self.assertRaises(ValueError):
            self.build(
                annulus_edges_over_mass=(AlwaysEqualFloat(inner), outer)
            )
        tiny_budget = replace(self.policy, maximum_direction_evaluations=447)
        with self.assertRaisesRegex(ValueError, "448 direction evaluations"):
            self.build(policy=tiny_budget)

    def test_trace_options_and_termination_require_exact_canonical_schema(self) -> None:
        boolean_float = RayTraceOptions(absolute_tolerance=True)
        with self.assertRaises(KerrReturningRadiationKernelVerificationError):
            self.build(ray_options=boolean_float)
        integer_float = SurfaceEventOptions(absolute_tolerance=1)
        with self.assertRaises(KerrReturningRadiationKernelVerificationError):
            self.build(surface_options=integer_float)
        tampered_termination = KerrOblateTermination.horizon_worldtube(
            self.metric,
            escape_radius_m=20.0,
            offset_m=0.02,
        )
        object.__setattr__(
            tampered_termination,
            "escape_radius_m",
            AlwaysEqualFloat(tampered_termination.escape_radius_m),
        )
        with self.assertRaises(KerrReturningRadiationKernelVerificationError):
            self.build(termination=tampered_termination)

    def test_return_outside_declared_grid_fails_closed(self) -> None:
        def outside(*args):
            source_face = args[6]
            del source_face
            return kernel_module._DirectionTransport(
                "return-upper",
                UPPER,
                math.nextafter(self.edges[-1], math.inf),
                1.0,
                1.0,
                "a" * 64,
                UPPER,
                math.nextafter(self.edges[-1], math.inf),
            )

        with patch.object(kernel_module, "_trace_direction", side_effect=outside):
            with self.assertRaisesRegex(KerrReturningRadiationKernelError, "outside"):
                self.build()

    def test_fine_coarse_receiver_annulus_split_fails_inside_integrator(self) -> None:
        middle = 0.5 * math.fsum(self.edges)

        def split_receiver(*args):
            return kernel_module._DirectionTransport(
                "return-upper",
                UPPER,
                math.nextafter(middle, -math.inf),
                1.0,
                1.0,
                _identity_for(*args[6:]),
                UPPER,
                math.nextafter(middle, math.inf),
            )

        with patch.object(
            kernel_module,
            "_trace_direction",
            side_effect=split_receiver,
        ):
            with self.assertRaisesRegex(
                KerrReturningRadiationKernelConvergenceError,
                "different receiver annuli",
            ):
                self.build(
                    annulus_edges_over_mass=(
                        self.edges[0],
                        middle,
                        self.edges[-1],
                    )
                )

    def test_exact_internal_and_outer_bin_edges_have_deterministic_ownership(self) -> None:
        middle = 0.5 * math.fsum(self.edges)
        edges = (self.edges[0], middle, self.edges[-1])
        self.assertEqual(kernel_module._receiver_bin_index(self.edges[0], edges), 0)
        self.assertEqual(kernel_module._receiver_bin_index(middle, edges), 1)
        self.assertEqual(kernel_module._receiver_bin_index(self.edges[-1], edges), 1)

    def test_capture_escape_and_plunge_are_diagnostics_only(self) -> None:
        fate_field = {
            "captured": "captured",
            "escaped": "escaped",
            "plunge-sink": "plunge_sink",
        }
        for fate, field in fate_field.items():
            def nonreturning(*args, selected_fate=fate):
                return kernel_module._DirectionTransport(
                    selected_fate,
                    None,
                    None,
                    None,
                    0.0,
                    _identity_for(selected_fate, *args[6:]),
                )

            with self.subTest(fate=fate), patch.object(
                kernel_module,
                "_trace_direction",
                side_effect=nonreturning,
            ):
                result = self.build()
                for block in (
                    result.upper_receiver_upper_emitter_coefficients,
                    result.upper_receiver_lower_emitter_coefficients,
                    result.lower_receiver_upper_emitter_coefficients,
                    result.lower_receiver_lower_emitter_coefficients,
                ):
                    self.assertEqual(block, ((0.0,),))
                self.assertEqual(
                    result.upper_emitter_g2_returned_power_columns,
                    (0.0,),
                )
                self.assertEqual(
                    result.lower_emitter_g2_returned_power_columns,
                    (0.0,),
                )
                self.assertEqual(
                    getattr(result.upper_emitter_fate_fractions[0], field),
                    1.0,
                )
                self.assertEqual(
                    getattr(result.lower_emitter_fate_fractions[0], field),
                    1.0,
                )

    def test_periodic_phase_disagreement_fails_closed(self) -> None:
        def narrow_sector(*args):
            source_radius = args[7]
            psi = args[9]
            if psi < 0.4:
                return kernel_module._DirectionTransport(
                    "return-upper",
                    UPPER,
                    source_radius,
                    1.0,
                    1.0,
                    _identity_for(*args[6:]),
                    UPPER,
                    source_radius,
                )
            return kernel_module._DirectionTransport(
                "escaped", None, None, None, 0.0, _identity_for(*args[6:])
            )

        strict = replace(
            self.policy,
            absolute_tolerance=1.0e-6,
            relative_tolerance=1.0e-6,
        )
        with patch.object(kernel_module, "_trace_direction", side_effect=narrow_sector):
            with self.assertRaises(KerrReturningRadiationKernelConvergenceError):
                self.build(policy=strict)

    def test_adjacent_annulus_merging_preserves_area_energy_action(self) -> None:
        middle = 0.5 * math.fsum(self.edges)
        fine = self.build(
            annulus_edges_over_mass=(self.edges[0], middle, self.edges[-1])
        )
        projected = fine.coarsen_annuli(self.edges)
        self.assertIs(type(projected), KerrForwardReturningRadiationKernelProjection)
        verify_kerr_returning_radiation_kernel_projection(projected)
        with (
            patch.object(
                kernel_module,
                "verify_kerr_returning_radiation_kernel_projection",
                wraps=verify_kerr_returning_radiation_kernel_projection,
            ) as verifier,
            patch.object(
                kernel_module,
                "verify_and_reduce_kerr_returning_radiation_kernel_projection",
                wraps=verify_and_reduce_kerr_returning_radiation_kernel_projection,
            ) as combiner,
        ):
            reduced = projected.to_axisymmetric_energy_kernel()
        self.assertEqual(combiner.call_count, 1)
        self.assertEqual(verifier.call_count, 1)
        expected_coefficient = 0.5 * math.fsum(
            (
                projected.upper_receiver_upper_emitter_coefficients[0][0],
                projected.upper_receiver_lower_emitter_coefficients[0][0],
                projected.lower_receiver_upper_emitter_coefficients[0][0],
                projected.lower_receiver_lower_emitter_coefficients[0][0],
            )
        )
        expected_reduced = AxisymmetricReturningRadiationKernel(
            annulus_radii_over_mass=(
                projected.annulus_representative_radii_over_mass[0],
            ),
            receiver_emitter_coefficients=((expected_coefficient,),),
            ray_kernel_producer_id=(
                f"{IMPLEMENTATION_ID}:coarsened:"
                f"{projected.model_descriptor_sha256}"
            ),
        )
        self.assertEqual(
            reduced.canonical_descriptor_json,
            expected_reduced.canonical_descriptor_json,
        )
        self.assertAlmostEqual(
            projected.upper_emitter_g2_returned_power_columns[0], 4.0, places=13
        )
        self.assertAlmostEqual(
            projected.lower_emitter_g2_returned_power_columns[0], 4.0, places=13
        )
        projected.revalidate()
        with self.assertRaises(ValueError):
            fine.coarsen_annuli((self.edges[0], middle + 1.0e-8, self.edges[-1]))

    def test_one_real_kerr_direction_uses_public_primitive_and_g2(self) -> None:
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
        transport = _REAL_TRACE_DIRECTION(
            surface,
            termination,
            self.ray_options,
            self.surface_options,
            None,
            None,
            UPPER,
            8.0,
            0.02,
            0.5 * math.pi,
        )
        self.assertEqual(transport.fate, "return-upper")
        self.assertEqual(transport.receiver_face, UPPER)
        self.assertIsNotNone(transport.frequency_ratio)
        self.assertEqual(
            transport.g2.hex(),
            (transport.frequency_ratio * transport.frequency_ratio).hex(),
        )
        self.assertNotEqual(transport.g2.hex(), (transport.g2 * transport.g2).hex())
        self.assertEqual(transport.coarse_receiver_face, UPPER)
        self.assertIsNotNone(transport.coarse_receiver_radius_over_mass)
        seam = 0.5 * math.fsum(
            (
                transport.receiver_radius_over_mass,
                transport.coarse_receiver_radius_over_mass,
            )
        )
        seam_edges = (
            float(calibration.isco_radius_over_mass),
            seam,
            float(calibration.outer_radius_over_mass),
        )
        with self.assertRaisesRegex(
            KerrReturningRadiationKernelConvergenceError,
            "different receiver annuli",
        ):
            kernel_module._validated_return_receiver_bin(transport, seam_edges)

    def test_real_direction_callable_gate_is_frozen_against_single_rebinding(
        self,
    ) -> None:
        with (
            patch.object(
                kernel_module,
                "_trace_issued_kerr_returning_radiation_direction",
                side_effect=AssertionError("rebound issuer ran"),
            ),
            patch.object(
                kernel_module,
                "_consume_issued_kerr_returning_radiation_direction",
                side_effect=AssertionError("rebound consumer ran"),
            ),
        ):
            transport = _REAL_TRACE_DIRECTION(
                self.surface,
                self.termination,
                self.ray_options,
                self.surface_options,
                None,
                None,
                UPPER,
                7.0,
                0.02,
                0.5 * math.pi,
            )
        self.assertEqual(transport.fate, "return-upper")


if __name__ == "__main__":
    unittest.main()
