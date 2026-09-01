from __future__ import annotations

import ast
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest

from offline.kerr_finite_thickness_selected_oracle import (
    KerrFiniteThicknessSelectedOracleError,
    SCIENTIFIC_STATUS,
    configuration_from_sampler_descriptor,
    selected_ray_observed_intensities_nu,
    selected_ray_refined_observed_intensities_nu,
    trace_selected_ray,
    trace_selected_ray_refined,
)
from offline.kerr_selected_oracle import FixedRk4Options
import scripts.render_offline_kerr_finite_thickness_frame as renderer


ROOT = Path(__file__).resolve().parents[1]
FREQUENCIES_HZ = (5.0e14, 1.0e15)


class KerrFiniteThicknessSelectedOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        temporary = Path(cls.temporary.name)
        arguments = renderer.parse_args(
            [
                str(temporary / "unused-finite-selected-product"),
                "--cache",
                str(temporary / "unused-finite-selected-cache"),
                "--width",
                "1",
                "--height",
                "1",
                "--maximum-depth",
                "0",
                # A finite stretched worldtube avoids pretending that the
                # singular BL horizon chart has uniform fixed-step accuracy.
                "--horizon-offset-over-mass",
                "0.5",
                "--ray-maximum-affine-length-over-mass",
                "150",
                # This lower-face calibration ray has two real transparent
                # auxiliary crossings before the opaque hit.  A factor four
                # production coarse trace resolves the same topology without
                # making this selected-ray test needlessly slow.
                "--coarse-tolerance-multiplier",
                "4",
            ]
        )
        cls.plan = renderer.build_render_plan(arguments)
        cls.descriptor = cls.plan.sampler.descriptor()
        cls.configuration = configuration_from_sampler_descriptor(cls.descriptor)
        cls.options = FixedRk4Options(
            step_m=0.02,
            maximum_affine_length_m=150.0,
            maximum_steps=20_000,
        )
        selected = {
            "upper": (0.5, -0.5),
            "lower": (0.1, 0.1),
            "captured": (0.0, 0.0),
            "escaped": (0.5, 0.5),
        }
        cls.refinements = {
            name: trace_selected_ray_refined(
                cls.configuration,
                screen_x,
                screen_y,
                cls.options,
            )
            for name, (screen_x, screen_y) in selected.items()
        }
        cls.production = {
            name: cls.plan.sampler.sample(screen_x, screen_y, FREQUENCIES_HZ)
            for name, (screen_x, screen_y) in selected.items()
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_scope_declares_shared_and_independent_boundaries(self) -> None:
        self.assertIsInstance(SCIENTIFIC_STATUS, MappingProxyType)
        self.assertIn("canonical-BL", SCIENTIFIC_STATUS["sharedBlFramework"])
        self.assertTrue(SCIENTIFIC_STATUS["sharedPageThorneRadialScalar"])
        self.assertTrue(
            SCIENTIFIC_STATUS["independentlyImplementsFinitePhotosphere"]
        )
        self.assertTrue(
            SCIENTIFIC_STATUS[
                "independentlyImplementsOffEquatorialEmitterAndNormal"
            ]
        )
        self.assertTrue(SCIENTIFIC_STATUS["independentlyImplementsGAndSignedMu"])
        self.assertFalse(SCIENTIFIC_STATUS["usesProductionKerrSchildGeodesic"])
        self.assertFalse(
            SCIENTIFIC_STATUS["usesProductionAcceptedStepSurfaceLocator"]
        )
        self.assertFalse(SCIENTIFIC_STATUS["isFullFrameProof"])
        self.assertFalse(SCIENTIFIC_STATUS["isGeneralRelativisticMagnetohydrodynamics"])
        with self.assertRaises(TypeError):
            SCIENTIFIC_STATUS["isFullFrameProof"] = True  # type: ignore[index]

    def test_source_imports_only_independent_bl_framework_and_nt_scalar(self) -> None:
        source = ROOT / "offline" / "kerr_finite_thickness_selected_oracle.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        offline_imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("offline.")
        }
        self.assertEqual(
            offline_imports,
            {"offline.kerr_selected_oracle", "offline.novikov_thorne"},
        )

    def test_descriptor_is_strictly_replayed_and_content_hashed(self) -> None:
        encoded = json.dumps(
            self.descriptor,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.assertEqual(
            self.configuration.sampler_descriptor_sha256,
            hashlib.sha256(encoded).hexdigest(),
        )
        self.assertAlmostEqual(self.configuration.dimensionless_spin, 0.7)
        self.assertAlmostEqual(self.configuration.isco_radius_over_mass, 3.3931284701816304)
        self.assertEqual(self.configuration.outer_radius_over_mass, 25.0)
        self.assertEqual(
            self.configuration.photosphere_height_over_mass(
                self.configuration.isco_radius_over_mass
            ),
            0.0,
        )
        self.assertGreater(
            self.configuration.photosphere_height_over_mass(25.0),
            0.0,
        )

    def test_inconsistent_or_non_exact_descriptor_fails_closed(self) -> None:
        for foreign_version in (True, 1.0):
            wrong_version = copy.deepcopy(self.descriptor)
            wrong_version["version"] = foreign_version
            with self.subTest(version=repr(foreign_version)), self.assertRaisesRegex(
                KerrFiniteThicknessSelectedOracleError,
                "only version 1",
            ):
                configuration_from_sampler_descriptor(wrong_version)

        wrong_implementation = copy.deepcopy(self.descriptor)
        wrong_implementation["implementationId"] = "finite-approximation/v1"
        with self.assertRaisesRegex(
            KerrFiniteThicknessSelectedOracleError,
            "only .* is supported",
        ):
            configuration_from_sampler_descriptor(wrong_implementation)

        wrong_screen = copy.deepcopy(self.descriptor)
        wrong_screen["screenConvention"]["screenY"] = "opposite"
        with self.assertRaisesRegex(
            KerrFiniteThicknessSelectedOracleError,
            "expected exact",
        ):
            configuration_from_sampler_descriptor(wrong_screen)

        wrong_isco = copy.deepcopy(self.descriptor)
        wrong_isco["diskThermalProxy"]["iscoRadiusM"] *= 1.01
        with self.assertRaisesRegex(
            KerrFiniteThicknessSelectedOracleError,
            "ISCO disagrees",
        ):
            configuration_from_sampler_descriptor(wrong_isco)

        wrong_maximum_radius = copy.deepcopy(self.descriptor)
        wrong_maximum_radius["finiteThicknessSurface"][
            "maximumPhotosphereOblateRadiusM"
        ] *= 1.001
        with self.assertRaisesRegex(
            KerrFiniteThicknessSelectedOracleError,
            "maximum photosphere radius disagrees",
        ):
            configuration_from_sampler_descriptor(wrong_maximum_radius)

        boolean_mass = copy.deepcopy(self.descriptor)
        boolean_mass["metric"]["massM"] = True
        with self.assertRaisesRegex(
            KerrFiniteThicknessSelectedOracleError,
            "finite built-in number",
        ):
            configuration_from_sampler_descriptor(boolean_mass)

        class ForeignString(str):
            pass

        foreign_implementation = copy.deepcopy(self.descriptor)
        foreign_implementation["implementationId"] = ForeignString(
            self.descriptor["implementationId"]
        )
        with self.assertRaisesRegex(
            KerrFiniteThicknessSelectedOracleError,
            "exact finite JSON primitive",
        ):
            configuration_from_sampler_descriptor(foreign_implementation)

        class ForeignFloat(float):
            pass

        foreign_radius = copy.deepcopy(self.descriptor)
        foreign_radius["observer"]["radiusM"] = ForeignFloat(
            self.descriptor["observer"]["radiusM"]
        )
        with self.assertRaisesRegex(
            KerrFiniteThicknessSelectedOracleError,
            "exact finite JSON primitive",
        ):
            configuration_from_sampler_descriptor(foreign_radius)

        unknown_root = copy.deepcopy(self.descriptor)
        unknown_root["unboundPhysics"] = "forged"
        with self.assertRaisesRegex(
            KerrFiniteThicknessSelectedOracleError,
            "unexpected=.*unboundPhysics",
        ):
            configuration_from_sampler_descriptor(unknown_root)

        unknown_observer = copy.deepcopy(self.descriptor)
        unknown_observer["observer"]["unboundFrame"] = "forged"
        with self.assertRaisesRegex(
            KerrFiniteThicknessSelectedOracleError,
            "unexpected=.*unboundFrame",
        ):
            configuration_from_sampler_descriptor(unknown_observer)

        forged_velocity = copy.deepcopy(self.descriptor)
        forged_velocity["observer"]["fourVelocity"] = [1.0, 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(
            KerrFiniteThicknessSelectedOracleError,
            "independently rebuilt KS ZAMO",
        ):
            configuration_from_sampler_descriptor(forged_velocity)

        class ForeignDict(dict):
            pass

        with self.assertRaisesRegex(
            KerrFiniteThicknessSelectedOracleError,
            "exact object",
        ):
            configuration_from_sampler_descriptor(ForeignDict(self.descriptor))

    def test_four_real_selected_outcomes_and_h_half_h_convergence(self) -> None:
        upper = self.refinements["upper"]
        lower = self.refinements["lower"]
        captured = self.refinements["captured"]
        escaped = self.refinements["escaped"]
        for expected, refinement in (
            ("upper", upper),
            ("lower", lower),
            ("captured", captured),
            ("escaped", escaped),
        ):
            with self.subTest(expected=expected):
                self.assertTrue(refinement.outcome_agrees)
                self.assertTrue(refinement.face_agrees)
                self.assertEqual(refinement.coarse.outcome, expected)
                self.assertEqual(refinement.fine.outcome, expected)

        self.assertLess(upper.terminal_radius_difference_m, 1.0e-10)
        self.assertLess(upper.pseudo_cylindrical_radius_difference_m, 1.0e-10)
        self.assertLess(upper.relative_g_difference, 1.0e-12)
        self.assertLess(upper.signed_mu_difference, 1.0e-12)
        self.assertLess(lower.terminal_radius_difference_m, 2.0e-8)
        self.assertLess(lower.pseudo_cylindrical_radius_difference_m, 2.0e-8)
        self.assertLess(lower.relative_g_difference, 1.0e-9)
        self.assertLess(lower.signed_mu_difference, 1.0e-8)
        self.assertEqual(captured.terminal_radius_difference_m, 0.0)
        self.assertEqual(escaped.terminal_radius_difference_m, 0.0)
        self.assertGreater(captured.affine_length_difference_m, 0.0)
        self.assertLess(captured.affine_length_difference_m, 3.0e-8)
        self.assertGreater(escaped.affine_length_difference_m, 0.0)
        self.assertLess(escaped.affine_length_difference_m, 1.0e-9)
        self.assertAlmostEqual(
            captured.fine.terminal_radius_m,
            self.configuration.capture_radius_m,
            places=13,
        )
        self.assertAlmostEqual(
            escaped.fine.terminal_radius_m,
            self.configuration.escape_radius_m,
            places=13,
        )

        self.assertLess(upper.fine.maximum_hamiltonian_residual, 1.0e-12)
        self.assertLess(upper.fine.maximum_relative_carter_drift, 1.0e-12)
        self.assertLess(lower.fine.maximum_hamiltonian_residual, 1.0e-9)
        self.assertLess(lower.fine.maximum_relative_carter_drift, 1.0e-9)
        # Capture terminates on a deliberately finite stretched worldtube;
        # this bound is honest for fixed-step BL near its stiff horizon chart.
        self.assertLess(captured.fine.maximum_hamiltonian_residual, 1.0e-7)
        self.assertLess(captured.fine.maximum_relative_carter_drift, 2.0e-10)
        self.assertLess(escaped.fine.maximum_hamiltonian_residual, 1.0e-12)
        self.assertLess(escaped.fine.maximum_relative_carter_drift, 1.0e-12)

    def test_descriptor_binds_frequency_height_thermal_and_visibility_semantics(self) -> None:
        mutations = (
            (
                "observerFrequencyFrame",
                lambda value: value.__setitem__(
                    "observerFrequencyFrame",
                    "coordinate-frame",
                ),
                "observer frequency frame",
            ),
            (
                "heightRateIsIndependentOfThermalRate",
                lambda value: value["finiteThicknessSurface"].__setitem__(
                    "heightRateIsIndependentOfThermalRate",
                    False,
                ),
                "independently supplied",
            ),
            (
                "radialReference",
                lambda value: value["diskThermalProxy"].__setitem__(
                    "radialReference",
                    "spherical-event-radius",
                ),
                "thermal radial reference",
            ),
            (
                "materialClearance",
                lambda value: value["observer"]["materialClearance"].__setitem__(
                    "upperFaceSignedValue",
                    value["observer"]["materialClearance"][
                        "upperFaceSignedValue"
                    ]
                    + 0.1,
                ),
                "clearance disagrees",
            ),
            (
                "visibilityConstraints",
                lambda value: value["termination"][
                    "visibilityConstraints"
                ].__setitem__(
                    "escapeStrictlyOutsideMaximumPhotosphereOblateRadius",
                    False,
                ),
                "visibility contract",
            ),
        )
        for label, mutate, message in mutations:
            forged = copy.deepcopy(self.descriptor)
            mutate(forged)
            with self.subTest(label=label), self.assertRaisesRegex(
                KerrFiniteThicknessSelectedOracleError,
                message,
            ):
                configuration_from_sampler_descriptor(forged)

    def test_upper_and_lower_hits_are_off_equatorial_actual_face_events(self) -> None:
        for name in ("upper", "lower"):
            result = self.refinements[name].fine
            with self.subTest(face=name):
                self.assertEqual(result.face, name)
                self.assertIsNotNone(result.pseudo_cylindrical_radius_over_mass)
                rho = result.pseudo_cylindrical_radius_over_mass
                radius = result.terminal_radius_m / self.configuration.mass_m
                absolute_height = math.sqrt(max(0.0, radius * radius - rho * rho))
                self.assertGreater(absolute_height, 0.0)
                self.assertAlmostEqual(
                    absolute_height,
                    self.configuration.photosphere_height_over_mass(rho),
                    places=11,
                )
                self.assertGreater(result.frequency_shift_g, 0.0)
                self.assertGreater(result.signed_emission_angle_cosine, 0.0)
                self.assertLessEqual(result.signed_emission_angle_cosine, 1.0)

    def test_lower_first_visible_ray_preserves_transparent_crossing_topology(self) -> None:
        self.assertEqual(
            self.refinements["upper"].fine.transparent_surface_crossings,
            0,
        )
        self.assertEqual(
            self.refinements["lower"].coarse.transparent_surface_crossings,
            2,
        )
        self.assertEqual(
            self.refinements["lower"].fine.transparent_surface_crossings,
            2,
        )
        production_topology = json.loads(
            self.production["lower"].topology_signature
        )
        self.assertEqual(
            [entry["classification"] for entry in production_topology["crossings"]],
            [
                "inside-isco-transparent",
                "inside-isco-transparent",
                "opaque-lower-photosphere",
            ],
        )
        self.assertEqual(
            production_topology["crossings"][-1]["surfaceId"],
            "kerr-finite-thickness-lower-photosphere",
        )

    def test_independent_g_and_nt_d20_spectra_match_production_sampler(self) -> None:
        thresholds = {"upper": 1.0e-10, "lower": 5.0e-8}
        for name, maximum_relative in thresholds.items():
            refinement = self.refinements[name]
            production = self.production[name]
            independent = selected_ray_refined_observed_intensities_nu(
                self.configuration,
                refinement,
                FREQUENCIES_HZ,
            )
            with self.subTest(face=name):
                self.assertEqual(production.visible_source, "disk")
                self.assertIsNotNone(production.frequency_shift_g)
                self.assertLess(
                    abs(
                        refinement.fine.frequency_shift_g
                        - production.frequency_shift_g
                    )
                    / max(
                        abs(refinement.fine.frequency_shift_g),
                        abs(production.frequency_shift_g),
                    ),
                    maximum_relative,
                )
                self.assertLess(
                    independent.maximum_relative_difference,
                    2.0e-8,
                )
                for oracle_value, production_value in zip(
                    independent.fine_intensities_nu,
                    production.specific_intensities_nu,
                ):
                    self.assertGreater(oracle_value, 0.0)
                    self.assertLess(
                        abs(oracle_value - production_value)
                        / max(abs(oracle_value), abs(production_value)),
                        maximum_relative,
                    )

        upper_topology = json.loads(self.production["upper"].topology_signature)
        self.assertEqual(
            upper_topology["crossings"][0]["classification"],
            "opaque-upper-photosphere",
        )

    def test_capture_escape_and_d20_angular_contracts_are_distinct(self) -> None:
        captured = self.refinements["captured"].fine
        escaped = self.refinements["escaped"].fine
        self.assertEqual(self.production["captured"].visible_source, "captured-boundary")
        self.assertEqual(self.production["escaped"].visible_source, "escaped-boundary")
        self.assertEqual(
            selected_ray_observed_intensities_nu(
                self.configuration,
                captured,
                FREQUENCIES_HZ,
            ),
            (0.0, 0.0),
        )
        with self.assertRaisesRegex(
            KerrFiniteThicknessSelectedOracleError,
            "separately configured observer spectrum",
        ):
            selected_ray_observed_intensities_nu(
                self.configuration,
                escaped,
                FREQUENCIES_HZ,
            )

        upper = self.refinements["upper"].fine
        low_mu = replace(upper, signed_emission_angle_cosine=0.2)
        high_mu = replace(upper, signed_emission_angle_cosine=0.8)
        low_value = selected_ray_observed_intensities_nu(
            self.configuration,
            low_mu,
            (FREQUENCIES_HZ[0],),
        )[0]
        high_value = selected_ray_observed_intensities_nu(
            self.configuration,
            high_mu,
            (FREQUENCIES_HZ[0],),
        )[0]
        self.assertAlmostEqual(
            high_value / low_value,
            (0.5 + 0.75 * 0.8) / (0.5 + 0.75 * 0.2),
            places=13,
        )

    def test_public_inputs_reject_bool_subclasses_and_foreign_results(self) -> None:
        with self.assertRaisesRegex(ValueError, "screen_x"):
            trace_selected_ray(
                self.configuration,
                True,
                0.0,
                self.options,
            )
        with self.assertRaisesRegex(ValueError, "frequency 0"):
            selected_ray_observed_intensities_nu(
                self.configuration,
                self.refinements["upper"].fine,
                (True,),
            )
        with self.assertRaisesRegex(TypeError, "exact selected-ray result"):
            selected_ray_observed_intensities_nu(
                self.configuration,
                object(),  # type: ignore[arg-type]
                FREQUENCIES_HZ,
            )


if __name__ == "__main__":
    unittest.main()
