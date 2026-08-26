from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import scripts.render_offline_kerr_nt_frame as renderer
from scripts.verify_offline_kerr_nt_selected_rays import verify_selected_rays


class SelectedRayProductVerifierTests(unittest.TestCase):
    def test_real_product_selected_ray_has_honest_second_evidence_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "product"
            arguments = renderer.parse_args(
                [
                    str(output),
                    "--cache",
                    str(root / "cache"),
                    "--spin",
                    "0.7",
                    "--black-hole-mass-solar",
                    "1e8",
                    "--accretion-rate-kg-s",
                    "1e22",
                    "--inclination-deg",
                    "63.0253574644",
                    "--frequency-hz",
                    "5e14",
                    "--width",
                    "1",
                    "--height",
                    "1",
                    "--tile-width",
                    "1",
                    "--tile-height",
                    "1",
                    "--screen-x-min",
                    "0.49999",
                    "--screen-x-max",
                    "0.50001",
                    "--screen-y-min",
                    "-0.50001",
                    "--screen-y-max",
                    "-0.49999",
                    "--minimum-depth",
                    "0",
                    "--maximum-depth",
                    "0",
                    "--maximum-ray-evaluations",
                    "64",
                    "--radiance-guard-ceiling",
                    "100",
                    "--ray-absolute-tolerance",
                    "5e-10",
                    "--ray-relative-tolerance",
                    "5e-10",
                    "--ray-maximum-step",
                    "0.25",
                    "--ray-maximum-affine-length",
                    "100",
                    "--surface-absolute-tolerance",
                    "5e-10",
                    "--surface-relative-tolerance",
                    "5e-10",
                    "--surface-null-residual-limit",
                    "2e-7",
                    "--frequency-null-residual-limit",
                    "2e-7",
                ]
            )
            renderer.execute_render_plan(renderer.build_render_plan(arguments))
            report = verify_selected_rays(
                output / "manifest.json",
                screen_points=((0.5, -0.5),),
                step_m=0.01,
                maximum_affine_length_m=100.0,
            )

        self.assertTrue(report["structuralContractVerified"])
        self.assertTrue(report["selectedRayCalibrationVerified"])
        self.assertTrue(report["independentGeodesicIntegrator"])
        self.assertTrue(report["independentEventLocator"])
        self.assertFalse(report["productionTracerCalledByOracle"])
        self.assertTrue(report["productionSamplerUsedOnlyAsComparator"])
        self.assertEqual(
            report["spectralIndependence"],
            "shared-page-thorne-radial-scalar",
        )
        self.assertFalse(report["fullFramePhysicsProof"])
        self.assertFalse(report["isNumericalRelativitySolver"])
        self.assertFalse(report["isGeneralRelativisticMagnetohydrodynamics"])
        self.assertEqual(report["rayCount"], 1)
        self.assertEqual(report["rays"][0]["outcome"], "disk")
        self.assertFalse(report["rays"][0]["diskRadiusProductionPersisted"])
        self.assertLess(
            report["maximumRelativeGIndependentVsProduction"],
            1.0e-10,
        )
        self.assertLess(
            report["maximumRelativeIntensityIndependentVsProduction"],
            1.0e-9,
        )


if __name__ == "__main__":
    unittest.main()
