from __future__ import annotations

import _ctypes
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import offline.kerr_native_cpu_backend as backend_module
import offline.kerr_returning_radiation_native_cpu as wrapper_module
from offline.kerr_native_cpu_backend import (
    ABI_VERSION,
    DEFAULT_AUDIT_TOOL_PATH,
    DEFAULT_HEADER_PATH,
    DEFAULT_MAKEFILE_PATH,
    DEFAULT_SOURCE_PATH,
    KerrNativeCpuAuthenticationError,
    KerrNativeCpuCapacityError,
    StrictKerrCpuBackend,
)
from offline.kerr_returning_radiation_native_cpu import (
    evaluate_forward_direction_native_cpu,
    trace_forward_direction_native_cpu,
)
from tools.native.golden_cache import load_golden_corpus
from tools.native.phase_space_golden import (
    FROZEN_REFERENCE_SHA256,
    PhaseSpaceGoldenError,
    _coordinate_for_ordinal,
    _reference_definition,
    build_nested16_numeric_replay_context,
    load_reference_document,
)


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "native/cpu/build/libblackhole_cpu.dylib"
PHASE_GOLDEN = ROOT / "tools/native/nested16_phase_space_golden.json"
NARROW_GOLDEN = ROOT / "tools/native/nested16_golden_corpus.json"


@unittest.skipUnless(LIBRARY.is_file(), "native CPU ABI-v3 library is not built")
class KerrNativeCpuAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase = load_reference_document(
            PHASE_GOLDEN.absolute(),
            expected_sha256=FROZEN_REFERENCE_SHA256,
        )
        cls.narrow = load_golden_corpus(NARROW_GOLDEN.absolute())
        cls.numeric_replay = build_nested16_numeric_replay_context()
        cls.definition = cls.numeric_replay.definition
        cls.backend = StrictKerrCpuBackend(LIBRARY.absolute())

    def test_frozen_reference_remains_fail_closed_while_numeric_replay_is_honest(
        self,
    ) -> None:
        with (
            patch(
                "offline.kerr_returning_radiation_kernel_cached."
                "run_kernel_direction_cache",
                side_effect=AssertionError("cache runner is forbidden"),
            ),
            self.assertRaisesRegex(
                PhaseSpaceGoldenError,
                "does not reproduce the frozen cache job key",
            ),
        ):
            _reference_definition()
        descriptor = self.numeric_replay.descriptor()
        self.assertEqual(
            descriptor["classification"],
            "non-production-nested16-numeric-replay-context",
        )
        self.assertFalse(descriptor["productionQualified"])
        self.assertFalse(descriptor["frozenCacheJobIdentityClaimed"])
        self.assertFalse(descriptor["frozenScientificJobIdentityClaimed"])
        self.assertFalse(descriptor["temporaryJobKeysReported"])

    def test_descriptor_binds_runtime_sources_build_audit_and_exact_layouts(
        self,
    ) -> None:
        descriptor = self.backend.model_descriptor()
        self.assertEqual(descriptor["abiVersion"], ABI_VERSION)
        self.assertEqual(
            descriptor["structureSizes"],
            {
                "KerrRayOptions": 112,
                "KerrRayResult": 216,
                "KerrReturningModel": 80,
                "KerrSurfaceOptions": 72,
                "RayPathSegment": 208,
                "RecordedSurfaceCrossing": 128,
                "RuntimeContract": 44,
            },
        )
        expected = {
            "auditTool": DEFAULT_AUDIT_TOOL_PATH,
            "backendPython": Path(backend_module.__file__).absolute(),
            "ctypesRuntime": Path(_ctypes.__file__).resolve(strict=True),
            "header": DEFAULT_HEADER_PATH,
            "makefile": DEFAULT_MAKEFILE_PATH,
            "source": DEFAULT_SOURCE_PATH,
            "library": LIBRARY.absolute(),
            "mathRuntime": Path(math.__file__).resolve(strict=True),
            "pythonRuntime": Path(sys.executable).resolve(strict=True),
            "wrapperPython": Path(wrapper_module.__file__).absolute(),
        }
        for name, path in expected.items():
            artifact = descriptor["artifacts"][name]
            self.assertEqual(artifact["byteLength"], path.stat().st_size)
            self.assertEqual(
                artifact["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )
        self.assertIn("binary audit passed", descriptor["binaryAudit"]["report"])
        self.assertEqual(
            descriptor["binaryAudit"]["reportSha256"],
            hashlib.sha256(
                descriptor["binaryAudit"]["report"].encode("utf-8")
            ).hexdigest(),
        )
        self.backend.revalidate()

    def test_library_selection_is_explicit_and_symlinks_fail_closed(self) -> None:
        with self.assertRaisesRegex(TypeError, "exact absolute"):
            StrictKerrCpuBackend(Path("native/cpu/build/libblackhole_cpu.dylib"))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            link = Path(directory) / "linked.dylib"
            link.symlink_to(LIBRARY.absolute())
            with self.assertRaisesRegex(
                KerrNativeCpuAuthenticationError,
                "regular non-symlink",
            ):
                StrictKerrCpuBackend(link.absolute())

    def test_all_eight_primitives_and_cached_narrow_transports_are_exact(
        self,
    ) -> None:
        narrow_by_ordinal = {
            item["coordinate"]["ordinal"]: item["transport"]
            for item in self.narrow["records"]
        }
        self.assertEqual(
            self.phase["selectedOrdinals"],
            [369, 433, 726, 732, 736, 31263, 37331, 75785],
        )
        for record in self.phase["records"]:
            ordinal = record["coordinate"]["ordinal"]
            with self.subTest(ordinal=ordinal):
                coordinate = _coordinate_for_ordinal(
                    self.definition.plan,
                    ordinal,
                )
                transport = evaluate_forward_direction_native_cpu(
                    self.definition.scientific_context,
                    coordinate,
                    backend=self.backend,
                )
                self.assertEqual(
                    transport["transport"]["primitiveDescriptorSha256"],
                    record["referenceEvidence"]["primitiveDescriptorSha256"],
                )
                if ordinal in narrow_by_ordinal:
                    self.assertEqual(transport, narrow_by_ordinal[ordinal])

    def test_capacity_failure_never_calls_the_python_ray_fallback(self) -> None:
        limited = StrictKerrCpuBackend(
            LIBRARY.absolute(),
            segment_capacity=0,
            crossing_capacity=0,
        )
        coordinate = _coordinate_for_ordinal(self.definition.plan, 736)
        with patch(
            "offline.kerr_returning_radiation_rays."
            "trace_kerr_returning_radiation_direction",
            side_effect=AssertionError("Python fallback is forbidden"),
        ):
            with self.assertRaisesRegex(
                KerrNativeCpuCapacityError,
                "caller-owned",
            ):
                evaluate_forward_direction_native_cpu(
                    self.definition.scientific_context,
                    coordinate,
                    backend=limited,
                )

    def test_out_of_annulus_emitter_fails_before_native_or_python_fallback(
        self,
    ) -> None:
        identity = self.definition.scientific_context.identity
        outside = identity.surface.calibration.isco_radius_over_mass - 1.0e-6
        with (
            patch.object(
                self.backend,
                "trace_one_resolution",
                side_effect=AssertionError("native trace must not see bad launch"),
            ) as native_trace,
            patch(
                "offline.kerr_returning_radiation_rays."
                "trace_kerr_returning_radiation_direction",
                side_effect=AssertionError("Python fallback is forbidden"),
            ),
            self.assertRaisesRegex(ValueError, "ISCO seam"),
        ):
            trace_forward_direction_native_cpu(
                identity.surface,
                identity.termination,
                identity.fine_ray_options,
                identity.fine_surface_options,
                identity.coarse_ray_options,
                identity.coarse_surface_options,
                "upper",
                outside,
                0.5,
                0.0,
                backend=self.backend,
            )
        native_trace.assert_not_called()


if __name__ == "__main__":
    unittest.main()
