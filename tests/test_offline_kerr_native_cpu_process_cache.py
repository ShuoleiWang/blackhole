from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import offline.kerr_native_cpu_backend as backend_module
from offline.geodesic import RayTraceOptions, SurfaceEventOptions
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_finite_thickness import (
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_surface import (
    KerrFiniteThicknessMultiSurface,
)
from offline.kerr_native_cpu_backend import (
    DEFAULT_HEADER_PATH,
    FrozenKerrNativeCpuRuntimeBinding,
    KerrNativeCpuAuthenticationError,
    KerrNativeCpuRuntimeBinding,
    StrictKerrCpuBackend,
    consume_kerr_native_cpu_runtime_binding,
)
from offline.kerr_returning_radiation_native_cpu import (
    consume_forward_direction_native_cpu_runtime_binding,
)
from tools.native.golden_cache import load_golden_corpus
from tools.native.phase_space_golden import (
    FROZEN_REFERENCE_SHA256,
    load_reference_document,
)


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "native/cpu/build/libblackhole_cpu.dylib"
AUDIT_TOOL = ROOT / "native/cpu/tools/audit_binary.py"
NARROW_GOLDEN = ROOT / "tools/native/nested16_golden_corpus.json"
PHASE_GOLDEN = ROOT / "tools/native/nested16_phase_space_golden.json"


@unittest.skipUnless(LIBRARY.is_file(), "native CPU ABI-v3 library is not built")
class KerrNativeCpuProcessCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.narrow = load_golden_corpus(NARROW_GOLDEN.absolute())
        cls.phase = load_reference_document(
            PHASE_GOLDEN.absolute(),
            expected_sha256=FROZEN_REFERENCE_SHA256,
        )
        cls.record = next(
            item
            for item in cls.phase["records"]
            if item["coordinate"]["ordinal"] == 736
        )
        identity = cls.phase["reference"]["scientificDocument"][
            "scientificIdentity"
        ]
        cls.metric = KerrKerrSchildMetric(**identity["metric"])
        cls.calibration = StationaryKerrFiniteThicknessCalibration(
            **identity["surface"]["calibration"]
        )
        cls.surface = KerrFiniteThicknessMultiSurface(
            cls.metric,
            cls.calibration,
        )
        cls.termination = KerrOblateTermination(**identity["termination"])
        policy = cls.record["referenceEvidence"]["primitiveDescriptor"][
            "numericalPolicy"
        ]
        cls.fine_ray_options = RayTraceOptions(**policy["fineRayOptions"])
        cls.fine_surface_options = SurfaceEventOptions(
            **policy["fineSurfaceOptions"]
        )
        cls.coarse_ray_options = RayTraceOptions(**policy["coarseRayOptions"])
        cls.coarse_surface_options = SurfaceEventOptions(
            **policy["coarseSurfaceOptions"]
        )

    def setUp(self) -> None:
        backend_module._reset_process_local_backend_cache_for_tests()

    def tearDown(self) -> None:
        backend_module._reset_process_local_backend_cache_for_tests()

    def binding(self, library: Path = LIBRARY) -> FrozenKerrNativeCpuRuntimeBinding:
        probe = StrictKerrCpuBackend(library.absolute())
        return probe.runtime_binding()

    def test_process_local_backend_is_lazy_reused_and_path_free(self) -> None:
        binding = self.binding()
        self.assertIsInstance(binding, KerrNativeCpuRuntimeBinding)
        descriptor_json = json.dumps(
            binding.path_free_descriptor(),
            allow_nan=False,
            sort_keys=True,
        )
        self.assertNotIn(str(LIBRARY.absolute()), descriptor_json)
        with consume_kerr_native_cpu_runtime_binding(binding) as first:
            first.revalidate()
        with consume_kerr_native_cpu_runtime_binding(binding) as second:
            second.revalidate()
        self.assertIs(first, second)
        self.assertEqual(
            first.model_descriptor_sha256,
            binding.backend_descriptor_sha256,
        )

    def test_same_library_bytes_at_different_paths_share_path_free_identity(
        self,
    ) -> None:
        original = StrictKerrCpuBackend(LIBRARY.absolute())
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            renamed = Path(directory) / "renamed-whole-ray-backend.dylib"
            shutil.copyfile(LIBRARY, renamed)
            relocated = StrictKerrCpuBackend(renamed.absolute())
            self.assertEqual(
                original.model_descriptor_sha256,
                relocated.model_descriptor_sha256,
            )
            original_binding = original.runtime_binding()
            relocated_binding = relocated.runtime_binding()
            self.assertEqual(
                original_binding.path_free_descriptor(),
                relocated_binding.path_free_descriptor(),
            )
            with consume_kerr_native_cpu_runtime_binding(
                original_binding
            ) as first:
                pass
            with consume_kerr_native_cpu_runtime_binding(
                relocated_binding
            ) as second:
                pass
            self.assertIsNot(first, second)
            self.assertEqual(
                first.model_descriptor_sha256,
                second.model_descriptor_sha256,
            )

    def test_jobs_runtime_binding_protocol_is_consumed_without_adapter_shim(
        self,
    ) -> None:
        from offline.kerr_returning_radiation_kernel_jobs import (
            make_kernel_direction_evaluator_runtime_binding,
        )

        probe = StrictKerrCpuBackend(LIBRARY.absolute())
        binding = make_kernel_direction_evaluator_runtime_binding(
            evaluator_implementation_id="tests.native-cpu-process-cache/v1",
            library_path=LIBRARY.absolute(),
            backend_descriptor=probe.model_descriptor(),
            segment_capacity=100_000,
            crossing_capacity=100_000,
        )
        self.assertIsInstance(binding, KerrNativeCpuRuntimeBinding)
        with consume_kerr_native_cpu_runtime_binding(binding) as consumed:
            self.assertEqual(
                consumed.model_descriptor_sha256,
                probe.model_descriptor_sha256,
            )

    def test_pid_change_poisoned_inherited_loaded_image(self) -> None:
        binding = self.binding()
        with consume_kerr_native_cpu_runtime_binding(binding) as first:
            first.revalidate()
        process_id = os.getpid()
        with patch.object(backend_module.os, "getpid", return_value=process_id + 1):
            with self.assertRaisesRegex(
                KerrNativeCpuAuthenticationError,
                "poisoned",
            ):
                with consume_kerr_native_cpu_runtime_binding(binding):
                    self.fail("fork-inherited image was reused")
        with self.assertRaisesRegex(
            KerrNativeCpuAuthenticationError,
            "poisoned",
        ):
            with consume_kerr_native_cpu_runtime_binding(binding):
                self.fail("PID-flipped image path was reused")

    def test_descriptor_and_explicit_path_mismatches_fail_closed(self) -> None:
        binding = self.binding()
        replacement = "0" * 64
        if replacement == binding.backend_descriptor_sha256:
            replacement = "1" * 64
        wrong_descriptor = FrozenKerrNativeCpuRuntimeBinding(
            binding.library_path,
            binding.library_sha256,
            replacement,
            binding.segment_capacity,
            binding.crossing_capacity,
        )
        with self.assertRaisesRegex(
            KerrNativeCpuAuthenticationError,
            "differs from the bound backend descriptor",
        ):
            with consume_kerr_native_cpu_runtime_binding(wrong_descriptor):
                self.fail("descriptor mismatch unexpectedly yielded a backend")

        wrong_path = FrozenKerrNativeCpuRuntimeBinding(
            DEFAULT_HEADER_PATH,
            binding.library_sha256,
            binding.backend_descriptor_sha256,
            binding.segment_capacity,
            binding.crossing_capacity,
        )
        with self.assertRaisesRegex(
            KerrNativeCpuAuthenticationError,
            "binary failed|ABI-v3 surface",
        ):
            with consume_kerr_native_cpu_runtime_binding(wrong_path):
                self.fail("non-library path unexpectedly yielded a backend")

    def test_live_backend_cannot_be_pickled_or_copied(self) -> None:
        backend = StrictKerrCpuBackend(LIBRARY.absolute())
        with self.assertRaisesRegex(TypeError, "cannot be pickled"):
            pickle.dumps(backend)
        with self.assertRaisesRegex(TypeError, "cannot be copied"):
            copy.copy(backend)
        with self.assertRaisesRegex(TypeError, "cannot be copied"):
            copy.deepcopy(backend)
        binding_payload = pickle.dumps(backend.runtime_binding())
        restored = pickle.loads(binding_payload)
        self.assertIsInstance(restored, FrozenKerrNativeCpuRuntimeBinding)
        self.assertNotIsInstance(restored, StrictKerrCpuBackend)

    def test_artifact_drift_is_detected_after_bound_use(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            copied_library = Path(directory) / LIBRARY.name
            shutil.copyfile(LIBRARY, copied_library)
            binding = self.binding(copied_library)
            with self.assertRaisesRegex(
                KerrNativeCpuAuthenticationError,
                "changed after backend construction",
            ):
                with consume_kerr_native_cpu_runtime_binding(binding):
                    snapshot = copied_library.stat()
                    os.utime(
                        copied_library,
                        ns=(snapshot.st_atime_ns, snapshot.st_mtime_ns + 1),
                    )
            with self.assertRaisesRegex(
                KerrNativeCpuAuthenticationError,
                "poisoned",
            ):
                StrictKerrCpuBackend(copied_library.absolute())

    def test_runtime_consumer_preserves_current_narrow_adapter_exactness(self) -> None:
        binding = self.binding()
        sample = self.record["sample"]
        transport = consume_forward_direction_native_cpu_runtime_binding(
            binding,
            self.surface,
            self.termination,
            self.fine_ray_options,
            self.fine_surface_options,
            self.coarse_ray_options,
            self.coarse_surface_options,
            sample["sourceFace"],
            sample["sourceRadiusOverMass"],
            sample["emissionAngleCosine"],
            sample["tangentAzimuthRad"],
        )
        expected = next(
            item["transport"]["transport"]
            for item in self.narrow["records"]
            if item["coordinate"]["ordinal"] == 736
        )
        self.assertEqual(transport.fate, expected["fate"])
        self.assertEqual(transport.receiver_face, expected["receiverFace"])
        self.assertEqual(
            transport.receiver_radius_over_mass,
            expected["receiverRadiusOverMass"],
        )
        self.assertEqual(transport.frequency_ratio, expected["frequencyRatio"])
        self.assertEqual(transport.g2, expected["g2"])
        self.assertEqual(
            transport.primitive_descriptor_sha256,
            expected["primitiveDescriptorSha256"],
        )
        self.assertEqual(
            transport.coarse_receiver_face,
            expected["coarseReceiverFace"],
        )
        self.assertEqual(
            transport.coarse_receiver_radius_over_mass,
            expected["coarseReceiverRadiusOverMass"],
        )

    def test_runtime_consumer_module_does_not_eagerly_import_cached_helpers(
        self,
    ) -> None:
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        script = (
            "import sys; "
            "import offline.kerr_returning_radiation_native_cpu; "
            "assert 'offline.kerr_returning_radiation_kernel_cached' "
            "not in sys.modules"
        )
        completed = subprocess.run(
            (sys.executable, "-c", script),
            check=False,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_binary_audit_uses_bound_absolute_tools_with_hostile_path(self) -> None:
        environment = dict(os.environ)
        environment["PATH"] = "/definitely/not-an-audit-tool-directory"
        completed = subprocess.run(
            (sys.executable, str(AUDIT_TOOL), str(LIBRARY)),
            check=False,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        prefix = "binary audit passed: "
        self.assertTrue(completed.stdout.startswith(prefix), completed.stdout)
        report = json.loads(completed.stdout[len(prefix) :])
        self.assertEqual(set(report["tools"]), {"file", "nm", "otool"})
        for name, artifact in report["tools"].items():
            self.assertEqual(artifact["artifactName"], name)
            self.assertEqual(len(artifact["sha256"]), 64)
            self.assertGreater(artifact["byteLength"], 0)
        self.assertNotIn("/definitely/not-an-audit-tool-directory", completed.stdout)
        backend_report = StrictKerrCpuBackend(LIBRARY.absolute()).model_descriptor()[
            "binaryAudit"
        ]["report"]
        self.assertTrue(backend_report.startswith(prefix))
        self.assertEqual(
            json.loads(backend_report[len(prefix) :])["tools"],
            report["tools"],
        )

    def test_backend_audit_child_ignores_hostile_python_and_dyld_environment(
        self,
    ) -> None:
        hostile = {
            "DYLD_INSERT_LIBRARIES": "/definitely/not/a/library.dylib",
            "DYLD_LIBRARY_PATH": "/definitely/not/a/library/path",
            "PYTHONHOME": "/definitely/not/a/python/home",
            "PYTHONPATH": "/definitely/not/a/python/path",
            "PYTHONUSERBASE": "/definitely/not/a/user/site",
        }
        with patch.dict(os.environ, hostile, clear=False):
            descriptor = StrictKerrCpuBackend(
                LIBRARY.absolute()
            ).model_descriptor()
        report = descriptor["binaryAudit"]["report"]
        self.assertTrue(report.startswith("binary audit passed: "))
        for value in hostile.values():
            self.assertNotIn(value, report)

    def test_backend_rejects_noncanonical_forged_success_audit_report(self) -> None:
        forged = subprocess.CompletedProcess(
            args=(),
            returncode=0,
            stdout="binary audit passed: {}\n",
        )
        with patch.object(backend_module.subprocess, "run", return_value=forged):
            with self.assertRaisesRegex(
                KerrNativeCpuAuthenticationError,
                "unsupported schema",
            ):
                StrictKerrCpuBackend(LIBRARY.absolute())


if __name__ == "__main__":
    unittest.main()
