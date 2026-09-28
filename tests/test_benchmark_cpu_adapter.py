from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.native.benchmark_cpu_adapter import (
    CpuAdapterBenchmarkError,
    DEFAULT_NATIVE_TASK_START,
    SCHEMA,
    TASK_DIRECTION_COUNT,
    _parser,
    _selected_ordinals,
    main,
)
from tools.native.golden_cache import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
ISOLATED_ROOT = ROOT.parent
LIBRARY = ROOT / "native/cpu/build/libblackhole_cpu.dylib"
JOB_DIRECTORY = (
    ISOLATED_ROOT
    / "golden-cache"
    / "d8286face863a0755d70c20c1638c6a5dc4d9d9037c0227626fb805bb4d0f751"
)


class CpuAdapterBenchmarkTests(unittest.TestCase):
    def test_selection_defaults_and_python_budget_are_fail_closed(self) -> None:
        native = _parser().parse_args(
            [
                str(JOB_DIRECTORY.absolute()),
                "--dylib",
                str(LIBRARY.absolute()),
                "--output",
                str((ROOT / "unused-native-report.json").absolute()),
            ]
        )
        self.assertEqual(
            _selected_ordinals(native),
            tuple(
                range(
                    DEFAULT_NATIVE_TASK_START,
                    DEFAULT_NATIVE_TASK_START + TASK_DIRECTION_COUNT,
                )
            ),
        )
        python = _parser().parse_args(
            [
                str(JOB_DIRECTORY.absolute()),
                "--dylib",
                str(LIBRARY.absolute()),
                "--backend",
                "python",
                "--ordinal-range",
                "736:737",
                "--output",
                str((ROOT / "unused-python-report.json").absolute()),
            ]
        )
        with self.assertRaisesRegex(
            CpuAdapterBenchmarkError,
            "python-direction-limit",
        ):
            _selected_ordinals(python)

    @unittest.skipUnless(
        LIBRARY.is_file() and JOB_DIRECTORY.is_dir(),
        "isolated ABI-v3 library/cache evidence is unavailable",
    )
    def test_one_native_direction_is_authenticated_exact_and_cache_runner_free(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory) / "report.json"
            stdout = io.StringIO()
            arguments = [
                str(JOB_DIRECTORY.absolute()),
                "--dylib",
                str(LIBRARY.absolute()),
                "--backend",
                "native",
                "--ordinal-range",
                "736:737",
                "--repeat",
                "1",
                "--output",
                str(output.absolute()),
            ]
            with (
                patch(
                    "offline.kerr_returning_radiation_kernel_cached."
                    "run_kernel_direction_cache",
                    side_effect=AssertionError("cache runner is forbidden"),
                ),
                redirect_stdout(stdout),
            ):
                self.assertEqual(main(arguments), 0)
            report_payload = output.read_bytes()
            report = json.loads(report_payload)
            self.assertEqual(report["schema"], SCHEMA)
            self.assertEqual(report["selection"]["ordinals"], [736])
            self.assertEqual(report["fateDistribution"], {"captured": 1})
            self.assertTrue(report["qualification"]["allRepeatsByteExact"])
            self.assertTrue(report["qualification"]["allRepeatsQualified"])
            self.assertTrue(
                report["authentication"]["cacheReadOnlyStableBeforeAfter"]
            )
            self.assertTrue(report["backend"]["nativeNeverFallsBack"])
            self.assertEqual(
                report["numericReplayContext"]["classification"],
                "non-production-nested16-numeric-replay-context",
            )
            self.assertFalse(
                report["numericReplayContext"]["productionQualified"]
            )
            self.assertFalse(
                report["numericReplayContext"][
                    "frozenScientificJobIdentityClaimed"
                ]
            )
            self.assertFalse(
                report["qualification"][
                    "numericReplayContextProductionQualified"
                ]
            )
            self.assertFalse(
                report["qualification"]["temporaryScientificOrJobKeyReported"]
            )
            self.assertIn(
                "not the temporary numeric-replay context identity",
                report["authentication"]["scope"],
            )
            self.assertFalse(report["qualification"]["cacheRunnerInvoked"])
            self.assertFalse(report["qualification"]["cacheWrites"])
            self.assertTrue(report["qualification"]["toolSourceStable"])
            counters = report["repeats"][0]["transportCounters"]
            self.assertGreater(counters["fineAcceptedSteps"], 0)
            self.assertGreater(counters["coarseAcceptedSteps"], 0)
            basis_sha = report.pop("reportBasisSha256")
            self.assertEqual(
                basis_sha,
                hashlib.sha256(canonical_json_bytes(report)).hexdigest(),
            )
            terminal = json.loads(stdout.getvalue())
            self.assertTrue(terminal["qualified"])
            self.assertEqual(
                terminal["outputSha256"],
                hashlib.sha256(report_payload).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
