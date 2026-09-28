from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.native.golden_cache import load_golden_corpus
import tools.native.replay_cpu_adapter_cache as replay


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = (ROOT / "native/cpu/build/libblackhole_cpu.dylib").absolute()
GOLDEN = (ROOT / "tools/native/nested16_golden_corpus.json").absolute()


class NativeCpuCopiedCacheReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        corpus = load_golden_corpus(GOLDEN)
        record = next(
            item for item in corpus["records"] if item["coordinate"]["ordinal"] == 736
        )
        cls.selection = {
            **corpus,
            "records": [record],
            "selectedOrdinals": [736],
        }

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @unittest.skipUnless(LIBRARY.is_file(), "native CPU ABI-v3 library is not built")
    def test_one_direction_worker_smoke_is_byte_exact(self) -> None:
        definition = replay._native_definition(LIBRARY)
        replay._worker_initialize(
            definition.scientific_context,
            definition.plan,
        )
        result = replay._worker_replay_chunk(
            (self.selection["records"][0],)
        )
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["firstOrdinal"], 736)
        self.assertEqual(result["lastOrdinal"], 736)
        self.assertEqual(sum(result["fateDistribution"].values()), 1)

    def test_production_replay_has_no_evidence_injection_hook(self) -> None:
        self.assertEqual(tuple(inspect.signature(replay.run_replay).parameters), ("arguments",))
        with patch.object(replay._golden_cache, "extract_golden_corpus", lambda: {}):
            with self.assertRaisesRegex(
                replay.NativeCpuCacheReplayError,
                "helper callable",
            ):
                replay._helper_source_artifacts()
        with patch.object(replay, "_EXTRACT_GOLDEN_CORPUS_ENTRY", lambda: {}):
            with self.assertRaisesRegex(
                replay.NativeCpuCacheReplayError,
                "invoked replay helper binding changed",
            ):
                replay._helper_source_artifacts()

    def test_selection_and_output_boundaries_fail_closed(self) -> None:
        for value in ("-1:1", "1:1", "0:38145", "malformed"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    replay._selection(value)
        for value in ("0", "nan", "inf", "86401"):
            with self.subTest(deadline=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    replay._deadline_seconds(value)
        oversized = self.root / "oversized-artifact"
        oversized.touch()
        os.truncate(oversized, replay.MAXIMUM_REPLAY_ARTIFACT_BYTES + 1)
        with self.assertRaisesRegex(
            replay.NativeCpuCacheReplayError,
            "exceeds",
        ):
            replay._stable_artifact(oversized, "oversized test artifact")
        job = (self.root / "job").absolute()
        job.mkdir()
        with self.assertRaisesRegex(
            replay.NativeCpuCacheReplayError,
            "outside the cache job",
        ):
            replay._safe_write_new(job / "forbidden.json", b"{}\n", job)
        self.assertEqual(tuple(job.iterdir()), ())

        nested = job / "nested"
        nested.mkdir()
        linked_parent = self.root / "lexically-outside"
        linked_parent.symlink_to(job, target_is_directory=True)
        with self.assertRaisesRegex(
            replay.NativeCpuCacheReplayError,
            "symlink or unreadable ancestor",
        ):
            replay._safe_write_new(
                linked_parent / "nested" / "bypass.json",
                b"{}\n",
                job,
            )
        self.assertFalse((nested / "bypass.json").exists())

        case_job = self.root / "CaseIdentityProbe"
        case_job.mkdir()
        case_alias = self.root / "caseidentityprobe"
        with self.assertRaises(replay.NativeCpuCacheReplayError):
            replay._safe_write_new(
                case_alias / "case-bypass.json",
                b"{}\n",
                case_job,
            )
        self.assertFalse((case_job / "case-bypass.json").exists())


if __name__ == "__main__":
    unittest.main()
