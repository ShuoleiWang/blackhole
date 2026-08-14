from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import offline.authenticated_artifact as artifact_module
from offline.authenticated_artifact import (
    AuthenticatedArtifactChangedError,
    AuthenticatedArtifactError,
    AuthenticatedArtifactResourceError,
    authenticate_stable_artifact,
    resolve_stable_artifact_locator,
)


class AuthenticatedArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve(strict=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_current_regular_file_digest_matches_direct_sha256(self) -> None:
        path = self.root / "artifact.bin"
        payload = (b"authenticated-artifact\0" * 70000) + b"tail"
        path.write_bytes(payload)

        result = authenticate_stable_artifact(path, "test artifact")

        self.assertEqual(result.path, path)
        self.assertEqual(result.byte_length, len(payload))
        self.assertEqual(result.sha256, hashlib.sha256(payload).hexdigest())

    def test_rejects_final_and_ancestor_symlinks_without_reading(self) -> None:
        target_directory = self.root / "target"
        target_directory.mkdir()
        target = target_directory / "artifact.bin"
        target.write_bytes(b"sentinel")
        final_alias = self.root / "final-alias.bin"
        final_alias.symlink_to(target)
        ancestor_alias = self.root / "ancestor-alias"
        ancestor_alias.symlink_to(target_directory, target_is_directory=True)

        for path in (final_alias, ancestor_alias / target.name):
            with self.subTest(path=path), patch.object(
                artifact_module.os,
                "read",
                side_effect=AssertionError("symlink input must not be read"),
            ) as read, self.assertRaisesRegex(
                AuthenticatedArtifactError,
                "without following symbolic links",
            ):
                authenticate_stable_artifact(path, "symlink artifact")
            read.assert_not_called()
        self.assertEqual(target.read_bytes(), b"sentinel")

    def test_runtime_locator_binds_lexical_symlink_chain_and_target(self) -> None:
        target_directory = self.root / "runtime-target"
        target_directory.mkdir()
        target = target_directory / "runtime.bin"
        target.write_bytes(b"runtime")
        directory_alias = self.root / "runtime-directory-alias"
        directory_alias.symlink_to(target_directory, target_is_directory=True)
        locator = directory_alias / target.name

        first = resolve_stable_artifact_locator(locator, "runtime locator")
        second = resolve_stable_artifact_locator(locator, "runtime locator")

        self.assertEqual(first, second)
        self.assertEqual(first.locator_path, locator)
        self.assertEqual(first.resolved_path, target)
        self.assertEqual(len(first.chain_binding_sha256), 64)

    def test_rejects_fifo_without_blocking_or_reading(self) -> None:
        if not hasattr(os, "mkfifo"):
            self.skipTest("platform has no FIFO primitive")
        fifo = self.root / "artifact.fifo"
        os.mkfifo(fifo)
        with patch.object(
            artifact_module.os,
            "read",
            side_effect=AssertionError("FIFO must be rejected before reading"),
        ) as read, self.assertRaisesRegex(
            AuthenticatedArtifactError,
            "regular file",
        ):
            authenticate_stable_artifact(fifo, "FIFO artifact")
        read.assert_not_called()

    def test_per_file_and_total_caps_reject_before_first_read(self) -> None:
        path = self.root / "oversized.bin"
        path.write_bytes(b"123456")
        cases = (
            {
                "maximum_byte_length": 5,
                "preceding_total_byte_length": 0,
                "maximum_total_byte_length": 100,
            },
            {
                "maximum_byte_length": 100,
                "preceding_total_byte_length": 5,
                "maximum_total_byte_length": 10,
            },
        )
        for options in cases:
            with self.subTest(options=options), patch.object(
                artifact_module.os,
                "read",
                side_effect=AssertionError("byte cap must precede data reads"),
            ) as read, self.assertRaises(AuthenticatedArtifactResourceError):
                authenticate_stable_artifact(
                    path,
                    "oversized artifact",
                    **options,
                )
            read.assert_not_called()

    def test_growth_past_cap_is_bounded_and_fails_closed(self) -> None:
        path = self.root / "growing.bin"
        path.write_bytes(b"a" * 16)
        original_read = artifact_module.os.read
        mutated = False

        def grow_after_first_read(descriptor: int, count: int) -> bytes:
            nonlocal mutated
            block = original_read(descriptor, count)
            if not mutated:
                mutated = True
                with path.open("ab") as stream:
                    stream.write(b"b")
                    stream.flush()
                    os.fsync(stream.fileno())
            return block

        with patch.object(
            artifact_module.os,
            "read",
            side_effect=grow_after_first_read,
        ), self.assertRaises(AuthenticatedArtifactResourceError):
            authenticate_stable_artifact(
                path,
                "growing artifact",
                maximum_byte_length=16,
                maximum_total_byte_length=16,
            )

    def test_final_file_swap_is_detected_by_path_reopen(self) -> None:
        path = self.root / "artifact.bin"
        displaced = self.root / "displaced.bin"
        replacement = self.root / "replacement.bin"
        path.write_bytes(b"original")
        replacement.write_bytes(b"replacement")
        original_read = artifact_module.os.read
        mutated = False

        def swap_after_first_read(descriptor: int, count: int) -> bytes:
            nonlocal mutated
            block = original_read(descriptor, count)
            if not mutated:
                mutated = True
                path.rename(displaced)
                replacement.rename(path)
            return block

        with patch.object(
            artifact_module.os,
            "read",
            side_effect=swap_after_first_read,
        ), self.assertRaisesRegex(
            AuthenticatedArtifactChangedError,
            "changed while|path changed",
        ):
            authenticate_stable_artifact(path, "swapped artifact")

    def test_real_ancestor_directory_swap_is_detected(self) -> None:
        ancestor = self.root / "ancestor"
        ancestor.mkdir()
        path = ancestor / "artifact.bin"
        path.write_bytes(b"original")
        replacement = self.root / "replacement-directory"
        replacement.mkdir()
        (replacement / path.name).write_bytes(b"replacement")
        displaced = self.root / "displaced-directory"
        original_read = artifact_module.os.read
        mutated = False

        def swap_after_first_read(descriptor: int, count: int) -> bytes:
            nonlocal mutated
            block = original_read(descriptor, count)
            if not mutated:
                mutated = True
                ancestor.rename(displaced)
                replacement.rename(ancestor)
            return block

        with patch.object(
            artifact_module.os,
            "read",
            side_effect=swap_after_first_read,
        ), self.assertRaisesRegex(
            AuthenticatedArtifactChangedError,
            "path changed",
        ):
            authenticate_stable_artifact(path, "ancestor-swapped artifact")

    def test_missing_security_flag_and_relative_path_fail_before_open(self) -> None:
        path = self.root / "artifact.bin"
        path.write_bytes(b"payload")
        with patch.object(
            artifact_module.os,
            "O_NOFOLLOW",
            0,
        ), patch.object(
            artifact_module.os,
            "open",
            side_effect=AssertionError("missing flags must fail before path I/O"),
        ) as opened, self.assertRaisesRegex(
            AuthenticatedArtifactError,
            "O_NOFOLLOW",
        ):
            authenticate_stable_artifact(path, "flagless artifact")
        opened.assert_not_called()

        with patch.object(
            artifact_module.os,
            "open",
            side_effect=AssertionError("relative path must fail before path I/O"),
        ) as opened, self.assertRaises(TypeError):
            authenticate_stable_artifact(Path("relative.bin"), "relative artifact")
        opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
