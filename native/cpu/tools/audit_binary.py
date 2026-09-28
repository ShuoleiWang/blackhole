#!/usr/bin/env python3
"""Fail closed if the milestone-3 Mach-O violates its build contract."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
from pathlib import Path


EXPECTED_EXPORTS = {
    "_bh_cpu_bilinear4",
    "_bh_cpu_fsum",
    "_bh_cpu_get_runtime_contract",
    "_bh_cpu_hamiltonian_rhs",
    "_bh_cpu_hamiltonian_rhs_batch",
    "_bh_cpu_kerr_dopri54_probe_batch",
    "_bh_cpu_trace_kerr_returning_ray",
    "_bh_cpu_kerr_dopri54_step",
    "_bh_cpu_kerr_hamiltonian_rhs",
    "_bh_cpu_kerr_metric_sample",
    "_bh_cpu_matrix4_vector4",
    "_bh_cpu_metric_algebra_audit",
    "_bh_cpu_metric_sample_audit",
    "_bh_cpu_normalized_null_residual",
    "_bh_cpu_require_strict_fp",
    "_bh_cpu_status_string",
}
FORBIDDEN_FUSED_OPERATION = re.compile(
    r"\b(?:fmla|fmls|fmadd|fmsub|fnmadd|fnmsub|"
    r"vfmadd|vfmsub|vfnmadd|vfnmsub)[a-z0-9_.]*\b",
    re.IGNORECASE,
)
FIXED_TOOLS = {
    "file": Path("/usr/bin/file"),
    "nm": Path("/usr/bin/nm"),
    "otool": Path("/usr/bin/otool"),
}


def _tool_artifact(name: str, path: Path) -> dict[str, object]:
    descriptor = -1
    try:
        before_path = os.lstat(path)
        if (
            stat.S_ISLNK(before_path.st_mode)
            or not stat.S_ISREG(before_path.st_mode)
            or before_path.st_mode & 0o111 == 0
        ):
            raise RuntimeError(
                f"fixed audit tool {name!r} is not an executable regular file"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        digest = hashlib.sha256()
        length = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            length += len(block)
        after = os.fstat(descriptor)
        after_path = os.lstat(path)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        path_identity = (
            after_path.st_dev,
            after_path.st_ino,
            after_path.st_mode,
            after_path.st_size,
            after_path.st_mtime_ns,
            after_path.st_ctime_ns,
        )
        if (
            identity != after_identity
            or identity != path_identity
            or length != before.st_size
        ):
            raise RuntimeError(f"fixed audit tool {name!r} changed while read")
        return {
            "artifactName": path.name,
            "byteLength": length,
            "sha256": digest.hexdigest(),
        }
    except OSError as error:
        raise RuntimeError(
            f"fixed absolute audit tool {name!r} is unavailable: {path}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fixed_tool_artifacts() -> dict[str, dict[str, object]]:
    if platform.system() != "Darwin":
        raise RuntimeError("Mach-O binary audit requires Darwin fixed tools")
    return {
        name: _tool_artifact(name, path)
        for name, path in sorted(FIXED_TOOLS.items())
    }


def run(tool_name: str, *arguments: str) -> str:
    if tool_name not in FIXED_TOOLS:
        raise RuntimeError(f"unsupported fixed audit tool {tool_name!r}")
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "",
        "TMPDIR": "/private/tmp",
    }
    completed = subprocess.run(
        (str(FIXED_TOOLS[tool_name]), *arguments),
        check=True,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: audit_binary.py LIBRARY")
    library = Path(sys.argv[1]).resolve(strict=True)
    architecture = platform.machine()
    if architecture not in {"arm64", "x86_64"}:
        raise RuntimeError(f"unsupported build architecture: {architecture}")

    tool_artifacts = _fixed_tool_artifacts()
    file_description = run("file", str(library))
    if "Mach-O 64-bit" not in file_description or architecture not in file_description:
        raise RuntimeError(
            f"binary architecture does not match host {architecture}: "
            f"{file_description.strip()}"
        )

    load_commands = run("otool", "-l", str(library))
    if "__LLVM" in load_commands:
        raise RuntimeError("embedded LLVM/LTO payload is forbidden")

    disassembly = run("otool", "-tvV", str(library))
    fused_match = FORBIDDEN_FUSED_OPERATION.search(disassembly)
    if fused_match is not None:
        raise RuntimeError(
            f"forbidden fused floating operation: {fused_match.group(0)}"
        )

    exports = {
        line.strip().split()[-1]
        for line in run("nm", "-gU", str(library)).splitlines()
        if line.strip()
    }
    missing = EXPECTED_EXPORTS - exports
    if missing:
        raise RuntimeError(f"missing public ABI exports: {sorted(missing)}")
    unexpected = exports - EXPECTED_EXPORTS
    if unexpected:
        raise RuntimeError(f"unexpected public ABI exports: {sorted(unexpected)}")
    if _fixed_tool_artifacts() != tool_artifacts:
        raise RuntimeError("fixed audit tool artifacts changed during binary audit")

    report = {
        "architecture": architecture,
        "exports": len(EXPECTED_EXPORTS),
        "fusedFpInstructions": 0,
        "ltoPayloads": 0,
        "tools": tool_artifacts,
    }
    print(
        "binary audit passed: "
        + json.dumps(
            report,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
