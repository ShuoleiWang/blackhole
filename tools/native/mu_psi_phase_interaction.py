"""Read-only manifest analysis for the nested16 ``mu x psi x phase`` design.

This tool consumes two already-published native-CPU refinement checkpoints:

* the parent ``rho=16, mu=32, psi=64`` five-pass checkpoint; and
* the target ``rho=16, mu=16, psi=64`` five-pass checkpoint.

It never calls a ray evaluator.  An optional cache gate opens only the parent
job through the standard-library, ``O_NOFOLLOW`` golden-cache authenticator;
it cannot call a runner or publish cache content.  Both input checkpoints are
accepted only through the public checkpoint verifier with an external
whole-manifest SHA-256 and the same explicitly selected dylib.  The result is
a permanently non-qualifying diagnostic.  In particular, no edge,
interaction, cache substitution, or v2 comparison result can make it product
eligible or trigger an automatic escalation.

The retained five-pass summaries form these cells::

    A = parent full           (mu=32, psi=64, phase=0.0)
    B = parent phase-shifted  (mu=32, psi=64, phase=0.5)
    C = parent half-mu        (mu=16, psi=64, phase=0.0)
    D = target phase-shifted  (mu=16, psi=64, phase=0.5)
    E = parent half-psi       (mu=32, psi=32, phase=0.0)
    F = target half-psi       (mu=16, psi=32, phase=0.0)

The target full pass must reproduce C after pass identity is removed.  This
is intentionally not a raw summary-descriptor comparison because ``gridId``,
pass name, and pass index carry producer-pass identity.  The production sample
audit already excludes those identity fields and hashes physical samples and
transport only, so its raw SHA-256 is required to match exactly in addition to
all retained physical summary values and normalized pass evidence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from types import CodeType, FunctionType
from typing import Any, Final, Mapping, Sequence
import uuid

if __package__ in (None, ""):
    _BOOTSTRAP_ROOT = Path(__file__).absolute().parents[2]
    if str(_BOOTSTRAP_ROOT) not in sys.path:
        sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from offline.job import canonical_json_bytes
from offline.kerr_returning_radiation_convergence_v2 import (
    FATE_NAMES,
    KerrReturningRadiationConvergenceV2Policy,
    KerrReturningRadiationGridSummaryV2,
    compare_kerr_returning_radiation_grids_v2,
)
from offline.returning_radiation_fate_quadrature import (
    EmittedFluxDirectionNode,
    kerrbb_d20_emitted_flux_direction_nodes,
)
from offline.kerr_returning_radiation_refinement_checkpoint import (
    NATIVE_CPU_EVALUATOR_MODE,
    NATIVE_MANIFEST_SCHEMA,
    KerrReturningRadiationVerifiedRefinementCheckpoint,
    verify_kerr_returning_radiation_refinement_checkpoint,
)
from tools.native import golden_cache as _golden_cache


SCHEMA: Final = "blackhole.kerr-mu-psi-phase-interaction-analysis/v1"
IMPLEMENTATION_ID: Final = (
    "tools.native.mu-psi-phase-interaction/manifest-read-only-v1"
)
CLASSIFICATION: Final = (
    "read-only-non-qualifying-finite-grid-interaction-diagnostic"
)
REPORT_NAME_SUFFIX: Final = ".json"
MAXIMUM_SOURCE_FILE_BYTES: Final = 16 * 1024 * 1024
MAXIMUM_DYLIB_BYTES: Final = 256 * 1024 * 1024
MAXIMUM_REPORT_BYTES: Final = 64 * 1024 * 1024

_PATH_TYPE: Final = type(Path())
_MODULE_PATH: Final = Path(os.path.abspath(__file__))
_SOURCE_ROOT: Final = _MODULE_PATH.parents[2]
_SOURCE_LOGICAL_PATHS: Final = (
    "offline/job.py",
    "offline/kerr_returning_radiation_convergence_v2.py",
    "offline/kerr_returning_radiation_refinement_checkpoint.py",
    "offline/returning_radiation_fate_quadrature.py",
    "tools/native/golden_cache.py",
    "tools/native/mu_psi_phase_interaction.py",
)
_VERIFY_PUBLIC_ENTRY: Final = verify_kerr_returning_radiation_refinement_checkpoint
_VERIFY_CALL_ENTRY: Final = _VERIFY_PUBLIC_ENTRY
_PHASE_NODE_PUBLIC_ENTRY: Final = kerrbb_d20_emitted_flux_direction_nodes
_PHASE_NODE_CALL_ENTRY: Final = _PHASE_NODE_PUBLIC_ENTRY
_COMPARE_PUBLIC_ENTRY: Final = compare_kerr_returning_radiation_grids_v2
_COMPARE_CALL_ENTRY: Final = _COMPARE_PUBLIC_ENTRY
_CANONICAL_JSON_PUBLIC_ENTRY: Final = canonical_json_bytes
_STRICT_POLICY: Final = KerrReturningRadiationConvergenceV2Policy()
_GOLDEN_SECURE_JOB_DIRECTORY: Final = _golden_cache._secure_job_directory
_GOLDEN_READ_REGULAR_FILE_AT: Final = _golden_cache._read_regular_file_at
_GOLDEN_VALIDATE_JOB_DOCUMENT: Final = _golden_cache._validated_job_document
_GOLDEN_STABLE_DIRECTORY_NAMES: Final = _golden_cache._stable_directory_names
_GOLDEN_TASK_STEM: Final = _golden_cache._task_stem
_GOLDEN_REQUIRE_REGULAR_ENTRY: Final = _golden_cache._require_regular_entry
_GOLDEN_VALIDATE_RECEIPT: Final = _golden_cache._validated_receipt
_GOLDEN_VALIDATE_PAYLOAD_RECORDS: Final = _golden_cache._validated_payload_records
_GOLDEN_STABLE_STAT_IDENTITY: Final = _golden_cache._stable_stat_identity
_SUMMARY_KEYS: Final = {
    "coefficientIndexOrder",
    "emitterAreas",
    "fateFractionOrder",
    "fateFractions",
    "g2ReturnedPowerColumns",
    "gridId",
    "matrices",
    "maximumNormalizedSampleWeight",
    "properPowerEquation",
    "receiverAreas",
    "schema",
}
_PASS_NAMES: Final = (
    "full",
    "half-rho",
    "half-mu",
    "half-psi",
    "phase-shifted",
)


class MuPsiPhaseInteractionError(RuntimeError):
    """Raised when inputs cannot support the closed diagnostic report."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or len(value) != 64 or value != value.lower():
        raise MuPsiPhaseInteractionError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise MuPsiPhaseInteractionError(
            f"{label} must be a lowercase SHA-256 digest"
        ) from error
    if len(raw) != 32:
        raise MuPsiPhaseInteractionError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _require_exact_dict(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise MuPsiPhaseInteractionError(f"{label} has a non-exact shape")
    return value


def _positive_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _stable_regular_file(path: Path, label: str, maximum_bytes: int) -> bytes:
    if type(path) is not _PATH_TYPE or not path.is_absolute():
        raise MuPsiPhaseInteractionError(
            f"{label} must be an exact absolute platform Path"
        )
    absolute = Path(os.path.abspath(os.fspath(path)))
    if path != absolute or not path.name or path.name in (".", ".."):
        raise MuPsiPhaseInteractionError(f"{label} path is non-canonical")
    directory_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    directories = [os.open(os.sep, directory_flags)]
    descriptor = -1
    try:
        for component in path.parts[1:-1]:
            directories.append(
                os.open(component, directory_flags, dir_fd=directories[-1])
            )
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directories[-1],
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise MuPsiPhaseInteractionError(f"{label} is not a regular file")
        if before.st_size < 1 or before.st_size > maximum_bytes:
            raise MuPsiPhaseInteractionError(f"{label} exceeds its byte bound")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise MuPsiPhaseInteractionError(f"{label} ended while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise MuPsiPhaseInteractionError(f"{label} grew while reading")
        after = os.fstat(descriptor)
        identity = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise MuPsiPhaseInteractionError(f"{label} changed while reading")
        return b"".join(chunks)
    except OSError as error:
        raise MuPsiPhaseInteractionError(f"{label} is not securely readable") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        for directory in reversed(directories):
            os.close(directory)


def _artifact(path: Path, logical_path: str, maximum_bytes: int) -> dict[str, Any]:
    payload = _stable_regular_file(path, logical_path, maximum_bytes)
    return {
        "byteLength": len(payload),
        "logicalPath": logical_path,
        "sha256": _sha256_bytes(payload),
    }


def _source_snapshot() -> dict[str, Any]:
    if _MODULE_PATH != _SOURCE_ROOT / _SOURCE_LOGICAL_PATHS[-1]:
        raise MuPsiPhaseInteractionError(
            "interaction analyzer has no canonical checkout source identity"
        )
    artifacts = [
        _artifact(
            _SOURCE_ROOT / logical_path,
            logical_path,
            MAXIMUM_SOURCE_FILE_BYTES,
        )
        for logical_path in _SOURCE_LOGICAL_PATHS
    ]
    return {
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
        "manifestSha256": _canonical_sha256(artifacts),
    }


def _normalized_code_object(code: CodeType, logical_path: str) -> CodeType:
    constants = tuple(
        _normalized_code_object(value, logical_path)
        if type(value) is CodeType
        else value
        for value in code.co_consts
    )
    # Supplying ``co_code`` explicitly strips CPython's live adaptive/quickened
    # execution state before code-identity comparison, including when this verifier
    # snapshots its own currently executing code object.
    return code.replace(
        co_code=code.co_code,
        co_filename=logical_path,
        co_consts=constants,
    )


def _source_code_object(
    module_code: CodeType,
    qualname: str,
) -> CodeType:
    matches: list[CodeType] = []

    def visit(code: CodeType) -> None:
        if code.co_qualname == qualname:
            matches.append(code)
        for value in code.co_consts:
            if type(value) is CodeType:
                visit(value)

    visit(module_code)
    if len(matches) != 1:
        raise MuPsiPhaseInteractionError(
            f"source does not contain one exact callable {qualname}"
        )
    return matches[0]


def _constant_identity_document(value: Any, logical_path: str) -> dict[str, Any]:
    """Describe constants by value, independent of CPython interning state."""

    if type(value) is CodeType:
        return {
            "kind": "code",
            "value": _code_identity_document(value, logical_path),
        }
    if value is None:
        return {"kind": "none"}
    if type(value) is bool:
        return {"kind": "bool", "value": value}
    if type(value) is int:
        return {"decimal": str(value), "kind": "int"}
    if type(value) is float:
        return {"hex": value.hex(), "kind": "float"}
    if type(value) is complex:
        return {
            "imagHex": value.imag.hex(),
            "kind": "complex",
            "realHex": value.real.hex(),
        }
    if type(value) is str:
        payload = value.encode("utf-8", "surrogatepass")
        return {
            "byteLength": len(payload),
            "kind": "str",
            "sha256": _sha256_bytes(payload),
        }
    if type(value) is bytes:
        return {
            "byteLength": len(value),
            "kind": "bytes",
            "sha256": _sha256_bytes(value),
        }
    if type(value) is tuple:
        return {
            "items": [
                _constant_identity_document(item, logical_path) for item in value
            ],
            "kind": "tuple",
        }
    if type(value) is frozenset:
        items = [
            _constant_identity_document(item, logical_path) for item in value
        ]
        items.sort(key=canonical_json_bytes)
        return {"items": items, "kind": "frozenset"}
    if type(value) is dict:
        items = [
            {
                "key": _constant_identity_document(key, logical_path),
                "value": _constant_identity_document(item, logical_path),
            }
            for key, item in value.items()
        ]
        items.sort(key=canonical_json_bytes)
        return {"items": items, "kind": "dict"}
    raise MuPsiPhaseInteractionError(
        f"loaded callable contains unsupported constant type {type(value).__name__}"
    )


def _code_identity_document(code: CodeType, logical_path: str) -> dict[str, Any]:
    constants = [
        _constant_identity_document(value, logical_path) for value in code.co_consts
    ]
    return {
        "argCount": code.co_argcount,
        "bytecodeByteLength": len(code.co_code),
        "bytecodeSha256": _sha256_bytes(code.co_code),
        "cellVars": list(code.co_cellvars),
        "constants": constants,
        "exceptionTableByteLength": len(code.co_exceptiontable),
        "exceptionTableSha256": _sha256_bytes(code.co_exceptiontable),
        "firstLineNumber": code.co_firstlineno,
        "flags": code.co_flags,
        "freeVars": list(code.co_freevars),
        "keywordOnlyArgCount": code.co_kwonlyargcount,
        "lineTableByteLength": len(code.co_linetable),
        "lineTableSha256": _sha256_bytes(code.co_linetable),
        "logicalPath": logical_path,
        "name": code.co_name,
        "names": list(code.co_names),
        "nestedLocalCount": code.co_nlocals,
        "positionalOnlyArgCount": code.co_posonlyargcount,
        "qualname": code.co_qualname,
        "stackSize": code.co_stacksize,
        "varNames": list(code.co_varnames),
    }


def _module_function_bindings() -> tuple[tuple[str, FunctionType], ...]:
    return tuple(
        sorted(
            (
                (name, value)
                for name, value in globals().items()
                if type(value) is FunctionType
                and object.__getattribute__(value, "__module__") == __name__
            ),
            key=lambda item: item[0],
        )
    )


def _callable_snapshot() -> dict[str, Any]:
    """Bind loaded function code to freshly compiled, no-follow source bytes."""

    secure_job_source = getattr(
        _GOLDEN_SECURE_JOB_DIRECTORY,
        "__wrapped__",
        None,
    )
    analyzer_items: list[tuple[str, FunctionType, str]] = []
    seen_analyzer_functions: set[int] = set()
    for _binding_name, function in _module_function_bindings():
        if function is _callable_snapshot or id(function) in seen_analyzer_functions:
            continue
        seen_analyzer_functions.add(id(function))
        analyzer_items.append(
            (
                f"analyzer.{object.__getattribute__(function, '__name__')}",
                function,
                "tools/native/mu_psi_phase_interaction.py",
            )
        )
    analyzer_specifications = tuple(analyzer_items)
    external_specifications = (
        (
            "checkpoint.publicVerifier",
            _VERIFY_CALL_ENTRY,
            "offline/kerr_returning_radiation_refinement_checkpoint.py",
        ),
        (
            "convergence.v2Comparator",
            _COMPARE_CALL_ENTRY,
            "offline/kerr_returning_radiation_convergence_v2.py",
        ),
        (
            "quadrature.phaseNodeGenerator",
            _PHASE_NODE_CALL_ENTRY,
            "offline/returning_radiation_fate_quadrature.py",
        ),
        (
            "canonical.jsonEncoder",
            canonical_json_bytes,
            "offline/job.py",
        ),
        (
            "golden.secureJobDirectory",
            secure_job_source,
            "tools/native/golden_cache.py",
        ),
        (
            "golden.readFile",
            _GOLDEN_READ_REGULAR_FILE_AT,
            "tools/native/golden_cache.py",
        ),
        (
            "golden.validateJob",
            _GOLDEN_VALIDATE_JOB_DOCUMENT,
            "tools/native/golden_cache.py",
        ),
        (
            "golden.validateReceipt",
            _GOLDEN_VALIDATE_RECEIPT,
            "tools/native/golden_cache.py",
        ),
        (
            "golden.validatePayload",
            _GOLDEN_VALIDATE_PAYLOAD_RECORDS,
            "tools/native/golden_cache.py",
        ),
        (
            "golden.stableDirectoryNames",
            _GOLDEN_STABLE_DIRECTORY_NAMES,
            "tools/native/golden_cache.py",
        ),
        (
            "golden.taskStem",
            _GOLDEN_TASK_STEM,
            "tools/native/golden_cache.py",
        ),
        (
            "golden.requireRegularEntry",
            _GOLDEN_REQUIRE_REGULAR_ENTRY,
            "tools/native/golden_cache.py",
        ),
        (
            "golden.stableStatIdentity",
            _GOLDEN_STABLE_STAT_IDENTITY,
            "tools/native/golden_cache.py",
        ),
    )
    specifications = analyzer_specifications + external_specifications
    compiled: dict[str, CodeType] = {}
    entries: list[dict[str, Any]] = []
    for label, function, logical_path in specifications:
        if type(function) is not FunctionType:
            raise MuPsiPhaseInteractionError(
                f"loaded callable {label} has a foreign type"
            )
        code = object.__getattribute__(function, "__code__")
        qualname = object.__getattribute__(function, "__qualname__")
        if type(code) is not CodeType or type(qualname) is not str:
            raise MuPsiPhaseInteractionError(
                f"loaded callable {label} has no exact code identity"
            )
        module_code = compiled.get(logical_path)
        if module_code is None:
            source = _stable_regular_file(
                _SOURCE_ROOT / logical_path,
                f"callable source {logical_path}",
                MAXIMUM_SOURCE_FILE_BYTES,
            )
            try:
                module_code = compile(
                    source,
                    logical_path,
                    "exec",
                    dont_inherit=True,
                    optimize=sys.flags.optimize,
                )
            except (SyntaxError, ValueError) as error:
                raise MuPsiPhaseInteractionError(
                    f"callable source {logical_path} cannot be compiled"
                ) from error
            compiled[logical_path] = module_code
        source_code = _source_code_object(module_code, qualname)
        loaded_bytes = canonical_json_bytes(
            _code_identity_document(code, logical_path)
        )
        source_bytes = canonical_json_bytes(
            _code_identity_document(source_code, logical_path)
        )
        if loaded_bytes != source_bytes:
            raise MuPsiPhaseInteractionError(
                f"loaded callable {label} differs from reported source "
                f"({_sha256_bytes(loaded_bytes)} != "
                f"{_sha256_bytes(source_bytes)})"
            )
        defaults_payload = canonical_json_bytes(
            _constant_identity_document(
                (
                    object.__getattribute__(function, "__defaults__"),
                    object.__getattribute__(function, "__kwdefaults__"),
                ),
                logical_path,
            )
        )
        entries.append(
            {
                "label": label,
                "logicalPath": logical_path,
                "codeIdentityByteLength": len(loaded_bytes),
                "codeIdentitySha256": _sha256_bytes(loaded_bytes),
                "defaultsSha256": _sha256_bytes(defaults_payload),
                "qualname": qualname,
                "sourceCodeExact": True,
            }
        )
    return {
        "callableCount": len(entries),
        "entries": entries,
        "manifestSha256": _canonical_sha256(entries),
    }


def _idle_snapshot_verifier_identity() -> dict[str, Any]:
    """Snapshot ``_callable_snapshot`` only while that function is idle."""

    function = _CALLABLE_SNAPSHOT_CALL_ENTRY
    if type(function) is not FunctionType:
        raise MuPsiPhaseInteractionError(
            "callable snapshot verifier has a foreign type"
        )
    logical_path = "tools/native/mu_psi_phase_interaction.py"
    source = _stable_regular_file(
        _SOURCE_ROOT / logical_path,
        "callable snapshot verifier source",
        MAXIMUM_SOURCE_FILE_BYTES,
    )
    module_code = compile(
        source,
        logical_path,
        "exec",
        dont_inherit=True,
        optimize=sys.flags.optimize,
    )
    code = object.__getattribute__(function, "__code__")
    qualname = object.__getattribute__(function, "__qualname__")
    source_code = _source_code_object(module_code, qualname)
    loaded_bytes = canonical_json_bytes(_code_identity_document(code, logical_path))
    source_bytes = canonical_json_bytes(
        _code_identity_document(source_code, logical_path)
    )
    if loaded_bytes != source_bytes:
        raise MuPsiPhaseInteractionError(
            "callable snapshot verifier differs from reported source"
        )
    defaults_payload = canonical_json_bytes(
        _constant_identity_document(
            (
                object.__getattribute__(function, "__defaults__"),
                object.__getattribute__(function, "__kwdefaults__"),
            ),
            logical_path,
        )
    )
    return {
        "label": "analyzer._callable_snapshot",
        "logicalPath": logical_path,
        "codeIdentityByteLength": len(loaded_bytes),
        "codeIdentitySha256": _sha256_bytes(loaded_bytes),
        "defaultsSha256": _sha256_bytes(defaults_payload),
        "qualname": qualname,
        "sourceCodeExact": True,
    }


def _summary_from_document(
    value: Mapping[str, Any],
    descriptor_sha256: str,
    label: str,
) -> KerrReturningRadiationGridSummaryV2:
    if type(value) is not dict or set(value) != _SUMMARY_KEYS:
        raise MuPsiPhaseInteractionError(f"{label} summary shape is non-canonical")
    try:
        summary = KerrReturningRadiationGridSummaryV2(
            value["gridId"],
            tuple(value["receiverAreas"]),
            tuple(value["emitterAreas"]),
            tuple(tuple(row) for row in value["matrices"]),
            tuple(value["g2ReturnedPowerColumns"]),
            tuple(tuple(row) for row in value["fateFractions"]),
            value["maximumNormalizedSampleWeight"],
        )
        summary.revalidate()
    except (KeyError, TypeError, ValueError) as error:
        raise MuPsiPhaseInteractionError(
            f"{label} summary cannot be reconstructed through convergence v2"
        ) from error
    _require_sha256(descriptor_sha256, f"{label} summary descriptor SHA-256")
    if summary.model_descriptor_sha256 != descriptor_sha256:
        raise MuPsiPhaseInteractionError(
            f"{label} summary differs from its v2 descriptor SHA-256"
        )
    if canonical_json_bytes(dict(summary.descriptor())) != canonical_json_bytes(value):
        raise MuPsiPhaseInteractionError(
            f"{label} summary differs from its canonical v2 reconstruction"
        )
    return summary


def _physical_summary_document(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the producer-local grid identifier."""

    if type(value) is not dict or set(value) != _SUMMARY_KEYS:
        raise MuPsiPhaseInteractionError("physical summary input is malformed")
    return {key: deepcopy(item) for key, item in value.items() if key != "gridId"}


def _normalized_pass_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove pass identity while retaining the physical sample-audit digest."""

    if type(value) is not dict:
        raise MuPsiPhaseInteractionError("pass evidence must be an exact object")
    required = {"passIndex", "passName", "sampleAuditSha256"}
    if not required.issubset(value):
        raise MuPsiPhaseInteractionError("pass evidence lacks identity fields")
    _require_sha256(value["sampleAuditSha256"], "raw sample-audit SHA-256")
    result = {
        key: deepcopy(item)
        for key, item in value.items()
        if key not in ("passIndex", "passName")
    }
    witness = result.get("maximumNormalizedSampleWeightWitness")
    if type(witness) is not dict or not {"passIndex", "passName"}.issubset(witness):
        raise MuPsiPhaseInteractionError(
            "maximum sample-weight witness lacks pass identity"
        )
    result["maximumNormalizedSampleWeightWitness"] = {
        key: deepcopy(item)
        for key, item in witness.items()
        if key not in ("passIndex", "passName")
    }
    return result


def _checkpoint_parts(document: Mapping[str, Any], label: str) -> dict[str, Any]:
    if type(document) is not dict:
        raise MuPsiPhaseInteractionError(f"{label} checkpoint document is malformed")
    if (
        document.get("schema") != NATIVE_MANIFEST_SCHEMA
        or document.get("evaluatorMode") != NATIVE_CPU_EVALUATOR_MODE
    ):
        raise MuPsiPhaseInteractionError(
            f"{label} checkpoint is not a native-CPU v2 manifest"
        )
    try:
        authentication = document["authenticatedConvergenceV2"]
        summaries = authentication["summaries"]
        descriptor = authentication["descriptor"]
        pass_evidence = descriptor["passEvidence"]
        policy = authentication["policy"]
        producer = document["producer"]
        definition = producer["cacheDefinition"]
        plan = definition["scientificPlan"]
        scientific_identity = definition["scientificIdentity"]
        runtime = document["runtimeNumericBackend"]
        source_freeze = document["sourceFreeze"]
    except (KeyError, TypeError) as error:
        raise MuPsiPhaseInteractionError(
            f"{label} checkpoint lacks analyzer evidence"
        ) from error
    if (
        type(summaries) is not list
        or len(summaries) != 5
        or type(pass_evidence) is not list
        or len(pass_evidence) != 5
        or type(plan) is not dict
        or type(plan.get("passes")) is not list
        or len(plan["passes"]) != 5
    ):
        raise MuPsiPhaseInteractionError(
            f"{label} checkpoint does not retain the exact five-pass topology"
        )
    expected_policy_document = dict(_STRICT_POLICY.descriptor())
    if (
        type(policy) is not dict
        or policy.get("descriptor") != expected_policy_document
        or policy.get("descriptorSha256") != _STRICT_POLICY.model_descriptor_sha256
    ):
        raise MuPsiPhaseInteractionError(
            f"{label} checkpoint does not use the canonical strict v2 policy"
        )
    rebuilt_summaries: list[KerrReturningRadiationGridSummaryV2] = []
    for index, (summary_entry, evidence, pass_plan) in enumerate(
        zip(summaries, pass_evidence, plan["passes"])
    ):
        if type(summary_entry) is not dict or set(summary_entry) != {
            "descriptor",
            "descriptorSha256",
        }:
            raise MuPsiPhaseInteractionError(
                f"{label} summary {index} has a non-exact wrapper"
            )
        summary = _summary_from_document(
            summary_entry["descriptor"],
            summary_entry["descriptorSha256"],
            f"{label} pass {index}",
        )
        expected_name = _PASS_NAMES[index]
        if (
            type(evidence) is not dict
            or type(pass_plan) is not dict
            or evidence.get("passIndex") != index
            or evidence.get("passName") != expected_name
            or pass_plan.get("name") != expected_name
            or summary.grid_id != f"cached-{expected_name}"
        ):
            raise MuPsiPhaseInteractionError(
                f"{label} pass {index} identity is non-canonical"
            )
        for evidence_key, plan_key in (
            ("rhoOrder", "rhoOrder"),
            ("muOrder", "muOrder"),
            ("psiCount", "psiCount"),
            ("phaseCells", "phaseCells"),
        ):
            if evidence.get(evidence_key) != pass_plan.get(plan_key):
                raise MuPsiPhaseInteractionError(
                    f"{label} pass {index} evidence differs from its plan"
                )
        rebuilt_summaries.append(summary)
    return {
        "document": document,
        "summaries": tuple(rebuilt_summaries),
        "summaryEntries": tuple(summaries),
        "passEvidence": tuple(pass_evidence),
        "plan": plan,
        "scientificIdentity": scientific_identity,
        "runtime": runtime,
        "sourceFreeze": source_freeze,
    }


def _require_design(parent: Mapping[str, Any], target: Mapping[str, Any]) -> None:
    parent_plan = parent["plan"]
    target_plan = target["plan"]
    for label, plan, expected_mu in (
        ("parent", parent_plan, 32),
        ("target", target_plan, 16),
    ):
        full = plan["passes"][0]
        if (
            plan.get("formulation") != "forward"
            or full.get("rhoOrder") != 16
            or full.get("muOrder") != expected_mu
            or full.get("psiCount") != 64
            or full.get("phaseCells") != 0.0
        ):
            raise MuPsiPhaseInteractionError(
                f"{label} checkpoint has the wrong rho/mu/psi/phase design"
            )
    parent_document = parent["document"]
    if parent_document.get("qualified") is not False:
        raise MuPsiPhaseInteractionError(
            "parent checkpoint must be the retained non-qualified native manifest"
        )
    if type(target["document"].get("qualified")) is not bool:
        raise MuPsiPhaseInteractionError("target qualification flag is malformed")
    comparable_parent_plan = {
        key: deepcopy(value)
        for key, value in parent_plan.items()
        if key not in ("directionCount", "passes")
    }
    comparable_target_plan = {
        key: deepcopy(value)
        for key, value in target_plan.items()
        if key not in ("directionCount", "passes")
    }
    if comparable_parent_plan != comparable_target_plan:
        raise MuPsiPhaseInteractionError(
            "parent and target scientific-plan topology differs beyond grid order"
        )
    if parent["scientificIdentity"] != target["scientificIdentity"]:
        raise MuPsiPhaseInteractionError(
            "parent and target scientific geometry/options/source closure differ"
        )
    if parent["runtime"] != target["runtime"]:
        raise MuPsiPhaseInteractionError(
            "parent and target runtime numeric backend descriptors differ"
        )
    if parent["sourceFreeze"] != target["sourceFreeze"]:
        raise MuPsiPhaseInteractionError(
            "parent and target source-freeze snapshots differ"
        )


def _physical_summary_sha(summary: KerrReturningRadiationGridSummaryV2) -> str:
    return _canonical_sha256(_physical_summary_document(dict(summary.descriptor())))


def _cell_document(
    name: str,
    checkpoint: str,
    pass_index: int,
    summary: KerrReturningRadiationGridSummaryV2,
    evidence: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    pass_plan = plan["passes"][pass_index]
    return {
        "cell": name,
        "checkpoint": checkpoint,
        "gridId": summary.grid_id,
        "muOrder": pass_plan["muOrder"],
        "passIndex": pass_index,
        "passName": pass_plan["name"],
        "phaseCells": pass_plan["phaseCells"],
        "physicalSummarySha256": _physical_summary_sha(summary),
        "psiCount": pass_plan["psiCount"],
        "rhoOrder": pass_plan["rhoOrder"],
        "sampleAuditSha256": evidence["sampleAuditSha256"],
        "summaryDescriptorSha256": summary.model_descriptor_sha256,
    }


def _edge_document(
    left_name: str,
    left: KerrReturningRadiationGridSummaryV2,
    right_name: str,
    right: KerrReturningRadiationGridSummaryV2,
) -> dict[str, Any]:
    try:
        comparison = _COMPARE_CALL_ENTRY(
            left,
            right,
            _STRICT_POLICY,
        )
        comparison.revalidate()
    except (TypeError, ValueError) as error:
        raise MuPsiPhaseInteractionError(
            f"edge {left_name}->{right_name} cannot be compared through v2"
        ) from error
    return {
        "leftCell": left_name,
        "rightCell": right_name,
        "v2Comparison": dict(comparison.descriptor()),
        "v2ComparisonSha256": comparison.model_descriptor_sha256,
        "withinV2Tolerance": comparison.converged,
    }


def _observable_channels(
    summary: KerrReturningRadiationGridSummaryV2,
) -> dict[str, tuple[tuple[str | int, ...], float]]:
    channels: dict[str, tuple[tuple[str | int, ...], float]] = {}
    for receiver_index, row in enumerate(summary.proper_power_matrix):
        for source_index, value in enumerate(row):
            channels[f"properPower/{receiver_index}/{source_index}"] = (
                ("properPower", receiver_index, source_index),
                value,
            )
    for source_index, value in enumerate(summary.g2_columns):
        channels[f"g2/{source_index}"] = (("g2", source_index), value)
    for source_index, row in enumerate(summary.fate_fractions):
        for fate_index, value in enumerate(row):
            channels[f"fate/{source_index}/{fate_index}"] = (
                ("fate", source_index, FATE_NAMES[fate_index]),
                value,
            )
    return channels


def _interaction_document(
    name: str,
    baseline_name: str,
    first_name: str,
    second_name: str,
    joint_name: str,
    baseline: KerrReturningRadiationGridSummaryV2,
    first: KerrReturningRadiationGridSummaryV2,
    second: KerrReturningRadiationGridSummaryV2,
    joint: KerrReturningRadiationGridSummaryV2,
) -> dict[str, Any]:
    """Return ``joint - second - first + baseline`` diagnostics."""

    channel_sets = tuple(
        _observable_channels(item) for item in (baseline, first, second, joint)
    )
    keys = tuple(sorted(channel_sets[0]))
    if any(tuple(sorted(item)) != keys for item in channel_sets[1:]):
        raise MuPsiPhaseInteractionError(f"{name} channel shapes differ")
    signed_values: list[float] = []
    normalized_values: list[float] = []
    witnesses: list[dict[str, Any]] = []
    for key in keys:
        path = channel_sets[0][key][0]
        values = tuple(item[key][1] for item in channel_sets)
        interaction = _positive_zero(
            math.fsum((values[3], -values[2], -values[1], values[0]))
        )
        if not math.isfinite(interaction):
            raise MuPsiPhaseInteractionError(f"{name} interaction is non-finite")
        scale = max(abs(value) for value in values)
        normalized = 0.0 if scale == 0.0 else abs(interaction) / scale
        signed_values.append(interaction)
        normalized_values.append(normalized)
        witnesses.append(
            {
                "baseline": values[0],
                "first": values[1],
                "interaction": interaction,
                "joint": values[3],
                "path": list(path),
                "second": values[2],
                "scaleNormalizedAbsoluteInteraction": normalized,
            }
        )
    receiver_count = len(baseline.receiver_areas)
    source_count = len(baseline.emitter_areas)
    signed_proper_power = tuple(
        tuple(
            _positive_zero(
                math.fsum(
                    (
                        joint.proper_power_matrix[receiver_index][source_index],
                        -second.proper_power_matrix[receiver_index][source_index],
                        -first.proper_power_matrix[receiver_index][source_index],
                        baseline.proper_power_matrix[receiver_index][source_index],
                    )
                )
            )
            for source_index in range(source_count)
        )
        for receiver_index in range(receiver_count)
    )
    signed_g2 = tuple(
        _positive_zero(
            math.fsum(
                (
                    joint.g2_columns[source_index],
                    -second.g2_columns[source_index],
                    -first.g2_columns[source_index],
                    baseline.g2_columns[source_index],
                )
            )
        )
        for source_index in range(source_count)
    )
    signed_fates = tuple(
        tuple(
            _positive_zero(
                math.fsum(
                    (
                        joint.fate_fractions[source_index][fate_index],
                        -second.fate_fractions[source_index][fate_index],
                        -first.fate_fractions[source_index][fate_index],
                        baseline.fate_fractions[source_index][fate_index],
                    )
                )
            )
            for fate_index in range(len(FATE_NAMES))
        )
        for source_index in range(source_count)
    )
    tolerance_columns: list[dict[str, Any]] = []
    maximum_tolerance_units = 0.0
    for source_index in range(source_count):
        column_scale = max(
            baseline.g2_columns[source_index],
            first.g2_columns[source_index],
            second.g2_columns[source_index],
            joint.g2_columns[source_index],
            _STRICT_POLICY.g2_column_relative_floor,
        )
        g2_units = (
            abs(signed_g2[source_index])
            / column_scale
            / _STRICT_POLICY.g2_column_relative_tolerance
        )
        column_l1_units = (
            math.fsum(
                abs(signed_proper_power[receiver_index][source_index])
                for receiver_index in range(receiver_count)
            )
            / column_scale
            / _STRICT_POLICY.column_normalized_l1_tolerance
        )
        fate_row = signed_fates[source_index]
        fate_tv_units = (
            0.5 * math.fsum(abs(value) for value in fate_row)
            / _STRICT_POLICY.fate_total_variation_tolerance
        )
        fate_absolute_units = (
            max(abs(value) for value in fate_row)
            / _STRICT_POLICY.fate_component_absolute_tolerance
        )
        relative_units: list[float] = []
        for fate_index, interaction in enumerate(fate_row):
            fate_scale = max(
                baseline.fate_fractions[source_index][fate_index],
                first.fate_fractions[source_index][fate_index],
                second.fate_fractions[source_index][fate_index],
                joint.fate_fractions[source_index][fate_index],
            )
            if fate_scale >= _STRICT_POLICY.fate_component_relative_floor:
                relative_units.append(
                    abs(interaction)
                    / fate_scale
                    / _STRICT_POLICY.fate_component_relative_tolerance
                )
        fate_relative_units = max(relative_units) if relative_units else 0.0
        column_maximum = max(
            g2_units,
            column_l1_units,
            fate_tv_units,
            fate_absolute_units,
            fate_relative_units,
        )
        maximum_tolerance_units = max(maximum_tolerance_units, column_maximum)
        tolerance_columns.append(
            {
                "columnL1InteractionToleranceUnits": column_l1_units,
                "columnScale": column_scale,
                "fateAbsoluteInteractionToleranceUnits": fate_absolute_units,
                "fateRelativeInteractionToleranceUnits": fate_relative_units,
                "fateTotalVariationInteractionToleranceUnits": fate_tv_units,
                "g2InteractionToleranceUnits": g2_units,
                "maximumInteractionToleranceUnits": column_maximum,
                "sourceColumn": source_index,
            }
        )
    maximum_index = max(
        range(len(witnesses)),
        key=lambda index: (abs(signed_values[index]), keys[index]),
    )
    maximum_normalized_index = max(
        range(len(witnesses)),
        key=lambda index: (normalized_values[index], keys[index]),
    )
    square_mean = math.fsum(value * value for value in signed_values) / len(
        signed_values
    )
    return {
        "baselineCell": baseline_name,
        "componentCount": len(signed_values),
        "definition": f"{joint_name}-{second_name}-{first_name}+{baseline_name}",
        "firstEffectCell": first_name,
        "interactionName": name,
        "jointCell": joint_name,
        "l1AbsoluteInteraction": math.fsum(abs(value) for value in signed_values),
        "maximumAbsoluteInteraction": abs(signed_values[maximum_index]),
        "maximumAbsoluteWitness": witnesses[maximum_index],
        "maximumScaleNormalizedAbsoluteInteraction": normalized_values[
            maximum_normalized_index
        ],
        "maximumScaleNormalizedWitness": witnesses[maximum_normalized_index],
        "maximumInteractionToleranceUnits": maximum_tolerance_units,
        "rootMeanSquareInteraction": math.sqrt(square_mean),
        "secondEffectCell": second_name,
        "signedFateInteraction": signed_fates,
        "signedG2Interaction": signed_g2,
        "signedProperPowerInteraction": signed_proper_power,
        "toleranceUnitsBySourceColumn": tolerance_columns,
    }


def _library_artifact_from_document(document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        artifact = document["runtimeNumericBackend"]["descriptor"][
            "nativeWholeRay"
        ]["artifacts"]["library"]
    except (KeyError, TypeError) as error:
        raise MuPsiPhaseInteractionError(
            "native runtime descriptor lacks its dylib artifact"
        ) from error
    expected_keys = {"artifactName", "byteLength", "logicalPath", "sha256"}
    if type(artifact) is not dict or set(artifact) != expected_keys:
        raise MuPsiPhaseInteractionError("native dylib artifact shape is malformed")
    if artifact["logicalPath"] != "runtime/libblackhole_cpu.dylib":
        raise MuPsiPhaseInteractionError("native dylib artifact is not path-free")
    if type(artifact["byteLength"]) is not int or artifact["byteLength"] < 1:
        raise MuPsiPhaseInteractionError("native dylib artifact length is malformed")
    _require_sha256(artifact["sha256"], "native dylib artifact SHA-256")
    return deepcopy(artifact)


def _c_reproduction(
    parent: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    parent_summary = parent["summaries"][2]
    target_summary = target["summaries"][0]
    parent_physical = _physical_summary_document(dict(parent_summary.descriptor()))
    target_physical = _physical_summary_document(dict(target_summary.descriptor()))
    if canonical_json_bytes(parent_physical) != canonical_json_bytes(target_physical):
        raise MuPsiPhaseInteractionError(
            "target full pass does not exactly reproduce parent C physical summary"
        )
    parent_raw = parent["passEvidence"][2]["sampleAuditSha256"]
    target_raw = target["passEvidence"][0]["sampleAuditSha256"]
    if parent_raw != target_raw:
        raise MuPsiPhaseInteractionError(
            "target full pass does not reproduce parent C physical sample audit"
        )
    parent_evidence = _normalized_pass_evidence(parent["passEvidence"][2])
    target_evidence = _normalized_pass_evidence(target["passEvidence"][0])
    if canonical_json_bytes(parent_evidence) != canonical_json_bytes(target_evidence):
        raise MuPsiPhaseInteractionError(
            "target full pass does not reproduce parent C normalized pass evidence"
        )
    return {
        "exactPhysicalSummary": True,
        "excludedPassIdentityFields": [
            "summary.gridId",
            "passEvidence.passIndex",
            "passEvidence.passName",
            "passEvidence.maximumNormalizedSampleWeightWitness.passIndex",
            "passEvidence.maximumNormalizedSampleWeightWitness.passName",
        ],
        "normalizedPassEvidenceExact": True,
        "normalizedPassEvidenceSha256": _canonical_sha256(parent_evidence),
        "physicalSummarySha256": _canonical_sha256(parent_physical),
        "rawSampleAuditDigestEqualityRequired": True,
        "rawSampleAuditSha256": {
            "equal": parent_raw == target_raw,
            "parentC": parent_raw,
            "targetFull": target_raw,
        },
        "rawSampleAuditTreatment": (
            "production audit preimage contains physical node/transport/rho/source "
            "fields and excludes pass identity, so exact equality is a hard gate"
        ),
        "targetFullReproducesCell": "C",
    }


def _exact_node_document(node: EmittedFluxDirectionNode) -> dict[str, float]:
    if type(node) is not EmittedFluxDirectionNode:
        raise MuPsiPhaseInteractionError(
            "phase re-reduction received a foreign quadrature node"
        )
    return {
        "emissionAngleCosine": node.emission_angle_cosine,
        "normalizedEmittedFluxWeight": node.normalized_emitted_flux_weight,
        "tangentAzimuthRad": node.tangent_azimuth_rad,
    }


def _close_phase_half_weights(
    provisional: Sequence[float],
) -> tuple[tuple[float, ...], int, float]:
    """Apply the production angular-rule binary64 tail closure."""

    if type(provisional) not in (tuple, list) or not provisional:
        raise MuPsiPhaseInteractionError(
            "phase half-grid weights must be a non-empty exact sequence"
        )
    weights = list(provisional)
    if any(type(value) is not float or not math.isfinite(value) or value <= 0.0 for value in weights):
        raise MuPsiPhaseInteractionError(
            "phase half-grid weights must be finite positive binary64 values"
        )
    initial_total = math.fsum(weights)
    iterations = 0
    for _index in range(16):
        current_total = math.fsum(weights)
        if current_total == 1.0:
            break
        candidate = weights[-1] + (1.0 - current_total)
        if candidate == weights[-1]:
            candidate = math.nextafter(
                weights[-1],
                math.inf if current_total < 1.0 else -math.inf,
            )
        weights[-1] = candidate
        iterations += 1
    if math.fsum(weights).hex() != 1.0.hex() or any(value <= 0.0 for value in weights):
        raise MuPsiPhaseInteractionError(
            "phase half-grid weights cannot be closed exactly"
        )
    return tuple(weights), iterations, initial_total


def _phase_half_grid_rules() -> tuple[
    tuple[EmittedFluxDirectionNode, ...],
    tuple[EmittedFluxDirectionNode, ...],
    tuple[EmittedFluxDirectionNode, ...],
    dict[str, Any],
]:
    """Prove the phase64 even/odd rules reproduce canonical psi32 rules."""

    phase64 = _PHASE_NODE_CALL_ENTRY(
        32,
        64,
        phase_cells=0.5,
    )
    even_expected = _PHASE_NODE_CALL_ENTRY(
        32,
        32,
        phase_cells=0.0,
    )
    odd_expected = _PHASE_NODE_CALL_ENTRY(
        32,
        32,
        phase_cells=0.5,
    )
    parity_results: list[tuple[tuple[float, ...], int, float]] = []
    for parity in (0, 1):
        provisional = tuple(
            2.0
            * phase64[mu_index * 64 + 2 * psi_index + parity]
            .normalized_emitted_flux_weight
            for mu_index in range(32)
            for psi_index in range(32)
        )
        parity_results.append(_close_phase_half_weights(provisional))
    for parity, expected in enumerate((even_expected, odd_expected)):
        closed_weights = parity_results[parity][0]
        for mu_index in range(32):
            for psi_index in range(32):
                full = phase64[mu_index * 64 + 2 * psi_index + parity]
                half = expected[mu_index * 32 + psi_index]
                closed = closed_weights[mu_index * 32 + psi_index]
                if (
                    full.emission_angle_cosine.hex()
                    != half.emission_angle_cosine.hex()
                    or full.tangent_azimuth_rad.hex()
                    != half.tangent_azimuth_rad.hex()
                    or closed.hex() != half.normalized_emitted_flux_weight.hex()
                ):
                    raise MuPsiPhaseInteractionError(
                        "phase64 parity reclosure does not reproduce canonical psi32"
                    )
    evidence = {
        "even": {
            "derivedPhaseCells": 0.0,
            "initialDoubledWeightSum": parity_results[0][2],
            "tailClosureIterations": parity_results[0][1],
        },
        "odd": {
            "derivedPhaseCells": 0.5,
            "initialDoubledWeightSum": parity_results[1][2],
            "tailClosureIterations": parity_results[1][1],
        },
        "rule": (
            "select phase64 parity, multiply each angular weight by two, then "
            "apply the production binary64 final-weight closure"
        ),
    }
    return phase64, even_expected, odd_expected, evidence


def _normalized_phase_cache_record(
    record: Mapping[str, Any],
    *,
    source_pass: str,
    mapped_psi_index: int,
    source_node: EmittedFluxDirectionNode,
    derived_node: EmittedFluxDirectionNode,
) -> tuple[tuple[int, int, int, int, int], bytes]:
    """Remove pass/ordinal identity and bind one exact derived psi32 input."""

    if type(record) is not dict or set(record) != {"coordinate", "transport"}:
        raise MuPsiPhaseInteractionError("phase cache record shape is malformed")
    coordinate = record["coordinate"]
    wrapper = record["transport"]
    if type(coordinate) is not dict or type(wrapper) is not dict:
        raise MuPsiPhaseInteractionError("phase cache record is malformed")
    sample = wrapper.get("sample")
    if type(sample) is not dict or type(wrapper.get("transport")) is not dict:
        raise MuPsiPhaseInteractionError("phase cache transport is malformed")
    expected_index = 4 if source_pass == "B" else 3
    expected_name = "phase-shifted" if source_pass == "B" else "half-psi"
    if (
        coordinate.get("passIndex") != expected_index
        or coordinate.get("passName") != expected_name
        or sample.get("passIndex") != expected_index
        or sample.get("passName") != expected_name
        or sample.get("muIndex") != coordinate.get("muIndex")
        or sample.get("psiIndex") != coordinate.get("psiIndex")
    ):
        raise MuPsiPhaseInteractionError(
            f"phase cache {source_pass} record has stale pass identity"
        )
    expected_source = _exact_node_document(source_node)
    actual_source = {
        key: sample.get(key)
        for key in (
            "emissionAngleCosine",
            "normalizedEmittedFluxWeight",
            "tangentAzimuthRad",
        )
    }
    if canonical_json_bytes(actual_source) != canonical_json_bytes(expected_source):
        raise MuPsiPhaseInteractionError(
            f"phase cache {source_pass} sample differs from its canonical angular node"
        )
    key = (
        coordinate["faceIndex"],
        coordinate["annulusIndex"],
        coordinate["rhoIndex"],
        coordinate["muIndex"],
        mapped_psi_index,
    )
    normalized = {
        "coordinate": {
            "annulusIndex": coordinate["annulusIndex"],
            "face": coordinate["face"],
            "faceIndex": coordinate["faceIndex"],
            "muIndex": coordinate["muIndex"],
            "psiIndex": mapped_psi_index,
            "rhoIndex": coordinate["rhoIndex"],
        },
        "sample": {
            "emissionAngleCosine": derived_node.emission_angle_cosine,
            "muIndex": sample["muIndex"],
            "normalizedEmittedFluxWeight": (
                derived_node.normalized_emitted_flux_weight
            ),
            "psiIndex": mapped_psi_index,
            "rhoIndex": sample["rhoIndex"],
            "sourceAnnulusIndex": sample["sourceAnnulusIndex"],
            "sourceFace": sample["sourceFace"],
            "sourceRadiusOverMass": sample["sourceRadiusOverMass"],
            "tangentAzimuthRad": derived_node.tangent_azimuth_rad,
        },
        "transport": {
            "formulation": wrapper["formulation"],
            "schema": wrapper["schema"],
            "transport": deepcopy(wrapper["transport"]),
        },
    }
    return key, canonical_json_bytes(normalized)


def _cache_trust_from_parent_manifest(
    document: Mapping[str, Any],
    *,
    expected_payload_set_sha256: str,
    expected_direction_stream_sha256: str,
) -> tuple[_golden_cache.CacheTrust, dict[str, Any]]:
    try:
        definition = document["producer"]["cacheDefinition"]
        job_spec = definition["jobSpec"]
        cache_job_key = definition["cacheJobKey"]
        scientific_job_key = definition["scientificJobKey"]
        scientific_plan_sha256 = definition["scientificPlanSha256"]
        plan = definition["scientificPlan"]
        task_count = document["operationalExecution"]["taskCount"]
    except (KeyError, TypeError) as error:
        raise MuPsiPhaseInteractionError(
            "parent manifest lacks cache trust evidence"
        ) from error
    job_document = {"jobKey": cache_job_key, "spec": job_spec}
    try:
        trust = _golden_cache.CacheTrust(
            cache_job_key=cache_job_key,
            scientific_job_key=scientific_job_key,
            scientific_plan_sha256=scientific_plan_sha256,
            job_document_sha256=_sha256_bytes(canonical_json_bytes(job_document)),
            authenticated_task_count=task_count,
            authenticated_direction_count=plan["directionCount"],
            authenticated_payload_set_sha256=_require_sha256(
                expected_payload_set_sha256,
                "external parent cache payload-set SHA-256",
            ),
            authenticated_direction_stream_sha256=_require_sha256(
                expected_direction_stream_sha256,
                "external parent cache direction-stream SHA-256",
            ),
        )
    except (KeyError, TypeError, ValueError, _golden_cache.GoldenCacheError) as error:
        raise MuPsiPhaseInteractionError(
            "parent manifest contains invalid cache trust evidence"
        ) from error
    return trust, job_document


def _assert_golden_cache_bindings() -> None:
    bindings = (
        (_golden_cache._secure_job_directory, _GOLDEN_SECURE_JOB_DIRECTORY),
        (_golden_cache._read_regular_file_at, _GOLDEN_READ_REGULAR_FILE_AT),
        (_golden_cache._validated_job_document, _GOLDEN_VALIDATE_JOB_DOCUMENT),
        (_golden_cache._stable_directory_names, _GOLDEN_STABLE_DIRECTORY_NAMES),
        (_golden_cache._task_stem, _GOLDEN_TASK_STEM),
        (_golden_cache._require_regular_entry, _GOLDEN_REQUIRE_REGULAR_ENTRY),
        (_golden_cache._validated_receipt, _GOLDEN_VALIDATE_RECEIPT),
        (
            _golden_cache._validated_payload_records,
            _GOLDEN_VALIDATE_PAYLOAD_RECORDS,
        ),
        (_golden_cache._stable_stat_identity, _GOLDEN_STABLE_STAT_IDENTITY),
    )
    if any(current is not frozen for current, frozen in bindings):
        raise MuPsiPhaseInteractionError(
            "read-only golden-cache authenticator binding changed"
        )


def _phase_cache_replay(
    parent_document: Mapping[str, Any],
    parent_cache_job_path: Path,
    *,
    expected_payload_set_sha256: str,
    expected_direction_stream_sha256: str,
) -> dict[str, Any]:
    """Read-only authenticate B/E records and prove B-even reproduces E."""

    _assert_golden_cache_bindings()
    trust, job_document = _cache_trust_from_parent_manifest(
        parent_document,
        expected_payload_set_sha256=expected_payload_set_sha256,
        expected_direction_stream_sha256=expected_direction_stream_sha256,
    )
    phase64_nodes, even_nodes, odd_nodes, rule_evidence = _phase_half_grid_rules()
    e_records: dict[tuple[int, int, int, int, int], bytes] = {}
    e_stream = hashlib.sha256()
    even_stream = hashlib.sha256()
    odd_stream = hashlib.sha256()
    full_stream = hashlib.sha256()
    payload_inventory: list[dict[str, Any]] = []
    authenticated_records = 0
    authenticated_tasks = 0
    even_count = 0
    odd_count = 0

    try:
        with _GOLDEN_SECURE_JOB_DIRECTORY(parent_cache_job_path, trust) as session:
            job_payload = _GOLDEN_READ_REGULAR_FILE_AT(
                session.job_fd,
                "job.json",
                maximum_bytes=_golden_cache._MAXIMUM_JOB_BYTES,
                label="parent cache job document",
            )
            if job_payload != canonical_json_bytes(job_document):
                raise MuPsiPhaseInteractionError(
                    "parent cache job document differs from its manifest"
                )
            definition = _GOLDEN_VALIDATE_JOB_DOCUMENT(job_payload, trust)
            before = os.fstat(session.task_fd)
            names = _GOLDEN_STABLE_DIRECTORY_NAMES(
                session.task_fd,
                "parent cache task directory",
            )
            name_set = set(names)
            expected_stems = {_GOLDEN_TASK_STEM(task) for task in definition.tasks}
            payload_names = {f"{stem}.bin" for stem in expected_stems}
            receipt_names = {f"{stem}.receipt.json" for stem in expected_stems}
            lock_names = {f"{stem}.lock" for stem in expected_stems}
            unexpected = sorted(name_set - payload_names - receipt_names - lock_names)
            if unexpected:
                raise MuPsiPhaseInteractionError(
                    f"parent cache has unexpected entry {unexpected[0]!r}"
                )
            for name in names:
                _GOLDEN_REQUIRE_REGULAR_ENTRY(
                    session.task_fd,
                    name,
                    f"parent cache entry {name!r}",
                )
            for task in definition.tasks:
                stem = _GOLDEN_TASK_STEM(task)
                payload_name = f"{stem}.bin"
                receipt_name = f"{stem}.receipt.json"
                if payload_name not in name_set or receipt_name not in name_set:
                    raise MuPsiPhaseInteractionError(
                        "parent cache is incomplete; B-even/E proof requires every task"
                    )
                payload = _GOLDEN_READ_REGULAR_FILE_AT(
                    session.task_fd,
                    payload_name,
                    maximum_bytes=_golden_cache._MAXIMUM_TASK_PAYLOAD_BYTES,
                    label=f"parent cache payload {payload_name}",
                )
                receipt = _GOLDEN_READ_REGULAR_FILE_AT(
                    session.task_fd,
                    receipt_name,
                    maximum_bytes=_golden_cache._MAXIMUM_RECEIPT_BYTES,
                    label=f"parent cache receipt {receipt_name}",
                )
                payload_sha, _receipt_length = _GOLDEN_VALIDATE_RECEIPT(
                    payload,
                    receipt,
                    task=task,
                    payload_name=payload_name,
                    trust=trust,
                )
                records = _GOLDEN_VALIDATE_PAYLOAD_RECORDS(
                    payload,
                    definition=definition,
                    task=task,
                    trust=trust,
                )
                payload_inventory.append(
                    {
                        "byteLength": len(payload),
                        "name": payload_name,
                        "recordCount": len(records),
                        "sha256": payload_sha,
                    }
                )
                authenticated_tasks += 1
                authenticated_records += len(records)
                for record in records:
                    full_stream.update(canonical_json_bytes(record))
                    coordinate = record["coordinate"]
                    pass_index = coordinate["passIndex"]
                    mu_index = coordinate["muIndex"]
                    psi_index = coordinate["psiIndex"]
                    if pass_index == 3:
                        source_node = even_nodes[mu_index * 32 + psi_index]
                        key, normalized = _normalized_phase_cache_record(
                            record,
                            source_pass="E",
                            mapped_psi_index=psi_index,
                            source_node=source_node,
                            derived_node=source_node,
                        )
                        if key in e_records:
                            raise MuPsiPhaseInteractionError(
                                "parent cache duplicates one E reduction input"
                            )
                        e_records[key] = normalized
                        e_stream.update(normalized)
                    elif pass_index == 4:
                        parity = psi_index & 1
                        mapped = psi_index // 2
                        source_node = phase64_nodes[mu_index * 64 + psi_index]
                        derived_node = (even_nodes, odd_nodes)[parity][
                            mu_index * 32 + mapped
                        ]
                        key, normalized = _normalized_phase_cache_record(
                            record,
                            source_pass="B",
                            mapped_psi_index=mapped,
                            source_node=source_node,
                            derived_node=derived_node,
                        )
                        if parity == 0:
                            expected = e_records.pop(key, None)
                            if expected is None or normalized != expected:
                                raise MuPsiPhaseInteractionError(
                                    "B-even does not exactly reproduce E reduction input"
                                )
                            even_stream.update(normalized)
                            even_count += 1
                        else:
                            odd_stream.update(normalized)
                            odd_count += 1
            after = os.fstat(session.task_fd)
            if _GOLDEN_STABLE_STAT_IDENTITY(before) != _GOLDEN_STABLE_STAT_IDENTITY(after):
                raise MuPsiPhaseInteractionError(
                    "parent cache task directory changed during read-only replay"
                )
    except _golden_cache.GoldenCacheError as error:
        raise MuPsiPhaseInteractionError(
            "parent cache failed read-only authentication"
        ) from error

    expected_half_count = (
        2
        * int(parent_document["producer"]["cacheDefinition"]["scientificPlan"][
            "annulusCount"
        ])
        * 16
        * 32
        * 32
    )
    payload_set_sha256 = _canonical_sha256(payload_inventory)
    direction_stream_sha256 = full_stream.hexdigest()
    if (
        e_records
        or even_count != expected_half_count
        or odd_count != expected_half_count
        or even_stream.hexdigest() != e_stream.hexdigest()
        or authenticated_tasks != trust.authenticated_task_count
        or authenticated_records != trust.authenticated_direction_count
        or payload_set_sha256 != trust.authenticated_payload_set_sha256
        or direction_stream_sha256 != trust.authenticated_direction_stream_sha256
    ):
        raise MuPsiPhaseInteractionError(
            "parent cache B/E record coverage is incomplete or non-canonical"
        )
    e_physical_summary = _physical_summary_document(
        parent_document["authenticatedConvergenceV2"]["summaries"][3][
            "descriptor"
        ]
    )
    return {
        "bEven": {
            "derivedPhaseCells": 0.0,
            "derivedPsiCount": 32,
            "normalizedDirectionStreamSha256": even_stream.hexdigest(),
            "recordCount": even_count,
            "reproducesCell": "E",
            "reproducesEExactly": True,
        },
        "bOdd": {
            "derivedPhaseCells": 0.5,
            "derivedPsiCount": 32,
            "hasProductionFivePassComparator": False,
            "normalizedDirectionStreamSha256": odd_stream.hexdigest(),
            "recordCount": odd_count,
            "recordOnlyCell": "G",
        },
        "cacheAuthentication": {
            "authenticatedDirectionCount": authenticated_records,
            "authenticatedDirectionStreamSha256": direction_stream_sha256,
            "authenticatedPayloadSetSha256": payload_set_sha256,
            "authenticatedTaskCount": authenticated_tasks,
            "cacheJobKey": trust.cache_job_key,
            "externallyAnchoredSnapshot": True,
            "jobDocumentSha256": trust.job_document_sha256,
            "readOnly": True,
        },
        "e": {
            "manifestPhysicalSummarySha256": _canonical_sha256(e_physical_summary),
            "normalizedDirectionStreamSha256": e_stream.hexdigest(),
            "recordCount": expected_half_count,
        },
        "performed": True,
        "productionFivePassContractModified": False,
        "qualificationEffect": "none",
        "quadratureReclosure": rule_evidence,
    }


def _analysis_body(
    parent_document: Mapping[str, Any],
    target_document: Mapping[str, Any],
    *,
    parent_manifest_sha256: str,
    target_manifest_sha256: str,
    dylib_artifact: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    cache_replay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pure manifest-level analyzer used by the no-ray unit tests."""

    parent_sha = _require_sha256(
        parent_manifest_sha256,
        "parent external manifest SHA-256",
    )
    target_sha = _require_sha256(
        target_manifest_sha256,
        "target external manifest SHA-256",
    )
    parent = _checkpoint_parts(parent_document, "parent")
    target = _checkpoint_parts(target_document, "target")
    _require_design(parent, target)
    parent_library = _library_artifact_from_document(parent_document)
    target_library = _library_artifact_from_document(target_document)
    if parent_library != target_library or parent_library != dylib_artifact:
        raise MuPsiPhaseInteractionError(
            "parent, target, and selected dylib artifacts are not identical"
        )
    reproduction = _c_reproduction(parent, target)

    summaries = {
        "A": parent["summaries"][0],
        "B": parent["summaries"][4],
        "C": parent["summaries"][2],
        "D": target["summaries"][4],
        "E": parent["summaries"][3],
        "F": target["summaries"][3],
    }
    evidence = {
        "A": parent["passEvidence"][0],
        "B": parent["passEvidence"][4],
        "C": parent["passEvidence"][2],
        "D": target["passEvidence"][4],
        "E": parent["passEvidence"][3],
        "F": target["passEvidence"][3],
    }
    cells = {
        "A": _cell_document("A", "parent", 0, summaries["A"], evidence["A"], parent["plan"]),
        "B": _cell_document("B", "parent", 4, summaries["B"], evidence["B"], parent["plan"]),
        "C": _cell_document("C", "parent", 2, summaries["C"], evidence["C"], parent["plan"]),
        "D": _cell_document("D", "target", 4, summaries["D"], evidence["D"], target["plan"]),
        "E": _cell_document("E", "parent", 3, summaries["E"], evidence["E"], parent["plan"]),
        "F": _cell_document("F", "target", 3, summaries["F"], evidence["F"], target["plan"]),
    }
    edges = {
        "muAtPhaseHalf": _edge_document("B", summaries["B"], "D", summaries["D"]),
        "muAtPhaseZero": _edge_document("A", summaries["A"], "C", summaries["C"]),
        "muAtPsiHalf": _edge_document("E", summaries["E"], "F", summaries["F"]),
        "phaseAtMu16": _edge_document("C", summaries["C"], "D", summaries["D"]),
        "phaseAtMu32": _edge_document("A", summaries["A"], "B", summaries["B"]),
        "psiAtMu16": _edge_document("C", summaries["C"], "F", summaries["F"]),
        "psiAtMu32": _edge_document("A", summaries["A"], "E", summaries["E"]),
    }
    interactions = {
        "muByPhase": _interaction_document(
            "mu-by-phase",
            "A",
            "B",
            "C",
            "D",
            summaries["A"],
            summaries["B"],
            summaries["C"],
            summaries["D"],
        ),
        "muByPsi": _interaction_document(
            "mu-by-psi",
            "A",
            "E",
            "C",
            "F",
            summaries["A"],
            summaries["E"],
            summaries["C"],
            summaries["F"],
        ),
    }
    return {
        "automaticEscalation": False,
        "cacheReplay": (
            {
                "performed": False,
                "reason": (
                    "no parent cache job path was selected; manifest-level edge "
                    "and interaction diagnostics remain available"
                ),
            }
            if cache_replay is None
            else deepcopy(cache_replay)
        ),
        "cells": cells,
        "classification": CLASSIFICATION,
        "coverage": {
            "identifiedEdges": sorted(edges),
            "identifiedInteractions": ["muByPhase", "muByPsi"],
            "missingSummaryCells": [
                "mu32-psi32-phase0.5",
                "mu16-psi32-phase0.5",
            ],
            "muByPsiByPhaseIdentified": False,
            "psiByPhaseIdentified": False,
            "recordOnlyDerivedCells": (
                [] if cache_replay is None else ["G"]
            ),
            "reason": (
                "the production five-pass contract has no phase-shifted half-psi "
                "passes; this analyzer does not change that contract"
            ),
        },
        "edgeDiagnostics": edges,
        "implementationId": IMPLEMENTATION_ID,
        "inputs": {
            "nativeLibraryArtifact": deepcopy(dylib_artifact),
            "parent": {
                "checkpointId": parent_document.get("id"),
                "externallyAuthenticatedManifestSha256": parent_sha,
                "qualified": False,
            },
            "sameExplicitDylibForBothPublicVerifiers": True,
            "target": {
                "checkpointId": target_document.get("id"),
                "externallyAuthenticatedManifestSha256": target_sha,
                "qualified": target_document["qualified"],
            },
        },
        "interactionDiagnostics": interactions,
        "manifestLevelOnly": cache_replay is None,
        "productEligible": False,
        "productionFivePassContractModified": False,
        "productionQualified": False,
        "qualified": False,
        "reproductionGate": reproduction,
        "schema": SCHEMA,
        "scientificBoundary": {
            "containsReadOnlyCacheRecordSubstitution": cache_replay is not None,
            "containsRayEvaluation": False,
            "isIndependentPhysicsOracle": False,
            "isQualificationEvidence": False,
            "isReadOnlyEvidenceDiagnostic": True,
            "mayAuthorizeProduct": False,
            "mayEscalateAutomatically": False,
        },
        "sourceBinding": deepcopy(source_snapshot),
    }


def _complete_report(body: Mapping[str, Any]) -> dict[str, Any]:
    body_document = json.loads(canonical_json_bytes(body))
    body_sha = _canonical_sha256(body_document)
    return {
        **body_document,
        "id": f"kerr-mu-psi-phase-interaction-{body_sha[:24]}",
        "integrity": {"analysisBodySha256": body_sha},
    }


def _selected_dylib_artifact(
    dylib_path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _stable_regular_file(dylib_path, "native CPU dylib", MAXIMUM_DYLIB_BYTES)
    selected = {
        "artifactName": "libblackhole_cpu.dylib",
        "byteLength": len(payload),
        "logicalPath": "runtime/libblackhole_cpu.dylib",
        "sha256": _sha256_bytes(payload),
    }
    if selected != expected:
        raise MuPsiPhaseInteractionError(
            "selected native CPU dylib differs from checkpoint runtime artifact"
        )
    return selected


def _verify_checkpoint(
    manifest_path: Path,
    expected_sha256: str,
    dylib_path: Path,
) -> KerrReturningRadiationVerifiedRefinementCheckpoint:
    try:
        result = _VERIFY_CALL_ENTRY(
            manifest_path,
            expected_manifest_sha256=expected_sha256,
            native_library_path=dylib_path,
        )
    except Exception as error:
        raise MuPsiPhaseInteractionError(
            f"public checkpoint verifier rejected {manifest_path}"
        ) from error
    if type(result) is not KerrReturningRadiationVerifiedRefinementCheckpoint:
        raise MuPsiPhaseInteractionError(
            "public checkpoint verifier returned a foreign result type"
        )
    if result.manifest_sha256 != expected_sha256:
        raise MuPsiPhaseInteractionError(
            "public checkpoint verifier returned the wrong manifest digest"
        )
    if (
        result.manifest_path != manifest_path
        or result.output_directory != manifest_path.parent
        or type(result.document) is not dict
        or result.document.get("id") != result.checkpoint_id
        or type(result.document.get("qualified")) is not bool
        or result.document["qualified"] is not result.qualified
    ):
        raise MuPsiPhaseInteractionError(
            "public checkpoint verifier returned inconsistent checkpoint evidence"
        )
    return result


def _require_frozen_module_function_bindings() -> None:
    current = _MODULE_FUNCTION_BINDINGS_CALL_ENTRY()
    frozen = _FROZEN_MODULE_FUNCTION_BINDINGS
    if len(current) != len(frozen) or any(
        current_name != frozen_name or current_function is not frozen_function
        for (current_name, current_function), (frozen_name, frozen_function) in zip(
            current,
            frozen,
        )
    ):
        raise MuPsiPhaseInteractionError(
            "analyzer module helper binding changed after import"
        )


def _secure_output_parent(
    path: Path,
) -> tuple[Path, int, tuple[tuple[int, int], ...]]:
    if type(path) is not _PATH_TYPE or not path.is_absolute():
        raise MuPsiPhaseInteractionError(
            "output must be an exact absolute platform Path"
        )
    absolute = Path(os.path.abspath(os.fspath(path)))
    if path != absolute or not path.name or path.name in (".", ".."):
        raise MuPsiPhaseInteractionError("output path is non-canonical")
    if path.suffix != REPORT_NAME_SUFFIX:
        raise MuPsiPhaseInteractionError("output must use a .json suffix")
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors = [os.open(os.sep, flags)]
    try:
        for component in path.parent.parts[1:]:
            descriptors.append(os.open(component, flags, dir_fd=descriptors[-1]))
        parent = descriptors[-1]
        identities = tuple(
            (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
            for descriptor in descriptors
        )
        for descriptor in descriptors[:-1]:
            os.close(descriptor)
        return absolute, parent, identities
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _secure_directory_identity(path: Path, label: str) -> tuple[int, int]:
    if type(path) is not _PATH_TYPE or not path.is_absolute():
        raise MuPsiPhaseInteractionError(
            f"{label} must be an exact absolute platform Path"
        )
    absolute = Path(os.path.abspath(os.fspath(path)))
    if path != absolute or path == path.parent:
        raise MuPsiPhaseInteractionError(f"{label} path is non-canonical")
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors = [os.open(os.sep, flags)]
    try:
        for component in path.parts[1:]:
            descriptors.append(os.open(component, flags, dir_fd=descriptors[-1]))
        status = os.fstat(descriptors[-1])
        return status.st_dev, status.st_ino
    except OSError as error:
        raise MuPsiPhaseInteractionError(
            f"{label} is not a secure existing directory"
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _publish_new_report(
    path: Path,
    payload: bytes,
    *,
    forbidden_directory_identities: frozenset[tuple[int, int]] = frozenset(),
) -> None:
    if len(payload) < 1 or len(payload) > MAXIMUM_REPORT_BYTES:
        raise MuPsiPhaseInteractionError("analysis report exceeds its byte bound")
    if type(forbidden_directory_identities) is not frozenset:
        raise TypeError("forbidden_directory_identities must be an exact frozenset")
    absolute, parent_fd, ancestry = _secure_output_parent(path)
    if any(identity in forbidden_directory_identities for identity in ancestry):
        os.close(parent_fd)
        raise MuPsiPhaseInteractionError(
            "analysis output must not be inside an input checkpoint/cache directory"
        )
    temporary = f".{absolute.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    descriptor = -1
    linked = False
    validated = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written < 1:
                raise MuPsiPhaseInteractionError(
                    "analysis report write made no progress"
                )
            offset += written
        os.fsync(descriptor)
        temporary_status = os.fstat(descriptor)
        os.link(
            temporary,
            absolute.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        linked = True
        os.fsync(parent_fd)
        reopened_absolute, reopened_parent_fd, reopened_ancestry = (
            _secure_output_parent(path)
        )
        final_descriptor = -1
        try:
            if (
                reopened_absolute != absolute
                or any(
                    identity in forbidden_directory_identities
                    for identity in reopened_ancestry
                )
            ):
                raise MuPsiPhaseInteractionError(
                    "analysis output parent changed into an input directory"
                )
            original_parent_status = os.fstat(parent_fd)
            reopened_parent_status = os.fstat(reopened_parent_fd)
            if (
                original_parent_status.st_dev,
                original_parent_status.st_ino,
            ) != (
                reopened_parent_status.st_dev,
                reopened_parent_status.st_ino,
            ):
                raise MuPsiPhaseInteractionError(
                    "analysis output parent path identity changed during publication"
                )
            final_descriptor = os.open(
                absolute.name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=reopened_parent_fd,
            )
            final_status = os.fstat(final_descriptor)
            if (
                not stat.S_ISREG(final_status.st_mode)
                or (
                    final_status.st_dev,
                    final_status.st_ino,
                    final_status.st_size,
                )
                != (
                    temporary_status.st_dev,
                    temporary_status.st_ino,
                    len(payload),
                )
            ):
                raise MuPsiPhaseInteractionError(
                    "analysis output file identity differs from its fsynced temporary"
                )
            chunks: list[bytes] = []
            remaining = len(payload)
            while remaining:
                chunk = os.read(final_descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise MuPsiPhaseInteractionError(
                        "analysis output ended during final verification"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(final_descriptor, 1) or b"".join(chunks) != payload:
                raise MuPsiPhaseInteractionError(
                    "analysis output bytes differ from the canonical report"
                )
            validated = True
        finally:
            if final_descriptor >= 0:
                os.close(final_descriptor)
            os.close(reopened_parent_fd)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite analysis report {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if linked and not validated:
            try:
                os.unlink(absolute.name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        if linked:
            os.fsync(parent_fd)
        os.close(parent_fd)
    if not linked or not validated:
        raise MuPsiPhaseInteractionError("analysis report was not published")


def analyze_mu_psi_phase_interaction(
    parent_manifest_path: Path,
    parent_manifest_sha256: str,
    target_manifest_path: Path,
    target_manifest_sha256: str,
    native_library_path: Path,
    output_path: Path,
    parent_cache_job_path: Path | None = None,
    parent_cache_payload_set_sha256: str | None = None,
    parent_cache_direction_stream_sha256: str | None = None,
) -> dict[str, Any]:
    """Authenticate two manifests and atomically publish one new report."""

    if (
        verify_kerr_returning_radiation_refinement_checkpoint
        is not _VERIFY_PUBLIC_ENTRY
        or _VERIFY_CALL_ENTRY is not _VERIFY_PUBLIC_ENTRY
        or kerrbb_d20_emitted_flux_direction_nodes is not _PHASE_NODE_PUBLIC_ENTRY
        or _PHASE_NODE_CALL_ENTRY is not _PHASE_NODE_PUBLIC_ENTRY
        or compare_kerr_returning_radiation_grids_v2 is not _COMPARE_PUBLIC_ENTRY
        or _COMPARE_CALL_ENTRY is not _COMPARE_PUBLIC_ENTRY
        or canonical_json_bytes is not _CANONICAL_JSON_PUBLIC_ENTRY
        or _source_snapshot is not _SOURCE_SNAPSHOT_PUBLIC_ENTRY
        or _SOURCE_SNAPSHOT_CALL_ENTRY is not _SOURCE_SNAPSHOT_PUBLIC_ENTRY
        or _callable_snapshot is not _CALLABLE_SNAPSHOT_PUBLIC_ENTRY
        or _CALLABLE_SNAPSHOT_CALL_ENTRY is not _CALLABLE_SNAPSHOT_PUBLIC_ENTRY
        or _module_function_bindings is not _MODULE_FUNCTION_BINDINGS_PUBLIC_ENTRY
        or _MODULE_FUNCTION_BINDINGS_CALL_ENTRY
        is not _MODULE_FUNCTION_BINDINGS_PUBLIC_ENTRY
        or _require_frozen_module_function_bindings
        is not _MODULE_BINDING_GUARD_PUBLIC_ENTRY
        or _MODULE_BINDING_GUARD_CALL_ENTRY
        is not _MODULE_BINDING_GUARD_PUBLIC_ENTRY
        or _idle_snapshot_verifier_identity
        is not _IDLE_SNAPSHOT_VERIFIER_PUBLIC_ENTRY
        or _IDLE_SNAPSHOT_VERIFIER_CALL_ENTRY
        is not _IDLE_SNAPSHOT_VERIFIER_PUBLIC_ENTRY
    ):
        raise MuPsiPhaseInteractionError(
            "public checkpoint/quadrature binding changed"
        )
    _MODULE_BINDING_GUARD_CALL_ENTRY()
    if __name__ != "__main__":
        raise MuPsiPhaseInteractionError(
            "report publication is CLI-only in a fresh analyzer process"
        )
    parent_sha = _require_sha256(parent_manifest_sha256, "parent manifest SHA-256")
    target_sha = _require_sha256(target_manifest_sha256, "target manifest SHA-256")
    source_before = _SOURCE_SNAPSHOT_CALL_ENTRY()
    callable_before = _CALLABLE_SNAPSHOT_CALL_ENTRY()
    snapshot_verifier_before = _IDLE_SNAPSHOT_VERIFIER_CALL_ENTRY()
    if (
        source_before != _FROZEN_SOURCE_SNAPSHOT
        or callable_before != _FROZEN_CALLABLE_SNAPSHOT
        or snapshot_verifier_before != _FROZEN_SNAPSHOT_VERIFIER_IDENTITY
    ):
        raise MuPsiPhaseInteractionError(
            "analyzer loaded-code/source identity changed after import"
        )
    parent = _verify_checkpoint(parent_manifest_path, parent_sha, native_library_path)
    target = _verify_checkpoint(target_manifest_path, target_sha, native_library_path)
    parent_directory_identity = _secure_directory_identity(
        parent.output_directory,
        "parent checkpoint output",
    )
    target_directory_identity = _secure_directory_identity(
        target.output_directory,
        "target checkpoint output",
    )
    forbidden_directory_identities = {
        parent_directory_identity,
        target_directory_identity,
    }
    expected_library = _library_artifact_from_document(parent.document)
    selected_library = _selected_dylib_artifact(native_library_path, expected_library)
    cache_replay = None
    cache_directory_identity = None
    if parent_cache_job_path is not None:
        if type(parent_cache_job_path) is not _PATH_TYPE:
            raise TypeError("parent_cache_job_path must be None or an exact Path")
        cache_absolute = Path(os.path.abspath(os.fspath(parent_cache_job_path)))
        output_absolute = Path(os.path.abspath(os.fspath(output_path)))
        if cache_absolute == output_absolute or cache_absolute in output_absolute.parents:
            raise MuPsiPhaseInteractionError(
                "analysis output must not be written inside the parent cache job"
            )
        cache_directory_identity = _secure_directory_identity(
            parent_cache_job_path,
            "parent cache job",
        )
        forbidden_directory_identities.add(cache_directory_identity)
        if (
            parent_cache_payload_set_sha256 is None
            or parent_cache_direction_stream_sha256 is None
        ):
            raise MuPsiPhaseInteractionError(
                "parent cache replay requires external payload-set and "
                "direction-stream SHA-256 anchors"
            )
    elif (
        parent_cache_payload_set_sha256 is not None
        or parent_cache_direction_stream_sha256 is not None
    ):
        raise MuPsiPhaseInteractionError(
            "parent cache snapshot anchors require --parent-cache-job"
        )
    _preflight_path, preflight_fd, output_ancestry = _secure_output_parent(output_path)
    os.close(preflight_fd)
    if any(
        identity in forbidden_directory_identities for identity in output_ancestry
    ):
        raise MuPsiPhaseInteractionError(
            "analysis output must not be inside an input checkpoint/cache directory"
        )
    if parent_cache_job_path is not None:
        assert parent_cache_payload_set_sha256 is not None
        assert parent_cache_direction_stream_sha256 is not None
        cache_replay = _phase_cache_replay(
            parent.document,
            parent_cache_job_path,
            expected_payload_set_sha256=parent_cache_payload_set_sha256,
            expected_direction_stream_sha256=(
                parent_cache_direction_stream_sha256
            ),
        )
    body = _analysis_body(
        parent.document,
        target.document,
        parent_manifest_sha256=parent_sha,
        target_manifest_sha256=target_sha,
        dylib_artifact=selected_library,
        source_snapshot={
            "loadedCallables": callable_before,
            "snapshotVerifier": snapshot_verifier_before,
            "sourceClosure": source_before,
        },
        cache_replay=cache_replay,
    )
    _MODULE_BINDING_GUARD_CALL_ENTRY()
    source_after = _SOURCE_SNAPSHOT_CALL_ENTRY()
    callable_after = _CALLABLE_SNAPSHOT_CALL_ENTRY()
    snapshot_verifier_after = _IDLE_SNAPSHOT_VERIFIER_CALL_ENTRY()
    if (
        source_after != source_before
        or callable_after != callable_before
        or snapshot_verifier_after != snapshot_verifier_before
    ):
        raise MuPsiPhaseInteractionError(
            "analyzer source/loaded callables changed during analysis"
        )
    # Close both manifest and runtime races before publication.  These calls do
    # not start cache work; the public verifier reads only each closed manifest,
    # current source, and the explicitly selected dylib.
    closing_parent = _verify_checkpoint(
        parent_manifest_path,
        parent_sha,
        native_library_path,
    )
    closing_target = _verify_checkpoint(
        target_manifest_path,
        target_sha,
        native_library_path,
    )
    if (
        closing_parent.document != parent.document
        or closing_target.document != target.document
        or _selected_dylib_artifact(native_library_path, expected_library)
        != selected_library
        or _SOURCE_SNAPSHOT_CALL_ENTRY() != source_before
        or _CALLABLE_SNAPSHOT_CALL_ENTRY() != callable_before
        or _IDLE_SNAPSHOT_VERIFIER_CALL_ENTRY() != snapshot_verifier_before
        or _secure_directory_identity(
            parent.output_directory,
            "parent checkpoint output",
        )
        != parent_directory_identity
        or _secure_directory_identity(
            target.output_directory,
            "target checkpoint output",
        )
        != target_directory_identity
        or (
            parent_cache_job_path is not None
            and _secure_directory_identity(
                parent_cache_job_path,
                "parent cache job",
            )
            != cache_directory_identity
        )
    ):
        raise MuPsiPhaseInteractionError(
            "manifest, dylib, or analyzer source changed before publication"
        )
    report = _complete_report(body)
    payload = canonical_json_bytes(report)
    _publish_new_report(
        output_path,
        payload,
        forbidden_directory_identities=frozenset(forbidden_directory_identities),
    )
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only, permanently non-qualifying mu x psi x phase analysis "
            "of two externally authenticated native refinement manifests."
        )
    )
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--native-library", required=True, type=Path)
    parser.add_argument(
        "--parent-cache-job",
        type=Path,
        help=(
            "optional complete parent cache job directory for read-only B-even/E "
            "record-substitution proof"
        ),
    )
    parser.add_argument("--parent-cache-payload-set-sha256")
    parser.add_argument("--parent-cache-direction-stream-sha256")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = analyze_mu_psi_phase_interaction(
            args.parent_manifest,
            args.parent_sha256,
            args.target_manifest,
            args.target_sha256,
            args.native_library,
            args.output,
            args.parent_cache_job,
            args.parent_cache_payload_set_sha256,
            args.parent_cache_direction_stream_sha256,
        )
    except (MuPsiPhaseInteractionError, OSError, TypeError, ValueError) as error:
        print(f"mu-psi-phase interaction error: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "automaticEscalation": report["automaticEscalation"],
                "id": report["id"],
                "output": os.fspath(args.output),
                "productEligible": report["productEligible"],
                "qualified": report["qualified"],
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


_SOURCE_SNAPSHOT_PUBLIC_ENTRY: Final = _source_snapshot
_SOURCE_SNAPSHOT_CALL_ENTRY: Final = _SOURCE_SNAPSHOT_PUBLIC_ENTRY
_CALLABLE_SNAPSHOT_PUBLIC_ENTRY: Final = _callable_snapshot
_CALLABLE_SNAPSHOT_CALL_ENTRY: Final = _CALLABLE_SNAPSHOT_PUBLIC_ENTRY
_MODULE_FUNCTION_BINDINGS_PUBLIC_ENTRY: Final = _module_function_bindings
_MODULE_FUNCTION_BINDINGS_CALL_ENTRY: Final = _MODULE_FUNCTION_BINDINGS_PUBLIC_ENTRY
_MODULE_BINDING_GUARD_PUBLIC_ENTRY: Final = _require_frozen_module_function_bindings
_MODULE_BINDING_GUARD_CALL_ENTRY: Final = _MODULE_BINDING_GUARD_PUBLIC_ENTRY
_IDLE_SNAPSHOT_VERIFIER_PUBLIC_ENTRY: Final = _idle_snapshot_verifier_identity
_IDLE_SNAPSHOT_VERIFIER_CALL_ENTRY: Final = _IDLE_SNAPSHOT_VERIFIER_PUBLIC_ENTRY
_FROZEN_MODULE_FUNCTION_BINDINGS: Final = _MODULE_FUNCTION_BINDINGS_CALL_ENTRY()
_FROZEN_SOURCE_SNAPSHOT: Final = _SOURCE_SNAPSHOT_CALL_ENTRY()
_FROZEN_CALLABLE_SNAPSHOT: Final = _CALLABLE_SNAPSHOT_CALL_ENTRY()
_FROZEN_SNAPSHOT_VERIFIER_IDENTITY: Final = (
    _IDLE_SNAPSHOT_VERIFIER_CALL_ENTRY()
)


if __name__ == "__main__":
    raise SystemExit(main())
