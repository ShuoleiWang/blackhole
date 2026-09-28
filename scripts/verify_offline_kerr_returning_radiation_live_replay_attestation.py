#!/usr/bin/env python3
"""Verify historical returning-radiation live-replay attestation bytes.

This verifier authenticates the closed attestation artifact, its exact current
schema/source closure, and the separately supplied spectral product.  It never
owns a live sampler and never reruns frame, thermal, direction-cache, or
direction-ray physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, NoReturn, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import offline.kerr_returning_radiation_live_replay_attestation as attestation_module
from offline.kerr_returning_radiation_live_replay_attestation import (
    ATTESTATION_DIRECTORY_SUFFIX,
    ATTESTATION_SCHEMA,
    ATTESTATION_SCHEMA_ID,
    DEFAULT_ATTESTATION_SCHEMA,
    MANIFEST_NAME,
    MAXIMUM_ATTESTATION_MANIFEST_BYTES,
    MAXIMUM_ATTESTATION_SCHEMA_BYTES,
    MAXIMUM_JSON_NESTING_DEPTH,
    MAXIMUM_SPECTRAL_MANIFEST_BYTES,
    SIDECAR_NAME,
    read_stable_bounded_file,
    validated_live_replay_report,
)
from offline.job import canonical_json_bytes
from offline.spectral_product import SpectralProductPublication
from scripts.verify_nr_contract import (
    ContractError,
    audit_schema_dialect,
    validate_json_schema,
)
from scripts.verify_offline_spectral_frame import (
    validate_scientific_spectral_frame,
)


class ReturningRadiationLiveReplayAttestationContractError(ContractError):
    """Deterministic historical-attestation validation failure."""


def fail(path: str, message: str) -> NoReturn:
    raise ReturningRadiationLiveReplayAttestationContractError(f"{path}: {message}")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_json_nesting(payload: bytes, label: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for byte in payload:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):
            depth += 1
            if depth > MAXIMUM_JSON_NESTING_DEPTH:
                fail(label, "exceeds the fixed JSON nesting limit")
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth < 0:
                fail(label, "has invalid JSON nesting")


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if type(key) is not str or key in result:
                fail(label, f"duplicate or foreign JSON key {key!r}")
            result[key] = value
        return result

    def reject(value: str) -> NoReturn:
        fail(label, f"non-finite JSON number {value!r} is forbidden")

    _validate_json_nesting(payload, label)
    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject,
        )
    except (
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as error:
        fail(label, f"invalid UTF-8 JSON: {error}")
    if type(parsed) is not dict:
        fail(label, "JSON root must be an exact object")
    return parsed


def _artifact_root_identity(path: Path) -> tuple[int, int, int, int, int]:
    root = path.parent
    if root.name.endswith(ATTESTATION_DIRECTORY_SUFFIX) is not True:
        fail("$files", "attestation directory name lacks the fixed v1 suffix")
    if root.is_symlink():
        fail("$files", "symlinked attestation root is forbidden")
    try:
        metadata = os.lstat(root)
    except OSError as error:
        fail("$files", f"cannot inspect attestation root: {error}")
    if not stat.S_ISDIR(metadata.st_mode):
        fail("$files", "attestation root must be a directory")
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _closed_artifact_root(path: Path) -> tuple[int, int, int, int, int]:
    root = path.parent
    before = _artifact_root_identity(path)
    try:
        names = set(os.listdir(root))
    except OSError as error:
        fail("$files", f"cannot inspect attestation root: {error}")
    if names != {MANIFEST_NAME, SIDECAR_NAME}:
        fail("$files", "attestation root differs from its closed two-file tree")
    for name in names:
        candidate = root / name
        try:
            child = os.lstat(candidate)
        except OSError as error:
            fail("$files", f"cannot inspect {name}: {error}")
        if not stat.S_ISREG(child.st_mode):
            fail("$files", f"{name} must be a non-symlink regular file")
    after = _artifact_root_identity(path)
    if after != before:
        fail("$files", "attestation root changed during its closed-tree scan")
    return after


def _closed_artifact_snapshot(
    path: Path,
    label: str,
) -> tuple[tuple[int, int, int, int, int], bytes, bytes]:
    first_identity = _closed_artifact_root(path)
    first_manifest = read_stable_bounded_file(
        path,
        MAXIMUM_ATTESTATION_MANIFEST_BYTES,
        f"{label} attestation manifest",
    )
    first_sidecar = read_stable_bounded_file(
        path.parent / SIDECAR_NAME,
        4096,
        f"{label} attestation sidecar",
    )
    scanned_identity = _closed_artifact_root(path)
    closing_manifest = read_stable_bounded_file(
        path,
        MAXIMUM_ATTESTATION_MANIFEST_BYTES,
        f"closing {label} attestation manifest",
    )
    closing_sidecar = read_stable_bounded_file(
        path.parent / SIDECAR_NAME,
        4096,
        f"closing {label} attestation sidecar",
    )
    closing_identity = _artifact_root_identity(path)
    if (
        first_identity != scanned_identity
        or scanned_identity != closing_identity
        or first_manifest != closing_manifest
        or first_sidecar != closing_sidecar
    ):
        fail("$files", f"{label} attestation changed across its final self-closure")
    return closing_identity, closing_manifest, closing_sidecar


def _current_source_artifacts() -> tuple[Any, ...]:
    # Lazy import avoids adding a verifier -> renderer -> verifier import cycle.
    import scripts.render_offline_kerr_returning_radiation_frame as renderer

    # A clean standalone verifier has not traversed the renderer's lazy replay
    # edge.  Load that one canonical module before the unchanged exact-origin
    # gate authenticates the complete fixed producer closure.
    renderer._load_declared_source_modules_for_origin_gate()
    return renderer.collect_source_artifacts(renderer.ROOT)


def _source_identity_matches_current(
    spectral_manifest: dict[str, Any],
    current_sources: tuple[Any, ...],
) -> None:
    job_spec = spectral_manifest["producer"]["jobSpec"]
    declared_inputs = job_spec["inputs"]
    declared_source_documents = sorted(
        canonical_json_bytes(entry)
        for entry in declared_inputs
        if type(entry) is dict
        and type(entry.get("uri")) is str
        and entry["uri"].startswith("repo-source://")
    )
    current_source_documents = sorted(
        canonical_json_bytes(artifact.as_dict()) for artifact in current_sources
    )
    if declared_source_documents != current_source_documents:
        fail(
            "$.bindings.inputArtifactsSha256",
            "spectral JobSpec does not bind the exact current source artifact set",
        )
    current_hashes = sorted({artifact.sha256 for artifact in current_sources})
    if job_spec["producerSourceHashes"] != current_hashes:
        fail(
            "$.bindings.producerSourceHashesSha256",
            "spectral JobSpec source hashes differ from current exact source bytes",
        )


def validate_returning_radiation_live_replay_attestation(
    attestation_manifest_path: Path | str,
    spectral_manifest_path: Path | str,
    *,
    schema_path: Path | str = DEFAULT_ATTESTATION_SCHEMA,
) -> dict[str, Any]:
    """Validate historical evidence without performing any live replay."""

    path = Path(attestation_manifest_path).absolute()
    spectral_path = Path(spectral_manifest_path).absolute()
    schema = Path(schema_path).absolute()
    if path.name != MANIFEST_NAME or spectral_path.name != MANIFEST_NAME:
        fail("$", "attestation and spectral manifests must both be named manifest.json")
    root_identity, payload, sidecar = _closed_artifact_snapshot(path, "initial")

    schema_payload = read_stable_bounded_file(
        schema,
        MAXIMUM_ATTESTATION_SCHEMA_BYTES,
        "attestation schema",
    )
    default_schema_payload = read_stable_bounded_file(
        Path(DEFAULT_ATTESTATION_SCHEMA),
        MAXIMUM_ATTESTATION_SCHEMA_BYTES,
        "repository attestation schema",
    )
    if schema_payload != default_schema_payload:
        fail("$schema", "verifier requires the repository's exact current schema")
    schema_document = _strict_json(schema_payload, "$schema")
    if (
        schema_document.get("$schema")
        != "https://json-schema.org/draft/2020-12/schema"
        or schema_document.get("$id") != ATTESTATION_SCHEMA_ID
    ):
        fail("$schema", "unexpected attestation schema identity or dialect")
    audit_schema_dialect(schema_document)

    document = _strict_json(payload, "$")
    if canonical_json_bytes(document) != payload:
        fail("$", "attestation manifest is not canonical JSON")
    validate_json_schema(document, schema_document, schema_document)
    if document["schema"] != ATTESTATION_SCHEMA:
        fail("$.schema", "unsupported attestation schema")
    digest = _sha256(payload)
    if sidecar != f"{digest}  {MANIFEST_NAME}\n".encode("ascii"):
        fail("$.integrity", "manifest sidecar does not bind attestation bytes")

    try:
        spectral_report = validate_scientific_spectral_frame(spectral_path)
    except ContractError as error:
        fail("$.subject", f"spectral product is not structurally valid: {error}")
    spectral_payload = read_stable_bounded_file(
        spectral_path,
        MAXIMUM_SPECTRAL_MANIFEST_BYTES,
        "attested spectral manifest",
    )
    spectral_manifest = _strict_json(spectral_payload, "$spectral")
    publication = SpectralProductPublication(
        output_directory=spectral_path.parent,
        manifest_path=spectral_path,
        manifest_sha256=_sha256(spectral_payload),
        product_id=spectral_manifest["id"],
        product_sha256=spectral_manifest["integrity"]["productSha256"],
        tile_count=spectral_report["tileCount"],
        record_count=spectral_report["recordCount"],
    )
    try:
        report = validated_live_replay_report(document["evidence"], publication)
        expected = attestation_module._manifest_claims(
            publication,
            spectral_payload,
            report,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        fail("$", f"attestation bindings are invalid: {error}")
    if canonical_json_bytes(document) != canonical_json_bytes(expected):
        fail("$", "attestation is not the exact claim derived from its subject")

    current_sources_before = _current_source_artifacts()
    _source_identity_matches_current(spectral_manifest, current_sources_before)
    final_spectral_report = validate_scientific_spectral_frame(spectral_path)
    final_spectral_payload = read_stable_bounded_file(
        spectral_path,
        MAXIMUM_SPECTRAL_MANIFEST_BYTES,
        "final attested spectral manifest",
    )
    current_sources_after = _current_source_artifacts()
    final_root_identity, final_payload, final_sidecar = _closed_artifact_snapshot(
        path,
        "final",
    )
    if (
        final_spectral_report != spectral_report
        or final_spectral_payload != spectral_payload
        or final_payload != payload
        or final_sidecar != sidecar
        or current_sources_after != current_sources_before
        or final_root_identity != root_identity
    ):
        fail(
            "$",
            "attestation, subject, or current source bytes changed during verification",
        )
    return {
        "attestationId": document["id"],
        "attestationManifestSha256": digest,
        "attestationVerified": True,
        "directionCacheRecordsReplayed": False,
        "directionRaysRetraced": False,
        "independentPhysicsOracle": False,
        "physicsVerified": False,
        "sameCodeReplayAttested": True,
        "sameCodeReplayReperformed": False,
        "sourceArtifactsCurrentMatch": True,
        "spectralManifestSha256": document["subject"]["manifestSha256"],
        "status": "returning-radiation-live-replay-attestation-conformant",
        "thermalFixedPointReplayed": False,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attestation_manifest", type=Path)
    parser.add_argument("spectral_manifest", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_ATTESTATION_SCHEMA)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        report = validate_returning_radiation_live_replay_attestation(
            arguments.attestation_manifest,
            arguments.spectral_manifest,
            schema_path=arguments.schema,
        )
    except (ContractError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            f"returning live-replay attestation verification failed: {error}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ReturningRadiationLiveReplayAttestationContractError",
    "main",
    "validate_returning_radiation_live_replay_attestation",
)
