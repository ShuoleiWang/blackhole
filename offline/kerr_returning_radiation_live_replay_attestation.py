"""Crash-safe publication of returning-radiation live-replay evidence.

The publisher owns the canonical live replay call.  Callers supply the exact
live sampler and spectral publication, never a report.  The resulting sibling
directory is immutable, content-addressed by its canonical manifest bytes, and
kept outside the generic spectral product's deliberately closed file tree.

This is historical same-code evidence.  It is not an independent physics
oracle and it does not rerun the thermal fixed point, direction-cache records,
or direction rays.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Final, Mapping, NoReturn
import uuid

from offline.authenticated_artifact import authenticate_stable_artifact
from offline.job import canonical_json_bytes
from offline.kerr_returning_radiation_finite_thickness_frame import (
    KerrReturningRadiationFiniteThicknessRaySampler,
)
from offline.kerr_returning_radiation_frame_context import (
    ValidatedReturningThermalAuthority,
)
from offline.spectral_product import SpectralProductPublication


ROOT: Final = Path(__file__).resolve().parents[1]
ATTESTATION_SCHEMA: Final = (
    "blackhole.returning-radiation-live-replay-attestation/v1"
)
ATTESTATION_SCHEMA_ID: Final = (
    "https://github.com/ShuoleiWang/blackhole/schemas/"
    "offline-returning-radiation-live-replay-attestation-v1.schema.json"
)
DEFAULT_ATTESTATION_SCHEMA: Final = (
    ROOT
    / "schemas"
    / "offline-returning-radiation-live-replay-attestation-v1.schema.json"
)
MANIFEST_NAME: Final = "manifest.json"
SIDECAR_NAME: Final = "manifest.sha256"
ATTESTATION_DIRECTORY_SUFFIX: Final = ".live-replay-attestation-v1"
MAXIMUM_ATTESTATION_MANIFEST_BYTES: Final = 1024 * 1024
MAXIMUM_SPECTRAL_MANIFEST_BYTES: Final = 64 * 1024 * 1024
MAXIMUM_ATTESTATION_SCHEMA_BYTES: Final = 1024 * 1024
MAXIMUM_SOURCE_HASH_COUNT: Final = 64
MAXIMUM_INPUT_ARTIFACT_COUNT: Final = 66
MAXIMUM_JSON_NESTING_DEPTH: Final = 64
MAXIMUM_SUBJECT_TILE_COUNT: Final = 65_536
MAXIMUM_SUBJECT_RECORD_COUNT: Final = 16_777_216
_PATH_TYPE: Final = type(Path())

_TRUE_REPORT_FIELDS: Final = (
    "structuralContractVerified",
    "producerIdentityCurrentMatch",
    "jobSpecVerified",
    "samplerDescriptorLiveMatch",
    "numericBackendCurrentMatch",
    "sourceArtifactsCurrentMatch",
    "frameGeodesicsReplayed",
    "frozenThermalSnapshotBound",
    "pixelBytesExact",
)
_FALSE_REPORT_FIELDS: Final = (
    "thermalFixedPointReplayed",
    "directionCacheRecordsReplayed",
    "directionRaysRetraced",
    "independentPhysicsOracle",
)
_INTEGER_REPORT_FIELDS: Final = (
    "recordCount",
    "tileCount",
    "totalRaySamples",
    "totalFrameGeodesicsReplayed",
)
_TEXT_REPORT_FIELDS: Final = (
    "replayScope",
    "scientificScope",
    "sourceHashScope",
)
_REPORT_FIELDS: Final = {
    "id",
    "status",
    *_TRUE_REPORT_FIELDS,
    *_FALSE_REPORT_FIELDS,
    *_INTEGER_REPORT_FIELDS,
    *_TEXT_REPORT_FIELDS,
}
_THERMAL_IDENTITY_FIELDS: Final = {
    "axisymmetricKernelDescriptorSha256",
    "descriptorSha256",
    "novikovThorneDiskDescriptorSha256",
    "profileDescriptorSha256",
    "providerDescriptorSha256",
    "sourceEvidenceDescriptorSha256",
    "sourceKernelDescriptorSha256",
    "sourceKernelKind",
    "surfaceIdentityDescriptorSha256",
    "underlyingKernelDescriptorSha256",
}
_SAMPLER_DESCRIPTOR_ENTRY: Final = (
    KerrReturningRadiationFiniteThicknessRaySampler.descriptor
)
_AUTHORITY_REQUIRE_LIVE_ENTRY: Final = ValidatedReturningThermalAuthority.require_live


class ReturningRadiationLiveReplayAttestationError(RuntimeError):
    """Fail-closed live-replay attestation publication error."""


@dataclass(frozen=True, slots=True)
class ReturningRadiationLiveReplayAttestationPublication:
    output_directory: Path
    manifest_path: Path
    manifest_sha256: str
    attestation_id: str
    spectral_manifest_sha256: str
    live_replay_report: Mapping[str, Any]


def default_live_replay_attestation_directory(
    spectral_output_directory: Path | str,
) -> Path:
    output = Path(spectral_output_directory).absolute()
    return output.with_name(f"{output.name}{ATTESTATION_DIRECTORY_SUFFIX}")


def _fail(message: str) -> NoReturn:
    raise ReturningRadiationLiveReplayAttestationError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _exact_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypeError(f"{label} must be an exact lowercase SHA-256 string")
    return value


def _validate_spectral_publication(
    publication: SpectralProductPublication,
) -> None:
    if type(publication) is not SpectralProductPublication:
        raise TypeError("spectral_publication must have exact publication type")
    output = publication.output_directory
    manifest = publication.manifest_path
    if (
        type(output) is not _PATH_TYPE
        or not output.is_absolute()
        or output == output.parent
        or Path(os.path.abspath(os.fspath(output))) != output
    ):
        raise TypeError(
            "spectral publication output must be an exact lexical absolute Path"
        )
    if (
        type(manifest) is not _PATH_TYPE
        or not manifest.is_absolute()
        or Path(os.path.abspath(os.fspath(manifest))) != manifest
        or manifest.name != MANIFEST_NAME
        or manifest.parent != output
        or manifest != output / MANIFEST_NAME
    ):
        raise TypeError(
            "spectral publication manifest must be the exact output/manifest.json Path"
        )
    _exact_sha256(publication.manifest_sha256, "spectral manifest sha256")
    _exact_sha256(publication.product_sha256, "spectral product sha256")
    product_id = publication.product_id
    if (
        type(product_id) is not str
        or not product_id.startswith("scientific-spectral-frame-")
        or len(product_id) != len("scientific-spectral-frame-") + 24
        or any(
            character not in "0123456789abcdef"
            for character in product_id[-24:]
        )
    ):
        raise TypeError("spectral product id is malformed")
    if (
        type(publication.tile_count) is not int
        or publication.tile_count < 1
        or publication.tile_count > MAXIMUM_SUBJECT_TILE_COUNT
        or type(publication.record_count) is not int
        or publication.record_count < 1
        or publication.record_count > MAXIMUM_SUBJECT_RECORD_COUNT
    ):
        raise TypeError("spectral publication counters exceed their exact limits")


def _paths_equal_or_nested(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _canonical_hash(value: Any) -> str:
    try:
        return _sha256(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise ReturningRadiationLiveReplayAttestationError(
            f"attestation value is not finite canonical JSON: {error}"
        ) from error


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
                _fail(f"{label} exceeds its fixed JSON nesting limit")
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth < 0:
                _fail(f"{label} has invalid JSON nesting")


def _strict_canonical_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if type(key) is not str or key in result:
                _fail(f"{label} contains a duplicate or foreign JSON key")
            result[key] = value
        return result

    def reject(value: str) -> NoReturn:
        _fail(f"{label} contains forbidden non-finite JSON number {value!r}")

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
        _fail(f"{label} is not strict UTF-8 JSON: {error}")
    if type(parsed) is not dict:
        _fail(f"{label} root must be an exact object")
    try:
        encoded = canonical_json_bytes(parsed)
    except (TypeError, ValueError, RecursionError) as error:
        _fail(f"{label} is not finite canonical JSON: {error}")
    if encoded != payload:
        _fail(f"{label} is not in canonical JSON encoding")
    return parsed


def _read_nofollow(path: Path, maximum_bytes: int, label: str) -> bytes:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate == candidate.parent:
        raise TypeError(f"{label} path must be an absolute non-root path")
    required = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    values = tuple(getattr(os, name, None) for name in required)
    if any(type(value) is not int for value in values):
        _fail("secure attestation read primitives are unavailable")
    directory_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
    )
    directories: list[int] = []
    descriptor = -1
    try:
        directories.append(os.open(os.sep, directory_flags))
        for component in candidate.parts[1:-1]:
            if component in ("", ".", ".."):
                _fail(f"{label} path is not canonical")
            directories.append(
                os.open(component, directory_flags, dir_fd=directories[-1])
            )
        descriptor = os.open(
            candidate.name,
            file_flags,
            dir_fd=directories[-1],
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{label} must be a regular file")
        if before.st_size < 0 or before.st_size > maximum_bytes:
            _fail(f"{label} exceeds its fixed {maximum_bytes}-byte limit")
        chunks: list[bytes] = []
        length = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum_bytes - length + 1))
            if not block:
                break
            length += len(block)
            if length > maximum_bytes:
                _fail(f"{label} grew beyond its fixed byte limit")
            chunks.append(block)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after) or length != before.st_size:
            _fail(f"{label} changed while it was read")
        return b"".join(chunks)
    except OSError as error:
        raise ReturningRadiationLiveReplayAttestationError(
            f"cannot read {label} without following symlinks: {error}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        for directory in reversed(directories):
            os.close(directory)


def read_stable_bounded_file(path: Path, maximum_bytes: int, label: str) -> bytes:
    """Return one stable no-symlink byte snapshot bracketed by full hashes."""

    candidate = Path(path)
    before = authenticate_stable_artifact(
        candidate,
        label,
        maximum_byte_length=maximum_bytes,
        maximum_total_byte_length=maximum_bytes,
    )
    payload = _read_nofollow(candidate, maximum_bytes, label)
    after = authenticate_stable_artifact(
        candidate,
        label,
        maximum_byte_length=maximum_bytes,
        maximum_total_byte_length=maximum_bytes,
    )
    if (
        before != after
        or len(payload) != before.byte_length
        or _sha256(payload) != before.sha256
    ):
        _fail(f"{label} changed across its authenticated byte snapshot")
    return payload


def validated_live_replay_report(
    report: object,
    publication: SpectralProductPublication,
) -> dict[str, Any]:
    """Return an exact canonical report that cannot overclaim replay scope."""

    if type(publication) is not SpectralProductPublication:
        raise TypeError("publication must have exact SpectralProductPublication type")
    if type(report) is not dict or any(type(key) is not str for key in report):
        raise TypeError("live replay must return an exact string-keyed report object")
    if set(report) != _REPORT_FIELDS:
        _fail("live replay returned a non-exact report schema")
    if (
        type(report["id"]) is not str
        or report["id"] != publication.product_id
        or type(report["status"]) is not str
        or report["status"]
        != "returning-radiation-live-sampler-byte-replay-conformant"
    ):
        _fail("live replay report identity or status is invalid")
    if any(report[name] is not True for name in _TRUE_REPORT_FIELDS):
        _fail("live replay report lacks required positive evidence")
    if any(report[name] is not False for name in _FALSE_REPORT_FIELDS):
        _fail("live replay report overclaims its scientific evidence")
    if any(
        type(report[name]) is not int or report[name] < 1
        for name in _INTEGER_REPORT_FIELDS
    ):
        _fail("live replay report contains invalid counters")
    if (
        report["recordCount"] != publication.record_count
        or report["tileCount"] != publication.tile_count
        or report["recordCount"] > 16_777_216
        or report["tileCount"] > 65_536
        or report["totalRaySamples"] > 5_000_000
        or report["totalFrameGeodesicsReplayed"] > 10_000_000
        or report["totalFrameGeodesicsReplayed"] != 2 * report["totalRaySamples"]
    ):
        _fail("live replay counters do not close within fixed limits")
    if any(
        type(report[name]) is not str
        or not report[name]
        or len(report[name].encode("utf-8")) > 4096
        for name in _TEXT_REPORT_FIELDS
    ):
        _fail("live replay report contains an invalid scope statement")
    return json.loads(canonical_json_bytes(report))


_LIVE_REPLAY_FROZEN_ENTRY: Any | None = None
_STRUCTURAL_VERIFIER_FROZEN_ENTRY: Any | None = None


def _canonical_live_replay_entry() -> Any:
    from offline import kerr_returning_radiation_spectral_replay as replay_module

    global _LIVE_REPLAY_FROZEN_ENTRY
    current = replay_module.validate_returning_radiation_spectral_live_replay
    canonical = replay_module._CANONICAL_LIVE_REPLAY_VALIDATOR_ENTRY
    if current is not canonical:
        _fail("returning live-replay callable identity changed")
    if _LIVE_REPLAY_FROZEN_ENTRY is None:
        _LIVE_REPLAY_FROZEN_ENTRY = canonical
    elif _LIVE_REPLAY_FROZEN_ENTRY is not canonical:
        _fail("returning live-replay callable identity changed")
    return canonical


def _canonical_structural_verifier_entry() -> Any:
    from offline import kerr_returning_radiation_spectral_replay as replay_module
    from scripts import verify_offline_spectral_frame as verifier_module

    global _STRUCTURAL_VERIFIER_FROZEN_ENTRY
    current = verifier_module.validate_scientific_spectral_frame
    canonical = replay_module._STRUCTURAL_VERIFIER_ENTRY
    if current is not canonical:
        _fail("spectral structural-verifier callable identity changed")
    if _STRUCTURAL_VERIFIER_FROZEN_ENTRY is None:
        _STRUCTURAL_VERIFIER_FROZEN_ENTRY = canonical
    elif _STRUCTURAL_VERIFIER_FROZEN_ENTRY is not canonical:
        _fail("spectral structural-verifier callable identity changed")
    return canonical


def _validated_structural_subject(
    publication: SpectralProductPublication,
    schema_path: Path,
) -> bytes:
    entry = _canonical_structural_verifier_entry()
    report = entry(publication.manifest_path, schema_path)
    if (
        type(report) is not dict
        or set(report)
        != {
            "id",
            "physicsVerified",
            "provenanceScope",
            "recordCount",
            "status",
            "tileCount",
        }
        or report["id"] != publication.product_id
        or report["physicsVerified"] is not False
        or type(report["provenanceScope"]) is not str
        or not report["provenanceScope"]
        or report["recordCount"] != publication.record_count
        or report["status"]
        != "scientific-spectral-frame-structural-contract-conformant"
        or report["tileCount"] != publication.tile_count
    ):
        _fail("spectral structural verifier returned a non-exact report")
    if _canonical_structural_verifier_entry() is not entry:
        _fail("spectral structural-verifier callable identity changed")
    return canonical_json_bytes(report)


def _validated_sampler_live_identity(
    sampler: KerrReturningRadiationFiniteThicknessRaySampler,
) -> tuple[bytes, object]:
    if (
        type(sampler) is not KerrReturningRadiationFiniteThicknessRaySampler
        or KerrReturningRadiationFiniteThicknessRaySampler.descriptor
        is not _SAMPLER_DESCRIPTOR_ENTRY
        or type(sampler.authority) is not ValidatedReturningThermalAuthority
        or ValidatedReturningThermalAuthority.require_live
        is not _AUTHORITY_REQUIRE_LIVE_ENTRY
    ):
        _fail("returning sampler live-identity callable changed")
    snapshot = _AUTHORITY_REQUIRE_LIVE_ENTRY(sampler.authority)
    try:
        authenticated_snapshot = object.__getattribute__(
            sampler,
            "_authenticated_snapshot",
        )
    except AttributeError as error:
        raise ReturningRadiationLiveReplayAttestationError(
            "returning sampler lacks its authenticated snapshot"
        ) from error
    if snapshot is not authenticated_snapshot:
        _fail("returning sampler live authority identity changed")
    try:
        descriptor = canonical_json_bytes(_SAMPLER_DESCRIPTOR_ENTRY(sampler))
    except (TypeError, ValueError, RuntimeError) as error:
        raise ReturningRadiationLiveReplayAttestationError(
            f"returning sampler live descriptor is invalid: {error}"
        ) from error
    return descriptor, snapshot


def _manifest_claims(
    publication: SpectralProductPublication,
    manifest_payload: bytes,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _strict_canonical_json(manifest_payload, "spectral manifest")
    manifest_sha256 = _sha256(manifest_payload)
    try:
        job_spec = manifest["producer"]["jobSpec"]
        source_hashes = job_spec["producerSourceHashes"]
        inputs = job_spec["inputs"]
        thermal_identity = manifest["sampler"]["descriptor"]["authority"]
    except (KeyError, TypeError) as error:
        _fail(f"spectral manifest lacks attestation binding data: {error}")
    if (
        manifest_sha256 != publication.manifest_sha256
        or manifest.get("id") != publication.product_id
        or manifest.get("integrity", {}).get("productSha256")
        != publication.product_sha256
    ):
        _fail("spectral publication metadata differs from its manifest bytes")
    job_key = manifest["producer"]["jobKey"]
    if type(job_key) is not str or len(job_key) != 64:
        _fail("spectral tile job key is malformed")
    if _canonical_hash(job_spec) != job_key:
        _fail("spectral tile job key does not bind the exact JobSpec")
    if (
        type(source_hashes) is not list
        or not source_hashes
        or len(source_hashes) > MAXIMUM_SOURCE_HASH_COUNT
        or any(type(value) is not str or len(value) != 64 for value in source_hashes)
    ):
        _fail("producer source hash set exceeds its fixed schema")
    if type(inputs) is not list or len(inputs) > MAXIMUM_INPUT_ARTIFACT_COUNT:
        _fail("JobSpec input artifact set exceeds its fixed schema")
    if (
        type(thermal_identity) is not dict
        or set(thermal_identity) != _THERMAL_IDENTITY_FIELDS
    ):
        _fail("thermal snapshot identity has a non-exact schema")
    claims = {
        "bindings": {
            "inputArtifactsSha256": _canonical_hash(inputs),
            "numericBackendDescriptorSha256": manifest["runtimeNumericBackend"][
                "descriptorSha256"
            ],
            "producerSourceHashesSha256": _canonical_hash(source_hashes),
            "samplerDescriptorSha256": manifest["sampler"]["descriptorSha256"],
            "thermalSnapshotIdentity": json.loads(
                canonical_json_bytes(thermal_identity)
            ),
        },
        "directionCacheRecordsReplayed": False,
        "directionRaysRetraced": False,
        "evidence": dict(report),
        "independentPhysicsOracle": False,
        "physicsVerified": False,
        "sameCodeReplayAttested": True,
        "sameCodeReplayReperformed": False,
        "schema": ATTESTATION_SCHEMA,
        "scientificStatus": {
            "classification": (
                "historical same-code returning-radiation live sampler byte replay"
            ),
            "directionCacheRecordsReplayed": False,
            "directionRaysRetraced": False,
            "independentPhysicsOracle": False,
            "prohibitedClaim": (
                "This attestation must not be described as an independent physics "
                "oracle or as replay of the thermal fixed point, direction-cache "
                "records, or direction rays."
            ),
            "thermalFixedPointReplayed": False,
        },
        "subject": {
            "jobKey": job_key,
            "manifestByteLength": len(manifest_payload),
            "manifestSha256": manifest_sha256,
            "productId": manifest["id"],
            "productSha256": manifest["integrity"]["productSha256"],
            "schema": manifest["schema"],
        },
        "thermalFixedPointReplayed": False,
    }
    claims_sha256 = _canonical_hash(claims)
    return {
        **claims,
        "id": f"returning-radiation-live-replay-attestation-{claims_sha256[:24]}",
        "integrity": {"claimsSha256": claims_sha256},
    }


def _directory_open_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if type(no_follow) is not int or type(directory) is not int:
        _fail("secure attestation directory primitives are unavailable")
    return os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)


def _directory_identity(descriptor: int) -> tuple[int, int, int]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        _fail("attestation directory session lost its directory identity")
    return metadata.st_dev, metadata.st_ino, metadata.st_mode


def _open_absolute_directory_chain(path: Path, label: str) -> list[int]:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.anchor != os.sep:
        raise TypeError(f"{label} must be an absolute directory path")
    if any(component in ("", ".", "..") for component in candidate.parts[1:]):
        raise TypeError(f"{label} is not a canonical directory path")
    flags = _directory_open_flags()
    descriptors: list[int] = []
    try:
        descriptors.append(os.open(os.sep, flags))
        for component in candidate.parts[1:]:
            descriptors.append(
                os.open(component, flags, dir_fd=descriptors[-1])
            )
        return descriptors
    except OSError as error:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise ReturningRadiationLiveReplayAttestationError(
            f"cannot anchor {label} without following symlinks: {error}"
        ) from error


def _directory_chain_identity(
    descriptors: list[int],
) -> tuple[tuple[int, int, int], ...]:
    return tuple(_directory_identity(descriptor) for descriptor in descriptors)


def _assert_directory_path_identity(
    path: Path,
    expected: tuple[tuple[int, int, int], ...],
    label: str,
) -> None:
    reopened = _open_absolute_directory_chain(path, label)
    try:
        if _directory_chain_identity(reopened) != expected:
            _fail(f"{label} identity changed during attestation publication")
    finally:
        for descriptor in reversed(reopened):
            os.close(descriptor)


def _assert_absent_at(directory_descriptor: int, name: str, output: Path) -> None:
    try:
        os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise ReturningRadiationLiveReplayAttestationError(
            f"cannot inspect attestation destination {output}: {error}"
        ) from error
    raise FileExistsError(f"refusing to overwrite existing attestation {output}")


def _write_exclusive_at(
    directory_descriptor: int,
    name: str,
    payload: bytes,
) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=directory_descriptor,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written < 1:
                _fail("short write while publishing attestation")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _promote_directory_no_replace_at(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    destination: Path,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        function = library.renameatx_np
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            0x00000004,
        )
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            1,
        )
    else:
        _fail("platform lacks atomic no-replace directory publication")
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            f"refusing to overwrite existing attestation {destination}"
        )
    _fail(f"unable to publish attestation atomically: {os.strerror(number)}")


def _read_regular_at(
    directory_descriptor: int,
    name: str,
    maximum_bytes: int,
    label: str,
) -> tuple[bytes, tuple[int, int, int, int, int, int]]:
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_descriptor,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{label} must be a regular file")
        if before.st_size < 0 or before.st_size > maximum_bytes:
            _fail(f"{label} exceeds its fixed byte limit")
        chunks: list[bytes] = []
        length = 0
        while True:
            block = os.read(
                descriptor,
                min(1024 * 1024, maximum_bytes - length + 1),
            )
            if not block:
                break
            chunks.append(block)
            length += len(block)
            if length > maximum_bytes:
                _fail(f"{label} grew beyond its fixed byte limit")
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after) or length != before.st_size:
            _fail(f"{label} changed while it was read")
        return b"".join(chunks), identity(after)
    except OSError as error:
        raise ReturningRadiationLiveReplayAttestationError(
            f"cannot read {label} from the anchored directory: {error}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _closed_published_directory_snapshot(
    directory_descriptor: int,
) -> tuple[
    bytes,
    bytes,
    tuple[int, int, int],
]:
    before = os.fstat(directory_descriptor)
    try:
        names = set(os.listdir(directory_descriptor))
    except OSError as error:
        raise ReturningRadiationLiveReplayAttestationError(
            f"cannot scan published attestation directory: {error}"
        ) from error
    if names != {MANIFEST_NAME, SIDECAR_NAME}:
        _fail("published attestation differs from its closed two-file tree")
    manifest, manifest_identity = _read_regular_at(
        directory_descriptor,
        MANIFEST_NAME,
        MAXIMUM_ATTESTATION_MANIFEST_BYTES,
        "published attestation manifest",
    )
    sidecar, sidecar_identity = _read_regular_at(
        directory_descriptor,
        SIDECAR_NAME,
        4096,
        "published attestation sidecar",
    )
    closing_manifest, closing_manifest_identity = _read_regular_at(
        directory_descriptor,
        MANIFEST_NAME,
        MAXIMUM_ATTESTATION_MANIFEST_BYTES,
        "closing published attestation manifest",
    )
    closing_sidecar, closing_sidecar_identity = _read_regular_at(
        directory_descriptor,
        SIDECAR_NAME,
        4096,
        "closing published attestation sidecar",
    )
    after = os.fstat(directory_descriptor)
    directory_fields = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if (
        directory_fields(before) != directory_fields(after)
        or manifest != closing_manifest
        or sidecar != closing_sidecar
        or manifest_identity != closing_manifest_identity
        or sidecar_identity != closing_sidecar_identity
    ):
        _fail("published attestation changed during its final closed-tree scan")
    return manifest, sidecar, _directory_identity(directory_descriptor)


def _remove_staging_directory_at(
    parent_descriptor: int,
    staging_descriptor: int,
    staging_name: str,
) -> None:
    reopened = -1
    try:
        try:
            reopened = os.open(
                staging_name,
                _directory_open_flags(),
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError:
            return
        if _directory_identity(reopened) != _directory_identity(staging_descriptor):
            _fail("staging directory identity changed before cleanup")
        for name in os.listdir(reopened):
            os.unlink(name, dir_fd=reopened)
        os.fsync(reopened)
        os.rmdir(staging_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        if reopened >= 0:
            os.close(reopened)


def publish_returning_radiation_live_replay_attestation(
    output_directory: Path | str,
    *,
    spectral_publication: SpectralProductPublication,
    sampler: KerrReturningRadiationFiniteThicknessRaySampler,
    spectral_schema_path: Path | str,
) -> ReturningRadiationLiveReplayAttestationPublication:
    """Run canonical live replay and publish one immutable sibling directory."""

    _validate_spectral_publication(spectral_publication)
    if type(sampler) is not KerrReturningRadiationFiniteThicknessRaySampler:
        raise TypeError("sampler must have exact returning-radiation sampler type")
    output = Path(output_directory).absolute()
    if output == output.parent or output.name in ("", ".", ".."):
        raise TypeError("attestation output must be an absolute non-root path")
    if (
        _paths_equal_or_nested(output, spectral_publication.output_directory)
        or _paths_equal_or_nested(output, spectral_publication.manifest_path)
    ):
        raise ValueError(
            "attestation and spectral publication paths must be non-nested"
        )
    spectral_schema = Path(spectral_schema_path).absolute()

    parent_chain = _open_absolute_directory_chain(
        output.parent,
        "attestation parent",
    )
    parent_descriptor = parent_chain[-1]
    parent_chain_identity = _directory_chain_identity(parent_chain)
    staging_name = f".{output.name}.staging-{uuid.uuid4().hex}"
    staging_descriptor = -1
    staging_exists = False
    promoted = False
    try:
        _assert_absent_at(parent_descriptor, output.name, output)
        initial_sampler_descriptor, initial_sampler_snapshot = (
            _validated_sampler_live_identity(sampler)
        )
        structural_entry = _canonical_structural_verifier_entry()
        entry = _canonical_live_replay_entry()
        report = validated_live_replay_report(
            entry(
                spectral_publication.manifest_path,
                sampler=sampler,
                schema_path=spectral_schema,
            ),
            spectral_publication,
        )
        if _canonical_live_replay_entry() is not entry:
            _fail("returning live-replay callable identity changed")
        post_replay_descriptor, post_replay_snapshot = (
            _validated_sampler_live_identity(sampler)
        )
        if (
            post_replay_descriptor != initial_sampler_descriptor
            or post_replay_snapshot is not initial_sampler_snapshot
        ):
            _fail("returning sampler live identity changed during live replay")
        manifest_payload = read_stable_bounded_file(
            spectral_publication.manifest_path,
            MAXIMUM_SPECTRAL_MANIFEST_BYTES,
            "spectral manifest after live replay",
        )
        artifact = _manifest_claims(spectral_publication, manifest_payload, report)
        payload = canonical_json_bytes(artifact)
        if len(payload) > MAXIMUM_ATTESTATION_MANIFEST_BYTES:
            _fail("attestation manifest exceeds its fixed byte limit")
        digest = _sha256(payload)
        sidecar = f"{digest}  {MANIFEST_NAME}\n".encode("ascii")

        _assert_directory_path_identity(
            output.parent,
            parent_chain_identity,
            "attestation parent",
        )
        _assert_absent_at(parent_descriptor, output.name, output)
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_descriptor)
        staging_exists = True
        staging_descriptor = os.open(
            staging_name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
        _write_exclusive_at(staging_descriptor, MANIFEST_NAME, payload)
        _write_exclusive_at(staging_descriptor, SIDECAR_NAME, sidecar)
        os.fsync(staging_descriptor)
        staged_manifest, staged_sidecar, _staged_identity = (
            _closed_published_directory_snapshot(staging_descriptor)
        )
        if staged_manifest != payload or staged_sidecar != sidecar:
            _fail("staged attestation bytes differ from their canonical payload")
        _promote_directory_no_replace_at(
            parent_descriptor,
            staging_name,
            output.name,
            output,
        )
        staging_exists = False
        promoted = True
        os.fsync(parent_descriptor)

        structural_report = _validated_structural_subject(
            spectral_publication,
            spectral_schema,
        )
        if _canonical_structural_verifier_entry() is not structural_entry:
            _fail("spectral structural-verifier callable identity changed")

        published_manifest, published_sidecar, published_identity = (
            _closed_published_directory_snapshot(staging_descriptor)
        )
        final_subject = read_stable_bounded_file(
            spectral_publication.manifest_path,
            MAXIMUM_SPECTRAL_MANIFEST_BYTES,
            "spectral manifest after attestation promotion",
        )
        final_sampler_descriptor, final_sampler_snapshot = (
            _validated_sampler_live_identity(sampler)
        )
        if (
            published_manifest != payload
            or published_sidecar != sidecar
            or final_subject != manifest_payload
            or _manifest_claims(spectral_publication, final_subject, report)
            != artifact
            or final_sampler_descriptor != initial_sampler_descriptor
            or final_sampler_snapshot is not initial_sampler_snapshot
            or _canonical_live_replay_entry() is not entry
        ):
            _fail("attestation final subject, sampler, or published bytes changed")

        _assert_directory_path_identity(
            output.parent,
            parent_chain_identity,
            "attestation parent",
        )
        reopened_chain = _open_absolute_directory_chain(
            output,
            "published attestation",
        )
        try:
            if (
                _directory_chain_identity(reopened_chain[:-1])
                != parent_chain_identity
                or _directory_identity(reopened_chain[-1]) != published_identity
            ):
                _fail("published attestation path identity changed")
            reopened_manifest, reopened_sidecar, reopened_identity = (
                _closed_published_directory_snapshot(reopened_chain[-1])
            )
        finally:
            for descriptor in reversed(reopened_chain):
                os.close(descriptor)
        closing_subject = read_stable_bounded_file(
            spectral_publication.manifest_path,
            MAXIMUM_SPECTRAL_MANIFEST_BYTES,
            "closing spectral manifest after attestation promotion",
        )
        closing_sampler_descriptor, closing_sampler_snapshot = (
            _validated_sampler_live_identity(sampler)
        )
        closing_manifest, closing_sidecar, closing_identity = (
            _closed_published_directory_snapshot(staging_descriptor)
        )
        if (
            reopened_manifest != payload
            or reopened_sidecar != sidecar
            or reopened_identity != published_identity
            or closing_subject != manifest_payload
            or closing_sampler_descriptor != initial_sampler_descriptor
            or closing_sampler_snapshot is not initial_sampler_snapshot
            or closing_manifest != payload
            or closing_sidecar != sidecar
            or closing_identity != published_identity
            or _canonical_live_replay_entry() is not entry
        ):
            _fail("attestation changed during its final stable closure")
        _assert_directory_path_identity(
            output.parent,
            parent_chain_identity,
            "closing attestation parent",
        )
        closing_path_chain = _open_absolute_directory_chain(
            output,
            "closing published attestation",
        )
        try:
            if (
                _directory_chain_identity(closing_path_chain[:-1])
                != parent_chain_identity
                or _directory_identity(closing_path_chain[-1])
                != published_identity
            ):
                _fail("closing published attestation path identity changed")
        finally:
            for descriptor in reversed(closing_path_chain):
                os.close(descriptor)
        closing_structural_report = _validated_structural_subject(
            spectral_publication,
            spectral_schema,
        )
        post_structural_subject = read_stable_bounded_file(
            spectral_publication.manifest_path,
            MAXIMUM_SPECTRAL_MANIFEST_BYTES,
            "post-structural closing spectral manifest",
        )
        if (
            closing_structural_report != structural_report
            or post_structural_subject != manifest_payload
            or _canonical_live_replay_entry() is not entry
            or _canonical_structural_verifier_entry() is not structural_entry
        ):
            _fail("spectral subject changed across final structural closure")
    except BaseException:
        if staging_exists and staging_descriptor >= 0:
            try:
                _remove_staging_directory_at(
                    parent_descriptor,
                    staging_descriptor,
                    staging_name,
                )
            except (OSError, ReturningRadiationLiveReplayAttestationError):
                pass
        raise
    finally:
        if staging_descriptor >= 0:
            os.close(staging_descriptor)
        for descriptor in reversed(parent_chain):
            os.close(descriptor)
    if not promoted:
        _fail("attestation was not atomically promoted")
    return ReturningRadiationLiveReplayAttestationPublication(
        output_directory=output,
        manifest_path=output / MANIFEST_NAME,
        manifest_sha256=digest,
        attestation_id=artifact["id"],
        spectral_manifest_sha256=spectral_publication.manifest_sha256,
        live_replay_report=report,
    )


_PUBLISH_CANONICAL_ENTRY: Final = (
    publish_returning_radiation_live_replay_attestation
)


__all__ = (
    "ATTESTATION_DIRECTORY_SUFFIX",
    "ATTESTATION_SCHEMA",
    "ATTESTATION_SCHEMA_ID",
    "DEFAULT_ATTESTATION_SCHEMA",
    "MANIFEST_NAME",
    "MAXIMUM_ATTESTATION_MANIFEST_BYTES",
    "MAXIMUM_ATTESTATION_SCHEMA_BYTES",
    "MAXIMUM_JSON_NESTING_DEPTH",
    "MAXIMUM_SPECTRAL_MANIFEST_BYTES",
    "ReturningRadiationLiveReplayAttestationError",
    "ReturningRadiationLiveReplayAttestationPublication",
    "SIDECAR_NAME",
    "default_live_replay_attestation_directory",
    "publish_returning_radiation_live_replay_attestation",
    "read_stable_bounded_file",
    "validated_live_replay_report",
)
