"""Deterministic offline spectral composition for authenticated v1 ray endpoints.

This module is deliberately narrower than an NR or GR radiative-transfer
renderer.  It consumes a validated ``blackhole.nr-transfer-map/v1`` vacuum
endpoint product and applies Liouville's vacuum specific-intensity invariant to
an externally supplied spectral environment.  It does not integrate geodesics,
emission, absorption, plasma, or polarization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import stat
import struct
from typing import Any, Collection, Final, Mapping, Protocol, Sequence, runtime_checkable

from scripts.verify_nr_contract import validate_contract


OUTPUT_SCHEMA: Final = "blackhole.offline-vacuum-spectral/v1"
COMPOSITOR_ALGORITHM: Final = "vacuum-Liouville-specific-intensity/v1"
TRANSFER_SCHEMA: Final = "blackhole.nr-transfer-map/v1"
TRANSFER_RECORD: Final = struct.Struct("<7fBBH")
FLOAT32: Final = struct.Struct("<f")
RECORD_BYTES: Final = TRANSFER_RECORD.size
CANONICAL_NAN_FLOAT32: Final = bytes.fromhex("0000c07f")

OUTCOME_ESCAPED: Final = 0
OUTCOME_CAPTURED: Final = 1
OUTCOME_UNRESOLVED: Final = 2
OUTCOME_OUTSIDE_DOMAIN: Final = 3
OUTCOME_INTEGRATOR_FAILURE: Final = 4
OUTCOME_MISSING: Final = 255

VALID_DIRECTION: Final = 1 << 0
VALID_FREQUENCY_SHIFT: Final = 1 << 1

# Exact SI constants after the 2019 SI redefinition.
PLANCK_CONSTANT_J_S: Final = 6.62607015e-34
LIGHT_SPEED_M_S: Final = 299_792_458.0
BOLTZMANN_CONSTANT_J_K: Final = 1.380649e-23


class VacuumRenderError(RuntimeError):
    """Raised when a validated endpoint product cannot be composed safely."""


@runtime_checkable
class SpectralEnvironment(Protocol):
    """A distant source of emitted-frame spectral specific intensity.

    Returned values are ``I_nu`` in
    ``W m^-2 sr^-1 Hz^-1`` at ``emitted_frequency_hz``.  The direction is a
    canonical outgoing ICRS unit vector.  A descriptor must bind frequency to
    the transfer map's finite escape-boundary reference observer and direction
    to its declared ICRS continuation; neither is silently reinterpreted as a
    source at infinity.
    """

    def specific_intensity_nu(
        self,
        emitted_frequency_hz: float,
        direction_icrs: tuple[float, float, float],
    ) -> float:
        """Return emitted-frame spectral specific intensity."""

    def descriptor(self) -> Mapping[str, Any]:
        """Return deterministic JSON metadata for the output audit manifest."""


@dataclass(frozen=True)
class PlanckBlackbodyEnvironment:
    """An isotropic analytic Planck ``B_nu`` environment.

    ``normalization`` is a dimensionless multiplier.  The analytic environment
    is intentionally useful for scientific invariants and deterministic tests;
    it does not stand in for the repository's native-resolution photographic
    skies.
    """

    temperature_k: float
    normalization: float = 1.0

    def __post_init__(self) -> None:
        _positive_finite(self.temperature_k, "temperature_k")
        _positive_finite(self.normalization, "normalization")

    def specific_intensity_nu(
        self,
        emitted_frequency_hz: float,
        direction_icrs: tuple[float, float, float],
    ) -> float:
        del direction_icrs  # The declared environment is isotropic.
        frequency = _positive_finite(
            emitted_frequency_hz,
            "emitted_frequency_hz",
        )
        exponent = (
            PLANCK_CONSTANT_J_S
            * frequency
            / (BOLTZMANN_CONSTANT_J_K * self.temperature_k)
        )

        # Evaluate in log space so valid extreme frequencies underflow to zero
        # instead of overflowing an intermediate nu**3 or exp(x).
        log_denominator = (
            exponent
            if exponent > 50.0
            else math.log(math.expm1(exponent))
        )
        log_intensity = (
            math.log(2.0 * PLANCK_CONSTANT_J_S)
            - 2.0 * math.log(LIGHT_SPEED_M_S)
            + 3.0 * math.log(frequency)
            - log_denominator
            + math.log(self.normalization)
        )
        if log_intensity < -745.0:
            return 0.0
        if log_intensity > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
            raise VacuumRenderError("Planck environment intensity overflowed float64")
        intensity = math.exp(log_intensity)
        if not math.isfinite(intensity) or intensity < 0.0:
            raise VacuumRenderError("Planck environment returned invalid intensity")
        return intensity

    def descriptor(self) -> Mapping[str, Any]:
        return {
            "directionFrame": "ICRS-stored-escape-direction",
            "frequencyFrame": "transfer-map-escape-boundary-reference-observer",
            "implementationId": "Planck-B_nu-exact-SI-constants/v1",
            "kind": "isotropic-planck-blackbody",
            "normalization": self.normalization,
            "quantity": "spectral-specific-intensity-I_nu",
            "temperatureK": self.temperature_k,
            "units": "W m^-2 sr^-1 Hz^-1",
        }


@dataclass(frozen=True)
class TransferRecord:
    """Immutable decoded representation of the canonical 32-byte v1 ABI."""

    escape_direction_icrs: tuple[float, float, float]
    frequency_shift_g: float
    coordinate_lookback_time_m: float
    null_residual: float
    projection_error_px: float
    ray_outcome: int
    capture_target: int
    validity_mask: int
    raw_bytes: bytes


@dataclass(frozen=True)
class SpectralSample:
    """One composed spectrum together with its unmodified terminal ray state."""

    specific_intensity_nu: tuple[float, ...]
    ray_outcome: int
    capture_target: int
    validity_mask: int


@dataclass(frozen=True)
class VacuumRenderResult:
    """Paths and deterministic identity of a completed offline composition."""

    output_directory: Path
    manifest_path: Path
    manifest_sha256: str
    record_count: int
    chunk_count: int


def _positive_finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def validate_observer_frequencies(
    values: Sequence[float],
) -> tuple[float, ...]:
    """Return a finite, positive, strictly increasing frequency-bin tuple."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("observer frequencies must be a sequence")
    frequencies = tuple(
        _positive_finite(value, f"observer_frequencies_hz[{index}]")
        for index, value in enumerate(values)
    )
    if not frequencies:
        raise ValueError("at least one observer-frequency bin is required")
    if any(current <= previous for previous, current in zip(frequencies, frequencies[1:])):
        raise ValueError("observer-frequency bins must increase strictly")
    return frequencies


def decode_transfer_record(payload: bytes | bytearray | memoryview) -> TransferRecord:
    """Decode one canonical transfer record and retain an immutable byte copy."""

    raw = bytes(payload)
    if len(raw) != RECORD_BYTES:
        raise VacuumRenderError(
            f"a v1 transfer record must contain exactly {RECORD_BYTES} bytes"
        )
    (
        direction_x,
        direction_y,
        direction_z,
        frequency_shift,
        coordinate_lookback,
        null_residual,
        projection_error,
        outcome,
        capture_target,
        validity_mask,
    ) = TRANSFER_RECORD.unpack(raw)
    return TransferRecord(
        escape_direction_icrs=(direction_x, direction_y, direction_z),
        frequency_shift_g=frequency_shift,
        coordinate_lookback_time_m=coordinate_lookback,
        null_residual=null_residual,
        projection_error_px=projection_error,
        ray_outcome=outcome,
        capture_target=capture_target,
        validity_mask=validity_mask,
        raw_bytes=raw,
    )


def _escaped_transport_is_valid(record: TransferRecord) -> bool:
    if record.ray_outcome != OUTCOME_ESCAPED:
        return False
    required = VALID_DIRECTION | VALID_FREQUENCY_SHIFT
    if record.validity_mask & required != required:
        return False
    if (
        not math.isfinite(record.frequency_shift_g)
        or record.frequency_shift_g <= 0.0
        or any(not math.isfinite(value) for value in record.escape_direction_icrs)
    ):
        return False
    norm = math.sqrt(
        math.fsum(value * value for value in record.escape_direction_icrs)
    )
    return math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6)


def _record_state_is_self_consistent(
    record: TransferRecord,
    valid_capture_targets: frozenset[int] | None,
) -> bool:
    """Apply the immutable v1 state machine for the public record API.

    Full-dataset rendering has already passed the stronger contract validator.
    This local gate prevents callers of ``compose_vacuum_spectrum`` from
    turning a hand-constructed or corrupted captured record into trusted black.
    """

    try:
        if decode_transfer_record(record.raw_bytes) != record:
            return False
    except (TypeError, ValueError, VacuumRenderError, struct.error):
        return False

    floats = (
        *record.escape_direction_icrs,
        record.frequency_shift_g,
        record.coordinate_lookback_time_m,
        record.null_residual,
        record.projection_error_px,
    )
    if any(not math.isfinite(value) for value in floats):
        return False
    expected_masks = {
        OUTCOME_ESCAPED: 0x1F,
        OUTCOME_CAPTURED: (1 << 2) | (1 << 3) | (1 << 4),
        OUTCOME_UNRESOLVED: (1 << 3) | (1 << 4),
        OUTCOME_OUTSIDE_DOMAIN: (1 << 2) | (1 << 3) | (1 << 4),
        OUTCOME_INTEGRATOR_FAILURE: 1 << 3,
        OUTCOME_MISSING: 0,
    }
    if record.validity_mask != expected_masks.get(record.ray_outcome):
        return False
    if record.ray_outcome == OUTCOME_CAPTURED:
        if (
            valid_capture_targets is None
            or record.capture_target not in valid_capture_targets
        ):
            return False
    elif record.capture_target != 255:
        return False

    validity_fields = (
        (VALID_DIRECTION, 0, 3),
        (VALID_FREQUENCY_SHIFT, 12, 1),
        (1 << 2, 16, 1),
        (1 << 3, 20, 1),
        (1 << 4, 24, 1),
    )
    for bit, offset, components in validity_fields:
        if record.validity_mask & bit:
            continue
        for component in range(components):
            start = offset + component * 4
            if record.raw_bytes[start : start + 4] != b"\x00\x00\x00\x00":
                return False
    if record.validity_mask & (1 << 2) and record.coordinate_lookback_time_m < 0.0:
        return False
    if record.validity_mask & (1 << 3) and record.null_residual < 0.0:
        return False
    if record.validity_mask & (1 << 4) and record.projection_error_px < 0.0:
        return False
    return record.ray_outcome != OUTCOME_ESCAPED or _escaped_transport_is_valid(record)


def _compose_validated_frequencies(
    record: TransferRecord,
    observer_frequencies_hz: tuple[float, ...],
    environment: SpectralEnvironment,
    valid_capture_targets: frozenset[int] | None = None,
) -> SpectralSample:
    state_is_valid = _record_state_is_self_consistent(
        record,
        valid_capture_targets,
    )
    if not state_is_valid:
        values = (math.nan,) * len(observer_frequencies_hz)
    elif record.ray_outcome == OUTCOME_CAPTURED:
        values = (0.0,) * len(observer_frequencies_hz)
    elif record.ray_outcome != OUTCOME_ESCAPED:
        values = (math.nan,) * len(observer_frequencies_hz)
    else:
        direction = record.escape_direction_icrs
        frequency_shift = record.frequency_shift_g
        shift_cubed = frequency_shift * frequency_shift * frequency_shift
        emitted_frequencies = tuple(
            observer_frequency / frequency_shift
            for observer_frequency in observer_frequencies_hz
        )
        if (
            not math.isfinite(shift_cubed)
            or shift_cubed <= 0.0
            or any(
                not math.isfinite(emitted_frequency) or emitted_frequency <= 0.0
                for emitted_frequency in emitted_frequencies
            )
        ):
            return SpectralSample(
                specific_intensity_nu=(math.nan,) * len(observer_frequencies_hz),
                ray_outcome=record.ray_outcome,
                capture_target=record.capture_target,
                validity_mask=record.validity_mask,
            )
        composed: list[float] = []
        for emitted_frequency in emitted_frequencies:
            emitted_intensity = environment.specific_intensity_nu(
                emitted_frequency,
                direction,
            )
            if (
                isinstance(emitted_intensity, bool)
                or not isinstance(emitted_intensity, (int, float))
                or not math.isfinite(float(emitted_intensity))
                or float(emitted_intensity) < 0.0
            ):
                raise VacuumRenderError(
                    "spectral environment returned a non-finite or negative I_nu"
                )
            observed_intensity = shift_cubed * float(emitted_intensity)
            if not math.isfinite(observed_intensity) or observed_intensity < 0.0:
                raise VacuumRenderError(
                    "vacuum frequency transport produced an invalid I_nu"
                )
            composed.append(observed_intensity)
        values = tuple(composed)
    return SpectralSample(
        specific_intensity_nu=values,
        ray_outcome=record.ray_outcome,
        capture_target=record.capture_target,
        validity_mask=record.validity_mask,
    )


def compose_vacuum_spectrum(
    record: TransferRecord,
    observer_frequencies_hz: Sequence[float],
    environment: SpectralEnvironment,
    *,
    valid_capture_targets: Collection[int] | None = None,
) -> SpectralSample:
    """Compose one observer-frame spectrum without sampling unusable rays.

    Escaped records use ``nu_emit = nu_obs / g`` and
    ``I_nu_obs = g^3 I_nu_emit``.  Captured records are exactly zero only when
    their target code belongs to an explicitly supplied, validated target set.
    Without dataset context a captured record is unusable rather than silently
    trusting a globally meaningless uint8 code.
    Unresolved, missing, failed, outside-domain, or transport-invalid records
    return NaNs while retaining their original categorical state.
    """

    frequencies = validate_observer_frequencies(observer_frequencies_hz)
    targets: frozenset[int] | None = None
    if valid_capture_targets is not None:
        targets = frozenset(valid_capture_targets)
        if any(
            type(value) is not int or value < 0 or value >= OUTCOME_MISSING
            for value in targets
        ):
            raise ValueError("capture target codes must be integers in [0, 254]")
    return _compose_validated_frequencies(
        record,
        frequencies,
        environment,
        targets,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_environment_descriptor(
    environment: SpectralEnvironment,
) -> dict[str, Any]:
    try:
        descriptor = dict(environment.descriptor())
        encoded = _canonical_json_bytes(descriptor)
        decoded = json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as error:
        raise VacuumRenderError(
            f"spectral environment descriptor is not finite JSON: {error}"
        ) from error
    if not isinstance(decoded, dict) or not decoded:
        raise VacuumRenderError("spectral environment descriptor must be an object")
    if decoded.get("frequencyFrame") != (
        "transfer-map-escape-boundary-reference-observer"
    ):
        raise VacuumRenderError(
            "spectral environment must bind frequency to the transfer-map "
            "escape-boundary reference observer"
        )
    if decoded.get("directionFrame") != "ICRS-stored-escape-direction":
        raise VacuumRenderError(
            "spectral environment must consume the stored ICRS escape direction"
        )
    if not isinstance(decoded.get("implementationId"), str) or not decoded[
        "implementationId"
    ].strip():
        raise VacuumRenderError(
            "spectral environment descriptor needs a stable implementationId"
        )
    if decoded.get("quantity") != "spectral-specific-intensity-I_nu":
        raise VacuumRenderError(
            "spectral environment must return spectral-specific-intensity I_nu"
        )
    if decoded.get("units") != "W m^-2 sr^-1 Hz^-1":
        raise VacuumRenderError(
            "spectral environment must return I_nu in W m^-2 sr^-1 Hz^-1"
        )
    return decoded


def _normalized_dataset_uri(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise VacuumRenderError(f"{label} must be a normalized relative POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or "." in pure.parts
        or ".." in pure.parts
        or pure.as_posix() != value
    ):
        raise VacuumRenderError(f"{label} must stay inside the dataset directory")
    return pure


def _copy_regular_file_without_following_final_symlink(
    source: Path,
    destination: Path,
) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise VacuumRenderError(f"unable to snapshot input artifact {source}: {error}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise VacuumRenderError(f"input artifact is not a regular file: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(descriptor, "rb", closefd=False) as input_stream:
            with destination.open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
    finally:
        os.close(descriptor)


def _snapshot_transfer_dataset(
    source_manifest_path: Path,
    snapshot_root: Path,
) -> Path:
    """Copy one coherent candidate dataset for a second full validation.

    The first validation authorizes the source path.  This copy uses only
    normalized relative URIs and the returned snapshot is validated again before
    any environment call, so manifest/chunk races cannot reach composition.
    """

    snapshot_root.mkdir(mode=0o700)
    manifest_destination = snapshot_root / "manifest.json"
    _copy_regular_file_without_following_final_symlink(
        source_manifest_path,
        manifest_destination,
    )
    try:
        manifest = json.loads(manifest_destination.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VacuumRenderError(f"snapshotted manifest is invalid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise VacuumRenderError("snapshotted manifest root must be an object")

    artifact_uris: set[str] = set()
    sidecar_uri = manifest.get("integrity", {}).get("manifestSidecar")
    artifact_uris.add(
        _normalized_dataset_uri(sidecar_uri, "manifest sidecar URI").as_posix()
    )
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list):
        raise VacuumRenderError("snapshotted manifest chunks must be an array")
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise VacuumRenderError(f"chunk {index} must be an object")
        artifact_uris.add(
            _normalized_dataset_uri(chunk.get("uri"), f"chunk {index} URI").as_posix()
        )

    provenance = manifest.get("provenance", {})
    if provenance.get("artifactUriBase") == "manifest-directory":
        for index, artifact in enumerate(provenance.get("sourceArtifacts", [])):
            if isinstance(artifact, dict) and artifact.get("storage") == "bundled":
                artifact_uris.add(
                    _normalized_dataset_uri(
                        artifact.get("uri"),
                        f"source artifact {index} URI",
                    ).as_posix()
                )

    source_root = source_manifest_path.parent
    for uri in sorted(artifact_uris):
        pure = PurePosixPath(uri)
        _copy_regular_file_without_following_final_symlink(
            source_root.joinpath(*pure.parts),
            snapshot_root.joinpath(*pure.parts),
        )
    return manifest_destination


def _float32_bytes(value: float) -> bytes:
    if math.isnan(value):
        return CANONICAL_NAN_FLOAT32
    if not math.isfinite(value) or value < 0.0:
        raise VacuumRenderError("output radiance must be finite, non-negative, or NaN")
    try:
        payload = FLOAT32.pack(value)
    except (OverflowError, struct.error) as error:
        raise VacuumRenderError("output radiance exceeds float32 range") from error
    if not math.isfinite(FLOAT32.unpack(payload)[0]):
        raise VacuumRenderError("output radiance quantized to non-finite float32")
    return payload


def _write_and_hash(stream: Any, digest: Any, payload: bytes) -> None:
    stream.write(payload)
    digest.update(payload)


def render_offline_vacuum(
    manifest_path: Path | str,
    output_directory: Path | str,
    observer_frequencies_hz: Sequence[float],
    environment: SpectralEnvironment,
) -> VacuumRenderResult:
    """Validate, stream, and compose a deterministic spectral endpoint product.

    Contract validation intentionally runs before output-path creation and before
    any environment call.  The authorized dataset is then copied through safe
    relative paths into a private snapshot and fully validated a second time.
    Only authenticated snapshot chunks may reach the environment.
    """

    source_manifest_path = Path(manifest_path).resolve()

    # This must remain the first dataset operation.  It authenticates the
    # sidecar, provenance, chunk hashes, frames, records, and scientific gates.
    initial_manifest_sha256 = _sha256_file(source_manifest_path)
    validate_contract(source_manifest_path)
    if _sha256_file(source_manifest_path) != initial_manifest_sha256:
        raise VacuumRenderError("input manifest changed during contract validation")

    output_path = Path(output_directory).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.mkdir()
    except FileExistsError as error:
        raise VacuumRenderError("output directory must not already exist") from error

    try:
        snapshot_root = output_path / ".input-snapshot"
        snapshot_manifest_path = _snapshot_transfer_dataset(
            source_manifest_path,
            snapshot_root,
        )
        validation_report = validate_contract(snapshot_manifest_path)
        if _sha256_file(snapshot_manifest_path) != initial_manifest_sha256:
            raise VacuumRenderError(
                "snapshotted manifest differs from the validated input manifest"
            )
        source_manifest_path = snapshot_manifest_path
        source_manifest_bytes = source_manifest_path.read_bytes()
        source_manifest_sha256 = hashlib.sha256(source_manifest_bytes).hexdigest()
        source_manifest = json.loads(source_manifest_bytes)
        if source_manifest.get("schema") != TRANSFER_SCHEMA:
            raise VacuumRenderError("only blackhole.nr-transfer-map/v1 is supported")
        if source_manifest.get("renderable") is not True:
            raise VacuumRenderError("non-renderable protocol fixtures cannot be composed")
        if source_manifest.get("physicalSystem", {}).get("vacuum") is not True:
            raise VacuumRenderError("offline vacuum composition requires vacuum=true")
        layout = source_manifest.get("recordLayout", {})
        if (
            layout.get("structFormat") != "<7fBBH"
            or layout.get("recordBytes") != RECORD_BYTES
        ):
            raise VacuumRenderError("input does not use the immutable 32-byte v1 ABI")

        frequencies = validate_observer_frequencies(observer_frequencies_hz)
        if not isinstance(environment, PlanckBlackbodyEnvironment):
            raise VacuumRenderError(
                "offline vacuum spectral v1 products require the independently "
                "verifiable PlanckBlackbodyEnvironment"
            )
        environment_descriptor = _canonical_environment_descriptor(environment)
        compositor = {
            "algorithm": COMPOSITOR_ALGORITHM,
            "entryPoint": "scripts/render_offline_vacuum.py",
            "numericBackend": (
                "CPython float64 evaluation with explicit little-endian "
                "IEEE-754 float32 output"
            ),
            "sourceSha256": _sha256_file(Path(__file__).resolve()),
            "sourceUri": "offline/vacuum.py",
        }

        spectra_directory = output_path / "spectral"
        states_directory = output_path / "states"
        spectra_directory.mkdir()
        states_directory.mkdir()

        outcome_names = {
            int(code): name
            for name, code in source_manifest["recordLayout"]["rayOutcomes"].items()
        }
        valid_capture_targets = frozenset(
            int(target["code"])
            for target in source_manifest["captureTargets"]
        )
        outcome_counts = {name: 0 for name in sorted(outcome_names.values())}
        output_chunks: list[dict[str, Any]] = []
        total_records = 0
        unusable_records = 0

        for chunk_index, chunk in enumerate(source_manifest["chunks"]):
            tile = chunk["tile"]
            stem = (
                f"t{int(chunk['sampleIndex']):04d}"
                f"-y{int(tile['y']):06d}"
                f"-x{int(tile['x']):06d}"
            )
            spectral_uri = f"spectral/{stem}.f32"
            state_uri = f"states/{stem}.bin"
            spectral_path = output_path / spectral_uri
            state_path = output_path / state_uri
            input_uri = _normalized_dataset_uri(
                chunk["uri"],
                f"validated chunk {chunk_index} URI",
            )
            input_path = source_manifest_path.parent.joinpath(*input_uri.parts)

            input_digest = hashlib.sha256()
            spectral_digest = hashlib.sha256()
            state_digest = hashlib.sha256()
            input_bytes = 0
            spectral_bytes = 0
            state_bytes = 0
            input_flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                input_flags |= os.O_NOFOLLOW
            try:
                input_descriptor = os.open(input_path, input_flags)
            except OSError as error:
                raise VacuumRenderError(
                    f"unable to open snapshotted chunk {chunk_index}: {error}"
                ) from error
            with os.fdopen(input_descriptor, "rb") as input_stream:
                input_stat = os.fstat(input_stream.fileno())
                if (
                    not stat.S_ISREG(input_stat.st_mode)
                    or input_stat.st_size != int(chunk["byteLength"])
                ):
                    raise VacuumRenderError(
                        f"chunk {chunk_index} is not the validated regular file"
                    )
                authenticated_digest = hashlib.sha256()
                while block := input_stream.read(1024 * 1024):
                    authenticated_digest.update(block)
                if authenticated_digest.hexdigest() != chunk["sha256"]:
                    raise VacuumRenderError(
                        f"chunk {chunk_index} hash changed after snapshot validation"
                    )
                input_stream.seek(0)

                # Authentication and composition share the same open file
                # description.  No path reopen may swap bytes between the hash
                # gate and the first stateful environment call.
                with (
                    spectral_path.open("wb") as spectral_stream,
                    state_path.open("wb") as state_stream,
                ):
                    for record_index in range(int(chunk["recordCount"])):
                        raw_record = input_stream.read(RECORD_BYTES)
                        if len(raw_record) != RECORD_BYTES:
                            raise VacuumRenderError(
                                f"chunk {chunk_index} truncated at record {record_index}"
                            )
                        input_digest.update(raw_record)
                        input_bytes += len(raw_record)
                        record = decode_transfer_record(raw_record)
                        sample = _compose_validated_frequencies(
                            record,
                            frequencies,
                            environment,
                            valid_capture_targets,
                        )
                        outcome_name = outcome_names.get(record.ray_outcome)
                        if outcome_name is None:
                            raise VacuumRenderError(
                                "validated chunk contains unknown outcome "
                                f"{record.ray_outcome}"
                            )
                        outcome_counts[outcome_name] += 1
                        if any(
                            math.isnan(value)
                            for value in sample.specific_intensity_nu
                        ):
                            unusable_records += 1

                        for value in sample.specific_intensity_nu:
                            payload = _float32_bytes(value)
                            _write_and_hash(
                                spectral_stream,
                                spectral_digest,
                                payload,
                            )
                            spectral_bytes += len(payload)
                        # Retain the authenticated v1 state byte-for-byte.  This
                        # is stronger than copying only outcome/mask fields and
                        # keeps every terminal diagnostic without re-encoding.
                        state_payload = record.raw_bytes
                        _write_and_hash(state_stream, state_digest, state_payload)
                        state_bytes += len(state_payload)

                    if input_stream.read(1):
                        raise VacuumRenderError(
                            f"chunk {chunk_index} has trailing bytes"
                        )

            if input_bytes != int(chunk["byteLength"]):
                raise VacuumRenderError(
                    f"chunk {chunk_index} byte length changed after validation"
                )
            if input_digest.hexdigest() != chunk["sha256"]:
                raise VacuumRenderError(
                    f"chunk {chunk_index} hash changed after validation"
                )
            expected_spectral_bytes = (
                int(chunk["recordCount"]) * len(frequencies) * FLOAT32.size
            )
            expected_state_bytes = int(chunk["recordCount"]) * RECORD_BYTES
            if (
                spectral_bytes != expected_spectral_bytes
                or state_bytes != expected_state_bytes
            ):
                raise VacuumRenderError("internal output record-layout mismatch")

            total_records += int(chunk["recordCount"])
            output_chunks.append(
                {
                    "inputChunkSha256": chunk["sha256"],
                    "recordCount": int(chunk["recordCount"]),
                    "sampleIndex": int(chunk["sampleIndex"]),
                    "spectral": {
                        "byteLength": spectral_bytes,
                        "sha256": spectral_digest.hexdigest(),
                        "uri": spectral_uri,
                    },
                    "state": {
                        "byteLength": state_bytes,
                        "sha256": state_digest.hexdigest(),
                        "uri": state_uri,
                    },
                    "tile": {
                        "height": int(tile["height"]),
                        "width": int(tile["width"]),
                        "x": int(tile["x"]),
                        "y": int(tile["y"]),
                    },
                }
            )

        shutil.rmtree(snapshot_root)

        output_record_layout = {
            "spectral": {
                "byteOrder": "little-endian",
                "componentType": "float32",
                "componentsPerRecord": len(frequencies),
                "invalidEncoding": "canonical-quiet-NaN-0x7fc00000",
                "order": "record-major-then-observer-frequency-bin",
                "units": "W m^-2 sr^-1 Hz^-1",
            },
            "state": {
                "byteOrder": "little-endian",
                "policy": "byte-exact-authenticated-input-record-copy",
                "recordBytes": RECORD_BYTES,
                "schema": TRANSFER_SCHEMA,
                "structFormat": "<7fBBH",
            },
        }
        configuration = {
            "compositor": compositor,
            "environment": environment_descriptor,
            "inputManifestSha256": source_manifest_sha256,
            "observerFrequencyBinsHz": list(frequencies),
            "schema": OUTPUT_SCHEMA,
        }
        configuration_sha256 = hashlib.sha256(
            _canonical_json_bytes(configuration)
        ).hexdigest()
        product_identity = {
            "configurationSha256": configuration_sha256,
            "outputChunks": output_chunks,
            "recordLayout": output_record_layout,
        }
        product_digest = hashlib.sha256(
            _canonical_json_bytes(product_identity)
        ).hexdigest()
        product_id = f"offline-vacuum-{product_digest[:20]}"

        manifest_document = {
            "schema": OUTPUT_SCHEMA,
            "id": product_id,
            "scientificStatus": {
                "classification": "validated vacuum endpoint spectral composition",
                "description": (
                    "Observer-frequency I_nu bins composed from an authenticated "
                    "v1 vacuum transfer map and a declared spectral environment."
                ),
                "isNumericalRelativitySolver": False,
                "isGeneralRelativisticRadiativeTransfer": False,
                "isOpenExrScientificMaster": False,
                "prohibitedClaim": (
                    "Do not describe this compositor as an NR spacetime solver, "
                    "GRRT plasma calculation, or OpenEXR scientific master."
                ),
            },
            "inputTransferMap": {
                "datasetKind": source_manifest["datasetKind"],
                "id": source_manifest["id"],
                "manifestSha256": source_manifest_sha256,
                "schema": source_manifest["schema"],
                "scientificStatus": source_manifest["scientificStatus"],
                "validationStatus": validation_report["status"],
            },
            "compositor": compositor,
            "environment": environment_descriptor,
            "observerFrequencyBinsHz": list(frequencies),
            "transport": {
                "emittedFrequency": "nu_emit=nu_obs/g",
                "escapeBoundaryReferenceObserver": source_manifest[
                    "escapeBoundary"
                ]["referenceObserver"],
                "frequencyShiftConvention": source_manifest["escapeBoundary"][
                    "frequencyShiftConvention"
                ],
                "mode": COMPOSITOR_ALGORITHM,
                "observedSpecificIntensity": "I_nu_obs=g^3*I_nu_emit",
                "samplingPolicy": (
                    "sample environment only for escaped records with valid "
                    "ICRS direction and positive frequency shift"
                ),
                "storedEscapeDirection": source_manifest["escapeBoundary"][
                    "storedEscapeDirection"
                ],
            },
            "projection": {
                "heightPixels": int(source_manifest["projection"]["heightPixels"]),
                "imageOrigin": source_manifest["projection"]["imageOrigin"],
                "pixelSampleLocation": source_manifest["projection"][
                    "pixelSampleLocation"
                ],
                "widthPixels": int(source_manifest["projection"]["widthPixels"]),
            },
            "sampling": {
                "dimensionOrder": source_manifest["sampling"]["dimensionOrder"],
                "observationTimesM": source_manifest["sampling"][
                    "observationTimesM"
                ],
                "pixelOrder": source_manifest["sampling"]["pixelOrder"],
            },
            "recordLayout": output_record_layout,
            "outcomes": {
                "counts": outcome_counts,
                "policy": {
                    "captured": "all spectral bins are positive float32 zero",
                    "escaped": "vacuum spectral composition",
                    "unusable": "all spectral bins are canonical float32 NaN",
                },
                "recordCount": total_records,
                "unusableRecordCount": unusable_records,
            },
            "integrity": {
                "configurationSha256": configuration_sha256,
                "manifestSidecar": "manifest.sha256",
                "productSha256": product_digest,
            },
            "chunks": output_chunks,
        }
        manifest_bytes = _canonical_json_bytes(manifest_document)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        (output_path / "manifest.json").write_bytes(manifest_bytes)
        (output_path / "manifest.sha256").write_bytes(
            f"{manifest_sha256}  manifest.json\n".encode("ascii")
        )
    except BaseException:
        shutil.rmtree(output_path, ignore_errors=True)
        raise

    return VacuumRenderResult(
        output_directory=output_path,
        manifest_path=output_path / "manifest.json",
        manifest_sha256=manifest_sha256,
        record_count=total_records,
        chunk_count=len(output_chunks),
    )


__all__ = [
    "CANONICAL_NAN_FLOAT32",
    "OUTCOME_CAPTURED",
    "OUTCOME_ESCAPED",
    "OUTCOME_INTEGRATOR_FAILURE",
    "OUTCOME_MISSING",
    "OUTCOME_OUTSIDE_DOMAIN",
    "OUTCOME_UNRESOLVED",
    "OUTPUT_SCHEMA",
    "PlanckBlackbodyEnvironment",
    "SpectralEnvironment",
    "SpectralSample",
    "TransferRecord",
    "VALID_DIRECTION",
    "VALID_FREQUENCY_SHIFT",
    "VacuumRenderError",
    "VacuumRenderResult",
    "compose_vacuum_spectrum",
    "decode_transfer_record",
    "render_offline_vacuum",
    "validate_observer_frequencies",
]
