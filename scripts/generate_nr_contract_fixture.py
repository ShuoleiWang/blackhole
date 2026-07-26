#!/usr/bin/env python3
"""Generate the deterministic, non-renderable NR transfer-map contract fixture."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Final, Iterable


ROOT: Final = Path(__file__).resolve().parents[1]
SCHEMA_PATH: Final = ROOT / "schemas" / "nr-transfer-map-v1.schema.json"
OUTPUT_DIR: Final = (
    ROOT / "assets" / "transfer-maps" / "contract-fixture-v1"
)
CHUNK_DIR: Final = OUTPUT_DIR / "chunks"
CHUNK_NAME: Final = "t0000-y0000-x0000.bin"
CHUNK_PATH: Final = CHUNK_DIR / CHUNK_NAME
MANIFEST_PATH: Final = OUTPUT_DIR / "manifest.json"
MANIFEST_HASH_PATH: Final = OUTPUT_DIR / "manifest.sha256"

RECORD: Final = struct.Struct("<7fBBH")
RECORD_BYTES: Final = 32
WIDTH: Final = 4
HEIGHT: Final = 2

OUTCOME_ESCAPED: Final = 0
OUTCOME_CAPTURED: Final = 1
OUTCOME_UNRESOLVED: Final = 2
OUTCOME_OUTSIDE_DOMAIN: Final = 3
OUTCOME_INTEGRATOR_FAILURE: Final = 4
OUTCOME_MISSING: Final = 255

CAPTURE_A: Final = 0
CAPTURE_B: Final = 1
CAPTURE_NONE: Final = 255

VALID_DIRECTION: Final = 1 << 0
VALID_FREQUENCY_SHIFT: Final = 1 << 1
VALID_COORDINATE_LOOKBACK_TIME: Final = 1 << 2
VALID_NULL_RESIDUAL: Final = 1 << 3
VALID_PROJECTION_ERROR: Final = 1 << 4
VALID_ALL: Final = (
    VALID_DIRECTION
    | VALID_FREQUENCY_SHIFT
    | VALID_COORDINATE_LOOKBACK_TIME
    | VALID_NULL_RESIDUAL
    | VALID_PROJECTION_ERROR
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_artifact(role: str, path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "role": role,
        "storage": "bundled",
        "uri": path.relative_to(ROOT).as_posix(),
        "byteLength": len(payload),
        "sha256": sha256_bytes(payload),
    }


def pack_record(
    escape_direction: tuple[float, float, float],
    frequency_shift_g: float,
    coordinate_lookback_time_m: float,
    null_residual: float,
    projection_error_px: float,
    ray_outcome: int,
    capture_target: int,
    validity_mask: int,
) -> bytes:
    values: Iterable[float] = (
        *escape_direction,
        frequency_shift_g,
        coordinate_lookback_time_m,
        null_residual,
        projection_error_px,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("fixture records must contain only finite floats")
    if validity_mask & VALID_COORDINATE_LOOKBACK_TIME and coordinate_lookback_time_m < 0:
        raise ValueError("a valid coordinate lookback time must be non-negative")
    if validity_mask & VALID_DIRECTION:
        norm = math.sqrt(sum(component * component for component in escape_direction))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-7):
            raise ValueError("a valid escape direction must be normalized")
    elif escape_direction != (0.0, 0.0, 0.0):
        raise ValueError("an invalid escape direction must be positive zero")
    return RECORD.pack(
        *escape_direction,
        frequency_shift_g,
        coordinate_lookback_time_m,
        null_residual,
        projection_error_px,
        ray_outcome,
        capture_target,
        validity_mask,
    )


def fixture_chunk() -> bytes:
    """Return two rows of four records covering every terminal outcome."""
    records = [
        pack_record(
            (0.0, 0.0, 1.0),
            1.0,
            42.0,
            1.0e-10,
            0.02,
            OUTCOME_ESCAPED,
            CAPTURE_NONE,
            VALID_ALL,
        ),
        pack_record(
            (0.6, 0.0, 0.8),
            0.92,
            48.0,
            2.0e-10,
            0.03,
            OUTCOME_ESCAPED,
            CAPTURE_NONE,
            VALID_ALL,
        ),
        pack_record(
            (0.0, 0.0, 0.0),
            0.0,
            16.0,
            3.0e-10,
            0.04,
            OUTCOME_CAPTURED,
            CAPTURE_A,
            VALID_COORDINATE_LOOKBACK_TIME | VALID_NULL_RESIDUAL | VALID_PROJECTION_ERROR,
        ),
        pack_record(
            (0.0, 0.0, 0.0),
            0.0,
            17.0,
            4.0e-10,
            0.05,
            OUTCOME_CAPTURED,
            CAPTURE_B,
            VALID_COORDINATE_LOOKBACK_TIME | VALID_NULL_RESIDUAL | VALID_PROJECTION_ERROR,
        ),
        pack_record(
            (0.0, 0.0, 0.0),
            0.0,
            0.0,
            5.0e-7,
            0.8,
            OUTCOME_UNRESOLVED,
            CAPTURE_NONE,
            VALID_NULL_RESIDUAL | VALID_PROJECTION_ERROR,
        ),
        pack_record(
            (0.0, 0.0, 0.0),
            0.0,
            24.0,
            6.0e-9,
            1.2,
            OUTCOME_OUTSIDE_DOMAIN,
            CAPTURE_NONE,
            VALID_COORDINATE_LOOKBACK_TIME | VALID_NULL_RESIDUAL | VALID_PROJECTION_ERROR,
        ),
        pack_record(
            (0.0, 0.0, 0.0),
            0.0,
            0.0,
            2.0e-4,
            0.0,
            OUTCOME_INTEGRATOR_FAILURE,
            CAPTURE_NONE,
            VALID_NULL_RESIDUAL,
        ),
        pack_record(
            (0.0, 0.0, 0.0),
            0.0,
            0.0,
            0.0,
            0.0,
            OUTCOME_MISSING,
            CAPTURE_NONE,
            0,
        ),
    ]
    payload = b"".join(records)
    expected_bytes = WIDTH * HEIGHT * RECORD_BYTES
    if RECORD.size != RECORD_BYTES or len(payload) != expected_bytes:
        raise AssertionError(
            f"fixture layout mismatch: struct={RECORD.size}, payload={len(payload)}, "
            f"expected={expected_bytes}"
        )
    return payload


def manifest(chunk: bytes) -> dict[str, object]:
    identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    identity3 = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]
    camera_to_world = [
        1.0, 0.0, 0.0, 4.0,
        0.0, 1.0, 0.0, 3.0,
        0.0, 0.0, 1.0, 12.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    world_to_camera = [
        1.0, 0.0, 0.0, -4.0,
        0.0, 1.0, 0.0, -3.0,
        0.0, 0.0, 1.0, -12.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    minkowski_metric = [
        -1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]

    return {
        "schema": "blackhole.nr-transfer-map/v1",
        "id": "nr-contract-fixture-v1",
        "datasetKind": "synthetic-contract-fixture",
        "renderable": False,
        "scientificStatus": {
            "classification": "project-generated protocol fixture; not NR",
            "sourceIsNumericalRelativity": False,
            "derivedFromNearZoneSpacetime": False,
            "derivedWithSlowLightGeodesics": False,
            "description": (
                "A deterministic 2x4 record-layout fixture for validating data "
                "contracts, hashes, coordinate transforms, and invalid-ray handling."
            ),
            "prohibitedClaim": (
                "Do not render this fixture or describe it as a physical "
                "binary-black-hole simulation."
            ),
        },
        "physicalSystem": {
            "kind": "synthetic-contract-fixture",
            "vacuum": None,
            "componentIds": ["A", "B"],
            "parameterEpochProtocolM": None,
            "massRatioQ": None,
            "dimensionlessSpins": [],
            "eccentricity": None,
            "referenceOrbitalPhaseRad": None,
            "remnant": None,
            "notApplicableReason": (
                "The protocol fixture has capture identifiers but no physical "
                "binary parameters or remnant."
            ),
            "description": (
                "Two synthetic identifiers exercise capture classification; "
                "they are not physical black holes."
            ),
        },
        "provenance": {
            "origin": "project-generated",
            "project": "ShuoleiWang/blackhole",
            "datasetVersion": "1.0.0",
            "license": "NOASSERTION",
            "artifactUriBase": "repository-root",
            "sourceSimulation": {
                "kind": "none",
                "catalog": None,
                "identifier": "project-generated-nr-contract-fixture",
                "version": "1",
                "doi": None,
                "evolutionCode": None,
                "notApplicableReason": (
                    "The fixture is synthetic protocol test data and has no "
                    "source numerical-relativity simulation."
                ),
            },
            "generator": {
                "name": "generate_nr_contract_fixture.py",
                "version": "1.0.0",
                "uri": "scripts/generate_nr_contract_fixture.py",
                "command": "python3 scripts/generate_nr_contract_fixture.py",
                "codeRevision": (
                    f"sha256:{sha256_bytes(Path(__file__).resolve().read_bytes())}"
                ),
                "deterministic": True,
            },
            "sourceArtifacts": [
                source_artifact("generator-source", Path(__file__).resolve()),
                source_artifact("schema", SCHEMA_PATH),
            ],
        },
        "units": {
            "system": "geometric",
            "G": 1.0,
            "c": 1.0,
            "massNormalization": {
                "quantity": "synthetic unit mass",
                "symbol": "M",
                "value": 1.0,
                "definition": (
                    "A dimensionless unit chosen only to exercise protocol fields."
                ),
                "referenceEpochSourceM": None,
            },
            "coordinateTime": "M",
            "length": "M",
            "angle": "radian",
            "frequencyShift": "dimensionless",
        },
        "timeReference": {
            "sourceTimeAtProtocolZeroM": 0.0,
            "sourceTimeDirection": "future-increasing",
            "protocolTimeDefinition": (
                "t_protocol=t_source-sourceTimeAtProtocolZeroM"
            ),
            "zeroEvent": {
                "name": "synthetic fixture origin",
                "source": "project-generated",
                "description": (
                    "The only observer sample is assigned protocol time zero."
                ),
            },
            "waveformTimeMapping": {
                "status": "not-applicable",
                "sourceQuantity": None,
                "mapping": None,
                "notApplicableReason": (
                    "The synthetic fixture contains no waveform."
                ),
            },
        },
        "coordinates": {
            "metricSignature": "-+++",
            "nrChart": {
                "status": "synthetic",
                "gauge": "not applicable; synthetic Minkowski contract fixture",
                "coordinates": "Cartesian (t,x,y,z)",
                "timeSlicing": "constant synthetic coordinate-time slices",
            },
            "worldFrame": {
                "handedness": "right",
                "axisOrder": ["x", "y", "z"],
                "origin": "synthetic NR chart origin",
                "matrixConvention": (
                    "row-major spatial affine 4x4 matrices multiplying [x,y,z,1] "
                    "column vectors; not spacetime coordinate transforms"
                ),
                "nrToWorld": identity,
                "worldToNr": identity,
            },
            "sky": {
                "referenceFrame": "ICRS",
                "icrsAxes": {
                    "x": "ICRS right ascension 0 degrees, declination 0 degrees",
                    "y": "ICRS right ascension 90 degrees, declination 0 degrees",
                    "z": "ICRS north celestial pole",
                },
                "rotationConvention": (
                    "proper right-handed row-major 3x3 rotations multiplying "
                    "spatial column vectors"
                ),
                "worldToIcrs": identity3,
                "icrsToWorld": identity3,
                "projection": "equirectangular",
                "longitudeMapping": "u=fract(longitude/(2*pi)+0.5)",
                "latitudeMapping": "v=0.5-latitude/pi",
                "escapeDirectionFrame": "ICRS",
            },
        },
        "observer": {
            "tetradBasisOrder": ["time", "right", "up", "forward"],
            "tetradIndexConvention": (
                "e_(a)^mu; rows are local basis vectors in the NR coordinate basis"
            ),
            "samples": [
                {
                    "sampleIndex": 0,
                    "protocolTimeM": 0.0,
                    "eventNr": [0.0, 4.0, 3.0, 12.0],
                    "metricCovariantNr": minkowski_metric,
                    "metricContravariantNr": minkowski_metric,
                    "fourVelocityContravariantNr": [1.0, 0.0, 0.0, 0.0],
                    "properTimeM": 0.0,
                    "tetradContravariantNr": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
            ],
        },
        "camera": {
            "frameType": "affine-visualization-frame",
            "motion": "fixed",
            "matrixConvention": (
                "row-major spatial affine 4x4 matrices multiplying [x,y,z,1] "
                "column vectors; not spacetime coordinate transforms"
            ),
            "basisOrder": ["right", "up", "forward"],
            "cameraToWorld": camera_to_world,
            "worldToCamera": world_to_camera,
            "physicalRelation": (
                "The affine frame is for deterministic playback coordinates; "
                "physical ray initialization uses the observer tetrad."
            ),
        },
        "projection": {
            "model": "rectilinear-pinhole",
            "widthPixels": WIDTH,
            "heightPixels": HEIGHT,
            "verticalFieldOfViewRad": 0.8,
            "imageOrigin": "top-left",
            "pixelSampleLocation": "center",
            "aspectConvention": "aspect=widthPixels/heightPixels",
            "screenXFormula": (
                "screenX=((x+0.5)/widthPixels*2-1)*aspect*"
                "tan(verticalFieldOfViewRad/2)"
            ),
            "screenYFormula": (
                "screenY=(1-(y+0.5)/heightPixels*2)*"
                "tan(verticalFieldOfViewRad/2)"
            ),
            "rayTimeOrientation": "past-directed",
            "localRayConvention": (
                "k^(a)=(-1,normalize(screenX,screenY,1)) in the "
                "time/right/up/forward tetrad"
            ),
        },
        "sampling": {
            "observationTimesM": [0.0],
            "timeCoordinate": "protocol",
            "dimensionOrder": ["time", "y", "x"],
            "pixelOrder": "row-major",
            "tileOrder": "manifest-order",
            "interpolation": {
                "time": "none",
                "continuous": "validity-strict-linear",
                "escapeDirection": "validity-strict-linear-then-normalize",
                "categorical": "nearest-no-blend",
                "invalidRecords": "never-sample-sky",
            },
        },
        "rayIntegration": {
            "spacetimeMode": "synthetic",
            "spatialInterpolation": "not applicable; no spacetime samples",
            "temporalInterpolation": "not applicable; one synthetic sample",
            "integrator": {
                "name": "none",
                "method": "hand-authored protocol sentinel records",
            },
            "tolerances": {
                "absolute": None,
                "relative": None,
                "nullConstraint": None,
            },
            "initialNormalization": "u_observer·k_observer=1",
            "timeOrientation": "past-directed",
            "termination": {
                "escaped": "intersect escapeBoundary",
                "captured": "intersect a captureTargets surface",
                "unresolved": "step or affine-parameter budget exhausted",
                "outside-domain": (
                    "left the declared spacetime domain away from escapeBoundary"
                ),
                "integrator-failure": "non-finite state or tolerance failure",
                "missing": "record was not generated",
            },
            "integrationPrecision": "float64",
            "outputPrecision": "float32",
        },
        "escapeBoundary": {
            "surface": {
                "kind": "synthetic-coordinate-sphere",
                "centreWorldM": [0.0, 0.0, 0.0],
                "radiusM": 100.0,
            },
            "referenceObserver": {
                "kind": "synthetic-inertial",
                "definition": (
                    "Future-directed unit inertial observer u=(1,0,0,0) in "
                    "the synthetic Cartesian NR chart."
                ),
                "sourceArtifactRole": None,
            },
            "frequencyShiftConvention": (
                "g=(u_observer·k_observer)/(u_boundary·k_boundary)"
            ),
            "storedEscapeDirection": {
                "frame": "ICRS",
                "normalization": "unit Euclidean spatial vector",
                "continuationBeyondBoundary": (
                    "Synthetic straight continuation in the declared ICRS "
                    "asymptotic frame with no additional lensing."
                ),
            },
        },
        "recordLayout": {
            "storage": "raw-struct-array",
            "byteOrder": "little-endian",
            "structFormat": "<7fBBH",
            "recordBytes": RECORD_BYTES,
            "fields": [
                {
                    "name": "escapeDirection",
                    "offsetBytes": 0,
                    "componentType": "float32",
                    "components": 3,
                },
                {
                    "name": "frequencyShiftG",
                    "offsetBytes": 12,
                    "componentType": "float32",
                    "components": 1,
                },
                {
                    "name": "coordinateLookbackTimeM",
                    "offsetBytes": 16,
                    "componentType": "float32",
                    "components": 1,
                },
                {
                    "name": "nullResidual",
                    "offsetBytes": 20,
                    "componentType": "float32",
                    "components": 1,
                },
                {
                    "name": "projectionErrorPx",
                    "offsetBytes": 24,
                    "componentType": "float32",
                    "components": 1,
                },
                {
                    "name": "rayOutcome",
                    "offsetBytes": 28,
                    "componentType": "uint8",
                    "components": 1,
                },
                {
                    "name": "captureTarget",
                    "offsetBytes": 29,
                    "componentType": "uint8",
                    "components": 1,
                },
                {
                    "name": "validityMask",
                    "offsetBytes": 30,
                    "componentType": "uint16",
                    "components": 1,
                },
            ],
            "observableConventions": {
                "escapeDirection": (
                    "outgoing ICRS unit direction after the declared "
                    "continuation beyond escapeBoundary"
                ),
                "frequencyShiftG": (
                    "g=(u_observer·k_observer)/(u_boundary·k_boundary)"
                ),
                "coordinateLookbackTimeM": (
                    "t_observer_protocol-t_terminal_protocol >= 0; a "
                    "gauge-dependent coordinate quantity, not a physical "
                    "relative arrival-time delay"
                ),
                "nullResidual": (
                    "maximum absolute g_mu_nu*k^mu*k^nu along the ray after "
                    "normalizing u_observer·k_observer to one"
                ),
                "projectionErrorPx": (
                    "estimated image-plane displacement under the declared "
                    "geodesic or interpolation refinement"
                ),
                "validity": (
                    "validityMask is authoritative; fields whose bits are clear "
                    "must never affect rendering or interpolation"
                ),
            },
            "rayOutcomes": {
                "escaped": OUTCOME_ESCAPED,
                "captured": OUTCOME_CAPTURED,
                "unresolved": OUTCOME_UNRESOLVED,
                "outside-domain": OUTCOME_OUTSIDE_DOMAIN,
                "integrator-failure": OUTCOME_INTEGRATOR_FAILURE,
                "missing": OUTCOME_MISSING,
            },
            "captureTargetNone": CAPTURE_NONE,
            "validityBits": {
                "escapeDirection": 0,
                "frequencyShiftG": 1,
                "coordinateLookbackTimeM": 2,
                "nullResidual": 3,
                "projectionErrorPx": 4,
            },
            "invalidFloatPolicy": (
                "write-positive-zero-and-use-validity-mask-as-authority"
            ),
        },
        "captureTargets": [
            {
                "code": CAPTURE_A,
                "id": "A",
                "description": "Synthetic capture target A",
                "surfaceKind": "synthetic-coordinate-sphere",
                "validityIntervalProtocolM": [0.0, 0.0],
                "classificationPriority": 0,
                "sourceArtifactRole": None,
            },
            {
                "code": CAPTURE_B,
                "id": "B",
                "description": "Synthetic capture target B",
                "surfaceKind": "synthetic-coordinate-sphere",
                "validityIntervalProtocolM": [0.0, 0.0],
                "classificationPriority": 1,
                "sourceArtifactRole": None,
            },
        ],
        "accuracy": {
            "status": "not-measured",
            "notMeasuredReason": (
                "The fixture contains hand-authored synthetic records and no "
                "numerical-relativity or geodesic solution."
            ),
            "nrConvergence": {
                "quantity": "NR grid-convergence order",
                "status": "not-applicable",
                "method": None,
                "value": None,
            },
            "constraintNorms": {
                "quantity": "maximum dimensionless constraint norm",
                "status": "not-applicable",
                "method": None,
                "value": None,
            },
            "geodesicNullResidual": {
                "quantity": "p99 normalized null residual",
                "status": "not-measured",
                "method": "synthetic sentinel values exercise the binary layout",
                "value": None,
            },
            "interpolationError": {
                "quantity": "p95 projection interpolation error in pixels",
                "status": "not-measured",
                "method": "synthetic sentinel values exercise the binary layout",
                "value": None,
            },
            "unresolvedFraction": None,
            "outcomeFractions": {
                "escaped": None,
                "captured": None,
                "unresolved": None,
                "outside-domain": None,
                "integrator-failure": None,
                "missing": None,
                "unusable": None,
            },
            "fixtureAssertions": {
                "worldCameraRoundTripTolerance": 1.0e-12,
                "expectedOutcomeCounts": {
                    "escaped": 2,
                    "captured": 2,
                    "unresolved": 1,
                    "outside-domain": 1,
                    "integrator-failure": 1,
                    "missing": 1,
                },
            },
        },
        "integrity": {
            "algorithm": "sha256",
            "manifestSidecar": "manifest.sha256",
            "sidecarFormat": "<lowercase-hex><two-spaces>manifest.json<newline>",
            "chunkUriBase": "manifest-directory",
            "chunkHashes": "embedded-in-manifest",
        },
        "chunks": [
            {
                "sampleIndex": 0,
                "tile": {
                    "x": 0,
                    "y": 0,
                    "width": WIDTH,
                    "height": HEIGHT,
                },
                "uri": f"chunks/{CHUNK_NAME}",
                "recordCount": WIDTH * HEIGHT,
                "recordBytes": RECORD_BYTES,
                "byteLength": len(chunk),
                "sha256": sha256_bytes(chunk),
            }
        ],
    }


def main() -> None:
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(f"missing schema: {SCHEMA_PATH}")

    chunk = fixture_chunk()
    document = manifest(chunk)
    manifest_bytes = (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    manifest_hash = sha256_bytes(manifest_bytes)
    manifest_hash_bytes = f"{manifest_hash}  manifest.json\n".encode("ascii")

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_PATH.write_bytes(chunk)
    MANIFEST_PATH.write_bytes(manifest_bytes)
    MANIFEST_HASH_PATH.write_bytes(manifest_hash_bytes)

    print(f"Wrote {CHUNK_PATH.relative_to(ROOT)} ({len(chunk)} bytes)")
    print(f"Wrote {MANIFEST_PATH.relative_to(ROOT)} ({len(manifest_bytes)} bytes)")
    print(f"SHA-256 {manifest_hash}")


if __name__ == "__main__":
    main()
