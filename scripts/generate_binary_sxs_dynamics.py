#!/usr/bin/env python3
"""Generate the compact SXS:BBH:0001 dynamics track used by the browser.

The source HDF5 files are intentionally not bundled. Download the pinned Lev5
files from the URLs recorded below, install the offline-only ``numpy`` and
``h5py`` dependencies, and pass their local paths to this script.

The generated track changes only the binary's motion and waveform readout. It
does not contain a near-zone metric or null-geodesic transfer data and must not
be described as NR-backed ray tracing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:
    import h5py  # type: ignore[import-not-found]
    import numpy as np
except ImportError as error:  # pragma: no cover - exercised only without tools
    raise SystemExit(
        "This offline generator requires numpy and h5py. "
        "Install them in a temporary environment before running it."
    ) from error


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = ROOT / "assets" / "scenes"
MANIFEST_NAME = "binary-sxs-bbh-0001-v2.json"
SAMPLES_NAME = "binary-sxs-bbh-0001-v2.samples.json"

SOURCE_RECORD = "https://doi.org/10.5281/zenodo.3273935"
SOURCE_FILES = {
    "metadata": {
        "path": "SXS:BBH:0001/Lev5/metadata.json",
        "byteLength": 4_170,
        "md5": "099d4c93d9466fe4b7ecad6c94499cf3",
        "url": (
            "https://zenodo.org/api/records/3273935/files/"
            "SXS:BBH:0001/Lev5/metadata.json/content"
        ),
    },
    "horizons": {
        "path": "SXS:BBH:0001/Lev5/Horizons.h5",
        "byteLength": 3_501_232,
        "md5": "484ea88842209e64983793159bcc7d7c",
        "url": (
            "https://zenodo.org/api/records/3273935/files/"
            "SXS:BBH:0001/Lev5/Horizons.h5/content"
        ),
    },
    "waveform": {
        "path": (
            "SXS:BBH:0001/Lev5/"
            "rhOverM_Asymptotic_GeometricUnits_CoM.h5"
        ),
        "byteLength": 142_641_207,
        "md5": "c271e0b905c74f434f00c9b14f67850c",
        "url": (
            "https://zenodo.org/api/records/3273935/files/"
            "SXS:BBH:0001/Lev5/"
            "rhOverM_Asymptotic_GeometricUnits_CoM.h5/content"
        ),
    },
}

WAVEFORM_DATASET = "Extrapolated_N2.dir/Y_l2_m2.dat"
SAMPLE_SCHEMA = "blackhole.binary-dynamics-samples/v1"
TRACK_COLUMNS = [
    "tProtocolM",
    "separationM",
    "orbitalPhaseUnwrappedRad",
    "h22Real",
    "h22Imag",
    "renderTopologyBlend",
    "individualHorizonsValid",
]
CHANNEL_TOLERANCES = {
    "separationM": 0.003,
    "orbitalPhaseUnwrappedRad": 0.003,
    "h22Real": 0.0005,
    "h22Imag": 0.0005,
    "renderTopologyBlend": 0.0005,
}
RINGDOWN_END_AFTER_PEAK_M = 120.0
SLOW_MOTION_START_M = -160.0
SLOW_MOTION_END_M = 70.0
SLOW_MOTION_FACTOR = 0.12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path, algorithm: str) -> str:
    checksum = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def verify_source(path: Path, source: dict[str, Any]) -> dict[str, Any]:
    require(path.is_file(), f"source file does not exist: {path}")
    size = path.stat().st_size
    require(
        size == source["byteLength"],
        f"{path.name}: expected {source['byteLength']} bytes, found {size}",
    )
    actual_md5 = digest(path, "md5")
    require(
        actual_md5 == source["md5"],
        f"{path.name}: pinned MD5 mismatch ({actual_md5})",
    )
    return {
        **source,
        "sha256": digest(path, "sha256"),
    }


def strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=reject_constant)
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def smoothstep(value: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    clamped = np.clip(value, 0.0, 1.0)
    return clamped * clamped * (3.0 - 2.0 * clamped)


def rounded(value: float, digits: int = 9) -> float:
    result = round(float(value), digits)
    return 0.0 if result == 0.0 else result


def adaptive_indices(
    times: "np.ndarray[Any, Any]",
    channels: "np.ndarray[Any, Any]",
    tolerances: "np.ndarray[Any, Any]",
    mandatory_indices: Iterable[int],
) -> list[int]:
    """Return piecewise-linear knots satisfying all declared channel errors."""

    kept = {0, len(times) - 1, *mandatory_indices}

    def refine(first: int, last: int) -> None:
        if last <= first + 1:
            return
        duration = times[last] - times[first]
        require(duration > 0.0, "dense source times must increase strictly")
        local_times = times[first + 1 : last]
        weights = (local_times - times[first]) / duration
        prediction = (
            channels[first][None, :]
            + weights[:, None] * (channels[last] - channels[first])[None, :]
        )
        normalized_error = np.abs(
            channels[first + 1 : last] - prediction
        ) / tolerances[None, :]
        flattened_index = int(np.argmax(normalized_error))
        error = float(normalized_error.reshape(-1)[flattened_index])
        if error <= 1.0:
            return
        local_row = flattened_index // channels.shape[1]
        split = first + 1 + local_row
        kept.add(split)
        refine(first, split)
        refine(split, last)

    boundaries = sorted(kept)
    for first, last in zip(boundaries, boundaries[1:]):
        refine(first, last)
    return sorted(kept)


def interpolation_errors(
    dense_times: "np.ndarray[Any, Any]",
    dense_channels: "np.ndarray[Any, Any]",
    sample_times: "np.ndarray[Any, Any]",
    sample_channels: "np.ndarray[Any, Any]",
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for index, name in enumerate(CHANNEL_TOLERANCES):
        reconstructed = np.interp(
            dense_times,
            sample_times,
            sample_channels[:, index],
        )
        errors[name] = float(
            np.max(np.abs(reconstructed - dense_channels[:, index]))
        )
    return errors


def nearest_index(values: "np.ndarray[Any, Any]", target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def generate(
    metadata_path: Path,
    horizons_path: Path,
    waveform_path: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    source_artifacts = {
        "metadata": verify_source(metadata_path, SOURCE_FILES["metadata"]),
        "horizons": verify_source(horizons_path, SOURCE_FILES["horizons"]),
        "waveform": verify_source(waveform_path, SOURCE_FILES["waveform"]),
    }
    metadata = strict_json(metadata_path)

    require(
        metadata.get("alternative_names") == "SXS:BBH:0001",
        "metadata is not SXS:BBH:0001",
    )
    reference_mass_total = (
        float(metadata["reference_mass1"]) + float(metadata["reference_mass2"])
    )
    require(
        math.isclose(reference_mass_total, 1.0, rel_tol=0.0, abs_tol=1e-5),
        "unexpected reference total mass normalization",
    )

    with h5py.File(horizons_path, "r") as horizons:
        centre_a = np.asarray(
            horizons["AhA.dir/CoordCenterInertial.dat"][:],
            dtype=np.float64,
        )
        centre_b = np.asarray(
            horizons["AhB.dir/CoordCenterInertial.dat"][:],
            dtype=np.float64,
        )
        common_mass = np.asarray(
            horizons["AhC.dir/ChristodoulouMass.dat"][:],
            dtype=np.float64,
        )
        common_spin = np.asarray(
            horizons["AhC.dir/chiInertial.dat"][:],
            dtype=np.float64,
        )

    require(centre_a.shape == centre_b.shape, "A/B horizon grids differ")
    require(centre_a.shape[1] == 4, "horizon center records must have 4 columns")
    require(
        np.array_equal(centre_a[:, 0], centre_b[:, 0]),
        "A/B horizon times do not match exactly",
    )
    require(
        np.all(np.diff(centre_a[:, 0]) > 0.0),
        "horizon source times must increase strictly",
    )

    with h5py.File(waveform_path, "r") as waveform:
        h22 = np.asarray(waveform[WAVEFORM_DATASET][:], dtype=np.float64)

    require(h22.ndim == 2 and h22.shape[1] == 3, "h22 dataset must be Nx3")
    require(
        np.all(np.diff(h22[:, 0]) > 0.0),
        "waveform retarded times must increase strictly",
    )
    amplitude = np.hypot(h22[:, 1], h22[:, 2])
    waveform_peak_index = int(np.argmax(amplitude))
    waveform_peak_source_time_m = float(h22[waveform_peak_index, 0])
    waveform_peak_amplitude = float(amplitude[waveform_peak_index])

    source_horizon_times_m = centre_a[:, 0] / reference_mass_total
    relative_position = (
        centre_b[:, 1:4] - centre_a[:, 1:4]
    ) / reference_mass_total
    separation_m = np.linalg.norm(relative_position, axis=1)
    orbital_phase = np.unwrap(
        np.arctan2(relative_position[:, 1], relative_position[:, 0])
    )

    relaxation_source_time_m = (
        float(metadata["relaxation_time"]) / reference_mass_total
    )
    first_horizon_index = int(
        np.searchsorted(source_horizon_times_m, relaxation_source_time_m)
    )
    require(
        first_horizon_index < len(source_horizon_times_m) - 2,
        "relaxation time is outside the individual-horizon data",
    )
    source_horizon_times_m = source_horizon_times_m[first_horizon_index:]
    separation_m = separation_m[first_horizon_index:]
    orbital_phase = orbital_phase[first_horizon_index:]
    orbital_phase = orbital_phase - orbital_phase[0]

    horizon_protocol_times_m = (
        source_horizon_times_m - waveform_peak_source_time_m
    )
    waveform_protocol_times_m = h22[:, 0] - waveform_peak_source_time_m
    first_time_m = float(horizon_protocol_times_m[0])
    individual_horizons_last_m = float(horizon_protocol_times_m[-1])
    common_horizon_first_m = (
        float(metadata["common_horizon_time"]) / reference_mass_total
        - waveform_peak_source_time_m
    )
    final_time_m = RINGDOWN_END_AFTER_PEAK_M

    require(first_time_m < common_horizon_first_m < 0.0, "invalid event order")
    require(
        individual_horizons_last_m <= common_horizon_first_m,
        "unexpected individual/common horizon overlap for pinned source",
    )
    require(
        waveform_protocol_times_m[0] <= first_time_m
        and waveform_protocol_times_m[-1] >= final_time_m,
        "waveform does not cover the playback interval",
    )

    waveform_mask = (
        (waveform_protocol_times_m >= first_time_m)
        & (waveform_protocol_times_m <= final_time_m)
    )
    dense_times = np.unique(
        np.concatenate(
            (
                horizon_protocol_times_m,
                waveform_protocol_times_m[waveform_mask],
                np.asarray(
                    [
                        first_time_m,
                        individual_horizons_last_m,
                        common_horizon_first_m,
                        0.0,
                        final_time_m,
                        SLOW_MOTION_START_M,
                        SLOW_MOTION_END_M,
                    ],
                    dtype=np.float64,
                ),
            )
        )
    )
    dense_times = dense_times[
        (dense_times >= first_time_m) & (dense_times <= final_time_m)
    ]

    dense_separation = np.interp(
        dense_times,
        horizon_protocol_times_m,
        separation_m,
    )
    dense_phase = np.interp(
        dense_times,
        horizon_protocol_times_m,
        orbital_phase,
    )
    dense_h22_real = np.interp(
        dense_times,
        waveform_protocol_times_m,
        h22[:, 1],
    )
    dense_h22_imag = np.interp(
        dense_times,
        waveform_protocol_times_m,
        h22[:, 2],
    )
    dense_blend = smoothstep(
        (dense_times - common_horizon_first_m)
        / (0.0 - common_horizon_first_m)
    )
    dense_validity = (dense_times <= individual_horizons_last_m).astype(np.int8)
    dense_channels = np.column_stack(
        (
            dense_separation,
            dense_phase,
            dense_h22_real,
            dense_h22_imag,
            dense_blend,
        )
    )
    tolerances = np.asarray(
        [CHANNEL_TOLERANCES[name] for name in CHANNEL_TOLERANCES],
        dtype=np.float64,
    )
    event_times = [
        first_time_m,
        individual_horizons_last_m,
        common_horizon_first_m,
        0.0,
        final_time_m,
        SLOW_MOTION_START_M,
        SLOW_MOTION_END_M,
    ]
    mandatory_indices = [nearest_index(dense_times, time) for time in event_times]
    selected_indices = adaptive_indices(
        dense_times,
        dense_channels,
        tolerances,
        mandatory_indices,
    )

    selected_times = dense_times[selected_indices]
    selected_channels = dense_channels[selected_indices]
    selected_validity = dense_validity[selected_indices]
    sample_rows = [
        [
            rounded(time_m),
            rounded(channels[0]),
            rounded(channels[1]),
            rounded(channels[2]),
            rounded(channels[3]),
            rounded(channels[4]),
            int(validity),
        ]
        for time_m, channels, validity in zip(
            selected_times,
            selected_channels,
            selected_validity,
        )
    ]
    # Re-read rounded values so reported bounds cover the exact bundled bytes.
    rounded_times = np.asarray([row[0] for row in sample_rows], dtype=np.float64)
    rounded_channels = np.asarray(
        [row[1:6] for row in sample_rows],
        dtype=np.float64,
    )
    measured_errors = interpolation_errors(
        dense_times,
        dense_channels,
        rounded_times,
        rounded_channels,
    )
    for name, tolerance in CHANNEL_TOLERANCES.items():
        require(
            measured_errors[name] <= tolerance * 1.001,
            f"downsampled {name} error exceeds tolerance",
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    samples_path = output_directory / SAMPLES_NAME
    manifest_path = output_directory / MANIFEST_NAME
    samples_payload = {
        "schema": SAMPLE_SCHEMA,
        "columns": TRACK_COLUMNS,
        "samples": sample_rows,
    }
    samples_bytes = (
        json.dumps(
            samples_payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    samples_path.write_bytes(samples_bytes)

    remnant_mass = float(metadata["remnant_mass"])
    remnant_spin = [float(value) for value in metadata["remnant_dimensionless_spin"]]
    measured_final_mass = float(common_mass[-1, 1]) / reference_mass_total
    measured_final_spin = [float(value) for value in common_spin[-1, 1:4]]
    body_mass_total = (
        float(metadata["reference_mass1"]) + float(metadata["reference_mass2"])
    )
    body_masses = [
        float(metadata["reference_mass1"]) / body_mass_total,
        float(metadata["reference_mass2"]) / body_mass_total,
    ]

    manifest = {
        "schema": "blackhole.binary-scene/v2",
        "id": "binary-sxs-bbh-0001-v2",
        "title": "SXS:BBH:0001 NR-driven binary dynamics preview",
        "scientificStatus": {
            "classification": (
                "NR-driven dynamics with weak-field fast-light rendering"
            ),
            "dynamicsSourceIsNumericalRelativity": True,
            "nearZoneSpacetimeUsedForLightPropagation": False,
            "slowLightGeodesicsIncluded": False,
            "description": (
                "Orbital coordinate separation and phase come from the two "
                "SXS apparent-horizon centroids, while the waveform strip uses "
                "the CoM-corrected extrapolated N=2 (2,2) strain mode. The "
                "existing lens shader remains a weak-field fast-light preview."
            ),
            "prohibitedClaim": (
                "Do not describe the rendered image as NR ray tracing, a solved "
                "binary spacetime image, or a quantitatively accurate merger "
                "shadow."
            ),
        },
        "source": {
            "catalog": "SXS",
            "simulation": "SXS:BBH:0001",
            "level": "Lev5",
            "recordDoi": SOURCE_RECORD,
            "recordId": 3273935,
            "license": {
                "spdx": None,
                "status": "not-declared-in-pinned-zenodo-record",
                "attributionRequired": True,
            },
            "artifacts": source_artifacts,
            "waveformDataset": WAVEFORM_DATASET,
        },
        "units": {
            "system": "geometric",
            "G": 1.0,
            "c": 1.0,
            "massUnit": "M",
            "timeUnit": "M",
            "distanceUnit": "M",
            "referenceMassDefinition": (
                "sum of the two Christodoulou masses at relaxation time"
            ),
            "referenceMassTotalInCodeUnits": reference_mass_total,
        },
        "physicalSystem": {
            "massRatioQ": float(metadata["initial_mass_ratio"]),
            "referenceEccentricity": float(metadata["reference_eccentricity"]),
            "referenceTimeCodeUnits": float(metadata["reference_time"]),
            "bodies": [
                {
                    "id": "A",
                    "massFraction": body_masses[0],
                    "dimensionlessSpin": [
                        float(value)
                        for value in metadata["reference_dimensionless_spin1"]
                    ],
                    "orbitPositionScale": -body_masses[1],
                },
                {
                    "id": "B",
                    "massFraction": body_masses[1],
                    "dimensionlessSpin": [
                        float(value)
                        for value in metadata["reference_dimensionless_spin2"]
                    ],
                    "orbitPositionScale": body_masses[0],
                },
            ],
            "remnant": {
                "massFraction": remnant_mass,
                "dimensionlessSpin": remnant_spin,
                "metadataSource": "SXS:BBH:0001 Lev5 metadata.json",
                "finalHorizonDiagnostic": {
                    "sourceTimeCodeUnits": float(common_mass[-1, 0]),
                    "massFraction": measured_final_mass,
                    "dimensionlessSpin": measured_final_spin,
                },
            },
        },
        "timeReference": {
            "protocolZeroEvent": "maximum amplitude of Extrapolated_N2 h(2,2)",
            "waveformPeakSourceTimeM": waveform_peak_source_time_m,
            "waveformPeakAmplitude": waveform_peak_amplitude,
            "waveformMapping": (
                "t_protocol=t_retarded_normalized-t_waveform_peak"
            ),
            "horizonMapping": (
                "t_protocol=t_coordinate_code/referenceMassTotal"
                "-t_waveform_peak"
            ),
            "alignmentCaveat": (
                "Horizon coordinate time and extrapolated waveform retarded "
                "time are distinct source coordinates. Their published numeric "
                "time origins are retained; this is not a gauge-invariant "
                "light-travel-time measurement."
            ),
        },
        "events": {
            "relaxation": {
                "tProtocolM": first_time_m,
                "source": "metadata relaxation_time",
            },
            "individualHorizonsLast": {
                "tProtocolM": individual_horizons_last_m,
                "source": "last common A/B CoordCenterInertial sample",
            },
            "commonApparentHorizonFirst": {
                "tProtocolM": common_horizon_first_m,
                "source": "metadata common_horizon_time",
            },
            "waveformPeak": {
                "tProtocolM": 0.0,
                "source": "maximum abs(Extrapolated_N2 h(2,2))",
            },
            "playbackEnd": {
                "tProtocolM": final_time_m,
                "source": "configured ringdown display endpoint",
            },
        },
        "dynamics": {
            "asset": {
                "uri": SAMPLES_NAME,
                "schema": SAMPLE_SCHEMA,
                "encoding": "utf-8 minified JSON",
                "sha256": hashlib.sha256(samples_bytes).hexdigest(),
                "byteLength": len(samples_bytes),
                "sampleCount": len(sample_rows),
                "columns": TRACK_COLUMNS,
            },
            "interpolation": "piecewise linear with validity-strict horizon hold",
            "firstTimeM": float(rounded_times[0]),
            "finalTimeM": float(rounded_times[-1]),
            "sourceChannels": {
                "separationM": (
                    "Euclidean distance between SXS AhA/AhB inertial-frame "
                    "coordinate centroids; gauge-dependent"
                ),
                "orbitalPhaseUnwrappedRad": (
                    "unwrapped atan2(delta_y, delta_x), shifted to zero at "
                    "the first bundled sample; gauge-dependent"
                ),
                "h22": (
                    "complex r h_22 / M from the CoM-corrected N=2 "
                    "extrapolated waveform"
                ),
            },
            "postHorizonPolicy": (
                "hold the last valid A/B separation and phase; remove the "
                "two-centre renderer only through renderTopologyBlend"
            ),
            "renderTransition": {
                "quantity": "presentation proxy, not a horizon observable",
                "kind": "smoothstep",
                "startEvent": "commonApparentHorizonFirst",
                "completeEvent": "waveformPeak",
            },
            "measuredMaxInterpolationError": {
                key: rounded(value, 12)
                for key, value in measured_errors.items()
            },
            "declaredMaxInterpolationError": CHANNEL_TOLERANCES,
        },
        "rendererAdapter": {
            "shaderBundleId": "binary-approx-v1",
            "lightPropagation": "weak-field-fast-light",
            "positionsFrozenPerRay": True,
            "nearZoneMetricConsumed": False,
            "stateAbi": [
                "separationM",
                "orbitalPhaseUnwrappedRad",
                "renderTopologyBlend",
                "reserved",
            ],
            "massAbi": [
                "bodyAMassFraction",
                "bodyBMassFraction",
                "remnantMassFraction",
                "reserved",
            ],
        },
        "playback": {
            "cycleDurationSecondsAtNominalRate": 36.0,
            "endHoldSeconds": 2.5,
            "loop": True,
            "scrubCoordinate": "linear protocol time",
            "slowMotion": {
                "enabledByDefault": True,
                "startTimeM": SLOW_MOTION_START_M,
                "endTimeM": SLOW_MOTION_END_M,
                "rateMultiplier": SLOW_MOTION_FACTOR,
                "status": (
                    "presentation-only wall-time mapping; not gravitational "
                    "time dilation"
                ),
            },
        },
        "rendererDefaults": {
            "observerRadiusM": 42.0,
            "initialViewingInclinationDeg": 57.0,
            "fieldOfViewDeg": 52.0,
            "raySteps": 512,
            "exposure": 1.08,
        },
        "generation": {
            "generator": "scripts/generate_binary_sxs_dynamics.py",
            "generatorSha256": digest(Path(__file__).resolve(), "sha256"),
            "command": (
                "python3 scripts/generate_binary_sxs_dynamics.py "
                "--metadata metadata.json --horizons Horizons.h5 "
                "--waveform rhOverM_Asymptotic_GeometricUnits_CoM.h5"
            ),
            "deterministic": True,
        },
    }
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, samples_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--horizons", required=True, type=Path)
    parser.add_argument("--waveform", required=True, type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    manifest_path, samples_path = generate(
        arguments.metadata,
        arguments.horizons,
        arguments.waveform,
        arguments.output_directory,
    )
    manifest = strict_json(manifest_path)
    def display_path(path: Path) -> Path:
        try:
            return path.relative_to(ROOT)
        except ValueError:
            return path

    print("SXS binary dynamics assets generated")
    print(f"  manifest = {display_path(manifest_path)}")
    print(f"  samples = {display_path(samples_path)}")
    print(f"  count = {manifest['dynamics']['asset']['sampleCount']}")
    print(
        "  protocol range = "
        f"{manifest['dynamics']['firstTimeM']:.6f} M .. "
        f"{manifest['dynamics']['finalTimeM']:.6f} M"
    )
    print(
        "  common horizon / waveform peak = "
        f"{manifest['events']['commonApparentHorizonFirst']['tProtocolM']:.6f} M "
        "/ 0 M"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
