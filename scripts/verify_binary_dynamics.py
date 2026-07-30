#!/usr/bin/env python3
"""Validate the bundled SXS-driven binary dynamics and playback contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets" / "scenes" / "binary-sxs-bbh-0001-v2.json"
GENERATOR = ROOT / "scripts" / "generate_binary_sxs_dynamics.py"
EXPECTED_COLUMNS = [
    "tProtocolM",
    "separationM",
    "orbitalPhaseUnwrappedRad",
    "h22Real",
    "h22Imag",
    "renderTopologyBlend",
    "individualHorizonsValid",
]
ERROR_CHANNELS = {
    "separationM",
    "orbitalPhaseUnwrappedRad",
    "h22Real",
    "h22Imag",
    "renderTopologyBlend",
}
DECLARED_ERROR_LIMITS = {
    "separationM": 0.003,
    "orbitalPhaseUnwrappedRad": 0.003,
    "h22Real": 0.0005,
    "h22Imag": 0.0005,
    "renderTopologyBlend": 0.0005,
}
EXPECTED_STATE_ABI = [
    "separationM",
    "orbitalPhaseUnwrappedRad",
    "renderTopologyBlend",
    "reserved",
]
EXPECTED_MASS_ABI = [
    "bodyAMassFraction",
    "bodyBMassFraction",
    "remnantMassFraction",
    "reserved",
]
PINNED_SOURCE = {
    "metadata": (
        4_170,
        "099d4c93d9466fe4b7ecad6c94499cf3",
        "329d0643f9d33361eafaeae7ef1818dcda3311b33477ecef4f002ead17f42668",
    ),
    "horizons": (
        3_501_232,
        "484ea88842209e64983793159bcc7d7c",
        "cf97de4a60a4cd5c6a56f219ea9fa81f1849647f134250e95ae79e40be4dd957",
    ),
    "waveform": (
        142_641_207,
        "c271e0b905c74f434f00c9b14f67850c",
        "d760add0693e458781f8db9958b4669971e816d7c026cdbe5f09b7d8fd6bd21f",
    ),
}
PINNED_SOURCE_PATHS = {
    "metadata": "SXS:BBH:0001/Lev5/metadata.json",
    "horizons": "SXS:BBH:0001/Lev5/Horizons.h5",
    "waveform": (
        "SXS:BBH:0001/Lev5/"
        "rhOverM_Asymptotic_GeometricUnits_CoM.h5"
    ),
}
PINNED_REMNANT_MASS = 0.951609417715
PINNED_REMNANT_SPIN_Z = 0.686461676493
PINNED_REFERENCE_MASS_TOTAL = 1.000000327684


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValidationError(f"{path}: non-finite JSON constant {value!r}")

    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=object_without_duplicates,
                parse_constant=reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read {path}: {error}") from error


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def finite(value: Any, path: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{path} must be numeric",
    )
    numeric = float(value)
    require(math.isfinite(numeric), f"{path} must be finite")
    return numeric


def integer(value: Any, path: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{path} must be an integer",
    )
    return value


def boolean(value: Any, path: str) -> bool:
    require(isinstance(value, bool), f"{path} must be boolean")
    return value


def nonempty_string(value: Any, path: str) -> str:
    require(
        isinstance(value, str) and bool(value.strip()),
        f"{path} must be a non-empty string",
    )
    return value


def vector3(value: Any, path: str) -> list[float]:
    require(isinstance(value, list) and len(value) == 3, f"{path} must have 3 values")
    return [finite(component, f"{path}[{index}]") for index, component in enumerate(value)]


def dimensionless_spin(value: Any, path: str) -> list[float]:
    spin = vector3(value, path)
    magnitude = math.sqrt(sum(component * component for component in spin))
    require(magnitude <= 1.0 + 1.0e-12, f"{path} magnitude must not exceed one")
    return spin


def sha256_text(value: Any, path: str) -> str:
    digest = nonempty_string(value, path)
    require(
        len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{path} must be a lowercase SHA-256",
    )
    return digest


def exact_keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{path} must be an object")
    actual = set(value)
    require(
        actual == expected,
        f"{path} keys differ: missing={sorted(expected - actual)}, "
        f"unknown={sorted(actual - expected)}",
    )
    return value


def linear_sample(rows: list[list[Any]], time_m: float) -> list[float]:
    if time_m <= rows[0][0]:
        return [float(value) for value in rows[0][:6]]
    if time_m >= rows[-1][0]:
        return [float(value) for value in rows[-1][:6]]
    lower = 0
    upper = len(rows) - 1
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if rows[middle][0] <= time_m:
            lower = middle
        else:
            upper = middle
    first = rows[lower]
    second = rows[upper]
    weight = (time_m - first[0]) / (second[0] - first[0])
    return [
        float(first[index] + (second[index] - first[index]) * weight)
        for index in range(6)
    ]


def verify_physical_system(manifest: dict[str, Any]) -> None:
    physical = exact_keys(
        manifest["physicalSystem"],
        {
            "bodies",
            "massRatioQ",
            "referenceEccentricity",
            "referenceTimeCodeUnits",
            "remnant",
        },
        "$.physicalSystem",
    )
    bodies = physical["bodies"]
    require(
        isinstance(bodies, list) and len(bodies) == 2,
        "$.physicalSystem.bodies must contain exactly A and B",
    )
    masses: list[float] = []
    position_scales: list[float] = []
    for index, expected_id in enumerate(("A", "B")):
        path = f"$.physicalSystem.bodies[{index}]"
        body = exact_keys(
            bodies[index],
            {
                "id",
                "massFraction",
                "dimensionlessSpin",
                "orbitPositionScale",
            },
            path,
        )
        require(body["id"] == expected_id, f"{path}.id must be {expected_id!r}")
        mass = finite(body["massFraction"], f"{path}.massFraction")
        require(0.0 < mass < 1.0, f"{path}.massFraction must be in (0,1)")
        masses.append(mass)
        dimensionless_spin(body["dimensionlessSpin"], f"{path}.dimensionlessSpin")
        position_scales.append(
            finite(body["orbitPositionScale"], f"{path}.orbitPositionScale")
        )
    require(
        math.isclose(sum(masses), 1.0, rel_tol=0.0, abs_tol=1.0e-12),
        "$.physicalSystem body mass fractions must sum to one",
    )
    require(
        position_scales[0] < 0.0 < position_scales[1],
        "$.physicalSystem orbit position scales must straddle the barycentre",
    )
    require(
        math.isclose(
            position_scales[1] - position_scales[0],
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            masses[0] * position_scales[0]
            + masses[1] * position_scales[1],
            0.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "$.physicalSystem orbit position scales violate the mass ABI",
    )

    mass_ratio = finite(physical["massRatioQ"], "$.physicalSystem.massRatioQ")
    require(
        mass_ratio > 0.0 and abs(mass_ratio - 1.0) <= 1.0e-6,
        "$.physicalSystem.massRatioQ is not the pinned equal-mass system",
    )
    eccentricity = finite(
        physical["referenceEccentricity"],
        "$.physicalSystem.referenceEccentricity",
    )
    require(
        0.0 <= eccentricity < 0.01,
        "$.physicalSystem.referenceEccentricity is outside the quasi-circular gate",
    )
    require(
        finite(
            physical["referenceTimeCodeUnits"],
            "$.physicalSystem.referenceTimeCodeUnits",
        )
        >= 0.0,
        "$.physicalSystem.referenceTimeCodeUnits must be non-negative",
    )

    remnant = exact_keys(
        physical["remnant"],
        {
            "massFraction",
            "dimensionlessSpin",
            "finalHorizonDiagnostic",
            "metadataSource",
        },
        "$.physicalSystem.remnant",
    )
    remnant_mass = finite(
        remnant["massFraction"],
        "$.physicalSystem.remnant.massFraction",
    )
    require(
        0.0 < remnant_mass <= 1.0,
        "$.physicalSystem.remnant.massFraction must be in (0,1]",
    )
    dimensionless_spin(
        remnant["dimensionlessSpin"],
        "$.physicalSystem.remnant.dimensionlessSpin",
    )
    require(
        remnant["metadataSource"] == "SXS:BBH:0001 Lev5 metadata.json",
        "$.physicalSystem.remnant.metadataSource drifted",
    )
    diagnostic = exact_keys(
        remnant["finalHorizonDiagnostic"],
        {"massFraction", "dimensionlessSpin", "sourceTimeCodeUnits"},
        "$.physicalSystem.remnant.finalHorizonDiagnostic",
    )
    diagnostic_mass = finite(
        diagnostic["massFraction"],
        "$.physicalSystem.remnant.finalHorizonDiagnostic.massFraction",
    )
    require(
        0.0 < diagnostic_mass <= 1.0,
        "$.physicalSystem.remnant final horizon mass must be in (0,1]",
    )
    dimensionless_spin(
        diagnostic["dimensionlessSpin"],
        "$.physicalSystem.remnant.finalHorizonDiagnostic.dimensionlessSpin",
    )
    require(
        finite(
            diagnostic["sourceTimeCodeUnits"],
            "$.physicalSystem.remnant.finalHorizonDiagnostic.sourceTimeCodeUnits",
        )
        > finite(
            physical["referenceTimeCodeUnits"],
            "$.physicalSystem.referenceTimeCodeUnits",
        ),
        "$.physicalSystem.remnant final diagnostic must follow the reference time",
    )


def verify_time_and_events(manifest: dict[str, Any]) -> None:
    time_reference = exact_keys(
        manifest["timeReference"],
        {
            "protocolZeroEvent",
            "waveformMapping",
            "horizonMapping",
            "waveformPeakSourceTimeM",
            "waveformPeakAmplitude",
            "alignmentCaveat",
        },
        "$.timeReference",
    )
    for name in ("waveformMapping", "horizonMapping", "alignmentCaveat"):
        nonempty_string(time_reference[name], f"$.timeReference.{name}")
    peak_time = finite(
        time_reference["waveformPeakSourceTimeM"],
        "$.timeReference.waveformPeakSourceTimeM",
    )
    require(peak_time > 0.0, "$.timeReference waveform source peak must be positive")
    peak_amplitude = finite(
        time_reference["waveformPeakAmplitude"],
        "$.timeReference.waveformPeakAmplitude",
    )
    require(
        0.0 < peak_amplitude < 1.0,
        "$.timeReference waveform peak amplitude must be in (0,1)",
    )

    events = exact_keys(
        manifest["events"],
        {
            "relaxation",
            "individualHorizonsLast",
            "commonApparentHorizonFirst",
            "waveformPeak",
            "playbackEnd",
        },
        "$.events",
    )
    for name, event in events.items():
        path = f"$.events.{name}"
        exact_keys(event, {"source", "tProtocolM"}, path)
        nonempty_string(event["source"], f"{path}.source")
        finite(event["tProtocolM"], f"{path}.tProtocolM")


def verify_dynamics_contract(manifest: dict[str, Any]) -> None:
    dynamics = exact_keys(
        manifest["dynamics"],
        {
            "asset",
            "firstTimeM",
            "finalTimeM",
            "interpolation",
            "sourceChannels",
            "postHorizonPolicy",
            "renderTransition",
            "measuredMaxInterpolationError",
            "declaredMaxInterpolationError",
        },
        "$.dynamics",
    )
    asset = exact_keys(
        dynamics["asset"],
        {
            "schema",
            "uri",
            "encoding",
            "columns",
            "sampleCount",
            "byteLength",
            "sha256",
        },
        "$.dynamics.asset",
    )
    require(
        asset["schema"] == "blackhole.binary-dynamics-samples/v1",
        "$.dynamics.asset.schema changed",
    )
    require(
        asset["encoding"] == "utf-8 minified JSON",
        "$.dynamics.asset.encoding changed",
    )
    require(asset["columns"] == EXPECTED_COLUMNS, "$.dynamics.asset.columns changed")
    require(
        integer(asset["sampleCount"], "$.dynamics.asset.sampleCount") >= 100,
        "$.dynamics.asset.sampleCount is implausibly small",
    )
    require(
        integer(asset["byteLength"], "$.dynamics.asset.byteLength") > 0,
        "$.dynamics.asset.byteLength must be positive",
    )
    sha256_text(asset["sha256"], "$.dynamics.asset.sha256")

    first_time = finite(dynamics["firstTimeM"], "$.dynamics.firstTimeM")
    final_time = finite(dynamics["finalTimeM"], "$.dynamics.finalTimeM")
    require(
        first_time < 0.0 < final_time,
        "$.dynamics time range must contain the waveform peak",
    )
    require(
        math.isclose(
            first_time,
            finite(
                manifest["events"]["relaxation"]["tProtocolM"],
                "$.events.relaxation.tProtocolM",
            ),
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        and math.isclose(
            final_time,
            finite(
                manifest["events"]["playbackEnd"]["tProtocolM"],
                "$.events.playbackEnd.tProtocolM",
            ),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "$.dynamics time range must be bounded by relaxation and playbackEnd",
    )
    require(
        dynamics["interpolation"]
        == "piecewise linear with validity-strict horizon hold",
        "$.dynamics.interpolation changed",
    )
    nonempty_string(dynamics["postHorizonPolicy"], "$.dynamics.postHorizonPolicy")

    source_channels = exact_keys(
        dynamics["sourceChannels"],
        {"separationM", "orbitalPhaseUnwrappedRad", "h22"},
        "$.dynamics.sourceChannels",
    )
    for name, value in source_channels.items():
        nonempty_string(value, f"$.dynamics.sourceChannels.{name}")

    transition = exact_keys(
        dynamics["renderTransition"],
        {"kind", "startEvent", "completeEvent", "quantity"},
        "$.dynamics.renderTransition",
    )
    require(
        transition["kind"] == "smoothstep"
        and transition["startEvent"] == "commonApparentHorizonFirst"
        and transition["completeEvent"] == "waveformPeak",
        "$.dynamics.renderTransition ABI changed",
    )
    require(
        "presentation proxy" in nonempty_string(
            transition["quantity"],
            "$.dynamics.renderTransition.quantity",
        ),
        "$.dynamics.renderTransition must remain labelled as a presentation proxy",
    )

    for map_name in (
        "measuredMaxInterpolationError",
        "declaredMaxInterpolationError",
    ):
        error_map = exact_keys(
            dynamics[map_name],
            ERROR_CHANNELS,
            f"$.dynamics.{map_name}",
        )
        for channel, value in error_map.items():
            numeric = finite(value, f"$.dynamics.{map_name}.{channel}")
            require(
                numeric >= 0.0,
                f"$.dynamics.{map_name}.{channel} must be non-negative",
            )
            if map_name == "declaredMaxInterpolationError":
                require(
                    0.0 < numeric <= DECLARED_ERROR_LIMITS[channel],
                    f"$.dynamics.{map_name}.{channel} exceeds the v2 error budget",
                )


def verify_renderer_contract(manifest: dict[str, Any]) -> None:
    renderer = exact_keys(
        manifest["rendererAdapter"],
        {
            "shaderBundleId",
            "lightPropagation",
            "nearZoneMetricConsumed",
            "positionsFrozenPerRay",
            "stateAbi",
            "massAbi",
        },
        "$.rendererAdapter",
    )
    boolean(
        renderer["nearZoneMetricConsumed"],
        "$.rendererAdapter.nearZoneMetricConsumed",
    )
    boolean(
        renderer["positionsFrozenPerRay"],
        "$.rendererAdapter.positionsFrozenPerRay",
    )
    require(
        renderer["nearZoneMetricConsumed"] is False
        and renderer["positionsFrozenPerRay"] is True,
        "$.rendererAdapter fast-light boundary changed",
    )
    require(
        renderer["stateAbi"] == EXPECTED_STATE_ABI,
        "$.rendererAdapter.stateAbi changed",
    )
    require(
        renderer["massAbi"] == EXPECTED_MASS_ABI,
        "$.rendererAdapter.massAbi changed",
    )

    defaults = exact_keys(
        manifest["rendererDefaults"],
        {
            "observerRadiusM",
            "fieldOfViewDeg",
            "initialViewingInclinationDeg",
            "exposure",
            "raySteps",
        },
        "$.rendererDefaults",
    )
    require(
        finite(defaults["observerRadiusM"], "$.rendererDefaults.observerRadiusM")
        > 2.0,
        "$.rendererDefaults.observerRadiusM must stay outside 2 M",
    )
    field_of_view = finite(
        defaults["fieldOfViewDeg"],
        "$.rendererDefaults.fieldOfViewDeg",
    )
    require(
        1.0 < field_of_view < 179.0,
        "$.rendererDefaults.fieldOfViewDeg must be in (1,179)",
    )
    inclination = finite(
        defaults["initialViewingInclinationDeg"],
        "$.rendererDefaults.initialViewingInclinationDeg",
    )
    require(
        0.0 <= inclination <= 180.0,
        "$.rendererDefaults.initialViewingInclinationDeg must be in [0,180]",
    )
    exposure = finite(defaults["exposure"], "$.rendererDefaults.exposure")
    require(
        0.0 < exposure <= 16.0,
        "$.rendererDefaults.exposure must be in (0,16]",
    )
    ray_steps = integer(defaults["raySteps"], "$.rendererDefaults.raySteps")
    require(
        64 < ray_steps <= 512,
        "$.rendererDefaults.raySteps must be an integer in [65,512]",
    )


def verify_generation_contract(manifest: dict[str, Any]) -> None:
    generation = exact_keys(
        manifest["generation"],
        {"command", "deterministic", "generator", "generatorSha256"},
        "$.generation",
    )
    require(
        generation["generator"] == "scripts/generate_binary_sxs_dynamics.py",
        "$.generation.generator changed",
    )
    nonempty_string(generation["command"], "$.generation.command")
    require(
        boolean(generation["deterministic"], "$.generation.deterministic") is True,
        "$.generation must remain deterministic",
    )
    sha256_text(generation["generatorSha256"], "$.generation.generatorSha256")


def verify_manifest(manifest: Any) -> None:
    exact_keys(
        manifest,
        {
            "schema",
            "id",
            "title",
            "scientificStatus",
            "source",
            "units",
            "physicalSystem",
            "timeReference",
            "events",
            "dynamics",
            "rendererAdapter",
            "playback",
            "rendererDefaults",
            "generation",
        },
        "$",
    )
    require(
        manifest["schema"] == "blackhole.binary-scene/v2",
        "unexpected manifest schema",
    )
    require(
        manifest["id"] == "binary-sxs-bbh-0001-v2",
        "unexpected manifest id",
    )
    nonempty_string(manifest["title"], "$.title")
    status = exact_keys(
        manifest["scientificStatus"],
        {
            "classification",
            "description",
            "dynamicsSourceIsNumericalRelativity",
            "nearZoneSpacetimeUsedForLightPropagation",
            "slowLightGeodesicsIncluded",
            "prohibitedClaim",
        },
        "$.scientificStatus",
    )
    nonempty_string(status["description"], "$.scientificStatus.description")
    require(
        status.get("classification")
        == "NR-driven dynamics with weak-field fast-light rendering",
        "scientific classification drifted",
    )
    require(
        status.get("dynamicsSourceIsNumericalRelativity") is True,
        "NR dynamics source must be explicit",
    )
    require(
        status.get("nearZoneSpacetimeUsedForLightPropagation") is False
        and status.get("slowLightGeodesicsIncluded") is False,
        "manifest overclaims NR light propagation",
    )
    prohibited = str(status.get("prohibitedClaim", "")).lower()
    require(
        "do not" in prohibited and "nr ray tracing" in prohibited,
        "prohibited image claim is missing",
    )

    source = exact_keys(
        manifest["source"],
        {
            "catalog",
            "simulation",
            "level",
            "recordId",
            "recordDoi",
            "license",
            "waveformDataset",
            "artifacts",
        },
        "$.source",
    )
    require(
        source.get("catalog") == "SXS"
        and source.get("simulation") == "SXS:BBH:0001"
        and source.get("level") == "Lev5"
        and source.get("recordId") == 3_273_935
        and source.get("recordDoi")
        == "https://doi.org/10.5281/zenodo.3273935",
        "pinned SXS source drifted",
    )
    license_data = exact_keys(
        source["license"],
        {"status", "spdx", "attributionRequired"},
        "$.source.license",
    )
    require(
        license_data["status"] == "not-declared-in-pinned-zenodo-record"
        and license_data["spdx"] is None
        and boolean(
            license_data["attributionRequired"],
            "$.source.license.attributionRequired",
        )
        is True,
        "source license declaration drifted",
    )
    require(
        source.get("waveformDataset")
        == "Extrapolated_N2.dir/Y_l2_m2.dat",
        "waveform mode/extrapolation order drifted",
    )
    artifacts = source.get("artifacts")
    require(
        isinstance(artifacts, dict) and set(artifacts) == set(PINNED_SOURCE),
        "source artifact roles changed",
    )
    for role, (byte_length, md5, expected_sha256) in PINNED_SOURCE.items():
        artifact = exact_keys(
            artifacts[role],
            {"url", "path", "byteLength", "md5", "sha256"},
            f"$.source.artifacts.{role}",
        )
        expected_path = PINNED_SOURCE_PATHS[role]
        require(
            artifact.get("byteLength") == byte_length
            and artifact.get("md5") == md5
            and artifact.get("sha256") == expected_sha256,
            f"{role} pinned size/MD5/SHA-256 drifted",
        )
        require(
            artifact.get("path") == expected_path
            and artifact.get("url")
            == (
                "https://zenodo.org/api/records/3273935/files/"
                f"{expected_path}/content"
            ),
            f"{role} source URL is not pinned",
        )
        sha256_text(artifact["sha256"], f"$.source.artifacts.{role}.sha256")

    units = exact_keys(
        manifest["units"],
        {
            "system",
            "G",
            "c",
            "massUnit",
            "distanceUnit",
            "timeUnit",
            "referenceMassDefinition",
            "referenceMassTotalInCodeUnits",
        },
        "$.units",
    )
    require(
        units["system"] == "geometric"
        and finite(units["G"], "$.units.G") == 1.0
        and finite(units["c"], "$.units.c") == 1.0
        and units["massUnit"] == "M"
        and units["distanceUnit"] == "M"
        and units["timeUnit"] == "M",
        "geometric-unit ABI changed",
    )
    nonempty_string(
        units["referenceMassDefinition"],
        "$.units.referenceMassDefinition",
    )
    require(
        math.isclose(
            finite(
                units["referenceMassTotalInCodeUnits"],
                "$.units.referenceMassTotalInCodeUnits",
            ),
            PINNED_REFERENCE_MASS_TOTAL,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "$.units.referenceMassTotalInCodeUnits drifted from pinned metadata",
    )

    verify_physical_system(manifest)
    verify_time_and_events(manifest)
    verify_dynamics_contract(manifest)
    verify_renderer_contract(manifest)
    verify_generation_contract(manifest)

    time_reference = manifest["timeReference"]
    require(
        time_reference.get("protocolZeroEvent")
        == "maximum amplitude of Extrapolated_N2 h(2,2)",
        "protocol zero is not the waveform peak",
    )
    require(
        "distinct source coordinates"
        in str(time_reference.get("alignmentCaveat", "")),
        "waveform/horizon time-coordinate caveat is missing",
    )

    renderer = manifest["rendererAdapter"]
    require(
        renderer.get("shaderBundleId") == "binary-approx-v1"
        and renderer.get("lightPropagation") == "weak-field-fast-light"
        and renderer.get("nearZoneMetricConsumed") is False,
        "renderer adapter boundary drifted",
    )


def verify_samples(manifest: dict[str, Any]) -> dict[str, Any]:
    asset = manifest["dynamics"]["asset"]
    require(asset.get("uri") == "binary-sxs-bbh-0001-v2.samples.json", "asset URI changed")
    require(asset.get("columns") == EXPECTED_COLUMNS, "manifest columns changed")
    samples_path = MANIFEST.parent / asset["uri"]
    require(samples_path.is_file(), "sample asset is missing")
    require(
        samples_path.stat().st_size == asset["byteLength"],
        "sample byte length mismatch",
    )
    require(sha256(samples_path) == asset["sha256"], "sample SHA-256 mismatch")
    payload = strict_json(samples_path)
    exact_keys(payload, {"schema", "columns", "samples"}, "$samples")
    require(
        payload["schema"] == "blackhole.binary-dynamics-samples/v1",
        "sample schema changed",
    )
    require(payload["columns"] == EXPECTED_COLUMNS, "sample columns changed")
    rows = payload["samples"]
    require(isinstance(rows, list), "samples must be an array")
    require(len(rows) == asset["sampleCount"], "sample count mismatch")
    require(len(rows) >= 100, "track is implausibly sparse")

    previous_time = -math.inf
    previous_phase = -math.inf
    previous_blend = -math.inf
    invalid_seen = False
    first_invalid_values: tuple[float, float] | None = None
    for index, row in enumerate(rows):
        require(
            isinstance(row, list) and len(row) == len(EXPECTED_COLUMNS),
            f"sample {index} has the wrong width",
        )
        values = [
            finite(value, f"samples[{index}][{column}]")
            for column, value in enumerate(row[:6])
        ]
        validity = row[6]
        require(validity in (0, 1) and not isinstance(validity, bool), f"sample {index} validity is invalid")
        require(values[0] > previous_time, f"sample {index} time does not increase")
        require(values[1] > 0, f"sample {index} separation is invalid")
        require(values[2] >= previous_phase, f"sample {index} phase runs backward")
        require(
            0 <= values[5] <= 1 and values[5] >= previous_blend,
            f"sample {index} render blend is invalid",
        )
        if validity == 0:
            if not invalid_seen:
                first_invalid_values = (values[1], values[2])
            invalid_seen = True
            require(
                first_invalid_values is not None
                and math.isclose(values[1], first_invalid_values[0], abs_tol=1e-9)
                and math.isclose(values[2], first_invalid_values[1], abs_tol=1e-9),
                f"sample {index} invents post-horizon A/B motion",
            )
        else:
            require(not invalid_seen, f"sample {index} restores expired horizons")
        previous_time = values[0]
        previous_phase = values[2]
        previous_blend = values[5]

    require(
        math.isclose(rows[0][0], manifest["dynamics"]["firstTimeM"], abs_tol=1e-9)
        and math.isclose(rows[-1][0], manifest["dynamics"]["finalTimeM"], abs_tol=1e-9),
        "sample range disagrees with manifest",
    )
    require(rows[0][5] == 0 and rows[-1][5] == 1, "render transition is incomplete")
    return payload


def verify_source_acceptance(
    manifest: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    rows = payload["samples"]
    events = manifest["events"]
    ordered = [
        finite(events[name]["tProtocolM"], f"events.{name}.tProtocolM")
        for name in (
            "relaxation",
            "individualHorizonsLast",
            "commonApparentHorizonFirst",
            "waveformPeak",
            "playbackEnd",
        )
    ]
    require(
        all(current > previous for previous, current in zip(ordered, ordered[1:])),
        "physical/source events are not ordered",
    )
    require(ordered[3] == 0, "waveform peak must be protocol t=0")
    peak = linear_sample(rows, 0.0)
    peak_amplitude = math.hypot(peak[3], peak[4])
    require(
        math.isclose(
            peak_amplitude,
            manifest["timeReference"]["waveformPeakAmplitude"],
            rel_tol=0,
            abs_tol=2e-9,
        ),
        "bundled h22 peak amplitude disagrees with source",
    )
    sampled_peak = max(
        rows,
        key=lambda row: math.hypot(float(row[3]), float(row[4])),
    )
    require(
        math.isclose(sampled_peak[0], 0.0, abs_tol=1e-12),
        "bundled waveform maximum is not at protocol t=0",
    )

    remnant = manifest["physicalSystem"]["remnant"]
    mass = finite(remnant["massFraction"], "remnant.massFraction")
    spin = [finite(value, "remnant.dimensionlessSpin") for value in remnant["dimensionlessSpin"]]
    require(
        math.isclose(mass, PINNED_REMNANT_MASS, abs_tol=1e-12),
        "remnant mass disagrees with pinned metadata",
    )
    require(
        math.isclose(spin[2], PINNED_REMNANT_SPIN_Z, abs_tol=1e-12),
        "remnant spin disagrees with pinned metadata",
    )
    final_diagnostic = remnant["finalHorizonDiagnostic"]
    require(
        abs(final_diagnostic["massFraction"] - mass) / mass < 1e-3,
        "metadata and final horizon mass differ by >= 0.1%",
    )
    require(
        abs(final_diagnostic["dimensionlessSpin"][2] - spin[2]) / spin[2] < 1e-3,
        "metadata and final horizon spin differ by >= 0.1%",
    )

    measured = manifest["dynamics"]["measuredMaxInterpolationError"]
    declared = manifest["dynamics"]["declaredMaxInterpolationError"]
    require(
        measured["orbitalPhaseUnwrappedRad"] < 0.01,
        "orbital phase interpolation residual exceeds 0.01 rad",
    )
    for channel in (
        "separationM",
        "orbitalPhaseUnwrappedRad",
        "h22Real",
        "h22Imag",
        "renderTopologyBlend",
    ):
        require(
            0 <= measured[channel] <= declared[channel],
            f"{channel} measured interpolation error exceeds declaration",
        )


def verify_playback(manifest: dict[str, Any]) -> None:
    playback = exact_keys(
        manifest["playback"],
        {
            "cycleDurationSecondsAtNominalRate",
            "endHoldSeconds",
            "loop",
            "scrubCoordinate",
            "slowMotion",
        },
        "$.playback",
    )
    require(playback.get("scrubCoordinate") == "linear protocol time", "scrub coordinate changed")
    require(
        boolean(playback.get("loop"), "$.playback.loop") is True,
        "binary playback must loop",
    )
    cycle_duration = finite(
        playback.get("cycleDurationSecondsAtNominalRate"),
        "$.playback.cycleDurationSecondsAtNominalRate",
    )
    require(
        cycle_duration > 0.0,
        "playback cycle duration must be positive",
    )
    end_hold = finite(playback.get("endHoldSeconds"), "playback.endHoldSeconds")
    require(
        0 <= end_hold < cycle_duration,
        "end hold must be non-negative and shorter than the nominal cycle",
    )
    slow = exact_keys(
        playback["slowMotion"],
        {
            "enabledByDefault",
            "startTimeM",
            "endTimeM",
            "rateMultiplier",
            "status",
        },
        "$.playback.slowMotion",
    )
    boolean(slow["enabledByDefault"], "$.playback.slowMotion.enabledByDefault")
    start = finite(slow["startTimeM"], "slowMotion.startTimeM")
    end = finite(slow["endTimeM"], "slowMotion.endTimeM")
    multiplier = finite(slow["rateMultiplier"], "slowMotion.rateMultiplier")
    require(start < 0 < end, "slow-motion window must contain the waveform peak")
    require(
        manifest["dynamics"]["firstTimeM"] <= start
        and end <= manifest["dynamics"]["finalTimeM"],
        "slow-motion window is outside the playback range",
    )
    require(0 < multiplier < 1, "slow-motion multiplier must be in (0,1)")
    require(
        "presentation-only"
        in nonempty_string(slow.get("status"), "$.playback.slowMotion.status"),
        "slow motion is not labelled as presentation-only",
    )


def verify_fail_closed_mutation_guards(manifest: dict[str, Any]) -> None:
    """Prove that high-risk manifest mutations are rejected by this verifier."""

    mutations: tuple[tuple[str, tuple[Any, ...], Any], ...] = (
        (
            "negative body mass fraction",
            ("physicalSystem", "bodies", 0, "massFraction"),
            -4.0 / 9.0,
        ),
        (
            "body mass fractions no longer sum to one",
            ("physicalSystem", "bodies", 0, "massFraction"),
            0.4,
        ),
        (
            "negative observer radius",
            ("rendererDefaults", "observerRadiusM"),
            -1.0,
        ),
        (
            "negative ray-step budget",
            ("rendererDefaults", "raySteps"),
            -5,
        ),
        (
            "zero playback cycle duration",
            ("playback", "cycleDurationSecondsAtNominalRate"),
            0.0,
        ),
        (
            "untrusted source digest",
            ("source", "artifacts", "metadata", "sha256"),
            "0" * 64,
        ),
        (
            "shader mass ABI drift",
            ("rendererAdapter", "massAbi", 0),
            "unexpected",
        ),
        (
            "shader state ABI drift",
            ("rendererAdapter", "stateAbi", 0),
            "unexpected",
        ),
    )
    for name, path, replacement in mutations:
        candidate = copy.deepcopy(manifest)
        parent: Any = candidate
        for component in path[:-1]:
            parent = parent[component]
        parent[path[-1]] = replacement
        try:
            verify_manifest(candidate)
            verify_playback(candidate)
        except ValidationError:
            continue
        raise ValidationError(f"internal fail-closed guard accepted {name}")

    unknown_key = copy.deepcopy(manifest)
    unknown_key["dynamics"]["unexpected"] = 1
    try:
        verify_manifest(unknown_key)
    except ValidationError:
        return
    raise ValidationError("internal fail-closed guard accepted an unknown dynamics key")


def main() -> int:
    manifest = strict_json(MANIFEST)
    verify_manifest(manifest)
    verify_fail_closed_mutation_guards(manifest)
    payload = verify_samples(manifest)
    verify_source_acceptance(manifest, payload)
    verify_playback(manifest)
    require(
        manifest["generation"]["generatorSha256"] == sha256(GENERATOR),
        "generator changed without regenerating the assets",
    )
    print("SXS binary dynamics checks passed")
    print(f"  manifest = {MANIFEST.relative_to(ROOT)}")
    print(f"  samples = {len(payload['samples'])}")
    print(
        "  time range = "
        f"{payload['samples'][0][0]:.3f} M .. "
        f"{payload['samples'][-1][0]:.3f} M"
    )
    print(
        "  common horizon / h22 peak = "
        f"{manifest['events']['commonApparentHorizonFirst']['tProtocolM']:.6f} M "
        "/ 0 M"
    )
    print(
        "  max orbital phase residual = "
        f"{manifest['dynamics']['measuredMaxInterpolationError']['orbitalPhaseUnwrappedRad']:.3e} rad"
    )
    print(
        "  remnant mass / spin-z = "
        f"{manifest['physicalSystem']['remnant']['massFraction']:.12f} / "
        f"{manifest['physicalSystem']['remnant']['dimensionlessSpin'][2]:.12f}"
    )
    print(
        "  source-manifest renderer boundary = legacy weak-field fast-light; "
        "the WebGPU strong-field layer is verified separately"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"SXS binary dynamics validation failed: {error}")
        raise SystemExit(1) from error
