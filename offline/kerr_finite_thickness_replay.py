"""Deterministic replay of finite-thickness exact-Kerr spectral products.

This verifier reconstructs the closed
``kerr-finite-thickness-spectral-ray-sampler/v1`` configuration from an
authenticated scientific-frame manifest, reruns the real adaptive producer,
and requires every public pixel record to be byte-identical.  It is a
same-code-family consistency check, not an independent physics oracle.  The
supported producer is a stationary prescribed photosphere with an equatorial
Novikov--Thorne thermal proxy; it does not include returning radiation, a
solved atmosphere, GRMHD, or numerical relativity.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, NoReturn

from offline.adaptive_frame import integrate_spectral_pixel
from offline.cie_color import (
    CIE_ROW_COUNT,
    DEFAULT_CIE_CSV,
    DEFAULT_CIE_METADATA,
    cie_1931_frequency_grid_hz,
    load_authenticated_cie_1931_2deg,
)
from offline.job import InputArtifact, canonical_json_bytes
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination
from offline.kerr_disk import StationaryNovikovThorneDisk
from offline.kerr_finite_thickness import (
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_frame import (
    IMPLEMENTATION_ID,
    KerrFiniteThicknessRaySampler,
)
from offline.kerr_finite_thickness_surface import (
    KerrFiniteThicknessMultiSurface,
)
from offline.kerr_nt_replay import (
    DEFAULT_REPLAY_LIMITS,
    MAXIMUM_OFFICIAL_CIE_FREQUENCY_BINS,
    KerrNtReplayError,
    ReplayResourceLimits,
    _adaptive_options,
    _boolean,
    _grid,
    _integer,
    _mapping,
    _number,
    _preflight_manifest,
    _ray_options,
    _read_relative_file,
    _read_stable_regular,
    _record_differences,
    _string,
    _surface_options,
)
from offline.kerr_disk_frame import (
    DarkEscapedObserverSpectrum,
    PowerLawEscapedObserverSpectrum,
)
from offline.spectral_frame import (
    SpectralPixelLayout,
    pack_adaptive_pixel,
    unpack_spectral_pixel,
)
from offline.spectral_product import (
    ADAPTIVE_TILE_ALGORITHM_VERSION,
    ADAPTIVE_TILE_PRODUCER_ID,
    PRODUCT_SCHEMA,
    default_numeric_backend_descriptor,
)
from scripts.render_offline_kerr_finite_thickness_frame import (
    CIE_CSV_INPUT_URI,
    CIE_METADATA_INPUT_URI,
    PRODUCER_SOURCE_FILES,
)
from scripts.verify_offline_spectral_frame import (
    DEFAULT_SCHEMA,
    validate_scientific_spectral_frame,
)


ROOT = Path(__file__).resolve().parents[1]

# The shared bounded-artifact helpers deliberately use this exact exception.
# Export a finite-thickness-specific public spelling without forking the common
# TOCTOU and resource-envelope implementation.
KerrFiniteThicknessReplayError = KerrNtReplayError


def _fail(path: str, message: str) -> NoReturn:
    raise KerrFiniteThicknessReplayError(f"{path}: {message}")


def _escaped_spectrum(value: Any):
    wrapper_path = "$.sampler.descriptor.escapedObserverSpectrum"
    wrapper = _mapping(value, wrapper_path)
    descriptor = _mapping(wrapper.get("descriptor"), f"{wrapper_path}.descriptor")
    implementation = _string(
        descriptor.get("implementationId"),
        f"{wrapper_path}.descriptor.implementationId",
    )
    if implementation == "dark-observer-frame-escape-spectrum/v1":
        return DarkEscapedObserverSpectrum()
    if implementation == "power-law-observer-frame-escape-spectrum/v1":
        try:
            return PowerLawEscapedObserverSpectrum(
                reference_specific_intensity_nu=_number(
                    descriptor["referenceSpecificIntensityNu"],
                    f"{wrapper_path}.descriptor.referenceSpecificIntensityNu",
                ),
                reference_frequency_hz=_number(
                    descriptor["referenceFrequencyHz"],
                    f"{wrapper_path}.descriptor.referenceFrequencyHz",
                ),
                spectral_index=_number(
                    descriptor["spectralIndex"],
                    f"{wrapper_path}.descriptor.spectralIndex",
                ),
            )
        except KeyError as error:
            _fail(
                f"{wrapper_path}.descriptor",
                f"missing power-law field {error.args[0]!r}",
            )
    _fail(
        f"{wrapper_path}.descriptor.implementationId",
        f"unsupported closed escape spectrum {implementation!r}",
    )


def reconstruct_kerr_finite_thickness_sampler(
    descriptor: Mapping[str, Any],
) -> KerrFiniteThicknessRaySampler:
    """Reconstruct and content-round-trip the supported production sampler."""

    raw = _mapping(descriptor, "$.sampler.descriptor")
    if raw.get("implementationId") != IMPLEMENTATION_ID:
        _fail(
            "$.sampler.descriptor.implementationId",
            f"only {IMPLEMENTATION_ID!r} can be replayed",
        )
    if _integer(raw.get("version"), "$.sampler.descriptor.version") != 1:
        _fail("$.sampler.descriptor.version", "only sampler version 1 is supported")

    try:
        metric_raw = _mapping(raw["metric"], "$.sampler.descriptor.metric")
        if (
            _string(metric_raw["sourceId"], "$.sampler.descriptor.metric.sourceId")
            != "analytic-kerr-kerr-schild"
            or _boolean(
                metric_raw["timeDependent"],
                "$.sampler.descriptor.metric.timeDependent",
            )
        ):
            _fail(
                "$.sampler.descriptor.metric",
                "replay requires the stationary analytic Kerr-Schild provider",
            )
        metric = KerrKerrSchildMetric(
            mass_m=_number(metric_raw["massM"], "$.sampler.descriptor.metric.massM"),
            spin_a_m=_number(
                metric_raw["signedSpinAM"],
                "$.sampler.descriptor.metric.signedSpinAM",
            ),
            singularity_guard_m=_number(
                metric_raw["singularityGuardM"],
                "$.sampler.descriptor.metric.singularityGuardM",
            ),
        )

        finite_raw = _mapping(
            raw["finiteThicknessSurface"],
            "$.sampler.descriptor.finiteThicknessSurface",
        )
        calibration = StationaryKerrFiniteThicknessCalibration(
            dimensionless_spin=_number(
                finite_raw["dimensionlessSpinMagnitude"],
                "$.sampler.descriptor.finiteThicknessSurface."
                "dimensionlessSpinMagnitude",
            ),
            eddington_scaled_mass_accretion_rate=_number(
                finite_raw["eddingtonScaledMassAccretionRate"],
                "$.sampler.descriptor.finiteThicknessSurface."
                "eddingtonScaledMassAccretionRate",
            ),
            orientation=_string(
                finite_raw["orientation"],
                "$.sampler.descriptor.finiteThicknessSurface.orientation",
            ),
            outer_radius_over_mass=_number(
                finite_raw["outerRadiusOverMass"],
                "$.sampler.descriptor.finiteThicknessSurface.outerRadiusOverMass",
            ),
            thinness_gate_maximum_h_over_rho=_number(
                finite_raw["thinnessGateMaximumHOverRho"],
                "$.sampler.descriptor.finiteThicknessSurface."
                "thinnessGateMaximumHOverRho",
            ),
        )
        surface = KerrFiniteThicknessMultiSurface(metric, calibration)

        termination_raw = _mapping(
            raw["termination"], "$.sampler.descriptor.termination"
        )
        capture_target = _string(
            termination_raw["captureTargetId"],
            "$.sampler.descriptor.termination.captureTargetId",
        )
        escape_target = _string(
            termination_raw["escapeTargetId"],
            "$.sampler.descriptor.termination.escapeTargetId",
        )
        if capture_target not in {
            "analytic-kerr-event-horizon",
            "analytic-kerr-stretched-horizon",
        } or escape_target != "analytic-kerr-escape-worldtube":
            _fail(
                "$.sampler.descriptor.termination",
                "unsupported exact Kerr terminal-surface identity",
            )
        termination_spin = _number(
            termination_raw["spinAM"],
            "$.sampler.descriptor.termination.spinAM",
        )
        capture_radius = _number(
            termination_raw["captureRadiusM"],
            "$.sampler.descriptor.termination.captureRadiusM",
        )
        horizon_radius = metric.outer_horizon_radius_m
        if capture_target == "analytic-kerr-event-horizon":
            if capture_radius.hex() != horizon_radius.hex():
                _fail(
                    "$.sampler.descriptor.termination.captureRadiusM",
                    "event-horizon target requires the exact Kerr outer-horizon "
                    "radius",
                )
        elif capture_radius <= horizon_radius:
            _fail(
                "$.sampler.descriptor.termination.captureRadiusM",
                "stretched-horizon target must lie strictly outside the Kerr "
                "horizon",
            )
        termination = KerrOblateTermination(
            spin_a_m=termination_spin,
            capture_radius_m=capture_radius,
            escape_radius_m=_number(
                termination_raw["escapeRadiusM"],
                "$.sampler.descriptor.termination.escapeRadiusM",
            ),
            capture_target_id=capture_target,
            escape_target_id=escape_target,
        )

        disk_raw = _mapping(
            raw["diskThermalProxy"],
            "$.sampler.descriptor.diskThermalProxy",
        )
        disk = StationaryNovikovThorneDisk(
            metric=metric,
            black_hole_mass_kg=_number(
                disk_raw["blackHoleMassKg"],
                "$.sampler.descriptor.diskThermalProxy.blackHoleMassKg",
            ),
            mass_accretion_rate_kg_s=_number(
                disk_raw["massAccretionRateKgS"],
                "$.sampler.descriptor.diskThermalProxy.massAccretionRateKgS",
            ),
            orientation=_string(
                disk_raw["orientation"],
                "$.sampler.descriptor.diskThermalProxy.orientation",
            ),
            colour_correction=_number(
                disk_raw["colourCorrection"],
                "$.sampler.descriptor.diskThermalProxy.colourCorrection",
            ),
        )

        ray_raw = _mapping(raw["rayOptions"], "$.sampler.descriptor.rayOptions")
        surface_raw = _mapping(
            raw["surfaceOptions"], "$.sampler.descriptor.surfaceOptions"
        )
        fine_options = _ray_options(
            ray_raw["fine"], "$.sampler.descriptor.rayOptions.fine"
        )
        fine_surface_options = _surface_options(
            surface_raw["fine"], "$.sampler.descriptor.surfaceOptions.fine"
        )
        observer_raw = _mapping(raw["observer"], "$.sampler.descriptor.observer")
        convergence = _mapping(
            raw["convergence"], "$.sampler.descriptor.convergence"
        )
        transfer = _mapping(
            raw["frequencyTransfer"],
            "$.sampler.descriptor.frequencyTransfer",
        )
        sampler = KerrFiniteThicknessRaySampler(
            metric=metric,
            observer_radius_m=_number(
                observer_raw["radiusM"], "$.sampler.descriptor.observer.radiusM"
            ),
            termination=termination,
            surface=surface,
            disk=disk,
            escaped_observer_spectrum=_escaped_spectrum(
                raw["escapedObserverSpectrum"]
            ),
            fine_options=fine_options,
            surface_options=fine_surface_options,
            observer_theta_rad=_number(
                observer_raw["thetaRad"], "$.sampler.descriptor.observer.thetaRad"
            ),
            observer_phi_ks_rad=_number(
                observer_raw["phiKsRad"], "$.sampler.descriptor.observer.phiKsRad"
            ),
            observer_coordinate_time_m=_number(
                observer_raw["coordinateTimeM"],
                "$.sampler.descriptor.observer.coordinateTimeM",
            ),
            coarse_tolerance_multiplier=_number(
                convergence["coarseToleranceMultiplier"],
                "$.sampler.descriptor.convergence.coarseToleranceMultiplier",
            ),
            terminal_event_tolerance_m=_number(
                convergence["terminalEventToleranceM"],
                "$.sampler.descriptor.convergence.terminalEventToleranceM",
            ),
            terminal_covector_tolerance=_number(
                convergence["terminalCovectorTolerance"],
                "$.sampler.descriptor.convergence.terminalCovectorTolerance",
            ),
            disk_radius_absolute_tolerance_m=_number(
                convergence["diskRadiusAbsoluteToleranceM"],
                "$.sampler.descriptor.convergence.diskRadiusAbsoluteToleranceM",
            ),
            disk_radius_relative_tolerance=_number(
                convergence["diskRadiusRelativeTolerance"],
                "$.sampler.descriptor.convergence.diskRadiusRelativeTolerance",
            ),
            frequency_shift_relative_tolerance=_number(
                convergence["frequencyShiftRelativeTolerance"],
                "$.sampler.descriptor.convergence.frequencyShiftRelativeTolerance",
            ),
            emission_cosine_absolute_tolerance=_number(
                convergence["emissionCosineAbsoluteTolerance"],
                "$.sampler.descriptor.convergence.emissionCosineAbsoluteTolerance",
            ),
            specific_intensity_absolute_tolerance=_number(
                convergence["specificIntensityAbsoluteTolerance"],
                "$.sampler.descriptor.convergence.specificIntensityAbsoluteTolerance",
            ),
            specific_intensity_relative_tolerance=_number(
                convergence["specificIntensityRelativeTolerance"],
                "$.sampler.descriptor.convergence.specificIntensityRelativeTolerance",
            ),
            escape_direction_tolerance_rad=_number(
                convergence["escapeDirectionToleranceRad"],
                "$.sampler.descriptor.convergence.escapeDirectionToleranceRad",
            ),
            frequency_null_residual_limit=_number(
                transfer["nullResidualLimit"],
                "$.sampler.descriptor.frequencyTransfer.nullResidualLimit",
            ),
            conserved_quantity_tolerance=_number(
                transfer["conservedQuantityTolerance"],
                "$.sampler.descriptor.frequencyTransfer.conservedQuantityTolerance",
            ),
            recorded_path_absolute_tolerance=_number(
                transfer["requestedRecordedPathAbsoluteTolerance"],
                "$.sampler.descriptor.frequencyTransfer."
                "requestedRecordedPathAbsoluteTolerance",
            ),
            recorded_path_relative_tolerance=_number(
                transfer["requestedRecordedPathRelativeTolerance"],
                "$.sampler.descriptor.frequencyTransfer."
                "requestedRecordedPathRelativeTolerance",
            ),
            boundary_value_tolerance_m=_number(
                transfer["boundaryValueToleranceM"],
                "$.sampler.descriptor.frequencyTransfer.boundaryValueToleranceM",
            ),
            emitter_event_tolerance_m=_number(
                transfer["emitterEventToleranceM"],
                "$.sampler.descriptor.frequencyTransfer.emitterEventToleranceM",
            ),
        )
    except KeyError as error:
        _fail("$.sampler.descriptor", f"missing required field {error.args[0]!r}")
    except KerrFiniteThicknessReplayError:
        raise
    except (ArithmeticError, TypeError, ValueError) as error:
        _fail("$.sampler.descriptor", f"cannot reconstruct sampler: {error}")

    if canonical_json_bytes(sampler.descriptor()) != canonical_json_bytes(raw):
        _fail(
            "$.sampler.descriptor",
            "descriptor is not the exact content-complete output of the "
            "supported finite-thickness sampler",
        )
    return sampler


def _current_input_snapshot(
    manifest: Mapping[str, Any],
    source_root: Path,
    maximum_source_file_bytes: int,
) -> tuple[tuple[InputArtifact, ...], tuple[InputArtifact, ...]]:
    """Authenticate current producer source and official CIE bytes."""

    sources: list[InputArtifact] = []
    for relative in PRODUCER_SOURCE_FILES:
        payload = _read_stable_regular(
            source_root / relative,
            maximum_source_file_bytes,
            f"source:{relative.as_posix()}",
        )
        sources.append(
            InputArtifact(
                f"repo-source://{relative.as_posix()}",
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            )
        )
    science: list[InputArtifact] = []
    for uri, path in (
        (CIE_CSV_INPUT_URI, Path(DEFAULT_CIE_CSV)),
        (CIE_METADATA_INPUT_URI, Path(DEFAULT_CIE_METADATA)),
    ):
        payload = _read_stable_regular(
            path,
            maximum_source_file_bytes,
            f"science:{uri}",
        )
        science.append(
            InputArtifact(uri, len(payload), hashlib.sha256(payload).hexdigest())
        )

    current_sources = tuple(sorted(sources))
    current_science = tuple(sorted(science))
    try:
        job_spec = manifest["producer"]["jobSpec"]
        declared_inputs = job_spec["inputs"]
        declared_source_hashes = job_spec["producerSourceHashes"]
    except (KeyError, TypeError) as error:
        _fail("$.producer.jobSpec.inputs", f"cannot audit input snapshot: {error}")
    expected_inputs = [
        artifact.as_dict()
        for artifact in sorted((*current_sources, *current_science))
    ]
    if declared_inputs != expected_inputs:
        _fail(
            "$.producer.jobSpec.inputs",
            "declared producer/CIE inputs do not exactly match this replay checkout",
        )
    expected_source_hashes = sorted(
        {artifact.sha256 for artifact in current_sources}
    )
    if declared_source_hashes != expected_source_hashes:
        _fail(
            "$.producer.jobSpec.producerSourceHashes",
            "producer source hashes do not exactly match current source artifacts",
        )
    return current_sources, current_science


def _official_cie_frequencies() -> tuple[float, ...]:
    try:
        table = load_authenticated_cie_1931_2deg(
            Path(DEFAULT_CIE_CSV),
            Path(DEFAULT_CIE_METADATA),
        )
        frequencies = cie_1931_frequency_grid_hz(table)
    except (ArithmeticError, OSError, TypeError, ValueError) as error:
        _fail("$.observerFrequencyBinsHz", f"cannot load official CIE grid: {error}")
    if len(frequencies) != CIE_ROW_COUNT:
        _fail(
            "$.observerFrequencyBinsHz",
            f"official CIE grid must contain exactly {CIE_ROW_COUNT} bins",
        )
    return frequencies


def _enforce_sampler_limits(
    sampler: KerrFiniteThicknessRaySampler,
    limits: ReplayResourceLimits,
) -> None:
    ray_options = (
        ("fine", sampler.fine_options),
        ("coarseDerived", sampler._coarse_ray_options),
    )
    for label, options in ray_options:
        checks = (
            (
                options.maximum_accepted_steps,
                limits.maximum_ray_accepted_steps,
                "maximumAcceptedSteps",
            ),
            (
                options.maximum_rejected_steps,
                limits.maximum_ray_rejected_steps,
                "maximumRejectedSteps",
            ),
            (
                options.event_maximum_iterations,
                limits.maximum_ray_event_iterations,
                "eventMaximumIterations",
            ),
        )
        for actual, maximum, field in checks:
            if actual > maximum:
                _fail(
                    f"$.sampler.descriptor.rayOptions.{label}.{field}",
                    f"value exceeds replay limit {maximum}",
                )
        if options.maximum_affine_length > limits.maximum_affine_length_m:
            _fail(
                f"$.sampler.descriptor.rayOptions.{label}.maximumAffineLength",
                f"value exceeds replay limit {limits.maximum_affine_length_m}",
            )

    surface_options = (
        ("fine", sampler.surface_options),
        ("coarseDerived", sampler._coarse_surface_options),
    )
    for label, options in surface_options:
        checks = (
            (
                options.maximum_iterations,
                limits.maximum_surface_iterations,
                "maximumIterations",
            ),
            (
                options.maximum_reintegrations,
                limits.maximum_surface_reintegrations,
                "maximumReintegrations",
            ),
            (
                options.subdivisions_per_segment,
                limits.maximum_surface_subdivisions_per_segment,
                "subdivisionsPerSegment",
            ),
        )
        for actual, maximum, field in checks:
            if actual > maximum:
                _fail(
                    f"$.sampler.descriptor.surfaceOptions.{label}.{field}",
                    f"value exceeds replay limit {maximum}",
                )


def validate_kerr_finite_thickness_replay(
    manifest_path: Path | str,
    schema_path: Path | str = DEFAULT_SCHEMA,
    *,
    limits: ReplayResourceLimits = DEFAULT_REPLAY_LIMITS,
    source_root: Path | str = ROOT,
) -> dict[str, Any]:
    """Authenticate, reconstruct, and byte-replay one finite-height frame."""

    if not isinstance(limits, ReplayResourceLimits):
        raise TypeError("limits must be ReplayResourceLimits")
    path = Path(manifest_path).absolute()
    if path.name != "manifest.json":
        _fail("$", "v1 replay input must be named 'manifest.json'")
    schema = Path(schema_path).absolute()
    schema_payload = _read_stable_regular(
        schema, limits.maximum_manifest_bytes, "$schema"
    )
    default_schema_payload = _read_stable_regular(
        Path(DEFAULT_SCHEMA).absolute(),
        limits.maximum_manifest_bytes,
        "$defaultSchema",
    )
    if schema_payload != default_schema_payload:
        _fail("$schema", "replay requires the repository's exact strict v1 schema")

    manifest, manifest_payload = _preflight_manifest(path, limits)
    structural_report = validate_scientific_spectral_frame(path, schema)
    if _read_stable_regular(path, limits.maximum_manifest_bytes, "$") != manifest_payload:
        _fail("$", "manifest changed after structural verification")
    if manifest.get("schema") != PRODUCT_SCHEMA:
        _fail("$.schema", f"unsupported product schema {manifest.get('schema')!r}")
    producer = manifest["producer"]
    if (
        producer["id"] != ADAPTIVE_TILE_PRODUCER_ID
        or producer["algorithmVersion"] != ADAPTIVE_TILE_ALGORITHM_VERSION
    ):
        _fail("$.producer", "unsupported adaptive tile producer identity")

    source_root_path = Path(source_root).absolute()
    if source_root_path.resolve() != ROOT.resolve():
        _fail(
            "$.producer.jobSpec.inputs",
            "source replay is restricted to the checkout that loaded this verifier",
        )
    sources_before, science_before = _current_input_snapshot(
        manifest, source_root_path, limits.maximum_source_file_bytes
    )

    official_frequencies = _official_cie_frequencies()
    manifest_frequencies = tuple(manifest["observerFrequencyBinsHz"])
    if manifest_frequencies != official_frequencies:
        _fail(
            "$.observerFrequencyBinsHz",
            "finite-thickness v1 replay requires the exact official CIE 471-bin grid",
        )
    layout = SpectralPixelLayout(manifest_frequencies)
    if dict(layout.descriptor()) != manifest["pixelLayout"]:
        _fail("$.pixelLayout", "layout does not round-trip exactly")
    grid = _grid(manifest["frame"])
    options = _adaptive_options(manifest["adaptivePixelOptions"])
    current_backend = default_numeric_backend_descriptor()
    if manifest["runtimeNumericBackend"]["descriptor"] != current_backend:
        _fail(
            "$.runtimeNumericBackend.descriptor",
            "byte-exact replay requires the declared current CPython/binary64 backend",
        )
    sampler = reconstruct_kerr_finite_thickness_sampler(
        manifest["sampler"]["descriptor"]
    )
    _enforce_sampler_limits(sampler, limits)

    declared_records = structural_report["recordCount"]
    if declared_records * options.maximum_ray_evaluations * 2 > (
        limits.maximum_total_ray_evaluations
    ):
        _fail(
            "$.adaptivePixelOptions.maximumRayEvaluations",
            "declared fine-plus-coarse geodesic budget exceeds replay resource limit",
        )

    record_count = 0
    total_ray_samples = 0
    maximum_accepted_steps = 0
    maximum_rejected_steps = 0
    root = path.parent
    for tile_index, entry in enumerate(manifest["tiles"]):
        tile = entry["tile"]
        tile_path = f"$.tiles[{tile_index}]"
        payload = _read_relative_file(
            root,
            entry["payload"]["uri"],
            entry["payload"]["byteLength"],
            f"{tile_path}.payload.uri",
        )
        if hashlib.sha256(payload).hexdigest() != entry["payload"]["sha256"]:
            _fail(f"{tile_path}.payload.sha256", "tile changed after authentication")
        for local_index in range(entry["recordCount"]):
            local_y, local_x = divmod(local_index, tile["width"])
            x = tile["x"] + local_x
            y = tile["y"] + local_y
            x_min, x_max, y_min, y_max = grid.pixel_bounds(x, y)
            try:
                result = integrate_spectral_pixel(
                    sampler,
                    layout.observer_frequencies_hz,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    options=options,
                )
                expected = pack_adaptive_pixel(layout, result, options)
            except (ArithmeticError, RuntimeError, TypeError, ValueError) as error:
                _fail(
                    f"{tile_path}.records[{local_index}]",
                    f"replay computation failed closed: {error}",
                )
            offset = local_index * layout.record_bytes
            actual = payload[offset : offset + layout.record_bytes]
            if actual != expected:
                differences = ", ".join(
                    _record_differences(actual, expected, layout)
                )
                _fail(
                    f"{tile_path}.records[{local_index}]",
                    "deterministic numerical replay mismatch in " + differences,
                )
            record = unpack_spectral_pixel(layout, expected)
            record_count += 1
            total_ray_samples += record.sample_count
            maximum_accepted_steps = max(
                maximum_accepted_steps, record.maximum_accepted_steps
            )
            maximum_rejected_steps = max(
                maximum_rejected_steps, record.maximum_rejected_steps
            )
        if _read_relative_file(
            root,
            entry["payload"]["uri"],
            entry["payload"]["byteLength"],
            f"{tile_path}.payload.uri",
        ) != payload:
            _fail(f"{tile_path}.payload.uri", "tile changed during replay")

    sources_after, science_after = _current_input_snapshot(
        manifest, source_root_path, limits.maximum_source_file_bytes
    )
    if sources_after != sources_before or science_after != science_before:
        _fail("$.producer.jobSpec.inputs", "producer/CIE input changed during replay")
    if _read_stable_regular(path, limits.maximum_manifest_bytes, "$") != manifest_payload:
        _fail("$", "manifest changed during replay")
    if _read_stable_regular(
        schema, limits.maximum_manifest_bytes, "$schema"
    ) != schema_payload:
        _fail("$schema", "schema changed during replay")
    final_structural_report = validate_scientific_spectral_frame(path, schema)
    if final_structural_report != structural_report:
        _fail("$", "structural evidence changed during replay")
    sources_final, science_final = _current_input_snapshot(
        manifest, source_root_path, limits.maximum_source_file_bytes
    )
    if sources_final != sources_before or science_final != science_before:
        _fail("$.producer.jobSpec.inputs", "producer/CIE input changed during replay")
    if default_numeric_backend_descriptor() != current_backend:
        _fail("$.runtimeNumericBackend.descriptor", "numeric backend changed during replay")
    if _official_cie_frequencies() != official_frequencies:
        _fail("$.observerFrequencyBinsHz", "official CIE input changed during replay")
    if record_count != structural_report["recordCount"]:
        _fail("$.tiles", "replayed record count disagrees with structural evidence")

    return {
        "id": manifest["id"],
        "independentPhysicsOracle": False,
        "includesReturningRadiation": False,
        "isCompleteGeneralRelativisticRadiativeTransfer": False,
        "isGeneralRelativisticMagnetohydrodynamics": False,
        "isNumericalRelativitySolver": False,
        "maximumAcceptedSteps": maximum_accepted_steps,
        "maximumRejectedSteps": maximum_rejected_steps,
        "numericBackendCurrentMatch": True,
        "officialCie471CurrentMatch": True,
        "physicsReplayVerified": True,
        "productBoundSourceHashesCurrentMatch": True,
        "recordCount": record_count,
        "replayScope": (
            "same-code-family deterministic numerical replay; byte-exact public "
            "spectral ABI, aggregate source coverage/masks, and diagnostics"
        ),
        "scientificScope": (
            "stationary prescribed finite-height photosphere with equatorial "
            "Novikov-Thorne thermal proxy; no returning radiation, solved "
            "atmosphere, GRMHD, complete GRRT, or numerical relativity"
        ),
        "sourceArtifactCount": len(sources_before),
        "scienceArtifactCount": len(science_before),
        "sourceHashScope": (
            "exact current bytes for every declared producer source and official "
            "CIE input; source-closure completeness remains a producer contract"
        ),
        "status": (
            "exact-kerr-finite-thickness-deterministic-numerical-replay-conformant"
        ),
        "structuralContractVerified": True,
        "tamperDetectionScope": (
            "all replayed payload bytes and closed configuration/output "
            "consistency; a fully resealed byte-equivalent alternative manifest "
            "requires an external expected hash or signature"
        ),
        "tileCount": structural_report["tileCount"],
        "totalGeodesicTraces": total_ray_samples * 2,
        "totalRaySamples": total_ray_samples,
    }


__all__ = [
    "DEFAULT_REPLAY_LIMITS",
    "KerrFiniteThicknessReplayError",
    "MAXIMUM_OFFICIAL_CIE_FREQUENCY_BINS",
    "ReplayResourceLimits",
    "reconstruct_kerr_finite_thickness_sampler",
    "validate_kerr_finite_thickness_replay",
]
