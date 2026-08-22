#!/usr/bin/env python3
"""Publish an authenticated finite-thickness exact-Kerr spectral frame.

The renderer uses the exact stationary Kerr metric, the prescribed stationary
Zhou--Taylor--Reynolds upper/lower photospheres, an equatorial
Novikov--Thorne/Page--Thorne thermal proxy evaluated at matching
pseudo-cylindrical radius, the flux-conserving KERRBB D20 angular law, and
independent fine/coarse whole-ray convergence.  Its observer-frequency axis is
the exact authenticated 471-bin CIE 1931 2-degree visible grid.

This is a first-visible scalar ``I_nu`` product.  It is not numerical
relativity, GRMHD, a solved atmosphere or vertical structure, polarization,
returning-radiation transport, or complete GRRT.  The finite photosphere is a
stationary prescribed calibration; the height-calibration accretion rate and
the SI thermal-disk accretion rate remain independent caller inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.adaptive_frame import AdaptivePixelOptions  # noqa: E402
from offline.cie_color import (  # noqa: E402
    CIE_DATASET_DOI,
    CIE_ROW_COUNT,
    DEFAULT_CIE_CSV,
    DEFAULT_CIE_METADATA,
    cie_1931_frequency_grid_hz,
    load_authenticated_cie_1931_2deg,
)
from offline.geodesic import RayTraceOptions, SurfaceEventOptions  # noqa: E402
from offline.job import InputArtifact, JobRun, JobSpec, TaskKey, run_job  # noqa: E402
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination  # noqa: E402
from offline.kerr_disk import StationaryNovikovThorneDisk  # noqa: E402
from offline.kerr_disk_frame import DarkEscapedObserverSpectrum  # noqa: E402
from offline.kerr_finite_thickness import (  # noqa: E402
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_frame import (  # noqa: E402
    KerrFiniteThicknessRaySampler,
)
from offline.kerr_finite_thickness_surface import (  # noqa: E402
    KerrFiniteThicknessMultiSurface,
)
from offline.novikov_thorne import PROGRADE, RETROGRADE  # noqa: E402
from offline.spectral_frame import SpectralPixelLayout  # noqa: E402
from offline.spectral_product import (  # noqa: E402
    AdaptiveSpectralTileProducer,
    SpectralFrameGrid,
    SpectralProductPublication,
    build_spectral_job_spec,
    default_numeric_backend_descriptor,
    publish_spectral_product,
)
from scripts.verify_offline_spectral_frame import (  # noqa: E402
    DEFAULT_SCHEMA as DEFAULT_SPECTRAL_SCHEMA,
    validate_scientific_spectral_frame,
)


SOLAR_MASS_KG = 1.98847e30
CIE_CSV_INPUT_URI = f"doi:{CIE_DATASET_DOI}#CIE_xyz_1931_2deg.csv"
CIE_METADATA_INPUT_URI = (
    f"doi:{CIE_DATASET_DOI}#CIE_xyz_1931_2deg.csv_metadata.json"
)

# Complete repository-owned Python closure that can affect tile payloads or
# their scientific identity.  CIE resources are authenticated separately as
# scientific inputs rather than being mislabeled as producer source.
PRODUCER_SOURCE_FILES = (
    Path("offline/__init__.py"),
    Path("offline/adaptive_frame.py"),
    Path("offline/cie_color.py"),
    Path("offline/disk_atmosphere.py"),
    Path("offline/geodesic.py"),
    Path("offline/job.py"),
    Path("offline/kerr.py"),
    Path("offline/kerr_disk.py"),
    Path("offline/kerr_disk_early_stop.py"),
    Path("offline/kerr_disk_frame.py"),
    Path("offline/kerr_disk_transfer.py"),
    Path("offline/kerr_finite_thickness.py"),
    Path("offline/kerr_finite_thickness_emitter.py"),
    Path("offline/kerr_finite_thickness_frame.py"),
    Path("offline/kerr_finite_thickness_replay_certificate.py"),
    Path("offline/kerr_finite_thickness_surface.py"),
    Path("offline/kerr_finite_thickness_transfer.py"),
    Path("offline/novikov_thorne.py"),
    Path("offline/radiative_transfer.py"),
    Path("offline/spacetime.py"),
    Path("offline/spectral_frame.py"),
    Path("offline/spectral_product.py"),
    Path("scripts/render_offline_kerr_finite_thickness_frame.py"),
)


@dataclass(frozen=True, slots=True)
class BoundInputStableSpectralTileProducer:
    """Reject a source or authenticated CIE edit around every tile."""

    inner: AdaptiveSpectralTileProducer
    source_artifacts: tuple[InputArtifact, ...]
    science_artifacts: tuple[InputArtifact, ...]
    source_root: Path
    cie_csv_path: Path
    cie_metadata_path: Path

    def __call__(self, spec: JobSpec, key: TaskKey) -> bytes:
        assert_bound_inputs_stable(
            self.source_artifacts,
            self.science_artifacts,
            source_root=self.source_root,
            cie_csv_path=self.cie_csv_path,
            cie_metadata_path=self.cie_metadata_path,
        )
        payload = self.inner(spec, key)
        assert_bound_inputs_stable(
            self.source_artifacts,
            self.science_artifacts,
            source_root=self.source_root,
            cie_csv_path=self.cie_csv_path,
            cie_metadata_path=self.cie_metadata_path,
        )
        return payload


@dataclass(frozen=True, slots=True)
class KerrFiniteThicknessFramePlan:
    """Fully bound scientific work plus non-scientific scheduling policy."""

    output_directory: Path
    cache_root: Path
    jobs: int
    max_in_flight: int | None
    layout: SpectralPixelLayout
    grid: SpectralFrameGrid
    adaptive_options: AdaptivePixelOptions
    sampler: KerrFiniteThicknessRaySampler
    numeric_backend: Mapping[str, object]
    source_artifacts: tuple[InputArtifact, ...]
    science_artifacts: tuple[InputArtifact, ...]
    job_spec: JobSpec
    producer: BoundInputStableSpectralTileProducer
    verification_schema: Path


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _regular_input(path: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise OSError(f"{label} must be a regular non-symlink file: {candidate}")
    return candidate


def collect_source_artifacts(
    source_root: Path = ROOT,
) -> tuple[InputArtifact, ...]:
    """Hash every repository file in the declared producer closure."""

    root = Path(source_root)
    artifacts: list[InputArtifact] = []
    for relative in PRODUCER_SOURCE_FILES:
        path = _regular_input(root / relative, "producer source")
        artifacts.append(
            InputArtifact.from_path(
                f"repo-source://{relative.as_posix()}",
                path,
            )
        )
    return tuple(artifacts)


def collect_science_artifacts(
    cie_csv_path: Path,
    cie_metadata_path: Path,
) -> tuple[InputArtifact, ...]:
    """Bind the exact authenticated CIE bytes as scientific inputs."""

    csv_path = _regular_input(cie_csv_path, "CIE CSV")
    metadata_path = _regular_input(cie_metadata_path, "CIE metadata")
    return tuple(
        sorted(
            (
                InputArtifact.from_path(CIE_CSV_INPUT_URI, csv_path),
                InputArtifact.from_path(CIE_METADATA_INPUT_URI, metadata_path),
            )
        )
    )


def assert_bound_inputs_stable(
    expected_sources: Sequence[InputArtifact],
    expected_science: Sequence[InputArtifact],
    *,
    source_root: Path = ROOT,
    cie_csv_path: Path = DEFAULT_CIE_CSV,
    cie_metadata_path: Path = DEFAULT_CIE_METADATA,
) -> None:
    """Fail if producer code or CIE bytes changed after JobSpec creation."""

    current_sources = collect_source_artifacts(source_root)
    current_science = collect_science_artifacts(cie_csv_path, cie_metadata_path)
    if tuple(expected_sources) != current_sources:
        raise RuntimeError(
            "producer source files changed after the JobSpec was constructed"
        )
    if tuple(expected_science) != current_science:
        raise RuntimeError(
            "authenticated CIE inputs changed after the JobSpec was constructed"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output",
        type=Path,
        help="new product directory; existing paths are never overwritten",
    )
    parser.add_argument(
        "--cache",
        required=True,
        type=Path,
        help="resumable content-addressed task-cache root",
    )

    cie = parser.add_argument_group("authenticated CIE 471-bin frequency grid")
    cie.add_argument("--cie-csv", type=Path, default=DEFAULT_CIE_CSV)
    cie.add_argument("--cie-metadata", type=Path, default=DEFAULT_CIE_METADATA)

    physics = parser.add_argument_group("stationary Kerr and disk physics")
    physics.add_argument(
        "--metric-mass-m",
        type=float,
        default=1.0,
        help="geometric Kerr mass scale M used by coordinates and termination",
    )
    physics.add_argument(
        "--spin",
        "--dimensionless-spin",
        dest="spin",
        type=float,
        default=0.7,
        help="signed Kerr spin a/M; finite-height calibration uses |a/M|",
    )
    mass = physics.add_mutually_exclusive_group()
    mass.add_argument(
        "--black-hole-mass-solar",
        type=float,
        default=1.0e8,
        help="SI thermal-spectrum black-hole mass in solar masses",
    )
    mass.add_argument(
        "--black-hole-mass-kg",
        type=float,
        default=None,
        help="SI thermal-spectrum black-hole mass in kilograms",
    )
    physics.add_argument(
        "--thermal-accretion-rate-kg-s",
        type=float,
        default=1.0e22,
        help="SI Novikov-Thorne rest-mass accretion rate; not inferred from dotm",
    )
    physics.add_argument(
        "--height-accretion-rate-eddington",
        "--dotm",
        dest="height_accretion_rate_eddington",
        type=float,
        default=0.05,
        help="dimensionless finite-height calibration rate, independently supplied",
    )
    physics.add_argument(
        "--outer-radius-over-mass",
        "--rout-over-mass",
        dest="outer_radius_over_mass",
        type=float,
        default=25.0,
        help="physical photosphere outer pseudo-cylindrical radius rho_out/M",
    )
    physics.add_argument(
        "--thinness-gate-maximum-h-over-rho",
        type=float,
        default=0.25,
        help="caller claim gate for pressure-scale H/rho; cannot loosen module policy",
    )
    physics.add_argument(
        "--orientation",
        choices=(PROGRADE, RETROGRADE),
        default=PROGRADE,
    )
    physics.add_argument("--colour-correction", type=float, default=1.7)
    physics.add_argument(
        "--singularity-guard-over-mass",
        type=float,
        default=1.0e-9,
    )

    observer = parser.add_argument_group("ZAMO observer and exact worldtubes")
    observer.add_argument("--observer-radius-over-mass", type=float, default=30.0)
    observer.add_argument("--inclination-deg", type=float, default=60.0)
    observer.add_argument("--observer-phi-deg", type=float, default=0.0)
    observer.add_argument(
        "--observer-coordinate-time-over-mass",
        type=float,
        default=0.0,
    )
    observer.add_argument("--escape-radius-over-mass", type=float, default=50.0)
    observer.add_argument("--horizon-offset-over-mass", type=float, default=0.02)

    frame = parser.add_argument_group("small bounded spectral frame")
    frame.add_argument("--width-pixels", "--width", dest="width_pixels", type=_positive_integer, default=1)
    frame.add_argument("--height-pixels", "--height", dest="height_pixels", type=_positive_integer, default=1)
    frame.add_argument(
        "--screen-x-min",
        type=float,
        default=0.49999,
        help="left edge of the bounded calibration cell",
    )
    frame.add_argument(
        "--screen-x-max",
        type=float,
        default=0.50001,
        help="right edge of the bounded calibration cell",
    )
    frame.add_argument(
        "--screen-y-min",
        type=float,
        default=-0.50001,
        help="lower edge of the bounded calibration cell",
    )
    frame.add_argument(
        "--screen-y-max",
        type=float,
        default=-0.49999,
        help="upper edge of the bounded calibration cell",
    )
    frame.add_argument("--tile-width", type=_positive_integer, default=1)
    frame.add_argument("--tile-height", type=_positive_integer, default=1)
    frame.add_argument("--jobs", type=_positive_integer, default=1)
    frame.add_argument("--max-in-flight", type=_positive_integer, default=1)

    ray = parser.add_argument_group("fine ray and N/2N surface localization")
    ray.add_argument("--ray-absolute-tolerance", type=float, default=5.0e-10)
    ray.add_argument("--ray-relative-tolerance", type=float, default=5.0e-10)
    ray.add_argument("--ray-initial-step-over-mass", type=float, default=0.05)
    ray.add_argument("--ray-minimum-step-over-mass", type=float, default=1.0e-8)
    ray.add_argument("--ray-maximum-step-over-mass", type=float, default=0.25)
    ray.add_argument("--ray-maximum-affine-length-over-mass", type=float, default=300.0)
    ray.add_argument("--ray-maximum-accepted-steps", type=_positive_integer, default=100_000)
    ray.add_argument("--ray-maximum-rejected-steps", type=_positive_integer, default=100_000)
    ray.add_argument("--ray-null-residual-limit", type=float, default=2.0e-7)
    ray.add_argument("--ray-metric-interpolation-error-limit", type=float, default=1.0e-7)
    ray.add_argument("--ray-event-value-tolerance-over-mass", type=float, default=1.0e-9)
    ray.add_argument("--ray-event-affine-tolerance-over-mass", type=float, default=1.0e-10)
    ray.add_argument("--ray-event-maximum-iterations", type=_positive_integer, default=64)
    ray.add_argument("--surface-absolute-tolerance", type=float, default=5.0e-10)
    ray.add_argument("--surface-relative-tolerance", type=float, default=5.0e-10)
    ray.add_argument("--surface-null-residual-limit", type=float, default=2.0e-7)
    ray.add_argument("--surface-metric-interpolation-error-limit", type=float, default=1.0e-7)
    ray.add_argument("--surface-value-tolerance", type=float, default=1.0e-9)
    ray.add_argument("--surface-affine-tolerance-over-mass", type=float, default=1.0e-10)
    ray.add_argument("--surface-maximum-iterations", type=_positive_integer, default=64)
    ray.add_argument("--surface-maximum-reintegrations", type=_positive_integer, default=100_000)
    ray.add_argument(
        "--surface-subdivisions-per-segment",
        type=_positive_integer,
        default=4,
        help="even N; agreement with 2N is finite evidence, not caustic completeness",
    )

    convergence = parser.add_argument_group("fine/coarse and transfer gates")
    convergence.add_argument("--coarse-tolerance-multiplier", type=float, default=32.0)
    convergence.add_argument("--terminal-event-tolerance-over-mass", type=float, default=2.0e-5)
    convergence.add_argument("--terminal-covector-tolerance", type=float, default=2.0e-5)
    convergence.add_argument("--disk-radius-absolute-tolerance-over-mass", type=float, default=0.0)
    convergence.add_argument("--disk-radius-relative-tolerance", type=float, default=2.0e-5)
    convergence.add_argument("--frequency-shift-relative-tolerance", type=float, default=2.0e-5)
    convergence.add_argument("--emission-cosine-absolute-tolerance", type=float, default=2.0e-5)
    convergence.add_argument("--specific-intensity-absolute-tolerance", type=float, default=0.0)
    convergence.add_argument("--specific-intensity-relative-tolerance", type=float, default=2.0e-4)
    convergence.add_argument("--escape-direction-tolerance-rad", type=float, default=2.0e-5)
    convergence.add_argument("--frequency-null-residual-limit", type=float, default=2.0e-7)
    convergence.add_argument("--conserved-quantity-tolerance", type=float, default=2.0e-7)
    convergence.add_argument("--recorded-path-absolute-tolerance", type=float, default=2.0e-10)
    convergence.add_argument("--recorded-path-relative-tolerance", type=float, default=2.0e-10)
    convergence.add_argument("--boundary-value-tolerance-over-mass", type=float, default=None)
    convergence.add_argument("--emitter-event-tolerance-over-mass", type=float, default=None)

    adaptive = parser.add_argument_group("bounded adaptive pixel integration")
    adaptive.add_argument("--minimum-depth", type=_non_negative_integer, default=0)
    adaptive.add_argument("--maximum-depth", type=_non_negative_integer, default=0)
    adaptive.add_argument("--maximum-ray-evaluations", type=_positive_integer, default=32)
    adaptive.add_argument("--radiance-absolute-tolerance", type=float, default=0.0)
    adaptive.add_argument("--radiance-relative-tolerance", type=float, default=1.0e-3)
    adaptive.add_argument("--radiance-guard-ceiling", type=float, default=1.0)
    adaptive.add_argument("--unresolved-solid-angle-fraction-tolerance", type=float, default=0.0)
    adaptive.add_argument("--weighted-log-g-tolerance", type=float, default=1.0e-3)
    adaptive.add_argument("--weighted-direction-tolerance-rad", type=float, default=1.0e-4)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_render_plan(
    arguments: argparse.Namespace,
    *,
    source_root: Path = ROOT,
) -> KerrFiniteThicknessFramePlan:
    """Construct the complete job identity without tracing or writing output."""

    cie_csv = Path(arguments.cie_csv).absolute()
    cie_metadata = Path(arguments.cie_metadata).absolute()
    cie_table = load_authenticated_cie_1931_2deg(cie_csv, cie_metadata)
    frequencies = cie_1931_frequency_grid_hz(cie_table)
    if len(frequencies) != CIE_ROW_COUNT:
        raise AssertionError("authenticated CIE grid does not contain 471 bins")
    layout = SpectralPixelLayout(frequencies)
    grid = SpectralFrameGrid(
        width_pixels=arguments.width_pixels,
        height_pixels=arguments.height_pixels,
        screen_x_min=arguments.screen_x_min,
        screen_x_max=arguments.screen_x_max,
        screen_y_min=arguments.screen_y_min,
        screen_y_max=arguments.screen_y_max,
    )
    adaptive_options = AdaptivePixelOptions(
        minimum_depth=arguments.minimum_depth,
        maximum_depth=arguments.maximum_depth,
        maximum_ray_evaluations=arguments.maximum_ray_evaluations,
        radiance_absolute_tolerances=(arguments.radiance_absolute_tolerance,)
        * layout.frequency_count,
        radiance_relative_tolerance=arguments.radiance_relative_tolerance,
        unresolved_solid_angle_fraction_tolerance=(
            arguments.unresolved_solid_angle_fraction_tolerance
        ),
        weighted_log_g_tolerance=arguments.weighted_log_g_tolerance,
        weighted_direction_tolerance_rad=arguments.weighted_direction_tolerance_rad,
        radiance_guard_ceilings=(arguments.radiance_guard_ceiling,)
        * layout.frequency_count,
    )

    mass_m = float(arguments.metric_mass_m)
    metric = KerrKerrSchildMetric(
        mass_m=mass_m,
        spin_a_m=arguments.spin * mass_m,
        singularity_guard_m=arguments.singularity_guard_over_mass * mass_m,
    )
    calibration = StationaryKerrFiniteThicknessCalibration(
        dimensionless_spin=abs(arguments.spin),
        eddington_scaled_mass_accretion_rate=(
            arguments.height_accretion_rate_eddington
        ),
        orientation=arguments.orientation,
        outer_radius_over_mass=arguments.outer_radius_over_mass,
        thinness_gate_maximum_h_over_rho=(
            arguments.thinness_gate_maximum_h_over_rho
        ),
    )
    surface = KerrFiniteThicknessMultiSurface(metric, calibration)
    termination = KerrOblateTermination.horizon_worldtube(
        metric,
        escape_radius_m=arguments.escape_radius_over_mass * mass_m,
        offset_m=arguments.horizon_offset_over_mass * mass_m,
    )
    black_hole_mass_kg = (
        arguments.black_hole_mass_kg
        if arguments.black_hole_mass_kg is not None
        else arguments.black_hole_mass_solar * SOLAR_MASS_KG
    )
    disk = StationaryNovikovThorneDisk(
        metric=metric,
        black_hole_mass_kg=black_hole_mass_kg,
        mass_accretion_rate_kg_s=arguments.thermal_accretion_rate_kg_s,
        orientation=arguments.orientation,
        colour_correction=arguments.colour_correction,
    )
    ray_options = RayTraceOptions(
        absolute_tolerance=arguments.ray_absolute_tolerance,
        relative_tolerance=arguments.ray_relative_tolerance,
        initial_step=arguments.ray_initial_step_over_mass * mass_m,
        minimum_step=arguments.ray_minimum_step_over_mass * mass_m,
        maximum_step=arguments.ray_maximum_step_over_mass * mass_m,
        maximum_affine_length=(
            arguments.ray_maximum_affine_length_over_mass * mass_m
        ),
        maximum_accepted_steps=arguments.ray_maximum_accepted_steps,
        maximum_rejected_steps=arguments.ray_maximum_rejected_steps,
        null_residual_limit=arguments.ray_null_residual_limit,
        metric_interpolation_error_limit=(
            arguments.ray_metric_interpolation_error_limit
        ),
        event_value_tolerance=(
            arguments.ray_event_value_tolerance_over_mass * mass_m
        ),
        event_affine_tolerance=(
            arguments.ray_event_affine_tolerance_over_mass * mass_m
        ),
        event_maximum_iterations=arguments.ray_event_maximum_iterations,
        record_path=True,
    )
    surface_options = SurfaceEventOptions(
        absolute_tolerance=arguments.surface_absolute_tolerance,
        relative_tolerance=arguments.surface_relative_tolerance,
        null_residual_limit=arguments.surface_null_residual_limit,
        metric_interpolation_error_limit=(
            arguments.surface_metric_interpolation_error_limit
        ),
        surface_value_tolerance=arguments.surface_value_tolerance,
        affine_tolerance=(
            arguments.surface_affine_tolerance_over_mass * mass_m
        ),
        maximum_iterations=arguments.surface_maximum_iterations,
        maximum_reintegrations=arguments.surface_maximum_reintegrations,
        subdivisions_per_segment=arguments.surface_subdivisions_per_segment,
    )
    boundary_tolerance = (
        None
        if arguments.boundary_value_tolerance_over_mass is None
        else arguments.boundary_value_tolerance_over_mass * mass_m
    )
    emitter_tolerance = (
        None
        if arguments.emitter_event_tolerance_over_mass is None
        else arguments.emitter_event_tolerance_over_mass * mass_m
    )
    sampler = KerrFiniteThicknessRaySampler(
        metric=metric,
        observer_radius_m=arguments.observer_radius_over_mass * mass_m,
        termination=termination,
        surface=surface,
        disk=disk,
        escaped_observer_spectrum=DarkEscapedObserverSpectrum(),
        fine_options=ray_options,
        surface_options=surface_options,
        observer_theta_rad=math.radians(arguments.inclination_deg),
        observer_phi_ks_rad=math.radians(arguments.observer_phi_deg),
        observer_coordinate_time_m=(
            arguments.observer_coordinate_time_over_mass * mass_m
        ),
        coarse_tolerance_multiplier=arguments.coarse_tolerance_multiplier,
        terminal_event_tolerance_m=(
            arguments.terminal_event_tolerance_over_mass * mass_m
        ),
        terminal_covector_tolerance=arguments.terminal_covector_tolerance,
        disk_radius_absolute_tolerance_m=(
            arguments.disk_radius_absolute_tolerance_over_mass * mass_m
        ),
        disk_radius_relative_tolerance=arguments.disk_radius_relative_tolerance,
        frequency_shift_relative_tolerance=(
            arguments.frequency_shift_relative_tolerance
        ),
        emission_cosine_absolute_tolerance=(
            arguments.emission_cosine_absolute_tolerance
        ),
        specific_intensity_absolute_tolerance=(
            arguments.specific_intensity_absolute_tolerance
        ),
        specific_intensity_relative_tolerance=(
            arguments.specific_intensity_relative_tolerance
        ),
        escape_direction_tolerance_rad=arguments.escape_direction_tolerance_rad,
        frequency_null_residual_limit=arguments.frequency_null_residual_limit,
        conserved_quantity_tolerance=arguments.conserved_quantity_tolerance,
        recorded_path_absolute_tolerance=(
            arguments.recorded_path_absolute_tolerance
        ),
        recorded_path_relative_tolerance=(
            arguments.recorded_path_relative_tolerance
        ),
        boundary_value_tolerance_m=boundary_tolerance,
        emitter_event_tolerance_m=emitter_tolerance,
    )

    numeric_backend = default_numeric_backend_descriptor()
    source_artifacts = collect_source_artifacts(source_root)
    science_artifacts = collect_science_artifacts(cie_csv, cie_metadata)
    source_hashes = tuple(
        sorted({artifact.sha256 for artifact in source_artifacts})
    )
    all_inputs = tuple(sorted((*source_artifacts, *science_artifacts)))
    job_spec = build_spectral_job_spec(
        layout,
        grid,
        adaptive_options,
        sampler.descriptor(),
        tile_width=arguments.tile_width,
        tile_height=arguments.tile_height,
        numeric_backend=numeric_backend,
        inputs=all_inputs,
        producer_source_hashes=source_hashes,
    )
    producer = BoundInputStableSpectralTileProducer(
        inner=AdaptiveSpectralTileProducer(
            sampler,
            layout,
            grid,
            adaptive_options,
            numeric_backend,
        ),
        source_artifacts=source_artifacts,
        science_artifacts=science_artifacts,
        source_root=Path(source_root).absolute(),
        cie_csv_path=cie_csv,
        cie_metadata_path=cie_metadata,
    )
    return KerrFiniteThicknessFramePlan(
        output_directory=Path(arguments.output).absolute(),
        cache_root=Path(arguments.cache).absolute(),
        jobs=arguments.jobs,
        max_in_flight=arguments.max_in_flight,
        layout=layout,
        grid=grid,
        adaptive_options=adaptive_options,
        sampler=sampler,
        numeric_backend=numeric_backend,
        source_artifacts=source_artifacts,
        science_artifacts=science_artifacts,
        job_spec=job_spec,
        producer=producer,
        verification_schema=DEFAULT_SPECTRAL_SCHEMA.absolute(),
    )


def execute_render_plan(
    plan: KerrFiniteThicknessFramePlan,
) -> tuple[JobRun, SpectralProductPublication, Mapping[str, Any]]:
    """Resume tiles, publish atomically, then run the strict structure verifier."""

    output = plan.output_directory
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output {output}")
    assert_bound_inputs_stable(
        plan.source_artifacts,
        plan.science_artifacts,
        source_root=plan.producer.source_root,
        cie_csv_path=plan.producer.cie_csv_path,
        cie_metadata_path=plan.producer.cie_metadata_path,
    )
    job_run = run_job(
        plan.job_spec,
        plan.producer,
        plan.cache_root,
        jobs=plan.jobs,
        max_in_flight=plan.max_in_flight,
    )
    assert_bound_inputs_stable(
        plan.source_artifacts,
        plan.science_artifacts,
        source_root=plan.producer.source_root,
        cie_csv_path=plan.producer.cie_csv_path,
        cie_metadata_path=plan.producer.cie_metadata_path,
    )
    publication = publish_spectral_product(
        output,
        job_spec=plan.job_spec,
        job_run=job_run,
        layout=plan.layout,
        grid=plan.grid,
        options=plan.adaptive_options,
        sampler_descriptor=plan.sampler.descriptor(),
        numeric_backend=plan.numeric_backend,
    )
    verification = validate_scientific_spectral_frame(
        publication.manifest_path,
        plan.verification_schema,
    )
    if verification.get("status") != (
        "scientific-spectral-frame-structural-contract-conformant"
    ):
        raise RuntimeError("strict spectral-frame verifier returned a bad status")
    return job_run, publication, verification


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        plan = build_render_plan(arguments)
        job_run, publication, verification = execute_render_plan(plan)
    except Exception as error:
        print(
            f"Offline finite-thickness exact-Kerr frame failed: {error}",
            file=sys.stderr,
        )
        return 1

    print("Offline finite-thickness exact-Kerr CIE-grid spectral frame completed")
    print(f"  manifest = {publication.manifest_path}")
    print(f"  manifest sha256 = {publication.manifest_sha256}")
    print(f"  product id = {publication.product_id}")
    print(f"  job key = {job_run.job_key}")
    print(f"  strict verifier = {verification['status']}")
    print(
        f"  frequencies = {plan.layout.frequency_count} authenticated CIE bins; "
        f"records = {publication.record_count}; tiles = {publication.tile_count}; "
        f"reused tasks = {job_run.reused_tasks}; "
        f"executed tasks = {job_run.executed_tasks}"
    )
    print(
        "  scope = exact stationary Kerr + prescribed stationary finite-height "
        "photosphere + equatorial NT-at-rho scalar proxy; no NR, GRMHD, solved "
        "atmosphere, polarization, returning radiation, or complete GRRT"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
