#!/usr/bin/env python3
"""Render an authenticated exact-Kerr Novikov--Thorne spectral frame.

The command produces linear observer-frame ``I_nu`` tiles.  It uses the exact
stationary Kerr metric, a zero-torque Novikov--Thorne thin disk, the
flux-conserving KERRBB D20 angular law, independent fine/coarse ray traces, and
the finite-stencil adaptive pixel integrator.  It is not NR, GRMHD, polarized
transfer, returning radiation, or a solved disk atmosphere.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.adaptive_frame import AdaptivePixelOptions  # noqa: E402
from offline.disk_atmosphere import (  # noqa: E402
    FluxConservingLinearLimbDarkening,
)
from offline.geodesic import RayTraceOptions, SurfaceEventOptions  # noqa: E402
from offline.job import (  # noqa: E402
    InputArtifact,
    JobRun,
    JobSpec,
    TaskKey,
    run_job,
)
from offline.kerr import (  # noqa: E402
    KerrKerrSchildMetric,
    KerrOblateTermination,
)
from offline.kerr_disk import StationaryNovikovThorneDisk  # noqa: E402
from offline.kerr_disk_frame import (  # noqa: E402
    DarkEscapedObserverSpectrum,
    KerrDiskRaySampler,
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


SOLAR_MASS_KG = 1.98847e30

# This is the complete repository-owned Python closure that can affect a tile
# payload or its scientific identity.  Unrelated vacuum/NR producers are not
# included, so their edits do not invalidate an exact-Kerr NT cache.
PRODUCER_SOURCE_FILES = (
    Path("offline/__init__.py"),
    Path("offline/adaptive_frame.py"),
    Path("offline/disk_atmosphere.py"),
    Path("offline/geodesic.py"),
    Path("offline/job.py"),
    Path("offline/kerr.py"),
    Path("offline/kerr_disk.py"),
    Path("offline/kerr_disk_early_stop.py"),
    Path("offline/kerr_disk_frame.py"),
    Path("offline/kerr_disk_transfer.py"),
    Path("offline/novikov_thorne.py"),
    Path("offline/radiative_transfer.py"),
    Path("offline/spacetime.py"),
    Path("offline/spectral_frame.py"),
    Path("offline/spectral_product.py"),
    Path("scripts/render_offline_kerr_nt_frame.py"),
)


@dataclass(frozen=True, slots=True)
class SourceStableSpectralTileProducer:
    """Keep a concurrent source edit from contaminating a resumable cache."""

    inner: AdaptiveSpectralTileProducer
    source_artifacts: tuple[InputArtifact, ...]
    source_root: Path

    def __call__(self, spec: JobSpec, key: TaskKey) -> bytes:
        assert_source_snapshot_stable(self.source_artifacts, self.source_root)
        payload = self.inner(spec, key)
        assert_source_snapshot_stable(self.source_artifacts, self.source_root)
        return payload


@dataclass(frozen=True, slots=True)
class KerrNtFramePlan:
    """Fully bound scientific job plus non-scientific scheduling policy."""

    output_directory: Path
    cache_root: Path
    jobs: int
    max_in_flight: int | None
    layout: SpectralPixelLayout
    grid: SpectralFrameGrid
    adaptive_options: AdaptivePixelOptions
    sampler: KerrDiskRaySampler
    numeric_backend: Mapping[str, object]
    source_artifacts: tuple[InputArtifact, ...]
    job_spec: JobSpec
    producer: SourceStableSpectralTileProducer


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def collect_source_artifacts(
    source_root: Path = ROOT,
) -> tuple[InputArtifact, ...]:
    """Hash every repository source in the declared producer closure."""

    root = Path(source_root)
    artifacts: list[InputArtifact] = []
    for relative in PRODUCER_SOURCE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise OSError(f"producer source must be a regular file: {path}")
        artifacts.append(
            InputArtifact.from_path(
                f"repo-source://{relative.as_posix()}",
                path,
            )
        )
    return tuple(artifacts)


def assert_source_snapshot_stable(
    expected: Sequence[InputArtifact],
    source_root: Path = ROOT,
) -> None:
    """Fail before publication if a producer source changed during the run."""

    current = collect_source_artifacts(source_root)
    if tuple(expected) != current:
        raise RuntimeError(
            "producer source files changed after the JobSpec was constructed"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output",
        type=Path,
        help="new product directory; an existing path is never overwritten",
    )
    parser.add_argument(
        "--cache",
        required=True,
        type=Path,
        help="resumable content-addressed task-cache root",
    )

    physics = parser.add_argument_group("Kerr and disk physics")
    physics.add_argument(
        "--spin",
        "--dimensionless-spin",
        dest="spin",
        type=float,
        default=0.7,
        help="signed dimensionless Kerr spin a/M; Novikov-Thorne needs |a/M| < 1",
    )
    mass = physics.add_mutually_exclusive_group()
    mass.add_argument(
        "--black-hole-mass-solar",
        type=float,
        default=1.0e8,
        help="black-hole mass used for the SI disk spectrum, in solar masses",
    )
    mass.add_argument(
        "--black-hole-mass-kg",
        type=float,
        default=None,
        help="black-hole mass used for the SI disk spectrum, in kilograms",
    )
    physics.add_argument(
        "--accretion-rate-kg-s",
        type=float,
        default=1.0e22,
        help="total rest-mass accretion rate in kg/s",
    )
    physics.add_argument(
        "--orientation",
        choices=(PROGRADE, RETROGRADE),
        default=PROGRADE,
        help="disk orbital orientation relative to the black-hole spin axis",
    )
    physics.add_argument(
        "--colour-correction",
        type=float,
        default=1.7,
        help="constant diluted-blackbody colour correction",
    )
    physics.add_argument(
        "--inclination-deg",
        type=float,
        default=60.0,
        help=(
            "observer polar inclination from +spin axis; exact edge-on is "
            "rejected for the zero-thickness disk"
        ),
    )
    physics.add_argument(
        "--observer-radius-m",
        type=float,
        default=30.0,
        help="ZAMO radius in geometric coordinates with metric M=1",
    )
    physics.add_argument(
        "--disk-outer-radius-m",
        type=float,
        default=25.0,
        help="opaque disk outer radius in geometric coordinates with M=1",
    )
    physics.add_argument(
        "--escape-radius-m",
        type=float,
        default=50.0,
        help="dark escape worldtube radius in geometric coordinates with M=1",
    )
    physics.add_argument(
        "--horizon-offset-m",
        type=float,
        default=0.02,
        help="capture-worldtube offset outside r+ in geometric M=1 units",
    )
    physics.add_argument(
        "--singularity-guard-m",
        type=float,
        default=1.0e-9,
        help="exact-Kerr ring singularity guard in geometric M=1 units",
    )

    frame = parser.add_argument_group("spectral frame and tiling")
    frame.add_argument(
        "--frequency-hz",
        action="append",
        required=True,
        type=float,
        help="observer-frame frequency in Hz; repeat in strictly increasing order",
    )
    frame.add_argument(
        "--width-pixels",
        "--width",
        dest="width_pixels",
        type=_positive_integer,
        default=64,
    )
    frame.add_argument(
        "--height-pixels",
        "--height",
        dest="height_pixels",
        type=_positive_integer,
        default=64,
    )
    frame.add_argument("--screen-x-min", type=float, default=-1.0)
    frame.add_argument("--screen-x-max", type=float, default=1.0)
    frame.add_argument("--screen-y-min", type=float, default=-1.0)
    frame.add_argument("--screen-y-max", type=float, default=1.0)
    frame.add_argument(
        "--tile-width",
        type=_positive_integer,
        default=8,
    )
    frame.add_argument(
        "--tile-height",
        type=_positive_integer,
        default=8,
    )
    frame.add_argument("--jobs", type=_positive_integer, default=1)
    frame.add_argument(
        "--max-in-flight",
        type=_positive_integer,
        default=None,
        help="bounded submitted tasks; omitted means 2 * jobs",
    )

    ray = parser.add_argument_group("fine ray and surface tolerances")
    ray.add_argument("--ray-absolute-tolerance", type=float, default=5.0e-10)
    ray.add_argument("--ray-relative-tolerance", type=float, default=5.0e-10)
    ray.add_argument("--ray-initial-step", type=float, default=0.05)
    ray.add_argument("--ray-minimum-step", type=float, default=1.0e-8)
    ray.add_argument("--ray-maximum-step", type=float, default=0.25)
    ray.add_argument(
        "--ray-maximum-affine-length",
        type=float,
        default=300.0,
    )
    ray.add_argument(
        "--ray-maximum-accepted-steps",
        type=_positive_integer,
        default=100_000,
    )
    ray.add_argument(
        "--ray-maximum-rejected-steps",
        type=_positive_integer,
        default=100_000,
    )
    ray.add_argument("--ray-null-residual-limit", type=float, default=2.0e-7)
    ray.add_argument(
        "--ray-metric-interpolation-error-limit",
        type=float,
        default=1.0e-7,
    )
    ray.add_argument("--ray-event-value-tolerance", type=float, default=1.0e-9)
    ray.add_argument("--ray-event-affine-tolerance", type=float, default=1.0e-10)
    ray.add_argument(
        "--ray-event-maximum-iterations",
        type=_positive_integer,
        default=64,
    )
    ray.add_argument("--surface-absolute-tolerance", type=float, default=5.0e-10)
    ray.add_argument("--surface-relative-tolerance", type=float, default=5.0e-10)
    ray.add_argument(
        "--surface-null-residual-limit",
        type=float,
        default=2.0e-7,
    )
    ray.add_argument(
        "--surface-metric-interpolation-error-limit",
        type=float,
        default=1.0e-7,
    )
    ray.add_argument("--surface-value-tolerance", type=float, default=1.0e-9)
    ray.add_argument("--surface-affine-tolerance", type=float, default=1.0e-10)
    ray.add_argument(
        "--surface-maximum-iterations",
        type=_positive_integer,
        default=64,
    )
    ray.add_argument(
        "--surface-maximum-reintegrations",
        type=_positive_integer,
        default=100_000,
    )
    ray.add_argument(
        "--surface-subdivisions-per-segment",
        type=_positive_integer,
        default=2,
        help="must be even; finite probing is not a caustic-completeness proof",
    )

    convergence = parser.add_argument_group("fine/coarse transfer tolerances")
    convergence.add_argument(
        "--coarse-tolerance-multiplier",
        type=float,
        default=32.0,
    )
    convergence.add_argument(
        "--terminal-event-tolerance-m",
        type=float,
        default=2.0e-5,
    )
    convergence.add_argument(
        "--terminal-covector-tolerance",
        type=float,
        default=2.0e-5,
    )
    convergence.add_argument(
        "--disk-radius-absolute-tolerance-m",
        type=float,
        default=0.0,
    )
    convergence.add_argument(
        "--disk-radius-relative-tolerance",
        type=float,
        default=2.0e-5,
    )
    convergence.add_argument(
        "--frequency-shift-relative-tolerance",
        type=float,
        default=2.0e-5,
    )
    convergence.add_argument(
        "--emission-angle-absolute-tolerance",
        type=float,
        default=2.0e-5,
    )
    convergence.add_argument(
        "--specific-intensity-absolute-tolerance",
        type=float,
        default=0.0,
    )
    convergence.add_argument(
        "--specific-intensity-relative-tolerance",
        type=float,
        default=2.0e-4,
    )
    convergence.add_argument(
        "--escape-direction-tolerance-rad",
        type=float,
        default=2.0e-5,
    )
    convergence.add_argument(
        "--frequency-null-residual-limit",
        type=float,
        default=1.0e-7,
    )
    convergence.add_argument(
        "--conserved-quantity-tolerance",
        type=float,
        default=1.0e-7,
    )
    convergence.add_argument(
        "--emitter-event-tolerance-m",
        type=float,
        default=None,
        help="omitted uses 1e-8 * metric M",
    )

    adaptive = parser.add_argument_group("adaptive pixel tolerances")
    adaptive.add_argument("--minimum-depth", type=int, default=0)
    adaptive.add_argument("--maximum-depth", type=int, default=3)
    adaptive.add_argument(
        "--maximum-ray-evaluations",
        type=_positive_integer,
        default=2_000,
    )
    adaptive.add_argument(
        "--radiance-absolute-tolerance",
        type=float,
        default=0.0,
        help="one scalar repeated for every frequency bin",
    )
    adaptive.add_argument(
        "--radiance-relative-tolerance",
        type=float,
        default=1.0e-3,
    )
    adaptive.add_argument(
        "--radiance-guard-ceiling",
        type=float,
        default=1.0,
        help="per-bin physical ceiling used for unresolved-topology error guards",
    )
    adaptive.add_argument(
        "--unresolved-solid-angle-fraction-tolerance",
        type=float,
        default=0.0,
    )
    adaptive.add_argument(
        "--weighted-log-g-tolerance",
        type=float,
        default=1.0e-3,
    )
    adaptive.add_argument(
        "--weighted-direction-tolerance-rad",
        type=float,
        default=1.0e-4,
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_render_plan(
    arguments: argparse.Namespace,
    *,
    source_root: Path = ROOT,
) -> KerrNtFramePlan:
    """Construct all bound objects without tracing a ray or writing a file."""

    frequencies = tuple(arguments.frequency_hz)
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
        radiance_absolute_tolerances=(
            arguments.radiance_absolute_tolerance,
        )
        * layout.frequency_count,
        radiance_relative_tolerance=arguments.radiance_relative_tolerance,
        unresolved_solid_angle_fraction_tolerance=(
            arguments.unresolved_solid_angle_fraction_tolerance
        ),
        weighted_log_g_tolerance=arguments.weighted_log_g_tolerance,
        weighted_direction_tolerance_rad=(
            arguments.weighted_direction_tolerance_rad
        ),
        radiance_guard_ceilings=(arguments.radiance_guard_ceiling,)
        * layout.frequency_count,
    )

    metric = KerrKerrSchildMetric(
        mass_m=1.0,
        spin_a_m=arguments.spin,
        singularity_guard_m=arguments.singularity_guard_m,
    )
    termination = KerrOblateTermination.horizon_worldtube(
        metric,
        escape_radius_m=arguments.escape_radius_m,
        offset_m=arguments.horizon_offset_m,
    )
    black_hole_mass_kg = (
        arguments.black_hole_mass_kg
        if arguments.black_hole_mass_kg is not None
        else arguments.black_hole_mass_solar * SOLAR_MASS_KG
    )
    disk = StationaryNovikovThorneDisk(
        metric=metric,
        black_hole_mass_kg=black_hole_mass_kg,
        mass_accretion_rate_kg_s=arguments.accretion_rate_kg_s,
        orientation=arguments.orientation,
        colour_correction=arguments.colour_correction,
    )
    ray_options = RayTraceOptions(
        absolute_tolerance=arguments.ray_absolute_tolerance,
        relative_tolerance=arguments.ray_relative_tolerance,
        initial_step=arguments.ray_initial_step,
        minimum_step=arguments.ray_minimum_step,
        maximum_step=arguments.ray_maximum_step,
        maximum_affine_length=arguments.ray_maximum_affine_length,
        maximum_accepted_steps=arguments.ray_maximum_accepted_steps,
        maximum_rejected_steps=arguments.ray_maximum_rejected_steps,
        null_residual_limit=arguments.ray_null_residual_limit,
        metric_interpolation_error_limit=(
            arguments.ray_metric_interpolation_error_limit
        ),
        event_value_tolerance=arguments.ray_event_value_tolerance,
        event_affine_tolerance=arguments.ray_event_affine_tolerance,
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
        affine_tolerance=arguments.surface_affine_tolerance,
        maximum_iterations=arguments.surface_maximum_iterations,
        maximum_reintegrations=arguments.surface_maximum_reintegrations,
        subdivisions_per_segment=arguments.surface_subdivisions_per_segment,
    )
    sampler = KerrDiskRaySampler(
        metric=metric,
        observer_radius_m=arguments.observer_radius_m,
        termination=termination,
        disk=disk,
        outer_radius_m=arguments.disk_outer_radius_m,
        escaped_observer_spectrum=DarkEscapedObserverSpectrum(),
        fine_options=ray_options,
        surface_options=surface_options,
        angular_emission_law=FluxConservingLinearLimbDarkening(),
        observer_theta_rad=math.radians(arguments.inclination_deg),
        coarse_tolerance_multiplier=arguments.coarse_tolerance_multiplier,
        terminal_event_tolerance_m=arguments.terminal_event_tolerance_m,
        terminal_covector_tolerance=arguments.terminal_covector_tolerance,
        disk_radius_absolute_tolerance_m=(
            arguments.disk_radius_absolute_tolerance_m
        ),
        disk_radius_relative_tolerance=(
            arguments.disk_radius_relative_tolerance
        ),
        frequency_shift_relative_tolerance=(
            arguments.frequency_shift_relative_tolerance
        ),
        emission_angle_absolute_tolerance=(
            arguments.emission_angle_absolute_tolerance
        ),
        specific_intensity_absolute_tolerance=(
            arguments.specific_intensity_absolute_tolerance
        ),
        specific_intensity_relative_tolerance=(
            arguments.specific_intensity_relative_tolerance
        ),
        escape_direction_tolerance_rad=(
            arguments.escape_direction_tolerance_rad
        ),
        frequency_null_residual_limit=(
            arguments.frequency_null_residual_limit
        ),
        conserved_quantity_tolerance=(
            arguments.conserved_quantity_tolerance
        ),
        emitter_event_tolerance_m=arguments.emitter_event_tolerance_m,
    )
    numeric_backend = default_numeric_backend_descriptor()
    source_artifacts = collect_source_artifacts(source_root)
    source_hashes = tuple(
        sorted({artifact.sha256 for artifact in source_artifacts})
    )
    job_spec = build_spectral_job_spec(
        layout,
        grid,
        adaptive_options,
        sampler.descriptor(),
        tile_width=arguments.tile_width,
        tile_height=arguments.tile_height,
        numeric_backend=numeric_backend,
        inputs=source_artifacts,
        producer_source_hashes=source_hashes,
    )
    producer = SourceStableSpectralTileProducer(
        inner=AdaptiveSpectralTileProducer(
            sampler,
            layout,
            grid,
            adaptive_options,
            numeric_backend,
        ),
        source_artifacts=source_artifacts,
        source_root=Path(source_root).absolute(),
    )
    return KerrNtFramePlan(
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
        job_spec=job_spec,
        producer=producer,
    )


def execute_render_plan(
    plan: KerrNtFramePlan,
    *,
    source_root: Path = ROOT,
) -> tuple[JobRun, SpectralProductPublication]:
    """Resume missing tiles, then atomically publish a new product."""

    output = plan.output_directory
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output {output}")
    assert_source_snapshot_stable(plan.source_artifacts, source_root)
    job_run = run_job(
        plan.job_spec,
        plan.producer,
        plan.cache_root,
        jobs=plan.jobs,
        max_in_flight=plan.max_in_flight,
    )
    assert_source_snapshot_stable(plan.source_artifacts, source_root)
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
    return job_run, publication


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        plan = build_render_plan(arguments)
        job_run, publication = execute_render_plan(plan)
    except Exception as error:
        print(f"Offline exact-Kerr NT frame failed: {error}", file=sys.stderr)
        return 1

    print("Offline exact-Kerr Novikov-Thorne spectral frame completed")
    print(f"  manifest = {publication.manifest_path}")
    print(f"  manifest sha256 = {publication.manifest_sha256}")
    print(f"  product id = {publication.product_id}")
    print(f"  job key = {job_run.job_key}")
    print(
        f"  records = {publication.record_count}; tiles = {publication.tile_count}; "
        f"reused tasks = {job_run.reused_tasks}; "
        f"executed tasks = {job_run.executed_tasks}"
    )
    print(
        "  scope = exact stationary Kerr + zero-torque Novikov-Thorne scalar "
        "thin disk with D20 angular law; not NR, GRMHD, polarization, "
        "returning radiation, or a solved atmosphere"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
