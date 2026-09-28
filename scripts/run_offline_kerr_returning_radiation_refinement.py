#!/usr/bin/env python3
"""Run a kernel-only authenticated Kerr returning-radiation refinement.

Exit status 0 means the published five-grid checkpoint is v2-qualified, 2
means a complete source-current checkpoint was published but is not qualified,
and 1 means execution/authentication/publication failed.  No CIE table, camera,
thermal profile, spectrum, tile, frame, or display product is constructed.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Sequence
from types import ModuleType
from importlib.machinery import ModuleSpec, SourceFileLoader


# This resolved path is bootstrap-only so direct ``python scripts/...`` can
# import the package.  The security gate below uses the unresolved invocation
# path plus the checkpoint module's independently anchored source root.
_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
ROOT = _BOOTSTRAP_ROOT
while str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from offline.geodesic import RayTraceOptions, SurfaceEventOptions  # noqa: E402
from offline.kerr import KerrKerrSchildMetric, KerrOblateTermination  # noqa: E402
from offline.kerr_finite_thickness import (  # noqa: E402
    StationaryKerrFiniteThicknessCalibration,
)
from offline.kerr_finite_thickness_area import (  # noqa: E402
    KerrFiniteThicknessAreaQuadraturePolicy,
)
from offline.kerr_finite_thickness_surface import (  # noqa: E402
    KerrFiniteThicknessMultiSurface,
)
from offline.kerr_returning_radiation_convergence_v2 import (  # noqa: E402
    KerrReturningRadiationConvergenceV2Policy,
)
import offline.kerr_returning_radiation_convergence_v2 as _convergence  # noqa: E402
from offline.kerr_returning_radiation_kernel import (  # noqa: E402
    KerrReturningRadiationKernelPolicy,
)
from offline.kerr_returning_radiation_refinement_checkpoint import (  # noqa: E402
    KerrReturningRadiationRefinementPlan,
    execute_kerr_returning_radiation_refinement_checkpoint,
)
import offline.kerr_returning_radiation_refinement_checkpoint as _checkpoint  # noqa: E402
from offline.novikov_thorne import PROGRADE, RETROGRADE  # noqa: E402


ROOT = _checkpoint._SOURCE_ROOT
_EXPECTED_CLI_FILE = ROOT / "scripts/run_offline_kerr_returning_radiation_refinement.py"
_CLI_FILE = Path(os.path.abspath(__file__))
_EXECUTING_MODULE_NAME = __name__
_EXECUTING_MODULE = sys.modules.get(_EXECUTING_MODULE_NAME)
_EXECUTING_MODULE_SPEC = globals().get("__spec__")
_EXECUTING_MODULE_LOADER = globals().get("__loader__")
_ORIGIN_MODULES = (
    (
        "offline.kerr_returning_radiation_convergence_v2",
        _convergence,
        ROOT / "offline/kerr_returning_radiation_convergence_v2.py",
    ),
    (
        "offline.kerr_returning_radiation_refinement_checkpoint",
        _checkpoint,
        ROOT / "offline/kerr_returning_radiation_refinement_checkpoint.py",
    ),
    (_EXECUTING_MODULE_NAME, _EXECUTING_MODULE, _CLI_FILE),
)
_EXECUTE_REFINEMENT_CHECKPOINT_PUBLIC_ENTRY = (
    _checkpoint._EXECUTE_REFINEMENT_CHECKPOINT_CANONICAL_ENTRY
)
_EXECUTE_REFINEMENT_CHECKPOINT_CALL_ENTRY = (
    _EXECUTE_REFINEMENT_CHECKPOINT_PUBLIC_ENTRY
)


def _require_exact_cli_module_origins() -> None:
    if _EXECUTING_MODULE_NAME not in (
        "__main__",
        "scripts.run_offline_kerr_returning_radiation_refinement",
    ):
        raise RuntimeError("refinement CLI has an unsupported module identity")
    if _CLI_FILE != _EXPECTED_CLI_FILE:
        raise RuntimeError("refinement CLI was not invoked by its exact checkout path")
    if _EXECUTING_MODULE_NAME == "__main__":
        if (
            _EXECUTING_MODULE_SPEC is not None
            or type(_EXECUTING_MODULE_LOADER) is not SourceFileLoader
            or type(_EXECUTING_MODULE_LOADER.name) is not str
            or _EXECUTING_MODULE_LOADER.name != "__main__"
            or type(_EXECUTING_MODULE_LOADER.path) is not str
            or Path(_EXECUTING_MODULE_LOADER.path) != _EXPECTED_CLI_FILE
        ):
            raise RuntimeError("refinement CLI has a foreign direct-loader identity")
    elif (
        type(_EXECUTING_MODULE_SPEC) is not ModuleSpec
        or type(_EXECUTING_MODULE_LOADER) is not SourceFileLoader
        or _EXECUTING_MODULE_SPEC.loader is not _EXECUTING_MODULE_LOADER
        or _EXECUTING_MODULE_SPEC.name != _EXECUTING_MODULE_NAME
        or Path(_EXECUTING_MODULE_SPEC.origin) != _EXPECTED_CLI_FILE
        or _EXECUTING_MODULE_LOADER.name != _EXECUTING_MODULE_NAME
        or Path(_EXECUTING_MODULE_LOADER.path) != _EXPECTED_CLI_FILE
    ):
        raise RuntimeError("refinement CLI has a foreign import identity")
    for module_name, expected_module, expected_file in _ORIGIN_MODULES:
        module = sys.modules.get(module_name)
        if type(module) is not ModuleType or module is not expected_module:
            raise RuntimeError(
                f"refinement source module {module_name} is not the exact loaded module"
            )
        try:
            raw_origin = object.__getattribute__(module, "__file__")
        except AttributeError as error:
            raise RuntimeError(
                f"refinement source module {module_name} has no file origin"
            ) from error
        if (
            type(raw_origin) is not str
            or not Path(raw_origin).is_absolute()
            or Path(raw_origin) != expected_file
        ):
            raise RuntimeError(
                f"refinement source module {module_name} was not loaded from "
                "the exact renderer ROOT"
            )
    _checkpoint._require_exact_source_module_origins()
    _checkpoint._assert_runtime_bindings()
    if (
        _checkpoint._EXECUTE_REFINEMENT_CHECKPOINT_CANONICAL_ENTRY
        is not _EXECUTE_REFINEMENT_CHECKPOINT_PUBLIC_ENTRY
        or _checkpoint.execute_kerr_returning_radiation_refinement_checkpoint
        is not _EXECUTE_REFINEMENT_CHECKPOINT_PUBLIC_ENTRY
        or execute_kerr_returning_radiation_refinement_checkpoint
        is not _EXECUTE_REFINEMENT_CHECKPOINT_PUBLIC_ENTRY
    ):
        raise RuntimeError("refinement checkpoint execute runtime binding changed")


def _checkpoint_cli_origin_guard() -> None:
    """Let the checkpoint front door recheck this exact CLI before cache I/O."""

    _require_exact_cli_module_origins()


_CHECKPOINT_CLI_ORIGIN_GUARD_PUBLIC_ENTRY = _checkpoint_cli_origin_guard


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "output",
        type=Path,
        help="new absolute checkpoint directory; never overwritten",
    )
    parser.add_argument(
        "--kernel-cache",
        required=True,
        type=Path,
        help="separate absolute resumable production-forward cache root",
    )
    native = parser.add_argument_group("explicit native forward evaluator")
    native.add_argument(
        "--native-kerr-cpu-library",
        type=Path,
        default=None,
        help=(
            "exact absolute ABI-v3 dylib path; presence selects native-cpu "
            "without fallback"
        ),
    )
    native.add_argument(
        "--native-kerr-cpu-segment-capacity",
        type=_positive_integer,
        default=None,
    )
    native.add_argument(
        "--native-kerr-cpu-crossing-capacity",
        type=_positive_integer,
        default=None,
    )

    physics = parser.add_argument_group("stationary finite-height Kerr physics")
    physics.add_argument("--metric-mass-m", type=float, default=1.0)
    physics.add_argument(
        "--spin", "--dimensionless-spin", dest="spin", type=float, default=0.7
    )
    physics.add_argument(
        "--height-accretion-rate-eddington",
        "--dotm",
        dest="height_accretion_rate_eddington",
        type=float,
        default=0.05,
    )
    physics.add_argument(
        "--outer-radius-over-mass",
        "--rout-over-mass",
        dest="outer_radius_over_mass",
        type=float,
        default=25.0,
    )
    physics.add_argument(
        "--thinness-gate-maximum-h-over-rho", type=float, default=0.25
    )
    physics.add_argument(
        "--orientation", choices=(PROGRADE, RETROGRADE), default=PROGRADE
    )
    physics.add_argument("--singularity-guard-over-mass", type=float, default=1.0e-9)
    physics.add_argument("--escape-radius-over-mass", type=float, default=50.0)
    physics.add_argument("--horizon-offset-over-mass", type=float, default=0.02)

    annuli = parser.add_argument_group("finite-volume annuli")
    choice = annuli.add_mutually_exclusive_group()
    choice.add_argument("--annulus-count", type=_positive_integer, default=1)
    choice.add_argument(
        "--annulus-edges-over-mass", type=float, nargs="+", default=None
    )

    ray = parser.add_argument_group("fine ray and surface event controls")
    ray.add_argument("--ray-absolute-tolerance", type=float, default=5.0e-10)
    ray.add_argument("--ray-relative-tolerance", type=float, default=5.0e-10)
    ray.add_argument("--ray-initial-step-over-mass", type=float, default=0.05)
    ray.add_argument("--ray-minimum-step-over-mass", type=float, default=1.0e-8)
    ray.add_argument("--ray-maximum-step-over-mass", type=float, default=0.25)
    ray.add_argument(
        "--ray-maximum-affine-length-over-mass", type=float, default=300.0
    )
    ray.add_argument(
        "--ray-maximum-accepted-steps", type=_positive_integer, default=100_000
    )
    ray.add_argument(
        "--ray-maximum-rejected-steps", type=_positive_integer, default=100_000
    )
    ray.add_argument("--ray-null-residual-limit", type=float, default=2.0e-7)
    ray.add_argument(
        "--ray-metric-interpolation-error-limit", type=float, default=1.0e-7
    )
    ray.add_argument(
        "--ray-event-value-tolerance-over-mass", type=float, default=1.0e-9
    )
    ray.add_argument(
        "--ray-event-affine-tolerance-over-mass", type=float, default=1.0e-10
    )
    ray.add_argument(
        "--ray-event-maximum-iterations", type=_positive_integer, default=64
    )
    ray.add_argument("--surface-absolute-tolerance", type=float, default=5.0e-10)
    ray.add_argument("--surface-relative-tolerance", type=float, default=5.0e-10)
    ray.add_argument("--surface-null-residual-limit", type=float, default=2.0e-7)
    ray.add_argument(
        "--surface-metric-interpolation-error-limit", type=float, default=1.0e-7
    )
    ray.add_argument("--surface-value-tolerance", type=float, default=1.0e-9)
    ray.add_argument(
        "--surface-affine-tolerance-over-mass", type=float, default=1.0e-10
    )
    ray.add_argument(
        "--surface-maximum-iterations", type=_positive_integer, default=64
    )
    ray.add_argument(
        "--surface-maximum-reintegrations", type=_positive_integer, default=100_000
    )
    ray.add_argument(
        "--surface-subdivisions-per-segment", type=_positive_integer, default=4
    )

    kernel = parser.add_argument_group("forward five-grid kernel")
    kernel.add_argument("--kernel-rho-order", type=_positive_integer, default=8)
    kernel.add_argument("--kernel-mu-order", type=_positive_integer, default=16)
    kernel.add_argument("--kernel-psi-count", type=_positive_integer, default=32)
    kernel.add_argument("--kernel-absolute-tolerance", type=float, default=2.0e-2)
    kernel.add_argument("--kernel-relative-tolerance", type=float, default=5.0e-2)
    kernel.add_argument(
        "--kernel-symmetry-absolute-tolerance", type=float, default=2.0e-8
    )
    kernel.add_argument(
        "--kernel-symmetry-relative-tolerance", type=float, default=2.0e-7
    )
    kernel.add_argument(
        "--kernel-maximum-direction-evaluations",
        type=_positive_integer,
        default=2_000_000,
    )
    kernel.add_argument(
        "--kernel-maximum-whole-ray-traces",
        type=_positive_integer,
        default=8_000_000,
    )
    kernel.add_argument(
        "--kernel-directions-per-task", type=_positive_integer, default=64
    )
    kernel.add_argument("--kernel-jobs", type=_positive_integer, default=1)
    kernel.add_argument("--kernel-max-in-flight", type=_positive_integer, default=None)

    area = parser.add_argument_group("finite-height proper annulus area")
    area.add_argument("--area-gauss-legendre-order", type=_positive_integer, default=24)
    area.add_argument("--area-relative-tolerance", type=float, default=2.0e-10)
    area.add_argument(
        "--area-absolute-tolerance-over-mass-squared", type=float, default=2.0e-11
    )
    area.add_argument(
        "--area-maximum-point-evaluations", type=_positive_integer, default=384
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _annulus_edges(
    arguments: argparse.Namespace,
    calibration: StationaryKerrFiniteThicknessCalibration,
) -> tuple[float, ...]:
    inner = float(calibration.isco_radius_over_mass)
    outer = float(calibration.outer_radius_over_mass)
    if arguments.annulus_edges_over_mass is None:
        count = arguments.annulus_count
        span = outer - inner
        edges = (
            inner,
            *(math.fsum((inner, span * index / count)) for index in range(1, count)),
            outer,
        )
    else:
        edges = tuple(float(value) for value in arguments.annulus_edges_over_mass)
    if len(edges) < 2 or any(not math.isfinite(value) for value in edges):
        raise ValueError("annulus edges must contain at least two finite values")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("annulus edges must be strictly increasing")
    if edges[0].hex() != inner.hex() or edges[-1].hex() != outer.hex():
        raise ValueError("annulus edges must exactly cover ISCO through R_out")
    return edges


def build_refinement_plan(
    arguments: argparse.Namespace,
) -> KerrReturningRadiationRefinementPlan:
    """Build only the stationary transport and finite-grid refinement inputs."""

    _require_exact_cli_module_origins()
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
        affine_tolerance=arguments.surface_affine_tolerance_over_mass * mass_m,
        maximum_iterations=arguments.surface_maximum_iterations,
        maximum_reintegrations=arguments.surface_maximum_reintegrations,
        subdivisions_per_segment=arguments.surface_subdivisions_per_segment,
    )
    kernel_policy = KerrReturningRadiationKernelPolicy(
        rho_order=arguments.kernel_rho_order,
        mu_order=arguments.kernel_mu_order,
        psi_count=arguments.kernel_psi_count,
        absolute_tolerance=arguments.kernel_absolute_tolerance,
        relative_tolerance=arguments.kernel_relative_tolerance,
        symmetry_absolute_tolerance=(
            arguments.kernel_symmetry_absolute_tolerance
        ),
        symmetry_relative_tolerance=(
            arguments.kernel_symmetry_relative_tolerance
        ),
        maximum_direction_evaluations=(
            arguments.kernel_maximum_direction_evaluations
        ),
        maximum_whole_ray_traces=arguments.kernel_maximum_whole_ray_traces,
    )
    area_policy = KerrFiniteThicknessAreaQuadraturePolicy(
        gauss_legendre_order=arguments.area_gauss_legendre_order,
        relative_tolerance=arguments.area_relative_tolerance,
        absolute_tolerance_over_mass_squared=(
            arguments.area_absolute_tolerance_over_mass_squared
        ),
        maximum_point_evaluations=arguments.area_maximum_point_evaluations,
    )
    native_library = arguments.native_kerr_cpu_library
    if native_library is None:
        if (
            arguments.native_kerr_cpu_segment_capacity is not None
            or arguments.native_kerr_cpu_crossing_capacity is not None
        ):
            raise ValueError(
                "native capacities require --native-kerr-cpu-library"
            )
        evaluator_mode = _checkpoint.PYTHON_EVALUATOR_MODE
        native_segment_capacity = None
        native_crossing_capacity = None
    else:
        if type(native_library) is not type(Path()) or not native_library.is_absolute():
            raise ValueError(
                "--native-kerr-cpu-library must be an exact absolute path"
            )
        evaluator_mode = _checkpoint.NATIVE_CPU_EVALUATOR_MODE
        native_segment_capacity = (
            100_000
            if arguments.native_kerr_cpu_segment_capacity is None
            else arguments.native_kerr_cpu_segment_capacity
        )
        native_crossing_capacity = (
            100_000
            if arguments.native_kerr_cpu_crossing_capacity is None
            else arguments.native_kerr_cpu_crossing_capacity
        )
    return KerrReturningRadiationRefinementPlan(
        output_directory=Path(arguments.output),
        cache_root=Path(arguments.kernel_cache),
        surface=surface,
        termination=termination,
        annulus_edges_over_mass=_annulus_edges(arguments, calibration),
        ray_options=ray_options,
        surface_options=surface_options,
        kernel_policy=kernel_policy,
        area_policy=area_policy,
        convergence_policy=KerrReturningRadiationConvergenceV2Policy(),
        evaluator_mode=evaluator_mode,
        native_library_path=native_library,
        native_segment_capacity=native_segment_capacity,
        native_crossing_capacity=native_crossing_capacity,
        directions_per_task=arguments.kernel_directions_per_task,
        jobs=arguments.kernel_jobs,
        max_in_flight=arguments.kernel_max_in_flight,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        _require_exact_cli_module_origins()
        arguments = parse_args(argv)
        plan = build_refinement_plan(arguments)
        _require_exact_cli_module_origins()
        original_guard = _checkpoint._CLI_ORIGIN_GUARD_CALL_ENTRY
        _checkpoint._CLI_ORIGIN_GUARD_CALL_ENTRY = (
            _CHECKPOINT_CLI_ORIGIN_GUARD_PUBLIC_ENTRY
        )
        try:
            publication = _EXECUTE_REFINEMENT_CHECKPOINT_CALL_ENTRY(plan)
        finally:
            _checkpoint._CLI_ORIGIN_GUARD_CALL_ENTRY = original_guard
        _require_exact_cli_module_origins()
    except Exception as error:
        print(f"Offline Kerr returning-radiation refinement failed: {error}", file=sys.stderr)
        return 1
    print("Offline Kerr returning-radiation refinement checkpoint published")
    print(f"  manifest = {publication.manifest_path}")
    print(f"  manifest sha256 = {publication.manifest_sha256}")
    print(f"  checkpoint id = {publication.checkpoint_id}")
    print(f"  v2 qualified = {publication.qualified}")
    print(
        "  scope = kernel-only same-code five-grid evidence; no thermal, "
        "spectrum, CIE, camera, tile, frame, or display product"
    )
    return 0 if publication.qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
