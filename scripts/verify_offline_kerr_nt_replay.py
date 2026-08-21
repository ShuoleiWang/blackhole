#!/usr/bin/env python3
"""Replay-verify an authenticated exact-Kerr Novikov--Thorne frame.

The result is a byte-exact deterministic numerical replay with the same local
production code family.  It is not an independent analytic physics oracle,
numerical relativity, GRMHD, polarization, or returning-radiation validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.kerr_nt_replay import (  # noqa: E402
    DEFAULT_REPLAY_LIMITS,
    KerrNtReplayError,
    MAXIMUM_OFFICIAL_CIE_FREQUENCY_BINS,
    ReplayResourceLimits,
    validate_kerr_nt_replay,
)
from scripts.verify_nr_contract import ContractError  # noqa: E402
from scripts.verify_offline_spectral_frame import DEFAULT_SCHEMA  # noqa: E402


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _cie_frequency_limit(value: str) -> int:
    parsed = _positive_integer(value)
    if parsed > MAXIMUM_OFFICIAL_CIE_FREQUENCY_BINS:
        raise argparse.ArgumentTypeError(
            "value may not exceed the official CIE "
            f"{MAXIMUM_OFFICIAL_CIE_FREQUENCY_BINS}-bin grid"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--max-manifest-bytes",
        type=_positive_integer,
        default=DEFAULT_REPLAY_LIMITS.maximum_manifest_bytes,
    )
    parser.add_argument(
        "--max-source-file-bytes",
        type=_positive_integer,
        default=DEFAULT_REPLAY_LIMITS.maximum_source_file_bytes,
    )
    parser.add_argument(
        "--max-product-bytes",
        type=_positive_integer,
        default=DEFAULT_REPLAY_LIMITS.maximum_product_bytes,
    )
    parser.add_argument(
        "--max-tile-bytes",
        type=_positive_integer,
        default=DEFAULT_REPLAY_LIMITS.maximum_tile_bytes,
    )
    parser.add_argument(
        "--max-tiles",
        type=_positive_integer,
        default=DEFAULT_REPLAY_LIMITS.maximum_tiles,
    )
    parser.add_argument(
        "--max-records",
        type=_positive_integer,
        default=DEFAULT_REPLAY_LIMITS.maximum_records,
    )
    parser.add_argument(
        "--maximum-frequency-bins",
        "--max-frequency-bins",
        dest="max_frequency_bins",
        type=_cie_frequency_limit,
        default=DEFAULT_REPLAY_LIMITS.maximum_frequency_bins,
    )
    parser.add_argument(
        "--max-total-ray-evaluations",
        type=_positive_integer,
        default=DEFAULT_REPLAY_LIMITS.maximum_total_ray_evaluations,
    )
    parser.add_argument(
        "--max-adaptive-depth",
        type=_non_negative_integer,
        default=DEFAULT_REPLAY_LIMITS.maximum_adaptive_depth,
    )
    parser.add_argument(
        "--max-ray-accepted-steps",
        type=_positive_integer,
        default=DEFAULT_REPLAY_LIMITS.maximum_ray_accepted_steps,
    )
    parser.add_argument(
        "--max-ray-rejected-steps",
        type=_positive_integer,
        default=DEFAULT_REPLAY_LIMITS.maximum_ray_rejected_steps,
    )
    parser.add_argument(
        "--max-ray-event-iterations",
        type=_positive_integer,
        default=DEFAULT_REPLAY_LIMITS.maximum_ray_event_iterations,
    )
    parser.add_argument(
        "--max-surface-iterations",
        type=_positive_integer,
        default=DEFAULT_REPLAY_LIMITS.maximum_surface_iterations,
    )
    parser.add_argument(
        "--max-surface-reintegrations",
        type=_positive_integer,
        default=DEFAULT_REPLAY_LIMITS.maximum_surface_reintegrations,
    )
    parser.add_argument(
        "--max-surface-subdivisions-per-segment",
        type=_positive_integer,
        default=(
            DEFAULT_REPLAY_LIMITS.maximum_surface_subdivisions_per_segment
        ),
    )
    parser.add_argument(
        "--max-affine-length-m",
        type=_positive_float,
        default=DEFAULT_REPLAY_LIMITS.maximum_affine_length_m,
    )
    return parser


def _limits(arguments: argparse.Namespace) -> ReplayResourceLimits:
    return ReplayResourceLimits(
        maximum_manifest_bytes=arguments.max_manifest_bytes,
        maximum_source_file_bytes=arguments.max_source_file_bytes,
        maximum_product_bytes=arguments.max_product_bytes,
        maximum_tile_bytes=arguments.max_tile_bytes,
        maximum_tiles=arguments.max_tiles,
        maximum_records=arguments.max_records,
        maximum_frequency_bins=arguments.max_frequency_bins,
        maximum_total_ray_evaluations=arguments.max_total_ray_evaluations,
        maximum_adaptive_depth=arguments.max_adaptive_depth,
        maximum_ray_accepted_steps=arguments.max_ray_accepted_steps,
        maximum_ray_rejected_steps=arguments.max_ray_rejected_steps,
        maximum_ray_event_iterations=arguments.max_ray_event_iterations,
        maximum_surface_iterations=arguments.max_surface_iterations,
        maximum_surface_reintegrations=arguments.max_surface_reintegrations,
        maximum_surface_subdivisions_per_segment=(
            arguments.max_surface_subdivisions_per_segment
        ),
        maximum_affine_length_m=arguments.max_affine_length_m,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = validate_kerr_nt_replay(
            arguments.manifest,
            arguments.schema,
            limits=_limits(arguments),
        )
    except (
        ArithmeticError,
        ContractError,
        KerrNtReplayError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"offline Kerr/NT replay validation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
