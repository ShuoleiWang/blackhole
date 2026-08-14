#!/usr/bin/env python3
"""Compose authenticated vacuum transfer endpoints into spectral float32 tiles."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.vacuum import (  # noqa: E402
    PlanckBlackbodyEnvironment,
    VacuumRenderError,
    render_offline_vacuum,
)
from scripts.verify_nr_contract import ContractError  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        help="authenticated blackhole.nr-transfer-map/v1 manifest",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="new output directory (must not already exist)",
    )
    parser.add_argument(
        "--frequency-hz",
        action="append",
        required=True,
        type=float,
        help=(
            "observer-frame frequency bin in Hz; repeat in strictly increasing "
            "order"
        ),
    )
    parser.add_argument(
        "--temperature-k",
        type=float,
        default=6500.0,
        help="isotropic analytic Planck-environment temperature (default: 6500)",
    )
    parser.add_argument(
        "--normalization",
        type=float,
        default=1.0,
        help="dimensionless Planck I_nu multiplier (default: 1)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        environment = PlanckBlackbodyEnvironment(
            arguments.temperature_k,
            arguments.normalization,
        )
        result = render_offline_vacuum(
            arguments.manifest,
            arguments.output,
            arguments.frequency_hz,
            environment,
        )
    except (ContractError, VacuumRenderError, ValueError, OSError) as error:
        print(f"Offline vacuum spectral composition failed: {error}", file=sys.stderr)
        return 1

    print("Offline vacuum spectral composition completed")
    print(f"  manifest = {result.manifest_path}")
    print(f"  manifest sha256 = {result.manifest_sha256}")
    print(
        f"  records = {result.record_count}; chunks = {result.chunk_count}; "
        f"frequency bins = {len(arguments.frequency_hz)}"
    )
    print(
        "  scope = authenticated vacuum endpoint composition only; "
        "not an NR solver, GRRT plasma calculation, or OpenEXR master"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
