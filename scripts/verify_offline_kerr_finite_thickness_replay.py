#!/usr/bin/env python3
"""Replay-verify a finite-thickness exact-Kerr spectral frame.

The result is byte-exact deterministic numerical replay with the same local
production code family.  It is not an independent physics oracle, numerical
relativity, GRMHD, a solved atmosphere, complete GRRT, or returning-radiation
validation.
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

from offline.kerr_finite_thickness_replay import (  # noqa: E402
    KerrFiniteThicknessReplayError,
    validate_kerr_finite_thickness_replay,
)
from scripts.verify_nr_contract import ContractError  # noqa: E402
import scripts.verify_offline_kerr_nt_replay as common_cli  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Return the shared bounded replay CLI with finite-height scope text."""

    parser = common_cli.build_parser()
    parser.description = __doc__
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = validate_kerr_finite_thickness_replay(
            arguments.manifest,
            arguments.schema,
            limits=common_cli._limits(arguments),
        )
    except (
        ArithmeticError,
        ContractError,
        KerrFiniteThicknessReplayError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"offline finite-thickness Kerr replay validation failed: {error}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
