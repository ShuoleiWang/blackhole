#!/usr/bin/env python3
"""Convert scientific CIE XYZ v1 to unclamped scene-linear sRGB v1.

This CLI wrapper is part of the converter's authenticated source closure.
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

from offline.cie_product import CieProductError
from offline.linear_rgb_product import (
    LinearRgbProductError,
    convert_cie_xyz_product_to_linear_srgb,
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_cie_xyz_manifest", type=Path)
    parser.add_argument("input_spectral_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        publication = convert_cie_xyz_product_to_linear_srgb(
            arguments.input_cie_xyz_manifest,
            arguments.input_spectral_manifest,
            arguments.output_directory,
        )
    except (
        CieProductError,
        LinearRgbProductError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"offline linear-sRGB conversion failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "id": publication.product_id,
                "manifest": str(publication.manifest_path),
                "manifestSha256": publication.manifest_sha256,
                "productSha256": publication.product_sha256,
                "recordCount": publication.record_count,
                "tileCount": publication.tile_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
