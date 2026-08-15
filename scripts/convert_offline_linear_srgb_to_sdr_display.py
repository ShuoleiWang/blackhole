#!/usr/bin/env python3
"""Convert authenticated linear-sRGB v1 to an SDR PPM16 quicklook v1.

This CLI wrapper is part of the display transform's authenticated source closure.
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
from offline.display_product import (
    DisplayProductError,
    convert_linear_srgb_to_sdr_display,
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_linear_srgb_manifest", type=Path)
    parser.add_argument("input_cie_xyz_manifest", type=Path)
    parser.add_argument("input_spectral_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--exposure",
        type=float,
        required=True,
        help="Fixed positive manual exposure multiplier.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        publication = convert_linear_srgb_to_sdr_display(
            arguments.input_linear_srgb_manifest,
            arguments.input_cie_xyz_manifest,
            arguments.input_spectral_manifest,
            arguments.output_directory,
            exposure=arguments.exposure,
        )
    except (
        CieProductError,
        DisplayProductError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"offline SDR display conversion failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "id": publication.product_id,
                "imageCount": publication.image_count,
                "manifest": str(publication.manifest_path),
                "manifestSha256": publication.manifest_sha256,
                "pixelCount": publication.pixel_count,
                "productSha256": publication.product_sha256,
                "tileCount": publication.tile_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
