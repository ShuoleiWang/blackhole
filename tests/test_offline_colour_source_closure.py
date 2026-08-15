from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

from offline.cie_product import CONVERTER_SOURCE_FILES as CIE_SOURCE_FILES
from offline.display_product import DISPLAY_SOURCE_FILES
from offline.linear_rgb_product import (
    CONVERTER_SOURCE_FILES as LINEAR_SOURCE_FILES,
)


ROOT = Path(__file__).resolve().parents[1]


class OfflineColourSourceClosureTests(unittest.TestCase):
    def test_clean_imports_exactly_match_each_declared_python_closure(self) -> None:
        probe = """
import importlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
for name in sys.argv[2:]:
    importlib.import_module(name)
paths = set()
for module in tuple(sys.modules.values()):
    source = getattr(module, "__file__", None)
    if source is None:
        continue
    try:
        relative = Path(source).resolve().relative_to(root)
    except (OSError, ValueError):
        continue
    if relative.suffix == ".py":
        paths.add(relative.as_posix())
print(json.dumps(sorted(paths)))
"""
        cases = (
            (
                "cie",
                CIE_SOURCE_FILES,
                "scripts.convert_offline_spectral_to_cie_xyz",
                "scripts.verify_offline_cie_xyz",
            ),
            (
                "linear",
                LINEAR_SOURCE_FILES,
                "scripts.convert_offline_cie_xyz_to_linear_srgb",
                "scripts.verify_offline_linear_srgb",
            ),
            (
                "display",
                DISPLAY_SOURCE_FILES,
                "scripts.convert_offline_linear_srgb_to_sdr_display",
                "scripts.verify_offline_sdr_display",
            ),
        )
        for label, closure, cli_module, verifier_module in cases:
            with self.subTest(layer=label):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        probe,
                        str(ROOT),
                        cli_module,
                        verifier_module,
                    ],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                imported = tuple(json.loads(completed.stdout))
                expected = tuple(uri for uri in closure if uri.endswith(".py"))
                self.assertEqual(imported, expected)


if __name__ == "__main__":
    unittest.main()
