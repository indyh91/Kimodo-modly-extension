"""Regression test for Modly's runpy-based setup launcher.

Modly can run setup.py from a different working directory without adding the
extension root to sys.path. This test deliberately isolates Python's import path
and avoids invoking setup() or installing dependencies.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SetupEntrypointImportTests(unittest.TestCase):
    def test_runpy_from_unrelated_directory_imports_sibling_host_compat(self):
        probe = (
            "import pathlib, runpy, sys\n"
            "entry = pathlib.Path(sys.argv[1]).resolve()\n"
            "runpy.run_path(str(entry), run_name='modly_import_probe')\n"
            "import host_compat\n"
            "assert pathlib.Path(host_compat.__file__).resolve() == "
            "entry.with_name('host_compat.py')\n"
        )
        with tempfile.TemporaryDirectory() as outside_dir:
            result = subprocess.run(
                [sys.executable, "-I", "-c", probe, str(ROOT / "setup.py")],
                cwd=outside_dir,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
