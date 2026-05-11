import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import setup as kimodo_setup  # noqa: E402


class SetupPathTests(unittest.TestCase):
    def test_windows_venv_executables_use_scripts_directory(self):
        venv_dir = Path("venv")

        with mock.patch.object(kimodo_setup.platform, "system", return_value="Windows"):
            self.assertEqual(kimodo_setup.venv_python(venv_dir), venv_dir / "Scripts" / "python.exe")
            self.assertEqual(kimodo_setup.venv_pip(venv_dir), venv_dir / "Scripts" / "pip.exe")

    def test_posix_venv_executables_use_bin_directory(self):
        venv_dir = Path("venv")

        with mock.patch.object(kimodo_setup.platform, "system", return_value="Linux"):
            self.assertEqual(kimodo_setup.venv_python(venv_dir), venv_dir / "bin" / "python")
            self.assertEqual(kimodo_setup.venv_pip(venv_dir), venv_dir / "bin" / "pip")

    def test_pip_install_uses_list_form_python_module_command_without_subprocess(self):
        calls = []

        def capture(command, *, env=None):
            calls.append((command, env))

        install_env = {"TEST_ENV": "1"}

        with mock.patch.object(kimodo_setup, "run", side_effect=capture):
            kimodo_setup.pip_install(Path("venv") / "Scripts" / "python.exe", "--upgrade", "pip", env=install_env)

        self.assertEqual(
            calls,
            [
                (
                    [str(Path("venv") / "Scripts" / "python.exe"), "-m", "pip", "install", "--upgrade", "pip"],
                    install_env,
                )
            ],
        )
        self.assertIsInstance(calls[0][0], list)

    def test_torch_install_uses_list_form_command_without_subprocess(self):
        compatibility = type(
            "Compatibility",
            (),
            {
                "wheel_urls": (),
                "torch_packages": ("torch==2.6.0", "torchvision==0.21.0"),
                "torch_index": "https://download.pytorch.org/whl/cu124",
            },
        )()
        calls = []

        with mock.patch.object(kimodo_setup, "run", side_effect=lambda command, *, env=None: calls.append(command)):
            kimodo_setup.install_torch(Path("venv") / "bin" / "python", compatibility)

        self.assertEqual(
            calls,
            [
                [
                    str(Path("venv") / "bin" / "python"),
                    "-m",
                    "pip",
                    "install",
                    "torch==2.6.0",
                    "torchvision==0.21.0",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu124",
                ]
            ],
        )
        self.assertIsInstance(calls[0], list)


if __name__ == "__main__":
    unittest.main()
