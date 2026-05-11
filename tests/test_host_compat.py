import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from host_compat import (  # noqa: E402
    CU124_INDEX,
    EXPERIMENTAL,
    SUPPORTED,
    X86_CU124_PACKAGES,
    resolve_host_compatibility,
)


class HostCompatibilityTests(unittest.TestCase):
    def test_windows_amd64_is_normalized_and_classified_experimental(self):
        facts, compatibility = resolve_host_compatibility(
            system="Windows",
            machine="AMD64",
            gpu_sm=70,
            cuda_version=124,
            python_tag="cp312",
            gpu_name="RTX test GPU",
        )

        self.assertEqual(facts.machine, "x86_64")
        self.assertEqual(facts.platform_key, "windows/x86_64")
        self.assertEqual(compatibility.state, EXPERIMENTAL)
        self.assertEqual(compatibility.platform_key, "windows/x86_64")
        self.assertIsNotNone(compatibility.lane)
        self.assertEqual(compatibility.torch_index, CU124_INDEX)
        self.assertEqual(compatibility.torch_packages, X86_CU124_PACKAGES)

    def test_windows_x64_alias_preserves_experimental_dict_notes(self):
        _facts, compatibility = resolve_host_compatibility(
            system="Windows",
            machine="x64",
            gpu_sm=70,
            cuda_version=124,
            python_tag="cp312",
        )

        payload = compatibility.as_dict()

        self.assertEqual(payload["state"], "experimental")
        self.assertEqual(payload["platform_key"], "windows/x86_64")
        self.assertIn("Windows x86_64 is experimental and has not been validated on a real Windows host.", payload["notes"])
        self.assertIn("Setup can still fail on Windows because of CUDA, PyTorch, DLL loading, or gated model access.", payload["notes"])

    def test_linux_x86_64_supported_lane_is_unchanged(self):
        _facts, compatibility = resolve_host_compatibility(
            system="Linux",
            machine="x86_64",
            gpu_sm=70,
            cuda_version=124,
            python_tag="cp312",
        )

        self.assertEqual(compatibility.state, SUPPORTED)
        self.assertEqual(compatibility.platform_key, "linux/x86_64")
        self.assertEqual(compatibility.lane, "linux-x86_64-cu124")
        self.assertEqual(compatibility.reason, "Linux x86_64 + NVIDIA SM 70+ is a supported Kimodo runtime lane.")
        self.assertEqual(compatibility.torch_index, CU124_INDEX)
        self.assertEqual(compatibility.torch_packages, X86_CU124_PACKAGES)
        self.assertEqual(compatibility.wheel_urls, ())

    def test_linux_arm64_experimental_lane_is_not_expanded_to_supported(self):
        _facts, compatibility = resolve_host_compatibility(
            system="Linux",
            machine="aarch64",
            gpu_sm=70,
            cuda_version=124,
            python_tag="cp312",
        )

        self.assertEqual(compatibility.state, EXPERIMENTAL)
        self.assertNotEqual(compatibility.state, SUPPORTED)
        self.assertEqual(compatibility.platform_key, "linux/arm64")
        self.assertEqual(compatibility.lane, "linux-arm64-cu124")
        self.assertEqual(compatibility.torch_index, CU124_INDEX)
        self.assertEqual(compatibility.torch_packages, ())
        self.assertGreater(len(compatibility.wheel_urls), 0)


if __name__ == "__main__":
    unittest.main()
