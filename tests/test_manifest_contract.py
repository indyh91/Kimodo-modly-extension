import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest.json"


def read_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


class ManifestContractTests(unittest.TestCase):
    def test_animate_rigged_mesh_multi_inputs_have_labels_for_renderer_fallbacks(self):
        manifest = read_manifest()
        node = next(node for node in manifest["nodes"] if node["id"] == "animate-rigged-mesh")

        self.assertEqual(node["input"], "text")
        self.assertEqual(node["output"], "mesh")

        ports = {port["name"]: port for port in node["inputs"]}
        self.assertEqual(set(ports), {"prompt", "rigged_mesh"})

        self.assertEqual(ports["prompt"]["type"], "text")
        self.assertEqual(ports["rigged_mesh"]["type"], "mesh")

        for port in ports.values():
            self.assertIsInstance(port.get("label"), str)
            self.assertTrue(port["label"].strip())
            self.assertTrue(port["required"])


if __name__ == "__main__":
    unittest.main()
