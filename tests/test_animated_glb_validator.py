import base64
import copy
import json
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glb_animation_exporter import GlbAnimationExporter  # noqa: E402
from retargeting import JointMapping, RetargetChannel, RetargetClip  # noqa: E402
from rig_inspector import inspect_rig  # noqa: E402
from validation.animated_glb_validator import main, validate_bundle, validate_gltf_document  # noqa: E402


FIXTURES = ROOT / "tests" / "fixtures"


class AnimatedGlbValidatorTests(unittest.TestCase):
    def test_positive_bundle_validates_generated_animated_glb(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._bundle(Path(tmp), animated_suffix=".glb")

            report = validate_bundle(bundle)

        self.assertTrue(report["passed"], report)
        self.assertEqual(report["animation"]["name"], "Kimodo Motion")
        self.assertEqual(report["animation"]["target_paths"], ["translation"])
        self.assertGreaterEqual(report["animation"]["changed_samples"], 1)
        self.assertEqual(report["artifacts"]["missing"], [])
        self.assertEqual(report["metadata"]["retarget_status"], "success")

    def test_positive_gltf_bundle_uses_same_validation_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._bundle(Path(tmp), animated_suffix=".gltf")

            report = validate_bundle(bundle, animated_name="animated.gltf")

        self.assertTrue(report["passed"], report)
        self.assertEqual(report["animation"]["name"], "Kimodo Motion")

    def test_no_animations_fails_with_stable_code(self):
        document = self._base_document()
        document["animations"] = []

        report = validate_gltf_document(document)

        self.assertFalse(report["passed"])
        self.assertIn("no_animation_clips", self._codes(report))

    def test_static_noop_animation_fails_with_stable_code(self):
        document = self._animated_document(values=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])

        report = validate_gltf_document(document)

        self.assertFalse(report["passed"])
        self.assertIn("static_noop_animation", self._codes(report))

    def test_malformed_reference_fails_with_stable_code(self):
        document = self._animated_document()
        document["animations"][0]["channels"][0]["sampler"] = 99

        report = validate_gltf_document(document)

        self.assertFalse(report["passed"])
        self.assertIn("bad_animation_reference", self._codes(report))

    def test_malformed_accessor_fails_with_stable_code(self):
        document = self._animated_document()
        output_accessor = document["animations"][0]["samplers"][0]["output"]
        document["accessors"][output_accessor]["componentType"] = 5123

        report = validate_gltf_document(document)

        self.assertFalse(report["passed"])
        self.assertIn("malformed_accessor", self._codes(report))

    def test_external_buffer_uri_fails_closed_with_stable_code(self):
        document = self._animated_document()
        for buffer in document["buffers"]:
            buffer["uri"] = "external-motion-buffer.bin"

        report = validate_gltf_document(document)

        self.assertFalse(report["passed"])
        self.assertIn("malformed_accessor", self._codes(report))
        self.assertTrue(
            any("external buffer URIs are unsupported" in error["message"] for error in report["errors"]),
            report,
        )

    def test_invalid_glb_header_and_chunk_fail_with_stable_code(self):
        cases = {
            "bad_header": b"not-a-glb",
            "chunk_exceeds_bounds": struct.pack("<4sIIII", b"glTF", 2, 20, 16, 0x4E4F534A),
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    bundle = self._bundle(Path(tmp), animated_suffix=".glb")
                    (bundle / "animated.glb").write_bytes(payload)

                    report = validate_bundle(bundle)

                self.assertFalse(report["passed"])
                self.assertIn("invalid_glb_structure", self._codes(report))

    def test_missing_metadata_artifact_fails_bundle_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._bundle(Path(tmp), animated_suffix=".glb")
            (bundle / "metadata.json").unlink()

            report = validate_bundle(bundle)

        self.assertFalse(report["passed"])
        self.assertIn("missing_artifact", self._codes(report))

    def test_metadata_mismatch_fails_bundle_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._bundle(Path(tmp), animated_suffix=".glb")
            metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
            metadata["retarget_status"] = "failed"
            (bundle / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            report = validate_bundle(bundle)

        self.assertFalse(report["passed"])
        self.assertIn("metadata_mismatch", self._codes(report))

    def test_cli_writes_json_report_and_returns_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._bundle(Path(tmp), animated_suffix=".glb")
            output = Path(tmp) / "report.json"

            exit_code = main(["--bundle", str(bundle), "--json-report", str(output)])

            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertTrue(report["passed"])

    def _bundle(self, root: Path, *, animated_suffix: str) -> Path:
        bundle = root / "bundle"
        bundle.mkdir()
        for name, payload in {
            "motion.npz": b"npz",
            "motion.bvh": b"bvh",
            "preview.glb": b"preview",
        }.items():
            (bundle / name).write_bytes(payload)
        rig_path = bundle / "tiny_rigged.gltf"
        shutil.copyfile(FIXTURES / "tiny_rigged.gltf", rig_path)
        rig = inspect_rig(rig_path, source_kind="load_mesh_existing")
        clip = RetargetClip(
            name="Kimodo Motion",
            duration=1.0,
            frame_time=1.0,
            root_translations=((0.0, 0.0, 0.0), (0.25, 0.0, 0.0)),
            channels=(RetargetChannel(target_joint="Hips", path="translation", times=(0.0, 1.0), values=((0.0, 0.0, 0.0), (0.25, 0.0, 0.0))),),
            mapping=JointMapping(confidence="compatible", pairs={"Hips": "Hips"}, warnings=()),
            warnings=("root_motion_mode_preserve_translation",),
        )
        animated_name = f"animated{animated_suffix}"
        GlbAnimationExporter().write_animation(rig, clip, bundle / animated_name)
        artifacts = ["animated.glb", "metadata.json", "motion.bvh", "motion.npz", "preview.glb"]
        if animated_name != "animated.glb":
            artifacts = [animated_name if item == "animated.glb" else item for item in artifacts]
        (bundle / "metadata.json").write_text(
            json.dumps({
                "retarget_status": "success",
                "source_rigged_mesh": str(rig_path),
                "source_kind": "load_mesh_existing",
                "classification": "humanoid",
                "mapping_confidence": "compatible",
                "warnings": ["root_motion_mode_preserve_translation"],
                "artifacts": artifacts,
            }),
            encoding="utf-8",
        )
        return bundle

    def _animated_document(self, *, values=None):
        document = self._base_document()
        values = values or [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        times_payload = struct.pack("<2f", 0.0, 1.0)
        values_payload = struct.pack("<6f", *(component for value in values for component in value))
        time_accessor = self._append_buffer(document, times_payload, "SCALAR", 2)
        output_accessor = self._append_buffer(document, values_payload, "VEC3", 2)
        document["animations"] = [{
            "name": "Kimodo Motion",
            "samplers": [{"input": time_accessor, "output": output_accessor, "interpolation": "LINEAR"}],
            "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
        }]
        return document

    def _base_document(self):
        document = copy.deepcopy(json.loads((FIXTURES / "tiny_rigged.gltf").read_text(encoding="utf-8")))
        document.setdefault("animations", [])
        return document

    def _append_buffer(self, document, payload, accessor_type, count):
        buffer_index = len(document["buffers"])
        document["buffers"].append({
            "byteLength": len(payload),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(payload).decode("ascii"),
        })
        buffer_view_index = len(document["bufferViews"])
        document["bufferViews"].append({"buffer": buffer_index, "byteOffset": 0, "byteLength": len(payload)})
        accessor_index = len(document["accessors"])
        document["accessors"].append({"bufferView": buffer_view_index, "componentType": 5126, "count": count, "type": accessor_type})
        return accessor_index

    def _codes(self, report):
        return {error["code"] for error in report["errors"]}


if __name__ == "__main__":
    unittest.main()
