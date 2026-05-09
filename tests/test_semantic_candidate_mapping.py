import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generator  # noqa: E402
from generator import KimodoRuntimeError  # noqa: E402
from kimodo_bvh import parse_bvh  # noqa: E402
from retarget_errors import MAPPING_INCOMPATIBLE, RetargetError  # noqa: E402
from retarget_metadata import retarget_success_metadata  # noqa: E402
from retargeting import build_joint_mapping, build_retarget_clip  # noqa: E402
from rig_inspector import inspect_rig  # noqa: E402


FIXTURES = ROOT / "tests" / "fixtures"


class SemanticCandidateMappingTests(unittest.TestCase):
    def test_core_semantic_candidates_build_basic_mapping_and_report_candidate_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "candidate_core_only.gltf"
            self._write_candidate_rig(rig_path, ["candidate_hips", "candidate_spine", "candidate_head"])
            self._write_semantic_candidate_sidecar(
                rig_path,
                roles={
                    "hips": self._candidate("hips", 0, "candidate_hips", 0.91),
                    "spine": self._candidate("spine", 1, "candidate_spine", 0.89),
                    "head": self._candidate("head", 2, "candidate_head", 0.87),
                },
            )
            motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")
            rig = inspect_rig(rig_path)

        mapping = build_joint_mapping(motion, rig)
        clip = build_retarget_clip(motion, rig)
        metadata = retarget_success_metadata(Path("animated.glb"), rig=rig, clip=clip, warnings=clip.warnings)

        self.assertIn("Hips", mapping.pairs)
        self.assertEqual(mapping.pairs["Hips"], "candidate_hips")
        self.assertEqual(mapping.pairs["Spine"], "candidate_spine")
        self.assertEqual(mapping.pairs["Head"], "candidate_head")
        self.assertIn(mapping.confidence, {"semantic_candidate", "best_effort"})
        self.assertIn(metadata["animation_mapping_status"], {"semantic_candidate", "best_effort"})
        self.assertEqual(metadata["contract_status"], "contract_missing_or_untrusted")
        self.assertEqual(metadata["stabilization_status"], "not_evaluated")

    def test_partial_limb_candidates_degrade_with_coverage_warnings_not_mapping_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "candidate_partial_limbs.gltf"
            self._write_candidate_rig(
                rig_path,
                [
                    "sem_hips",
                    "sem_spine",
                    "sem_head",
                    "sem_left_upper_arm",
                    "sem_right_upper_leg",
                ],
            )
            self._write_semantic_candidate_sidecar(
                rig_path,
                roles={
                    "hips": self._candidate("hips", 0, "sem_hips", 0.92),
                    "spine": self._candidate("spine", 1, "sem_spine", 0.90),
                    "head": self._candidate("head", 2, "sem_head", 0.88),
                    "left_upper_arm": self._candidate("left_upper_arm", 3, "sem_left_upper_arm", 0.74),
                    "right_upper_leg": self._candidate("right_upper_leg", 4, "sem_right_upper_leg", 0.72),
                },
            )
            motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")
            rig = inspect_rig(rig_path)

        mapping = build_joint_mapping(motion, rig)

        self.assertEqual(mapping.pairs["Hips"], "sem_hips")
        self.assertEqual(mapping.pairs["Spine"], "sem_spine")
        self.assertEqual(mapping.pairs["Head"], "sem_head")
        self.assertEqual(mapping.pairs["LeftArm"], "sem_left_upper_arm")
        self.assertEqual(mapping.pairs["RightUpLeg"], "sem_right_upper_leg")
        self.assertIn("semantic_candidate_mapping", mapping.warnings)
        self.assertIn("semantic_candidate_missing_limb_role:left_upper_leg", mapping.warnings)
        self.assertIn("semantic_candidate_missing_limb_role:right_upper_arm", mapping.warnings)
        self.assertEqual(mapping.coverage.get("semantic_candidate_core"), "3/3")
        self.assertEqual(mapping.coverage.get("semantic_candidate_limb_groups"), "2/4")

    def test_semantic_candidate_mapping_does_not_cross_trust_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "candidate_untrusted_boundary.gltf"
            self._write_candidate_rig(rig_path, ["sc_hips", "sc_spine", "sc_head"])
            self._write_semantic_candidate_sidecar(
                rig_path,
                roles={
                    "hips": self._candidate("hips", 0, "sc_hips", 0.91),
                    "spine": self._candidate("spine", 1, "sc_spine", 0.89),
                    "head": self._candidate("head", 2, "sc_head", 0.87),
                },
            )
            motion = parse_bvh(FIXTURES / "synthetic_walk.bvh")
            rig = inspect_rig(rig_path)

        clip = build_retarget_clip(motion, rig)
        metadata = retarget_success_metadata(Path("animated.glb"), rig=rig, clip=clip, warnings=clip.warnings)

        self.assertEqual(rig.contract_status, "contract_missing_or_untrusted")
        self.assertEqual(rig.humanoid_roles, ())
        self.assertEqual(rig.humanoid_chains, ())
        self.assertNotIn("retarget_calibration", metadata["diagnostics"])
        self.assertEqual(metadata["contract_status"], "contract_missing_or_untrusted")
        self.assertEqual(metadata["animation_mapping_status"], "semantic_candidate")
        self.assertIn(metadata["stabilization_status"], {"not_evaluated", "ineligible"})

    def test_insufficient_semantic_candidates_use_preview_fallback_when_preview_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            rig_path = bundle / "candidate_no_core.gltf"
            metadata_path = bundle / "metadata.json"
            self._write_motion_bundle(bundle)
            (bundle / "preview.glb").write_bytes(b"preview")
            self._write_candidate_rig(rig_path, ["only_hips", "only_spine", "accessory"])
            self._write_semantic_candidate_sidecar(
                rig_path,
                roles={
                    "hips": self._candidate("hips", 0, "only_hips", 0.93),
                    "spine": self._candidate("spine", 1, "only_spine", 0.91),
                    "left_hand": self._candidate("left_hand", 2, "accessory", 0.70),
                },
                diagnostics=["insufficient_core_candidates"],
            )
            request = {"rigged_mesh_path": rig_path, "rigged_mesh_source_kind": "unknown_mesh_source", "prompt": "Walk forward"}

            output = generator.retarget_motion_bundle(bundle, metadata_path, request, {"output_stem_name": "motion", "num_samples": 1})
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(output.name, "preview.glb")
        self.assertEqual(metadata["runtime_status"], "success_with_fallback")
        self.assertEqual(metadata["retarget_error_code"], MAPPING_INCOMPATIBLE)
        self.assertEqual(metadata["animation_mapping_status"], "failed")
        self.assertEqual(metadata["fallback_output"], "preview.glb")

    def test_insufficient_semantic_candidates_without_safe_output_remain_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            rig_path = bundle / "candidate_no_core_no_preview.gltf"
            metadata_path = bundle / "metadata.json"
            self._write_motion_bundle(bundle)
            self._write_candidate_rig(rig_path, ["only_hips", "only_spine", "accessory"])
            self._write_semantic_candidate_sidecar(
                rig_path,
                roles={
                    "hips": self._candidate("hips", 0, "only_hips", 0.93),
                    "spine": self._candidate("spine", 1, "only_spine", 0.91),
                    "left_hand": self._candidate("left_hand", 2, "accessory", 0.70),
                },
                diagnostics=["insufficient_core_candidates"],
            )
            request = {"rigged_mesh_path": rig_path, "rigged_mesh_source_kind": "unknown_mesh_source", "prompt": "Walk forward"}

            with self.assertRaises(KimodoRuntimeError) as ctx:
                generator.retarget_motion_bundle(bundle, metadata_path, request, {"output_stem_name": "motion", "num_samples": 1})
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(ctx.exception.code, MAPPING_INCOMPATIBLE)
        self.assertEqual(metadata["retarget_status"], "failed")
        self.assertEqual(metadata["retarget_error_code"], MAPPING_INCOMPATIBLE)
        self.assertNotIn("fallback_output", metadata)

    def test_trusted_contract_mapping_is_preferred_when_semantic_candidates_also_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "trusted_contract_with_candidates.gltf"
            bvh_path = Path(tmp) / "full_contract_motion.bvh"
            self._write_trusted_contract_rig(rig_path)
            self._write_full_contract_bvh(bvh_path)
            self._write_trusted_contract_and_semantic_sidecar(rig_path)
            motion = parse_bvh(bvh_path)
            rig = inspect_rig(rig_path)

        mapping = build_joint_mapping(motion, rig)
        clip = build_retarget_clip(motion, rig)
        metadata = retarget_success_metadata(Path("animated.glb"), rig=rig, clip=clip, warnings=clip.warnings)

        self.assertEqual(rig.contract_status, "trusted")
        self.assertEqual(mapping.confidence, "calibrated")
        self.assertEqual(mapping.pairs["Hips"], "contract_hips")
        self.assertEqual(metadata["contract_status"], "trusted")
        self.assertEqual(metadata["animation_mapping_status"], "trusted_contract")
        self.assertNotEqual(mapping.pairs["Hips"], "candidate_hips")

    def _write_motion_bundle(self, bundle: Path) -> None:
        (bundle / "metadata.json").write_text(json.dumps({"canonical_motion_artifact": "motion.npz"}), encoding="utf-8")
        (bundle / "motion.npz").write_bytes(b"npz")
        (bundle / "motion.bvh").write_text((FIXTURES / "synthetic_walk.bvh").read_text(encoding="utf-8"), encoding="utf-8")

    def _write_candidate_rig(self, path: Path, joint_names: list[str]) -> None:
        nodes = []
        for index, name in enumerate(joint_names):
            node = {"name": name}
            if index + 1 < len(joint_names):
                node["children"] = [index + 1]
            nodes.append(node)
        path.write_text(
            json.dumps({"asset": {"version": "2.0"}, "nodes": nodes, "skins": [{"joints": list(range(len(nodes))), "inverseBindMatrices": 0}]}),
            encoding="utf-8",
        )

    def _write_semantic_candidate_sidecar(self, path: Path, *, roles: dict[str, dict], diagnostics: list[str] | None = None) -> None:
        payload = {
            "metadata_mode": "auto",
            "semantic_candidates": {
                "schema": "unirig.semantic_candidates.v1",
                "trust": {"trusted": False, "stabilization_eligible": False},
                "status": "candidate",
                "roles": roles,
                "chains": {"torso": ["hips", "spine", "head"]},
                "diagnostics": diagnostics or [],
            },
        }
        path.with_suffix(".rigmeta.json").write_text(json.dumps(payload), encoding="utf-8")

    def _candidate(self, role: str, node_index: int, node_name: str, confidence: float) -> dict:
        return {
            "role": role,
            "node_id": node_name,
            "node_index": node_index,
            "node_name": node_name,
            "confidence": confidence,
            "reasons": ["synthetic_semantic_mapping_fixture"],
            "source": "semantic_humanoid_resolver",
            "unsafe_flags": [],
            "rejected": False,
        }

    def _contract_roles(self) -> list[str]:
        return [
            "hips",
            "spine",
            "chest",
            "neck",
            "head",
            "left_upper_arm",
            "left_forearm",
            "left_hand",
            "right_upper_arm",
            "right_forearm",
            "right_hand",
            "left_upper_leg",
            "left_lower_leg",
            "left_foot",
            "right_upper_leg",
            "right_lower_leg",
            "right_foot",
        ]

    def _write_trusted_contract_rig(self, path: Path) -> None:
        nodes = []
        for index, role in enumerate(self._contract_roles()):
            node = {"name": f"contract_{role}"}
            if index + 1 < len(self._contract_roles()):
                node["children"] = [index + 1]
            nodes.append(node)
        nodes.extend({"name": name} for name in ("candidate_hips", "candidate_spine", "candidate_head"))
        path.write_text(
            json.dumps({"asset": {"version": "2.0"}, "nodes": nodes, "skins": [{"joints": list(range(len(nodes))), "inverseBindMatrices": 0}]}),
            encoding="utf-8",
        )

    def _write_trusted_contract_and_semantic_sidecar(self, path: Path) -> None:
        roles = self._contract_roles()
        contract = {
            "schema": "modly.humanoid.v1",
            "roles": {role: {"node_index": index, "node_name": f"contract_{role}"} for index, role in enumerate(roles)},
            "chains": {
                "torso": ["hips", "spine", "chest", "neck", "head"],
                "left_arm": ["left_upper_arm", "left_forearm", "left_hand"],
                "right_arm": ["right_upper_arm", "right_forearm", "right_hand"],
                "left_leg": ["left_upper_leg", "left_lower_leg", "left_foot"],
                "right_leg": ["right_upper_leg", "right_lower_leg", "right_foot"],
            },
            "basis_hints": {"up_axis": "Y"},
            "rest": {"source": "synthetic"},
        }
        payload = {
            "humanoid_contract": contract,
            "semantic_candidates": {
                "schema": "unirig.semantic_candidates.v1",
                "trust": {"trusted": False, "stabilization_eligible": False},
                "status": "candidate",
                "roles": {
                    "hips": self._candidate("hips", 17, "candidate_hips", 0.95),
                    "spine": self._candidate("spine", 18, "candidate_spine", 0.94),
                    "head": self._candidate("head", 19, "candidate_head", 0.93),
                },
                "chains": {"torso": ["hips", "spine", "head"]},
                "diagnostics": [],
            },
        }
        path.with_suffix(".rigmeta.json").write_text(json.dumps(payload), encoding="utf-8")

    def _write_full_contract_bvh(self, path: Path) -> None:
        hierarchy = """HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0.0 1.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT Chest
    {
      OFFSET 0.0 1.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT Neck
      {
        OFFSET 0.0 1.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Head
        {
          OFFSET 0.0 0.5 0.0
          CHANNELS 3 Zrotation Xrotation Yrotation
        }
      }
      JOINT LeftUpperArm
      {
        OFFSET -1.0 0.5 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftForeArm
        {
          OFFSET 0.0 1.0 0.0
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT LeftHand
          {
            OFFSET 0.0 0.5 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
          }
        }
      }
      JOINT RightUpperArm
      {
        OFFSET 1.0 0.5 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightForeArm
        {
          OFFSET 0.0 1.0 0.0
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT RightHand
          {
            OFFSET 0.0 0.5 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
          }
        }
      }
    }
  }
  JOINT LeftUpperLeg
  {
    OFFSET -0.5 -1.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftLowerLeg
    {
      OFFSET 0.0 -1.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT LeftFoot
      {
        OFFSET 0.0 -0.5 0.5
        CHANNELS 3 Zrotation Xrotation Yrotation
      }
    }
  }
  JOINT RightUpperLeg
  {
    OFFSET 0.5 -1.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT RightLowerLeg
    {
      OFFSET 0.0 -1.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT RightFoot
      {
        OFFSET 0.0 -0.5 0.5
        CHANNELS 3 Zrotation Xrotation Yrotation
      }
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.0333333
"""
        frame = " ".join("0.0" for _ in range(54))
        path.write_text(hierarchy + frame + "\n" + frame + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
