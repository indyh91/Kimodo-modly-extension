import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rig_inspector import inspect_rig  # noqa: E402


class SemanticCandidateIngestionTests(unittest.TestCase):
    def test_sidecar_semantic_candidates_parse_separately_from_trusted_humanoid_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "semantic_candidate_rig.gltf"
            self._write_candidate_rig(rig_path)
            self._write_semantic_candidate_sidecar(
                rig_path,
                roles={
                    "hips": self._candidate("hips", 0, 0.91),
                    "spine": self._candidate("spine", 1, 0.88),
                    "head": self._candidate("head", 2, 0.86),
                    "left_hand": self._candidate("left_hand", 3, 0.70),
                },
            )

            rig = inspect_rig(rig_path)

        candidates = {binding.role: binding for binding in rig.semantic_candidate_bindings}
        self.assertEqual(rig.contract_status, "contract_missing_or_untrusted")
        self.assertEqual(rig.humanoid_roles, ())
        self.assertEqual(rig.humanoid_chains, ())
        self.assertEqual(set(candidates), {"hips", "spine", "head", "left_hand"})
        self.assertEqual(candidates["hips"].node_name, "bone_0")
        self.assertEqual(candidates["hips"].node_id, "bone_0")
        self.assertAlmostEqual(candidates["hips"].confidence, 0.91)
        self.assertIn("semantic_candidates_validated", rig.semantic_candidate_diagnostics)
        self.assertIn("torso", {chain.name for chain in rig.semantic_candidate_chains})

    def test_insufficient_core_candidates_preserve_diagnostics_without_trust_or_stabilization_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "semantic_candidate_no_core.gltf"
            self._write_candidate_rig(rig_path)
            self._write_semantic_candidate_sidecar(
                rig_path,
                roles={
                    "hips": self._candidate("hips", 0, 0.92),
                    "spine": self._candidate("spine", 1, 0.90),
                    "head": self._candidate("head", 2, 0.89, unsafe_flags=["passive_or_accessory_core"]),
                    "left_hand": self._candidate("left_hand", 3, 0.45),
                },
                status="blocked",
                diagnostics=["insufficient_core_candidates", "semantic_candidate_core_unsafe:head"],
            )

            rig = inspect_rig(rig_path)

        self.assertEqual(rig.contract_status, "contract_missing_or_untrusted")
        self.assertEqual(rig.humanoid_roles, ())
        self.assertEqual(rig.semantic_candidate_bindings, ())
        self.assertIn("insufficient_core_candidates", rig.semantic_candidate_diagnostics)
        self.assertIn("semantic_candidate_core_unsafe:head", rig.semantic_candidate_diagnostics)
        self.assertIn("semantic_candidates_core_incomplete", rig.semantic_candidate_diagnostics)

    def test_low_confidence_core_candidate_is_diagnostic_only_and_never_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig_path = Path(tmp) / "semantic_candidate_low_confidence.gltf"
            self._write_candidate_rig(rig_path)
            self._write_semantic_candidate_sidecar(
                rig_path,
                roles={
                    "hips": self._candidate("hips", 0, 0.91),
                    "spine": self._candidate("spine", 1, 0.50),
                    "head": self._candidate("head", 2, 0.86),
                },
            )

            rig = inspect_rig(rig_path)

        self.assertEqual(rig.contract_status, "contract_missing_or_untrusted")
        self.assertEqual(rig.humanoid_roles, ())
        self.assertEqual(rig.semantic_candidate_bindings, ())
        self.assertIn("semantic_candidate_low_confidence:spine", rig.semantic_candidate_diagnostics)
        self.assertIn("semantic_candidates_core_incomplete", rig.semantic_candidate_diagnostics)

    def _write_candidate_rig(self, path: Path) -> None:
        path.write_text(
            json.dumps({
                "asset": {"version": "2.0"},
                "nodes": [
                    {"name": "bone_0", "children": [1]},
                    {"name": "bone_1", "children": [2, 3]},
                    {"name": "bone_2"},
                    {"name": "bone_3"},
                ],
                "skins": [{"joints": [0, 1, 2, 3], "inverseBindMatrices": 0}],
            }),
            encoding="utf-8",
        )

    def _write_semantic_candidate_sidecar(
        self,
        path: Path,
        *,
        roles: dict[str, dict],
        status: str = "candidate",
        diagnostics: list[str] | None = None,
    ) -> None:
        payload = {
            "metadata_mode": "auto",
            "semantic_candidates": {
                "schema": "unirig.semantic_candidates.v1",
                "trust": {"trusted": False, "stabilization_eligible": False},
                "status": status,
                "roles": roles,
                "chains": {"torso": ["hips", "spine", "head"]},
                "diagnostics": diagnostics or [],
            },
        }
        path.with_suffix(".rigmeta.json").write_text(json.dumps(payload), encoding="utf-8")

    def _candidate(self, role: str, node_index: int, confidence: float, *, unsafe_flags: list[str] | None = None) -> dict:
        return {
            "role": role,
            "node_id": f"bone_{node_index}",
            "node_index": node_index,
            "node_name": f"bone_{node_index}",
            "confidence": confidence,
            "reasons": ["synthetic_semantic_fixture"],
            "source": "semantic_humanoid_resolver",
            "unsafe_flags": unsafe_flags or [],
            "rejected": False,
        }


if __name__ == "__main__":
    unittest.main()
