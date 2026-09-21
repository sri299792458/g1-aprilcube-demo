from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "artifacts/grasp_atlas/cube40_viral_v1/right/arm_grasp_pool.yaml"
H50 = (
    ROOT
    / "artifacts/grasp_shortlists/cube_tripod_right_v1/h50/shortlist.yaml"
)


def test_h50_shortlist_has_candidate_specific_isaac_closure_states() -> None:
    pool = yaml.safe_load(POOL.read_text())
    shortlist = yaml.safe_load(H50.read_text())

    assert pool["format"] == "g1_aprilcube_arm_grasp_pool"
    assert pool["candidate_count"] == len(pool["candidates"]) == 3178
    assert shortlist["candidate_count"] == len(shortlist["candidates"]) == 372
    assert pool["source"]["isaac_closed_q_policy"] == (
        "exact_simulated_joint_state_at_closed_before_tug"
    )

    pool_by_id = {item["candidate_id"]: item for item in pool["candidates"]}
    assert len(pool_by_id) == pool["candidate_count"]
    expected_joints = {
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
    }
    for candidate in shortlist["candidates"]:
        assert set(pool_by_id[candidate["candidate_id"]]["isaac_closed_q"]) == (
            expected_joints
        )
