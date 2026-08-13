from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import trimesh.transformations as tra


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_supported_arm_grasp_pool import build  # noqa: E402


def test_supported_pool_copies_original_goal_pose_and_only_pass_records():
    report_path = (
        ROOT
        / "artifacts/grasp_support/u_legs_right_upright_replay1_isaac_v1/report.json"
    )
    mesh_path = ROOT / "generated/aprilcube_parts/u_legs/grasp_mesh.obj"
    report = json.loads(report_path.read_text())
    pool = build(
        report_path,
        mesh_path,
        "right_upright_test",
        "right",
    )

    assert pool["candidate_count"] == report["pass_count"] == 365
    assert pool["family_count"] == 14
    assert len({item["candidate_id"] for item in pool["candidates"]}) == 365
    assert len(
        {item["family_id"] for item in pool["candidates"][: pool["family_count"]]}
    ) == pool["family_count"]

    by_id = {record["candidate_id"]: record for record in report["records"]}
    candidate = pool["candidates"][0]
    pose = candidate["object_T_G"]
    actual = tra.quaternion_matrix(
        [pose["orientation"]["w"], *pose["orientation"]["xyz"]]
    )
    actual[:3, 3] = pose["position"]
    source = by_id[candidate["candidate_id"]]
    assert source["result"]["passed"]
    assert np.allclose(actual, source["input"]["object_T_G"])
    assert not np.allclose(actual, source["result"]["final_object_T_G"])
