from __future__ import annotations

import numpy as np
import trimesh.transformations as tra

from g1_aprilcube_demo.planning.grasp_audit import (
    CandidateAudit,
    local_z_pregrasp,
    support_plane_audit,
)
from g1_aprilcube_demo.planning.grasp_goalset import GraspCandidate


def candidate(matrix: np.ndarray) -> GraspCandidate:
    return GraspCandidate("candidate", "family", 0.5, matrix)


def test_local_z_pregrasp_uses_candidate_orientation():
    grasp = tra.translation_matrix([1.0, 2.0, 3.0]) @ tra.rotation_matrix(
        np.pi / 2.0, [0.0, 1.0, 0.0]
    )
    pregrasp = local_z_pregrasp(grasp, -0.1)
    assert np.allclose(pregrasp[:3, :3], grasp[:3, :3])
    assert np.allclose(pregrasp[:3, 3], [0.9, 2.0, 3.0])


def test_support_plane_audit_preserves_pose_and_names_first_failed_gate():
    object_T_G = tra.translation_matrix([0.0, 0.0, 0.02])
    hand_vertices = np.asarray(
        [[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]], dtype=np.float64
    )
    item = candidate(object_T_G)
    original = item.object_T_G.copy()
    records = support_plane_audit(
        [item],
        np.eye(4),
        hand_vertices,
        support_z_m=0.0,
        approach_offset_m=-0.1,
    )
    assert records[0].final_support_clearance_m == 0.01
    assert np.isclose(records[0].pregrasp_support_clearance_m, -0.09)
    assert records[0].support_plane_clear is False
    assert records[0].failure_gate == "support_plane"
    assert np.array_equal(item.object_T_G, original)


def test_complete_pickup_is_authoritative_over_standalone_ik_diagnostic():
    identity = np.eye(4).tolist()
    record = CandidateAudit(
        "candidate",
        "family",
        0.5,
        identity,
        identity,
        0.01,
        0.01,
        True,
        final_endpoint_ik=True,
        pregrasp_endpoint_ik=False,
        pickup_plan=True,
    )
    record.finish()
    assert record.passed is True
    assert record.failure_gate is None
