from __future__ import annotations

from pathlib import Path

import numpy as np

from g1_aprilcube_demo.planning.grasp_goalset import (
    GraspCandidate,
    progressive_candidate_rounds,
)
from g1_aprilcube_demo.planning.task_grasp import (
    annotate_candidate,
    load_semantic_voxels,
)


ROOT = Path(__file__).resolve().parents[1]


def candidate(
    candidate_id: str,
    family_id: str,
    position: tuple[float, float, float],
    direction: tuple[float, float, float],
    role: str | None = None,
) -> GraspCandidate:
    matrix = np.eye(4)
    matrix[:3, 3] = position
    matrix[:3, 2] = direction
    # The tests use cardinal directions, so construct an orthogonal basis for
    # the two cases below without introducing a rotation helper dependency.
    if np.allclose(direction, [0.0, 0.0, -1.0]):
        matrix[:3, 0] = [1.0, 0.0, 0.0]
        matrix[:3, 1] = [0.0, -1.0, 0.0]
    elif np.allclose(direction, [1.0, 0.0, 0.0]):
        matrix[:3, 0] = [0.0, 1.0, 0.0]
        matrix[:3, 1] = [0.0, 0.0, 1.0]
    return GraspCandidate(candidate_id, family_id, 0.0, matrix, role)


def test_owned_voxels_distinguish_u_legs_from_bridge():
    voxels = load_semantic_voxels(
        ROOT / "config/aprilcube_parts/u_legs.yaml"
    )
    assert len(voxels) == 7
    owners = {voxel.index: voxel.component for voxel in voxels}
    assert owners[(0, 0, 0)] == "left_leg"
    assert owners[(2, 0, 0)] == "right_leg"
    # Later cuboid ownership makes the shared top row unambiguously bridge.
    assert owners[(0, 0, 2)] == "hip_bridge"
    assert owners[(2, 0, 2)] == "hip_bridge"


def test_approach_annotation_hits_named_u_regions_and_enforces_allowed_set():
    voxels = load_semantic_voxels(
        ROOT / "config/aprilcube_parts/u_legs.yaml"
    )
    left_leg = annotate_candidate(
        candidate("left", "f", [-0.18, 0.0, -0.045], [1.0, 0.0, 0.0]),
        voxels,
    )
    bridge = annotate_candidate(
        candidate("bridge", "f", [0.0, 0.0, 0.18], [0.0, 0.0, -1.0]),
        voxels,
    )
    assert left_leg.target_components == ("left_leg",)
    assert left_leg.allowed_by({"left_leg", "right_leg"})
    assert bridge.target_components == ("hip_bridge",)
    assert not bridge.allowed_by({"left_leg", "right_leg"})


def test_progressive_expansion_preserves_every_member_once():
    values = [
        candidate("a0", "a", [0, 0, 0], [1, 0, 0], "primary"),
        candidate(
            "b0",
            "b",
            [0, 0, 0],
            [1, 0, 0],
            "primary",
        ),
        candidate(
            "a1",
            "a",
            [0, 0, 0],
            [1, 0, 0],
            "translation_diverse_backup",
        ),
        candidate(
            "b1",
            "b",
            [0, 0, 0],
            [1, 0, 0],
            "pose_diverse_backup",
        ),
        candidate("a2", "a", [0, 0, 0], [1, 0, 0]),
        candidate("b2", "b", [0, 0, 0], [1, 0, 0]),
        candidate("a3", "a", [0, 0, 0], [1, 0, 0]),
    ]
    rounds = progressive_candidate_rounds(values)
    assert [
        (item.round_id, [candidate.candidate_id for candidate in item.candidates])
        for item in rounds
    ] == [
        ("primary", ["a0", "b0"]),
        ("translation_backup", ["a1"]),
        ("pose_backup", ["b1"]),
        ("remaining_000", ["a2", "b2"]),
        ("remaining_001", ["a3"]),
    ]
    flattened = [
        candidate.candidate_id
        for item in rounds
        for candidate in item.candidates
    ]
    assert len(flattened) == len(set(flattened)) == len(values)
