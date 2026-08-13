from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


END2END = Path(__file__).resolve().parents[1] / "third_party" / "GraspGenX" / "end2end"
if str(END2END) not in sys.path:
    sys.path.insert(0, str(END2END))

from placement_goals import generate_inside_goals, proper_axis_rotations  # noqa: E402
from registry import _open_bin_primitives  # noqa: E402


def _cube_vertices(side: float) -> np.ndarray:
    half = side / 2.0
    return np.asarray(
        [
            [x, y, z]
            for x in (-half, half)
            for y in (-half, half)
            for z in (-half, half)
        ],
        dtype=np.float64,
    )


def test_axis_aligned_rotations_are_24_unique_proper_rotations():
    rotations = proper_axis_rotations()
    assert len(rotations) == 24
    assert np.allclose(rotations[0], np.eye(3))
    assert len({tuple(rotation.reshape(-1)) for rotation in rotations}) == 24
    for rotation in rotations:
        assert np.allclose(rotation.T @ rotation, np.eye(3))
        assert np.isclose(np.linalg.det(rotation), 1.0)


def test_inside_goal_preserves_selected_object_to_tool_transform():
    object_T_tool = np.eye(4)
    object_T_tool[:3, 3] = [0.06, -0.02, 0.01]
    target_world_T = np.eye(4)
    target_world_T[:3, 3] = [0.48, -0.25, 0.69]
    params = {
        "width": 0.22,
        "depth": 0.22,
        "height": 0.10,
        "thickness": 0.015,
        "angle": 0.15,
    }
    goals = generate_inside_goals(
        object_vertices=_cube_vertices(0.045),
        initial_world_T_object=np.eye(4),
        object_T_tool=object_T_tool,
        target_world_T=target_world_T,
        target_params=params,
        config={
            "orientation_policy": "free_axis_aligned",
            "max_candidates": 24,
            "containment_margin_m": 0.01,
            "release_clearance_above_rim_m": 0.04,
            "pre_place_extra_clearance_m": 0.08,
        },
    )

    assert len(goals) == 24
    for goal in goals:
        recovered = np.linalg.inv(goal.world_T_object) @ goal.world_T_tool
        assert np.allclose(recovered, object_T_tool)
        pre_recovered = (
            np.linalg.inv(goal.pre_place_world_T_object)
            @ goal.pre_place_world_T_tool
        )
        assert np.allclose(pre_recovered, object_T_tool)
        # Target +Z is world +Z in this fixture: release is 4 cm above
        # the 10 cm rim and pre-place is another 8 cm higher.
        release_vertices = (
            np.column_stack((_cube_vertices(0.045), np.ones(8)))
            @ goal.world_T_object.T
        )[:, :3]
        assert np.isclose(release_vertices[:, 2].min(), 0.69 + 0.10 + 0.04)
        assert np.isclose(
            goal.pre_place_world_T_object[2, 3]
            - goal.world_T_object[2, 3],
            0.08,
        )


def test_open_bin_collision_is_floor_plus_four_walls():
    obstacles = _open_bin_primitives(
        "bin",
        {
            "translation": [0.48, -0.25, 0.69],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        {
            "width": 0.22,
            "depth": 0.22,
            "height": 0.10,
            "thickness": 0.015,
            "angle": 0.15,
        },
    )
    assert [obstacle.name for obstacle in obstacles] == [
        "bin_floor",
        "bin_wall_pos_x",
        "bin_wall_neg_x",
        "bin_wall_pos_y",
        "bin_wall_neg_y",
    ]
    assert all(obstacle.type == "cuboid" for obstacle in obstacles)
    assert obstacles[0].dims == [0.22, 0.22, 0.015]
    # The floor is 7.5 mm above the bin origin; no solid body occupies the
    # center all the way to the rim as the old AABB representation did.
    assert np.allclose(obstacles[0].pose[:3], [0.48, -0.25, 0.6975])
