from __future__ import annotations

import numpy as np

from tools.render_full_assembly import (
    _finger_profiles,
    _joint_values,
    _object_poses,
    _sample_indices,
)


def test_trajectory_sampling_preserves_motion_endpoints_and_holds():
    motion = np.column_stack((np.linspace(0.0, 1.0, 22), np.zeros(22)))
    selected = _sample_indices(motion, motion_frames=6, hold_frames=12)
    assert len(selected) == 6
    assert selected[0] == 0
    assert selected[-1] == 21

    hold = np.repeat([[0.25, -0.5]], 18, axis=0)
    selected = _sample_indices(hold, motion_frames=6, hold_frames=12)
    assert len(selected) == 12
    assert selected[0] == 0
    assert selected[-1] == 17


def test_render_uses_current_dex3_open_close_profiles():
    profiles = _finger_profiles()
    values = _joint_values(
        ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"],
        np.array([0.1, -0.2]),
        {"left": 0.5, "right": 1.0},
        profiles,
    )
    assert values["left_shoulder_pitch_joint"] == 0.1
    assert values["right_shoulder_pitch_joint"] == -0.2
    assert np.isclose(
        values["left_hand_middle_0_joint"],
        0.5 * profiles["left"]["close"]["left_hand_middle_0_joint"],
    )
    assert values["right_hand_middle_0_joint"] == profiles["right"]["close"][
        "right_hand_middle_0_joint"
    ]


def test_attached_object_pose_follows_the_saved_grasp_frame_contract():
    world_T_grasp = np.eye(4)
    world_T_grasp[:3, 3] = [0.4, -0.1, 0.9]
    grasp_T_object = np.eye(4)
    grasp_T_object[:3, 3] = [0.02, 0.03, -0.04]
    poses = _object_poses(
        {
            "part": {
                "world_T_object": None,
                "hand": "right",
                "grasp_T_object": grasp_T_object.tolist(),
            }
        },
        {"right_hand_grasp_frame": world_T_grasp},
    )
    assert np.allclose(poses["part"], world_T_grasp @ grasp_T_object)
