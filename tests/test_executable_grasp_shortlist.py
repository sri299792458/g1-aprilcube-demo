from __future__ import annotations

import numpy as np

from g1_aprilcube_demo.grasping.executable_shortlist import (
    contact_group_force,
    relative_pose_change,
)


def test_relative_pose_change_is_frame_invariant_for_translation_and_rotation():
    # Use a nontrivial fixed hand frame. The closed trace stores
    # inverse(world_T_object) @ world_T_G, not world_T_G itself.
    initial_object_T_G = np.eye(4)
    initial_object_T_G[:3, 3] = [0.1, -0.2, 0.3]
    angle = np.deg2rad(30.0)
    world_T_closed_object = np.eye(4)
    world_T_closed_object[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    world_T_closed_object[:3, 3] = [0.003, 0.004, 0.0]
    closed_object_T_G = (
        np.linalg.inv(world_T_closed_object) @ initial_object_T_G
    )
    translation, rotation = relative_pose_change(
        initial_object_T_G,
        closed_object_T_G,
    )
    assert np.isclose(translation, 0.005)
    assert np.isclose(rotation, 30.0)


def test_contact_group_force_uses_body_level_max_without_vector_cancellation():
    contacts = [
        {"hand_link": "thumb", "contact_force_magnitude_N": 1.2},
        {"hand_link": "thumb", "contact_force_magnitude_N": 2.4},
        {"hand_link": "index", "contact_force_magnitude_N": 3.1},
    ]
    assert contact_group_force(contacts, {"thumb"}) == 2.4
    assert contact_group_force(contacts, {"missing"}) == 0.0
