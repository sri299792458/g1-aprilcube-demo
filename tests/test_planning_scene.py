from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
import yourdfpy

from g1_aprilcube_demo.planning import load_planning_scene


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "config/planning/unibot_seated_aprilcube_v1.yaml"


def test_scene_uses_complete_observed_g1_state_and_named_table_reference():
    scene = load_planning_scene(SCENE)
    assert scene.planning_frame == "base_link"
    assert len(scene.start_joint_positions) == 43
    assert np.isclose(scene.table_top_z_m, 0.00652595, atol=5e-9)
    assert scene.start_joint_positions["waist_pitch_joint"] == 0.1707620471715927
    assert scene.start_joint_positions["right_shoulder_pitch_joint"] == -0.33483925461769104
    assert all(
        scene.start_joint_positions[name] == 0.0
        for name in scene.start_joint_positions
        if name.startswith(("right_hand_", "left_hand_"))
    )


def test_scattered_parts_rest_on_table_without_overlapping():
    scene = load_planning_scene(SCENE)
    transforms = scene.world_part_transforms()
    bounds = {}
    for part_id, part in scene.parts.items():
        mesh = trimesh.load(part.mesh, force="mesh", process=False)
        vertices = trimesh.transform_points(mesh.vertices, transforms[part_id])
        bounds[part_id] = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
        assert np.isclose(vertices[:, 2].min(), scene.table_top_z_m, atol=1e-10)

    for index, first in enumerate(sorted(bounds)):
        for second in sorted(bounds)[index + 1 :]:
            overlap = np.minimum(bounds[first][1], bounds[second][1]) - np.maximum(
                bounds[first][0], bounds[second][0]
            )
            assert not np.all(overlap > 0.0), f"{first} overlaps {second}"


def test_current_dex3_and_loose_parts_clear_the_planning_scene():
    scene = load_planning_scene(SCENE)
    robot = yourdfpy.URDF.load(
        str(scene.urdf),
        build_scene_graph=True,
        load_meshes=False,
        build_collision_scene_graph=True,
        load_collision_meshes=True,
    )
    robot.update_cfg(dict(scene.start_joint_positions))
    robot_collision = trimesh.collision.CollisionManager()
    for name, mesh in robot.collision_scene.geometry.items():
        transform = robot.collision_scene.graph.get(
            frame_from=robot.collision_scene.graph.base_frame, frame_to=name
        )[0]
        robot_collision.add_object(name, mesh, transform)

    table = trimesh.creation.box(
        extents=[*scene.table_size_xy_m, scene.table_thickness_m]
    )
    table_transform = trimesh.transformations.translation_matrix(
        [
            *scene.table_center_xy_m,
            scene.table_top_z_m - 0.5 * scene.table_thickness_m,
        ]
    )
    assert not robot_collision.in_collision_single(table, table_transform)
    assert robot_collision.min_distance_single(table, table_transform) >= (
        scene.minimum_robot_clearance_m
    )

    for part_id, part in scene.parts.items():
        mesh = trimesh.load(part.mesh, force="mesh", process=False)
        assert not robot_collision.in_collision_single(
            mesh, scene.world_part_transforms()[part_id]
        ), part_id
