from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh.transformations as tra
import yaml

from g1_aprilcube_demo.planning.grasp_goalset import load_grasp_pool, world_grasps
from g1_aprilcube_demo.planning.curobo_backend import CuroboBackend
from g1_aprilcube_demo.planning.workspace import placement_samples, workspace_samples
from g1_aprilcube_demo.runtime import RuntimeObservationError, load_observation


ROOT = Path(__file__).resolve().parents[1]
MESHES = {
    "t_body": ROOT / "generated/aprilcube_parts/t_body/grasp_mesh.obj",
    "u_legs": ROOT / "generated/aprilcube_parts/u_legs/grasp_mesh.obj",
    "cube_head": ROOT / "generated/aprilcube_parts/cube_head/grasp_mesh.obj",
}


def test_runtime_observations_are_distinct_supported_scenes_without_fixture():
    nominal = load_observation(
        ROOT / "config/observations/t_u_cube_nominal_v1.yaml", MESHES
    )
    shuffled = load_observation(
        ROOT / "config/observations/t_u_cube_shuffled_v1.yaml", MESHES
    )
    assert np.isclose(nominal.table.top_z, 0.84)
    assert np.isclose(shuffled.table.top_z, 0.84)
    assert any(
        not np.allclose(nominal.world_T_objects[name], shuffled.world_T_objects[name])
        for name in MESHES
    )
    for observation in (nominal, shuffled):
        # T and U are genuinely lying on a broad face: their canonical
        # thickness axis (object Y) is vertical in the world, not object Z.
        for name in ("t_body", "u_legs"):
            rotation = observation.world_T_objects[name][:3, :3]
            assert np.isclose(abs(rotation[2, 1]), 1.0, atol=1e-6)
            assert np.isclose(rotation[2, 2], 0.0, atol=1e-6)
    assert not np.allclose(
        nominal.world_T_objects["t_body"][:3, :3],
        shuffled.world_T_objects["t_body"][:3, :3],
    )
    config = yaml.safe_load(
        (ROOT / "config/planning/t_u_cube_runtime_v2.yaml").read_text()
    )
    assert "fixtures" not in config
    assert "initial_world_pose" not in str(config)


def test_observation_rejects_overlap_and_non_normalized_quaternion(tmp_path):
    source = yaml.safe_load(
        (ROOT / "config/observations/t_u_cube_nominal_v1.yaml").read_text()
    )
    source["objects"]["cube_head"]["translation"][:2] = source["objects"]["u_legs"][
        "translation"
    ][:2]
    overlap = tmp_path / "overlap.yaml"
    overlap.write_text(yaml.safe_dump(source))
    with pytest.raises(RuntimeObservationError, match="overlap"):
        load_observation(overlap, MESHES)

    source = yaml.safe_load(
        (ROOT / "config/observations/t_u_cube_nominal_v1.yaml").read_text()
    )
    source["objects"]["cube_head"]["quaternion_xyzw"] = [0, 0, 0, 2]
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(source))
    with pytest.raises(RuntimeObservationError, match="quaternion norm"):
        load_observation(invalid, MESHES)

    source = yaml.safe_load(
        (ROOT / "config/observations/t_u_cube_nominal_v1.yaml").read_text()
    )
    source["objects"]["cube_head"]["translation"][0] = 1.1
    unsupported = tmp_path / "unsupported.yaml"
    unsupported.write_text(yaml.safe_dump(source))
    with pytest.raises(RuntimeObservationError, match="tabletop XY bounds"):
        load_observation(unsupported, MESHES)


def test_atlas_world_transform_is_exact_and_does_not_mutate_candidate():
    pool = load_grasp_pool(
        ROOT / "artifacts/grasp_atlas/cube_viral_v1/right/arm_grasp_pool.yaml"
    )
    original = pool[0].object_T_G.copy()
    world_T_object = tra.translation_matrix([0.3, -0.2, 0.7]) @ tra.rotation_matrix(
        0.4, [0, 0, 1]
    )
    actual = world_grasps(world_T_object, pool[:1])[0]
    assert np.allclose(actual, world_T_object @ original)
    assert np.array_equal(pool[0].object_T_G, original)


def test_workspace_is_bounded_center_first_and_placement_height_is_derived():
    cfg = yaml.safe_load(
        (ROOT / "config/planning/t_u_cube_runtime_v2.yaml").read_text()
    )["workspace"]
    work = workspace_samples(cfg["u_stage"])
    assert len(work) == 81
    assert np.allclose(work[0][1][:3, 3], [0.42, 0.0, 1.02])
    placements = placement_samples(cfg["placement"], 0.84, -0.225)
    assert len(placements) == 27
    assert all(np.isclose(matrix[2, 3], 1.068) for _, matrix in placements)


def test_new_runtime_has_no_project_ik_or_cartesian_interpolator():
    paths = [
        ROOT / "g1_aprilcube_demo/planning/curobo_backend.py",
        ROOT / "g1_aprilcube_demo/planning/runtime_assembly.py",
    ]
    function_names = set()
    calls = set()
    for path in paths:
        tree = ast.parse(path.read_text())
        function_names.update(
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert function_names.isdisjoint({"_ik_targets", "solve_ik", "inverse_kinematics"})
    assert not any("interpol" in name.lower() for name in function_names)
    assert "plan_grasp" in calls
    assert "plan_pose" in calls
    assert "linear_motion" in calls


def test_stage_robot_locks_exact_descriptor_hand_state():
    config = yaml.safe_load(
        (ROOT / "config/planning/t_u_cube_runtime_v2.yaml").read_text()
    )
    backend = object.__new__(CuroboBackend)
    backend.robot_document = yaml.safe_load(
        (ROOT / config["robot"]["model"]).read_text()
    )
    backend.arm_joint_names = list(config["robot"]["arm_joint_names"])
    backend.hand_profiles = {
        side: json.loads((ROOT / path).read_text())
        for side, path in config["robot"]["hand_profiles"].items()
    }
    robot = backend._stage_robot(
        "right",
        np.zeros(len(backend.arm_joint_names)),
        {"left": 1.0, "right": 0.0},
    )
    locks = robot["robot_cfg"]["kinematics"]["lock_joints"]
    assert {
        name: locks[name] for name in backend.hand_profiles["left"]["close"]
    } == backend.hand_profiles["left"]["close"]
    assert {
        name: locks[name] for name in backend.hand_profiles["right"]["open"]
    } == backend.hand_profiles["right"]["open"]
