from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh.transformations as tra
import yaml

from g1_aprilcube_demo.planning.grasp_goalset import (
    GraspCandidate,
    goal_tool_pose,
    load_grasp_pool,
    world_grasps,
)
from g1_aprilcube_demo.planning.curobo_backend import CuroboBackend
from g1_aprilcube_demo.planning.runtime_assembly import (
    GoalsetPickupDomain,
    RoleAssignment,
    RuntimeAssemblyPlanner,
)
from g1_aprilcube_demo.planning.workspace import placement_samples, workspace_samples
from g1_aprilcube_demo.runtime import RuntimeObservationError, load_observation


ROOT = Path(__file__).resolve().parents[1]
MESHES = {
    "t_body": ROOT / "generated/aprilcube_parts/t_body/grasp_mesh.obj",
    "u_legs": ROOT / "generated/aprilcube_parts/u_legs/grasp_mesh.obj",
    "cube_head": ROOT / "generated/aprilcube_parts/cube_head/grasp_mesh.obj",
}


def test_runtime_observations_are_distinct_upright_scenes_without_fixture():
    nominal = load_observation(
        ROOT / "config/observations/t_u_cube_nominal_v1.yaml", MESHES
    )
    shuffled = load_observation(
        ROOT / "config/observations/t_u_cube_shuffled_v1.yaml", MESHES
    )
    assert np.isclose(nominal.table.top_z, 0.70)
    assert np.isclose(shuffled.table.top_z, 0.70)
    assert any(
        not np.allclose(nominal.world_T_objects[name], shuffled.world_T_objects[name])
        for name in MESHES
    )
    for observation in (nominal, shuffled):
        # T and U preserve canonical up. In particular, the U uses the named
        # supported-pickup condition: upright on both leg ends.
        for name in ("t_body", "u_legs"):
            rotation = observation.world_T_objects[name][:3, :3]
            assert np.isclose(rotation[2, 2], 1.0, atol=1e-6)
            assert np.isclose(rotation[2, 0], 0.0, atol=1e-6)
            assert np.isclose(rotation[2, 1], 0.0, atol=1e-6)
        assert np.isclose(
            observation.world_T_objects["u_legs"][2, 3], 0.7675
        )
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


def test_goal_pose_keeps_candidates_as_alternatives_in_one_problem():
    first = tra.translation_matrix([0.1, 0.2, 0.3])
    second = tra.translation_matrix([-0.4, 0.5, 0.6])
    base_T_world = tra.translation_matrix([1.0, 0.0, 0.0])
    goal = goal_tool_pose(
        {"tool": [first, second]}, base_T_world, device="cpu"
    )

    assert goal.position.shape == (1, 1, 1, 2, 3)
    assert goal.quaternion.shape == (1, 1, 1, 2, 4)
    assert np.allclose(
        goal.position[0, 0, 0].numpy(),
        [[1.1, 0.2, 0.3], [0.6, 0.5, 0.6]],
    )


def test_runtime_lazily_requeries_native_goalsets_without_family_gate():
    candidates = []
    for index in range(5):
        pose = tra.translation_matrix([float(index), 0.0, 0.0])
        candidates.append(
            GraspCandidate(
                candidate_id=f"candidate_{index}",
                family_id="same_family",
                score=0.0,
                object_T_G=pose,
            )
        )

    class FakeGoalsetPlanner:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def plan_grasp(self, matrices, **_kwargs):
            self.calls.append(len(matrices))
            # The second chunk has no feasible alternative. The first chunk
            # yields its middle member, then a different remaining member on
            # the next lazy round.
            if len(self.calls) == 1:
                return SimpleNamespace(selected_index=1)
            if len(self.calls) == 2:
                return None
            return SimpleNamespace(selected_index=0)

    goalset_planner = FakeGoalsetPlanner()

    class FakeBackend:
        @staticmethod
        def scene(*_args):
            return {}

        @staticmethod
        def stage(*_args, **_kwargs):
            return goalset_planner

    planner = object.__new__(RuntimeAssemblyPlanner)
    planner.pools = {("part", "right"): tuple(candidates)}
    planner.cfg = {
        "planner": {"candidate_goalset_size": 3},
        "motion": {"pick_approach_local_z_m": -0.1},
    }
    planner.observation = SimpleNamespace(
        world_T_objects={"part": np.eye(4)},
        table=SimpleNamespace(center=np.zeros(3), dimensions=np.ones(3)),
    )
    planner.backend = FakeBackend()
    planner.initial_q = np.zeros(14)
    planner.contact_links = {"right": ("finger",)}
    planner.run = SimpleNamespace(qualification=[])

    domain = planner._new_pickup_domain("part", "right")
    assert [[item.candidate_id for item in chunk] for chunk in domain.chunks] == [
        ["candidate_0", "candidate_1", "candidate_2"],
        ["candidate_3", "candidate_4"],
    ]

    first = planner._expand_pickup_domain(domain)
    second = planner._expand_pickup_domain(domain)

    assert [candidate.candidate_id for candidate in first] == ["candidate_1"]
    assert [candidate.candidate_id for candidate in second] == ["candidate_0"]
    assert [candidate.candidate_id for candidate in domain.selected] == [
        "candidate_1",
        "candidate_0",
    ]
    assert goalset_planner.calls == [3, 2, 2]
    assert [item.candidate_id for item in domain.chunks[0]] == ["candidate_2"]
    assert domain.chunks[1] == []
    assert planner.run.qualification[0]["goalset_capacity"] == 3
    assert planner.run.qualification[0]["atlas_candidate_count"] == 5


def test_mode_search_rejects_the_failed_prefix_at_its_dependency_scope():
    def candidate(name):
        return GraspCandidate(name, "one_family", 0.0, np.eye(4))

    t = candidate("t")
    u1, u2 = candidate("u1"), candidate("u2")
    c1, c2 = candidate("c1"), candidate("c2")
    domains = {
        "t_body": GoalsetPickupDomain("t_body", "left", [], [t]),
        "u_legs": GoalsetPickupDomain("u_legs", "right", [], [u1, u2]),
        "cube_head": GoalsetPickupDomain(
            "cube_head", "right", [], [c1, c2]
        ),
    }

    class Context:
        def __enter__(self):
            return SimpleNamespace()

        def __exit__(self, *_exc):
            return False

    planner = object.__new__(RuntimeAssemblyPlanner)
    planner.initial_q = np.zeros(14)
    planner.backend = SimpleNamespace(coupled=lambda *_args, **_kwargs: Context())
    planner._cached_pair_mode = (
        lambda _coupled, _assignment, _t, child, grasp: (
            f"{child}_{grasp.candidate_id}",
            np.eye(4),
            np.zeros(14),
        )
    )
    assignment = RoleAssignment("left", "right")

    mode = planner._find_selected_mode(
        assignment,
        domains,
        {},
        set(),
        {("t", "u1")},
        set(),
    )
    assert (mode.u_grasp.candidate_id, mode.cube_grasp.candidate_id) == (
        "u2",
        "c1",
    )

    mode = planner._find_selected_mode(
        assignment,
        domains,
        {},
        set(),
        {("t", "u1")},
        {("left", "right", "t", "u2", "c1")},
    )
    assert (mode.u_grasp.candidate_id, mode.cube_grasp.candidate_id) == (
        "u2",
        "c2",
    )
    assert (
        planner._find_selected_mode(
            assignment,
            domains,
            {},
            {"t"},
            set(),
            set(),
        )
        is None
    )


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
