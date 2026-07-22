from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from g1_aprilcube_demo.assembly import load_assembly_task


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "config/tasks/t_u_cube_humanoid_v1.yaml"


def test_task_compiles_to_expected_state_sequence():
    task = load_assembly_task(TASK_PATH)
    stages = task.compile()
    assert [stage.step_id for stage in stages] == [
        "pick_t",
        "pick_u",
        "mate_u_to_t",
        "pick_head",
        "mate_head_to_t",
        "place_complete",
    ]
    assert stages[0].after.hand_payloads["holder"] == ("t_body",)
    assert stages[2].after.hand_payloads["holder"] == ("t_body", "u_legs")
    assert stages[2].after.hand_payloads["worker"] == ()
    assert stages[4].after.hand_payloads["holder"] == (
        "cube_head",
        "t_body",
        "u_legs",
    )
    assert stages[-1].after.hand_payloads == {"holder": (), "worker": ()}
    assert stages[-1].after.placed_assembly == (
        "cube_head",
        "t_body",
        "u_legs",
    )


def test_mating_transforms_are_exact_and_final_height_is_360_mm():
    task = load_assembly_task(TASK_PATH)
    transforms = task.member_transforms()
    bounds = []
    for part_id, part in task.parts.items():
        mesh = trimesh.load(part.mesh, force="mesh", process=False)
        corners = trimesh.bounds.corners(mesh.bounds)
        world = (
            transforms[part_id]
            @ np.column_stack((corners, np.ones(len(corners)))).T
        ).T[:, :3]
        bounds.append(np.stack((world.min(axis=0), world.max(axis=0))))
    combined_min = np.min([bound[0] for bound in bounds], axis=0)
    combined_max = np.max([bound[1] for bound in bounds], axis=0)
    assert np.isclose(combined_max[2] - combined_min[2], 0.360, atol=1e-12)

    u = task.connections["u_to_t"]
    head = task.connections["head_to_t"]
    assert np.allclose(
        (u.parent_T_child.matrix @ [*u.child_contact_point_m, 1.0])[:3],
        u.parent_contact_point_m,
    )
    assert np.allclose(
        (head.parent_T_child.matrix @ [*head.child_contact_point_m, 1.0])[:3],
        head.parent_contact_point_m,
    )


def test_readiness_selects_left_holder_and_right_worker():
    report = load_assembly_task(TASK_PATH).readiness_report()
    assert report["motion_planning_ready"]
    assignments = {
        tuple(sorted(item["role_to_hand"].items())): item for item in report["assignments"]
    }
    right_holder = assignments[(('holder', 'right'), ('worker', 'left'))]
    assert {item["part"] for item in right_holder["missing_grasp_pools"]} == {
        "u_legs",
        "cube_head",
    }
    left_holder = assignments[(('holder', 'left'), ('worker', 'right'))]
    assert left_holder["ready"]
    assert left_holder["missing_grasp_pools"] == []


def test_pick_and_mate_commands_do_not_depend_on_newton():
    task = load_assembly_task(TASK_PATH)
    command_kinds = [
        command.kind for stage in task.compile() for command in stage.commands
    ]
    assert "select_qualified_grasp" in command_kinds
    assert "plan_mate_contact" in command_kinds
    assert all("newton" not in kind.lower() for kind in command_kinds)
