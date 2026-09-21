#!/usr/bin/env python3
"""Fail loudly if our Isaac qualifier diverges from the pinned VIRAL code."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPGEN_SCRIPTS = (
    PROJECT_ROOT / "third_party/GraspDataGen/scripts/graspgen"
)
DEFAULT_VIRAL_REPO = Path(
    "/home/srinivas/Desktop/g1pilot-workspace/reference_repos/"
    "GR00T-VisualSim2Real"
)
ATLAS_CONFIGS = (
    PROJECT_ROOT / "config/grasp_atlas/cube_viral_v1.yaml",
    PROJECT_ROOT / "config/grasp_atlas/t_body_viral_v1.yaml",
    PROJECT_ROOT / "config/grasp_atlas/u_legs_viral_v1.yaml",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, label: str) -> None:
    require(
        math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9),
        f"{label}: expected {expected}, got {actual}",
    )


def regex_value(mapping: dict[str, float], joint_name: str, label: str) -> float:
    matches = [value for pattern, value in mapping.items() if re.fullmatch(pattern, joint_name)]
    require(len(matches) == 1, f"{label}: {joint_name} matched {len(matches)} entries")
    return float(matches[0])


def substring_gain(mapping: dict[str, float], joint_name: str, label: str) -> float:
    stem = joint_name.removesuffix("_joint")
    matches = [value for key, value in mapping.items() if key in stem]
    require(len(matches) == 1, f"{label}: {joint_name} matched {len(matches)} entries")
    return float(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viral-repo", type=Path, default=DEFAULT_VIRAL_REPO)
    args = parser.parse_args()
    viral_repo = args.viral_repo.resolve()

    sys.path.insert(0, str(GRASPGEN_SCRIPTS))
    from simulation_profiles import VIRAL_G1_DEX3_PROFILE, get_profile

    profile = get_profile(VIRAL_G1_DEX3_PROFILE)
    assert profile is not None
    expected_commit = profile["source"]["commit"]
    actual_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=viral_repo, text=True
    ).strip()
    require(actual_commit == expected_commit, f"VIRAL commit: {actual_commit}")

    robot_path = viral_repo / profile["source"]["robot_config"]
    simulator_path = viral_repo / profile["source"]["simulator_config"]
    adapter_path = viral_repo / profile["source"]["adapter"]
    robot = yaml.safe_load(robot_path.read_text())["robot"]
    simulator = yaml.safe_load(simulator_path.read_text())["simulator"]["config"]
    adapter = adapter_path.read_text()

    simulation = profile["simulation"]
    close(simulator["sim"]["fps"], simulation["physics_hz"], "physics_hz")
    require(
        simulator["sim"]["control_decimation"]
        == simulation["control_decimation"],
        "control_decimation mismatch",
    )
    close(
        simulation["physics_hz"] / simulation["control_decimation"],
        simulation["control_hz"],
        "control_hz",
    )
    require(
        simulator["sim"]["physx"]["solver_type"] == simulation["solver_type"],
        "solver_type mismatch",
    )
    require("max_position_iteration_count=255" in adapter, "active PhysX max-position bound changed")
    require("max_velocity_iteration_count=255" in adapter, "active PhysX max-velocity bound changed")
    require("gpu_max_rigid_patch_count=10 * 2**15" in adapter, "rigid-patch bound changed")
    require("actuators[\"all\"] = ImplicitActuatorCfg(" in adapter, "active actuator is not implicit PD")
    require("dof_armature_list[i] * 3" in adapter, "armature x3 adapter scaling changed")
    require("dof_joint_friction_list[i] * 0" in adapter, "adapter friction-zeroing changed")
    require("self.simulator_config.sim.physx.contact_offset" not in adapter, "global contact_offset became active")
    require(simulator["actuators"]["actuation_mode"] == "idealpd", "declared actuation mode changed")

    articulation = profile["articulation"]
    require(
        articulation["enabled_self_collisions"]
        is (not bool(robot["asset"]["self_collisions"])),
        "self-collision mapping mismatch",
    )
    require(
        articulation["solver_position_iterations"]
        == simulator["sim"]["physx"]["num_position_iterations"],
        "articulation position iterations mismatch",
    )
    require(
        articulation["solver_velocity_iterations"]
        == simulator["sim"]["physx"]["num_velocity_iterations"],
        "articulation velocity iterations mismatch",
    )
    close(
        articulation["max_depenetration_velocity"],
        simulator["sim"]["physx"]["max_depenetration_velocity"],
        "max_depenetration_velocity",
    )

    actuator = profile["actuator"]
    dof_names = robot["dof_names"]
    right_hand_names = robot["right_hand_dof_names"]
    require(len(right_hand_names) == 7, "expected seven right Dex3 joints")
    for name in right_hand_names:
        index = dof_names.index(name)
        close(
            regex_value(actuator["stiffness"], name, "stiffness profile"),
            substring_gain(robot["control"]["stiffness"], name, "VIRAL stiffness"),
            f"{name} stiffness",
        )
        close(
            regex_value(actuator["damping"], name, "damping profile"),
            substring_gain(robot["control"]["damping"], name, "VIRAL damping"),
            f"{name} damping",
        )
        close(
            regex_value(actuator["effort_limit_sim"], name, "effort profile"),
            robot["dof_effort_limit_list"][index],
            f"{name} effort",
        )
        close(
            regex_value(actuator["velocity_limit_sim"], name, "velocity profile"),
            robot["dof_vel_limit_list"][index],
            f"{name} velocity",
        )
        close(
            regex_value(actuator["armature"], name, "armature profile"),
            robot["dof_armature_list"][index] * 3.0,
            f"{name} armature",
        )
        close(
            regex_value(actuator["friction"], name, "friction profile"),
            robot["dof_joint_friction_list"][index] * 0.0,
            f"{name} friction",
        )

    object_profile = profile["object"]
    require("contact_offset=0.002" in (
        viral_repo
        / "gr00t/rl/data/tasks/walk_stand_place_grasp_turn_homie/scenario_cfg/isaacsim.py"
    ).read_text(), "active task-object 2 mm contact offset changed")
    for config_path in ATLAS_CONFIGS:
        config = yaml.safe_load(config_path.read_text())
        contract = config["physics"]["object_contract"]
        require(
            contract["collision_approximation"]
            == object_profile["collision_approximation"],
            f"{config_path.name}: collision approximation mismatch",
        )
        for config_key, profile_key in (
            ("static_friction", "static_friction"),
            ("dynamic_friction", "dynamic_friction"),
            ("restitution", "restitution"),
            ("contact_offset_m", "contact_offset"),
            ("rest_offset_m", "rest_offset"),
            ("linear_damping", "linear_damping"),
            ("angular_damping", "angular_damping"),
            ("max_depenetration_velocity_mps", "max_depenetration_velocity"),
        ):
            close(contract[config_key], object_profile[profile_key], f"{config_path.name} {config_key}")
        require(
            contract["gravity_enabled"] is (not object_profile["disable_gravity"]),
            f"{config_path.name}: gravity mismatch",
        )

    print(f"PASS: {VIRAL_G1_DEX3_PROFILE}")
    print(f"  source commit: {actual_commit}")
    print("  7 right-Dex3 actuator rows match YAML and executed x3/x0 adapter transforms")
    print("  200 Hz / 50 Hz source command rate / TGS / 4+0 solver contract matches")
    print("  AprilCube object contract is explicit and consistent across cube/T/U")
    print("  declared-but-dormant idealpd, 0.01 contact offset, and 0.95 effort scale remain excluded")


if __name__ == "__main__":
    main()
