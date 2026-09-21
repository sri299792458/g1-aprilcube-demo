from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh.transformations as tra
import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tools/build_g1_dual_arm_model.py"
URDF = ROOT / "generated/robot/g1_fixed_torso_dual_dex3.urdf"
CONFIG = ROOT / "generated/robot/g1_fixed_torso_dual_dex3.yml"


def setup_module():
    subprocess.run([str(ROOT / ".venv/bin/python"), str(BUILDER)], check=True)


def test_model_has_only_two_arms_and_two_hands_movable():
    root = ET.parse(URDF).getroot()
    movable = [
        joint.get("name")
        for joint in root.findall("joint")
        if joint.get("type") != "fixed"
    ]
    assert len(movable) == 28
    assert sum("shoulder" in joint for joint in movable) == 6
    assert sum("hand_" in joint for joint in movable) == 14
    assert not any("hip" in joint or "knee" in joint or "ankle" in joint for joint in movable)


def test_each_robot_tool_is_exact_inverse_of_its_descriptor_transform():
    robot = ET.parse(URDF).getroot()
    joints = {joint.get("name"): joint for joint in robot.findall("joint")}
    for side in ("left", "right"):
        descriptor = ET.parse(
            ROOT / f"third_party/GraspGenX/assets/x_grippers/dex3_rev1_{side}/gripper.urdf"
        ).getroot()
        d_origin = descriptor.find("joint/origin")
        d_xyz = [float(v) for v in d_origin.get("xyz").split()]
        d_rpy = [float(v) for v in d_origin.get("rpy").split()]
        grasp_T_palm = tra.euler_matrix(*d_rpy)
        grasp_T_palm[:3, 3] = d_xyz
        r_origin = joints[f"{side}_hand_palm_to_grasp_frame"].find("origin")
        r_xyz = [float(v) for v in r_origin.get("xyz").split()]
        r_rpy = [float(v) for v in r_origin.get("rpy").split()]
        palm_T_grasp = tra.euler_matrix(*r_rpy)
        palm_T_grasp[:3, 3] = r_xyz
        assert np.allclose(grasp_T_palm @ palm_T_grasp, np.eye(4), atol=1e-10)


def test_config_exposes_both_tools_and_independent_attachment_slots():
    kin = yaml.safe_load(CONFIG.read_text())["robot_cfg"]["kinematics"]
    assert kin["tool_frames"] == [
        "left_hand_grasp_frame",
        "right_hand_grasp_frame",
    ]
    assert kin["extra_collision_spheres"] == {
        "left_attached_object": 96,
        "right_attached_object": 96,
    }
    assert kin["extra_links"]["left_attached_object"]["parent_link_name"] == "left_hand_grasp_frame"
    assert kin["extra_links"]["right_attached_object"]["parent_link_name"] == "right_hand_grasp_frame"
    assert len(kin["lock_joints"]) == 14
