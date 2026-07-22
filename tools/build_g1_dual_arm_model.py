#!/usr/bin/env python3
"""Derive the fixed-torso, dual-arm current-Dex3 cuRobo model.

The source of robot geometry and dynamics is NVIDIA/cuRobo's official Unitree
G1 rev-1.0 asset.  This project-owned builder only removes the legs/sensors,
fixes the waist, exposes the two validated GraspGenX G frames as tools, and
adds one standard cuRobo attachment-sphere slot per hand.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh.transformations as tra
import yaml


ROOT = Path(__file__).resolve().parents[1]
CUROBO = ROOT / "third_party/GraspGenX/ext/curobo"
SOURCE_DIR = CUROBO / "curobo/content/assets/robot/g1"
SOURCE_URDF = SOURCE_DIR / "g1_29dof_with_hand_rev_1_0.urdf"
SOURCE_CONFIG = CUROBO / "curobo/content/configs/robot/unitree_g1.yml"
OUTPUT_DIR = ROOT / "generated/robot"
OUTPUT_URDF = OUTPUT_DIR / "g1_fixed_torso_dual_dex3.urdf"
OUTPUT_CONFIG = OUTPUT_DIR / "g1_fixed_torso_dual_dex3.yml"

ARM_JOINTS = {
    side: [
        f"{side}_shoulder_pitch_joint",
        f"{side}_shoulder_roll_joint",
        f"{side}_shoulder_yaw_joint",
        f"{side}_elbow_joint",
        f"{side}_wrist_roll_joint",
        f"{side}_wrist_pitch_joint",
        f"{side}_wrist_yaw_joint",
    ]
    for side in ("left", "right")
}
HAND_JOINTS = {
    side: [
        f"{side}_hand_thumb_0_joint",
        f"{side}_hand_thumb_1_joint",
        f"{side}_hand_thumb_2_joint",
        f"{side}_hand_middle_0_joint",
        f"{side}_hand_middle_1_joint",
        f"{side}_hand_index_0_joint",
        f"{side}_hand_index_1_joint",
    ]
    for side in ("left", "right")
}
TOOL_FRAMES = {side: f"{side}_hand_grasp_frame" for side in ("left", "right")}
ATTACHMENT_LINKS = {
    side: f"{side}_attached_object" for side in ("left", "right")
}
PRUNE_ROOT_JOINTS = {
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "logo_joint",
    "head_joint",
    "imu_in_torso_joint",
    "imu_in_pelvis_joint",
    "d435_joint",
    "mid360_joint",
}


def joint_links(joint: ET.Element) -> tuple[str, str]:
    return (
        str(joint.find("parent").get("link")),
        str(joint.find("child").get("link")),
    )


def descendants(root: ET.Element, root_joint_names: set[str]) -> set[str]:
    children: dict[str, list[str]] = {}
    removed: set[str] = set()
    frontier: list[str] = []
    for joint in root.findall("joint"):
        parent, child = joint_links(joint)
        children.setdefault(parent, []).append(child)
        if joint.get("name") in root_joint_names:
            removed.add(child)
            frontier.append(child)
    while frontier:
        for child in children.get(frontier.pop(), []):
            if child not in removed:
                removed.add(child)
                frontier.append(child)
    return removed


def make_fixed(joint: ET.Element, position: float = 0.0) -> None:
    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    xyz = np.asarray([float(x) for x in origin.get("xyz", "0 0 0").split()])
    rpy = np.asarray([float(x) for x in origin.get("rpy", "0 0 0").split()])
    transform = tra.euler_matrix(*rpy)
    transform[:3, 3] = xyz
    axis_node = joint.find("axis")
    axis = np.asarray(
        [float(x) for x in axis_node.get("xyz", "1 0 0").split()]
        if axis_node is not None
        else [1.0, 0.0, 0.0]
    )
    transform = transform @ tra.rotation_matrix(position, axis)
    origin.set("xyz", " ".join(f"{x:.12g}" for x in transform[:3, 3]))
    origin.set(
        "rpy", " ".join(f"{x:.12g}" for x in tra.euler_from_matrix(transform))
    )
    joint.set("type", "fixed")
    for tag in ("axis", "limit", "dynamics", "safety_controller", "mimic"):
        node = joint.find(tag)
        if node is not None:
            joint.remove(node)


def add_tool(root: ET.Element, side: str) -> None:
    descriptor = (
        ROOT
        / f"third_party/GraspGenX/assets/x_grippers/dex3_rev1_{side}/gripper.urdf"
    )
    descriptor_joint = ET.parse(descriptor).getroot().find("joint")
    origin = descriptor_joint.find("origin")
    xyz = np.asarray([float(x) for x in origin.get("xyz", "0 0 0").split()])
    rpy = np.asarray([float(x) for x in origin.get("rpy", "0 0 0").split()])
    grasp_T_palm = tra.euler_matrix(*rpy)
    grasp_T_palm[:3, 3] = xyz
    palm_T_grasp = tra.inverse_matrix(grasp_T_palm)
    palm_rpy = tra.euler_from_matrix(palm_T_grasp)
    frame = TOOL_FRAMES[side]
    ET.SubElement(root, "link", {"name": frame})
    joint = ET.SubElement(
        root,
        "joint",
        {"name": f"{side}_hand_palm_to_grasp_frame", "type": "fixed"},
    )
    ET.SubElement(joint, "parent", {"link": f"{side}_hand_palm_link"})
    ET.SubElement(joint, "child", {"link": frame})
    ET.SubElement(
        joint,
        "origin",
        {
            "xyz": " ".join(f"{x:.12g}" for x in palm_T_grasp[:3, 3]),
            "rpy": " ".join(f"{x:.12g}" for x in palm_rpy),
        },
    )


def build_urdf() -> set[str]:
    tree = ET.parse(SOURCE_URDF)
    root = tree.getroot()
    root.set("name", "g1_fixed_torso_dual_dex3")
    removed = descendants(root, PRUNE_ROOT_JOINTS)
    movable = set(sum(ARM_JOINTS.values(), []) + sum(HAND_JOINTS.values(), []))
    for joint in list(root.findall("joint")):
        parent, child = joint_links(joint)
        if joint.get("name") in PRUNE_ROOT_JOINTS or parent in removed or child in removed:
            root.remove(joint)
        elif joint.get("type") != "fixed" and joint.get("name") not in movable:
            make_fixed(joint)
    for link in list(root.findall("link")):
        if link.get("name") in removed:
            root.remove(link)
    for side in ("left", "right"):
        add_tool(root, side)
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename and not Path(filename).is_absolute():
            mesh.set("filename", str((SOURCE_DIR / filename).resolve()))
    actual = {
        str(joint.get("name"))
        for joint in root.findall("joint")
        if joint.get("type") != "fixed"
    }
    if actual != movable:
        raise RuntimeError(f"Unexpected movable joints: {sorted(actual ^ movable)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, "  ")
    tree.write(OUTPUT_URDF, encoding="utf-8", xml_declaration=True)
    return {str(link.get("name")) for link in root.findall("link")}


def build_config(kept_links: set[str]) -> None:
    source = yaml.safe_load(SOURCE_CONFIG.read_text())
    kin = source.get("robot_cfg", source)["kinematics"]
    robot_collision_links = [
        link for link in kin["collision_link_names"] if link in kept_links
    ]
    collision_links = robot_collision_links + list(ATTACHMENT_LINKS.values())
    collision_spheres = {
        link: spheres
        for link, spheres in kin["collision_spheres"].items()
        if link in robot_collision_links
    }
    self_ignore = {
        link: [other for other in others if other in robot_collision_links]
        for link, others in kin.get("self_collision_ignore", {}).items()
        if link in robot_collision_links
    }
    self_buffer = {
        link: value
        for link, value in kin.get("self_collision_buffer", {}).items()
        if link in robot_collision_links
    }
    extra_links = {}
    for side in ("left", "right"):
        attachment = ATTACHMENT_LINKS[side]
        hand_links = [
            link for link in robot_collision_links if link.startswith(f"{side}_hand_")
        ]
        extra_links[attachment] = {
            "link_name": attachment,
            "joint_name": f"{side}_attachment_joint",
            "joint_type": "FIXED",
            "parent_link_name": TOOL_FRAMES[side],
            "fixed_transform": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
        self_ignore[attachment] = hand_links
        self_buffer[attachment] = 0.0
        for link in hand_links:
            self_ignore.setdefault(link, [])
            if attachment not in self_ignore[link]:
                self_ignore[link].append(attachment)

    planning_joints = ARM_JOINTS["left"] + ARM_JOINTS["right"]
    hand_joints = HAND_JOINTS["left"] + HAND_JOINTS["right"]
    output = {
        "robot_cfg": {
            "kinematics": {
                "urdf_path": str(OUTPUT_URDF.resolve()),
                "asset_root_path": str(SOURCE_DIR.resolve()),
                "base_link": "base_link",
                "collision_link_names": collision_links,
                "collision_spheres": collision_spheres,
                "extra_collision_spheres": {
                    ATTACHMENT_LINKS["left"]: 96,
                    ATTACHMENT_LINKS["right"]: 96,
                },
                "extra_links": extra_links,
                "collision_sphere_buffer": float(
                    kin.get("collision_sphere_buffer", 0.0)
                ),
                "cspace": {
                    "joint_names": planning_joints + hand_joints,
                    "cspace_distance_weight": [1.0] * 28,
                    "null_space_weight": [1.0] * 28,
                    "max_acceleration": [10.0] * 28,
                    "max_jerk": [500.0] * 28,
                    "position_limit_clip": 0.0,
                    "default_joint_position": [0.0] * 28,
                },
                "tool_frames": list(TOOL_FRAMES.values()),
                "lock_joints": {joint: 0.0 for joint in hand_joints},
                "mesh_link_names": [
                    link
                    for link in kin.get("mesh_link_names", robot_collision_links)
                    if link in robot_collision_links
                ],
                "self_collision_buffer": self_buffer,
                "self_collision_ignore": self_ignore,
                "use_global_cumul": bool(kin.get("use_global_cumul", True)),
                "load_meshes": bool(kin.get("load_meshes", False)),
            }
        }
    }
    OUTPUT_CONFIG.write_text(yaml.safe_dump(output, sort_keys=False))


def main() -> None:
    kept = build_urdf()
    build_config(kept)
    print(f"wrote {OUTPUT_URDF}")
    print(f"wrote {OUTPUT_CONFIG}")


if __name__ == "__main__":
    main()
