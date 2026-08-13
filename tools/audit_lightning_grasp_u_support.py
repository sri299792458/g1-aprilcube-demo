#!/usr/bin/env python3
"""Audit Lightning-Grasp Dex3 solutions against the U's broad-face supports.

Lightning Grasp searches in the hand/object system and does not model a table.
This tool performs the missing, deliberately narrow check without changing the
generated grasp:

1. place the exact U mesh in each configured broad-face support;
2. recover the hand-base pose implied by Lightning Grasp's G_T_object output;
3. forward-kinematically place the exact articulated Dex3 collision geometry;
4. measure table clearance and hand/object penetration.

It does not invent a pregrasp, approach direction, or closing trajectory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import fcl
import numpy as np
import pyrender
import torch
import trimesh
import yaml
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
LIGHTNING_ROOT = ROOT / "third_party/lightning-grasp"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LIGHTNING_ROOT))

from g1_aprilcube_demo.grasping.support_atlas import (  # noqa: E402
    configured_support_conditions,
)
from lygra.kinematics import batch_fk, build_kinematics_tree  # noqa: E402
from lygra.mesh import RobotMesh  # noqa: E402


DEFAULT_INPUT = (
    ROOT
    / "artifacts/lightning_grasp/u_legs_dex3_right_exhaustive_seed20260727.npz"
)
DEFAULT_URDF = (
    ROOT
    / "third_party/GraspGenX/assets/x_grippers/dex3_rev1_right/gripper.urdf"
)
DEFAULT_OBJECT = ROOT / "generated/aprilcube_parts/u_legs/grasp_mesh.obj"
DEFAULT_SUPPORT_CONFIG = ROOT / "config/grasp_support/u_legs_right_v1.yaml"
DEFAULT_REPORT = ROOT / "artifacts/lightning_grasp/u_legs_broad_face_audit.json"
DEFAULT_VISUAL = ROOT / "docs/assets/lightning_grasp_dex3_u_broad_face_audit.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--object-mesh", type=Path, default=DEFAULT_OBJECT)
    parser.add_argument("--support-config", type=Path, default=DEFAULT_SUPPORT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--visual", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--table-tolerance-m", type=float, default=1.0e-6)
    parser.add_argument(
        "--penetration-tolerance-m",
        type=float,
        default=0.002,
        help="Paper's practical hand/object penetration margin",
    )
    parser.add_argument("--render-count", type=int, default=8)
    return parser.parse_args()


def mesh_to_bvh(mesh: trimesh.Trimesh) -> fcl.BVHModel:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    model = fcl.BVHModel()
    model.beginModel(len(vertices), len(faces))
    model.addSubModel(vertices, faces)
    model.endModel()
    return model


def fcl_transform(matrix: np.ndarray) -> fcl.Transform:
    return fcl.Transform(
        np.asarray(matrix[:3, :3], dtype=np.float64),
        np.asarray(matrix[:3, 3], dtype=np.float64),
    )


def exact_link_collision_meshes(
    robot_mesh: RobotMesh, tree: Any
) -> dict[str, trimesh.Trimesh]:
    """Load URDF collision geometry without Lightning's convex-hull conversion."""

    meshes: dict[str, trimesh.Trimesh] = {}
    for link_name in tree.links:
        link = next(link for link in robot_mesh.robot.links if link.name == link_name)
        parts = []
        for collision in link.collisions:
            mesh = robot_mesh.create_trimesh_from_data(collision, convex=False)
            if mesh is None:
                continue
            mesh = mesh.copy()
            mesh.apply_transform(
                np.eye(4) if collision.origin is None else collision.origin
            )
            parts.append(mesh)
        if parts:
            meshes[link_name] = trimesh.util.concatenate(parts)
    return meshes


def collision_metrics(
    *,
    object_collision: fcl.CollisionObject,
    link_bvhs: dict[str, fcl.BVHModel],
    support_t_g: np.ndarray,
    g_t_links: np.ndarray,
    tree: Any,
) -> tuple[float, list[str], int]:
    max_depth = 0.0
    colliding_links: list[str] = []
    contact_count = 0
    request = fcl.CollisionRequest(num_max_contacts=128, enable_contact=True)
    for link_id, g_t_link in enumerate(g_t_links):
        link_name = tree.links[link_id]
        bvh = link_bvhs.get(link_name)
        if bvh is None:
            continue
        support_t_link = support_t_g @ g_t_link
        link_object = fcl.CollisionObject(bvh, fcl_transform(support_t_link))
        result = fcl.CollisionResult()
        fcl.collide(link_object, object_collision, request, result)
        if not result.is_collision:
            continue
        colliding_links.append(link_name)
        contact_count += len(result.contacts)
        if result.contacts:
            max_depth = max(
                max_depth,
                max(float(contact.penetration_depth) for contact in result.contacts),
            )
    return max_depth, colliding_links, contact_count


def hand_minimum_z(
    *,
    collision_meshes: dict[str, trimesh.Trimesh],
    support_t_g: np.ndarray,
    g_t_links: np.ndarray,
    tree: Any,
) -> tuple[float, str]:
    minimum = float("inf")
    minimum_link = ""
    for link_id, g_t_link in enumerate(g_t_links):
        link_name = tree.links[link_id]
        mesh = collision_meshes.get(link_name)
        if mesh is None:
            continue
        support_t_link = support_t_g @ g_t_link
        vertices = trimesh.transform_points(mesh.vertices, support_t_link)
        link_minimum = float(vertices[:, 2].min())
        if link_minimum < minimum:
            minimum = link_minimum
            minimum_link = link_name
    return minimum, minimum_link


def look_at(camera_position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - camera_position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    matrix = np.eye(4)
    matrix[:3, 0] = right
    matrix[:3, 1] = up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = camera_position
    return matrix


def posed_hand_visual_mesh(
    *,
    robot_mesh: RobotMesh,
    support_t_g: np.ndarray,
    g_t_links: np.ndarray,
    tree: Any,
) -> trimesh.Trimesh:
    parts = []
    for link_id, g_t_link in enumerate(g_t_links):
        mesh = robot_mesh.get_link_visual_mesh(tree.links[link_id])
        if len(mesh.vertices) == 0:
            continue
        mesh.apply_transform(support_t_g @ g_t_link)
        parts.append(mesh)
    return trimesh.util.concatenate(parts)


def render_trial(
    *,
    object_mesh: trimesh.Trimesh,
    support_t_object: np.ndarray,
    hand_mesh: trimesh.Trimesh,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    scene = pyrender.Scene(
        bg_color=[0.96, 0.95, 0.92, 1.0],
        ambient_light=[0.55, 0.55, 0.55],
    )
    table = trimesh.creation.box(extents=[0.34, 0.30, 0.012])
    table.apply_translation([0.0, 0.0, -0.006])
    table_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.56, 0.36, 0.20, 1.0], roughnessFactor=0.85
    )
    hand_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.38, 0.44, 0.51, 1.0], roughnessFactor=0.75
    )
    object_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.10, 0.72, 0.79, 1.0], roughnessFactor=0.7
    )
    scene.add(pyrender.Mesh.from_trimesh(table, material=table_material, smooth=False))
    scene.add(
        pyrender.Mesh.from_trimesh(object_mesh, material=object_material, smooth=False),
        pose=support_t_object,
    )
    scene.add(
        pyrender.Mesh.from_trimesh(hand_mesh, material=hand_material, smooth=False)
    )
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(44.0))
    scene.add(
        camera,
        pose=look_at(
            np.array([0.28, -0.30, 0.22]),
            np.array([0.0, 0.0, 0.045]),
        ),
    )
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.2)
    scene.add(light, pose=look_at(np.array([0.20, -0.18, 0.35]), np.zeros(3)))
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, _ = renderer.render(scene)
    finally:
        renderer.delete()
    return color


def contact_sheet(
    *,
    selected: list[dict[str, Any]],
    q: np.ndarray,
    fk: np.ndarray,
    supports_by_id: dict[str, Any],
    object_mesh: trimesh.Trimesh,
    robot_mesh: RobotMesh,
    tree: Any,
    output: Path,
) -> None:
    if not selected:
        return
    panels = []
    font = ImageFont.load_default(size=20)
    for trial in selected:
        candidate_index = int(trial["candidate_index"])
        support = supports_by_id[trial["support_id"]]
        hand_mesh = posed_hand_visual_mesh(
            robot_mesh=robot_mesh,
            support_t_g=np.asarray(trial["support_T_G"], dtype=np.float64),
            g_t_links=fk[candidate_index],
            tree=tree,
        )
        rendered = render_trial(
            object_mesh=object_mesh,
            support_t_object=support.support_T_object,
            hand_mesh=hand_mesh,
        )
        image = Image.fromarray(rendered)
        draw = ImageDraw.Draw(image)
        label = (
            f"#{candidate_index}  {support.label}\n"
            f"table clearance {1000 * trial['table_clearance_m']:+.1f} mm  "
            f"max object penetration {1000 * trial['max_object_penetration_m']:.1f} mm"
        )
        draw.rounded_rectangle((12, 12, 628, 66), radius=8, fill=(255, 255, 255, 225))
        draw.multiline_text((22, 18), label, font=font, fill=(24, 35, 48))
        panels.append(image)
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (640 * columns, 480 * rows), (245, 243, 238))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * 640, (index // columns) * 480))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    args = parse_args()
    for attribute in (
        "input",
        "urdf",
        "object_mesh",
        "support_config",
        "output",
        "visual",
    ):
        setattr(args, attribute, getattr(args, attribute).resolve())
    data = np.load(args.input)
    q = np.asarray(data["q"], dtype=np.float32)
    g_t_object = np.asarray(data["object_pose"], dtype=np.float64)
    joint_names = [str(value) for value in data["active_joint_names"].tolist()]

    tree = build_kinematics_tree(str(args.urdf), active_joint_names=joint_names)
    with torch.no_grad():
        fk = (
            batch_fk(tree, torch.from_numpy(q).cuda())["link"]
            .detach()
            .cpu()
            .numpy()
        )
    robot_mesh = RobotMesh(str(args.urdf))
    collision_meshes = exact_link_collision_meshes(robot_mesh, tree)
    link_bvhs = {name: mesh_to_bvh(mesh) for name, mesh in collision_meshes.items()}

    loaded_object = trimesh.load(args.object_mesh, force="mesh")
    if not isinstance(loaded_object, trimesh.Trimesh):
        raise TypeError(f"Expected one object mesh at {args.object_mesh}")
    object_mesh = loaded_object

    config = yaml.safe_load(args.support_config.read_text())
    supports = configured_support_conditions(
        object_mesh, entries=config["supports"]["orientations"]
    )
    broad_supports = [
        support for support in supports if support.symmetry_class == "broad_face"
    ]
    supports_by_id = {support.support_id: support for support in broad_supports}

    trials = []
    for support in broad_supports:
        object_collision = fcl.CollisionObject(
            mesh_to_bvh(object_mesh), fcl_transform(support.support_T_object)
        )
        for candidate_index in range(len(q)):
            support_t_g = support.support_T_object @ np.linalg.inv(
                g_t_object[candidate_index]
            )
            clearance, minimum_link = hand_minimum_z(
                collision_meshes=collision_meshes,
                support_t_g=support_t_g,
                g_t_links=fk[candidate_index],
                tree=tree,
            )
            max_depth, colliding_links, contact_count = collision_metrics(
                object_collision=object_collision,
                link_bvhs=link_bvhs,
                support_t_g=support_t_g,
                g_t_links=fk[candidate_index],
                tree=tree,
            )
            table_clear = clearance >= -args.table_tolerance_m
            penetration_acceptable = max_depth <= args.penetration_tolerance_m
            trials.append(
                {
                    "candidate_index": candidate_index,
                    "support_id": support.support_id,
                    "support_label": support.label,
                    "support_T_G": support_t_g.tolist(),
                    "table_clearance_m": clearance,
                    "minimum_clearance_link": minimum_link,
                    "table_clear": table_clear,
                    "max_object_penetration_m": max_depth,
                    "penetration_acceptable": penetration_acceptable,
                    "colliding_object_links": colliding_links,
                    "object_contact_count": contact_count,
                    "final_geometry_eligible": bool(
                        table_clear and penetration_acceptable
                    ),
                }
            )

    eligible = [trial for trial in trials if trial["final_geometry_eligible"]]
    table_clear_trials = [trial for trial in trials if trial["table_clear"]]
    candidate_eligible = sorted(
        {int(trial["candidate_index"]) for trial in eligible}
    )

    ranked = sorted(
        eligible if eligible else table_clear_trials if table_clear_trials else trials,
        key=lambda trial: (
            trial["max_object_penetration_m"],
            -trial["table_clearance_m"],
            trial["candidate_index"],
        ),
    )
    selected = ranked[: max(0, args.render_count)]

    report = {
        "schema_version": 1,
        "scope": (
            "Lightning-Grasp final articulated hand configurations under the "
            "two broad-face U tabletop supports"
        ),
        "inputs": {
            "result_npz": str(args.input.relative_to(ROOT)),
            "urdf": str(args.urdf.relative_to(ROOT)),
            "object_mesh": str(args.object_mesh.relative_to(ROOT)),
            "support_config": str(args.support_config.relative_to(ROOT)),
            "candidate_count": len(q),
            "joint_names": joint_names,
        },
        "contract": {
            "lightning_output_pose": "G_T_object",
            "support_hand_pose": "support_T_G = support_T_object @ inverse(G_T_object)",
            "table_clearance": "minimum exact URDF collision-mesh vertex z",
            "table_tolerance_m": args.table_tolerance_m,
            "hand_object_measurement": "exact URDF collision meshes against exact U mesh with FCL",
            "penetration_tolerance_m": args.penetration_tolerance_m,
            "important_exclusion": (
                "No pregrasp, approach, closure trajectory, dynamics, or retention test"
            ),
        },
        "summary": {
            "support_count": len(broad_supports),
            "candidate_support_pairs": len(trials),
            "table_clear_pair_count": len(table_clear_trials),
            "final_geometry_eligible_pair_count": len(eligible),
            "final_geometry_eligible_unique_candidate_count": len(candidate_eligible),
            "minimum_table_clearance_m": min(
                trial["table_clearance_m"] for trial in trials
            ),
            "maximum_table_clearance_m": max(
                trial["table_clearance_m"] for trial in trials
            ),
            "minimum_max_object_penetration_m": min(
                trial["max_object_penetration_m"] for trial in trials
            ),
            "maximum_max_object_penetration_m": max(
                trial["max_object_penetration_m"] for trial in trials
            ),
        },
        "selected_for_visual_review": selected,
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    contact_sheet(
        selected=selected,
        q=q,
        fk=fk,
        supports_by_id=supports_by_id,
        object_mesh=object_mesh,
        robot_mesh=robot_mesh,
        tree=tree,
        output=args.visual,
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved: {args.output}")
    if selected:
        print(f"Saved: {args.visual}")


if __name__ == "__main__":
    main()
