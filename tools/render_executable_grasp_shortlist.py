#!/usr/bin/env python3
"""Render every execution-qualified grasp before cuRobo chooses among them.

Each panel shows the unchanged GraspGenX proposal with the descriptor-open
Dex3, followed by the actual Isaac ``closed_before_tug`` relative state.  The
closed view is not a guessed kinematic endpoint: its joint positions and
object-to-grasp transform are copied from the recorded physics trace.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrender
import trimesh
import trimesh.transformations as tra
import yaml
import yourdfpy


ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = (248, 247, 243)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def material(color: tuple[int, int, int], alpha: float = 1.0):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(*[channel / 255.0 for channel in color], alpha),
        metallicFactor=0.03,
        roughnessFactor=0.72,
        alphaMode="BLEND" if alpha < 1.0 else "OPAQUE",
    )


def pose_matrix(document: dict) -> np.ndarray:
    orientation = document["orientation"]
    matrix = tra.quaternion_matrix(
        [orientation["w"], *orientation["xyz"]]
    )
    matrix[:3, 3] = document["position"]
    return matrix


def add_mesh(
    scene: pyrender.Scene,
    mesh: trimesh.Trimesh,
    pose: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 1.0,
) -> None:
    scene.add(
        pyrender.Mesh.from_trimesh(
            mesh,
            material=material(color, alpha),
            smooth=False,
        ),
        pose=pose,
    )


def add_hand(
    scene: pyrender.Scene,
    robot: yourdfpy.URDF,
    joints: dict[str, float],
    object_T_G: np.ndarray,
) -> None:
    robot.update_cfg(joints)
    for name, mesh in robot.scene.geometry.items():
        G_T_mesh = robot.scene.graph.get(
            frame_from=robot.scene.graph.base_frame,
            frame_to=name,
        )[0]
        add_mesh(scene, mesh, object_T_G @ G_T_mesh, (67, 78, 91))


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    backward = eye - target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.asarray([0.0, 0.0, 1.0]), backward)
    if np.linalg.norm(right) < 1.0e-8:
        right = np.asarray([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = backward
    pose[:3, 3] = eye
    return pose


def render_state(
    *,
    object_mesh: trimesh.Trimesh,
    world_T_object: np.ndarray,
    robot: yourdfpy.URDF,
    joints: dict[str, float],
    object_T_G: np.ndarray,
    support_z: float,
) -> Image.Image:
    scene = pyrender.Scene(
        bg_color=np.asarray([*BACKGROUND, 255], dtype=np.uint8),
        ambient_light=np.asarray([0.58, 0.58, 0.58, 1.0]),
    )
    table = trimesh.creation.box(extents=[0.24, 0.24, 0.012])
    table_pose = tra.translation_matrix([0.0, 0.0, support_z - 0.006])
    add_mesh(scene, table, table_pose, (183, 145, 106))
    add_mesh(scene, object_mesh, world_T_object, (228, 154, 55))
    add_hand(scene, robot, joints, object_T_G)

    bounds = scene.bounds
    target = 0.5 * (bounds[0] + bounds[1])
    diagonal = max(float(np.linalg.norm(bounds[1] - bounds[0])), 0.18)
    direction = np.asarray([0.82, -1.0, 0.72])
    direction /= np.linalg.norm(direction)
    camera_pose = look_at(target + direction * diagonal * 1.45, target)
    scene.add(
        pyrender.PerspectiveCamera(yfov=np.deg2rad(36.0), znear=0.01, zfar=3.0),
        pose=camera_pose,
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
        pose=camera_pose,
    )
    fill = look_at(target + np.asarray([-0.3, 0.2, 0.4]), target)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=1.4),
        pose=fill,
    )
    renderer = pyrender.OffscreenRenderer(620, 470)
    try:
        color, _ = renderer.render(
            scene,
            flags=pyrender.RenderFlags.RGBA
            | pyrender.RenderFlags.SKIP_CULL_FACES,
        )
    finally:
        renderer.delete()
    return Image.fromarray(color, mode="RGBA").convert("RGB")


def candidate_panel(
    *,
    index: int,
    candidate: dict,
    object_mesh: trimesh.Trimesh,
    robot: yourdfpy.URDF,
    open_q: dict[str, float],
    support_z: float,
) -> Image.Image:
    evidence = candidate["execution_evidence"]
    open_image = render_state(
        object_mesh=object_mesh,
        world_T_object=np.eye(4),
        robot=robot,
        joints=open_q,
        object_T_G=pose_matrix(candidate["object_T_G"]),
        support_z=support_z,
    )
    closed_image = render_state(
        object_mesh=object_mesh,
        world_T_object=np.asarray(evidence["isaac_closed_world_T_object"]),
        robot=robot,
        joints=evidence["isaac_closed_q"],
        object_T_G=pose_matrix(candidate["object_T_G"]),
        support_z=support_z,
    )

    panel = Image.new("RGB", (1240, 610), BACKGROUND)
    panel.paste(open_image, (0, 105))
    panel.paste(closed_image, (620, 105))
    draw = ImageDraw.Draw(panel)
    short_id = candidate["candidate_id"].replace("cube_head__", "")
    draw.text((22, 14), f"#{index + 1}  {short_id}", fill=(28, 43, 58), font=font(24, True))
    draw.text(
        (22, 52),
        f"GraspGenX score {candidate['graspgenx_score']:.3f}   ·   "
        f"closure moved cube {1000.0 * evidence['closure_translation_m']:.1f} mm, "
        f"{evidence['closure_rotation_deg']:.1f}°",
        fill=(75, 86, 98),
        font=font(17),
    )
    draw.text((18, 566), "UNCHANGED PROPOSAL · descriptor-open hand", fill=(45, 61, 76), font=font(16, True))
    draw.text((638, 566), "RECORDED ISAAC STATE · immediately after closure", fill=(45, 61, 76), font=font(16, True))
    forces = evidence["closure_contact_group_max_force_N"]
    metrics = (
        f"thumb/opposing contact {forces['thumb']:.2f}/{forces['opposing']:.2f} N   ·   "
        f"closed hand/object table clearance "
        f"{1000.0 * evidence['closed_hand_table_clearance_m']:.1f}/"
        f"{1000.0 * evidence['closed_object_table_clearance_m']:.1f} mm"
    )
    draw.text((22, 79), metrics, fill=(75, 86, 98), font=font(15))
    return panel


def contact_sheet(panels: list[Image.Image], output: Path) -> None:
    thumb_width = 820
    thumbs = []
    for panel in panels:
        height = round(panel.height * thumb_width / panel.width)
        thumbs.append(panel.resize((thumb_width, height), Image.Resampling.LANCZOS))
    rows = math.ceil(len(thumbs) / 2)
    header = 110
    sheet = Image.new(
        "RGB",
        (1640, header + rows * thumbs[0].height),
        (237, 235, 230),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 16), "Cube executable-grasp shortlist", fill=(25, 41, 56), font=font(32, True))
    draw.text(
        (24, 62),
        "Every pose below enters the same cuRobo goal set; no family labels and no visual ranking.",
        fill=(73, 82, 91),
        font=font(18),
    )
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 2) * thumb_width, header + (index // 2) * thumb.height))
    sheet.save(output, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shortlist",
        type=Path,
        default=ROOT / "artifacts/grasp_shortlists/cube_right_executable_v1/shortlist.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/grasp_shortlists/cube_right_executable_v1/visual",
    )
    args = parser.parse_args()
    shortlist = yaml.safe_load(args.shortlist.resolve().read_text())
    config_path = ROOT / shortlist["source"]["config"]
    config = yaml.safe_load(config_path.read_text())
    object_mesh = trimesh.load(ROOT / shortlist["object_mesh"], force="mesh", process=False)
    hand_root = (ROOT / config["hand"]["urdf"]).parent
    robot = yourdfpy.URDF.load(
        str(hand_root / "gripper.urdf"),
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    descriptor = json.loads((hand_root / "config.json").read_text())
    support_z = float(object_mesh.bounds[0, 2])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    panels = []
    for index, candidate in enumerate(shortlist["candidates"]):
        panel = candidate_panel(
            index=index,
            candidate=candidate,
            object_mesh=object_mesh,
            robot=robot,
            open_q=descriptor["open"],
            support_z=support_z,
        )
        panel.save(output / f"candidate_{index + 1:02d}.png", quality=94)
        panels.append(panel)
        print(f"Rendered {index + 1}/{len(shortlist['candidates'])}", flush=True)
    contact_sheet(panels, output / "contact_sheet.png")
    print(output / "contact_sheet.png")


if __name__ == "__main__":
    main()
