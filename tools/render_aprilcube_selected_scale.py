"""Render the selected 45 mm AprilCube task scale beside exact open Dex3.

This is a physical-size visualization only.  It does not place the hand at a
grasp candidate and never renders the terminal close posture against an object.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyrender
import trimesh
import trimesh.transformations as tra
import yourdfpy
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
HAND_ROOT = ROOT / "third_party/GraspGenX/assets/x_grippers/dex3_rev1_right"
PARTS_ROOT = ROOT / "generated/aprilcube_parts"
OUTPUT = ROOT / "docs/assets/aprilcube_45mm_scale.png"
BACKGROUND = (248, 247, 243)
VOXEL_M = 0.045


def material(color: tuple[int, int, int], alpha: float = 1.0):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(*[value / 255.0 for value in color], alpha),
        metallicFactor=0.03,
        roughnessFactor=0.72,
        alphaMode="BLEND" if alpha < 1.0 else "OPAQUE",
    )


def add_mesh(scene: pyrender.Scene, mesh: trimesh.Trimesh, pose: np.ndarray,
             color: tuple[int, int, int] | None = None, alpha: float = 1.0) -> None:
    if color is None:
        rendered = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    else:
        rendered = pyrender.Mesh.from_trimesh(
            mesh, material=material(color, alpha), smooth=False
        )
    scene.add(rendered, pose=pose)


def add_hand(scene: pyrender.Scene, robot: yourdfpy.URDF, pose: np.ndarray) -> None:
    for name, mesh in robot.scene.geometry.items():
        root_to_mesh = robot.scene.graph.get(
            frame_from=robot.scene.graph.base_frame, frame_to=name
        )[0]
        add_mesh(scene, mesh, pose @ root_to_mesh, (77, 89, 103), 0.76)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    backward = eye - target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0.0, 0.0, 1.0]), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    result = np.eye(4)
    result[:3, 0] = right
    result[:3, 1] = up
    result[:3, 2] = backward
    result[:3, 3] = eye
    return result


def render_scene(scene: pyrender.Scene, eye: np.ndarray, target: np.ndarray,
                 width: int = 700, height: int = 570) -> Image.Image:
    camera_pose = look_at(eye, target)
    scene.add(
        pyrender.PerspectiveCamera(yfov=np.deg2rad(34.0), znear=0.01, zfar=5.0),
        pose=camera_pose,
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=3.2),
        pose=camera_pose,
    )
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    return Image.fromarray(rgba, mode="RGBA").convert("RGB")


def scene() -> pyrender.Scene:
    return pyrender.Scene(
        bg_color=np.array([*BACKGROUND, 255], dtype=np.uint8),
        ambient_light=np.array([0.56, 0.56, 0.56, 1.0]),
    )


def load_part(name: str) -> trimesh.Trimesh:
    return trimesh.load(
        PARTS_ROOT / name / "mujoco/cube.obj", force="mesh", process=False
    )


def add_ground(scene_: pyrender.Scene, width: float, depth: float) -> None:
    plane = trimesh.creation.box(extents=[width, depth, 0.004])
    add_mesh(scene_, plane, tra.translation_matrix([0.0, 0.0, -0.004]),
             (221, 215, 202))


def loose_parts(parts: dict[str, trimesh.Trimesh]) -> Image.Image:
    view = scene()
    add_ground(view, 0.52, 0.24)
    placements = (
        ("t_body", -0.17, 0.090),
        ("u_legs", 0.02, 0.0675),
        ("cube_head", 0.16, 0.0225),
    )
    for name, x, z in placements:
        add_mesh(view, parts[name], tra.translation_matrix([x, 0.0, z]))
    return render_scene(
        view, np.array([0.48, -0.72, 0.37]), np.array([0.0, 0.0, 0.09])
    )


def assembled_with_hand(parts: dict[str, trimesh.Trimesh],
                        robot: yourdfpy.URDF) -> Image.Image:
    view = scene()
    add_ground(view, 0.52, 0.24)
    add_hand(view, robot, tra.translation_matrix([-0.15, 0.0, 0.0]))
    x = 0.095
    add_mesh(view, parts["u_legs"], tra.translation_matrix([x, 0.0, 1.5 * VOXEL_M]))
    add_mesh(view, parts["t_body"], tra.translation_matrix([x, 0.0, 5.0 * VOXEL_M]))
    add_mesh(view, parts["cube_head"], tra.translation_matrix([x, 0.0, 7.5 * VOXEL_M]))
    return render_scene(
        view, np.array([0.53, -0.78, 0.43]), np.array([0.0, 0.0, 0.18])
    )


def label(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    panel = Image.new("RGB", (image.width, image.height + 82), BACKGROUND)
    panel.paste(image, (0, 82))
    draw = ImageDraw.Draw(panel)
    draw.text((17, 11), title, fill=(24, 43, 59),
              font=ImageFont.truetype("DejaVuSans-Bold.ttf", 24))
    draw.text((17, 48), subtitle, fill=(68, 76, 84),
              font=ImageFont.truetype("DejaVuSans.ttf", 16))
    return panel


def compose(left: Image.Image, right: Image.Image) -> None:
    margin, header, footer = 20, 118, 96
    width = left.width + right.width + 3 * margin
    height = header + left.height + 2 * margin + footer
    sheet = Image.new("RGB", (width, height), (235, 233, 227))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 15), "Selected physical scale: 45 mm AprilCube voxel",
              fill=(21, 39, 55), font=ImageFont.truetype("DejaVuSans-Bold.ttf", 33))
    draw.text((margin, 63),
              "exact generated parts and exact current open Dex3 shown at one metric scale",
              fill=(69, 76, 84), font=ImageFont.truetype("DejaVuSans.ttf", 18))
    sheet.paste(left, (margin, header))
    sheet.paste(right, (2 * margin + left.width, header))
    y = header + left.height + margin
    draw.rounded_rectangle((margin, y, width - margin, height - 16), radius=14,
                           fill=(255, 248, 226))
    draw.text((margin + 16, y + 10),
              "T 135×45×180 mm · U 135×45×135 mm · head 45 mm · assembled height 360 mm",
              fill=(104, 68, 9), font=ImageFont.truetype("DejaVuSans-Bold.ttf", 17))
    draw.text((margin + 16, y + 42),
              "SIZE VIEW ONLY — no grasp pose, closure, contact, magnet gap, or robot reachability is asserted",
              fill=(126, 75, 25), font=ImageFont.truetype("DejaVuSans.ttf", 16))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT)


def main() -> None:
    config = json.loads((HAND_ROOT / "config.json").read_text())
    robot = yourdfpy.URDF.load(
        str(HAND_ROOT / "gripper.urdf"),
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    robot.update_cfg(config["open"])
    parts = {name: load_part(name) for name in ("t_body", "u_legs", "cube_head")}
    left = label(loose_parts(parts), "Loose task parts", "actual print geometry · all upright only for scale")
    right = label(assembled_with_hand(parts, robot), "Completed figure + Dex3", "same metric scale · open hand is a reference, not a grasp")
    compose(left, right)
    print(OUTPUT)


if __name__ == "__main__":
    main()
