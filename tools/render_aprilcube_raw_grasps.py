"""Render raw GraspGenX frames with the exact *open* Dex3 geometry.

The fixed terminal close pose is intentionally not rendered against an object.
It would pass through an object that should stop the real fingers on contact and
therefore cannot establish grasp validity, contact quality, or object scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyrender
import trimesh
import trimesh.transformations as tra
import yaml
import yourdfpy
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARTS_ROOT = PROJECT_ROOT / "generated/aprilcube_parts"
GRASPS_ROOT = PROJECT_ROOT / "artifacts/aprilcube_raw_grasps"
HAND_ROOT = (
    PROJECT_ROOT
    / "third_party/GraspGenX/assets/x_grippers/dex3_rev1_right"
)
OUTPUT = PROJECT_ROOT / "docs/assets/aprilcube_raw_grasp_audit.png"
PARTS = (
    ("t_body", "T BODY · holder part", (70, 163, 188)),
    ("u_legs", "U LEGS · first attachment", (91, 167, 118)),
    ("cube_head", "CUBE HEAD · final attachment", (222, 155, 57)),
)
BACKGROUND = (248, 247, 243)


def material(color: tuple[int, int, int], alpha: float = 1.0):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(*[channel / 255.0 for channel in color], alpha),
        metallicFactor=0.03,
        roughnessFactor=0.72,
        alphaMode="BLEND" if alpha < 1.0 else "OPAQUE",
    )


def add_mesh(scene: pyrender.Scene, mesh: trimesh.Trimesh, pose: np.ndarray,
             color: tuple[int, int, int] | None = None, alpha: float = 1.0):
    if color is None:
        rendered = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    else:
        rendered = pyrender.Mesh.from_trimesh(
            mesh, material=material(color, alpha), smooth=False
        )
    scene.add(rendered, pose=pose)


def add_hand(scene: pyrender.Scene, robot: yourdfpy.URDF, joints: dict,
             root_pose: np.ndarray, alpha: float):
    robot.update_cfg(joints)
    for name, mesh in robot.scene.geometry.items():
        root_to_mesh = robot.scene.graph.get(
            frame_from=robot.scene.graph.base_frame, frame_to=name
        )[0]
        add_mesh(scene, mesh, root_pose @ root_to_mesh, (79, 91, 105), alpha)


def add_frame(scene: pyrender.Scene, pose: np.ndarray, scale: float = 0.027):
    axis = trimesh.creation.axis(
        transform=pose, origin_size=0.0025, axis_radius=0.0008, axis_length=scale
    )
    for mesh in axis.dump() if isinstance(axis, trimesh.Scene) else [axis]:
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))


def add_sweep_box(scene: pyrender.Scene, pose: np.ndarray, sweep: dict):
    box = trimesh.creation.box(extents=np.asarray(sweep["extents"]))
    transform = pose @ tra.translation_matrix(sweep["offset"])
    add_mesh(scene, box, transform, (40, 165, 220), 0.12)


def load_grasps(path: Path) -> list[tuple[np.ndarray, float]]:
    data = yaml.safe_load(path.read_text())["grasps"]
    result = []
    for entry in data.values():
        quaternion = entry["orientation"]
        transform = tra.quaternion_matrix(
            [quaternion["w"], *quaternion["xyz"]]
        )
        transform[:3, 3] = entry["position"]
        result.append((transform, float(entry["confidence"])))
    return result


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    backward = eye - target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0.0, 0.0, 1.0]), backward)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = backward
    pose[:3, 3] = eye
    return pose


def camera_from_scene(scene: pyrender.Scene, direction: np.ndarray) -> np.ndarray:
    bounds = scene.bounds
    center = bounds.mean(axis=0)
    radius = max(float(np.linalg.norm(bounds[1] - bounds[0]) / 2.0), 0.06)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    return look_at(center + direction * radius * 2.9, center)


def render_panel(
    object_mesh: trimesh.Trimesh,
    robot: yourdfpy.URDF,
    state: dict | None,
    candidates: list[tuple[np.ndarray, float]],
    show_many_frames: bool,
    show_sweep: bool,
    camera_direction: np.ndarray,
) -> Image.Image:
    scene = pyrender.Scene(
        bg_color=np.array([*BACKGROUND, 255], dtype=np.uint8),
        ambient_light=np.array([0.55, 0.55, 0.55, 1.0]),
    )
    add_mesh(scene, object_mesh, np.eye(4))
    if show_many_frames:
        for pose, _score in candidates[:5]:
            add_frame(scene, pose)
    else:
        best_pose = candidates[0][0]
        add_hand(scene, robot, state, best_pose, 0.62)
        add_frame(scene, best_pose, 0.035)
        if show_sweep:
            add_sweep_box(scene, best_pose, CONFIG["sweep_volume"])

    camera_pose = camera_from_scene(scene, camera_direction)
    scene.add(
        pyrender.PerspectiveCamera(yfov=np.deg2rad(35), znear=0.01, zfar=5.0),
        pose=camera_pose,
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=3.1),
        pose=camera_pose,
    )
    renderer = pyrender.OffscreenRenderer(viewport_width=430, viewport_height=350)
    rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    return Image.fromarray(rgba, mode="RGBA").convert("RGB")


def label(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    panel = Image.new("RGB", (image.width, image.height + 72), BACKGROUND)
    panel.paste(image, (0, 72))
    draw = ImageDraw.Draw(panel)
    draw.text((16, 10), title, fill=(24, 43, 59),
              font=ImageFont.truetype("DejaVuSans-Bold.ttf", 20))
    draw.text((16, 41), subtitle, fill=(68, 76, 84),
              font=ImageFont.truetype("DejaVuSans.ttf", 14))
    return panel


def compose(rows: list[list[Image.Image]]) -> None:
    margin, header, footer = 18, 112, 92
    panel_w, panel_h = rows[0][0].size
    width = panel_w * 4 + margin * 5
    height = panel_h * 3 + margin * 4 + header + footer
    sheet = Image.new("RGB", (width, height), (235, 233, 227))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 15), "Raw GraspGenX proposals on the actual AprilCube parts",
              fill=(21, 39, 55), font=ImageFont.truetype("DejaVuSans-Bold.ttf", 32))
    draw.text((margin, 61),
              "one canonical candidate set serves both Dex3 hands · rendered with exact current hand geometry",
              fill=(69, 76, 84), font=ImageFont.truetype("DejaVuSans.ttf", 18))
    y0 = header + margin
    for row_index, row in enumerate(rows):
        for column, panel in enumerate(row):
            sheet.paste(
                panel,
                (margin + column * (panel_w + margin), y0 + row_index * (panel_h + margin)),
            )
    footer_y = height - footer + 8
    draw.rounded_rectangle(
        (margin, footer_y, width - margin, height - 16), radius=14, fill=(255, 239, 228)
    )
    draw.text((margin + 16, footer_y + 10),
              "RAW ≠ VALID — open hand only; no candidate has passed contact-aware closure, collision, approach, IK, or table checks",
              fill=(133, 58, 29), font=ImageFont.truetype("DejaVuSans-Bold.ttf", 17))
    draw.text((margin + 16, footer_y + 40),
              "No terminal-close overlay is shown: a fixed endpoint can penetrate an object that should stop the real fingers",
              fill=(91, 67, 55), font=ImageFont.truetype("DejaVuSans.ttf", 15))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT)


CONFIG = json.loads((HAND_ROOT / "config.json").read_text())


def main() -> None:
    robot = yourdfpy.URDF.load(
        str(HAND_ROOT / "gripper.urdf"),
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    rows = []
    for part, part_title, _color in PARTS:
        object_mesh = trimesh.load(
            PARTS_ROOT / part / "mujoco/cube.obj", force="mesh", process=False
        )
        candidates = load_grasps(GRASPS_ROOT / f"{part}.yaml")
        best_score = candidates[0][1]
        rows.append(
            [
                label(
                    render_panel(object_mesh, robot, None, candidates, True, False,
                                 np.array([1.2, -1.5, 1.0])),
                    part_title,
                    f"top 5 returned frames · best score {best_score:.3f}",
                ),
                label(
                    render_panel(object_mesh, robot, CONFIG["open"], candidates, False, True,
                                 np.array([1.2, -1.5, 1.0])),
                    "BEST RAW · open",
                    "blue box is descriptor conditioning, not contact",
                ),
                label(
                    render_panel(object_mesh, robot, CONFIG["open"], candidates, False, False,
                                 np.array([0.0, -1.0, 0.18])),
                    "BEST RAW · open side",
                    "scale and clearance view only",
                ),
                label(
                    render_panel(object_mesh, robot, CONFIG["open"], candidates, False, False,
                                 np.array([-1.0, 1.2, 0.8])),
                    "BEST RAW · open reverse",
                    "returned frame, not a validated grasp",
                ),
            ]
        )
    compose(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
