#!/usr/bin/env python3
"""Render the exact scene that the first cuRobo planning gate will consume."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrender
import trimesh
import yourdfpy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g1_aprilcube_demo.planning import load_planning_scene


DEFAULT_SCENE = ROOT / "config/planning/unibot_seated_aprilcube_v1.yaml"
BACKGROUND = (247, 246, 242)


def material(color: tuple[int, int, int], roughness: float = 0.72):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=tuple(value / 255.0 for value in color) + (1.0,),
        metallicFactor=0.02,
        roughnessFactor=roughness,
    )


def add_mesh(
    scene: pyrender.Scene,
    mesh: trimesh.Trimesh,
    pose: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    scene.add(
        pyrender.Mesh.from_trimesh(
            mesh, material=material(color), smooth=False
        ),
        pose=pose,
    )


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    backward = eye - target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0.0, 0.0, 1.0]), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = backward
    pose[:3, 3] = eye
    return pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = load_planning_scene(args.scene)
    output = args.output.resolve() if args.output else spec.output

    robot = yourdfpy.URDF.load(
        str(spec.urdf),
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    robot.update_cfg(dict(spec.start_joint_positions))

    view = pyrender.Scene(
        bg_color=np.array([*BACKGROUND, 255], dtype=np.uint8),
        ambient_light=np.array([0.52, 0.52, 0.52, 1.0]),
    )
    for name, mesh in robot.scene.geometry.items():
        base_T_mesh = robot.scene.graph.get(
            frame_from=robot.scene.graph.base_frame, frame_to=name
        )[0]
        add_mesh(view, mesh, base_T_mesh, (161, 168, 174))

    robot_floor_z = float(robot.scene.bounds[0, 2])
    floor = trimesh.creation.box(extents=[2.0, 2.0, 0.012])
    add_mesh(
        view,
        floor,
        trimesh.transformations.translation_matrix([0.35, 0.0, robot_floor_z - 0.008]),
        (221, 217, 208),
    )

    table_size = [*spec.table_size_xy_m, spec.table_thickness_m]
    table = trimesh.creation.box(extents=table_size)
    table_pose = trimesh.transformations.translation_matrix(
        [
            spec.table_center_xy_m[0],
            spec.table_center_xy_m[1],
            spec.table_top_z_m - 0.5 * spec.table_thickness_m,
        ]
    )
    add_mesh(view, table, table_pose, (125, 88, 58))

    leg_height = spec.table_top_z_m - spec.table_thickness_m - robot_floor_z
    if leg_height > 0.05:
        inset = 0.055
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                x = spec.table_center_xy_m[0] + sx * (
                    0.5 * spec.table_size_xy_m[0] - inset
                )
                y = spec.table_center_xy_m[1] + sy * (
                    0.5 * spec.table_size_xy_m[1] - inset
                )
                leg = trimesh.creation.box(extents=[0.045, 0.045, leg_height])
                pose = trimesh.transformations.translation_matrix(
                    [x, y, robot_floor_z + 0.5 * leg_height]
                )
                add_mesh(view, leg, pose, (72, 68, 64))

    part_transforms = spec.world_part_transforms()
    for part_id, part in spec.parts.items():
        mesh = trimesh.load(part.mesh, force="mesh", process=False)
        add_mesh(view, mesh, part_transforms[part_id], part.color_rgb)

    camera_pose = look_at(
        np.asarray(spec.camera_eye_m), np.asarray(spec.camera_target_m)
    )
    view.add(
        pyrender.PerspectiveCamera(yfov=np.deg2rad(38.0), znear=0.01, zfar=6.0),
        pose=camera_pose,
    )
    view.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=3.4),
        pose=camera_pose,
    )
    fill_pose = look_at(np.array([0.1, 1.2, 1.25]), np.array([0.2, 0.0, 0.0]))
    view.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=1.8),
        pose=fill_pose,
    )

    width, height = spec.image_size_px
    renderer = pyrender.OffscreenRenderer(width, height)
    rgba, _ = renderer.render(view, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    image = Image.fromarray(rgba, mode="RGBA").convert("RGB")

    header = 118
    result = Image.new("RGB", (width, height + header), BACKGROUND)
    result.paste(image, (0, header))
    draw = ImageDraw.Draw(result)
    draw.text(
        (28, 18),
        "UniBot-seeded G1 assembly planning scene",
        fill=(24, 43, 59),
        font=ImageFont.truetype("DejaVuSans-Bold.ttf", 34),
    )
    draw.text(
        (29, 67),
        "observed seated joint state · level table reference · actual 45 mm AprilCube T / U / cube meshes",
        fill=(69, 76, 84),
        font=ImageFont.truetype("DejaVuSans.ttf", 19),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    print(output)
    print(f"planning frame: {spec.planning_frame}")
    print(f"Dex3-clear table top z: {spec.table_top_z_m:.8f} m")
    print(f"robot floor z: {robot_floor_z:.8f} m")
    for part_id, transform in part_transforms.items():
        print(f"{part_id}: xyz={transform[:3, 3].tolist()}")


if __name__ == "__main__":
    main()
