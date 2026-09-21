"""Render the official Unitree Dex3 +X axis and an Rx(pi) roll sequence.

The source URDF defaults to the locally checked-out official Unitree
``xr_teleoperate`` repository. Pass ``--unitree-hand-dir`` to use another
checkout of the same assets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyrender
import trimesh
import trimesh.transformations as tra
import yourdfpy
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAND_DIR = Path(
    "/home/srinivas/Desktop/g1pilot-workspace/reference_repos/"
    "xr_teleoperate/assets/unitree_hand"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/assets/dex3_rx_pi_explainer.png"
BACKGROUND = (248, 247, 243)


def material(color: tuple[int, int, int], alpha: float = 1.0):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(*[channel / 255.0 for channel in color], alpha),
        metallicFactor=0.03,
        roughnessFactor=0.72,
        alphaMode="BLEND" if alpha < 1.0 else "OPAQUE",
    )


def add_mesh(
    scene: pyrender.Scene,
    mesh: trimesh.Trimesh,
    pose: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 1.0,
):
    scene.add(
        pyrender.Mesh.from_trimesh(
            mesh,
            material=material(color, alpha),
            smooth=False,
        ),
        pose=pose,
    )


def load_hand_in_palm_frame(hand_dir: Path, side: str) -> trimesh.Trimesh:
    urdf_path = hand_dir / f"unitree_dex3_{side}.urdf"
    robot = yourdfpy.URDF.load(
        str(urdf_path),
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    base = robot.scene.graph.base_frame
    palm = f"{side}_hand_palm_link"
    base_t_palm = robot.scene.graph.get(frame_from=base, frame_to=palm)[0]
    palm_t_base = np.linalg.inv(base_t_palm)

    parts: list[trimesh.Trimesh] = []
    prefix = f"{side}_hand_"
    for geometry_name, geometry in robot.scene.geometry.items():
        if not geometry_name.startswith(prefix):
            continue
        base_t_geometry = robot.scene.graph.get(
            frame_from=base,
            frame_to=geometry_name,
        )[0]
        part = geometry.copy()
        part.apply_transform(palm_t_base @ base_t_geometry)
        parts.append(part)
    if not parts:
        raise RuntimeError(f"No {side} hand geometry found in {urdf_path}")
    return trimesh.util.concatenate(parts)


def align_z_to(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    vector /= np.linalg.norm(vector)
    return trimesh.geometry.align_vectors([0.0, 0.0, 1.0], vector)


def add_arrow(
    scene: pyrender.Scene,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    radius: float = 0.0022,
):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    vector = end - start
    length = float(np.linalg.norm(vector))
    direction = vector / length
    head_length = min(0.018, length * 0.22)
    shaft_length = length - head_length
    orientation = align_z_to(direction)

    shaft = trimesh.creation.cylinder(radius=radius, height=shaft_length, sections=24)
    shaft_pose = orientation.copy()
    shaft_pose[:3, 3] = start + direction * (shaft_length / 2.0)
    add_mesh(scene, shaft, shaft_pose, color)

    head = trimesh.creation.cone(radius=radius * 2.5, height=head_length, sections=24)
    head_pose = orientation.copy()
    head_pose[:3, 3] = start + direction * shaft_length
    add_mesh(scene, head, head_pose, color)


def add_axes(scene: pyrender.Scene, long_x: bool):
    add_arrow(
        scene,
        np.array([-0.025, 0.0, 0.0]),
        np.array([0.225 if long_x else 0.115, 0.0, 0.0]),
        (216, 64, 55),
        radius=0.0028,
    )
    if long_x:
        add_arrow(scene, np.zeros(3), np.array([0.0, 0.075, 0.0]), (42, 153, 83))
        add_arrow(scene, np.zeros(3), np.array([0.0, 0.0, 0.075]), (54, 108, 190))
        origin = trimesh.creation.icosphere(radius=0.005, subdivisions=2)
        add_mesh(scene, origin, np.eye(4), (30, 36, 43))


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    backward = eye - target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0.0, 0.0, 1.0]), backward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = backward
    pose[:3, 3] = eye
    return pose


def render_hand(
    hand: trimesh.Trimesh,
    angle_deg: float,
    show_all_axes: bool,
    width: int,
    height: int,
) -> Image.Image:
    scene = pyrender.Scene(
        bg_color=np.array([*BACKGROUND, 255], dtype=np.uint8),
        ambient_light=np.array([0.58, 0.58, 0.58, 1.0]),
    )
    hand_pose = tra.rotation_matrix(np.deg2rad(angle_deg), [1.0, 0.0, 0.0])
    add_mesh(scene, hand, hand_pose, (79, 155, 188))
    add_axes(scene, long_x=show_all_axes)

    camera_pose = look_at(
        np.array([0.31, -0.31, 0.235]),
        np.array([0.080, 0.010, 0.0]),
    )
    scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(38.0)), pose=camera_pose)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=3.2),
        pose=camera_pose,
    )
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    return Image.fromarray(rgba, mode="RGBA").convert("RGB")


def title_panel(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    panel = Image.new("RGB", (image.width, image.height + 88), BACKGROUND)
    panel.paste(image, (0, 88))
    draw = ImageDraw.Draw(panel)
    draw.text(
        (22, 14),
        title,
        fill=(24, 43, 59),
        font=ImageFont.truetype("DejaVuSans-Bold.ttf", 25),
    )
    draw.text(
        (22, 50),
        subtitle,
        fill=(68, 76, 84),
        font=ImageFont.truetype("DejaVuSans.ttf", 17),
    )
    return panel


def compose(axis_panel: Image.Image, sequence: list[Image.Image], output: Path):
    margin = 24
    header = 112
    sequence_width = 370
    sequence_height = 300
    sheet_width = axis_panel.width + sequence_width * 2 + margin * 4
    sheet_height = header + max(axis_panel.height, sequence_height * 2 + margin) + margin * 2
    sheet = Image.new("RGB", (sheet_width, sheet_height), (235, 233, 227))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (margin, 17),
        "Dex3  Rx(180°): a half-roll around the palm's +X axis",
        fill=(21, 39, 55),
        font=ImageFont.truetype("DejaVuSans-Bold.ttf", 34),
    )
    draw.text(
        (margin, 66),
        "The red axis stays fixed; the hand rolls around it from 0° to 180°.",
        fill=(69, 76, 84),
        font=ImageFont.truetype("DejaVuSans.ttf", 20),
    )
    y0 = header + margin
    sheet.paste(axis_panel, (margin, y0))
    x0 = margin * 2 + axis_panel.width
    positions = [
        (x0, y0),
        (x0 + sequence_width + margin, y0),
        (x0, y0 + sequence_height + margin),
        (x0 + sequence_width + margin, y0 + sequence_height + margin),
    ]
    for panel, position in zip(sequence, positions, strict=True):
        sheet.paste(panel, position)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unitree-hand-dir", type=Path, default=DEFAULT_HAND_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    right_hand = load_hand_in_palm_frame(args.unitree_hand_dir, "right")
    axis = title_panel(
        render_hand(right_hand, 0.0, True, 650, 625),
        "+X is the wrist-to-fingertips direction",
        "red +X · green +Y · blue +Z · black dot = palm-frame origin",
    )
    sequence = []
    for angle in (0, 60, 120, 180):
        image = render_hand(right_hand, float(angle), False, 370, 244)
        subtitle = "start" if angle == 0 else ("same geometry as official left hand" if angle == 180 else "continue rolling")
        sequence.append(title_panel(image, f"{angle}° about +X", subtitle))
    compose(axis, sequence, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
