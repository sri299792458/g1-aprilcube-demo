"""Render the stock GraspGenX Unitree descriptor and one saved candidate.

The script does not import project manipulation code. It consumes only:

* the pinned GraspGenX checkout;
* its managed gripper_descriptions checkout;
* a YAML emitted by the upstream object-mesh demo.

Set ``GRASPGENX_GRIPPER_CFG_DIR`` when gripper assets live in a shared cache.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyrender
import trimesh
import trimesh.transformations as tra
import yaml
import yourdfpy
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPGENX_ROOT = PROJECT_ROOT / "third_party/GraspGenX"
GRIPPER_DESCRIPTIONS_ROOT = Path(
    os.environ.get(
        "GRASPGENX_GRIPPER_CFG_DIR",
        GRASPGENX_ROOT / "ext/gripper_descriptions",
    )
)
HAND_DIR = (
    GRIPPER_DESCRIPTIONS_ROOT
    / "gripper_descriptions/assets/x_grippers/unitree_g1"
)
URDF = HAND_DIR / "gripper.urdf"
CONFIG = HAND_DIR / "config.json"
BOX = GRASPGENX_ROOT / "assets/sample_data/object_mesh/box.obj"
GRASPS = PROJECT_ROOT / "artifacts/upstream_probe/unitree_box_grasps.yml"
OUTPUT_DIR = PROJECT_ROOT / "docs/assets"
OUTPUT = OUTPUT_DIR / "graspgenx_unitree_upstream_probe.png"
OUTPUT_MULTIVIEW = OUTPUT_DIR / "graspgenx_unitree_candidate_multiview.png"


def matrix_from_saved_grasp(entry: dict) -> np.ndarray:
    q = entry["orientation"]
    transform = tra.quaternion_matrix([q["w"], *q["xyz"]])
    transform[:3, 3] = entry["position"]
    return transform


def material(color: tuple[int, int, int], alpha: float = 1.0):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(*[c / 255.0 for c in color], alpha),
        metallicFactor=0.05,
        roughnessFactor=0.72,
        alphaMode="BLEND" if alpha < 1.0 else "OPAQUE",
    )


def add_trimesh(
    scene: pyrender.Scene,
    mesh: trimesh.Trimesh,
    pose: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 1.0,
):
    scene.add(
        pyrender.Mesh.from_trimesh(mesh, material=material(color, alpha), smooth=False),
        pose=pose,
    )


def add_frame(scene: pyrender.Scene, transform: np.ndarray, length: float, palm: bool):
    axis = trimesh.creation.axis(
        transform=transform,
        origin_size=0.004 if palm else 0.005,
        axis_radius=0.0012,
        axis_length=length,
    )
    meshes = axis.dump() if isinstance(axis, trimesh.Scene) else [axis]
    for mesh in meshes:
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    sphere = trimesh.creation.icosphere(radius=0.005 if palm else 0.006, subdivisions=2)
    add_trimesh(
        scene,
        sphere,
        tra.translation_matrix(transform[:3, 3]),
        (194, 79, 203) if palm else (25, 25, 25),
    )


def add_hand(
    scene: pyrender.Scene,
    robot: yourdfpy.URDF,
    joints: dict,
    root_pose: np.ndarray,
    color: tuple[int, int, int],
):
    robot.update_cfg(joints)
    for geometry_name, mesh in robot.scene.geometry.items():
        root_to_geometry = robot.scene.graph.get(geometry_name)[0]
        add_trimesh(scene, mesh, root_pose @ root_to_geometry, color)


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    backward = eye - target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0.0, 0.0, 1.0]), backward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = backward
    pose[:3, 3] = eye
    return pose


def render_panel(
    robot: yourdfpy.URDF,
    joints: dict,
    root_pose: np.ndarray,
    palm_from_root: np.ndarray,
    hand_color: tuple[int, int, int],
    box_mesh: trimesh.Trimesh | None,
    show_sweep: bool,
    camera_eye: np.ndarray,
    camera_target: np.ndarray,
) -> Image.Image:
    scene = pyrender.Scene(
        bg_color=np.array([248, 247, 243, 255], dtype=np.uint8),
        ambient_light=np.array([0.55, 0.55, 0.55, 1.0]),
    )
    add_hand(scene, robot, joints, root_pose, hand_color)
    if box_mesh is not None:
        add_trimesh(scene, box_mesh, np.eye(4), (215, 177, 113), 0.48)
    add_frame(scene, root_pose, 0.045, palm=False)
    add_frame(scene, root_pose @ palm_from_root, 0.03, palm=True)

    if show_sweep:
        with CONFIG.open() as stream:
            sweep = json.load(stream)["sweep_volume"]
        sweep_box = trimesh.creation.box(extents=np.asarray(sweep["extents"]))
        add_trimesh(
            scene,
            sweep_box,
            root_pose @ tra.translation_matrix(sweep["offset"]),
            (70, 210, 225),
            0.20,
        )

    camera_pose = look_at(camera_eye, camera_target)
    scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(38.0)), pose=camera_pose)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
        pose=camera_pose,
    )
    renderer = pyrender.OffscreenRenderer(viewport_width=720, viewport_height=590)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    return Image.fromarray(color, mode="RGBA").convert("RGB")


def label_panel(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 82), (248, 247, 243))
    canvas.paste(image, (0, 82))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (22, 14),
        title,
        fill=(26, 44, 61),
        font=ImageFont.truetype("DejaVuSans-Bold.ttf", 25),
    )
    draw.text(
        (22, 49),
        subtitle,
        fill=(70, 77, 84),
        font=ImageFont.truetype("DejaVuSans.ttf", 17),
    )
    return canvas


def compose(panels: list[Image.Image], title: str, output: Path):
    margin = 24
    width = panels[0].width * 2 + margin * 3
    height = panels[0].height * 2 + margin * 3 + 72
    sheet = Image.new("RGB", (width, height), (236, 234, 228))
    ImageDraw.Draw(sheet).text(
        (margin, 18),
        title,
        fill=(23, 42, 58),
        font=ImageFont.truetype("DejaVuSans-Bold.ttf", 34),
    )
    y0 = 72 + margin
    sheet.paste(panels[0], (margin, y0))
    sheet.paste(panels[1], (panels[0].width + 2 * margin, y0))
    sheet.paste(panels[2], (margin, y0 + panels[0].height + margin))
    sheet.paste(
        panels[3],
        (panels[0].width + 2 * margin, y0 + panels[0].height + margin),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main():
    for path in (URDF, CONFIG, BOX, GRASPS):
        if not path.is_file():
            raise FileNotFoundError(path)
    if URDF.stat().st_size < 1000:
        raise RuntimeError(f"descriptor URDF is unexpectedly small: {URDF}")
    mesh_probe = HAND_DIR / "meshes/right_palm_link.STL"
    if not mesh_probe.is_file() or mesh_probe.stat().st_size < 1000:
        raise RuntimeError(
            "Unitree meshes are missing or still Git LFS pointers; materialize "
            f"the upstream assets before rendering: {mesh_probe}"
        )

    with CONFIG.open() as stream:
        config = json.load(stream)
    with GRASPS.open() as stream:
        saved = yaml.safe_load(stream)["grasps"]
    best_name, best = max(saved.items(), key=lambda item: item[1]["confidence"])
    best_pose = matrix_from_saved_grasp(best)

    robot = yourdfpy.URDF.load(
        str(URDF),
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    palm_from_root = robot.scene.graph.get(
        frame_from="world", frame_to="right_palm_link"
    )[0]
    halfway = {
        name: 0.5 * (config["open"][name] + config["close"][name])
        for name in config["open"]
    }
    box_mesh = trimesh.load(str(BOX), force="mesh")
    canonical_eye = np.array([0.32, -0.31, 0.24])
    canonical_target = np.array([0.015, 0.0, 0.055])
    grasp_eye = np.array([0.32, -0.32, 0.24])
    grasp_target = np.array([0.0, 0.0, 0.015])

    overview = [
        label_panel(
            render_panel(
                robot, config["open"], np.eye(4), palm_from_root,
                (69, 143, 209), None, True, canonical_eye, canonical_target,
            ),
            "A · stock descriptor, open",
            "black origin = returned grasp frame; magenta origin = right_palm_link",
        ),
        label_panel(
            render_panel(
                robot, halfway, np.eye(4), palm_from_root,
                (77, 174, 133), None, True, canonical_eye, canonical_target,
            ),
            "B · stock descriptor, halfway",
            "50% interpolation from the upstream open endpoint to close endpoint",
        ),
        label_panel(
            render_panel(
                robot, config["open"], best_pose, palm_from_root,
                (69, 143, 209), box_mesh, False, grasp_eye, grasp_target,
            ),
            f"C · {best_name} open at predicted pose",
            f"highest saved score = {best['confidence']:.3f}; upstream 10 cm box",
        ),
        label_panel(
            render_panel(
                robot, config["close"], best_pose, palm_from_root,
                (224, 116, 78), box_mesh, False, grasp_eye, grasp_target,
            ),
            f"D · {best_name} closed at the same pose",
            "GraspGenX supplied the frame; descriptor joints supplied the hand shape",
        ),
    ]
    compose(overview, "Upstream-only GraspGenX → Unitree G1 probe", OUTPUT)

    view_specs = [
        (config["open"], (69, 143, 209), grasp_eye, "A · open, palm-side view"),
        (config["close"], (224, 116, 78), grasp_eye, "B · closed, palm-side view"),
        (config["open"], (69, 143, 209), np.array([-0.32, 0.32, 0.24]),
         "C · open, object-side view"),
        (config["close"], (224, 116, 78), np.array([0.02, 0.38, 0.24]),
         "D · closed, side view"),
    ]
    multiview = [
        label_panel(
            render_panel(
                robot, joints, best_pose, palm_from_root, color, box_mesh, False,
                eye, grasp_target,
            ),
            title,
            f"same {best_name}, score {best['confidence']:.3f}; translucent 10 cm box",
        )
        for joints, color, eye, title in view_specs
    ]
    compose(multiview, "One returned grasp, inspected from multiple views", OUTPUT_MULTIVIEW)

    np.set_printoptions(precision=6, suppress=True)
    print(f"saved: {OUTPUT}")
    print(f"saved: {OUTPUT_MULTIVIEW}")
    print(f"best: {best_name} score={best['confidence']:.6f}")
    print("grasp_root_T_right_palm_link=")
    print(palm_from_root)
    print("object_T_grasp_root=")
    print(best_pose)
    print("object_T_right_palm_link=")
    print(best_pose @ palm_from_root)


if __name__ == "__main__":
    main()
