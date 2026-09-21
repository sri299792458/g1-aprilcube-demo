"""Build and visually audit GraspGenX descriptors for current Unitree Dex3.

This is a deterministic wrapper around the output contract of GraspGenX's
``scripts/gripper_config_wizard.py``.  The interactive wizard remains the tool
for changing the sweep boxes.  This wrapper fixes all inputs we already know:

* exact official Unitree URDF and mesh revisions;
* the canonical GraspGenX frame for both mirrored hands;
* the provisional, hardware-demonstrated GR00T open/close profiles;
* a sweep vector explicitly validated against the released checkpoint.

It intentionally removes the standalone URDF's teleoperation-only auxiliary
links.  Only the palm, seven actuated joints, and seven finger links belong in
the GraspGenX descriptor.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyrender
import trimesh
import yourdfpy
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPGENX_ROOT = PROJECT_ROOT / "third_party/GraspGenX"
MANIFEST = PROJECT_ROOT / "config/dex3_rev1_descriptor.json"
DEFAULT_CACHE = PROJECT_ROOT / ".cache/unitree_xr_teleoperate"
DEFAULT_OUTPUT = GRASPGENX_ROOT / "assets/x_grippers"
DEFAULT_AUDIT = PROJECT_ROOT / "artifacts/dex3_rev1_descriptor/audit.json"
DEFAULT_IMAGE = PROJECT_ROOT / "docs/assets/dex3_rev1_descriptor_states.png"
BACKGROUND = (248, 247, 243)
DISTAL_LINKS = {
    "right": (
        "right_hand_thumb_2_link.STL",
        "right_hand_middle_1_link.STL",
        "right_hand_index_1_link.STL",
    ),
    "left": (
        "left_hand_thumb_2_link.STL",
        "left_hand_middle_1_link.STL",
        "left_hand_index_1_link.STL",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_sources(manifest: dict, source_root: Path | None, cache: Path) -> Path:
    source = source_root.resolve() if source_root else cache.resolve()
    repository = manifest["unitree_source"]["repository"]
    commit = manifest["unitree_source"]["commit"]
    for relative, expected in manifest["unitree_source"]["files"].items():
        path = source / relative
        if not path.is_file() and source_root is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"{repository}/raw/{commit}/{relative}"
            print(f"download {url}")
            urllib.request.urlretrieve(url, path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned Unitree source: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Hash mismatch for {path}: expected {expected}, received {actual}"
            )
    return source


def make_descriptor_urdf(source_urdf: Path, side: str, frame: dict, output: Path) -> None:
    source_robot = ET.parse(source_urdf).getroot()
    prefix = f"{side}_hand_"
    links = {element.get("name"): element for element in source_robot.findall("link")}
    joints = [
        element
        for element in source_robot.findall("joint")
        if element.get("type") != "fixed" and element.get("name", "").startswith(prefix)
    ]
    if len(joints) != 7:
        raise RuntimeError(f"Expected seven {side} hand joints, found {len(joints)}")
    palm = f"{side}_hand_palm_link"
    selected_links = [palm] + [joint.find("child").get("link") for joint in joints]

    robot = ET.Element("robot", {"name": f"dex3_rev1_{side}"})
    ET.SubElement(robot, "link", {"name": "world"})
    fixed = ET.SubElement(robot, "joint", {"name": f"world_to_{palm}", "type": "fixed"})
    ET.SubElement(fixed, "parent", {"link": "world"})
    ET.SubElement(fixed, "child", {"link": palm})
    ET.SubElement(
        fixed,
        "origin",
        {
            "xyz": " ".join(f"{value:.16g}" for value in frame["xyz"]),
            "rpy": " ".join(f"{value:.16g}" for value in frame["rpy"]),
        },
    )
    for name in selected_links:
        link = copy.deepcopy(links[name])
        inertial = link.find("inertial")
        if inertial is not None:
            origin = inertial.find("origin")
            inertia = inertial.find("inertia")
            if origin is not None and inertia is not None:
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                if not np.allclose(rpy, 0.0):
                    # URDF inertia entries are expressed in the inertial
                    # frame.  Rotate the tensor into the link frame so
                    # importers do not need to handle a rotated inertia
                    # origin (Newton 1.0 currently mishandles this case).
                    I = np.array(
                        [
                            [float(inertia.get("ixx")), float(inertia.get("ixy")), float(inertia.get("ixz"))],
                            [float(inertia.get("ixy")), float(inertia.get("iyy")), float(inertia.get("iyz"))],
                            [float(inertia.get("ixz")), float(inertia.get("iyz")), float(inertia.get("izz"))],
                        ]
                    )
                    R = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")[:3, :3]
                    I_link = R @ I @ R.T
                    origin.set("rpy", "0 0 0")
                    for key, value in {
                        "ixx": I_link[0, 0],
                        "ixy": I_link[0, 1],
                        "ixz": I_link[0, 2],
                        "iyy": I_link[1, 1],
                        "iyz": I_link[1, 2],
                        "izz": I_link[2, 2],
                    }.items():
                        inertia.set(key, f"{value:.16g}")
        robot.append(link)
    for joint in joints:
        robot.append(copy.deepcopy(joint))
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def copy_meshes(source: Path, side: str, destination: Path) -> None:
    meshes = source / "assets/unitree_hand/meshes"
    out = destination / "meshes"
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(meshes.glob(f"{side}_hand_*_link.STL")):
        shutil.copy2(path, out / path.name)


def load_robot(urdf: Path) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        str(urdf),
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=True,
        load_collision_meshes=True,
    )


def geometry_centroid_in_frame(
    robot: yourdfpy.URDF,
    joints: dict[str, float],
    geometry_name: str,
    frame_from: str,
) -> np.ndarray:
    robot.update_cfg(joints)
    geometry = robot.scene.geometry[geometry_name]
    frame_T_geometry = robot.scene.graph.get(
        frame_from=frame_from,
        frame_to=geometry_name,
    )[0]
    return trimesh.transformations.transform_points(
        geometry.vertices, frame_T_geometry
    ).mean(axis=0)


def derive_frame_and_sweep(
    source_urdf: Path,
    side: str,
    frame_spec: dict,
    profile: dict,
    sweep_spec: dict,
) -> tuple[dict, dict, dict]:
    """Derive the canonical origin from pinned geometry, never a magic number."""
    source_robot = load_robot(source_urdf)
    palm = f"{side}_hand_palm_link"
    R_G0_P = np.eye(4)
    R_G0_P[:3, :3] = np.asarray(frame_spec["rotation"], dtype=float)
    centroids_G0 = []
    for name in DISTAL_LINKS[side]:
        centroid_P = geometry_centroid_in_frame(
            source_robot, profile["close"], name, frame_from=palm
        )
        centroids_G0.append(
            trimesh.transformations.transform_points(
                np.asarray([centroid_P]), R_G0_P
            )[0]
        )
    thumb_x = float(centroids_G0[0][0])
    opponents_x = float(np.mean([centroids_G0[1][0], centroids_G0[2][0]]))
    midpoint_x = 0.5 * (thumb_x + opponents_x)
    translation_x = -midpoint_x

    frame = copy.deepcopy(frame_spec)
    frame["xyz"] = [translation_x, 0.0, 0.0]
    sweep = copy.deepcopy(sweep_spec)
    # Sweep offsets are already expressed in the final canonical G frame.  Do
    # not shift them again when deriving the fixed G_T_P translation: the
    # released checkpoint consumes these twelve numbers directly, independent
    # of the URDF, and their physics validation used these exact resolved
    # values.
    derivation = {
        "unshifted_thumb_centroid_x_m": thumb_x,
        "unshifted_opponents_mean_centroid_x_m": opponents_x,
        "unshifted_pinch_midpoint_x_m": midpoint_x,
        "canonical_translation_x_m": translation_x,
    }
    return frame, sweep, derivation


def merged_scene_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    parts: list[trimesh.Trimesh] = []
    for name, geometry in scene.geometry.items():
        transform = scene.graph.get(frame_from=scene.graph.base_frame, frame_to=name)[0]
        part = geometry.copy()
        part.apply_transform(transform)
        parts.append(part)
    if not parts:
        raise RuntimeError("URDF scene contains no geometry")
    return trimesh.util.concatenate(parts)


def joint_limits(urdf: Path) -> dict[str, tuple[float, float]]:
    root = ET.parse(urdf).getroot()
    result = {}
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        result[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    return result


def validate_profile(urdf: Path, profile: dict) -> dict:
    limits = joint_limits(urdf)
    if set(profile["open"]) != set(limits) or set(profile["close"]) != set(limits):
        raise RuntimeError(f"Profile joints do not match {urdf}")
    margins = {}
    for name, (lower, upper) in limits.items():
        for state in ("open", "close"):
            value = profile[state][name]
            if not lower <= value <= upper:
                raise RuntimeError(f"{name} {state}={value} outside [{lower}, {upper}]")
        margins[name] = min(
            profile["close"][name] - lower,
            upper - profile["close"][name],
        )
    return {"joint_count": len(limits), "minimum_close_limit_margin_rad": min(margins.values())}


def descriptor_config(robot: yourdfpy.URDF, profile: dict, sweep: dict) -> dict:
    robot.update_cfg(profile["open"])
    bbox = merged_scene_mesh(robot.scene).bounds.tolist()
    return {
        "open": profile["open"],
        "close": profile["close"],
        "fingertip": sweep["offset"],
        "sweep_volume": {
            "extents": sweep["extents"],
            "offset": sweep["offset"],
            "extents2": sweep["extents2"],
            "offset2": sweep["offset2"],
        },
        "links": list(robot.link_map),
        "standoff": [0.0, sweep["extents"][2] / 2.0],
        "bbox": bbox,
        "symmetric": False,
        "type": "revolute_3f",
        "base_rotation": np.eye(4).tolist(),
        "review_status": sweep["status"],
    }


def export_state_meshes(robot: yourdfpy.URDF, profile: dict, destination: Path) -> dict[str, trimesh.Trimesh]:
    states = {
        "open": profile["open"],
        "half": {
            name: 0.5 * (profile["open"][name] + profile["close"][name])
            for name in profile["open"]
        },
        "closed": profile["close"],
    }
    result = {}
    for name, joints in states.items():
        robot.update_cfg(joints)
        result[name] = merged_scene_mesh(robot.scene)
    robot.update_cfg(profile["open"])
    result["open"].export(destination / "vis_mesh.obj")
    merged_scene_mesh(robot.collision_scene).export(destination / "coll_mesh.obj")
    return result


def mat(color: tuple[int, int, int], alpha: float = 1.0):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(*[channel / 255.0 for channel in color], alpha),
        metallicFactor=0.03,
        roughnessFactor=0.72,
        alphaMode="BLEND" if alpha < 1.0 else "OPAQUE",
    )


def add_mesh(scene: pyrender.Scene, mesh: trimesh.Trimesh, color, alpha=1.0):
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=mat(color, alpha), smooth=False))


def add_box(scene: pyrender.Scene, extents: list[float], offset: list[float], color) -> None:
    box = trimesh.creation.box(extents=np.asarray(extents))
    box.apply_translation(offset)
    add_mesh(scene, box, color, 0.13)
    corners = trimesh.bounds.corners(np.array(box.bounds))
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
             (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for start, end in edges:
        vector = corners[end] - corners[start]
        length = np.linalg.norm(vector)
        cylinder = trimesh.creation.cylinder(radius=0.00075, height=length, sections=12)
        cylinder.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], vector))
        cylinder.apply_translation((corners[start] + corners[end]) / 2.0)
        add_mesh(scene, cylinder, color)


def add_axes(scene: pyrender.Scene) -> None:
    axis = trimesh.creation.axis(origin_size=0.0035, axis_radius=0.0012, axis_length=0.055)
    for mesh in axis.dump() if isinstance(axis, trimesh.Scene) else [axis]:
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))


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


def render_panel(mesh: trimesh.Trimesh, hand_color, sweep: tuple | None) -> Image.Image:
    scene = pyrender.Scene(
        bg_color=np.array([*BACKGROUND, 255], dtype=np.uint8),
        ambient_light=np.array([0.58, 0.58, 0.58, 1.0]),
    )
    add_mesh(scene, mesh, hand_color)
    if sweep:
        add_box(scene, sweep[0], sweep[1], sweep[2])
    add_axes(scene)
    camera_pose = look_at(np.array([0.27, -0.28, 0.24]), np.array([0.03, 0.0, 0.08]))
    scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(38.0)), pose=camera_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.2), pose=camera_pose)
    renderer = pyrender.OffscreenRenderer(viewport_width=520, viewport_height=430)
    rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    return Image.fromarray(rgba, mode="RGBA").convert("RGB")


def label(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    panel = Image.new("RGB", (image.width, image.height + 76), BACKGROUND)
    panel.paste(image, (0, 76))
    draw = ImageDraw.Draw(panel)
    draw.text((18, 12), title, fill=(24, 43, 59), font=ImageFont.truetype("DejaVuSans-Bold.ttf", 23))
    draw.text((18, 45), subtitle, fill=(68, 76, 84), font=ImageFont.truetype("DejaVuSans.ttf", 15))
    return panel


def render_audit(all_meshes: dict, sweep: dict, output: Path) -> None:
    panels = []
    for side, color in (("right", (64, 145, 194)), ("left", (223, 115, 91))):
        for state in ("open", "half", "closed"):
            box = None
            if state == "open":
                box = (sweep["extents"], sweep["offset"], (45, 150, 220))
            elif state == "half":
                box = (sweep["extents2"], sweep["offset2"], (238, 158, 30))
            panels.append(
                label(
                    render_panel(all_meshes[side][state], color, box),
                    f"{side.upper()} · {state}",
                    "released-checkpoint morphology proxy · physics validated"
                    if box else "GR00T demonstrated endpoint",
                )
            )
    margin, header, footer = 20, 112, 100
    width = panels[0].width * 3 + margin * 4
    height = panels[0].height * 2 + margin * 3 + header + footer
    sheet = Image.new("RGB", (width, height), (235, 233, 227))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 16), "Current Unitree Dex3 → GraspGenX descriptor audit",
              fill=(21, 39, 55), font=ImageFont.truetype("DejaVuSans-Bold.ttf", 34))
    draw.text((margin, 65), "red +X = descriptor aperture · blue +Z = approach · same pose frame for both hands",
              fill=(69, 76, 84), font=ImageFont.truetype("DejaVuSans.ttf", 19))
    y0 = header + margin
    for index, panel in enumerate(panels):
        row, column = divmod(index, 3)
        sheet.paste(panel, (margin + column * (panel.width + margin), y0 + row * (panel.height + margin)))
    footer_y = height - footer + 10
    draw.rounded_rectangle((margin, footer_y, width - margin, height - 18), radius=14, fill=(255, 248, 226))
    draw.text((margin + 18, footer_y + 12), "CHECKPOINT CONTRACT — proxy vector validated with the exact current hand",
              fill=(126, 78, 8), font=ImageFont.truetype("DejaVuSans-Bold.ttf", 18))
    draw.text((margin + 18, footer_y + 42), "Blue/open, orange/half; proxy encodes learned morphology, not the literal L-shaped finger gap",
              fill=(91, 72, 38), font=ImageFont.truetype("DejaVuSans.ttf", 16))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def build_side(manifest: dict, source: Path, output_root: Path, side: str) -> tuple[dict, dict]:
    name = manifest["descriptor_names"][side]
    destination = output_root / name
    destination.mkdir(parents=True, exist_ok=True)
    source_urdf = source / f"assets/unitree_hand/unitree_dex3_{side}.urdf"
    urdf = destination / "gripper.urdf"
    profile = manifest["finger_profile"][side]
    frame, sweep, origin_derivation = derive_frame_and_sweep(
        source_urdf,
        side,
        manifest["canonical_frame"][side],
        profile,
        manifest["sweep_volume"],
    )
    make_descriptor_urdf(source_urdf, side, frame, urdf)
    copy_meshes(source, side, destination)
    shutil.copy2(source / "LICENSE", destination / "SOURCE_LICENSE")

    validation = validate_profile(urdf, profile)
    robot = load_robot(urdf)
    config = descriptor_config(robot, profile, sweep)
    with (destination / "config.json").open("w") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    meshes = export_state_meshes(robot, profile, destination)
    validation.update(
        {
            "descriptor": name,
            "urdf_sha256": sha256(urdf),
            "config_sha256": sha256(destination / "config.json"),
            "visual_mesh_sha256": sha256(destination / "vis_mesh.obj"),
            "collision_mesh_sha256": sha256(destination / "coll_mesh.obj"),
            "open_bbox_m": meshes["open"].bounds.tolist(),
            "canonical_origin_derivation": origin_derivation,
            "resolved_sweep_volume": sweep,
        }
    )
    return meshes, validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--source-root", type=Path, default=None,
                        help="Existing xr_teleoperate checkout; otherwise download pinned assets")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    source = acquire_sources(manifest, args.source_root, args.cache)
    all_meshes, validations = {}, {}
    for side in ("right", "left"):
        all_meshes[side], validations[side] = build_side(
            manifest, source, args.output_root, side
        )
    right_bounds = np.asarray(all_meshes["right"]["open"].bounds)
    left_bounds = np.asarray(all_meshes["left"]["open"].bounds)
    overlay_bound_error = float(np.max(np.abs(right_bounds - left_bounds)))
    if overlay_bound_error > 5e-5:
        raise RuntimeError(f"Canonical left/right bounds differ by {overlay_bound_error} m")
    audit = {
        "schema_version": 1,
        "manifest_sha256": sha256(args.manifest),
        "unitree_commit": manifest["unitree_source"]["commit"],
        "finger_profile_commit": manifest["finger_profile"]["source_commit"],
        "sweep_review_status": manifest["sweep_volume"]["status"],
        "canonical_left_right_open_bbox_max_error_m": overlay_bound_error,
        "hands": validations,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("w") as stream:
        json.dump(audit, stream, indent=2)
        stream.write("\n")
    render_audit(
        all_meshes,
        validations["right"]["resolved_sweep_volume"],
        args.image,
    )
    print(f"descriptors: {args.output_root}")
    print(f"audit:       {args.audit}")
    print(f"image:       {args.image}")


if __name__ == "__main__":
    main()
