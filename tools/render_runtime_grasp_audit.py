#!/usr/bin/env python3
"""Render every support-clear candidate from a runtime grasp audit.

The dark hand is the exact open descriptor visual mesh at the final
``world_T_G``.  The translucent blue hand is the same mesh at the named
local-Z pregrasp.  No closed-hand pose is shown because kinematic closure
through an object is not a contact simulation.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrender
import trimesh


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_aprilcube_demo.assembly import load_assembly_task  # noqa: E402
from g1_aprilcube_demo.runtime import load_observation  # noqa: E402


PART_COLORS = {
    "t_body": (50, 157, 183),
    "u_legs": (75, 160, 105),
    "cube_head": (221, 148, 48),
}
FAILURE_COLORS = {
    "final_endpoint_ik": (190, 64, 56),
    "pregrasp_endpoint_ik": (154, 92, 180),
    "pickup_plan": (213, 116, 43),
    "not_evaluated": (110, 118, 128),
}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default="artifacts/runtime_grasp_audit/nominal/audit.json",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def material(color: tuple[int, int, int], alpha: float = 1.0):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(*[value / 255.0 for value in color], alpha),
        metallicFactor=0.02,
        roughnessFactor=0.78,
        alphaMode="BLEND" if alpha < 1.0 else "OPAQUE",
    )


def add_mesh(
    scene: pyrender.Scene,
    mesh: trimesh.Trimesh,
    pose: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 1.0,
) -> None:
    scene.add(
        pyrender.Mesh.from_trimesh(
            mesh, material=material(color, alpha), smooth=False
        ),
        pose=pose,
    )


def cylinder_between(start: np.ndarray, end: np.ndarray) -> trimesh.Trimesh:
    delta = end - start
    length = float(np.linalg.norm(delta))
    cylinder = trimesh.creation.cylinder(radius=0.0022, height=length, sections=16)
    direction = delta / length
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)
    transform[:3, 3] = 0.5 * (start + end)
    cylinder.apply_transform(transform)
    return cylinder


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


def render_view(
    *,
    hand: trimesh.Trimesh,
    object_meshes: dict[str, trimesh.Trimesh],
    object_poses: dict[str, np.ndarray],
    target_part: str,
    table_center: tuple[float, float, float],
    table_dimensions: tuple[float, float, float],
    final_pose: np.ndarray,
    pregrasp_pose: np.ndarray,
    final_color: tuple[int, int, int],
    direction: np.ndarray,
) -> Image.Image:
    scene = pyrender.Scene(
        bg_color=np.asarray([248, 247, 243, 255], dtype=np.uint8),
        ambient_light=np.asarray([0.58, 0.58, 0.58]),
    )
    table = trimesh.creation.box(extents=np.asarray(table_dimensions))
    table_pose = np.eye(4)
    table_pose[:3, 3] = table_center
    add_mesh(scene, table, table_pose, (182, 143, 103))

    for part, mesh in object_meshes.items():
        if part == target_part:
            add_mesh(scene, mesh, object_poses[part], PART_COLORS[part])
        else:
            add_mesh(scene, mesh, object_poses[part], (160, 166, 171), 0.22)

    add_mesh(scene, hand, pregrasp_pose, (73, 174, 206), 0.20)
    add_mesh(scene, hand, final_pose, final_color, 0.88)
    add_mesh(
        scene,
        cylinder_between(pregrasp_pose[:3, 3], final_pose[:3, 3]),
        np.eye(4),
        (238, 172, 34),
    )
    axes = trimesh.creation.axis(
        transform=final_pose,
        origin_size=0.003,
        axis_radius=0.001,
        axis_length=0.035,
    )
    for item in axes.dump() if isinstance(axes, trimesh.Scene) else [axes]:
        scene.add(pyrender.Mesh.from_trimesh(item, smooth=False))

    target = object_poses[target_part][:3, 3] + np.asarray([0.0, 0.0, 0.035])
    direction = direction / np.linalg.norm(direction)
    camera_pose = look_at(target + 0.48 * direction, target)
    scene.add(
        pyrender.PerspectiveCamera(yfov=np.deg2rad(38.0), znear=0.02, zfar=4.0),
        pose=camera_pose,
    )
    light_pose = camera_pose.copy()
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.2), pose=light_pose)
    fill_pose = look_at(target + np.asarray([-0.3, 0.25, 0.5]), target)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.3), pose=fill_pose)

    renderer = pyrender.OffscreenRenderer(560, 400)
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
    part: str,
    record: dict,
    hand: trimesh.Trimesh,
    object_meshes: dict[str, trimesh.Trimesh],
    object_poses: dict[str, np.ndarray],
    table_center: tuple[float, float, float],
    table_dimensions: tuple[float, float, float],
) -> Image.Image:
    final_pose = np.asarray(record["world_T_G"], dtype=np.float64)
    pregrasp_pose = np.asarray(record["world_T_pregrasp_G"], dtype=np.float64)
    passed = record["pickup_plan"] is True
    status = (
        "PICKUP PLAN PASS"
        if passed
        else f"FAIL · {record['failure_gate'].replace('_', ' ').upper()}"
    )
    final_color = (40, 137, 83) if passed else FAILURE_COLORS.get(
        record["failure_gate"], (176, 69, 60)
    )
    images = [
        render_view(
            hand=hand,
            object_meshes=object_meshes,
            object_poses=object_poses,
            target_part=part,
            table_center=table_center,
            table_dimensions=table_dimensions,
            final_pose=final_pose,
            pregrasp_pose=pregrasp_pose,
            final_color=final_color,
            direction=np.asarray([0.75, -0.78, 0.52]),
        ),
        render_view(
            hand=hand,
            object_meshes=object_meshes,
            object_poses=object_poses,
            target_part=part,
            table_center=table_center,
            table_dimensions=table_dimensions,
            final_pose=final_pose,
            pregrasp_pose=pregrasp_pose,
            final_color=final_color,
            direction=np.asarray([0.0, -0.04, 1.0]),
        ),
    ]
    panel = Image.new("RGB", (1120, 500), (248, 247, 243))
    panel.paste(images[0], (0, 100))
    panel.paste(images[1], (560, 100))
    draw = ImageDraw.Draw(panel)
    draw.text((20, 14), record["candidate_id"], fill=(31, 45, 60), font=font(24, True))
    draw.text(
        (20, 50),
        f"{part} · {record['family_id']} · score {record['graspgenx_score']:.3f}",
        fill=(76, 87, 99),
        font=font(17),
    )
    status_width = draw.textbbox((0, 0), status, font=font(20, True))[2]
    draw.rounded_rectangle(
        (1090 - status_width, 18, 1102, 60),
        radius=10,
        fill=final_color,
    )
    draw.text(
        (1096 - status_width, 27),
        status,
        fill=(255, 255, 255),
        font=font(20, True),
    )
    final_mm = 1000.0 * record["final_support_clearance_m"]
    pre_mm = 1000.0 * record["pregrasp_support_clearance_m"]
    draw.text(
        (580, 66),
        f"support clearance: final {final_mm:+.1f} mm · pregrasp {pre_mm:+.1f} mm",
        fill=(76, 87, 99),
        font=font(15),
    )
    draw.text((16, 466), "three-quarter view", fill=(57, 68, 80), font=font(15, True))
    draw.text((576, 466), "top view", fill=(57, 68, 80), font=font(15, True))
    return panel


def contact_sheet(paths: list[Path], output: Path, columns: int) -> None:
    thumbs: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((560, 250), Image.Resampling.LANCZOS)
        thumbs.append(image.copy())
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * 560, rows * 250), (238, 237, 233))
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % columns) * 560, (index // columns) * 250))
    sheet.save(output, quality=92)


def write_index(report: dict, part_paths: dict[str, list[Path]], output: Path) -> None:
    cards = []
    for part, paths in part_paths.items():
        summary = report["parts"][part]["summary"]
        figures = "\n".join(
            f'<a href="{html.escape(path.name)}"><img loading="lazy" '
            f'src="{html.escape(path.name)}"></a>'
            for path in paths
        )
        cards.append(
            f"""
            <section>
              <h2>{html.escape(part)}</h2>
              <p>{summary['atlas_candidates']} physics-qualified atlas candidates
              → {summary['support_plane_clear']} support-clear
              → {summary['pickup_plan']} complete cuRobo pickups.</p>
              <p><a href="{part}_contact_sheet.png">Open the contact sheet</a></p>
              <div class="grid">{figures}</div>
            </section>
            """
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Runtime grasp audit</title>
<style>
body {{ margin: 32px auto; max-width: 1500px; padding: 0 24px;
       background:#f8f7f3; color:#1f2d3c; font:17px/1.5 system-ui,sans-serif; }}
h1 {{ margin-bottom:6px; }} h2 {{ margin-top:50px; }}
.note {{ background:white; border-radius:14px; padding:18px 22px; }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.grid img {{ width:100%; border-radius:10px; box-shadow:0 2px 10px #0002; }}
code {{ background:#e9ecef; padding:2px 5px; border-radius:4px; }}
</style></head><body>
<h1>Scene-conditioned GraspGenX audit</h1>
<p>{html.escape(report['observation_id'])}</p>
<div class="note">
Dark hand: exact open Dex3 descriptor mesh at immutable <code>world_T_G</code>.
Blue hand: the same mesh at the −10 cm descriptor-local-Z pregrasp.
Yellow: approach segment. Closed fingers are intentionally not rendered.
The complete upstream <code>plan_grasp</code> result is authoritative.
Connector keep-outs and assembly compatibility are not evaluated here.
</div>
{''.join(cards)}
</body></html>"""
    output.write_text(document)


def main() -> None:
    args = parse_args()
    report_path = project_path(args.report)
    report = json.loads(report_path.read_text())
    output = (
        project_path(args.output_dir)
        if args.output_dir
        else report_path.parent / "visual"
    )
    output.mkdir(parents=True, exist_ok=True)

    task = load_assembly_task(project_path(report["task"]))
    object_meshes = {
        part: trimesh.load(spec.mesh, force="mesh", process=False)
        for part, spec in task.parts.items()
    }
    observation = load_observation(
        project_path(report["observation"]),
        {part: spec.mesh for part, spec in task.parts.items()},
    )

    part_paths: dict[str, list[Path]] = {}
    for part, part_report in report["parts"].items():
        side = part_report["assigned_hand_for_this_checkpoint"]
        hand = trimesh.load(
            ROOT
            / f"third_party/GraspGenX/assets/x_grippers/"
            f"dex3_rev1_{side}/vis_mesh.obj",
            force="mesh",
            process=False,
        )
        part_paths[part] = []
        eligible = [
            record
            for record in part_report["candidates"]
            if record["support_plane_clear"]
        ]
        for index, record in enumerate(eligible):
            panel = candidate_panel(
                part=part,
                record=record,
                hand=hand,
                object_meshes=object_meshes,
                object_poses=dict(observation.world_T_objects),
                table_center=observation.table.center,
                table_dimensions=observation.table.dimensions,
            )
            path = output / f"{part}_{index:03d}_{record['candidate_id']}.png"
            panel.save(path, quality=94)
            part_paths[part].append(path)
            print(f"[render] {part} {index + 1}/{len(eligible)}", flush=True)
        contact_sheet(
            part_paths[part],
            output / f"{part}_contact_sheet.png",
            columns=2 if len(eligible) <= 6 else 4,
        )
    write_index(report, part_paths, output / "index.html")
    print(output / "index.html")


if __name__ == "__main__":
    main()
