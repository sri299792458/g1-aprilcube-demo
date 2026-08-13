#!/usr/bin/env python3
"""Build stable-support-conditioned proposal buckets before grasp physics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_aprilcube_demo.grasping.support_atlas import (
    MeshCollisionGate,
    TargetRegionClassifier,
    build_buckets,
    configured_support_conditions,
    evaluate_support,
    load_raw_candidates,
    load_raw_candidates_many,
    semantic_voxels,
    sha256,
)


DEFAULT_CONFIG = ROOT / "config/grasp_support/u_legs_right_v1.yaml"
BACKGROUND = (248, 247, 243)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def raw_provenance(raw_directories: list[Path]) -> list[dict[str, Any]]:
    output = []
    for raw_directory in raw_directories:
        for path in sorted(raw_directory.glob("shard_*.yaml")):
            provenance = path.with_suffix(".provenance.json")
            output.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "provenance": (
                        str(provenance.relative_to(ROOT))
                        if provenance.is_file()
                        else None
                    ),
                    "provenance_sha256": (
                        sha256(provenance) if provenance.is_file() else None
                    ),
                }
            )
    return output


def make_document(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported support-atlas configuration schema")
    object_cfg = config["object"]
    hand_cfg = config["hand"]
    proposal_cfg = config["proposals"]
    support_cfg = config["supports"]
    gate_cfg = config["geometric_gates"]

    object_mesh_path = project_path(object_cfg["mesh"])
    hand_mesh_path = project_path(hand_cfg["open_collision_mesh"])
    geometry_path = project_path(object_cfg["geometry_config"])
    if "raw_directories" in proposal_cfg:
        if "raw_directory" in proposal_cfg:
            raise ValueError(
                "Configure proposals.raw_directory or raw_directories, not both"
            )
        raw_directories = [
            project_path(value) for value in proposal_cfg["raw_directories"]
        ]
    else:
        raw_directories = [project_path(proposal_cfg["raw_directory"])]
    for required in (object_mesh_path, hand_mesh_path, geometry_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    object_mesh = trimesh.load(object_mesh_path, force="mesh", process=False)
    hand_mesh = trimesh.load(hand_mesh_path, force="mesh", process=False)
    if not isinstance(object_mesh, trimesh.Trimesh) or not object_mesh.is_watertight:
        raise ValueError("Support conditioning requires one watertight object mesh")
    if not isinstance(hand_mesh, trimesh.Trimesh):
        raise ValueError("Support conditioning requires one open-hand collision mesh")

    candidates = (
        load_raw_candidates(raw_directories[0])
        if len(raw_directories) == 1
        else load_raw_candidates_many(raw_directories)
    )
    expected = int(proposal_cfg["expected_count"])
    if len(candidates) != expected:
        raise ValueError(f"Expected {expected} raw candidates, found {len(candidates)}")
    supports = configured_support_conditions(
        object_mesh,
        entries=support_cfg["orientations"],
    )
    included_classes = support_cfg.get("include_symmetry_classes")
    if included_classes is not None:
        included = {str(value) for value in included_classes}
        available = {support.symmetry_class for support in supports}
        unknown = included - available
        if unknown:
            raise ValueError(
                f"Unknown included support symmetry classes: {sorted(unknown)}"
            )
        supports = tuple(
            support for support in supports
            if support.symmetry_class in included
        )
        if not supports:
            raise ValueError("Support selection removed every configured support")
    classifier = TargetRegionClassifier(
        object_mesh,
        semantic_voxels(geometry_path),
        surface_tolerance_m=float(gate_cfg["semantic_surface_tolerance_m"]),
    )
    collision = MeshCollisionGate(hand_mesh, object_mesh)
    results = []
    for support in supports:
        print(
            f"[support-atlas] {support.support_id} ({support.label}): "
            f"{len(candidates)} raw candidates",
            flush=True,
        )
        result = evaluate_support(
            support=support,
            candidates=candidates,
            hand_mesh=hand_mesh,
            collision=collision,
            classifier=classifier,
            approach_offset_m=float(gate_cfg["pregrasp_offset_local_z_m"]),
            corridor_step_m=float(gate_cfg["approach_corridor_step_m"]),
            table_tolerance_m=float(gate_cfg["table_penetration_tolerance_m"]),
        )
        print(
            f"[support-atlas] {support.support_id}: "
            f"{result['survivor_count']} geometric survivors",
            flush=True,
        )
        results.append(result)

    buckets = build_buckets(results)
    survivor_count = sum(int(value["survivor_count"]) for value in results)
    if sum(int(bucket["member_count"]) for bucket in buckets) != survivor_count:
        raise RuntimeError("Bucket membership does not cover every survivor")
    return {
        "schema_version": 1,
        "status": "pre_physics_support_conditioned_proposals",
        "support_atlas_id": config["support_atlas_id"],
        "object_id": object_cfg["id"],
        "hand_side": hand_cfg["side"],
        "contract": {
            "candidate_transform": "immutable raw GraspGenX object_T_G",
            "support_plane": "infinite tabletop z=0; XY translation and yaw factor out",
            "table_gate": "exact open-hand vertex minimum at final and pregrasp endpoints",
            "target_gate": "exact open-hand/object FCL mesh intersection",
            "approach_gate": (
                "straight negative-local-Z pregrasp to grasp, sampled at the "
                f"configured {float(gate_cfg['approach_corridor_step_m']):.6f} m resolution"
            ),
            "bucket_policy": (
                "support + semantic approach target + surface relation + approach "
                "sector; every survivor retained exactly once"
            ),
            "not_claimed": [
                "finger closure success",
                "force closure or payload retention",
                "physics-qualified contact family",
                "G1 arm reachability",
            ],
        },
        "source": {
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256(config_path),
            "object_mesh": str(object_mesh_path.relative_to(ROOT)),
            "object_mesh_sha256": sha256(object_mesh_path),
            "geometry_config": str(geometry_path.relative_to(ROOT)),
            "geometry_config_sha256": sha256(geometry_path),
            "open_hand_collision_mesh": str(hand_mesh_path.relative_to(ROOT)),
            "open_hand_collision_mesh_sha256": sha256(hand_mesh_path),
            "raw_directories": [
                str(path.relative_to(ROOT)) for path in raw_directories
            ],
            "raw_shards": raw_provenance(raw_directories),
        },
        "raw_candidate_count": len(candidates),
        "stable_support_count": len(supports),
        "support_symmetry_classes": sorted(
            {support.symmetry_class for support in supports}
        ),
        "support_conditioned_pair_count": len(candidates) * len(supports),
        "geometric_survivor_count": survivor_count,
        "proposal_bucket_count": len(buckets),
        "supports": results,
        "proposal_buckets": buckets,
    }


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(
        f"/usr/share/fonts/truetype/dejavu/{name}", size
    )


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
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


def render_support_visual(document: dict[str, Any], output: Path) -> None:
    """Render the exact U mesh in each of its six configured supports."""

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender

    object_mesh = trimesh.load(
        project_path(document["source"]["object_mesh"]),
        force="mesh",
        process=False,
    )
    object_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.23, 0.63, 0.40, 1.0),
        metallicFactor=0.02,
        roughnessFactor=0.74,
    )
    table_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.67, 0.48, 0.32, 1.0),
        metallicFactor=0.01,
        roughnessFactor=0.82,
    )
    panel_width, render_height, label_height = 480, 340, 84
    panels: list[Image.Image] = []
    for result in document["supports"]:
        scene = pyrender.Scene(
            bg_color=np.asarray([*BACKGROUND, 255], dtype=np.uint8),
            ambient_light=np.asarray([0.58, 0.58, 0.58]),
        )
        table = trimesh.creation.box(extents=[0.32, 0.28, 0.014])
        table_pose = trimesh.transformations.translation_matrix(
            [0.0, 0.0, -0.007]
        )
        scene.add(
            pyrender.Mesh.from_trimesh(
                table, material=table_material, smooth=False
            ),
            pose=table_pose,
        )
        object_pose = np.asarray(
            result["support"]["support_T_object"], dtype=np.float64
        )
        scene.add(
            pyrender.Mesh.from_trimesh(
                object_mesh, material=object_material, smooth=False
            ),
            pose=object_pose,
        )
        target = np.asarray([0.0, 0.0, 0.045])
        camera_pose = _look_at(
            np.asarray([0.245, -0.285, 0.205]), target
        )
        scene.add(
            pyrender.PerspectiveCamera(
                yfov=np.deg2rad(36.0), znear=0.01, zfar=2.0
            ),
            pose=camera_pose,
        )
        scene.add(
            pyrender.DirectionalLight(
                color=np.ones(3), intensity=3.0
            ),
            pose=camera_pose,
        )
        fill_pose = _look_at(
            np.asarray([-0.24, 0.18, 0.30]), target
        )
        scene.add(
            pyrender.DirectionalLight(
                color=np.ones(3), intensity=1.5
            ),
            pose=fill_pose,
        )
        renderer = pyrender.OffscreenRenderer(panel_width, render_height)
        try:
            rgba, _ = renderer.render(
                scene,
                flags=(
                    pyrender.RenderFlags.RGBA
                    | pyrender.RenderFlags.SKIP_CULL_FACES
                ),
            )
        finally:
            renderer.delete()
        panel = Image.new(
            "RGB", (panel_width, render_height + label_height), BACKGROUND
        )
        panel.paste(
            Image.fromarray(rgba, mode="RGBA").convert("RGB"), (0, 0)
        )
        draw = ImageDraw.Draw(panel)
        support = result["support"]
        draw.text(
            (22, render_height + 10),
            support["label"].replace("_", " "),
            fill=(27, 49, 65),
            font=_font(22, bold=True),
        )
        draw.text(
            (22, render_height + 43),
            (
                f"table-up object {support['table_up_sector']}  ·  "
                f"{result['survivor_count']:,} geometry-clear"
            ),
            fill=(76, 83, 90),
            font=_font(17),
        )
        panels.append(panel)

    header_height = 112
    columns = min(3, max(1, len(panels)))
    rows = (len(panels) + columns - 1) // columns
    montage = Image.new(
        "RGB",
        (
            columns * panel_width,
            header_height + rows * (render_height + label_height),
        ),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(montage)
    draw.text(
        (32, 18),
        f"Selected U tabletop supports ({len(panels)})",
        fill=(24, 43, 59),
        font=_font(34, bold=True),
    )
    draw.text(
        (33, 66),
        (
            "Exact 45 mm-voxel AprilCube mesh · table contact at z = 0 · "
            "counts are geometric candidates, not physics successes"
        ),
        fill=(69, 76, 84),
        font=_font(19),
    )
    for index, panel in enumerate(panels):
        x = (index % columns) * panel_width
        y = header_height + (index // columns) * (render_height + label_height)
        montage.paste(panel, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output)


def markdown_report(
    document: dict[str, Any], *, support_visual_reference: str
) -> str:
    lines = [
        f"# U support-conditioned {document['hand_side']}-Dex3 proposal audit",
        "",
        "This is a pre-physics geometric audit. It does **not** claim that any",
        "candidate closes on, lifts, or retains the U.",
        "",
        "## Scope",
        "",
        f"- Raw immutable GraspGenX candidates: **{document['raw_candidate_count']:,}**",
        f"- Stable tabletop orientations: **{document['stable_support_count']}**",
        f"- Candidate/support pairs checked: **{document['support_conditioned_pair_count']:,}**",
        f"- Geometric survivors: **{document['geometric_survivor_count']:,}**",
        f"- Pre-physics proposal buckets: **{document['proposal_bucket_count']}**",
        "",
        f"![Selected U tabletop supports]({support_visual_reference})",
        "",
        "Absolute tabletop height, XY translation, and in-plane yaw are absent",
        "because they do not change object–support clearance. They belong to arm",
        "reachability, not this object-relative support audit.",
        "",
        "## Stable supports and gates",
        "",
        "| Table-up object axis | Label | Equivalence class | Final table-clear | Pregrasp table-clear | Final object-clear | Full corridor-clear |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for result in document["supports"]:
        support = result["support"]
        counts = result["stage_counts"]
        lines.append(
            "| {axis} | {label} | {symmetry} | {final:,} | {pre:,} | "
            "{object_clear:,} | {corridor:,} |".format(
                axis=support["table_up_sector"],
                label=support["label"],
                symmetry=support["symmetry_class"],
                final=int(counts.get("final_table_clear", 0)),
                pre=int(counts.get("pregrasp_table_clear", 0)),
                object_clear=int(counts.get("final_object_clear", 0)),
                corridor=int(counts.get("approach_corridor_clear", 0)),
            )
        )
    lines.extend(
        [
            "",
            "Only the support conditions selected by this experiment are evaluated.",
            "Geometric symmetries remain recorded as equivalence classes rather",
            "than silently deleting tag-distinguishable poses.",
            "",
            "## Survivor distribution by semantic U component",
            "",
            "| Support | Hip bridge | Left leg | Right leg | Unresolved | Total |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in document["supports"]:
        component_counts = Counter(
            survivor["target_region"]["component"]
            for survivor in result["survivors"]
        )
        lines.append(
            f"| {result['support']['label']} | "
            f"{component_counts['hip_bridge']:,} | "
            f"{component_counts['left_leg']:,} | "
            f"{component_counts['right_leg']:,} | "
            f"{component_counts['unresolved']:,} | "
            f"{result['survivor_count']:,} |"
        )
    lines.extend(
        [
            "",
            f"The JSON artifact retains all {document['proposal_bucket_count']} exact buckets, including surface,",
            "cavity/exterior relation, support relation, approach direction, and",
            "every concrete member ID. Nothing is selected or discarded by the",
            "bucketing stage.",
            "",
            "These are proposal buckets, not final grasp families. Final families",
            "must be constructed only after physical closure and retention succeed.",
            "",
            "## Required next gate",
            "",
            "Run every geometric survivor in the corrected table-supported Isaac",
            "test: collision-free pregrasp, complete approach, finger closure,",
            "vertical lift, and hold under gravity. Only those successes may enter",
            "the final object-centric family library.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text())
    document = make_document(config_path)
    output = (
        project_path(args.output)
        if args.output is not None
        else project_path(config["outputs"]["artifact"])
    )
    report = (
        project_path(args.report)
        if args.report is not None
        else project_path(config["outputs"]["report"])
    )
    support_visual = project_path(config["outputs"]["support_visual"])
    support_visual_reference = os.path.relpath(
        support_visual, start=report.parent
    )
    atomic_write(output, json.dumps(document, indent=2) + "\n")
    render_support_visual(document, support_visual)
    atomic_write(
        report,
        markdown_report(
            document,
            support_visual_reference=support_visual_reference,
        ),
    )
    print(
        f"[support-atlas] wrote {output}: "
        f"{document['geometric_survivor_count']} survivors in "
        f"{document['proposal_bucket_count']} buckets",
        flush=True,
    )
    print(f"[support-atlas] wrote {report}", flush=True)
    print(f"[support-atlas] wrote {support_visual}", flush=True)


if __name__ == "__main__":
    main()
