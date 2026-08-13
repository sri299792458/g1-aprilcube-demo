#!/usr/bin/env python3
"""Build one evidence-backed runtime grasp shortlist."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import trimesh
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_aprilcube_demo.grasping.executable_shortlist import (  # noqa: E402
    OpenHandGeometry,
    build_shortlist,
    load_trace_records,
    sha256,
)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/grasp_shortlists/cube_right_executable_v1.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text())
    object_mesh_path = project_path(config["object"]["mesh"])
    hand_visual_path = project_path(config["hand"]["visual_mesh"])
    hand_collision_path = project_path(config["hand"]["collision_mesh"])
    hand_urdf_path = project_path(config["hand"]["urdf"])
    source_pool_path = project_path(config["source"]["arm_grasp_pool"])
    source_pool = yaml.safe_load(source_pool_path.read_text())
    trace_patterns = [
        str(project_path(value))
        for value in config["source"]["contact_trace_globs"]
    ]

    object_mesh = trimesh.load(object_mesh_path, force="mesh")
    geometry = OpenHandGeometry(
        object_mesh=object_mesh,
        hand_visual_mesh=trimesh.load(hand_visual_path, force="mesh"),
        hand_collision_mesh=trimesh.load(hand_collision_path, force="mesh"),
        hand_urdf_path=hand_urdf_path,
        approach_distance_m=float(
            config["execution_contract"]["approach_distance_m"]
        ),
        numerical_tolerance_m=float(
            config["execution_contract"]["numerical_geometry_tolerance_m"]
        ),
    )
    trace_records = load_trace_records(trace_patterns)
    document = build_shortlist(
        config=config,
        source_pool=source_pool,
        trace_records=trace_records,
        geometry=geometry,
        object_mesh_path=object_mesh_path,
        source_paths={
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256(config_path),
            "arm_grasp_pool": str(source_pool_path.relative_to(ROOT)),
            "arm_grasp_pool_sha256": sha256(source_pool_path),
            "contact_trace_globs": config["source"]["contact_trace_globs"],
        },
    )
    output = project_path(args.output or config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(document, sort_keys=False))
    print(
        f"Wrote {output}: {document['candidate_count']} executable candidates "
        f"from {document['audit']['source_candidate_count']} retention passes"
    )


if __name__ == "__main__":
    main()
