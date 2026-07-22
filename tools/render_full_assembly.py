#!/usr/bin/env python3
"""Render the saved full-assembly plan with the existing GraspGenX renderer.

The planner owns all motion and object-state decisions.  This tool only
reconstructs current-URDF visual-link poses and attached-object poses from the
saved trajectories, then hands the resulting JSON to GraspGenX's existing EGL
MP4 renderer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import trimesh
import yaml


ROOT = Path(__file__).resolve().parents[1]
GRASPGENX = ROOT / "third_party/GraspGenX"
sys.path.insert(0, str(GRASPGENX / "end2end"))

from trajectory_visualizer import URDFFK  # noqa: E402


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sample_indices(q: np.ndarray, motion_frames: int, hold_frames: int) -> np.ndarray:
    """Keep state transitions exact while limiting long solver trajectories."""
    if len(q) <= 1:
        return np.array([0], dtype=np.int64)
    limit = hold_frames if np.allclose(q[0], q[-1]) else motion_frames
    if len(q) <= limit:
        return np.arange(len(q), dtype=np.int64)
    return np.unique(np.linspace(0, len(q) - 1, limit).round().astype(np.int64))


def _finger_profiles() -> dict[str, dict[str, dict[str, float]]]:
    output = {}
    for side in ("left", "right"):
        path = GRASPGENX / f"assets/x_grippers/dex3_rev1_{side}/config.json"
        doc = json.loads(path.read_text())
        output[side] = {"open": doc["open"], "close": doc["close"]}
    return output


def _joint_values(
    arm_names: list[str],
    arm_q: np.ndarray,
    closed: dict[str, float],
    profiles: dict[str, dict[str, dict[str, float]]],
) -> dict[str, float]:
    values = {name: float(value) for name, value in zip(arm_names, arm_q)}
    for side in ("left", "right"):
        fraction = float(closed[side])
        opened = profiles[side]["open"]
        for name, close_value in profiles[side]["close"].items():
            open_value = float(opened[name])
            values[name] = open_value + fraction * (float(close_value) - open_value)
    return values


def _object_poses(
    object_states: dict[str, Any],
    tool_poses: dict[str, np.ndarray],
) -> dict[str, list[list[float]]]:
    output = {}
    for name, state in object_states.items():
        if state["world_T_object"] is not None:
            world_T_object = np.asarray(state["world_T_object"], dtype=np.float64)
        else:
            side = str(state["hand"])
            world_T_object = tool_poses[f"{side}_hand_grasp_frame"] @ np.asarray(
                state["grasp_T_object"], dtype=np.float64
            )
        output[name] = world_T_object.tolist()
    return output


def _write_box(path: Path, dimensions: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=np.asarray(dimensions, dtype=np.float64)).export(path)


def export_trajectory(
    config_path: Path,
    run_dir: Path,
    motion_frames: int,
    hold_frames: int,
) -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    report = json.loads((run_dir / "planning_report.json").read_text())
    state = json.loads((run_dir / "render_state.json").read_text())
    arrays = np.load(run_dir / "arm_trajectories.npz")

    urdf_path = _project_path(cfg["robot"]["urdf"])
    fk = URDFFK(str(urdf_path), asset_root=str(urdf_path.parent))
    arm_names = list(report["arm_joint_names"])
    profiles = _finger_profiles()
    base_T = np.eye(4)
    base_T[:3, 3] = np.asarray(
        cfg["robot"]["base_world_translation_m"], dtype=np.float64
    )

    static_dir = run_dir / "static_meshes"
    table_mesh = static_dir / "table.obj"
    _write_box(table_mesh, cfg["table"]["dimensions_m"])
    table_pose = np.eye(4)
    table_pose[:3, 3] = cfg["table"]["center_world_m"]
    static: dict[str, Any] = {
        "table": {
            "mesh_rel": str(table_mesh),
            "transform": table_pose.tolist(),
        }
    }
    for name, fixture in cfg.get("fixtures", {}).items():
        mesh_path = static_dir / f"{name}.obj"
        _write_box(mesh_path, fixture["dimensions_m"])
        pose = np.eye(4)
        pose[:3, 3] = fixture["center_world_m"]
        static[name] = {"mesh_rel": str(mesh_path), "transform": pose.tolist()}

    objects = [
        {"id": name, "mesh_rel": str(_project_path(part["mesh"]))}
        for name, part in cfg["parts"].items()
    ]
    frames = []
    timeline = []
    for segment_index, segment in enumerate(state["segments"]):
        q = arrays[segment["trajectory_key"]]
        sampled = _sample_indices(q, motion_frames, hold_frames)
        first_output_frame = len(frames)
        for q_index in sampled:
            joints = _joint_values(
                arm_names, q[int(q_index)], segment["hand_closed"], profiles
            )
            parts = [
                {
                    "name": visual.link_name,
                    "mesh_rel": visual.mesh_rel,
                    "transform": pose.tolist(),
                }
                for visual, pose in fk.link_poses_with_visual_offset(
                    joints, base_T=base_T
                )
            ]
            tools = fk.fk(
                joints,
                base_T=base_T,
                link_names=["left_hand_grasp_frame", "right_hand_grasp_frame"],
            )
            frames.append(
                {
                    "parts": parts,
                    "object_poses": _object_poses(segment["objects"], tools),
                    "stage": segment["name"],
                }
            )
        timeline.append(
            {
                "segment_index": segment_index,
                "name": segment["name"],
                "selected_candidate": segment["selected_candidate"],
                "output_frames": [first_output_frame, len(frames) - 1],
                "source_frames": int(len(q)),
            }
        )

    trajectory = {
        "schema_version": 1,
        "base_dir": str(ROOT),
        "fps": int(cfg["render"]["fps"]),
        "camera": {
            "eye": cfg["render"]["camera_eye_world_m"],
            "target": cfg["render"]["camera_target_world_m"],
            "up": [0.0, 0.0, 1.0],
        },
        "background_color": [0.97, 0.97, 0.96, 1.0],
        "static": static,
        "objects": objects,
        "frames": frames,
    }
    trajectory_path = run_dir / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory) + "\n")
    (run_dir / "timeline.json").write_text(json.dumps(timeline, indent=2) + "\n")
    return trajectory_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/planning/t_u_cube_full_assembly_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--motion-frames", type=int, default=6)
    parser.add_argument("--hold-frames", type=int, default=12)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    run_dir = args.run_dir or _project_path(cfg["render"]["output_dir"])
    trajectory_path = export_trajectory(
        args.config.resolve(), run_dir.resolve(), args.motion_frames, args.hold_frames
    )
    if args.no_render:
        print(f"trajectory exported: {trajectory_path}")
        return

    width, height = cfg["render"]["resolution"]
    output = run_dir / "full_assembly.mp4"
    subprocess.run(
        [
            sys.executable,
            str(GRASPGENX / "end2end/render_trajectory_mp4.py"),
            "--trajectory",
            str(trajectory_path),
            "--output",
            str(output),
            "--resolution",
            f"{width}x{height}",
            "--fps",
            str(cfg["render"]["fps"]),
            "--no-texture",
        ],
        check=True,
    )
    print(f"full assembly video: {output}")


if __name__ == "__main__":
    main()
