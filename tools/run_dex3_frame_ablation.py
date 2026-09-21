#!/usr/bin/env python3
"""Test a geometry-derived Dex3 grasp frame without changing the canonical asset.

The current Dex3 open posture is L-shaped.  This diagnostic derives a frame in
which +X follows the open thumb-to-opponents separation and +Z is the
corresponding approach bisector, obtains the open/half boxes from the upstream
GraspGenX wizard's estimator using only the three terminal finger geometries,
and asks the released model for candidates in that frame.

Every returned ``object_T_G_aligned`` is converted back to the current asset's
unchanged frame before it is written.  The existing current-Dex3 Isaac asset can
therefore evaluate all variants without a second hand model or a hand-authored
grasp transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import yourdfpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPGENX_ROOT = PROJECT_ROOT / "third_party/GraspGenX"
sys.path.insert(0, str(GRASPGENX_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from graspgenx.dataset.eval_utils import save_to_isaac_grasp_format  # noqa: E402
from graspgenx.grasp_server import GraspGenXSampler  # noqa: E402
from graspgenx.utils.checkpoint_io import load_model_cfg  # noqa: E402
from graspgenx.x_grippers import make_sweep_volume_gripper_info  # noqa: E402
from run_aprilcube_raw_grasps import sample_centered  # noqa: E402
from scripts.gripper_config_wizard import (  # noqa: E402
    estimate_inner_sweep_volume,
    interpolate_joint_states,
)


TERMINAL_GEOMETRIES = (
    "right_hand_thumb_2_link.STL",
    "right_hand_middle_1_link.STL",
    "right_hand_index_1_link.STL",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry_centroid(robot: yourdfpy.URDF, name: str) -> np.ndarray:
    geometry = robot.scene.geometry[name]
    root_T_geometry = robot.scene.graph.get(name)[0]
    return trimesh.transformations.transform_points(
        geometry.vertices, root_T_geometry
    ).mean(axis=0)


def aligned_frame(robot: yourdfpy.URDF, open_joints: dict[str, float]) -> tuple[np.ndarray, dict]:
    """Return ``G_aligned_T_G_current`` and its measured construction evidence."""
    robot.update_cfg(open_joints)
    centroids = np.asarray(
        [geometry_centroid(robot, name) for name in TERMINAL_GEOMETRIES]
    )
    thumb = centroids[0]
    opponents = centroids[1:].mean(axis=0)
    separation = thumb - opponents

    # Preserve +Y (the index/middle separation axis) and rotate in XZ so the
    # thumb/opponents separation has no Z component in the aligned frame.
    angle = float(np.arctan2(separation[2], separation[0]))
    transform = trimesh.transformations.rotation_matrix(angle, [0.0, 1.0, 0.0])
    aligned_separation = transform[:3, :3] @ separation
    evidence = {
        "terminal_geometry_names": list(TERMINAL_GEOMETRIES),
        "terminal_centroids_in_current_G_m": centroids.tolist(),
        "thumb_minus_opponents_in_current_G_m": separation.tolist(),
        "rotation_about_y_rad": angle,
        "rotation_about_y_deg": float(np.rad2deg(angle)),
        "thumb_minus_opponents_in_aligned_G_m": aligned_separation.tolist(),
    }
    return transform, evidence


def estimate_boxes(
    robot: yourdfpy.URDF,
    open_joints: dict[str, float],
    close_joints: dict[str, float],
    frame: np.ndarray,
) -> dict:
    half_joints = interpolate_joint_states(open_joints, close_joints, 0.5)
    open_extents, open_offset, open_axis = estimate_inner_sweep_volume(
        robot,
        open_joints,
        list(TERMINAL_GEOMETRIES),
        base_T=frame,
    )
    mid_extents, mid_offset, mid_axis = estimate_inner_sweep_volume(
        robot,
        half_joints,
        list(TERMINAL_GEOMETRIES),
        base_T=frame,
    )
    return {
        "extents": open_extents,
        "offset": open_offset,
        "extents2": mid_extents,
        "offset2": mid_offset,
        "estimated_open_closing_axis": open_axis,
        "estimated_mid_closing_axis": mid_axis,
    }


def shifted_variant(
    name: str,
    base_frame: np.ndarray,
    base_boxes: dict,
    translation: np.ndarray,
) -> dict:
    frame = base_frame.copy()
    frame[:3, 3] = translation
    return {
        "name": name,
        "aligned_G_T_current_G": frame,
        "sweep_volume": {
            "extents": list(base_boxes["extents"]),
            "offset": (np.asarray(base_boxes["offset"]) + translation).tolist(),
            "extents2": list(base_boxes["extents2"]),
            "offset2": (np.asarray(base_boxes["offset2"]) + translation).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument(
        "--current-config",
        type=Path,
        default=GRASPGENX_ROOT / "assets/x_grippers/dex3_rev1_right/config.json",
    )
    parser.add_argument(
        "--current-urdf",
        type=Path,
        default=GRASPGENX_ROOT / "assets/x_grippers/dex3_rev1_right/gripper.urdf",
    )
    parser.add_argument(
        "--released-config",
        type=Path,
        default=GRASPGENX_ROOT
        / "ext/gripper_descriptions/gripper_descriptions/assets/x_grippers/unitree_g1/config.json",
    )
    parser.add_argument(
        "--object-mesh",
        type=Path,
        default=PROJECT_ROOT / "generated/aprilcube_parts/cube_head/grasp_mesh.obj",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/dex3_frame_ablation/raw",
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--sample-points", type=int, default=3500)
    parser.add_argument("--num-grasps", type=int, default=480)
    parser.add_argument("--top-k", type=int, default=120)
    parser.add_argument(
        "--variants",
        default="",
        help="Optional comma-separated variant names to run; empty runs every variant.",
    )
    args = parser.parse_args()

    current = json.loads(args.current_config.read_text())
    released = json.loads(args.released_config.read_text())
    robot = yourdfpy.URDF.load(
        str(args.current_urdf), build_scene_graph=True, load_meshes=True
    )
    base_frame, frame_evidence = aligned_frame(robot, current["open"])
    boxes = estimate_boxes(robot, current["open"], current["close"], base_frame)
    if boxes["estimated_open_closing_axis"] != 0:
        raise RuntimeError("Derived frame did not align the open separation with +X")

    open_offset = np.asarray(boxes["offset"], dtype=float)
    center_xy = np.array([-open_offset[0], -open_offset[1], 0.0])
    center_xy_z70 = center_xy.copy()
    center_xy_z70[2] = 0.070 - open_offset[2]
    variants = [
        shifted_variant("aligned_rotate_only", base_frame, boxes, np.zeros(3)),
        shifted_variant("aligned_center_xy", base_frame, boxes, center_xy),
        shifted_variant("aligned_center_xy_z70", base_frame, boxes, center_xy_z70),
    ]

    # The released checkpoints use X as aperture and Z as approach depth.  In
    # the current L-shaped open posture the wizard detects the physical gap on
    # current-frame Z.  These variants keep the execution frame unchanged and
    # test whether that measured aperture belongs in the descriptor's semantic
    # X slot.  The 60/40 mm transverse/depth values are not invented: they are
    # the corresponding values of the released Unitree descriptor.
    identity = np.eye(4)
    current_gap_open = estimate_inner_sweep_volume(
        robot, current["open"], list(TERMINAL_GEOMETRIES)
    )
    current_gap_mid = estimate_inner_sweep_volume(
        robot,
        interpolate_joint_states(current["open"], current["close"], 0.5),
        list(TERMINAL_GEOMETRIES),
    )
    semantic_dimensions = {
        "extents": [
            current_gap_open[0][current_gap_open[2]],
            released["sweep_volume"]["extents"][1],
            released["sweep_volume"]["extents"][2],
        ],
        "extents2": [
            current_gap_mid[0][current_gap_mid[2]],
            released["sweep_volume"]["extents2"][1],
            released["sweep_volume"]["extents2"][2],
        ],
    }
    semantic_geometry_offsets = {
        **semantic_dimensions,
        "offset": [0.0, 0.0, current_gap_open[1][2]],
        "offset2": [0.0, 0.0, current_gap_mid[1][2]],
    }
    released_offsets = {
        **semantic_dimensions,
        "offset": list(released["sweep_volume"]["offset"]),
        "offset2": list(released["sweep_volume"]["offset2"]),
    }
    released_dimensions_geometry_offsets = {
        "extents": list(released["sweep_volume"]["extents"]),
        "offset": [0.0, 0.0, current_gap_open[1][2]],
        "extents2": list(released["sweep_volume"]["extents2"]),
        "offset2": [0.0, 0.0, current_gap_mid[1][2]],
    }
    for name, sweep in (
        ("current_frame_semantic_geometry", semantic_geometry_offsets),
        ("current_frame_semantic_released_offsets", released_offsets),
        (
            "current_frame_released_dims_geometry_offsets",
            released_dimensions_geometry_offsets,
        ),
        ("current_frame_released_sweep", released["sweep_volume"]),
    ):
        variants.append(
            {
                "name": name,
                "aligned_G_T_current_G": identity.copy(),
                "sweep_volume": {
                    key: list(sweep[key])
                    for key in ("extents", "offset", "extents2", "offset2")
                },
            }
        )

    if args.variants:
        requested_variants = {
            value.strip() for value in args.variants.split(",") if value.strip()
        }
        available_variants = {variant["name"] for variant in variants}
        unknown_variants = sorted(requested_variants - available_variants)
        if unknown_variants:
            raise ValueError(f"Unknown variants: {unknown_variants}")
        variants = [
            variant for variant in variants if variant["name"] in requested_variants
        ]

    model_cfg = load_model_cfg(
        str(args.checkpoints.resolve() / "gen"),
        str(args.checkpoints.resolve() / "dis"),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    points, center_transform, _ = sample_centered(
        args.object_mesh, args.sample_points, args.seed
    )
    summary = {
        "schema_version": 1,
        "purpose": "canonical-frame ablation for the current L-shaped Dex3 open posture",
        "seed": args.seed,
        "sample_points": args.sample_points,
        "requested_grasps": args.num_grasps,
        "retained_top_k": args.top_k,
        "current_config": str(args.current_config.resolve()),
        "current_config_sha256": sha256(args.current_config),
        "current_urdf": str(args.current_urdf.resolve()),
        "current_urdf_sha256": sha256(args.current_urdf),
        "released_config": str(args.released_config.resolve()),
        "released_config_sha256": sha256(args.released_config),
        "object_mesh": str(args.object_mesh.resolve()),
        "object_mesh_sha256": sha256(args.object_mesh),
        "frame_derivation": frame_evidence,
        "upstream_wizard_terminal_box_estimate_before_origin_shift": boxes,
        "upstream_wizard_terminal_box_estimate_in_current_frame": {
            "open": {
                "extents": current_gap_open[0],
                "offset": current_gap_open[1],
                "closing_axis": current_gap_open[2],
            },
            "mid": {
                "extents": current_gap_mid[0],
                "offset": current_gap_mid[1],
                "closing_axis": current_gap_mid[2],
            },
        },
        "variants": {},
    }

    shared_model = None
    for variant in variants:
        name = variant["name"]
        sweep = variant["sweep_volume"]
        gripper_info = make_sweep_volume_gripper_info(
            extents_open=sweep["extents"],
            offset_open=sweep["offset"],
            extents_mid=sweep["extents2"],
            offset_mid=sweep["offset2"],
            gripper_type=2,
            fingertip_depth=0.07,
            name=name,
        )
        sampler = GraspGenXSampler(
            model_cfg, gripper_info=gripper_info, model=shared_model
        )
        if shared_model is None:
            shared_model = sampler.model

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        grasps, confidences = GraspGenXSampler.run_inference(
            points,
            sampler,
            grasp_threshold=-1.0,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.top_k,
            min_grasps=args.top_k,
            max_tries=1,
            remove_outliers=False,
        )
        if len(grasps) != args.top_k:
            raise RuntimeError(f"{name}: expected {args.top_k} grasps, got {len(grasps)}")

        centered_T_aligned = grasps.detach().cpu().numpy()
        centered_T_aligned[:, 3, 3] = 1.0
        aligned_T_current = variant["aligned_G_T_current_G"]
        object_T_current = np.asarray(
            [
                np.linalg.inv(center_transform)
                @ centered_pose
                @ aligned_T_current
                for centered_pose in centered_T_aligned
            ]
        )
        confidence_array = confidences.detach().cpu().numpy()
        variant_dir = args.output_root / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        output = variant_dir / "cube_head.yaml"
        save_to_isaac_grasp_format(object_T_current, confidence_array, str(output))
        summary["variants"][name] = {
            "aligned_G_T_current_G": aligned_T_current.tolist(),
            "sweep_volume": sweep,
            "output_frame": "current canonical G used by the unchanged Isaac hand asset",
            "pose_conversion": "object_T_current_G = object_T_aligned_G @ aligned_G_T_current_G",
            "output": str(output.resolve()),
            "output_sha256": sha256(output),
            "confidence_min": float(confidence_array.min()),
            "confidence_max": float(confidence_array.max()),
            "confidence_mean": float(confidence_array.mean()),
        }
        print(
            f"{name}: confidence {confidence_array.min():.3f}-"
            f"{confidence_array.max():.3f}, mean {confidence_array.mean():.3f}"
        )

    summary_path = args.output_root.parent / "inference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
