#!/usr/bin/env python3
"""Run all released/current sweep-volume triplet combinations on one cube sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPGENX_ROOT = PROJECT_ROOT / "third_party/GraspGenX"
sys.path.insert(0, str(GRASPGENX_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from graspgenx.dataset.eval_utils import save_to_isaac_grasp_format  # noqa: E402
from graspgenx.grasp_server import GraspGenXSampler  # noqa: E402
from graspgenx.utils.checkpoint_io import load_model_cfg  # noqa: E402
from graspgenx.x_grippers import make_sweep_volume_gripper_info  # noqa: E402
from run_aprilcube_raw_grasps import sample_centered  # noqa: E402


GROUPS = ("extents", "offset", "extents2", "offset2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument(
        "--released-config",
        type=Path,
        default=GRASPGENX_ROOT
        / "ext/gripper_descriptions/gripper_descriptions/assets/x_grippers/unitree_g1/config.json",
    )
    parser.add_argument(
        "--current-config",
        type=Path,
        default=GRASPGENX_ROOT / "assets/x_grippers/dex3_rev1_right/config.json",
    )
    parser.add_argument(
        "--object-mesh",
        type=Path,
        default=PROJECT_ROOT / "generated/aprilcube_parts/cube_head/grasp_mesh.obj",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/dex3_sweep_ablation/raw",
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--sample-points", type=int, default=3500)
    parser.add_argument("--num-grasps", type=int, default=480)
    parser.add_argument("--top-k", type=int, default=120)
    args = parser.parse_args()

    released = json.loads(args.released_config.read_text())
    current = json.loads(args.current_config.read_text())
    released_sweep = released["sweep_volume"]
    current_sweep = current["sweep_volume"]
    model_cfg = load_model_cfg(
        str(args.checkpoints.resolve() / "gen"),
        str(args.checkpoints.resolve() / "dis"),
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "meaning": "mask bits follow extents,offset,extents2,offset2; 1 selects released, 0 current",
        "seed": args.seed,
        "sample_points": args.sample_points,
        "requested_grasps": args.num_grasps,
        "retained_top_k": args.top_k,
        "object_mesh": str(args.object_mesh.resolve()),
        "object_mesh_sha256": sha256(args.object_mesh),
        "released_config": str(args.released_config.resolve()),
        "released_config_sha256": sha256(args.released_config),
        "current_config": str(args.current_config.resolve()),
        "current_config_sha256": sha256(args.current_config),
        "variants": {},
    }

    shared_model = None
    for mask_value in range(16):
        mask = f"{mask_value:04b}"
        sweep = {}
        source_by_group = {}
        for bit_index, group in enumerate(GROUPS):
            use_released = mask[bit_index] == "1"
            sweep[group] = list(
                (released_sweep if use_released else current_sweep)[group]
            )
            source_by_group[group] = "released" if use_released else "current"

        params = {
            "extents_open": sweep["extents"],
            "offset_open": sweep["offset"],
            "extents_mid": sweep["extents2"],
            "offset_mid": sweep["offset2"],
            "gripper_type": 2,
            # This field is not consumed by the released sweep_volume_v2
            # generator/discriminator; keep it fixed across every ablation.
            "fingertip_depth": 0.07,
        }
        gripper_info = make_sweep_volume_gripper_info(
            extents_open=params["extents_open"],
            offset_open=params["offset_open"],
            extents_mid=params["extents_mid"],
            offset_mid=params["offset_mid"],
            gripper_type=params["gripper_type"],
            fingertip_depth=params["fingertip_depth"],
            name=f"sweep_ablation_{mask}",
        )
        sampler = GraspGenXSampler(
            model_cfg,
            gripper_info=gripper_info,
            model=shared_model,
        )
        if shared_model is None:
            shared_model = sampler.model

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        points, center_transform, _ = sample_centered(
            args.object_mesh,
            args.sample_points,
            args.seed,
        )
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
            raise RuntimeError(f"{mask}: expected {args.top_k} grasps, got {len(grasps)}")
        grasp_array = grasps.detach().cpu().numpy()
        confidence_array = confidences.detach().cpu().numpy()
        grasp_array[:, 3, 3] = 1.0
        object_grasps = np.asarray(
            [np.linalg.inv(center_transform) @ grasp for grasp in grasp_array]
        )
        variant_dir = args.output_root / f"mask_{mask}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        output = variant_dir / "cube_head.yaml"
        save_to_isaac_grasp_format(object_grasps, confidence_array, str(output))
        summary["variants"][mask] = {
            "source_by_group": source_by_group,
            "sweep_volume": sweep,
            "output": str(output.resolve()),
            "output_sha256": sha256(output),
            "confidence_min": float(confidence_array.min()),
            "confidence_max": float(confidence_array.max()),
            "confidence_mean": float(confidence_array.mean()),
        }
        print(
            f"mask {mask}: confidence "
            f"{confidence_array.min():.3f}-{confidence_array.max():.3f}, "
            f"mean {confidence_array.mean():.3f}"
        )

    summary_path = args.output_root.parent / "inference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
