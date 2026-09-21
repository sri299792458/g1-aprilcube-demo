#!/usr/bin/env python3
"""Import a GraspGenX hand descriptor with Isaac Lab's URDF converter."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf",
        type=Path,
        default=repo / "third_party/GraspGenX/assets/x_grippers/dex3_rev1_right/gripper.urdf",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "third_party/GraspDataGen/bots/dex3_rev1_right",
    )
    parser.add_argument("--usd-file-name", default="dex3_rev1_right.usd")
    parser.add_argument(
        "--merge-fixed-joints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Merge fixed URDF links during import. The hand-only default keeps "
            "this disabled so the descriptor palm remains a separately named "
            "PhysX articulation body."
        ),
    )
    parser.add_argument(
        "--self-collision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable collisions between links of the imported hand articulation. "
            "The GraspGenX Newton playback contract disables these collisions."
        ),
    )
    parser.add_argument(
        "--high-stiffness-joint-pattern",
        default="right_hand_thumb_0_joint",
    )
    parser.add_argument(
        "--remaining-joint-pattern",
        default="right_hand_(thumb_[12]|middle_[01]|index_[01])_joint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulation_app = AppLauncher(headless=True).app
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        args.output_dir.mkdir(parents=True, exist_ok=True)
        cfg = UrdfConverterCfg(
            asset_path=str(args.urdf.resolve()),
            usd_dir=str(args.output_dir.resolve()),
            usd_file_name=args.usd_file_name,
            force_usd_conversion=True,
            make_instanceable=True,
            fix_base=True,
            root_link_name="world",
            merge_fixed_joints=args.merge_fixed_joints,
            collider_type="convex_decomposition",
            self_collision=args.self_collision,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness={
                        args.high_stiffness_joint_pattern: 2.0,
                        args.remaining_joint_pattern: 0.5,
                    },
                    damping={
                        args.high_stiffness_joint_pattern: 0.1,
                        args.remaining_joint_pattern: 0.1,
                    },
                ),
            ),
        )
        converter = UrdfConverter(cfg)
        print(converter.usd_path)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
