#!/usr/bin/env python3
"""Validate Dex3 controller profiles against the generated exact-hand URDFs."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "dex3_newton_control_profiles.yaml"
DESCRIPTOR_ROOT = ROOT / "generated" / "graspgenx_assets" / "x_grippers"


def full_name(side: str, suffix: str) -> str:
    return f"{side}_{suffix}"


def finite_nonnegative(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative, got {value!r}")
    return number


def actuated_urdf_joints(path: Path) -> set[str]:
    root = ET.parse(path).getroot()
    return {
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") not in {"fixed", "floating"}
    }


def validate_profile(
    name: str,
    profile: dict[str, Any],
    suffixes: list[str],
    side: str,
) -> None:
    gains = profile["gains"]
    if set(gains) != set(suffixes):
        raise ValueError(f"{name}: gains must cover exactly the seven joint suffixes")
    for suffix, values in gains.items():
        finite_nonnegative(values["kp"], f"{name}.{suffix}.kp")
        finite_nonnegative(values["kd"], f"{name}.{suffix}.kd")

    effort = profile["effort_limit_nm"]
    if "default" in effort:
        finite_nonnegative(effort["default"], f"{name}.effort.default")
    elif set(effort) == set(suffixes):
        for suffix, value in effort.items():
            finite_nonnegative(value, f"{name}.{suffix}.effort")
    else:
        raise ValueError(f"{name}: effort limits must be default or seven-joint map")

    armature = profile["armature_kg_m2"]
    selected = armature.get(side, armature)
    if "default" in selected:
        finite_nonnegative(selected["default"], f"{name}.armature.default")
    elif set(selected) == set(suffixes):
        for suffix, value in selected.items():
            finite_nonnegative(value, f"{name}.{side}.{suffix}.armature")
    else:
        raise ValueError(f"{name}: armature must be default or per-side seven-joint map")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    data = yaml.safe_load(args.config.read_text())
    suffixes = list(data["joint_suffixes"])
    if len(suffixes) != 7 or len(set(suffixes)) != 7:
        raise ValueError("joint_suffixes must contain seven unique Dex3 joints")

    for side in ("right", "left"):
        urdf = DESCRIPTOR_ROOT / f"dex3_rev1_{side}" / "gripper.urdf"
        expected = {full_name(side, suffix) for suffix in suffixes}
        actual = actuated_urdf_joints(urdf)
        if actual != expected:
            raise ValueError(
                f"{side} URDF joint mismatch: missing={sorted(expected-actual)}, "
                f"extra={sorted(actual-expected)}"
            )
        for name, profile in data["profiles"].items():
            validate_profile(name, profile, suffixes, side)
        print(f"{side}: 7/7 URDF joints and all profiles valid")

    print("No profile has been physics-qualified; validation is structural only.")


if __name__ == "__main__":
    main()
