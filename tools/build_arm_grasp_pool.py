#!/usr/bin/env python3
"""Build a compact arm-planning pool from a completed intrinsic grasp atlas.

The Isaac result YAML stores the gripper's final simulated pose at the top
level.  That is deliberately *not* an arm-planning goal.  This program copies
the original, immutable ``object_T_G`` from each Isaac input and uses the
ordinary simulator result only to decide whether that candidate passed.

The output remains a lossless subset of the GraspGenX proposals: no pose is
averaged, mirrored, shifted, or regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def pose_from_grasp(grasp: dict[str, Any]) -> dict[str, Any]:
    source = grasp.get("graspgenx_source") or {}
    pose = source.get("object_T_G")
    if pose is None:
        pose = {
            "position": grasp["position"],
            "orientation": grasp["orientation"],
        }
    position = np.asarray(pose["position"], dtype=np.float64)
    quat = np.asarray(
        [pose["orientation"]["w"], *pose["orientation"]["xyz"]],
        dtype=np.float64,
    )
    if position.shape != (3,) or quat.shape != (4,):
        raise ValueError("Expected a 3-vector position and wxyz quaternion")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quat)):
        raise ValueError("Non-finite object_T_G")
    norm = float(np.linalg.norm(quat))
    if not np.isclose(norm, 1.0, atol=1e-5):
        raise ValueError(f"Non-unit object_T_G quaternion: norm={norm}")
    return {
        "position": position.tolist(),
        "orientation": {"w": float(quat[0]), "xyz": quat[1:].tolist()},
    }


def family_index(families_doc: dict[str, Any]) -> tuple[dict[str, dict], dict[str, str]]:
    families: dict[str, dict] = {}
    member_to_family: dict[str, str] = {}
    for family in families_doc["families"]:
        family_id = family["family_id"]
        families[family_id] = family
        for candidate_id in family["member_ids"]:
            if candidate_id in member_to_family:
                raise ValueError(f"Candidate belongs to two families: {candidate_id}")
            member_to_family[candidate_id] = family_id
    return families, member_to_family


def ordered_ids(
    families: dict[str, dict], candidates: dict[str, dict]
) -> list[str]:
    """Family-balanced deterministic order without changing neural scores."""
    output: list[str] = []
    seen: set[str] = set()

    # First expose one primary from every family, then each type of stored
    # pose-diverse backup.  This gives a bounded cuRobo goalset broad family
    # coverage without pretending that representatives are the whole atlas.
    for role in ("primary", "translation_diverse_backup", "pose_diverse_backup"):
        for family_id in sorted(families):
            family = families[family_id]
            for representative in family["representatives"]:
                candidate_id = representative["candidate_id"]
                if representative["role"] == role and candidate_id in candidates:
                    if candidate_id not in seen:
                        output.append(candidate_id)
                        seen.add(candidate_id)

    # Then round-robin the remaining members, ranked *within* each family by
    # their unchanged GraspGenX score.  Large families cannot monopolize the
    # beginning of the planning pool.
    remaining: dict[str, list[str]] = {}
    for family_id in sorted(families):
        ids = [
            value
            for value in families[family_id]["member_ids"]
            if value in candidates and value not in seen
        ]
        ids.sort(key=lambda value: (-candidates[value]["graspgenx_score"], value))
        remaining[family_id] = ids
    depth = 0
    while True:
        appended = False
        for family_id in sorted(remaining):
            values = remaining[family_id]
            if depth < len(values):
                candidate_id = values[depth]
                output.append(candidate_id)
                seen.add(candidate_id)
                appended = True
        if not appended:
            break
        depth += 1

    if len(output) != len(candidates) or len(seen) != len(candidates):
        missing = sorted(set(candidates) - seen)
        raise ValueError(f"Ordering lost candidates: {missing[:5]}")
    return output


def build(config_path: Path, hand_side: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    atlas_root = project_path(config["artifacts_root"])
    families_path = atlas_root / hand_side / "families.json"
    import json

    families_doc = json.loads(families_path.read_text())
    families, member_to_family = family_index(families_doc)

    role_by_id: dict[str, str] = {}
    for family in families.values():
        for representative in family["representatives"]:
            role_by_id[representative["candidate_id"]] = representative["role"]

    candidates: dict[str, dict] = {}
    input_hashes: dict[str, str] = {}
    result_hashes: dict[str, str] = {}
    input_paths = sorted((atlas_root / hand_side / "isaac_inputs").glob("shard_*.yaml"))
    if not input_paths:
        raise FileNotFoundError(f"No Isaac inputs under {atlas_root / hand_side}")

    output_directory = config["physics"].get(
        "qualification_output_directory", "physics_outputs"
    )
    object_mesh_stem = project_path(config["object"]["mesh"]).stem
    for input_path in input_paths:
        shard_name = input_path.stem
        result_path = (
            atlas_root
            / hand_side
            / output_directory
            / shard_name
            / "grasp_sim_data"
            / f"dex3_rev1_{hand_side}"
            / f"{object_mesh_stem}.yaml"
        )
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        input_doc = yaml.safe_load(input_path.read_text())
        result_doc = yaml.safe_load(result_path.read_text())
        input_grasps = input_doc["grasps"]
        result_grasps = result_doc["grasps"]
        if set(input_grasps) != set(result_grasps):
            raise ValueError(f"Input/result candidate mismatch in {shard_name}")
        input_hashes[str(input_path.relative_to(PROJECT_ROOT))] = sha256(input_path)
        result_hashes[str(result_path.relative_to(PROJECT_ROOT))] = sha256(result_path)

        for candidate_id, result in result_grasps.items():
            passed = float(result.get("confidence", 0.0)) == 1.0
            if not passed:
                continue
            if candidate_id not in member_to_family:
                raise ValueError(f"Passing candidate absent from families: {candidate_id}")
            source_grasp = input_grasps[candidate_id]
            source_meta = source_grasp.get("graspgenx_source") or {}
            if source_meta.get("candidate_id", candidate_id) != candidate_id:
                raise ValueError(f"Candidate identity changed: {candidate_id}")
            result_meta = result.get("graspgenx_source") or {}
            source_hash = source_meta.get("candidate_content_sha256")
            if result_meta.get("candidate_content_sha256") != source_hash:
                raise ValueError(f"Candidate provenance changed: {candidate_id}")
            if candidate_id in candidates:
                raise ValueError(f"Duplicate passing candidate: {candidate_id}")
            family_id = member_to_family[candidate_id]
            candidates[candidate_id] = {
                "candidate_id": candidate_id,
                "candidate_content_sha256": source_hash,
                "family_id": family_id,
                "representative_role": role_by_id.get(candidate_id),
                "graspgenx_score": float(source_meta.get("confidence", source_grasp["confidence"])),
                "object_T_G": pose_from_grasp(source_grasp),
            }

    expected = int(families_doc["pass_count"])
    if len(candidates) != expected:
        raise ValueError(f"Expected {expected} passes, collected {len(candidates)}")
    order = ordered_ids(families, candidates)
    mesh_path = project_path(config["object"]["mesh"])
    output_candidates = [candidates[candidate_id] for candidate_id in order]
    return {
        "format": "g1_aprilcube_arm_grasp_pool",
        "format_version": 1,
        "atlas_id": config["atlas_id"],
        "hand_side": hand_side,
        "object_id": config["object"]["id"],
        "object_mesh": str(mesh_path.relative_to(PROJECT_ROOT)),
        "object_mesh_sha256": sha256(mesh_path),
        "candidate_count": len(output_candidates),
        "family_count": int(families_doc["family_count"]),
        "ordering_policy": (
            "all_family_primaries_then_translation_backups_then_rotation_backups_"
            "then_family_round_robin_by_unchanged_graspgenx_score"
        ),
        "source": {
            "atlas_config": str(config_path.relative_to(PROJECT_ROOT)),
            "atlas_config_sha256": sha256(config_path),
            "families": str(families_path.relative_to(PROJECT_ROOT)),
            "families_sha256": sha256(families_path),
            "isaac_input_sha256": input_hashes,
            "physics_result_sha256": result_hashes,
            "pose_policy": "original_object_T_G_copied_from_isaac_input",
            "admission_policy": "ordinary_isaac_physics_confidence_equals_one",
        },
        "candidates": output_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hand-side", choices=("right", "left"), default="right")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = project_path(args.config)
    output = build(config_path, args.hand_side)
    if args.output is None:
        config = yaml.safe_load(config_path.read_text())
        output_path = (
            project_path(config["artifacts_root"])
            / args.hand_side
            / "arm_grasp_pool.yaml"
        )
    else:
        output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(output, sort_keys=False))
    print(
        f"Wrote {output_path}: {output['candidate_count']} physics-passing "
        f"candidates in {output['family_count']} families"
    )


if __name__ == "__main__":
    main()
