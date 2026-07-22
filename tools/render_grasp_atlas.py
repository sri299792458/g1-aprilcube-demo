#!/usr/bin/env python3
"""Replay real atlas representatives in Isaac and write one sequential MP4."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config/grasp_atlas/cube_v1.yaml"
GRASP_DATA_GEN = PROJECT_ROOT / "third_party/GraspDataGen"
APPROACH_ORDER = {value: index for index, value in enumerate((
    "+X", "-X", "+Y", "-Y", "+Z", "-Z"
))}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(contents)
        temporary = Path(stream.name)
    temporary.replace(path)


def family_sort_key(family: dict) -> tuple:
    signature = family["signature"]
    digits = tuple(signature["digit_chains"])
    return (
        APPROACH_ORDER[signature["approach_sector"]],
        bool(signature["palm_contact"]),
        len(digits),
        digits,
        -int(family["member_count"]),
        family["family_id"],
    )


def load_source_grasps(input_dir: Path) -> tuple[dict, dict[str, dict], list[Path]]:
    paths = sorted(input_dir.glob("shard_*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No Isaac input shards under {input_dir}")
    base = None
    grasps: dict[str, dict] = {}
    for path in paths:
        payload = yaml.safe_load(path.read_text())
        if base is None:
            base = copy.deepcopy(payload)
        for candidate_id, grasp in payload["grasps"].items():
            if candidate_id in grasps:
                raise ValueError(f"Duplicate candidate {candidate_id} in {path}")
            grasps[candidate_id] = grasp
    assert base is not None
    return base, grasps, paths


def build_review_input(
    *, families: dict, base: dict, source_grasps: dict[str, dict], source_paths: list[Path]
) -> tuple[dict, list[dict]]:
    selected = []
    review_grasps = {}
    for family in sorted(families["families"], key=family_sort_key):
        primaries = [
            value for value in family["representatives"] if value["role"] == "primary"
        ]
        if len(primaries) != 1:
            raise ValueError(f"{family['family_id']} has {len(primaries)} primaries")
        representative = primaries[0]
        candidate_id = representative["candidate_id"]
        if candidate_id not in source_grasps:
            raise KeyError(f"Representative missing from Isaac inputs: {candidate_id}")
        grasp = copy.deepcopy(source_grasps[candidate_id])
        if grasp.get("graspgenx_source", {}).get("candidate_content_sha256") != \
                representative["candidate_content_sha256"]:
            raise ValueError(f"Candidate provenance changed: {candidate_id}")
        signature = family["signature"]
        metadata = {
            "family_id": family["family_id"],
            "member_count": int(family["member_count"]),
            "digit_chains": signature["digit_chains"],
            "palm_contact": bool(signature["palm_contact"]),
            "approach_sector": signature["approach_sector"],
            "diagnostic_broad_faces_by_chain": representative.get(
                "diagnostic_broad_faces_by_chain", {}
            ),
        }
        grasp["grasp_atlas_family"] = metadata
        review_grasps[candidate_id] = grasp
        selected.append({"candidate_id": candidate_id, **metadata})

    output = copy.deepcopy(base)
    output["created_with"] = "g1_aprilcube_grasp_atlas_review"
    output["grasps"] = review_grasps
    output["source"] = {
        "policy": "primary physics-passing representative from each right-hand family",
        "isaac_inputs": [str(path.relative_to(PROJECT_ROOT)) for path in source_paths],
        "transform_policy": "copied_verbatim_from_qualified_isaac_input",
    }
    output["grasp_atlas_review"] = {
        "schema_version": 1,
        "hand_side": families["hand_side"],
        "family_count": len(selected),
        "selected": selected,
    }
    return output, selected


def ffprobe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
            "-show_entries", "format=duration", "-of", "json", str(path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hand-side", choices=("right",), default="right")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs/assets/dex3_cube_grasp_families_right.mp4",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="Replace an existing review MP4 and its reproducible replay artifacts.",
    )
    parser.add_argument(
        "--allow-replay-failures",
        action="store_true",
        help=(
            "Retain one-shot physics-passing representatives that fail the "
            "independent review replay. Their MP4 segments and manifest "
            "entries remain explicit FAIL evidence in this diagnostic rerun."
        ),
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.resolve().read_text())
    artifacts_root = project_path(config["artifacts_root"])
    hand_dir = artifacts_root / args.hand_side
    families_path = hand_dir / "families.json"
    families = json.loads(families_path.read_text())
    if families["hand_side"] != args.hand_side:
        raise ValueError("Family file hand side does not match request")
    base, source_grasps, source_paths = load_source_grasps(hand_dir / "isaac_inputs")
    review_input, selected = build_review_input(
        families=families,
        base=base,
        source_grasps=source_grasps,
        source_paths=source_paths,
    )

    review_dir = hand_dir / "review"
    input_path = review_dir / "primary_representatives_input.yaml"
    result_dir = review_dir / "isaac_replay"
    output = args.output.resolve()
    result_yaml = (
        result_dir / "grasp_sim_data" / "dex3_rev1_right" / "grasp_mesh.yaml"
    )
    existing = [path for path in (output, input_path, result_yaml) if path.exists()]
    if existing and not args.replace:
        raise FileExistsError(
            "Refusing to replace review artifacts; use --replace intentionally: "
            + ", ".join(str(path) for path in existing)
        )
    atomic_write_text(input_path, yaml.safe_dump(review_input, sort_keys=False))

    command = [
        str(GRASP_DATA_GEN / ".venv/bin/python"),
        "scripts/graspgen/grasp_sim.py",
        "--grasp_file", str(input_path),
        "--max_num_envs", str(len(selected)),
        "--headless", "--enable_cameras",
        "--output_failed_grasp_locations",
        "--capture_video", str(output),
        "--capture_video_fps", str(args.fps),
        "--capture_video_layout", "sequential",
    ]
    environment = os.environ.copy()
    environment["OMNI_KIT_ACCEPT_EULA"] = "Y"
    environment["GRASP_DATASET_DIR"] = str(result_dir)
    print(f"Replaying {len(selected)} right-hand family representatives in Isaac")
    subprocess.run(command, cwd=GRASP_DATA_GEN, env=environment, check=True)

    if not result_yaml.exists():
        raise RuntimeError(
            "Isaac replay returned without writing its expected physics result: "
            f"{result_yaml}"
        )
    if not output.exists():
        raise RuntimeError(
            "Isaac replay returned without writing its expected video: "
            f"{output}"
        )

    replay = yaml.safe_load(result_yaml.read_text())
    replay_grasps = replay["grasps"]
    missing = [item["candidate_id"] for item in selected if item["candidate_id"] not in replay_grasps]
    failed = [
        item["candidate_id"]
        for item in selected
        if float(replay_grasps.get(item["candidate_id"], {}).get("confidence", 0.0)) != 1.0
    ]
    if missing or (failed and not args.allow_replay_failures):
        raise RuntimeError(
            f"Representative replay did not reproduce stored PASS: missing={missing}, failed={failed}"
        )
    media = ffprobe(output)
    selected_with_results = [
        {
            **item,
            "replay_passed": (
                float(replay_grasps[item["candidate_id"]]["confidence"]) == 1.0
            ),
        }
        for item in selected
    ]
    manifest = {
        "schema_version": 1,
        "hand_side": args.hand_side,
        "family_count": len(selected),
        "all_replays_passed": not failed,
        "replay_pass_count": len(selected) - len(failed),
        "replay_fail_count": len(failed),
        "failed_replays": failed,
        "input": str(input_path.relative_to(PROJECT_ROOT)),
        "physics_output": str(result_yaml.relative_to(PROJECT_ROOT)),
        "video": str(output.relative_to(PROJECT_ROOT)),
        "video_probe": media,
        "selected": selected_with_results,
    }
    manifest_path = review_dir / "review_manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    if failed:
        print(
            f"Completed sequential review MP4 with {len(failed)} explicit "
            f"replay failures: {output}"
        )
    else:
        print(f"Verified sequential review MP4: {output}")
    print(json.dumps(media, indent=2))


if __name__ == "__main__":
    main()
