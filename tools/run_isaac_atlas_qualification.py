#!/usr/bin/env python3
"""Run resumable VIRAL-profile Isaac qualification over existing raw shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASP_DATA_GEN = PROJECT_ROOT / "third_party/GraspDataGen"
DEFAULT_CONFIGS = (
    PROJECT_ROOT / "config/grasp_atlas/cube_viral_v1.yaml",
    PROJECT_ROOT / "config/grasp_atlas/t_body_viral_v1.yaml",
    PROJECT_ROOT / "config/grasp_atlas/u_legs_viral_v1.yaml",
)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_shards(value: str, count: int) -> list[int]:
    if value == "all":
        return list(range(count))
    result = sorted({int(token) for token in value.split(",") if token.strip()})
    if not result or result[0] < 0 or result[-1] >= count:
        raise ValueError(f"Shard selection must be inside [0, {count - 1}]")
    return result


def expected_result_path(run_dir: Path, hand_side: str) -> Path:
    return (
        run_dir
        / "grasp_sim_data"
        / f"dex3_rev1_{hand_side}"
        / "grasp_mesh.yaml"
    )


def validate_completed_run(
    *,
    input_path: Path,
    result_path: Path,
    trace_path: Path,
    expected_count: int,
    simulation_profile: str,
) -> dict:
    source = yaml.safe_load(input_path.read_text())
    result = yaml.safe_load(result_path.read_text())
    source_ids = list(source["grasps"])[:expected_count]
    all_result_ids = list(result["grasps"])
    result_ids = all_result_ids[:expected_count]
    if result_ids != source_ids:
        raise ValueError(
            f"Result candidate identity/order mismatch in {result_path}"
        )
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    if len(records) != expected_count:
        raise ValueError(
            f"{trace_path} contains {len(records)} trials; expected {expected_count}"
        )
    trace_ids = [record["candidate_id"] for record in records]
    if trace_ids != source_ids:
        raise ValueError(f"Trace candidate identity/order mismatch in {trace_path}")
    bad_profiles = [
        record["candidate_id"]
        for record in records
        if record["physics"].get("simulation_profile") != simulation_profile
    ]
    if bad_profiles:
        raise ValueError(
            f"Trace records do not carry {simulation_profile}: {bad_profiles[:5]}"
        )
    for record in records:
        if record["physics"].get("contact_force_measurement") != (
            "max_norm_over_object_filtered_body_pairs"
        ):
            raise ValueError(
                f"Trace {record['candidate_id']} lacks the body-pair scalar contract"
            )
        for phase in record["phases"]:
            for contact in phase["contacts"]:
                magnitude = contact.get("contact_force_magnitude_N")
                if magnitude is None or float(magnitude) < 0.0:
                    raise ValueError(
                        f"Trace {record['candidate_id']} has an invalid contact scalar"
                    )
        final_phase = record["phases"][-1]
        if final_phase["name"] != "after_tug_5_final":
            raise ValueError(
                f"Trace {record['candidate_id']} has the wrong qualified phase"
            )
        active_digit_chains = {
            digit
            for contact in final_phase["contacts"]
            for digit in ("thumb", "index", "middle")
            if f"_hand_{digit}_" in contact["physx_body"]
            and float(contact["contact_force_magnitude_N"]) > 0.0
        }
        scalar_pass = len(active_digit_chains) >= int(
            record["physics"]["min_contact_groups"]
        )
        if scalar_pass is not bool(record["result"]["passed"]):
            raise ValueError(
                f"Trace {record['candidate_id']} scalar contacts disagree with PASS"
            )
    passes = sum(bool(record["result"]["passed"]) for record in records)
    ordinary_passes = sum(
        float(result["grasps"][candidate_id]["confidence"]) == 1.0
        for candidate_id in result_ids
    )
    if passes != ordinary_passes:
        raise ValueError(
            f"Ordinary/trace verdict mismatch: {ordinary_passes} vs {passes}"
        )
    return {
        "trials": expected_count,
        "passes": passes,
        "fails": expected_count - passes,
        "input_sha256": sha256(input_path),
        "result_sha256": sha256(result_path),
        "trace_sha256": sha256(trace_path),
    }


def run_shard(
    *,
    config_path: Path,
    config: dict,
    hand_side: str,
    shard_index: int,
    max_grasps: int,
    resume: bool,
) -> dict:
    artifacts_root = project_path(config["artifacts_root"])
    input_path = (
        artifacts_root
        / hand_side
        / "isaac_inputs"
        / f"shard_{shard_index:03d}.yaml"
    )
    if not input_path.exists():
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tools/build_dex3_isaac_grasp_input.py"),
                "--atlas-config",
                str(config_path),
                "--hand-side",
                hand_side,
                "--shard-index",
                str(shard_index),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    output_directory = (
        f"physics_smoke_body_scalar_{max_grasps:03d}"
        if max_grasps
        else config["physics"].get(
            "qualification_output_directory", "physics_outputs"
        )
    )
    run_root = artifacts_root / hand_side / output_directory
    run_dir = run_root / f"shard_{shard_index:03d}"
    trace_path = run_root / f"shard_{shard_index:03d}.contact_trace.jsonl"
    result_path = expected_result_path(run_dir, hand_side)
    log_path = run_dir / "run.log"
    expected_count = max_grasps or int(config["generation"]["batch_size"])
    existing = [path for path in (result_path, trace_path) if path.exists()]
    if len(existing) == 2 and resume:
        return validate_completed_run(
            input_path=input_path,
            result_path=result_path,
            trace_path=trace_path,
            expected_count=expected_count,
            simulation_profile=config["physics"]["simulation_profile"],
        )
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an incomplete qualification shard: "
            + ", ".join(str(path) for path in existing)
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    physics = config["physics"]
    command = [
        str(GRASP_DATA_GEN / ".venv/bin/python"),
        "scripts/graspgen/grasp_sim.py",
        "--grasp_file",
        str(input_path),
        "--simulation_profile",
        physics["simulation_profile"],
        "--fps",
        str(physics["fps"]),
        "--max_num_envs",
        str(min(expected_count, int(physics["max_num_envs"]))),
        "--force_magnitude",
        str(physics["force_magnitude_g"]),
        "--initial_grasp_duration",
        str(physics["initial_grasp_duration_s"]),
        "--tug_sequences",
        json.dumps(physics["tug_sequences"]),
        "--contact_trace",
        str(trace_path),
        "--contact_trace_mode",
        physics.get("contact_trace_mode", "detailed"),
        "--contact_trace_max_points_per_pair",
        str(physics["detailed_point_budget_per_body_environment"]),
        "--headless",
        "--output_failed_grasp_locations",
    ]
    if max_grasps:
        command.extend(("--max_num_grasps", str(max_grasps)))
    environment = os.environ.copy()
    environment["OMNI_KIT_ACCEPT_EULA"] = "Y"
    environment["GRASP_DATASET_DIR"] = str(run_dir)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=GRASP_DATA_GEN,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Isaac qualification failed for {config['atlas_id']} shard "
            f"{shard_index}; inspect {log_path}"
        )
    if not result_path.is_file() or not trace_path.is_file():
        raise RuntimeError(
            f"Isaac returned without complete outputs for {config['atlas_id']} "
            f"shard {shard_index}; inspect {log_path}"
        )
    return validate_completed_run(
        input_path=input_path,
        result_path=result_path,
        trace_path=trace_path,
        expected_count=expected_count,
        simulation_profile=physics["simulation_profile"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append")
    parser.add_argument(
        "--hand-side",
        choices=("right", "left"),
        default="right",
        help="Physical Dex3 side to qualify with its matching Isaac asset.",
    )
    parser.add_argument("--shards", default="all")
    parser.add_argument(
        "--max-grasps",
        type=int,
        default=0,
        help="Smoke-test only: evaluate this many leading candidates per shard.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Validate and skip completed shards."
    )
    args = parser.parse_args()
    if args.max_grasps < 0:
        raise ValueError("--max-grasps must be nonnegative")
    config_paths = [path.resolve() for path in (args.config or DEFAULT_CONFIGS)]
    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text())
        seeds = config["generation"]["seeds"]
        shards = parse_shards(args.shards, len(seeds))
        summaries = {}
        for shard_index in shards:
            print(f"{config['atlas_id']}: qualifying shard {shard_index:03d}")
            summary = run_shard(
                config_path=config_path,
                config=config,
                hand_side=args.hand_side,
                shard_index=shard_index,
                max_grasps=args.max_grasps,
                resume=args.resume,
            )
            summaries[f"{shard_index:03d}"] = summary
            print(
                f"  {summary['passes']}/{summary['trials']} PASS; "
                f"{summary['fails']} FAIL"
            )
        manifest = {
            "schema_version": 1,
            "atlas_id": config["atlas_id"],
            "simulation_profile": config["physics"]["simulation_profile"],
            "contact_trace_mode": config["physics"].get(
                "contact_trace_mode", "detailed"
            ),
            "max_grasps_per_shard": args.max_grasps or None,
            "shards": summaries,
        }
        manifest_path = (
            project_path(config["artifacts_root"])
            / args.hand_side
            / (
                f"qualification_smoke_body_scalar_{args.max_grasps:03d}_manifest.json"
                if args.max_grasps
                else "qualification_run_body_scalar_v2_manifest.json"
            )
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
