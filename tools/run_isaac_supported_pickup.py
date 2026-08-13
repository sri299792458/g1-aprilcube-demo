#!/usr/bin/env python3
"""Run table-supported Isaac pickup qualification over support-atlas trials."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
GRASP_DATA_GEN = ROOT / "third_party/GraspDataGen"
DEFAULT_CONFIG = (
    ROOT
    / "config/grasp_support/u_legs_right_broad_face_isaac_v1.yaml"
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from build_dex3_isaac_grasp_input import (  # noqa: E402
    build_output,
    side_contract,
)
from g1_aprilcube_demo.grasping.support_atlas import (  # noqa: E402
    pose_document,
)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported supported-pickup configuration schema")
    if float(config["physics"]["env_spacing_m"]) <= 0.0:
        raise ValueError("physics.env_spacing_m must be positive")
    pass_contract = config["supported_pickup"]["pass_contract"]
    if int(pass_contract["min_final_object_contact_digit_groups"]) != int(
        config["physics"]["min_contact_groups"]
    ):
        raise ValueError(
            "Physics and supported-pickup digit-group requirements disagree"
        )
    if not bool(pass_contract["reject_any_hand_table_contact"]):
        raise ValueError("The supported-pickup contract must reject hand/table contact")
    if not bool(
        pass_contract["reject_object_table_contact_during_final_hold"]
    ):
        raise ValueError(
            "The supported-pickup contract must require the object off the table"
        )
    return config


def report_metadata(config: dict[str, Any]) -> dict[str, str]:
    configured = config.get("report", {})
    experiment_id = str(config["experiment_id"])
    return {
        "title": str(
            configured.get(
                "title",
                f"{experiment_id} supported-pickup qualification",
            )
        ),
        "support_scope": str(
            configured.get(
                "support_scope",
                f"the support selected by {experiment_id}",
            )
        ),
    }


def selected_trials(config: dict[str, Any]) -> list[dict[str, Any]]:
    document = json.loads(
        project_path(config["support_atlas"]).read_text()
    )
    requested_class = config["support_symmetry_class"]
    trials = []
    for support in document["supports"]:
        if support["support"]["symmetry_class"] != requested_class:
            continue
        support_pose = pose_document(
            np.asarray(
                support["support"]["support_T_object"],
                dtype=np.float64,
            )
        )
        for survivor in support["survivors"]:
            trials.append(
                {
                    "candidate_id": survivor["candidate_id"],
                    "candidate_content_sha256": survivor[
                        "candidate_content_sha256"
                    ],
                    "support_id": support["support"]["support_id"],
                    "support_label": support["support"]["label"],
                    "support_T_object": support_pose,
                    "support_T_G": survivor["support_T_G"],
                    "support_T_pregrasp_G": survivor[
                        "support_T_pregrasp_G"
                    ],
                    "proposal_bucket_id": survivor[
                        "proposal_bucket_id"
                    ],
                    "target_region": survivor["target_region"],
                    "approach_sector_object": survivor[
                        "approach_sector_object"
                    ],
                }
            )
    trials.sort(
        key=lambda trial: (trial["support_id"], trial["candidate_id"])
    )
    identities = [
        (trial["support_id"], trial["candidate_id"]) for trial in trials
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate support-conditioned trial identity")
    expected = sum(
        support["survivor_count"]
        for support in document["supports"]
        if support["support"]["symmetry_class"] == requested_class
    )
    if len(trials) != expected:
        raise ValueError(
            f"Selected {len(trials)} trials but atlas reports {expected}"
        )
    selection = config.get("selection", {})
    requested = selection.get("trials")
    source_report_path = selection.get("passed_source_report")
    if requested is not None and source_report_path is not None:
        raise ValueError(
            "selection.trials and selection.passed_source_report are exclusive"
        )
    if source_report_path is not None:
        source_path = project_path(source_report_path)
        expected_hash = str(
            selection["passed_source_report_sha256"]
        )
        actual_hash = sha256(source_path)
        if actual_hash != expected_hash:
            raise ValueError(
                "The configured source PASS report content has changed"
            )
        source_report = json.loads(source_path.read_text())
        if source_report.get("status") != "complete":
            raise ValueError("The source PASS report is incomplete")
        requested = [
            {
                "support_label": record["supported_pickup"][
                    "support_label"
                ],
                "candidate_id": record["candidate_id"],
            }
            for record in source_report["records"]
            if bool(record["result"]["passed"])
        ]
    if requested is not None:
        available = {
            (trial["support_label"], trial["candidate_id"]): trial
            for trial in trials
        }
        requested_keys = [
            (
                str(record["support_label"]),
                str(record["candidate_id"]),
            )
            for record in requested
        ]
        if len(requested_keys) != len(set(requested_keys)):
            raise ValueError("Duplicate configured supported-pickup trial")
        missing = [key for key in requested_keys if key not in available]
        if missing:
            raise ValueError(
                f"Configured supported-pickup trials are absent: {missing}"
            )
        trials = [available[key] for key in requested_keys]
    return trials


def raw_candidates(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    support_atlas = json.loads(
        project_path(config["support_atlas"]).read_text()
    )
    raw_paths = [
        project_path(record["path"])
        for record in support_atlas["source"]["raw_shards"]
    ]
    output: dict[str, dict[str, Any]] = {}
    for path in raw_paths:
        document = yaml.safe_load(path.read_text())
        for candidate_id, candidate in document["grasps"].items():
            if candidate_id in output:
                raise ValueError(f"Duplicate raw candidate {candidate_id}")
            output[candidate_id] = candidate
    return output


def build_chunk_input(
    *,
    config_path: Path,
    config: dict[str, Any],
    trials: list[dict[str, Any]],
    raw_by_id: dict[str, dict[str, Any]],
    chunk_index: int,
    output: Path,
) -> None:
    input_root = output.parent
    derived_raw = input_root / f"chunk_{chunk_index:03d}.raw.yaml"
    raw_document = {
        "format": "isaac_grasp",
        "format_version": "1.0",
        "created_with": "graspgenx",
        "grasps": {},
    }
    for trial in trials:
        candidate_id = trial["candidate_id"]
        if candidate_id not in raw_by_id:
            raise KeyError(candidate_id)
        candidate = copy.deepcopy(raw_by_id[candidate_id])
        generation = candidate.get("graspgenx_generation", {})
        if generation.get("candidate_content_sha256") != trial[
            "candidate_content_sha256"
        ]:
            raise ValueError(
                f"Candidate content identity changed: {candidate_id}"
            )
        raw_document["grasps"][candidate_id] = candidate
    atomic_write(derived_raw, yaml.safe_dump(raw_document, sort_keys=False))

    side = side_contract(ROOT, config["hand"]["side"])
    result = build_output(
        raw_path=derived_raw,
        hand_config=side["hand_config"],
        gripper_usd=side["gripper_usd"],
        object_mesh=project_path(config["object"]["mesh"]),
        object_mass=float(config["object"]["mass_kg"]),
        finger_colliders=side["finger_colliders"],
        finger_contact_groups=side["finger_contact_groups"],
        min_contact_groups=int(config["physics"]["min_contact_groups"]),
        approach_axis=2,
        open_limit="lower",
        atlas_metadata={
            "schema_version": 1,
            "atlas_id": config["experiment_id"],
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256(config_path),
            "object_id": config["object"]["id"],
            "hand_side": config["hand"]["side"],
            "shard_index": chunk_index,
            "simulation_profile": config["physics"]["simulation_profile"],
            "qualification_mode": "supported_pickup",
        },
        contact_trace_links=side["contact_trace_links"],
        contact_trace_link_aliases=side[
            "contact_trace_link_aliases"
        ],
    )
    result["supported_pickup"] = copy.deepcopy(
        config["supported_pickup"]
    )
    result["supported_pickup_experiment"] = {
        "experiment_id": config["experiment_id"],
        "support_atlas": str(
            project_path(config["support_atlas"]).relative_to(ROOT)
        ),
        "support_atlas_sha256": sha256(
            project_path(config["support_atlas"])
        ),
        "report": report_metadata(config),
        "support_symmetry_class": config["support_symmetry_class"],
        "chunk_index": chunk_index,
        "trial_count": len(trials),
    }
    by_id = {trial["candidate_id"]: trial for trial in trials}
    for candidate_id, grasp in result["grasps"].items():
        source_score = float(grasp["confidence"])
        if not np.isclose(
            source_score,
            float(grasp["graspgenx_source"]["confidence"]),
        ):
            raise ValueError("GraspGenX score changed during input conversion")
        # In Isaac Grasp files confidence is an admission/result field. Keep
        # the neural score under graspgenx_source and admit every selected
        # support-conditioned trial.
        grasp["confidence"] = 1.0
        grasp["supported_pickup"] = copy.deepcopy(by_id[candidate_id])
    atomic_write(output, yaml.safe_dump(result, sort_keys=False))


def expected_result(run_directory: Path, side: str) -> Path:
    return (
        run_directory
        / "grasp_sim_data"
        / f"dex3_rev1_{side}"
        / "grasp_mesh.yaml"
    )


def run_chunk(
    *,
    config: dict[str, Any],
    input_path: Path,
    run_directory: Path,
    trace_path: Path,
    video_path: Path | None,
    resume: bool,
) -> tuple[Path, Path, Path | None]:
    result_path = expected_result(
        run_directory, config["hand"]["side"]
    )
    required = [result_path, trace_path]
    if video_path is not None:
        required.append(video_path)
    if resume and all(path.is_file() for path in required):
        return result_path, trace_path, video_path
    existing = [path for path in required if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite incomplete supported-pickup outputs: "
            + ", ".join(str(path) for path in existing)
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    input_document = yaml.safe_load(input_path.read_text())
    trial_count = len(input_document["grasps"])
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
        str(trial_count),
        "--max_num_grasps",
        str(trial_count),
        "--env_spacing",
        str(physics["env_spacing_m"]),
        "--force_magnitude",
        "1.0",
        "--initial_grasp_duration",
        "1.0",
        "--tug_sequences",
        "[]",
        "--contact_trace",
        str(trace_path),
        "--contact_trace_mode",
        physics["contact_trace_mode"],
        "--contact_trace_max_points_per_pair",
        str(physics["contact_trace_max_points_per_pair"]),
        "--headless",
        "--output_failed_grasp_locations",
    ]
    if video_path is not None:
        command.extend(
            [
                "--enable_cameras",
                "--capture_video",
                str(video_path),
                "--capture_video_fps",
                str(config["capture"]["fps"]),
                "--capture_video_layout",
                config["capture"]["layout"],
            ]
        )
    environment = os.environ.copy()
    environment["OMNI_KIT_ACCEPT_EULA"] = "Y"
    environment["GRASP_DATASET_DIR"] = str(run_directory)
    log_path = run_directory / "run.log"
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
            f"Isaac supported pickup failed; inspect {log_path}"
        )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Isaac exited without required outputs {missing}; inspect {log_path}"
        )
    return result_path, trace_path, video_path


def validate_chunk(
    *,
    input_path: Path,
    result_path: Path,
    trace_path: Path,
) -> list[dict[str, Any]]:
    source = yaml.safe_load(input_path.read_text())
    result = yaml.safe_load(result_path.read_text())
    source_ids = list(source["grasps"])
    result_ids = list(result["grasps"])
    if result_ids != source_ids:
        raise ValueError("Isaac result identity/order differs from input")
    records = [
        json.loads(line) for line in trace_path.read_text().splitlines()
    ]
    if [record["candidate_id"] for record in records] != source_ids:
        raise ValueError("Isaac trace identity/order differs from input")
    expected_phases = [
        "settled_on_support",
        "approach_complete",
        "closed_before_lift",
        "lift_complete",
        "final_hold",
    ]
    for record in records:
        if record["physics"]["mode"] != "supported_pickup":
            raise ValueError("Trace lacks the supported-pickup mode")
        if [phase["name"] for phase in record["phases"]] != expected_phases:
            raise ValueError("Trace has incorrect supported-pickup phases")
        for phase in record["phases"]:
            if "world_T_object" not in phase or "world_T_G" not in phase:
                raise ValueError("Trace lacks world-frame phase poses")
        approach_G = np.asarray(
            record["phases"][1]["world_T_G"], dtype=np.float64
        )
        lifted_G = np.asarray(
            record["phases"][3]["world_T_G"], dtype=np.float64
        )
        commanded_lift = float(
            record["physics"]["supported_pickup_contract"][
                "lift_distance_m"
            ]
        )
        actual_lift = lifted_G[:3, 3] - approach_G[:3, 3]
        if not np.allclose(
            actual_lift,
            [0.0, 0.0, commanded_lift],
            atol=1.0e-4,
            rtol=0.0,
        ):
            raise ValueError(
                "Supported-pickup hand did not execute the commanded lift"
            )
        if (
            float(
                record["result"][
                    "final_gripper_command_position_error_m"
                ]
            )
            > 1.0e-4
        ):
            raise ValueError(
                "Supported-pickup hand did not hold its final position"
            )
        if (
            float(
                record["result"][
                    "final_gripper_command_orientation_error_rad"
                ]
            )
            > 1.0e-3
        ):
            raise ValueError(
                "Supported-pickup hand did not hold its final orientation"
            )
        verdict = bool(record["result"]["passed"])
        recomputed = (
            bool(record["result"]["digit_contact_pass"])
            and not bool(record["result"]["hand_table_contact_any"])
            and not bool(
                record["result"][
                    "object_table_contact_during_final_hold"
                ]
            )
        )
        if verdict != recomputed:
            raise ValueError("Supported-pickup verdict contract mismatch")
        ordinary = bool(
            result["grasps"][record["candidate_id"]]["confidence"]
        )
        if ordinary != verdict:
            raise ValueError("Ordinary Isaac result disagrees with trace")
    return records


def concatenate_videos(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".txt", delete=False
    ) as stream:
        for path in paths:
            escaped = str(path.resolve()).replace("'", "'\\''")
            stream.write(f"file '{escaped}'\n")
        manifest = Path(stream.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest),
                "-c",
                "copy",
                str(output),
            ],
            check=True,
        )
    finally:
        manifest.unlink(missing_ok=True)


def make_report(
    *,
    config_path: Path,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    video: Path | None,
) -> dict[str, Any]:
    by_support: dict[str, Counter[str]] = {}
    by_component: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    for record in records:
        support = record["supported_pickup"]["support_label"]
        support_counts = by_support.setdefault(support, Counter())
        passed = bool(record["result"]["passed"])
        support_counts["trials"] += 1
        support_counts["passes" if passed else "fails"] += 1
        component = record["supported_pickup"]["target_region"].get(
            "component", "unresolved"
        )
        by_component[f"{component}:{'pass' if passed else 'fail'}"] += 1
        reasons = []
        if not bool(record["result"]["digit_contact_pass"]):
            reasons.append("insufficient_final_digit_contacts")
        if bool(record["result"]["hand_table_contact_any"]):
            reasons.append("hand_table_contact")
        if bool(
            record["result"]["object_table_contact_during_final_hold"]
        ):
            reasons.append("object_on_table_during_final_hold")
        if not reasons:
            reasons.append("pass")
        failures["+".join(reasons)] += 1
    passes = sum(bool(record["result"]["passed"]) for record in records)
    digit_contact_passes = sum(
        bool(record["result"]["digit_contact_pass"]) for record in records
    )
    hand_table_contacts = sum(
        bool(record["result"]["hand_table_contact_any"])
        for record in records
    )
    object_table_contacts = sum(
        bool(
            record["result"][
                "object_table_contact_during_final_hold"
            ]
        )
        for record in records
    )
    final_table_forces = [
        float(
            record["result"][
                "max_final_hold_object_table_contact_force_N"
            ]
        )
        for record in records
    ]
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "complete",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "support_atlas": config["support_atlas"],
        "support_atlas_sha256": sha256(
            project_path(config["support_atlas"])
        ),
        "report": report_metadata(config),
        "trial_count": len(records),
        "pass_count": passes,
        "fail_count": len(records) - passes,
        "digit_contact_pass_count": digit_contact_passes,
        "hand_table_contact_count": hand_table_contacts,
        "object_table_final_hold_contact_count": object_table_contacts,
        "final_object_table_contact_force_range_N": (
            {
                "min": min(final_table_forces),
                "max": max(final_table_forces),
            }
            if final_table_forces
            else None
        ),
        "by_support": {
            support: dict(counts)
            for support, counts in sorted(by_support.items())
        },
        "by_target_component_and_verdict": dict(sorted(by_component.items())),
        "verdict_combinations": dict(sorted(failures.items())),
        "video": (
            str(video.relative_to(ROOT)) if video is not None else None
        ),
        "records": records,
    }


def markdown_report(report: dict[str, Any]) -> str:
    report_config = report["report"]
    lines = [
        f"# {report_config['title']}",
        "",
        "This is the table-supported Isaac/PhysX result, not a render or",
        "collision-only prediction.",
        "",
        f"- Trials: **{report['trial_count']}**",
        f"- PASS: **{report['pass_count']}**",
        f"- FAIL: **{report['fail_count']}**",
        f"- Retained by at least two digit chains: "
        f"**{report['digit_contact_pass_count']}**",
        f"- Hand/table collision: **{report['hand_table_contact_count']}**",
        f"- U/table contact in final hold: "
        f"**{report['object_table_final_hold_contact_count']}**",
        "",
    ]
    if report["video"]:
        lines.extend(
            [
                f"Video: [`{report['video']}`](assets/{Path(report['video']).name})",
                "",
            ]
        )
    lines.extend(
        [
            "## Results by selected support",
            "",
            "| Support | Trials | PASS | FAIL |",
            "|---|---:|---:|---:|",
        ]
    )
    for support, counts in report["by_support"].items():
        lines.append(
            f"| {support} | {counts.get('trials', 0)} | "
            f"{counts.get('passes', 0)} | {counts.get('fails', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Physical pass contract",
            "",
            "A trial passes only when all three statements are true:",
            "",
            "1. At least two Dex3 digit chains contact the U after the 20 cm lift",
            "   and one-second final hold.",
            "2. No Dex3 hand link contacted the table during settle, approach,",
            "   closure, lift, or hold.",
            "3. The U did not contact the table during the final elevated hold.",
            "",
            "## Verdict combinations",
            "",
            "| Verdict/reasons | Trials |",
            "|---|---:|",
        ]
    )
    for reason, count in report["verdict_combinations"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "Every concrete trial and its five named phase snapshots remain in",
            "the machine-readable report JSON.",
            "",
            "## Conclusion",
            "",
        ]
    )
    if report["pass_count"]:
        lines.extend(
            [
                f"{report['pass_count']} of {report['trial_count']} "
                "geometry-clear proposals passed the complete physical",
                f"contract for {report_config['support_scope']}. These",
                "records form the support-conditioned physics library; they",
                "are eligible for family construction and later cuRobo",
                "reachability checks, but are not automatically executable.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"None of the {report['trial_count']} geometry-clear proposals",
                "passed the complete physical contract for",
                f"{report_config['support_scope']}. The selected runtime",
                "library is empty for this proposal set, hand, and controller",
                "profile.",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="Run only the first N trials under a separate smoke output root.",
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    trials = selected_trials(config)
    if args.smoke:
        if args.smoke <= 0 or args.smoke > len(trials):
            raise ValueError("--smoke is outside the trial count")
        trials = trials[: args.smoke]
    raw_by_id = raw_candidates(config)
    base_root = project_path(config["outputs"]["artifacts_root"])
    artifacts_root = (
        base_root / f"smoke_{args.smoke:03d}"
        if args.smoke
        else base_root / "full"
    )
    inputs_root = artifacts_root / "inputs"
    runs_root = artifacts_root / "runs"
    chunk_size = int(config["physics"]["max_num_envs"])
    chunks = [
        trials[index : index + chunk_size]
        for index in range(0, len(trials), chunk_size)
    ]
    input_paths = []
    for chunk_index, chunk in enumerate(chunks):
        input_path = inputs_root / f"chunk_{chunk_index:03d}.yaml"
        if input_path.exists():
            existing = yaml.safe_load(input_path.read_text())
            existing_ids = list(existing["grasps"])
            requested_ids = [trial["candidate_id"] for trial in chunk]
            if existing_ids != requested_ids:
                raise RuntimeError(
                    f"Existing input differs from requested trials: {input_path}"
                )
            expected_config_hash = sha256(config_path)
            actual_config_hash = existing.get("grasp_atlas", {}).get(
                "config_sha256"
            )
            if actual_config_hash != expected_config_hash:
                raise RuntimeError(
                    "Existing input was built from a different configuration: "
                    f"{input_path}"
                )
            if existing.get("supported_pickup") != config["supported_pickup"]:
                raise RuntimeError(
                    "Existing input has a different supported-pickup contract: "
                    f"{input_path}"
                )
        else:
            build_chunk_input(
                config_path=config_path,
                config=config,
                trials=chunk,
                raw_by_id=raw_by_id,
                chunk_index=chunk_index,
                output=input_path,
            )
        input_paths.append(input_path)
    print(
        f"[supported-pickup] built {len(trials)} trials in "
        f"{len(chunks)} chunks",
        flush=True,
    )
    if args.build_only:
        return

    all_records = []
    video_chunks = []
    for chunk_index, input_path in enumerate(input_paths):
        run_directory = runs_root / f"chunk_{chunk_index:03d}"
        trace_path = runs_root / f"chunk_{chunk_index:03d}.trace.jsonl"
        video_path = (
            runs_root / f"chunk_{chunk_index:03d}.mp4"
            if bool(config["capture"]["enabled"]) and not args.no_video
            else None
        )
        print(
            f"[supported-pickup] Isaac chunk {chunk_index + 1}/"
            f"{len(input_paths)}",
            flush=True,
        )
        result_path, trace_path, video_path = run_chunk(
            config=config,
            input_path=input_path,
            run_directory=run_directory,
            trace_path=trace_path,
            video_path=video_path,
            resume=args.resume,
        )
        all_records.extend(
            validate_chunk(
                input_path=input_path,
                result_path=result_path,
                trace_path=trace_path,
            )
        )
        if video_path is not None:
            video_chunks.append(video_path)

    final_video = None
    if video_chunks and not args.smoke:
        final_video = project_path(config["outputs"]["video"])
        if final_video.exists() and not args.resume:
            raise FileExistsError(final_video)
        if not final_video.exists():
            concatenate_videos(video_chunks, final_video)

    report = make_report(
        config_path=config_path,
        config=config,
        records=all_records,
        video=final_video,
    )
    report_json = (
        artifacts_root / "report.json"
        if args.smoke
        else project_path(config["outputs"]["report_json"])
    )
    report_markdown = (
        artifacts_root / "report.md"
        if args.smoke
        else project_path(config["outputs"]["report_markdown"])
    )
    atomic_write(report_json, json.dumps(report, indent=2) + "\n")
    atomic_write(report_markdown, markdown_report(report))
    print(
        f"[supported-pickup] {report['pass_count']} PASS / "
        f"{report['fail_count']} FAIL",
        flush=True,
    )
    print(f"[supported-pickup] report: {report_json}", flush=True)
    if final_video is not None:
        print(f"[supported-pickup] video: {final_video}", flush=True)


if __name__ == "__main__":
    main()
