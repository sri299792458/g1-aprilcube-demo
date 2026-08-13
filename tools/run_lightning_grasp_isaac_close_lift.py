#!/usr/bin/env python3
"""Physically qualify table-clear Lightning-Grasp U candidates in Isaac.

Lightning Grasp outputs a final hand/object transform and a target articulated
configuration.  It does not output a pregrasp or approach trajectory.  This
experiment therefore performs the narrow claim we can test without inventing
one: keep the hand base at the generated final pose, command the exact
Lightning-Grasp q from the standard open q, lift 20 cm, and hold under gravity.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from build_dex3_isaac_grasp_input import (  # noqa: E402
    sha256,
    sha256_usd_package,
    side_contract,
)
from g1_aprilcube_demo.grasping.support_atlas import (  # noqa: E402
    configured_support_conditions,
    pose_document,
)
from run_isaac_supported_pickup import (  # noqa: E402
    atomic_write,
    run_chunk,
    validate_chunk,
)


DEFAULT_CONFIG = (
    ROOT
    / "config/grasp_support/u_legs_right_lightning_grasp_close_lift_isaac_v1.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete Isaac outputs and regenerate reports",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def pose_fields(matrix: np.ndarray) -> dict[str, Any]:
    document = pose_document(np.asarray(matrix, dtype=np.float64))
    return {
        "position": copy.deepcopy(document["position"]),
        "orientation": copy.deepcopy(document["orientation"]),
    }


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported Lightning-Grasp Isaac config schema")
    if config["source"]["selection"] != "every final_geometry_eligible pair":
        raise ValueError("This runner only supports the audited eligible selection")
    return config


def build_input(
    *,
    config_path: Path,
    config: dict[str, Any],
    output: Path,
) -> list[str]:
    npz_path = project_path(config["source"]["result_npz"])
    audit_path = project_path(config["source"]["broad_face_audit"])
    object_path = project_path(config["object"]["mesh"])
    data = np.load(npz_path)
    q = np.asarray(data["q"], dtype=np.float64)
    g_t_object = np.asarray(data["object_pose"], dtype=np.float64)
    joint_names = [str(value) for value in data["active_joint_names"].tolist()]

    audit = json.loads(audit_path.read_text())
    eligible = [
        trial
        for trial in audit["trials"]
        if bool(trial["final_geometry_eligible"])
    ]
    if len(eligible) != int(
        audit["summary"]["final_geometry_eligible_pair_count"]
    ):
        raise ValueError("Audit eligible selection count changed")

    baseline_report_path = config["source"].get("baseline_report")
    if baseline_report_path is not None:
        baseline_report = json.loads(project_path(baseline_report_path).read_text())
        admissible = {
            (
                record["supported_pickup"]["support_id"],
                int(record["supported_pickup"]["target_region"]["candidate_index"]),
            )
            for record in baseline_report["records"]
            if not bool(record["result"]["hand_table_contact_any"])
        }
        eligible = [
            trial
            for trial in eligible
            if (trial["support_id"], int(trial["candidate_index"])) in admissible
        ]
        if not eligible:
            raise ValueError("Baseline table-contact selection removed every candidate")
    identities = [
        (trial["support_id"], int(trial["candidate_index"]))
        for trial in eligible
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate Lightning-Grasp support/candidate pair")

    support_config = yaml.safe_load(
        (ROOT / "config/grasp_support/u_legs_right_v1.yaml").read_text()
    )
    object_mesh = trimesh.load(object_path, force="mesh")
    supports = configured_support_conditions(
        object_mesh, entries=support_config["supports"]["orientations"]
    )
    supports_by_id = {support.support_id: support for support in supports}

    side = side_contract(ROOT, config["hand"]["side"])
    hand_config = json.loads(side["hand_config"].read_text())
    open_q = hand_config["open"]
    if list(open_q) != joint_names:
        raise ValueError(
            "Lightning-Grasp q order and the Isaac hand joint contract differ"
        )
    urdf_path = (
        ROOT
        / "third_party/GraspGenX/assets/x_grippers/dex3_rev1_right/gripper.urdf"
    )
    urdf_root = ET.parse(urdf_path).getroot()
    limits = {}
    for joint in urdf_root.findall("joint"):
        limit = joint.find("limit")
        if limit is None or joint.get("name") not in joint_names:
            continue
        limits[str(joint.get("name"))] = (
            float(limit.get("lower")),
            float(limit.get("upper")),
        )
    if set(limits) != set(joint_names):
        raise ValueError("Could not recover every Dex3 joint limit")
    overclosure_scales = [
        float(value)
        for value in config.get("execution", {}).get(
            "overclosure_scales", [0.0]
        )
    ]
    if not overclosure_scales or any(value < 0.0 for value in overclosure_scales):
        raise ValueError("Overclosure scales must be a nonempty nonnegative list")

    grasps = {}
    for trial in eligible:
        candidate_index = int(trial["candidate_index"])
        support = supports_by_id[trial["support_id"]]
        object_t_g = np.linalg.inv(g_t_object[candidate_index])
        support_t_g = np.asarray(trial["support_T_G"], dtype=np.float64)
        if not np.allclose(
            support_t_g,
            support.support_T_object @ object_t_g,
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise ValueError(
                f"Support transform mismatch for candidate {candidate_index}"
            )
        final_pose = pose_document(support_t_g)
        object_pose = pose_document(support.support_T_object)
        contact_q = {
            joint_name: float(q[candidate_index, joint_index])
            for joint_index, joint_name in enumerate(joint_names)
        }
        for overclosure_scale in overclosure_scales:
            candidate_id = (
                f"lightning_u_{support.table_up_sector.replace('+', 'pos').replace('-', 'neg')}"
                f"_candidate_{candidate_index:04d}"
            )
            if len(overclosure_scales) > 1 or overclosure_scale != 0.0:
                candidate_id += f"_overclose_{int(round(1000 * overclosure_scale)):04d}"
            command_q = {}
            for joint_name in joint_names:
                value = float(open_q[joint_name]) + (1.0 + overclosure_scale) * (
                    contact_q[joint_name] - float(open_q[joint_name])
                )
                lower, upper = limits[joint_name]
                command_q[joint_name] = float(np.clip(value, lower, upper))
            grasps[candidate_id] = {
                "confidence": 1.0,
                **pose_fields(object_t_g),
                "lightning_grasp_source": {
                    "candidate_index": candidate_index,
                    "result_npz": str(npz_path.relative_to(ROOT)),
                    "result_npz_sha256": sha256(npz_path),
                    "G_T_object": g_t_object[candidate_index].tolist(),
                    "object_T_G": object_t_g.tolist(),
                    "contact_q": contact_q,
                    "command_q": command_q,
                    "overclosure_scale": overclosure_scale,
                    "overclosure_contract": (
                        "clip(q_open + (1 + scale) * "
                        "(q_contact - q_open), joint_limits)"
                    ),
                    "final_table_clearance_m": float(
                        trial["table_clearance_m"]
                    ),
                    "max_object_penetration_m": float(
                        trial["max_object_penetration_m"]
                    ),
                },
                "pregrasp_cspace_position": copy.deepcopy(open_q),
                "cspace_position": command_q,
                "supported_pickup": {
                    "candidate_id": candidate_id,
                    "support_id": support.support_id,
                    "support_label": support.label,
                    "support_T_object": object_pose,
                    # Zero base motion is intentional: Lightning Grasp provides
                    # no pregrasp or approach, so only closure/retention is tested.
                    "support_T_pregrasp_G": copy.deepcopy(final_pose),
                    "support_T_G": final_pose,
                    "proposal_bucket_id": "lightning_grasp_final_geometry",
                    "target_region": {
                        "generator": "lightning_grasp",
                        "candidate_index": candidate_index,
                        "qualification_scope": (
                            "overclosure_sweep_close_in_place_then_lift"
                            if overclosure_scale > 0.0
                            else "close_in_place_then_lift"
                        ),
                        "overclosure_scale": overclosure_scale,
                    },
                    "approach_sector_object": "none_close_in_place",
                },
            }

    input_document = {
        "format": "isaac_grasp",
        "format_version": "1.0",
        "created_with": "lightning_grasp_adapter",
        "object_file": str(object_path),
        "object_scale": 1.0,
        "obj2usd_use_existing_usd": False,
        "obj2usd_mass": float(config["object"]["mass_kg"]),
        "gripper_file": str(side["gripper_usd"]),
        "finger_colliders": side["finger_colliders"],
        "open_limit": "lower",
        "use_cspace_position_as_target": True,
        "approach_axis": 2,
        "source": {
            "generator": "lightning_grasp",
            "lightning_repository_commit": "af43818e864b0389c97b73429e5e60de2a2de593",
            "result_npz": str(npz_path),
            "result_npz_sha256": sha256(npz_path),
            "broad_face_audit": str(audit_path),
            "broad_face_audit_sha256": sha256(audit_path),
            "hand_config": str(side["hand_config"]),
            "hand_config_sha256": sha256(side["hand_config"]),
            "gripper_usd": str(side["gripper_usd"]),
            "gripper_usd_sha256": sha256(side["gripper_usd"]),
            "gripper_usd_package_sha256": sha256_usd_package(
                side["gripper_usd"]
            ),
            "object_mesh": str(object_path),
            "object_mesh_sha256": sha256(object_path),
            "transform_policy": (
                "object_T_G=inverse(Lightning G_T_object); "
                "target q copied exactly; zero-motion base pregrasp"
            ),
        },
        "finger_contact_groups": side["finger_contact_groups"],
        "min_contact_groups": int(config["physics"]["min_contact_groups"]),
        "contact_trace_links": side["contact_trace_links"],
        "contact_trace_link_aliases": side[
            "contact_trace_link_aliases"
        ],
        "supported_pickup": copy.deepcopy(config["supported_pickup"]),
        "supported_pickup_experiment": {
            "experiment_id": config["experiment_id"],
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256(config_path),
            "qualification_scope": (
                "overclosure_sweep_close_in_place_then_lift"
                if any(value > 0.0 for value in overclosure_scales)
                else "close_in_place_then_lift"
            ),
            "overclosure_scales": overclosure_scales,
            "trial_count": len(grasps),
        },
        "grasps": grasps,
    }
    atomic_write(output, yaml.safe_dump(input_document, sort_keys=False))
    return list(grasps)


def markdown_report(
    *,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    video: Path,
) -> str:
    overclosure_scales = config.get("execution", {}).get(
        "overclosure_scales", [0.0]
    )
    passes = [record for record in records if record["result"]["passed"]]
    digit_passes = [
        record for record in records if record["result"]["digit_contact_pass"]
    ]
    table_contacts = [
        record
        for record in records
        if record["result"]["hand_table_contact_any"]
    ]
    lines = [
        "# Lightning-Grasp Dex3 broad-face U Isaac close/lift test",
        "",
        "This experiment tests the exact final pose and seven-joint configuration",
        "returned by Lightning Grasp. The hand base does not execute an invented",
        "pregrasp approach: it remains at the final pose, closes from the standard",
        "Dex3 open configuration toward Lightning's `q`, lifts 20 cm, and holds under",
        "gravity.",
        "",
        f"- Trials: **{len(records)}**",
        f"- Full PASS: **{len(passes)}**",
        f"- Retained with at least two digit groups: **{len(digit_passes)}**",
        f"- Any hand/table contact: **{len(table_contacts)}**",
        f"- Overclosure scales: **{overclosure_scales}**",
        f"- Review video: [`{video.name}`](assets/{video.name})",
        "",
        "A PASS establishes table-supported close/lift retention for this final",
        "grasp. It does not establish an arm-reachable collision-free pregrasp or",
        "approach trajectory, because Lightning Grasp does not return one.",
        "",
        "| Candidate | Support | PASS | Two digit groups | Hand/table | Object/table at hold |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for record in records:
        result = record["result"]
        metadata = record["supported_pickup"]
        lines.append(
            f"| `{record['candidate_id']}` | {metadata['support_label']} | "
            f"{'yes' if result['passed'] else 'no'} | "
            f"{'yes' if result['digit_contact_pass'] else 'no'} | "
            f"{'yes' if result['hand_table_contact_any'] else 'no'} | "
            f"{'yes' if result['object_table_contact_during_final_hold'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    output_root = project_path(config["outputs"]["artifacts_root"])
    input_path = output_root / "input.yaml"
    run_directory = output_root / "run"
    trace_path = output_root / "trace.jsonl"
    video_path = project_path(config["outputs"]["video"])
    report_path = project_path(config["outputs"]["report_json"])
    markdown_path = project_path(config["outputs"]["report_markdown"])

    candidate_ids = build_input(
        config_path=config_path,
        config=config,
        output=input_path,
    )
    result_path, trace_path, _ = run_chunk(
        config=config,
        input_path=input_path,
        run_directory=run_directory,
        trace_path=trace_path,
        video_path=video_path,
        resume=args.resume,
    )
    records = validate_chunk(
        input_path=input_path,
        result_path=result_path,
        trace_path=trace_path,
    )
    if [record["candidate_id"] for record in records] != candidate_ids:
        raise ValueError("Isaac returned candidates in a different order")
    qualification_scope = (
        "overclosure_sweep_close_in_place_then_lift"
        if any(
            float(value) > 0.0
            for value in config.get("execution", {}).get(
                "overclosure_scales", [0.0]
            )
        )
        else "close_in_place_then_lift"
    )
    by_overclosure_scale: dict[str, dict[str, int]] = {}
    for record in records:
        scale = float(
            record["supported_pickup"]["target_region"].get(
                "overclosure_scale", 0.0
            )
        )
        counts = by_overclosure_scale.setdefault(
            f"{scale:.6f}",
            {"trials": 0, "passes": 0, "digit_contact_passes": 0, "hand_table_contacts": 0},
        )
        counts["trials"] += 1
        counts["passes"] += int(bool(record["result"]["passed"]))
        counts["digit_contact_passes"] += int(
            bool(record["result"]["digit_contact_pass"])
        )
        counts["hand_table_contacts"] += int(
            bool(record["result"]["hand_table_contact_any"])
        )
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "complete",
        "qualification_scope": qualification_scope,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "input": str(input_path.relative_to(ROOT)),
        "input_sha256": sha256(input_path),
        "trial_count": len(records),
        "pass_count": sum(bool(record["result"]["passed"]) for record in records),
        "digit_contact_pass_count": sum(
            bool(record["result"]["digit_contact_pass"]) for record in records
        ),
        "hand_table_contact_count": sum(
            bool(record["result"]["hand_table_contact_any"]) for record in records
        ),
        "object_table_final_hold_contact_count": sum(
            bool(
                record["result"]["object_table_contact_during_final_hold"]
            )
            for record in records
        ),
        "by_overclosure_scale": by_overclosure_scale,
        "video": str(video_path.relative_to(ROOT)),
        "records": records,
    }
    atomic_write(report_path, json.dumps(report, indent=2) + "\n")
    atomic_write(
        markdown_path,
        markdown_report(config=config, records=records, video=video_path),
    )
    print(json.dumps({key: report[key] for key in (
        "trial_count",
        "pass_count",
        "digit_contact_pass_count",
        "hand_table_contact_count",
        "object_table_final_hold_contact_count",
    )}, indent=2))
    print(f"Saved: {report_path}")
    print(f"Saved: {markdown_path}")
    print(f"Saved: {video_path}")


if __name__ == "__main__":
    main()
