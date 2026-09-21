#!/usr/bin/env python3
"""Audit the existing GraspGenX atlas in one measured tabletop observation.

This command does not generate, alter, rank, or truncate grasps.  It records
the result of named support, endpoint-IK, and complete cuRobo pickup gates for
every existing candidate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_aprilcube_demo.planning.grasp_audit import (
    CandidateAudit,
    summarize,
    support_plane_audit,
)
from g1_aprilcube_demo.planning.grasp_goalset import GraspCandidate
from g1_aprilcube_demo.planning.runtime_assembly import RuntimeAssemblyPlanner


ASSIGNMENTS = {
    "t_body": "left",
    "u_legs": "right",
    "cube_head": "right",
}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/planning/t_u_cube_runtime_v2.yaml",
    )
    parser.add_argument(
        "--observation",
        default="config/observations/t_u_cube_nominal_v1.yaml",
    )
    parser.add_argument(
        "--task",
        default="config/tasks/t_u_cube_humanoid_v1.yaml",
    )
    parser.add_argument(
        "--output",
        default="artifacts/runtime_grasp_audit/nominal/audit.json",
    )
    parser.add_argument(
        "--support-only",
        action="store_true",
        help="Skip GPU IK and motion planning; useful for fast geometry checks.",
    )
    return parser.parse_args()


def matching_candidate(
    pool: tuple[GraspCandidate, ...], candidate_id: str
) -> GraspCandidate:
    return next(item for item in pool if item.candidate_id == candidate_id)


def run_curobo_gates(
    planner: RuntimeAssemblyPlanner,
    part: str,
    side: str,
    records: list[CandidateAudit],
) -> None:
    """Apply the exact runtime start state, scene, and upstream planner."""

    approach_offset = float(planner.cfg["motion"]["pick_approach_local_z_m"])
    contact_links = planner.contact_links[side]
    eligible = [item for item in records if item.support_plane_clear]
    if not eligible:
        return

    with planner.backend.stage(
        side,
        planner.initial_q,
        planner._scene(),
        {"left": None, "right": None},
        max_goalset=1,
        hand_closed={"left": 0.0, "right": 0.0},
    ) as stage:
        for index, record in enumerate(eligible, start=1):
            print(
                f"[audit] {part} {index}/{len(eligible)}: {record.candidate_id}",
                flush=True,
            )
            candidate = matching_candidate(
                planner.pools[part], record.candidate_id
            )
            motion = stage.plan_grasp(
                [
                    planner.observation.world_T_objects[part]
                    @ candidate.object_T_G
                ],
                approach_offset_m=approach_offset,
                contact_links=contact_links,
            )
            record.pickup_plan = motion is not None
            if motion is not None:
                # A successful native plan contains both subtrajectories, so
                # separate endpoint probes cannot veto it.
                record.final_endpoint_ik = True
                record.pregrasp_endpoint_ik = True
            else:
                # These are failure diagnostics only.  They deliberately run
                # after the authoritative complete pickup request.
                final = stage.check_endpoint(
                    np.asarray(record.world_T_G),
                    disable_links=contact_links,
                )
                pregrasp = stage.check_endpoint(
                    np.asarray(record.world_T_pregrasp_G)
                )
                record.final_endpoint_ik = final.success
                record.final_position_error_m = final.position_error_m
                record.final_orientation_error_rad = final.orientation_error_rad
                record.pregrasp_endpoint_ik = pregrasp.success
                record.pregrasp_position_error_m = pregrasp.position_error_m
                record.pregrasp_orientation_error_rad = pregrasp.orientation_error_rad
            record.finish()


def main() -> None:
    args = parse_args()
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    planner = RuntimeAssemblyPlanner(args.config, args.observation, args.task)
    try:
        parts: dict[str, dict] = {}
        approach_offset = float(planner.cfg["motion"]["pick_approach_local_z_m"])
        for part, side in ASSIGNMENTS.items():
            descriptor = project_path(
                f"third_party/GraspGenX/assets/x_grippers/"
                f"dex3_rev1_{side}/coll_mesh.obj"
            )
            hand = trimesh.load(descriptor, force="mesh", process=False)
            records = support_plane_audit(
                planner.pools[part],
                planner.observation.world_T_objects[part],
                np.asarray(hand.vertices),
                support_z_m=planner.observation.table.top_z,
                approach_offset_m=approach_offset,
            )
            if not args.support_only:
                run_curobo_gates(planner, part, side, records)
            parts[part] = {
                "assigned_hand_for_this_checkpoint": side,
                "atlas_path": str(
                    planner.task.parts[part].grasp_pools[side].relative_to(ROOT)
                ),
                "open_hand_collision_mesh": str(descriptor.relative_to(ROOT)),
                "summary": summarize(records),
                "candidates": [item.to_dict() for item in records],
            }
            print(f"[audit] {part}: {parts[part]['summary']}", flush=True)

        payload = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "observation": str(planner.observation_path.relative_to(ROOT)),
            "observation_id": planner.observation.observation_id,
            "planning_config": str(planner.config_path.relative_to(ROOT)),
            "task": str(planner.task_path.relative_to(ROOT)),
            "approach_contract": {
                "axis": "GraspGenX descriptor-frame local Z",
                "offset_m": approach_offset,
                "source": "motion.pick_approach_local_z_m",
            },
            "gate_contract": {
                "support_plane": (
                    "Exact open descriptor collision mesh at final grasp and "
                    "local-Z pregrasp must remain above the tabletop."
                ),
                "final_endpoint_ik": (
                    "Failure diagnostic only: upstream cuRobo IK in the full "
                    "scene, with configured terminal contact-link spheres "
                    "disabled. It never vetoes a successful plan_grasp."
                ),
                "pregrasp_endpoint_ik": (
                    "Failure diagnostic only: upstream cuRobo IK in the full "
                    "scene with all robot collision spheres enabled. It never "
                    "vetoes a successful plan_grasp."
                ),
                "pickup_plan": (
                    "Unchanged upstream MotionPlanner.plan_grasp from the "
                    "configured seated-ready arm state; authoritative after "
                    "the analytic support gate."
                ),
                "task_connector_keepouts": "not_evaluated_at_this_checkpoint",
                "mate_and_place_compatibility": "not_evaluated_at_this_checkpoint",
            },
            "candidate_policy": (
                "Existing diffusion candidates only; object_T_G is unchanged; "
                "no GraspMoE, manual grasp, pose tuning, ranking, or truncation."
            ),
            "parts": parts,
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(output)
    finally:
        planner.close()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
