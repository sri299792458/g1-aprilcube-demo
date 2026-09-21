#!/usr/bin/env python3
"""Build an arm-planning pool from a supported-pickup PASS ledger.

The output copies each candidate's original object_T_G from the Isaac input
record. The simulated final hand/object pose is never used as a motion goal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import trimesh.transformations as tra
import yaml


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pose_document(matrix: np.ndarray) -> dict[str, Any]:
    quaternion = tra.quaternion_from_matrix(matrix)
    return {
        "position": matrix[:3, 3].tolist(),
        "orientation": {
            "w": float(quaternion[0]),
            "xyz": quaternion[1:].tolist(),
        },
    }


def family_balanced_order(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        families[record["supported_pickup"]["proposal_bucket_id"]].append(record)
    for values in families.values():
        values.sort(
            key=lambda item: (
                -float(item["input"]["graspgenx_score"]),
                item["candidate_id"],
            )
        )
    ordered = []
    depth = 0
    while True:
        appended = False
        for family_id in sorted(families):
            values = families[family_id]
            if depth < len(values):
                ordered.append(values[depth])
                appended = True
        if not appended:
            return ordered
        depth += 1


def build(
    report_path: Path,
    mesh_path: Path,
    atlas_id: str,
    hand_side: str,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    passing = [
        record for record in report["records"] if bool(record["result"]["passed"])
    ]
    if len(passing) != int(report["pass_count"]):
        raise ValueError("Report pass_count disagrees with its concrete records")
    if not passing:
        raise ValueError("Cannot build an arm pool from an empty PASS ledger")
    if {record["hand_side"] for record in passing} != {hand_side}:
        raise ValueError("PASS records do not match the requested physical hand")
    object_ids = {record["object_id"] for record in passing}
    if len(object_ids) != 1:
        raise ValueError(f"PASS records contain multiple objects: {object_ids}")

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    candidates = []
    for record in family_balanced_order(passing):
        candidate_id = record["candidate_id"]
        content_hash = record["candidate_content_sha256"]
        if candidate_id in seen_ids or content_hash in seen_hashes:
            raise ValueError(f"Duplicate supported candidate: {candidate_id}")
        seen_ids.add(candidate_id)
        seen_hashes.add(content_hash)
        matrix = np.asarray(record["input"]["object_T_G"], dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"Invalid object_T_G for {candidate_id}")
        family_id = record["supported_pickup"]["proposal_bucket_id"]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_content_sha256": content_hash,
                "family_id": family_id,
                "representative_role": (
                    "primary"
                    if not any(
                        item["family_id"] == family_id for item in candidates
                    )
                    else None
                ),
                "graspgenx_score": float(record["input"]["graspgenx_score"]),
                "object_T_G": pose_document(matrix),
            }
        )

    family_count = len({item["family_id"] for item in candidates})
    return {
        "format": "g1_aprilcube_arm_grasp_pool",
        "format_version": 1,
        "atlas_id": atlas_id,
        "hand_side": hand_side,
        "object_id": next(iter(object_ids)),
        "object_mesh": str(mesh_path.relative_to(ROOT)),
        "object_mesh_sha256": sha256(mesh_path),
        "candidate_count": len(candidates),
        "family_count": family_count,
        "ordering_policy": (
            "supported_pickup_proposal_bucket_round_robin_by_unchanged_"
            "graspgenx_score"
        ),
        "source": {
            "supported_pickup_report": str(report_path.relative_to(ROOT)),
            "supported_pickup_report_sha256": sha256(report_path),
            "support_atlas": report["support_atlas"],
            "support_atlas_sha256": report["support_atlas_sha256"],
            "pose_policy": "original_object_T_G_copied_from_supported_pickup_input",
            "admission_policy": (
                "supported_pickup_result_passed_after_approach_close_lift_hold"
            ),
        },
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--atlas-id", required=True)
    parser.add_argument("--hand-side", choices=("left", "right"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = project_path(args.report)
    mesh = project_path(args.mesh)
    output = project_path(args.output)
    document = build(report, mesh, args.atlas_id, args.hand_side)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(document, sort_keys=False))
    print(
        f"Wrote {output}: {document['candidate_count']} support-qualified "
        f"candidates in {document['family_count']} proposal buckets"
    )


if __name__ == "__main__":
    main()
