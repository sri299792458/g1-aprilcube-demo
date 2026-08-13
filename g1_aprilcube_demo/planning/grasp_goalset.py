"""Immutable GraspGenX atlas transforms and cuRobo goal-set construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import trimesh.transformations as tra
import yaml


@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    family_id: str
    score: float
    object_T_G: np.ndarray
    representative_role: str | None = None


def load_grasp_pool(path: str | Path) -> tuple[GraspCandidate, ...]:
    document = yaml.safe_load(Path(path).read_text())
    candidates = []
    for entry in document["candidates"]:
        pose = entry["object_T_G"]
        orientation = pose["orientation"]
        matrix = tra.quaternion_matrix([orientation["w"], *orientation["xyz"]])
        matrix[:3, 3] = np.asarray(pose["position"], dtype=np.float64)
        matrix.setflags(write=False)
        candidates.append(
            GraspCandidate(
                candidate_id=str(entry["candidate_id"]),
                family_id=str(entry["family_id"]),
                score=float(entry["graspgenx_score"]),
                object_T_G=matrix,
                representative_role=(
                    None
                    if entry.get("representative_role") is None
                    else str(entry["representative_role"])
                ),
            )
        )
    return tuple(candidates)


@dataclass(frozen=True)
class CandidateExpansionRound:
    """One family-balanced progressive planning round."""

    round_id: str
    stage: str
    depth: int
    candidates: tuple[GraspCandidate, ...]


def progressive_candidate_rounds(
    candidates: Sequence[GraspCandidate],
) -> tuple[CandidateExpansionRound, ...]:
    """Expose primaries, diverse backups, then remaining family members.

    Every candidate appears exactly once.  The first three rounds are the
    explicit atlas representatives.  Later rounds take one remaining member
    per family in the pool's deterministic within-family order.
    """

    family_order: list[str] = []
    by_family: dict[str, list[GraspCandidate]] = {}
    for candidate in candidates:
        if candidate.family_id not in by_family:
            family_order.append(candidate.family_id)
            by_family[candidate.family_id] = []
        by_family[candidate.family_id].append(candidate)

    output: list[CandidateExpansionRound] = []
    seen: set[str] = set()
    roles = (
        ("primary", "primary"),
        ("translation_backup", "translation_diverse_backup"),
        ("pose_backup", "pose_diverse_backup"),
    )
    for round_id, role in roles:
        values = tuple(
            candidate
            for family in family_order
            for candidate in by_family[family]
            if candidate.representative_role == role
            and candidate.candidate_id not in seen
        )
        seen.update(candidate.candidate_id for candidate in values)
        if values:
            output.append(
                CandidateExpansionRound(round_id, "representative", 0, values)
            )

    remaining = {
        family: [
            candidate
            for candidate in by_family[family]
            if candidate.candidate_id not in seen
        ]
        for family in family_order
    }
    depth = 0
    while any(depth < len(values) for values in remaining.values()):
        values = tuple(
            remaining[family][depth]
            for family in family_order
            if depth < len(remaining[family])
        )
        output.append(
            CandidateExpansionRound(
                round_id=f"remaining_{depth:03d}",
                stage="remaining",
                depth=depth,
                candidates=values,
            )
        )
        seen.update(candidate.candidate_id for candidate in values)
        depth += 1

    expected = {candidate.candidate_id for candidate in candidates}
    if seen != expected:
        raise ValueError(
            "Progressive expansion lost candidates: "
            f"{sorted(expected - seen)[:5]}"
        )
    return tuple(output)


def world_grasps(
    world_T_object: np.ndarray, candidates: Iterable[GraspCandidate]
) -> list[np.ndarray]:
    return [world_T_object @ item.object_T_G for item in candidates]


def goal_tool_pose(
    frame_to_matrices: dict[str, Sequence[np.ndarray]],
    base_T_world: np.ndarray,
    *,
    device: str = "cuda",
):
    """Build one problem whose goal-set dimension is shared by all frames."""

    from curobo.types import GoalToolPose

    frames = list(frame_to_matrices)
    counts = {len(frame_to_matrices[frame]) for frame in frames}
    if len(counts) != 1:
        raise ValueError("Every tool frame must supply the same goal-set length")
    count = counts.pop()
    if count < 1:
        raise ValueError("A goal set cannot be empty")
    positions = torch.empty((1, 1, len(frames), count, 3), device=device)
    quaternions = torch.empty((1, 1, len(frames), count, 4), device=device)
    for link_index, frame in enumerate(frames):
        for goal_index, world_T_G in enumerate(frame_to_matrices[frame]):
            base_T_G = base_T_world @ world_T_G
            positions[0, 0, link_index, goal_index] = torch.as_tensor(
                base_T_G[:3, 3], device=device, dtype=torch.float32
            )
            quaternions[0, 0, link_index, goal_index] = torch.as_tensor(
                tra.quaternion_from_matrix(base_T_G), device=device, dtype=torch.float32
            )
    return GoalToolPose(frames, positions, quaternions)
