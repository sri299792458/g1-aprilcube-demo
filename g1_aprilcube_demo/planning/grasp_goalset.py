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
            )
        )
    return tuple(candidates)


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
