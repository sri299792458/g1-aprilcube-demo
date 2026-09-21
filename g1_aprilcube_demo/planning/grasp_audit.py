"""Scene-conditioned admission records for immutable GraspGenX candidates.

The physics atlas answers an intrinsic question: can the isolated hand retain
the object?  This module adds the first scene question without changing the
candidate transform: does the exact open descriptor hand clear the support
plane at both the grasp and its named local-Z pregrasp?

Robot IK and motion planning remain upstream cuRobo responsibilities.  Their
results are attached to these records by ``tools/audit_runtime_grasps.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import trimesh
import trimesh.transformations as tra

from .grasp_goalset import GraspCandidate


@dataclass
class CandidateAudit:
    """One immutable atlas candidate's ordered admission-gate results."""

    candidate_id: str
    family_id: str
    graspgenx_score: float
    world_T_G: list[list[float]]
    world_T_pregrasp_G: list[list[float]]
    final_support_clearance_m: float
    pregrasp_support_clearance_m: float
    support_plane_clear: bool
    final_endpoint_ik: bool | None = None
    final_position_error_m: float | None = None
    final_orientation_error_rad: float | None = None
    pregrasp_endpoint_ik: bool | None = None
    pregrasp_position_error_m: float | None = None
    pregrasp_orientation_error_rad: float | None = None
    pickup_plan: bool | None = None
    failure_gate: str | None = None

    def finish(self) -> None:
        """Set the first failed gate after all applicable results are known."""

        if not self.support_plane_clear:
            self.failure_gate = "support_plane"
        elif self.pickup_plan is True:
            # The native plan_grasp result is authoritative: it contains both
            # a valid pregrasp trajectory and a valid terminal approach.
            self.failure_gate = None
        elif self.final_endpoint_ik is False:
            self.failure_gate = "final_endpoint_ik"
        elif self.pregrasp_endpoint_ik is False:
            self.failure_gate = "pregrasp_endpoint_ik"
        elif self.pickup_plan is False:
            self.failure_gate = "pickup_plan"
        else:
            self.failure_gate = "not_evaluated"

    @property
    def passed(self) -> bool:
        return self.pickup_plan is True

    def to_dict(self) -> dict:
        return asdict(self)


def local_z_pregrasp(world_T_G: np.ndarray, offset_m: float) -> np.ndarray:
    """Translate a grasp along its own Z axis, preserving its orientation."""

    return np.asarray(world_T_G, dtype=np.float64) @ tra.translation_matrix(
        [0.0, 0.0, float(offset_m)]
    )


def support_clearance(
    hand_vertices_G: np.ndarray, world_T_G: np.ndarray, support_z_m: float
) -> float:
    """Return the lowest open-hand vertex height relative to a support plane."""

    vertices_world = trimesh.transform_points(hand_vertices_G, world_T_G)
    return float(vertices_world[:, 2].min() - support_z_m)


def support_plane_audit(
    candidates: Iterable[GraspCandidate],
    world_T_object: np.ndarray,
    hand_vertices_G: np.ndarray,
    *,
    support_z_m: float,
    approach_offset_m: float,
    tolerance_m: float = 0.0,
) -> list[CandidateAudit]:
    """Evaluate every candidate without mutating or reranking the atlas."""

    records: list[CandidateAudit] = []
    for candidate in candidates:
        world_T_G = np.asarray(world_T_object) @ candidate.object_T_G
        world_T_pregrasp = local_z_pregrasp(world_T_G, approach_offset_m)
        final_clearance = support_clearance(
            hand_vertices_G, world_T_G, support_z_m
        )
        pregrasp_clearance = support_clearance(
            hand_vertices_G, world_T_pregrasp, support_z_m
        )
        record = CandidateAudit(
            candidate_id=candidate.candidate_id,
            family_id=candidate.family_id,
            graspgenx_score=candidate.score,
            world_T_G=world_T_G.tolist(),
            world_T_pregrasp_G=world_T_pregrasp.tolist(),
            final_support_clearance_m=final_clearance,
            pregrasp_support_clearance_m=pregrasp_clearance,
            support_plane_clear=(
                final_clearance >= -tolerance_m
                and pregrasp_clearance >= -tolerance_m
            ),
        )
        record.finish()
        records.append(record)
    return records


def summarize(records: Iterable[CandidateAudit]) -> dict[str, int]:
    values = list(records)
    return {
        "atlas_candidates": len(values),
        "support_plane_clear": sum(item.support_plane_clear for item in values),
        "final_endpoint_ik": sum(item.final_endpoint_ik is True for item in values),
        "pregrasp_endpoint_ik": sum(
            item.pregrasp_endpoint_ik is True for item in values
        ),
        "pickup_plan": sum(item.pickup_plan is True for item in values),
    }
