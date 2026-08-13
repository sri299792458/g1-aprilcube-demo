from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from g1_aprilcube_demo.planning.runtime_assembly import (
    RoleAssignment,
    candidates_by_family,
    ready_role_assignments,
)
from g1_aprilcube_demo.planning.grasp_goalset import GraspCandidate


def test_pair_targets_are_converted_from_roles_to_curobo_left_right_order():
    holder = np.eye(4)
    holder[0, 3] = 1.0
    worker = np.eye(4)
    worker[0, 3] = 2.0

    left_holder = RoleAssignment("left", "right")
    left, right = left_holder.ordered_pair_targets(holder, worker)
    assert left is holder
    assert right is worker

    right_holder = RoleAssignment("right", "left")
    left, right = right_holder.ordered_pair_targets(holder, worker)
    assert left is worker
    assert right is holder


def test_only_complete_evidence_backed_assignments_reach_runtime_search():
    task = SimpleNamespace(
        readiness_report=lambda: {
            "assignments": [
                {
                    "role_to_hand": {"holder": "left", "worker": "right"},
                    "ready": True,
                    "missing_grasp_pools": [],
                },
                {
                    "role_to_hand": {"holder": "right", "worker": "left"},
                    "ready": False,
                    "missing_grasp_pools": [
                        {"role": "worker", "hand": "left", "part": "u_legs"}
                    ],
                },
            ]
        }
    )

    assert ready_role_assignments(task) == [RoleAssignment("left", "right")]


def test_grasp_families_are_explicit_partitions_not_mixed_batches():
    def candidate(candidate_id: str, family_id: str) -> GraspCandidate:
        return GraspCandidate(candidate_id, family_id, 0.0, np.eye(4))

    grouped = candidates_by_family(
        [
            candidate("a0", "family_a"),
            candidate("b0", "family_b"),
            candidate("a1", "family_a"),
            candidate("c0", "family_c"),
            candidate("b1", "family_b"),
        ]
    )

    actual = [
        (family, [item.candidate_id for item in values])
        for family, values in grouped
    ]
    assert actual == [
        ("family_a", ["a0", "a1"]),
        ("family_b", ["b0", "b1"]),
        ("family_c", ["c0"]),
    ]
