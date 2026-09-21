from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from g1_aprilcube_demo.grasping.support_atlas import (
    TargetRegionClassifier,
    build_buckets,
    configured_support_conditions,
    evaluate_support,
    load_raw_candidates,
    load_raw_candidates_many,
    minimum_world_z_values,
    semantic_voxels,
)
from g1_aprilcube_demo.grasping.support_atlas import MeshCollisionGate


ROOT = Path(__file__).resolve().parents[1]
U_MESH = ROOT / "generated/aprilcube_parts/u_legs/grasp_mesh.obj"
U_GEOMETRY = ROOT / "config/aprilcube_parts/u_legs.yaml"
HAND_MESH = (
    ROOT
    / "third_party/GraspGenX/assets/x_grippers/dex3_rev1_right/coll_mesh.obj"
)
RAW = ROOT / "artifacts/grasp_atlas/u_legs_v1/raw"


def u_fixture():
    mesh = trimesh.load(U_MESH, force="mesh", process=False)
    supports = configured_support_conditions(
        mesh,
        entries=[
            {"table_up_object": "+X", "label": "left", "symmetry_class": "side"},
            {"table_up_object": "-X", "label": "right", "symmetry_class": "side"},
            {"table_up_object": "+Y", "label": "broad_a", "symmetry_class": "broad"},
            {"table_up_object": "-Y", "label": "broad_b", "symmetry_class": "broad"},
            {"table_up_object": "+Z", "label": "legs", "symmetry_class": "legs"},
            {"table_up_object": "-Z", "label": "bridge", "symmetry_class": "bridge"},
        ],
    )
    return mesh, supports


def grasp_with_z(origin, direction):
    z = np.asarray(direction, dtype=np.float64)
    z /= np.linalg.norm(z)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(reference @ z)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    x = np.cross(reference, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((x, y, z))
    result[:3, 3] = origin
    return result


def test_u_has_six_stable_supports_in_four_geometric_symmetry_classes():
    _, supports = u_fixture()
    assert {item.table_up_sector for item in supports} == {
        "+X",
        "-X",
        "+Y",
        "-Y",
        "+Z",
        "-Z",
    }
    assert {item.symmetry_class for item in supports} == {
        "side",
        "broad",
        "legs",
        "bridge",
    }
    for support in supports:
        transformed = trimesh.transform_points(
            trimesh.load(U_MESH, force="mesh", process=False).vertices,
            support.support_T_object,
        )
        assert np.isclose(transformed[:, 2].min(), 0.0, atol=1e-8)


def test_semantic_target_region_distinguishes_u_legs_and_bridge_cavity():
    mesh, _ = u_fixture()
    classifier = TargetRegionClassifier(
        mesh,
        semantic_voxels(U_GEOMETRY),
        surface_tolerance_m=0.0005,
    )
    table_up = np.array([0.0, 1.0, 0.0])
    left = classifier.classify(
        grasp_with_z([0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]), table_up
    )
    right = classifier.classify(
        grasp_with_z([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]), table_up
    )
    bridge = classifier.classify(
        grasp_with_z([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]), table_up
    )
    assert (left["component"], left["surface_sector"], left["surface_relation"]) == (
        "left_leg",
        "+X",
        "cavity",
    )
    assert (
        right["component"],
        right["surface_sector"],
        right["surface_relation"],
    ) == ("right_leg", "-X", "cavity")
    assert (
        bridge["component"],
        bridge["surface_sector"],
        bridge["surface_relation"],
    ) == ("hip_bridge", "-Z", "cavity")


def test_known_broad_face_candidate_enters_support_conditioned_survivors():
    mesh, supports = u_fixture()
    support = next(item for item in supports if item.table_up_sector == "+Y")
    all_candidates = load_raw_candidates(RAW)
    by_id = {candidate.candidate_id: candidate for candidate in all_candidates}
    clear = by_id["u_legs__seed_0000000019__sample_254"]
    below_table = by_id["u_legs__seed_0000000019__sample_000"]
    hand = trimesh.load(HAND_MESH, force="mesh", process=False)
    result = evaluate_support(
        support=support,
        candidates=[clear, below_table],
        hand_mesh=hand,
        collision=MeshCollisionGate(hand, mesh),
        classifier=TargetRegionClassifier(
            mesh,
            semantic_voxels(U_GEOMETRY),
            surface_tolerance_m=0.0005,
        ),
        approach_offset_m=-0.10,
        corridor_step_m=0.001,
        table_tolerance_m=0.000001,
    )
    assert result["survivor_count"] == 1
    assert result["survivors"][0]["candidate_id"] == clear.candidate_id
    assert result["first_rejection_reason_counts"] == {"final_table": 1}


def test_batched_hull_minimum_z_matches_every_dense_hand_vertex():
    hand = trimesh.load(HAND_MESH, force="mesh", process=False)
    candidates = load_raw_candidates(RAW)[:32]
    matrices = np.stack([candidate.object_T_G for candidate in candidates])
    measured = minimum_world_z_values(hand, matrices, batch_size=7)
    expected = np.asarray(
        [
            trimesh.transformations.transform_points(
                hand.vertices, matrix
            )[:, 2].min()
            for matrix in matrices
        ]
    )
    np.testing.assert_allclose(measured, expected, atol=1e-12, rtol=0.0)


def test_bucketing_retains_every_support_conditioned_member():
    key_a = {
        "support_id": "a",
        "support_symmetry_class": "a",
        "component": "left_leg",
        "surface_sector": "+X",
        "surface_relation": "cavity",
        "support_relation": "lateral",
        "approach_sector": "-X",
    }
    key_b = dict(key_a, support_id="b", support_symmetry_class="b")
    support_results = [
        {
            "survivors": [
                {
                    "candidate_id": "one",
                    "proposal_bucket_id": "proposal_a",
                    "proposal_bucket_key": key_a,
                },
                {
                    "candidate_id": "two",
                    "proposal_bucket_id": "proposal_a",
                    "proposal_bucket_key": key_a,
                },
            ]
        },
        {
            "survivors": [
                {
                    "candidate_id": "one",
                    "proposal_bucket_id": "proposal_b",
                    "proposal_bucket_key": key_b,
                }
            ]
        },
    ]
    buckets = build_buckets(support_results)
    assert sum(bucket["member_count"] for bucket in buckets) == 3
    assert {bucket["proposal_bucket_id"] for bucket in buckets} == {
        "proposal_a",
        "proposal_b",
    }


def test_multiple_raw_sources_reject_duplicate_candidate_ids():
    with np.testing.assert_raises_regex(
        ValueError, "Duplicate candidate ID across raw sources"
    ):
        load_raw_candidates_many([RAW, RAW])
