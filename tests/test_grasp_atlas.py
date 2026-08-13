from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml

from tools import build_dex3_isaac_grasp_input as isaac_input
from tools import build_grasp_atlas as atlas
from tools import run_isaac_atlas_qualification as qualification


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/grasp_atlas/cube_v1.yaml"


def atlas_fixture():
    config = atlas.load_config(CONFIG_PATH)
    surface = atlas.build_surface_regions(config)
    mesh = trimesh.load(
        ROOT / config["object"]["mesh"], force="mesh", process=False
    )
    return config, surface, mesh


def test_six_face_centers_map_to_named_regions():
    config, surface, mesh = atlas_fixture()
    expected = {
        "+X": [0.0225, 0.0, 0.0],
        "-X": [-0.0225, 0.0, 0.0],
        "+Y": [0.0, 0.0225, 0.0],
        "-Y": [0.0, -0.0225, 0.0],
        "+Z": [0.0, 0.0, 0.0225],
        "-Z": [0.0, 0.0, -0.0225],
    }
    for face, point in expected.items():
        mapped = atlas.map_contact_point(
            point,
            regions=surface["regions"],
            mesh=mesh,
            mesh_tolerance_m=config["surface_mapping"]["mesh_tolerance_m"],
            fillet_radius_m=config["object"]["fillet_radius_m"],
            floating_point_tolerance_m=config["surface_mapping"][
                "floating_point_tolerance_m"
            ],
        )
        assert mapped["primary_region"].endswith("/" + face)
        assert mapped["mapping_valid"]
        assert np.allclose(mapped["uv_clamped"], [0.5, 0.5], atol=1e-12)


def test_printed_40mm_cuboid_config_maps_all_six_exact_faces():
    config = atlas.load_config(ROOT / "config/grasp_atlas/cube40_viral_v1.yaml")
    surface = atlas.build_surface_regions(config)

    assert {region["april_tag_id"] for region in surface["regions"]} == set(range(6))
    assert {region["face"] for region in surface["regions"]} == {
        "+X", "-X", "+Y", "-Y", "+Z", "-Z"
    }
    for region in surface["regions"]:
        assert np.isclose(region["basis"]["u_length_m"], 0.04)
        assert np.isclose(region["basis"]["v_length_m"], 0.04)


def test_isaac_result_name_follows_the_configured_mesh_stem(tmp_path):
    path = qualification.expected_result_path(tmp_path, "right", "cube")
    assert path == tmp_path / "grasp_sim_data/dex3_rev1_right/cube.yaml"


def test_ideal_corner_retains_three_broad_face_labels():
    config, surface, mesh = atlas_fixture()
    mapped = atlas.map_contact_point(
        [0.0225, 0.0225, 0.0225],
        regions=surface["regions"],
        mesh=mesh,
        mesh_tolerance_m=config["surface_mapping"]["mesh_tolerance_m"],
        fillet_radius_m=config["object"]["fillet_radius_m"],
        floating_point_tolerance_m=config["surface_mapping"][
            "floating_point_tolerance_m"
        ],
    )
    nearby_faces = {region.rsplit("/", 1)[-1] for region in mapped["nearby_regions_within_fillet_radius"]}
    assert nearby_faces == {"+X", "+Y", "+Z"}


def make_phase(name: str, side: str = "right") -> dict:
    zero = [0.0, 0.0, 0.0]
    return {
        "name": name,
        "contacts": [
            {
                "hand_link": f"{side}_hand_thumb_2_link",
                "net_normal_force_world_N": [0.1, 0.0, 0.0],
                "points": [],
            },
            {
                "hand_link": f"{side}_hand_index_1_link",
                "net_normal_force_world_N": [0.0, 0.1, 0.0],
                "points": [],
            },
            {
                "hand_link": f"{side}_hand_middle_1_link",
                "net_normal_force_world_N": zero,
                "points": [],
            },
        ],
    }


def make_trial(candidate_id: str, translation_x: float = 0.0) -> dict:
    transform = np.eye(4)
    transform[0, 3] = translation_x
    phase_names = [
        "closed_before_tug",
        "after_tug_1",
        "after_tug_2",
        "after_tug_3",
        "after_tug_4",
        "after_tug_5_final",
    ]
    return {
        "candidate_id": candidate_id,
        "candidate_content_sha256": candidate_id,
        "hand_side": "right",
        "input": {"object_T_G": transform.tolist(), "graspgenx_score": 0.5},
        "result": {"passed": True, "final_q": {}},
        "phases": [make_phase(name) for name in phase_names],
    }


def test_family_construction_uses_coarse_chains_and_is_order_independent():
    trials = [make_trial("b", 0.01), make_trial("a", 0.00)]
    mapped = []
    for trial in trials:
        trial = copy.deepcopy(trial)
        trial["contact_persistence"] = atlas.annotate_persistence(trial)
        trial["contact_signature"] = atlas.build_signature(trial)
        trial["family_key_sha256"] = atlas.stable_hash(trial["contact_signature"])
        mapped.append(trial)

    forward = atlas.build_families(mapped, voxel_size_m=0.045, hand_side="right")
    reverse = atlas.build_families(
        list(reversed(mapped)), voxel_size_m=0.045, hand_side="right"
    )
    assert forward == reverse
    assert forward["family_count"] == 1
    signature = forward["families"][0]["signature"]
    assert signature["digit_chains"] == ["index", "thumb"]
    assert signature["palm_contact"] is False
    assert forward["families"][0]["diagnostic_broad_faces_by_chain"] == {
        "index": [],
        "thumb": [],
    }
    assert forward["families"][0]["representatives"][0]["candidate_id"] == "a"


def test_family_signature_uses_the_physics_qualified_final_phase():
    trial = make_trial("contact_transition")
    closed = trial["phases"][0]
    final = trial["phases"][-1]
    closed["contacts"][1]["net_normal_force_world_N"] = [0.0, 0.0, 0.0]
    final["contacts"][1]["net_normal_force_world_N"] = [0.0, 0.1, 0.0]

    assert sorted(atlas.coarse_phase_contacts(closed, "right")) == ["thumb"]
    signature = atlas.build_signature(trial)
    assert signature["digit_chains"] == ["index", "thumb"]


def test_body_pair_scalar_prevents_vector_cancellation_from_erasing_contact():
    phase = make_phase("after_tug_5_final")
    thumb = phase["contacts"][0]
    thumb["net_normal_force_world_N"] = [0.0, 0.0, 0.0]
    thumb["contact_force_magnitude_N"] = 0.75
    index = phase["contacts"][1]
    index["contact_force_magnitude_N"] = 0.0

    assert sorted(atlas.coarse_phase_contacts(phase, "right")) == ["thumb"]


def test_raw_transform_is_unchanged_for_both_exact_hands(tmp_path):
    raw = {
        "format": "isaac_grasp",
        "format_version": "1.0",
        "grasps": {
            "cube_head__seed_0000000019__sample_000": {
                "confidence": 0.75,
                "position": [0.012, -0.034, 0.056],
                "orientation": {"w": 0.5, "xyz": [0.5, -0.5, 0.5]},
                "graspgenx_generation": {
                    "candidate_id": "cube_head__seed_0000000019__sample_000",
                    "candidate_content_sha256": "abc",
                    "batch_index": 0,
                    "generation_seed": 19,
                    "sample_index": 0,
                },
            }
        },
    }
    raw_path = tmp_path / "raw.yaml"
    raw_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    expected = raw["grasps"]["cube_head__seed_0000000019__sample_000"]

    for side in ("right", "left"):
        contract = isaac_input.side_contract(ROOT, side)
        built = isaac_input.build_output(
            raw_path=raw_path,
            hand_config=contract["hand_config"],
            gripper_usd=contract["gripper_usd"],
            object_mesh=ROOT / "generated/aprilcube_parts/cube_head/grasp_mesh.obj",
            object_mass=0.030,
            finger_colliders=contract["finger_colliders"],
            finger_contact_groups=contract["finger_contact_groups"],
            min_contact_groups=2,
            approach_axis=2,
            open_limit="lower",
            contact_trace_links=contract["contact_trace_links"],
            contact_trace_link_aliases=contract["contact_trace_link_aliases"],
        )
        actual = built["grasps"]["cube_head__seed_0000000019__sample_000"]
        assert actual["position"] == expected["position"]
        assert actual["orientation"] == expected["orientation"]
        assert actual["graspgenx_source"]["object_T_G"] == {
            "position": expected["position"],
            "orientation": expected["orientation"],
        }
        serialized = json.dumps(actual)
        wrong_side = "left_hand" if side == "right" else "right_hand"
        assert wrong_side not in serialized
