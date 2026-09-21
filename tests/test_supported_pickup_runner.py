from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "tools/run_isaac_supported_pickup.py"
CONFIG_PATH = (
    ROOT
    / "config/grasp_support/u_legs_right_broad_face_isaac_v1.yaml"
)
UPRIGHT_CONFIG_PATH = (
    ROOT
    / "config/grasp_support/u_legs_right_upright_isaac_v1.yaml"
)
UPRIGHT_REVIEW_CONFIG_PATH = (
    ROOT
    / "config/grasp_support/u_legs_right_upright_review2_isaac_v1.yaml"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_isaac_supported_pickup", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def transform(z: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def passing_trace_record() -> dict:
    names_and_z = [
        ("settled_on_support", 0.20),
        ("approach_complete", 0.10),
        ("closed_before_lift", 0.10),
        ("lift_complete", 0.30),
        ("final_hold", 0.30),
    ]
    return {
        "candidate_id": "candidate",
        "physics": {
            "mode": "supported_pickup",
            "supported_pickup_contract": {"lift_distance_m": 0.20},
        },
        "phases": [
            {
                "name": name,
                "world_T_object": transform(0.02),
                "world_T_G": transform(z),
            }
            for name, z in names_and_z
        ],
        "result": {
            "passed": True,
            "digit_contact_pass": True,
            "hand_table_contact_any": False,
            "object_table_contact_during_final_hold": False,
            "final_gripper_command_position_error_m": 0.0,
            "final_gripper_command_orientation_error_rad": 0.0,
            "max_hand_table_contact_force_N": 0.0,
            "max_final_hold_object_table_contact_force_N": 0.0,
        },
        "supported_pickup": {
            "support_label": "broad_minus_y_face_down",
            "target_region": {"component": "left_leg"},
        },
    }


def write_validation_fixture(tmp_path: Path, record: dict) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "input.yaml"
    result_path = tmp_path / "result.yaml"
    trace_path = tmp_path / "trace.jsonl"
    input_path.write_text(
        yaml.safe_dump({"grasps": {"candidate": {"confidence": 1.0}}})
    )
    result_path.write_text(
        yaml.safe_dump(
            {
                "grasps": {
                    "candidate": {
                        "confidence": float(record["result"]["passed"])
                    }
                }
            }
        )
    )
    trace_path.write_text(json.dumps(record) + "\n")
    return input_path, result_path, trace_path


def test_broad_face_selection_contains_both_explicit_supports():
    runner = load_runner()
    config = runner.load_config(CONFIG_PATH)
    trials = runner.selected_trials(config)
    counts = Counter(trial["support_label"] for trial in trials)
    assert len(trials) == 42
    assert counts == {
        "broad_minus_y_face_down": 33,
        "broad_plus_y_face_down": 9,
    }


def test_upright_selection_keeps_all_survivors_and_review_is_explicit():
    runner = load_runner()
    upright = runner.load_config(UPRIGHT_CONFIG_PATH)
    assert len(runner.selected_trials(upright)) == 1837

    review = runner.load_config(UPRIGHT_REVIEW_CONFIG_PATH)
    selected = runner.selected_trials(review)
    requested = review["selection"]["trials"]
    assert len(selected) == 14
    assert [trial["candidate_id"] for trial in selected] == [
        record["candidate_id"] for record in requested
    ]


def test_trace_validation_proves_the_commanded_vertical_lift(tmp_path):
    runner = load_runner()
    record = passing_trace_record()
    paths = write_validation_fixture(tmp_path, record)
    assert runner.validate_chunk(
        input_path=paths[0],
        result_path=paths[1],
        trace_path=paths[2],
    ) == [record]

    bad_record = copy.deepcopy(record)
    bad_record["phases"][3]["world_T_G"][2][3] = 0.29
    bad_paths = write_validation_fixture(tmp_path / "bad", bad_record)
    with pytest.raises(
        ValueError,
        match="did not execute the commanded lift",
    ):
        runner.validate_chunk(
            input_path=bad_paths[0],
            result_path=bad_paths[1],
            trace_path=bad_paths[2],
        )


def test_report_counts_each_independent_physical_gate():
    runner = load_runner()
    config = runner.load_config(CONFIG_PATH)
    passed = passing_trace_record()
    failed = copy.deepcopy(passed)
    failed["candidate_id"] = "failed"
    failed["result"].update(
        {
            "passed": False,
            "digit_contact_pass": False,
            "hand_table_contact_any": True,
            "object_table_contact_during_final_hold": True,
            "max_hand_table_contact_force_N": 10.0,
            "max_final_hold_object_table_contact_force_N": 2.0,
        }
    )
    report = runner.make_report(
        config_path=CONFIG_PATH,
        config=config,
        records=[passed, failed],
        video=None,
    )
    assert report["pass_count"] == 1
    assert report["digit_contact_pass_count"] == 1
    assert report["hand_table_contact_count"] == 1
    assert report["object_table_final_hold_contact_count"] == 1
    assert report["final_object_table_contact_force_range_N"] == {
        "min": 0.0,
        "max": 2.0,
    }
    assert report["report"] == runner.report_metadata(config)
