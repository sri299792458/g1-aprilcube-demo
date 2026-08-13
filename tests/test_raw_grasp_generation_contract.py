from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "tools/run_aprilcube_raw_grasps.py"


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "run_aprilcube_raw_grasps", GENERATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compact_seed_schedule_and_partial_final_shard_are_exact():
    generator = load_generator()
    config = {
        "batch_size": 256,
        "total_candidate_count": 95_904,
        "seeds": {"start": 1_000_019, "step": 10, "count": 375},
    }
    seeds = generator.configured_seeds(config)
    counts = generator.configured_batch_counts(config, seeds)
    assert len(seeds) == 375
    assert len(set(seeds)) == 375
    assert counts[:-1] == [256] * 374
    assert counts[-1] == 160
    assert sum(counts) == 95_904


def test_seed_schedule_must_exactly_cover_requested_batch_count():
    generator = load_generator()
    config = {
        "batch_size": 256,
        "total_candidate_count": 256,
        "seeds": {"start": 1_000, "step": 1, "count": 2},
    }
    seeds = generator.configured_seeds(config)
    with pytest.raises(ValueError, match="exactly ceil"):
        generator.configured_batch_counts(config, seeds)
