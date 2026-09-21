#!/usr/bin/env python3
"""Plan one observed T/U/cube scene and save an executable artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_aprilcube_demo.planning.runtime_assembly import (  # noqa: E402
    RuntimeAssemblyPlanner,
    save_run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/planning/t_u_cube_runtime_v2.yaml",
    )
    parser.add_argument(
        "--observation",
        type=Path,
        default=ROOT / "config/observations/t_u_cube_nominal_v1.yaml",
    )
    parser.add_argument(
        "--task",
        type=Path,
        default=ROOT / "config/tasks/t_u_cube_humanoid_v1.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = args.output or ROOT / config["render"]["output_dir"] / args.observation.stem
    planner = RuntimeAssemblyPlanner(args.config, args.observation, args.task)
    try:
        run = planner.execute()
        save_run(run, output)
        print(f"runtime assembly planned: {output / 'planning_report.json'}")
    finally:
        planner.close()


if __name__ == "__main__":
    main()
