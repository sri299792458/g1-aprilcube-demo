#!/usr/bin/env python3
"""Plan the full T/U/cube assembly and save the reproducible run artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from g1_aprilcube_demo.planning.assembly_runner import CuroboAssemblyRunner, save_run


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/planning/t_u_cube_full_assembly_v1.yaml",
    )
    parser.add_argument(
        "--task",
        type=Path,
        default=ROOT / "config/tasks/t_u_cube_humanoid_v1.yaml",
    )
    args = parser.parse_args()
    runner = CuroboAssemblyRunner(args.config, args.task)
    try:
        run = runner.execute()
        output = ROOT / runner.cfg["render"]["output_dir"]
        save_run(run, output)
        print(f"full assembly planned: {output / 'planning_report.json'}")
    finally:
        runner.close()


if __name__ == "__main__":
    main()
