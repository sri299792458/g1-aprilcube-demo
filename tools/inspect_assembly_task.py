"""Validate and compile the project-owned assembly task contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_aprilcube_demo.assembly import load_assembly_task


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK = ROOT / "config/tasks/t_u_cube_humanoid_v1.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts/assembly_task/t_u_cube_humanoid_v1/compiled_task.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    task = load_assembly_task(args.task)
    compiled = task.to_compiled_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compiled, indent=2) + "\n")

    print(f"task: {task.task_id}")
    for stage in task.compile():
        print(f"{stage.step_id}: {stage.action}")
        for command in stage.commands:
            print(f"  - {command.kind}")
    print(json.dumps(compiled["readiness"], indent=2))
    print(f"compiled: {args.output}")


if __name__ == "__main__":
    main()
