"""Load and compile the fixed T/U/cube assembly task.

This module owns task semantics only.  It deliberately does not import cuRobo,
Isaac, Newton, ROS, or a robot model.  A motion backend consumes the compiled
commands; it may not invent object-state transitions on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TaskSpecError(ValueError):
    """The task file is internally inconsistent or references stale data."""


def _tuple(values: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise TaskSpecError(f"{label} must contain {size} numbers")
    result = tuple(float(value) for value in values)
    if not np.isfinite(result).all():
        raise TaskSpecError(f"{label} contains a non-finite number")
    return result


@dataclass(frozen=True)
class Transform:
    """Rigid transform with an XYZW quaternion."""

    translation: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], label: str) -> "Transform":
        translation = _tuple(value.get("translation"), 3, f"{label}.translation")
        quaternion = _tuple(
            value.get("quaternion_xyzw"), 4, f"{label}.quaternion_xyzw"
        )
        norm = float(np.linalg.norm(quaternion))
        if not np.isclose(norm, 1.0, atol=1e-7):
            raise TaskSpecError(f"{label} quaternion norm is {norm}, expected 1")
        return cls(translation=translation, quaternion_xyzw=quaternion)

    @property
    def matrix(self) -> np.ndarray:
        x, y, z, w = self.quaternion_xyzw
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        result[:3, 3] = self.translation
        return result


@dataclass(frozen=True)
class PartSpec:
    part_id: str
    mesh: Path
    geometry_config: Path
    grasp_pools: Mapping[str, Path]
    allowed_grasp_cuboids: tuple[str, ...]
    keep_clear_connections: tuple[str, ...]


@dataclass(frozen=True)
class ConnectionSpec:
    connection_id: str
    parent: str
    child: str
    parent_T_child: Transform
    parent_contact_point_m: tuple[float, float, float]
    child_contact_point_m: tuple[float, float, float]
    parent_outward_normal: tuple[float, float, float]
    child_outward_normal: tuple[float, float, float]


@dataclass(frozen=True)
class SequenceStep:
    step_id: str
    action: str
    role: str | None = None
    part: str | None = None
    connection: str | None = None


@dataclass(frozen=True)
class TaskSnapshot:
    loose_parts: tuple[str, ...]
    hand_payloads: Mapping[str, tuple[str, ...]]
    placed_assembly: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "loose_parts": list(self.loose_parts),
            "hand_payloads": {
                role: list(parts) for role, parts in self.hand_payloads.items()
            },
            "placed_assembly": list(self.placed_assembly),
        }


@dataclass(frozen=True)
class CompiledCommand:
    kind: str
    role: str | None = None
    part: str | None = None
    connection: str | None = None
    stationary_role: str | None = None
    scene_effect: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "kind": self.kind,
                "role": self.role,
                "part": self.part,
                "connection": self.connection,
                "stationary_role": self.stationary_role,
                "scene_effect": self.scene_effect,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class CompiledStage:
    step_id: str
    action: str
    before: TaskSnapshot
    commands: tuple[CompiledCommand, ...]
    after: TaskSnapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "before": self.before.to_dict(),
            "commands": [command.to_dict() for command in self.commands],
            "after": self.after.to_dict(),
        }


@dataclass(frozen=True)
class AssemblyTask:
    task_id: str
    root_part: str
    final_object_id: str
    roles: tuple[str, ...]
    physical_hands: tuple[str, ...]
    parts: Mapping[str, PartSpec]
    connections: Mapping[str, ConnectionSpec]
    sequence: tuple[SequenceStep, ...]
    source_path: Path

    def _snapshot(
        self,
        loose: set[str],
        hands: Mapping[str, set[str]],
        placed: set[str],
    ) -> TaskSnapshot:
        return TaskSnapshot(
            loose_parts=tuple(sorted(loose)),
            hand_payloads={role: tuple(sorted(hands[role])) for role in self.roles},
            placed_assembly=tuple(sorted(placed)),
        )

    def compile(self) -> tuple[CompiledStage, ...]:
        """Compile high-level steps into explicit planning and scene operations."""

        loose = set(self.parts)
        hands = {role: set() for role in self.roles}
        placed: set[str] = set()
        stages: list[CompiledStage] = []

        for step in self.sequence:
            before = self._snapshot(loose, hands, placed)
            if step.action == "pick":
                if step.role not in hands or step.part not in self.parts:
                    raise TaskSpecError(f"Invalid pick step {step.step_id}")
                if step.part not in loose:
                    raise TaskSpecError(
                        f"{step.step_id} cannot pick non-loose part {step.part}"
                    )
                if hands[step.role]:
                    raise TaskSpecError(
                        f"{step.step_id} requires empty {step.role} hand"
                    )
                commands = (
                    CompiledCommand("select_qualified_grasp", step.role, step.part),
                    CompiledCommand("plan_to_pregrasp", step.role, step.part),
                    CompiledCommand("plan_contact_approach", step.role, step.part),
                    CompiledCommand("close_hand", step.role, step.part),
                    CompiledCommand(
                        "attach_part",
                        step.role,
                        step.part,
                        scene_effect="remove loose world object; add hand attachment",
                    ),
                    CompiledCommand("plan_retract", step.role, step.part),
                )
                loose.remove(step.part)
                hands[step.role].add(step.part)

            elif step.action == "mate":
                if step.connection not in self.connections:
                    raise TaskSpecError(
                        f"{step.step_id} references unknown connection {step.connection}"
                    )
                connection = self.connections[step.connection]
                holder, worker = "holder", "worker"
                if connection.parent not in hands[holder]:
                    raise TaskSpecError(
                        f"{step.step_id} requires {connection.parent} in holder payload"
                    )
                if hands[worker] != {connection.child}:
                    raise TaskSpecError(
                        f"{step.step_id} requires worker to hold only {connection.child}"
                    )
                commands = (
                    CompiledCommand(
                        "plan_mate_precontact",
                        worker,
                        connection.child,
                        connection.connection_id,
                        stationary_role=holder,
                    ),
                    CompiledCommand(
                        "plan_mate_contact",
                        worker,
                        connection.child,
                        connection.connection_id,
                        stationary_role=holder,
                    ),
                    CompiledCommand("open_hand", worker, connection.child),
                    CompiledCommand(
                        "transfer_to_holder_composite",
                        worker,
                        connection.child,
                        connection.connection_id,
                        stationary_role=holder,
                        scene_effect=(
                            "remove worker attachment; replace holder collision model "
                            "with updated composite"
                        ),
                    ),
                    CompiledCommand(
                        "plan_retreat", worker, connection.child, stationary_role=holder
                    ),
                )
                hands[holder].add(connection.child)
                hands[worker].clear()

            elif step.action == "place":
                role = step.role
                if role not in hands:
                    raise TaskSpecError(f"Invalid place role in {step.step_id}")
                expected = set(self.parts)
                if hands[role] != expected:
                    raise TaskSpecError(
                        f"{step.step_id} requires complete assembly {sorted(expected)}, "
                        f"got {sorted(hands[role])}"
                    )
                commands = (
                    CompiledCommand("plan_place_precontact", role),
                    CompiledCommand("plan_place_contact", role),
                    CompiledCommand("open_hand", role),
                    CompiledCommand(
                        "detach_complete_assembly",
                        role,
                        scene_effect="remove hand attachment; add placed composite to world",
                    ),
                    CompiledCommand("plan_retreat", role),
                )
                placed = set(hands[role])
                hands[role].clear()

            else:
                raise TaskSpecError(
                    f"{step.step_id} has unsupported action {step.action!r}"
                )

            stages.append(
                CompiledStage(
                    step_id=step.step_id,
                    action=step.action,
                    before=before,
                    commands=commands,
                    after=self._snapshot(loose, hands, placed),
                )
            )

        if placed != set(self.parts) or loose or any(hands.values()):
            raise TaskSpecError("Sequence does not finish with one placed complete assembly")
        return tuple(stages)

    def member_transforms(self) -> Mapping[str, np.ndarray]:
        """Return root-part transforms for every member of the final assembly."""

        result: dict[str, np.ndarray] = {self.root_part: np.eye(4)}
        remaining = dict(self.connections)
        while remaining:
            progressed = False
            for connection_id, connection in list(remaining.items()):
                if connection.parent not in result:
                    continue
                result[connection.child] = (
                    result[connection.parent] @ connection.parent_T_child.matrix
                )
                del remaining[connection_id]
                progressed = True
            if not progressed:
                raise TaskSpecError(
                    "Connections do not form a tree rooted at " + self.root_part
                )
        if set(result) != set(self.parts):
            raise TaskSpecError(
                f"Assembly tree covers {sorted(result)}, expected {sorted(self.parts)}"
            )
        return result

    def readiness_report(self) -> dict[str, Any]:
        """Report which holder/worker hand assignments have all required atlases."""

        picks_by_role = {
            role: [
                step.part
                for step in self.sequence
                if step.action == "pick" and step.role == role
            ]
            for role in self.roles
        }
        assignments = []
        for hands in itertools.permutations(self.physical_hands, len(self.roles)):
            role_to_hand = dict(zip(self.roles, hands))
            missing = []
            for role, parts in picks_by_role.items():
                hand = role_to_hand[role]
                for part_id in parts:
                    if hand not in self.parts[part_id].grasp_pools:
                        missing.append({"role": role, "hand": hand, "part": part_id})
            assignments.append(
                {
                    "role_to_hand": role_to_hand,
                    "ready": not missing,
                    "missing_grasp_pools": missing,
                }
            )
        return {
            "task_id": self.task_id,
            "assembly_geometry_ready": True,
            "motion_planning_ready": any(item["ready"] for item in assignments),
            "assignments": assignments,
        }

    def to_compiled_dict(self) -> dict[str, Any]:
        transforms = self.member_transforms()
        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "source": str(self.source_path.relative_to(PROJECT_ROOT)),
            "root_part": self.root_part,
            "final_object_id": self.final_object_id,
            "member_transforms_root_T_part": {
                part_id: transform.tolist() for part_id, transform in transforms.items()
            },
            "stages": [stage.to_dict() for stage in self.compile()],
            "readiness": self.readiness_report(),
        }


def _project_path(value: str, label: str) -> Path:
    path = (PROJECT_ROOT / value).resolve()
    if not path.is_file():
        raise TaskSpecError(f"{label} does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_assembly_task(path: Path | str) -> AssemblyTask:
    """Load, validate, and return a project assembly task."""

    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = (PROJECT_ROOT / source_path).resolve()
    payload = yaml.safe_load(source_path.read_text())
    if payload.get("schema_version") != 1:
        raise TaskSpecError("Only assembly task schema_version 1 is supported")

    parts: dict[str, PartSpec] = {}
    geometry_cuboids: dict[str, set[str]] = {}
    for part_id, value in payload["parts"].items():
        mesh = _project_path(value["mesh"], f"parts.{part_id}.mesh")
        geometry_config = _project_path(
            value["geometry_config"], f"parts.{part_id}.geometry_config"
        )
        geometry = yaml.safe_load(geometry_config.read_text())
        cuboids = {
            str(cuboid["name"])
            for cuboid in geometry.get("shape", {}).get("cuboids", [])
        }
        geometry_cuboids[part_id] = cuboids

        pools: dict[str, Path] = {}
        for hand, pool_value in value.get("grasp_pools", {}).items():
            pool = _project_path(pool_value, f"parts.{part_id}.grasp_pools.{hand}")
            header = yaml.safe_load(pool.read_text())
            if header.get("format") != "g1_aprilcube_arm_grasp_pool":
                raise TaskSpecError(f"Unsupported grasp-pool format in {pool}")
            if header.get("hand_side") != hand:
                raise TaskSpecError(f"{pool} declares hand {header.get('hand_side')}, not {hand}")
            if header.get("object_id") != part_id:
                raise TaskSpecError(
                    f"{pool} declares object {header.get('object_id')}, not {part_id}"
                )
            if header.get("object_mesh_sha256") != _sha256(mesh):
                raise TaskSpecError(f"Mesh hash does not match grasp pool {pool}")
            pools[str(hand)] = pool

        constraints = value.get("grasp_constraints", {})
        allowed = tuple(str(item) for item in constraints.get("allowed_cuboids", []))
        unknown = set(allowed) - cuboids
        if unknown:
            raise TaskSpecError(
                f"Part {part_id} has unknown allowed grasp cuboids {sorted(unknown)}"
            )
        parts[part_id] = PartSpec(
            part_id=part_id,
            mesh=mesh,
            geometry_config=geometry_config,
            grasp_pools=pools,
            allowed_grasp_cuboids=allowed,
            keep_clear_connections=tuple(
                str(item) for item in constraints.get("keep_clear_connections", [])
            ),
        )

    connections: dict[str, ConnectionSpec] = {}
    for connection_id, value in payload["connections"].items():
        connection = ConnectionSpec(
            connection_id=connection_id,
            parent=str(value["parent"]),
            child=str(value["child"]),
            parent_T_child=Transform.from_mapping(
                value["parent_T_child"], f"connections.{connection_id}.parent_T_child"
            ),
            parent_contact_point_m=_tuple(
                value["parent_contact_point_m"],
                3,
                f"connections.{connection_id}.parent_contact_point_m",
            ),
            child_contact_point_m=_tuple(
                value["child_contact_point_m"],
                3,
                f"connections.{connection_id}.child_contact_point_m",
            ),
            parent_outward_normal=_tuple(
                value["parent_outward_normal"],
                3,
                f"connections.{connection_id}.parent_outward_normal",
            ),
            child_outward_normal=_tuple(
                value["child_outward_normal"],
                3,
                f"connections.{connection_id}.child_outward_normal",
            ),
        )
        if connection.parent not in parts or connection.child not in parts:
            raise TaskSpecError(f"Connection {connection_id} references an unknown part")
        transformed_point = (
            connection.parent_T_child.matrix
            @ np.array([*connection.child_contact_point_m, 1.0])
        )[:3]
        if not np.allclose(
            transformed_point, connection.parent_contact_point_m, atol=1e-9
        ):
            raise TaskSpecError(
                f"Connection {connection_id} contact points do not coincide"
            )
        transformed_normal = (
            connection.parent_T_child.matrix[:3, :3]
            @ np.asarray(connection.child_outward_normal)
        )
        if not np.allclose(
            transformed_normal,
            -np.asarray(connection.parent_outward_normal),
            atol=1e-9,
        ):
            raise TaskSpecError(
                f"Connection {connection_id} contact normals are not opposed"
            )
        connections[connection_id] = connection

    for part in parts.values():
        unknown = set(part.keep_clear_connections) - set(connections)
        if unknown:
            raise TaskSpecError(
                f"Part {part.part_id} references unknown keep-clear connections {sorted(unknown)}"
            )

    sequence = tuple(
        SequenceStep(
            step_id=str(value["id"]),
            action=str(value["action"]),
            role=value.get("role"),
            part=value.get("part"),
            connection=value.get("connection"),
        )
        for value in payload["sequence"]
    )
    step_ids = [step.step_id for step in sequence]
    if len(step_ids) != len(set(step_ids)):
        raise TaskSpecError("Sequence step IDs must be unique")

    task = AssemblyTask(
        task_id=str(payload["task_id"]),
        root_part=str(payload["assembly"]["root_part"]),
        final_object_id=str(payload["assembly"]["final_object_id"]),
        roles=tuple(str(value) for value in payload["roles"]),
        physical_hands=tuple(str(value) for value in payload["physical_hands"]),
        parts=parts,
        connections=connections,
        sequence=sequence,
        source_path=source_path,
    )
    if task.roles != ("holder", "worker"):
        raise TaskSpecError("Schema v1 requires roles [holder, worker]")
    if task.root_part not in parts:
        raise TaskSpecError(f"Unknown assembly root part {task.root_part}")
    task.member_transforms()
    task.compile()
    return task
