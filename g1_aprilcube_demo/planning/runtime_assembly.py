"""Runtime-pose-conditioned T/U/cube assembly using upstream cuRobo stages."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
import yaml

from g1_aprilcube_demo.assembly import load_assembly_task
from g1_aprilcube_demo.runtime import load_observation
from .curobo_backend import CuroboBackend, PayloadGeometry, TOOL, SLOT
from .grasp_goalset import GraspCandidate, load_grasp_pool, world_grasps
from .workspace import placement_samples, workspace_samples


ROOT = Path(__file__).resolve().parents[2]
MODE_CACHE_CONTRACT = b"connector-mode-v3-diverse-closed-dex3"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


@dataclass
class ObjectState:
    world_T_object: np.ndarray | None = None
    hand: str | None = None
    grasp_T_object: np.ndarray | None = None

    def clone(self) -> "ObjectState":
        return ObjectState(
            None if self.world_T_object is None else self.world_T_object.copy(),
            self.hand,
            None if self.grasp_T_object is None else self.grasp_T_object.copy(),
        )


@dataclass
class Segment:
    name: str
    q: np.ndarray
    objects: dict[str, ObjectState]
    hand_closed: dict[str, float]
    selected_candidate: str | None = None


@dataclass
class RuntimeAssemblyRun:
    config_path: Path
    observation_path: Path
    task_path: Path
    arm_joint_names: list[str]
    observation_id: str
    segments: list[Segment] = field(default_factory=list)
    selected: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "success": True,
            "planner": "upstream cuRobo MotionPlanner; runtime observation conditioned",
            "config": str(self.config_path.relative_to(ROOT)),
            "observation": str(self.observation_path.relative_to(ROOT)),
            "observation_id": self.observation_id,
            "task": str(self.task_path.relative_to(ROOT)),
            "arm_joint_names": self.arm_joint_names,
            "selected_grasps": self.selected,
            "events": self.events,
            "segments": [
                {"name": segment.name, "frames": int(len(segment.q))}
                for segment in self.segments
            ],
        }


@dataclass
class _Snapshot:
    q: np.ndarray
    loose: set[str]
    placed: set[str]
    objects: dict[str, ObjectState]
    payloads: dict[str, PayloadGeometry | None]
    closed: dict[str, float]
    grasps: dict[str, GraspCandidate]
    selected: dict[str, dict[str, Any]]
    segment_count: int
    event_count: int


@dataclass(frozen=True)
class AssemblyMode:
    t_grasp: GraspCandidate
    u_grasp: GraspCandidate
    cube_grasp: GraspCandidate
    u_work_id: str
    u_world_T_root: np.ndarray
    u_precontact_q: np.ndarray
    head_work_id: str
    head_world_T_root: np.ndarray
    head_precontact_q: np.ndarray


class RuntimeAssemblyPlanner:
    """Plan the complete task before any eventual hardware execution."""

    def __init__(
        self, config_path: str | Path, observation_path: str | Path, task_path: str | Path
    ):
        self.config_path = project_path(config_path)
        self.observation_path = project_path(observation_path)
        self.task_path = project_path(task_path)
        self.cfg = yaml.safe_load(self.config_path.read_text())
        if self.cfg.get("schema_version") != 2:
            raise ValueError("Runtime planner requires schema_version 2")
        self.task = load_assembly_task(self.task_path)
        self.compiled_stages = self.task.compile()
        part_meshes = {
            name: project_path(self.task.parts[name].mesh) for name in self.task.parts
        }
        self.observation = load_observation(self.observation_path, part_meshes)
        if self.observation.planning_frame != "world":
            raise ValueError("This prototype currently requires planning_frame: world")

        robot_cfg = self.cfg["robot"]
        self.arm_joint_names = list(robot_cfg["arm_joint_names"])
        self.initial_q = np.asarray(robot_cfg["start_arm_joint_positions"], dtype=np.float64)
        if self.initial_q.shape != (len(self.arm_joint_names),):
            raise ValueError("start_arm_joint_positions does not match arm_joint_names")
        self.base_world = np.eye(4)
        self.base_world[:3, 3] = np.asarray(robot_cfg["base_world_translation_m"])

        self.pools: dict[str, tuple[GraspCandidate, ...]] = {}
        for part, hand in (("t_body", "left"), ("u_legs", "right"), ("cube_head", "right")):
            path = project_path(self.task.parts[part].grasp_pools[hand])
            self.pools[part] = load_grasp_pool(path)

        self.backend = CuroboBackend(
            robot_config=project_path(robot_cfg["model"]),
            base_world=self.base_world,
            arm_joint_names=self.arm_joint_names,
            part_meshes=part_meshes,
            part_geometry={
                name: project_path(part.geometry_config)
                for name, part in self.task.parts.items()
            },
            hand_profiles={
                side: project_path(path)
                for side, path in robot_cfg["hand_profiles"].items()
            },
            planner_options=self.cfg["planner"],
        )
        self.q = self.initial_q.copy()
        self.loose = set(self.task.parts)
        self.placed: set[str] = set()
        self.objects = {
            name: ObjectState(world_T_object=matrix.copy())
            for name, matrix in self.observation.world_T_objects.items()
        }
        self.payloads: dict[str, PayloadGeometry | None] = {"left": None, "right": None}
        self.closed = {"left": 0.0, "right": 0.0}
        self.grasps: dict[str, GraspCandidate] = {}
        self.run = RuntimeAssemblyRun(
            self.config_path,
            self.observation_path,
            self.task_path,
            self.arm_joint_names,
            self.observation.observation_id,
        )
        self.contact_links = {
            side: tuple(self.cfg["robot"]["contact_links"][side])
            for side in ("left", "right")
        }

    def close(self) -> None:
        # Stage planners are context-managed; the backend retains only tensors.
        torch.cuda.synchronize()

    def _scene(self) -> dict[str, Any]:
        loose_poses = {
            part: state.world_T_object
            for part, state in self.objects.items()
            if state.world_T_object is not None
        }
        return self.backend.scene(
            self.observation.table.center,
            self.observation.table.dimensions,
            loose_poses,
        )

    def _snapshot_objects(self) -> dict[str, ObjectState]:
        return {name: state.clone() for name, state in self.objects.items()}

    def _snapshot(self) -> _Snapshot:
        return _Snapshot(
            self.q.copy(), set(self.loose), set(self.placed), self._snapshot_objects(),
            copy.deepcopy(self.payloads), dict(self.closed), dict(self.grasps),
            copy.deepcopy(self.run.selected), len(self.run.segments), len(self.run.events),
        )

    def _restore(self, snapshot: _Snapshot) -> None:
        self.q = snapshot.q.copy()
        self.loose = set(snapshot.loose)
        self.placed = set(snapshot.placed)
        self.objects = {name: state.clone() for name, state in snapshot.objects.items()}
        self.payloads = copy.deepcopy(snapshot.payloads)
        self.closed = dict(snapshot.closed)
        self.grasps = dict(snapshot.grasps)
        self.run.selected = copy.deepcopy(snapshot.selected)
        del self.run.segments[snapshot.segment_count :]
        del self.run.events[snapshot.event_count :]

    def _record(
        self, name: str, q: np.ndarray, candidate: str | None = None
    ) -> None:
        self.run.segments.append(
            Segment(name, np.asarray(q), self._snapshot_objects(), dict(self.closed), candidate)
        )

    def _event(self, stage: str, **details: Any) -> None:
        self.run.events.append({"stage": stage, "success": True, **details})
        print(f"[runtime] complete: {stage}", flush=True)

    def _hold(self, name: str, multiplier: int = 1) -> None:
        frames = int(self.cfg["render"]["hold_frames"]) * multiplier
        self._record(name, np.repeat(self.q[None, :], frames, axis=0))

    def _finger_motion(self, side: str, target: float, name: str) -> None:
        frames = int(self.cfg["render"]["hold_frames"])
        start = self.closed[side]
        for fraction in np.linspace(start, target, frames + 1)[1:]:
            self.closed[side] = float(fraction)
            self._record(name, self.q[None, :])

    def _validate_task_state(self, stage_index: int) -> None:
        """Assert runtime payload transitions match the compiled task exactly."""

        def members(side: str) -> list[str]:
            payload = self.payloads[side]
            return [] if payload is None else sorted(payload.members)

        actual = {
            "loose_parts": sorted(self.loose),
            "hand_payloads": {
                "holder": members("left"),
                "worker": members("right"),
            },
            "placed_assembly": sorted(self.placed),
        }
        compiled = self.compiled_stages[stage_index]
        expected = compiled.after.to_dict()
        if actual != expected:
            raise RuntimeError(
                f"Runtime state after {compiled.step_id} differs from task compiler: "
                f"actual={actual}, expected={expected}"
            )
        self._event(
            f"{compiled.step_id}: task state verified",
            task_step=compiled.step_id,
            task_state=actual,
        )

    def _accept(self, name: str, motion: Any, candidate: str | None = None) -> bool:
        if motion is None:
            return False
        self._record(name, motion.q_full, candidate)
        self.q = motion.q_full[-1].copy()
        self._event(name, frames=int(len(motion.q_full)), candidate=candidate)
        return True

    def _pick(
        self,
        part: str,
        side: str,
        excluded: set[str],
        required_candidate: GraspCandidate | None = None,
    ) -> GraspCandidate | None:
        before_pick = self._snapshot()
        batch_size = int(self.cfg["planner"]["candidate_goalset_size"])
        candidates = (
            [required_candidate]
            if required_candidate is not None
            else [item for item in self.pools[part] if item.candidate_id not in excluded]
        )
        object_pose = self.objects[part].world_T_object
        assert object_pose is not None
        chosen: GraspCandidate | None = None
        with self.backend.stage(
            side,
            self.q,
            self._scene(),
            self.payloads,
            max_goalset=batch_size,
            hand_closed=self.closed,
        ) as planner:
            for offset in range(0, len(candidates), batch_size):
                batch = candidates[offset : offset + batch_size]
                print(
                    f"[runtime] {part}: native plan_grasp candidates "
                    f"{offset + 1}-{offset + len(batch)}/{len(candidates)}",
                    flush=True,
                )
                selection = planner.plan_grasp(
                    world_grasps(object_pose, batch),
                    approach_offset_m=float(self.cfg["motion"]["pick_approach_local_z_m"]),
                    contact_links=self.contact_links[side],
                )
                if selection is None:
                    continue
                chosen = batch[selection.selected_index]
                approach = planner.grasp_submotion(selection.result, "approach")
                contact = planner.grasp_submotion(selection.result, "grasp")
                if approach is None or contact is None:
                    continue
                self._accept(f"{part}: move to pregrasp", approach, chosen.candidate_id)
                self._accept(f"{part}: constrained grasp approach", contact, chosen.candidate_id)
                print(f"[runtime] selected {part}: {chosen.candidate_id}", flush=True)
                break

        if chosen is None:
            return None

        self._finger_motion(side, 1.0, f"{part}: close {side} Dex3")
        grasp_T_object = tra.inverse_matrix(chosen.object_T_G)
        self.grasps[part] = chosen
        self.loose.remove(part)
        self.objects[part] = ObjectState(hand=side, grasp_T_object=grasp_T_object)
        self.payloads[side] = PayloadGeometry({part: grasp_T_object})
        contact_world_T_G = object_pose @ chosen.object_T_G
        retract = contact_world_T_G.copy()
        retract[2, 3] += float(self.cfg["motion"]["pick_retract_world_z_m"])
        with self.backend.stage(
            side,
            self.q,
            self._scene(),
            self.payloads,
            max_goalset=1,
            hand_closed=self.closed,
        ) as planner:
            motion = planner.plan_linear(
                retract,
                axis="z",
                in_tool_frame=False,
                # The newly attached object begins in intentional support
                # contact. cuRobo has no pairwise ACM, so only the table is
                # disabled for this constrained upward separation.
                disable_obstacles=["table"],
            )
            if not self._accept(f"{part}: payload-aware vertical retract", motion):
                excluded.add(chosen.candidate_id)
                self._restore(before_pick)
                if required_candidate is not None:
                    return None
                return self._pick(part, side, excluded)
        self.run.selected[part] = {
            "hand": side,
            "candidate_id": chosen.candidate_id,
            "family_id": chosen.family_id,
            "graspgenx_score": chosen.score,
            "object_T_G": chosen.object_T_G.tolist(),
            "qualification": "Isaac/PhysX VIRAL-profile PASS + cuRobo plan_grasp",
        }
        self._hold(f"{part}: retained")
        return chosen

    def _root_grasp_pose(self, world_T_root: np.ndarray) -> np.ndarray:
        return world_T_root @ self.grasps[self.task.root_part].object_T_G

    def _child_grasp_pose(self, child: str, world_T_root: np.ndarray) -> np.ndarray:
        return (
            world_T_root
            @ self.task.member_transforms()[child]
            @ self.grasps[child].object_T_G
        )

    def _move_side(self, side: str, target: np.ndarray, name: str) -> bool:
        with self.backend.stage(
            side,
            self.q,
            self._scene(),
            self.payloads,
            max_goalset=1,
            hand_closed=self.closed,
        ) as planner:
            return self._accept(name, planner.plan_pose(target))

    def _move_side_cspace(self, side: str, target_q: np.ndarray, name: str) -> bool:
        with self.backend.stage(
            side,
            self.q,
            self._scene(),
            self.payloads,
            max_goalset=1,
            hand_closed=self.closed,
        ) as planner:
            return self._accept(name, planner.plan_cspace(target_q))

    def _mate(
        self,
        child: str,
        connection: str,
        required_work_pose: tuple[str, np.ndarray] | None = None,
        required_precontact_q: np.ndarray | None = None,
    ) -> bool:
        workspace_key = "u_stage" if child == "u_legs" else "head_stage"
        samples = (
            [required_work_pose]
            if required_work_pose is not None
            else workspace_samples(self.cfg["workspace"][workspace_key])
        )
        offset = float(self.cfg["motion"]["mate_precontact_world_z_m"][child])
        selected: tuple[str, np.ndarray, np.ndarray, np.ndarray] | None = None

        if required_precontact_q is not None and required_work_pose is not None:
            sample_id, world_T_root = required_work_pose
            left_target = self._root_grasp_pose(world_T_root)
            contact = self._child_grasp_pose(child, world_T_root)
            precontact = contact.copy()
            precontact[2, 3] += offset
            selected = (sample_id, world_T_root, left_target, precontact)
        else:
            # Each candidate is a singleton coupled problem. See the live spec:
            # the goal-set dimension cannot preserve left/right pair identity.
            with self.backend.coupled(
                self.q, self._scene(), self.payloads, hand_closed=self.closed
            ) as coupled:
                for sample_id, world_T_root in samples:
                    left_target = self._root_grasp_pose(world_T_root)
                    contact = self._child_grasp_pose(child, world_T_root)
                    precontact = contact.copy()
                    precontact[2, 3] += offset
                    if coupled.plan_pair(left_target, precontact) is not None:
                        selected = (sample_id, world_T_root, left_target, precontact)
                        break
        if selected is None:
            return False
        sample_id, world_T_root, left_target, precontact = selected
        self._event(
            f"{connection}: coupled mode qualified",
            work_pose_id=sample_id,
            world_T_root=world_T_root.tolist(),
            coupled_goalset_size=1,
        )

        holder_move = False
        if required_precontact_q is not None:
            holder_move = self._move_side_cspace(
                "left",
                required_precontact_q,
                f"{connection}: move holder to qualified configuration",
            )
            if not holder_move:
                self._event(
                    f"{connection}: holder endpoint fallback selected",
                    reason="paired IK joint target has no path from realized prior state",
                    invariant="world grasp-frame precontact target unchanged",
                )
                holder_move = self._move_side(
                    "left",
                    left_target,
                    f"{connection}: move holder to qualified pose via alternate IK",
                )
        else:
            holder_move = self._move_side(
                "left", left_target, f"{connection}: move holder to work pose"
            )
        if not holder_move:
            return False
        worker_move = False
        if required_precontact_q is not None:
            worker_move = self._move_side_cspace(
                "right",
                required_precontact_q,
                f"{connection}: move child to qualified precontact",
            )
            if not worker_move:
                self._event(
                    f"{connection}: worker endpoint fallback selected",
                    reason="paired IK joint target has no path from realized prior state",
                    invariant="world grasp-frame precontact target unchanged",
                )
                worker_move = self._move_side(
                    "right",
                    precontact,
                    f"{connection}: move child to precontact via alternate IK",
                )
        else:
            worker_move = self._move_side(
                "right", precontact, f"{connection}: move child to precontact"
            )
        if not worker_move:
            return False
        contact = self._child_grasp_pose(child, world_T_root)
        with self.backend.stage(
            "right",
            self.q,
            self._scene(),
            self.payloads,
            max_goalset=1,
            hand_closed=self.closed,
        ) as planner:
            motion = planner.plan_linear(
                contact,
                axis="z",
                in_tool_frame=False,
                disable_links=[SLOT["left"]],
            )
            if not self._accept(f"{connection}: constrained connector approach", motion):
                return False
        self._hold(f"{connection}: connector aligned")
        self._finger_motion("right", 0.0, f"{connection}: release worker")

        root_grasp_T_root = tra.inverse_matrix(self.grasps[self.task.root_part].object_T_G)
        member_transforms = self.task.member_transforms()
        prior_members = set(self.payloads["left"].members)
        combined_members = prior_members | {child}
        holder_members = {
            part: root_grasp_T_root @ member_transforms[part]
            for part in sorted(combined_members)
        }
        self.payloads["right"] = None
        self.payloads["left"] = PayloadGeometry(holder_members)
        for part, grasp_T_part in holder_members.items():
            self.objects[part] = ObjectState(hand="left", grasp_T_object=grasp_T_part)
        self._event(
            f"{connection}: attachment transfer",
            transition="right child payload -> left composite payload",
            members=sorted(holder_members),
        )

        with self.backend.stage(
            "right",
            self.q,
            self._scene(),
            self.payloads,
            max_goalset=1,
            hand_closed=self.closed,
        ) as planner:
            retreat = planner.plan_linear(
                precontact,
                axis="z",
                in_tool_frame=False,
                disable_links=[SLOT["left"]],
            )
            if not self._accept(f"{connection}: constrained worker retreat", retreat):
                return False
        self._hold(f"{connection}: composite retained")
        return True

    def _qualified_pick_representatives(
        self, part: str, side: str, maximum: int
    ) -> list[GraspCandidate]:
        """Return cuRobo-reachable representatives without changing task state."""

        batch_size = int(self.cfg["planner"]["candidate_goalset_size"])
        candidates = self.pools[part]
        object_pose = self.observation.world_T_objects[part]
        empty_payloads = {"left": None, "right": None}
        selected: list[GraspCandidate] = []
        initial_scene = self.backend.scene(
            self.observation.table.center,
            self.observation.table.dimensions,
            self.observation.world_T_objects,
        )
        with self.backend.stage(
            side,
            self.initial_q,
            initial_scene,
            empty_payloads,
            max_goalset=batch_size,
            hand_closed={"left": 0.0, "right": 0.0},
        ) as planner:
            for offset in range(0, len(candidates), batch_size):
                batch = candidates[offset : offset + batch_size]
                result = planner.plan_grasp(
                    world_grasps(object_pose, batch),
                    approach_offset_m=float(self.cfg["motion"]["pick_approach_local_z_m"]),
                    contact_links=self.contact_links[side],
                )
                if result is not None:
                    candidate = batch[result.selected_index]
                    if candidate.candidate_id not in {item.candidate_id for item in selected}:
                        selected.append(candidate)
                        print(
                            f"[runtime] pickup-qualified {part}: {candidate.candidate_id}",
                            flush=True,
                        )
                        if len(selected) >= maximum:
                            break
        return selected

    def _pair_payloads(
        self,
        t_grasp: GraspCandidate,
        child: str,
        child_grasp: GraspCandidate,
    ) -> dict[str, PayloadGeometry | None]:
        grasp_T_root = tra.inverse_matrix(t_grasp.object_T_G)
        root_members = {"t_body"}
        if child == "cube_head":
            root_members.add("u_legs")
        members = {
            part: grasp_T_root @ self.task.member_transforms()[part]
            for part in sorted(root_members)
        }
        return {
            "left": PayloadGeometry(members),
            "right": PayloadGeometry({child: tra.inverse_matrix(child_grasp.object_T_G)}),
        }

    def _qualify_pair_mode(
        self,
        coupled: Any,
        t_grasp: GraspCandidate,
        child: str,
        child_grasp: GraspCandidate,
    ) -> tuple[str, np.ndarray, np.ndarray] | None:
        key = "u_stage" if child == "u_legs" else "head_stage"
        offset = float(self.cfg["motion"]["mate_precontact_world_z_m"][child])
        coupled.set_payloads(self._pair_payloads(t_grasp, child, child_grasp))
        for sample_id, world_T_root in workspace_samples(self.cfg["workspace"][key]):
            left_target = world_T_root @ t_grasp.object_T_G
            contact = (
                world_T_root
                @ self.task.member_transforms()[child]
                @ child_grasp.object_T_G
            )
            precontact = contact.copy()
            precontact[2, 3] += offset
            precontact_q = coupled.solve_pair_ik(left_target, precontact)
            if precontact_q is None:
                continue
            # Qualify the actual locked-holder linear approach from the exact
            # collision-free paired IK configuration, not merely its endpoint.
            with self.backend.stage(
                "right",
                precontact_q,
                coupled.backend.scene(
                    self.observation.table.center,
                    self.observation.table.dimensions,
                    {},
                ),
                self._pair_payloads(t_grasp, child, child_grasp),
                max_goalset=1,
                hand_closed={"left": 1.0, "right": 1.0},
            ) as approach:
                if approach.plan_linear(
                    contact,
                    axis="z",
                    in_tool_frame=False,
                    disable_links=[SLOT["left"]],
                    max_attempts=4,
                ) is not None:
                    return sample_id, world_T_root, precontact_q
        return None

    def qualify_modes(self) -> list[AssemblyMode]:
        """Find root/worker grasp combinations that support both connectors."""

        robot_cfg = self.cfg["robot"]
        cache_inputs = [
            self.config_path,
            self.observation_path,
            self.task_path,
            Path(__file__).with_name("curobo_backend.py").resolve(),
            Path(__file__).with_name("grasp_goalset.py").resolve(),
            Path(__file__).with_name("workspace.py").resolve(),
            project_path(robot_cfg["model"]),
            project_path(robot_cfg["urdf"]),
            *(project_path(path) for path in robot_cfg["hand_profiles"].values()),
        ]
        for part, hand in (("t_body", "left"), ("u_legs", "right"), ("cube_head", "right")):
            specification = self.task.parts[part]
            cache_inputs.extend(
                [
                    project_path(specification.grasp_pools[hand]),
                    project_path(specification.mesh),
                    project_path(specification.geometry_config),
                ]
            )
        digest = hashlib.sha256()
        digest.update(MODE_CACHE_CONTRACT)
        for path in cache_inputs:
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
        cache_path = ROOT / "artifacts/runtime_assembly/mode_cache" / f"{digest.hexdigest()}.json"
        if cache_path.is_file():
            document = json.loads(cache_path.read_text())
            by_part = {
                part: {candidate.candidate_id: candidate for candidate in candidates}
                for part, candidates in self.pools.items()
            }
            modes = [
                AssemblyMode(
                    by_part["t_body"][item["t_grasp"]],
                    by_part["u_legs"][item["u_grasp"]],
                    by_part["cube_head"][item["cube_grasp"]],
                    item["u_work_id"],
                    np.asarray(item["u_world_T_root"], dtype=np.float64),
                    np.asarray(item["u_precontact_q"], dtype=np.float64),
                    item["head_work_id"],
                    np.asarray(item["head_world_T_root"], dtype=np.float64),
                    np.asarray(item["head_precontact_q"], dtype=np.float64),
                )
                for item in document["modes"]
            ]
            print(f"[runtime] loaded {len(modes)} qualified modes from {cache_path}", flush=True)
            return modes

        limits = self.cfg["search"]["pickup_representatives"]
        t_candidates = self._qualified_pick_representatives(
            "t_body", "left", int(limits["t_body"])
        )
        u_candidates = self._qualified_pick_representatives(
            "u_legs", "right", int(limits["u_legs"])
        )
        cube_candidates = self._qualified_pick_representatives(
            "cube_head", "right", int(limits["cube_head"])
        )
        if not t_candidates or not u_candidates or not cube_candidates:
            raise RuntimeError(
                "Mode qualification found no runtime-pickable representative for "
                f"T={len(t_candidates)}, U={len(u_candidates)}, cube={len(cube_candidates)}"
            )

        empty_scene = self.backend.scene(
            self.observation.table.center, self.observation.table.dimensions, {}
        )
        maximum_modes = int(self.cfg["search"]["maximum_qualified_modes"])
        modes: list[AssemblyMode] = []
        with self.backend.coupled(
            self.initial_q,
            empty_scene,
            {"left": None, "right": None},
            hand_closed={"left": 1.0, "right": 1.0},
        ) as coupled:
            for t_grasp in t_candidates:
                head_modes = []
                for cube_grasp in cube_candidates:
                    head = self._qualify_pair_mode(
                        coupled, t_grasp, "cube_head", cube_grasp
                    )
                    if head is not None:
                        head_modes.append((cube_grasp, head))
                        break
                if not head_modes:
                    continue
                u_modes = []
                for u_grasp in u_candidates:
                    u_mode = self._qualify_pair_mode(
                        coupled, t_grasp, "u_legs", u_grasp
                    )
                    if u_mode is not None:
                        u_modes.append((u_grasp, u_mode))
                        break
                if not u_modes:
                    continue
                cube_grasp, (head_id, head_pose, head_q) = head_modes[0]
                u_grasp, (u_id, u_pose, u_q) = u_modes[0]
                chosen = AssemblyMode(
                    t_grasp,
                    u_grasp,
                    cube_grasp,
                    u_id,
                    u_pose.copy(),
                    u_q.copy(),
                    head_id,
                    head_pose.copy(),
                    head_q.copy(),
                )
                modes.append(chosen)
                print(
                    "[runtime] diverse complete mode qualified: "
                    f"T={chosen.t_grasp.candidate_id}, "
                    f"U={chosen.u_grasp.candidate_id}, "
                    f"cube={chosen.cube_grasp.candidate_id}",
                    flush=True,
                )
                if len(modes) >= maximum_modes:
                    self._write_mode_cache(cache_path, digest.hexdigest(), modes)
                    return modes
        if modes:
            self._write_mode_cache(cache_path, digest.hexdigest(), modes)
        return modes

    @staticmethod
    def _write_mode_cache(
        cache_path: Path, input_sha256: str, modes: Sequence[AssemblyMode]
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "input_sha256": input_sha256,
                    "modes": [
                        {
                            "t_grasp": mode.t_grasp.candidate_id,
                            "u_grasp": mode.u_grasp.candidate_id,
                            "cube_grasp": mode.cube_grasp.candidate_id,
                            "u_work_id": mode.u_work_id,
                            "u_world_T_root": mode.u_world_T_root.tolist(),
                            "u_precontact_q": mode.u_precontact_q.tolist(),
                            "head_work_id": mode.head_work_id,
                            "head_world_T_root": mode.head_world_T_root.tolist(),
                            "head_precontact_q": mode.head_precontact_q.tolist(),
                        }
                        for mode in modes
                    ],
                },
                indent=2,
            )
            + "\n"
        )

    def _assembly_min_z(self) -> float:
        bounds = []
        transforms = self.task.member_transforms()
        for part, spec in self.task.parts.items():
            mesh = trimesh.load(spec.mesh, force="mesh", process=False)
            vertices = trimesh.transform_points(mesh.vertices, transforms[part])
            bounds.append(float(vertices[:, 2].min()))
        return min(bounds)

    def _place(self) -> bool:
        samples = placement_samples(
            self.cfg["workspace"]["placement"],
            self.observation.table.top_z,
            self._assembly_min_z(),
        )
        selected: tuple[str, np.ndarray, np.ndarray, Any] | None = None
        with self.backend.stage(
            "left",
            self.q,
            self._scene(),
            self.payloads,
            max_goalset=1,
            hand_closed=self.closed,
        ) as planner:
            current_tool = planner.current_tool_world()
            print(
                "[runtime] placement start: "
                f"left_tool_xyz={current_tool[:3, 3].tolist()}",
                flush=True,
            )
            current_world_T_root = current_tool @ tra.inverse_matrix(
                self.grasps[self.task.root_part].object_T_G
            )
            for sample_id, world_T_root in samples:
                print(f"[runtime] place candidate: {sample_id}", flush=True)
                contact_target = self._root_grasp_pose(world_T_root)
                precontact_root = world_T_root.copy()
                precontact_root[2, 3] = max(
                    float(current_world_T_root[2, 3]),
                    float(world_T_root[2, 3])
                    + float(self.cfg["motion"]["placement_minimum_precontact_clearance_m"]),
                )
                precontact_target = self._root_grasp_pose(precontact_root)
                motion = planner.plan_pose(precontact_target)
                if motion is not None:
                    selected = (sample_id, world_T_root, contact_target, motion)
                    break
        if selected is not None:
            sample_id, world_T_root, target, motion = selected
            self._accept("place: payload-aware transfer to precontact", motion)
            with self.backend.stage(
                "left",
                self.q,
                self._scene(),
                self.payloads,
                max_goalset=1,
                hand_closed=self.closed,
            ) as planner:
                descent = planner.plan_linear(
                    target,
                    axis="z",
                    in_tool_frame=False,
                    disable_obstacles=["table"],
                )
                if not self._accept("place: constrained support descent", descent):
                    return False
            self._event(
                "place: selected support pose",
                placement_pose_id=sample_id,
                world_T_root=world_T_root.tolist(),
            )
            self._hold("place: complete assembly above table")
            self._finger_motion("left", 0.0, "place: release complete assembly")
            for part, root_T_part in self.task.member_transforms().items():
                self.objects[part] = ObjectState(
                    world_T_object=world_T_root @ root_T_part
                )
            self.placed = set(self.task.parts)
            self.payloads["left"] = None
            retreat = target.copy()
            retreat[2, 3] += float(self.cfg["motion"]["final_retreat_world_z_m"])
            with self.backend.stage(
                "left",
                self.q,
                self._scene(),
                self.payloads,
                max_goalset=1,
                hand_closed=self.closed,
            ) as planner:
                motion = planner.plan_linear(
                    retreat,
                    axis="z",
                    in_tool_frame=False,
                    # The new world copies begin in intentional contact with
                    # the just-opened holder. Hide only those copies for this
                    # bounded separating motion; table and robot collisions
                    # remain live and every obstacle is restored on return.
                    disable_obstacles=self.backend.part_obstacle_names(
                        sorted(self.task.parts)
                    ),
                )
                if not self._accept("place: empty holder retreat", motion):
                    return False
            self._hold("final reveal", multiplier=2)
            return True
        return False

    def _reset_attempt(self) -> None:
        self.q = self.initial_q.copy()
        self.loose = set(self.task.parts)
        self.placed = set()
        self.objects = {
            name: ObjectState(world_T_object=matrix.copy())
            for name, matrix in self.observation.world_T_objects.items()
        }
        self.payloads = {"left": None, "right": None}
        self.closed = {"left": 0.0, "right": 0.0}
        self.grasps = {}
        self.run.segments.clear()
        self.run.events.clear()
        self.run.selected.clear()

    def execute(self) -> RuntimeAssemblyRun:
        modes = self.qualify_modes()
        if not modes:
            raise RuntimeError("No T/U/cube grasp combination supports both connector modes")
        last_reason = "no qualified mode was execution-feasible"
        for attempt, mode in enumerate(modes):
            self._reset_attempt()
            self._event(
                "qualified assembly mode selected",
                qualified_mode_attempt=attempt + 1,
                t_grasp=mode.t_grasp.candidate_id,
                u_grasp=mode.u_grasp.candidate_id,
                cube_grasp=mode.cube_grasp.candidate_id,
                u_work_pose=mode.u_work_id,
                head_work_pose=mode.head_work_id,
            )
            self._hold("start: observed loose parts")
            if self._pick("t_body", "left", set(), mode.t_grasp) is None:
                last_reason = "qualified T pickup failed during complete planning"
                continue
            self._validate_task_state(0)
            if self._pick("u_legs", "right", set(), mode.u_grasp) is None:
                last_reason = "qualified U pickup failed with held T"
                continue
            self._validate_task_state(1)
            if not self._mate(
                "u_legs",
                "u_to_t",
                (mode.u_work_id, mode.u_world_T_root),
                mode.u_precontact_q,
            ):
                last_reason = "qualified U mate failed during sequential planning"
                continue
            self._validate_task_state(2)
            if self._pick("cube_head", "right", set(), mode.cube_grasp) is None:
                last_reason = "qualified cube pickup failed with held T+U"
                continue
            self._validate_task_state(3)
            if not self._mate(
                "cube_head",
                "head_to_t",
                (mode.head_work_id, mode.head_world_T_root),
                mode.head_precontact_q,
            ):
                last_reason = "qualified head mate failed during sequential planning"
                continue
            self._validate_task_state(4)
            if not self._place():
                last_reason = "no complete-assembly placement mode"
                continue
            self._validate_task_state(5)
            self._event(
                "complete",
                qualified_mode_attempt=attempt + 1,
                observation_id=self.observation.observation_id,
            )
            return self.run
        raise RuntimeError(
            f"No complete assembly plan after {len(modes)} qualified modes: {last_reason}"
        )


def save_run(run: RuntimeAssemblyRun, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "planning_report.json").write_text(
        json.dumps(run.report(), indent=2) + "\n"
    )
    arrays = {
        f"segment_{index:03d}": segment.q
        for index, segment in enumerate(run.segments)
    }
    np.savez_compressed(directory / "arm_trajectories.npz", **arrays)
    render_state = {
        "schema_version": 2,
        "segments": [
            {
                "trajectory_key": f"segment_{index:03d}",
                "name": segment.name,
                "selected_candidate": segment.selected_candidate,
                "hand_closed": segment.hand_closed,
                "objects": {
                    name: {
                        "world_T_object": None if state.world_T_object is None else state.world_T_object.tolist(),
                        "hand": state.hand,
                        "grasp_T_object": None if state.grasp_T_object is None else state.grasp_T_object.tolist(),
                    }
                    for name, state in segment.objects.items()
                },
            }
            for index, segment in enumerate(run.segments)
        ],
    }
    (directory / "render_state.json").write_text(
        json.dumps(render_state, indent=2) + "\n"
    )
