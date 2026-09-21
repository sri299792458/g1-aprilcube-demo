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
MODE_CACHE_CONTRACT = b"connector-mode-v10-native-curobo-goalset-lazy-constraints"


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
    role_to_hand: dict[str, str] = field(default_factory=dict)
    planning_cost: float | None = None
    qualification: list[dict[str, Any]] = field(default_factory=list)
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
            "role_to_hand": self.role_to_hand,
            "planning_cost": self.planning_cost,
            "qualification": self.qualification,
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
    holder_hand: str
    worker_hand: str

    @property
    def role_to_hand(self) -> dict[str, str]:
        return {"holder": self.holder_hand, "worker": self.worker_hand}


@dataclass(frozen=True)
class RoleAssignment:
    """One complete mapping of semantic task roles to physical hands."""

    holder_hand: str
    worker_hand: str

    @property
    def role_to_hand(self) -> dict[str, str]:
        return {"holder": self.holder_hand, "worker": self.worker_hand}

    @property
    def assignment_id(self) -> str:
        return f"holder-{self.holder_hand}__worker-{self.worker_hand}"

    def ordered_pair_targets(
        self, holder_target: np.ndarray, worker_target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return targets in the fixed left/right order expected by cuRobo."""

        if self.holder_hand == "left":
            return holder_target, worker_target
        return worker_target, holder_target


@dataclass
class GoalsetPickupDomain:
    """Lazy discrete domain selected by native cuRobo goal-set planning."""

    part: str
    side: str
    chunks: list[list[GraspCandidate]]
    selected: list[GraspCandidate] = field(default_factory=list)
    round_index: int = 0

    @property
    def has_remaining(self) -> bool:
        return any(self.chunks)


def ready_role_assignments(task: Any) -> list[RoleAssignment]:
    """Return only assignments backed by a declared pool for every role pick."""

    ready = []
    for item in task.readiness_report()["assignments"]:
        if not item["ready"]:
            continue
        mapping = item["role_to_hand"]
        ready.append(RoleAssignment(mapping["holder"], mapping["worker"]))
    return ready


def candidates_by_family(
    candidates: Sequence[GraspCandidate],
) -> list[tuple[str, list[GraspCandidate]]]:
    """Partition a pool without losing its deterministic family order."""

    grouped: dict[str, list[GraspCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.family_id, []).append(candidate)
    return list(grouped.items())


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

        self.assignments = ready_role_assignments(self.task)
        if not self.assignments:
            missing = self.task.readiness_report()["assignments"]
            raise ValueError(
                "No holder/worker assignment has a grasp pool for every required pick: "
                f"{missing}"
            )
        self.role_to_hand = self.assignments[0].role_to_hand

        self.pools: dict[tuple[str, str], tuple[GraspCandidate, ...]] = {}
        for part, specification in self.task.parts.items():
            for hand, pool_path in specification.grasp_pools.items():
                key = (part, hand)
                self.pools[key] = load_grasp_pool(project_path(pool_path))

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
        self._pair_mode_cache: dict[
            tuple[str, str, str, str, str], tuple[str, np.ndarray, np.ndarray] | None
        ] = {}
        self._mode_search_domains: dict[
            str, dict[str, GoalsetPickupDomain]
        ] = {}
        self._constraint_cache_digest_value: str | None = None

    def _constraint_cache_digest(self) -> str:
        """Hash every input that can change an exact cuRobo verdict."""

        if self._constraint_cache_digest_value is not None:
            return self._constraint_cache_digest_value
        robot_cfg = self.cfg["robot"]
        inputs = [
            self.config_path,
            self.observation_path,
            self.task_path,
            Path(__file__).resolve(),
            Path(__file__).with_name("curobo_backend.py").resolve(),
            Path(__file__).with_name("grasp_goalset.py").resolve(),
            Path(__file__).with_name("workspace.py").resolve(),
            project_path(robot_cfg["model"]),
            project_path(robot_cfg["urdf"]),
            *(project_path(path) for path in robot_cfg["hand_profiles"].values()),
        ]
        for specification in self.task.parts.values():
            inputs.extend(
                [
                    project_path(specification.mesh),
                    project_path(specification.geometry_config),
                    *(
                        project_path(pool)
                        for pool in specification.grasp_pools.values()
                    ),
                ]
            )
        digest = hashlib.sha256()
        digest.update(MODE_CACHE_CONTRACT)
        for path in sorted(set(inputs)):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
        self._constraint_cache_digest_value = digest.hexdigest()
        return self._constraint_cache_digest_value

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
                role: members(self.role_to_hand[role]) for role in self.task.roles
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
        goalset_capacity = int(self.cfg["planner"]["candidate_goalset_size"])
        candidates = (
            [required_candidate]
            if required_candidate is not None
            else [
                item
                for item in self.pools[(part, side)]
                if item.candidate_id not in excluded
            ]
        )
        object_pose = self.objects[part].world_T_object
        assert object_pose is not None
        chosen: GraspCandidate | None = None
        with self.backend.stage(
            side,
            self.q,
            self._scene(),
            self.payloads,
            max_goalset=goalset_capacity,
            hand_closed=self.closed,
        ) as planner:
            for offset in range(0, len(candidates), goalset_capacity):
                goalset = candidates[offset : offset + goalset_capacity]
                print(
                    f"[runtime] {part}: native plan_grasp candidates "
                    f"{offset + 1}-{offset + len(goalset)}/{len(candidates)}",
                    flush=True,
                )
                selection = planner.plan_grasp(
                    world_grasps(object_pose, goalset),
                    approach_offset_m=float(self.cfg["motion"]["pick_approach_local_z_m"]),
                    contact_links=self.contact_links[side],
                )
                if selection is None:
                    continue
                chosen = goalset[selection.selected_index]
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
        holder = self.role_to_hand["holder"]
        worker = self.role_to_hand["worker"]
        assignment = RoleAssignment(holder, worker)
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
            holder_target = self._root_grasp_pose(world_T_root)
            contact = self._child_grasp_pose(child, world_T_root)
            precontact = contact.copy()
            precontact[2, 3] += offset
            selected = (sample_id, world_T_root, holder_target, precontact)
        else:
            # Each candidate is a singleton coupled problem. See the live spec:
            # the goal-set dimension cannot preserve left/right pair identity.
            with self.backend.coupled(
                self.q, self._scene(), self.payloads, hand_closed=self.closed
            ) as coupled:
                for sample_id, world_T_root in samples:
                    holder_target = self._root_grasp_pose(world_T_root)
                    contact = self._child_grasp_pose(child, world_T_root)
                    precontact = contact.copy()
                    precontact[2, 3] += offset
                    left_target, right_target = assignment.ordered_pair_targets(
                        holder_target, precontact
                    )
                    if coupled.plan_pair(left_target, right_target) is not None:
                        selected = (
                            sample_id,
                            world_T_root,
                            holder_target,
                            precontact,
                        )
                        break
        if selected is None:
            return False
        sample_id, world_T_root, holder_target, precontact = selected
        self._event(
            f"{connection}: coupled mode qualified",
            work_pose_id=sample_id,
            world_T_root=world_T_root.tolist(),
            coupled_goalset_size=1,
        )

        holder_move = False
        if required_precontact_q is not None:
            holder_move = self._move_side_cspace(
                holder,
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
                    holder,
                    holder_target,
                    f"{connection}: move holder to qualified pose via alternate IK",
                )
        else:
            holder_move = self._move_side(
                holder, holder_target, f"{connection}: move holder to work pose"
            )
        if not holder_move:
            return False
        worker_move = False
        if required_precontact_q is not None:
            worker_move = self._move_side_cspace(
                worker,
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
                    worker,
                    precontact,
                    f"{connection}: move child to precontact via alternate IK",
                )
        else:
            worker_move = self._move_side(
                worker, precontact, f"{connection}: move child to precontact"
            )
        if not worker_move:
            return False
        contact = self._child_grasp_pose(child, world_T_root)
        with self.backend.stage(
            worker,
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
                disable_links=[SLOT[holder]],
            )
            if not self._accept(f"{connection}: constrained connector approach", motion):
                return False
        self._hold(f"{connection}: connector aligned")
        self._finger_motion(worker, 0.0, f"{connection}: release worker")

        root_grasp_T_root = tra.inverse_matrix(self.grasps[self.task.root_part].object_T_G)
        member_transforms = self.task.member_transforms()
        prior_payload = self.payloads[holder]
        assert prior_payload is not None
        prior_members = set(prior_payload.members)
        combined_members = prior_members | {child}
        holder_members = {
            part: root_grasp_T_root @ member_transforms[part]
            for part in sorted(combined_members)
        }
        self.payloads[worker] = None
        self.payloads[holder] = PayloadGeometry(holder_members)
        for part, grasp_T_part in holder_members.items():
            self.objects[part] = ObjectState(
                hand=holder, grasp_T_object=grasp_T_part
            )
        self._event(
            f"{connection}: attachment transfer",
            transition=f"{worker} child payload -> {holder} composite payload",
            members=sorted(holder_members),
        )

        with self.backend.stage(
            worker,
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
                disable_links=[SLOT[holder]],
            )
            if not self._accept(f"{connection}: constrained worker retreat", retreat):
                return False
        self._hold(f"{connection}: composite retained")
        return True

    def _new_pickup_domain(self, part: str, side: str) -> GoalsetPickupDomain:
        """Partition one atlas only by upstream goal-set buffer capacity."""

        capacity = int(self.cfg["planner"]["candidate_goalset_size"])
        candidates = self.pools[(part, side)]
        chunks = [
            list(candidates[offset : offset + capacity])
            for offset in range(0, len(candidates), capacity)
        ]
        return GoalsetPickupDomain(part, side, chunks)

    def _expand_pickup_domain(
        self, domain: GoalsetPickupDomain
    ) -> list[GraspCandidate]:
        """Ask cuRobo for one additional feasible grasp from every goal set.

        A selected grasp is removed from its chunk. If downstream task
        constraints reject it, a later round asks cuRobo to choose another
        remaining member. A no-solution result exhausts that whole goal set.
        """

        if not domain.has_remaining:
            return []
        object_pose = self.observation.world_T_objects[domain.part]
        scene = self.backend.scene(
            self.observation.table.center,
            self.observation.table.dimensions,
            self.observation.world_T_objects,
        )
        added: list[GraspCandidate] = []
        requests: list[dict[str, Any]] = []
        capacity = int(self.cfg["planner"]["candidate_goalset_size"])
        with self.backend.stage(
            domain.side,
            self.initial_q,
            scene,
            {"left": None, "right": None},
            max_goalset=capacity,
            hand_closed={"left": 0.0, "right": 0.0},
        ) as planner:
            for chunk_index, remaining in enumerate(domain.chunks):
                if not remaining:
                    continue
                candidate_count = len(remaining)
                result = planner.plan_grasp(
                    world_grasps(object_pose, remaining),
                    approach_offset_m=float(
                        self.cfg["motion"]["pick_approach_local_z_m"]
                    ),
                    contact_links=self.contact_links[domain.side],
                )
                if result is None:
                    requests.append(
                        {
                            "chunk_index": chunk_index,
                            "goalset_size": candidate_count,
                            "selected_candidate_id": None,
                            "status": "no_curobo_solution",
                        }
                    )
                    remaining.clear()
                    continue
                selected = remaining.pop(result.selected_index)
                domain.selected.append(selected)
                added.append(selected)
                requests.append(
                    {
                        "chunk_index": chunk_index,
                        "goalset_size": candidate_count,
                        "selected_candidate_id": selected.candidate_id,
                        "selected_goalset_index": result.selected_index,
                        "status": "selected",
                    }
                )
        self.run.qualification.append(
            {
                "gate": "native-curobo-lazy-goalset-pickup",
                "part": domain.part,
                "hand": domain.side,
                "round_index": domain.round_index,
                "atlas_candidate_count": len(self.pools[(domain.part, domain.side)]),
                "goalset_capacity": capacity,
                "selected_this_round": len(added),
                "selected_total": len(domain.selected),
                "remaining_candidate_count": sum(map(len, domain.chunks)),
                "requests": requests,
            }
        )
        print(
            f"[runtime] {domain.part}/{domain.side} cuRobo goal-set round "
            f"{domain.round_index}: selected {len(added)}, "
            f"total {len(domain.selected)}",
            flush=True,
        )
        domain.round_index += 1
        return added

    def _pair_payloads(
        self,
        assignment: RoleAssignment,
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
            assignment.holder_hand: PayloadGeometry(members),
            assignment.worker_hand: PayloadGeometry(
                {child: tra.inverse_matrix(child_grasp.object_T_G)}
            ),
        }

    def _qualify_pair_mode(
        self,
        coupled: Any,
        assignment: RoleAssignment,
        t_grasp: GraspCandidate,
        child: str,
        child_grasp: GraspCandidate,
    ) -> tuple[str, np.ndarray, np.ndarray] | None:
        key = "u_stage" if child == "u_legs" else "head_stage"
        offset = float(self.cfg["motion"]["mate_precontact_world_z_m"][child])
        coupled.set_payloads(
            self._pair_payloads(assignment, t_grasp, child, child_grasp)
        )
        for sample_id, world_T_root in workspace_samples(self.cfg["workspace"][key]):
            holder_target = world_T_root @ t_grasp.object_T_G
            contact = (
                world_T_root
                @ self.task.member_transforms()[child]
                @ child_grasp.object_T_G
            )
            precontact = contact.copy()
            precontact[2, 3] += offset
            left_target, right_target = assignment.ordered_pair_targets(
                holder_target, precontact
            )
            precontact_q = coupled.solve_pair_ik(left_target, right_target)
            if precontact_q is None:
                continue
            # Qualify the actual locked-holder linear approach from the exact
            # collision-free paired IK configuration, not merely its endpoint.
            with self.backend.stage(
                assignment.worker_hand,
                precontact_q,
                coupled.backend.scene(
                    self.observation.table.center,
                    self.observation.table.dimensions,
                    {},
                ),
                self._pair_payloads(assignment, t_grasp, child, child_grasp),
                max_goalset=1,
                hand_closed={"left": 1.0, "right": 1.0},
            ) as approach:
                if approach.plan_linear(
                    contact,
                    axis="z",
                    in_tool_frame=False,
                    disable_links=[SLOT[assignment.holder_hand]],
                    max_attempts=4,
                ) is not None:
                    return sample_id, world_T_root, precontact_q
        return None

    def _cached_pair_mode(
        self,
        coupled: Any,
        assignment: RoleAssignment,
        t_grasp: GraspCandidate,
        child: str,
        child_grasp: GraspCandidate,
    ) -> tuple[str, np.ndarray, np.ndarray] | None:
        """Memoize exact cuRobo pair constraints without approximating them."""

        key = (
            assignment.holder_hand,
            assignment.worker_hand,
            t_grasp.candidate_id,
            child,
            child_grasp.candidate_id,
        )
        if key not in self._pair_mode_cache:
            self._pair_mode_cache[key] = self._qualify_pair_mode(
                coupled, assignment, t_grasp, child, child_grasp
            )
        return self._pair_mode_cache[key]

    def _find_selected_mode(
        self,
        assignment: RoleAssignment,
        domains: Mapping[str, GoalsetPickupDomain],
        empty_scene: Mapping[str, Any],
        rejected_t: set[str],
        rejected_tu: set[tuple[str, str]],
        rejected_modes: set[tuple[str, str, str, str, str]],
    ) -> AssemblyMode | None:
        """Return the first exact assembly-compatible selected grasp triple.

        The candidates in ``domains[*].selected`` were chosen by cuRobo from
        disjoint goal sets. This method adds the task's exact downstream
        constraints: paired arm IK, payload geometry, and the locked-holder
        linear connector approach. Pair verdicts are memoized, so a later
        lazy expansion evaluates only genuinely new combinations.
        """

        with self.backend.coupled(
            self.initial_q,
            empty_scene,
            {"left": None, "right": None},
            hand_closed={"left": 1.0, "right": 1.0},
        ) as coupled:
            for t_grasp in domains["t_body"].selected:
                if t_grasp.candidate_id in rejected_t:
                    continue
                for u_grasp in domains["u_legs"].selected:
                    if (
                        t_grasp.candidate_id,
                        u_grasp.candidate_id,
                    ) in rejected_tu:
                        continue
                    u_mode = self._cached_pair_mode(
                        coupled,
                        assignment,
                        t_grasp,
                        "u_legs",
                        u_grasp,
                    )
                    if u_mode is None:
                        continue
                    for cube_grasp in domains["cube_head"].selected:
                        key = self._mode_key(
                            assignment, t_grasp, u_grasp, cube_grasp
                        )
                        if key in rejected_modes:
                            continue
                        head_mode = self._cached_pair_mode(
                            coupled,
                            assignment,
                            t_grasp,
                            "cube_head",
                            cube_grasp,
                        )
                        if head_mode is None:
                            continue

                        u_id, u_pose, u_q = u_mode
                        head_id, head_pose, head_q = head_mode
                        return AssemblyMode(
                            t_grasp,
                            u_grasp,
                            cube_grasp,
                            u_id,
                            u_pose.copy(),
                            u_q.copy(),
                            head_id,
                            head_pose.copy(),
                            head_q.copy(),
                            assignment.holder_hand,
                            assignment.worker_hand,
                        )
        return None

    @staticmethod
    def _mode_key(
        assignment: RoleAssignment,
        t_grasp: GraspCandidate,
        u_grasp: GraspCandidate,
        cube_grasp: GraspCandidate,
    ) -> tuple[str, str, str, str, str]:
        return (
            assignment.holder_hand,
            assignment.worker_hand,
            t_grasp.candidate_id,
            u_grasp.candidate_id,
            cube_grasp.candidate_id,
        )

    def _domains_for_assignment(
        self, assignment: RoleAssignment
    ) -> dict[str, GoalsetPickupDomain]:
        domains = self._mode_search_domains.get(assignment.assignment_id)
        if domains is None:
            domains = {
                "t_body": self._new_pickup_domain(
                    "t_body", assignment.holder_hand
                ),
                "u_legs": self._new_pickup_domain(
                    "u_legs", assignment.worker_hand
                ),
                "cube_head": self._new_pickup_domain(
                    "cube_head", assignment.worker_hand
                ),
            }
            self._mode_search_domains[assignment.assignment_id] = domains
        return domains

    def _next_qualified_mode(
        self,
        assignment: RoleAssignment,
        rejected_t: set[str],
        rejected_tu: set[tuple[str, str]],
        rejected_modes: set[tuple[str, str, str, str, str]],
        empty_scene: Mapping[str, Any],
    ) -> AssemblyMode | None:
        """Lazily expose grasps until one new complete mode is found."""

        domains = self._domains_for_assignment(assignment)
        while True:
            if all(domain.selected for domain in domains.values()):
                chosen = self._find_selected_mode(
                    assignment,
                    domains,
                    empty_scene,
                    rejected_t,
                    rejected_tu,
                    rejected_modes,
                )
                if chosen is not None:
                    return chosen

            if not any(domain.has_remaining for domain in domains.values()):
                print(
                    "[runtime] exhausted native cuRobo goal-set search without "
                    f"another complete mode: {assignment.assignment_id}; "
                    f"selected="
                    f"{ {part: len(domain.selected) for part, domain in domains.items()} }",
                    flush=True,
                )
                return None

            for domain in domains.values():
                if domain.has_remaining:
                    self._expand_pickup_domain(domain)

            selected_counts = {
                part: len(domain.selected) for part, domain in domains.items()
            }
            if any(
                not domain.selected and not domain.has_remaining
                for domain in domains.values()
            ):
                print(
                    "[runtime] assignment has an empty cuRobo pickup domain: "
                    f"{assignment.assignment_id}; selected={selected_counts}",
                    flush=True,
                )
                return None
            print(
                "[runtime] searching exact assembly constraints over "
                f"cuRobo-selected candidates for {assignment.assignment_id}; "
                f"selected={selected_counts}",
                flush=True,
            )

    def qualify_modes(self) -> list[AssemblyMode]:
        """Find complete grasp/mate modes for every evidence-backed assignment."""

        input_sha256 = self._constraint_cache_digest()
        cache_path = (
            ROOT
            / "artifacts/runtime_assembly/constraint_cache"
            / input_sha256
            / "complete_modes.json"
        )
        if cache_path.is_file():
            document = json.loads(cache_path.read_text())
            by_part_hand = {
                key: {
                    candidate.candidate_id: candidate for candidate in candidates
                }
                for key, candidates in self.pools.items()
            }
            modes = [
                AssemblyMode(
                    by_part_hand[("t_body", item["holder_hand"])][item["t_grasp"]],
                    by_part_hand[("u_legs", item["worker_hand"])][item["u_grasp"]],
                    by_part_hand[("cube_head", item["worker_hand"])][
                        item["cube_grasp"]
                    ],
                    item["u_work_id"],
                    np.asarray(item["u_world_T_root"], dtype=np.float64),
                    np.asarray(item["u_precontact_q"], dtype=np.float64),
                    item["head_work_id"],
                    np.asarray(item["head_world_T_root"], dtype=np.float64),
                    np.asarray(item["head_precontact_q"], dtype=np.float64),
                    item["holder_hand"],
                    item["worker_hand"],
                )
                for item in document["modes"]
            ]
            print(f"[runtime] loaded {len(modes)} qualified modes from {cache_path}", flush=True)
            return modes

        empty_scene = self.backend.scene(
            self.observation.table.center, self.observation.table.dimensions, {}
        )
        modes: list[AssemblyMode] = []
        for assignment in self.assignments:
            print(
                f"[runtime] qualifying assignment {assignment.assignment_id}",
                flush=True,
            )
            chosen = self._next_qualified_mode(
                assignment, set(), set(), set(), empty_scene
            )
            if chosen is None:
                continue

            modes.append(chosen)
            print(
                "[runtime] task-compatible complete mode qualified: "
                f"assignment={assignment.assignment_id}, "
                f"T={chosen.t_grasp.candidate_id}, "
                f"U={chosen.u_grasp.candidate_id}, "
                f"cube={chosen.cube_grasp.candidate_id}",
                flush=True,
            )
        if modes:
            self._write_mode_cache(cache_path, input_sha256, modes)
        return modes

    @staticmethod
    def _write_mode_cache(
        cache_path: Path, input_sha256: str, modes: Sequence[AssemblyMode]
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "schema_version": 3,
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
                            "holder_hand": mode.holder_hand,
                            "worker_hand": mode.worker_hand,
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
        holder = self.role_to_hand["holder"]
        samples = placement_samples(
            self.cfg["workspace"]["placement"],
            self.observation.table.top_z,
            self._assembly_min_z(),
        )
        selected: tuple[str, np.ndarray, np.ndarray, Any] | None = None
        with self.backend.stage(
            holder,
            self.q,
            self._scene(),
            self.payloads,
            max_goalset=1,
            hand_closed=self.closed,
        ) as planner:
            current_tool = planner.current_tool_world()
            print(
                "[runtime] placement start: "
                f"{holder}_tool_xyz={current_tool[:3, 3].tolist()}",
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
                holder,
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
            self._finger_motion(holder, 0.0, "place: release complete assembly")
            for part, root_T_part in self.task.member_transforms().items():
                self.objects[part] = ObjectState(
                    world_T_object=world_T_root @ root_T_part
                )
            self.placed = set(self.task.parts)
            self.payloads[holder] = None
            retreat = target.copy()
            retreat[2, 3] += float(self.cfg["motion"]["final_retreat_world_z_m"])
            with self.backend.stage(
                holder,
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

    def _reset_attempt(self, role_to_hand: Mapping[str, str]) -> None:
        self.role_to_hand = dict(role_to_hand)
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
        self.run.role_to_hand = dict(role_to_hand)
        self.run.planning_cost = None

    def _planning_cost(self) -> float:
        """Joint-space arc length of a complete collision-checked plan."""

        cost = 0.0
        previous = self.initial_q
        for segment in self.run.segments:
            q = np.asarray(segment.q, dtype=np.float64)
            if q.size == 0:
                continue
            cost += float(np.linalg.norm(q[0] - previous))
            if len(q) > 1:
                cost += float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
            previous = q[-1]
        return cost

    def _plan_complete_mode(
        self, mode: AssemblyMode, attempt: int
    ) -> tuple[RuntimeAssemblyRun | None, str, str]:
        """Plan one complete sequence without issuing any hardware command."""

        self._reset_attempt(mode.role_to_hand)
        self._event(
            "qualified assembly mode selected",
            qualified_mode_attempt=attempt,
            role_to_hand=mode.role_to_hand,
            t_grasp=mode.t_grasp.candidate_id,
            u_grasp=mode.u_grasp.candidate_id,
            cube_grasp=mode.cube_grasp.candidate_id,
            u_work_pose=mode.u_work_id,
            head_work_pose=mode.head_work_id,
        )
        self._hold("start: observed loose parts")
        if self._pick("t_body", mode.holder_hand, set(), mode.t_grasp) is None:
            return (
                None,
                "pick_t",
                "qualified T pickup failed during complete planning",
            )
        self._validate_task_state(0)
        if self._pick("u_legs", mode.worker_hand, set(), mode.u_grasp) is None:
            return None, "pick_u", "qualified U pickup failed with held T"
        self._validate_task_state(1)
        if not self._mate(
            "u_legs",
            "u_to_t",
            (mode.u_work_id, mode.u_world_T_root),
            mode.u_precontact_q,
        ):
            return (
                None,
                "mate_u_to_t",
                "qualified U mate failed during sequential planning",
            )
        self._validate_task_state(2)
        if self._pick(
            "cube_head", mode.worker_hand, set(), mode.cube_grasp
        ) is None:
            return (
                None,
                "pick_head",
                "qualified cube pickup failed with held T+U",
            )
        self._validate_task_state(3)
        if not self._mate(
            "cube_head",
            "head_to_t",
            (mode.head_work_id, mode.head_world_T_root),
            mode.head_precontact_q,
        ):
            return (
                None,
                "mate_head_to_t",
                "qualified head mate failed during sequential planning",
            )
        self._validate_task_state(4)
        if not self._place():
            return None, "place_complete", "no complete-assembly placement mode"
        self._validate_task_state(5)
        self._event(
            "complete",
            qualified_mode_attempt=attempt,
            observation_id=self.observation.observation_id,
        )
        cost = self._planning_cost()
        self.run.planning_cost = cost
        self._event(
            "complete plan scored",
            role_to_hand=mode.role_to_hand,
            joint_space_arc_length=cost,
        )
        return copy.deepcopy(self.run), "", ""

    def execute(self) -> RuntimeAssemblyRun:
        modes = self.qualify_modes()
        if not modes:
            raise RuntimeError("No T/U/cube grasp combination supports both connector modes")
        last_reason = "no qualified mode was execution-feasible"
        successful_runs: list[
            tuple[float, str, RuntimeAssemblyRun, AssemblyMode]
        ] = []
        empty_scene = self.backend.scene(
            self.observation.table.center, self.observation.table.dimensions, {}
        )
        attempt = 0
        for initial_mode in modes:
            assignment = RoleAssignment(
                initial_mode.holder_hand, initial_mode.worker_hand
            )
            rejected_t: set[str] = set()
            rejected_tu: set[tuple[str, str]] = set()
            rejected_modes: set[tuple[str, str, str, str, str]] = set()
            mode: AssemblyMode | None = initial_mode
            while mode is not None:
                attempt += 1
                planned, failed_stage, reason = self._plan_complete_mode(
                    mode, attempt
                )
                if planned is not None:
                    cost = float(planned.planning_cost)
                    successful_runs.append(
                        (cost, assignment.assignment_id, planned, mode)
                    )
                    # One complete plan is sufficient for this role
                    # assignment. The other assignment is still tested so the
                    # final choice can use whole-plan joint-space cost.
                    break

                last_reason = reason
                if failed_stage == "pick_t":
                    rejected_t.add(mode.t_grasp.candidate_id)
                elif failed_stage in {"pick_u", "mate_u_to_t"}:
                    rejected_tu.add(
                        (
                            mode.t_grasp.candidate_id,
                            mode.u_grasp.candidate_id,
                        )
                    )
                else:
                    rejected_modes.add(
                        self._mode_key(
                            assignment,
                            mode.t_grasp,
                            mode.u_grasp,
                            mode.cube_grasp,
                        )
                    )
                self.run.qualification.append(
                    {
                        "gate": "complete-sequence-backtrack",
                        "assignment": assignment.assignment_id,
                        "t_grasp": mode.t_grasp.candidate_id,
                        "u_grasp": mode.u_grasp.candidate_id,
                        "cube_grasp": mode.cube_grasp.candidate_id,
                        "failed_stage": failed_stage,
                        "reason": reason,
                    }
                )
                print(
                    f"[runtime] complete mode rejected: {reason}; searching "
                    f"next mode for {assignment.assignment_id}",
                    flush=True,
                )
                mode = self._next_qualified_mode(
                    assignment,
                    rejected_t,
                    rejected_tu,
                    rejected_modes,
                    empty_scene,
                )
        if successful_runs:
            successful_runs.sort(key=lambda item: (item[0], item[1]))
            _, _, best_run, best_mode = successful_runs[0]
            successful_modes = [item[3] for item in successful_runs]
            input_sha256 = self._constraint_cache_digest()
            cache_path = (
                ROOT
                / "artifacts/runtime_assembly/constraint_cache"
                / input_sha256
                / "complete_modes.json"
            )
            # Qualification may expose a connector-compatible mode whose
            # realized sequential prefix later fails. Replace that provisional
            # cache with modes that completed the entire in-memory plan so the
            # next identical run starts from proven executable choices.
            self._write_mode_cache(cache_path, input_sha256, successful_modes)
            self.run = best_run
            self.role_to_hand = best_mode.role_to_hand
            self._event(
                "role assignment chosen",
                role_to_hand=best_mode.role_to_hand,
                selection_policy=(
                    "minimum joint-space arc length among complete "
                    "collision-checked plans"
                ),
                complete_plan_count=len(successful_runs),
                alternatives=[
                    {
                        "role_to_hand": item[3].role_to_hand,
                        "joint_space_arc_length": item[0],
                    }
                    for item in successful_runs
                ],
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
