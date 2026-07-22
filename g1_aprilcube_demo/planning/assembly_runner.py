"""Clean-cuRobo executor for the fixed T/U/cube assembly demonstration.

This module contains project policy (scene state, grasp selection, attachment
transfers, and stage sequencing).  It deliberately uses cuRobo's public
MotionPlanner and AttachmentManager without patching either one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
import yaml

from g1_aprilcube_demo.assembly import load_assembly_task


ROOT = Path(__file__).resolve().parents[2]
TOOL = {"left": "left_hand_grasp_frame", "right": "right_hand_grasp_frame"}
SLOT = {"left": "left_attached_object", "right": "right_attached_object"}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def axis_angle(values: list[float]) -> np.ndarray:
    axis = np.asarray(values[:3], dtype=np.float64)
    angle = np.deg2rad(float(values[3]))
    if np.linalg.norm(axis) == 0.0:
        return np.eye(4)
    return tra.rotation_matrix(angle, axis)


def transform(value: dict[str, Any]) -> np.ndarray:
    result = axis_angle(value["rotation_axis_angle_deg"])
    result[:3, 3] = np.asarray(value["translation"], dtype=np.float64)
    return result


def matrix_to_pose_list(matrix: np.ndarray) -> list[float]:
    q = tra.quaternion_from_matrix(matrix)
    return [*matrix[:3, 3].tolist(), *q.tolist()]


def pose_from_pool(candidate: dict[str, Any]) -> np.ndarray:
    value = candidate["object_T_G"]
    q = value["orientation"]
    result = tra.quaternion_matrix([q["w"], *q["xyz"]])
    result[:3, 3] = value["position"]
    return result


def local_offset(matrix: np.ndarray, xyz: list[float]) -> np.ndarray:
    offset = np.eye(4)
    offset[:3, 3] = xyz
    return matrix @ offset


def world_offset(matrix: np.ndarray, xyz: list[float]) -> np.ndarray:
    offset = np.eye(4)
    offset[:3, 3] = xyz
    return offset @ matrix


@dataclass
class ObjectState:
    """Pose rule used to reconstruct each part in every rendered frame."""

    world_T_object: np.ndarray | None = None
    hand: str | None = None
    grasp_T_object: np.ndarray | None = None

    def copy(self) -> "ObjectState":
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
class AssemblyRun:
    config_path: Path
    task_path: Path
    arm_joint_names: list[str]
    segments: list[Segment] = field(default_factory=list)
    selected: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "success": True,
            "planner": "clean cuRobo MotionPlanner (OMPL is not used in this prototype)",
            "config": str(self.config_path.relative_to(ROOT)),
            "task": str(self.task_path.relative_to(ROOT)),
            "arm_joint_names": self.arm_joint_names,
            "selected_grasps": self.selected,
            "events": self.events,
            "segments": [
                {"name": item.name, "frames": int(len(item.q))}
                for item in self.segments
            ],
        }


class CuroboAssemblyRunner:
    def __init__(self, config_path: str | Path, task_path: str | Path):
        # Imports are delayed so task/spec unit tests remain usable without CUDA.
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import JointState

        self._JointState = JointState
        self.config_path = project_path(config_path)
        self.task_path = project_path(task_path)
        self.cfg = yaml.safe_load(self.config_path.read_text())
        self.task = load_assembly_task(self.task_path)
        self.base_world = np.eye(4)
        self.base_world[:3, 3] = self.cfg["robot"]["base_world_translation_m"]
        self.world_base = tra.inverse_matrix(self.base_world)
        self.part_mesh = {
            name: project_path(value["mesh"])
            for name, value in self.cfg["parts"].items()
        }
        self.part_cuboids = {
            name: self._load_part_cuboids(self.task.parts[name].geometry_config)
            for name in self.cfg["parts"]
        }
        self.initial_world = {
            name: transform(value["initial_world_pose"])
            for name, value in self.cfg["parts"].items()
        }
        self.pools = {
            name: yaml.safe_load(project_path(value["grasp_pool"]).read_text())
            for name, value in self.cfg["parts"].items()
        }
        model = str(project_path(self.cfg["robot"]["model"]))
        scene = self._scene(set(self.cfg["parts"]))
        goalset = int(self.cfg["motion"]["candidate_goalset_size"])
        planner_cfg = MotionPlannerCfg.create(
            robot=model,
            scene_model=scene,
            collision_cache={"obb": 8, "mesh": 8},
            max_goalset=goalset,
            num_ik_seeds=32,
            num_trajopt_seeds=4,
        )
        self.planner = MotionPlanner(planner_cfg)
        # Collision-aware free-space transfers need the released graph planner
        # as a fallback when a straight TrajOpt seed passes through a loose
        # part. Warm it once; Cartesian contact stages are still waypointed.
        self.planner.warmup(enable_graph=True, num_warmup_iterations=1)
        from curobo._src.collision.attachment_manager import AttachmentManager

        # The clean checkpoint's MotionPlanner property forwards to a solver
        # attribute that is not constructed. Use the released manager class
        # explicitly with the planner's shared kinematics/collision tensors.
        self.attachments = AttachmentManager(
            self.planner.kinematics,
            self.planner.scene_collision_checker,
            self.planner.device_cfg,
        )
        # The current multi-tool pose solver rejects even a one-arm 4 cm probe
        # when both tools are constrained together. Solve each independent arm
        # with the same robot model, then submit the complete 14-joint goal to
        # the main collision-aware c-space planner. The nonmoving seven joints
        # are copied exactly from the current state and audited per segment.
        from curobo._src.solver.solver_ik import IKSolver

        robot_doc = yaml.safe_load(project_path(self.cfg["robot"]["model"]).read_text())
        self.side_ik = {}
        self.side_solution_columns: dict[str, tuple[list[int], list[int]]] = {}
        for side in ("left", "right"):
            side_robot = copy.deepcopy(robot_doc)
            side_robot["robot_cfg"]["kinematics"]["tool_frames"] = [TOOL[side]]
            side_cfg = MotionPlannerCfg.create(
                robot=side_robot,
                scene_model=None,
                max_goalset=goalset,
                # A 48-pose GraspGenX goalset needs several seeds per pose.
                # Sixty-four total seeds left most candidates with only one
                # chance and made reachability depend on batch composition.
                num_ik_seeds=256,
                num_trajopt_seeds=1,
                # Selection alternates between 48-pose goalsets and singleton
                # Cartesian waypoints; dynamic CUDA-graph shape reset is not
                # supported by the released standalone IKSolver.
                use_cuda_graph=False,
            )
            solver = IKSolver(side_cfg.ik_solver_config, None)
            self.side_ik[side] = solver
            moving_names = [
                name for name in self.planner.joint_names if name.startswith(f"{side}_")
            ]
            main_columns = [self.planner.joint_names.index(name) for name in moving_names]
            solver_columns = [solver.joint_names.index(name) for name in moving_names]
            self.side_solution_columns[side] = (main_columns, solver_columns)
        self.q = JointState.from_position(
            torch.tensor(
                [self.cfg["robot"]["start_arm_joint_positions"]],
                device="cuda",
                dtype=torch.float32,
            ),
            joint_names=self.planner.joint_names,
        )
        self.loose = set(self.cfg["parts"])
        self.objects = {
            name: ObjectState(world_T_object=value.copy())
            for name, value in self.initial_world.items()
        }
        self.closed = {"left": 0.0, "right": 0.0}
        self.run = AssemblyRun(
            config_path=self.config_path,
            task_path=self.task_path,
            arm_joint_names=list(self.planner.joint_names),
        )
        self.object_spheres = self._fit_part_spheres()
        self.chosen_object_T_G: dict[str, np.ndarray] = {}
        initial_fk = self._fk()
        self.ready_world = {
            side: self.base_world @ self._pose_matrix(initial_fk[TOOL[side]])
            for side in ("left", "right")
        }

    def close(self) -> None:
        for solver in self.side_ik.values():
            solver.destroy()
        self.planner.destroy()

    @staticmethod
    def _load_part_cuboids(path: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Return exact voxel-union cuboids in the generated part frame."""
        doc = yaml.safe_load(path.read_text())
        voxel = float(doc["shape"]["voxel_size_mm"]) / 1000.0
        entries = doc["shape"]["cuboids"]
        lows = np.asarray([entry["origin"] for entry in entries], dtype=np.float64)
        highs = lows + np.asarray([entry["size"] for entry in entries], dtype=np.float64)
        union_center = 0.5 * (lows.min(axis=0) + highs.max(axis=0))
        output = []
        for entry in entries:
            origin = np.asarray(entry["origin"], dtype=np.float64)
            size = np.asarray(entry["size"], dtype=np.float64)
            center = (origin + 0.5 * size - union_center) * voxel
            output.append((str(entry["name"]), center, size * voxel))
        return output

    def _scene(self, loose: set[str]) -> dict[str, Any]:
        table = self.cfg["table"]
        table_T = np.eye(4)
        table_T[:3, 3] = table["center_world_m"]
        table_T = self.world_base @ table_T
        output: dict[str, Any] = {
            "cuboid": {
                "table": {
                    "dims": table["dimensions_m"],
                    "pose": matrix_to_pose_list(table_T),
                }
            },
        }
        for name, fixture in self.cfg.get("fixtures", {}).items():
            fixture_T = np.eye(4)
            fixture_T[:3, 3] = fixture["center_world_m"]
            output["cuboid"][name] = {
                "dims": fixture["dimensions_m"],
                "pose": matrix_to_pose_list(self.world_base @ fixture_T),
            }
        for name in sorted(loose):
            pose = (
                self.world_base @ self.objects[name].world_T_object
                if hasattr(self, "objects")
                else self.world_base @ self.initial_world[name]
            )
            for cuboid_name, center, dims in self.part_cuboids[name]:
                local = np.eye(4)
                local[:3, 3] = center
                output["cuboid"][f"{name}__{cuboid_name}"] = {
                    "dims": dims.tolist(),
                    "pose": matrix_to_pose_list(pose @ local),
                }
        return output

    def _update_world(self, include_target: str | None = None) -> None:
        from curobo.scene import Scene

        loose = set(self.loose)
        if include_target is not None:
            loose.add(include_target)
        self.planner.update_world(Scene.create(self._scene(loose)))

    def _fit_part_spheres(self) -> dict[str, torch.Tensor]:
        from curobo.scene import Mesh
        from curobo._src.geom.sphere_fit.types import SphereFitType

        counts = {"t_body": 32, "u_legs": 32, "cube_head": 8}
        result = {}
        for name, path in self.part_mesh.items():
            result[name] = self.attachments.fit_spheres(
                [Mesh(name=name, file_path=str(path), pose=[0, 0, 0, 1, 0, 0, 0])],
                num_spheres=counts[name],
                surface_radius=0.004,
                sphere_fit_type=SphereFitType.MORPHIT,
            )
        return result

    def _fk(self):
        return self.planner.compute_kinematics(self.q).tool_poses.to_dict()

    def _pose_matrix(self, pose) -> np.ndarray:
        p = pose.position.reshape(-1, 3)[0].detach().cpu().numpy()
        q = pose.quaternion.reshape(-1, 4)[0].detach().cpu().numpy()
        matrix = tra.quaternion_matrix(q)
        matrix[:3, 3] = p
        return matrix

    def _goal(self, moving: str, matrices: list[np.ndarray]):
        from curobo._src.types.tool_pose import GoalToolPose

        frames = [TOOL["left"], TOOL["right"]]
        current = self._fk()
        count = len(matrices)
        positions = torch.empty((1, 1, 2, count, 3), device="cuda")
        quaternions = torch.empty((1, 1, 2, count, 4), device="cuda")
        for link_index, side in enumerate(("left", "right")):
            values = (
                matrices
                if side == moving
                else [self.base_world @ self._pose_matrix(current[TOOL[side]])] * count
            )
            for index, matrix in enumerate(values):
                pose = self.world_base @ matrix
                positions[0, 0, link_index, index] = torch.as_tensor(
                    pose[:3, 3], device="cuda", dtype=torch.float32
                )
                quaternions[0, 0, link_index, index] = torch.as_tensor(
                    tra.quaternion_from_matrix(pose), device="cuda", dtype=torch.float32
                )
        return GoalToolPose(frames, positions, quaternions)

    def _side_goal(self, side: str, matrices: list[np.ndarray]):
        from curobo._src.types.tool_pose import GoalToolPose

        count = len(matrices)
        positions = torch.empty((1, 1, 1, count, 3), device="cuda")
        quaternions = torch.empty((1, 1, 1, count, 4), device="cuda")
        for index, matrix in enumerate(matrices):
            pose = self.world_base @ matrix
            positions[0, 0, 0, index] = torch.as_tensor(
                pose[:3, 3], device="cuda", dtype=torch.float32
            )
            quaternions[0, 0, 0, index] = torch.as_tensor(
                tra.quaternion_from_matrix(pose), device="cuda", dtype=torch.float32
            )
        return GoalToolPose([TOOL[side]], positions, quaternions)

    def _trajectory(self, result) -> np.ndarray:
        state = (
            result.interpolated_trajectory
            if result.interpolated_trajectory is not None
            else result.js_solution
        )
        state = self.planner.kinematics.get_active_js(state)
        values = state.position.detach().cpu().numpy()
        while values.ndim > 2:
            values = values[0]
        if result.interpolated_trajectory is not None and getattr(result, "interpolated_last_tstep", None) is not None:
            last = int(torch.as_tensor(result.interpolated_last_tstep).reshape(-1)[0].item())
            values = values[: last + 1]
        return values

    def _snapshot(self) -> dict[str, ObjectState]:
        return {name: value.copy() for name, value in self.objects.items()}

    def _record(self, name: str, q: np.ndarray, selected: str | None = None) -> None:
        self.run.segments.append(
            Segment(name, q, self._snapshot(), dict(self.closed), selected)
        )

    def _hold(self, name: str, frames: int | None = None) -> None:
        count = frames or int(self.cfg["render"]["hold_frames"])
        q = self.q.position.detach().cpu().numpy().reshape(1, -1)
        self._record(name, np.repeat(q, count, axis=0))

    def _accept_result(self, name: str, result, selected: str | None = None) -> None:
        if result is None or not bool(torch.as_tensor(result.success).any()):
            status = getattr(result, "status", None)
            raise RuntimeError(f"cuRobo failed at {name}: {status}")
        trajectory = self._trajectory(result)
        self._record(name, trajectory, selected)
        self.q = self._JointState.from_position(
            torch.as_tensor(
                trajectory[-1:], device="cuda", dtype=torch.float32
            ),
            joint_names=self.planner.joint_names,
        )
        self.run.events.append({"stage": name, "success": True, "frames": len(trajectory)})
        print(f"[assembly] complete: {name} ({len(trajectory)} trajectory frames)", flush=True)

    def _accept_trajectory(
        self, name: str, trajectory: np.ndarray, selected: str | None = None
    ) -> None:
        self._record(name, trajectory, selected)
        self.q = self._JointState.from_position(
            torch.as_tensor(trajectory[-1:], device="cuda", dtype=torch.float32),
            joint_names=self.planner.joint_names,
        )
        self.run.events.append({"stage": name, "success": True, "frames": len(trajectory)})
        print(f"[assembly] complete: {name} ({len(trajectory)} trajectory frames)", flush=True)

    def _plan_joint_targets(
        self,
        name: str,
        side: str,
        targets: list[torch.Tensor],
        attempts: int,
    ):
        stationary = slice(7, 14) if side == "left" else slice(0, 7)
        print(f"[assembly] planning: {name} ({len(targets)} IK endpoints)", flush=True)
        for target in targets[:8]:
            target_state = self._JointState.from_position(
                target, joint_names=self.planner.joint_names
            )
            result = self.planner.plan_cspace(
                target_state, self.q, max_attempts=attempts
            )
            if result is None or not bool(torch.as_tensor(result.success).any()):
                continue
            trajectory = self._trajectory(result)
            stationary_start = self.q.position[0, stationary].detach().cpu().numpy()
            drift = float(np.max(np.abs(trajectory[:, stationary] - stationary_start)))
            if drift > 1e-3:
                projected = trajectory.copy()
                projected[:, stationary] = stationary_start
                projected_tensor = torch.as_tensor(
                    projected, device="cuda", dtype=torch.float32
                )
                valid = self.planner.graph_planner.check_samples_feasibility(
                    projected_tensor
                ).reshape(-1)
                if not bool(valid.all()):
                    print(
                        f"[assembly] rejected {name}: stationary projection "
                        f"collides ({int(valid.sum())}/{len(valid)} valid)",
                        flush=True,
                    )
                    continue
                self._accept_trajectory(name, projected)
                self.run.events[-1]["stationary_projection_applied"] = True
            else:
                self._accept_result(name, result)
            self.run.events[-1]["stationary_arm_max_drift_rad"] = drift
            return result
        return None

    def _ik_targets(
        self, side: str, matrices: list[np.ndarray]
    ) -> dict[int, list[torch.Tensor]]:
        """Collision-free whole-robot endpoints grouped by goalset member."""
        solver = self.side_ik[side]
        ik = self.side_ik[side].solve_pose(
            self._side_goal(side, matrices),
            # The right-tool solver orders its active chain before the left
            # chain. Reorder the seed by name; passing the main left-first
            # tensor here makes a nearby right-hand waypoint look like a
            # distant IK branch even if the returned solution is remapped.
            current_state=self.q.reorder(solver.joint_names),
            return_seeds=256,
        )
        success = torch.as_tensor(ik.success).reshape(-1)
        solutions = ik.solution.reshape(-1, len(self.planner.joint_names))
        indices = (
            torch.as_tensor(ik.goalset_index).reshape(-1)
            if ik.goalset_index is not None
            else torch.zeros(len(solutions), device=solutions.device, dtype=torch.long)
        )
        main_columns, solver_columns = self.side_solution_columns[side]
        stationary_columns = [
            index for index in range(len(self.planner.joint_names)) if index not in main_columns
        ]
        states: list[torch.Tensor] = []
        source_indices: list[int] = []
        for seed_index in torch.nonzero(success, as_tuple=False).reshape(-1).tolist():
            q = self.q.position.clone()
            q[:, main_columns] = solutions[seed_index, solver_columns]
            q[:, stationary_columns] = self.q.position[:, stationary_columns]
            states.append(q)
            source_indices.append(int(indices[seed_index].item()))
        if not states:
            return {}
        stacked = torch.cat(states, dim=0)
        valid = self.planner.graph_planner.check_samples_feasibility(stacked).reshape(-1)
        grouped: dict[int, list[torch.Tensor]] = {}
        current = self.q.position
        valid_rows = torch.nonzero(valid, as_tuple=False).reshape(-1).tolist()
        valid_rows.sort(
            key=lambda row: float(torch.linalg.vector_norm(stacked[row : row + 1] - current).item())
        )
        for row in valid_rows:
            grouped.setdefault(source_indices[row], []).append(stacked[row : row + 1])
        return grouped

    def _plan(self, name: str, side: str, matrices: list[np.ndarray], attempts: int = 5):
        grouped = self._ik_targets(side, matrices)
        for goal_index in sorted(grouped):
            result = self._plan_joint_targets(
                name, side, grouped[goal_index], attempts
            )
            if result is not None:
                result.goalset_index = torch.tensor([goal_index], device="cuda")
                return result
        return None

    def _feasible_goal_indices(
        self, side: str, matrices: list[np.ndarray]
    ) -> set[int]:
        """Return goalset members with at least one collision-free IK state."""
        return set(self._ik_targets(side, matrices))

    def _linear_plan(
        self,
        name: str,
        side: str,
        matrix: np.ndarray,
        axis: str,
        in_tool_frame: bool,
        hidden_slots: tuple[str, ...] = (),
        steps: int = 4,
    ) -> None:
        for slot in hidden_slots:
            self.planner.disable_link_collision([SLOT[slot]])
        try:
            current_base = self._pose_matrix(self._fk()[TOOL[side]])
            current = self.base_world @ current_base
            q0 = tra.quaternion_from_matrix(current)
            q1 = tra.quaternion_from_matrix(matrix)
            for index, alpha in enumerate(np.linspace(1.0 / steps, 1.0, steps), start=1):
                waypoint = np.eye(4)
                waypoint[:3, 3] = (1.0 - alpha) * current[:3, 3] + alpha * matrix[:3, 3]
                waypoint[:3, :3] = tra.quaternion_matrix(
                    tra.quaternion_slerp(q0, q1, float(alpha))
                )[:3, :3]
                result = self._plan(
                    f"{name} [{index}/{steps}]", side, [waypoint], attempts=6
                )
                if result is None:
                    raise RuntimeError(
                        f"cuRobo failed at linear waypoint {name} {index}/{steps}"
                    )
        finally:
            for slot in hidden_slots:
                self.planner.enable_link_collision([SLOT[slot]])

    def _candidate_subset(self, part: str) -> list[dict[str, Any]]:
        candidates = self.pools[part]["candidates"]
        allowed = set(self.task.parts[part].allowed_grasp_cuboids)
        if allowed:
            candidates = [
                item
                for item in candidates
                if self._approach_target_cuboid(part, pose_from_pool(item)) in allowed
            ]
        if not candidates:
            raise RuntimeError(f"No qualified candidates satisfy {part} constraints")
        return candidates

    def _approach_target_cuboid(
        self, part: str, object_T_G: np.ndarray
    ) -> str | None:
        """Return the first part cuboid hit by the candidate's approach ray.

        GraspGenX's G origin is not a contact point.  A region constraint must
        therefore not classify a grasp from ``G.translation`` alone.  The
        released cuRobo/GraspGenX contract approaches from negative local Z to
        G, so the positive local-Z ray at the exact grasp identifies the piece
        of object geometry that the hand approaches.  This is only a coarse
        task-region label; Isaac/PhysX remains the grasp-success authority.
        """
        origin = object_T_G[:3, 3]
        direction = object_T_G[:3, 2]
        best: tuple[float, str] | None = None
        for name, center, dims in self.part_cuboids[part]:
            low = center - 0.5 * dims
            high = center + 0.5 * dims
            t_min, t_max = 0.0, float("inf")
            for axis in range(3):
                if abs(direction[axis]) < 1e-9:
                    if origin[axis] < low[axis] or origin[axis] > high[axis]:
                        t_min = float("inf")
                        break
                    continue
                near = (low[axis] - origin[axis]) / direction[axis]
                far = (high[axis] - origin[axis]) / direction[axis]
                if near > far:
                    near, far = far, near
                t_min = max(t_min, near)
                t_max = min(t_max, far)
                if t_min > t_max:
                    break
            if t_min <= t_max and t_max >= 0.0:
                hit = max(0.0, t_min)
                if best is None or hit < best[0]:
                    best = (hit, name)
        return None if best is None else best[1]

    def _pick(self, part: str, side: str) -> None:
        candidates = self._candidate_subset(part)
        object_pose = self.objects[part].world_T_object
        pre_offset = float(self.cfg["motion"]["pick_pregrasp_offset_local_z_m"])
        # The target remains collision-live for the free-space transfer to
        # pregrasp. It is hidden only while checking/planning the final exact
        # contact approach; all other parts and the table remain live.
        self._update_world()
        batch_size = int(self.cfg["motion"]["candidate_goalset_size"])
        result = None
        chosen = None
        chosen_grasp = None
        for offset in range(0, len(candidates), batch_size):
            batch = candidates[offset : offset + batch_size]
            grasps = [object_pose @ pose_from_pool(item) for item in batch]
            pregrasps = [local_offset(matrix, [0, 0, pre_offset]) for matrix in grasps]
            pre_targets = self._ik_targets(side, pregrasps)
            self.loose.remove(part)
            self._update_world()
            grasp_targets = self._ik_targets(side, grasps)
            future_pre_targets: dict[int, list[torch.Tensor]] | None = None
            future_contact_targets: dict[int, list[torch.Tensor]] | None = None
            if part != self.task.root_part:
                member = self.task.member_transforms()[part]
                holder_cfg = (
                    self.cfg["motion"]["holder_u_attach_t_world_pose"]
                    if part == "u_legs"
                    else self.cfg["motion"]["holder_upright_t_world_pose"]
                )
                future_world_T_part = transform(holder_cfg) @ member
                future_contacts = [
                    future_world_T_part @ pose_from_pool(item) for item in batch
                ]
                future_pres = [
                    world_offset(
                        matrix,
                        [
                            0,
                            0,
                            float(
                                self.cfg["motion"][
                                    "mate_precontact_offsets_world_z_m"
                                ][part]
                            ),
                        ],
                    )
                    for matrix in future_contacts
                ]
                future_pre_targets = self._ik_targets(side, future_pres)
                # Exact connector contact intentionally touches the holder's
                # attached composite. cuRobo has link-level enable/disable but
                # no pairwise ACM for two attachment slots, so mirror the
                # connector stage: keep the holder live at precontact, hide it
                # only while qualifying the exact-contact endpoint.
                self.planner.disable_link_collision([SLOT["left"]])
                try:
                    future_contact_targets = self._ik_targets(side, future_contacts)
                finally:
                    self.planner.enable_link_collision([SLOT["left"]])
            self.loose.add(part)
            self._update_world()
            feasible = set(pre_targets) & set(grasp_targets)
            if future_pre_targets is not None and future_contact_targets is not None:
                feasible &= set(future_pre_targets) & set(future_contact_targets)
            if not feasible:
                continue
            selected_local = min(feasible)
            attempt = self._plan_joint_targets(
                f"{part}: move to qualified pregrasp",
                side,
                pre_targets[selected_local],
                4,
            )
            if attempt is None:
                clearance = world_offset(
                    pregrasps[selected_local],
                    [
                        0,
                        0,
                        float(self.cfg["motion"]["pregrasp_clearance_world_z_m"]),
                    ],
                )
                self._linear_plan(
                    f"{part}: move above qualified pregrasp",
                    side,
                    clearance,
                    axis="z",
                    in_tool_frame=False,
                    steps=8,
                )
                self._linear_plan(
                    f"{part}: descend to qualified pregrasp",
                    side,
                    pregrasps[selected_local],
                    axis="z",
                    in_tool_frame=False,
                )
                attempt = True
            if attempt is None:
                continue
            result = attempt
            chosen = batch[selected_local]
            chosen_grasp = grasps[selected_local]
            break
        if result is None or chosen is None or chosen_grasp is None:
            raise RuntimeError(f"No collision-free {side} pregrasp found for {part}")
        self.run.segments[-1].selected_candidate = chosen["candidate_id"]
        print(
            f"[assembly] selected {part} grasp {chosen['candidate_id']}",
            flush=True,
        )

        # Only the manipulation target is removed for the qualified straight
        # contact approach; table and every other object remain live.
        self.loose.remove(part)
        self._update_world()
        self._linear_plan(
            f"{part}: straight contact approach",
            side,
            chosen_grasp,
            axis="z",
            in_tool_frame=True,
        )
        self._hold(f"{part}: at grasp")
        close_frames = int(self.cfg["render"]["hold_frames"])
        q = self.q.position.detach().cpu().numpy().reshape(1, -1)
        for i in range(close_frames):
            self.closed[side] = (i + 1) / close_frames
            self._record(f"{part}: close {side} Dex3", q)

        object_T_G = pose_from_pool(chosen)
        grasp_T_object = tra.inverse_matrix(object_T_G)
        self.chosen_object_T_G[part] = object_T_G
        self.objects[part] = ObjectState(hand=side, grasp_T_object=grasp_T_object)
        self._set_payload(side, {part: grasp_T_object})
        self.planner.disable_link_collision([SLOT[side]])
        lift = world_offset(chosen_grasp, [0, 0, float(self.cfg["motion"]["pick_retract_world_z_m"])])
        self._linear_plan(f"{part}: vertical retract", side, lift, axis="z", in_tool_frame=False)
        self.planner.enable_link_collision([SLOT[side]])
        self._hold(f"{part}: retained after retract")
        self.run.selected[part] = {
            "hand": side,
            "candidate_id": chosen["candidate_id"],
            "family_id": chosen["family_id"],
            "graspgenx_score": chosen["graspgenx_score"],
            "object_T_G": object_T_G.tolist(),
            "qualification": "Isaac/PhysX VIRAL-profile PASS",
        }

    def _set_payload(self, side: str, members: dict[str, np.ndarray]) -> None:
        spheres = []
        for part, grasp_T_part in members.items():
            source = self.object_spheres[part]
            centers = source[:, :3].detach().cpu().numpy()
            transformed = trimesh.transform_points(centers, grasp_T_part)
            value = source.clone()
            value[:, :3] = torch.as_tensor(transformed, device=value.device, dtype=value.dtype)
            spheres.append(value)
        combined = torch.cat(spheres, dim=0)
        self.attachments.update(
            combined, self.q, link_name=SLOT[side], world_objects_pose_offset=None
        )

    def _clear_payload(self, side: str) -> None:
        params = self.planner.kinematics.config.kinematics_config
        params.reset_link_spheres(SLOT[side])

    def _move_root_part(self, name: str, world_T_t: np.ndarray) -> None:
        left_goal = world_T_t @ self.chosen_object_T_G["t_body"]
        self._update_world()
        result = self._plan(name, "left", [left_goal], attempts=7)
        if result is None:
            self._linear_plan(
                name,
                "left",
                left_goal,
                axis="z",
                in_tool_frame=False,
                steps=12,
            )

    def _park_worker(self, name: str) -> None:
        self._update_world()
        result = self._plan(name, "right", [self.ready_world["right"]], attempts=8)
        if result is None:
            self._linear_plan(
                name,
                "right",
                self.ready_world["right"],
                axis="z",
                in_tool_frame=False,
                steps=8,
            )

    def _mate(self, child: str, connection: str) -> None:
        member = self.task.member_transforms()[child]
        holder_cfg = (
            self.cfg["motion"]["holder_u_attach_t_world_pose"]
            if child == "u_legs"
            else self.cfg["motion"]["holder_upright_t_world_pose"]
        )
        world_T_t = transform(holder_cfg)
        world_T_child = world_T_t @ member
        child_T_G = self.chosen_object_T_G[child]
        contact_G = world_T_child @ child_T_G
        pre_G = world_offset(
            contact_G,
            [
                0,
                0,
                float(
                    self.cfg["motion"]["mate_precontact_offsets_world_z_m"][child]
                ),
            ],
        )
        self._update_world()
        precontact_result = self._plan(
            f"{connection}: move child to precontact", "right", [pre_G], attempts=8
        )
        if precontact_result is None:
            # A single c-space seed (or simultaneous translate+rotate) can
            # sweep a large carried part through the holder. Route it using
            # the conventional payload sequence while keeping both payloads
            # live: raise, rotate away, traverse above the target, descend.
            current = self.base_world @ self._pose_matrix(self._fk()[TOOL["right"]])
            transit_z = max(float(current[2, 3]), float(pre_G[2, 3])) + float(
                self.cfg["motion"]["mate_transit_clearance_world_z_m"]
            )
            raised = current.copy()
            raised[2, 3] = transit_z
            self._linear_plan(
                f"{connection}: raise carried child clear",
                "right",
                raised,
                axis="z",
                in_tool_frame=False,
            )
            rotated = raised.copy()
            rotated[:3, :3] = pre_G[:3, :3]
            self._linear_plan(
                f"{connection}: orient carried child in clear space",
                "right",
                rotated,
                axis="z",
                in_tool_frame=False,
                steps=8,
            )
            above = pre_G.copy()
            above[2, 3] = transit_z
            self._linear_plan(
                f"{connection}: traverse above precontact",
                "right",
                above,
                axis="z",
                in_tool_frame=False,
                steps=6,
            )
            self._linear_plan(
                f"{connection}: descend to precontact",
                "right",
                pre_G,
                axis="z",
                in_tool_frame=False,
            )
        self._linear_plan(
            f"{connection}: connector approach",
            "right",
            contact_G,
            axis="z",
            in_tool_frame=False,
            hidden_slots=("left",),
        )
        self._hold(f"{connection}: aligned")

        open_frames = int(self.cfg["render"]["hold_frames"])
        q = self.q.position.detach().cpu().numpy().reshape(1, -1)
        for i in range(open_frames):
            self.closed["right"] = 1.0 - (i + 1) / open_frames
            self._record(f"{connection}: magnetic snap and release", q)

        t_G = self.chosen_object_T_G["t_body"]
        grasp_T_t = tra.inverse_matrix(t_G)
        members = {"t_body": grasp_T_t}
        for part, t_T_part in self.task.member_transforms().items():
            if part == "t_body":
                continue
            if part == child or self.objects[part].hand == "left":
                members[part] = grasp_T_t @ t_T_part
        self._clear_payload("right")
        self._set_payload("left", members)
        for part, grasp_T_part in members.items():
            self.objects[part] = ObjectState(hand="left", grasp_T_object=grasp_T_part)

        # The retreat begins in intentional hand/composite contact. Hide only
        # the holder payload until the worker has backed out along the exact
        # reverse GraspGenX approach direction.
        current_right = self._pose_matrix(self._fk()[TOOL["right"]])
        current_right_world = self.base_world @ current_right
        retreat = local_offset(
            current_right_world,
            [0, 0, float(self.cfg["motion"]["worker_retreat_local_z_m"])],
        )
        self._linear_plan(
            f"{connection}: worker retreat",
            "right",
            retreat,
            axis="z",
            in_tool_frame=True,
            hidden_slots=("left",),
        )
        self._hold(f"{connection}: composite retained by holder")
        self.run.events.append(
            {
                "stage": connection,
                "scene_transition": "right child attachment -> left composite attachment",
                "members": sorted(members),
            }
        )

    def execute(self) -> AssemblyRun:
        self._hold("start: three loose parts")
        self._pick("t_body", "left")
        u_attach = transform(self.cfg["motion"]["holder_u_attach_t_world_pose"])
        self._move_root_part("holder: stage upright T for U attachment", u_attach)

        self._pick("u_legs", "right")
        self._mate("u_legs", "u_to_t")

        self._park_worker("worker: park clear of composite/head workspace")
        upright = transform(self.cfg["motion"]["holder_upright_t_world_pose"])
        self._move_root_part("holder: lower T+U to head-assembly height", upright)

        self._pick("cube_head", "right")
        self._mate("cube_head", "head_to_t")

        placed = transform(self.cfg["motion"]["placed_t_world_pose"])
        self._move_root_part("holder: place complete assembly", placed)
        self._hold("complete assembly at placement")
        q = self.q.position.detach().cpu().numpy().reshape(1, -1)
        frames = int(self.cfg["render"]["hold_frames"])
        for i in range(frames):
            self.closed["left"] = 1.0 - (i + 1) / frames
            self._record("release complete assembly", q)

        for part, t_T_part in self.task.member_transforms().items():
            self.objects[part] = ObjectState(world_T_object=placed @ t_T_part)
        self._clear_payload("left")
        current_left = self._pose_matrix(self._fk()[TOOL["left"]])
        retreat = world_offset(
            self.base_world @ current_left,
            [0, 0, float(self.cfg["motion"]["retreat_world_z_m"])],
        )
        self._linear_plan(
            "holder: retreat from placed assembly",
            "left",
            retreat,
            axis="z",
            in_tool_frame=False,
        )
        self._hold("final reveal", frames * 2)
        self.run.events.append(
            {
                "stage": "complete",
                "scene_transition": "left composite attachment -> placed world assembly",
            }
        )
        return self.run


def save_run(run: AssemblyRun, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "planning_report.json").write_text(json.dumps(run.report(), indent=2) + "\n")
    arrays = {f"segment_{index:03d}": item.q for index, item in enumerate(run.segments)}
    np.savez_compressed(directory / "arm_trajectories.npz", **arrays)
    render_state = {
        "schema_version": 1,
        "segments": [
            {
                "trajectory_key": f"segment_{index:03d}",
                "name": segment.name,
                "selected_candidate": segment.selected_candidate,
                "hand_closed": segment.hand_closed,
                "objects": {
                    name: {
                        "world_T_object": (
                            None
                            if state.world_T_object is None
                            else state.world_T_object.tolist()
                        ),
                        "hand": state.hand,
                        "grasp_T_object": (
                            None
                            if state.grasp_T_object is None
                            else state.grasp_T_object.tolist()
                        ),
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
