"""A narrow project adapter over the released cuRobo manipulation APIs.

This module does not solve IK, interpolate Cartesian waypoints, or rank grasp
geometry. It only translates project frames/state into cuRobo objects and
translates successful cuRobo trajectories back into the 14-arm-joint runtime
state.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
import yaml

from .grasp_goalset import goal_tool_pose


TOOL = {"left": "left_hand_grasp_frame", "right": "right_hand_grasp_frame"}
SLOT = {"left": "left_attached_object", "right": "right_attached_object"}


def matrix_to_pose_list(matrix: np.ndarray) -> list[float]:
    return [*matrix[:3, 3].tolist(), *tra.quaternion_from_matrix(matrix).tolist()]


def load_part_cuboids(path: str | Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load the exact AprilCube voxel union in the generated mesh frame."""

    document = yaml.safe_load(Path(path).read_text())
    voxel = float(document["shape"]["voxel_size_mm"]) / 1000.0
    entries = document["shape"]["cuboids"]
    low = np.asarray([entry["origin"] for entry in entries], dtype=np.float64)
    high = low + np.asarray([entry["size"] for entry in entries], dtype=np.float64)
    union_center = 0.5 * (low.min(axis=0) + high.max(axis=0))
    output = []
    for entry in entries:
        origin = np.asarray(entry["origin"], dtype=np.float64)
        size = np.asarray(entry["size"], dtype=np.float64)
        center = (origin + 0.5 * size - union_center) * voxel
        output.append((str(entry["name"]), center, size * voxel))
    return output


@dataclass(frozen=True)
class PayloadGeometry:
    """Part sphere locations expressed in one hand's grasp frame."""

    members: Mapping[str, np.ndarray]


@dataclass
class PlannedMotion:
    q_full: np.ndarray
    result: Any
    selected_index: int | None = None


class CuroboBackend:
    """Create upstream planners with runtime scenes and explicit locked arms."""

    def __init__(
        self,
        *,
        robot_config: str | Path,
        base_world: np.ndarray,
        arm_joint_names: Sequence[str],
        part_meshes: Mapping[str, str | Path],
        part_geometry: Mapping[str, str | Path],
        hand_profiles: Mapping[str, str | Path],
        planner_options: Mapping[str, Any],
    ):
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.scene import Mesh
        from curobo._src.geom.sphere_fit.types import SphereFitType

        self._MotionPlanner = MotionPlanner
        self._MotionPlannerCfg = MotionPlannerCfg
        self.robot_path = Path(robot_config).resolve()
        self.robot_document = yaml.safe_load(self.robot_path.read_text())
        self.base_world = np.asarray(base_world, dtype=np.float64)
        self.world_base = tra.inverse_matrix(self.base_world)
        self.arm_joint_names = list(arm_joint_names)
        self.part_meshes = {name: Path(path).resolve() for name, path in part_meshes.items()}
        self.part_cuboids = {
            name: load_part_cuboids(path) for name, path in part_geometry.items()
        }
        self.hand_profiles: dict[str, dict[str, dict[str, float]]] = {}
        for side in ("left", "right"):
            document = json.loads(Path(hand_profiles[side]).read_text())
            profile = {
                state: {name: float(value) for name, value in document[state].items()}
                for state in ("open", "close")
            }
            if set(profile["open"]) != set(profile["close"]):
                raise ValueError(f"{side} Dex3 open/close profiles name different joints")
            if len(profile["open"]) != 7:
                raise ValueError(f"{side} Dex3 profile must contain exactly seven joints")
            self.hand_profiles[side] = profile
        self.options = dict(planner_options)

        # Sphere fitting is the one expensive geometry operation the project
        # delegates to the released AttachmentManager. The resulting spheres
        # are immutable and reused by every short-lived stage planner.
        bootstrap_cfg = self._planner_cfg(
            copy.deepcopy(self.robot_document), scene=None, max_goalset=1
        )
        bootstrap = MotionPlanner(bootstrap_cfg)
        counts = dict(self.options.get("attachment_spheres", {}))
        self.part_spheres: dict[str, torch.Tensor] = {}
        try:
            for name, mesh_path in self.part_meshes.items():
                self.part_spheres[name] = bootstrap.trajopt_solver.core.attachment_manager.fit_spheres(
                    [Mesh(name=name, file_path=str(mesh_path), pose=[0, 0, 0, 1, 0, 0, 0])],
                    num_spheres=int(counts.get(name, 16)),
                    surface_radius=float(self.options.get("attachment_surface_radius_m", 0.004)),
                    sphere_fit_type=SphereFitType.MORPHIT,
                ).clone()
        finally:
            bootstrap.destroy()

    def _planner_cfg(self, robot: dict[str, Any], scene: dict[str, Any] | None, max_goalset: int):
        return self._MotionPlannerCfg.create(
            robot=robot,
            scene_model=scene,
            collision_cache=dict(self.options.get("collision_cache", {"obb": 16, "mesh": 8})),
            max_goalset=max_goalset,
            num_ik_seeds=int(self.options.get("num_ik_seeds", 32)),
            num_trajopt_seeds=int(self.options.get("num_trajopt_seeds", 4)),
            position_tolerance=float(self.options.get("position_tolerance_m", 0.005)),
            orientation_tolerance=float(self.options.get("orientation_tolerance_rad", 0.05)),
            use_cuda_graph=bool(self.options.get("use_cuda_graph", True)),
        )

    def scene(
        self,
        table_center: Sequence[float],
        table_dimensions: Sequence[float],
        loose_world_poses: Mapping[str, np.ndarray],
    ) -> dict[str, Any]:
        table_T = np.eye(4)
        table_T[:3, 3] = table_center
        output: dict[str, Any] = {
            "cuboid": {
                "table": {
                    "dims": list(table_dimensions),
                    "pose": matrix_to_pose_list(self.world_base @ table_T),
                }
            }
        }
        for part in sorted(loose_world_poses):
            base_T_part = self.world_base @ loose_world_poses[part]
            for cuboid_name, center, dimensions in self.part_cuboids[part]:
                part_T_cuboid = np.eye(4)
                part_T_cuboid[:3, 3] = center
                output["cuboid"][f"{part}__{cuboid_name}"] = {
                    "dims": dimensions.tolist(),
                    "pose": matrix_to_pose_list(base_T_part @ part_T_cuboid),
                }
        return output

    def _stage_robot(
        self,
        moving_side: str,
        q_full: np.ndarray,
        hand_closed: Mapping[str, float] | None,
    ) -> dict[str, Any]:
        robot = copy.deepcopy(self.robot_document)
        kin = robot["robot_cfg"]["kinematics"]
        kin["tool_frames"] = [TOOL[moving_side]]
        stationary = "right" if moving_side == "left" else "left"
        locks = kin.setdefault("lock_joints", {})
        for name, value in zip(self.arm_joint_names, q_full):
            if name.startswith(f"{stationary}_"):
                locks[name] = float(value)
        for side in ("left", "right"):
            fraction = float((hand_closed or {}).get(side, 0.0))
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"{side} hand_closed must be in [0, 1], got {fraction}")
            profile = self.hand_profiles[side]
            for name, open_value in profile["open"].items():
                close_value = profile["close"][name]
                locks[name] = open_value + fraction * (close_value - open_value)
        return robot

    def stage(
        self,
        moving_side: str,
        q_full: np.ndarray,
        scene: dict[str, Any],
        payloads: Mapping[str, PayloadGeometry | None],
        *,
        max_goalset: int,
        hand_closed: Mapping[str, float] | None = None,
    ) -> "StagePlanner":
        robot = self._stage_robot(moving_side, q_full, hand_closed)
        planner = self._MotionPlanner(self._planner_cfg(robot, scene, max_goalset))
        if bool(self.options.get("warmup", True)):
            planner.warmup(enable_graph=True, num_warmup_iterations=1)
        stage = StagePlanner(self, planner, moving_side, q_full)
        stage.set_payloads(payloads)
        return stage

    def coupled(
        self,
        q_full: np.ndarray,
        scene: dict[str, Any],
        payloads: Mapping[str, PayloadGeometry | None],
        hand_closed: Mapping[str, float] | None = None,
    ) -> "CoupledPlanner":
        robot = copy.deepcopy(self.robot_document)
        locks = robot["robot_cfg"]["kinematics"].setdefault("lock_joints", {})
        for side in ("left", "right"):
            fraction = float((hand_closed or {}).get(side, 0.0))
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"{side} hand_closed must be in [0, 1], got {fraction}")
            profile = self.hand_profiles[side]
            for name, open_value in profile["open"].items():
                locks[name] = open_value + fraction * (profile["close"][name] - open_value)
        planner = self._MotionPlanner(
            self._planner_cfg(robot, scene, max_goalset=1)
        )
        coupled = CoupledPlanner(self, planner, q_full)
        coupled.set_payloads(payloads)
        return coupled

    def payload_spheres(self, payload: PayloadGeometry) -> torch.Tensor:
        values = []
        for part, grasp_T_part in payload.members.items():
            source = self.part_spheres[part]
            transformed = source.clone()
            centers = trimesh.transform_points(
                source[:, :3].detach().cpu().numpy(), grasp_T_part
            )
            transformed[:, :3] = torch.as_tensor(
                centers, device=source.device, dtype=source.dtype
            )
            values.append(transformed)
        return torch.cat(values, dim=0)

    def part_obstacle_names(self, parts: Sequence[str]) -> list[str]:
        """Return exact scene names for the voxel cuboids of named parts."""

        return [
            f"{part}__{cuboid_name}"
            for part in parts
            for cuboid_name, _center, _dimensions in self.part_cuboids[part]
        ]


class _PlannerBase:
    def __init__(self, backend: CuroboBackend, planner: Any, q_full: np.ndarray):
        from curobo.types import JointState

        self.backend = backend
        self.planner = planner
        self._JointState = JointState
        self.active_names = list(planner.joint_names)
        indices = [backend.arm_joint_names.index(name) for name in self.active_names]
        self.q_active = JointState.from_position(
            torch.as_tensor(q_full[indices][None, :], device="cuda", dtype=torch.float32),
            joint_names=self.active_names,
        )
        self._full_template = np.asarray(q_full, dtype=np.float64).copy()

    def set_current(self, q_full: np.ndarray) -> None:
        self._full_template = np.asarray(q_full, dtype=np.float64).copy()
        indices = [self.backend.arm_joint_names.index(name) for name in self.active_names]
        self.q_active = self._JointState.from_position(
            torch.as_tensor(
                self._full_template[indices][None, :], device="cuda", dtype=torch.float32
            ),
            joint_names=self.active_names,
        )

    def close(self) -> None:
        self.planner.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _update_one_manager(self, manager: Any, payloads: Mapping[str, PayloadGeometry | None]) -> None:
        params = manager.kinematics_params
        for side in ("left", "right"):
            params.reset_link_spheres(SLOT[side])
            payload = payloads.get(side)
            if payload is not None:
                manager.update(
                    self.backend.payload_spheres(payload),
                    self.q_active,
                    link_name=SLOT[side],
                    world_objects_pose_offset=None,
                )

    def set_payloads(self, payloads: Mapping[str, PayloadGeometry | None]) -> None:
        # MotionPlanner currently exposes only TrajOpt's manager. IK owns a
        # separate Kinematics instance, so both upstream managers must receive
        # the identical attachment tensor for collision-correct goal solving.
        self._update_one_manager(self.planner.ik_solver.core.attachment_manager, payloads)
        self._update_one_manager(self.planner.trajopt_solver.core.attachment_manager, payloads)

    def _set_link_collision(self, links: Sequence[str], enabled: bool) -> None:
        method = "enable_link_spheres" if enabled else "disable_link_spheres"
        for solver in (self.planner.ik_solver, self.planner.trajopt_solver):
            params = solver.kinematics.config.kinematics_config
            for link in links:
                getattr(params, method)(link)

    def _trajectory(self, result: Any) -> np.ndarray:
        state = (
            result.interpolated_trajectory
            if result.interpolated_trajectory is not None
            else result.js_solution
        )
        state = self.planner.kinematics.get_active_js(state)
        q = state.position.detach().cpu().numpy()
        while q.ndim > 2:
            q = q[0]
        if result.interpolated_trajectory is not None and result.interpolated_last_tstep is not None:
            last = int(torch.as_tensor(result.interpolated_last_tstep).reshape(-1)[0].item())
            q = q[: last + 1]
        return q

    def _expand(self, active_q: np.ndarray) -> np.ndarray:
        full = np.repeat(self._full_template[None, :], len(active_q), axis=0)
        for active_index, name in enumerate(self.active_names):
            full[:, self.backend.arm_joint_names.index(name)] = active_q[:, active_index]
        return full

    def accept(self, result: Any, selected_index: int | None = None) -> PlannedMotion | None:
        if result is None or not bool(torch.as_tensor(result.success).any()):
            return None
        active_q = self._trajectory(result)
        full_q = self._expand(active_q)
        return PlannedMotion(full_q, result, selected_index)


class StagePlanner(_PlannerBase):
    def __init__(
        self, backend: CuroboBackend, planner: Any, moving_side: str, q_full: np.ndarray
    ):
        super().__init__(backend, planner, q_full)
        self.side = moving_side

    def plan_grasp(
        self,
        world_T_grasps: Sequence[np.ndarray],
        *,
        approach_offset_m: float,
        contact_links: Sequence[str],
        approach_in_tool_frame: bool = True,
    ) -> PlannedMotion | None:
        # Make each atlas slice an independent deterministic solver request.
        # Otherwise RNG state advances with the number of earlier failed
        # slices and candidate feasibility becomes batch-order dependent.
        self.planner.reset_seed()
        goal = goal_tool_pose(
            {TOOL[self.side]: world_T_grasps}, self.backend.world_base
        )
        # MotionPlanner toggles TrajOpt contact links internally. Mirror that
        # state on its separately-owned IK kinematics for the goal-set solve.
        self._set_link_collision(contact_links, False)
        try:
            result = self.planner.plan_grasp(
                goal,
                self.q_active,
                grasp_approach_axis="z",
                grasp_approach_offset=float(approach_offset_m),
                grasp_approach_in_tool_frame=approach_in_tool_frame,
                plan_grasp_to_lift=False,
                disable_collision_links=list(contact_links),
            )
        finally:
            self._set_link_collision(contact_links, True)
        if result is None or not bool(torch.as_tensor(result.success).any()):
            return None
        index = int(torch.as_tensor(result.goalset_index).reshape(-1)[0].item())
        # Preserve both upstream subtrajectories; the runtime records them as
        # separate semantic segments after expanding the active joint names.
        return PlannedMotion(np.empty((0, len(self.backend.arm_joint_names))), result, index)

    def grasp_submotion(self, grasp_result: Any, attribute: str) -> PlannedMotion | None:
        trajectory = getattr(grasp_result, f"{attribute}_interpolated_trajectory")
        if trajectory is None:
            trajectory = getattr(grasp_result, f"{attribute}_trajectory")
        if trajectory is None:
            return None
        state = self.planner.kinematics.get_active_js(trajectory)
        active_q = state.position.detach().cpu().numpy()
        while active_q.ndim > 2:
            active_q = active_q[0]
        last_value = getattr(grasp_result, f"{attribute}_interpolated_last_tstep")
        if getattr(grasp_result, f"{attribute}_interpolated_trajectory") is not None and last_value is not None:
            last = int(torch.as_tensor(last_value).reshape(-1)[0].item())
            active_q = active_q[: last + 1]
        return PlannedMotion(self._expand(active_q), grasp_result)

    def plan_pose(self, world_T_G: np.ndarray, *, max_attempts: int = 8) -> PlannedMotion | None:
        self.planner.reset_seed()
        goal = goal_tool_pose({TOOL[self.side]: [world_T_G]}, self.backend.world_base)
        return self.accept(
            self.planner.plan_pose(goal, self.q_active, max_attempts=max_attempts)
        )

    def plan_cspace(self, q_full_goal: np.ndarray, *, max_attempts: int = 8) -> PlannedMotion | None:
        indices = [self.backend.arm_joint_names.index(name) for name in self.active_names]
        goal = self._JointState.from_position(
            torch.as_tensor(
                np.asarray(q_full_goal)[indices][None, :],
                device="cuda",
                dtype=torch.float32,
            ),
            joint_names=self.active_names,
        )
        return self.accept(
            self.planner.plan_cspace(goal, self.q_active, max_attempts=max_attempts)
        )

    def current_tool_world(self) -> np.ndarray:
        pose = self.planner.compute_kinematics(self.q_active).tool_poses.get_link_pose(
            TOOL[self.side]
        )
        position = pose.position.reshape(-1, 3)[0].detach().cpu().numpy()
        quaternion = pose.quaternion.reshape(-1, 4)[0].detach().cpu().numpy()
        base_T_tool = tra.quaternion_matrix(quaternion)
        base_T_tool[:3, 3] = position
        return self.backend.base_world @ base_T_tool

    def plan_linear(
        self,
        world_T_G: np.ndarray,
        *,
        axis: str,
        in_tool_frame: bool,
        disable_links: Sequence[str] = (),
        disable_obstacles: Sequence[str] = (),
        max_attempts: int = 8,
    ) -> PlannedMotion | None:
        from curobo.types import ToolPoseCriteria

        frame = TOOL[self.side]
        criterion = ToolPoseCriteria.linear_motion(
            axis=axis, non_terminal_scale=1.0, project_distance_to_goal=in_tool_frame
        )
        self.planner.update_tool_pose_criteria({frame: criterion})
        self._set_link_collision(disable_links, False)
        for obstacle in disable_obstacles:
            self.planner.scene_collision_checker.enable_obstacle(obstacle, False)
        try:
            return self.plan_pose(world_T_G, max_attempts=max_attempts)
        finally:
            for obstacle in disable_obstacles:
                self.planner.scene_collision_checker.enable_obstacle(obstacle, True)
            self._set_link_collision(disable_links, True)
            self.planner.update_tool_pose_criteria({frame: ToolPoseCriteria()})


class CoupledPlanner(_PlannerBase):
    def solve_pair_ik(
        self,
        left_world_T_G: np.ndarray,
        right_world_T_G: np.ndarray,
        *,
        disable_links: Sequence[str] = (),
    ) -> np.ndarray | None:
        """Collision-check one coupled endpoint without imposing a fake start path."""

        self.planner.reset_seed()
        goal = goal_tool_pose(
            {
                TOOL["left"]: [left_world_T_G],
                TOOL["right"]: [right_world_T_G],
            },
            self.backend.world_base,
        )
        self._set_link_collision(disable_links, False)
        try:
            result = self.planner.ik_solver.solve_pose(
                goal,
                return_seeds=1,
                current_state=self.q_active,
            )
            success = torch.as_tensor(result.success)
            if not bool(success.any()):
                return None
            solution = result.solution[success][0].detach().cpu().numpy().reshape(-1)
            full = self._full_template.copy()
            for index, name in enumerate(self.active_names):
                full[self.backend.arm_joint_names.index(name)] = solution[index]
            return full
        finally:
            self._set_link_collision(disable_links, True)

    def plan_pair(
        self,
        left_world_T_G: np.ndarray,
        right_world_T_G: np.ndarray,
        *,
        disable_links: Sequence[str] = (),
        max_attempts: int = 6,
    ) -> PlannedMotion | None:
        self.planner.reset_seed()
        goal = goal_tool_pose(
            {
                TOOL["left"]: [left_world_T_G],
                TOOL["right"]: [right_world_T_G],
            },
            self.backend.world_base,
        )
        self._set_link_collision(disable_links, False)
        try:
            return self.accept(
                self.planner.plan_pose(goal, self.q_active, max_attempts=max_attempts)
            )
        finally:
            self._set_link_collision(disable_links, True)
