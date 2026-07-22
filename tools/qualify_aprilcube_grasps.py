#!/usr/bin/env python3
"""Contact-qualify raw GraspGenX poses with exact Dex3 kinematics and Newton.

The pipeline intentionally keeps three claims separate:

1. The fully open exact hand must not intersect the target mesh.
2. A sampled kinematic close must sweep both the thumb and opposing fingers
   through the target. This is only an inexpensive eligibility prefilter.
3. Newton closes the actuated exact hand around a free rigid object. Only this
   stage can pass a candidate, based on final opposing contacts and limited
   object displacement.

The simulation uses the candidate exactly as returned by GraspGenX. The
descriptor URDF supplies ``G_T_link``; no candidate-specific palm offset or
hand-authored grasp pose is introduced here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import newton
import coacd
import numpy as np
import trimesh
import trimesh.transformations as tra
import warp as wp
import yaml
import yourdfpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config/grasp_qualification.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/aprilcube_qualified_grasps/results.json"
DESCRIPTOR_ROOT = PROJECT_ROOT / "third_party/GraspGenX/assets/x_grippers"
PARTS_ROOT = PROJECT_ROOT / "generated/aprilcube_parts"
RAW_ROOT = PROJECT_ROOT / "artifacts/aprilcube_raw_grasps"


@dataclass(frozen=True)
class Candidate:
    """One raw GraspGenX candidate expressed as ``object_T_G``."""

    candidate_id: str
    rank: int
    confidence: float
    object_T_G: np.ndarray


@dataclass
class PhysicsScene:
    """One reusable exact-hand/object Newton model for a part."""

    model: newton.Model
    object_body: int
    object_joint: int
    hand_joint_indices: dict[str, int]
    open_cfg: dict[str, float]
    close_cfg: dict[str, float]
    shape_pairs: Any
    original_object_mesh: trimesh.Trimesh


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bare(name: Any) -> str:
    return str(name).split("/")[-1]


def transform_to_wp(matrix: np.ndarray) -> wp.transform:
    quat_wxyz = tra.quaternion_from_matrix(matrix)
    return wp.transform(
        wp.vec3(*[float(value) for value in matrix[:3, 3]]),
        wp.quat(
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
            float(quat_wxyz[0]),
        ),
    )


def wp_transform_array_to_matrix(value: np.ndarray) -> np.ndarray:
    matrix = tra.quaternion_matrix([value[6], value[3], value[4], value[5]])
    matrix[:3, 3] = value[:3]
    return matrix


def matrix_to_joint_free(matrix: np.ndarray) -> np.ndarray:
    quat_wxyz = tra.quaternion_from_matrix(matrix)
    return np.array(
        [
            matrix[0, 3],
            matrix[1, 3],
            matrix[2, 3],
            quat_wxyz[1],
            quat_wxyz[2],
            quat_wxyz[3],
            quat_wxyz[0],
        ],
        dtype=np.float32,
    )


def transform_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    return matrix[:3, :3] @ point + matrix[:3, 3]


def trimesh_to_newton(mesh: trimesh.Trimesh) -> newton.Mesh:
    return newton.Mesh(
        np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        np.ascontiguousarray(mesh.faces, dtype=np.int32).reshape(-1),
        compute_inertia=True,
    )


def load_candidates(path: Path) -> list[Candidate]:
    entries = yaml.safe_load(path.read_text())["grasps"]
    candidates = []
    for rank, (candidate_id, entry) in enumerate(entries.items()):
        quaternion = entry["orientation"]
        object_T_G = tra.quaternion_matrix(
            [quaternion["w"], *quaternion["xyz"]]
        )
        object_T_G[:3, 3] = entry["position"]
        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                rank=rank,
                confidence=float(entry["confidence"]),
                object_T_G=object_T_G,
            )
        )
    return candidates


def load_hand(urdf_path: Path) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        str(urdf_path),
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=True,
        load_collision_meshes=True,
    )


def link_group(link: str) -> str:
    if "thumb" in link:
        return "thumb"
    if "index" in link or "middle" in link:
        return "opponent"
    if "palm" in link:
        return "palm"
    return "other"


def collision_meshes_by_group(
    robot: yourdfpy.URDF,
    joints: dict[str, float],
) -> dict[str, trimesh.Trimesh]:
    """Return collision geometry baked into the canonical ``G`` frame."""
    robot.update_cfg(joints)
    pieces: dict[str, list[trimesh.Trimesh]] = {
        "all": [],
        "thumb": [],
        "opponent": [],
        "palm": [],
        "other": [],
    }
    scene = robot.collision_scene
    for geometry_name, geometry in scene.geometry.items():
        link = scene.graph.transforms.parents[geometry_name]
        G_T_geometry = scene.graph.get(
            frame_from=scene.graph.base_frame,
            frame_to=geometry_name,
        )[0]
        mesh = geometry.copy()
        mesh.apply_transform(G_T_geometry)
        pieces["all"].append(mesh)
        pieces[link_group(link)].append(mesh)
    result = {}
    for group, group_pieces in pieces.items():
        if group_pieces:
            result[group] = trimesh.util.concatenate(group_pieces)
    return result


def build_fcl_manager(mesh: trimesh.Trimesh, name: str):
    manager = trimesh.collision.CollisionManager()
    manager.add_object(name, mesh)
    return manager


def geometric_prefilter(
    candidates: list[Candidate],
    object_mesh: trimesh.Trimesh,
    robot: yourdfpy.URDF,
    descriptor: dict,
    sweep_samples: int,
) -> list[dict[str, Any]]:
    """Reject open collisions and candidates lacking a two-sided close sweep."""
    object_manager = build_fcl_manager(object_mesh, "object")

    open_mesh = collision_meshes_by_group(robot, descriptor["open"])["all"]
    open_manager = build_fcl_manager(open_mesh, "open_hand")

    sweep_managers: list[tuple[float, Any, Any]] = []
    for alpha in np.linspace(1.0 / sweep_samples, 1.0, sweep_samples):
        joints = {
            name: (1.0 - alpha) * descriptor["open"][name]
            + alpha * descriptor["close"][name]
            for name in descriptor["open"]
        }
        groups = collision_meshes_by_group(robot, joints)
        sweep_managers.append(
            (
                float(alpha),
                build_fcl_manager(groups["thumb"], f"thumb_{alpha:.3f}"),
                build_fcl_manager(groups["opponent"], f"opponent_{alpha:.3f}"),
            )
        )

    results = []
    for candidate in candidates:
        open_manager.set_transform("open_hand", candidate.object_T_G)
        open_collision = bool(open_manager.in_collision_other(object_manager))
        thumb_alphas = []
        opponent_alphas = []
        if not open_collision:
            for alpha, thumb_manager, opponent_manager in sweep_managers:
                thumb_name = next(iter(thumb_manager._objs))
                opponent_name = next(iter(opponent_manager._objs))
                thumb_manager.set_transform(thumb_name, candidate.object_T_G)
                opponent_manager.set_transform(opponent_name, candidate.object_T_G)
                if thumb_manager.in_collision_other(object_manager):
                    thumb_alphas.append(alpha)
                if opponent_manager.in_collision_other(object_manager):
                    opponent_alphas.append(alpha)

        reasons = []
        if open_collision:
            reasons.append("open_hand_collision")
        else:
            if not thumb_alphas:
                reasons.append("thumb_never_reaches_object")
            if not opponent_alphas:
                reasons.append("opponent_never_reaches_object")
        results.append(
            {
                "candidate_id": candidate.candidate_id,
                "rank": candidate.rank,
                "confidence": candidate.confidence,
                "open_collision": open_collision,
                "thumb_first_sweep_alpha": min(thumb_alphas, default=None),
                "opponent_first_sweep_alpha": min(opponent_alphas, default=None),
                "eligible_for_physics": not reasons,
                "reasons": reasons,
            }
        )
    return results


def deterministic_mass_properties(
    builder: newton.ModelBuilder,
    body: int,
    mesh: trimesh.Trimesh,
    mass: float,
) -> None:
    source = mesh.copy()
    volume = float(source.volume) if source.is_volume else 0.0
    if volume <= 1.0e-9:
        source = source.convex_hull
        volume = float(source.volume)
    source.density = mass / volume
    inertia = np.asarray(source.moment_inertia, dtype=np.float64)
    inertia_wp = wp.mat33(*[float(value) for value in inertia.reshape(-1)])
    builder.body_mass[body] = mass
    builder.body_inv_mass[body] = 1.0 / mass
    builder.body_inertia[body] = inertia_wp
    builder.body_inv_inertia[body] = wp.inverse(inertia_wp)
    builder.body_com[body] = wp.vec3(
        *[float(value) for value in np.asarray(source.center_mass)]
    )


def restore_urdf_mass_properties(
    builder: newton.ModelBuilder,
    urdf_path: Path,
) -> None:
    """Restore authoritative URDF inertia after collider approximation.

    Newton's mesh-approximation pass updates body mass properties from the
    generated colliders.  For this qualification experiment the collision
    geometry may change representation, but the hand dynamics should continue
    to use the official URDF inertials.  Reconstructing the inertia in the link
    frame also avoids a nonsymmetric palm tensor produced by the current
    Newton URDF import for its nonzero inertial-frame RPY.
    """
    root = ET.parse(urdf_path).getroot()
    body_by_name = {
        bare(label): index for index, label in enumerate(builder.body_label)
    }
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None or link.attrib["name"] not in body_by_name:
            continue
        mass_node = inertial.find("mass")
        inertia_node = inertial.find("inertia")
        if mass_node is None or inertia_node is None:
            continue
        origin = inertial.find("origin")
        xyz = np.zeros(3, dtype=float)
        rpy = np.zeros(3, dtype=float)
        if origin is not None:
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        values = {name: float(inertia_node.attrib[name]) for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
        inertia_inertial = np.array(
            [
                [values["ixx"], values["ixy"], values["ixz"]],
                [values["ixy"], values["iyy"], values["iyz"]],
                [values["ixz"], values["iyz"], values["izz"]],
            ],
            dtype=float,
        )
        link_R_inertial = tra.euler_matrix(*rpy, axes="sxyz")[:3, :3]
        inertia_link = link_R_inertial @ inertia_inertial @ link_R_inertial.T
        inertia_link = 0.5 * (inertia_link + inertia_link.T)
        eigenvalues = np.linalg.eigvalsh(inertia_link)
        if np.min(eigenvalues) <= 0.0:
            raise ValueError(
                f"URDF inertia for {link.attrib['name']} is not positive definite: {eigenvalues}"
            )
        body = body_by_name[link.attrib["name"]]
        mass = float(mass_node.attrib["value"])
        inertia_wp = wp.mat33(*[float(value) for value in inertia_link.reshape(-1)])
        builder.body_mass[body] = mass
        builder.body_inv_mass[body] = 1.0 / mass
        builder.body_inertia[body] = inertia_wp
        builder.body_inv_inertia[body] = wp.inverse(inertia_wp)
        builder.body_com[body] = wp.vec3(*[float(value) for value in xyz])


def build_physics_scene(
    urdf_path: Path,
    descriptor: dict,
    object_mesh: trimesh.Trimesh,
    cfg: dict,
) -> PhysicsScene:
    sim = cfg["simulation"]
    geometry_cfg = cfg["geometry"]
    coacd.set_log_level("off")

    builder = newton.ModelBuilder(gravity=0.0)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
        ke=float(sim["contact_ke"]),
        kd=float(sim["contact_kd"]),
        kf=float(sim["contact_kf"]),
        mu=float(sim["finger_friction"]),
        is_hydroelastic=False,
    )
    builder.add_urdf(
        str(urdf_path),
        floating=False,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
        collapse_fixed_joints=False,
    )

    mesh_shapes = [
        index
        for index in range(builder.shape_count)
        if int(builder.shape_type[index]) == int(newton.GeoType.MESH)
    ]
    coacd_shapes = []
    hull_shapes = []
    for shape in mesh_shapes:
        body = int(builder.shape_body[shape])
        label = bare(builder.body_label[body]) if body >= 0 else "world"
        if any(key in label for key in geometry_cfg["coacd_link_keywords"]):
            coacd_shapes.append(shape)
        else:
            hull_shapes.append(shape)
    if hull_shapes:
        builder.approximate_meshes(
            method="convex_hull",
            shape_indices=hull_shapes,
            keep_visual_shapes=False,
        )
    if coacd_shapes:
        builder.approximate_meshes(
            method="coacd",
            shape_indices=coacd_shapes,
            keep_visual_shapes=False,
            threshold=float(geometry_cfg["coacd_threshold"]),
        )

    restore_urdf_mass_properties(builder, urdf_path)

    bare_to_joint = {
        bare(label): index for index, label in enumerate(builder.joint_label)
    }
    for name in descriptor["open"]:
        joint = bare_to_joint[name]
        qd_offset = int(builder.joint_qd_start[joint])
        builder.joint_target_ke[qd_offset] = float(sim["finger_kp"])
        builder.joint_target_kd[qd_offset] = float(sim["finger_kd"])
        builder.joint_target_mode[qd_offset] = int(newton.JointTargetMode.POSITION)

    object_cfg = newton.ModelBuilder.ShapeConfig(
        ke=float(sim["contact_ke"]),
        kd=float(sim["contact_kd"]),
        kf=float(sim["contact_kf"]),
        mu=float(sim["object_friction"]),
        density=1000.0,
        is_hydroelastic=False,
    )
    object_body = builder.add_body(
        xform=wp.transform_identity(),
        mass=float(sim["object_mass_kg"]),
        label="qualification_object",
    )
    object_joint = builder.joint_count - 1
    object_shape = builder.add_shape_mesh(
        body=object_body,
        xform=wp.transform_identity(),
        mesh=trimesh_to_newton(object_mesh),
        cfg=object_cfg,
        label="qualification_object_shape",
    )
    builder.approximate_meshes(
        method="coacd",
        shape_indices=[object_shape],
        keep_visual_shapes=False,
        threshold=float(geometry_cfg["coacd_threshold"]),
    )
    deterministic_mass_properties(
        builder,
        object_body,
        object_mesh,
        float(sim["object_mass_kg"]),
    )
    object_qd_start = int(builder.joint_qd_start[object_joint])
    for offset in range(object_qd_start, object_qd_start + 6):
        builder.joint_armature[offset] = float(sim["object_armature"])

    model = builder.finalize()
    model_joint_names = {
        bare(label): index for index, label in enumerate(model.joint_label)
    }
    hand_joint_indices = {
        name: model_joint_names[name] for name in descriptor["open"]
    }

    shape_body = model.shape_body.numpy()
    object_shapes = np.flatnonzero(shape_body == object_body)
    hand_shapes = np.flatnonzero(
        (shape_body >= 0) & (shape_body != object_body)
    )
    pairs = np.asarray(
        [(int(object_shape), int(hand_shape)) for object_shape in object_shapes for hand_shape in hand_shapes],
        dtype=np.int32,
    )
    if not len(pairs):
        raise RuntimeError("No object-hand collision pairs were constructed")
    shape_pairs = wp.array(pairs, dtype=wp.vec2i, device=model.device)

    return PhysicsScene(
        model=model,
        object_body=object_body,
        object_joint=object_joint,
        hand_joint_indices=hand_joint_indices,
        open_cfg={name: float(value) for name, value in descriptor["open"].items()},
        close_cfg={name: float(value) for name, value in descriptor["close"].items()},
        shape_pairs=shape_pairs,
        original_object_mesh=object_mesh,
    )


def initialize_candidate(
    scene: PhysicsScene,
    candidate: Candidate,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = scene.model
    joint_q = model.joint_q.numpy().copy()
    joint_qd = np.zeros_like(model.joint_qd.numpy())
    joint_q_start = model.joint_q_start.numpy()
    for name, value in scene.open_cfg.items():
        joint = scene.hand_joint_indices[name]
        joint_q[int(joint_q_start[joint])] = value

    G_T_object = np.linalg.inv(candidate.object_T_G)
    object_q_start = int(joint_q_start[scene.object_joint])
    joint_q[object_q_start : object_q_start + 7] = matrix_to_joint_free(G_T_object)
    model.joint_q.assign(joint_q)
    model.joint_qd.assign(joint_qd)
    return joint_q, joint_qd, G_T_object


def contact_summary(
    scene: PhysicsScene,
    state: newton.State,
    contacts: newton.Contacts,
    active_contact_max_separation_m: float,
) -> dict[str, Any]:
    model = scene.model
    count = min(
        int(contacts.rigid_contact_count.numpy()[0]),
        int(contacts.rigid_contact_max),
    )
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    points0 = contacts.rigid_contact_point0.numpy()[:count]
    points1 = contacts.rigid_contact_point1.numpy()[:count]
    normals = contacts.rigid_contact_normal.numpy()[:count]
    shape_body = model.shape_body.numpy()
    body_q = state.body_q.numpy()

    per_link: dict[str, dict[str, list[np.ndarray]]] = {}
    buffered_object_hand_pairs = 0
    signed_separations = []
    for index in range(count):
        body0 = int(shape_body[int(shape0[index])])
        body1 = int(shape_body[int(shape1[index])])
        if body0 == scene.object_body and body1 != scene.object_body:
            hand_body = body1
            object_directed_normal = normals[index]
        elif body1 == scene.object_body and body0 != scene.object_body:
            hand_body = body0
            object_directed_normal = -normals[index]
        else:
            continue

        G_T_body0 = np.eye(4) if body0 < 0 else wp_transform_array_to_matrix(body_q[body0])
        G_T_body1 = np.eye(4) if body1 < 0 else wp_transform_array_to_matrix(body_q[body1])
        G_point0 = transform_point(G_T_body0, np.asarray(points0[index], dtype=float))
        G_point1 = transform_point(G_T_body1, np.asarray(points1[index], dtype=float))
        normal = np.asarray(normals[index], dtype=float)
        # Newton stores ``normal = -a_to_b``.  The signed surface separation
        # used by its writer is therefore ``-dot(point1-point0, normal)``.
        # Contact buffers can contain separated candidate pairs; only a
        # non-positive (or tiny tolerance-positive) separation is touching.
        signed_separation = -float(np.dot(G_point1 - G_point0, normal))
        signed_separations.append(signed_separation)
        buffered_object_hand_pairs += 1
        if signed_separation > active_contact_max_separation_m:
            continue

        link = "world" if hand_body < 0 else bare(model.body_label[hand_body])
        point = 0.5 * (G_point0 + G_point1)
        normal = np.asarray(object_directed_normal, dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm > 1.0e-8:
            normal /= norm
        entry = per_link.setdefault(
            link,
            {"points": [], "normals": [], "signed_separations": []},
        )
        entry["points"].append(point)
        entry["normals"].append(normal)
        entry["signed_separations"].append(signed_separation)

    links = {}
    group_normals: dict[str, list[np.ndarray]] = {"thumb": [], "opponent": [], "palm": [], "other": []}
    group_points: dict[str, list[np.ndarray]] = {"thumb": [], "opponent": [], "palm": [], "other": []}
    for link, values in sorted(per_link.items()):
        points = np.asarray(values["points"], dtype=float)
        link_normals = np.asarray(values["normals"], dtype=float)
        point = points.mean(axis=0)
        normal = link_normals.mean(axis=0)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > 1.0e-8:
            normal /= normal_norm
        group = link_group(link)
        group_points[group].append(point)
        group_normals[group].append(normal)
        links[link] = {
            "group": group,
            "active_contact_count": int(len(points)),
            "minimum_signed_separation_m": float(
                np.min(values["signed_separations"])
            ),
            "centroid_G": point.tolist(),
            "mean_normal_G": normal.tolist(),
        }

    group_contacts = {}
    for group in group_normals:
        if group_normals[group]:
            group_contacts[group] = {
                "link_count": len(group_normals[group]),
                "centroid_G": np.asarray(group_points[group]).mean(axis=0).tolist(),
                "link_mean_normals_G": np.asarray(group_normals[group]).tolist(),
            }

    opposing_dot = None
    if group_normals["thumb"] and group_normals["opponent"]:
        opposing_dot = min(
            float(np.dot(thumb_normal, opponent_normal))
            for thumb_normal in group_normals["thumb"]
            for opponent_normal in group_normals["opponent"]
        )

    return {
        "buffered_object_hand_pair_count": buffered_object_hand_pairs,
        "active_object_hand_contact_count": sum(
            value["active_contact_count"] for value in links.values()
        ),
        "active_contact_max_separation_m": active_contact_max_separation_m,
        "minimum_buffered_signed_separation_m": (
            float(np.min(signed_separations)) if signed_separations else None
        ),
        "links": links,
        "groups": group_contacts,
        "opposing_normal_dot": opposing_dot,
        "opposing_angle_degrees": (
            math.degrees(math.acos(float(np.clip(opposing_dot, -1.0, 1.0))))
            if opposing_dot is not None
            else None
        ),
    }


def simulate_candidate(
    scene: PhysicsScene,
    candidate: Candidate,
    cfg: dict,
) -> dict[str, Any]:
    sim = cfg["simulation"]
    acceptance = cfg["acceptance"]
    model = scene.model
    _joint_q, _joint_qd, initial_G_T_object = initialize_candidate(scene, candidate)

    states = [model.state(), model.state()]
    newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
    control = model.control()
    joint_q_start = model.joint_q_start.numpy()
    targets = control.joint_target_pos.numpy()
    for name, value in scene.open_cfg.items():
        targets[int(joint_q_start[scene.hand_joint_indices[name]])] = value
    control.joint_target_pos.assign(targets)

    capacity = int(sim["contact_capacity"])
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        solver="newton",
        integrator="implicitfast",
        cone="elliptic",
        iterations=int(sim["solver_iterations"]),
        ls_iterations=int(sim["solver_line_search_iterations"]),
        impratio=float(sim["solver_impratio"]),
        njmax=capacity,
        nconmax=capacity,
    )
    collision_pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=True,
        broad_phase="explicit",
        shape_pairs_filtered=scene.shape_pairs,
        rigid_contact_max=capacity,
    )
    contacts = newton.Contacts(
        rigid_contact_max=capacity,
        soft_contact_max=0,
        device=model.device,
    )

    def snapshot(step: int, alpha: float) -> dict[str, Any]:
        collision_pipeline.collide(states[0], contacts)
        wp.synchronize()
        summary = contact_summary(
            scene,
            states[0],
            contacts,
            float(sim["active_contact_max_separation_m"]),
        )
        body_q = states[0].body_q.numpy()
        G_T_object = wp_transform_array_to_matrix(body_q[scene.object_body])
        q_values = states[0].joint_q.numpy()
        return {
            "step": step,
            "time_seconds": step * float(sim["dt_seconds"]),
            "target_close_fraction": alpha,
            "object_translation_m": float(
                np.linalg.norm(
                    G_T_object[:3, 3] - initial_G_T_object[:3, 3]
                )
            ),
            "G_T_object": G_T_object.tolist(),
            "joints": {
                name: float(q_values[int(joint_q_start[joint])])
                for name, joint in scene.hand_joint_indices.items()
            },
            "active_contact_count": summary["active_object_hand_contact_count"],
            "active_contact_groups": sorted(summary["groups"]),
            "minimum_buffered_signed_separation_m": summary[
                "minimum_buffered_signed_separation_m"
            ],
        }

    dt = float(sim["dt_seconds"])
    close_steps = int(round(float(sim["close_seconds"]) / dt))
    hold_steps = int(round(float(sim["hold_seconds"]) / dt))
    total_steps = close_steps + hold_steps
    collide_every = int(sim["collide_every_steps"])
    trace_every = int(sim["trace_interval_steps"])
    trace = [snapshot(0, 0.0)]
    for step in range(total_steps):
        alpha = min(1.0, (step + 1) / close_steps)
        targets = control.joint_target_pos.numpy()
        for name in scene.open_cfg:
            target = (1.0 - alpha) * scene.open_cfg[name] + alpha * scene.close_cfg[name]
            targets[int(joint_q_start[scene.hand_joint_indices[name]])] = target
        control.joint_target_pos.assign(targets)
        states[0].clear_forces()
        if step % collide_every == 0:
            collision_pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, dt)
        states[0], states[1] = states[1], states[0]
        completed_steps = step + 1
        if completed_steps % trace_every == 0 or completed_steps == total_steps:
            trace.append(snapshot(completed_steps, alpha))

    collision_pipeline.collide(states[0], contacts)
    wp.synchronize()
    summary = contact_summary(
        scene,
        states[0],
        contacts,
        float(sim["active_contact_max_separation_m"]),
    )

    final_body_q = states[0].body_q.numpy()
    final_G_T_object = wp_transform_array_to_matrix(final_body_q[scene.object_body])
    translation = float(
        np.linalg.norm(final_G_T_object[:3, 3] - initial_G_T_object[:3, 3])
    )
    rotation_delta = final_G_T_object[:3, :3] @ initial_G_T_object[:3, :3].T
    rotation_angle = math.degrees(
        math.acos(float(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)))
    )

    final_joint_q = states[0].joint_q.numpy()
    joints = {
        name: float(final_joint_q[int(joint_q_start[joint])])
        for name, joint in scene.hand_joint_indices.items()
    }

    reasons = []
    for required_group in acceptance["required_contact_groups"]:
        if required_group not in summary["groups"]:
            reasons.append(f"missing_{required_group}_contact")
    if translation > float(acceptance["max_object_translation_m"]):
        reasons.append("object_pushed_too_far")
    opposing_dot = summary["opposing_normal_dot"]
    if opposing_dot is None:
        reasons.append("no_opposing_contact_pair")
    elif opposing_dot > float(acceptance["max_opposing_normal_dot"]):
        reasons.append("contact_normals_not_opposed")

    return {
        "candidate_id": candidate.candidate_id,
        "rank": candidate.rank,
        "confidence": candidate.confidence,
        "passed": not reasons,
        "reasons": reasons,
        "initial_object_T_G": candidate.object_T_G.tolist(),
        "initial_G_T_object": initial_G_T_object.tolist(),
        "final_G_T_object": final_G_T_object.tolist(),
        "object_translation_m": translation,
        "object_rotation_degrees": rotation_angle,
        "final_joints": joints,
        "contacts": summary,
        "trace": trace,
    }


def qualify_part(
    part: str,
    candidates: list[Candidate],
    robot: yourdfpy.URDF,
    descriptor: dict,
    urdf_path: Path,
    cfg: dict,
) -> dict[str, Any]:
    mesh_path = PARTS_ROOT / part / "grasp_mesh.obj"
    object_mesh = trimesh.load(mesh_path, force="mesh", process=False)
    geometric = geometric_prefilter(
        candidates,
        object_mesh,
        robot,
        descriptor,
        int(cfg["geometry"]["closing_sweep_samples"]),
    )
    eligible_ids = {
        result["candidate_id"] for result in geometric if result["eligible_for_physics"]
    }
    eligible = [candidate for candidate in candidates if candidate.candidate_id in eligible_ids]
    eligible = eligible[: int(cfg["acceptance"]["max_physics_candidates_per_part"])]

    print(
        f"{part}: {sum(not row['open_collision'] for row in geometric)}/{len(candidates)} open-clear, "
        f"{len(eligible_ids)} two-sided sweep, simulating top {len(eligible)}",
        flush=True,
    )

    physics = []
    if eligible:
        scene = build_physics_scene(urdf_path, descriptor, object_mesh, cfg)
        for index, candidate in enumerate(eligible, start=1):
            result = simulate_candidate(scene, candidate, cfg)
            physics.append(result)
            state = "PASS" if result["passed"] else "fail"
            print(
                f"  [{index:02d}/{len(eligible):02d}] {candidate.candidate_id} "
                f"score={candidate.confidence:.3f} {state} "
                f"move={1000.0 * result['object_translation_m']:.1f} mm "
                f"angle={result['contacts']['opposing_angle_degrees']}",
                flush=True,
            )

    passed = [result for result in physics if result["passed"]]
    selected = passed[0]["candidate_id"] if passed else (physics[0]["candidate_id"] if physics else None)
    return {
        "mesh": str(mesh_path.relative_to(PROJECT_ROOT)),
        "mesh_sha256": sha256(mesh_path),
        "raw_candidates": len(candidates),
        "open_clear_candidates": sum(not row["open_collision"] for row in geometric),
        "two_sided_sweep_candidates": len(eligible_ids),
        "physics_candidates": len(physics),
        "passed_candidates": len(passed),
        "selected_candidate": selected,
        "selected_is_pass": bool(passed),
        "geometric_prefilter": geometric,
        "physics": physics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    side = cfg["hand_side"]
    hand_root = DESCRIPTOR_ROOT / f"dex3_rev1_{side}"
    urdf_path = hand_root / "gripper.urdf"
    descriptor_path = hand_root / "config.json"
    descriptor = json.loads(descriptor_path.read_text())
    robot = load_hand(urdf_path)

    output = {
        "schema_version": 1,
        "status": "physics_contact_qualification",
        "claim_boundary": (
            "Pass means exact open geometry is clear and Newton produced final "
            "thumb/opponent contacts with opposed normals without pushing the free "
            "object beyond the configured limit. It does not imply arm reachability, "
            "table clearance, hardware force control, or task success."
        ),
        "config": str(args.config.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256(args.config),
        "hand": {
            "side_simulated": side,
            "descriptor": str(descriptor_path.relative_to(PROJECT_ROOT)),
            "descriptor_sha256": sha256(descriptor_path),
            "urdf": str(urdf_path.relative_to(PROJECT_ROOT)),
            "urdf_sha256": sha256(urdf_path),
            "canonical_left_right_equivalence": (
                "Descriptor audit establishes coincident canonical geometry; results "
                "are reusable for either hand before side-specific arm IK."
            ),
        },
        "parts": {},
    }

    for part in cfg["parts"]:
        candidates = load_candidates(RAW_ROOT / f"{part}.yaml")
        output["parts"][part] = qualify_part(
            part,
            candidates,
            robot,
            descriptor,
            urdf_path,
            cfg,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()
