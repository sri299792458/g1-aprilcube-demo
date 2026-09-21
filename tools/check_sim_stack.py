#!/usr/bin/env python3
"""Verify the locked GraspGenX/Newton/MuJoCo-Warp GPU stack with contact."""

from __future__ import annotations

import sys

import graspgenx
import mujoco
import mujoco_warp
import newton
import torch
import warp as wp


def main() -> None:
    """Drop a sphere onto a plane through the same solver path as playback."""
    wp.init()

    print(f"python:       {sys.version.split()[0]} ({sys.executable})")
    print(f"graspgenx:    {graspgenx.__file__}")
    print(f"newton:       {newton.__version__}")
    print(f"warp:         {wp.__version__}")
    print(f"mujoco:       {mujoco.__version__}")
    print(f"mujoco-warp:  {mujoco_warp.__version__}")
    print(f"torch:        {torch.__version__}")
    print(f"warp device:  {wp.get_device()}")

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA device")
    if not wp.get_device().is_cuda:
        raise RuntimeError("Warp did not select a CUDA device")

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 1000.0
    builder.add_ground_plane()

    radius = 0.05
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.30), wp.quat_identity()),
        mass=0.25,
        label="contact_smoke_sphere",
    )
    builder.add_shape_sphere(body=body, radius=radius)

    model = builder.finalize()
    state_in, state_out = model.state(), model.state()
    control = model.control()
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        solver="newton",
        integrator="implicitfast",
        cone="elliptic",
        iterations=100,
        ls_iterations=50,
        impratio=1000.0,
        njmax=4096,
        nconmax=4096,
    )
    collision_pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=True,
        broad_phase="explicit",
    )
    contacts = newton.Contacts(
        rigid_contact_max=1024,
        soft_contact_max=0,
        device=model.device,
    )

    for _ in range(240):
        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        state_in, state_out = state_out, state_in

    wp.synchronize()
    final_z = float(state_in.body_q.numpy()[body, 2])
    print(f"contact:      final z={final_z:.6f} m; expected about {radius:.6f} m")
    if not 0.9 * radius < final_z < 1.2 * radius:
        raise RuntimeError("sphere did not settle on the ground plane")

    print("PASS: GraspGenX imports and Newton/MuJoCo-Warp contact simulation works")


if __name__ == "__main__":
    main()
