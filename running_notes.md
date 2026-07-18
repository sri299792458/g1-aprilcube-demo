# Running notes

## 2026-07-18 — clean restart, checkpoint 0

All implementation artifacts from the earlier prototype are excluded. They are
not migration inputs and are not compatibility requirements.

Pinned and audited:

- GraspGenX: `b9429097728cb1c430dd78b92edf17ba318aad03`
- Gripper descriptions used for the probe:
  `19a03c00d19aeaf052d0f6801f0041982d676e8a`
- Gripper: upstream `unitree_g1`, right hand only
- Probe object: upstream 0.10 m cube mesh
- Model output: 40 candidate `4x4` transforms and 40 confidence scores

Confirmed by code trace and render:

1. A returned matrix is the pose of the GraspGenX canonical root `G` in the
   input point-cloud frame `F`: `F_T_G`.
2. It is not a palm pose, wrist pose, pregrasp, or arm trajectory.
3. The Unitree descriptor URDF contains the fixed transform
   `G_T_right_palm_link` as its `world_joint`.
4. Therefore `F_T_right_palm_link = F_T_G @ G_T_right_palm_link`.
5. The released checkpoint is conditioned by the 12 open/half-open sweep-box
   numbers. It does not read the URDF or seven joint endpoints during inference.
6. The descriptor's open/close joint dictionaries are separate geometry and
   execution endpoints. Closing is not synthesized per object by GraspGenX.

Problems found before any project code was written:

- The upstream mesh demo README mentions a `--planner` flag that its current
  argument parser does not accept.
- Unitree hand meshes initially existed only as Git LFS pointer files. Inference
  could still run from the sweep descriptor, but the resulting empty visual
  meshes would have made visual validation invalid.
- The shipped Unitree descriptor is right-hand-only. Left-hand handling remains
  an explicit open item; it will not be guessed.
- A high confidence is not proof of collision-free closure or arm reachability.

Next gate: review the visual probe and contract. After approval, generate and
render candidates for the actual AprilCube meshes before introducing MoveIt.
