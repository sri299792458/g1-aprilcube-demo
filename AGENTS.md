# Maintaining the offline G1 grasp tools

Read README.md and the relevant guide under docs/ before changing a pipeline.
Preserve immutable grasp IDs, object_T_G transforms, source hashes, hand/shape
identity and the qualification profile. Distinguish intrinsic simulation,
supported-pickup simulation, kinematic replay and real robot results.

The 45 mm assembly study and 40 mm cube pool are different geometries. Recorded
isaac_closed_q values are achieved simulation states, not hardware closing
commands. The later tabletop guide documents the physical correction.

Repository maintenance does not authorize robot commands, new GPU inference,
simulation trials or calibration experiments. Some GraspGenX imports download
assets; inspect imports before treating a tool's help command as read-only.
Keep source pins and the locked environment unchanged unless a dependency
upgrade is explicitly part of the task.

Keep running notes and internal proposals private under .local/. Public docs
should describe implemented behavior, technical evidence and limitations.
Record generated-artifact requirements honestly rather than inventing missing
results or replacing them with a different experiment's files.
