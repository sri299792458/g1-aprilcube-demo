# U support-conditioned right-Dex3 proposal audit

This is a pre-physics geometric audit. It does **not** claim that any
candidate closes on, lifts, or retains the U.

## Scope

- Raw immutable GraspGenX candidates: **4,096**
- Stable tabletop orientations: **6**
- Candidate/support pairs checked: **24,576**
- Geometric survivors: **6,630**
- Pre-physics proposal buckets: **164**

![Six explicit U tabletop supports](assets/u_legs_six_tabletop_supports.png)

Absolute tabletop height, XY translation, and in-plane yaw are absent
because they do not change object–support clearance. They belong to arm
reachability, not this object-relative support audit.

## Stable supports and gates

| Table-up object axis | Label | Equivalence class | Final table-clear | Pregrasp table-clear | Final object-clear | Full corridor-clear |
|---|---|---|---:|---:|---:|---:|
| +X | left_outer_leg_down | outer_leg_side | 2,241 | 2,175 | 1,695 | 1,691 |
| -X | right_outer_leg_down | outer_leg_side | 2,024 | 1,944 | 1,498 | 1,496 |
| +Y | broad_minus_y_face_down | broad_face | 62 | 61 | 33 | 33 |
| -Y | broad_plus_y_face_down | broad_face | 34 | 34 | 9 | 9 |
| +Z | upright_on_leg_ends | leg_ends | 2,412 | 2,311 | 1,840 | 1,837 |
| -Z | inverted_on_hip_bridge | hip_bridge | 2,160 | 2,072 | 1,570 | 1,564 |

The two broad-face supports, the two outside-leg supports, upright
leg-end support, and inverted hip-bridge support remain separately
auditable. Geometric symmetries are recorded as equivalence classes
rather than silently deleting tag-distinguishable poses.

## Survivor distribution by semantic U component

| Support | Hip bridge | Left leg | Right leg | Unresolved | Total |
|---|---:|---:|---:|---:|---:|
| left_outer_leg_down | 791 | 19 | 872 | 9 | 1,691 |
| right_outer_leg_down | 701 | 781 | 5 | 9 | 1,496 |
| broad_minus_y_face_down | 10 | 9 | 9 | 5 | 33 |
| broad_plus_y_face_down | 3 | 1 | 2 | 3 | 9 |
| upright_on_leg_ends | 1,380 | 242 | 200 | 15 | 1,837 |
| inverted_on_hip_bridge | 47 | 721 | 793 | 3 | 1,564 |

The JSON artifact retains all 164 exact buckets, including surface,
cavity/exterior relation, support relation, approach direction, and
every concrete member ID. Nothing is selected or discarded by the
bucketing stage.

These are proposal buckets, not final grasp families. Final families
must be constructed only after physical closure and retention succeed.

## Physics result and remaining gate

The 42 broad-face survivors have completed the corrected table-supported
Isaac test. The result is **0/42 PASS**: no candidate retained the U after the
20 cm lift, and 14 also contacted the table. The full ledger and video are
`artifacts/grasp_support/u_legs_right_broad_face_isaac_v1/report.json` and
`docs/assets/dex3_u_broad_face_supported_pickup_all42.mp4`.

The broad-face runtime library is therefore empty. The upright-on-leg-ends
support has since produced 365 candidates that passed the same physical
contract twice under different batch layouts. The two outer-leg-side supports
and inverted hip-bridge support remain geometry-only and must pass the same
approach, closure, lift, and hold contract before any of their members may
enter the final object-centric family library.
