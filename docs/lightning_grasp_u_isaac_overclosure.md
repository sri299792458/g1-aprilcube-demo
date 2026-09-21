# Lightning-Grasp Dex3 broad-face U overclosure diagnostic

This experiment tests the exact final pose and seven-joint configuration
returned by Lightning Grasp. The hand base does not execute an invented
pregrasp approach: it remains at the final pose, closes from the standard
Dex3 open configuration toward Lightning's `q`, lifts 20 cm, and holds under
gravity.

- Trials: **20**
- Full PASS: **0**
- Retained with at least two digit groups: **0**
- Any hand/table contact: **0**
- Overclosure scales: **[0.05, 0.1, 0.2, 0.35, 0.5]**
- Review video: [`lightning_grasp_dex3_u_overclosure_isaac20.mp4`](assets/lightning_grasp_dex3_u_overclosure_isaac20.mp4)

A PASS establishes table-supported close/lift retention for this final
grasp. It does not establish an arm-reachable collision-free pregrasp or
approach trajectory, because Lightning Grasp does not return one.

| Candidate | Support | PASS | Two digit groups | Hand/table | Object/table at hold |
|---|---|---:|---:|---:|---:|
| `lightning_u_posY_candidate_0000_overclose_0050` | broad_minus_y_face_down | no | no | no | yes |
| `lightning_u_posY_candidate_0000_overclose_0100` | broad_minus_y_face_down | no | no | no | yes |
| `lightning_u_posY_candidate_0000_overclose_0200` | broad_minus_y_face_down | no | no | no | yes |
| `lightning_u_posY_candidate_0000_overclose_0350` | broad_minus_y_face_down | no | no | no | yes |
| `lightning_u_posY_candidate_0000_overclose_0500` | broad_minus_y_face_down | no | no | no | yes |
| `lightning_u_posY_candidate_0001_overclose_0050` | broad_minus_y_face_down | no | no | no | yes |
| `lightning_u_posY_candidate_0001_overclose_0100` | broad_minus_y_face_down | no | no | no | yes |
| `lightning_u_posY_candidate_0001_overclose_0200` | broad_minus_y_face_down | no | no | no | yes |
| `lightning_u_posY_candidate_0001_overclose_0350` | broad_minus_y_face_down | no | no | no | yes |
| `lightning_u_posY_candidate_0001_overclose_0500` | broad_minus_y_face_down | no | no | no | yes |
| `lightning_u_negY_candidate_0052_overclose_0050` | broad_plus_y_face_down | no | no | no | yes |
| `lightning_u_negY_candidate_0052_overclose_0100` | broad_plus_y_face_down | no | no | no | yes |
| `lightning_u_negY_candidate_0052_overclose_0200` | broad_plus_y_face_down | no | no | no | yes |
| `lightning_u_negY_candidate_0052_overclose_0350` | broad_plus_y_face_down | no | no | no | yes |
| `lightning_u_negY_candidate_0052_overclose_0500` | broad_plus_y_face_down | no | no | no | yes |
| `lightning_u_negY_candidate_0053_overclose_0050` | broad_plus_y_face_down | no | no | no | yes |
| `lightning_u_negY_candidate_0053_overclose_0100` | broad_plus_y_face_down | no | no | no | yes |
| `lightning_u_negY_candidate_0053_overclose_0200` | broad_plus_y_face_down | no | no | no | yes |
| `lightning_u_negY_candidate_0053_overclose_0350` | broad_plus_y_face_down | no | no | no | yes |
| `lightning_u_negY_candidate_0053_overclose_0500` | broad_plus_y_face_down | no | no | no | yes |
