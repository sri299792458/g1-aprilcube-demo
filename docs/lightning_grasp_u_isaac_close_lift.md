# Lightning-Grasp Dex3 broad-face U Isaac close/lift test

This experiment tests the exact final pose and seven-joint configuration
returned by Lightning Grasp. The hand base does not execute an invented
pregrasp approach: it remains at the final pose, closes from the standard
Dex3 open configuration to Lightning's `q`, lifts 20 cm, and holds under
gravity.

- Trials: **14**
- Full PASS: **0**
- Retained with at least two digit groups: **0**
- Any hand/table contact: **10**
- Review video: [`lightning_grasp_dex3_u_close_lift_isaac14.mp4`](assets/lightning_grasp_dex3_u_close_lift_isaac14.mp4)

A PASS establishes table-supported close/lift retention for this final
grasp. It does not establish an arm-reachable collision-free pregrasp or
approach trajectory, because Lightning Grasp does not return one.

| Candidate | Support | PASS | Two digit groups | Hand/table | Object/table at hold |
|---|---|---:|---:|---:|---:|
| `lightning_u_posY_candidate_0000` | broad_minus_y_face_down | no | no | no | yes |
| `lightning_u_posY_candidate_0001` | broad_minus_y_face_down | no | no | no | yes |
| `lightning_u_posY_candidate_0110` | broad_minus_y_face_down | no | no | yes | yes |
| `lightning_u_posY_candidate_0156` | broad_minus_y_face_down | no | no | yes | yes |
| `lightning_u_posY_candidate_0157` | broad_minus_y_face_down | no | no | yes | yes |
| `lightning_u_posY_candidate_0158` | broad_minus_y_face_down | no | no | yes | yes |
| `lightning_u_posY_candidate_0199` | broad_minus_y_face_down | no | no | yes | yes |
| `lightning_u_posY_candidate_0200` | broad_minus_y_face_down | no | no | yes | yes |
| `lightning_u_posY_candidate_0201` | broad_minus_y_face_down | no | no | yes | yes |
| `lightning_u_negY_candidate_0052` | broad_plus_y_face_down | no | no | no | yes |
| `lightning_u_negY_candidate_0053` | broad_plus_y_face_down | no | no | no | yes |
| `lightning_u_negY_candidate_0094` | broad_plus_y_face_down | no | no | yes | yes |
| `lightning_u_negY_candidate_0100` | broad_plus_y_face_down | no | no | yes | yes |
| `lightning_u_negY_candidate_0104` | broad_plus_y_face_down | no | no | yes | yes |
