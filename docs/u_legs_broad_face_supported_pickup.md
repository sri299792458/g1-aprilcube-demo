# Right-Dex3 broad-face U supported-pickup qualification

This is the table-supported Isaac/PhysX result, not a render or
collision-only prediction.

- Trials: **42**
- PASS: **0**
- FAIL: **42**
- Retained by at least two digit chains: **0**
- Hand/table collision: **14**
- U/table contact in final hold: **42**

Video: [`docs/assets/dex3_u_broad_face_supported_pickup_all42.mp4`](assets/dex3_u_broad_face_supported_pickup_all42.mp4)

## Results by broad-face support

| Support | Trials | PASS | FAIL |
|---|---:|---:|---:|
| broad_minus_y_face_down | 33 | 0 | 33 |
| broad_plus_y_face_down | 9 | 0 | 9 |

## Physical pass contract

A trial passes only when all three statements are true:

1. At least two Dex3 digit chains contact the U after the 20 cm lift
   and one-second final hold.
2. No Dex3 hand link contacted the table during settle, approach,
   closure, lift, or hold.
3. The U did not contact the table during the final elevated hold.

## Verdict combinations

| Verdict/reasons | Trials |
|---|---:|
| insufficient_final_digit_contacts+hand_table_contact+object_on_table_during_final_hold | 14 |
| insufficient_final_digit_contacts+object_on_table_during_final_hold | 28 |

Every concrete trial and its five named phase snapshots remain in
the machine-readable report JSON.

## Conclusion

None of the 42 geometry-clear broad-face proposals is an
admissible supported pickup for this right Dex3 and controller
profile. This rules out these two flat supports for the current
4,096-proposal atlas; it does not claim that a Dex3 can never
pick a flat U with a different grasp generator or manipulation
strategy.

The next candidate orientation should be selected from the
separately enumerated outer-leg, leg-end, or hip-bridge supports
and must pass this same physical contract before entering
cuRobo.
