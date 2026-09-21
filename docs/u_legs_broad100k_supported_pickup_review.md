# Right-Dex3 broad-face U 100K failure review

This is the table-supported Isaac/PhysX result, not a render or
collision-only prediction.

- Trials: **10**
- PASS: **0**
- FAIL: **10**
- Retained by at least two digit chains: **1**
- Hand/table collision: **7**
- U/table contact in final hold: **9**

Video: [`docs/assets/dex3_u_broad100k_supported_pickup_review10.mp4`](assets/dex3_u_broad100k_supported_pickup_review10.mp4)

## Results by selected support

| Support | Trials | PASS | FAIL |
|---|---:|---:|---:|
| broad_minus_y_face_down | 7 | 0 | 7 |
| broad_plus_y_face_down | 3 | 0 | 3 |

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
| hand_table_contact | 1 |
| insufficient_final_digit_contacts+hand_table_contact+object_on_table_during_final_hold | 6 |
| insufficient_final_digit_contacts+object_on_table_during_final_hold | 3 |

Every concrete trial and its five named phase snapshots remain in
the machine-readable report JSON.

## Conclusion

None of these ten visual diagnostics passed. The first segment is the only
member of the exhaustive 983-trial run that retained the U through the final
hold; it is still an invalid pickup because a hand link contacted the table.
The other nine illustrate the dominant failure mode: the hand follows the
commanded lift while the U remains on or returns to the table.

This video is a human-readable diagnostic only. The exhaustive 983-trial
ledger—not this ten-member selection—owns the zero-PASS verdict.
