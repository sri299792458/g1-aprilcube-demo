# Right-Dex3 54 mm U broad-face supported-pickup qualification

This is the table-supported Isaac/PhysX result, not a render or
collision-only prediction.

- Trials: **60**
- PASS: **0**
- FAIL: **60**
- Retained by at least two digit chains: **0**
- Hand/table collision: **18**
- U/table contact in final hold: **60**

## Results by selected support

| Support | Trials | PASS | FAIL |
|---|---:|---:|---:|
| broad_minus_y_face_down | 52 | 0 | 52 |
| broad_plus_y_face_down | 8 | 0 | 8 |

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
| insufficient_final_digit_contacts+hand_table_contact+object_on_table_during_final_hold | 18 |
| insufficient_final_digit_contacts+object_on_table_during_final_hold | 42 |

Every concrete trial and its five named phase snapshots remain in
the machine-readable report JSON.

## Conclusion

None of the 60 geometry-clear proposals
passed the complete physical contract for
every broad-face geometry-clear proposal from 4,096 raw candidates. The selected runtime
library is empty for this proposal set, hand, and controller
profile.
