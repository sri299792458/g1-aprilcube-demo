# Right-Dex3 58.5 mm U broad-face supported-pickup qualification

This is the table-supported Isaac/PhysX result, not a render or
collision-only prediction.

- Trials: **98**
- PASS: **1**
- FAIL: **97**
- Retained by at least two digit chains: **1**
- Hand/table collision: **12**
- U/table contact in final hold: **97**

## Results by selected support

| Support | Trials | PASS | FAIL |
|---|---:|---:|---:|
| broad_minus_y_face_down | 90 | 0 | 90 |
| broad_plus_y_face_down | 8 | 1 | 7 |

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
| insufficient_final_digit_contacts+hand_table_contact+object_on_table_during_final_hold | 12 |
| insufficient_final_digit_contacts+object_on_table_during_final_hold | 85 |
| pass | 1 |

Every concrete trial and its five named phase snapshots remain in
the machine-readable report JSON.

## Conclusion

1 of 98 geometry-clear proposals passed the complete physical
contract for every broad-face geometry-clear proposal from 4,096 raw candidates. These
records form the support-conditioned physics library; they
are eligible for family construction and later cuRobo
reachability checks, but are not automatically executable.
