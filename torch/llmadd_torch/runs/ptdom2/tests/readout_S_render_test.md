<!-- render_findings_partR.py from ../readout_S.json (part S); every number is read from that JSON; † = mean / count of a JSON per-seed list (readout_ptdom2 mn / wn) -->
**ptdom2 Part S 结果**（曲线面积，%；Holm 族 = ['countries', 'people', 'cities', 'months']（4 个）；final = True）：

| 领域 | 批 | 先验@lr 对 none@lr（面积） | 真 − none：均值 [boot90]，两半，赢，Holm p | 真 − 打乱：均值 [boot90]，赢，Holm p | final3：真 − none / 真 − 打乱 | 同 lr 增益 | 判词 | Tier 与标签 |
|---|---|---|---|---|---|---|---|---|
| 国家 | B1 | only-countries@2e-3 93.6 对 none@2e-3 88.8（打乱 78.8） | **+4.8** [+3.1, +6.8]，两半 +6.1 / +3.5，8/8，Holm p .006 | **+14.8** [+13.7, +16.0]，8/8，Holm p < .001 | +3.7†（8/8†） / +8.4†（8/8†） | @2e-3（ptdom seed 0–3）MISSING S.rows.countries.same_lr_2e3_seeds0_3 | PASS；几何：GEOMETRY-SPECIFIC；reading 见下 | Tier MISSING S.rows.countries.tier |
| 人物 | B1 | only-people@2e-3 93.8 对 none@2e-3 92.6（打乱 91.9） | **+1.2** [+0.9, +1.6]，两半 +1.1 / +1.3，8/8，Holm p .002 | **+1.9** [+1.6, +2.3]，8/8，Holm p < .001 | +0.7†（6/8†） / +1.5†（7/8†） | @2e-3（ptdom seed 0–3）MISSING S.rows.people.same_lr_2e3_seeds0_3 | PASS；几何：GEOMETRY-SPECIFIC；reading 见下 | Tier MISSING S.rows.people.tier，FLOOR (both <= 99.9)，真 vs 打乱：FLOOR (both <= 99.9) |
| 城市 | A | only-cities@2e-3 88.9 对 none@4e-3 84.8（打乱 72.2） | **+4.1** [+1.5, +6.7]，两半 +5.7 / +2.6，7/8，Holm p .022 | **+16.7** [+14.1, +19.7]，8/8，Holm p < .001 | +2.8†（6/8†） / +13.1†（8/8†） | same-lr PASS 行（descriptive: only-cities@2e-3 - none@2e-3 (not F100's contrast: F100's none pick is 4e-3; raw p, no Holm)）：PASS，+3.4 [+2.0, +4.9]，7/8，p（未校正）.004 | PASS；几何：GEOMETRY-SPECIFIC；reading 见下 | Tier MISSING S.rows.cities.tier |
| 月份 | A | only-months@2e-3 5.0 对 none@4e-3 0.7（打乱 0.7） | **+4.3** [+3.2, +5.2]，两半 +4.0 / +4.6，8/8，Holm p < .001 | **+4.3** [+3.3, +5.2]，8/8，Holm p < .001 | +1.5†（7/8†） / +1.5†（7/8†） | same-lr PASS 行（descriptive: only-months@2e-3 - none@2e-3 (not F100's contrast: F100's none pick is 4e-3; raw p, no Holm)）：PASS，+4.3 [+3.2, +5.3]，8/8，p（未校正）< .001 | PASS；几何：GEOMETRY-SPECIFIC；reading 见下 | Tier MISSING S.rows.months.tier，FLOOR (both <= 13.3)，真 vs 打乱：FLOOR (both <= 13.3) |

描述性（不是条件；† = JSON 逐 seed 列表的均值 / 计数）：
- 国家：
  - 打乱 − none@2e-3（描述性）：面积 −10.03†（0/8†），boot90 MISSING（JSON 不存；读数器只在打印时算，见读数 txt），final3 −4.73†
  - 窗口划分（窗口内 / 窗口外；两者之和 = 面积增益）：对 none +2.56† / +2.23†（窗口内 eval 26 次）；对打乱 +8.13† / +6.68†（窗口内 eval 26 次）；打乱对 none −5.57† / −4.45†（窗口内 eval 26 次）
  - LM 占比 a/(a+b)（cell 中位，seed 8–15）：only-countries@2e-3 0.902；only-countries-shuf@2e-3 0.892；none@2e-3 1.000
  - aux gate（每 cell 中位 ≥ 0.5）：prior@P s8-11 0.896（最低 run 0.892，裁剪 < 1 的比例 0.000）；prior@P s12-15 0.906（最低 run 0.881，裁剪 < 1 的比例 0.000）；prior@P s8-15 0.902（最低 run 0.881，裁剪 < 1 的比例 0.000）；shuf@P s8-15 (descriptive) 0.892（最低 run 0.868，裁剪 < 1 的比例 0.000）
  - 难题子集 final3：对 none +8.66†，对打乱 +18.13†
  - 真 − none 规则细项：单侧 t p .003，Holm .006，双侧 p .006，t95 [+1.9, +7.7]；条件 boot90_lo_gt0 ok，halves_gt0 ok，p_holm_lt ok，wins_ge ok
  - 真 − 打乱 规则细项：单侧 t p < .001，Holm < .001，双侧 p < .001，t95 [+13.1, +16.6]；条件 boot90_lo_gt0 ok，halves_gt0 ok，p_holm_lt ok，wins_ge ok
  - FLOOR 线 77.2（chance 50.0，answer-frequency line 70.5，shorter-option line 49.5，shortcut baseline (every held-out question) 72.2）；真 vs none：无；真 vs 打乱：无
  - 难题子集捷径基线（描述性）：only-countries@2e-3 93.8 对基线 12.4（n 2332）；only-countries-shuf@2e-3 75.5 对基线 12.4（n 2332）；none@2e-3 85.3 对基线 12.4（n 2332）
  - 打乱目标对真目标的 CKA（8 seed 范围）：0.021 .. 0.059
  - 双路径检查（A2.4，描述性）：8/8 seed 满足
  - reading（读数器原文）：PASS and GEOMETRY-SPECIFIC, but the geometry-dependence sentence is withheld (A3.2 / A4): the shuffled arm got worse: shuffled - none@2e-3 mean -10.03, boot90 [-12.41, -7.24] (the upper bound must be >= 0, A4.2a) | UNDETERMINED at n = 8: a harm of the shuffled arm is not excluded: shuffled - none@2e-3 boot90 [-12.41, -7.24] (the lower bound must be >= -1, A5c)
  - reading 没印出的原因：the shuffled arm got worse: shuffled - none@2e-3 mean -10.03, boot90 [-12.41, -7.24] (the upper bound must be >= 0, A4.2a)；UNDETERMINED at n = 8: a harm of the shuffled arm is not excluded: shuffled - none@2e-3 boot90 [-12.41, -7.24] (the lower bound must be >= -1, A5c)
- 人物：
  - 打乱 − none@2e-3（描述性）：面积 −0.71†（1/8†），boot90 MISSING（JSON 不存；读数器只在打印时算，见读数 txt），final3 −0.86†
  - 窗口划分（窗口内 / 窗口外；两者之和 = 面积增益）：对 none +0.63† / +0.58†（窗口内 eval 26 次）；对打乱 +0.87† / +1.06†（窗口内 eval 26 次）；打乱对 none −0.24† / −0.47†（窗口内 eval 26 次）
  - LM 占比 a/(a+b)（cell 中位，seed 8–15）：only-people@2e-3 0.976；only-people-shuf@2e-3 0.977；none@2e-3 1.000
  - aux gate（每 cell 中位 ≥ 0.5）：prior@P s8-11 0.975（最低 run 0.974，裁剪 < 1 的比例 0.000）；prior@P s12-15 0.978（最低 run 0.974，裁剪 < 1 的比例 0.000）；prior@P s8-15 0.976（最低 run 0.974，裁剪 < 1 的比例 0.000）；shuf@P s8-15 (descriptive) 0.977（最低 run 0.973，裁剪 < 1 的比例 0.000）
  - 难题子集 final3：对 none +3.42†，对打乱 +8.19†
  - 真 − none 规则细项：单侧 t p < .001，Holm .002，双侧 p .001，t95 [+0.7, +1.7]；条件 boot90_lo_gt0 ok，halves_gt0 ok，p_holm_lt ok，wins_ge ok
  - 真 − 打乱 规则细项：单侧 t p < .001，Holm < .001，双侧 p < .001，t95 [+1.4, +2.5]；条件 boot90_lo_gt0 ok，halves_gt0 ok，p_holm_lt ok，wins_ge ok
  - FLOOR 线 99.9（chance 50.0，answer-frequency line 91.7，shorter-option line 46.6，shortcut baseline (every held-out question) 94.9）；真 vs none：['FLOOR (both <= 99.9)']；真 vs 打乱：['FLOOR (both <= 99.9)']
  - 难题子集捷径基线（描述性）：only-people@2e-3 77.5 对基线 64.1（n 580）；only-people-shuf@2e-3 69.1 对基线 64.1（n 580）；none@2e-3 74.1 对基线 64.1（n 580）
  - 打乱目标对真目标的 CKA（8 seed 范围）：0.001 .. 0.046
  - 双路径检查（A2.4，描述性）：8/8 seed 满足
  - reading（读数器原文）：PASS and GEOMETRY-SPECIFIC, but the geometry-dependence sentence is withheld (A3.2 / A4): FLOOR: FLOOR (both <= 99.9); FLOOR (both <= 99.9) | the shuffled arm got worse: shuffled - none@2e-3 mean -0.71, boot90 [-1.07, -0.37] (the upper bound must be >= 0, A4.2a) | UNDETERMINED at n = 8: a harm of the shuffled arm is not excluded: shuffled - none@2e-3 boot90 [-1.07, -0.37] (the lower bound must be >= -1, A5c)
  - reading 没印出的原因：FLOOR: FLOOR (both <= 99.9); FLOOR (both <= 99.9)；the shuffled arm got worse: shuffled - none@2e-3 mean -0.71, boot90 [-1.07, -0.37] (the upper bound must be >= 0, A4.2a)；UNDETERMINED at n = 8: a harm of the shuffled arm is not excluded: shuffled - none@2e-3 boot90 [-1.07, -0.37] (the lower bound must be >= -1, A5c)
- 城市：
  - 打乱 − none@2e-3（描述性）：面积 −13.31†（0/8†），boot90 MISSING（JSON 不存；读数器只在打印时算，见读数 txt），final3 −11.14†
  - 窗口划分（窗口内 / 窗口外；两者之和 = 面积增益）：对 none +2.07† / +2.06†（窗口内 eval 26 次）；对打乱 +7.43† / +9.29†（窗口内 eval 26 次）；打乱对 none −5.14† / −8.18†（窗口内 eval 26 次）
  - LM 占比 a/(a+b)（cell 中位，seed 8–15）：only-cities@2e-3 0.905；only-cities-shuf@2e-3 0.897；none@4e-3 1.000
  - aux gate（每 cell 中位 ≥ 0.5）：prior@P s8-11 0.901（最低 run 0.881，裁剪 < 1 的比例 0.000）；prior@P s12-15 0.906（最低 run 0.874，裁剪 < 1 的比例 0.000）；prior@P s8-15 0.905（最低 run 0.874，裁剪 < 1 的比例 0.000）；shuf@P s8-15 (descriptive) 0.897（最低 run 0.863，裁剪 < 1 的比例 0.000）
  - 难题子集 final3：对 none +6.52†，对打乱 +25.40†
  - 真 − none 规则细项：单侧 t p .022，Holm .022，双侧 p .044，t95 [+0.1, +8.1]；条件 boot90_lo_gt0 ok，halves_gt0 ok，p_holm_lt ok，wins_ge ok
  - 真 − 打乱 规则细项：单侧 t p < .001，Holm < .001，双侧 p < .001，t95 [+12.5, +20.9]；条件 boot90_lo_gt0 ok，halves_gt0 ok，p_holm_lt ok，wins_ge ok
  - FLOOR 线 73.6（chance 50.0，answer-frequency line 66.1，shorter-option line 53.1，shortcut baseline (every held-out question) 68.6）；真 vs none：无；真 vs 打乱：无
  - 难题子集捷径基线（描述性）：only-cities@2e-3 85.6 对基线 13.7（n 2262）；only-cities-shuf@2e-3 60.4 对基线 13.7（n 2262）；none@4e-3 79.3 对基线 13.7（n 2262）；none@2e-3 82.3 对基线 13.7（n 2262）
  - 打乱目标对真目标的 CKA（8 seed 范围）：0.020 .. 0.095
  - 双路径检查（A2.4，描述性）：8/8 seed 满足
  - reading（读数器原文）：PASS and GEOMETRY-SPECIFIC, but the geometry-dependence sentence is withheld (A3.2 / A4): only learns faster: final3 gain vs none@4e-3 mean +2.78, boot90 lower -0.60; vs shuffled mean +13.10, boot90 lower +8.89 (both paired bootstrap 90% lower bounds must be > 0, A4.1) | the shuffled arm got worse: shuffled - none@2e-3 mean -13.31, boot90 [-16.12, -10.51] (the upper bound must be >= 0, A4.2a) | UNDETERMINED at n = 8: a harm of the shuffled arm is not excluded: shuffled - none@2e-3 boot90 [-16.12, -10.51] (the lower bound must be >= -1, A5c)
  - reading 没印出的原因：only learns faster: final3 gain vs none@4e-3 mean +2.78, boot90 lower -0.60; vs shuffled mean +13.10, boot90 lower +8.89 (both paired bootstrap 90% lower bounds must be > 0, A4.1)；the shuffled arm got worse: shuffled - none@2e-3 mean -13.31, boot90 [-16.12, -10.51] (the upper bound must be >= 0, A4.2a)；UNDETERMINED at n = 8: a harm of the shuffled arm is not excluded: shuffled - none@2e-3 boot90 [-16.12, -10.51] (the lower bound must be >= -1, A5c)
- 月份：
  - 打乱 − none@2e-3（描述性）：面积 −0.02†（3/8†），boot90 MISSING（JSON 不存；读数器只在打印时算，见读数 txt），final3 −0.06†
  - 窗口划分（窗口内 / 窗口外；两者之和 = 面积增益）：对 none +2.97† / +1.30†（窗口内 eval 26 次）；对打乱 +2.86† / +1.41†（窗口内 eval 26 次）；打乱对 none +0.05† / −0.07†（窗口内 eval 26 次）
  - LM 占比 a/(a+b)（cell 中位，seed 8–15）：only-months@2e-3 0.827；only-months-shuf@2e-3 0.836；none@4e-3 1.000
  - aux gate（每 cell 中位 ≥ 0.5）：prior@P s8-11 0.845（最低 run 0.784，裁剪 < 1 的比例 0.000）；prior@P s12-15 0.819（最低 run 0.782，裁剪 < 1 的比例 0.000）；prior@P s8-15 0.827（最低 run 0.782，裁剪 < 1 的比例 0.000）；shuf@P s8-15 (descriptive) 0.836（最低 run 0.777，裁剪 < 1 的比例 0.000）
  - 难题子集 final3：对 none -，对打乱 -
  - 真 − none 规则细项：单侧 t p < .001，Holm < .001，双侧 p < .001，t95 [+2.7, +5.8]；条件 boot90_lo_gt0 ok，halves_gt0 ok，p_holm_lt ok，wins_ge ok
  - 真 − 打乱 规则细项：单侧 t p < .001，Holm < .001，双侧 p < .001，t95 [+2.8, +5.8]；条件 boot90_lo_gt0 ok，halves_gt0 ok，p_holm_lt ok，wins_ge ok
  - FLOOR 线 13.3（chance 8.3，shortest-span line 8.3）；真 vs none：['FLOOR (both <= 13.3)']；真 vs 打乱：['FLOOR (both <= 13.3)']
  - 打乱目标对真目标的 CKA（8 seed 范围）：0.056 .. 0.125
  - 双路径检查（A2.4，描述性）：8/8 seed 满足
  - reading（读数器原文）：PASS and GEOMETRY-SPECIFIC, but the geometry-dependence sentence is withheld (A3.2 / A4): FLOOR: FLOOR (both <= 13.3); FLOOR (both <= 13.3)
  - reading 没印出的原因：FLOOR: FLOOR (both <= 13.3); FLOOR (both <= 13.3)
