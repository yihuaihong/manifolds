<!-- render_findings_partR.py from tests/partR_final.json (part R); every number is read from that JSON; † = mean / count of a JSON per-seed list (readout_ptdom2 mn / wn) -->
**ptdom2 Part R 结果**（曲线面积，%；Holm 族 = ['landmarks', 'compass', 'prefixes', 'ranks', 'scalar_adj']（5 个）；final = True）：
> consistency：picks_R_after1.json agrees；picks_R_after2.json agrees

| 领域 | 批 | 先验@lr 对 none@lr（面积） | 真 − none：均值 [boot90]，两半，赢，Holm p | 真 − 打乱：均值 [boot90]，赢，Holm p | final3：真 − none / 真 − 打乱 | 同 lr 增益 | 判词 | Tier 与标签 |
|---|---|---|---|---|---|---|---|---|
| stars | - | - | - | - | - | - | NOT FINAL：none pick 8e-3 / argmax 1.6e-2 at its outermost lr 1.6e-2 | - |
| landmarks | A | only-landmarks@2e-3 76.6 对 none@8e-3 77.8（打乱 69.5） | −1.2 [−2.8, +0.4]，两半 −1.2 / −1.2，3/8，Holm p 1.000 | **+7.1** [+4.4, +9.7]，7/8，Holm p .011 | −4.3†（1/8†） / +5.7†（6/8†） | @2e-3（ptdom seed 0–3）[+6.6, +6.1, +4.2, +8.8] | NOT SHOWN；几何：GEOMETRY-SPECIFIC；reading 见下 | Tier L，共享位点（见注） |
| compass | B2 | only-compass@1e-3 81.8 对 none@2e-3 79.0（打乱 74.6） | +2.8 [−1.6, +7.1]，两半 −2.4 / +7.9，4/8，Holm p .546 | **+7.1** [+1.6, +12.2]，6/8，Holm p .043 | +3.2†（5/8†） / +6.9†（7/8†） | @2e-3（ptdom seed 0–3）[−2.7, +0.3, −0.7, +13.9] | NOT SHOWN；几何：GEOMETRY-SPECIFIC；reading 见下 | Tier L，FLOOR (both <= 89.7)，真 vs 打乱：FLOOR (both <= 89.7)，先验在最低 lr |
| prefixes | B2 | only-prefixes@1e-3 83.4 对 none@1e-3 84.3（打乱 76.7） | −0.9 [−3.8, +2.1]，两半 +1.1 / −2.9，3/8，Holm p 1.000 | **+6.7** [+3.6, +10.1]，7/8，Holm p .022 | −1.0†（3/8†） / +5.0†（5/8†） | @2e-3（ptdom seed 0–3）[+3.9, +3.0, +0.9, +11.0]；@1e-3（seed 8–15）−0.9† | NOT SHOWN；几何：GEOMETRY-SPECIFIC；reading 见下 | Tier L，FLOOR (both <= 89.5)，真 vs 打乱：FLOOR (both <= 89.5)，先验在最低 lr，共享位点（见注） |
| ranks | B2 | only-ranks@1e-3 93.9 对 none@4e-3 90.1（打乱 78.1） | +3.7 [−0.7, +7.7]，两半 +4.8 / +2.7，7/8，Holm p .546 | **+15.7** [+7.5, +26.7]，8/8，Holm p .043 | −3.2†（3/8†） / +12.0†（5/8†） | @2e-3（ptdom seed 0–3）[+8.3, +7.3, +4.3, +5.3] | NOT SHOWN；几何：GEOMETRY-SPECIFIC；reading 见下 | Tier L，FLOOR (both <= 98.0)，真 vs 打乱：FLOOR (both <= 98.0) |
| scalar_adj | B3 | only-scalar_adj@1e-3 88.3 对 none@4e-3 85.5（打乱 82.4） | +2.8 [−0.8, +6.0]，两半 +2.1 / +3.5，6/8，Holm p .546 | **+5.9** [+3.3, +8.4]，7/8，Holm p .017 | −0.5†（5/8†） / +3.8†（5/8†） | @2e-3（ptdom seed 0–3）[+12.3, +2.5, +12.2, +6.1] | NOT SHOWN；几何：GEOMETRY-SPECIFIC；reading 见下 | Tier L，FLOOR (both <= 92.2)，真 vs 打乱：FLOOR (both <= 92.2)，先验在最低 lr |

描述性（不是条件；† = JSON 逐 seed 列表的均值 / 计数）：
- landmarks：
  - 打乱 − none@8e-3（描述性；lr 混淆：P ≠ N）：面积 −8.30†（0/8†），boot90 MISSING（JSON 不存；读数器只在打印时算，见读数 txt），final3 −10.02†
  - 窗口划分（窗口内 / 窗口外；两者之和 = 面积增益）：对 none +0.62† / −1.80†（窗口内 eval 26 次）；对打乱 +2.95† / +4.17†（窗口内 eval 26 次）；打乱对 none −2.34† / −5.97†（窗口内 eval 26 次）
  - LM 占比 a/(a+b)（cell 中位，seed 8–15）：only-landmarks@2e-3 0.916；only-landmarks-shuf@2e-3 0.915；none@8e-3 1.000
  - aux gate（每 cell 中位 ≥ 0.5）：prior@P s8-11 0.919（最低 run 0.908，裁剪 < 1 的比例 0.000）；prior@P s12-15 0.914（最低 run 0.910，裁剪 < 1 的比例 0.000）；prior@P s8-15 0.916（最低 run 0.908，裁剪 < 1 的比例 0.000）；shuf@P s8-15 (descriptive) 0.915（最低 run 0.893，裁剪 < 1 的比例 0.000）
  - 难题子集 final3：对 none −8.12†，对打乱 +9.40†
  - 真 − none 规则细项：单侧 t p .848，Holm 1.000，双侧 p .304，t95 [−3.7, +1.3]；条件 boot90_lo_gt0 NO，halves_gt0 NO，p_holm_lt NO，wins_ge NO
  - 真 − 打乱 规则细项：单侧 t p .002，Holm .011，双侧 p .004，t95 [+3.1, +11.2]；条件 boot90_lo_gt0 ok，halves_gt0 ok，p_holm_lt ok，wins_ge ok
  - FLOOR 线 70.1（chance 50.0，answer-frequency line 64.0，shorter-option line 50.3，shortcut baseline (every held-out question) 65.1）；真 vs none：无；真 vs 打乱：无
  - 难题子集捷径基线（描述性）：only-landmarks@2e-3 65.6 对基线 9.0（n 2510）；only-landmarks-shuf@2e-3 56.8 对基线 9.0（n 2510）；none@8e-3 73.9 对基线 9.0（n 2510）
  - 打乱目标对真目标的 CKA（8 seed 范围）：0.017 .. 0.057
  - 双路径检查（A2.4，描述性）：8/8 seed 满足
  - reading（读数器原文）：GEOMETRY-SPECIFIC without PASS: true > shuffled, but no gain vs none shown (the shuffled prior may hurt)
  - 注：16/76 items share their site token (>= 20%): a NOT SHOWN reads 'with this site' (ptdom A1.3)
- compass：
  - 打乱 − none@2e-3（描述性；lr 混淆：P ≠ N）：面积 −4.37†（4/8†），boot90 MISSING（JSON 不存；读数器只在打印时算，见读数 txt），final3 −3.73†
  - 窗口划分（窗口内 / 窗口外；两者之和 = 面积增益）：对 none +1.33† / +1.42†（窗口内 eval 26 次）；对打乱 +3.12† / +4.01†（窗口内 eval 26 次）；打乱对 none −1.78† / −2.58†（窗口内 eval 26 次）
  - LM 占比 a/(a+b)（cell 中位，seed 8–15）：only-compass@1e-3 0.944；only-compass-shuf@1e-3 0.951；none@2e-3 1.000
  - aux gate（每 cell 中位 ≥ 0.5）：prior@P s8-11 0.945（最低 run 0.936，裁剪 < 1 的比例 0.000）；prior@P s12-15 0.944（最低 run 0.940，裁剪 < 1 的比例 0.000）；prior@P s8-15 0.944（最低 run 0.936，裁剪 < 1 的比例 0.000）；shuf@P s8-15 (descriptive) 0.951（最低 run 0.932，裁剪 < 1 的比例 0.000）
  - 难题子集 final3：对 none +6.99†，对打乱 +10.30†
  - 真 − none 规则细项：单侧 t p .181，Holm .546，双侧 p .363，t95 [−3.9, +9.5]；条件 boot90_lo_gt0 NO，halves_gt0 NO，p_holm_lt NO，wins_ge NO
  - 真 − 打乱 规则细项：单侧 t p .038，Holm .043，双侧 p .075，t95 [−0.9, +15.2]；条件 boot90_lo_gt0 ok，halves_gt0 ok，p_holm_lt ok，wins_ge ok
  - FLOOR 线 89.7（chance 50.0，answer-frequency line 50.0，shorter-option line 50.0，shortcut baseline (every held-out question) 84.7）；真 vs none：['FLOOR (both <= 89.7)']；真 vs 打乱：['FLOOR (both <= 89.7)']
  - 难题子集捷径基线（描述性）：only-compass@1e-3 86.5 对基线 74.3（n 494）；only-compass-shuf@1e-3 76.5 对基线 74.3（n 494）；none@2e-3 79.9 对基线 74.3（n 494）
  - 打乱目标对真目标的 CKA（8 seed 范围）：0.062 .. 0.125
  - 双路径检查（A2.4，描述性）：8/8 seed 满足
  - reading（读数器原文）：GEOMETRY-SPECIFIC without PASS: true > shuffled, but no gain vs none shown (the shuffled prior may hurt)
  - 注：PRIOR AT ITS LOWEST LR 1e-3 (the prior has no lower extension: its best lr may lie lower; conservative for a PASS)
- prefixes：
  - 打乱 − none@1e-3（描述性）：面积 −7.64†（0/8†），boot90 MISSING（JSON 不存；读数器只在打印时算，见读数 txt），final3 −6.01†
  - 窗口划分（窗口内 / 窗口外；两者之和 = 面积增益）：对 none −0.32† / −0.63†（窗口内 eval 26 次）；对打乱 +2.93† / +3.77†（窗口内 eval 26 次）；打乱对 none −3.24† / −4.40†（窗口内 eval 26 次）
  - LM 占比 a/(a+b)（cell 中位，seed 8–15）：only-prefixes@1e-3 0.974；only-prefixes-shuf@1e-3 0.982；none@1e-3 1.000
  - aux gate（每 cell 中位 ≥ 0.5）：prior@P s8-11 0.975（最低 run 0.961，裁剪 < 1 的比例 0.000）；prior@P s12-15 0.974（最低 run 0.971，裁剪 < 1 的比例 0.000）；prior@P s8-15 0.974（最低 run 0.961，裁剪 < 1 的比例 0.000）；shuf@P s8-15 (descriptive) 0.982（最低 run 0.977，裁剪 < 1 的比例 0.000）
  - 难题子集 final3：对 none −6.93†，对打乱 +17.53†
  - 真 − none 规则细项：单侧 t p .684，Holm 1.000，双侧 p .632，t95 [−5.4, +3.5]；条件 boot90_lo_gt0 NO，halves_gt0 NO，p_holm_lt NO，wins_ge NO
  - 真 − 打乱 规则细项：单侧 t p .007，Holm .022，双侧 p .015，t95 [+1.8, +11.6]；条件 boot90_lo_gt0 ok，halves_gt0 ok，p_holm_lt ok，wins_ge ok
  - FLOOR 线 89.5（chance 50.0，answer-frequency line 83.0，shorter-option line 41.7，shortcut baseline (every held-out question) 84.5）；真 vs none：['FLOOR (both <= 89.5)']；真 vs 打乱：['FLOOR (both <= 89.5)']
  - 难题子集捷径基线（描述性）：only-prefixes@1e-3 64.4 对基线 50.0（n 60）；only-prefixes-shuf@1e-3 45.4 对基线 50.0（n 60）；none@1e-3 65.4 对基线 50.0（n 60）
  - 打乱目标对真目标的 CKA（8 seed 范围）：0.001 .. 0.125
  - 双路径检查（A2.4，描述性）：8/8 seed 满足
  - reading（读数器原文）：GEOMETRY-SPECIFIC without PASS: true > shuffled, but no gain vs none shown (the shuffled prior may hurt)
  - 注：PRIOR AT ITS LOWEST LR 1e-3 (the prior has no lower extension: its best lr may lie lower; conservative for a PASS)；3/11 items share their site token (>= 20%): a NOT SHOWN reads 'with this site' (ptdom A1.3)
- ranks：
  - 打乱 − none@4e-3（描述性；lr 混淆：P ≠ N）：面积 −12.01†（1/8†），boot90 MISSING（JSON 不存；读数器只在打印时算，见读数 txt），final3 −15.23†
  - 窗口划分（窗口内 / 窗口外；两者之和 = 面积增益）：对 none +3.05† / +0.68†（窗口内 eval 26 次）；对打乱 +7.20† / +8.54†（窗口内 eval 26 次）；打乱对 none −4.15† / −7.86†（窗口内 eval 26 次）
  - LM 占比 a/(a+b)（cell 中位，seed 8–15）：only-ranks@1e-3 0.977；only-ranks-shuf@1e-3 0.975；none@4e-3 1.000
  - aux gate（每 cell 中位 ≥ 0.5）：prior@P s8-11 0.977（最低 run 0.973，裁剪 < 1 的比例 0.000）；prior@P s12-15 0.977（最低 run 0.970，裁剪 < 1 的比例 0.000）；prior@P s8-15 0.977（最低 run 0.970，裁剪 < 1 的比例 0.000）；shuf@P s8-15 (descriptive) 0.975（最低 run 0.970，裁剪 < 1 的比例 0.000）
  - 难题子集 final3：对 none -，对打乱 -
  - 真 − none 规则细项：单侧 t p .109，Holm .546，双侧 p .218，t95 [−2.8, +10.2]；条件 boot90_lo_gt0 NO，halves_gt0 ok，p_holm_lt NO，wins_ge ok
  - 真 − 打乱 规则细项：单侧 t p .021，Holm .043，双侧 p .043，t95 [+0.7, +30.8]；条件 boot90_lo_gt0 ok，halves_gt0 ok，p_holm_lt ok，wins_ge ok
  - FLOOR 线 98.0（chance 50.0，answer-frequency line 90.6，shorter-option line 47.7，shortcut baseline (every held-out question) 93.0）；真 vs none：['FLOOR (both <= 98.0)']；真 vs 打乱：['FLOOR (both <= 98.0)']
  - 难题子集捷径基线（描述性）：only-ranks@1e-3 75.2 对基线 60.0（n 20）；only-ranks-shuf@1e-3 37.1 对基线 60.0（n 20）；none@4e-3 79.4 对基线 60.0（n 20）
  - 打乱目标对真目标的 CKA（8 seed 范围）：0.002 .. 0.082
  - 双路径检查（A2.4，描述性）：8/8 seed 满足
  - reading（读数器原文）：GEOMETRY-SPECIFIC without PASS: true > shuffled, but no gain vs none shown (the shuffled prior may hurt)
- scalar_adj：
  - 打乱 − none@4e-3（描述性；lr 混淆：P ≠ N）：面积 −3.10†（1/8†），boot90 MISSING（JSON 不存；读数器只在打印时算，见读数 txt），final3 −4.32†
  - 窗口划分（窗口内 / 窗口外；两者之和 = 面积增益）：对 none +1.93† / +0.89†（窗口内 eval 26 次）；对打乱 +2.56† / +3.35†（窗口内 eval 26 次）；打乱对 none −0.63† / −2.47†（窗口内 eval 26 次）
  - LM 占比 a/(a+b)（cell 中位，seed 8–15）：only-scalar_adj@1e-3 0.968；only-scalar_adj-shuf@1e-3 0.966；none@4e-3 1.000
  - aux gate（每 cell 中位 ≥ 0.5）：prior@P s8-11 0.968（最低 run 0.967，裁剪 < 1 的比例 0.000）；prior@P s12-15 0.965（最低 run 0.963，裁剪 < 1 的比例 0.000）；prior@P s8-15 0.968（最低 run 0.963，裁剪 < 1 的比例 0.000）；shuf@P s8-15 (descriptive) 0.966（最低 run 0.961，裁剪 < 1 的比例 0.000）
  - 难题子集 final3：对 none −1.06†，对打乱 +7.96†
  - 真 − none 规则细项：单侧 t p .125，Holm .546，双侧 p .250，t95 [−2.5, +8.1]；条件 boot90_lo_gt0 NO，halves_gt0 ok，p_holm_lt NO，wins_ge ok
  - 真 − 打乱 规则细项：单侧 t p .004，Holm .017，双侧 p .008，t95 [+2.1, +9.8]；条件 boot90_lo_gt0 ok，halves_gt0 ok，p_holm_lt ok，wins_ge ok
  - FLOOR 线 92.2（chance 50.0，answer-frequency line 84.4，shorter-option line 46.0，shortcut baseline (every held-out question) 87.2）；真 vs none：['FLOOR (both <= 92.2)']；真 vs 打乱：['FLOOR (both <= 92.2)']
  - 难题子集捷径基线（描述性）：only-scalar_adj@1e-3 64.3 对基线 57.5（n 106）；only-scalar_adj-shuf@1e-3 57.2 对基线 57.5（n 106）；none@4e-3 70.4 对基线 57.5（n 106）
  - 打乱目标对真目标的 CKA（8 seed 范围）：0.097 .. 0.136
  - 双路径检查（A2.4，描述性）：8/8 seed 满足
  - reading（读数器原文）：GEOMETRY-SPECIFIC without PASS: true > shuffled, but no gain vs none shown (the shuffled prior may hurt)
  - 注：PRIOR AT ITS LOWEST LR 1e-3 (the prior has no lower extension: its best lr may lie lower; conservative for a PASS)
