<!-- render_findings_partR.py from mutated (part R); every number is read from that JSON; † = mean / count of a JSON per-seed list (readout_ptdom2 mn / wn) -->
**ptdom2 Part R 结果**（曲线面积，%；Holm 族 = ['landmarks', 'compass', 'prefixes', 'ranks', 'scalar_adj']（5 个）；final = False）：

> **PROVISIONAL**（final = False）
> problems（14 条，去重后 10 条）：B2 only-compass@1e-3 s15: no run；B2 only-compass-shuf@1e-3 s15: no run；B2 only-prefixes@1e-3 s15: no run；B2 only-prefixes-shuf@1e-3 s15: no run；B2 only-ranks@1e-3 s15: no run；B2 none@4e-3 s15: no check.json；B2 only-ranks-shuf@1e-3 s15: no run；B3 only-scalar_adj@1e-3 s15: no run；B3 none@4e-3 s15: no run；B3 only-scalar_adj-shuf@1e-3 s15: no run
> consistency：picks_R_after1.json agrees；picks_R_after2.json agrees

| 领域 | 批 | 先验@lr 对 none@lr（面积） | 真 − none：均值 [boot90]，两半，赢，Holm p | 真 − 打乱：均值 [boot90]，赢，Holm p | final3：真 − none / 真 − 打乱 | 同 lr 增益 | 判词 | Tier 与标签 |
|---|---|---|---|---|---|---|---|---|
| stars | - | - | - | - | - | - | NOT FINAL：none pick 8e-3 / argmax 1.6e-2 at its outermost lr 1.6e-2 | - |
| landmarks | A | only-landmarks@2e-3 MISSING R.rows.landmarks.floor.means 对 none@8e-3 MISSING R.rows.landmarks.floor.means（打乱 69.5） | −1.2 MISSING R.rows.landmarks.rule_gain.boot90，两半 −1.2 / −1.2，3/8，Holm p .848 | +7.1 [+4.4, +9.7]，MISSING R.rows.landmarks.rule_geo.wins/8，Holm p .002 | −4.3†（1/8†） / +5.7†（6/8†） | @2e-3（ptdom seed 0–3）[+6.6, +6.1, +4.2, +8.8] | PROVISIONAL (NOT SHOWN if final; Holm over the 1 complete of the 5 domains (A3.3: the family is reduced))；几何：PROVISIONAL (GEOMETRY-SPECIFIC if final) | Tier MISSING R.rows.landmarks.tier，共享位点（见注） |
| compass | B2 | only-compass@1e-3 对 none@2e-3：-（缺 run） | -（缺 run） | -（缺 run） | -（-） / -（-） | @2e-3（ptdom seed 0–3）[−2.7, +0.3, −0.7, +13.9] | PROVISIONAL (missing runs)；几何：PROVISIONAL (missing runs) | Tier L，先验在最低 lr |
| prefixes | B2 | only-prefixes@1e-3 对 none@1e-3：-（缺 run） | -（缺 run） | -（缺 run） | -（-） / -（-） | @2e-3（ptdom seed 0–3）[+3.9, +3.0, +0.9, +11.0]；@1e-3（seed 8–15）−0.4† | PROVISIONAL (missing runs)；几何：PROVISIONAL (missing runs) | Tier L，先验在最低 lr，共享位点（见注） |
| ranks | B2 | only-ranks@1e-3 对 none@4e-3：-（缺 run） | -（缺 run） | -（缺 run） | -（-） / -（-） | @2e-3（ptdom seed 0–3）[+8.3, +7.3, +4.3, +5.3] | PROVISIONAL (missing runs)；几何：PROVISIONAL (missing runs) | Tier L |
| scalar_adj | B3 | only-scalar_adj@1e-3 对 none@4e-3：-（缺 run） | -（缺 run） | -（缺 run） | -（-） / -（-） | @2e-3（ptdom seed 0–3）[+12.3, +2.5, +12.2, +6.1] | PROVISIONAL (missing runs)；几何：PROVISIONAL (missing runs) | Tier L，先验在最低 lr |

描述性（不是条件；† = JSON 逐 seed 列表的均值 / 计数）：
- landmarks：
  - 打乱 − none@8e-3（描述性；lr 混淆：P ≠ N）：面积 −8.30†（0/8†），boot90 MISSING（JSON 不存；读数器只在打印时算，见读数 txt），final3 −10.02†
  - 窗口划分：MISSING R.rows.landmarks.split
  - LM 占比：MISSING R.rows.landmarks.lm_share
  - aux gate（每 cell 中位 ≥ 0.5）：prior@P s8-11 0.919（最低 run 0.908，裁剪 < 1 的比例 0.000）；prior@P s12-15 0.914（最低 run 0.910，裁剪 < 1 的比例 0.000）；prior@P s8-15 0.916（最低 run 0.908，裁剪 < 1 的比例 0.000）；shuf@P s8-15 (descriptive) 0.915（最低 run 0.893，裁剪 < 1 的比例 0.000）
  - 难题子集 final3：对 none −8.12†，对打乱 +9.40†
  - 真 − none 规则细项：单侧 t p .848，Holm .848，双侧 p .304，t95 [−3.7, +1.3]；条件 boot90_lo_gt0 NO，halves_gt0 NO，p_holm_lt NO，wins_ge NO
  - 真 − 打乱 规则细项：单侧 t p .002，Holm .002，双侧 p .004，t95 [+3.1, +11.2]；条件 boot90_lo_gt0 ok，halves_gt0 ok，p_holm_lt ok，wins_ge ok
  - FLOOR 线 70.1（chance 50.0，answer-frequency line 64.0，shorter-option line 50.3，shortcut baseline (every held-out question) 65.1）；真 vs none：无；真 vs 打乱：无
  - 难题子集捷径基线（描述性）：only-landmarks@2e-3 65.6 对基线 9.0（n 2510）；only-landmarks-shuf@2e-3 56.8 对基线 9.0（n 2510）；none@8e-3 73.9 对基线 9.0（n 2510）
  - 打乱目标对真目标的 CKA（8 seed 范围）：0.017 .. 0.057
  - 双路径检查（A2.4，描述性）：8/8 seed 满足
  - final 时会是：['NOT SHOWN', 'GEOMETRY-SPECIFIC']（Holm over the 1 complete of the 5 domains (A3.3: the family is reduced)）
  - 注：16/76 items share their site token (>= 20%): a NOT SHOWN reads 'with this site' (ptdom A1.3)
- compass：
  - 打乱 − none@2e-3（描述性；lr 混淆：P ≠ N）：-（缺 run）
  - 窗口划分（窗口内 / 窗口外；两者之和 = 面积增益）：对 none -；对打乱 -；打乱对 none -
  - LM 占比 a/(a+b)（cell 中位，seed 8–15）：only-compass@1e-3 0.945；only-compass-shuf@1e-3 0.950；none@2e-3 1.000
  - 难题子集 final3：对 none -，对打乱 -
  - 打乱目标对真目标的 CKA（8 seed 范围）：0.062 .. 0.125
  - 注：PRIOR AT ITS LOWEST LR 1e-3 (the prior has no lower extension: its best lr may lie lower; conservative for a PASS)
- prefixes：
  - 打乱 − none@1e-3（描述性）：-（缺 run）
  - 窗口划分（窗口内 / 窗口外；两者之和 = 面积增益）：对 none -；对打乱 -；打乱对 none -
  - LM 占比 a/(a+b)（cell 中位，seed 8–15）：only-prefixes@1e-3 0.974；only-prefixes-shuf@1e-3 0.982；none@1e-3 1.000
  - 难题子集 final3：对 none -，对打乱 -
  - 打乱目标对真目标的 CKA（8 seed 范围）：0.001 .. 0.125
  - 注：PRIOR AT ITS LOWEST LR 1e-3 (the prior has no lower extension: its best lr may lie lower; conservative for a PASS)；3/11 items share their site token (>= 20%): a NOT SHOWN reads 'with this site' (ptdom A1.3)
- ranks：
  - 打乱 − none@4e-3（描述性；lr 混淆：P ≠ N）：-（缺 run）
  - 窗口划分（窗口内 / 窗口外；两者之和 = 面积增益）：对 none -；对打乱 -；打乱对 none -
  - LM 占比 a/(a+b)（cell 中位，seed 8–15）：only-ranks@1e-3 0.978；only-ranks-shuf@1e-3 0.975；none@4e-3 1.000
  - 难题子集 final3：对 none -，对打乱 -
  - 打乱目标对真目标的 CKA（8 seed 范围）：0.002 .. 0.082
- scalar_adj：
  - 打乱 − none@4e-3（描述性；lr 混淆：P ≠ N）：-（缺 run）
  - 窗口划分（窗口内 / 窗口外；两者之和 = 面积增益）：对 none -；对打乱 -；打乱对 none -
  - LM 占比 a/(a+b)（cell 中位，seed 8–15）：only-scalar_adj@1e-3 0.968；only-scalar_adj-shuf@1e-3 0.967；none@4e-3 1.000
  - 难题子集 final3：对 none -，对打乱 -
  - 打乱目标对真目标的 CKA（8 seed 范围）：0.097 .. 0.136
  - 注：PRIOR AT ITS LOWEST LR 1e-3 (the prior has no lower extension: its best lr may lie lower; conservative for a PASS)
