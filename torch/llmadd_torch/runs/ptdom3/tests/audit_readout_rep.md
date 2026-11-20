# 审查：readout_ptdom3r.py / render_findings_ptdom3r.py（exp-auditor，2026-11-19）

审查对象：runs/ptdom3/readout_ptdom3r.py（sha 417b7f61ce40），runs/ptdom3/render_findings_ptdom3r.py（sha 19667584b357）。
依赖：readout_ptdom3.py 9abd89ad75a7（冻结），pick_pt6v2.py 7a2e73faa8c8。审查档位为探索档，属于新 read-out 代码的一次代码审查。

## 方法

**独立重算脚本**：/scratch/yh6210/tmp_audit/readout_ptdom3r/indep.py。
- 它不 import readout_ptdom3、readout_ptdom3r 或 pick_pt6v2。
- 用钉住的 code/tools/toy3.py 重建 split，参数取 run JSON 里记录的 split_seed、split_mode、train_frac。
- 由 `hits_<d>_held` 自己算出每题准确率，再自己实现 I1/I2、p/N、配对 g、wins、d、boot90（default_rng(0)，B 20000）、final3 g、chance、FLOOR/CEILING、E1/E2/E4/E5、次指标和 n_tf/n_twin。

**比对 1**：对照 tests/readout_rep_partial_20261119T1504.json。只用该时刻状态为 VALID 的 30 个格子。

**比对 2**：对照 15:21Z 的现读。命令为 `--allow_partial --no_crosscheck --out /scratch/yh6210/tmp_audit/readout_ptdom3r/now`，共 31 个 run，此时 hcube s2 已完成。

**结果**：两次比对都是 TOTAL diffs 0，6 个领域全部 EQUAL。
- 比对字段：p、N、配对 seed、逐 seed g、均值、wins、k、d、boot90、final3 g、次指标 g 和 final3、chance、标签、E1/E2/E4/E5、`_cells` 全部值、seed_info n_tf/n_twin。
- 在所有 run、所有领域上，「hits 按 read-out mask 平均」和 curve 值的最大差为 5.6e-17。说明 token 位置和每题 hits 的顺序与 curve 是同一个量。
- 18 个 (domain, seed) 上，各臂的 split_seed、split_mode、train_frac 完全一致，所以配对是同一个 split。
- 每个 seed 的 split 不同，这是设计如此，例如 hcube s1 的 split_seed 为 1310321579。

**自测**：readout_ptdom3r --selftest 23/23 PASS。render --selftest 显式给 JSON 路径时 6/6 PASS；不给路径时崩溃，见次要 m1。

## 委派问题逐条回答

**(1) 打印的数和原始 run JSON 是否一致？**
一致，见上面两次比对，0 个差异。N 的选择和 I1、I2 一致。
- 屏幕期 none lr（2e-3…8e-3）只有 s0，覆盖 1，不到最佳覆盖 3 的一半，不算 present。所以 N 只在 1.6e-2 和 3.2e-2 中选。
- torus、hcube、box5 的 N 都是 1.6e-2。三者 none 均值依次为 68.9 vs 65.1、69.1 vs 67.1、20.2 vs 17.5（1.6e-2 vs 3.2e-2）。
- seed 1、2 的 twin-free mask：
  - hcube 为 66/300/366、72/298/370（twin-free / twin / 合计）；d6 为 31/31、21/42；box5 无 twin。
  - 重建出的 held 题目和 read-out mask 与 run 记录逐题相同（assert 通过）。

**(2) strict 模式会不会在缺 run 时出判定，或在不该拒绝时拒绝？**
- 缺 run 时出判定：不会。REP11 和 SCREEN_NEEDED 共 21 格，任一格不是 VALID/FLAGGED 都拒绝。
- 应该放行时拒绝：会，见重要 I3。
- 现状：only-box5@1e-3 s1/s2 的目录还不存在，属于材料缺失，不是代码问题。

**(3) 和 pick_pt6v2 的 cross-check 是否比的是同一字段？**
是同一字段。但它只是一致性检查：两边共用 readout_ptdom3.dom_numbers，规则逻辑也是拷贝来的，所以抓不到原始数或 mask 的错。它也不比 tags、chance、final3、boot90、grade 和次指标。这一缺口本次已由独立重算补上（见上）；终读时再跑一次 indep.py 即可。

**(4) 会不会让真效应消失或造出假效应？**
代码层面没有发现。解读层面有两处，见 I1、I2。

## 阻断

未发现。

## 重要

**I1** render_findings_ptdom3r.py:100：「**结果**」的单位写成「read-out 准确率在 prior window 上的 area」。实际 area 是全部 75 次 eval（step 0–18309）的均值，与 F103 的「曲线面积」同义。
- 读者会把 torus 的 +12.6 当成窗口内的增益。实际窗口内增益是 +44.0。
- 修法：改成「75 次 eval 的曲线面积」。

**I2** torus 的「增益」措辞把两段方向相反的效应混在一起；render_findings_ptdom3r.py:77、:156-158。

逐段重算，配对，每 seed 依次为 s0/s1/s2：

| 领域 | 窗口内（step 916–7323）g | 窗口外 g | final3 g |
|---|---|---|---|
| torus | +29.2/+49.7/+52.9，均值 +44.0 | −2.6/−4.3/−5.3，均值 −4.0 | −12.2，0/3 |
| hcube（twin-free） | 均值 +47.6 | 均值 +6.8，3/3 | +4.6，2/3 |
| box5（只有 s0） | +56.3 | +21.2 | — |

- torus 的 prior 臂在窗口末约 97–99。窗口结束后它自己掉到 80–86（脚本：/scratch/yh6210/tmp_audit/readout_ptdom3r/curves.py），none@1.6e-2 则升到 94–98。
- box5 s0 的 prior 臂从 87.3（step 5750）掉到 38.0。
- hcube 的 prior 臂保持在 96–98。
- :156-158 的限定语「prior window 内 prior 臂更高」没有被计算，只要反号就照模板写出：
  - 目前对 torus 恰好成立；
  - 若出现 mg<0<f3，写出的方向会是反的。
- 建议：
  - read-out 输出 RO3.dom_numbers 已有的 in/out 两段 g；
  - 标题在 torus 处写成「窗口内加速、窗口后回落」，不要只写「增益」。

**I3** readout_ptdom3r.py:378-385 对 REP11 的 INVALID 一律拒绝。pick_pt6v2.py:405-407 会豁免 rr≥1（第二次失败）的格子。所以按登记规则本该出终读时，read-out 永远只能出 PROVISIONAL。
- 检查：构造一个 -r1 INVALID 的 REP11 格子，跑 W 自测。

**I4** readout_ptdom3r.py:435-465：cross-check 只能证明「和 pick 一致」，证明不了「对」。见问题 (3)。

## 次要

**m1** render_findings_ptdom3r.py:193 不给路径时，用 `max(glob("readout_rep_partial_*.json"))` 取文件，会取到 `*.pick.json`。之后在 :115-117 因 `_MISS` 哨兵为真值，在迭代时 TypeError 崩溃；:125 有同样的问题。key 缺失时会崩，不会打出 MISSING。

**m2** readout_ptdom3r.py:415：run_pick 不检查 rc，可能读到旧的 .pick.json。`--root` 也没有传给 pick，pick 固定读 runs/ptdom3/out。

**m3** render :88 把「53M」写死了。

**m4** 标题里的「supports」是 read-out 自己的分级，不是 E1–E5 的 eligible。目前两者一致。

**m5** box5 的 N 在 1.6e-2 和 3.2e-2 之间差距不大（20.2 vs 17.5），按规则没问题。

## 通用陷阱逐项

| 陷阱 | 结论 |
|---|---|
| train/eval 不一致 | 未发现。所有 run 都是 75 次 eval，最后一步 18309；两臂同一个 split。 |
| 记忆 | hcube twin-free g +20.9 大于全集 g +16.6，torus 的 tf g +12.7 与全集 +12.6 接近。效应不来自 triple twin。 |
| dead loss | 只报告：LM share 0.99–1.00，clip 1.00，即 prior 梯度约占更新的 1%，但效应很大。 |
| 单 seed | box5 只有 s0。 |
| token 位置 | 已核对，误差 5.6e-17。 |
| 同名异义 | 有：area 的单位标签，即 I1。 |
| 静默丢样本 | 未发现。31 个 run 全部读入；每个 seed 的 n 都打印出来了（box5 150/153/181、hcube 368/366/370、torus 489/491/482）。 |
| 修 bug 的副作用 | 未发现。 |

## 证据撑得住的最强结论（15:21Z，31 run）

在登记的 pick 规则下比较：调好 lr 的 prior 对调好 lr 的 none，53M toy3，seed 0–2。
- torus：read-out 曲线面积 +12.6 pp，3/3，d 3.50，boot90 [+10.5, +14.8]。
- hcube（twin-free）：+20.9 pp，3/3，d 3.48，boot90 [+16.9, +24.9]。
- 两者 E1–E5 全过。
- 增益集中在 prior window 内：torus +44.0，hcube +47.6。
- torus 的 prior 臂在窗口后回落，final3 为 −12.2（0/3）；hcube 的 final3 为 +4.6（2/3）。
- box5 只有 s0（+33.4），属于初步结果。

15:09 的 render 草稿标题把 hcube 写成 +18.0（2 seeds），已经过时；现在是 3 seeds、+20.9。
