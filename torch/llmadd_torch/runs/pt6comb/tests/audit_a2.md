# pt6comb Amendment A2 审查（对抗式，全新审查员，只读）— 2026-11-10

对象：AUDIT.md「Amendment A2」；readout_pt6comb.py 42d7b6158b04、make_jobs_pt6comb.py 08e197092b2c、select_pt6comb.py 75b4f0c00915
（sha 与 AUDIT A2 表一致）；tests/ 下 A2 相关测试与日志；上一轮审查 /scratch/yh6210/tmp_audit/pt6comb_audit/audit_build.md。
没有读任何 pt6comb run 的数值（审查期间冒烟作业 22595628 开始运行并写了 smoke/ 下的文件，未打开）。
除本文件外没有写项目文件；我的检查脚本和输出都在 /scratch/yh6210/tmp_audit/pt6comb_a2/：

| 脚本 / 输出 | 内容 | 结果 |
|---|---|---|
| chk_cells.py / chk_cells.out | C 格子集合、名字、seed、生成器与读出两份 c_cells、选点不读 seed 8–15、L1 / 冒烟文件逐字节回归 | 10/10 PASS |
| chk_verdict.py / chk_verdict.out | verdict() 与我独立实现的 A2 规则（scipy）在 20 种 (P, N) × 40 个随机世界上对比 | 4800 个判词 0 分歧 |
| sim_a2.py / sim_a2.out | 干扰规则的判定概率、HELPS 零效应误判率（正态 / t3 噪声） | 见第 2 问 |
| mut_a2.py / mut_a2.out | 我自己的 20 条变异（在 ./mut/ 的副本上），看项目 --selftest 和我的双路径能否抓住 | 见第 3 问 |
| selftest_rerun.out | 在当前代码上重跑 --selftest | 56/56，与 tests/selftest_readout.out 逐行相同 |

---------------------------------------------------------------------------------------------------------------------------------

## 结论速览
- **阻断：未发现。** 格子集合、h / u / g 的格子和 seed 配对、HELPS 与「interference not detected」的判定，都和 A2 文字一致（两条独立路径对上）。
  L1 的 24 个作业文件和已提交冒烟的作业 / body 文件都和现在的生成器渲染逐字节相同；其他批次没有被改动。
- **重要（1 条）**：A2 把 h 的 all6 臂从 all6@P 换成了 all6@2e-3，但 A2.1 辅助损失 gate 仍然只查 all6@P。
  all6@2e-3 就算被先验主导，干扰判词照样打印。
- **次要（6 条）**：
  - 选点 note 里还写着过时的 "NO INTERFERENCE"，会写进 picks 文件；
  - 报告表格的 all6 / none 列取的是 P / N 的值，h / u 用的 2e-3 水平没打印；
  - 新增的 16 个 run 会连带卡住 HELPS 的 strict 读出；
  - 自测有几处没咬住（同 lr 那一对缺失、表格里的判词字样、gate 查的是哪一臂、边界）；
  - 标题里的 lr 写法容易误读；
  - 同 lr 的 LM 代价行没打印。

---------------------------------------------------------------------------------------------------------------------------------

## 1. 实现与 AUDIT A2 文字是否一致

**格子集合**
- 每个可达的 (P, N) 都检查了：P ∈ {1e-3, 2e-3, 4e-3, 8e-3}，N ∈ {1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2}，共 20 种。
  - 生成器的 c_cells（make_jobs_pt6comb.py:492–499）和读出的 c_cells（readout_pt6comb.py:541–548）逐项相同；
  - 集合 = {none@N, all6@P, none@2e-3, all6@2e-3, 六个 only-m@2e-3}；
  - 格子数 8 / 9 / 10，即 64 / 72 / 80 个 run；
  - 前两项依次是 none@N、all6@P；
  - 8 个 seed 上名字全部唯一。（chk_cells C1）
- P = N（比如 4e-3 / 4e-3）时两臂不同，不会被合并，得到 80 个 run，正确。
- seed 只取 8–15，和 SEL 不相交；120 个 C 名字没有一个和 L1 / L1x 的名字重复。（C2、C3）
- 用我自己放在审查目录里的 chmod 444 假 picks 文件跑生成器的 cells('C')，6 种 (P, N) 分别得到 72 / 80 / 64 / 72 / 80 / 80 个 run。
  缺了同 lr 那一对的旧式 C 列表会被拒；把生成器的 c_cells 改成不含这一对，也会被 :488 的断言拦下。（C4–C6）
- **seed 8–15 从没被选点读过**：
  - 在 seed 8–15 上植入 none@2e-3、all6@2e-3、all6@4e-3 三种格子，数值取极端，并且有的无效、有的 gate 不过、data_sha 不同；
  - decide() 的输出逐字段不变，integrity(seeds=SEL) 不报问题。（C7、C8）

**配对与判定**
- 我用 scipy 独立实现了 A2 的规则，在每种 (P, N) 下 40 个随机世界（每格每 seed 独立取值）上，和 verdict() 逐项对比：
  - g = all6@P − none@N；h = all6@2e-3 − only-m@2e-3；u = only-m@2e-3 − none@2e-3；都按同一 seed 配对；
  - HELPS；「interference not detected」，以及它是靠哪个子句通过的（margin / p）；
  - u ≤ 0 时 margin 为 None，并打出说明；
  - 同 lr 增益 all6@2e-3 − none@2e-3 的均值和 wins；
  - all6@P − only-m 和它的 lr 标记；lr_conf；
  - headline 的名单；报告里的标题字样。
  - 结果：4800 个判词 **0 分歧**（chk_verdict.out）。
- 判词公式和 AUDIT A2 的登记文字逐字一致：
  - readout_pt6comb.py:526–533：`helps = p < .05 and wins >= 6`；`u_pos = mean u > 0`；`by_m = u_pos and mean h >= -0.5·mean u`；`by_p = p(H1 h<0) >= .05`；判词 = by_m 或 by_p。
  - 标题（:626–627）= "k of 6 manifolds HELPS with no interference detected (at the same lr 2e-3)"，和 A2 第 4 条逐字一致。
- A2 第 2 条说「u ≤ 0 时判词与 A0 的字面读法相同」：已证实。在 224 万次模拟抽样里，A0 和 A2 给出的判词 0 处不同（sim_a2.out 最后一列）。

## 2. 模拟：判定概率
模型：每个 seed 上三臂的噪声相互独立，seed 效应在配对差里抵消。h 和 u 共用同一个 only-m run，所以 corr(h, u) = −0.5。
上一轮审查假设 h 和 u 独立，这一点和它不同。每种情形抽 40,000 次；向量化的规则和真实的 manifold_verdict 在 600 次抽样上 0 处不一致。

表中数字 = P(interference not detected)，括号里是 t3 噪声下的值。

| 情形 | 比较类噪声（u 4.1，差的 sd 4.6） | s4_hard 噪声（u 26，sd 19） | u = 0.5 sd | u = 2 sd |
|---|---|---|---|---|
| all6 把单先验增益全部抹掉（h = −u） | **0.27**（0.20） | 0.04（0.05） | 0.65 | 0.00 |
| 抹掉 75% | 0.50（0.39） | 0.21 | 0.76 | 0.06 |
| 抹掉一半（正好在 margin 上） | 0.73（0.67） | 0.59 | 0.85 | 0.51 |
| 完全不受影响（h = 0） | 0.97（0.98） | 0.99 | 0.96 | 1.00 |
| u = 0，h = 0 | 0.95 | 0.95 | 0.95 | 0.95 |
| u = 0，all6 比两者都低 0.5U | 0.69 | 0.46 | 0.84 | 0.19 |
| u = −0.5U，h = −0.5U | 0.69 | 0.46 | 0.84 | 0.18 |

- u ≤ 0 时只剩 t 子句：
  - 真实 h = 0 时，判「not detected」的概率约 0.95，符合设计；
  - 真实干扰时，检出率完全取决于噪声大小。比较类噪声下，h = −0.5U 时只有约 31% 能检出。
- 比较类领域「全部抹掉」仍有 27% 判「not detected」，和 A1 记录的 0.28 一致。
  A2 改成「not detected」的措辞，并把子句和区间都打印出来，正好对应这一点。规则本身保持不变（Pending 14 维持默认）。
- **HELPS 在零效应下**：
  - 单个流形误判率 0.044（t3：0.042；只看 p < .05 是 0.049，wins ≥ 6 再挡掉一点）。
  - 六个流形不校正、且共用 none@N 和 all6@P 的 run 时，至少一个误判 HELPS 的概率：
    - 正态噪声：run 内相关 0 时为 0.25，相关 0.5 时为 0.18；
    - t3 噪声：0.22 / 0.17。
  - 用 Holm 时（相关 0）为 0.049。
  - 这些都在已登记的「不校正，Holm 只作参考列」范围内，这里只给出数字。

## 3. 测试和变异能不能咬住
- 项目自带的 6 条 A2 变异，我核对过替换的文本都只出现一次，日志里全部被抓（mutation_readout.out 31/31）。
- 我自己的 20 条变异（mut_a2.out）：
  - **自测抓到 9 条**：
    - 用格子映射把 h 换到 all6@P；
    - 用格子映射把 u 换到对 none@N；
    - 同 lr 增益改成对 none@N；
    - u ≤ 0 的说明永远不打；
    - margin 用 g 算；
    - u_pos 用 g 判；
    - 标题字样改回 NO INTERFERENCE；
    - C 列表顺序调换；
    - u 的符号反了。
  - **自测没抓到、我的双路径抓到 2 条**：
    - h_at_P 的 lr 标记改成 P ≠ N；
    - h 被标成「lr-confounded」（A1 的旧标签）。
  - **两者都没抓到 9 条**：
    - 三条边界改动：`>=` 改成 `>`、`u > 0` 改成 `>= 0`、`p >= .05` 改成 `> .05`。只影响恰好落在边界上的情形，对连续数据几乎不会发生。
    - **缺失列表不含同 lr 的那一对**（`uniq`，readout_pt6comb.py:557–558）。见次要 m4。
    - final3 的 h / u、secondary 的 h 回到 P / N。这些都只是描述行。
    - **表格里的判词字样改回 "NO INTERFERENCE"**（:671）。V21 只检查 HEADLINE 那一段（:1061）。
    - **gate 改查 all6@2e-3 而不是 all6@P**。V12 用的是 P = 2e-3，这时两者是同一个格子，测试分不出来。
- 结论：决定判词的路径都被钉住了，包括 g / h / u 用哪个格子、margin、u_pos、t 的方向、合并、C 列表。
  没钉住的是描述行、有效性守卫和措辞，见 m4。

## 4. 副作用 / 回归
- **L1 的 24 个作业文件**：磁盘上的 mtime 是 17:05，在 A2 之前。用 A2 版生成器的 render6 在进程内重新渲染（不写盘），和磁盘文件**逐字节相同**（R1）。
- **已提交的冒烟**：body_*.sh 两个文件和 job_pt6comb-smoke.sbatch 与重新渲染逐字节相同（R2）。
  smoke_check 用到的 RP.series_of 可以正常导入，自测 56/56。
- **其他批次**：17:10 之后，runs/pt6comb 之外被改动的文件只有 make_jobs_pt6comb.py，以及一个无关的 watch/watch2_q.txt。
  make_jobs_ptdom.py 6061b299129c、make_jobs_ptdom2.py 62d680144854、readout_ptdom.py 1e2f17522e47 在导入时都通过了 sha 断言。没有新生成的 .pyc。
- tests/*.out 的时间都晚于两份代码最后一次修改（readout 17:57:42、生成器 17:58:11），只有 selftest_readout.out（17:57:46）早于生成器那次修改。我重跑的结果和它逐行相同。
- 文字残留：submit_pt6comb.sh:73 打印的提示还是「64 runs」（不是断言，不影响提交）；AUDIT §3.5 / §4 / §10 的旧文字已被「A2 holds」覆盖。

---------------------------------------------------------------------------------------------------------------------------------

## 问题清单

### 阻断
未发现。

### 重要
**I1. h 用的 all6@2e-3 不过 A2.1 辅助损失 gate；A2 把 h 换了臂，但没把 gate 一起带过去。**
- 位置：
  - readout_pt6comb.py:603–606：每半 seed 各做一次 gate，只查 `get("all6", s)`，也就是 all6@P；
  - :571：h 用的是 `all62`；
  - :608–618：all6@2e-3 的 LM 份额只作描述打印，报告里也不写 ok / FAIL；
  - AUDIT A2 对此没有说明。
- 为什么：
  - A0 / A1 里 h 用的是 all6@P，按两半 seed 过 gate，任一半失败就不出判词。
  - A2 之后，P ≠ 2e-3 时 h 的 all6 臂没有任何 gate。
  - 如果 all6@2e-3 在 L1 就因为份额 < 0.5 被选点排除，A2 仍会在 C 里跑它，并用它出「INTERFERENCE / not detected」。这时的判词是关于一个被先验主导的臂，违背「先验只是辅助 loss」这条原则。
  - Pending 7 只把 only-m 的 gate 定成了描述性，没有覆盖 all6。
- 验证：
  - 读代码可以确认；
  - 变异 "gate per half on all6@2e-3 instead of all6@P" 在自测和我的双路径下都没被抓到，因为 V12 用的 P = 2e-3。
- 建议（协调者定，最好在 picks 写出之前）：
  - all6@2e-3 也按两半 seed 过 gate，失败就打印 "all6@2e-3 GATE"，不出干扰判词；
  - 或者明确登记为描述性（和 Pending 7 一样），并在报告里打印 ok / FAIL；
  - 另外加一条 P ≠ 2e-3 的自测。
- 置信度：读代码确认。影响取决于 all6@2e-3 的 LM 份额（未知）。

### 次要
**m1. 选点 note 里残留 "NO INTERFERENCE"，而且内容已经过时。**
- 位置：readout_pt6comb.py:474–476，「PRIOR AT ITS LOWEST LR … conservative for HELPS and for NO INTERFERENCE」。
- 为什么：
  - A2 之后 h 和 P 无关，所以「for NO INTERFERENCE」不再成立；A2 第 4 条也不允许再用这个词。
  - 这条 note 会写进 picks 文件（strip() 包含 notes），也会出现在报告的 selection 段。
  - **必须在 select_pt6comb.py 写 picks_after_L1.json 之前改**，写出之后再改，--verify 会判 DIFFERENT。
  - 同类残留还有 docstring :28 和 submit_pt6comb.sh:73 的「64 runs」。
- 置信度：确认。

**m2. 报告表格的 all6 / none / only 三列是 all6@P、none@N 的水平，h / u 却是用 2e-3 的水平算的；2e-3 的两个水平已经算了（:582 的 levels），但没打印。**
- 位置：readout_pt6comb.py:676–679。
- 为什么：P ≠ 2e-3 时，读者用表格里的数做不出 h / u，容易把 all6 − only 当成 h。
- 建议：加两列 all6@2e-3 / none@2e-3，或者给每列标上 lr。
- 置信度：确认。

**m3. strict 读出现在也要求 none@2e-3 / all6@2e-3 都齐全且有效，HELPS 会被它不用的 16 个 run 卡住。**
- 位置：readout_pt6comb.py:557–563、:775–780。
- 为什么：
  - 已经完成的 run 永不重跑（上一轮审查 m4）。所以同 lr 那一对里只要有一个 run 完成了但无效，全部判词都只能是 PROVISIONAL。
  - 这是流程问题，不影响测量本身。
- 置信度：确认。

**m4. 自测没钉住的几处（见第 3 问）。**
- **缺失列表不含同 lr 那一对**：
  - 这样改之后，一个完成但无效的 all6@2e-3 run 会被悄悄拿来算 h：
    - load() 对所有完成的 run 都算指标，不管是否有效（:379–382）；
    - get() 不检查有效性；
    - 唯一的守卫就是 `uniq` 里包含这一对。
  - 现在的代码是对的，但没有测试。
- **表格里的判词字样**：可以悄悄改回 "NO INTERFERENCE" 而不被发现，V21 只查 HEADLINE 段。
- **gate 查哪一臂**：见 I1。
- 建议加三条自测：
  - P、N 都 ≠ 2e-3 时，删掉或设无效一个 all6@2e-3 run，期望它出现在 missing 里；
  - 整份报告不含 "NO INTERFERENCE"；
  - P ≠ 2e-3 时 h_at_P 和 lr_conf 的标记。
- 置信度：变异实测。

**m5. 标题「HELPS with no interference detected (at the same lr 2e-3)」容易被读成 HELPS 也是在 2e-3 下判的。**
- 为什么：
  - 实际上 HELPS 用的是 all6@P − none@N。P ≠ 2e-3 时，一个流形可以被计入标题，同时它的同 lr 增益 all6@2e-3 − none@2e-3 约为 0。
  - 同 lr 增益那一行已经打印，但标题本身不提示这一点。
  - 措辞是 A2 原文规定的，只建议在旁边注明 "HELPS at all6@P vs none@N"。
- 置信度：确认（属于读法风险）。

**m6. LM 代价没有同 lr 的行。**
- 位置：readout_pt6comb.py:608–618。
- 为什么：
  - seed 8–15 上现在总是有 all6@2e-3 和 none@2e-3，但 all6@2e-3 的 LM 代价是和 none@N 比的，打的是「lr-confounded」，同 lr 的那一行没打印。Pending 11 的前提已经变了。
  - none@2e-3 那一行是对 none@N 比的，内容纯粹是 lr 的效应。
  - 只影响描述。
- 置信度：确认。

## 一句话
A2 的实现和登记文字一致，回归干净：两条独立路径 0 分歧，L1 和冒烟文件逐字节不变，选点仍然读不到 seed 8–15。
唯一实质性的缺口是：换到 all6@2e-3 之后，h 那一臂的 gate 没有跟着换（I1）。
另外有一条过时的 note 会写进 picks 文件（m1），必须在 select 之前处理。
