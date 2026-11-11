# ptdom Amendment A5 对抗审查（全新审查者，没参与设计；2026-11-04）

**范围**：AUDIT.md 的 Amendment A5（A5.1–A5.5），对照 tests/audit_a4.md 的阻断 1、重要 2、次要 3/4/6；make_jobs_ptdom.py、readout_ptdom.py、select_ptdom.py、submit_ptdom.sh 与 superseded_A4/scripts 的 diff；240 个 stage-1 作业文件与 superseded_A4/jobs；tests/a5/ 的全部脚本和日志；4 个 GPU pilot 的 run JSON。

**纪律与披露（请协调者知悉）**：
- 没有提交、没有取消任何 slurm 作业；runs/ptdom 下没有 HOLD / picks / SUBMITTED_ / out（已核对）。
- **两处非本意的写入，必须说明**：
  1. 我重跑 `tests/a5/jobcheck_a5.py`（为了复核 24/24）时，它在第 52 行调用 `make_jobs_ptdom.py --batch X --stage 1`，**把 runs/ptdom 下 240 个 stage-1 作业文件原地重渲染了一遍**（mtime 变成 11:13–11:14）。内容我核过没变：重渲染前后我各跑了一次自己的逐文件对比脚本 jobs_cmp.py，结论相同（heredoc sha 12dff9487798、intent = A4 + dp_tol0/dp_tol、其余逐行等于 A4 版本），6 个抽查文件的字节数与重渲染前的 ls 一致；生成器是确定性的。除这 240 个文件外没有别的项目文件被改（find -newermt 核对）。
  2. 我用系统 python3（3.13）import 项目模块，在 runs/ptdom/__pycache__、llmadd_torch/__pycache__、runs/ptdom/superseded_A4/scripts/__pycache__ 里写了 `*.cpython-313.pyc` 字节码缓存。之后的运行都加了 PYTHONDONTWRITEBYTECODE=1。这些缓存无害，可以删。
- 我的脚本都在 session scratchpad `.../scratchpad/audit_a5/`（节点本地临时目录，方法和数字都写在下文）：jobs_cmp.py（240 文件逐个对比）、cx_audit_a5.py + cx_mx4.py（我自己的 fixB 反例 MX1–MX4，真实 screen / chain_step / readout / MJ.cells，A4 读数做对照）、mc_a5.py / mc_alt.py / mc_alt2.py（管线模拟，真实 screen / chain_step / asks / rule，新 RNG 种子，另加三类零假设）、bpb_lr.py、e0.py、keys.py。模拟在 claude_host 上 nice 19 跑。

**总判断**：
- **阻断 1（双路容差）修好了**：规则 = ptheld Amendment DP 的数值（eval 0 ≤ 0.05；之后每个打分 token ≤ 0.5 nat，按领域最长片段取上界），2-way 和 k-span 都覆盖；184 个 ptheld run 和 4 个 pilot 的验证我复现了（输出逐字相同）。与 DP 的一处差别：超限只记 FLAG、不判无效（次要 2）。
- **重要 2（fixB）实现正确**：按决定的规则实现；我自己构造的 3 个反例里 A4 给假 PASS、A5 给 NOT SHOWN / NOT FINAL，包括"F 单元与另一领域同一步的 E 重合"这条边界路径。
- **240 个作业文件**：只改了 intent（+dp_tol0 / dp_tol，数值正确）和 CHECK_PY heredoc；训练命令、SBATCH 头、输入检查、ARGS 都没动。对 run 的有效性来说实际上只放宽（eval 0 收紧了，但超限只记 FLAG，不会让 run 无效；新增的只有"NaN / 缺字段 → 无效"）。
- **假阳性率 ≤ 0.05：在协调者的模型族（平坦 + 线性共享趋势）里基本成立，但擦线**：我换种子复现，trend +1 时为 0.051–0.054。**模型族之外有新问题（重要 1）**：lr 响应是凹形（内部最优点，这是 lr 扫描的常态）时，即使两臂完全相同，P(PASS | 晋级) 也有 0.06–0.09。机制与 fixB 无关，A4 起就存在；SF 0.05 在多数情形下也压不到 0.05 以下（0.057–0.064）。
- **现在能不能提交第一阶段：可以**。条件见文末。

---

## 问题清单（按严重程度）

### 重要 1（新发现；A4 起就有，不是 A5 引入）：lr 响应为凹形时，P(PASS | 晋级) 超过 0.05，两臂完全相同也一样
- **位置**：
  - 选点规则：readout_ptdom.py:538 `select_unit`，code/tools/readout_rules.py:128-136（top2：前两名差 ≤ 2 分时按 bpb 选）；
  - 判词：readout_ptdom.py:1021-1035 → `rule`（第 (d) 条，seeds 4-7 的单边 t，SF_ALPHA 0.10，readout_ptdom.py:118）；
  - 校准：AUDIT.md A4.3、A5.2，只模拟了平坦和线性共享趋势两种 lr 响应。
- **机制**：
  - screen 在 seeds 0-3 上要求 4/4 正增益和同 lr 正增益。这个条件偏向留下"先验选在自己的最优 lr、none 因噪声或 bpb 平局选偏一档"的情形。
  - 之后的重选用的还是同一批 seeds 0-3，所以 lr 选择基本不变，两臂在选点上的真实差距（例如 +1 分）原样进入 seeds 4-7。
  - seeds 4-7 只是没被噪声选择污染，**lr 的选择照样带进去了**。所以 (d) 这条"无选择"检验挡不住这条路径。
  - 平坦响应下各 lr 真值相同，这条路径不存在；线性趋势只体现了它的一部分。
- **模拟**（真实 A5 管线；协调者的噪声模型 σ 3 / 5 / 10，in-training bpb 独立随机，除非另注；单领域名义水平，Holm 族 1）：

  | 零假设（各臂在自己最优 lr 处的真值相等） | n 晋级 | P(PASS \| 晋级) SF 0.10 | SF 0.05 | SF 0.10 + 同 lr 条件 † |
  |---|---|---|---|---|
  | 两臂相同，最优点 2e-3，曲率 1 分/档²，σ 3（两次独立运行） | 7188 / 5477 | **0.081 (se 0.003) / 0.080** | 0.064 | 0.068 |
  | 最优点 none 4e-3 / 先验 1e-3，曲率 1，σ 3（两次） | 2003 / 1921 | 0.067 (0.006) / 0.048 (0.005) | 0.038 | 0.046 |
  | 同上，σ 5 | 2073 | 0.063 (0.005) | | |
  | 同上，曲率 2，σ 3 | 1480 | **0.093 (0.008)** | | |
  | 同上，曲率 0.5，σ 3 | 2366 | 0.053 (0.005) | | |
  | 同上，曲率 1，σ 3，bpb 呈 U 形 ‡ | 2944 | 0.055 (0.004) | | |
  | （对照）最优点 none 4e-3 / 先验 5e-4，曲率 0.5 / 1 | 332 / 196 | 0.018 / 0.046 | | |
  | （对照）协调者模型：共享趋势 +1，σ 3 | 5829 | 0.053 | 0.043 | 0.044 |
  | （对照）功效：平坦，δ +6，σ 3：无条件 P(PASS) | 33335 | 0.315 | 0.290 | 0.315 |

  更多两臂相同的情形（mc_alt2.py，各 30000 次批模拟）：

  | 两臂相同 | n 晋级 | SF 0.10 | SF 0.05 | SF 0.10 + 同 lr |
  |---|---|---|---|---|
  | 最优点 2e-3，曲率 2，σ 5 | 5669 | **0.077** | 0.062 | 0.064 |
  | 最优点 2e-3，曲率 3，σ 10（rb6 领域读出的噪声水平） | 4781 | **0.078** | 0.063 | 0.063 |
  | 最优点 4e-3，曲率 3，σ 10 | 1090 | 0.072 | 0.057 | 0.065 |
  | 最优点 2e-3，曲率 1，σ 3，U 形 bpb（平局恰好偏向读出最优点，对规则最有利） | 10436 / 7892（两次） | **0.059 (se 0.002) / 0.059** | 0.046 | 0.056 |

  - † 同 lr 条件：先验最终选点 P 处，seeds 4-7 上 prior@P − none@P 的均值 > 0（none@P 缺 seeds 4-7 时补抽）。它和主检验共用先验的 seeds 4-7，二者相关，所以几乎不起作用。
  - ‡ U 形 bpb：按 rb6 53M none 的实测（bpb 1e-3 1.206、3e-3 1.206、8e-3 1.2275、1.6e-2 1.255；seed sd 0.005）建模为 1.206 + 0.0055·x²，x = log2(lr/2e-3)，两臂共用。
  - 在 PASS 的样本里，选点处的真实差距平均 +1.0（两臂相同、最优点 2e-3）、+0.95（曲率 2）。说明这些 PASS 来自 lr 选择，不来自噪声。
  - 两臂相同时，σ 从 3 到 10、曲率从 1 到 3，P(PASS | 晋级) 都稳定在 0.072–0.081；只有在 bpb 平局恰好偏向读出最优点时才降到 0.059。
  - 两臂相同的零假设下，单领域无条件 P(PASS) 0.0021–0.0024，24 个领域合计约 0.05–0.06 次假 PASS（族错误率约 5–6%）；平坦零假设下约 0.01。
- **确定性反例 MX4**（cx_mx4.py；真实代码，fake run，抖动 ±0.15）：
  - 设定：两臂完全相同，最优点 2e-3（1e-3 / 4e-3 低 1 分）；none@2e-3 在 seeds 0-3 上 −1.5；none@4e-3 和 prior@2e-3 的 bpb 略好。
  - 结果：screen ADVANCE（prior 2e-3，none 4e-3）→ 读数 **PASS**，gain +0.93，seeds 4-7 +0.95。A4 代码给出同样的 PASS。
  - 抖动很小，所以 seeds 4-7 检验几乎必过。这个反例只用来说明路径存在；发生频率看上表。
- **与 fixB 的关系**：
  - 我的反例 MX2（见核对 2）是同一机制的 fixB 版：none@P 已经跑过，而且是 none 的 argmax，但 top2 + bpb 平局把 none 留在一个幸运的内部 lr，结果 PASS。
  - 协调者模型在 trend +1 时的假 PASS 里，fixB 只占 25%（159/628），68% 是"先验选点低于 none 选点、都在 none 的范围内"。A5.2 把余量归因于 fixB 重选 + 平局，这只对了一部分。
- **为什么不是阻断**：
  - 它只影响 stage 2 之后的判词规则，不改任何 stage-1 作业文件；
  - 只要在读任何 stage-1 结果（`select --after 1`）之前由用户决定并预注册，就不损害预注册。
- **建议（交用户定，任何结果出来之前；改哪一种都要先用真实管线模拟）**：
  - (a) 接受并写明：凹形响应下每个晋级领域的假 PASS 约 0.06–0.09；
  - (b) 改判词，让 none 的比较格不再由 seeds 0-3 单独决定。例如：none 在前两名 lr 都补跑 seeds 4-7，seeds 4-7 的增益对两者中较好的那个计算；或者把同 lr 比较做成和主检验同等的必要条件（同样的 SF 检验）。成本是每个晋级领域约 4–8 个 run；
  - (c) 如果想在 stage 1 解决（让两臂网格对称，例如先验也跑 4e-3），必须在提交前决定。这也会顺带缓解 A5.3 的问题。
  - 只把 SF 降到 0.05 不一定够：bpb 与 lr 无关时，两臂相同仍有 0.057–0.064；U 形 bpb 时为 0.046。
- **置信度**：
  - 机制：高（真实代码的确定性反例，加两次独立模拟）；
  - 数值：中。依赖曲率 / σ 的比值和 bpb 与 lr 的关系，真实曲率未知。rb6 的读出对 lr 的变化相当大（例如 clock_readout lam1：5e-4 46.2、4e-3 58.9、8e-3 51.2；σ 约 5–10），所以曲率 / σ ≈ 0.3 的设定不算极端。

### 次要 2：eval 0 超过 0.05 也只记 FLAG，不判无效；与 ptheld Amendment DP（"(i) 或 (ii) 失败 → run 无效"）不同
- **位置**：make_jobs_ptdom.py:398（eval 0 与后续用同一个 flag 检查）、:526-528（FLAGC）；readout_ptdom.py:111、322。
- **为什么**：
  - eval 0 是唯一能真正识别索引错误的已知答案：近均匀模型上，片段错位约差 0.5 nat/token，多一个或少一个 token 约差 11 nats。
  - A5.1 写的是"Rule (= Amendment DP)"，但后果与 DP 不同：一旦出现 GPU 特有的索引错误，每个 run 都只会被 FLAG，判词照常给出（"no rule reads it"）。
- **实际风险低**：同一 trainer、同一 B200 的 4 个 pilot 上，eval 0 最大值 0.017（2-way 与 k-span 各集合，含 8 token 的 people），比阈值有 3 倍余量；ptheld 552 个值最大 0.0195。
- **建议（可选）**：只让 (ii)（后续 eval）只记 FLAG；(i)（eval 0）保持判无效，或者至少在任一 run 的 eval 0 超限时让读数 PROVISIONAL。
- **置信度**：代码事实，高；影响，低。

### 次要 3：tests/a5/jobcheck_a5.py 有副作用，会重渲染生产作业文件
- **位置**：tests/a5/jobcheck_a5.py:52（`make_jobs_ptdom.py --batch b --stage 1`，在 runs/ptdom 原地写 60 个 job 文件）。A4 版很可能也一样，没核。
- **为什么**：这是一个"测试"，却会覆盖即将提交的文件。如果生成器在审查之后被改动，重跑测试就会静默地把已审过的作业文件换掉；它也让 mtime 失去溯源价值。我这次就是因此误写了 240 个文件（内容未变，见上文披露）。
- **建议**：渲染到临时目录（例如 `JD` 可配置），或者测试只读现有文件。
- **置信度**：高。

### 次要 4：F 单元恰好与另一领域同一步的 E 重合时，select 不列出这个待跑的 fixB；chain_report 还把它说成 "F"
- **位置**：select_ptdom.py:134（只在 `pk["F"]` 非空时列出）；readout_ptdom.py:785（F 去掉 E 中已有的单元）、1176-1178（文案写 "asked (F ...)"）。
- **反例 MX3**：compass 的 none 走 EXT2 → 5e-4（E），zodiac 的 P = 5e-4。
  - after 2：`F = []`，`E = [none@5e-4]`，zodiac fixb_pending；
  - select 不打印 zodiac 在等这个格；
  - 读数正确：after 2 给出 NO VERDICT；after 3 none 重选到 5e-4，NOT FINAL；生成器只渲染一次 none@5e-4（8 个 run）。
- **影响**：只是外观，逻辑正确。
- **置信度**：高。

### 次要 5：240 个作业文件的头注释仍写 "(Amendments A3, A4)"
- **位置**：make_jobs_ptdom.py:660。
- **说明**：只是注释，CHECK_PY 和 intent 都是 A5 的。修它会让"只改了 intent + CHECK_PY"的核对变复杂，可以不修。

---

## 逐题核对

### 1. 阻断 1：双路容差是否按 Amendment DP 改对，并在 ptheld 和 pilot 上验证
**改对了**（与 DP 的差别见次要 2）。
- **CHECK_PY**（make_jobs_ptdom.py:376-417）：
  - eval 0 的阈值是 `dp_tol0` 0.05，之后是 `dp_tol[d]` = 0.5 × SPAN_TOK[d]；
  - 覆盖 2-way 的 `lp_dp_maxdiff_<d>_<held|seen>`，以及 k-span 每个集合（含 months 的 heldE 和 p1–p4）的 `lpk_dp_maxdiff_*` 与 `lpk_own_maxdiff_*`；
  - `dom_of` 按最长前缀映射到领域，我核过 24 个领域名之间没有前缀冲突；
  - 字段缺失或非有限值单独成为一项检查，判无效；超限的 run 只记 FLAG（`valid_data_flagged`）；loader 的 DP_FLAG_CHECKS 与 CHECK_PY 的 FLAGC 一致。
- **打分 token 数**：
  - 我读了 trainer（pretrain_range.py:724-760、1150-1175、1837-1877）：2-way 的 dp 取 "other option" 行前 4 行的单段 |差|；k-span 的 dp 取第一个 eval 行的前 4 个候选行；own 取所有 eval 行。每段都是 `' <item>\n'`，paraphrase 断言与训练行同段。
  - 所以"领域最长片段"是每一行的上界，只会更宽，不会更严。SPAN_TOK 与 span_tok.py、pilot 的 kspan_tokens / item_tokens 一致。
- **验证（复现）**：
  - ptheld_dp_a5.py 重跑，输出与 tests/a5/ptheld_dp_a5.out **逐字相同**：184 run、38,640 值，新规则 0 个超限（旧规则 130/184 个 run）；最大值 / 容差 0.926；eval 0 最大 0.0195。
  - pilot：720 个值 0 超；eval 0 最大 0.017；后续最大比值 0.084。
  - check_py_a5.py 重跑 **62/62**（在 pilot 自己的 run JSON、gpu.csv、offline_lm.json 上，用作业同一个解释器 open_r1）。
- **残余**：2-token 领域（months、compass、life_stages，容差 1.0）后期偶有超限可能。ptheld 的 3-token 集合每 token 最大约 0.46，所以 batch A months 每批约 10 万个值里可能有个位数的 FLAG。这不影响有效性。
- **pilot 确认是否要重做**：不需要。
  - CHECK_PY 是 run 文件的纯函数；heredoc 用 `<<'PY'` 引号，intent 放在单引号里，dp_tol 的 JSON 里没有单引号；240 个文件的 heredoc 与 CHECK_PY 逐字节相同。
  - 所以 check_py_a5 的 CPU 重推导与作业内运行等价，可以替代 A5 的确认。
  - 这一点由协调者决定；check_ok_<X> 仍是 A4 的记录。

### 2. 重要 2：fixB 是否实现正确（先验选点超出 none 的范围 → 补跑 none 并重选）
**实现正确，与决定一致**（readout_ptdom.py:760-786）。
- **规则逐条核对**：
  - **触发条件**：`need` = none 自己的链已走完（nx None），且 P 落在 grid ∪ 该领域自己的外扩之外。
  - **复用与待跑**：none@P 的 8 个 seed 都已被问过 → 立即重选（interp. a）；否则挂起，记入 F。F 去掉与 E 重复的单元，C 去掉与 F 重复的单元。
  - **重选**：`select_unit` 在 seeds 0-3 上，对 range ∪ {P} 选点。
  - **NOT FINAL**：A4 的 not_final，或 P ∈ {5e-4, 1.6e-2} 且 P 是重选后的 pick 或 argmax（interp. b）。
  - **不再外扩**：P 不进 exts，所以 fixB 的 lr 不会触发新的外扩（interp. c）。
- **链的时序**：
  - 先验从 after 2 起就固定（网格只有 2 档，argmax 必在边上，必有且只有一次外扩）。
  - 所以 F 只会出现在 after 2（stage 2b），只会是 none@1e-3 或 none@5e-4（FIXB_LR 断言）；链在 after 4 之前一定结束。
- **读数**：
  - fixB 挂起时给 NO VERDICT，并列入 PROVISIONAL；
  - 一致性检查比较 F；
  - A4 的 picks 文件被拒，生成器也拒（amendment）。
- **我自己的反例**（cx_audit_a5.py；真实代码，同样的 run 同时过 A4 读数）：

  | 反例 | 设定（都没有先验效应） | A4 | A5 |
  |---|---|---|---|
  | MX1 | 共享趋势 +2/档；none@8e-3 在 seeds 0-3 幸运 +5 → none 向上外扩到 1.6e-2，范围 [2e-3, 1.6e-2]；P = 1e-3 | PASS（prior@1e-3 对 none@2e-3，+2） | F = none@1e-3（8 个 run）→ after 2 NO VERDICT → none 重选到 1e-3 → **NOT SHOWN**（gain −0.00） |
  | MX2b | 趋势 +1；none@4e-3 幸运 +1.5（内部，不外扩）；P = 1e-3；bpb 相同 | PASS | F = none@1e-3 → **NOT SHOWN** |
  | MX3 | F 单元与 compass 的 EXT2 在同一步：zodiac 无效应、趋势 +2；compass 真效应 +5 | zodiac PASS | after 2：E = [none@5e-4]，F = []，zodiac NO VERDICT；after 3 复用该格，none 重选到 5e-4 → **NOT FINAL**；生成器只渲染一次 none@5e-4 |
  | MX2 | 同 MX2b，但 none@4e-3 的 bpb 好 0.01 | PASS | F = none@1e-3 已跑，none 的 argmax 是 1e-3，但 top2 平局按 bpb 选了 4e-3 → **PASS（假）**，带 FIXB 标签（重要 1 的机制） |

- **结论**：fixB 堵住了 audit_a4 那条"none 从没跑过 P"的路径，但它堵不住"跑过了却因为选择噪声没选上"的路径（重要 1）。
- 协调者的 selftest 46/46、cx_a5 6/6 我都重跑过，结果一致。

### 3. 模拟的假阳性率是否 ≤ 0.05
- **在协调者的模型内**：
  - 平坦零假设：0.034–0.041 ✓。
  - 线性共享趋势：我换了新种子复现，σ 3 trend +1 为 **0.054 (se 0.002)**，σ 5 trend +1 为 0.051，σ 3 trend +0.75 为 0.052。与 A5.2 合并后，σ 3 trend +1 约 0.053（95% 区间约 0.050–0.056）。也就是**略高于 0.05**，不是"≤ 0.05"。A5.5 写的"在目标附近、模拟误差内"基本属实，但偏乐观。
  - 换成贴近真实的 U 形 bpb：trend +1 为 0.043 / 0.044（σ 3 / 5），平坦 0.037，trend +2 为 0.025，都 ≤ 0.05。
- **在模型之外（凹形 lr 响应）**：0.05–0.09，见重要 1。**所以"≤ 0.05"这个说法只在平坦或线性响应下成立。**
- **功效**：A5 对 A4 功效约低 5%（相对），我复现无条件 P(PASS) 0.315（σ 3，δ +6），与 A5.2 一致。SF 0.05 会降到 0.290。

### 4. 240 个作业文件的改动是否只是放宽容差
**是。我独立逐文件核对过**（jobs_cmp.py；重渲染前后各一次）：
- 240/240 文件名与 superseded_A4/jobs 一一对应；
- heredoc 与当前 CHECK_PY 逐字节相同（sha256[:12] 12dff9487798），A4 的 heredoc 与 A4 的 CHECK_PY 相同；
- heredoc 外只有 intent 那一行不同。intent 解析后 = A4 intent + `dp_tol0` 0.05 + `dp_tol` = {d: 0.5·SPAN_TOK[d]}，键与该批的 ext 一致；
- SBATCH 头、HOLD 守卫、sha 检查、输入检查、ARGS、训练命令都与 A4 逐字相同。

**对有效性的影响**：
- 后续 eval 放宽（0.25 → 1.0–4.0）；
- eval 0 收紧（< 0.25 → ≤ 0.05），但超限只记 FLAG，所以不会让 run 无效；
- 新增"NaN 判无效"。A4 用 `max(..., inf)`，缺字段时本来就失败；NaN 在 A4 里取决于在 max 中的位置，可能漏过；
- exit 码语义不变：只有 FLAG 类失败时 exit 4，Slurm 记 FAILED；loader 据 check.json 判为有效；rerun_plan 不重跑已完成的 run。

### 5. 有没有引入新问题
- 未发现会让 stage 1 出错的新问题。
- 新的次要问题见次要 2–5。重要 1 不是 A5 引入的，是 A4 起就有、A5 的校准没覆盖到的。
- 其他核过、没问题的点：
  - `dom_of` 的空集不会发生（pilot 的所有 lpk 键都以领域名开头）；
  - FIXB_LR 断言；cells() 拒收 stage 2 里的 F、先验臂的 F、FIXB_LR 之外的 F；
  - `--allow_partial` 时 sprobs 使读数 PROVISIONAL（次要 3 已修）；
  - A5.3 注释是纯描述；
  - submit_ptdom.sh 只改了日志路径和说明；
  - aux_gate_check.py 未改。

---

## audit_a4 各项现状

| audit_a4 项 | 现状 |
|---|---|
| 阻断 1 双路容差 | 已修，已复现验证（次要 2：eval 0 只记 FLAG） |
| 重要 2 lr 混淆残余（none 未跑过 P） | 已修（fixB，MX1 / MX2b / MX3）；更一般的选点残余见重要 1 |
| 次要 3 `--allow_partial` | 已修 |
| 次要 4 平局容差 | 未改，理由成立：pilot 只记计数、不记 gap，改它需要 Rev H |
| 次要 6 NOT ADVANCE 的方向 | 已写入 screen（A5.3 注释） |
| 次要 5、7、8、9 | 未在 A5 范围内；次要 8（bpb 与 lr 无关）我用 U 形 bpb 补模拟了：结论更有利 |

---

## 现在能不能提交第一阶段

**可以提交 stage 1**，前提是：
1. 协调者接受 check_py_a5 在 pilot 真实 B200 文件上的 CPU 重推导，作为 A5 的 pilot 确认。我认为它与作业内运行等价，不需要新 pilot。
2. 重要 1 必须在读任何 stage-1 结果（`select_ptdom.py --after 1`）之前由用户决定并预注册：(a) 接受并写明，或 (b) 改判词（需要先用真实管线模拟）。只有选 (c)（stage 1 让两臂网格对称）时，才必须在提交前决定。
