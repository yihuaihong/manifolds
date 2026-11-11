# pt2same Amendment A2（A1 敏感性读数的定义）及 readout_pt2same_a1.py 对抗审查（2026-11-02）

审查员：全新子 agent，没有参与 A2 的设计。全程只读，只写了本文件。没有改动其他文件，没有提交或取消作业，没有打开任何 pt2same 结果，包括 runs/pt2same/out 下的 run JSON、check.json、offline_lm.json 和作业日志；out/ 只看了目录名和 stat 元数据。测试都在 scratchpad 的临时目录里跑，用的是合成树。运行时设置了 PYTHONDONTWRITEBYTECODE=1，跑完后核对过：项目目录里没有新文件，`__pycache__` 也没有新增。

审查用到的脚本放在 scratchpad（会话私有目录，可能被清理）：
- `ce_a2.py`：13 个反例。
- `ce_primary_nan.py`：1 个越界探针，测的是主读数。
- 日志：`selftest_rerun.log`、`ce_a2.log`。

## 结论先行
- **阻断：未发现。**
- **重要：未发现。**
- **次要：6 条（m1–m6）。**
  - m1：A2 写作期间，主批次 seed 0 的部分 run JSON 已经在盘上，其中包括 A1 的直接输入 53M none@8e-3 s0。「没读过」这一点无法核实。
  - m2：A1 加载器不直接核对 run JSON 里的 args.lr / seed / tag。
  - m3：自测有覆盖缺口。我用反例补测了，代码行为全部正确。
  - m4：A1 作业如果因 infra 原因丢失并换名重跑，脚本读不到，会永久输出 not computed。
  - m5：解读措辞问题。LOSO 只修正 Torch 上的重新选 lr，不修正 @8e-3 一侧本身带有的 Burst 选择偏差。
  - m6：8 个新 A1 格子没有 expected-range 规则。
- **范围外（次要）：** 冻结的主读数 `load_run` 会把 NaN area 当作 ok 放行，最后打印出「no size trend shown (CI includes 0)」。实际发生的可能性很低，但这条路径是静默的。

| 问 | 结论 |
|---|---|
| 1 | 是。A2 在任何 A1 结果出现之前写好：8 个 A1 作业到 18:57 都还是 PENDING，out/ 下也没有它们的目录。附带说明见 m1。 |
| 2 | 定义完整，可以直接执行。 |
| 3 | 代码实现了定义。没有静默回退，seed 没有配错，area 和 bpb 取法正确，没有把 4 seed 和 8 seed 混在一起比较。次要问题见 m2、m4。 |
| 4 | 14/14 已复现，逐行相同，只有临时路径不同。另跑了 13 个反例，结果全部符合预期；自测本身的缺口见 m3。 |
| 5 | 未发现任何会改动主判词的路径。 |

---

## Q1. A2 是否在任何 A1 结果出现之前写好？

**结论：是。** 置信度高。

| 对象 | 时间（EST，mtime） |
|---|---|
| A1 的 8 个作业 21696761–768（sacct） | 18:21:27–28 提交；18:57 查询时全部 PENDING，Start = Unknown |
| runs/pt2same/out/ 下的目录 | 只有主批次的目录（4 个 s0，外加 18:58 新出现的 190m-none-s1），**没有** lr4e-3 或 lr1.6e-2 的目录 |
| readout_pt2same_a1.py | 18:50:38，sha256 a358df929e82…，与 A2 记录的一致 |
| tests/known_a1.log、tests/selftest_a1.log | 18:50:43、18:51:07，都晚于脚本的最后修改，所以对应的是最终版本 |
| AUDIT.md（A2 是文件末尾一节） | 18:52:12 |
| readout.json、readout_a1.json | 不存在 |
| FROZEN.sha256 | `sha256sum -c` 全部 OK；readout_pt2same.py 为 bf7cced43238…，与 A1 脚本里断言的值一致 |

**次要 m1：A2 写作时，主批次 seed 0 的部分结果已经在盘上。**
- **位置：** AUDIT.md:213-214（「4 main seed-0 jobs running (nothing of this batch read)」）；runs/pt2same/code/tools/pretrain_range.py:1969-1984（每次 eval 之后都会重写整个 run JSON）。
- **为什么：** 主作业 21696726/727/728/729 的开始时间是 18:22:44、18:26:12、18:26:12、18:39:06，它们的 run JSON 在每个 eval 后都会更新。我观察到 mtime 从 18:52 变到了 18:58。所以 A2 写作期间（约 18:39–18:52），**A1 的直接输入**已经有部分曲线在盘上：53M none@8e-3 s0（这是 A1 里 8e-3 一侧和 Δ53 的来源）、53M win-ctx s0、190M s0。A2 只写了「running」，没有写「部分 curve 已在盘上」。「nothing read」无法核实：这个文件系统的 atime 等于 mtime，作业每次写入都会刷新它；/scratch 上读文件本来也不刷新 atime。
- **为什么只是次要：**
  - A2 的每个选择都能追溯到结果出现之前写好的 tests/audit_a1.md 第 5a 节（18:39:19），包括 seeds 0-3、rule 1 加 bpb 并列裁决、开区间不扩展、LOSO、not computed、单独的 JSON。
  - 构建者自己做的三项决定里，两项让规则更严（要求主读数有判词、要求有 offline bpb）。第三项是不给 CI，而 audit_a1 本来就把 CI 列为可选。这三项都不会因为看了 s0 的部分曲线而朝某个方向偏。
- **建议：** 在 AUDIT 里补一句：「A2 写作时，主批次 4 个 s0 run 的部分 run JSON 已在盘上（训练器每个 eval 重写一次），未打开」。以后写类似的修正案，最好在第一个作业开始之前写。
- **置信度：** 事实部分高；对结论有影响的可能性低。

## Q2. 定义是否完整、可执行？

**结论：完整，可以执行。** 置信度高。逐项核对如下：
- **seeds：** (a) 只用 seed 0-3，两侧都是（8e-3 一侧也只用 0-3）；明文禁止拼成 8 个 seed；A1 节不打印 8 seed 斜率（AUDIT.md:223-225）。
- **选 lr：** (b) 照 readout_rules.best_lr 执行。并列的定义是「与最高值差在 2 以内」（含等于 2，即 `m >= top - 2`）。并列时比较 offline 全量 val bpb 的均值，取法逐条照抄快照里的 `pt2main_summary.py:102-104`（我核对过，快照第 102-104 行正是这段）；bpb 也完全相等时取较低的 lr。
- **开区间：** `bracket` 判为 'open' 时打印「open bracket at the <edge> edge, not extended」，判为 'note' 时打印 note。各 fold 不判 bracket。
- **恒等式权重：** w53 = (x53 − x̄)/Σ(xi − x̄)²，x = ln N_nonemb。我独立算出 [-0.316653, +0.047304, +0.269349]，见 CE11。恒等式的方向也对：none 改用 best lr 后，Δ53 的均值变化量是 (n8 − nb)，所以斜率变化 = w53 × (n8 − nb)。当 nb > n8 时，斜率上升 0.3167 × 差值。
- **LOSO：** (d) 对每个留出的 seed h，在另外 3 个 seed 上按 rule 1 选 lr（包括 3 个 seed 上的 bpb 并列裁决），取 seed h 在该 lr 下的 area，4 个值取均值；斜率再算一次，同一个恒等式也要断言成立。我按定义手算了 LOSO 已知答案（[44,10,10,10] / [14,16,16,16] / [5]×4 → 各 fold 依次选 8e-3、4e-3、4e-3、4e-3 → 11.0），与日志一致。
- **not computed：** (f) 列出了全部触发条件：12 个 run 中任何一个没过 load_run 的检查、area 非有限、缺少有效的 offline 条目或 offline 双路径不一致、主读数没有判词。触发后 (b) 到 (e) 每一行都打印 not computed，不回退到 8e-3，也不改用更少的 run。发散的 run 照常算作数据。
- **分隔：** (g) 结果写进单独的 JSON；拒绝写入 readout.json；不给 CI、不下判词、不用等价性措辞。

A2 没写但不影响执行的地方，见 m4（换名重跑）和 m6（新格子没有 range 规则）。

## Q3. 代码是否实现了定义？

**结论：实现了。** 没有发现静默回退、seed 配错、area 或 bpb 取错、4 seed 与 8 seed 混用。置信度高。

- **路径。** `a1_paths`（readout_pt2same_a1.py:59-70）拼出的 torch 路径，与 A1 作业文件里的 OUT 和 TAG（a1/job_*-s1.sbatch:23）以及 run 文件名（:66）一致。在 8e-3 下，它还断言自己的结果等于 `RO.run_paths`，这是一条双路径校验。Burst 路径的拼法与 `RO.BURST_TASK` 一致；known_a1 在真实的 Burst 文件上逐 seed 复现了 audit_a1 Q2 的数值。
- **area。** 来自 `RO.load_run`，即 RR.area 对 75 个 eval、6 个 bucket 取均值，要求 finished、check pass、与 check.json 的差在 0.01 以内。另外加了 `isfinite` 检查（:80-81），比主读数更严，见 CE1。
- **bpb。** offline_lm.json 取 `chk.parent`，也就是作业的 `$OUT/offline_lm.json`（sbatch:64）。按 tag `_<tag>`、seed 和 `max_windows == 0` 过滤，再经过 `RR.offline_bpb(…, diverged)`，与 offline_lm.py:82-84 写入的字段一致（tag 取自 a.tag，seed 取自 a.seed）。之后再和 check.json 的 `info.bpb_full` 做双路径比较，两边都是 NaN 时视为相等。in-training bpb 取 `curve[-1]["val_bits_per_byte"]`，与 pt2main_summary:108 相同。发散的判定用 `RR.diverged_at`，与 CHECK_PY 的 `bad` 定义相同（sbatch:92）。
- **seed 配对。** `prior53[:4] - nb`：两者都按 seed 0..3 的顺序排列（RO.load 按 SEEDS 顺序读，load_a1 按 S4 顺序读）。各 fold 用 `T[lr]["area"][h]`，这里 h 既是 S4 中的下标，也就是 seed 本身。8e-3 的 area 断言等于主读数加载器读到的值（:258-260）。
- **4 seed 与 8 seed。** `slope_h1` 只取 S4 的 seed（:114）。`slope_h1(G)` 断言等于主读数的 `slope["h1"]`（:256），所以 190M 或 505M 如果误用了 8 个 seed 就会直接崩溃。CE9 把所有格子的 seed 4-7 都改掉之后，主读数的 8 seed 斜率从 +0.143 变成 +0.409，A1 的全部输出逐位不变。
- **not computed。** 自测覆盖了以下情况：失败、缺失、缺 offline 条目、area 双路径不一致、主读数无判词。CE1–CE3 和 CE7 另外补测了 NaN area、|orig_minus_logged| ≥ 0.002、bpb 双路径不一致、PENDING。所有情况都输出 not computed，并列出原因。主读数 PENDING 而且带了 --investigated 时会正常计算。readout.json 是带 note 生成的、A1 却不带 note 运行时，输出 STOP，rc 为 3。

**次要 m2：A1 加载器不直接核对 run JSON 自己的 args.lr / seed / tag。**
- **位置：** readout_pt2same_a1.py:73-93。
- **为什么：** lr 和 seed 是否正确，完全靠两件事：check.json 的 pass（CHECK_PY 会拿 run args 和 intent 比对），以及 offline 条目的 tag 和 seed 能对上。CE8a 把 8e-3 s0 的整个目录拷进 4e-3 s0 的位置，被 offline tag 拦下了，输出 not computed，这是正确的。CE8b 则把 offline tag 改成 4e-3，同时让 run JSON 里写着 lr 0.008、tag 8e-3，结果被接受了。这需要人为篡改，或者 CHECK_PY 本身失效，所以只算次要。
- **建议（可选）：** 在 `load_a1_run` 里加一条断言：`r["args"]["lr"] == float(lr) and r["args"]["seed"] == s and r["args"]["tag"] == f"_{tag}"`。这是第三条路径，加了不改变任何定义。
- **置信度：** 高（已复现）。实际影响：低。

**次要 m4：A1 作业换名重跑后脚本读不到。**
- **位置：** readout_pt2same_a1.py:59-70，路径是写死的；AUDIT.md:201-202，A1 第 6 条规定「重跑要换新 run 名」。
- **为什么：** 抢占后 requeue 会在原目录续跑，没有问题。但如果遇到 TIMEOUT 或节点故障，按 A1 第 6 条要换名重跑，而 A1 脚本只认原来的名字，于是会永久输出 not computed（原因是「incomplete … no run JSON」）。这种失败方式是安全的，不会静默出错，但 A2 没有规定重跑后怎么读。主读数也有同样的缺口，不在 A2 范围内。
- **建议：** 一旦发生，先写修正案说明换名后的映射，再改脚本（附 sha）；不要现场手改。
- **置信度：** 高。

## Q4. 自测是否真的覆盖了这些情形？

我在 scratchpad 里用 open_r1 环境重跑了 `--selftest`：14/14 PASS，输出与 tests/selftest_a1.log 逐行相同，只有临时路径不同。逐项核对断言：
- best = 8e-3 时，三个斜率在 1e-12 以内相等，各 fold 都选 8e-3，LOSO 均值为 23。这里 seed 0-3 的均值不同于 seed 4-7 的 18–21，所以如果误用了 8 个 seed，这一条会失败。
- 上沿和下沿的 open 都测了。
- offline bpb 与 in-training bpb 方向相反时，取 offline。
- 失败 run 的情形会断言主判词与 readout.json 逐字节相同。
- 恒等式与 polyfit 暴力解对照。
- LOSO 已知答案、STOP、readout.json 拒写、RO 全局变量不变。

这些断言都是实打实的，没有空转。

**次要 m3：自测没有覆盖的分支（我用反例补测了，代码全部正确）。**
- **位置：** readout_pt2same_a1.py:318-445；`_write` 在 :278-289，其中 `orig_minus_logged` 恒为 0.0，run JSON 也没有 args.lr / seed / tag 和 divergence。

| 反例 | 情形 | 结果 |
|---|---|---|
| CE1 | NaN area | A1 not computed（RO.load_run 单独判为 ok） |
| CE2 | \|orig_minus_logged\| = 0.01 | not computed |
| CE3 | offline bpb ≠ check.json 的 bpb_full | not computed |
| CE4 | 发散 run 处在并列中（8e-3 s1 的 lm 为 NaN，bpb 为 NaN） | computed；8e-3 的 bpb 记为 inf，选 4e-3，area 照常计入 |
| CE5 | bpb 完全相等 | 选较低的 lr |
| CE6 | note bracket（argmax 在下沿、差 ≤ 2，按 bpb 选 8e-3） | 打印 note |
| CE7 | 主读数 PENDING / 带 --investigated / readout.json 与运行时的 note 不一致 | not computed / computed / STOP rc 3 |
| CE8a / CE8b | 见 m2 | 拦下 / 接受 |
| CE9 | 改动所有格子的 seed 4-7 | A1 输出不变 |
| CE10 | 缺一个 Burst run | 表格显示 n/a，A1 照常计算 |
| CE11 | 权重手算 | 一致 |
| CE12 | 用指向 readout.json 的软链做 --json | 被拒绝 |

- **建议：** 把 CE1–CE7 和 CE10 并进 `--selftest`，并在 FROZEN 之外另记新的 sha。不补也不会导致测错，因为代码行为已经验证过。
- **置信度：** 高。

## Q5. 有没有任何路径会改动主判词？

**结论：未发现。** 置信度高。
- `RO.read(roots, …, js=None)`（:239）不写文件。A1 脚本只写 `--json` 指定的文件。`check_js` 在任何计算之前就拒绝 readout.json 和 `--primary-json` 指定的文件，按 resolve 之后的路径比较，所以软链也会被拒（CE12）。
- RO 的全局变量（PICKS、ROOTS、F86、BURST_TASK、N_NONEMB、SIZES、SEEDS、H1、H2）在运行前做了深拷贝，运行后断言没有变化（:236、:270）。A1 用的是自己的 `a1_paths`，不改 `RO.PICKS`。
- 主读数的路径完全由 PICKS 拼出，不用 glob，所以 A1 的 run 不会被读进去（audit_a1 Q5 已经核实过）。readout_pt2same.py 的 sha 与 FROZEN 一致，并且在导入时断言。`sys.dont_write_bytecode` 在导入之前就设好了，`__pycache__` 不会新增文件。
- 主读数没有判词时，A1 只输出 not computed，不会反过来影响主判词。A1 的 JSON 只记录 `primary_verdict` 原文，不含 8 seed 斜率。

## 其他次要项

**次要 m5：「selection bias (see LOSO)」的措辞可能让人以为 LOSO 修正了全部选择偏差。**
- **位置：** AUDIT.md:240-244、:258-259。
- **为什么：** LOSO 只修正 Torch 上 best-of-3 的样本内偏差。8e-3 一侧本身也是在 seed 0-3 上选出来的：Burst 的选择基于 seed 0-3，而 Torch seed 0-3 与 Burst seed 0-3 的 init 和数据相同；F86 里 53M none@8e-3 在 seed 0-3 上是 23.1，在 seed 4-7 上只有 13.8。所以「S0-3(best) − S0-3(8e-3)」或「LOSO − 8e-3」的差值，混合了站点造成的最优 lr 移动和两种选择效应，不能读成纯粹的站点效应。另外，190M 和 505M 的 S0-3 也都在各自的选择 seed 上。
- **建议：** FINDINGS 写成「seeds 0-3, descriptive；best-of-3 带样本内选择偏差（LOSO 行修正这一项）；@8e-3 一侧本身也是 seed 0-3 上的 Burst pick（winner's curse，见 Design）」。
- **置信度：** 高（属于解读问题）。

**次要 m6：8 个新 A1 格子没有 expected-range 或 investigate 规则。**
- **位置：** AUDIT.md:249-254。
- **为什么：** 主读数的 rule 2 只检查 4 个主格子。如果 A1 的某个 run 通过了 CHECK_PY，但数值异常（比如 Torch 4e-3 的 area 远高于 Burst 的 14.75），它会直接进入 rule 1，然后带动 A1 的斜率。(e) 表里逐 seed 打印的 Torch − Burst 可以起到人工核对的作用，但没有规定「先查再读」。这是描述性分析，不影响主判词，所以只算次要。
- **建议：** 读 A1 时先看 (e) 表，Torch − Burst 超过约 ±15 的格子按 experiment-audit 的「反常结果」流程处理之后再引用。
- **置信度：** 中。

**范围外（次要）：冻结的主读数会把 NaN area 当作 ok。**
- **位置：** readout_pt2same.py:89-91。`abs(nan - x) > 0.01` 的结果为 False，所以 NaN 会通过检查。
- **为什么：** 用 `ce_primary_nan.py` 测试：把 505M none s5 的 held_* 设为 NaN，`load_run` 返回 `('ok', nan)`，主判词打印「no size trend shown (the 8-seed 90% CI includes 0 …)」，而斜率和 CI 实际都是 NaN。这是静默的错误判词。held_* 是 exact-match 比例，评测集大小固定，实际出现 NaN 的可能性很低。A1 在 :80-81 有自己的 `isfinite` 检查，在 :255 有 `tolist ==` 断言，遇到 NaN 会响亮失败，所以不会受影响。
- **建议：** 主读数是冻结的，不能改。读主结果时附加一个人工检查：48 个 Torch area 全部有限。如果要补规则，就另写修正案。
- **置信度：** 代码路径高；实际发生的可能性低。

## 构建者的三项决定
1. **主读数必须有判词，否则 A1 输出 not computed。** 这和 audit_a1 5a 第 2 步一致，我同意。代价是：如果 53M none@8e-3 超出范围，调查期间看不到 A1 的 lr 表。若用户希望把 A1 的 (e) 表当作调查工具，应另写修正案，例如只在调查期间开放 (e) 表；不要临时绕开这个规则。
2. **要求有 offline bpb。** 这和 pt2main rule 6 一致：pt2main_summary 在缺 offline 条目时会把 best lr 判为 WITHHELD。主批次的 CHECK_PY 本来就要求 offline 双路径，所以实际上不会额外挡掉什么。我同意。
3. **不给 CI。** audit_a1 5a(e) 原本就把 CI 列为可选（「如果要给 CI」），而且 4 个 seed 只有 35 种不同的重抽组合。我同意。

## 复现
```
SP=<scratchpad>; TMPDIR=$SP/tmp PYTHONDONTWRITEBYTECODE=1 /scratch/yh6210/miniconda3/envs/open_r1/bin/python \
  /scratch/yh6210/llmadd_torch/runs/pt2same/readout_pt2same_a1.py --selftest      # 14/14
TMPDIR=$SP/tmp PYTHONDONTWRITEBYTECODE=1 .../python $SP/ce_a2.py                  # 13/13 as expected
sacct -j 21696761,...,21696768 -X --format=JobID,State,Start   # 全部 PENDING（2026-11-02 18:57）
```
