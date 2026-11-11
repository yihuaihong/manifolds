# pt2same Amendment A1 对抗审查（8 个 53M none lr 敏感性作业，2026-11-02）

审查员：全新子 agent，没参与设计。只读：除本文件外没有改任何文件，没有提交或取消作业，没有运行 A1 生成器（它会改写 a1/JOBS.sha256）。
独立核对脚本放在 scratchpad 里，没有写进项目目录：
- `a1_check.py`：在内存里重渲染，做归一化 diff，argparse 模拟，核对 Burst 孪生 run，查撞名，查读数路径。结果 0 failures。
- `a1_curse.py`：估计 Burst 上 3 个 lr、4 个 seed 的选择偏差。

运行时禁用了 bytecode，确认没有在项目目录里留下 `__pycache__` 写入。

## 结论先行
- **阻断：未发现。** 8 个 A1 作业和同 seed 的主作业 53M none@8e-3 相比，把 lr 字符串和 Burst 孪生任务名替换回去之后，作业体逐行相同。包括：
  - HOLD、sha 校验、COMMIT、GPU 守卫、数据 sha、model 行、2 个 operand site、无 target 的断言；
  - DRY 哈希（与 lr 无关）、step-0 lm 参照、CHECK_PY 全文、#SBATCH 头（时限 02:00:00、nice 20、requeue、account、QoS）。

  注释块按词比较也相同。Slurm 实际持有的 8 个 batch script（`scontrol write batch_script`）与磁盘文件、a1/JOBS.sha256、我在内存里的重渲染三者逐字节相同。8 个孪生 run 都选对了。render() 里的孪生断言在 A1 下仍然全部生效，包装函数没有绕过它们。主读数不会读到 a1 的 run。
- **重要（1 条）：** A1 的描述性读数还没实现。照 AUDIT 的字面意思（「best-of-3 … and the slope recomputed with that cell」）去写，很容易测错：
  - 把 4 seed 的斜率拿去和 8 seed 的主斜率比；
  - 或者把 seed 0-3 用新 lr、seed 4-7 用 8e-3 的结果拼在一起；
  - 另外，在同样 4 个 seed 上选 lr 再评估同样 4 个 seed，会带约 +2.5~2.8 area 点的选择偏差，相当于斜率 +0.8~0.9。

  需要在结果出来前把定义写死。具体做法见 Q5。
- **次要（6 条）：** 没有 submit_a1.sh，提交时绕过了清单校验和 FROZEN 校验（事后核对无害）；A1 的 out 目录放在主读数根目录 runs/pt2same/out 下；A1 生成器跳过了跨批次撞名断言（我手工核对无撞名）；HOLD 与主批次共用；注释写着「F86's pick」，生成器也指错了；磁盘估计没有算进 A1 的 +1.7 GB。

---

## Q1. A1 作业和主作业（53M none@8e-3）相比，除了 lr，还有没有别的差异？

**结论：未发现 lr 以外的实质差异（阻断/重要：无）。**

| 项 | 结果 | 证据 |
|---|---|---|
| args | 与主作业只差 `--lr`、`--out`、`--tag`。与 Burst 孪生 run 的 run JSON args 相比，emulate 后正好差 {data, out, tag, tokenizer}，8/8 都是这样，与主批次相同 | a1_check §3 |
| tag / out 路径 | tag 为 `pt2same-53m-lr{4e-3,1.6e-2}-none`，OUT 为 `runs/pt2same/out/pt2same-53m-lr…-none-s{0-3}`。和主批次 32 个名字不重，和其他批次 1326 个 `runs/*/out*/*` 目录也不重；8 个 job name、log 名都唯一；run_guard 按输出路径 claim（pretrain_range.py:487-489），不会互相挡 | a1_check §4 |
| step-0 lm 参照 | 与主作业同 seed 的参照相同（11.616473 / 11.587234 / 11.565472 / 11.602105）。已用 Burst 验证 step-0 lm 与 lr 无关：seed 0-3 在 4e-3、8e-3、1.6e-2 三个 lr 下的 step-0 lm 逐位相等，孪生 run 的 step-0 lm 减去参照全为 0 | a1_check §3 |
| DRY 摘要 | 哈希与主作业同 seed 相同，并且等于各自孪生 run 的 dry_inputs.txt 前 9 行的哈希（8/8） | a1_check §2、§3 |
| 检查脚本 CHECK_PY | 全文相同。intent JSON 只差 lr（0.004 / 0.016）和 tag，所以 CHECK_PY 会独立核对 trainer 实际记录下来的 lr，是 lr 的第二条路径 | 归一化 diff |
| GPU 守卫、HOLD、sha | 都相同（`*A100*80GB*`、`$R/HOLD`、WANT_SHA 逐模块检查、COMMIT）。bash -n 8/8 通过 | 归一化 diff |
| #SBATCH | 用 scontrol 核对：Nice=20、a100_tandon、gpu48、<slurm-account>、02:00:00、Requeue=1、8 CPU、64G、gres/gpu:1 | scontrol show job |
| sha 校验 | a1/JOBS.sha256 与 8 个文件一致，Slurm 持有的脚本也与之一致（逐字节）。FROZEN.sha256 当前全部 OK；主 JOBS.sha256 也 OK | sha256sum、scontrol write batch_script |

**次要 1a：没有 submit_a1.sh，A1 提交没有经过清单校验和 FROZEN 校验。**
- **位置：** make_jobs_pt2same_a1.py:4（docstring 写的是「a1/submit_a1.sh --go does」，但 a1/ 下没有这个脚本）、:46（清单是生成器拿自己刚写出的文件算出来的，只能自证）。对照 submit_pt2same.sh:19-21。
- **为什么：** 提交顺序是 1.6e-2（21696761-764）在前、4e-3（765-768）在后，和清单顺序相反，说明是直接循环 sbatch，没走清单。所以提交时既没有 `sha256sum -c`，也没有 FROZEN 校验。
- **检查：** 我事后用 `scontrol write batch_script <id> -` 和文件比对，8/8 相同。AUDIT 的 Submission 段可以补一句：A1 是怎么提交的，以及事后核对的结果。
- **置信度：** 高。实际影响：无。

**次要 1b：A1 的 out 目录在主读数根目录 runs/pt2same/out/ 下，不在 a1/ 下。**
- **位置：** make_jobs_pt2same.py:179（`OUT = R / "out" / name`）；a1 作业第 23 行。
- **为什么：** readout_pt2same.py 只按 PICKS 拼路径，不 glob（见 Q5），所以主判词安全。但用 glob 数 `out/*/check.json` 的进度监控或「32 个完成」之类的计数会数成 40；以后的工具如果按目录名推断 cell，也可能把 lr4e-3 当成主 cell。目录名里没有 a1 标记。
- **检查：** 监控和计数只认主 JOBS.sha256 里的 32 个名字；也可以用 `ls runs/pt2same/out | grep -c lr8e-3-none`。
- **置信度：** 高。实际影响：低。

**次要 1c：A1 生成器跳过了 main() 的跨批次撞名断言和 stray 检查。**
- **位置：** make_jobs_pt2same.py:335-338、:347-348（只在 main() 里执行），A1 生成器只调用了 render()。
- **检查：** 已手工核对，A1 的 8 个名字和 1 个 tag 不在任何 `runs/*/out*/*` 目录里，a1/ 下也没有多余的 job 文件。
- **置信度：** 高。实际影响：无。

**次要 1d：HOLD 共用。** a1 作业检查的是 `$R/HOLD`（R = runs/pt2same，作业第 23 和 25 行）。想只停 A1 就必须同时停掉主批次还在排队的作业，反过来也一样。如果要分开控制，只能 scancel（按规则那需要用户来做）。置信度：高。

**次要 1e：** AUDIT 的磁盘估计（Cost and disk 节，约 15.5 GB）没有算 A1 的 8 × 约 0.21 GB ≈ 1.7 GB。时间 6 A100-h 已经写进 Amendment。置信度：高。

## Q2. Burst 孪生 run 选对了吗？render() 的孪生断言在 A1 下还有效吗？

**结论：未发现问题。孪生选对了，断言没有被绕过。**

- **孪生是哪几个：** 4e-3 对应 350/351/352/353-pt2b-lr4e-3-none-s0..3（BLOCKS 的基础任务）；1.6e-2 对应 362/363/364/365-pt2b-lr1.6e-2-none-s0..3（EXTRA，open-bracket 规则加的）。我做了两次独立核对：
  1. 直接用正则扫 Burst 的 tasks/*.sh 文件名，不经过 G2.tasks()；
  2. 用 G2.tasks() 列出 pt2b none 的全部 lr（1e-3、2e-3、4e-3、8e-3、1.6e-2）。

  两者一致：每个 seed 恰好一个任务，seed 4-7 没有这两个 lr 的任务，也没有写法不同但数值相等的 lr（比如 0.004）。
- **孪生 run 的事实：** run JSON 里 args.lr = 0.004 / 0.016，seed、target none、arith_lines 1、tag 都对。check.json pass，已完成 75 个 eval，code_sha a3bce3ab63c1，节点 A100-SXM4-40GB。area：4e-3 是 26.12 / 8.48 / 9.92 / 14.47（均值 14.75），1.6e-2 是 18.33 / 12.57 / 20.02 / 9.89（均值 15.20），都和各自的 check.json 一致。
- **render() 的断言（make_jobs_pt2same.py:186-218）都还在起作用：** render() 的 lr 来自被改写的 `PICKS[size][arm]`，bname 取自包装函数给的 bt，之后的每条断言都去读磁盘上的 Burst 文件，和包装函数怎么选无关：
  - :188，bname 的格式必须是 `NNN-pt2b-lr{lr}-none-s{seed}`；
  - :192，任务文件里 SEED / TAG / ARM 那一行；
  - :195，ARGS 替换后与任务文件逐字符相等；
  - :197-207，flag 表（包括 `--lr`）；
  - :209-213，intent 与任务文件内嵌的 intent 完全相等（包括 lr 的浮点值、seed、tag）；
  - :216-218，DRY 值出现在任务文件里，并且等于 Burst dry_inputs.txt 的哈希。

  所以只要包装函数选错了 lr 或 seed，这些断言就会失败。A1 生成器在 :37 还额外断言了 `--lr {lr} ` 和 name。我在内存里重渲染了一遍，8/8 与文件逐字节相同。
- **render() 不检查的内容（主批次同样不检查）：** 孪生 run 是否完成、是否 pass。主批次的 args_diff.py 补了这一项，A1 没有对应测试。我已经补查：8/8 pass。
- **PICKS 改写的副作用：** 在 :33 改写，:44 还原（不是 exception-safe，但进程随后就退出）。render() 和 intent() 是 PICKS 唯一的消费者。readout_pt2same.PICKS 是另一个对象，不受影响（已断言）。

## Q3. burst_twins() 去掉 32 格完整性断言，会不会让某个检查失效？

**结论：未发现失效。**

- burst_tasks() 在 make_jobs_pt2same.py:128 的完整性断言防的是两件事：期望的格子缺了孪生，或者多出了不该有的键（只可能是 pick 那个 lr 下的其他 seed）。
- 在 A1 里，这两件事分别由以下几处兜住：
  - :25 的重复键断言（同一个 lr 和 seed 不会对到两个任务）；
  - :27 的 `>= {0,1,2,3}`（缺 seed 就失败）；
  - render() 的 `btasks[...]`（缺键会 KeyError）；
  - Q2 列出的格式和 intent 断言（选错就失败）。
- `>=` 允许 seed 4-7 存在，但 A1 只渲染 0-3，而且这两个 lr 实际上也没有 4-7 的任务。
- A1 根本不调用 burst_tasks() 和 jobs()，所以 PICKS 被改写期间，主批次的 32 格表不会被重算或污染。readout 的 BURST_TASK 是写死的，A1 也不碰它。

## Q4. 「F86's pick, fixed in advance」只是注释，还是会被当真？

**次要。只是注释，没有任何程序读它。但它在出处上有两处是错的。**

- **位置：** make_jobs_pt2same.py:235-240（about 注释）；a1 作业第 13-17 行。
- **没有程序读它：** 我 grep 了 llmadd_torch 下所有 .py 和 .sh，「fixed in advance / F86's pick」只出现在生成器本身和 tests/e2e 的副本里：
  - readout_pt2same.py 不读作业文件；
  - readout_known.py 和 render_check.py 只 glob `runs/pt2same/job_*.sbatch`，不递归进 a1/；
  - CHECK_PY 的 intent 和 check.json 的 info 里没有 pick 字段；
  - 作业日志里打印的 INTENT 行写的是中性的「none lr 4e-3 seed 0 … Burst twin 350-pt2b-lr4e-3-none-s0」，注释本身不会进日志。
- **出处上的两处错误：**
  1. 「lr … (F86's pick)」，对 A1 不对。
  2. 「generated by llmadd_torch/make_jobs_pt2same.py (change the generator, not this file)」，实际是 runs/pt2same/make_jobs_pt2same_a1.py 生成的。直接重跑主生成器得不到这 8 个文件。
- **风险：** 以后有人或 agent 从作业头抽取出处（写 PROVENANCE.md 或 FINDINGS）时，可能把 4e-3 或 1.6e-2 记成 pick。
- **检查：** AUDIT 的 Submission 段已经写明（AUDIT.md:209-210）。建议在 PROVENANCE 和 FINDINGS 里写 A1 时也注明「生成器 = make_jobs_pt2same_a1.py；lr 是敏感性 lr，不是 pick」。已经提交的文件不能改，也不需要改。
- **置信度：** 高。

## Q5. 读数会不会把 a1 的 run 误读进主判词？A1 的敏感性读数该怎么加？

**主判词被 a1 污染：未发现。**
- readout_pt2same.py 的路径完全由写死的 `PICKS`（:42-43）和 `run_paths`（:61-74）拼出，没有 glob、listdir 或 iterdir。
- 我把 160 条主读数路径（torch 48 × 2，burst 32 × 2）全部列出来检查，没有一条含 lr4e-3 或 lr1.6e-2。
- 完整性循环只覆盖 PICKS 里的格子，所以 A1 的 run 缺失或失败不会挡住主判词，A1 的值也不会混进主判词。
- `RO.PICKS` 与生成器的 `M.PICKS` 是不同的对象。

**重要 5a：A1 读数还没实现，按字面写很容易测错。要在任何 A1 结果出来之前把定义写进 AUDIT。**
- **位置：** AUDIT.md:193-196（A1 第 3 条：「best-of-{4e-3, 8e-3, 1.6e-2} on Torch seeds 0-3 by pt2main rule 1, and the slope recomputed with that cell」）；readout_pt2same.py:118-137（boot_idx 和 slope_table 都把 8 seed 写死了）。
- **为什么会测错（三个陷阱）：**
  1. **seed 数对不上。** 新的 lr 只有 seed 0-3。拿「替换后的 4 seed 斜率」去比主判词的 8 seed 斜率，会把 lr 的影响和 seed 半段的影响混在一起（Burst 上 53M none@8e-3 的 seed 0-3 均值 23.1，seed 4-7 是 13.8）。如果把 seed 0-3 用 best lr、seed 4-7 用 8e-3 拼成 8 个，同一个格子里就混了两个 lr。如果把 4 个 seed 的数组塞进 RO.slope_table：长度 4 会触发 IndexError，这是好事；但一旦有人为了让它跑通而补齐到 8 个，它就会静默算出结果。
  2. **选择偏差。** 在同样 4 个 seed 上取 3 个 lr 中的最大值，再用这 4 个 seed 评估，none 的均值会系统性偏高。用 Burst 的数据估计：每 seed 的 area 在不同 lr 之间几乎不相关（相关系数 -0.53 / 0.26 / -0.46，平均 -0.24；同一个 seed 在不同 lr 下的结果基本是独立抽样），seed SD 约 6。在「三个 lr 真值相等」的零假设下，max-of-3 的偏差为 **+2.5 到 +2.8 area 点**。OLS 权重 w53 = -0.317，所以斜率会被推高 **+0.8 到 +0.9**，方向为正。这正好和 I2 担心的「迁移代价让斜率偏负」方向相反，而且量级接近 build 审查说的「约 4 点就能翻判词」。n=4 时 lr 之间差异的 SE 约 4.3 点，规则 1 在 Torch 上重新选 lr 的分辨力很低。
  3. **规则 1 的 bracket 子句没有说明。** 如果 best 落在 4e-3 或 1.6e-2 这个边缘，并且比邻居高出 2 以上，按 pt2main 规则就要往外再加一个 lr。A1 没有预注册这一步，所以必须说明：「bracket 开口，不扩展，只作描述」。
- **建议的实现（不碰主判词）：**
  1. 写一个新文件，比如 runs/pt2same/readout_pt2same_a1.py。**不要改 readout_pt2same.py**：它在 FROZEN.sha256 里，一改 submit_pt2same.sh:21 就会拒绝，而且等于改了预注册。新脚本启动时断言 readout_pt2same.py 的 sha256 仍是 bf7cced43238…，输出写到单独的 readout_a1.json，不写进 readout.json。
  2. `import readout_pt2same as RO` 只读使用。**绝不修改** RO.PICKS、RO.ROOTS、RO.F86：比如设 `RO.PICKS["53m"]["none"]="4e-3"` 再调 RO.read()，主判词就会被换成 A1 的格子，seed 4-7 缺失还会变成 no verdict。脚本结束时断言 RO.PICKS 没变。先跑 RO.read(RO.ROOTS) 得到主判词，并断言它与已保存的 readout.json 的 verdict 和 slope 相同。主判词不可用时（no verdict 或 PENDING），A1 部分只打印「primary not available」，不计算。
  3. 数据：12 个 run，即 53M none 在 {4e-3, 8e-3, 1.6e-2} 下的 seed 0-3。8e-3 用主批次的同一批文件。路径按 RO.run_paths 的格式手工拼，lr 作为显式参数传入：`out/pt2same-53m-lr{lr}-none-s{s}/pt2same-53m-lr{lr}-none/run_whole_none_seed{s}_pt2same-53m-lr{lr}-none.json`。每个 run 用 `RO.load_run` 做同样的完整性检查（finished、check pass、area 双路径误差 0.01 以内）。任何一个不是 ok，敏感性就输出「not computed」并列出原因，**不允许静默退回 8e-3**。发散的 run 按 pt2main 的规定算作数据（CHECK_PY 对发散会判 pass，见 make_pt2main_tasks.py:241）。
  4. 规则 1：`RR.best_lr({lr: {"area": [4], "off": [...], "intr": [...]}})`，near_rule 用默认的 "within"，TIE 为 2。offline bpb 的取法照抄 pt2main_summary.py:102-104：在 offline_lm.json 里找 tag、seed 都匹配且 max_windows == 0 的条目，再用 `RR.offline_bpb(entry, diverged)`。in-training bpb 取 `curve[-1]["val_bits_per_byte"]`。然后用 `RR.bracket(lrs, mean, best)`，结果是 open 时打印「bracket open, not extended」。
  5. 输出（全部只作描述）：
     - (a) 3 个 lr 的 Torch 表：seed 0-3 的逐 seed 值和均值，并排放 Burst 的 14.75 / 23.10 / 15.20 以及逐 seed 的 Torch − Burst。这一项直接回答「站点有没有移动最优 lr」，也就是 I2 的问题。
     - (b) 规则 1 的 best、near 集合，以及 tie-break 用的是 offline 还是 in-training bpb。
     - (c) **只做 seed 0-3 对 seed 0-3 的比较**：`S_h1_primary = RO.slope_table(G, RO.SIZES)["h1"]`，再算 `S_h1_A1`：把 G["53m"][0:4] 换成 win-ctx@2e-3 减去 none@best（逐 seed 配对），其余尺寸不变，三个尺寸各取 seed 0-3 的均值后做 OLS。打印恒等式 `S_h1_A1 − S_h1_primary = +0.317 × (none@best 的均值 − none@8e-3 的均值)` 作为自检。**绝不拿它和 8 seed 的 sl["all"] 比较。**
     - (d) 选择偏差的修正行：leave-one-seed-out。对每个 s，在另外 3 个 seed 上用规则 1 选 lr，然后取 s 在该 lr 下的 area。4 个值的均值作为基本无偏的「Torch 上选出的 none」，用它再算一条 S_h1。
     - (e) 如果要给 CI，自己写 4 seed 的联合 bootstrap（`np.random.default_rng(0).integers(0, 4, (10000, 4))`），并注明 4 seed 只有 35 种不同的重抽组合，CI 只供参考。不要复用 RO.slope_table，它的 H1、H2 和 boot_idx 都是按 8 seed 写的。
  6. 自测，只用合成数据，照 RO.selftest 的 _fake_tree 写法：
     - best = 8e-3 时，S_h1_A1 == S_h1_primary，误差在 1e-12 以内；
     - 4e-3 比其他 lr 高 5 以上时，替换生效，斜率变化等于 0.317 × 差值；
     - 两个 lr 相差在 2 以内时，按 bpb 打破并列；
     - 1.6e-2 在边缘并高出 2 以上时，打印 bracket open；
     - 缺一个 A1 run 时输出 not computed，且主判词和 readout.json 逐字节不变；
     - 跑完后 RO.PICKS 不变。
  7. 写 FINDINGS 时，把 A1 写成描述性内容，明确说明它是 seed 0-3 的斜率、带选择偏差（附 LOSO 那一行），并且不改变主判词。
- **检查：** 把上面第 5 步 (c) 的恒等式和第 6 步的自测写进 tests/，并在任何 A1 结果出来之前把定义追加进 AUDIT（另立一节 Amendment，不改 A1 原文）。
- **置信度：** 高（选择偏差的量级是用 Burst 数据模拟出来的，取决于 seed SD 约 6、跨 lr 不相关这两个假设）。

## 核对清单（可复现）
- 渲染一致：在内存里用 render() 重渲染（PICKS 改为 4e-3 / 1.6e-2，孪生取自任务文件名），8/8 与 a1 文件逐字节相同；a1/JOBS.sha256 与文件一致；Slurm 持有的脚本与文件一致（8/8）。
- 与主作业的差异：把 lr 字符串、孪生名、intent 里的 lr 浮点值替换回去后，作业体与同 seed 的主作业逐行相同；注释按词比较也相同（8/8）。
- 孪生：任务文件和 G2.tasks() 一致，每个 seed 恰好一个；run args 的 lr / seed 正确；pass 并已完成；code_sha a3bce3ab63c1；args 只差 {data, out, tag, tokenizer}；step-0 lm 与参照差 0，且与 lr 无关（逐位相同）；DRY 一致。
- 撞名：与主批次 32 个名字不重，与其他批次 1326 个 run 目录不重；job name 唯一。
- 读数：160 条主读数路径都不含 A1 的 lr；RO.PICKS 独立。
- FROZEN.sha256 当前全部 OK；make_jobs_pt2same_a1.py 不在 FROZEN 里（sha256 1a3922dd…）。

## 最可能测错的地方
作业本身没问题。风险在还没写的 A1 读数：一个 4 seed、在同样 seed 上选过 lr 的「替换后斜率」，很容易被拿去和 8 seed 的主斜率比，或者被拼成 8 seed。这样得到的是「lr 加 seed 半段加选择偏差」三者的混合，偏差方向为正，约 +0.8 到 +0.9，和要检验的效应同量级。要在 A1 结果出来之前把定义写死：只比 seed 0-3 对 seed 0-3，附 LOSO 修正，写成独立脚本，不修改 RO 的全局变量。
