# pt2same 提交前对抗审查（build 层，2026-11-02）

审查员：全新子 agent，没参与设计。只读：除了本文件，没有改任何文件，没有提交作业，也没有取消作业。
看过的材料：AUDIT.md、make_jobs_pt2same.py、32 个 job_*.sbatch（diff 了 190M win-ctx s4 和 53M none s0 两个，与 pt2s505 同 seed 的 job 也逐行 diff 过）、readout_pt2same.py、submit_pt2same.sh、tests/*.log、pt2s505 的 AUDIT 和 run 结果、Burst 的 32 个孪生 run（run JSON、check.json、node.txt、PINNED.txt、log_tail），以及 Burst 仓库里 64abc19 到 7f1d06b 的 trainer diff（共 1766 行）。
独立重算用的脚本都在 scratchpad 里，没有写进项目目录：Burst 各 lr 格子的 seed 0-3 均值、pt2s505 各格子的均值和 bpb、32 个孪生 run 的 code_sha / lr / seed / pass、逐 seed 的 Delta、斜率杠杆、混合站点斜率。

## 结论先行
- **阻断：未发现。** 作业和 Burst 孪生 run 的训练条件一致：args 只差 data / tokenizer / out / tag，数据 sha 相同，seed 流相同，模型形状相同，都是 bf16，torch 都是 2.10.0+cu128，驱动都是 610.43.02，eval 集、窗口 [[916,1],[7324,0]]、λ 5、ctx 位点也都相同。作业和 pt2s505 是同一个 trainer（93e22bf4c4c0）、同一个 job 体、同一个分区和 GPU 型号（pt2s505 的 32 个 run 全在 A100-SXM4-80GB 上）。读数脚本与预注册一致，没有静默丢 run，area 没算错，seed 配对也没错。
- **重要（3 条）：** 斜率几乎只由 53M 和 505M 两个点决定，而 53M 的 none 恰好是最脆弱的格子；三个尺寸的 lr 来源不对称，AUDIT 里「pick 附近是平的」这句话对 53M none 不成立；Delta 实际上约等于常数减 none 的 area。
- 次要的有 9 条，见下文。

---

## 重要

### I1. 主斜率实际上只是 53M 对 505M 的比较；190M 的 16 个作业对裁决几乎没有贡献
- **位置：** readout_pt2same.py:41（N_NONEMB）、:128-137（slope_table）；AUDIT.md:108-115（规则 4/5）。
- **事实：** x = ln N_nonemb = 16.466 / 18.545 / 19.814。OLS 斜率是三个尺寸 Delta 的加权和，权重为 **-0.317 / +0.047 / +0.269**，190M 只占权重绝对值之和的 **7.5%**。换成总参数量 N，权重是 -0.459 / +0.040 / +0.420，结论不变。
- **为什么会测错：** 53M 的 8-seed Delta 每移动 4 个 area 点，斜率就变 -1.27。用 Burst 53M/190M 加 Torch 505M 的现有数据重算混合站点斜率，得到 +0.30，90% CI [-0.91, +1.50]，bootstrap SE 0.74，所以 CI 半宽约 1.2。也就是说，53M 的 Delta 变化约 4 点就能把裁决从「no size trend shown」翻成「SHOWN」，或者反过来。而 53M none@8e-3 本身的 seed SD 是 7.2，lr 敏感性约 8 点，正好是最可能动这么多的格子（见 I2）。「三个尺寸的斜率」这个名字会让人以为 190M 参与了裁决。
- **可验证的检查：** `w = xc/(xc@xc)`，xc = ln N_nonemb 去均值。提交前在 AUDIT 里写明权重，并预注册描述性的两两对比 Δ505-Δ53、Δ190-Δ53、Δ505-Δ190（用同一个联合 bootstrap），让 190M 的信息可见。
- **置信度：** 高。这是代数结果，不是猜测。

### I2. lr 沿用 F86 的 pick：数字本身对，但来源对不同尺寸不对称；AUDIT「pick 附近是平的」不适用于 53M none
- **位置：** make_jobs_pt2same.py:67（PICKS）；AUDIT.md:36-41（「the ties above ... say the area is flat near each pick」）。
- **核对结果（独立重算，readout_rules.area，Burst seed 0-3）：**

  | 格子 | lr 网格上的 seed 0-3 均值 | pick | 网格边缘情况 |
  |---|---|---|---|
  | 53M none | 1e-3 9.91 / 2e-3 14.21 / 4e-3 14.75 / **8e-3 23.10** / 1.6e-2 15.20 | 8e-3 | 尖峰，比两侧高 8.4 / 7.9 点，没有并列 |
  | 53M win-ctx | 5e-4 65.77 / 1e-3 70.19 / **2e-3 69.83** / 4e-3 64.20 | 2e-3（与 1e-3 并列，按 bpb 选） | 区间闭合 |
  | 190M none | 6e-4 13.55 / 1.2e-3 19.87 / 2.4e-3 22.24 / **4.8e-3 26.84** / 9.6e-3 28.78 | 4.8e-3（与 9.6e-3 并列，按 bpb 选） | area 随 lr 单调上升，argmax 在上沿 |
  | 190M win-ctx | **6e-4 78.56** / 1.2e-3 76.91 / 2.4e-3 72.86 | 6e-4 | argmax 在下沿，没有跑 3e-4 |
  | 505M none（Torch） | **4.8e-3 24.70** / 9.6e-3 25.59 / 1.92e-2 15.49 | 4.8e-3（并列，按 bpb 选） | 选中的是下沿那个点 |
  | 505M win-ctx（Torch） | 3e-4 78.12 / **6e-4 76.72** / 1.2e-3 75.19 | 6e-4（并列，按 bpb 选） | argmax 在下沿 |

  32 个孪生 run 的 args.lr 分别是 0.008 / 0.002 / 0.0048 / 0.0006，seed 都对，code_sha 都是 a3bce3ab63c1，check 全部 PASS，与 F86 的表和 R02 补记一致。readout_known.log 的 4 个 8-seed 均值（18.43 / 69.91 / 29.06 / 78.45）也重算一致。**选择本身符合 pt2main rule 1：并列的 lr 取 bpb 更低的那个。**
- **为什么会测错：**
  1. **迁移代价只落在 53M/190M。** 505M 的 pick 就是在 Torch 上选的，53M/190M 的 pick 是在 Burst 上选的。如果换站点改变了最优 lr，代价主要落在 lr 曲线陡的 none 上（53M none 的峰尖高约 8 点），prior 的曲线比较平（±2 到 6）。结果是 53M（以及 190M）的 Delta 偏大，而 505M 没有这个偏差。**偏差方向是让斜率偏负**，即「增益随 N 变小」。
  2. **winner's curse 的结构随尺寸不同。** 505M 的 seed 0-3 就是当初用来选 lr 的那几个 run，被原样复用。53M/190M 的 seed 0-3 是用同样 init 和数据的全新重跑，只部分继承 curse。AUDIT 写了「seeds 4-7 selection-free」，这句是对的，但没写 seed 0-3 这一半的偏差大小随尺寸不同。从数据看，505M 的 curse 很小（none 24.70 对 24.19，prior 76.72 对 78.13），53M none 的 curse 很大（23.10 对 13.77）。
  3. AUDIT 用「并列 = 平」论证 pick 挪动的代价小，但 53M none 根本没有并列，这个论证不成立。另外，190M none 在 area 上并没有夹住最优（单调上升，最高点在上沿），190M 和 505M 的 win-ctx 的 area argmax 都在下沿。所以「各自最优 lr」的误差在网格内不超过 2 点，在网格外是未知数。
- **可验证的检查：** 结果出来后先看 53M none 在 Torch 上两个 seed 半段的值。如果 seed 0-3 的均值又是约 23、而 seed 4-7 约 14，说明 curse 来自 init/数据；如果两半都约 14，说明是噪声或站点。另外打印 Δ 的 h1、h2 斜率和逐尺寸表（规则 5 已要求）。提交前在 AUDIT 的 Known risk 里补上两点：「53M none 无并列、峰尖约 8 点」和「迁移代价只在 53/190M，偏差方向为负」。不建议改设计，因为重选 lr 会把站点和 lr 混在一起，这一点 AUDIT 的理由成立。
- **置信度：** 数字核对是高；偏差方向是中（前提是站点真的移动了最优 lr）。

### I3. Delta 主要是常数减 none 的 area，「先验增益随尺寸」的裁决读的其实是 none 的变化
- **位置：** readout_pt2same.py:147-148（verdict 措辞）、:226-229（每个 arm 的斜率，只是描述性输出）。
- **事实：** 逐 seed 的 SD（Burst 53/190M，Torch 505M）：prior 0.9 / 1.1 / 2.0，none 7.2 / 7.9 / 5.4。prior 在 70-79 之间，接近天花板，Delta 的 SD（7.6 / 7.6 / 6.0）几乎全部来自 none。
- **为什么会测错：** 输出写成 SHOWN（+/-）：the prior's gain is larger/smaller at larger N。实际可识别的是 none 在各自 pick 上的 area 随尺寸怎么变，加上 prior 天花板随尺寸的小幅变化。这样写进 FINDINGS 容易被读成「先验机制随规模变强/变弱」。
- **可验证的检查：** 读数时并排列出 none 和 prior 的 area 斜率（脚本已经打印），FINDINGS 条目明写「Delta 斜率 ≈ -(none 斜率) + prior 斜率」；也可以加一个 prior 斜率是否接近 0 的描述。
- **置信度：** 高。这是数据事实，影响的是措辞。

---

## 次要

### M1. 站点效应和 run-to-run 不确定性分不开；符号约定与 AUDIT 相反（同名不同义）
- **位置：** readout_pt2same.py:233-244（`d = t - bu`，即 Torch - Burst）；AUDIT.md:12 引用的 P1b 是 Burst - Torch（-1.55）。
- **原因：**
  - 32/32 个 step-0 lm 与 Burst 逐位相同，说明两边前向 kernel 一样，差别只来自反向的非确定性（atomics），然后在训练中被放大。设计里没有同站点的重复 run，所以「per-seed range」基本是混沌噪声，只有均值差和它的 CI 才是系统性站点效应的估计。
  - 当初 P1b 失败比较的是 Burst A100-40GB 加 a3bce3ab63c1，对 Torch **L40S** 加 79e9b410dcc3（readout_rules.check_2pct），同时混了 GPU、trainer 和噪声三样，不能直接当作 A100-80GB 也会不同的证据。
  - 本批也同时换了 trainer（a3bce3ab63c1 → 93e22bf4c4c0），所以「站点效应」= 站点 + trainer 版本。
- **检查：** FINDINGS 里写清「Torch - Burst」，并注明「无同站点重复 run，per-seed 差主要是非确定性」。可选：同一个 A100-80GB 上把一个 53M seed 跑两次，估计非确定性的量级。
- **置信度：** 中高。

### M2. 预期范围 ±15 太宽，保护不了裁决
- **位置：** readout_pt2same.py:46、:186-195；AUDIT.md:101-105。
- **原因：** 斜率的 CI 半宽约 1.2，53M Delta 动 4 点就能翻裁决（I1）。而 53M none 从 18.4 漂到 33 仍算「in range」，不会触发调查。这条规则只能挡住数量级的错误（比如训练坏掉），挡不住会改变结论的系统偏移。
- **检查：** 加一个描述性标记（不改裁决规则也可以）：|Torch 均值 - Burst 均值| 超过 2×SE 时打印 NOTE。none 的 SE ≈ SD/√8 ≈ 2.6，prior 约 0.4-0.7。
- **置信度：** 高。

### M3. λ 固定为 5，先验的有效强度随尺寸下降
- **位置：** AUDIT.md:29-31。另见 FINDINGS F97 R28 note：300M token 时 LM 梯度占比 0.81 → 0.91。
- **原因：** 斜率混合了两件事：模型容量变化，以及「同一个 λ 在大模型里占的更新份额更小」。这不是测错，但结论必须带上「at fixed λ 5」。
- **检查：** CHECK_PY 的 info 里没有 grad_lm / grad_prior。读数时可以从 run JSON 的 curve 取窗口内的 grad_prior/grad_lm 中位数，按尺寸列一张描述表。
- **置信度：** 中。

### M4. 网格边缘：「各自最优 lr」在 3 个格子里没有被真正夹住
- 190M none 单调上升到 9.6e-3；190M 和 505M 的 win-ctx 的 argmax 在下沿。
- 这是 F86 和 pt2s505 留下来的问题，这里不重复展开（见 I2 表）。会让 Delta 在 190M 偏大（none 低估），也可能偏小（prior 低估）。每个格子的影响不超过约 2 点加上网格外的未知部分。
- **置信度：** 高（事实），中（影响大小）。

### M5. 重跑说明自相矛盾
- **位置：** AUDIT.md:97-100 说基础设施问题导致的丢失「删掉 SUBMITTED_ flag 再 --go」；submit_pt2same.sh:8 说「A submitted job is never resubmitted; a re-run needs a new run name」。
- **原因：** 机制上 AUDIT 的做法可行：脚本只跳过有 flag 的 job，run_guard 会接管已死作业的 claim，trainer 从 ckpt 续跑。但两份文件给协调者的指令不一致，可能导致协调者误改 generator 换名字，结果读数路径找不到 run，变成 incomplete。
- **检查：** 统一其中一处的文字。
- **置信度：** 高。

### M6. AUDIT.md:27 说「Seed s = the same init ... as pt2s505 seed s」，这是错的
- 尺寸不同，参数形状就不同，init 不可能相同。相同的只有数据流和 held-out split。
- 读数的联合 bootstrap 依据的是「同一数据流和 split」，所以没有错，只是措辞问题。

### M7. GPU 守卫和设备回退的小漏洞
- **位置：** job 第 37 行、make_jobs_pt2same.py:61（`*A100*80GB*`）；code/tools/pretrain_range.py:492（`dev = "cuda" if torch.cuda.is_available() else "cpu"`）。
- **原因：**
  - 这个通配模式也能匹配「NVIDIA A100 80GB PCIe」和「A100-SXM4-80GB MIG …」。目前 a100_tandon 的 43 个节点都是 gres gpu:a100，所以没有实际风险。
  - 守卫用的是 nvidia-smi，没有检查 torch.cuda。如果 CUDA 在 torch 里不可用，trainer 会静默退到 CPU 上训练。这种情况会 TIMEOUT，结果是 incomplete，不会产生错误的数字，所以不算静默错误。
  - 环境版本没有在运行时 assert，只记录了 Python 版本。我核对过：open_r1 的 torch 2.10.0、transformers 4.57.6、numpy 2.0.1 自 2026-04 起没有变过；驱动在 node.txt 里有记录，但没有 assert。
- **检查：** 结果出来后核对 32 个 node.txt：GPU 都是 A100-SXM4-80GB，驱动都是 610.43.02（与 pt2s505 一致）；gpu.csv 的利用率中位数大于 0。
- **置信度：** 高（属于低风险项）。

### M8. trainer 换版只在 CPU 上做过回归
- **位置：** pt2s505/AUDIT.md:72（d64×3、seq 64、40 步、单线程，prior 只在 1 个 eval 打开）。
- **原因：** 新增代码都受 --domains / --held_operands / --site emb / --wd / --lam_decay 控制。我读了 diff，默认路径上只有 `batch(s)` 的签名、`lam_t = lam_num` 和 `weight_decay=args.wd` 这几处变化，看不出功能差别。但 GPU 上的 fused AdamW 和 bf16 加 grad ckpt 这条路径没有做过逐值对比。好在 32/32 的 step-0 lm 逐位相同，前向是等价的。
- **检查（可选）：** 在同一张 A100-80GB 上，用 a3bce3ab63c1 和 93e22bf4c4c0 各跑 53M seed 0 none 到 `--stop_after 250`，对比 lm 和 grad_lm。
- **置信度：** 低风险。

### M9. 读数的输入没有冻结
- FROZEN.sha256 只冻结了代码，没有冻结 pt2s505/out 和 llmadd-burst-ro/results。如果这两个目录在读数前被改动（比如重跑、git pull），读数会静默使用新的值。
- **检查：** 读数时把这 48 个 run JSON 的 sha256 打印到 readout.json。
- **置信度：** 低风险。

---

## 逐题回答

1. **新作业与 Burst 原 run 的训练条件差在哪里**
   - run args 只差 data / tokenizer / out / tag，31/35 个 key 相同（args_diff.log；我抽查了 382-…-s4 的 run JSON 与 job ARGS，一致）。
   - 数据：两边都检查同一对 sha256。tokenizer：DRY 的 9 行逐行相同。tag 和 out 只用于输出文件名（pretrain_range.py:486,1815,2001）。meta.json 只用于 digit_tokens 的 assert。
   - seed 流：rng_split(s)、1000+s、2000+s、vr(11)、er(5) 都与模型无关；DRY 逐行相同；step-0 lm 32/32 逐位相同。
   - 模型形状和 n_params 相同；grad_ckpt 只看参数量（>150e6，:917），lm_chunk 只看 bs·seq·vocab（:1690），都不随显存变化，因此 Burst 40GB 和 Torch 80GB 一致。
   - 精度：两边都是 bf16 autocast；torch 都是 2.10.0+cu128（Burst 331-env311 的日志）；驱动都是 610.43.02。
   - eval 集、75 个 eval、窗口 [[916,1],[7324,0]]、λ 5、ctx 位点都相同。
   - **真正不同的只有三样：** GPU（A100-SXM4-40GB 对 80GB，同为 sm_80）、trainer 版本（a3bce3ab63c1 对 93e22bf4c4c0，默认路径在 CPU 上逐位相同）、CPU 数量（12 对 8，不影响结果）。另外 CHECK_PY 的 info.pinned_calls 在 Burst 数的是 PINNED.txt，在 Torch 数的是 COMMIT 的行数，属于同名不同义，但只是 info，不参与任何检查。
2. **与 pt2s505 是否同 trainer、同 job 体、同硬件：** 是。trainer 的 sha 和所有依赖模块都与 pt2s505 相同（WANT_SHA 是同一个对象）。job 体逐行 diff 只差 job 名、时限、注释、路径、ARGS、model 行、DRY、INTENT 和 GPU case。GPU 守卫比 pt2s505 更严，但 pt2s505 的 32 个 run 实际都跑在 A100-SXM4-80GB 上，驱动 610.43.02，每个都没有重启。分区、账户、QoS、nice 都相同，环境自 4 月起没有变。剩下的风险是 11 月以后集群驱动有没有升级（见 M7）。
3. **lr 选择：** 核对正确，与 F86 和 run JSON 一致（见 I2 表）。偏差方向：53M/190M 的 lr 是在 Burst 上选的，换站点可能带来的代价主要压低 none（尤其 53M 的尖峰），使 Delta 在小尺寸偏大、斜率偏负；505M 没有这个代价，但它的 seed 0-3 就是选 lr 用的 run（不过实测 curse 很小）。此外 190M none 和 190M/505M win-ctx 的 area 最优在网格边缘或网格外。
4. **读数与预注册：** 一致。完整性检查覆盖 48 个 run（finished、check pass、area 双路误差 0.01 以内）；Burst 的缺口只会关掉站点表，并且会打印出来，不会静默丢 run。area 用的是 readout_rules.area（6 个 bucket、全部 eval）。Delta 按 seed 下标配对；Burst 任务号映射核对正确（350-353/370-373、366-369/358-361、382-385）。联合 bootstrap 对所有尺寸共用同一组 idx（default_rng(0)）。两个半段的规则和 slope = 0 的分支都正确。问题不在实现，在规则的效力：见 I1、M2。
5. **会不会在非 A100-80GB 上训练，或在 HOLD / sha 不符时仍训练：** 不会。每次启动（包括 requeue）依次检查 HOLD、sha 和 COMMIT、GPU 型号、数据 sha，然后才进入 dry run 和训练；e2e 测试的 C/D/E/G 四个负对照都在训练前退出。漏洞只有 M7：PCIe 或 MIG 名字也能通过守卫（当前分区里没有这种卡），以及 torch 没有 CUDA 时会退到 CPU（结果是 TIMEOUT，不会静默产出数字）。HOLD 只在作业启动时检查，这是设计如此。
6. **意图说明没写、但代码依赖的假设：**
   - 190M 在主斜率里的权重只有 7.5%（I1）；
   - 非确定性是零均值的，且各尺寸量级相当（M1）；
   - 驱动和环境在 10 月与 11 月之间没有变化（运行时不 assert，M7）；
   - pt2s505 和 Burst 的结果目录在读数前不会变（M9）；
   - 「pick 附近是平的」（对 53M none 不成立，I2）；
   - λ 的有效份额与尺寸无关（M3）。
7. **陷阱清单：**
   - **训练/评测分布：** 一致。位置在行尾，eval 用 format 0，属于训练用过的 4 种格式之一。
   - **记忆：** held-out 的 pair 从不出现在训练行里。b11 的 pair 会出现在自然文本里，但 b11 的 area 没有单调的尺寸趋势（Burst 53M/190M + Torch 505M 的 none 分别是 27.4 / 44.0 / 35.4），所以 b11 不会单独造出斜率；仍建议把不含 b11 的 area5 斜率作为描述性输出。
   - **静默失效的损失：** CHECK_PY 检查了 geo > 0、0 < grad_prior < inf、prior_used = on；none 组检查了 prior 从未打开。
   - **单 seed：** 每个格子 8 个 seed，但 53M 每个 seed 都在 Burst 跑过，而且有 curse（I2）。
   - **token 位置：** 记分在答案加换行，dry 行打印了这个位置。
   - **同名不同义：** 站点效应的符号（M1）；「N」在格子名里是总参数量、在斜率里是 N_nonemb；pinned_calls 的含义。
   - **静默过滤：** 未发现。
   - **修 bug 的副作用：** trainer 改版只在 CPU 上回归过（M8）。

## 最可能测错的地方
主裁决几乎完全由 53M 对 505M 的 Delta 差决定（190M 只占 7.5% 的权重），而 53M 的 none@8e-3 是一个在 Burst 上选出的、带 winner's curse 的尖峰（seed 0-3 是 23.1，seed 4-7 是 13.8，两侧邻居约 15）。它在 Torch 上只要偏移约 4 点，斜率就会移动一个 CI 半宽。所以这个实验最可能测到的是「53M none 在迁移过来的 lr 上落在哪里」，而不是 prior 增益随尺寸的趋势。
