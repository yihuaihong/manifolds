# pt300x 构建审查（全新、盲审、只读；2026-11-02）

审查对象：make_jobs_pt300x.py e8fd563c335a，select_pt300x.py c4f07df1e18c，readout_pt300x.py c595c9dbb70c，
submit_pt300x.sh c3fbf808e3fd，64 个 stage-A job 文件，code7（a3bce3ab63c1）。对照：make_jobs_pttpp.py b4bfa6eb248d
（= FROZEN.txt）、runs/pttpp、runs/ptlr、runs/ptlr_1b、FINDINGS F87 / F97 FIXSLOPE。
没有改任何文件，没有提交或取消作业。只写了本文件；独立计算的脚本放在会话 scratchpad（leverage.py、power_c.py、
crit_region.py、alts.py、xcheck.py），都只读 run JSON。

## 摘要
- **阻断：未发现。** 训练条件、种子流、步数、eval、窗口、CHECK7、GPU 守卫、HOLD、sha 检查、B200 续跑都和 F87 / pttpp 一致，
  代码层面没发现会让训练或读数静默出错的 bug。四个复用格子的数我独立重算了一遍：4 点斜率 +3.252，残差 90% 区间
  [+0.102, +6.401]，和 KA 一致。
- **重要 1（最可能测错）：判词几乎在登记前就定了。** 两个新尺寸都在内部，只占斜率权重的约 23%。在合理范围内
  （93M 增益 15–35，294M 15–40），准则 (a)、(b) 一次都不会失败，只有 (c) 能失败。即使 53M→294M 完全不涨
  （两个新点都约 23.2），(c) 仍有 74–90% 的概率通过，判 "SHOWN (increase)"。
- **重要 2：294M 放 B200。** 能区分“线性增长”和“294M 之前平、之后跳升”的新点只有 294M 一个：按线性它约 29.6，按平台约
  23.5，相差约 6 点。桥测得 B200 的 gap 偏移是 +4.3，方向和趋势相同，量级也相同；桥本身只有 4 个 seed，none 的容差是 8
  点，pttpp AUDIT 自己写着“只能检出粗大偏移”。另外，294M 在 pttpp 里是 L40S、在这里是 B200，同一尺寸跨批次混用了硬件。
- **重要 3：读数没报 LM 梯度占比**（manifold_aux_loss_principle 要求报），pttpp 读数里有的“无 b11 面积”也丢了。pttpp 里
  93M 的 LM 占比中位数只有 0.54–0.56，其他尺寸是 0.80–0.91；有 36–38% 的 on-eval 先验梯度大于 LM 梯度。所以 λ=5 在 93M
  上很可能是另一种剂量。
- 次要 11 条，见下。

## 问题清单（按严重程度）

### 阻断
未发现。

### 重要

**I1 判词几乎由登记前已看过的四个点决定；新数据只能靠 (c) 起作用，而且区分不了“平台后跳升”**
- 位置：AUDIT.md:141-149（Decision rule）、AUDIT.md:166-169（Rough expectation）、readout_pt300x.py:195-207（verdict）、
  :309-311。
- 为什么会测错：
  - 六点 OLS 的斜率权重是 53M −0.192、93M −0.099、190M −0.013、294M +0.040、505M +0.096、1.07B +0.169，两个新点合计只占
    |权重| 的 23%。
  - 网格扫描 93M 增益 15–35、294M 增益 15–40（每格 2.5，共 99 种组合，各新格沿用 190M / 505M 的 seed 偏差形状），
    得到 SHOWN 55 次、仅 (c) 失败 44 次，(a)、(b) 失败 0 次。
  - 只看均值做 Monte Carlo，新点的 SD 取 2.0 或 3.41：
    - 两个新点都落在 4 点线上时，(c) 通过的概率是 0.97–1.00；
    - **53M 到 294M 完全平（两点都约 23.2）时，仍有 0.74–0.90**，b 的中位数 +3.1。
  - 两个新点单独算斜率：即使正好落在线上，也只有 +3.25，seed 90% 区间 [−0.25, +7.40]，自己没有检验力。
  - 因此 "SHOWN (increase) … across six sizes" 主要是把 F97 FIXSLOPE（事后、非预注册）再认证一遍。F97 记录的形状本来就是
    “53M→190M 几乎平（+0.35/e 倍），190M→505M +6.8”，而这个规则分不出线性和“平台后跳升”。
  - AUDIT 写了 “only partly confirmatory”，但没写 (a)、(b) 实际上已定、(c) 在零效应时也大概率通过。
- 可验证的检查：scratchpad 的 power_c.py 和 crit_region.py，它们用 readout_pt300x 自己的 slope_block / verdict，并读真实复用
  JSON。
- 建议：任何 pt300x 结果出来之前，以 amendment 的形式选一项（是否改由用户决定）：
  1. 加一个只看新数据的预注册准则，例如 93M、294M 的增益各自落在 4 点线（a、b 现在就固定）的 90% 预测区间内，并且
     294M − 190M > 0；
  2. 或者把判词改成“与 FIXSLOPE 一致 / 不一致”，不叫 SHOWN；
  3. 至少在判词行旁边打印 “P((c) passes | flat to 294M) ≈ 0.74–0.90”。
- 置信度：高（独立计算，用的是脚本自己的统计函数）。

**I2 294M 用 B200：唯一有区分力的新点带着同方向、同量级的硬件偏移；294M 跨批次混用硬件**
- 位置：make_jobs_pt300x.py:62（HW）、AUDIT.md:83-95；依据见 runs/pttpp/readout_full_20261030T2118.txt:113-117 和
  runs/pttpp/AUDIT.md:152-157。
- 为什么会测错：
  - 桥的数据：505M、4 个 seed，none 在 B200 上低 4.0（逐 seed −13.0、+1.0、−2.7、−1.5，SE 约 3.2），gap 是 +31.7 对
    +27.4。“PASS”只说明偏移小于 8 点。
  - pttpp AUDIT:157 原话：“The bridge detects only gross shifts (4 seeds)”。
  - 偏移方向：B200 上 none 更低 → 增益更大。两个 B200 尺寸都在 ln N 的高端，B200 指示变量和 ln N 的相关是 0.62。
  - 对斜率本身影响很小（294M +1 点 → b +0.04，AUDIT 的算法是对的），但对 I1 里唯一有区分力的比较影响很大：294M 是
    “约 29.6”还是“约 23.5”，差约 6 点；一个 +4 的 B200 偏移就能把平台情形的 294M 从 23.5 推到约 27.8。
  - 灵敏度 (i)、(ii) 只作描述，判词不依赖它们。
  - 跨批次：pttpp 的 294M 在 0.745 / 0.318 两条线上是 L40S，pt300x 的 294M@3e8 是 B200。以后若把 pt300x 并进 F97 回归
    （gain ~ a + b ln N + c ln T）或做 294M 的 token 比较，c 会吸收硬件差异。这和用户“同一尺寸不混卡”的原意相冲突，只是
    批内一致。
  - 桥只在 F87 的 L40S 最优 lr 上测过，没测 B200 是否改变 lr 最优点或发散阈值；294M 的 lr 是在 B200 上选的。
- 可验证的检查：readout 输出 (ii) L40S-only 斜率及其 df-2 区间；把 hardware (i) 的 c 和 294M − 190M 的差并排看。
- 建议：用户已经定了“越快越好”，这里只报告，不擅自改。可以选的做法：
  - 预注册判词必须同时写出 L40S-only 的 b 及其符号；
  - 在 AUDIT 里登记 294M 跨批次硬件不一致，以及任何跨批次回归都要带硬件指示变量；
  - 如果改到 L40S：每个 run 约 4.07 h，stage A 约 137 L40S-h。
- 置信度：中高。

**I3 读数没报 LM 梯度占比和裁剪系数（项目规则）；也丢了 pttpp 读数里的“无 b11”行**
- 位置：readout_pt300x.py:320-351（descriptive 部分没有 grad 和 B5）；对照 runs/pttpp/readout/tools/pttpp_summary.py:514
  （grad_stats）和 :700-712（print_grad），以及 :694-697（“without b11”）。
- 为什么会测错：
  - 记忆规则 manifold_aux_loss_principle 要求每个先验 arm 都报窗口内 LM 占比和 clip，先验梯度超过 LM 时要标出来。
  - pttpp 读数第 133-134 行：93M 在 tpp 0.318 / 0.745 两条线上，LM 占比中位数 0.54 / 0.56，先验 > LM 的 on-eval 占
    36% / 38%。其他尺寸是 0.79–0.93（53M@21.2 0.81、190M@2.65 0.88、505M@0.745 0.91、1.07B@0.318 0.91）。
  - 如果 93M@3e8 也这样，93M 这个点就处在“先验主导更新”的另一种状态，“λ=5”在不同尺寸上意义不同。
  - 对 FIXSLOPE 混杂 6（“固定 λ 不等于固定剂量”），本批没有加任何测量。
  - trainer 在每个 prior-on 的 eval 都记了 grad_lm / grad_prior / grad_cos（pretrain_range.py 的 grad_info），数据都在，
    只是读数没算。
  - 这也违反了 audit_port_fixes：兄弟脚本 pttpp 读数里已有的两项诊断没有继承。
- 可验证的检查：读数之后用 pttpp_summary.grad_stats 的定义，对 6 个格子的先验 arm 算窗口内 LM 占比中位数。
- 建议：结果出来前加为描述性输出（不进判词），93M 占比 < 0.6 时标注。另外加 6 个格子的 B5（无 b11）面积斜率。
- 置信度：缺失这件事是高；对 93M@3e8 有影响是中。

### 次要

**M1 8 seed 均值里有 4 个是选 lr 用的 seed（winner's curse），大小随尺寸变**
- 位置：readout_pt300x.py:104（seed 0–3 来自 stage A / X）、AUDIT.md:137-139。
- 为什么：none 的单 seed SD 在 6.1–9.7 之间（53M 6.13、190M 6.24、505M 9.65、1.07B 7.57，我的计算）。4 seed 均值在 2 个
  并列 lr 中取最大，粗估 seed 0–3 的 none 被抬高约 0.56σ_mean ≈ 1.7–2.7 点，8 seed 增益相应被压低一半，而且 σ 越大
  压得越多（大尺寸）。1.07B 是在 2 个 seed 上选的。方向对正斜率偏保守，但会改变 (c) 的残差。
- 检查：单列 seed 4–7（无选择）的斜率和它的残差区间，现在 (b) 只看符号。置信度：中（估计值）。

**M2 factor-2 网格对 none 粗、对先验细；各尺寸的档位对齐不同**
- 位置：make_jobs_pt300x.py:65-66。
- 为什么：F87 里 53M none 偏离最优一档约掉 8 点（4e-3 −7.8，1.6e-2 −8.4），先验几乎平（≤ 1.9）。按插值最优点，93M none
  离最近一档 0.35 档（6.13e-3），294M 是 0.49 档（6.73e-3）或 0.25 档（5.69e-3）。按约 8 点/档² 的曲率，离散化让 none
  少约 1–2 点，增益相应偏大，大小因尺寸而异，会加进 (c) 的尺寸层残差。这就是 FIXSLOPE 混杂 3，但没有量化。
- 检查：每个尺寸对 none 的 4 个 lr 均值拟合 log-lr 二次曲线，估计离档损失。置信度：中。

**M3 lr swap 灵敏度静默漏掉 FINDINGS 列出的并列先验 lr**
- 位置：readout_pt300x.py:79-87（只搜 ptlr 的 `{pre}-lr*-7pct-{arm}` 目录，seed 0–3 不全就静默跳过）。
- 为什么：alts.py 实测，复用格子只找到 53M win-ctx 4e-3、190M win-ctx 2.4e-3 两个并列备选。FINDINGS 列的并列先验 lr 是
  53M 1e-3（ptgw；ptlr 里的 1e-3 目录只有 seed 4–7）、190M 6e-4（ptsize999）、505M 3e-4，全都没进来。所以
  “N of M swaps change the sign”会少数。只影响描述性输出。
- 检查：alts.py 的输出。置信度：高。

**M4 “1/3 和 2/3 深度”在新尺寸上是同名不同义**
- 位置：code7/tools/pretrain_range.py:271-272（n_layers // 3），make_jobs_pt300x.py:163。
- 为什么：93M 是 3、6 / 10 层（0.30、0.60），294M 是 6、12 / 19 层（0.316、0.632）；其余尺寸都正好是 1/3、2/3。先验和
  读出层的位置因此偏浅。AUDIT 写了层号，但没列为偏差。先验接近天花板，影响多半很小。置信度：高（事实），影响：低。

**M5 bootstrap 在每个尺寸内独立抽 seed，没考虑同一个 seed 在各尺寸共享数据流和 held split**
- 位置：readout_pt300x.py:150-153。
- 为什么：pretrain_range.py 的 held_out(rng(seed)) 和 data_rng(1000+seed) 在所有尺寸上相同，F97 的逐 seed 斜率也是这么
  配对的。若跨尺寸正相关，独立抽样会高估斜率方差，所以偏保守，不会制造假阳性。照登记执行即可，写明“偏保守”。
  置信度：高。

**M6 OWN_FAILURES 没有 F87 读数那样的 WARNING**
- 位置：select_pt300x.py:51-55、:104-106。
- 为什么：拼错的名字会被静默忽略，那个 run 会一直显示 wait；列入的 run 即使已经完成，也照样按失败计分（bpb 记最差），
  而且不做 integrity 检查。F87 的 ptscale_summary 对这两种情况都会打印 WARNING。置信度：高。

**M7 unstable 的参考 lr 固定是 grid[0]**
- 位置：select_pt300x.py:145。
- 为什么：none 向下做了 X（1.2e-3）之后，“该 arm 最低的 none lr”就变了，但参考仍是 2.4e-3。只影响标记。置信度：高。

**M8 被 HOLD 或 TIMEOUT 打断的 B200 run 显示为 running，不是 AUDIT 说的 missing**
- 位置：AUDIT.md:206-208；make_jobs_pt300x.py:253。
- 为什么：重新排队的 B200 作业遇到 HOLD 会 exit 0；超过 3 h TIMEOUT 的作业不会自动 requeue。两种情况都会留下部分 JSON
  和 ckpt，select 会一直显示 "running"。不会静默丢 run，但也不会提示需要重投。本批没有 hook / watcher，只能靠人工发现。
  置信度：高。

**M9 读数依赖 /scratch 上 64 个复用 JSON（10-16 到 10-26 写入），没有备份和 sha 清单**
- 为什么：/scratch 会 purge，读文件不刷新 atime，而且已用 94%。文件丢了会得到 NOT DETERMINABLE（不静默），但会卡住结论。
  建议提交前先 tar 到 /archive，并记下 sha256。置信度：中。

**M10 “F87 rule 1 exactly”只是选点规则一样，扩展规则比 F87 严**
- 为什么：F87 的 505M none 和 pttpp 的 294M@0.745 none 都扩展过两次，而本批规定一次 X 之后仍开着就 STOP，读数就是
  NOT DETERMINABLE。none 最优 lr 的插值本身不确定，出现 STOP 的概率不小。这是流程问题，不会测错。置信度：中。

**M11 b11 的 held pair 在自然文本里出现（每 20 个里 13–17 个），而且污染程度随尺寸变化**
- 为什么：token 数固定时，大模型更会记住自然文本里的和，none 的 b11 可能随尺寸变高，增益斜率会被压低（方向保守）。
  pttpp 读数有“without b11”行，本批没有（见 I3）。置信度：中。

## 逐题回答

### 1. 和 F87 300M 格子（ptlr）、pttpp 同形状格子相比，训练条件哪里不一致
| 维度 | F87 ptlr（53M/190M/505M） | ptlr_1b（1.07B） | pttpp 93M / 294M（0.745 线） | pt300x 93M / 294M |
|---|---|---|---|---|
| tokens / steps | 默认 3e8 / 18310 | 同 | 31.3M / 1908、156.8M / 9570 | 默认 3e8 / 18310 ✓ |
| eval | 每 250 步，75 次 | 同 | 每 26 / 131 步，75 次 | 每 250 步，75 次 ✓ |
| warmup | 183 | 183 | 19 / 95 | 183 ✓ |
| 窗口 | 916..7323 | 同 | 按比例 | 916..7323 ✓（events、on_frac 由 CHECK7 断言） |
| λ / site | 5 / ctx | 同 | 同 | 同 ✓ |
| 层 | 1/3、2/3 | 12、24 / 36 | 3,6 / 10；6,12 / 19 | 同 pttpp（M4） |
| 数据 / seed 流 | pretrain_data；rng 2000+s、1000+s、split s | 同 | 同 | 同 ✓（运行时检查 data sha；CPU dry digest = pttpp 日志） |
| 精度 | bf16 autocast，190M 以上 grad ckpt，lm_chunk 0 | 同 | 同 | 同 ✓（93M off、294M on，由 CHECK7 断言） |
| 环境 | open_r1：torch 2.10.0+cu128、transformers 4.57.6（2026-04-14 安装，之后没变） | 同 | 同 | 同 ✓ |
| ckpt / save | 2000 / 每个 run 都存 | 500 / 只在选中的 lr | 0 / 只在 C | L40S 0、B200 500 / 只在 C |
| 硬件 | L40S | B200 | L40S | 93M L40S、294M B200（I2） |
- ckpt_every 和 save_model 只影响续跑粒度和模型落盘：torch.save 不推进 RNG，续跑时恢复 model、opt、rng、data_rng、
  torch / cuda rng、dyn 和 curve，见 pretrain_range.py:492-498 和 :583。所以不改变训练数字。
- 其余都一致；step-0 lm 参考在 F87 和 pttpp 之间逐位相同（190M s0–3），说明初始化和前向没有漂移。

### 2. lr 网格和选择规则
- 选点规则和 F87 rule 1 一致：seed 0–3 面积均值取最大，前两名差 ≤ 2 时选训练中 bpb 更低的（near_rule 'top2'）。F87 的
  ptlr 格子也没有离线全量 bpb，FINDINGS 写的是“只有训练中 65k token 的 bpb”，所以“只用训练中 bpb”和 F87 一致。
- 开口判定用 readout_rules.bracket，diverged / unstable / own failure 的处理和 F87 rule 2 一致。
- 唯一不同：扩展只允许一次（M10）。
- 网格：每个都覆盖插值点两侧，而且各有至少一档：
  - 93M none 6.13e-3，在 4.8e-3 和 9.6e-3 之间；
  - 93M win-ctx 1.53e-3；
  - 294M none 6.73e-3 / 5.69e-3；
  - 294M win-ctx 1.2e-3 正好在网格上，上方只有一档，但 pttpp 的上界论证（token 更少时 lr 更高或相同，所以 3e8 下最优
    ≤ 2.4e-3）成立。
- 没有发现明显偏向某一 arm 的网格。结构性不对称有两点：factor-2 网格对 none 粗、对先验细（M2）；none 的噪声大，选点本身
  就噪（M1）。
- 先验在各 lr 上近乎平，前两名几乎总在 2 点以内，所以它的 lr 实际上是由 65k token 的训练中 bpb 决定的，F87 也是这样。

### 3. 预注册分析和读数脚本
- 写清楚、可执行。(a) percentile bootstrap、(b) 两半 seed 的符号、(c) b ± t(0.95, 4)·sqrt(SSR/4/Sxx)，代码分别在
  readout_pt300x.py:139-147 和 :156-207，和 AUDIT 一致。NOT DETERMINABLE 和 INVESTIGATE 的条件也已实现。
- 我用自己的代码重算 4 点：b +3.252，区间 [+0.102, +6.401]，与 KA 一致。
- 没有发现静默丢 run：
  - 复用格子缺 seed 或未完成 → problem；
  - 新格子 missing / running / nocheck → notfinal，withheld → problem；
  - own failure 计分；
  - 判词要求 6 个格子全部 ok。
- seed 配对：seed 0–3 来自 sel.recs[(pick, s)]，seed 4–7 来自 C，顺序是 0..7，ap − an 按位相减 ✓。
- area 定义：readout_rules.area 和 CHECK7 的 area 都是对 75 次 eval 取平均，两条路径比较的容差 0.006 ✓。
- 1.07B：pick 是 2.4e-3 / 6e-4（REUSED 有断言），seed 0–7 是 stage A（0–1）+ B（2–3）+ C（4–7）各一个文件 ✓；swap
  不变动它的 lr（有说明）✓。
- 问题在“测什么”而不在代码：见 I1（判词基本已定）、M3（swap 漏项）、I3（缺诊断）。
- 小问题：“the coordinator records the investigation here before anyone reads the verdict”无法强制，脚本会在同一屏把
  verdict 打出来。

### 4. 硬件、HOLD、sha、B200 续跑、硬件桥
- **卡型：**
  - 93M：`--partition=l40s_public`，加上 `case *L40S*`（exit 4）；
  - 294M：`--constraint=b200`，加上 `case *B200*`（exit 4）；
  - X / C 用同一个 job_x 渲染；没有绕过守卫的路径 ✓。
- **HOLD 和 sha：** HOLD、trainer sha（exit 3）、data sha256（exit 3）都在训练开始前检查 ✓。HOLD 只在开头查，运行中的作业
  不会停（有说明）。运行时只检查 pretrain_range.py 的 sha，不检查 numlib / pretrain_helix / run_guard（快照只读，F87 也
  这样）。PYTHONPATH=pylib 里没有会遮蔽 torch / transformers / numpy 的包 ✓。
- **B200 续跑：**
  - job 头和 pttpp 的 B200 作业逐行相同（只差名字），trainer 也一样，所以可靠性和 pttpp 1.07B 相同。
  - ckpt 存在 (s+1)%500==0 的步上，这时最近一次 eval 是 s−249，续跑从 s+1 开始，curve 也从 ckpt 恢复，所以 eval 不重复、
    不缺；CHECK7 断言 75 次 eval 的步号正好对得上，select 的 integrity 也会再查一遍。
  - JSON 写到一半被杀：续跑后在下一次 eval 时重写；run_guard 碰到坏 JSON 当作未完成处理 ✓。
  - 风险只在 TIMEOUT / HOLD 后卡住、需要人工重投（M8）。
- **505M 硬件桥是否支持把 294M 放 B200：** 部分支持。它只排除了大于约 8 点的 none 偏移，点估计的方向是“B200 抬高增益
  +4.3”，正好和待检验的趋势同向；量级和 294M 这一点的区分量（约 6 点）相当。偏差方向：斜率偏大、(c) 更容易通过（I2）。

### 5. 意图说明没写、代码却依赖的假设
1. 判词把 6 个尺寸均值当成独立、同方差的点，但 4 个点的值在登记前已知（I1）。
2. B200 ≈ L40S，只用 4 个 seed 在 505M 上验证过（I2）；B200 的软件栈从 bridge（10-30）以来没变。
3. 先验剂量（LM 占比）在各尺寸可比（I3）；93M 在 pttpp 是例外。
4. 64 个复用 JSON 一直在、没被改（有 FINDINGS 校验，但没有备份，M9）。
5. trainer 的默认参数（tokens 3e8、eval_every 250、lam 5、window）不变：快照只读，CHECK7 也逐项比对 ✓。
6. 93M / 294M 的 step-0 lm 参考是 CPU 值（pttpp_refs："CPU for 93M / 294M"），L40S 上差在 1.2e-4 以内；B200 上未实测
   294M，但 505M 上 B200 与 L40S 差在 5e-5 以内，容差 1e-3 足够。
7. 同一个 seed 在各尺寸上数据流和 held split 相同（M5）。
8. x 轴取 ln N_nonemb，嵌入是 tied、V=100278。按 F97，改用总参数时斜率是 +4.70，“每 e 倍”的数值依赖这个定义。
9. 一次 X 就够闭合（M10）。
10. 判 unstable 的参考是 grid[0]（M7）。

### 6. 陷阱清单
- 训练 / 评测分布不一致：和 F87 相同（eval 是 format 0，放在 val 文本行末；训练用 4 种格式、每行 4 道），各尺寸一样。
  新问题：未发现。
- 记忆：b11 的自然文本污染随尺寸变化（M11）。held pair 不进训练行（draw_pair 拒绝）✓。
- 静默失效的损失：CHECK7 断言窗口 events、on_frac、geo > 0、0 < grad_prior < inf，none 从不开 ✓。未发现。
- 单 seed：未发现（每格 8 seed，选择用 4 seed）。
- token 位置：trainer 没改，s=0 时断言 operand 和 answer 的位置 ✓。未发现。
- 同名不同义：
  - “层 1/3、2/3”（M4）；
  - “λ=5”在不同尺寸剂量不同（I3）；
  - “294M”在 pttpp 是 L40S、在这里是 B200（I2）；
  - “F87 rule 1 exactly”只管选点、不管扩展（M10）；
  - “five sizes without 1.07B”不是 L40S-only（读数里已注明）✓。
- 静默过滤：lr swap 的备选（M3）；OWN_FAILURES 写错名字时被忽略（M6）。主分析里未发现。
- 修 bug 的副作用：没改 trainer；generator 只读导入 pttpp，FROZEN 26/26，pttpp 的 302 个 job 文件按位重渲一致。未发现。

## 我做的独立检查（scratchpad，只读）
- xcheck.py：用 numpy.polyfit 重算 4 点斜率和残差区间，结果和 KA 一致。
- leverage.py：用真实复用格子算斜率权重、6 种情形的判词，以及 46×46 网格。
- crit_region.py：在合理范围内统计各准则的失败次数：(a) 0、(b) 0、(c) 44 / 99。
- power_c.py：(c) 在“落在线上”和“平到 294M”两种情形下的通过率；B200 指示变量与 ln N 的相关 0.616，VIF 1.61。
- alts.py：读数在复用格子上实际找到哪些 lr 备选（M3）。
- 核对了环境里 torch / transformers 的版本和安装时间，pylib 没有遮蔽包，l40s 的 test-only 会抢占的作业 21595074 属于别的
  用户。

## 一句话
这个实验最可能测错的地方：六点判词基本由登记前已看过的四个点决定。两个新尺寸都在内部，只占约 23% 的斜率权重；即使
53M→294M 完全不涨，判词仍有 74–90% 的概率是 "SHOWN (increase)"。唯一能区分“线性增长”和“294M 前平、之后跳升”的
294M 点又跑在 B200 上，硬件桥测得的偏移（增益 +4.3，与趋势同向）和这一区分量（约 6 点）同量级。
