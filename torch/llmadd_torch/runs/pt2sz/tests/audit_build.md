# pt2sz 构建审查（全新、对抗、只读；2026-11-05，提交前）

审查对象（sha256[:12]，与 AUDIT.md「Files at writing」一致）：make_jobs_pt2sz.py af2ba3425158、select_pt2sz.py ab95a029492c、
readout_pt2sz.py 37ddcba5105a、submit_pt2sz.sh 2a0ee7915e6f、pt2sz_refs.json bb92a90edc26、tests/sim_null.py 14a0e39e4cd1、
64 个 stage-A job 文件（拼接 sha 4c5266f2ea5b）、runs/pt2sz/code（7f1d06b，pretrain_range.py 93e22bf4c4c0）。
对照（只读）：runs/pt2same、runs/pt2s505、runs/pttpp（2% tpp 线）、runs/pt300x/tests/audit_build.md、FINDINGS F98 / F99。

没有改任何文件，没有提交或取消实验作业。本文件是唯一写入的项目文件。独立计算都在 /scratch/yh6210/tmp_audit_pt2sz/
（sim_indep.py / sim_u.py / sim_offset*.py / wc.py；只读 run JSON 和 pt2same readout.json，**不导入** readout_pt2sz 或
sim_null）；CPU 部分通过 cpu_run.sh 跑在 cpu_short（作业 22006117、22007193）。

## 摘要
- **阻断：未发现。** 训练 / 评测条件、数据流、held split、eval 行、步数、窗口、CHECK_PY、GPU 守卫、HOLD、sha 检查、
  续跑、area 定义，新尺寸与三个复用格子逐项一致；读数里的斜率、within-size bootstrap、halves、残差 t 区间与 AUDIT.md
  写的一致；没找到会让数字静默出错的代码 bug。
- **重要 I1（最可能测错）：选 lr 带来的格子级偏移和要测的效应一样大。** 两个新点能区分「沿旧线增长」和「不增长」的
  对比只有 294M − 93M ≈ +2.3 点（约 1 个 seed SE）。新格子的 pick 在相邻档之间换一档，增益就动 0.9–1.4 点（505M
  Torch 实测）。模拟（真实效应为零、只让 294M win-ctx 的 pick 低一档，先验 +1.4）：P(SHOWN) 从 0.36 升到 **0.56**；
  93M −1.0、294M +1.4 时是 **0.69**，与「新点正好在旧线上」的检验力（0.68）相同。也就是说，单靠选点就能造出一个
  与真实增长无法区分的 SHOWN。新旧格子的选点规则、网格位置、硬件也不同（旧 53M / 190M 是 Burst 上按另一条规则选的）。
  294M win-ctx 的网格比两个邻居往下多伸一档（1.5e-4），面积又随 lr 单调下降（190M、505M 都是这样），所以「top2 落在
  1.5e-4 / 3e-4、bpb 选 3e-4」这条路径是现实的，方向偏向 SHOWN。
- **重要 I2：判词大半在登记前已定（独立复核了 AUDIT 的模拟）。** 在三个旧格子已观测的条件下，新点完全不涨时，
  主判据读成 SHOWN (increase) 的概率是 0.18–0.40；新点正好在旧线上时是 0.68。所以无论结果如何，似然比只有约 2–4：
  SHOWN 不能叫「确认」，NOT SHOWN 也不能叫「推翻」。(b) 在条件零假设下 98% 通过，形同虚设。AUDIT 的数字我独立重算
  后一致（差 ≤ 0.03）。
- 次要 12 条，见下。

## 问题清单（按严重程度）

### 阻断
未发现。

### 重要

**I1 选 lr 造成的格子级偏移（规则、网格位置、seed 0–3 复用）与新点要分辨的差同量级；判词没有任何一项计入它**
- 严重程度：重要。
- 位置：make_jobs_pt2sz.py:98-99（GRID）；AUDIT.md:84-89（294M 网格「两档在下」的理由）、:121-126（rule choice）、
  :187-190（known differences）；select_pt2sz.py:157-160（F87 rule 1：top2 + 最后一次 eval 的 in-training bpb）；
  readout_pt2sz.py:123（格子的 seed 0–3 就是选点用的 stage-A run）、:197-234（判据里只有 seed 误差和 5 点残差）。
- 为什么会测错：
  - 要测的量是「各 arm 在自己 pick 上的增益」。pick 由噪声和网格边界决定。505M Torch 网格（seed 0–3，wc.py）：none 4.8e-3
    24.70 / 9.6e-3 25.59（差 0.9，靠 bpb 选），win-ctx 3e-4 78.12 / 6e-4 76.72 / 1.2e-3 75.19（每档约 1.4–1.5，都在 2 点内，
    靠 bpb 选）。用同一规则做参数化重抽，505M none 有 37% 会选到 9.6e-3，win-ctx 3e-4 / 6e-4 / 1.2e-3 是 34% / 56% / 10%。
    换一档时整个 8-seed 格子的增益移动 0.9–1.4 点（seed 4–7 也跟着在那一档上跑）。
  - 新点的信号：旧线预测 93M 49.27、294M 51.59，平线是 50.31，各差约 1 点；能分辨两者的 294M − 93M 是 +2.31，seed SE
    约 2.4–2.7。所以一次换档的偏移就和信号一样大。
  - 独立模拟（sim_offset_small.py，条件在 F98 三格上，新点真实值都在 50.31，正态 seed 噪声 σ 5.48，固定 bootstrap 抽样同登记
    代码；n=1500）：
    | 新格子的系统偏移（无真实尺寸效应） | P(SHOWN increase) |
    |---|---|
    | 无（AUDIT 的 C 情形） | 0.36 |
    | 294M 先验 +1.4（win-ctx 选低一档） | **0.56** |
    | 294M none +0.9（none 选高一档，增益 −0.9） | 0.24 |
    | 93M 先验 +1.4 | 0.20 |
    | 93M −1.0、294M +1.4 | **0.69** |
    | 294M +2.0 | 0.61 |
    | 参考：新点正好在旧线上（真实增长） | 0.68 |
  - 方向上有一条现实路径：294M win-ctx 网格 1.5e-4–1.2e-3，比 190M（Burst 网格最低 6e-4）和 505M（最低 3e-4）都低。若面积
    像 190M、505M 那样随 lr 单调下降、每档不到 2 点，top2 就是 {1.5e-4, 3e-4}，bpb 再选较高的 3e-4。这时 294M 的先验落在
    比两个邻居低一档的位置，增益约 +1.4，正是上表第二行。93M win-ctx 网格同样往下多伸到 3e-4（190M 没跑过 6e-4 以下）。
  - 旧格子的 pick 来源不同：53M / 190M 是 Burst A100-40GB 上用 pt2main rule 1（within 2，offline bpb）在 F86 网格上选的，
    跨站点同 seed 相关只有约 0.35；505M 和新格子是 Torch 上、在同一批 seed 0–3 上选的。AUDIT 把「规则不同」列为已知差异，
    但没量化，也没给对应的读数。
  - winner's curse 本身对增益影响不大（wc.py：505M seed 0–3 在 pick 上 none 抬高约 +1.2、先验 +0.9，8-seed 增益净约 −0.15），
    主要问题是「换到哪一档」，不是同一档上的抬高。
  - (a) 的 bootstrap 只含 seed 误差；(c) 理论上包含格子级散布，但只有 3 个自由度，而且 3 个旧格子的散布是固定的。
- 能验证的检查（stage A 结束、写 C 之前即可做，不需新 run）：
  1. 每个新 arm 打印 seed 0–3 在每一档的面积和 bpb；标出面积是否在网格上单调（294M win-ctx：1.5e-4 > 3e-4 > 6e-4 吗？）。
  2. 用 seed 0–3 算新格子增益的选点灵敏度：登记规则的 pick、pt2main rule 1 的 pick（select 已打印，但没算增益）、面积
     argmax、邻居的 lr（294M：none 4.8e-3 / win-ctx 6e-4）。把这几种增益的极差和判词并排打印。
  3. 用 sim_offset_small.py 的做法，把实测的换档幅度代进去，报告「无尺寸效应、只有选点偏移」时的 SHOWN 概率。
- 建议（是否改由用户决定）：在任何 pt2sz 结果之前，以 amendment 加入 2. 的描述性输出；或者登记一个稳健性条件，例如
  「换成邻居 lr / argmax 后，seed 0–3 斜率仍同号且 (a) 仍成立」才写 SHOWN。
- 置信度：机制读代码确认；幅度来自真实 505M 网格的参数化估计和模拟（中高）。

**I2 主判据的结论大半在登记前已定；新点只能把赔率改变 2–4 倍（独立复核 AUDIT 的模拟）**
- 严重程度：重要（AUDIT 已写明，这里是独立复核，并补充两个推论）。
- 位置：AUDIT.md:192-214（模拟）、:156-163（判据）；readout_pt2sz.py:221-234（verdict）、:386-388；tests/sim_null.py。
- 为什么会测错：
  - 斜率权重 53M −0.262、93M −0.108、190M +0.034、294M +0.122、505M +0.214（我复算一致），旧点占 |w| 的 69%。新点完全平、
    恰在旧点均值 50.31 时，点估计 b 就已经是 +1.17（Σ w·y 直接算）。
  - SHOWN 区域（sim_null 的网格）要求 294M 比 93M 高约 1–6 点；新点相等时网格对角线上没有一格 SHOWN。所以 SHOWN 与否几乎
    只取决于 294M − 93M 的噪声（加上 I1 的选点偏移）。
  - (b) 由旧格子的两半（+1.84 / +1.03）锁定：条件零假设下 98% 通过。
- 独立复核（sim_indep.py，自写 slope / bootstrap / halves / 残差 t，只读 readout.json；n=4000）：
  | 情形 | 我的 | AUDIT（sim_null.out） |
  |---|---|---|
  | C：新点平在旧均值 50.31，固定 bootstrap 抽样（同登记代码） | 0.364 | 0.362 |
  | 同上，每次重新抽 bootstrap | 0.396 | — |
  | 同上，新点 seed 噪声按 ρ=0.25 条件在旧格子残差上（AUDIT 写的局限） | 0.380 | — |
  | 同上，额外格子级散布 τ=1.6 / 3.0 | 0.335 / 0.267 | 0.338（τ 1.6） |
  | 同上，新格子 seed SD ×1.5 | 0.236 | — |
  | C：平在 53M 水平 47.99 | 0.181 | 0.161–0.198 |
  | C：平在 505M 水平 52.98 | 0.195 | 0.180–0.233 |
  | 平台：93M = 53M 的 47.99、294M = 190M 的 49.95（新点不比下邻居高） | 0.504 | — |
  | U：五格全抽、真实值全平，τ 0 / 1.6 | 0.021 / 0.032 | 0.025 / 0.031 |
  | L：新点在旧线上（检验力） | 0.681 | 0.692 |
  - 方法本身没问题：用的是登记代码自己的统计函数，场景定义清楚；AUDIT 提到的「忽略新旧 seed 相关」影响 ≤ 0.02。
  - 补充 1：模拟只放了 seed 噪声和对称的 τ，没有放 I1 的选点偏移。加上后，零假设下的 SHOWN 概率可以到 0.56–0.69。
  - 补充 2：SHOWN 的似然比 = P(SHOWN | 在线上) / P(SHOWN | 平) ≈ 0.68 / (0.18–0.40) ≈ 1.7–3.8；NOT SHOWN ≈ 0.32 / (0.60–0.82)
    ≈ 0.4–0.5。读数时应把这两个数和判词写在一起。
- 能验证的检查：重跑 sim_indep.py / sim_offset_small.py（/scratch/yh6210/tmp_audit_pt2sz/）。
- 建议：FINDINGS 里的判词旁边写「新点不涨时也有 0.18–0.40 会读成 SHOWN（选点偏移下可到 0.56–0.69）」；new-vs-old-line
  那一栏才是新证据，但它同样受 I1 影响。
- 置信度：高（两套独立实现，结果一致）。

### 次要

**M1 新尺寸上「n_layers // 3、2·n_layers // 3」同名不同义（先验层和读出层偏浅），AUDIT 的已知差异里没列**
- 位置：pretrain_range.py:921-922；make_jobs_pt2sz.py:221；AUDIT.md:21-27（表里只有层号）、:187-190。
- 为什么：93M 是第 3、6 / 10 层（0.30、0.60），294M 是 6、12 / 19（0.316、0.632），三个旧尺寸都正好是 1/3、2/3。这和
  pt300x 构建审查 M4 是同一点，没有沿用到本 AUDIT。两个新点的权重几乎抵消（−0.108 / +0.122），对 b 影响很小，但会直接
  进入「新点 vs 旧线」那一栏（AUDIT 说那是新证据）。旁证：F99（7%）里同一个 93M 形状比 4 点线高 4.5，原因未查明。
- 检查：在 known differences 里补上这一条；读数时把 93M 的残差和这个已知差异写在一起。置信度：事实高，影响低–中。

**M2 seed 0–3 的 winner's curse 没对新格子量化；seed 4–7（无选择）的斜率只看符号**
- 位置：readout_pt2sz.py:123、:387（halves 只取点估计）；AUDIT.md:188-189（只提了 53M none）。
- 为什么：新格子和 505M 的 seed 0–3 就是选点用的 run，53M / 190M 只受部分影响（Burst 选的）。估计幅度见 I1（8-seed 增益净
  约 −0.15，小）。
- 检查：描述性地打印 seed 4–7 单独的 5 点斜率和它的残差区间。置信度：中。

**M3 93M 的 LM 梯度占比 < 0.6 只标注，没有后果**
- 位置：readout_pt2sz.py:64、:433-442；AUDIT.md:178-180。
- 为什么：pttpp 里 93M 形状在 7% 低 token 线上 LM 占比只有 0.54 / 0.56（其他尺寸 ≥ 0.79）；2% 整体更低（F98 0.68–0.76，
  pttpp 2% 0.58–0.75）。若 93M@3e8 2% 低于 0.6，这个点就是「先验主导更新」（manifold_aux_loss_principle），λ=5 在 93M 上等于
  另一种剂量，但它照样进斜率。F99 的 93M@3e8 7% 是 0.86，所以不一定会发生。
- 检查：stage A 后看 93M win-ctx 的 grad_lm / grad_prior。建议（用户定）：LM 占比 < 0.6 的新格子和超出期望范围同样处理，
  判词带上「INVESTIGATE BEFORE READING」。置信度：中。

**M4 trainer 先删 ckpt、后存模型；run_guard 只看 curve 判断「已完成」**
- 位置：pretrain_range.py:1997-2001；run_guard.py:19-23；job 体（make_jobs_pt2sz.py:390-392，set -e）。
- 为什么：存模型的那几秒里被抢占时，run JSON 已经完整、ckpt 已删、模型还没写。requeue 后 trainer 直接退出，offline_lm
  找不到模型，set -e 让作业失败，不会有 check.json，select 一直显示 nocheck；按 rule 5 重投也一样。只影响能否跑完，不会测错；
  与 pt2same 相同，窗口很短。检查：出现 nocheck 且模型缺失时按此诊断。置信度：读代码确认。

**M5 OWN_FAILURES 只认 seed 0–3 的名字；C 阶段（seed 4–7）的自身失败无法计分**
- 位置：select_pt2sz.py:56-64（names 只含 G.SEL）；readout_pt2sz.py:123-127。
- 为什么：C run 自己出错时没有对应规则，读数会一直是 NOT DETERMINABLE（不会测错，但会卡住）。AUDIT rule 5 也没覆盖 C。
- 建议：事先写明 C 阶段自身失败怎么处理。置信度：高。

**M6 own failure 不做 integrity 检查；它的「无 b11」增益用的是不完整曲线的均值**
- 位置：select_pt2sz.py:118-120；readout_pt2sz.py:424-431。
- 为什么：列入 OWN_FAILURES 的 run 直接 status ok，参数不对也不会发现；d5 用 R.area(p_, R.B5)，是已有 eval 的均值，而主面积
  是 75 个槽、缺的记 0，两种口径不同（只影响描述性的 no-b11 行）。置信度：高。

**M7 HOLD 的说明不完整**
- 位置：AUDIT.md:258-259；job 体 make_jobs_pt2sz.py:359。
- 为什么：AUDIT 写「遇到 HOLD 退出 0、没有 run JSON、选点显示 missing」。但一个被抢占后 requeue 的作业遇到 HOLD 时已经有部分
  JSON 和 ckpt，select 显示的是 running，不是 missing（pt300x 构建审查 M8，没改到文字里）。置信度：高。

**M8 磁盘写满中途崩溃，算 own failure 还是 infrastructure，没有事先规定**
- 位置：AUDIT.md:104-112（rule 5）、:248-251（Disk）；submit_pt2sz.sh:31-33（只在提交时查一次 97%）。
- 为什么：/scratch 已用 94%，同时还有 ptsz（L40S，含 731M）在提交。torch.save 写 ckpt 失败会让 trainer 抛异常；如果被列成
  own failure，那个 lr 的缺失 eval 记 0，会把选点推离它（直接影响 I1）。建议：写明「磁盘 / 文件系统错误 = infrastructure」。
  置信度：中。

**M9 选点不查 node.txt，读数才查；shell 守卫与读数对 GPU 名的要求不一致**
- 位置：select_pt2sz.py:111-143（没有 node_ok）；readout_pt2sz.py:72-80（要求恰好 "NVIDIA A100-SXM4-80GB"）；job 体
  `*A100*80GB*`（make_jobs_pt2sz.py:365）。
- 为什么：守卫会放过 "A100 80GB PCIe"，读数会拒绝，而且要等 C 跑完才发现。按 AUDIT，a100_tandon 全是 SXM4，实际无影响。
  置信度：高（不一致）/ 影响低。

**M10 Python 环境没有固定也没有记录**
- 位置：job 体 make_jobs_pt2sz.py:367（PYTHONPATH pylib，open_r1 env）。
- 为什么：trainer 和模块 sha 都查了，但 torch（2.10.0）/ transformers（4.57.6）版本没查。现在与 F98 时相同（site-packages
  4 月以后没改，pylib 10-13 以后没改）。stage A 和 C 之间如果环境变了（升级或 /scratch flush 后重装），只有 step-0 lm
  （只测前向）可能发现。建议：node.txt 里记 `python -c 'import torch, transformers; print(...)'`。置信度：中。

**M11 复用输入在可被 purge 的 /scratch 上，没有备份和 sha 清单**
- 位置：readout_pt2sz.py:45-47、:57-59（readout.json、F98 的 48 个 run JSON）；make_jobs_pttpp.dry2（渲染 X / C 时要读 Burst
  的 dry 日志 /scratch/yh6210/llmadd-burst-ro）；refs 依赖 pttpp tpp2-294m 的 run。
- 为什么：文件丢了会得到 NOT DETERMINABLE 或渲染断言失败，不会静默；但会卡住（pt300x 构建审查 M9，没沿用）。Burst 目录是
  GitHub 仓库的 clone，可以恢复。建议：结果前把 F98 的 48 个 run 目录 tar 到 /archive 并记 sha256。置信度：中。

**M12 「期望范围 ±15」太宽，只能拦住粗大错误**
- 位置：readout_pt2sz.py:61-63、:357-368。
- 为什么：增益允许 33–68，none 允许 5.6–43。几点的测量偏差（I1 的量级）完全不会触发。另外 INVESTIGATE 标记是加在判词后面，
  判词仍然照常打印（:389-390），AUDIT 说「先调查再读判词」只能靠人自觉。置信度：高。

## 逐题回答

### 1. 训练 / 评测条件哪里不一致？新尺寸与三个复用格子之间
- 序列位置、前缀长度、模板、数值范围：未发现不一致。trainer 是同一个 93e22bf4c4c0；训练行 = train.bin 文本 + 行尾 1 行
  算术（4 种 format 随机）；eval 行 = val.bin 文本 + 行尾 format 0 的一行，长度 512，teacher forcing，打分范围是答案加换行；
  held pair 不进算术行。93M 的 dry 文件除 model 行外与 pt2same 190M 同 seed 的逐行相同，294M 与 pttpp tpp2-294m 逐行相同
  （tests/dry_digest.out，我抽看了 tests/dry 下的解码行）。
- 数据切分、数据流：同 seed 在所有尺寸上用同一 held split（rng seed）、同一数据流（1000+s / 2000+s）、同一 eval 文本偏移
  （rng 11）。所以跨尺寸同 seed 相关（ρ≈0.25），within-size bootstrap 对此偏保守（AUDIT 已写）。
- 模型状态、精度、分片：bf16 autocast、单卡、AdamW、clip 1.0、1% warmup、cosine 到 0.1，全部是 trainer 默认，相同。
  grad ckpt：93M 关（同 53M），294M 开（同 190M / 505M），由 n_params > 150e6 决定。
- 作业体：与 pt2same 190M 的 diff 只有名字、nice、时限、注释、路径、INTENT、ARGS、model 行、DRY hash、CHECK_PY 的 run 文件 /
  intent，以及尾部的 RC 捕获、删模型、exit $RC（我自己 diff 了一个 294M none 作业）。`... | tee || RC=$?` 在 pipefail 下
  能拿到 python 的 rc 4，失败时不删模型。
- GPU：同为 Torch a100_tandon A100-SXM4-80GB（守卫 + 读数查 node.txt）。区别是 F98 的 53M / 190M lr 是在 Burst 上选的。
- 评测、area 定义：同一个 readout_rules.area（e91ee8a8a846），75 个 eval 全部取均值，含 step 0。
- **真正不同的**只有：形状（含 M1 的相对深度）、lr 规则 / 网格 / 选点用的硬件（I1）、tpp（AUDIT 已写）。

### 2. 数字看起来合理但其实测错了的地方
- token 位置、off-by-one：未发现（evaluate 用 logits_to_keep=max_ans+1，预测位与目标位对齐；CHECK_PY 与 readout 的 area
  同一公式，双路容差 0.006）。
- 指标名实不符、平均方式：未发现。gain 是逐 seed 配对的 win-ctx − none；斜率是 5 个 8-seed 均值的等权 OLS。
- 静默丢样本：未发现。缺 run、未完成、没 check 都会阻止判词；own failure 按 75 槽计 0（M5、M6 是边角）。
- lr 选择规则不同造成的不公平：**是，见 I1。** 这是本批最可能「数字合理但测错」的地方。
- 斜率、bootstrap、残差区间的实现：逐行对照 AUDIT.md:156-163，一致——(a) 每个尺寸一组 (10000, 8) 的 default_rng(0) 抽样，
  按 N 升序，5/95 百分位；(b) seed 0–3 与 4–7 的点估计同号；(c) b ± t(0.95, 3)·sqrt(SSR/3/Sxx)。我的独立实现复现了 AUDIT
  的模拟数字（I2 表）。F98 统计用的 joint bootstrap = readout_pt2same.boot_idx，一致。

### 3. 记忆、泄漏、偶然
- 泄漏：held pair 不进算术行，但 b11（及部分 b12）会在自然文本里出现，大模型更会记住，none 的 b11 可能随尺寸上升、压低增益
  斜率。读数有「无 b11」描述行，方向偏保守。
- 偶然：seed 0–3 同时用于选点和估计（M2）；pick 本身受噪声支配（I1：505M none 换档概率约 37%）。F98 I5 的「少数学得晚的
  none seed」在新格子里也可能出现，读数打印了中位数斜率。
- 误判 SHOWN 的概率：**AUDIT 的方法正确，数字我独立复现**（条件零假设 0.18–0.40，无条件 0.02–0.03，检验力 0.68）。但它没
  包含选点偏移；加上一档的偏移后，零假设下可到 0.56–0.69（I1）。

### 4. 可能静默失效的 loss / 梯度
- 未发现。先验 loss 通过前向 hook 取 geo_layers 的输出；CHECK_PY 要求每个 on-eval 的 geo > 0、0 < grad_prior < ∞，ctx 下
  prior_used_frac = prior_on_frac，none 从不开启；win-ctx 与 none 的 step-0 lm 逐位相同。294M 的 grad ckpt（非 reentrant）
  与 190M / 505M 同一条路径，已有实证（这些格子先验 arm 面积远高于 none）。clip 1.0 会同时缩放两项，读数报告 LM 占比和
  clip 系数（M3）。

### 5. 意图说明没写、代码却依赖的假设
- 参考值：294M 用 pttpp tpp2-294m 在 A100-80GB 上的 GPU 值（CPU−GPU ≤ 1e-4，3e8 与 tpp 参数下 CPU 值逐位相同）；93M 只有
  CPU 值（校准误差 ≤ 1.8e-4，容差 1e-3）。不合格只会 WITHHELD，不会测错。
- 删模型的时机：只在 check 通过后删，所以格子 seed 0–3 的模型都不保留（只留 C 的 16 个）。以后结果层审查或任何需要重新评测
  模型的分析，seed 0–3 都做不了。AUDIT 写了，属设计选择。M4 的时序问题。
- 重投 / 续跑：依赖 run_guard 用 squeue 判断「另一个作业是否在跑」，squeue 超时就当作在跑（新作业直接退出）。改 flag 名后
  `--run` 重投时，job 文件若与渲染不同会被改写（flag 不存在了）。C 阶段自身失败没有规则（M5）；磁盘满的分类没规定（M8）。
- nice 与优先级：nice 1000 只影响排队；结果不受影响。时间顺序是 294M 全部在前，硬件相同，不构成混杂。
- 磁盘：提交时查一次 < 97%，作业里不查；峰值约 16 × (3.5 GB ckpt + 1.2 GB 模型)，再加上同时提交的 ptsz（M8）。
- 环境版本没固定（M10）；复用输入没备份（M11）。

### 6. 陷阱清单
| # | 陷阱 | 结论 |
|---|---|---|
| 1 | 训练 / 评测分布不一致 | 未发现（同 trainer、同行布局、format 0 在训练分布内；dry 行逐行对上） |
| 2 | 记忆代替泛化 | held 不进训练行；b11 的自然文本泄漏随尺寸变化，有无 b11 描述行；seed 0–3 选点后复用（M2） |
| 3 | 静默失效的 loss / 梯度 | 未发现（CHECK_PY 逐 eval 查 geo / grad_prior；none 从不开） |
| 4 | 单 seed 偶然 | 8 seed；但选点本身受噪声支配（I1），F98 I5 的晚学 seed 可能重现 |
| 5 | 读错 token 位置 | 未发现 |
| 6 | 同名不同义 | 「各 arm 的 pick」新旧规则 / 网格 / 硬件不同（I1）；「1/3、2/3 层」在 93M / 294M 不是 1/3、2/3（M1）；「λ=5」剂量随尺寸变（M3）；LM 占比两种定义都打印了 |
| 7 | 静默过滤样本 | 未发现；缺失会阻止判词。边角：C 的 own failure 无法计分（M5） |
| 8 | 修 bug 的副作用 | 未发现：冻结文件 sha 断言；regress_frozen 重渲染 pt2same / pt2s505 / pttpp 全部作业逐字节相同；只新写 runs/pt2sz 和 pt2sz_refs.json。尾部改成 `|| RC=$?`，e2e T1–T3 覆盖 |

## 我做的独立检查（/scratch/yh6210/tmp_audit_pt2sz/，只读）
- sim_indep.py（cpu_short 作业 22006117，n=4000）：自写判据，复现 AUDIT 的 C / L 情形，并加了重新抽样、条件相关、τ=3、
  seed SD×1.5、平台情形。sim_u.py（作业 22007193）：U 情形 0.021 / 0.032。sim_offset_small.py（本地，n=1500，SE 约 0.013）：选点偏移表（I1）；
  同一脚本 n=4000 的 cpu_short 作业 22007305 后来跑完，结果一致（SE 约 0.008）：0.562 / 0.233 / 0.186 / 0.691 / 0.612。
- wc.py：505M（pt2s505）和 pttpp 2% 294M / 190M 网格的逐档面积、bpb、按 F87 rule 1 参数化重抽的换档概率和 winner's curse。
- 复算斜率权重、F98 三格斜率 +1.4354、新点平时的点估计 b = +1.17、旧线对新点的预测 49.27 / 51.59、lr 插值 6.13e-3 / 1.07e-3。
- 读代码：生成器全文、select / readout 全文、trainer 的参数 / eval / batch / 续跑 / 结尾、run_guard、CHECK_PY、refs 脚本；
  diff 了 pt2same 190M 与 pt2sz 294M 的作业体；核对文件 sha 与 AUDIT 一致；看了 tests/ 下全部 .out。
- 看了环境版本（torch 2.10.0 / transformers 4.57.6，4 月以后未改）和旧 run 的 GPU 利用率（97–99%）。

## 一句话
这个实验最可能测错的地方：新点的增益是在「各 arm 自己选出的 lr」上测的，而选点（新规则、往下多伸一档的网格、在同一批
seed 0–3 上选）带来的 0.9–1.4 点格子级偏移，与两个新点能分辨「增长 / 不增长」的差（294M − 93M ≈ 2.3 点）同量级，单靠
选点就能把 SHOWN 的概率从 0.36 推到 0.56–0.69；再加上旧点已经锁定了大半判据，SHOWN / NOT SHOWN 更可能反映选点和噪声，
而不是尺寸效应。
