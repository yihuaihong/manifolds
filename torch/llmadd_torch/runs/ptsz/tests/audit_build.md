# ptsz 构建审查（对抗式，全新子 agent，只读）— 2026-11-05

审查对象：runs/ptsz/AUDIT.md（预注册）、make_jobs_ptsz.py、select_ptsz.py、readout_ptsz.py、submit_ptsz.sh、smoke/、
tests/、scale_work/ptsz/、ptsz_refs.json、128 个 stage-A 作业，以及它们依赖的冻结 trainer（runs/ptsz/code7/tools/
pretrain_range.py a3bce3ab63c1、run_guard.py、pretrain_helix.py）。对照：runs/ptlr（F87）、runs/pt300x（F99）、
runs/ptlr_1b、FINDINGS F87 / F97 / F99。

我自己做的检查都在 /scratch/yh6210/tmp_audit_ptsz/ 下，没有动项目文件（本文件除外）：
- extract.py → areas.json：用自己写的 area 实现，从复用和相关的 3e8 7% run JSON（ptlr、pt300x、ptlr_1b）里重算每个
  lr、每个 seed 的 area / bpb / val_loss。
- sim_audit.py → sim_audit.out（CPU 作业 22018309，cpu_run.sh）：自己实现一遍判定规则（OLS、seed bootstrap 用规则里的
  固定 default_rng(0) 抽样、两半、df 6 残差 t），没有 import readout_ptsz / readout_rules。KA：旧的四点 L40S 斜率
  +2.1608 = readout_ka。在此基础上跑 (M1) build 的零假设模型，和 (M2) 显式的选 lr 流程（每个 arm 4 档网格，真值是
  二次曲线，seed 噪声从各 arm 自己的池子里重采样，F87 rule 1、X、STOP、seed 4–7）。
- seedcorr.py、robust.py：看跨尺寸的 seed 相关性，以及单个 seed 对复用格子斜率的影响。
- 这个目录里的 indep_sim.py、cells_empirical.*、run_*.log 不是我写的（时间戳 19:05–19:23，是别的会话留下的），我
  没用它们；它们的 model A 数字和 AUDIT 也一致。

**总评：没发现阻断。** 代码里没找到会静默测错的 bug：生成器、CHECK7、integrity re-check、选 lr 和读出，都和
AUDIT.md 的文字逐条对得上。风险集中在三处：推断层面（次检验、主检验对已看过点的依赖、均值增益对单个 seed 敏感）、
731M 显存，以及几处没预注册的人工判断。

---

## 重要

### I1. 均值增益被个别「晚学会」的 none seed 主导；杠杆最大的 731M 正是最容易出这种 seed 的地方
- 位置：readout_ptsz.py:217-241（slope_block 用 8 seed 均值），描述性部分 readout_ptsz.py:441-498（有留一尺寸，没有
  留一 seed，也没有中位数）；AUDIT.md:157-164。
- 为什么会测错：area 实际上是「多早学会」的指标。以 505M none@9.6e-3 为例，seed 0 在 6k–10k 步完成跃迁（area 62.1），
  seed 5 到 ~14k 步才跃迁（area 34.2，最后一次 eval 0.79）。只这一个 seed 就把 505M 的增益从中位数 28.7 拉到均值
  32.2。复用的四个 L40S 点（robust.py）：
  - 均值斜率 +2.16，中位数斜率 +1.01，20% 截尾均值斜率 +1.17；
  - 只去掉 505M seed 5，斜率就变成 +1.21；
  - F99 六点：均值 +2.79，中位数 +1.82。
  731M 的 ln N 最大（20.24），tpp 最低（0.49），none 的跃迁最可能晚。一个晚学会的 seed 会让该格增益多出约 +3 点，
  对斜率的影响也最大。所以「SHOWN (increase)」可能主要反映的是「大模型的 none 更常在 3e8 token 内学得晚」，不是
  整体分布的移动。预注册的统计量是均值，按规则不算错，但 AUDIT 没写这层含义，读出里也没有对应的稳健性数字。
- 验证：在读任何 ptsz 结果之前写 amendment，在读出里加描述性行：
  - 每格中位数 / 截尾均值的斜率（主检验和次检验各一个）；
  - 每格 none area < 45 的 seed 数；
  - 留一 seed 的斜率范围。
  然后在 4 个复用格子上复现上面的数字（+2.16 / +1.01 / +1.17 / +1.21）。
- 置信度：数据确认（复用格子）；对新格子是推测。

### I2. 次检验（唯一针对 F99 I1 的检验）没有尺寸层面的误差控制，叫「confirmatory」名不副实
- 位置：readout_ptsz.py:428-430（`verdict(m3, g1, g2, use_c=False)`）；AUDIT.md:166-168、211-216。
- 为什么会测错：(a) 只含 seed 误差，(b) 三个点同号很容易满足，df 1 的残差区间只打印不判定。按我自己的实现独立重算，
  新点不增长时判成 SHOWN (increase) 的概率：

  | tau（尺寸离散） | 0 | 1.85 | 3.12 |
  |---|---|---|---|
  | M1（build 的模型） | 0.074 | 0.137 | 0.198 |
  | M2（显式选 lr 流程） | 0.081 | 0.120 | 0.169 |
  | AUDIT 原表 | 0.074 | 0.115 | 0.186 |

  即使没有尺寸离散（tau 0），也是名义水平的约 1.5–2 倍：8 个 seed 的百分位 bootstrap 低估方差（因子 7/8），none
  的 seed 分布又是重尾。检验力只有约 0.5–0.6（AUDIT 原表）。
- 验证：sim_audit.out 的 SECONDARY 列。要改的话，得在看结果前写 amendment，例如 se² = bootstrap 方差 + tau6²/Sxx
  （tau6 = 1.85 事先固定，来自 F99 的残差）；或者把次检验改叫描述性。不改的话，任何「次检验 SHOWN」旁边都必须写上
  0.12–0.20 的误判率。
- 置信度：读代码确认 + 独立模拟。

### I3. 主检验能不能判成 SHOWN，主要取决于「不增长时停在哪个水平」
- 位置：AUDIT.md:198-200、207-210；readout_ptsz.py:401-419。
- 为什么会测错：四个已看过的 L40S 点自身斜率 +2.16；新点的平均 ln N（19.25）比旧点（18.10）大。只要新点平停在旧点
  均值以上，就会被读成增长。我的扫描（tau 1.85，新点平在 mu0）：

  | mu0 | 24 | 26 | 26.85（旧四点均值） | 28 | 28.77（F99 六点均值） | 30 | 32 |
  |---|---|---|---|---|---|---|---|
  | 主检验误判 SHOWN | 0.009 | 0.080 | 0.147 | 0.263 | 0.354 | 0.466 | 0.582 |

  AUDIT 自己给的期望区间（gain ~20–40，中心约 26 / 28.5 / 31.5 / 34.5）就落在误判率高的那一段。
  - M1 复现：AUDIT 的 0.117 / 0.127 / 0.143（mu0 26.85）对应我的 0.117 / 0.147 / 0.147；0.382 / 0.322 / 0.273
    （mu0 28.77）对应我的 0.367 / 0.354 / 0.305；无条件 0.032–0.049 对应我的 0.035–0.047；都在 2000 次模拟的
    MC 误差内。
  - M2（选 lr 流程）：主检验条件误判 0.132–0.144，无条件 0.034–0.049。
- 验证：sim_audit.out 末尾的 mu0 扫描。AUDIT「What the verdicts may be called」要写明：主检验 SHOWN、次检验 NOT
  SHOWN 时，只能说「新尺寸没落到旧趋势以下」。
- 置信度：模拟。

### I4. 731M 在 L40S 上的显存没验证，冒烟的通过标准也没有余量
- 位置：tests/smoke_report.py:120（`"731m reserved < total"`）；pretrain_range.py:546-554（grad_info 在 zero_grad
  之前）、:492-498（续跑时 `st` 一直被引用）、:466（lm_chunk 自动判成 0）。
- 为什么会测错 / 失效：
  - 判据形同虚设：分配器不可能 reserve 超过 total；真 OOM 时「A ok」已经是 False，这条判据从不起作用。所以只要 210 / 310
    步没 OOM 就算 PASS，哪怕余量只剩 0.1 GiB。
  - 我的估算：峰值 allocated 约 37 GB。拆开是：静态 11.7（参数 + 上一步的梯度 + Adam），block 检查点 2.2，
    logits 链 9.9 保留 + 13.1 瞬时，后面的 grad_info 阶段再加 g1/g2/v1/v2 约 11.7。续跑时再加约 2.9 GB（`st["model"]`）
    到约 40 GB。L40S 可用约 47.5 GB。AUDIT 用 reserved 线性外推得到 48.2 GB，偏悲观。所以大概率放得下，但余量只有
    7–10 GB，而且没有任何长程（18k 步）证据。
- 验证：冒烟跑完后（现在的作业是 22011938，见 M2），把判据收紧到 A、B 的 max_reserved ≤ total − 2 GiB，并且 nvidia-smi
  最大 memory.used ≤ 44 GiB；把余量写进 AUDIT。不要用 `--no-smoke`。
- 置信度：机制是读代码确认的；数字是推测。

### I5. OOM 的位置和后果都只落在 win-ctx 一臂，而它该怎么归类没有预注册
- 位置：pretrain_range.py:546（`and on and usable`，none 臂永远不走 grad_info）；AUDIT.md:143-144；select_ptsz.py:122-133。
- 为什么会测错：731M 的第一次 grad_info 在第 1000 步（窗口 916..7323，每 250 步 eval，约 28 分钟），在第一个 ckpt
  （1999 步）之前。所以 OOM 后无法续跑；`--requeue` 只管节点故障，作业直接 FAILED。
  - 选 lr 那边，这些 run 的 JSON 没跑完，又没有 check 文件，会永远显示「running / wait」，没有任何报警。
  - none 臂不受影响，照常跑完。
  - 之后要人工判：
    - 列进 OWN_FAILURES：缺失的 eval 记 0，win-ctx 面积崩掉，731M 增益变负，判词可能翻成 NOT SHOWN 甚至 SHOWN
      (decrease)；
    - 当基础设施问题重投：会再在第 1000 步 OOM；
    - 改用 A3 的 `--lm_chunk_tokens 8192`：CHECK7 的 intent 是 lm_chunk 0 / lm_chunk_tokens -1，不改 intent 就过不了，
      而且只改一臂会让两臂的 loss 路径不同。
- 验证：现在就写进 AUDIT：「确定性的 OOM 不算 own failure；该格 NOT DETERMINABLE；回退方案由用户定，并且两臂同时用」。
- 置信度：读代码确认。

---

## 次要

- **M1 新旧格子的 lr 网格不同，但造成的格子级偏移很小。**
  - 网格差异：
    - 505M win-ctx 只跑过 1.2e-3 一个 lr（ptlr/out 只有这一档），190M win-ctx 在 ptlr 里只有 2 档；
    - 53M 用的是 1e-3·2^k 阶梯；
    - 505M none 网格缺 2.4e-3；
    - 新格子是连续 4 档 + X。
  - 模拟结果：选 lr 对 8 seed 增益的平均偏移是 −0.35（网格居中），none 最优在下沿时 −0.23、在上沿时 +0.07。先验臂
    1 档对 4 档的赢家诅咒约 +0.1（先验 4 seed 均值 SD ≈ 0.28，κ_prior ≈ 0.55 点/档²）。合计 ≤ 0.5 点，远小于 tau 和
    斜率 × Δx（≈ 2.5 点）。加入选 lr 后误判率不变（见 I2 / I3）。
  - none 最优偏一档时 STOP 概率约 6%，居中时 0.1%。
  - seed 0–3 的增益平均比 seed 4–7 低 2.8：这是 none 选点的赢家诅咒，属于各格共同的平移，对两半各自的斜率 (b) 没有
    影响。
  - 置信度：模拟。数据来源：none 的 κ 中位数 2.67 点/档²，seed SD 7.85。
- **M2 冒烟作业号和 nice 已过时。** AUDIT.md:253、267、334、355 写的都是 21999324，但 sacct 显示它已经 CANCELLED；现在在
  排队的是 22011938（PENDING，命令行用 `--nice=100` 重投，而 job_smoke731.sbatch:7 写的是 1000；见
  runs/ptsz/SUBMITTED_smoke731）。按 AUDIT 跑 `smoke_report.py 21999324` 会因为没有日志直接报错（报错是显式的，不会
  静默）。「冒烟比同 nice 的批次作业早，所以先开始」这一推理也不再成立。要更新 AUDIT，用 22011938 跑 smoke_report。
  置信度：确认（sacct）。
- **M3 「double path」其实不独立。** CHECK7 的 area 和 `R.area` 是同一个公式、同一个 JSON（select_ptsz.py:141-144），只能
  发现 check 之后文件被改。对新格子真正独立的重算要放到结果层审查里做（F99 就是这么做的）。置信度：读代码确认。
- **M4 形状与 ln N 混在一起。**
  - 先验层 / 读出层的相对深度不一样：136M 是 3/11 = 0.27 和 7/11 = 0.64，其他格子是 0.30–0.33 和 0.60–0.67。
  - 深度序列是 6、10、11、12、19、20、24、29，391M（20 层）几乎和 294M（19 层）一样深。
  - 先验臂接近天花板，影响大概不大，但残差离散里有一部分是形状造成的。置信度：读代码确认，影响是推测。
- **M5 391M 的 step-0 lm 参考值只有 CPU 值，没有 GPU 校准。** 294M 由 pttpp 校准，731M / 136M 由冒烟校准。要是 391M 上
  GPU 和 CPU 差 > 1e-3，所有 391M run 都会被 WITHHELD（报错是显式的）。风险低。
- **M6 tie-break 用的 in-training bpb 只来自 8×16 条 val 序列**（约 65k token；pretrain_range.py:404-413）。旧网格里它随
  lr 单调上升，所以「前两名差 ≤ 2 → bpb 低者」实际上就是「取较低的 lr」。所有格子同一规则，不影响可比性。
- **M7 磁盘。**
  - 97% 线只在提交时查一次（submit_ptsz.sh:42-44），现在 /scratch 94%。
  - 失败的 run 不会删 ckpt（只有成功时才 unlink，pretrain_range.py:593-594），中途失败还可能留下 `.tmp`，731M 每个
    8.8 GB。
  - 写满（ENOSPC）时 `path.write_text` 会把 run JSON 截断（:579）：load_json 读到 None，显示成「missing」。
  - 731M 那一波运行期间要盯 myquota，并清掉失败 run 的 ckpt。
- **M8 run_guard 在 Slurm 很慢时可能让重投的作业「空跑」。** squeue 超时会返回 True（run_guard.py:27-32）：带旧 claim
  文件、新作业号的重投作业会打印「being computed by running job X」直接退出，CHECK7 在旧的半截 JSON 上失败，select 那边
  永远显示 running。只影响「基础设施损失后重投」的情形。重投前先看 / 删 `.claim`，并在新日志里 grep「run guard:」。
- **M9 /scratch 的 60 天清理不认读取。** 读文件不刷新 atime，复用的 F87 JSON（写于 10-16 到 10-22）从约 12-15 起可能被
  清理。读出必须在此之前做，或者先归档。
- **M10 续跑时 `sec` 会重置**（pretrain_range.py:488）：续跑过的 run 的耗时被低估，只影响成本估计。
- **M11 INVESTIGATE 的 ±15 带很宽**（先验 73–106；731M none 35–75），只能抓到严重失败。

---

## 逐题回答

### 1. 训练 / 评测条件哪里不一致
- 继承自 F87、8 个格子和两臂都完全相同的部分：
  - eval 是 val 文本 + 一行，只用 format 0（"Q: a + b = "），这一行放在 512 窗口的末尾；
  - 训练是 train 文本 + 4 行、4 种 format 随机；
  - 评分是 teacher-forced 的「答案 + 换行」完全匹配（pretrain_range.py:383-400 已核对位置正确）；
  - bf16 autocast 训练和评测一致，没有分片，eval 前切到 model.eval()。
  这些不会在新旧格子之间造成差别。**未发现**新的不一致。
- 新 4 格 vs 复用 4 格（L40S）的逐项对比：

  | 项目 | 新格子 | 复用格子 | 结论 |
  |---|---|---|---|
  | trainer | sha a3bce3ab63c1 | 同（integrity 会查） | 一致 |
  | 作业体 | 有 GPU guard、data sha、CHECK7 | F87 作业没有这三样 | 我核对了 56 个复用 F87 日志：全部「NVIDIA L40S」、0 次续跑、0 次 OOM、都有「done」 |
  | 数据流 | data_rng 1000+s、rng 2000+s、split 由 seed 决定 | 同 | 一致；两臂逐位相同（ctx 先验不消耗随机数） |
  | 评测 / area 定义 | readout_rules.area | 同 | 一致；KA 复现 F99 到 4 位小数，我自己的 area 实现也复现了 |
  | ckpt / 续跑 | 294M+ 用 2000、可能因节点故障续跑；136M 用 0 | F87 用 2000、从未续跑；93M 用 0 | 续跑只带来 GPU 非确定性，不带来偏差 |
  | 梯度检查点 | 按 >150M 规则开 | 同一规则 | 数学等价；hook 在重算时会再次触发，对 ctx 位点无害，值相同 |
  | 先验层位置 | n//3、2n//3 | 同一规则 | 相对深度有差（M4） |
  | 软件环境 | torch 2.10.0 / transformers 4.57.6 | 同 | site-packages 日期 2026-04，和 F87 时一样；CPU step-0 逐位相等 |
  | lr 网格 | 连续 4 档 + X | 不同（M1） | 偏移很小 |

### 2. 数字合理但测错的地方
- 指标实现和名字对得上：area、gain、斜率、bootstrap（每格 10000×8，default_rng(0)，按 N 升序）、两半、(c)
  （t(0.95, df 6)，se = sqrt(SSR/6/Sxx)）、次检验 (a)(b)。这些都是我逐行核对过，并用独立实现复现的
  （readout_ptsz.py:185-264、401-437）。
- 没有静默丢样本：own failure 计分，缺 run 判 NOT DETERMINABLE。平均方式是格子均值等权 OLS，和 AUDIT 一致。
- 跨尺寸的 seed 相关性约等于 0（6 个复用格子的非对角均值 −0.01），所以每个尺寸独立重采样没问题。
- 真正的问题在 I1（均值对单个 seed 敏感）和 M1（lr 网格可比性小偏差）。M3 的「double path」名不副实。

### 3. 记忆 / 泄漏 / 偶然
- 泄漏：held-out 对的两种顺序都不进训练（pretrain_range.py:214-223），b11 对会在自然文本里出现（B5 只做描述性），
  **未发现**新问题。
- 偶然：见 I1，单个 seed 就能把四点斜率减半。
- 误判率：见 I2 / I3。选 lr 造成的格子级偏移见 M1。

### 4. 静默失效的 loss / 梯度
- 先验是否真在起作用由 CHECK7 断言（geo > 0、0 < grad_prior、events、on_frac）；裁剪对两项 loss 同比例缩放；
  CKA 的 Gram 在 autocast 下是 bf16 matmul（pretrain_helix.py:51-55），但所有格子一样。除显存外**未发现**静默失效。
- 显存：见 I4。OOM 的位置和后果见 I5。冒烟能挡住「210 步内就 OOM」，挡不住「余量太小」。

### 5. 代码暗中依赖、意图说明没写的假设
- 参考值：391M 没有 GPU 校准（M5）；期望区间来自已看过的邻居。
- 续跑：续跑时 `st` 多占 2.9 GB，`sec` 重置（M10），run_guard 依赖 squeue（M8）。
- 磁盘：失败 run 残留 ckpt，写满时 JSON 被截断（M7）；/scratch 清理（M9）。
- nice 与排队：冒烟作业号和 nice 已变（M2）；按 testonly，nice 1000 的作业要 11-11 才开始，加上 49–58 h 运行和手工选点，
  用户预计的 11-10 / 11-11 出结果做不到。
- 手工步骤：OOM / 崩溃怎么归类（I5）；基础设施损失后怎么重投没写（SUBMITTED 标记会挡住 `--submit`，只能手工 sbatch，
  并注意 M8）。

### 6. 陷阱清单
| # | 陷阱 | 结论 |
|---|---|---|
| 1 | 训练 / 评测分布不一致 | 继承 F87，各格一致，未发现新问题 |
| 2 | 记忆代替泛化 | held-out 两种顺序都不训练，未发现 |
| 3 | 静默失效的 loss / 梯度 | 只有 731M 显存这个风险（I4、I5），其余由 CHECK7 断言 |
| 4 | 单 seed 偶然 | **有**：I1（505M seed 5 把四点斜率从 +1.21 拉到 +2.16） |
| 5 | 读错 token 位置 | 第 0 步断言了操作数和答案能解码（pretrain_range.py:507-511），dry digest 和 pttpp 日志逐行相同，未发现 |
| 6 | 同名不同义 | 294M L40S 与 B200 用不同 cid；「double path」不独立（M3） |
| 7 | 静默过滤样本 | 未发现 |
| 8 | 修 bug 的副作用 | regress_frozen 通过，冻结 sha 断言，未发现；冒烟作业号过时（M2） |

---

这个实验最可能测错的地方：731M 是杠杆最大的点，那里只要一两个 none seed 学得晚，均值增益就能多出 3 点以上；而唯一针对新尺寸的次检验只计 seed 误差（新点不增长时约 12–20% 会误判 SHOWN），所以「次检验 SHOWN」很可能测到的是 731M 个别 none seed 的偶然，而不是增益随尺寸增长。
