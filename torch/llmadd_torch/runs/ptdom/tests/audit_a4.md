# ptdom Amendment A4 对抗审查（全新审查者，没参与设计；只读；2026-11-04）

**范围**：
- AUDIT.md 的 Amendment A4（A4.1–A4.9），对照上一轮 tests/audit_a3.md 的阻断 1、重要 2–4、次要 5–14；
- make_jobs_ptdom.py（含 CHECK_PY）与 superseded_A3/scripts 的 diff；runs/ptdom/{readout_ptdom.py, select_ptdom.py, aux_gate_check.py, submit_ptdom.sh}；240 个 stage-1 作业文件与 4 个 pilot 作业文件；
- worktree llmadd-burst-ptdom 的 Rev G：`git diff 1208bad 11f61b9`（只改 tools/pretrain_range.py，+117/−4）；快照 runs/ptdom/code 的 pretrain_range.py 与 worktree 同为 d87a1502cee8；
- GPU 试跑：runs/ptdom/pilot/ 的 4 个 run JSON、check.json/check.txt、check_ok_<X>、logs/llmadd_ptdom-pilot-<X>_*.out；tests/a4/ 的全部日志。

**纪律**：只读。项目里只写了本文件；没有提交项目作业、没有取消作业，runs/ptdom 下没有新的 HOLD / picks 文件（已核对）。唯一提交的是 1 个审查用的 cpu_short 模拟作业 21851404，只打印、不写项目；它一直在排队，结果没用到（见重要 2）。
- 我自己的脚本：session scratchpad `/state/partition1/job-20790752/claude-4550886/-home-yh6210/74cbe2e4-1709-4a6a-b82e-817d33fbdd73/scratchpad/a4/`（cx_a4.py、cx_select.py、pilot_series.py、pilot_dist.py、ptheld_dp3.py、gT.py、ktok.py）。scratchpad 是节点本地临时目录，所以下文写全了方法和数字。
- 需要上计算节点的模拟脚本放在 `/scratch/yh6210/tmp_audit/ptdom_a4_audit/`（sim_fix.py、sim_local.py；临时审查目录，不是项目文件）。
- 这些脚本只 import 项目代码；假 run、picks、HOLD 全写在临时目录里（select_ptdom.py 调用时都显式给了 --root / --picks_dir / --hold / --subs_json）。

**总判断**：
- 上一轮的阻断 1（k-way 平局）**修好了**：并列感知逻辑正确，GPU 上 4 个 pilot 0 违例；全段打分实现正确，回归逐位一致。
- 重要 3（缺批次静默给判词）**修好了**。次要 5、7、8、11、12、13 **修好了**。
- 重要 2（lr 混淆）**只修了一半**：两端外扩、NOT FINAL、同 lr 条件 (iv) 都按决定实现了；但规则只在 none 一侧设防，留下一条路——先验的最终选点可以落在 none 根本没跑过的 lr 上，零效应也能拿到 PASS。我用真实代码构造出来了（重要 2）。
- 重要 4（模拟噪声）**修好了**：σ 3–10，加了只看 seeds 4-7 的条件。平坦零假设下 P(PASS | 晋级) ≈ 0.035–0.042 ≤ 0.05（我换种子重跑过）。共享 lr 趋势下仍然超标（0.05–0.15），根源同重要 2。
- **新的阻断 1**：2-way 与 k-span 的双路容差 0.25 nats，是用 400 步的 pilot（"早期区"）标定的。本项目自己的 ptheld 数据显示：同一对计算路径在训练后期，bf16 舍入差会超过 0.25。多数 stage-1 run 很可能因此判无效，整批选点卡死，出口和 A3 阻断 1 一样是死胡同（A3 次要 6 的老问题没有真正解决）。

---

## 问题清单（按严重程度排序）

### 阻断 1（新）：2-way / k-span 的 GPU 双路容差 0.25 nats 只在训练早期验证过，训练后期 bf16 舍入会频繁超过它。超过则 run 无效、选点拒绝、没有预注册出路
- **位置**：
  - make_jobs_ptdom.py:352-356：2-way，`max over pre evals × COMP × {held, seen}` 的 `lp_dp_maxdiff < 0.25`；
  - make_jobs_ptdom.py:357-365：k-span，`lpk_dp_maxdiff` / `lpk_own_maxdiff` 的最大值 < 0.25；
  - 两条路径都在 trainer：pretrain_range.py:1844、1868（`ans_logp_direct(x[:4])`：4 行的前向，再逐位置过 head）对比 `span_logp`（250 行前向，带 logits_to_keep）；
  - readout_ptdom.py:307-311：只容忍 CKA 一项失败，其余失败一律 invalid；
  - readout_ptdom.py:527-533、634：`select_unit` 要求每个 seed 都 valid，否则 "selection refused"。
- **为什么**：两条路径算的是同一个量，只是 kernel 形状不同，所以 logits 会舍入到相邻的 bf16 值。模型训好以后 |logit| 变大，一个 ulp 是 0.0625–0.25 nats；一段有 2–5 个 token，再在全部 eval × 全部集合上取最大值。
- **证据**：
  1. **本项目已经踩过同一个坑**。runs/ptheld/AUDIT.md:462-476 "Amendment DP"（2026-10-20）：同样的两条路径（250 行 GEMM 对比 8 行 + 逐位置 GEMV），差异按 bf16 ulp 量化（多为 0.0625 / 0.125）。CPU 标定的 0.05 在 GPU 上 64/64 个 run 失败，之后改成"step 0 ≤ 0.05；之后每个打分 token ≤ 0.5 nat"。
  2. **我在 184 个已完成的 53M ptheld GPU run 上实测**（75 次 eval，`logp_dp_maxdiff_*`，3 个集合；脚本 ptheld_dp3.py）：
     - step ≥ 1250：每个 (eval, 集合) 值的中位数 0.05–0.09，p99 0.24–0.35，**1.75% 的值 > 0.25**；
     - **71% 的 run 在某次 eval 超过 0.25**（这还只有 3 个集合）；
     - step ≤ 500：中位数 0.011，p90 0.055，最大 0.155。
  3. **ptdom 的 4 个 pilot 都在早期区**（≤ 400 步，val_loss 6.4）：2-way 中位数 0.007、p90 0.043、最大 0.080；k-span 中位数 0.007、p90 0.036、最大 0.084。这正是 ptheld 的早期区数值，**对 1250–18310 步没有任何信息**。
  4. **ptdom 每个 run 的取值点更多**：2-way 有 6–12 个集合（A 10、B1 12、B2 6、B3 6），k-span 有 6–11 个，乘 75 次 eval，每项检查 450–900 个值。按 4 行（ptheld 是 8 行）粗算，每个值超限的概率约减半，约 0.9%，则 P(run 超限) ≈ 1 − 0.991^450 ≈ 0.98。即使比 ptheld 低 10 倍，也约 0.5。
- **反例（cx_a4.py CX3，用真实 load / screen / readout）**：只让 1 个 stage-1 run（none@4e-3 s2）"只差 2-way 双路"失败：
  - load：`valid False`；
  - screen：`selection refused: unit B2 none|zodiac lacks valid runs ['B2 none@4e-3 s2']`；
  - readout：PROVISIONAL，`batches read []`。
- **后果**：每个批次都有比较类领域，所以 240 个 stage-1 run 全都暴露在这项检查下。大部分 run 很可能被判无效，144 B200-h 之后 `select --after 1` 对每批都拒绝；唯一出口是提交之后再改一个预注册的检查。
- **修法（便宜，提交前做，不动 trainer）**：
  - (a) 在 CHECK_PY 采用 Amendment DP 的规则：eval 0（近均匀模型，索引错误约 11 nats）|diff| ≤ 0.05；之后每次 eval ≤ 0.5 nat × 该集合的打分 token 数（或按 bf16 ulp 计）。
  - (b) 和/或预注册"只差 GPU 标定类容差（双路、分解、step-0 lm）的 run 算有效数据，照常记录"——这是 audit_a3 次要 6 的建议，A4 没采纳。
  - 然后重渲染 240 个作业，重跑 check_py_a4 / jobcheck_a4。
  - 想要真卡证据的话：提交前先跑 1 个全长的 B200 作业（例如 B2 none@4e-3 s0，约 0.5 h），读它的 check.json。
- **置信度**：机制，高（同函数、同结构，本仓库已诊断过一次）；"多数 run 失败"，中高（ptdom 的 span 与 logit 尺度和 ptheld 不完全相同，但 pilot 早期区的数值和 ptheld 早期区一致）。
- **其他 GPU 检查我也查了**：
  - 分解双路：72 个全长 ptheld2t GPU run 的窗口内最大值 7e-4 ≪ 0.05，安全；
  - none 臂 `grad_lm_share == 1.0`：trainer 直接赋常数（pretrain_range.py:2233、2452），不是计算出来的，安全；
  - `lpk_own_maxdiff`：pilot 全程恰好为 0（同形状同 kernel），安全；
  - k-way 已知答案：并列时 gap 必为 0，安全（见核对 1）。

### 重要 2：lr 混淆的残余——只在 none 一侧设防。先验的最终选点可以落在 none 没跑过的 lr，零效应拿 PASS，标签只有 EDGE
- **位置**：
  - readout_ptdom.py:545-571（`extension` / `next_ext` / `not_final` 只看 none 的最外 lr 5e-4 / 1.6e-2）；
  - readout_ptdom.py:725（prior 的 EDGE 只是标签）；
  - readout_ptdom.py:950-962（判词不看先验选点 lr 处的同 lr 增益）；
  - AUDIT.md A4.2 / A4.3。
- **构造的反例**：cx_a4.py CX1。真实的 screen → stage 2 → after 2 → readout；同一流程用 select_ptdom.py CLI 再跑一遍（cx_select.py），picks 文件一致。设定如下：
  - **没有任何先验效应**。两臂、各领域共用 lr 趋势 +2.5 分/每降一档（`60 + 2.5·(−log2(lr/2e-3))`），另加 ±0.3 的确定性抖动。
  - **只有一次噪声事件**：none@4e-3 在 seeds 0-3 上幸运 +4（61.5，高于 none@2e-3 的 60、低于 prior@1e-3 的 62.5）。于是 none 的 stage-1 argmax 落在网格内部（4e-3），**不触发任何下沿外扩**。
  - 先验的同 lr 格（2e-3，seeds 0-3）+0.4，满足条件 (iv)；这个 4/4 事件在零效应下概率是 1/16。
  - **结果**：
    - screen：ADVANCE（gain +2.4，4/4；same lr +0.4，4/4）；先验外扩 1e-3 → 5e-4；none 无外扩（argmax 4e-3；pick 2e-3 由 bpb 平局规则选出，落在它没外扩的网格下沿）；
    - after 2：only-zodiac@5e-4（EDGE），none@2e-3；
    - **读数（测试族 B2）：PASS**。gain +4.9（8/8），t p 1e-9，seeds 4-7 +5.0，单边 p 4e-5，tags 只有 `EDGE (prior ...)`；compass 同样 PASS。
    - none 从没跑过 2e-3 以下；判词行打印的同 lr 增益只有 @2e-3 +0.4（4 个 seed）。
- **为什么**：NOT FINAL 只问"none 的 pick / argmax 是否在 none 最外的 lr"，不问"先验选点的 lr 在不在 none 跑过的范围内"。一旦 none 的 argmax 靠噪声落在网格内部（σ 5、4-seed 均值时这很常见），先验可以比 none 低两档。
- **模拟**（sim_fix.py：a4_null_mc 的模型，真实函数，**另取 RNG 种子**）。PASS 规则是 A4 的（SF 0.10）。fixA = 先验最终选点 lr 落在该领域 none 跑过的 lr 范围之外时，判 NOT FINAL。fixB = 此时在先验选点 lr 补跑 none（seeds 0-7），再按 F87 规则 1 重选 none，然后用 A4 规则。

  | 情形（单领域名义水平） | n 晋级 | P(PASS \| 晋级)：A4 | fixA | fixB | 先验选点落在 none 范围外的比例（晋级中） |
  |---|---|---|---|---|---|
  | 零假设、平坦，σ 5 | 1510 | 0.042 (se 0.005) | 0.041 | 0.044 | 0.085 |
  | 零假设、平坦，σ 3 | 1759 | 0.034 (se 0.004) | 0.031 | 0.031 | 0.099 |
  | 共享趋势 +1/档，σ 3 | 2410 | **0.078** | 0.031 | 0.045 | 0.25 |
  | 共享趋势 +2/档，σ 5 | 2631 | **0.097** | 0.031 | 0.042 | 0.23 |
  | 共享趋势 +3/档，σ 5 | 3275 | **0.155** | 0.027 | 0.036 | 0.27 |
  | 功效 δ +6，σ 3：无条件 P(PASS) | 5024 | 0.331 | 0.215 | 0.314 | 0.35 |
  | 功效 δ +10，σ 5：无条件 P(PASS) | 5035 | 0.336 | 0.227 | 0.323 | 0.34 |

  - A4 一列与 A4.3 的表在误差内一致：趋势 +1 σ 3 为 0.088、+2 σ 5 为 0.105、+3 σ 5 为 0.147；功效为 0.332 / 0.337。
  - **fixB** 在共享趋势下把错判率压回 ≤ 0.05，功效只损失约 5%（相对）；**fixA** 也压得住，但功效损失约 35%（相对）。
  - 跑法：本机 nice 19 单进程，sim_local.py。我另外提交了一个只给审查用的 cpu_short 作业（21851404，sim_fix.py 全量，只打印、不写项目），一直在排队，结果没有用到；它是多余的，协调者可以 scancel（我按规则没有取消）。

- **建议（任何结果出来之前，交用户决定；不改 stage-1 作业文件）**：
  - 采用 fixB（成本：每批几个 none 格 × 8 seed，none 格在领域间共享；表中可见功效损失小），或者 fixA（更便宜，功效损失较大）；
  - 或者至少采用 A4.9 (1) 的 SF 0.05。注意 SF 0.05 只在趋势 ≤ +0.5/档时把错判率压到 0.05 以下，堵不上这条路径本身。
- **置信度**：机制，高（真实代码复现）；真实数据里有多常见，取决于 lr 趋势方向，未知。rb6 的 none 是高 lr 更好，那是保守方向；但 rb6 的先验臂在 dim 上是低 lr 更好（5e-4 82.2 vs 4e-3 63.9，a4_noise.out）。

### 次要 3：`--allow_partial` 时比较类子读数的问题（sub-read-out problems）不进 PROVISIONAL，也不打印，判词仍可以是最终的
- **位置**：readout_ptdom.py:1548-1550（sprobs 只在非 allow_partial 时拦截）；readout() 不接收 sprobs；893 行 `final_ok` 只看 integrity；1563 行 sprobs 只写进 JSON，report() 不打印。
- **影响**：table sha 不符、重算序列与日志不符时，只要加了 `--allow_partial`，就能给出最终 PASS；而且缺失的 subs 会让 shortcut 条件静默少算 seed。
- **修法**：把 sprobs 传进 readout()，非空时加进 provisional，并在报告头打印。
- **置信度**：高（读代码；cx_a4.py CX4 确认 readout 源码里没有 sprobs）。

### 次要 4：并列容差比必要的宽
- **位置**：pretrain_range.py:1817（`2^-7 × max|logit|`）；tests/a4/kway_tie_test.py 的 r4。
- **为什么**：
  - exact match 命中时，答案 token 是同一个 bf16 logits 张量上的词表 argmax，所以任何"k-way 选了别的"都必然是恰好相等（gap = 0）。
  - 现在的容差是 1–2 个 ulp（不是注释说的"一个 bf16 step"）。它会把"严格大 1 ulp"这种只有出错才会出现的情况（r4）也记成 tie。
- **影响**：索引错误的 gap 远大于 1 ulp，实际几乎不影响；容差可以改成 `gap == 0`。
- **置信度**：高。

### 次要 5：GPU pilot 没走到的路径
- **位置**：pilot 设置（make_jobs_ptdom.py:201-206）；pilot/check_ok_*；4 个 pilot 都是先验臂。
- **哪些没走到**：
  - stage-1 的"不存模型"分支：pilot 存了模型，check_py_a4 只在 pilot 上反向测了这条检查会失败；
  - none 臂的 "no prior ever" 检查；
  - 训练后期区（阻断 1）。
- **影响**：前两项逻辑上是安全的。trainer 只有带 `--save_model` 才存模型，ckpt 在结束时删除（pretrain_range.py:2503-2506）；`grad_lm_share` 是常数赋值。所以只有后期双路是真风险。

### 次要 6：功效表只算了平坦 lr 响应；在 rb6 的方向（none 在高 lr 更好）上，中等真效应在 stage 1 就过不了
- **反例（cx_a4.py CX5）**：真实效应 +3，共享趋势 −2.5 分/每升一档（高 lr 更好）。zodiac / compass 在 stage 1 都 NOT ADVANCE（gain −1.85）。原因：stage 1 里先验最高只到 2e-3，none 到 8e-3；先验的 4e-3 外扩只发生在晋级之后。
- **影响**：这是保守方向，不是错判。但应在报告里写明：在这个方向上，NOT ADVANCE 不是"无效应"的证据。
- **置信度**：高。

### 次要 7：顺序比较类的 FLOOR 仍然必然触发（audit_a3 次要 9 的前半没改）
- **位置**：readout_ptdom.py:972-978。answer-frequency 线 85–100，加 5 以后超过 100。只是描述性标签。

### 次要 8：NOT FINAL 的比例偏高，而且模拟里的 bpb 与 lr 无关
- **数字**：零假设下 P(NOT FINAL | 晋级) 0.24–0.28；有效应时 0.15–0.20（A4.3 表；我重跑为 0.26–0.28）。
- **模拟的缺口**：模拟里 in-training bpb 与 lr 独立抽取。真实 bpb 对 lr 有很强的趋势，2 分内的平局会被系统地推向 bpb 最好的 lr。如果那是 1.6e-2 或 5e-4，NOT FINAL 会更多。这是功效问题，不是错判率问题。

### 次要 9：`select_ptdom.py --after k` 越过已完成的链时，给出 AssertionError traceback，而不是一行干净的说明
- cx_select.py："picks_B2_after2.json asked for nothing"。只是外观问题。

---

## 逐题核对

### 1. k 路判别的平局容差；改为对整个条目片段打分是否正确
**正确**（次要 4 除外）。

**全段打分（读 diff 逐行核对）**：
- 候选片段：OFFSET 类是 `enc(" item\n")` 整体编码；months 是 `' month'` 加 `'\n'`。
- 建 eval 集时断言：答案片段 = eval 行的打分片段（`ids[a0:] == KSPAN[d][a_]`）。
- 断言：候选两两不互为前缀；每个片段的首 token 等于 k-way 候选。
- 候选行沿用 2-way 约定：同一个文本结尾和 numbers 行，文本开头前移到 512；答案自己的候选行 = eval 行，逐 token 断言。
- 判中规则：严格 >，并列算错，计入 `tiesk`。
- `S_.view(-1, nK_)` 的行列顺序与构造顺序一致；`span_logp` 窗口 `dom_max_k ≥` 片段长度。
- metrics 写成 `<d>_readout_kspan` / `_seen_kspan`；CHECK_PY 要求每次 eval 都有 `*_kspan`；loader 的 READ / EVAL_KEY 已切换。
- 单 token 条目数我从 pilot JSON 重算：months 12/12、zodiac 3/12、planets 7/8、elements 23/36、chinese_zodiac 11/12、geo_periods 1/12、greek 21/24，与 A4.1 一致。
- 打分片段带换行符（与决定原文的 `' <item>'` 不同）：这是 ft36 的形式，也是防前缀所必需的，我同意。

**并列感知的已知答案**：
- exact match 用同一个 logits 张量的词表 argmax，k-way 用候选内 argmax，两者看的是同一位置（`dom_max_ans − ln`，我手推过）。
- 所以 exact match 命中而 k-way 不同，只可能是恰好相等，gap = 0。violation 只会来自索引错误。

**GPU 证据**：
- 4 个 pilot 的 ties 为 25 / 0 / 14 / 35，violation 0。exact match 确有命中（geo_periods_seen 到 0.40，planets_seen 到 0.30），所以这项检查真的被检验过。
- `lpk_own_maxdiff` 恰好为 0；`lpk_dp` ≤ 0.084。

**回归**（reg_a4.out）：
- default / rb6all6 / rb6none 对比 origin/main：IDENTICAL；
- A（stars / months / none）与 B2（elements / none）对比 Rev F：除新增键外 IDENTICAL，36 个张量逐位相同。

### 2. lr 混淆：两端外扩、NOT FINAL、2e-3 同 lr 配对增益
**都按决定实现了，但有残余**（见重要 2）。
- 已实现：EXT_LR 两端外扩；EXT2 = 5e-4，只在 1e-3 仍是 argmax 时触发；OUTER；`not_final` = pick 或 argmax 落在已跑过的最外 lr。
- 生成器的 `cells()` 与读数的 `asks()` 一致：E 用 seeds 0-7，C 用 seeds 4-7；2b 只允许 none@5e-4；2c 不允许外扩。
- 我的 CX2：none 2e-3 → 1e-3 → 5e-4 → NOT FINAL，链在 after 3 完成。
- 条件 (iv)：selftest 里的 languages 被它拦下；我的 CX1 说明零效应下它只是一道 1/16 的门槛。

### 3. 校准模拟的噪声是否真实；第二阶段规则的假阳性率是否 ≤ 0.05
**噪声**：
- 用 rb6 53M 实测：每个 run 的 sd，numbers 读出 4.5，领域读出约 10。配对差 /√2 ≈ 格内 sd，所以独立噪声的假设有依据。
- σ 3 / 5 / 10 加 t3 重尾，覆盖了合理范围；模拟用的是真实的 screen / chain_step / rule。
- 未建模：bpb 的 lr 趋势（次要 8）、lr 间方差不齐。后者不影响 seeds 4-7 的"无选择"性质。

**错判率**：

| 情形 | P(PASS \| 晋级) | 结论 |
|---|---|---|
| 平坦零假设（我换种子重跑） | σ 5：0.042（se 0.005）；σ 3：0.034（se 0.004）。A4 写的是 0.035–0.036 | ≤ 0.05 成立 |
| 共享趋势 +1 / +2 / +3 分每档 | A4 表 0.06–0.15；我重跑 0.078 / 0.097 / 0.155 | 不成立（根源见重要 2；fixB 压回 0.036–0.045） |

Holm 会把实际族内的错判率进一步压低，所以按单领域名义水平算的数字是上界。

### 4. 读数是否会因为缺批次而静默给出判词
**不会**（次要 3 除外）。
- 我的 CLI 流程：只有 B2 在盘上，用设计的族（四批）时 PROVISIONAL，原因列出 A / B1 / B3 缺席，Holm 族没有缩小；用测试族（B2）时 final。
- 链不完整、picks 文件缺失或不符、批次 screen 被拒，都进入 provisional。integrity 问题：不加 allow_partial 时直接拒绝；加了也不给最终判词。

### 5. 每批独立的 HOLD
**通过**。
- 作业开头检查 `HOLD` 或 `HOLD_<批次>`。
- select 的默认路径是 `HOLD_<b>`。我的 GATE 反例：zodiac 所有先验格 LM share 0.3 → picks 写出，HOLD 写到指定的临时路径，exit 2。
- 生成器 `--submit` 两个都查；watcher 只点名失败格所在的批次；`submit --go` 两个都查，并要求 `check_ok_<X>`。

### 6. stage 1 不存模型
**通过**。
- 240 个文件的 ARGS 都没有 `--save_model`（`--save_model` 只出现在注释里），没有 offline LM 步骤。
- intent 里 `save_model: false`；CHECK_PY 断言没有 `model_*.pt`、没有 offline 文件。trainer 只在带 flag 时保存，ckpt 在结束时删除。
- 读数在 stage-1 格上用 in-training bpb（best_lr 只在两格都有 offline 时才用 offline）。

### 7. 四批都在 B200，每批一种卡
**通过**。
- 240/240：`--constraint=b200`、运行时守卫 `*B200*`、account 520、3 h、preemption=yes、requeue。
- singleton 槽：A 用 0–3、B1 用 4–7、B2 用 8–11、B3 用 12–15，每槽 15 个 run。HWOF 全是 b200，CHECK_PY 检查 GPU 名；stage 2 用同一个 HWOF。

### 8. GPU 试跑：每一项作业内检查是否都在真卡上通过
**通过的部分**：
- 4/4 的 check.txt 全 PASS（35 / 34 / 35 / 35，0 FAIL），GPU 都是 NVIDIA B200；torch 2.10.0，transformers 4.57.6。
- 输入检查块（exit 3 的那些）通过，否则不会进入训练。
- 244 个作业文件（240 + 4）的 CHECK_PY 字节一致（sha256[:12] 2b58f025882b）。

**但有两点**：
- pilot 只到 400 步，没有覆盖后期双路（阻断 1）；
- 没走到 none 臂和不存模型分支（次要 5）。

---

## A3 各项现状

| A3 项 | 现状 |
|---|---|
| 阻断 1 k-way 平局 | 已修（核对 1） |
| 重要 2 lr 不对称 | 部分修：残余路径见重要 2 |
| 重要 3 缺批次 | 已修（allow_partial 小洞见次要 3） |
| 重要 4 模拟噪声 | 已修（平坦零假设 ≤ 0.05；趋势下不成立，见重要 2） |
| 次要 5 首 token | 已修（全段为主读数，首 token 为副线，tied-embedding 说明已打印） |
| 次要 6 GPU 容差 | **未真正修**：pilot 在早期区，升级为阻断 1 |
| 次要 7 (iii) | 已修（合并计数，≥ 30，否则 NOT TESTABLE） |
| 次要 8 全局 HOLD | 已修 |
| 次要 9 FLOOR | 部分修（加了 shortcut / 消去 / 最短片段线；顺序类必然触发仍在，次要 7） |
| 次要 10 抢占窗口 | stage 1 已无关；stage 2 照旧（已接受） |
| 次要 11 DIFFER | 已修（进入 PROVISIONAL） |
| 次要 12 /scratch | stage 1 不存模型；现在 95.6%（剩约 220 GB），stage 2 约 30–65 GB，够用 |
| 次要 13 GEOK | 已入记录（check_py_a4） |
| 次要 14 参照 | 已接受（Rev G 下重验了 seed 0；每个作业仍有 DRY 哈希 exit 3） |

---

## 现在能不能提交第一阶段

**还不能。**

**必须先做**：修阻断 1。在 CHECK_PY 里把 2-way / k-span 双路改成 Amendment DP 式的容差（eval 0 ≤ 0.05，之后按打分 token 数给 bf16 容差），并/或预注册"只差 GPU 容差的 run 算数据"；然后重渲染 240 个作业，重跑 check_py_a4 / jobcheck_a4。这是一小时内的改动，不动 trainer、不动数据、不动 picks 逻辑。

**做完即可提交 stage 1**。另外，重要 2（fixA / fixB 或 SF 0.05）必须在读任何 stage-1 结果之前由用户决定并预注册；它不改 stage-1 作业文件，所以不阻塞 stage 1 的提交本身。
