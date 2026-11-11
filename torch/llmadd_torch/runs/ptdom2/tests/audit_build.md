# ptdom2 build + A1：全新对抗审查（2026-11-08，只读）

审查人：全新子 agent，未参与设计。范围：AUDIT.md（含 A1 与协调者的两条解释）、make_jobs_ptdom2.py、readout_ptdom2.py、
select_ptdom2.py、submit_ptdom2.sh、pilot/、复用的 make_jobs_ptdom.py / readout_ptdom.py / readout_rules.best_lr、训练器
pretrain_range.py（11f61b9，d87a1502cee8）的 --prior_shuf 路径，tests/*.out。
写入：只有本文件；所有检查脚本与中间文件在 /scratch/yh6210/tmp_audit_ptdom2/（cells.json = ptdom 384 个有效 run 的 area 抽取）。
没有提交、取消任何作业，没有写 HOLD / picks。审查时 GPU 冒烟作业 22355337 仍 PENDING（未分配节点），没有任何 pilot 结果可读。

## 结论先行

最可能测错的地方：**Part S 的 GEOMETRY-SPECIFIC**。真臂的 8 个 seed 全部被 F100 的选择挑过（seed 0–3 选 lr + 进 stage 2，seed 4–7
是 PASS 的判定 seed：「seed 4–7 全为正、Holm p < 0.1」），而打乱臂是新跑的；在「目标无关」的零假设下判词也偏向 GEOMETRY-SPECIFIC
（模拟：countries / cities 的假阳性率是无选择时的 2–5 倍）。A1.11 的「seed 4–7 未被选择」前提是错的。其次，GEOMETRY-SPECIFIC
分不开「真目标有用」和「打乱目标有害 / 实际剂量更重」，而读数没有给打乱臂对 none 的对比（Part S 连 none 都不打印）。

机制层面（打乱只换对应、剂量、数据、seed、位点、窗口、门槛的实现、作业体孪生、R1 / R1x / fixB / NOT FINAL、Holm / 两半 /
bootstrap / 单侧 t / 6/8）我按代码逐条核过，没有发现会让数字算错的 bug；发现的代码问题都在 --allow_partial 与描述性输出上（次要）。

## 问题清单（按严重程度）

### 阻断 1. Part S 的真臂在 8 个 seed 上全部是「被选中的」，打乱臂不是；赢家诅咒没有被处理，A1.11 的标注是错的
- 位置：AUDIT.md:281-283（Pending 11：「seeds 4-7 are unselected」）、:358-360（A1.11「not used by F100's lr selection」）；
  readout_ptdom2.py:644-652（verdict_S 用 seed 0–7 全部 8 个值做四个条件）、:646-647 与 :721（seed 4–7 行的标签）。
- 为什么会测错：S 的四个领域正是 F100 的四个 PASS。F100 的 PASS 条件（FINDINGS.md F100「两阶段规则」）包括 8-seed bootstrap、
  两半同号、t95、**seed 4–7 全为正且其单侧 t 的 Holm p < 0.1**；进 stage 2 还要求 seed 0–3 配对增益 4/4、同 lr 2e-3 增益 4/4；
  lr 2e-3 又是在 seed 0–3 上选的。所以 only-d@2e-3 在 seed 0–7 上的值都是「以真臂偏高为条件」的样本，而 only-d-shuf@2e-3 是新的。
  即便真目标与打乱目标完全等效（H0_geom），true − shuf 也系统性为正。两半条件挡不住，因为两半都被选过；A1.11 的 seed 4–7 行
  同样被选过，标签「未被选择使用」会误导读者把它当成无选择的复核。
- 模拟（/scratch/yh6210/tmp_audit_ptdom2/sim_s_selection.py，20,000 次；方差来自 ptdom 自己的格子：seed 共享效应 + run 噪声；
  近似 F100 的选择：seed 0–3 top2 2 分内取高 lr、pick 必须是 2e-3、stage-1 两个 4/4、PASS 的 8-seed 与 seed 4–7 条件；H0_geom 下
  打乱臂与真臂同分布）：

  | 领域 | sd_seed / sd_run | H0 下 true − shuf 的平均偏差 | P(GEOMETRY-SPECIFIC)（Holm 最严步 .025 … 最松步 .10） |
  |---|---|---|---|
  | countries | 1.29 / 2.51 | +1.03 | 0.121 … 0.256 |
  | cities | 0.68 / 2.14 | +0.80 | 0.109 … 0.213 |
  | months | 0.00 / 1.11 | +0.24 | 0.055 … 0.127 |
  | people | 0.62 / 0.63 | +0.08 | 0.041 … 0.097 |
  | 无选择的基线（sim_null_base.py） | | 0 | 0.024 … 0.066 |

  用观测均值当真值会低估选择偏差（观测均值本身被选高了），所以这是偏保守的估计。
- 能验证的检查：重跑 sim_s_selection.py；或直接比较：S 结果出来后，seed 0–7 的 true − shuf 与新 seed 上的 true − shuf。
- 建议（用户定）：S 的判定 seed 改成新的 8–15，两臂都跑（+32 个真臂 run；注意 B1 没有 seed 8–15 的 CPU 参考，
  ptdom2_refs 只有 A / B2 / B3，需要先跑 refs_ptdom2.py 的 B1）；或保留 0–7 但把 S 的词降为描述性的「选中 seed 上的上界」，
  不进 FINDINGS 当作几何特异的证据。无论哪种，A1.11 的标签都要改。
- 置信度：选择事实 = 读代码 / FINDINGS 确认；偏差大小 = 模拟（推测，近似 F100 规则）。

### 重要 2. GEOMETRY-SPECIFIC 分不开「真几何有用」与「打乱目标有害 / 剂量更重」；读数里没有打乱臂对 none 的对比
- 位置：readout_ptdom2.py:616-653（verdict_S 完全不读 none）、:684-691（Part R 只打印三个臂的均值，没有 shuf − none 的统计）、
  :578 / :629（打乱臂的 aux gate 只是描述）；AUDIT.md:97-98（「computed and reported for every FINAL domain, whatever its PASS
  verdict」）、:113-115（词义）。
- 为什么会测错：打乱目标不是「无信息」，而是把错误几何强加在条目状态上，与数据本身教的结构对抗——ptdom 的 none 臂自己就学到了
  CKA 0.13–0.69 的真结构（natcka.py：compass 0.53、months 0.64、countries 0.44）。打乱先验很可能在窗口内（75 次 eval 里占 26 次）
  压低读出；它的 1 − CKA 更难降，窗口内的先验梯度可能持续更大（λ 相同，但实际更新份额不同），而打乱臂的 gate 不过也不影响判词。
  这些都会让 true − shuf > 0，而真先验本身可能毫无帮助。Part R 里一个 NOT SHOWN（vs none）+ GEOMETRY-SPECIFIC 的领域，正是这种情形。
- 植入（planted.py，走真实的 load_all / verdict_R）：true = none、shuf = none − 3 → 六个领域全部「NOT SHOWN；GEOMETRY-SPECIFIC」。
  Monte Carlo（Holm × 6，sd_run 2.5、sd_seed 1）：打乱臂比 none 低 3 分、真臂 = none → P(GS) 0.589、P(PASS) 0.017；低 2 分 → 0.257。
  另：只学得快、终态无差（area +3、final3 0）→ PASS + GEOMETRY-SPECIFIC（按设计，area 是主指标；final3 只描述）。
- 建议（结果前登记，描述性即可）：
  - S：shuf@2e-3 − none@2e-3（ptdom 的 none@2e-3 seed 0–7 两个 batch 都有，不用新 run）；
  - R：P = N 时 shuf@P − none@N 同 lr；P ≠ N 时它带 lr 混杂，要么加 none@P（每 batch 8 个 run），要么标注；
  - 两个对比的面积增益拆成窗口内 / 窗口后；两臂的窗口 LM 份额中位数并排打印。
  - 读法：GEOMETRY-SPECIFIC 只有在 PASS（R）或 shuf ≥ none（S，在噪声内）时才读成「真几何有用」；shuf 明显 < none 时读成
    「错误目标有害」。
- 置信度：缺对比 = 读代码确认；后果 = 植入 + 模拟；「打乱先验在窗口内有害」本身是推测（等数据）。

### 重要 3. Part S 没有 FLOOR / CEILING 标签
- 位置：readout_ptdom2.py:616-653 与 :712-727（verdict_S / report 的 S 段没有调用 floor_tags）；AUDIT.md:106-112（「Printed beside
  every verdict ... FLOOR / CEILING」）。
- 为什么会测错：F100 里 months（面积约 5，机会 8.3，两臂终态都低于机会）和 people（FLOOR 线 100.4）都是 FLOOR。S 的
  GEOMETRY-SPECIFIC 打印时没有 FLOOR 标签，正好诱发 F100「不得写」第 2 条（把月份当成 FLOOR 以上可测的效应）。
- 检查：`grep -n floor runs/ptdom2/readout_ptdom2.py` 只在 verdict_R 里出现。
- 置信度：读代码确认。

### 次要 4. --allow_partial 下，integrity / manifest / R1pre / 子读出的问题不会让判词变成 PROVISIONAL，报告里也不显示
- 位置：readout_ptdom2.py:1103-1105、:1109-1111（只在没有 --allow_partial 时 exit）、:1124（report 不接收 probs，只进 JSON）。
- AUDIT.md:102-105 说拒绝的情形在 --allow_partial 下打印 PROVISIONAL。植入：一个打乱 run 的 data_sha 在一次 eval 不同 →
  RO.integrity 报「batch B2 seed 9: data_sha series differ」，但 --allow_partial 的判词是 PASS / GEOMETRY-SPECIFIC，报告文本不提。
- 正式读数不带 --allow_partial，所以不影响登记的路径；但「先看看」时容易读错。改法：probs 非空时所有判词加 PROVISIONAL 并打印 probs。
- 置信度：读代码 + 植入确认（planted.py 第 3 段）。

### 次要 5. --allow_partial 下缺一个打乱 run 时 report() 崩溃
- 位置：readout_ptdom2.py:573-574（floor_geo 在 gs 为 None 时是 None）、:698-700（直接取 x['floor_geo']['tags']）。
- 植入：删 ptdom2-B2-only-ranks-shuf-lr4e-3-s11 → TypeError: 'NoneType' object is not subscriptable（partial_crash.py）。
- 置信度：确认。

### 次要 6. 协调者要求的「PRIOR AT ITS LOWEST LR」描述行没有实现
- 位置：AUDIT.md:409-411；readout_ptdom2.py / select_ptdom2.py 里 grep 不到（只有 :435 的「below none's lowest lr」注记，
  而且那条只会出现在非 FINAL 的领域上）。
- 置信度：读代码确认。

### 次要 7. 打乱目标的强度随 seed、随领域变化很大，判词旁不打印
- 事实：shuf_cka_to_true（每个 run JSON 都记）seed 0–7：ranks 0.002–0.145、prefixes 0.001–0.080、months 0.026–0.132、compass
  0.037–0.099、people ≤ 0.027；scalar_adj（raw Gram、5 维、31 项）规划格子全在 0.097–0.136（tests/shufcka.out）。1 维目标的线性
  CKA = r²，所以 ranks / prefixes 的个别 seed 打乱目标仍保留 |r| 约 0.3–0.38 的真顺序。方向是保守的（偏向 NOT SHOWN），但给
  g′ 加了 seed 间方差。load2 已把它读进 rec（readout_ptdom2.py:159），report 不打印。建议每个领域打印 8 个值。
- 置信度：读日志确认（tdim2.py）。

### 次要 8. 「绑定词嵌入的注意事项对两臂相同，所以几何对比不受影响」说过头了
- 位置：AUDIT.md:115-116；readout_ptdom2.py:728-731。
- 理由：单 token 条目（compass 100%、scalar_adj 94%、months 12/12）的先验位点就是读出候选的输出行。真目标把这些行排成带答案的
  几何（Tier L），打乱恰好去掉了这条通路。所以这些领域的 GEOMETRY-SPECIFIC 区分不了「模型内部用了几何」与「答案几何被直接写进
  分类行」。建议措辞改成「两臂都移动输出行，但只有真臂按答案几何移动」。
- 置信度：推测（机制推理，未做 untie 对照）。

### 次要 9. 读数没有「打乱臂没有拟合到真目标」的双路检查
- pilot 的 P2b / P2c 只在两个 pilot run 上检查打乱臂对真目标的 eval CKA；正式 run 都记录了 cka_<d>，但读数没有断言。训练器路径已经
  由 CPU trace（T1–T5）核过，风险低。建议给每个 S / R2 打乱 run 加一条：窗口末 eval CKA（真目标）≤ 配对 none + 0.10，且 ≤ 真臂 − 0.2。
- 置信度：读代码确认（缺这条检查）。

### 次要 10. 发散的 run 是有效数据，打乱臂发散会直接变成 GEOMETRY-SPECIFIC；报告不显示 diverged_at
- 位置：readout_ptdom2.py:136-137（记录 diverged_at）、:195-200（照样 valid）、report 里不打印。λ = 1 下发散的可能性小。建议在判词旁打印。
- 置信度：读代码确认。

### 次要 11. pilot 的 body 在提交后被改写过（mtime 21:29:56，SUBMITTED 21:20:31）
- 和「a submitted pilot's files are never rewritten」不一致。我核过：现在的两个 body 和生成器当前的渲染逐字节相同（pilot_body_cmp.py，
  sha 469443e3b64e / 9255c61ea4ca = A1 表）；作业还在 PENDING，bash 运行时读的就是这两个文件，pass 规则写在前面。只影响溯源。
- 置信度：确认。

### 次要 12. 闸门的小漏洞
- select_ptdom2.py:122-125 打印的「next: make_jobs_ptdom2.py ... --submit」绕过 submit_ptdom2.sh 的 pilot / verify / 配额 / 自测闸门；
  应改成打印 submit_ptdom2.sh 的命令。
- pilot 只覆盖 B1 / B2 的打乱臂（countries、compass），A（cities、months、stars、landmarks）与 B3（scalar_adj）的打乱臂没有 GPU 冒烟，
  只有 CPU trace。
- submit_ptdom2.sh 的 pilot 闸门只认 pilot_check.out 里的「PILOT PASS」，不认生成器 / CHECK_PY_SHUF 的 sha；以后再改生成器，
  pilot 仍然算过。
- 置信度：读代码确认。

## 逐题回答

### 1. 打乱臂是否只换了「条目 → 坐标」的对应
- 训练器（读代码 + tests/shuf_trace.out）：DTAB_ACT[d] = F[qperm] 只在 d ∈ DSHUF 时生效（pretrain_range.py:1097）；训练项
  dgeo[d] = dom_prior(d, DTAB_ACT[d])（:2366）；加权梯度 act[d] = vec(LAMD[d]·g_)（:2224）；geo_<d> 的日志用 dgeo 或 DTAB_ACT（:2244）；
  dom_grads 的「true」项与 evaluate() 的 cka_<d> 用 DTAB（:2208）。far_perm 在两臂都会调用（RNG 消耗相同），data_sha 相同（T5）。
  --save_model 只在训练结束后保存（:2505），不影响训练。
- 剂量：λ（ARGS → lam_domains → domains.<d>.lam，CHECK_PY 两路）、窗口（prior_events / on_frac）、位点和 4-site / 3-item 门槛
  （T2：同步、同 item id）、lr、数据（data_sha 每次 eval、step-0 lm）、seed 都一致。我用 intent_diff.py 对比了 ptdom 孪生作业：
  intent 只差 args.tag、args.prior_shuf、shuf（seed 4–7 的孪生另差 save_model，A1.9 登记过）；ARGS 只多 --prior_shuf。
  **不一致的是实际剂量**：λ 相同，但打乱目标更难拟合，窗口内先验梯度与 LM 份额可能不同；打乱臂的 aux gate 只描述（问题 2）。
- CKA：82 个规划格子 shuf_cka_to_true 最大 0.136（< 0.15，训练器开头会 assert）；每个 seed、每个领域都成立；强度不均（问题 7）。
- 打乱臂里哪里还在用真目标、或反过来：训练项、加权梯度、grad_prior（因此 aux gate）、A6（A1.4 的相邻规则读 geo，即打乱目标）、
  CKA 上升门槛（打乱目标的 1 − geo）都用打乱目标；只有 gnorm_true_<d>、gcos_lm|<d>、eval 的 cka_<d> 用真目标，且都不进判词。
  真臂里没有任何地方用打乱目标。**未发现错用。**

### 2. Part S 的配对是否严格可比
- 同一快照（runs/ptdom2/code 与 runs/ptdom/code/tools 逐字节相同，生成器 assert）、同一数据（读数的 RO.integrity 按 batch × seed
  比较所有有效 run 的 data_sha 序列与 step-0 lm，ptdom2 的 run 也在内）、同一 seed、同一 CPU 参考。
- 环境：ptdom S 真臂的 check.json 记录 python 3.11.13、torch 2.10.0、transformers 4.57.6、numpy 2.0.1、tokenizers 0.22.2，与当前
  open_r1 环境相同；32 个 S 真臂 run 的驱动都是 610.43.02，没有一个被抢占重启（node.txt）。stage-1 / stage-2 作业体只差
  --save_model 与 offline LM，训练路径相同。**未发现系统差异**；但读数不比较两臂的版本 / 驱动，建议加一条断言（便宜）。pilot 的 P3
  （窗口前 |lm − 孪生|）会顺带显示 GPU 是否逐位确定。
- 赢家诅咒：**没有被正确处理**，见阻断 1。

### 3. Part R 的选点规则
- R1 选点、R1pre、R1x、fixB、NOT FINAL 与 AUDIT + A1 一致（逐条核了 decide_R:342-450）：A1.2 的 pick 或 argmax 触发、A1.1a 的
  none 上下各一步、R1pre 计作已用的那一步、A1.1b 的 fixB（含本轮 none 请求后再判范围）、(ii) R1x 后任何边都是终点、(iii) gate 不过的
  下一个 lr 把边关上、A1.3 的共享 none 网格、A1.8 的 GATE 不写 HOLD、A1.7 按两半看 pick 格子的 gate。
- seed 8–15：select_unit / gate_cells / none_asks_from_ptdom 都只读 SEL；R1x 的重推导只读 PRIOR_LRS0 ∪ {4e-3} 与 none 网格，
  不读 R1x 之外的格子。**确认从未进入选点。**
- 统计：rule() 的 bootstrap（readout_ptdom.boot_ci，10,000 次、default_rng(0)、百分位）、两半（seed 顺序的前 4 / 后 4）、单侧 t
  （sf_test）、Holm（每个对比单独、族 = FINAL 域、AUX GATE FAILED 的域留在族里）、赢 ≥ 6/8（严格 > 0）都正确；none 共享下 Holm
  仍有效。门槛不过的处理与 A1.7 / A1.8 一致。
- 唯一缺口：「PRIOR AT ITS LOWEST LR」没打印（次要 6）。

### 4. 判词会不会在不该过时通过（自己的模拟 / 植入）
- 真效应为 0（三臂同分布，sd_run 2.5、sd_seed 1、Holm × 6）：P(PASS) 0.017、P(GS) 0.016 —— 规则本身偏保守，不会乱过。
- 只学得快、终态无差：PASS + GEOMETRY-SPECIFIC 都会给（area 是主指标，按设计）；final3 与 hard final3 只描述。
- 打乱臂「失效」成有害：真臂 = none、打乱 = none − 3 → 六个领域都是 NOT SHOWN + GEOMETRY-SPECIFIC（问题 2）。
- 打乱臂「失效」成没打乱：CHECK_PY_SHUF（domains.shuf、shuf_cka_to_true < 0.15、prior_used_frac、gnorm_prior > 0）和 CPU trace
  会挡住；即使漏过，结果也偏向 NOT SHOWN（保守）。
- Part S 的选择偏差：见阻断 1 的表。
- --allow_partial 下的完整性问题：会给出正式的判词词（次要 4）。

### 5. 作业体孪生与闸门
- render_diff（386 个 ptdom 作业逐字节重渲染、全部 ptdom2 作业 = stage-1 孪生 + 登记的替换）我抽查了 S 的 s0 / s5 与 months s6：
  diff 只有作业名、日志名、注释、R / OUT / TAG、HOLD 行、ARGS 的 --prior_shuf、dry 规格里的 shuf true、CHECK_PY_SHUF 的编辑；
  intent 只差 tag / prior_shuf / shuf（s5、s6 另差 save_model，A1.9）。**一致。**
- CHECK_PY_SHUF：不放过——λ 比值检查对打乱臂取消，但 ARGS / lam_domains / 分解双路仍绑住 λ（--lam 2 的 CPU 测试 2.0000000）；
  不误杀——CKA 上升门槛属于 A5.1 的可标记有效；gs_ 只要求「记录到的地方」> 0；A6 换成相邻规则。只看到一个低概率风险：打乱臂在
  只有 3 个条目的 batch 上 bf16 超调、相邻 eval 又不 ≤ 0.005，会变成无效且不重跑（领域被拒，不是假阳性）。
- 闸门：HOLD / HOLD_<part>（作业开始时与 --submit 时）、配额 < 97%、快照与 sha、自测、pilot PASS、--check、一个名字只交一次、
  R1x / R2 的 picks 只读且 --verify 一致——都可靠。漏洞见次要 12。

### 6. 没写进意图、代码却依赖的假设；陷阱清单
- 隐含假设：(a) F100 的 seed 4–7 未被选择（错，阻断 1）；(b) 打乱目标本身无害、实际剂量与真目标相当（未写，问题 2）；
  (c) CKA < 0.15 的打乱等于「无几何」（1 维小领域仍留 |r| 约 0.3，问题 7）；(d) 两次跑同一配置可比（环境今天确认相同，读数没断言）；
  (e) --save_model 不影响训练（代码确认成立）。
- 训练 / 评测分布不一致：未发现（两臂同训练行、同 held-out 读出）。
- 记忆代替泛化：继承的问题——ranks 的 held-out 题被训练答案频率启发式全部答对（heuristic_ansfreq 1.0、n_hard 0），prefixes 0.79 /
  n_hard 10、people 0.92；这些领域的任何词都是在捷径可解的读出上；FLOOR 标签在 R 有、在 S 没有（问题 3）。
- 静默失效的损失 / 梯度：未发现（prior_used_frac ≥ 95%、gnorm_prior > 0、geo 记录率、分解双路）。
- 单 seed 偶然：未发现（8 seed + 两半 + 6/8）。
- 读错 token 位置：未发现（两臂位点相同，T2）。
- 同名不同义：geo_<d> 在打乱臂是对打乱目标的拟合、在真臂是对真目标的；CHECK、A6、pilot 都已按臂区分。check.json 的
  final_cka / cka_rise_true 在打乱臂指真目标，名字容易误读（不进判词）。
- 静默过滤样本：dom_prior 在 < 4 位点 / < 3 条目时返回 None——两臂同步跳过（T2），未发现偏差。
- 修 bug 的副作用：A1.4 的相邻规则只对打乱臂放宽有效性；它只决定 run 有效还是领域被拒，不会悄悄删样本，未发现偏差。

## 复现
所有脚本在 /scratch/yh6210/tmp_audit_ptdom2/（PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib，open_r1 python -B）：
dump_cells.py（→ cells.json）、show_s.py、sim_s_selection.py 20000、sim_null_base.py、planted.py、partial_crash.py、natcka.py、
tdim.py / tdim2.py、intent_diff.py、pilot_body_cmp.py。
