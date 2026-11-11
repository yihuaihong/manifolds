# ptdom2 Amendment A2：全新对抗审查（2026-11-08，只读）

审查人：全新子 agent，没有参与 A2 的修订。范围：AUDIT.md 的 Amendment A2 及其后的 coordinator 段（对照 A0 / A1 与上一轮审查
tests/audit_build.md）、make_jobs_ptdom2.py（34e7fd02cbe6）、readout_ptdom2.py（4b29de5bcac3）、select_ptdom2.py（1c2dfc6adda0）、
submit_ptdom2.sh（adcb18e10422）、ptdom2_refs_*.json、tests/*.py 与 *.out，以及复用的 make_jobs_ptdom.py / readout_ptdom.py。
开审时核对过：四个文件的 sha 与 AUDIT A2 表一致；ptdom2_refs_B1.json = 08963dd58475；runs/ptdom2/out 为空；除 pilot 外没有
SUBMITTED_ 标记；GPU 冒烟 22355337 仍 PENDING。
写入：只有本文件。全部检查脚本、合成数据与日志在 /scratch/yh6210/tmp_audit_ptdom2_a2/（见文末「复现」）。没有提交、取消任何作业，
没有写 HOLD / picks。

## 结论先行

最可能测错的地方：**Part S 的 PASS 词**。A2.1 把它登记成「F100 的 PASS 在新 seed 上的复现」，但对城市、月份，F100 的 PASS 对照是
**none@4e-3**（none 在这两个领域选中的 lr），A2 用的是 none@2e-3；在 ptdom 的 seed 0–7 上 none@2e-3 比 none@4e-3 低 1.32（城市，6/8）
和 0.84（月份，8/8），月份的 final3 甚至因此翻号。所以城市、月份的 S PASS 是同 lr 对比，偏向 PASS（模拟：对最佳 none 没有真实增益
时，P(PASS) 城市 0.20、月份 0.26，名义约 0.025），而且登记的读法「PASS 且 GEOMETRY-SPECIFIC = 增益依赖正确几何」直接用它。
GEOMETRY-SPECIFIC（真目标 vs 打乱目标，同 lr）不受这一条影响。

其次，登记的读法（A2.2）只看 PASS 与 GEOMETRY-SPECIFIC 两个词，不看 FLOOR、final3 / 窗口拆分、打乱臂相对 none 的伤害和打乱臂的
LM 份额。而 F100 已经给 4 个 S 领域里的 3 个记录了会让这句话说过头的情形：人物 FLOOR 线 > 100，月份 FLOOR 且增益只在窗口内，城市
主要是学得更快。

上一轮的阻断 1（赢家诅咒）已经真正消除：S 的判词只读 seed 8–15，三臂都是新跑的 ptdom2 run，没有任何选点代码路径读过这些 seed。
重要 2 / 3 的描述行（shuf − none、窗口拆分、LM 份额、FLOOR / CEILING）也都按定义实现了，数值正确。判词规则、Holm、A1.7 门槛、拒绝
与 PROVISIONAL 的实现，经我自己的植入与模拟核过，没有发现会让数字算错的 bug。作业体与 ptdom 孪生一致，名字不冲突。

## 问题清单（按严重程度）

### 阻断 1. Part S 的 PASS 不是 F100 的对照：城市、月份用 none@2e-3，而 F100 的 none 是 4e-3。偏向 PASS，「复现」的标签不成立
- 位置：AUDIT.md:430（「PASS on only-d@2e-3 - none@2e-3 (= F100's PASS replicated on fresh seeds)」）；readout_ptdom2.py:692–699
  （verdict_S 用 `contrast_domain(R_, subs, b, d, lr, lr, ...)`，P = N = S_LR = 2e-3；docstring：「PASS (only-d - none: F100 on fresh
  seeds)」）；make_jobs_ptdom2.py:357–359（S 只跑 none@2e-3）。
- 事实（runs/ptdom/readout.txt:139、:148，FINDINGS F100 表）：F100 的四个 PASS 是 countries / people vs none@2e-3，
  **cities / months vs none@4e-3**（batch A 这两个领域 none 的 pick 都是 4e-3）。
- 为什么会测错：对城市、月份，S 的「PASS」是同 lr（2e-3）增益，也就是 F100 的筛选条件 (iv)，不是 F100 的判词对照。none@2e-3 是较弱的
  对照（tmp_audit_ptdom2_a2/none_lr.py，ptdom seed 0–7）：

  | 领域 | F100 none pick | none@2e-3 − none@pick | 真臂 − none@2e-3 | 真臂 − none@pick | final3：真 / none@2e-3 / none@pick |
  |---|---|---|---|---|---|
  | 城市 | 4e-3 | −1.32（6/8 更低） | +4.80 | +3.48 | 93.9 / 91.3 / 92.6 |
  | 月份 | 4e-3 | −0.84（8/8 更低） | +4.74 | +3.90 | 0.85 / 0.02 / 1.53（对 none@pick 是 −0.68） |
  | 国家 / 人物 | 2e-3 | 0 | = | = | 一致 |

  模拟（sim_rules.py，走真实的 apply_rules，Holm × 4，sd_run 取 ptdom 的格子）：真臂 = F100 的最佳 none（对它没有增益）、只差
  none@2e-3 的那段缺口时，P(PASS) 城市 0.201、月份 0.256；真正的零假设下约 0.023–0.029。也就是说，城市、月份的 S PASS 有约 8–10 倍
  的「lr 带来的 PASS」。登记的读法（A2.2）在 PASS 且 GEOMETRY-SPECIFIC 时写「the gain depends on the correct geometry」，用的正是这个
  PASS。R 部分做对了（PASS 对 none@N，N = none 自己的 pick；P ≠ N 时 shuf − none 标为 lr-confounded），S 部分没有同样处理。
- 能验证的检查：`grep -n "none@4e-3" runs/ptdom/readout.txt`（月份 :139、城市 :148）；重跑 tmp_audit_ptdom2_a2/none_lr.py 与
  sim_rules.py。
- 改法（用户定，提交前、任何 S 结果之前）：(a) 给 batch A 加 none@4e-3 的 seed 8–15（8 个 run，约 3.7 B200-h；R2 若选中 A 的 none@4e-3
  也能共用），城市、月份的 PASS 对 none@4e-3，none@2e-3 只留给 shuf − none（同 lr）；或 (b) 不加 run，把 S 的 PASS 改名为「同 lr PASS
  (2e-3)」，删掉「= F100's PASS replicated」，城市、月份旁边标「lr-confounded：F100 的 none 是 4e-3」，并且不把它写成 F100 的复现。
  GEOMETRY-SPECIFIC 本身两种改法都不受影响。
- 置信度：对照不一致 = 读代码 + 读 ptdom 读数确认；偏差大小 = ptdom 数据 + 模拟。定级为阻断，是因为它要在任何 S 结果出来之前改登记；
  改法本身很便宜。

### 重要 2. 登记的读法只看两个词：在 FLOOR、只学得快、打乱臂有害或剂量过重时，仍写「增益依赖正确的几何」
- 位置：readout_ptdom2.py:630–634（READING 只按 (PASS, GS) 两个布尔值取文本）、:655（`reading = READING[(ok1, ok2)]`）；
  AUDIT.md:444–446。
- 为什么会测错：这一行是预注册的「怎么写」，会被直接搬进 FINDINGS。可是它不看旁边已经算出来的 FLOOR 标签、final3 / 窗口拆分、
  shuf − none 和打乱臂的 LM 份额。我植入的例子（planted_s.py，走真实的 load_all → verdict_S → report）全部给出「PASS and
  GEOMETRY-SPECIFIC: the gain depends on the correct geometry」：
  - 月份式 FLOOR（两臂约 2–6，线 13.3）：FLOOR 标签打印了，读法照写。F100 的月份本来就是 FLOOR，而且 84% 的增益在窗口内；
  - 人物式 FLOOR 线 > 100（捷径 95.4）：「the tag holds unconditionally」照印，读法照写。F100 的人物本来就是这样；
  - 只学得快（面积 +3，final3 0）：读法照写，final3 那行是 −0.08。F100 的城市「基本是学得更快」；
  - 真臂 +1.5、打乱臂 −4（vs none）：读法照写；true − shuf 里约 73% 来自打乱臂的伤害。模拟（真 +1、打乱 −2）：P(两个词都给)
    国家 0.13、城市 0.17、月份 0.39、人物 0.81；
  - 打乱臂 LM 份额 0.2（先验梯度主导更新，剂量远重于真臂的 0.9）：打乱臂的门槛只描述，GEOMETRY-SPECIFIC 照给，读法照写。
  上一轮审查建议的条件是「GEOMETRY-SPECIFIC 只有在 PASS（R）或 shuf ≥ none（S，在噪声内）时才读成真几何有用；shuf 明显 < none 时读成
  错误目标有害」。A2 只保留了 PASS 这一半。再加上阻断 1，S 的 4 个领域里有 3 个（人物 FLOOR、月份 FLOOR / 窗口内、城市 学得快 +
  lr 混杂），F100 已经记录过会让这句话说过头的原因。
- 能验证的检查：tmp_audit_ptdom2_a2/planted_s.py 的 floor_m / floor_p / fast / harm_small / gate_shuf 五个世界（planted_s.out）。
- 改法（结果前登记，仍然只是描述，不改判词）：读法按条件降级。FLOOR → 「FLOOR：不可读成几何依赖」；final3 增益在噪声内或窗口内份额
  ≥ 80% → 「依赖正确几何的是学得快 / 窗口内的读出」；shuf − none 的 90% bootstrap 上界 < 0 → 「真 > 打乱，其中打乱有害占 x%」；
  打乱臂门槛不过 → 「剂量不匹配」。
- 置信度：读代码 + 植入确认；「真实结果会落进这些情形」是推测，但 F100 的三个领域已经是这些情形。

### 次要 3. 新规则的测试有盲区：7 个变异自测抓不到
- 位置：readout_ptdom2.py 的 selftest（:852–1197）；tests/mutation_readout.py（22 个变异里没有这些）。
- 我自己的变异（mut_a2.py，临时副本放在审查目录，import 通过软链指向未改动的真文件；对照副本 49/49 PASS）：
  - **没抓到**：x1 WIN 多一个 eval（(916, 7500)，自测的容差 0.05 太松，:1123）；x6 打乱臂的 LM 份额换成真臂的格子；x7 去掉 FLOOR 线 > 100
    的注释；x8 CEILING 永不打印；x9 去掉 true vs shuffled 的 FLOOR（floor_geo）；x11 去掉「S 三臂必须是 ptdom2 run」的检查；x13 cka 双路
    的条件恒为真（:607，植入只有全过的一种情形）。
  - 抓到：x2 WIN 少一个 eval、x3 GS 的 Holm 用了 PASS 的、x4 S 的 PASS 对打乱臂、x5 shuf − none 符号反了、x10 门槛读打乱臂、x14 读法忽略 GS。
  - x12（S 的 seed 倒序）是等价变异（两半的集合不变），不算盲区。
- 都是描述行或不会发生的情形，不影响判词；但 WIN 是判读「增益在窗口内 / 外」的关键，见次要 4。
- 置信度：确认（mut_a2.out）。

### 次要 4. 窗口拆分的 WIN 是硬编码常数，没有和 run JSON 的 prior_events 双路核对
- 位置：readout_ptdom2.py:559（WIN = (916, 7323)）、:562–577（win_split 只读常数）。
- 事实：在真实 run 上核对过（win_check.py，ptdom-A-only-cities-lr2e-3-s0）：prior_events [[916, 1], [7324, 0]]，prior_on 的 eval
  正好是 1000..7250 的 26 个 = WIN 内的 eval。**现在一致**，in + out = 面积增益也成立（series_problems 双路，在 ptdom 的 384 个格子上
  无问题）。
- 风险：常数与训练器解耦。窗口一旦改动，不会报错；自测也抓不到「多一个 eval」（次要 3 的 x1）。
- 检查 / 改法：在 load2 或 win_split 里 assert：先验臂的 `[e.step for e in c if e.prior_on]` = WIN 内的 eval 步，且
  `r["prior_events"] == [[916, 1], [7324, 0]]`。
- 置信度：确认。

### 次要 5. FLOOR 线 > 100 的注释是写死的结论，不一定成立，来源也可能不是训练频率捷径
- 位置：readout_ptdom2.py:551–552。
- 为什么：线 > 100 只说明 max(lines) > 95，不说明「捷径在两臂之上」。两臂可以在 max(lines) 与 max(lines) + 5 之间（例如捷径 95.4、
  两臂 97 / 96），仍然是 FLOOR，但注释会说「the training-frequency shortcut is above both arms」。另外，让线超过 100 的也可能是
  答案频率启发式（ranks 的 heuristic_ansfreq = 1.0 → 线 105），不是训练频率捷径。F100 的第 5 条只是人物的个案。
- 改法：注释写成「the best non-geometric line {max(lines):.1f} > 95: FLOOR holds whenever both arms <= {line}」，并写明是哪条线。
  这一条没有测试（次要 3 的 x7）。
- 置信度：读代码确认。

### 次要 6. --allow_partial 下「PROVISIONAL (PASS if final)」用的是缩小后的 Holm 族
- 位置：readout_ptdom2.py:639–643（ok_fam = 有完整数据的领域；Holm 只在 ok_fam 上做）。
- 植入：月份缺一个打乱 run → 其余三个领域的 would-be 词按 Holm × 3 算。而正式判词一定是 Holm × 4。readout_ptdom 的原则是「Holm 族
  从不悄悄缩小」。不影响正式读数（严格模式拒绝），但「先看看」时 would-be 词偏松。
- 改法：缺失领域的 p 按 1 参与 Holm，或者 would-be 词注明「Holm over n of 4」。
- 置信度：读代码 + 植入确认（planted_s.out 的 miss_partial）。

### 次要 7. 打乱臂的 lr 不是为它选的：2e-3 是 F100 在真臂上（seed 0–3）挑的
- 位置：AUDIT.md A2.1；make_jobs_ptdom2.S_LR。
- 为什么：同 lr 是「剂量匹配」的设计选择，seed 层面已经没有赢家诅咒。但 lr 层面有不对称：2e-3 是真臂的最优或近最优（四个领域的
  argmax 是 1e-3 或 2e-3），而 none 在城市、月份偏好 4e-3。如果打乱臂的行为更像「none + 伤害」，它在 2e-3 上可能离自己的最优更远，
  true − shuf 会带一点 lr 不对称。量级小于阻断 1，方向偏向 GEOMETRY-SPECIFIC。
- 检查：S 结果出来后，看 shuf@2e-3 − none@2e-3 与 none 的 lr 曲线（none@2e-3 vs 4e-3 的差）是否同号同量级。
- 置信度：推测。

### 次要 8. 文档 / 溯源上的残留
- readout_ptdom2.py:22–23 的模块 docstring（「PRE-REGISTERED」那段）仍写 Part S = ptdom 真臂 seed 0–7 vs ptdom2 打乱臂；
  make_jobs_ptdom2.py:11–12 仍写 S = seeds 0–7、32 runs。代码行为是 A2 的（我核过），只是登记文本和代码不一致。
- tests/refs_ptdom2.py 在 22:32:04 改过，晚于它生成 ptdom2_refs_B1.json（22:30:59）。A2 表里的 ab5685dba69a 是改后的版本（大概率只改了
  shufcka 的格子表，:229），生成 B1 参考的那一版没有留档。参考错了作业会 exit 3（DRY rows / lm0 / table sha 在作业里核对），不会静默。
- submit_ptdom2.sh 在 22:45:55 最后一次修改，晚于所有 A2 测试日志；A2 版本没有跑过打印模式（tests/submit_check_S.out 还是 A0 的
  32 个带 --save_model 的作业，下次运行会覆盖）。`bash -n` 通过；闸门逻辑我逐行读过，没有问题。
- S0 的 32 个作业文件已经渲染在 runs/ptdom2/（job_ptdom2-*-shuf-lr2e-3-s0..7.sbatch），`submit_ptdom2.sh --part S0 --go` 可以提交。
  协调者决定不跑 S0，但没有 HOLD_S0（我不写 HOLD；建议由协调者决定是否放一个）。
- 队列里还有两个 20:14 提交的 CPU 作业 ptdom2refsB1（22348231）/ ptdom2refsB2（22348230），PENDING。它们跑 `run_refs_ptdom2.sh B1 / B2`，
  会遇到 refs_B1.done / refs_B2.done 后直接退出，无害；只是占队列。
- 置信度：确认。

### 次要 9. 读数之间的耦合（不会测错，但会互相阻塞）
- readout_ptdom2.main 的 probs（integrity / manifest / R1pre / series）是全局的：R2 的任何一个 run 数据不一致，`--part S` 也会被拒；
  反过来，select_ptdom2 --after R1 会对 batch A 的 seed 8–15（S 的城市 / 月份 / none 格子）做 integrity。
- 当 A 的 none pick 是 2e-3 时，R2 依赖 S 跑过 A none@2e-3 的 seed 8–15（cells("R2") 不再渲染它）。如果 S 没跑，读数会在缺 run 时
  拒绝（报错清楚，不会静默）；提交脚本不提醒这一点。
- 置信度：读代码确认。

## 逐题回答

### 1. 阻断 1（赢家诅咒）是否真正消除
- **S 的判词只读 seed 8–15**：verdict_S → contrast_domain(..., R2_SEEDS)（readout_ptdom2.py:699；R2_SEEDS = S_SEEDS2 = 8..15）；两半是
  g[:4] / g[4:] = 8–11 / 12–15；A1.7 门槛在 8–11、12–15 上分别判（:623、:657）。seed 0–7 只出现在描述行 seeds0_7_selected 里，而且
  只有 S0 跑过才有（S0 不跑 → None → 不打印）。变异 m11（S 读 0–7）被自测抓到。**确认。**
- **从未被任何选点 / 筛选读过**：decide_R、gate_cells、unit / select_unit、none_asks_from_ptdom 都只读 SEL（0–3）；check_existing 只看
  source == ptdom；select_ptdom2 只对 SEL 打印无效 run。读 seed 8–15 的只有 RO.integrity（数据一致性，不是选择）和 CPU 参考
  （不含任何结果）。F100 的 lr 2e-3 和四个领域是在 seed 0–7 上选的，这在新 seed 上不构成 seed 层面的偏差（lr 层面的不对称见次要 7）。
  **确认。**
- **三臂同 seed 严格可比**：
  - 数据：同 batch、同 seed 的所有有效 run 都被 RO.integrity 核对 data_sha（每次 eval）和 step-0 lm（植入：一个打乱 run 在一次 eval 的
    data_sha 被改 → 拒绝；--allow_partial → 全部 PROVISIONAL 且打印 PROBLEMS）。
  - 快照：同一个 runs/ptdom2/code（作业里核 sha），code_sha = 训练器文件的 sha，三臂相同。
  - 作业体：80 个 S 作业的 ARGS 和 intent 我独立比过 ptdom 的 stage-1 孪生（intent_args_diff.py）：none / 真臂除 seed、tag 和 seed 相关的
    参考字段外完全相同；打乱臂只多 args.prior_shuf 与 intent.shuf。seed 12–15 的孪生（ptdom s4–7）多 --save_model（A1.9 已登记）。
  - CPU 参考：一个 batch、一个 seed 对应一份参考（refs2），三臂共用；B1 的 seed 8–15 参考用 none@4e-3 与两条打乱臂（seed 8）生成。
    真臂在 seed 8–15 上没有单独的 CPU dry 测试，但 DRY 行与臂无关（T5：数据相同），作业内核对不过就 exit 3。
  - 剂量：λ、窗口、位点、lr 相同；**实际更新份额不一定相同**（打乱目标更难拟合）。A2 只把它作为描述性的 LM 份额打印出来，打乱臂的门槛
    也只是描述（见重要 2 的 gate_shuf 植入）。
- **none@2e-3 按批共享**：cells("S") 每个 seed 渲染一个 B1 none@2e-3 和一个 A none@2e-3（make_jobs_ptdom2.py:357–359）；cells("R2") 去掉
  S 已有的格子（:388；render_diff：合成 picks 里 A none@2e-3 的 8 个 seed 被去掉，144 → 136）。同名 = 同一个 run，读数里是同一个 key；
  R2 不会重复渲染。重跑走 `--part S --rerun`（-r1，读数取最新，对 S、R 都成立）。唯一的副作用是 R2 依赖 S（次要 9）。**与 R2 不冲突。**
- 结论：赢家诅咒已经消除。残留的是阻断 1（PASS 的对照 lr）和次要 7（lr 只为真臂选过）。

### 2. 重要 2、3 是否正确实现
- **shuf − none**：S 用同 lr 的 none@2e-3；R 用 none@N，P ≠ N 时打印「lr-confounded」（readout_ptdom2.py:599、:744）。变异 x5（符号反）被抓到。正确。
- **窗口内 / 外拆分**：in = Σ_{窗口内 eval} 100·Δ / 75，out = Σ_{其余} 100·Δ / 75，相加 = 100·mean(Δ) = 面积增益。前提是面积 =
  100 × series 的均值，A2 的 series_problems 对每个有效 run 双路核对（ptdom 的 384 个格子上无问题：真实目录上 --allow_partial 读数没有
  PROBLEMS 行）。窗口定义与训练器一致：真实 run 的 prior_on eval = 1000..7250 共 26 个 = WIN（win_check.out）。WIN 是常数，没有双路核对
  （次要 4）。
- **LM 份额**：真臂、打乱臂 = RO.cell_gate 的中位数（每个 run 窗口内 a / (a + b) 的中位数，再在 seed 8–15 上取中位数）；none 写死 1.0。
  正确，但没有测试（x6）。
- **FLOOR / CEILING**：定义与 readout_ptdom 相同（两臂面积 ≤ max(机会、启发式、整题捷径基线、k 步最短跨度线) + 5；CEILING = 参照臂
  ≥ 95，参照臂对 none 对比是 none、对打乱对比是打乱臂）。月份的线：真实 run 的 chance_readout = 1/12、heuristic_shortest_kspan =
  0.0833 → 13.33，与 A2.3 的「13.3」一致；月份没有消除基线（ptdom 读数「elimination baseline -」），所以不缺线。线 > 100 的注释措辞
  说过头（次要 5）。CEILING、floor_geo、注释都没有测试（x7–x9）。
- **登记的读法会不会在不该时给出「增益依赖正确的几何」**：会，见阻断 1（城市、月份的 PASS 带 lr 混杂）和重要 2（FLOOR、只学得快、打乱臂
  有害或剂量重时照写）。只有打乱有害、真臂 = none 时，读法正确降级为「GEOMETRY-SPECIFIC without PASS ... may hurt」（植入 harm，4/4）。

### 3. 判词实现（R 与 S）
- 四条规则（rule，:467–476）：bootstrap 90% 下界 > 0（RO.boot_ci，10,000 次，default_rng(0)，百分位）、两半均值 > 0、单侧 t（RO.sf_test，
  df 7）的 Holm p < 0.10、≥ 6/8 严格 > 0。两个对比各自做 Holm（:642–643；变异 x3「GS 用 PASS 的 Holm」被抓到）。族：R = FINAL 的领域
  （NOT FINAL / GATE 不在族里），S = 四个领域。A1.7：真臂（pick 格子）在两半分别过门槛，否则两个词都变成「AUX GATE FAILED at the pick」，
  领域留在族里（:657–661；植入 gate：S 四个领域都正确）。NOT FINAL 只对 R。拒绝与 PROVISIONAL 见下。**没有发现算错。**
- 我的植入（planted_s.py，13/14；那 1 个「FAIL」是我的期望写得太严：5 个零效应世界 40 个词里有 3 个不是 NOT SHOWN，和下面模拟的
  名义率一致，不是代码问题）：
  - 零效应：见模拟。每个词约 0.023–0.033，两个词都给约 0.003–0.005（Holm × 4），偏保守。
  - 只有打乱有害（真 = none，打乱 = none − 3）：四个领域都是 NOT SHOWN + GEOMETRY-SPECIFIC，读法写「may hurt」。正确。
    模拟（打乱 −2）：P(GS) 国家 0.43、城市 0.51、月份 0.92、人物 1.00；P(PASS) 约 0.025。
  - 只学得快（面积 +3，final3 0）：PASS + GEOMETRY-SPECIFIC（按设计，面积是主指标），读法「depends on the correct geometry」，final3 行
    −0.08（重要 2）。
  - 缺 run（A only-months-shuf s13）：严格模式拒绝；--allow_partial 四个领域都 PROVISIONAL（would-be 词的 Holm 缩小，次要 6）。
  - 完整性不符（一个打乱 run 的 data_sha 在一次 eval 被改）：拒绝；--allow_partial 全部 PROVISIONAL，报告打印 PROBLEMS。
  - 身份不符（-shuf 目录里 args.prior_shuf 为空）：拒绝。已完成 run 后又有 -r1 目录：拒绝（integrity：earlier finished）。
  - 门槛：真臂在 seed 12–15 的 LM 份额 0.3 → 「AUX GATE FAILED at the pick」；打乱臂份额 0.2 → 只描述，判词照给。
- 模拟（sim_rules.py，每个场景 4000 次，真实 apply_rules；[P(PASS), P(GS), P(两者)]）：
  - null：国家 [.026, .029, .004]、人物 [.029, .033, .005]、城市 [.026, .024, .003]、月份 [.023, .024, .003]
  - 打乱 −2：国家 [.024, .427, .020]、人物 [.024, 1.0, .024]、城市 [.025, .505, .019]、月份 [.027, .917, .026]
  - 真 +1 / 打乱 −2：国家 [.147, .717, .134]、人物 [.805, 1.0, .805]、城市 [.178, .818, .171]、月份 [.385, .998, .385]
  - S 的 lr 混杂（真臂 = F100 的最佳 none）：城市 P(PASS) .201、月份 .256；国家 / 人物 ≈ null

### 4. 渲染、名字、闸门
- 作业体：render_diff.out（17/17）覆盖 S 80、S0 32、R1 24、R1pre 8、R1x 40、R2 136、pilot 2。我另外直接 diff 了磁盘上的作业文件
  （B1 none / 真臂 / A 月份 seed 8 vs ptdom seed 0）：只差作业名、日志名、注释、R / OUT / TAG / SEED、HOLD_S 行、DRY 哈希、run 文件名；
  80 个 S 作业的 ARGS / intent 逐项比过（intent_args_diff.out），只有登记的差异。**一致。**
- 名字 / out 目录：runs/*/out/ 里没有 ptdom2-* 或 -s8..-s15 的已有目录；runs/ptdom2/out 为空；生成器在渲染时 assert 不和其他 batch 的
  out 目录同名；R2 不渲染 S 的格子。**未发现冲突。**
- 闸门（submit_ptdom2.sh，逐行读过，`bash -n` 通过）：HOLD / HOLD_<part>、配额 < 97%（现在 95.86%）、快照与 sha、自测、pilot PASS、
  --check、一个 part 只提交一次（看 --print 里的 SUBMITTED）、R1x / R2 的 picks 只读且 --verify 一致。**可靠。** 小缺口：A2 版本没跑过
  打印模式；S0 没有闸门；R2 不检查 S 是否跑过（次要 8、9）；pilot 闸门不绑 sha（协调者已记录，不处理）。

### 5. A2 是否引入新的副作用或回归
- R 的判词改成走 contrast_domain / apply_rules：自测的 R 部分（+6、零、两半、Holm × 6、A1.7、NOT FINAL、缺 run）仍然全过；我的对照副本
  49/49。select / decide_R 没动规则，只加了「PRIOR AT ITS LOWEST LR」注记（summarize_pick 不比较 notes，不影响 picks 一致性）。
- `--json`：在植入世界上 json.dumps(default=conv) 正常（json_dump.py）。
- 修 bug 的副作用：--allow_partial 的 Holm 缩小（次要 6）；main 的全局 probs 让 S 与 R 互相阻塞（次要 9）。
- 测试还咬得住吗：A2 的 22 个变异都被抓到；但新规则里 7 个变异抓不到（次要 3），主要是描述行和 FLOOR 的分支。
- **未发现会改变判词数字的回归。**

### 6. 陷阱清单
- 训练 / 评测分布不一致：未发现（三臂同训练行、同 held-out 读出，data_sha 每次 eval 核对）。
- 记忆代替泛化：继承的问题仍在。比较类领域是 Tier L；人物的训练频率捷径（95.4）高于两臂；ranks 的 held-out 题被答案频率启发式全部
  答对。新的 tied-embedding 句子（readout_ptdom2.py:805–807）措辞正确。读法会在 FLOOR 时照写「依赖几何」（重要 2）。
- 静默失效的损失 / 梯度：未发现会改判词的。打乱臂的梯度在作业内核对 > 0（CHECK_PY_SHUF），λ 走 ARGS / lam_domains / 分解双路。打乱臂的
  实际更新份额只描述（重要 2）。
- 单 seed 偶然：未发现（8 seed、两半、6/8、bootstrap）。
- 读错位置：位点两臂相同（shuf_trace T2）。窗口拆分的 eval 位置现在与训练器一致，但是靠常数（次要 4）。
- 同名不同义：**S 的「PASS」≠ F100 的 PASS（城市、月份，阻断 1）**；LM 份额里 none 的 1.0 是占位值，不是测出来的；FLOOR > 100 注释的「训练频率
  捷径」可能是答案频率线（次要 5）；S 用的是 R2_SEEDS 这个名字（值 = S_SEEDS2）。
- 静默过滤样本：严格模式下没有（缺 / 无效 run 都拒绝）。描述行在缺数据时打印「-」（win_split / hard3_pair 返回 None），只在
  --allow_partial 下出现。
- 修 bug 的副作用：见第 5 题。

## 复现
/scratch/yh6210/tmp_audit_ptdom2_a2/（PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib，open_r1 python -B）：
- none_lr.py：S 领域 none 各 lr 的面积与 final3（读上一轮审查的 cells.json = ptdom 384 个有效 run 的抽取）；
- sim_rules.py 4000 → sim_rules.out：真实 apply_rules 的 Monte Carlo；
- planted_s.py → planted_s.out（合成世界在 worlds/）；json_dump.py；
- mut_a2.py → mut_a2.out（变异副本在 mut/runs/ptdom2，软链指向未改动的真文件）；
- win_check.py（真实 run 的窗口 / series = area）；months_meta.py；intent_args_diff.py → intent_args_diff.out。
