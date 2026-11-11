# ptdom2 Amendment A3：全新对抗审查（2026-11-09，只读）

审查人：全新子 agent，没有参与 A3 的修订。范围：AUDIT.md「Amendment A3」（对照 A0–A2 与前两轮审查 tests/audit_build.md、
tests/audit_a2.md）、make_jobs_ptdom2.py（62d680144854）、readout_ptdom2.py（1584aed5dd92）、select_ptdom2.py（1c2dfc6adda0）、
submit_ptdom2.sh（fb10e0897d31）、tests/*.py 与 *.out、runs/ptdom/readout.json / readout.txt。开审时核对过：四个文件的 sha 与
AUDIT A3 表一致；runs/ptdom2/out 为空；除 pilot 外没有 SUBMITTED_ 标记。
写入：只有本文件。全部检查脚本、合成数据与日志在 /scratch/yh6210/tmp_audit_ptdom2_a3/（见文末「复现」；TMPDIR 也在那里，python -B，
PYTHONDONTWRITEBYTECODE=1）。审完核对过：四个代码文件的 sha 审查前后不变；runs/ptdom2 下我没有写任何文件（审查期间新出现的
pilot/ 输出、code/tools/__pycache__（00:09–00:17）和 AUDIT.md 末尾的「GPU pilot result」段（00:22）来自 pilot 作业和协调者，不是本审查）。
没有提交、取消任何作业，没有写 HOLD / picks。pilot 结果不在本审查范围，没有用它下任何结论。

## 结论先行

A3 的 (1) 做对了：Part S 的 PASS 对每个领域都用 F100 的 none pick（国家 / 人物 2e-3，城市 / 月份 4e-3，和 runs/ptdom/readout.json、
readout.txt:139/148/153/163 一致）；check_s_none 在不一致时真的会让读数拒绝（我改 S_NONE、改 readout.json 副本的 pick_none / pick_prior
三种方式都报出问题，main 里进 probs → SystemExit）；新增的 A none@4e-3 seed 8–15 共 8 个格子，S 共 88 个、无重复、有 CPU 参考、
作业体与 ptdom 孪生只差登记的替换，R2 在 A 的 none pick 是 2e-3 或 4e-3 时都不重复渲染；同 lr 描述行用的是 only-d@2e-3 − none@2e-3。
判词（四条规则、两半、Holm、门槛、NOT FINAL、拒绝 / PROVISIONAL）A3 没有改动，零效应下每个词约 0.023–0.027，正确。

问题集中在 (2)，也就是「增益依赖正确的几何」这句话：
- 五个条件里有两个是对噪声不敏感的点估计门槛，「只学得快」这一条在它要挡的情形下约 1/3 的概率挡不住（自测里的已知答案也是靠
  rng 种子碰巧过的：换 rng 2、3、4、6 自测就在这一条上 FAIL）；
- 没有上界：打乱臂拿到大部分增益时（真 +4、打乱 +3）照样印这句话；
- A3.1 把 PASS 的 none 和 shuffled − none 的 none 分开后，新出现一条崩溃路径：A none@2e-3（城市 / 月份只用于描述）缺失或无效时，
  严格模式要么报一个误导的拒绝理由，要么直接 TypeError，--allow_partial 也 TypeError。

最可能测错的地方：**A3.2 的「only learns faster」条件（两个 final3 增益均值 > 0）**——它是符号检验而不是噪声检验，纯「学得快」的世界
里约 1/3 的概率照印「the gain depends on the correct geometry」，而这正是 A3 要挡的情形。

## 问题清单（按严重程度）

### 重要 1. 「only learns faster」条件是点估计符号检验：纯学得快时约 1/3 照印那句话；自测的已知答案靠 rng 种子碰巧通过
- 位置：readout_ptdom2.py:666–668（`not (f_n > 0 and f_s > 0)`，f_n / f_s 是 8 个 seed 的 final3 增益均值）；AUDIT.md A3.2
  （「the final3 gains true - none and true - shuffled both > 0 (mean)」）；自测 :1209（fast 世界）与 :930（`default_rng(1)`）。
- 为什么会测错：这一条要挡的是「面积增益只来自学得更快、终态没有差别」。可在 final3 真实增益为 0 时，两个均值各自约一半概率 > 0，
  两者共用真臂的噪声（正相关），同时 > 0 的概率约 1/3。所以 PASS 且 GEOMETRY-SPECIFIC 成立的纯学得快世界里，约 1/3 照印那句话。
- 模拟（sim_reading.py，3000 次，真实的 apply_rules + reading_conditions，Holm × 4；每 run 噪声取 ptdom seed 0–7 的真实值：
  面积 / final3 每臂 sd 国家 1.40 / 2.09、城市 2.02 / 2.52，run 内相关 0.85；人物、月份带 F100 的 FLOOR，永远不印）：

  | 场景（相对 none） | 国家 P(PASS∧GS) / P(印这句话) | 城市 |
  |---|---|---|
  | 零效应 | 0.004 / 0.004 | 0.003 / 0.002 |
  | 只学得快：面积 +3，final3 0 | 0.975 / **0.317** | 0.757 / **0.272** |
  | 只学得快：面积 +5，final3 0 | 1.000 / **0.326** | 0.995 / **0.306** |
  | F100 城市式「大部分学得快」：面积 +3.5，final3 +1.3 | 0.992 / 0.760 | 0.879 / 0.606 |
  | 真几何效应 +3（面积与 final3），打乱 = none | 0.972 / 0.909 | 0.760 / 0.643 |

  F100 的城市（面积 +3.5、final3 +1.3，audit_a2 说「基本是学得更快」）按这条规则大多会印这句话；那是登记的写法，不是 bug，但应知道
  这一条只挡「终态完全没差」，挡不住「大部分是学得快」。
- 自测的已知答案不稳：mut_a3.py 只把自测的 `default_rng(1)` 换成 2..9（其他一字不改），rng 2、3、4、6 下自测 FAIL 在
  「A3.2 fast (area +3, final3 0) -> no sentence」这一条（另有 rng 6、9 FAIL 在 A2 S 的植入组合，见次要 5）。也就是说 fast 世界的
  final3 增益只是噪声，61/61 是 rng(1) 恰好给了一个负均值。变异 m24（去掉 final3 条件）被「抓到」也是同一个运气。
- 能验证的检查：重跑 tmp_audit_ptdom2_a3/sim_reading.py 与 mut_a3.py 的 r2..r9 行；planted_a3.out 的 W2（城市面积 +3、final3 +0.3
  → 印这句话）。
- 改法（结果前登记，仍只是描述）：把条件改成噪声感知的，例如两个 final3 增益的 90% bootstrap 下界 > 0（或单侧 t p < 0.10），或
  final3 增益 ≥ 面积增益的某个比例；自测的 fast 世界把 final3 增益设成确定的 0 或负值（不要叠加噪声后再取符号）。
- 置信度：读代码确认 + 模拟 + 自测换种子复现。

### 重要 2. 没有上界：打乱臂拿到大部分增益时照印「the gain depends on the correct geometry」；−1 的下界对噪声不敏感
- 位置：readout_ptdom2.py:669–671（只有 `sn < SHUF_WORSE` 一侧）、:657（SHUF_WORSE = −1.0，一个点估计门槛）；AUDIT.md A3.2。
- 为什么会测错：
  - 上界缺失：真 − none = +4、打乱 − none = +3 时，GEOMETRY-SPECIFIC 只要过了（真 − 打乱 = +1），这句话就印出来；但 3/4 的增益不需要
    正确的几何。planted_a3.out W1：国家 gain +3.85、shuffled − none +3.18 → 「PASS and GEOMETRY-SPECIFIC: the gain depends on the
    correct geometry」。模拟：真 +4 / 打乱 +3 时 P(印) = P(GS)（国家 0.306、城市 0.189）；真 +4 / 打乱 +2 时 0.809 / 0.563。
  - −1 的下界只看均值：真 +1.5 / 打乱 −0.9 时 P(印) 国家 0.364、城市 0.232，此时 true − shuf 的 37% 来自打乱臂的伤害（W3 也印）；
    反过来，真实的 +5 几何效应（打乱 = none）因为 shuffled − none 均值的噪声（sd 约 0.7 / 1.0）掉到 −1 以下，被误挡 8%（国家）/ 17%（城市）。
  - 城市 / 月份的 shuffled − none 按登记用 none@2e-3（同 lr），而 none@2e-3 在 ptdom 上比 F100 的 none（4e-3）低 1.32 / 0.84：
    一个比最佳 none 低 1.8 的打乱臂只比 none@2e-3 低 0.5，照过这一条（W4；模拟「A gap」场景城市 P(印) 0.624）。这是 A3.1 登记的
    选择，只是要知道它对城市 / 月份更宽。
- 能验证的检查：planted_a3.out W1 / W3 / W4；sim_reading.out 的 shuf gets most / half、harm under the bar、real geometry +5、A gap 行。
- 改法（描述层面，结果前登记）：把句子改成带份额的「x of the y-point gain depends on the correct geometry」（(真 − 打乱) / (真 − none)），
  或加上界（例如 shuffled − none 的 90% bootstrap 上界 < 真 − none 的一半）；下界改成 shuffled − none 的 90% bootstrap 下界 ≥ −1，
  或注明「点估计」。
- 置信度：读代码确认 + 植入 + 模拟；「打乱臂会拿到增益」本身是推测（等数据）。

### 重要 3. A3.1 新引入的崩溃路径：城市 / 月份的 A none@2e-3（只用于描述）缺失或无效时，S 读数崩溃或给出误导的拒绝
- 位置：readout_ptdom2.py:624（`gsn = pair(..., NS_ ..., [])`：none@2e-3 缺失 / 无效不进 problems）、:627–628（`have` 要求 NS_ 的
  run 也有效）、:649–650（`have` 为假 → `gate = None`）、:710–711（apply_rules 对 None 取下标）；:756–759（来源检查对缺失的 none@2e-3
  报「must be ptdom2 runs」）。
- 为什么会测错：A2 里 N = N_same，none 缺了 gain 就是 None、领域走 PROVISIONAL / 拒绝。A3.1 让城市 / 月份的 PASS 用 none@4e-3、描述行用
  none@2e-3，于是出现 gain / geo_gain 都在、但 `have` 为假的情形。植入（planted_a3.py，真实 load_all → verdict_S）：
  - A none@2e-3 s11 缺失，严格模式：拒绝，但理由是「cities: Part S's arms at seeds 8-15 must be ptdom2 runs」（误导）；
  - 同上，--allow_partial：`TypeError: 'NoneType' object is not subscriptable`（:711）；
  - A none@2e-3 s11 存在但无效（一个 check 失败），严格模式和 --allow_partial：都是同一个 TypeError，没有任何拒绝理由；
  - 对照：A none@4e-3 s11 缺失 → 正常拒绝 / PROVISIONAL；打乱臂无效 → 正常拒绝。
  按 A2.3「跑完的 run 不重跑」，一个跑完但无效的 A none@2e-3 会让整个 Part S（包括与它无关的国家 / 人物 PASS）永久读不出来，只能在
  结果出来之后改代码。
- 能验证的检查：planted_a3.out 的 W7 / W8 / W9 / W11。
- 改法：gate 不依赖 `have`（只看 T_ / S_）；shuffled − none 的格子缺失 / 无效时进 problems（严格模式给出正确的拒绝理由）或把这一行
  记作「-」并在 A3.2 条件里按失败处理；来源检查只看存在的 run。补一个自测：none@2e-3 缺失 / 无效。
- 置信度：读代码 + 植入确认。

### 次要 4. 新规则的测试盲区：9 个 A3 相关变异自测抓不到
- 位置：readout_ptdom2.py 的自测（:923–1315）；tests/mutation_readout.py。
- mut_a3.py（临时副本在审查目录，import 通过软链指向未改动的真文件；对照副本 61/61）：
  - **没抓到**：y1 check_s_none 恒返回 []（拒绝路径从未被测：自测只在真数据上断言它返回 []；我另外直接核过它会拒绝，
    check_s_none_test.out）；y2 final3 条件 and → or；y3 只查 vs none 的 final3；y4 SHUF_WORSE −1 → −3.5；y5 LM_DIFF 0.1 → 0.5；
    y6 shuffled − none 的数值改成对 none@N（标签仍是 2e-3）；y7 非 final（--allow_partial / problems）时保留这句话；y8 门槛失败时保留
    这句话；y11 S 的 FLOOR（真 vs none）改用 none@2e-3。
  - 抓到：y9 同 lr 行改用 none@N、y10 剂量条件读成真 vs none、y12 门槛不分两半。
- 都不影响现在代码的行为（现在的实现在这些点上是对的，planted_a3 W6 / W9 / W10 核过 y7 / y8 的正确行为），但下一次修改时这些行为没有
  测试保护；y1 正是 A3.1 新增的「拒绝」。
- 置信度：确认（mut_a3.out）。

### 次要 5. 自测的植入已知答案依赖 rng 种子；A3 的若干测试日志不是在最终文件上跑的
- rng：除重要 1 的 fast 世界外，rng 6、9 下「A2 S: planted ... months (0, 0) -> neither」那组也 FAIL（零效应的词以名义率出现假阳性，
  一组里有 4 个零效应词）。现在 rng(1) 固定，提交闸门是确定的；但「自测 PASS」部分是运气，改规则或改世界时会突然 FAIL。
- 溯源：readout_ptdom2.py 最后修改于 23:44:00，晚于 mutation_readout.out（23:34）、real_known.out（23:37）、readout_partial_noruns_a3.out
  （23:40，其报告头仍是「Amendments A1, A2」，现文件是「A1, A2, A3」）、plumbing.out（23:41）、submit_print_a3.out（23:43）；只有
  selftest_readout.out（23:44:10）是在最终文件上跑的。AUDIT A3 表把它们都记在 1584aed5dd92 名下。我在最终文件上重跑了：real_known
  15/15 PASS、plumbing 12/12 PASS（real_known_rerun.out、plumbing_rerun.out），A3 的 34 个变异全部仍被抓到、对照 61/61
  （mut_suite_current.out）。所以 A3 表的结论在最终文件上成立，只是记录的日志不是它的。
- 置信度：确认（mtime + 重跑）。

### 次要 6. 措辞残留：报告里永远出现「the gain depends on the correct geometry only when both hold (A2.2)」
- 位置：readout_ptdom2.py:870–872（报告末尾的 Reading 段，每份报告都印）；:807（「reading (pre-registered, A2.2)」）；:718（PROVISIONAL
  标签「(A3.3: the family is reduced)」在家族完整、只因外部问题而 PROVISIONAL 时也印）；:671（withheld 理由「shuffled - none」不写是哪个
  none，城市 / 月份是 none@2e-3，而 PASS 是 none@4e-3）。
- 为什么：末尾那句仍是 A2.2 的「两个词都成立即可」，与 A3.2 的五条件矛盾；按字符串检索这句话的读者或脚本在每份报告里都会命中
  （planted_a3.out 每个世界都多出 1 次）。
- 改法：末尾改成「only when all five A3.2 conditions hold (see each domain's reading line)」；标签改 A3.2；reduced 只在 n < N 时印。
- 置信度：读代码确认。

### 次要 7. WIN 断言用 assert：一个异常的先验 run 会让整个读数和 select_ptdom2 崩溃，而不是把这个 run 标成无效
- 位置：readout_ptdom2.py:189–192。
- 事实：在 ptdom 的 384 个格子上成立（real_known 重跑 PASS），A3.3 的意图（窗口常数和训练器对齐）达到了。风险只在出现异常 run 时：
  AssertionError 会挡住 --allow_partial 和 R1 的选点（select_ptdom2 --after R1 也走 load_all）。是响亮的失败，不会静默。
- 置信度：读代码确认；触发概率低（推测）。

### 次要 8. 剂量条件可能系统性地挡掉真实效应（推测，提交前可查）
- 位置：readout_ptdom2.py:672–675；AUDIT A3.2。
- 为什么：|LM 份额(真) − LM 份额(打乱)| ≤ 0.1 是确定性门槛。ptdom 真臂 @2e-3 的份额：国家 0.897、人物 0.974、城市 0.904、月份 0.830
  （noise.out）。打乱目标更难拟合，如果打乱臂的先验梯度系统性更大（份额低 0.1 以上），这句话会在所有 seed 组合下被挡，与效应大小无关
  （模拟：+5 的真实效应、份额差 0.12 → 印的概率 0）。这按 A3.2 是「剂量不同」的正确降级，但应事先知道它会不会注定发生。
- 检查：pilot 的 P5（打乱臂 on-eval 份额）对比 ptdom only-countries@2e-3 s0 同一批 on-eval 的份额（我没有读 pilot 输出，留给协调者）。
- 置信度：推测。

### 次要 9. docstring 残留（A1 起）
- make_jobs_ptdom2.py:16–17（R1x：「prior 8e-3 where 4e-3 is the argmax; none one step (x2) above」「0-36 runs」）与 select_ptdom2.py:16–17
  （「E = only-d@8e-3 where 4e-3 is the prior's area argmax」）仍是 A0 的触发规则，A1 已改成 pick 或 argmax、加了下降一步和 fixB、
  R1x 最多 40 个。A3.3 只声称改了 S 的 docstring，代码行为是 A1 的（自测 / plumbing 核过）。
- 置信度：确认。

## 逐题回答

### 1. A3 (1) 是否正确
- **每个领域用的是 F100 的 none pick**：verdict_S 取 `N = M2.S_NONE[d]`（:754），contrast_domain 的 PASS 对比是 only-d@2e-3 − none@N
  （:622）。S_NONE = {countries 2e-3, people 2e-3, cities 4e-3, months 4e-3}，与 runs/ptdom/readout.json 的 pick_none 和 readout.txt
  的 F100 判词行（months :139「vs none@4e-3」、cities :148、countries :153「vs none@2e-3」、people :163）一致；F100 的 pick_prior 四个
  都是 2e-3。注意 readout.txt:67 的 stage-1 筛选行里人物写的是「vs none@4e-3」，那是 stage-1 的 pick，最终判词（:163）与 readout.json
  都是 2e-3：S_NONE 用的是最终 pick，对。**确认。**
- **不一致时真的拒绝**：check_s_none_test.out：S_NONE 改城市 2e-3 → 报错；readout.json 副本把月份 pick_none 改 8e-3 → 报错；把人物
  pick_prior 改 1e-3 → 报错。main（:1344–1346）把它并进 probs，非 --allow_partial 即 SystemExit；--allow_partial 下 PROBLEMS 行 +
  全部 PROVISIONAL。**确认**（但拒绝路径没有测试：次要 4 的 y1）。
- **新增 none@4e-3 seeds 8–15**：cells("S") = 88 个、互不重复、seed 8–15，none 格子 = B1 2e-3、A 2e-3、A 4e-3（check_s_none_test.out (e)）；
  CPU 参考按 batch / seed（refs2：ptdom2_refs_A.json 覆盖 8–15，渲染时 assert）；作业体：job_ptdom2-A-none-lr4e-3-s9 与 ptdom 的
  job_ptdom-A-none-lr4e-3-s1 直接 diff，只差作业名 / slot、日志名、注释、R / OUT / TAG / SEED、HOLD_S、ARGS 的路径、DRY 哈希、run 文件名；
  render_diff 17/17、dryblock 30/30（含 s9）。与 R2：r2_dedup.out 用合成 picks（chmod 444，在审查目录）：A 的 none pick 4e-3 或 2e-3 时
  R2 不渲染 A 的 none（112 = 120 − 8），8e-3 时照常渲染，与 Part S 重叠 0；B2 / B3 不受影响。名字不与其他 batch 冲突（生成器 assert）。
  **未发现冲突、重复或缺参考。**
- **同 lr 描述行**：:760–766，only-d@2e-3 − none@2e-3（R2_SEEDS），四条规则用原始单侧 p、不做 Holm，标签写明「not F100's contrast」；
  只对 N ≠ lr（城市、月份）打印。shuffled − none 对四个领域都是 none@2e-3（N_same = lr）。**格子正确**（y9 变异被抓到）。
- 副作用见重要 3（none@2e-3 缺失 / 无效时崩溃）。

### 2. A3 (2) 是否正确
- **实现与登记逐条一致**（:660–676、:705–709）：FLOOR = floor（真 vs F100 none）或 floor_geo（真 vs 打乱）任一带 FLOOR 标签；final3 =
  两个增益的 8-seed 均值都 > 0（真 − none@N、真 − 打乱）；shuffled − none = 面积均值 ≥ −1（S：none@2e-3；R：none@N）；LM 份额 =
  RO.cell_gate 在 seed 8–15 上的格子中位数（每 run 窗口内 a / (a + b) 的中位数再取中位数；a = grad_lm，b = grad_prior，打乱臂的
  grad_prior 是加权后的打乱项梯度，pretrain_range.py:2224–2230，与真臂同义），|差| ≤ 0.1；只在 PASS 且 GS 时计算；门槛失败和非 final
  时 reading = None（:715、:720）。数据都是 seed 8–15、窗口与训练器一致（WIN 断言）。
- **能否在不该时打印**（planted_a3.out，真实 load_all → verdict_S → report；sim_reading.out）：
  - FLOOR：人物（线 > 100）与月份（线 13.3）按 F100 的数据几乎注定 FLOOR → 永远不印。正确（自测 floor_p / floor_m）。也就是说这句话
    实际上只可能出现在国家和城市。
  - 只学得快：**会**，约 1/3（重要 1）。
  - 打乱臂有害：−3 时正确挡住（模拟 P(印) 0.001 / 0.014）；−0.9 时照印（重要 2）。
  - 打乱臂拿到大部分增益：**会**（重要 2）。
  - 剂量不同：份额差 0.11 挡住（W5 国家），0.10 不挡（W5 城市，浮点 0.0999）。正确。
  - 门槛失败：reading = None（W6）。正确。
  - 缺 run：none@4e-3 / 打乱臂 / 真臂缺失 → 拒绝 / PROVISIONAL，不印（W9、W11）；none@2e-3 缺失或无效 → 崩溃（重要 3）。
  - --allow_partial：所有词 PROVISIONAL，reading = None（W9、W10，其中 W10 的国家 / 城市 would-be 是 PASS + GS）。正确。
- **真实效应会不会被误挡**：+3 的真几何效应（面积与 final3，打乱 = none、剂量相同）P(印 | PASS∧GS) 国家 0.94、城市 0.85；+5 时 0.92 /
  0.83（误挡主要来自 shuffled − none 均值的噪声低于 −1，城市 17%）。份额差 > 0.1 时一律挡住（次要 8）。

### 3. 判词整体
- A3 对 rule / apply_rules 的改动只在 reading 与 PROVISIONAL 标签（我逐行读过 :685–722）。四条规则（bootstrap 90% 下界 > 0、两半均值
  > 0、单侧 t 的 Holm p < 0.10、≥ 6/8）、两个对比分别 Holm、S 族 = 4、R 族 = FINAL 的领域、A1.7 两半门槛、NOT FINAL、拒绝、PROVISIONAL
  与 A2 相同；自测 R / S 部分、real_known 重跑 15/15、plumbing 重跑 12/12 都过。
- 零效应误判率（sim_reading.out null，3000 次，真实规则）：P(PASS) 0.023–0.027，P(GS) 0.023–0.024，P(两者) ≤ 0.004，P(印那句话)
  ≤ 0.004。偏保守，正确。
- S 的 PASS 现在对 F100 的最佳 none：对城市 / 月份比 A2（none@2e-3）严格，消除了 audit_a2 阻断 1 的 lr 偏差。

### 4. 渲染与提交
- S 88 个作业：render_diff 17/17（与 ptdom stage-1 孪生逐行一致，只差登记的替换）；我另外直接 diff 了 A none@4e-3 s9（见第 1 题）。
- S0：生成器 --submit 在任何 sbatch 之前 assert --allow_s0（make_jobs_ptdom2.py:490，--rerun 路径同样经过这里）；submit_ptdom2.sh:30 在
  --go 且无 --allow_s0 时 exit 3（s0_gate_a3.out 两条路径）。可靠。残留：32 个 S0 作业文件已渲染在 runs/ptdom2/，直接 sbatch 绕过一切闸门
  （所有 part 都一样），没有 HOLD_S0（audit_a2 已记）。
- submit 闸门（逐行读过，bash 5.2，`set -euo pipefail` 下空数组展开无问题）：HOLD / HOLD_<part>、配额 < 97%、快照与 sha、自测、
  `^PILOT PASS`（pilot_check.py:93 只在全部通过时印「PILOT PASS」，单条规则行以「PASS 」开头，不会误匹配）、--check、未提交过、R1x / R2 的
  picks 只读且 --verify。可靠。自测闸门在 rng(1) 下是确定的（次要 5）。
- 变异测试是否覆盖新规则：A3 的 34 个变异在最终文件上重跑，全部被抓到（mut_suite_current.out）；但新规则仍有 9 个盲区（次要 4），
  而且 m24（final3 条件）被抓到靠的是 rng(1)（重要 1）。

### 5. A3 的副作用 / 回归
- 重要 3（新崩溃路径）；次要 7（WIN 断言让异常 run 挡住全部读数与选点）；次要 6（措辞残留）；次要 5（测试日志早于最终文件）。
- 耦合：check_s_none 进入全局 probs，S_NONE 一旦与 F100 不符，Part R 的读数也被拒（与 A2 的全局 probs 同类，audit_a2 次要 9）。
- 未发现会改变判词数字的回归。

### 6. 陷阱清单
- 训练 / 评测分布不一致：未发现（新格子同训练行、同 held-out 读出，data_sha 每次 eval 由 integrity 核对，包括 A none@4e-3）。
- 记忆代替泛化：继承问题（Tier L、tied embeddings、人物的训练频率捷径）。A3.2 的 FLOOR 条件让人物、月份永远拿不到那句话，处理正确；
  比较类领域的 Tier L 措辞未变。
- 静默失效：未发现。打乱臂的 grad_prior 是打乱项的加权梯度（与真臂同义，LM 份额可比）。
- 单 seed 偶然：判词层面未发现（8 seed、两半、6/8、bootstrap）；**自测层面有**：已知答案依赖 rng 种子（重要 1、次要 5）。
- 读错位置：WIN 现在与训练器的 prior_on / prior_events 双路断言（:189–192），ptdom 384 格子成立。未发现。
- 同名不同义：「shuffled − none」在城市 / 月份是 none@2e-3，而 PASS 与 FLOOR 的「none」是 none@4e-3，withheld 理由不写 lr（次要 6）；
  报告末尾「only when both hold (A2.2)」与 A3.2 不同义（次要 6）；none 的 LM 份额 1.0 是占位值（已标注）。
- 静默过滤样本：未发现（严格模式缺 / 无效都拒绝，例外是重要 3 的崩溃，响亮而非静默）。
- 修 bug 的副作用：重要 3（A3.1 拆分 N 与 N_same 的直接后果）、次要 7。

## 复现
/scratch/yh6210/tmp_audit_ptdom2_a3/（open_r1 python -B，PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib，TMPDIR=./tmp）：
- noise.py → noise.out / noise.json：ptdom seed 0–7 的 S 对比噪声与真臂 LM 份额（读上一轮审查的 cells.json）；
- sim_reading.py 3000 → sim_reading.out：真实 apply_rules + reading_conditions 的 Monte Carlo；
- planted_a3.py → planted_a3.out（合成世界在 worlds/）：W1–W11；
- mut_a3.py → mut_a3.out：y1–y12 与 rng 2–9 的自测（副本在 mut_<k>/）；
- mut_suite_current.py → mut_suite_current.out：A3 的 34 个变异在最终 readout_ptdom2.py 上重跑（副本在 mut_s<k>/）；
- check_s_none_test.py → check_s_none_test.out；r2_dedup.py → r2_dedup.out（合成 picks 在 tmp/）；
- real_known_rerun.out、plumbing_rerun.out：tests/real_known.py 与 tests/plumbing.py 在最终文件上重跑（PYTHONDONTWRITEBYTECODE=1）。
