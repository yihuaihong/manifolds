# ptdom2 Amendment A4：全新对抗审查（2026-11-09，只读）

审查人：全新子 agent，没有参与 A4 的修订。范围：AUDIT.md「Amendment A4」（对照 A3 与 tests/audit_a3.md）、readout_ptdom2.py
（21797ec55b8f）、select_ptdom2.py（78c4a3a757ba）、tests/ 下的 selftest / selftest_rngs.py / mutation_readout.py / real_known.py /
plumbing.py 及其 *.out、rerun_a4.sh / rerun_a4.log。A3 版读数（1584aed5dd92）取自 A3 审查目录里未改动的副本
（tmp_audit_ptdom2_a3/mut_s0），diff 见 tmp_audit_ptdom2_a4/prev/readout_A3_to_A4.diff；wip_a4/ 与现行文件逐字节相同。
数据纪律：没有读任何 ptdom2 run 的面积、增益或 final3 数值；对真实 ptdom2 树只做了结构检查（字段是否存在、有效性类别计数、
prior_events 是否等于 WIN），对两份 --allow_partial 日志只数了行（没有读数字）。噪声参数取自 A3 审查从 **ptdom（不是 ptdom2）**
seed 0–7 提取的 noise.json。
写入：只有本文件。全部检查脚本与输出在 /scratch/yh6210/tmp_audit_ptdom2_a4/（见文末「复现」）。审查前后四个代码文件的 sha 不变
（readout 21797ec55b8f、select 78c4a3a757ba、make_jobs_ptdom2 62d680144854、submit fb10e0897d31）；没有提交、取消任何 ptdom2 作业，
没有写 HOLD / picks。

## 结论先行

- **判词本身没有被 A4 改动**：rule / Holm / 门槛 / 家族 / NOT FINAL / decide_R 的代码未变（diff 只动了 ok_fam 里一个冗余的 gate 条件
  和 PROVISIONAL 标签）。差分测试：A3 与 A4 的 apply_rules 在 4000 组随机家族（16000 行，含缺失、门槛失败、外部 problem）上判词
  逐个相同，final 标志 0 处不同；完整管道 verdict_S 在 8 个带噪世界（32 个领域）上判词、增益、shuffled − none 全同。零效应下
  P(PASS) 0.023–0.028、P(GS) 0.018–0.029（2000 次模拟），仍约 0.025。
- **A4.1 有效**：纯「学得快」（面积 +3 或 +5，final3 0）时打印那句话的概率从 A3 的 0.24–0.34 降到 0.018–0.025。
- **A4.3 / A4.4 的实现正确**：A none@2e-3 缺失、无效、未跑完、8 个 seed 全缺时，严格模式和 --allow_partial 都不拒绝、不 TypeError，
  城市 / 月份照常判词，只有需要它的描述行写「missing / invalid」，那句话被扣下并写明是哪个格子；B1 none@2e-3（国家 / 人物的 PASS 臂）
  缺失或无效时严格模式拒绝，理由准确地写出格子和失败的检查；窗口不符（prior_events 或 prior_on）时该 run 无效、理由是
  「window (A4.4)」，不崩溃。真实 ptdom2 树上 28 个 run 全部有效，22 个先验臂的 prior_events 全部等于 WIN。
- 问题集中在 **A4.2 的两个条件在 ptdom 量级的噪声下挡不住它们要挡的情形**（只影响那句描述性的话，不影响判词）：
  - A4.2a「打乱臂没有被证明有害」把举证责任放在「有害」一侧。真 +1.5 / 打乱 −0.9 这个被 AUDIT 写成「significant harm → 不印」
    的植入例子，只在自测的 sd 0.3 噪声下才显著；在 ptdom 噪声下，国家 / 城市 / 月份拿到 PASS∧GS 后仍有 44–47% 印那句话；
    打乱 −1.5（真 − 打乱的一半来自打乱臂的伤害）时仍有 28–31%。城市这一条反而比 A3 的 −1 点估计门槛更松。
  - A4.2b「打乱臂拿走大部分增益」是点估计，再叠加 GS 的选择效应：真 +4 / 打乱 +3（打乱臂拿走 3/4）时，拿到 PASS∧GS 的城市有 34%、
    国家有 17% 照印那句话；打乱拿走 60% 时为 50% / 33%。
- 检验力：真几何效应 +3（打乱 = none）时，在 PASS∧GS 的条件下印那句话的概率是国家 0.77、城市 0.70、人物 0.93、月份 0.91；
  +5 时 0.89–0.92。上限约 0.91 来自 A4.2a 的百分位 bootstrap 在 n = 8 时的假「有害」率（约 8–10%，名义 5%）。
- 变异测试：我另造的 19 个 A4 相关变异里有 7 个自测抓不到，其中最重要的是「A4.2a 改成均值 ≥ 0」（自测分不出置信区间规则和点估计规则）。

最可能测错的地方：**A4.2a / A4.2b 这两条「打乱臂」条件**——在真实噪声下，伤害约一半的对比或拿走 3/4 增益的打乱臂，仍有约 1/3
的机会让读数写出「the gain depends on the correct geometry」；自测里对应的已知答案只在 sd 0.3 的低噪声世界里成立。

## 问题清单（按严重程度）

阻断：未发现。

### 重要 1. A4.2a 在 ptdom 噪声下挡不住中等程度的伤害；AUDIT 里的「significant harm」植入例子只在自测噪声下成立
- 位置：readout_ptdom2.py:698–706（`lo_, hi_ = RO.boot_ci(sn)`；`if not hi_ >= 0` 才扣下）；AUDIT.md:635–640（A4.2(a) 和「true +1.5 /
  shuffled -0.9 (significant harm) -> no sentence, reason (a)」）；自测 :1359 / :1369（people 世界）与 :1231 / :1234（打乱臂噪声
  zm(0.3)，none 噪声按 seed 共享、在对比中抵消）。
- 为什么会测错：这一条的意图是「真 − 打乱的差距不能主要来自打乱臂变差」。实现是「打乱 − none 的 90% bootstrap 上界 ≥ 0」，即只有伤害
  显著时才扣下。n = 8、每臂 sd 1.4–2.0（ptdom 的国家 / 城市）时，上界约为均值 + 1.1（国家）/ + 1.6（城市），所以小于这个量级的伤害
  通过。同时打乱臂变差会让 A4.2b 自动通过（真 − 打乱 > 真 − none），没有别的条件兜底。
- 模拟（sim_a4.py，2000 次，真实 apply_rules + reading_conditions，Holm × 4，不加 FLOOR；A 批的 none@2e-3 是独立的 run）：

  | 场景（相对 none） | 国家 P(印) / P(印 \| PASS∧GS) | 城市 | 月份 | 人物 | A3 规则 P(印) 国家 / 城市 |
  |---|---|---|---|---|---|
  | 真 +1.5 / 打乱 −0.9 | 0.269 / 0.474 | 0.135 / 0.438 | 0.318 / 0.422 | 0.111 / 0.113 | 0.378 / 0.160 |
  | 真 +1.5 / 打乱 −1.5 | 0.153 / 0.281 | 0.114 / 0.309 | 0.099 / 0.134 | 0.002 / 0.002 | 0.185 / 0.099 |
  | 真 +1.5 / 打乱 −3 | 0.003 / 0.005 | 0.024 / 0.067 | 0.001 | 0 | 0.002 / 0.009 |

  打乱 −1.5 时真 − 打乱 = 3，其中一半是伤害；城市仍有 31% 的 PASS∧GS 印那句话，比 A3 的 −1 点估计门槛（0.099）还略松。人物噪声小
  （每臂 sd 0.64），所以挡得住；而人物、月份按 F100 的数据本来就带 FLOOR，那句话实际只可能出现在国家和城市——恰好是噪声大的两个。
- 和 audit_a3 的关系：audit_a3 重要 2 指出「真 +1.5 / 打乱 −0.9 时照印（国家 0.364）」。A4 后国家 0.269、城市 0.135（下降主要来自
  A4.1，不是 A4.2a），这个情形没有被解决；AUDIT 的植入例子让人以为它被解决了。
- 能验证的检查：`sim_a4_local.out` 的 harm 三行；自测 people 世界的 shuffled − none sd 是 0.3（:1231），把 zm(0.3) 换成 zm(1.4) 后这条
  已知答案会失败。
- 改法（结果前登记，仍是描述层面）：把举证责任放到「没有伤害」一侧，例如 shuffled − none 的 90% bootstrap **下界** ≥ −δ（δ 取 1，或
  0.25 × mean(真 − none)），或者把伤害份额 max(0, mean(none − shuffled)) / mean(真 − shuffled) ≤ 0.25 写成条件、并在句子里印出这个份额；
  自测里加一个 ptdom 量级噪声的伤害世界。代价：真效应下会多扣一些（下界规则在打乱 = none 时约扣掉 P(下界 < −δ)）。
- 置信度：读代码确认 + 模拟（模拟的行构造与真实管道在 24/24 个带噪 (领域, 抽样) 上逐值一致，planted_a4_local.out 末行）。

### 重要 2. A4.2b 是点估计，叠加 GS 的选择效应后，打乱臂拿走大部分增益时仍常印那句话
- 位置：readout_ptdom2.py:707–710（`gs_ >= SHARE_MIN * g_`，两个 8-seed 均值）、:675（SHARE_MIN 0.5）；AUDIT.md:638。
- 为什么会测错：GEOMETRY-SPECIFIC 要求真 − 打乱显著 > 0。在打乱臂拿走大部分增益的世界里，能过 GS 的抽样恰好是真 − 打乱被高估的那些，
  这些抽样里点估计 `gs_ ≥ 0.5 g_` 也更容易成立。条件只看均值、不看噪声，所以拿到两个词之后，打乱臂的份额被系统性低估。
- 模拟（同上）：

  | 场景 | 国家 P(印) / P(印 \| PASS∧GS) | 城市 | 月份 | 人物 |
  |---|---|---|---|---|
  | 真 +4 / 打乱 +3（拿走 75%） | 0.056 / 0.170 | 0.070 / 0.344 | 0.017 / 0.037 | 0.001 |
  | 真 +4 / 打乱 +2.4（拿走 60%） | 0.218 / 0.332 | 0.209 / 0.497 | 0.201 / 0.251 | 0.074 |
  | 真 +4 / 打乱 +2（边界 50%） | 0.426 / 0.528 | 0.321 / 0.567 | 0.477 / 0.511 | 0.499 |

  A3 没有这一条（P(印) 0.33 / 0.20 / 0.44 / 0.80），A4 降低了无条件概率；但读者看到的是「PASS 且 GEOMETRY-SPECIFIC」之后的那句话，
  这时城市在 75% 世界里有 1/3 的概率被写成「增益依赖正确的几何」。
- 能验证的检查：`sim_a4_local.out` 的 shuf takes most / 60% / half 三行。
- 改法：用配对的噪声感知版本，例如每个 seed 的 (真 − 打乱) − 0.5 × (真 − none)（= 0.5 真 − 打乱 + 0.5 none）的 90% bootstrap 下界 > 0；
  或者不做门槛、把份额 mean(真 − 打乱) / mean(真 − none) 和它的 bootstrap 区间直接印进那句话（「x of the y-point gain …」，audit_a3 的建议）。
- 置信度：读代码确认 + 模拟；「打乱臂会拿到增益」本身是推测（等数据）。

### 次要 3. 自测分不出 A4.2a 的区间规则和点估计规则（以及其他 6 个 A4 盲区）
- 位置：自测 :994（zm：每臂噪声在 8 个 seed 上严格零均值）、:1231；tests/mutation_readout.py 的 z 系列。
- mut_a4.py（副本在审查目录，import 通过软链指向未改动的真文件；对照 82/82）：
  - **没抓到**：w3 A4.2a 改成「均值 ≥ 0」（零均值噪声让阳性对照的 shuffled − none 均值恰为 0，区间规则和均值规则给出同一答案；在真实
    噪声下二者差很多：均值规则会扣掉约一半的真效应）；w4 / w27 A4.2b 改用 none@2e-3（A4 世界里城市的 none2 偏移为 0，none@2e-3 与
    none@4e-3 在这一条上没有被区分）；w9 只删掉 prior_on 与 WIN 的比较（自测的窗口篡改只改 prior_events，fake2 的 prior_on 永远来自
    固定窗口 :949）；w11 main 里不再把 check_s_none 并入 probs（自测只直接调用函数，从不走 main 的拒绝组合）；w15 LM_DIFF 0.1 → 0.05
    （没有份额差落在 0.05–0.1 之间的植入世界）；w14 `hi_ > 0`（边界，无实际意义）。
  - 抓到：w1 用上界代替下界、w2 用下界代替上界、w5 SHARE_MIN 0.75、w6 方向反转、w7 A4.1 改用面积、w10 删 prior_events 比较、w12 缺失的
    描述 none 不再扣句子、w13 删「descriptive only」行、w17 只查对打乱的 final3、w21 窗口只查打乱臂、w25 A4.2a 改用 final3、w26 A4.1 的
    vs none 改用 none@2e-3。
- 为什么：现在的实现在这些点上是对的，但下次修改时这些行为没有测试保护；w3 正是重要 1 所在的那条规则。另外，零均值噪声让 selftest_rngs
  的 10/10 变得几乎是必然的——它证明了已知答案不依赖 rng，但不再检验任何噪声下的行为（那部分只有本审查的模拟在做）。
- 改法：加一个打乱 − none 均值略负、噪声大（上界 > 0、均值 < 0）的世界，登记的规则应该印句子；加一个城市 none2 ≠ 0 的 A4.2b 世界；
  加一个只改 prior_on 的窗口篡改；把 main 的 probs 组合抽成函数并测它。
- 置信度：确认（mut_a4_local.out）。

### 次要 4. A4.3 的「refuses nothing」只在 verdict_S 内成立；main 的全局检查仍会被一个无效的描述格子触发
- 位置：readout_ptdom2.py:1522–1525（Rsub 取所有带 cmp 的 run，不看 valid）→ readout_ptdom.py:635–645（subsets_all 对每个 run 做
  sub-read-out 校验，问题进 sprobs，严格模式整份拒绝）；readout_ptdom.integrity 的 earlier_finished 也对所有记录生效。
- 为什么：如果 A none@2e-3 的某个 run 无效、且它的逐题记录本身对不上（例如 table sha 不同，或 -r1 重跑与一个已跑完的旧目录并存），
  严格模式会整份拒绝（Part R 和 S 一起），PROVISIONAL 模式下所有判词变 PROVISIONAL。这是 A2 以来就有的行为，对判词臂是正确的保护；
  只是 A4.3 的「描述格子不影响任何判词」在这个角落不成立。self-test 只测 verdict_S，不走 main。
- 检查：在合成世界里给一个 A none@2e-3 run 植入与快照不同的 table sha，走 main 的 subsets_all 路径（需要 RO.domain_meta，审查里没跑，
  这一条是读代码的结论）。
- 置信度：读代码确认；触发概率低（推测）。

### 次要 5. 读数卫生：wip_a4 的 --allow_partial 也在真实树上跑过，AUDIT 没有记
- 位置：runs/ptdom2/wip_a4/tests/readout_partial_a4.out（00:58 EST，wip 代码，默认根 = 真实 runs/ptdom2/out）；AUDIT.md:664–666 只记了
  06:07 UTC 那一次。
- 结构检查（只数行，没读数字）：两份日志都是 4 个 S 领域「PROVISIONAL (missing runs)」、没有逐 seed 增益行、没有 Traceback、没有
  PROVISIONAL 之外的判词词；但每份都有 4 行「shuffled - none@」描述行，若当时那两个臂 8 个 seed 都齐了，那里会有数字。
- 改法：在 AUDIT 里补记这一次；读数前别打开这两份日志（或移出 tests/）。
- 置信度：确认（文件时间与默认根）。

### 次要 6. 措辞
- 拒绝理由里同一个格子重复两次（例如「B1 none@2e-3 s11: no run; B1 none@2e-3 s11: no run」：国家、人物各报一次），无害。
- 一个先验臂因窗口不符而无效时，--allow_partial 写「PROVISIONAL (missing runs)」，其实是 invalid；完整家族下由 S 来源检查（内部
  problem）导致的 PROVISIONAL 标签写「an outside problem stands」。都不改变判词。
- 置信度：确认（planted_a4_local.out P5 / P6）。

## 逐题回答

### 1. A4.1 / A4.2 的实现与文字是否一致；模拟
- **抽样 / seed / 方向 / 上下界**：三处都用 readout_ptdom.boot_ci（10,000 次、default_rng(0)、百分位 5 / 95）作用在逐 seed 的配对差上
  （pair(...)[0] = 臂 1 − 臂 2，seed 对齐）。A4.1：`boot_ci(gain_final3)[0] > 0` 且 `boot_ci(geo_final3)[0] > 0`，方向真 − none@N、真 − 打乱
  （:692–697）。A4.2a：`boot_ci(shuf_none)[1] >= 0`，方向打乱 − none（:698–706）。A4.2b：`mean(geo_gain) >= 0.5 × mean(gain)`（:707–710）。
  与 AUDIT.md:632–640 的文字逐条一致。
- **所用 none 格子**：A4.1 与 A4.2b 用 none@N（S：F100 的 pick，城市 / 月份 4e-3；R：pick_none）；A4.2a 用 NS_（S：none@2e-3；R：none@N，
  P ≠ N 时描述行标 lr-confounded，理由文字写出 none@lr）。变异 w26（A4.1 改用 none@2e-3）被抓到；w27（A4.2b 改用 none@2e-3）没被抓到
  （次要 3）。
- **R 与 S 的区别**：只在 N_same。R 的打乱 − none 与 PASS 用同一个 none run（模拟里国家 / 人物就是这种结构）；S 的城市 / 月份用另一个
  run（none@2e-3）。规则函数相同。
- **模拟结果**（sim_a4_local.out；P(印 | PASS∧GS) 括号内；无 FLOOR）：

  | 场景 | 国家 | 城市 | 月份 | 人物 |
  |---|---|---|---|---|
  | 零效应 | 0.003 | 0.001 | 0.002 | 0.002 |
  | 纯学得快 面积 +3，final3 0 | 0.021 (0.021) | 0.020 (0.026) | 0.024 | 0.018 |
  | 纯学得快 面积 +5，final3 0 | 0.022 | 0.021 | 0.018 | 0.025 |
  | 大部分学得快 面积 +3.5，final3 +1.3 | 0.236 (0.238) | 0.143 (0.163) | 0.675 | 0.856 |
  | 打乱有害 真 +1.5 / 打乱 −0.9 | 0.269 (0.474) | 0.135 (0.438) | 0.318 (0.422) | 0.111 |
  | 打乱有害 真 +1.5 / 打乱 −1.5 | 0.153 (0.281) | 0.114 (0.309) | 0.099 | 0.002 |
  | 打乱拿走 75% 真 +4 / 打乱 +3 | 0.056 (0.170) | 0.070 (0.344) | 0.017 | 0.001 |
  | 打乱拿走 60% 真 +4 / 打乱 +2.4 | 0.218 (0.332) | 0.209 (0.497) | 0.201 | 0.074 |
  | 剂量不同 真 +5，份额差 0.12 | 0 | 0 | 0 | 0 |
  | 真几何 +3（面积与 final3），打乱 = none | 0.751 (0.772) | 0.531 (0.701) | 0.910 (0.913) | 0.928 |
  | 真几何 +5，打乱 = none | 0.916 (0.916) | 0.887 (0.895) | 0.918 | 0.914 |
  | A 批 none@2e-3 比 none@4e-3 低 ptdom 的差距，打乱 = none@2e-3 − 0.5，真 +3 | 0.669 | 0.610 | 0.690 | 0.477 |

  （「大部分学得快」里人物、月份噪声小，final3 +1.3 是可检出的真实终态差，所以照印；按 F100 的数据这两个领域带 FLOOR，不会印。）
- **真效应是否被过度挡掉**：+3 时 PASS∧GS 之后被扣下的比例国家 23%、城市 30%，其中 A4.1（final3 下界）16% / 20%，A4.2a 9% / 10%；
  +5 时只剩 A4.2a 的约 8–9%（打乱 = none 时百分位 bootstrap 在 n = 8 下约 8.5% 的上界 < 0，名义是 5%）。剂量条件按登记一律挡住
  （份额差 0.12）。这是 A4.1 的代价，可以接受，但应知道 +3 的真实几何效应在城市只有约一半的无条件概率拿到那句话。

### 2. A4.3
- 严格模式与 --allow_partial（planted_a4_local.out P1，真实管道 fake2 → load_all → integrity + series_problems → verdict_S → report）：
  A none@2e-3 s11 缺失 / 无效（data_sha 检查失败）/ 未跑完（40 个 eval）/ 8 个 seed 全缺，四种情形都不拒绝、不崩溃，main 的 integrity
  问题为空（integrity 只看有效 run），城市、月份按 none@4e-3 照常 PASS + GEOMETRY-SPECIFIC，国家、人物不受影响（照印那句话），
  城市、月份那句话被扣下，理由「shuffled - none@2e-3 missing / invalid (none@2e-3 s11: no run)」；需要它的 6 类行
  （A4.3 说明行、shuffled − none、同 lr PASS 行、窗口拆分的 shuffled vs none、double path、shortcut 的 none@2e-3 项）写「missing / invalid」。
- B1 none@2e-3（国家 / 人物的 PASS 臂，N = N_same）：缺失 → 严格模式拒绝「B1 none@2e-3 s11: no run」，不再出现 A3 那条误导的
  「must be ptdom2 runs」；无效 → 「B1 none@2e-3 s11: failed check: data_sha after step 0 = the CPU reference」；--allow_partial 下国家 / 人物
  PROVISIONAL (missing runs)，城市 / 月份 PROVISIONAL（Holm 家族缩小为 2）。A none@4e-3（城市 / 月份的 PASS 臂）无效 → 同样按格子拒绝。
  拒绝理由准确（P2–P4）。唯一的角落见次要 4。

### 3. A4.4
- 窗口不符：先验臂 prior_events 改成 [[916, 1], [7500, 0]]（P5，真臂）、prior_on 在 step 2500 关掉（P6，打乱臂）、prior_on 在 step 7500
  打开（P7）——三种都标成无效，理由以「window (A4.4): the run's prior window differs from WIN」开头并写出差异，严格模式按格子拒绝，
  --allow_partial 正常输出，没有 AssertionError。none 臂不做窗口检查（P8，设计如此；none 有没有被施加先验由 CHECK_PY / identity 管）。
  真实树：28 个 ptdom2 run 全部有效，22 个先验臂 prior_events 全等于 WIN，guard2 在 152 个作业文件上通过（real_struct_local.out）。
- check_s_none：接受植入的 read-out / S_NONE；none pick 不同、prior pick 不同、S_NONE 不同时各返回一条问题（自测 y1 已覆盖，
  我读代码核过）；默认参数仍读真实 runs/ptdom/readout.json 与 M2.S_NONE，main（:1517）把它并入 probs。main 的这一步没有测试（w11）。
- 变异覆盖：A4 新规则 12 个变异被抓到、7 个没被抓到（次要 3）。项目自己的 mutation_readout.out 59 行全 OK（58 个 + 对照），与 AUDIT 一致。

### 4. 判词与回归
- 判词规则未改（见结论）；A3 / A4 差分 16000 行 + 32 个管道领域判词全同（diff_a3_a4_local.out）；零效应误判率约 0.025。
- 修 bug 的副作用：B 的定位方式（:50）和 MANIFEST 改为 M2.R（:250）在正式目录下与 A3 等价；ok_fam 多出的 gate 条件是冗余的
  （gate 为 None 当且仅当真臂或打乱臂缺失，此时 geo_gain 已是 None）；winbad 先于「no check.json / unfinished」判断，对运行中的 run
  （逐 eval 的 prior_on 与截断的 WIN 比较）不会误判。作业文件不调用 readout / select（152 个作业文件里 0 处引用），
  所以 01:02 的替换不影响在跑的作业。未发现回归。
- 测试溯源：rerun_a4.log 的首尾 sha 都是 21797ec55b8f / 78c4a3a757ba，时间 06:04–06:09 UTC 晚于替换（01:02 EST = 06:02 UTC），
  各 *.out 的 mtime 一致；我在副本上重跑自测 82/82。

### 5. 陷阱清单
- 静默失效：未发现会静默改变判词或句子的路径。NaN（例如发散 run 的 final3）在三条新条件里都落到「扣下」一侧（`not NaN > 0`、
  `not NaN >= 0` 为真）。A4.2a 是「举证责任在伤害一侧」的设计选择，不是静默失效，但它在真实噪声下基本不起作用（重要 1）。
- 同名不同义：「shuffled − none」在 S 的城市 / 月份是 none@2e-3，PASS / A4.1 / A4.2b 的「none」是 none@4e-3——A4 的理由文字都写出了 lr，
  A3 审查次要 6 的这一项已修好；「PROVISIONAL (missing runs)」也用于 invalid（次要 6）。AUDIT 的「significant harm」在自测语境下才成立
  （重要 1）。
- 静默过滤样本：未发现。描述格子缺失只影响描述行并被点名；判词臂的缺失 / 无效都拒绝或 PROVISIONAL；窗口不符的 run 变无效并被点名。
- 修 bug 的副作用：未发现回归（见第 4 题）；次要 4 是 A4.3 修复不完整的角落。

## 复现
/scratch/yh6210/tmp_audit_ptdom2_a4/（open_r1 python -B，PYTHONPATH=/scratch/yh6210/llmadd_torch/pylib，PYTHONDONTWRITEBYTECODE=1；
临时合成世界在会话 scratchpad 的 tmp/ 里；本地 nice 19 跑，因为 cpu_short 一直排队）：
- base/、base_a3/：A4 / A3 读数的副本 + 指向真文件的软链；prev/readout_A3_to_A4.diff：A3 → A4 的完整 diff；
- sim_a4.py 2000 → sim_a4_local.out：Monte Carlo（真实 apply_rules / reading_conditions；boot_ci 用缓存索引的等价实现，启动时与原函数
  50 次逐值核对）；
- planted_a4.py（+ planted_a4_lib.py）→ planted_a4_local.out：P0–P8 与模拟的双路核对；
- mut_a4.py → mut_a4_local.out：w1–w27（副本在 mut_local/）；
- diff_a3_a4.py → diff_a3_a4_local.out：A3 / A4 判词差分；
- real_struct.py → real_struct_local.out：真实树的结构检查（不读数值）。
- 另：同样的检查也提交过 cpu_short（作业 22406463 a4mut、22406472 a4sim、22406483 a4plant、22406560 a4real、22407417 a4diff，
  审查结束时仍在排队；它们只写本审查目录，结果若出来会是 *_local.out 的重复，可以取消）。
