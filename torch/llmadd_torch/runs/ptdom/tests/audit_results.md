# ptdom 最终读数的对抗式审查（experiment-audit v，2026-11-07）

- 审查员：全新子 agent，没有参与设计或分析。
- 范围：readout.txt 6aeb5f6f56c0 / readout.json 3caeac06fe8d 的全部判词，以及支撑它们的代码、数据和决策链。
- 只读：除本文件外没有写任何项目文件；没有提交或取消作业；没有写 HOLD 或 picks。
- 检查脚本与日志都在 /scratch/yh6210/tmp_audit_ptdom_results/：
  - extract.py → runs.pkl；recompute.py → recompute_full.out；diff_screen.py；
  - floor_check.py；content_check.py → content_check.out；months_curve.py；comp_curve.py；
  - samelr_end.py；numbers_side.py；sens.py；window_check.py；hits_check.py；readout_rerun.txt/.json；
  - 训练器代码审查子 agent 的脚本在 sub_trainer/。

## 0. 结论

**数字没有算错，问题在判词的含义。**

我用自己写的脚本直接从 run JSON 重算了 24 个领域。area 由 curve 重算，不读 metrics。下面每一项都与 readout.json 逐项一致，0 处不一致：
- stage-1 的 pick、argmax、增益、同 lr 增益、final3、shortcut；
- 7 个 stage-2 领域的决策链（E / C / F / fixB / re-selection）；
- 8-seed 增益、boot90、两半、t95、p、Holm；
- seeds 4–7 的单侧 p 和 Holm；
- FLOOR 线和 CEILING。

用现行代码重跑 readout_ptdom.py，输出与 readout.txt、readout.json 逐字节相同。

但 4 个 PASS 的含义不一样：
- **months**：PASS 只在 prior 窗口内成立。到训练结束，两臂都低于机会水平，final3 增益 −0.7。
- **people**：PASS 落在训练频率捷径（95.4）之下。
- **cities**：主要是学得更快；训练末尾 hard 子集上增益约为 0。
- **countries**：只有它是持续、lr 匹配、并集中在 hard 子集上的增益。

**最可能测错的地方**：months 的 PASS 把"prior 开着时的暂时性超过机会水平"当成了"在预训练里有效"。

## 1. 问题清单（按严重程度排序）

| # | 严重程度 | 问题 | 位置 | 为什么会测错 | 验证检查 | 置信度 |
|---|---|---|---|---|---|---|
| 1 | 重要 | months 的 PASS 是窗口内的暂时效应。终态两臂都低于机会水平（prior 0.8，none 1.5，机会 8.3），final3 增益 −0.69（2/8）。面积增益 +3.90 中，84% 来自 26 个窗口内 eval（+3.27）。prior 臂 held tier S 在 step 2500 峰值 19.2%（8 seed 平均），none 约为 0。 | 主指标定义 AUDIT.md:115（area = 75 个 eval 的均值）；FLOOR 不改变判词 readout_ptdom.py:1180-1183；预注册预期 AUDIT.md:155（"months FLOOR / NOT SHOWN"）；FLOOR 的含义 AUDIT.md:144（"not measurable"） | area 对"窗口内学得快 / 暂时泛化"敏感。两臂都在背训练题，背题使 held 系统性落在机会水平之下（4.4）。PASS 与 FLOOR 同时成立，在预注册语义里自相矛盾：PASS 等于"有效"，FLOOR 等于"不可测"。 | months_curve.py：逐 eval 曲线与窗口前 / 内 / 后的面积分解；comp_curve.py：final3 | 数据确认 |
| 2 | 重要 | shortcut 条件 (iii) 近乎空检验。它只比较 prior 臂 hard 子集的 final3 与捷径基线；none 臂同样大幅超过基线：cities 87.4 对 13.8，countries 82.7 对 11.4，people 76.9 对 67.2，colours 97.8 对 69.6，animal_size 82.1 对 73.5。 | readout_ptdom.py:715-742（shortcut / comp_rows）、:1121、:1165 | 有 anchor 的领域按构造 h3_hard 只有 11–14%，任何训练过的模型都"满足"(iii)。所以"(iii) met"不能说明增益不是捷径造成的。A1.4 的本意是"增益在捷径失效处仍存在"（AUDIT.md:151），那应当看臂间的 hard 子集增益。 | floor_check.py（打印 none 臂的 pooled hard final3） | 读代码 + 数据确认 |
| 3 | 重要 | 面积把"速度"和"终态"混在一起。各 PASS 领域的终态（final3，8 seed，双侧 t）如下。cities 的 PASS 基本是学得更快。 | AUDIT.md:115；readout_ptdom.py:391-393 | 窗口占 75 个 eval 中的 26 个；prior 也加速了 seen 集（months 在 step 2000 时 +32）。 | comp_curve.py；final3 t 检验 | 数据确认 |
| 4 | 重要 | 数字题面积的副作用"vs none@pick"大多为负（months −2.65，cities −4.20，people −5.21，ranks −10.34），但这完全是 lr 效应。同 lr 2e-3 下 24 个领域均值 +0.29，12 个为负，符号检验 p 1.0。 | readout_ptdom.py:1243（numbers_area_vs_none_pick） | none 的数字题面积随 lr 单调上升（A：1e-3 为 9.4，8e-3 为 19.4；B2：1e-3 为 7.7，4e-3 为 21.3），而 prior 的 pick 偏低 lr。 | numbers_side.py | 数据确认 |
| 5 | 重要 | people 的 FLOOR 线是 100.4（> 100），这个标签无条件成立。真正的含义是：训练频率捷径 95.4 高于两臂（94.3 / 92.7）。animal_size（104.6）、ranks（98.8）同理。 | readout_ptdom.py:1177-1183 | 同一个 FLOOR 标签对 months 意味着"低于机会水平"，对顺序类领域意味着"捷径胜过两臂"。对一维 rank target，按训练胜率排序本身就是结构，二者不可分（A3.6）。 | floor_check.py | 读代码 + 数据确认 |
| 6 | 重要（已知，必须写进 FINDINGS） | 没有打乱 target 的剂量匹配对照；比较类领域属 Tier L（prior target 编码了 held 题的答案）。 | AUDIT.md:148-153、:703-707 | held 只从 LM 文本中扣出，没有从 prior target 中扣出。PASS 不是几何特异的：可能只是这些 token 上的额外梯度加快了学习，months 的 seen 加速就是一个迹象。 | 无（设计层） | 读代码确认 |
| 7 | 次要 | gate 度量的是 pre-clip 的梯度范数份额，不是 AdamW 的更新份额。tied embedding（--untie 默认关）下，months 12 个单 token 月份既是 site 也是读出候选，prior 能直接移动输出行。 | pretrain_range.py:514；pretrain_helix.py:70；readout_ptdom.py:166-181 | 全局 a/(a+b) ≥ 0.75 不代表 item 行的更新由 LM 主导；months 的暂时效应与这条直接路线一致。 | 子 agent：tiny 模型实测共享 storage | 读代码确认 / 机制为推测 |
| 8 | 次要 | months 没有消除基线。A4.5 写"months has no elimination line"，但 months 有同样的 Latin-square 结构，我算出 8 个 seed 为 37.5–44.2。 | readout_ptdom.py:2052（`d != "months"`）；AUDIT.md:932 | FLOOR 线 13.3 低估了非几何路线能达到的水平。标签不变（两臂都远低于它）。 | floor_check.py 末段 | 数据确认 |
| 9 | 次要 | 共享对照：months 与 cities 共用 A 的 none@4e-3；countries、animal_size、people 共用 B1 的 none@2e-3。cities 当初进 stage 2 靠 seed 2 的 +0.06 擦边。 | readout_ptdom.py:1099-1103 | Holm 对相关性有效，但"4 个 PASS"不是 4 次独立确认。 | sens.py；recompute_stage1.out | 数据确认 |
| 10 | 次要 | seeds 4–7 的检验只有 4 个点（df 3，单侧 t）。4/4 正号的符号检验最小 p 为 1/16，PASS 依赖正态假设。 | readout_ptdom.py:961-970 | 重尾时 p 偏小。A4.3 的 t3 重尾模拟显示校准尚可。 | — | 读代码确认 |
| 11 | 次要 | coordinator log 写"~17:00 EST B1 stage 2 complete, select --after 2"，但 picks_B1_after2.json 的 mtime 是 19:42。B1 stage 2 最后一个 check.json 在 17:52，2b 的 SUBMITTED_ 标记在 19:42–19:44。 | AUDIT.md:1473 | 只是时间记录不准，顺序正确。 | ls --time-style=full-iso | 数据确认 |
| 12 | 次要 | picks consistency 只比较 C / E / F / ADVANCE / amendment，不比较各单元的 pick 本身。 | readout_ptdom.py:1056-1057 | pick 本身不一致不会被这一步发现。我另做了比对：9 个文件全部一致。 | 见 1.5 节 | 读代码 + 数据确认 |
| 13 | 次要 | 同一 seed、同一 lr、prior 尚未开启时，两臂的 step-0 读出相差 1 行（例如 A s0 months：0.0991 对 0.1038），lm0 却相同。eval@0 是在第 0 步更新之后。 | — | GPU 非确定性。"配对"是数据和初始化配对，不是逐位配对；噪声已计入 seed 方差。 | 见第 4 节 | 数据确认 |
| 14 | 次要 | B1 里 countries / country_size 共用 70 个 item，animals / animal_size 共用 65 个。各自 held pair 中有 20–50% 是兄弟领域的训练 option pair（关系不同）。 | 子 agent 8(a) | 不泄漏答案；但"held pair 从未共现"只在单领域内成立，兄弟领域的 prior 也会移动同一批行。 | sub_trainer/check_sibling.py | 读代码 + 数据确认 |
| 15 | 次要 | months 的 tier E 在 best-vs-best 下 final3 −8.6（2/8），在同 lr 2e-3 下 +11.1（7/8）。 | — | 强 lr 依赖，只能作描述，不能写成伤害或帮助。 | samelr_end.py | 数据确认 |

各 PASS 领域的终态（第 3 条用）：
- months：−0.69（2/8）。
- cities：+1.31（6/8，p 0.33）；hard 子集 +0.13（5/8，p 0.97）。
- countries：+3.42（8/8，p 0.014）；hard 子集 +6.61（7/8，p 0.037）。
- people：+1.16（7/8，p 0.018）；hard 子集 +2.97（5/8，p 0.39）。

**阻断：未发现。** 没有数字或判词按预注册规则算错，也没有数据或流程问题使判词失效。第 1–6 条约束的是 FINDINGS 能怎么写，不影响判词本身。

## 2. 独立重算（问题 1）

### 2.1 有效性

- 磁盘上 386 个 run 目录 = 384 个 cell，外加 B3 none@2e-3 s0、none@4e-3 s0 的首投目录。这两个首投目录没有 run JSON（训练前 CUDA 故障），读出用的是 -r1。
- 所有 check.json 里只有 3 个失败项，全是 geo 检查，即那 3 个 A6 run。没有 CKA-rise 失败，也没有 DP 容差失败。
- 按我自己实现的 A5.1 / A6.1 规则，384 个 cell 全部有效。

### 2.2 area 与逐题 hits

- 我从 curve 重算的 area 等于 metrics 的 area（容差 1e-6）。
- 逐题 hits 串重算的准确率与记录值在 172,800 个（run, 领域, eval）序列上 0 处不符。

### 2.3 stage 1：24 个领域全量

下面各项都与 readout 一致（diff_screen.py 输出 "none"）：
- pick 与 argmax（top2 + in-training bpb 破平局）；
- seeds 0–3 的配对增益和 2e-3 同 lr 增益；
- final3；
- 判定；
- (iii) 的 pooled n、prior 值和 baseline 值。

17 个 NOT ADVANCE 领域全部复核，不只抽查 6 个，原因也一致：
- stars：3/4（s0 为 −8.84）；
- landmarks：2/4；
- languages：3/4，同 lr 也是 3/4；
- animals：2/4；
- country_size：同 lr 2/4；
- units：3/4；
- zodiac：2/4；
- compass：3/4，同 lr 2/4；
- planets：3/4，同 lr 2/4；
- elements：2/4；
- prefixes：3/4；
- chinese_zodiac：3/4；
- kinship：3/4；
- life_stages：2/4；
- geo_periods：2/4，同 lr 1/4；
- greek：3/4，同 lr 3/4；
- scalar_adj：3/4。

aux gate：我自己实现的 55 个 cell 中位数与 readout 全等，全部通过。

### 2.4 stage 2：7 个领域

| 领域 | pick（prior / none） | 增益 | 8-seed 的 p（Holm） | seeds 4–7 的 p（Holm） | 判词 |
|---|---|---|---|---|---|
| months | 2e-3 / 4e-3 | +3.90 | 0.0014 | 0.0220 | PASS [FLOOR 13.3] |
| colours | 2e-3 / 2e-3 | +2.06 | 0.0081 | 0.1069 | NOT SHOWN（seed 5 −0.21） |
| cities | 2e-3 / 4e-3 | +3.48 | 0.0217 | 0.0399 | PASS |
| countries | 2e-3 / 2e-3 | +3.85 | 0.0054 | 0.0355 | PASS |
| animal_size | 2e-3 / 2e-3 | +0.49 | 0.0054 | 0.1069 | NOT SHOWN [FLOOR, CEILING] |
| people | 2e-3 / 2e-3 | +1.54 | 0.0079 | 0.0399 | PASS [FLOOR 100.4] |
| ranks | 1e-3 / 4e-3（fixB 1e-3） | +5.94 | 0.0217 | 0.1069 | NOT SHOWN（seed 7 −1.91）[FLOOR] |

**不一致：未发现。**

## 3. 选 lr / 扩展 / fixB 与时间顺序（问题 2）

### 3.1 规则执行

- 'top2' + bpb 破平局按 AUDIT 执行。stage 1 没有 offline bpb，所以用 in-training bpb（A4.5）。
- 各领域的扩展与重选（与 A4.2 / A5.2 的文字一致）：
  - months：prior 1e-3 → 5e-4；
  - colours：prior → 4e-3，none 2e-3 → 1e-3；
  - cities：prior → 4e-3；
  - countries：prior → 5e-4，none → 1e-3；
  - animal_size：none → 1e-3 → 5e-4。第二步由 1e-3 的 97.4497 对 2e-3 的 97.4486 这 0.001 分的 argmax 触发，没有改变 pick。
  - people：prior → 4e-3，none → 1e-3；
  - ranks：prior → 5e-4，none → 1.6e-2，fixB none@1e-3 之后重选，pick 4e-3，argmax 8e-3。
- 7 个领域都不是 NOT FINAL，也没有 EDGE。
- 9 个 picks 文件的 pick 与我的独立 pick 一致。
- 要求运行的 cell 共 384 个，与磁盘上的完全相同。

### 3.2 时间顺序（文件 mtime，EST）

| 时间 | 事件 |
|---|---|
| 11-04 09:51–10:20 | A5 代码 |
| 11-04 11:57 | 提交 stage 1 |
| 11-04 16:13 | 用户选 option B |
| 11-06 00:49 | picks_A_after1 / picks_B1_after1（A5 代码） |
| 11-06 01:03 | A6 readout |
| 11-06 01:59 | A6.1 readout（1e2f17522e47，此后未改；select d851f9822efc 于 01:07，此后未改） |
| 11-06 15:10 | picks_B2_after1 / picks_B3_after1 |
| 11-06 15:28 | partial readout |
| 11-06 19:42 | picks_B1_after2 |
| 11-07 12:56–12:59 | picks_B1_after3、picks_B2_after2；partial readout |
| 11-07 21:39 | picks_B2_after3 |
| 11-07 21:40 | 最终 readout |

- partial readout 之后没有规则或代码变更。
- A6 的规则只依赖 geo、cka 和 grad_prior 三个字段。它放进来的 3 个 run 都不影响判词：
  - planets 和 geo_periods 本来就 NOT ADVANCE；
  - ranks@5e-4 s5 是扩展 cell 的 confirm seed，不进选择，也不进判词。
- **看了结果后才改规则或才做选择：未发现。**

## 4. 有效性（问题 3）与读出指标（问题 4）

### 4.1 A6 的三个 run 合规

我自己检查了每个 run 的 overshoot eval：

| run | geo | cka |
|---|---|---|
| planets@1e-3 s2 | −0.001147 | 1.000142 |
| ranks@5e-4 s5 | −0.001083 | 0.999498 |
| geo_periods@2e-3 s3 | −0.001129 | 0.999589 |

- 三个都落在 [−0.005, −0.001] × [0.995, 1.01] 内。
- 所有 on-eval 上 grad_prior 有限且 > 0。
- 没有 inactive 字段；numbers 的 geo 恒为 0。

### 4.2 其他有效性检查

- A5.1 标记：0 个。
- 静默丢弃 / 重复：未发现（asked = disk = valid = 384；每个 run 都 assert 了 args.seed、lr、tag、prior_domains、out 与目录名一致）。
- -r 目录：只有 B3 none s0 的两个，用对了。
- data_sha：同批同 seed 的各臂一致。
- code_sha：唯一，d87a1502cee8。

### 4.3 split 按内容核对

比较类领域（A、B1、B2 用 seeds 0–7，B3 用 0–3），以 ft36 题目全集加 run JSON 的 eval_held.q / seen.q 为准：
- eval_held = held key 的全部题；
- n_train 正确；
- seen ⊂ 训练池；
- held 的 (anchor, 无序 pair) 内容 ∩ 训练内容 = ∅，prompt 字符串也不相交；
- eval_cls 与 key 一一对应；
- answer-frequency / shorter / hard mask 与 trainer 一致。

months：seen 集就是全部 118 道训练题；held 的 53 个类 ∩ 训练的 79 个类 = 0（8 个 seed 都是）。

子 agent 按 dom_rng 重放了满 18,310 步的训练行采样：每一行都来自训练池，训练池里每道题都被抽到过。

**经由 LM 文本的泄漏：未发现。**（经由 prior target 的"设计内泄漏"即 Tier L，见第 1 节第 6 条。）

### 4.4 读出位置与打分

数据层面的已知答案：
- 训练末尾 seen 两臂都约为 100%；
- step 0 时 months k-span 约为 9%，等于机会水平；
- kspan 面积与 first-token 面积基本一致；
- ties 很少；
- EM > first-token k-way 的 191 个 eval 全是 bf16 平局（kway_em_bad = 0）。

代码层面（子 agent）：
- 对齐：span_logp（pretrain_range.py:1692-1697）从 answer 前一位起对齐。
- 判定：2-way 用严格大于（:1870）；k-span 是 own > max(others)（:1847-1849），候选互不为前缀（:720）。
- 双路径：eval 0 时差异 ≤ 0.03 nat；若有错位，差异会是约 11 nat。

**未发现位置错误。**

### 4.5 FLOOR 线

| 领域 | FLOOR 线 | 构成 |
|---|---|---|
| months | 13.33 | max(机会, shortest-span) + 5 |
| colours | 89.25 | |
| cities | 74.71 | |
| countries | 78.26 | |
| animal_size | 104.58 | |
| people | 100.42 | |
| ranks | 98.75 | |

都与 readout 一致。问题见第 1 节第 5、8 条。

### 4.6 months 两臂都低于机会水平时的 PASS 意味着什么

- **背题机制**：held tier S 类的答案，永远不是同一起点、同一 shift 下的训练答案。两臂到训练末尾都把训练题背到 100%，所以会系统性避开正确答案，落到机会水平之下。
- **只在窗口内超过机会水平**：prior 臂在 step 1500 / 2000 / 2500 分别为 11.3% / 14.7% / 19.2%；窗口末（7250）6.1%，step 10000 时 2.7%，终态 0.8%。
- **几何保留了但没被用上**：prior 装进去的几何一直在，终态 eval CKA 为 0.969（none 0.717）；但窗口关闭后，读出被背题压过。

所以这个 PASS 应写成："prior 开着时，held 的 k 步题暂时超过机会水平；窗口之后消失"。

## 5. gate 与先验施加范围（问题 5）

gate 按 cell 计算并通过：
- 7 个 pick 的逐 seed 中位数都 ≥ 0.749，且 seeds 0–3 和 4–7 各自通过；
- b > a 只出现在 months（208 个 eval 中有 4 个，单点最低 0.355）。

384 个 run 在数据层面全部满足下面各项：
- prior_events == [[916,1],[7324,0]]；
- prior_on 恰好在 916 ≤ step ≤ 7323 的 26 个 eval 上为 1；
- 窗口外 grad_prior 为 None 或 0；
- on-eval 上 grad_prior 等于 gnorm_prior_<active>（λ = 1）；
- gnorm_prior_* / geo_* 只出现在 active 领域；
- lam_domains 只有 active 为 1；
- none 臂没有任何 prior 字段。

代码层面（子 agent）：
- 窗口条件在 :2062；窗口外 loss 只有 LM 项（:2362-2364）。
- site 只来自 active 领域自己的行（:1608-1609、:2186-2194），并且只在 prompt 里，不在被打分的 answer span 上（:738-743、:1943）。
- inactive 领域只做诊断性的 autograd.grad，不回传。
- 分解与 backward 用同一个 batch、同一张图，pre-clip（:2195-2234、:2399）。

**问题：未发现。** 但 gate 度量的是什么有一个局限，见第 1 节第 7 条。

## 6. 哪些结论可能来自记忆、泄漏、lr 混杂或偶然；FINDINGS 能写 / 不得写（问题 6）

### 6.1 各类来源

- **记忆**：months 是背题动态（第 1 节第 1 条）；比较类领域不是纯背题（seen 约 100%，held 87–95%，内容核对无泄漏）。
- **泄漏**：经由 LM 文本的泄漏没有；经由 target 的有，即 Tier L（第 6 条）。
- **lr 混杂**：
  - best-vs-best 的 pick 中，months、cities、ranks 不同 lr；countries、people、colours、animal_size 同 lr。
  - 同 lr 增益方向一致：

    | 领域 | 同 lr 增益 |
    |---|---|
    | months | @1e-3 +7.0（4/4）；@2e-3 +4.7（8/8） |
    | cities | @2e-3 +4.8（8/8） |
    | countries | @5e-4 +8.2（8/8）；@1e-3 +6.1（4/4）；@2e-3 +3.8（8/8） |
    | people | @2e-3 +1.5（8/8）；@4e-3 +1.8（8/8） |
    | ranks | @1e-3 +7.7（7/8） |

  - 所以判词的方向不是 lr 造成的。数字题副作用则是 lr 造成的（第 4 条）。
  - 按用户 11-04 的决定，FINDINGS 要带上 A5 审计在"区间内有 lr 最优、两臂相同"情形下模拟的 P(PASS | advanced) = 0.07–0.08（论文不写）。
- **偶然**：4 个 PASS 的 seeds 4–7 Holm p 为 0.022–0.040，SF_ALPHA 取 0.05 也都过。第 9、10 条是需要的限定。
- **oracle 敏感性**：在 seeds 4–7 上把 none 换成它自己表现最好的 lr，countries 降为 +2.7（3/4，p 0.14），其余三个仍然 4/4。

### 6.2 能写

- 预注册判词原样记录（PASS 4 个，其中 2 个带 FLOOR；NOT SHOWN 3 个；17 个 NOT ADVANCE），并附带三条限定：
  - 没有打乱 target 的对照，所以不是几何特异的；
  - 比较类领域属 Tier L；
  - P(PASS | advanced) 为 0.07–0.08。
- **countries（最干净）**：在 53M 预训练中，sphere prior 提高了 countries 的 held 2-way，同 lr、持续到终态、集中在 hard 子集。数字：面积 +3.85（8/8），final3 +3.4（8/8），hard final3 +6.6（7/8），三个同 lr 对比全为正。这是 Tier L 的结论。
- **cities**：面积增益确认，但主要是学得更快；终态 +1.3（n.s.），hard 子集终态约为 0。
- **people**：+1.5，两臂都在训练频率捷径（95.4）之下，FLOOR，Tier L。
- **months**：prior 开着时，held tier S 暂时超过机会水平（峰值约 19%，none 约为 0），窗口之后消失，两臂终态都低于机会水平；面积 PASS 有 84% 来自窗口内。

### 6.3 不得写

- 不加限定地写"这 4 个流形先验在 53M 预训练里有帮助 / 被证明有效"。
- 写 months 的 prior 改善了月份推理，或任何关于 months 终态的说法；或把 months 当作 FLOOR 之上可测的效应。
- 以 (iii) 为证据写"增益不是捷径造成的"。
- 写"几何结构本身起作用"，或"模型发现了结构"。
- 写"NOT ADVANCE 的领域 prior 无效"：stars、compass、scalar_adj、prefixes、landmarks 的同 lr 增益为 +2.7 到 +10.7。
- 写"数字题面积下降"。最多只能写"lr 匹配下未见一致的数字题副作用"。

### 6.4 "Tier L"与"FLOOR"如何限定

- **Tier L**：每行都写"target 编码了 held 答案：给了结构，模型会用"。
- **FLOOR**：要分情况写：
  - months："两臂低于机会水平：背题主导"；
  - 顺序类领域："训练频率捷径高于两臂，线可以 > 100"。

## 7. 陷阱清单（问题 7）

1. **训练 / 评测分布不一致**：未发现会导致测错的不一致。
   - 两边用同一个 DLINE 模板；评测把问题放在 slot 1 行尾，前面是 val.bin 文本和一个 numbers 行。
   - 与训练的位置差约 10–25 个 token。
   - 评测里的 numbers 行按训练分布抽取，不含领域信息。
2. **记忆代替泛化**：months 是（第 1 条，重要）。比较类领域：未发现。
3. **静默失效的损失 / 梯度**：未发现。
   - 所有 on-eval 上 grad_prior > 0；
   - eval CKA 在窗口末尾 prior 臂 0.992–1.000，none 臂 0.20–0.68；终态 prior 臂仍有 0.88–0.97；
   - λ = 1 已核对。
4. **单 seed 偶然**：判词层面未发现。
   - 规则用 8 个 seed，其中 4 个不参与选择。
   - cities 当初进 stage 2 是擦边（第 9 条）。
   - colours / ranks 的 NOT SHOWN 各由一个 seed 决定，属正常判词。
5. **读错 token 位置**：未发现（数据层面和代码层面都查过，见 4.4）。
6. **同名不同义**：
   - FLOOR（第 5 条）；
   - area = 速度 + 终态（第 3 条）；
   - LM share：readout 里是 a/(a+b)，trainer 字段 grad_lm_share 是 a/|g_T|；
   - PASS 的字面含义"shown effective"不等于终态有效；
   - 两个 wrap mask 定义不同（:974 对 :1010），只影响描述性子集。
7. **静默过滤样本**：未发现。
   - 读出覆盖全部 held 题的两种顺序。
   - prior 在 site 不足时跳过：planets 在 4 个 run 里施加了 6406 或 6407 步（共 6408），有日志记录，满足 ≥ 95%。
8. **修 bug 的副作用**：未发现。
   - A3 修 'geo_' 前缀、A4 改用 full-span 读出，都在任何 run 之前。
   - A6 / A6.1 不影响任何判词。
   - picks_A/B1_after1 可以复现。

**最可能测错的地方**：months 的 PASS 把"prior 窗口内暂时超过机会水平"计成了"在预训练里有效"。到训练结束，两臂都低于机会水平，增益为 −0.7。
