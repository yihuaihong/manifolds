# ptdom Amendment A3 对抗审查（全新审查者，只读，2026-11-04）

**范围**：AUDIT.md 全文（重点 A3.1–A3.11）；make_jobs_ptdom.py（含 CHECK_PY）；runs/ptdom/{readout_ptdom.py, select_ptdom.py, aux_gate_check.py, submit_ptdom.sh}；240 个 stage-1 作业文件；worktree llmadd-burst-ptdom 的 Rev F 改动（`git diff 21cdc14 1208bad`，1 个文件 +157/−20，未推送，工作区 tools/ 无未提交改动）；runs/ptdom/code 快照（COMMIT 1208bad，pretrain_range.py e4f1c92482d2）；tests/a3/ 下的测试与日志；之前的 audit_build.md / audit_a1.md / audit_a2.md。

**纪律**：只读。除本文件外没有改任何项目文件，没有提交、没有取消作业。我自己的检查脚本都在 session 的 scratchpad（`/state/partition1/job-20790752/claude-4550886/-home-yh6210/74cbe2e4-1709-4a6a-b82e-817d33fbdd73/scratchpad/`：mc_a3.py、mc_a3_b.py、mc_a3_fix.py、cx_a3.py、cx_halves.py、geok_tamper.py、sites.py、tie_probe.py）。scratchpad 是临时目录，所以下面写全了方法和数字。这些脚本只 import 项目代码，在临时目录里造假 run，不写 runs/ptdom。

**总判断**：
- 数据、切分、答案、位点、各臂数据一致、批次 A 的回归、GPU 类型的隔离，我都逐项核对过，**没有发现错误**。
- 有 **1 个阻断项**：k-way 的已知答案检查在 GPU bf16 下很可能被 logit 并列触发。触发后 run 被判无效，整批的选点卡死，而且没有预注册的出路。
- 有 **4 个重要项**：
  - lr 范围不对称，把 A1.1 修掉的 lr 混淆又带了回来；
  - 严格读数会静默跳过没有 run 的批次；
  - 零假设模拟的噪声尺度偏小，给出的错判率偏乐观；
  - 判词条件在已晋级的前提下偏宽松（反保守）。
- 其余是次要项。

---

## 问题清单（按严重程度排序）

### 阻断 1：k-way 已知答案检查（整个 run ≤ 5 次违例）从没遇到过真正的命中。在 GPU bf16 下，logit 并列很可能让它失败，run 变成无效，整批无法选点
- **位置**：
  - make_jobs_ptdom.py:322-323（`kb <= 5`，对所有 `kway_em_bad_*` 在 75 次 eval 上求和）；
  - 快照 code/tools/pretrain_range.py:1736（exact match 用整个词表的 argmax）和 1749-1753、1765（k-way 用候选内的 argmax，计数 `kway_em_bad`）；
  - readout_ptdom.py:497-510（`select_unit` 要求每个 seed 都有 valid run，否则 "selection refused"）；
  - make_jobs_ptdom.py:647-649（已跑完的 run 永不重跑，检查失败也一样）。
- **为什么**：
  - eval 在 bf16 autocast 下做，logit 不上转（transformers 4.57.6 的 Olmo2ForCausalLM 带 `logits_to_keep` 时不转 float，我查过源码）。
  - 答案 token 和另一个候选的 bf16 logit 完全相等时，两个 argmax 的平局规则不同：
    - 词表 argmax 取 token id 小的那个；
    - 候选 argmax 取条目顺序靠前的那个。
  - 所以只要答案的 id 更小、而对手在条目顺序里更靠前，exact match 记为命中、k-way 记为未命中，算一次"违例"。这不是 bug，只是平局。
  - 阈值 5 是对整个 run 所有 k-way 集合（held、seen，months 还有 heldE 和 4 种格式）的 75 次 eval 求和。
- **证据**：
  1. **这个检查从没真正检验过。** CPU smoke 里所有 k-step 集合（B2z 的 zodiac / planets / elements，B3 的 chinese_zodiac / geo_periods / greek，两臂都算）在每次 eval 的 exact match 都是 0，`kway_em_bad` 当然是 0。
  2. **并列的实测频率。** 我在本项目一个训练好的 53M 模型上测过（ptheld2t `53m-lr1.6e-2-7pct-h54vf-none` seed 0；同样的 build_model，bf16 autocast，`logits_to_keep=128`；8192 个 val 位置；脚本 tie_probe.py）：
     - 前两名 logit 在 bf16 下**完全相等的位置占 1.76%**；
     - 在"不确定"的位置（top-1 概率 0.2–0.8）占 0.79%；
     - 最大 logit 的中位数 7.2（bf16 在 [4, 8) 的格距是 1/32）。
  3. **粗估。** B2 每次 eval 的 k-step held 约 490 行（zodiac 216、planets 48、elements 228）× 75 次 eval ≈ 3.7 万行次。设一半行次答案在前两名里且模型不确定，乘以 0.8% 的并列率，再乘以约 1/4 的"平局方向不一致"概率，**每个 run 约 35 次违例，远超 5**。即使我高估了 10 倍，也在阈值附近。
- **后果**：
  - CHECK 失败（不是只差 CKA 那一项）→ run 无效。
  - 下游 `select_unit` 拒绝选点，`rerun_plan` 又拒绝重跑已完成的 run。B2、B3 很可能还有批次 A 的 months，整批的 stage 1（约 156 L40S-h + 36 B200-h）跑完之后会卡死。
  - 唯一出路是看过数据之后再改一个预注册的检查。
- **可验证的检查**：
  - (a) 提交前在 CPU 上用 bf16 autocast，对一个训练过的模型构造 k-way 行，统计 `kway_em_bad`；
  - (b) 或者至少在 B2 第一个完成的作业的 check.json 里先看 `kway_em_bad` 的总数，再放其余作业。
- **修法（便宜，提交前做）**：
  - **只改 CHECK_PY（不动 trainer）**：把这一项改成 info，并在 loader 里像 CKA-only 失败那样容忍。CHECK_PY 是作业文本的一部分，所以 240 个作业文件要重渲染。
  - **或者改 trainer**：只在"某个候选的 logit 严格大于答案的 logit"时才计违例，这才是严格的已知答案。代价是新 commit，加上回归和快照重做。
- **置信度**：这个检查没被检验过：高。会出现并列：高（实测）。每个 run 超过 5 次：中。

### 重要 2：lr 可达范围不对称（先验臂 5e-4..4e-3，none 臂 1e-3..8e-3），A1.1 修掉的 lr 混淆又回来了。纯 lr 效应可以拿到 PASS，只多一个描述性的 EDGE 标签
- **位置**：AUDIT.md:666-670（A3.4）；make_jobs_ptdom.py:82-83（GRID1 / EXT_LR）；readout_ptdom.py:522-528（extension）和 684-689（rule），EDGE 只是标签（813）。
- **为什么**：
  - stage 1 时，先验臂能用到 1e-3，none 最低只有 2e-3。到 stage 2，先验臂还能再下探到 5e-4，none 只能到 1e-3。
  - 如果读数对两臂一样、随 lr 降低单调变好，"最好对最好"就会让先验臂多占一个 lr 档。
  - A1 审查的重要 1 正是这个问题。A1.1 的修法是 none 的网格覆盖 1e-3，并可外扩到 5e-4，A3 把它去掉了。
- **构造的反例（cx_a3.py，用真实的 screen / after2 / readout）**：
  - 设定：B2 的 compass，两臂完全相同的 lr 曲线 `area = 70 + 2.5 × (−log2(lr / 1e-3))`，**没有任何先验效应**。
  - 走完 stage 1 → 2 → 2b：
    - stage 1：ADVANCE（每个 seed 增益 +2.35）；
    - 外扩：先验到 5e-4，none 到 1e-3，两边都 moved、都是 EDGE；
    - 最终 only-compass@5e-4 对 none@1e-3：**PASS**（p ≈ 0），tags 只有 `EDGE (prior, none ...)`。
  - 同 lr 的对照数据其实就在盘上：none@1e-3 和 only-compass@1e-3 各 8 个 seed，增益约为 0。但判词不用它。
- **模拟（mc_a3.py / mc_a3_b.py；真实的 best_lr、extension、rule；无先验效应，两臂共用同一条 lr 趋势；σ 是每个 run 的噪声，单位为面积分）**：

  | σ | 每降一档 lr 的趋势 | P(ADVANCE) | P(PASS)（原始 p） |
  |---|---|---|---|
  | 5 | 0 | 0.028 | 0.0055 |
  | 5 | +0.5 | 0.041 | 0.0073 |
  | 5 | +1 | 0.060 | 0.014 |
  | 5 | +3 | 0.181 | 0.063 |
  | 3 | +1 | 0.109 | 0.027 |
  | 2 | +1 | 0.171 | 0.051 |
  | 1 | +1 | 0.36 | 0.17 |

  - 反方向的趋势（高 lr 更好）是保守的：σ 3、趋势 −1 时 P(PASS) = 0.0003。
- **修法（要在任何结果出来之前预注册；不影响 stage-1 作业文件）**：
  - **F1**：none 的最终 argmax 落在它到过的最低 lr（下沿 EDGE）时，判 NOT FINAL / lr-confounded，不判 PASS。
  - **F2**：此时 none 再外扩到 5e-4。
  - mc_a3_fix.py 的结果：σ 3、趋势 +2 时，P(PASS) 从 0.081 降到 0.054（F1）/ 0.061（F2）；σ 1、趋势 +1 时，从 0.17 降到 0.036 / 0.070。两种都只部分修复。
  - 建议再加一条同 lr 的读数增益：stage 1 两臂都跑了 2e-3、seeds 0-3，每个领域都有，现成免费。至少打印出来，最好作为 PASS 的附加条件。
- **置信度**：机制和反例：高。趋势在真实数据里有多常见：未知（rb6 里 none 是高 lr 更好，那是保守方向）。

### 重要 3：严格模式的读数静默跳过盘上没有 run 的批次。Holm 族变小，判词却不标 PROVISIONAL
- **位置**：readout_ptdom.py:710-712（`if not any(k[0] == b for k in R_): continue`）；main 1320（批次参数用默认的四个）；AUDIT.md:676-681（"Holm-adjusted over ALL advancing domains of ALL batches together"）。
- **构造的反例（cx_a3.py）**：盘上只有 B2。`readout(R_, subs)`（与 main() 同样调用，严格模式）**正常结束**：batches_read ['B2']，Holm family 1，integrity 无问题，compass 得到最终 PASS。报告里只有一行 "batches read"，没有任何警告。
- **为什么要紧**：
  - 例如 L40S 排队，A 或 B3 还没跑，读数就会用更小的 Holm 族给出"最终"判词；之后补上批次，判词可能翻转。
  - 只有某个批次 stage 1 跑了一部分时，`screen` 会因为缺 run 报错，这一点是对的。缺的是"整个批次不在"这个情形。
- **修法**：严格模式要求四个批次都完成 stage 1，并且每个批次要么没有晋级的领域，要么它的 stage 2/2b 齐全；否则拒绝，或者整体标 PROVISIONAL。
- **置信度**：高（已复现）。

### 重要 4：零假设模拟的噪声尺度偏小（audit_a2 重要 1 的同一个错误又出现了）。在"已晋级"的前提下，8 seed 判词是反保守的
- **位置**：tests/a3/a3_null_mc.py:35-39（每个 run 的噪声 sd 为 1 分；rho 只是加在同一 seed 上的共同效应）；AUDIT.md:661-663（"0.036 … ≤ 0.0037 … about 0.09 false PASS"）。
- **为什么**：
  - TIE = 2 分是绝对值。每个 run 的 sd 为 1 时，4 seed 均值的 sd 只有 0.5，前两名几乎总在 2 分以内，选点基本由随机的 bpb 打破平局决定，选点偏差被低估了。
  - 改变 rho 不会有任何效果：同一 seed 的共同效应在配对增益和选点里都会消掉。三行输出完全相同，这一点只是"看起来做过稳健性检查"。
  - audit_a2 测过 53M 的 4 seed 均值 sd，大约 2–3 分，对应每个 run 的 σ ≈ 4–6 分。
- **我的重跑（mc_a3.py，平坦零假设）**：

  | σ | P(ADVANCE) | P(PASS) | P(PASS \| 已晋级) | 已晋级时 seeds 0-3 的平均增益 | seeds 4-7 |
  |---|---|---|---|---|---|
  | 1 | 0.028 | 0.0022 | 0.081 | +0.57 | −0.07 |
  | 3 | 0.029 | 0.0043 | 0.147 | +2.4 | −0.17 |
  | 5 | 0.028 | 0.0055 | 0.196 | +4.2 | +0.09 |
  | 10 | 0.024 | 0.0040 | 0.170 | +8.1 | +0.54 |

  - 已晋级时，seeds 0-3 的增益被抬高约 0.8σ，seeds 4-7 约为 0。
  - 名义水平是 0.05，"已晋级"条件下的原始 p 错判率是它的 3–4 倍。Holm 会把它压下来，但 AUDIT 写的数字只在 σ = 1 时成立。
- **具体例子（cx_halves.py，用 `rule()`）**：增益 [6, 5, 7, 6 | 0.1, 0.2, 0.1, 0.3] → p 0.028，halves [+6.0, +0.175]，在 Holm 族为 1 时判 **PASS**。判词几乎全由被选过的 seeds 0-3 撑起来，未被选的一半只有 +0.2。
- **建议**：
  - 把 A3.3 的数字按 σ 3–5 改写，并告知用户。
  - 是否加一条只看 seeds 4-7 的条件（比如 seeds 4-7 的均值 ≥ seeds 0-3 均值的某个比例，或者 seeds 4-7 单独的单边检验），交用户决定。
  - 现在 seeds 4-7 只打印，不进判据。
- **置信度**：模拟本身高（用的是真实的选点和判词函数）；真实 σ 不确定，所以具体倍数为中。

---

### 次要 5：k-way 只比较首个 token。它和 F81 / ft36 的整段打分不是一回事（同名不同义），而且有些首 token 是通用 token
- **位置**：pretrain_range.py:692-695（`KWAY` = `' <item>\n'` 的首 token）；ft36.py 的文档注释（ft36 对 `" <answer>\n"` 求和，专门防前缀问题）。
- **事实（sites.py，用 tokenizer 重算）**：
  - zodiac：Aries → `' A'`，Taurus → `' T'`；
  - geo_periods：Cretaceous → `' C'`，Neogene → `' Ne'`，Quaternary → `' Qu'`；
  - elements：beryllium → `' b'`，krypton → `' k'`，vanadium → `' van'`，germanium → `' german'`；
  - greek：zeta → `' z'`，omicron → `' om'`；
  - chinese_zodiac：Rooster → `' Ro'`。
- **影响**：
  - 这些领域是按 F81（整段打分）的阳性挑出来的，这里换成了首 token 准确率。
  - 单 token 的条目（Gemini、Leo、Rat 等）的 k-way 候选就是先验位点所在的 token。词嵌入是 tied 的，先验会直接改动这些候选的输出行；多 token 的条目则不会。所以先验对读数的影响依条目而异。
  - 两臂一样，不构成增益的混淆，但解释时要叫"首 token k-way 准确率"。
- **建议**：判词行写明这一点。可选：在少量行上加一个整段打分的双路，作描述用。
- **置信度**：高（事实）；对结论的影响为低到中。

### 次要 6：其他只在 GPU 上才会遇到的容差检查也从没在 GPU 上跑过，而且出口和阻断 1 一样是死胡同
- **位置**：
  - step-0 lm ≤ 1e-3（make_jobs_ptdom.py:293，A1 次要 12：只在 A100 上验证过）；
  - 2-way 双路 < 0.25 nats，在所有 eval × 领域 × 集合上取最大值（319-321，Rev E 从没上过 GPU）；
  - 分解双路的最大值 < 0.05（331-334，B200 上没验证过）。
- **为什么**：任何一项在已完成的 run 里系统性失败，都会像阻断 1 那样卡住选点（select_unit 要求 valid，rerun_plan 拒绝重跑）。
- **建议**：提交前预注册"只差 GPU 标定类容差的 run 怎么处理"（例如像 CKA-only 那样算 valid 数据，或者允许换名重跑），并在第一个完成的 B200 / L40S 作业上先看 check.json。
- **置信度**：中。

### 次要 7：捷径条件 (iii) 可能只由 1 个 seed、2 道题决定；顺序比较类实际上无法检验
- **位置**：readout_ptdom.py:539-549（`shortcut`）。
- **反例（cx_a3.py）**：4 个 seed 里 3 个的 hard 子集为空，剩下一个 n_hard = 2 → `ok=True, n_seeds=1`。
- **影响**：
  - 表里的 hard 子集大小：ranks 0–6、life_stages 0–8、prefixes 4–12、kinship 10–26、country_size 7/8 个 seed 为空。
  - 对这几个领域，(iii) 基本是在抛硬币，或者必然失败。
- **建议**：屏选表里把这类领域标成"无法检验（answer-frequency 已解掉 85–100%）"，不要写成"无效"。这一点与构建者的开放问题 1 一致。
- **置信度**：高。

### 次要 8：HOLD 是全局的
- **位置**：select_ptdom.py:79-88（默认写 runs/ptdom/HOLD）；每个作业开头检查同一个文件。
- **影响**：
  - 一个批次在 select 时有臂所有格子都没过闸，会让**四个批次**所有待跑或被重排的作业在开头以 exit 0 退出。
  - 被抢占后重排的 B200 作业也会因此放弃已有的 ckpt，之后要用 -r1 从头重跑。
  - 只影响成本和时间，不影响正确性。

### 次要 9：FLOOR 标签和消去基线
- **位置**：readout_ptdom.py:802-805。
- **问题**：
  - 顺序比较类的 answer-frequency 线是 85–100，加 5 以后超过 100，FLOOR 必然触发。
  - k-step 类的消去基线（zodiac 47、chinese_zodiac 51、planets 28）不进 FLOOR，两臂都低于消去基线时不会有标记。
- 两者都只是描述，但会让判词行读起来误导。

### 次要 10：B200 抢占的一个小窗口
- **位置**：pretrain_range.py:2390-2394。
- **问题**：
  - 先 `ck.unlink()`，然后才保存模型。如果在这两步之间被抢占，run JSON 已经完整（75 次 eval），但没有模型文件。
  - 重排后 run_guard 判断"已完成"，offline_lm 因为没有模型而失败，于是没有 check.json。
  - A3.8 的手工补救要用到模型，走不通。
- 概率很小。

### 次要 11：picks 文件和重推导不一致时，严格读数只打印 "DIFFER"，不停下（readout_ptdom.py:723-733）
多数情况下，后面缺 seed 的检查会把它拦下来，但最好直接拒绝。

### 次要 12：/scratch 配额已用 95.5%
- `myquota`：4.78 / 5.00 TB，约剩 220 GB。
- 需求：stage 1 加 stage 2 约 400 个 run × 0.21 GB 的模型，再加 16 个 B200 并发的临时 ckpt，合计约 100 GB。
- 放得下，但很紧，其他项目也在写。提交前确认。

### 次要 13：GEOK 修复之后，check_py_a3 没有重做"none 臂出现先验键"的篡改测试
我补做了（geok_tamper.py，用 B3 smoke 的 none run 和 A3 的 CHECK_PY）：
- 不改：不报错；
- 植入 `geo_kinship` 或 `geo_geo_periods`：报错；
- 植入 `geo_periods`：不报错（它本来就不是先验键）。

**修复正确**，只是测试没进 A3 的记录。

### 次要 14：批次 A 的 53M 参照在 Rev F 下只重验了 seed 0 和 4
其余 6 个 seed 靠作业自己的 DRY 哈希检查（失败时 exit 3，是安全的方向）。可以接受，只记一笔。

---

## 逐题核对

### 1. 新领域的训练题、留出题、读出题是否一致；留出防泄漏；k 步类的答案和边界
**未发现问题。**
- **比较类（12 个，Rev E 路径）**：
  - key 是无序选项对 (b, e)。两种顺序共用同一个 key；有锚点的模板，同一选项对配所有锚点一起留出。
  - trainer 断言了两种顺序都在、不跨两边，留出对在训练里任何顺序都不出现。
  - 读出 = held 的全部题、两种顺序（训练模板，放在第 1 槽、行末），2-way 比较 `' <answer>\n'` 和 `' <other>\n'` 两段的 log-prob。
- **k 步类（6 个）**：
  - 循环类：k = 1..2n，key = (x, k mod n)。两种表面形式一起留出，k ≡ 0 的恒等类总在训练里，留出的每个移位在训练里至少有一个条目。
  - 线性类：key = (x, k)，只出 j = x + k < n 的题。交还 key 直到每个 x 和 k 都训练过。
  - 读出题 = held 全集。
- **答案语义**：我逐个核对了条目顺序：
  - zodiac Aries..Pisces；chinese_zodiac Rat..Pig；planets 从内到外；elements Z = 1..36；greek α..ω 24 个；geo_periods 从寒武纪到第四纪（从老到新，和"after = 之后"一致）；
  - ranks 从低到高；life_stages 从新生儿到老年；kinship 的代数（great-grandfather = +3）；prefixes 按指数；
  - people 用出生年配 "less"；animal_size 用 log 质量配 "more"；country_size 用 log 人口配 "more"；units / scalar_adj 是组内比较。
  - 循环回绕 (i + k) mod n、k > 20 用数字、k = 1 用单数，都正确。
  - trainer 里的答案断言是同一公式，属于同义反复；但公式本身是对的。
- **测试**：test_domains_a3（542/542）对 8 个 seed 重推导了切分、泄漏和答案。我抽查它的输出表，与 A3.1 的表一致。

### 2. 先验位点、共享位点、none 臂、各臂数据
**未发现问题。**
- **位点**：取 `' ' + name` 的最后一个 token。三处都断言了：off_line / anc_line 构建每一行时，`--dry_inputs` 对全部行，第一步训练时对真实 batch。dry 日志里可以看到，例如 'Rooster' → 'oster'，'great-grandfather' → 'father'（与 ' father' 的 token id 不同）。
- **共享位点**：我用 tokenizer 独立重算，结果与表一致：geo_periods 5/12、prefixes 3/11、elements 8/36、zodiac 2、people 10、countries 2、units 2、planets / greek / chinese_zodiac 0。20% 措辞线的三个新领域（geo_periods 42%、prefixes 27%、elements 22%）也一致。
- **none 臂**：没有 `--prior_domains`；CHECK_PY 的 "no prior ever" 一项按精确键检查（我的篡改测试见次要 13）。
- **各臂数据**：integrity 对同一批次、同一 seed 的所有 valid run，在每次 eval 比 data_sha，并比 step-0 lm；CPU 参照在 seed 0 上 none@4e-3 = only-<首个领域>@1e-3。先验相关的 RNG 不碰数据流。

### 3. 批次 A 是否逐字节不变；回归是否真的跑了
**未发现问题**（次要 14 除外）。
- reg_a3.out：cpu_short 作业 21837233 状态 COMPLETED，6 个配置全部 IDENTICAL：
  - default、rb6all6、rb6none 对比 93e22bf4c4c0（A1 时没跑的 rb6none 这次也跑了）；
  - batch A 的 stars / months / none 对比 Rev E bb34927f16d8。
  - 比较内容：run JSON 除 code_sha / out / sec 以外全部相同，模型 36 个张量逐位相同，dry 和 run 的 stdout 相同。
- run_refs_a3_A.out：Rev F 加 A3 的 ARGS，seed 0 和 4 的 DRY 行、切分、表 sha、lm0、dsha0 都等于 ptdom_refs.json。
- DRY 的 grep 新增了 "  k-way candidates "，但批次 A 没有这一行，所以哈希不变（已在 seed 0、4 验证）。

### 4. 两阶段规则；选择偏差；seed 4-7
- **写清楚了，代码也实现了。** 我对照了以下几处，与 A3.3 / A3.5 逐条一致：
  - `screen`：4/4 > 0；选中格子过闸；(iii) 只用 hard 子集非空的 seed。
  - `rule`：bootstrap 90%，default_rng(0)，百分位法；两半；t 检验；均值 > 0。
  - `holm`：selftest 对照了 step-down 规则。
- **我构造的反例**：CX-LR（重要 2）、CX-BATCH（重要 3）、CX-SC（次要 7）、cx_halves（重要 4）。
- **选择偏差**：
  - 在平坦零假设、σ 1–10 下，P(ADVANCE) 为 0.024–0.029，低于 1/16，因为 none 是 3 选 1、先验是 2 选 1。
  - 但已晋级时，seeds 0-3 的增益被抬高约 0.8σ，8 seed 的判词在这一前提下的原始错判率为 0.08–0.20（重要 4）。
- **seed 4-7**：选点和移动都只用 seeds 0-3（代码里 screen 和 after2 都是 `seeds=SEL`），所以在给定选点的前提下，seeds 4-7 不受选择影响。
- **但有两点**：
  - 判词的 (a) 和 (c) 把它们和被选过的 seeds 0-3 合在一起，seeds 4-7 只进"均值 > 0"这一条。
  - seeds 4-7 照样继承 lr 范围不对称带来的优势（重要 2）。

### 5. 辅助 loss 闸门
**未发现问题。**
- 按 (arm, lr) 格判；统计量 a / (a + b)；只用 prior_on 的 eval，并截在发散之前；缺字段、没有窗口内 eval、记录值与重推导之差超过 1e-3，都算失败。
- `select --after 1` / `--after 2` 在做决定前先算闸门（外扩格子也闸）；读数在给判词前用所选格子的全部完成 seed（0-7）再闸一次。
- watcher 按格子判；超过第 1000 步还没有窗口内 eval 的 run 判失败。
- audit_a2 的次要 2 / 3（合并 8 个 seed 取中位数、中位数的中位数）仍按设计未改，列在 A3.10 的开放问题里。

### 6. GPU 类型、HOLD / sha、B200 续跑
**未发现会在错卡上训练的作业。**
- 240 个文件：A 和 B3 是 60/60 的 l40s_public 分区，加 `*L40S*` 运行时守卫；B1 和 B2 是 60/60 的 `--constraint=b200`，加 `*B200*` 守卫。
- CHECK_PY 还会检查 gpu.csv 每个样本的 GPU 名。
- B200 的 singleton 槽位：B1 用 0–7，B2 用 8–15。
- HOLD 和 12 个 sha 在训练前检查（分别 exit 0 / exit 3）。HOLD 只在作业开始和重排后开始时检查，已经在跑的作业会跑完（按设计）。
- **B200 续跑**：每 500 步存 ckpt，含全部 RNG（rng、data_rng、dom_rng、order_rng、torch、cuda）和 dyn（data_sha、计数），写 tmp 再 rename；run_guard 认同一个 job id。
- **实证**：Torch B200 上 pt300x 的 84 个日志里，有 9 个出现 "resumed from"，被抢占后续上了。
- 小缺口见次要 8、次要 10。

### 7. Tier L 和措辞
**未发现问题。**
- COMP = ALL24 − KSTEP，18 个比较类（含没有锚点的模板）在屏选表、判词行、捷径行都带 [Tier L]，k-step 类带 [k-step]。
- 报告结尾一句是 "given the structure, the model uses it"。代码和 A3 文本里没有 "discover" 之类的词。
- 一个解释层面的提醒见次要 5：k-step 的单 token 候选与 tied 嵌入耦合。

### 8. 陷阱清单
- **训练和评测分布不一致**：评测放在第 1 槽、行末，用训练模板，和 rb6 一样。k-way 的首 token 打分不同于 F81 的整段打分（次要 5）。
- **记忆**：小表被反复见到（planets 12 题，每题约 1.6 万次；life_stages 18 题；ranks 26 题），留出只有 6–16 题，分辨率低；配对设计能处理。顺序比较类被频率捷径主导（次要 7）。
- **静默失效的损失或梯度**：位点少于 4 个时先验跳过，但 CHECK 要求 ≥ 95% 的 on-step 都用上了；gnorm_prior / gnorm_true = λ；分解双路。未发现问题。
- **单 seed**：没有。
- **token 位置**：k-way 的 logit 下标 `dom_max_ans − ln` 和答案首 token `x[:, L − ln]` 我手推过，正确；位点有断言。
- **同名不同义**："k-way" 和 F81 的 accuracy 不是一回事（次要 5）；各领域的"面积分"互相不可比，但 Holm 用的是 p 值。
- **静默过滤**：读数跳过缺席批次（重要 3）；(iii) 跳过 hard 子集为空的 seed（次要 7）；CKA-only 失败算 valid（按设计）。
- **修 bug 的副作用**：GEOK 修复已验证（次要 13）。另一个副作用：阻断 1 的已知答案检查是 A3 推广到多 token k-step 时才变得要紧的，但 smoke 没有一次命中，所以没测到。

---

## 现在能不能提交

**这个实验最可能测错的地方**有两处：
1. **运行层面**：k-step 批次（B2、B3，可能还有 A 的 months）的 k-way 已知答案检查，在 GPU bf16 下会被 logit 并列大量触发，run 被判无效、整批卡死（阻断 1）。
2. **判词层面**：lr 可达范围不对称，让纯 lr 效应冒充先验效应拿到 PASS（重要 2）。

**现在不能提交。** 先做一处小改，并补两条预注册；做完即可提交 stage 1：
1. 把 `kway_em_bad` 改成并列感知，或者在 CHECK_PY 里降为 info 并让 loader 容忍；然后重渲染 240 个作业文件。
2. 在任何结果出来之前写进预注册：
   - none 在下沿出现 EDGE 时怎么处理（F1 / F2），以及同 lr 的读数增益；
   - "四个批次都齐了才读判词"；
   - 只差 GPU 标定类容差的 run 算不算数据。

重要 4 的数字改写和告知可以和提交同时做。
