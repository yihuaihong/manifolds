# ptdom 构建对抗审查（全新审查者，只读，2026-11-03）

范围：AUDIT.md、make_jobs_ptdom.py、84 个 stage-A job、select_ptdom.py、readout_ptdom.py、submit_ptdom.sh；
trainer diff `75c7e1e..21cdc14`（tools/pretrain_range.py，+286/−31，worktree /scratch/yh6210/llmadd-burst-ptdom）；
tests/ 下的 dry log、smoke、refs。没有改任何文件，没有提交或取消作业（队列里没有 ptdom 作业，也没有 SUBMITTED_ 标记）。
自己做的检查脚本都放在 scratchpad：entail.py（目标是否带着答案）、lasttok.py（site token 撞车）、heur2.py（锚点条件捷径）、
smokechk.py（从 smoke 曲线重算读数和梯度诊断）。

**总判断：没有发现阻断问题。** 管线测的就是它声称要测的东西：读数 key 正确，2-way 和 k-way 的 logit 下标从代码推一遍是对的，
split 按 key 切、没有泄漏，none 臂确实没有先验，各臂数据相同，预注册的统计规则也照写实现了。
但有 4 条重要问题。它们会让 PASS 或 NOT SHOWN 的含义偏离本意，最好在投递前由用户定。

---

## 问题清单（按严重程度排序）

### 重要 1：两边 lr 网格不重叠，bracket 只看「按 bpb 打破平局后」的 pick。none 读数如果往低 lr 缓慢单调上升，这种情况永远不会被外扩，增益里会混进 lr 效应
- **位置**：make_jobs_ptdom.py:60（GRID：none 4e-3/8e-3/1.6e-2，先验 5e-4/1e-3/2e-3）；
  readout_ptdom.py:158-159（先 `best_lr(top2)`，再对 `b["best"]` 做 `bracket`）；
  code/tools/readout_rules.py:121-152（bracket 只有在「pick 在边上」且「领先邻居超过 2 点」时才判 open；
  argmax 在边上但领先不到 2 点只记 "note"，不外扩）；AUDIT.md:70-73、226-227。
- **为什么**：先验组的每个 lr 都比 none 的最低 lr 低 2 到 32 倍。设想 none 的锚点读数随 lr 下降每档只涨 1.5 点：
  4e-3 = 63、8e-3 = 61.5、1.6e-2 = 58。前两名相差不到 2，按 bpb 打破平局会选 8e-3（多半是 8e-3 的 bpb 更低）。
  8e-3 在网格内部，判为闭合，不进 stage X。none 也就从来没有在先验组所在的 lr 上被测过。
  这时 only-d@1e-3 − none@8e-3 里可能有好几点纯粹来自 lr，三条判据都能通过，结果就是一个假 PASS。
  反方向（先验组在 2e-3 上缓慢上升）只会让结果更保守。所以这个偏差只往「假 PASS」一个方向走。
  构建者提的 none@1e-3 只用于 LM 那一行，没有用于读数。
- **可验证的检查**：stage A 之后，对每个 `none|d` 单元打印 area 的 argmax 和 bracket 的 "note"。另外跑
  none@2e-3（最好再加 1e-3）、seed 0-3，共 4–8 个 run，约 5–10 L40S-h，把它在 d 上的读数和 none 的 pick 比。
  如果更高，这个 domain 的 gain 就不能写成先验效应。更干净的做法是投递前就预注册：none 在下沿出现 "note" 时也外扩，
  或者把 2e-3 直接放进 none 的 stage-A 网格。候选变多只会让 none 更强，对先验是保守的。
- **置信度**：这个缺口一定存在（代码确定）；它会不会实际造成偏差取决于锚点读数和 lr 的关系，目前未知，中等。

### 重要 2：「先验只是辅助 loss」这个前提在 53M 上没有验证，而且 stage C 前的闸门只写在文字里，代码里没有
- **位置**：AUDIT.md:148-149、230-236；readout_ptdom.py:343（FLAG 只是描述）；select_ptdom.py:48-56（不读 lm_share）；
  submit_ptdom.sh:26-34（后续阶段的命令里没有这一步）。
- **为什么**：smoke（tiny 模型，smokechk.py 重算）在窗口内的 LM share 是 0.083–0.099，clip_coef 是 0.10–0.12，
  也就是全局裁剪把整步更新压到了约 10%。step 0 时未加权的先验梯度是 LM 的 2.5–3.1 倍
  （gnorm_true 3.8–4.8 vs grad_lm 1.55）。rb6 在 53M 上未拟合的先验是 LM 的 1.9–4.3 倍。
  锚点先验有 60–89 个多 token 条目，拟合可能比 months 慢。用户的规则是「先验只是辅助 loss，LM 梯度必须主导」
  （2026-10-19 因为先验约占 2/3 更新否决了 all6 λ5）。如果 stage A 的先验组 LM share 中位数低于 0.5，
  60 个先验 run 就都跑在用户否决过的那种状态里。同时 clip 会降低这些 run 在窗口期的 LM 有效步长，
  这又和重要 1 的 lr 混淆叠在一起。
- **可验证的检查**：job 是按 seed 优先的顺序投的。前几个先验组 job 跑过第 1000、1250、1500 步的 eval
  （窗口内前 3 个 eval，大约开跑 15–20 分钟后）时，从 run log 读 `grad_lm_share` 和 `clip_coef`。
  如果中位数低于 0.5，touch HOLD，等用户决定。另外建议 select_ptdom.py 打印每个先验单元的 window median LM share，
  有 FLAG 时拒绝生成 stage C。
- **置信度**：风险存在，高；53M 上会不会真的触发，中等。

### 重要 3：先验 site 落在多 token 名字的最后一个子词，很多条目的 site 是同一个 token（token 层面的同名不同义），而且最严重的恰好是 AUDIT 称为「最干净检验」的 stars
- **位置**：pretrain_range.py:657 起（anc_line：site 取包含名字最后一个字符的 token）；AUDIT.md:59-63、139。
- **为什么**：lasttok.py 用 OLMo-2 tokenizer 数了和别的条目共用 site token 的条目：
  - stars 30/89（12 个 token）：'ux' = Acrux / Becrux / Gacrux / Pollux；'ab' = Girtab / Kochab / Marchab / Markab；
    'ar' = Achernar / Izar / Schedar；'a' = Agena / Ankaa / Shaula；等等。
  - landmarks 16/76：' Bridge' ×4、' Tower' ×3、' Falls' ×3、' Castle'、' Palace'、' House'。
  - cities 5/60：' City' = Ho Chi Minh City / Mexico City / New York City / New Taipei City / Quezon City。
  - languages 2/62（'ese'）；colours 0。

  这里的模型是从头训练的 53M、6 层，先验作用在第 2 和第 4 层。对同一个子词 token，状态只能靠 attention 区分条目，
  先验却要把同一个 token 拉向不同的目标坐标。ft36 用同样的 site 规则，但那是一个本来就认识这些名字的 7B 模型。
  这不是 bug，是忠实沿用 ft36 规则带来的可行性风险。stars 或 landmarks 判 NOT SHOWN 时，可能只是「这个 site 不可行」，
  不能写成「这个目标在预训练里没用」。
- **可验证的检查**：从保存的模型离线算 item-state CKA，把条目分成「site 唯一」和「site 共享」两组分别报告；
  或者在 AUDIT 里写明这两组的条目数，并把它列为解读 NOT SHOWN 时的前提。
  改 site（例如对名字的全部 token 取平均）会改设计，要用户定。
- **置信度**：撞车数字是确定的（tokenizer 实测）；它对结果的影响有多大，中等。

### 重要 4：colours 有一条构建者没记录的非几何捷径，能拿到 86%，「heuristic-hard 子集」也挡不住它
- **位置**：pretrain_range.py:791-802（只算了全局答案频率和更短选项两条启发线，以及它们的 hard mask）；
  AUDIT.md:94-96、134-135（「hard 子集显示增益在频率和长度都解不了的题上是否还在」）。
- **为什么**：heur2.py 在 trainer 的 split 上（seed 0 和 1）算了「锚点条件胜率」：
  对留出题「a：b 还是 c」，选训练里以 a 为锚点时胜率更高的那个选项。

  | 领域 | 全局答案频率（构建者的） | 锚点条件胜率 | 锚点条件胜率在构建者 hard 子集上 |
  |---|---|---|---|
  | colours（seed 0 / 1） | 55.6 / 52.4 | **86.0 / 84.3** | **71.0 / 69.4**（n 434 / 546） |
  | stars | 49.3 / 48.3 | 50.3 / 49.2 | 9–11 |
  | landmarks | 61.6 / 62.7 | 63.0 / 64.1 | 11–14 |
  | languages | 87.2 / 85.2 | 88.1 / 86.5 | 13 |
  | cities | 67.7 / 69.1 | 70.3 / 71.6 | 15 |

  colours 只有 24 个条目、233 个 key，每个锚点在训练里出现的次数很多。所以只记「谁和 red 一起时常赢」，
  不需要任何几何表征，就能答对留出题的 86%。

  后果：
  - colours 的 FLOOR 线（max(chance, 启发线) + 5 = 60.6）定得太低。
  - colours 的 hard 子集不是「捷径都失效的题」：上面 71% 仍然能被这条捷径答对。
  - 「增益在 hard 子集上还在」这个描述对 colours 不成立。

  主判词不受影响，因为两臂看到同样的数据，都能学这条捷径。但 colours 能留给先验的空间比 AUDIT 写的小得多。
- **可验证的检查**：重跑 heur2.py；把锚点条件胜率作为第三条启发线记进 info，并加进 hard mask（三条都错）。
  这只是 secondary 和描述性内容，可以在读数阶段补，不需要动 trainer；但要在投递前写进 AUDIT，否则就成了事后加的东西。
- **置信度**：数字是确定的；解读的置信度高。

### 次要 5：领域在一行里的搭配是固定的：months 永远和 stars 在同一行，landmarks 和 languages，colours 和 cities
- **位置**：pretrain_range.py:1402（`EXT[((s*32 + r)*2 + j) % 6]`）：64 mod 6 = 4，起点永远是偶数，所以一行里永远是
  (0,1)、(2,3)、(4,5) 这三种组合之一；dry log refs/dry_none_lr8e-3_s0.log:76-87 可以直接看到；AUDIT.md:23 只写了份额相等。
- **为什么**：主读数不受影响（eval 在 slot 1，因果 attention 看不到后面那一行）。
  但 slot-2 的上下文永远是同一个伙伴领域；跨领域矩阵（例如 only-stars 对 months 的影响）和同行共现混在一起，分不开。
- **检查**：在 AUDIT 和跨领域矩阵的说明里写明这个固定搭配。置信度：高。

### 次要 6：先验的 Gram 每个 batch 重新做列标准化，和 ft36 的「在全部条目上一次性标准化」不一样
- **位置**：pretrain_range.py:1432（prior → centred_gram(Fb)，按 batch 内的 site 行标准化）vs ft36.py:248（`Fz` 在全部条目上固定）；
  AUDIT.md:64-66（写了 "the trainer's centred_gram"）、133（「目标蕴含每道留出题的答案：F81 R2 99-100%」，那是 ft36 的目标）。
- **为什么 / 实测**：存储的目标列在全部条目上本来就是标准差 1，所以偏差不大。entail.py 模拟了 300 个 batch（每个 11 行）：
  按 batch 缩放后的目标对留出题答案的蕴含率，均值 stars 99.2、landmarks 99.1、colours 99.8、cities 98.3、languages 100，
  单个 batch 最低 92.6–95.0。Tier L 的前提仍然成立，只是 AUDIT 第 133 行的 99-100% 对这个 trainer 来说略有高估。
- **置信度**：高。

### 次要 7：GPU 上「先验确实进了 backward」的检查偏弱
- **位置**：make_jobs_ptdom.py:239-241（`grad_T_reldiff` 的中位数取全部 75 个 eval）；make_jobs_ptdom.py:264-268
  （`gnorm_prior / gnorm_true = λ`）配合 pretrain_range.py:1980（`act[d] = LAMD[d] * v[d]`）。
- **为什么**：75 个 eval 里有 49 个在窗口外，那里 grad_T 就是 grad_lm，相对差约为 0。所以中位数不可能被窗口内的错误拉上去，
  这条检查实际上不会失败。λ 比值检查是同义反复：日志和 loss（pretrain_range.py:2118）都乘的是 LAMD。
  CPU smoke 上窗口内的 reldiff 是 1.5e-4–6.3e-4，而且每个 eval 都 assert 过，所以代码路径已经在 CPU 上验证过。
  GPU 上唯一的行为层检查是 CKA 上升，而它失败了 run 仍然有效。
- **检查**：把中位数或最大值只取窗口内的 eval（`e["prior_on"]`）。置信度：高。

### 次要 8：FLOOR 标签用了启发线，「not measurable」这个说法可能把一个可测的零结果说成测不出
- **位置**：readout_ptdom.py:297-300；AUDIT.md:127。
- **为什么**：languages 的 floor 线是 92.2（87.2 + 5）。两臂 area 都在 80 多的话（area 还包含了训练早期），
  会被打上 FLOOR、写成「not measurable, never no effect」。可是在 50 到 92 之间其实是能测出差异的。
  建议把这类情况的标签改成 "BELOW HEURISTIC"，和真正的 chance 地板分开。置信度：高（只是措辞层面的问题）。

### 次要 9：picks 的「double path」其实是同一份代码跑两遍
- **位置**：select_ptdom.py:25、48（`RO.decide`）；readout_ptdom.py:247-254；AUDIT.md:106。
- **为什么**：它只能发现选完 lr 之后数据变了，发现不了选择逻辑本身的错误。逻辑本身已由 selftest 的已知答案覆盖。
  建议把 AUDIT 里的说法改成「一致性检查」，不要叫「double path」。置信度：高。

### 次要 10：每个领域的判词行没有带 Tier L 限定，限定只写在页脚
- **位置**：readout_ptdom.py:357-369。
- **为什么**：单独复制出来的一行 "stars PASS ... gain +x" 会丢掉「目标带着答案、只能读成注入」这层意思。
  建议在锚点领域的判词行上直接加 `[Tier L]` 标签。全部材料里没有任何「模型发现了结构」之类的说法。置信度：高。

### 次要 11：训练前没有校验 atlas 数据和环境版本
- **位置**：manifold_domains.py:25（`DATA = /scratch/yh6210/llmadd_torch/data/atlas`）；job 第 1 节只校验 12 个 .py 和 train.bin / val.bin。
- **为什么**：/scratch 会被 flush（memory 里有记录）。CSV 被删的话 dry_inputs 会报错（exit 3，安全）。
  CSV 被改的话，DRY 里解码出来的行多半会变（exit 3）；只改坐标、不改题目时，要等训练完的 table sha 检查才会 exit 4，
  这时 GPU 时间已经花掉了。torch、transformers、geonamescache、matplotlib 的版本只记录、不校验。置信度：中。

### 次要 12：step-0 lm 和 CPU 参考的容差 1e-3 只在 A100 上验证过
- **位置**：make_jobs_ptdom.py:209-210。
- **为什么**：rb6 在 A100 上用同样的容差通过了。L40S 上的 bf16 数值没测过。万一超出，所有 run 都会被 loader 排除
  （不只是标记一下），重投也会一模一样地再失败，整个管线会卡住。
- **检查**：看第一个跑完的 job 的 check.json。置信度：低到中。

### 次要 13：months 预期是 FLOOR，还占了 Holm 的一个名额
- **位置**：AUDIT.md:50-52、138。
- **为什么**：16 个以上的 run（约 21 L40S-h）花在一个预期必然 NOT SHOWN [FLOOR] 的领域上；
  其余五个领域的 Holm 门槛也因此从 α/5 变成 α/6。这是用户已经做过的决定，这里只记一笔成本和统计功效。置信度：高。

---

## 逐题回答

**1. 模板、位置、数值范围，以及留出是否按 key 防泄漏。**
- 锚点类的训练行和 eval 行都用 DLINE，模板相同（`ft36.pair_questions` 的 PAIR / PAIR2）。锚点类没有改写格式（`PARA_OF` 返回 ()）。
- 位置：eval 是 text + 一行 numbers 训练格式 + 题目放在行尾，对应训练里的 slot 1（pretrain_range.py:978）。
  训练中 slot 份额为 0.47–0.53。
- 条目范围：每个条目都在训练里出现过。n_answers_trained 等于条目数；只有 languages 是 59/62，有 2 道留出题的答案从没在训练里当过答案。
- key = 无序选项对 {b, c}。trainer 里 assert 了：没有 key 同时出现在两边，留出的选项对在训练里不以任何顺序、任何锚点出现，
  每道题的两种顺序都在同一侧。生成器测试独立重推了 split（52/52）。我读代码也确认了 `ft36.split` 的规则和 trainer 的 assert。
- 「同一锚点」：锚点本身不留出，留出题的锚点会带着别的选项对出现在训练里。这是设计（与 F81 相同）。
  它带来的锚点条件捷径见重要 4。
- months 就是 rb6 的 months：k = 1..23、k ≠ 12，tier S 按 (item, k mod 12) 留出，读出是 12 选 1，候选都是单 token（有 assert）。
  训练、留出、读出之间没有不一致。

**2. token 位置，以及 none 的先验是否真的关闭。**
- 规则是名字最后一个 token。在 dry log 里解码核对过：'Achern⟦ar⟧'、'Eiffel⟦ Tower⟧'、'Great Pyramid of G⟦iza⟧'、
  'Malay⟦alam⟧'、'mar⟦oon⟧'、'Quezon⟦ City⟧'、'Rio de⟦ Janeiro⟧'；months 落在起始月 ' August'。
  答案 span 上从不放 site（site 都在 "Answer:" 之前）。生成器测试解码了 191 个 site，全部正确。问题在于大量 site 撞车（重要 3）。
- none 臂的先验是真关的：DPRI 为空，prior_on 恒为 False（pretrain_range.py:1815），loss 里没有任何先验项（:2115-2118）。
  `dom_grads` 只用 `autograd.grad` 记 gnorm_true，不碰参数。job 里的 dry block 要求每个领域的 prior 都是 false；
  CHECK 要求 geo == 0、没有 gnorm_prior_ 键、LM share = 1。smoke 实测也是这样。

**3. 各臂数据是否相同、是否只差先验。**
- 是。数据流（2000+seed、4100、5000+seed，锚点类 4200+seed）都不依赖臂。
- smoke 两臂 11 个 eval 的 data_sha 全部相同，step-0 lm 也相同；不同臂的 DRY 哈希相同（refs）。
- job 文件之间的差别只有先验的 flag、lr，以及 none 里不起作用的默认值（lam 5、schedule whole、lam_rule norm：没有先验时都不起作用）。
- 唯一真正的系统差别是 lr（按设计如此）和窗口期的 clip（重要 1、2）。

**4. lr 选择是否公平，以及判据是否写清楚、是否实现。**
- 每个领域单独给 none 选 lr 是合理的：它给了 none 在该领域最强的基线，对先验是保守的。
  两边都是 3 选 1、都用 seed 0-3 选，赢家诅咒是对称的，再由两半 seed 的规则兜底。
- 问题在网格不重叠，加上 bracket 的 2 点容差（重要 1）。
- 判据写清楚了，也照写实现了：
  - (a) 10,000 次 bootstrap，`default_rng(0)`，取第 5 和第 95 百分位（readout_ptdom.py:202-206）；
  - (b) 两半 seed 均值都 > 0；
  - (c) 双侧单样本 t 检验，Holm 在六个领域上做 step-down，单调并封顶 1（:220-228）；
  - 另外要求总均值 > 0（:235）。
  selftest 有 31 个已知答案，覆盖 F98 的情况（bootstrap 排除 0 而 t 区间包含 0）、只过原始 p 不过 Holm、STOP 等。
  NOT FINAL 的领域按它手头的 seed 数算 p 进 Holm，这仍然是合法的 p 值，Holm 依然有效。

**5. Tier L 的定位和措辞。**
- 定位正确：AUDIT.md:8-9、132-135 和 readout 页脚都写了「给了结构就会用」，没有说「模型发现了结构」，
  也明确说了 PASS 不是几何特异的。
- 实测目标对留出答案的蕴含率：raw 和 ft36 的 99.3–100，这个 trainer 每个 batch 缩放后平均 98.3–100（次要 6）。
- 不足：判词行没有带 Tier L 标签（次要 10）；对 colours，「hard 子集」不能支持「捷径解不了」这种表述（重要 4）。

**6. 是否有作业会在非 L40S 上训练，或在 HOLD、sha 不符时仍然训练。**
- 没有。HOLD 是第一条命令（exit 0）。12 个文件的 sha、COMMIT、数据 sha256 都在训练前检查（exit 3）。
  L40S 检查用 `nvidia-smi` 的名字（exit 3），也在训练前。分区是 l40s_public。
  requeue 后脚本从头再跑，这些检查都会重做。
- 缺口：
  - 正在运行的 job 不会看 HOLD（按设计如此）；
  - atlas CSV 和包版本训练前不校验（次要 11）；
  - 训练后的 CHECK 失败是 exit 4，这时 GPU 时间已经花了（按设计如此）。

**7. 陷阱清单。**
- **训练和评测分布不一致**：未发现。slot 1 一致；eval 文本来自 val.bin（与 rb6 相同）。
- **记忆**：没有固定 batch。months 的 118 道题各重复约 1650 次，是有意为之，tier S 需要泛化。
  锚点类每道题约 108 次，用 SEEN < 80 标出。还有一条新的非几何捷径（重要 4）。
- **静默失效的 loss 或梯度**：
  - λ 生效了，但 GPU 上的检查偏弱（次要 7）；
  - 少于 4 个 site 或少于 3 个条目时会跳过这个 batch 的先验项，有记录，并要求 ≥ 95% 的窗口步用上了先验；
  - 先验项不在 no_grad 里（:2109-2118，处在 autocast 里，CKA 用 .float() 计算）；
  - clip 会压低更新（重要 2）。
- **单 seed**：没有这个问题，用了 8 个 seed，有两半规则和 Holm。
- **token 位置、同名不同义**：重要 3。
- **静默过滤**：`_pair_answer` 的 margin 过滤和 1500 道题的上限（subset seed 7）都和 ft36 相同；有记录，各 seed 一致。
- **修 bug 的副作用**：代码阅读确认，非锚点路径只改了 `PARA_OF`、`far_perm` 的默认 gram、`v[:len(qs)]`、ALL_DOMAINS 扩展，
  在 rb6 的输入下行为都不变。tiny 回归（default、rb6 all6、rb6 none）的 JSON 和模型逐位一致。
  基线 trainer 的 sha 我核对过：`75c7e1e:tools/pretrain_range.py` = 93e22bf4c4c0 = origin/main。
  rb6_505m 的读数是用冻结代码重跑的，它只能说明冻结的产物没有损坏，不能单独证明新 trainer 和旧 trainer 等价。
- **其他**：stars 的 smoke 读数是从 metrics 按曲线重算出来的，和 metrics 一致（smokechk.py：area 和曲线均值完全相同）。

---

**这个实验最可能测错的地方：** 先验组的 lr 全部比 none 的网格低。none 在下沿每档只差不到 2 点时，bracket 会把它判成闭合，
而且没有同 lr 的 none 读数，所以一个 PASS 可能其实是 lr 效应（加上窗口期 clip 的效应），而不是先验效应。
