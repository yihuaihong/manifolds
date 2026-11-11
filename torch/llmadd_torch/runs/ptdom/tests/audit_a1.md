# ptdom Amendment A1 对抗审查（全新审查者，只读，2026-11-04）

**范围**：AUDIT.md 末尾的 Amendment A1；make_jobs_ptdom.py、runs/ptdom/{select_ptdom.py, readout_ptdom.py, aux_gate_check.py, submit_ptdom.sh}；92 个 stage-A 作业文件；
逐条对照上一轮审查 tests/audit_build.md 的重要项 1–4 和次要项 5–13。

**纪律**：除本文件外没有写任何文件，没有提交或取消作业。
- 所有复现和反例都放在会话 scratchpad（`.../scratchpad/a1/`），并设了 `PYTHONDONTWRITEBYTECODE=1`。
- 结束时 runs/ptdom 下没有 HOLD、SUBMITTED_ 或 picks_* 文件。

**总判断：没有发现阻断问题。** A1 按它写的内容实现了，四个重要项都有对应的代码。
- 我独立重跑了以下测试，全部通过：selftest 47/47、aux_gate_test 8/8、plumbing 13/13（后两个原先是在 readout_ptdom.py 最后一次修改 21:46 之前跑的，现在在当前文件上确认通过）、92 个作业文件逐字节等于当前生成器的渲染、rb6 冻结回归。
- 但有 2 条新的重要问题：
  - 辅助 loss 闸门的聚合方式和阈值比用户规则松，构造反例可以绕过；
  - A1 新加的外扩触发条件会明显增加 X 和 STOP 的次数。
- 另有若干次要问题。

---

## 问题清单（按严重程度排序）

### 重要 1：辅助 loss 闸门按「臂」取中位数、阈值是 share < 0.5，选中 lr 上的 run 可以处在用户否决过的状态却照样过闸；先验梯度大于 LM 梯度时也不会报警

- **位置**：
  - readout_ptdom.py:126-141（`aux_gate`：先取每个 run 的窗口中位数，再对该臂所有 lr × seed 的 run 取中位数，然后和 0.5 比）；
  - select_ptdom.py:52-64；
  - readout_ptdom.py:650（side effects 里的 FLAG 只看 `min(lm_share) < 0.5`）；
  - aux_gate_check.py:151-157（早期检查对所有 lr、所有 run 的 eval 合并取中位数）；
  - AUDIT.md:289-293。
- **为什么**：
  1. **按臂聚合**：每个先验臂在 stage A 有 12 个 run（3 个 lr × 4 个 seed），闸门看的是这 12 个数的中位数。所以选中 lr 的 4 个 run 可以全部低于 0.5，闸门照样通过，这个单元也照样进入 C。
  2. **阈值比用户规则松**：用户规则（memory manifold_aux_loss_principle）写的是「flag any arm where the prior gradient exceeds the LM gradient」，rb6 的 `grad_split` 也专门输出了 `frac_prior_gt_lm`。而 share = |g_lm| / |g_lm + λg_p|，在 cos≈0 时，share < 0.5 对应的是 |λg_p| > 1.73 |g_lm|。所以先验梯度是 LM 的 1–1.73 倍时，闸门通过；gate_report 不打印 prior/LM，读数里的 FLAG 也不会触发。
  3. **反向梯度**：先验梯度和 LM 梯度反向时，|g_T| 变小，share 会大于 1，同样通过。
- **构造反例**（gate_cx.py；用 `readout_ptdom.fake_run` 造 92 个 stage-A run，再调用真实的 select_ptdom.py，`--root/--out/--hold` 都指向 scratchpad）：

  | 反例 | 构造 | 结果 |
  |---|---|---|
  | CX1 | only-stars 在 1e-3（area 最高、在网格内部、闭合）4 个 seed 的 LM share 都是 0.3，其余 lr 是 0.9 | 臂中位数 0.900 → **AUX GATE passed**，exit 0，写出 picks，`only-stars|stars pick 1e-3 decision C` |
  | CX1b | 12 个 run 里有 5 个是 0.3 | 同样通过 |
  | CX2 | |λg_p| = 1.5 |g_lm|，cos 0，share 0.555 | 通过；没有任何地方报警 |
  | CX3 | |g_lm| = |λg_p| = 1，cos −0.9，share 2.236 | 通过 |
  | CX6（早期检查） | seed 0 的 2e-3 是 0.3，5e-4 和 1e-3 是 0.9 | 合并中位数 0.900 → exit 0 |

  对照组表现正确：
  - CX4：只有窗口外的 eval 带低 share，被正确忽略，闸门通过；
  - CX5：一半窗口内 eval 是被跳过的 batch（share 1.0、没有 grad_prior），另一半是 0.3 → exit 2、打印 AUX GATE FAILED、HOLD 写入理由、没有 picks 文件。
- **可验证的检查 / 建议（需要用户定）**：
  - 闸门至少在 (arm, lr) 单元上判一次，尤其是选中 lr 的 4 个 seed，并在 stage C 的 run 上再判一次；
  - 同时打印并标记窗口中位数 `prior_over_lm > 1` 或 `frac_prior_gt_lm`（rb6 grad_split 里就有这个字段），让它和用户原话一致；
  - 阈值是 0.5 还是「先验梯度不超过 LM 梯度」，交给用户决定。
- **置信度**：
  - 代码事实和反例：高；
  - 和用户规则不一致：高；
  - 53M 上会不会真的出现（rb6 单先验在 53M 的窗口中位数 ≥ 0.98；锚点先验目前未知）：中低。

### 重要 2：A1 新加的外扩触发条件（area argmax 在网格边上，不要求领先幅度）用在了每个单元、两个边，而且 X 之后也用它判 STOP。lr 响应平坦时，X 和 NOT FINAL 都明显变多，"nominal 156 L40S-h" 基本不会发生

- **位置**：
  - readout_ptdom.py:374-391（`extend_rule`）、:432-437（`decide`：X 之后只要 `u["open"]` 就判 STOP，包括另一侧从没外扩过的边）；
  - AUDIT.md:266-274、353-362。
- **为什么**：
  - 上一轮审查的重要 1 只要求「none 在下沿出现 note 时也外扩」，也就是朝先验组 lr 的方向。A1 把这个条件推广到 12 个单元、两侧，并且 STOP 也不再要求 2 点的领先幅度。
  - 按 argmax 判断，和 area 的尺度无关。如果在噪声范围内 lr 对 area 没有影响：
    - 3 点网格的 argmax 落在边上的概率是 2/3；
    - 外扩之后，argmax 落在新边上的概率仍然不小。
- **蒙特卡洛**（stop_mc.py；调用真实的 `RR.best_lr('top2')` 和 `RO.extend_rule`，seed 均值的噪声 sd 为 σ，bpb 随机，每格 4000 次）：

  | 设定 | 先验单元 P(X) | 先验单元 P(STOP) | none 单元 P(X) | none 单元 P(STOP) | 一个领域 NOT FINAL 的概率 |
  |---|---|---|---|---|---|
  | 平坦，σ 0.5 | 0.66 | 0.16 | 0.41 | 0.07 | ≈ 0.22 |
  | 平坦，σ 1.5 | 0.67 | 0.19 | 0.43 | 0.09 | ≈ 0.25 |
  | 中心有弱峰（0.25 点/步²），σ 1 | 0.61 | 0.06 | 0.17 | 0 | ≈ 0.06 |
  | 中心峰 0.5 点/步²，σ 1 | 0.51 | — | — | — | ≈ 0.01 |

- **后果**：
  - 以弱峰估计，6 个先验单元约有 3–4 个会进 X，即 14–16 个 run，约 18–21 L40S-h，另加 none 的 X 和可能多出来的 C。实际期望成本明显高于 nominal 120 run / 156 h。
  - lr 响应平坦时，stars（AUDIT 称为最干净的检验）等主要领域约有两成概率最后是 NOT FINAL，没有判词。build 版的规则没有这个代价。
  - months 预期就在地板上，area 接近噪声，最容易触发这一条。
- **可验证的检查 / 建议（需要用户定）**：
  - 方案一：只对 none 单元的下沿用条件 (ii)，这是审查原本的诉求；
  - 方案二：保留 (ii) 用来决定是否外扩，但 X 之后的 STOP 仍然只用 bracket 的 'open'（2 点幅度）；
  - 无论选哪种，AUDIT 的成本表都应该给出「预期 X」的估计，而不只是 0–32。
- **置信度**：机制确定（代码）；具体概率取决于还不知道的 lr 响应，中等。

### 次要 3：早期检查 aux_gate_check 不把「无法双路核对」当成失败，和 A1.2 的写法以及 select 闸门都不一致
- **位置**：aux_gate_check.py:153-157（`bad` 只看合并中位数和 `max(dp) > 1e-3`）。
- **为什么**：A1.2 写的是 exit 1 if "< 0.5 (or unverifiable)"。
  - CX7：所有窗口内 eval 都没有 grad_cos，检查仍然 exit 0，打印 "rb6 double path max diff -"；
  - CX7b：同样的数据在 select 闸门里会失败（`dp_missing 312`）。
- 训练器只要写了 grad_prior 就会写 grad_cos（pretrain_range.py:1986），所以实际影响小。
- **检查**：`bad` 里加上「有 share 但没有 dp」的情况。置信度：高。

### 次要 4：stage C（seed 4–7）的 run 在出判词之前从来没有过闸
- **位置**：readout_ptdom.py:524、661（读数只打印闸门结果，判词不受影响）；select_ptdom.py 只在 `--stage A/AX` 时运行。
- **为什么**：选中 lr 的确认 seed 即使 share < 0.5，最终读数也只会多一行 "AUX GATE FAILED"，PASS 照样打印。
  这和 A1.2 的文字（"The read-out prints the same gate"）一致，但和闸门的目的不一致。
- **检查**：闸门失败时，在读数里把受影响领域的判词标成 HOLD（或者直接拒绝出判词）。置信度：高。

### 次要 5：分解双路检查用的是中位数，不是最大值
- **位置**：make_jobs_ptdom.py:273-278。
- **为什么**：只在少于一半的窗口内 eval 上出现的 grad_T 与 backward 不一致会被放过。CPU 上的差是 1.5e-4–6.3e-4，GPU 上用 max < 0.05 也有很大余量。
  已修好的部分是「只看窗口内」（反例：check_py_test 里 0.3 现在会失败）。
- **置信度**：高；实际风险低（分解和 backward 用的是同一个 loss）。

### 次要 6：stage A 进行中 touch HOLD 之后（A1 早期检查的标准动作），排队中的作业在开头 exit 0，但它们的 SUBMITTED_ 标记还在，所以 `make_jobs_ptdom.py --stage A --submit` 之后会跳过它们
- **位置**：make_jobs_ptdom.py:564-566（有 flag 就 continue）；make_jobs_ptdom.py:439（作业模板里的 HOLD → exit 0）。
- **为什么**：用户放行之后，没有写明怎么重新投递这些作业。select 会因为缺单元而拒绝（安全），但流程会卡住，或者有人手工 sbatch。
- **检查**：在 submit_ptdom.sh 或 AUDIT 里写明 HOLD 之后的重投流程，例如对「无 run JSON 且有 SUBMITTED_ 标记」的作业怎么办。置信度：高（只是操作层面）。

### 次要 7：同 lr 的 LM 行会把发散的 run 悄悄丢掉，而不是把它记成最差
- **位置**：readout_ptdom.py:626-630（`pair` 只保留有限值，inf − x 被过滤掉）。
- **为什么**：AUDIT 写的是「a diverged run is data (its bpb is the worst)」，这里却只会让 n 变小，wins 也随之缩小。打印里有 n，所以看得出来。
  vs none@pick 那一行在 build 版本里就是这样。
- **置信度**：高；影响小，只是描述性数字。

### 次要 8：colours 的 NOT SEPARABLE 规则用的是 hard 子集上的 AREA（包含训练早期接近随机的 eval），去和捷径基线 69.6（一个固定准确率）比，实际上可能让 colours 几乎不可能判 PASS
- **位置**：readout_ptdom.py:488-493、546、593-594；AUDIT.md:328-332。
- **为什么**：AUDIT 自己写了「conservative against a PASS」，这是有意的选择，不是 bug。但用户应该知道：colours 的 PASS 条件实际上接近「整个训练期间在 hard 子集上平均高于 69.6」，可能等于事先把 colours 判成 NOT SEPARABLE。
  - 想要不那么保守，可以改用 final3 和基线比。
  - 不改的话，在判词说明里写明这个门槛。
- **置信度**：规则的含义：高；会不会真把 colours 卡死：未知。

### 次要 9：两份日志的结尾措辞和实际做了什么不符
- **位置**：
  - tests/a1_refs_full.out 结尾是 "A1 REFS FULL RE-RUN PASS (21/21 …)"，但 18 个格子里实际重跑的只有 2 个，其余是拿存档文件比对；
  - tests/reg_trainer_a1.out 结尾是 "REGRESSION FAIL"（rb6none 没有跑）。
- AUDIT 表里两项都正确标成了 PARTIAL，只是单独看日志会被误导。置信度：高（只是措辞）。

---

## 逐题回答

**1. none 的 lr 网格和同 lr 的 LM 行**：已修好。
- 网格：make_jobs_ptdom.py:63 和 readout_ptdom.py:77 的 `GRID["none"]` 都是 1e-3 / 2e-3 / 4e-3 / 8e-3 / 1.6e-2，两边的 GRID / LADDER / ARMS 完全相同（jobs_cmp.py）。
- 作业文件：
  - 磁盘上正好 92 个，按 seed 优先排序，前 23 个都是 seed 0；
  - none 的 5 个 lr 各 4 个 seed；
  - 每个文件 ARGS 里的 `--lr/--seed` 和文件名一致；
  - 当前生成器在内存里渲染出的 92 份 = 磁盘文件逐字节，没有多余文件；
  - none@1e-3 和 none@8e-3 两个文件只差 lr、tag、out 路径和 intent 里的 lr。
- plan：A 92 / 119.6 L40S-h，nominal 120 / 156.0，worst 172 / 223.6，算术正确（X 最多 8 格 × 4，C 最多 (6+6) × 4）。
- 同 lr 的 LM 行（readout_ptdom.py:636-642）：
  - 两臂都跑过的每个 lr，计算 only-d@lr − none@lr；
  - 用离线 bpb ×1e3 和训练中 bpb ×1e3 两种量；
  - 按双方都 valid 的 seed 配对，符号正确；
  - X lr（例如 none@5e-4 和先验组的 5e-4）也会进来；
  - selftest 的已知答案 −5.00（8 个 seed）/ +10.00（4 个 seed）通过。
- 唯一的小问题是次要 7。
- lr 混淆本身已经不再单向偏向假 PASS：none 在下沿时 argmax 会触发外扩，在 TIE = 2 以内仍然靠 bpb 打破平局，这是两边对称的 F87 规则。代价见重要 2。

**2. 辅助 loss 闸门**：
- 字段读对了：
  - `grad_lm_share` = grad_lm / |g_lm + Σλ g_d|（pretrain_range.py:1988-1989）；
  - `grad_prior` = |Σ λ g_d|（加了权），所以 rb6 的重推公式成立；
  - ptdom 的 REVD = False（pretrain_range.py:510-511），`grad_lm_share` 不会在窗口内被改写成 1.0；
  - 真实 smoke JSON 窗口内的值是 0.083–0.099，窗口外 grad_prior 是 None。
- 只用窗口内的 eval：是（`window_split`：prior_on、有 grad_prior、发散之前）。CX4 和 CX5 验证了窗口外的 eval 和被跳过的 batch 都不进统计。
- 失败时：打印 AUX GATE FAILED、往 HOLD 追加理由（路径 = 作业检查的 `$R/HOLD`，也是生成器 `--submit` 检查的路径）、不写 picks、exit 2。这在 integrity 和 decide 之前执行（CX5 验证过）。
- 问题：聚合层级和阈值（重要 1）、早期检查的不一致（次要 3）、stage C 不过闸（次要 4）。

**3. 共享 site token 和唯一 site 读数**：正确。
- 我用第二条路径重新算了一遍（site_cx.py）：在每道题的完整 prompt 里用 offset mapping 找到 site token（训练器的规则），而不是单独 tokenize 名字。
  - 每个条目在所有出现位置的 site token 都一样，0 个例外；
  - 共享数量：stars 30/89（12 个 token）、landmarks 16/76（6 个）、languages 2/62、colours 0/24、cities 5/60；
  - 逐条目和 `anchor_meta` 的 site 完全相同（True）。
- 留出题里三个 site 都唯一的比例（seeds 0–3，split 由 ptdom_refs 的 split_seed 重推）：
  - stars 28–30%（seed 3 的 324 = A1 写的最小值）；
  - landmarks 49–52%；
  - languages 91–95%；
  - colours 100%；
  - cities 74–76%。
  
  和 A1.3 一致。
- `split_info` 的 `uniq` 是对锚点和两个选项的条目号取交集。`run_subsets` 用同一套 per-question hits 重算全集和 hard 子集，并要求和训练器记录的序列在 1e-9 内一致、面积在 1e-6 内一致，所以唯一 site 子集用的映射已经被交叉验证。
- stars 和 landmarks 的 NOT SHOWN 措辞实现了（readout_ptdom.py:496-498）。

**4. 训练频率捷径基线**：按 seed 计算，只用训练集。
- `split_info` 用的是每个 run 自己的 split_seed，`train` = 不在 held key 里的题，win 和 app 只在 train 上计数（readout_ptdom.py:286-306）。
- 和训练器的绑定：
  - 重推的答案频率 / 更短选项线必须在 1e-12 内等于训练器记录的值；
  - mask 必须逐位相等；
  - 两臂同一 seed 的基线必须相同，否则严格模式拒绝。
- colours 规则：比较 only-colours@pick 在 8 个 seed 上 hard area 的均值 > none 一侧 8 个 seed 的 h3_hard 均值，和 A1.4 一致（门槛含义见次要 8）。

**5. 次要项的修复**：

| 上一轮审查的条目 | A1 的处理 | 我的结论 |
|---|---|---|
| 7（分解双路只看窗口内；λ 从 $ARGS 取） | make_jobs_ptdom.py:273-278；`"$ARGS"` 作为第 8 个参数（作业文本第 80-81 行）；JARGS 解析和 `same_` 比较正确（`--lam 1` vs 1.0、`--save_model` 这类开关、多余或缺失的 flag 都会失败）；三个篡改用例失败正确 | 已修好（剩余见次要 5） |
| 9（double path 改名） | 改成 picks_consistency | 已修好 |
| 10（Tier L） | 锚点判词行和子行都带 [Tier L]，months 行写了 expected floor 和 Holm family | 已修好 |
| 13（months 留在 Holm 族） | 保留，行内写明 | 按用户决定 |
| 5、6、8、11、12 | 未决，在 A1.5 列出 | 如实记录，未发现新问题 |

**6. 旧的 rb6 读数和生成器**：仍然逐字节复现。我在 scratchpad 独立重跑了 reg_frozen.py 的副本，只改了输出目录：
- `make_rb6m_tasks.py --check`：96 个任务不变，0 个不同；
- 16 个 rb6_190m 作业文件逐字节相同；
- rb6_505m x1a 读数：我重跑出的 JSON 和 runs/rb6_505m/readout_505m_x1a.json 用 `cmp` 比**逐字节相同**；文本去掉 3 行页眉页脚后用 `diff` 比完全相同。
- snapshot 里 12 个文件的 sha256[:12] 都等于 WANT_SHA，readout_rules.py 是 e91ee8a8a846，没有改动。worktree 在 21cdc14，`git status` 干净。
- 修订者没跑完的两项（rb6none 回归、16 个参考格子）：训练器和 snapshot 的字节都没变，新 lr 的 DRY / step-0 和 seed 0 的参考相同，风险低。

---

## 对上一轮审查重要项的结论

| 上一轮条目 | 状态 |
|---|---|
| 1（lr 混淆） | 已修：网格重叠，argmax 在边上就外扩，有同 lr 的 LM 行。外扩规则推广过头，见重要 2 |
| 2（辅助 loss 闸门） | 代码里已经有了，写 HOLD 也正确。聚合层级和阈值比用户规则松，见重要 1 |
| 3（共享 site） | 已修（描述性的子读数 + 措辞），数字经第二条路径核实 |
| 4（colours 捷径） | 已修（按 seed 的基线 + NOT SEPARABLE）。门槛很保守，见次要 8 |

**提交前还剩的风险：** 主要是两件需要用户拍板的事。一是闸门按臂取中位数、阈值 0.5，比「先验梯度不得超过 LM 梯度」松，选中 lr 上的 run 可能处在被否决过的状态却仍然进入确认阶段；二是 A1 新的外扩/STOP 规则在 lr 响应平坦时会让 X 的成本明显超过 nominal，并让约两成的领域以 NOT FINAL 结束。其余都是次要或操作层面的问题。
