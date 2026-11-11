# ptdom Amendment A2 对抗审查（全新审查者，只读，2026-11-04）

**范围**：AUDIT.md 的 Amendment A2，以及 runs/ptdom/{select_ptdom.py, readout_ptdom.py, aux_gate_check.py, submit_ptdom.sh}、make_jobs_ptdom.py、92 个 stage-A 作业文件。逐条对照 tests/audit_a1.md 的重要 1–2 和次要 3–9。

**纪律**
- 除本文件外没有写任何文件，没有提交或取消作业。
- 所有反例都在内存里调用真实函数：`cell_gate`、`aux_gate`、`decide`、`select_unit`、`readout`，以及 `select_ptdom.main` 和 `aux_gate_check.main`。
- 跑两个 main 时，写盘调用（picks 文件、HOLD）被拦截到内存里；全程设了 `PYTHONDONTWRITEBYTECODE=1`。
- 读了修订者 scratchpad 里的 A1 副本（`jobs_A1/`、`make_jobs_ptdom.A1.py`），只读。
- 结束时 runs/ptdom 下没有 out/、HOLD、SUBMITTED_ 或 picks_* 文件。squeue、sacct、llmadd_torch/logs 和 Burst results 里都没有 ptdom 作业。
- 审查期间（04:21–04:31），另一个会话往 tests/ 写了几个文件：a2_verify.py、a2_verify_run.sh、a2_jobs_vs_a1.out、a3/（新领域 units / zodiac / greek / planets 的 tiny 运行）。这些都不是我写的。
- 被审查的四个脚本、make_jobs_ptdom.py 和 92 个作业文件在 03:30 之后都没有改动。
- AUDIT.md 在 04:31:29 被另一个会话改过一次，只增加了 138 字节，正好等于 Status 段两处 [A2] 标注的长度，也就是次要 8 已经被修掉；其余内容就是本报告审查的 A2 版本。
- 如果 A3 会改动这些文件，需要重新审查。

**总判断：没有发现阻断问题。**
- A1 审查的两个重要项在代码里都修好了，用我自己的反例逐条跑过。
- 剩下 1 条重要问题：A2.2 的成本表用的噪声偏小，预期成本和 NOT FINAL 比例都偏乐观。这只需要改文档，不动代码和作业文件。
- 另有 7 条次要问题。

---

## 问题清单（按严重程度排序）

### 重要 1：A2.2 成本模拟的噪声范围（4 个 seed 均值的 sd σ = 0.5–1.5 分）低于 53M 实测，所以预期成本和 NOT FINAL 比例偏乐观
- **位置**：tests/a2_cost_mc.py:38-42（SCEN）；AUDIT.md:456-489（方法、表格、"NOT FINAL 0–1.3%"、"166–181 L40S-h"）。
- **为什么**：现有 53M 数据里，4 个 seed 均值的 sd 普遍在 2–3 分以上。
  - pt2b none 的 numbers area（rb6m3 results 350–365；5 个 lr × seeds 0–3）：
    - 4-seed 均值的 sd 在 0.6–4.0 之间，中位约 2.4。
    - 相邻 lr 之间，4-seed 均值之差的 sd 为 1.5–5.9，约等于 √2·σ。也就是说 seed 效应在不同 lr 之间基本不共享，配对抵消不了。
  - rb6 53M 可测的领域读数（only-s4，seeds 0–3）：s4 2.6，months tier E 3.0，clock 6.0，dim 8.2。
- **我的重跑**：用修订者的模拟器（真实 `decide()`，300 次迭代）。

  | 设定 | prior P(X) | NOT FINAL | E[L40S-h]（p95） |
  |---|---|---|---|
  | 平坦，σ 2 | 0.33 | 0.037 | 184.8（197.6） |
  | 平坦，σ 3 | 0.47 | 0.065 | 189.5（202.8） |
  | 弱峰，σ 3 | 0.41 | 0.034 | 186.8 |
  | confound，σ 3 | 0.42 | 0.106 | 186.7 |
  | 对照：A1 规则，σ 3 | 0.66–0.67 | 0.18–0.27 | 191–192 |

- **结论**：
  - A2 的规则本身没问题：在高噪声下，NOT FINAL 的比例仍然比 A1 低 3–5 倍。
  - 但写进 AUDIT、给用户看的"预期 166–181 h、NOT FINAL 0–1.3%"偏乐观。按实测噪声，更接近 185–190 h（p95 约 200 h）、4–11%。
- **建议**：表格补上 σ 2 和 σ 3 两行，并告知用户。只改文档，不动代码。
- **置信度**：模拟结果高（真实 decide）。ptdom 锚点读数的真实 σ 未知，所以具体数值为中。

### 次要 2：stage C 的确认 seed 只按 8 个 seed 合并取中位数过闸。确认 seed 全部是先验主导时，照样出 PASS（A1 次要 4 只修好一部分）
- **位置**：readout_ptdom.py:647（`pg = cell_gate(...)`，选中格子 seeds 0–7 合并）、:711-713；AUDIT.md:433-434（"so the confirm seeds are gated too"）。
- **反例**（真实 `readout()`，各领域都植入 +19.8 的增益）：

  | 领域 | seeds 0–3 | seeds 4–7 | 8 seed 中位数 | 结果 |
  |---|---|---|---|---|
  | stars | 0.9 | 0.2（b = 4a） | 0.55 | **PASS** |
  | cities | 0.9 | 0.12（b = 7.3a） | 0.51 | **PASS** |
  | languages | 一半是 0.2，一半是 0.85 | 同左 | 0.525 | **PASS** |
  | landmarks | 0.9 | 0.05（b = 19a） | 0.475 | "AUX GATE FAILED at the pick"（正确） |

  side effects 那一行对前三个领域都打了 "FLAG: the prior dominates the update in a seed"，所以不是悄悄放过。
- **为什么是次要**：seeds 0–3 和 4–7 是可交换的，系统性不同的可能性低。但 AUDIT 的说法比实际强。
- **建议**：可以仿照判词的 halves 条件 (b)，在选中格子上对 seeds 4–7 单独再判一次中位数。需要用户决定。
- **置信度**：机制高，实际发生的可能性低。

### 次要 3：每格取「run 中位数的中位数」，一个 cell 可以有一半 seed（或每个 run 里近一半的 eval）是先验主导，照样过闸
- **位置**：readout_ptdom.py:160-169（`cell_gate`）；AUDIT.md:412-413。
- **反例**：
  - CX-A：4 个 run 的中位数为 [0.2, 0.2, 0.8, 0.8] → cell 中位数 0.5 → 通过。在 `decide()` 里 only-stars@1e-3 仍可入选并被选中。
  - CX-B：每个 run 的 26 个窗口内 eval 里，12 个是 b = 19a，14 个是 b = 0.67a → run 中位数 0.6 → 通过（b > a 的 eval 占 46%，cell 行里会打印）。
- **为什么**：用户原则写的是 "the LM objective must dominate every update … flag any arm where the prior gradient exceeds the LM gradient"。
  - flag 这部分做到了：cell 行打印 b > a 的比例，读数里有 FLAG。
  - 拦截只看中位数。这是协调者在 A2.1 明确选的口径，不是 bug。
- **置信度**：高（按设计）。

### 次要 4：watcher 把「没有任何窗口内 eval」的 run 一律当成「窗口还没开始」，不看它已经跑到第几步
- **位置**：aux_gate_check.py:59-62；AUDIT.md:440。
- **反例**（真实 `aux_gate_check.main`）：
  - 一个已跑完（step 18309）却从来没有 prior_on 的先验 run 被列为 "before the window (not judged yet)"，exit 2；
  - 加上另一个正常的 cell 后，exit 0（"passed so far"）。
- **影响小**：select 的闸门（"no on-window eval" → 该 cell 失败）和 CHECK_PY（prior_events / prior_on_frac）都会拦下它。
- **建议**：只有 last_step < 1000 才算 "before the window"，否则判失败。
- **置信度**：高。

### 次要 5：watcher 的 exit 1 →「touch HOLD」比 select 严，而且可能只依据 1–3 个最早的窗口内 eval
- **位置**：aux_gate_check.py:14、:87-88；submit_ptdom.sh:28-30；AUDIT.md:301-306、438-440。
- **为什么**：
  - 按 A2，select 遇到单个失败的 cell 只是把它排除，不写 HOLD。
  - watcher 却让协调者在任何一个 cell 失败时就 touch HOLD，这会让所有排队中的作业退出，之后要走 -r1 重投。
  - 反例：step 1500 时前 3 个窗口内 eval 是 0.30 / 0.35 / 0.45 → exit 1。
- **实测风险低**：rb6 53M 单先验（λ1、lr 1e-3，rb6m3 438–461 共 23 个 run）：
  - 前 3 个窗口内 eval 的中位数是 0.58–0.83，所以按 rb6 的拟合速度，step 1500 时不会误报；
  - 整个窗口的中位数是 0.82–0.92。
  - 锚点先验的拟合速度未知。
- **建议**：可以写明「exit 1 先报告用户，是否 HOLD 由用户定」，或者至少等窗口过半再据此 HOLD。操作层面的问题。
- **置信度**：机制高，误报概率中低。

### 次要 6：排除 cell 之后，bracket 会跨过被排除的格子比较；只剩一个可选格子时，没跑过的那一侧不会外扩
- **位置**：readout_ptdom.py:480-513（`select_unit`）。
- **反例**：
  - only-stars@1e-3（面积最高）失败 → 可选格子是 [5e-4, 2e-3]，两者相差 4 倍 → 2e-3 比 5e-4 高 5 分，判 open upper → X 4e-3。这个 X 多半是浪费：峰在被排除的 1e-3。
  - 5e-4 和 1e-3 都失败 → 只剩 2e-3，面积还在朝 2e-3 上升 → C，4e-3 从来没跑过。
- 两种情况都对 PASS 偏保守，只影响成本或者 lr 是否最优。
- **置信度**：高，实际发生的可能性低。

### 次要 7：重投路径有一个死角：训练已跑完，但没有 check.json
- **位置**：make_jobs_ptdom.py:571-573（拒绝重投已跑完的 run）；readout_ptdom.py:290-291（没有 check.json → invalid）。
- **为什么**：如果训练结束后在 offline_lm 或 CHECK_PY 阶段崩溃，会出现两头都走不通：
  - `--rerun` 拒绝重投（"a run is data"）；
  - select 又因为这个格子缺 valid run 而拒绝。
  - 现在没有工具单独重跑第 3–4 步。
- **发生概率低**：--time 3 h 对 1.3 h 的计划；被抢占后 `--requeue` 会从 checkpoint 续跑，再走完第 3–4 步。
- **建议**：在 AUDIT 里写一句手工补跑 offline_lm + CHECK_PY 的步骤。
- **置信度**：高（只是操作层面）。

### 次要 8（审查期间已修）：AUDIT 开头 Status 段（AUDIT.md:5-6）仍然写着 "an area argmax at a grid edge also triggers the extension"
这是 A1 的规则，已被 A2.2 取代，但这里没加 [A2] 标记。其他被取代的行都标了。只是措辞问题。
04:31 另一个会话在这里补上了 "[A2: only none's argmax at the LOWER edge at stage A; otherwise the 2-point margin, A2.2]"，gate 那一项也补了 [A2] 标注。现在已经正确。

（备注，不算问题：CHECK_PY 的分解检查改成取 max 之后，GPU 上的证据来自 Rev D 的路径。ptheld2t 的 72 个 run、5112 个先验生效的 eval 里，最大值是 7.10e-4，这个数我独立重算过，一致。ptdom 走的是 Rev E 的 `dom_grads` 路径，还没在 GPU 上跑过。万一超限，那个 run 会判为 invalid，不会产生悄悄出错的数字。所以第一个跑完的作业的 check.json 要看一眼。）

---

## 逐题核对

### 1. 辅助 loss 闸门（A1 重要 1、次要 3、次要 4）

| 要求 | 结论 | 依据（file:line + 我的反例） |
|---|---|---|
| 按 (arm, lr) 格判 | 已修 | `aux_gate` readout_ptdom.py:179-195；`decide` :522-526 排除失败格；select_ptdom.py:58。在内存里重做 CX1：1e-3 被排除，选点改变 |
| 用 a/(a+b) | 已修 | `window_split` :121-125。b = a 正好通过；0.4999 失败；反向梯度 b = 1.2a、cos −0.9 → 0.455 失败（A1 的统计量是 1.89）；b = 1.5a、cos −0.99 → 0.40 失败 |
| 只用窗口内 eval | 已修 | :118（prior_on，且在发散之前）。窗口外的 eval 即使 b = 100a 也被忽略；发散之后的 eval 不计入 |
| 缺字段算失败 | 已修 | :120、:147-157。只在 1 个 seed 的 1 个 eval 上缺 clip_coef → 失败；缺 grad_cos → 失败；grad_prior 为 NaN（lm 有限）→ 失败；logged share 偏 2e-3 → 失败，偏 5e-4 → 通过；没有 prior_on → select 判失败（watcher 的问题见次要 4）。watcher 现在也用 `cell_gate`（A1 次要 3 已修） |
| 选点前检查 | 已修 | stage A 和 AX 都检查，AX 时也覆盖 X 格。反例：X 格 4e-3（面积 70）的 share 是 0.3 → 被排除，"closed by the gate"，C 回到 2e-3 |
| 读判词前检查，含 stage C | 部分修好 | :647、:711-713 对选中格的 seeds 0–7 合并取中位数。确认 seed 全部是先验主导时照样 PASS，见次要 2 |
| 全臂失败 | 正确 | 跑真实的 `select_ptdom.main`：exit 2；HOLD 写明三个格子的理由；picks 照样写出，GATE = [none\|months, only-months\|months]；months 不进 C；其他单元不受影响 |
| 修订者请我确认的两点 | 可以接受 | ① 全臂失败时仍写 picks：写 HOLD 后，生成器会拒绝 `--submit`，其他单元的决定也与这一臂无关。② 下一个 lr 是失败格时那一侧算关闭：只会对 PASS 偏保守。代价见次要 6 |

**真实数据**：
- 冒烟 JSON 里 stars 的 a/(a+b) 是 [0.0901, 0.0770, 0.0896]，中位数 0.0896，与修订者的数字一致；双路差 1.4e-6。
- rb6 53M 单先验（λ1，lr 1e-3）：窗口中位数 0.82–0.92，b > a 的 eval 占 0–4%，都会通过。

### 2. 外扩规则（A1 重要 2）
- **已修**。`extend_rule`（readout_ptdom.py:452-471）只认 bracket 的 'open'（2 分余量）；none 下沿 argmax 的例外只在 stage A 生效（:503 `none_lower=(arm == "none" and not ext)`）；X 之后的 STOP 只看余量。
- 在真实 `decide()` 上逐条跑过：

  | 情形 | 结果 |
  |---|---|
  | none 的 argmax 在下沿，领先 0.5 | X 5e-4 |
  | none 的 argmax 在上沿，领先 0.5 | C |
  | none 的选点在上沿，开口领先 3.5 | X 3.2e-2 |
  | 先验臂的 argmax 在下沿，领先 1 | C |
  | 先验臂的选点在下沿，开口领先 3.5 | X 2.5e-4 |
  | X 之后 5e-4 比 1e-3 高 1 | C |
  | X 之后 5e-4 比 1e-3 高 3 | STOP |

- 生成器那边，`cells('X')` 断言 X 的 lr 在完整网格边外一步。我核对过：排除格子之后，X 的 lr 也不可能落到网格内部。
- **成本**：方法合理（真实 decide；A1 规则的对照能重现 A1 审查的数字），但 σ 范围偏小，见重要 1。

### 3. 次要 5–9

| A1 条目 | 结论 |
|---|---|
| 5（分解检查用中位数） | 已修：作业文本里是 `max(rd) < 0.05`；GPU 7.1e-4 重算一致 |
| 6（HOLD 后的重投） | 已修。<br>• `rerun_plan` 逻辑：最新的名字有 flag、不在队列里（按 id 或名字查，squeue 失败就拒绝）、没有跑完的 run JSON → 下一个 -r 名字。<br>• `--submit` 拒绝有 flag 的名字。<br>• 日志名 `%x_%j` 和 rerun_plan 拼出的 `jname_jid` 一致。<br>• loader 读最新目录（NAME 正则认 -r）。<br>• integrity 拒绝早期目录里已跑完的 run。<br>• 我在内存里渲染 rr=1：与 rr=0 恰好差 7 行，intent 相同。<br>• 死角见次要 7 |
| 7（同 lr 的 LM 行） | 已修：`pair_diff` :742-757，发散的 run 记为 ±inf，n 计入 |
| 8（colours 规则） | 已修，并且确实在任何结果之前登记（没有 out/、SUBMITTED_、日志、sacct 记录或 Burst 结果）。<br>单位一致：metrics 的 final3 是 ×100（pretrain_range.py `curve_metrics` 的 "final3": 100 × …），h3_hard 也是 ×100（readout_ptdom.py:427）。hard3 缺失时 → NOT SEPARABLE（保守） |
| 9（日志措辞） | 已修：a1_refs_full.out、reg_compare_a1_partial.out、reg_trainer_a1.out 都重新标注，并保留了原句 |

### 4. 作业文件（与上一版逐字节对比）
**有变化，而且变化恰好是文档里写的那些。**
- **A1 副本可信**：修订者 scratchpad 的 `jobs_A1/`（92 个）= `make_jobs_ptdom.A1.py` 在内存里渲染的结果，92/92 逐字节相同。
- **当前文件 = 当前生成器**：磁盘上 92 个文件 = 当前生成器在内存里的渲染，92/92 逐字节相同。
- **差异一致**：92 个文件与 A1 版的 diff 完全一样（diff 内容的 sha 都是 c19a73d2ed43），9 行，包括：
  - 注释 1 行 → 2 行；
  - 检查名里 median 改成 max；
  - `statistics.median(rd) < 0.05` 改成 `max(rd) < 0.05`；
  - info 里新增 `sn` 和 `lm_share_norm` 两行。
- **其余不变**：两个生成器的 cells、ARGS、intent 逐项相同；header 和输入检查没有变化。

---

## 现在能不能提交
**能提交。没有阻断项。**
- 建议提交前把 A2.2 的成本表补上 σ 2–3 两行（预期约 185–190 L40S-h，NOT FINAL 4–11%），并告诉用户。这只改文档，不动代码和作业文件。
- 次要 2–8 都不影响 stage A，可以在 stage A 运行期间按用户的决定处理。
