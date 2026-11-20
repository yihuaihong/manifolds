# pt6v2 Amendment V3 diff 复核（exp-auditor，2026-11-19）

审查对象：readout_pt6v2.py 从 bddf63acdf27 改到 cae52696322d，render_findings_pt6v2.py 从 20c106615baa 改到 995e97c66de6，新增 POSTHOC_V3.json 75bed7fe30ee 和 posthoc_v3_jobs.py 5457ba5d065f。make_jobs bb15d248afd8 和 CELLS.json ea9a9b01f081 没有改动。独立检查脚本是 /scratch/yh6210/tmp_audit/pt6v2_V3/check_v3.py，输出在 check_v3.out，79/79 ok。

**结论：没有阻断项，也没有重要项。** flag off 的输出没有变；flag on 的扩展数字和按登记规则独立重算的结果逐项一致；diff 只带来 4 条次要问题，都不改数字。

1. **flag off 不变**（实跑确认）。none@6.4e-2 的 s16、s17、s23 三个目录已经落盘。
   - off.txt 和 tests/readout_final_20261119T2001.txt 只差 2 行 sha；off.json 和最终 JSON 用 cmp 比对完全一致。
   - 新渲染器在 off.json 上的输出，和冻结的 render 20c106615baa 字节一致。
   - 原因：split_posthoc（readout_pt6v2.py:124, 912）把网格外的 none@6.4e-2 记录从 R_ 里拿掉；readout_pt6v2.py:915-916 的 glob 排除了 NL+PH。所以登记读数既不含这些 run，也不会因为它们被 strict 拒掉。
2. **扩展数字按登记规则计算**（实跑确认）。
   - 做法：用真实登记记录，植入 none@6.4e-2（从各 seed 的 none@3.2e-2 克隆后平移面积），和一份按 AUDIT.md + I2 独立写的实现比对。独立实现覆盖：argmax、平局取较高 lr、配对排名、逐 seed 回退、单侧 t（df 7）、wins ≥ 6、6 项 Holm、final3、boot90。
   - 六个场景：
     - B：8 个全有效。dim 精确平局，选 6.4e-2；cities g 从 +6.3 变成 −2.0，变号并失去 HELPS；box5 从 +19.7 变成 +14.1。
     - C：6 有效 + 2 缺失。
     - D：6 有效 + 2 无效。
     - E：4 有效 + 4 缺失。
     - F：4 有效 + 4 无效。
     - G：5 有效 + 3 缺失。
   - 结果：N_m、N_per_seed、g、wins、p、Holm、HELPS、final3、boot90 以及 sign_flip / lost_helps 全部一致。E 和 F 有效 seed 不过半，6.4e-2 出网格，扩展结果等于登记结果。
3. **PROVISIONAL 判定正确，登记块不受影响**（实跑确认）。
   - 有缺失时（C、E、G 和真实现状）标 PROVISIONAL，headline 带标签。
   - 无效时（D、F）不标 PROVISIONAL，该 seed 回退到它其他有效的 lr，和 I2 一致。
   - 每个场景跑完，res_r 的 JSON 快照都没变。
   - 接线检查：monkeypatch load_v2，给 readout(posthoc=) 喂 8 seed 全有效的植入数据。结果：块 1-2 文本等于 flag off 文本；JSON 去掉 posthoc_V3 后等于 flag off JSON，也等于冻结的最终 JSON；块 3 等于 extended_from 的输出。
4. **selftest**：51/51 checks，mutants 25/25 killed，耗时 37 s，跑完 tests/ 下没有残留的 mut_ 目录。3 个新测试场景和 3 个新 mutant 分别覆盖了三条路径：split、变号 / 失去 HELPS、缺失时标 PROVISIONAL 并回退。
5. **作业**（读文件 + sacct 确认）。
   - 8 份作业文本把 lr 和 slot 归一后，和同 seed 的登记 none@3.2e-2 作业逐字相同；sha12 和 POSTHOC_V3.json 一致；设备和 ALLOC 一致。
   - sacct 只多出 23783618-25 这 8 个作业。登记作业文件（mtime 11:55）、CELLS.json、make_jobs 都没改。
6. **真实现状**（20:4x 重读）。
   - s16、s17 有效：code_sha / data_sha / lm0 和同 seed 的 none@3.2e-2 一致，posthoc_problems 为空。
   - s23 还没写 check.json；s18-s22 在跑或排队。
   - 结果：2/8 有效，6.4e-2 出网格，扩展结果等于登记结果，标 PROVISIONAL。

次要（都不改数字，置信度都是读代码确认）：
- (a) render_findings_pt6v2.py:128-142：render_v3 的表头总是写「none 网格 [...,'6.4e-2']」，但不打印 extra_none 的 n_valid / in_grid / not_valid。如果最后有 ≥4 个 6.4e-2 run 无效（不是缺失），渲染出的表不带 PROVISIONAL，「变化」一栏写「无」，看起来像加了 6.4e-2 也没变化，实际是 6.4e-2 被 I2 剔出了网格。文本块 readout_pt6v2.py:886-891 有这句说明。检查办法：用场景 F 的 JSON 跑一次 render；建议在表里加一行 I2 状态。
- (b) readout_pt6v2.py:859：标签写「missing / invalid」，但只列缺失的 run，无效的 6.4e-2 不进这张表。
- (c) 6.4e-2 在网格内但只在 5-7 个 seed 有效时，I2 的配对排名只用这些 seed，登记 lr 之间的 argmax 也可能随之改变，块 3 会把它报成和登记结果不同。这是登记的 I2 规则本身的性质，N_m 在输出里看得到。
- (d) selftest 的 V3 部分直接调用 extended_from。readout() 的接线（spec、glob 排除、x_terminal(R_all)、posthoc_problems），以及 V3 下「无效不标 PROVISIONAL」「有效不过半出网格」两条路径，都没有 mutant 守护。本次 check_v3.py 已覆盖；而且接线一旦出错，表现是满屏 missing 的 PROVISIONAL，不会悄悄出错。

- **最可能测错的地方**：没发现数字算错。最可能误读的是 6.4e-2 因 ≥4 个 run 无效被 I2 剔出网格时，渲染表不说明这一点，「变化 无」会被读成「扩展网格不改变判定」。
- **可能被测没或写没的真效应**：未发现。扩展结果和登记结果走同一个 verdict_v2、同一组门槛；PROVISIONAL 只加标签，不拦读数。
- **证据撑得住的最强结论**：V3 代码可以直接用来出扩展读数。flag off 的 JSON 和渲染结果都与已审查的最终读数字节一致；flag on 在 7 个植入 / 真实场景下，N_m、g、wins、p、Holm、final3、boot90 和变化标记都与独立重算一致。目前 none@6.4e-2 只有 2/8 有效，扩展块等于登记块（6/6 HELPS）并标 PROVISIONAL；等 8 个 run 全部结束再读。
