# pick_pt6v2.py 代码审查（exp-auditor，2026-11-19 ~11:30Z，sha 9cf7c6b3763f）

- **审查对象**：AUDIT.md 的「Composition pick rule」（05:55）和「Pick rule readings」（07:10，I1–I4）。
- **方式**：只读。审查用的脚本和输出都在 `/scratch/yh6210/tmp_audit/pick_pt6v2/`：raw_g.py、load3_check.py、r2_inventory.py、order_check.py、pick_partial_audit.txt/json。

## 结论
**阻断：三类都未发现。** 在现有数据上，每个候选都按登记的规则和读法算对了。

**主要风险在 toy3 一侧的完整性判定。**
- 脚本把 INVALID 当作「已完成」，而 toy3 只有 3 个 seed，没有余量。
- 后果：一个复现 run 出问题，就足以把 torus 静默剔出，脚本照样 exit 0、交出 fallback 组合，中间没有人看。

## 已核实（独立检查）
**指标**
- Part R：对 landmarks / compass / ranks / scalar_adj 共 31 个 seed 对，直接从原始 run JSON 重算 per-seed g（取 metrics[<d>_readout].area，用最高的 -r<k>），与脚本输出逐个一致。
- toy3：在真实 none@1.6e-2 s0 上，t_val(hcube) 取到的是 tf_area 75.68（full area 是 79.70）；torus 取 full area 74.44。两者都与 readout_ptdom3 一致。

**解析**：在一棵合成 run 树上测了 load3。
- s1 / s2 / s15、-r2、lr 3.2e-2、tag 带 -s<seed>，都解析正确。
- prior_domains 不符的记为 INVALID。
- 没有 JSON 的更高 -r<k> 目录记为 UNFINISHED，会遮住下一级的有效 run（偏保守）。
- Part R 证据 seed 8–15 下的 21 个 (batch, arm, lr) 全部 valid。

**规则实现**：下列各项都符合登记。
- p / n 的选法（I1 present，I2 打平取高 lr）；
- 配对 seed 取交集；
- E4 `3*wins >= 2*k`；
- E5 用 ddof=1，以及 inf 的处理；
- 排序键 (−d, −mean g)；
- fallback 顺序 people → months；
- V2 只看入选者的 n 是否等于 3.2e-2。

手算 ranks 的 d = 2.04 / 6.68 = 0.31，与脚本一致。

**组合与下游**
- comp_doms 会按 TRAINER_ORDER 排序，所以任何顺序都能接受；脚本自带的 fallback 顺序就是 TRAINER_ORDER 限制在 ALLOWED 内的顺序。
- 严格模式下有 problem 时不写 JSON；auto 遇到 exit 2 就停。

**复现**
- `--allow_partial` 75 s 跑完，rc 0。
- 结果与 tests/pick_partial_E3src_20261119T1110.txt 相同，唯一差别是多了已经跑完的 compass s15（8 seeds，+2.76，4/8，d 0.34）。
- 现在没有合格候选，组合是 months, dim, cities, countries, people，V2 不触发。

## 重要
1. **toy3 把 INVALID 当作已完成，且 3 个 seed 零余量**（pick_pt6v2.py:394；selftest :763 明确断言了这一行为；:183-186 中 t_val 对非 OK 状态返回 None）。
   - 后果：以下复现 cell 中只要有一个是 INVALID（身份不符、check 失败、tf 读不出），配对 seed 就降到 2，E2 不过，torus OUT，脚本 exit 0 交出 fallback 组合：only-torus@1e-3 s1/s2、none@1.6e-2 s1/s2、none@3.2e-2 s0–2。
   - n 恰好取到覆盖少的那个 lr 时，k 同样会掉（:278 取的是交集）。
   - Part R 遇到同样情况会被 I4 拦住，两侧处理不对称。
   - 检查：交组合之前，看 toy3 各行的 k 是否都是 3，以及 replication 行里有没有 INVALID；或者让 toy3 的必需 cell 也走 I4（INVALID 且尝试次数 < 2 就拦）。
   - 置信度：读代码确认。
2. **竞态（与第 1 条同一根源）**：ptdom3 的 sbatch 先跑 trainer、写出完整 JSON（第 87 行），之后才跑 intent check 写 check.json（第 90–92 行）。
   - 这段时间里 RO3.status 返回 INVALID「no check.json」（readout_ptdom3.py:226-228），pick 把它当作已完成，静默丢掉这个 seed。
   - 如果作业在 check 阶段被抢占后重新排队，这个窗口会一直持续到重跑结束。
   - toy3 一侧完全不查 sacct。只在整批 sacct 都结束后才跑 pick，就能避开。
   - 修法：把「no check.json」当作 UNFINISHED，或者用 submitted.tsv 里的 job id 查 sacct。
   - 置信度：读代码确认；会不会触发取决于时机。
3. **活性**：toy3 没有「失败两次」的豁免（:359、:394）。
   - TIMEOUT、未重排的抢占、写到一半崩溃都会停在 UNFINISHED，严格模式会一直拒绝，直到有人重交。
   - 如果 only-torus@2e-3 s0 胜过 1e-3，I3 会要求 only-torus@2e-3 s1/s2，但 submitted.tsv 里只交了 @1e-3 s1/s2。这时脚本会拒绝并列出缺的 cell。这是拒绝，不会选错。
4. **E3 在哪里都不起约束作用。**
   - toy3：src_grid 写死了 3.2e-2（:246），top 恒 ≥ 3.2e-2，E3 恒为真（:315-317）。
   - 按 07:10 的读法（来源网格 ∪ present），none@3.2e-2 不 valid 时，n=1.6e-2 已在网格顶上，E3 应该不过；代码却让它过。
   - Part R：现有 cell 下所有 n 都低于 R1 网格的顶，E3 也不可能不过。
   - 目前不改变任何判定。
5. **selftest 38/38 没有走真实路径**（:586-601）。
   - fake 数据没有 sub 字段，所以 Part R 真正起约束的 shortcut-baseline FLOOR 线没被测到。
   - fake 里 area 等于 tf_area、chance 等于 chance_u，键用错了也测不出来。
   - load3、load_all、markers、sacct 都是注入的。
   - 上面的独立检查补上了现有数据下的这些缺口，但对以后的代码改动没有防护。
6. **第 10 问：已不可能改变判定的部分。** 5 个 Part R 候选已全部锁定为 OUT，不受仍在 PENDING 的 8 个 s15 作业影响：
   - landmarks、compass 已经有 8 个 seed；
   - prefixes 的 E4 现在 3/7，要过需要 6/8；
   - ranks、scalar_adj 即使 s15 area = 100，p 均值最多 94.1 / 89.3，仍低于 FLOOR 线 97.9 / 91.7。

   严格模式还在等 s15，以及 picks C 里那些从不进入 g 的 -shuf cell，这只会拖延时间。pick 现在只取决于 toy3，其中还活着的只有 torus。因此 I4 把运维 CANCELLED 的 r0 算作一次尝试（:443），对 pick 没有影响。等不等由主会话或用户定（登记要求 R2 完成）。

## 次要（合并）
- n 在证据 seed 上对 3 个 none lr 取 argmax（同一批样本上做选择），g 因此偏向不利于 prior。例如 prefixes 用 4e-3 得 −3.8，用 R1 的 1e-3 约为 −1.8。这是按登记规则。
- t_chance 取的是第一个 OK 记录（:190-195）。
- auto_read_pick.py 没有断言 complete / mode。
- markers 的 glob 用前缀匹配，s1 会匹配到 s15，但随后被锚定的正则过滤掉，没有问题。

## 结尾三行
- **最可能测错的地方**：toy3 一侧把 INVALID（包括 trainer 写完 JSON、check.json 还没写出的那段窗口）当作「已完成」，而 3 个 seed 没有余量。任何一个复现 run 出问题，torus 都会被静默剔出，脚本 exit 0 交出 fallback 组合。
- **可能被测没或写没的真效应**：
  - ranks（+2.05，6/7 胜，d 0.31，sd 主要来自 s13 的 −11.6）和 scalar_adj（+1.77，5/7）被 shortcut-baseline FLOOR 线和 d ≥ 0.5 挡掉；
  - seed-0 单 seed 筛选不合格的 toy3 域永远拿不到复现；
  - n 在同一批样本上取 argmax，使 g 偏低。

  这些都是按登记规则，只报告。
- **证据撑得住的最强结论**：在现有数据上，脚本逐项正确实现了登记的选择规则，31 个 Part R seed 对从原始 JSON 复现一致。5 个 Part R 候选不论 s15 结果如何都已 OUT，所以 pt6v2 的组合要么是 fallback（people + months），要么由 toy3 在 seed 0–2 上过线的域决定。目前只有 torus 一个线索（seed 0 上 +8.5，单 seed）。
