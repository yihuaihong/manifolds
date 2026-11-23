# V7b 结果审查（pt6v2 Amendment V7b：all6dropS = all6@2e-3 从 7324 起把 torus/box5 行原位换成等长正文，s16–23），2026-11-23，只读
独立重算：自写 loader，只沿用 dim strict 的定义（未导入 v7bread / v7read / v4final），断言四臂 eval 步、held 集、strict mask 逐 seed 相同。脚本和输出：/scratch/yh6210/tmp_audit/v7b_result/{recompute.py, recompute_out.txt, pooled.py, pooled_out.txt, extra.py, extra_out.txt}。

## Q1 数字：全部复现，没有算错
- 20.4（+20.38，sd 9.67，8/8，t p .001）；82%（0.817，bootstrap [0.645, 1.062]）；V7 的 0.765 [0.499, 0.958]；V7 − V7b −1.29 [−7.65, +5.33]，+4/−4；代价 4.57 [−1.17, +10.34]，+5/−3。CI 末位和 v7bread 差 ≤0.1，来自 bootstrap 的 RNG，不算错。
- V7b 只改了它声称要改的东西：
  - args 与登记 all6 只差 out / drop_domains_after / drop_fill，与 V7 只差 out / drop_fill；
  - 30 个 drop 前 eval 的 data_sha 三臂（all6 / V7 / V7b）全同，45 个 drop 后 eval 全不同；
  - 75 个 eval 上，kept 三领域的 slot_frac 和 dom_sites 与登记 all6 完全相等（差 0），所以 dim slot-1 = 0.4997；numbers / 三个 kept 领域的 token share 与 all6 相等；全部类型（含 text）的 token share 与 V7 相等；
  - fill 281242 行 = 2×140621 = cycling rule（10986 步 × 64 × 2/5）；fill 共 1687452 token，每行 6.0；drop 后 torus/box5 位点为 0；
  - 位置不变、filler 是正文：由冒烟 compare.txt（7324–7329 逐 token、逐位点）加代码确认（pretrain_range.py dom_row 中 `n` 不变，fill_ids 长度 = 被替换行）。
- 规则对不上：预写规则在两个 CI 都含的情况下判「F109 的 77% 不变，限定 2 换成实测 slot 效应」，计划句却以 82% 为首句数字。要么照规则写，要么改用下面的合并值，并注明偏离了规则。

## Q2 考虑项
1. **「不再能和零区分」写弱了真效应**（假阴性写法）。两次的代价是一致的：5.9 对 4.6，差 1.3 ± 6.5。V7b 显著性翻转，只是因为它的配对 sd 更大（8.95 对 7.18）。n=8 时，真实代价为 5 点的检验功效只有 27–40%，80% 功效的最小可检出效应是 8–10 点。逐 seed 合并 V7 和 V7b 后：代价 +5.2，bootstrap [+0.97, +9.09]，+6/−2，t p .052；保留比例 0.791 [0.647, 0.955]，上界 < 1。「at most a small part」作上界不成立：保留比例的下界 0.65，等于丢掉的部分最多可到 35%。
2. **合并的依据**：V7 和 V7b 共用同一组 all6、only-dim 分母；drop 前数据相同；slot 效应为空。所以两者是对同一个保留比例的两次重抽，逐 seed 平均能压低重抽噪声。合并后的数：+19.7 高于 only-dim（7/8），保留 0.79。
3. **slot 对照比 0.50 对 0.75 更强**（读代码确认：行内两行顺序由 order_rng.permutation 随机；V7 末步累计 dim slot-1 = 0.6506，反推 drop 后 0.751 = 0.25 + 0.50 单独成行）：
   - dim 行只和 cities、box5 同行。V7 里 box5 行被删后，dim 单独成行，排在 numbers 之后、行末，和评测布局（正文 + numbers + 题目，题目在 510–511）完全一样。drop 后一半的 dim 行处于这种布局，而登记 run 和 V7b 是 0%。
   - 即使这样，V7 仍比 V7b 低 1.3 点。I1 担心的方向（V7 被抬高）没有出现。
   - 「did not inflate」宜改成「没有可测的影响」：CI 上界 +5.4 点，约合 0.22 的份额。
4. **filler 本身的混杂**：
   - 它只改变一处 dim 的上下文：原 box5 行里排在 slot 2 的 dim（占 25%），前一行由 box5 行变成 6 token 的断句正文（dim 在 slot 1 时后面换成什么，在 causal attention 下不影响它自己的 loss），第二个 seed 流（7100+seed）只用来抽 filler。
   - 如果 filler 有害，S_b 会被压低，而不是抬高；实际 V7b ≥ V7。
   - bpb：V7b − V7 = +0.0079（6/8，p .045），但 s16 的 V7b 在 drop 前（7250）就已比 V7 差 0.033，是重抽差异，filler 的 LM 代价因此分不出来。V7b − all6 = +0.0025，是 null。
   - torus 掉得更彻底（drop 后均值 0.3 对 1.6）。这些都不影响 dim 的结论。
5. **设备**：
   - 7250 处数据相同：all6(B200) − V7b = +0.39（sd 4.6），V7 − V7b = +0.38，没有设备偏移。同卡型的对也照样分叉（s17 两次都在 A100，s18 两次都在 H100，drop 前 dim 最大差 18 / 11 点），所以主要是 nondeterminism，不必单独加限定。
   - 但 F109 限定 4 的「配对 SE 约 1.8」偏小了一倍：V7 − V7b 的 final3 sd 是 10.2（SE 3.6），代价的配对 SE 是 2.5–3.2。限定 4 的「解释不了 5.9」应删掉或改写。
6. **F109 两句的现状**：
   - 「约四分之一」应改为「约五分之一」：V7 23%、V7b 18%、合并 21%，CI 上界 35%。
   - 「约 4700 步内为零、之后增长」仍成立，V7b 还更晚：
     - 7324–12000 段代价 V7b −1.6（V7b 领先 6/8），合并 −1.0；
     - 12001–18309 段合并 +3.6 [+0.1, +6.9]；
     - 合并时间线：12000 +0.8 → 16500 +3.6 → 17500 +5.0 → 18309 +5.5。V7b 在 12000 前 ≤0，之后到 +4.8。

## 建议句
"With the dropped torus and grid lines replaced in place by text of the same length, so that every remaining line keeps its position and the dimension lines keep the registered 0.50 eval-slot share, the six priors still end 20.4 points above the dimensions prior alone on strict dimension questions (8 of 8 seeds), 82% of the gain with the problems kept (seed-bootstrap 95% CI 0.65–1.06). Matching the slots changed nothing measurable (V7 − V7b −1.3 points, CI −7.7 to +5.4), although V7 had trained half of its post-drop dimension lines in the exact evaluation layout. Pooling both runs per seed, dropping the problems keeps 79% of the gain (CI 0.65–0.96) and costs 5.2 points at the end of training (CI 1.0–9.1); the cost is zero for about 4700 steps after the drop and grows through the last third of training. So the post-window torus/grid practice carries about a fifth of the gain; the rest was in place before the drop."
限定集中一处：单一训练长度，代价到结束仍在涨；合并代价的 bootstrap CI 不含 0，但 t 检验 p = .05；n=8 时 MDE 约 8–10 点。

- **最可能测错的地方**：数字没测错。最可能写错的是那句「slot matched 后代价不再能和零区分」：它把一次功效 27–40% 检验的显著性翻转当成效应变化，而 V7 和 V7b 的代价只差 1.3 ± 6.5。
- **可能被测没或写没的真效应**：约 5 点的去题代价（合并 +5.2，CI 1.0–9.1，保留比例上界 0.955 < 1），被「no longer distinguishable from zero」和「at most a small part」写成了接近零。F109 限定 4 的 SE 1.8 低估了重抽噪声，这条限定的论证方向也因此不成立。
- **证据撑得住的最强结论**：去掉窗口后的 torus/box5 行，dim 的组合增益保留约八成：slot 匹配 82%（+20.4，8/8，CI 0.65–1.06），合并 79%（CI 0.65–0.96）。行位置和评测布局对结果没有可测影响（−1.3，CI −7.7 到 +5.4）。窗口后的练习在训练结束时撑着约五分之一的增益（5.2 点，CI 1.0–9.1），这部分代价在约 4700 步后才出现。计划句前两句的数字都对；后两句（「不再能和零区分」「at most a small part」）比证据弱，首句的 82% 也和预写规则的判定（77% 不变）不一致。
