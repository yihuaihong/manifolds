# V7 结果审查（pt6v2 Amendment V7 = all6@2e-3 在 7324 起去掉 torus/box5 LM 行，all6drop s16–23），2026-11-22，只读
独立重算（自写 loader，断言三臂 held_q 相同、drop=7324、data_sha drop 前相同 / 后不同、eval 步一致；strict n/seed 58–95）：/scratch/yh6210/tmp_audit/pt6v2_v7_result/{recompute.py, recompute_out.txt, timecourse.py, timecourse_out.txt}。与 out_v7/v7read.txt 一致。

## Q1 逐条核对（数字全部复现，没有算错）
1. 「比 only-dim 高 19.1（7/8）」：+19.09，sd 15.8，7/8（s23 −9.1），t p .011。复现。
2. 「curve average +11.9, 8/8」：复现。area 含 30 个配置相同的 drop 前 eval；drop 后段均值 +17.5（share 0.88）更对题，非错误。
3. 「+25.0 的 77%」：0.765 复现；seed bootstrap share 2.5/50/97.5% = 0.50/0.76/0.96。+25.0 的符号是 7/8 + 1 平（s23 两臂都是 57.54），FINDINGS.md:5432 写 8/8（本句没用这个符号计数）。
4. 「结束时代价 5.9（6/8）」：−5.86，6 负 / 1 平（s20）/ 1 正，t p .054，CI [−11.9, +0.1]。复现。
5. 「窗口后多掉 5.9（7/8）」：+5.85，7/8，t p .014，Wilcoxon .016，CI [+1.6, +10.1]（v7read.py:87，基线 = 7250 单点）。复现。但 7250 处两臂均值差 −0.01，4 和 5 是同一个 5.9 数了两次，不是两条证据；5 只是扣掉了每个 seed 的 re-draw offset。基线换成窗口末 4 个 eval（6500–7250）：+5.07，6/8，p .088，CI [−1.0, +11.1]，所以 7/8 和显著性依赖基线选法（7250 是预先写在 v7read 里的那个）。
6. 「约四分之一」：1−0.765 = 0.235 复现，bootstrap 4–50%。这只是终点值：7324–12000 段代价 +0.5（all6drop 不差，share 1.03），12001–18309 段 −4.6（6/8，share 0.81），18309 单点 −6.1。代价在 drop 后约 4700 步才出现，到最后仍在扩大。
7. 「大部分在先验开着时建成」：和数据对不上。7250 处两臂都只比 only-dim 高 +6.0；all6drop 在先验和 torus/box5 行都没有的情况下，差距扩大到 +19.1（only-dim 88.5→52.3，all6drop 94.5→71.2）。F105 补记二自己写的是「高出的部分出现在先验关掉之后」（FINDINGS.md:5416, 5432）。V7 支持的是：抗衰减的那部分在 drop 前已经就位，不靠窗口后的 torus/box5 练习维持。V7 也分不开「先验」和「窗口内的 torus/box5 行」，归因到先验要借 F106 的 all4（FINDINGS.md:5478）。

## 四个考虑项
- 设备：没有同卡的一对 run。all6drop 的 s16/18/19/20/22/23 在 H100，s17/21 在 A100；registered 的 s16–22 在 B200，s23 在 A100。HANDOFF_V7.md:102「s23 A100 in both」不对（all6drop s23 是 H100）。V5 同配置重跑（rerun − registered）：all6 f3 +3.4/−0.9，dim_drop −5.2/−4.3（两次都比原 run 掉得少，方向和 V7 相反）；only-dim f3 −4.6/−2.6。每 run 的 re-draw 噪声约 3–5 点且没有方向，8 seed 配对后 SE 约 1.8，解释不了 5.9。把 V5 当作第二个 all6 draw：s16 代价 −5.2，s17 −14.7，和 registered 配对的结果一致。
- 窗口末是否对齐：drop 前 data_sha 全同。7250 每 seed 差 0/−5.2/−1.2/+2.1/+4.6/+3.2/+2.4/−6.0（均值 −0.01，sd 3.9），和 V5 同配置重跑的 −1.7/−5.2 同一量级，属 GPU nondeterminism，不是 drop 前的配置差。check_v7 唯一的 FAIL 是 s17/21/23 的 step-0 lm（rel 1.8e-6/4.0e-6/1.9e-6，容差 1e-6 比设备数值噪声还紧），按 flag-only 处理是对的。

## 重要
- I1（推测）drop 同时改了 dim 行的位置（out_v7/check_v7.py:72-93 按设计校验过，但读数没有解释它）：all6drop 的 dim slot-1 share 全程 0.65（drop 后约 0.75），all6 和 only-dim 都是 0.50；domain eval 的题固定放在 slot 1（pretrain_range.py:247-249）。如果 slot-1 练习对 dim 评测有帮助，偏差会把 all6drop 往上推：share 偏高、代价偏低。drop 后头 4700 步 all6drop 略高（7500/9000 处 +2.0/+3.6）和这个解释相容。弱负对照：countries 同样是 0.65，差 −0.15（null）。检查：在已存的 checkpoint 上把 dim 题放到 slot 2（前面接一条 cities 行）评一次。
- I2 第 7 点需要改写，不是加限定（见上）。I3 第 4、5 点合成一条。I4 漏写了更强的事实：drop 后 400 步（7750）torus/box5 held 就降到 0–3.2%，final3 8/8 为 0.0，低于机会水平 1/36、1/25。这是 exact-match 全词表 argmax（pretrain_range.py:1847），所以是答案输出被压制，不能说成计算消失，但 dim 仍保留 77%。

## 次要
+25.0 的「8/8」实为 7/8 + 1 平；HANDOFF 的 s23 同卡说法错；area 版 share 0.88 被 drop 前段机械抬高；bpb all6drop 低 0.0054（7/8，−0.45%，多出的 text token），numbers/cities/countries 差 −0.48/+0.32/−0.15，均 null。

## 证据撑得住的最强说法（建议句）
「窗口关掉时去掉 torus 和 box5 的 LM 行，量纲格的组合增益大部分保留：训练结束时，strict 量纲格题上六先验仍比单加量纲格先验高 19.1 点（7/8），是保留这些行时 +25.0 的 77%，尽管 torus/box5 在 drop 后 400 步内就答不出了（8/8 为 0%）。去掉这些行的代价在 drop 后约 4700 步内为零，到训练结束为 5.9 点（相对窗口末多掉 5.9，7/8，95% CI 1.6–10.1）。所以窗口后的 torus/box5 练习在训练结束时撑着约四分之一的增益；其余部分在 drop 前已经就位，到窗口之后才显现。」
限定集中写一处：单一训练长度（四分之一是终点值，代价仍在扩大，bootstrap 4–50%）；dim 行位置随 drop 改变（I1，未量化，偏向「保留」）；归因到先验要借 F106 all4。设备混合不需要单独限定（re-draw 无方向，配对后解释不了效应）。原假设「dim 优势大部分消失」不成立：保留份额点估计 77%，bootstrap 下界 50%。

- **最可能测错的地方**：all6drop 和 all6 除了窗口后的 torus/box5 行，还差在 dim 行落在 eval 位置的比例（约 0.75 对 0.50）和 text token 数，两者都偏向「保留」，77% 可能略高（I1，未量化）。
- **可能被测没或写没的真效应**：句子没写 torus/box5 在 drop 后 400 步内就降到 0%，而 dim 仍保留 77%；也没写代价到 12000 步后才出现且仍在扩大，所以四分之一可能低估更长训练下的终值；第 7 点把「drop 前就位」误写成「先验开着时建成」，和 F105 冲突。
- **证据撑得住的最强结论**：训练结束时 dim strict 上六先验去掉窗口后的 torus/box5 行仍比 only-dim 高 19.1（7/8），是 +25.0 的 77%（bootstrap 50–96%）；去掉这些行的代价是窗口后多掉 5.9 点（7/8，CI 1.6–10.1），而且只在 drop 后约 4700 步才出现。作者那句的数字都对，「约四分之一」这个强度合适；第 7 点要改成「drop 前就位、窗口后显现」，同时补上 torus/box5 归零这一条。
