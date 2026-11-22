# ptsz 最终读数：结果审查（exp-auditor，2026-11-22，只读，没有重跑）
复算：/scratch/yh6210/tmp_audit/ptsz_final/{recompute,picks_swaps,extra,extra2}.py 和对应的 .out。只沿用了定义，没有 import 读数代码。沿用的定义：area 是 75 次 eval 上 6 个 held 桶 exact match 均值再乘 100（readout_rules 的 docstring）；gain 按 seed 配对；N_nonemb = n_params − 100278·d_model；OLS 等权；bootstrap 用 default_rng(0)，按 N 升序每个尺寸抽一次 (10000, 8) 下标，取第 5 / 95 百分位；(c) = b ± t(0.95, n−2)·sqrt(SSR/(n−2)/Sxx)；选 lr 用 F87 rule 1 的 top2：前两名相差 ≤2 时取 in-training bpb 较低的。

## (1) 数字：全部复现，没有算错
- 10 个 cell 的 none / prior / gain、seed CI、s0-3 / s4-7、8/8 胜场，都和 readout.txt 一致到打印精度。80 个 seed 对（160 个 run JSON）都是 75 evals，code a3bce3ab63c1，prior_events 相同。ptsz 的 64 个计数 run（4 尺寸 × 2 臂 × 8 seed）CHECK7 全部 pass。16 个 pick（其中 8 个新）按规则重算全部一致，没有 open bracket。
- PRIMARY：b +1.664，boot [+0.705, +2.675]（换成 rng(1) 是 [+0.72, +2.69]，MC 误差 ±0.02）；halves +1.155 / +2.173；resid90 df6 [+0.028, +3.300]，双侧 p 0.0955（单侧 0.048）；median +0.956，trimmed +1.193，LOSO-seed 极值 +1.320（去掉 505M s5）。
- 下面这些也都复现：SECONDARY +0.934 [−0.709, +2.680]，median −0.477，trimmed +0.405；10 点斜率 +2.299，加 B200 项后 +1.940，c +2.66（se 2.47）；bridge；arms −1.545 / +0.119；8 个 LOSO-size；18 个 lr swap；F99 six +2.787，p 0.0377；L40S-4 +2.161。
- 136M none s6/s7 因配额失败后重交：两个 JSON 完整，CHECK7 pass。AUDIT:147 规定「infrastructure loss is resubmitted, not listed」，所以不进 OWN_FAILURES，处理正确。读数 17:28 跑，晚于最后一个 C run 的 17:26；readout_sha cc44adc8329d 与 FIRST_READOUT.json 记录的一致。
- 计划句子里唯一有问题的数：「p = 0.10」是 0.0955 舍入来的，恰好等于 (c) 的门槛，读起来像没过。改成 p = 0.096（双侧）。

## (2) 证据撑得住的最强说法
- **gain 本身存在**：这一点不需要限定在一种卡上。9 个尺寸 53M–1.07B、10 个 cell（294M 在 L40S 和 B200 各一个），每个 cell +22.8 到 +36.1，80/80 个 seed 对全胜。计划句子写成「8 个尺寸、one GPU type、22.8–32.2」，说窄了：同一种卡只是斜率检验的条件。
- **随尺寸增长**：pre-registered primary 判 SHOWN (increase)，+1.66 / e-fold（median +0.96，trimmed +1.19），90% resid [+0.03, +3.30]，p = 0.096。离门槛很近，但按登记的规则成立，不需要另加限定。
- 预注册要求放在旁边的限定：
  - I3：必须照写，计划句子已经有了。这句话限定的是「primary SHOWN + secondary NOT SHOWN」这个组合，不只是 secondary；计划句子把它挂在次检验后面也可以。按字面看，4 个新点平均比旧 L40S-4 线低 1.1 点（391M 低 4.4），比 F99 六点线平均低 2.1，都在 seed 噪声以内（8-seed 均值的 SE 2.46）。所以照登记原话写「did not fall below」站得住。
  - 固定 token 混杂（AUDIT「What the verdicts may be called」）：这条是必需的，计划句子里没有。这条 ladder 上每个 non-embedding 参数的 token 数从 21.2 降到 0.49，尺寸和训练充分度是混在一起的；none 随尺寸变弱，正是这个混杂会预期的样子。另外不能写「prior 的作用随尺寸变大」，计划句子已经避开了。
  - F99 magnitude rule（FINDINGS F99 补记）：primary 已经合规。secondary 只写 +0.93 不合规，它的 median −0.48 符号相反，要么三个数一起写，要么只写 not shown。none 的 −1.55 也要配 median −0.83、trimmed −1.11，或者只报方向。
  - secondary 的 false-SHOWN rates（0.07 / 0.12 / 0.19 / 0.22，0.07–0.20，0.08–0.17）：RUNBOOK 第 5 步和 I2 要求写进 FINDINGS。但这组数针对的是 SHOWN 判定；这次是 NOT SHOWN，真正相关的是功效：对 F99 线 0.56–0.63，对 L40S-only 线 0.47–0.49。所以 NOT SHOWN 不能读成「新尺寸上没有增长」。它的 CI 包含 primary 的 +1.66，上界 +2.68 略低于 F99 的 +2.79。
  - few-seeds label：primary 没有触发（比值 0.57 / 0.72 / 0.79），不用挂。secondary 打印了 label，但按 amendment 2，这个 label 只对 SHOWN 判定有意义，也不用挂。
  - primary 的条件误判率：AUDIT:191-192 和 429-430 把它和 primary 判词放在一起，FINDINGS 照写；论文里有 I3 那一句就够了。登记的数是 0.12–0.14（mu0 = 旧四点均值 26.85）和 0.27–0.38（mu0 = F99 六点均值 28.77）；新 4 点实测均值 28.25，落在两者之间，偏后者。
- 逐项：
  - **10 点 all-hardware**：+2.30（median +1.44，trimmed +2.00），resid df8 [+0.91, +3.69]，p 0.015；加 B200 项后 +1.94 [+0.40, +3.48]，p 0.048；去掉 1.07B 后 9 点 +1.69 [+0.22, +3.16]，p 0.066。可以写成描述性的「与登记结果一致」，不能拿来替代检验。B200 两点之间的斜率是 +4.68。
  - **bridge（294M）**：gain −0.12 [−5.22, +4.69]（按 seed 配对是 [−3.1, +2.9]），none −0.24，prior −0.37 [−0.72, −0.02]，294M 上看不到硬件偏移。但 pt300x 在 505M 的 bridge 是 +4.3，所以只能说「294M 上合并两种卡没有被反驳」，不能说「硬件不重要」。
  - **lr swap**：18 个里 0 个改变符号，seeds 0-3 上的斜率在 +0.95…+1.53 之间（pick 处是 +1.155）。可以写「不是 tie band 内选 lr 造成的假象」，适用范围是 seeds 0-3、一次只换一个 lr；505M win-ctx 只有一个 lr，没有可换的。
  - **LOSO-size**：8 个斜率全为正（+1.18…+2.08），可以写「增长的方向不靠单一尺寸」。但只有 3/8 仍满足 (c)：去掉 53M、136M、294M、505M、731M 中任何一个都不满足。所以不能写「判定经得起 leave-one-size-out」。去掉 136M 后 (c) 是 [−0.11, +3.60]；正是 136M 让判定从 NOT SHOWN 变成 SHOWN，但它不改变斜率大小（+1.74 对 +1.66）。
  - **p = 0.038**：这是 F99 六个混合硬件尺寸的结果，六个点都在登记前看过，其中 4 个 L40S 点被主检验复用。论文的 headline 应改成 p = 0.096（8 个 L40S 尺寸，登记检验，双侧）。如果保留 0.038，必须标明是 F99 的六点拟合。不要换成 10 点的 0.015（描述性、混合硬件）。F99 的 +2.79 也不再作为 headline 量级。
- 建议的句子：
  "At a fixed 300M tokens with 7% arithmetic, the helix prior raises held-out accuracy averaged over training by 22.8 to 36.1 points at each of nine sizes from 53M to 1.07B (ten size-by-GPU cells, 8 of 8 seeds in each). On the eight L40S sizes the pre-registered test finds the gain increasing with size: +1.66 points per e-fold of non-embedding parameters (median-gain slope +0.96, trimmed +1.19; seed-bootstrap 90% CI [+0.71, +2.68], both halves positive, 90% residual interval [+0.03, +3.30], p = 0.096). The three sizes never seen before registration do not show it on their own (mean slope +0.93, median −0.48, trimmed +0.41; CI [−0.71, +2.68]), so this says only that the new sizes did not fall below the earlier trend. Along this ladder tokens per non-embedding parameter fall from 21 to 0.5, and the increase comes from the model without the prior getting weaker (−1.55 per e-fold; median −0.83, trimmed −1.11) while the prior arm stays at 89–90."

- **最可能测错的地方**：数字都没算错。最可能写错的是 secondary 只报了均值斜率 +0.93，而它的 median 斜率是 −0.48，符号相反；其次是 p 舍入成 0.10，恰好等于门槛。
- **可能被测没或写没的真效应**：(i) 加了「one GPU type」后，gain 存在性从 9 个尺寸 / 10 个 cell（+22.8–36.1，80/80）缩成了 8 个尺寸；(ii) secondary 的功效只有 0.47–0.63，它的 NOT SHOWN 不能读成新尺寸上没有增长；(iii) 写成 p = 0.10，让过了门槛的 SHOWN 看起来没过。
- **证据撑得住的最强结论**：在 3e8 tokens、7% 算术下，prior 在 53M–1.07B 的 9 个尺寸（10 个 cell）上都提升 +22.8–36.1 点（80/80 seed 对）。8 个 L40S 尺寸上，登记检验判 SHOWN (increase)：+1.66 / e-fold（median +0.96，trimmed +1.19），90% resid [+0.03, +3.30]，p = 0.096。按 I3，这只说明新尺寸没有落到旧趋势以下；尺寸和 tpp（21→0.49）是混在一起的。计划句子在 gain 的范围上说窄了，在 secondary 的量级上说过了。
