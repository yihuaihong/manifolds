# pt6v2 Amendment V8 结果审查（2026-11-23，exp-auditor，只读，未重跑）
V8 = only-dim@2e-3（只开量纲格先验）和 none@3.2e-2（不加先验、dim 面积最优 lr N_m）训练 2× / 4×，seed 16/17。自写脚本（strict 掩码与各指标按定义重写，不 import / exec v8read.py 或 v4final.py）：`/scratch/yh6210/tmp_audit/pt6v2_v8_result/recompute.py`、`recompute_out.txt`、`cka.py`。

## (1) 有没有算错或复现不出的数：没有
- 句中数字全部从 run JSON 复现：only-dim 2× / 4× strict final3 44.45 / 29.55（逐 seed 28.7、60.2 / 14.9、44.2）；all6 1× 8 seed 77.28（57.5–92.5）；none 4× 78.12（75.3、81.0）；none 4× numbers 49.49 对 all6 96.83。v8read.txt 其余各表也一致。「峰值在先验开着时」4/4 成立（峰值 step 6500 / 6750 / 7000 / 2250，都在 916–7323 内）。
- steps 36620 / 73240 = tokens / (32×512)。evals 148 / 294，最后一个在 36619 / 73239，没有重复（8 个都是 restart 0，没有 requeue），check.txt 全 PASS。
- 窗口：only-dim 4 个 run 的 prior_events 都是 [[916,1],[7324,0]]，prior_on 的 eval 恰好是 1000..7250（26 个），与 1× 相同；none 4 个都是 []。
- lr：曲线里没有记 lr。按 args（lr、tokens）和训练器公式（pretrain_range.py:1323-1326）算：warmup 366 / 732（1× 是 183）。step 18309 处 only-dim 的 lr 是 1.10e-3 / 1.74e-3（1× 2.0e-4），none 是 1.76e-2 / 2.78e-2（1× 3.2e-3），末端都是峰值的 0.1 倍。与 AUDIT 写的一致。
- 数据流：data_sha 是全部 batch 的滚动 sha256（pretrain_range.py:2395）。8 个长 run 与同 seed 的 1× 在全部 74 个共有 eval（≤18250）上相等；同 seed 不同臂在 step 250 也相等。所以到 1× 步数为止，数据和初始化逐位相同，差别只剩 lr 曲线和卡型。

## (2) 证据撑得住的最强说法
**重要-1　numbers 那半句混入了 lr 的影响（假阳性方向）。**
- none@3.2e-2 的 numbers 本来就差：1× 8 seed 只有 19.2；none 在 numbers 上最好的 lr（8e-3）也只有 63.8。
- only-dim@2e-3 同样没有 numbers 先验，但 2× 就到 95.5（逐 seed 93.0 / 98.0），4× 到 96.7（95.8 / 97.6），追平 all6 1× 的 96.8。
- 所以「without priors … far below on numbers」只对按 dim 选的 lr 成立，泛化成「没有先验 numbers 上不去」会被 only-dim 自己的数据推翻。none@3.2e-2 的 LM 也更差：4× 的 bpb 1.295，all6 1× 是 1.209。

**重要-2　漏掉的更强结论（假阴性方向）。**
- (a) only-dim 不是追不上，是追上了留不住。先验开着时 strict 峰值 93.5–100（均值 96.8–98.1），≈ all6 1× 的峰值 97.1；第 1250–2250 步就已经超过 77.3。
- (b) 训练越长反而越差。4× 两个 seed 都低于各自的 1×（14.9 对 77.0，44.2 对 58.9）；2× 只有 s16 更低。dim 几何也在流失：cka_dim final3 在 1× 的 8 个 seed 是 0.959–0.978，2× 是 0.86 / 0.87，4× 是 0.75 / 0.79，4 个长 run 全部低于 8 个 1× run。dim seen 一直约 100，说明是先验塑造的结构在消退、模型退回记忆，而不是没练到。
- (c) 没有任何长 run 能同时达到 all6 1× 的 dim 和 numbers：only-dim 有 numbers 没有 dim，none 有 dim 没有 numbers，逐 seed 都是如此。
- (d) none 在 4× 追平 dim 时几何不同：cka_dim 0.53 / 0.54，all6 是 0.97。这是描述性读数，跑前没有登记。

**重要-3　下降跟着拉长的 cosine 走，不是跟 token 数走。**
- 在 step 18250，数据逐位相同、窗口相同，only-dim s16 的 strict 在 x1 / x2 / x4 下是 78 / 34 / 14，s17 是 57 / 57 / 49；同一步的 cka 两个 seed 都从 0.97 降到 0.92–0.93。
- 所以这里的「多训练」实际是「先验关掉以后多了很多高 lr 步」。
- all6 没有跑长。all6 的先验同样在 7324 关，1× 下窗口后也掉 17.3 点（only-dim 掉 36.2）。因此不能写「联合先验抵抗长训练」。现在的比较是 all6 用 1/2–1/4 的算力胜出，对 all6 是保守的。

**重要-4　「needs 4×」只在 strict 集、这一个 lr 下成立。**
- 在全部 held dim 题上，none 2× 已经是 83.9，≥ all6 1× 的 83.0。
- lr 没有为 2× / 4× 重新调。建议写成「在试过的长度里，只有 4× 追平」。

**次要（合并）：**
- 比较基准：all6 的 s16 是 8 个 seed 里最高的（92.5），同 seed 比较时 only-dim 的差距更大（+42.7 / +57.6），所以用 8 seed 均值对 only-dim 是保守的。
- none 的 seed 配对没有信息量（1× 8 seed 上 corr(all6, none) = −0.56；none 的 s16 / s17 在 1× 是 55.4，≈ 8 seed 的 53.5），用 8 seed 均值是对的。若与同 seed 的 all6（87.2）比，none 4× 低 9.1（逐 seed −17.2 / −0.9）。
- only-dim 的 s16 / s17 是它 1× 的前两名（77.0 / 58.9，8 seed 均值 52.3），长 run 又从 B200 换到了 A100，所以「同 seed 67.9 → 44.5 → 29.5」里含回归均值。写下降时用 52.3 做 1× 参照，或者用 CKA。
- 卡型：长 run 在 A100（none s17 在 H100），1× 在 B200；由此带来的差异远小于上面这些效应。
- 其他领域（表 5）：torus 上两个长臂都超过 all6 1×（none 94.0–94.2，only-dim 97.1 / 94.2，all6 87.1）；cities / countries 上 all6 仍高 1–3 点；box5 噪声大（26–51 对 50.0）。

**建议写法**（第一句只带一个限定，其余限定集中在后面）：
"Training longer does not replace the joint priors: with up to 4× the training, no run without them reaches the six priors' 1× level on both dimensions and numbers. The dimensions prior alone reaches that level on dimensions while it is on (peak strict 97–98), but it does not hold it. At the end of 2× and 4× runs it scores 44.5 and 29.5, against 77.3 for the six priors at 1× (8 seeds), and its dimension geometry erodes (CKA 0.97 at 1× in all 8 seeds; 0.86/0.87 at 2×; 0.75/0.79 at 4×). Its numbers meanwhile reach the six priors' level (96.7 at 4× vs 96.8). The model without priors, at its best rate for dimensions, reaches the six priors' 1× mean on dimensions only at 4× (78.1 vs 77.3) and stays at 49.5 on numbers."

集中的限定：
- 每格 2 个 seed。
- 每臂只用一个 lr，cosine 随长度拉长，先验关掉后 lr 更高，下降跟着它走。
- all6 没有跑长。
- 49.5 是按 dim 选的 lr 下的 numbers（这个 lr 在 1× 只有 19.2）。
- 只看 strict 集（全 held 题上 none 2× 已经追平）。
- A100 / H100 对 B200。

- **最可能测错的地方**：数字都对；最容易被读错的是 numbers 那半句，49.5 来自 numbers 很差的 lr（1× 只有 19.2），而 only-dim 没有 numbers 先验，2× 就到 95.5。
- **可能被测没或写没的真效应**：原句的「追不上」漏写了「先验开着时追上了、关掉后留不住」，也漏写了「训练越长越差、几何流失」（CKA 4/4 低于全部 1× run）；「needs 4×」在全 held 题和重调 lr 后可能更短。
- **证据撑得住的最强结论**：在 4× 以内、试过的两个 lr 上，不加联合先验的模型都不能同时追平 all6 1× 的 dim（77.3）和 numbers（96.8）；only-dim 的 dim 随训练变长从峰值 97–98 降到 44.5 / 29.5，几何同步流失；none 只在 4× 追平 dim（78.1），numbers 停在 49.5。这比原句强：原句只说追不上，没写更长反而更差，也没写同时兼顾两个领域做不到。
