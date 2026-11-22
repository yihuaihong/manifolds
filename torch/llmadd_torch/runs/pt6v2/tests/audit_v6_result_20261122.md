# pt6v2 Amendment V6 结果审查（exp-auditor，2026-11-22，只读，未重跑）

范围：12 个 result.json 和 states_ans.npz，以及 aggregate.txt。我的脚本和输出都在 /scratch/yh6210/tmp_audit/pt6v2_v6_result/（recompute.py、compact.py、extract.py、z.py、z2.py，以及对应的 *.txt）。

**数字复核：没有算错的数。** 我自写代码，从 states_ans.npz 独立重算了以下几项，12 个 checkpoint 全部与 result.json 一致到 3 位小数：dose（h5 k3 上 torus / box5 / dim 三个子空间）、pc3 dim–box5 overlap（L5/L6）、clean acc 和 strict acc（n=58/77）。用存下的 35 个 null draw 重拟 drop~dose 直线，得到的 excess 也与存值一致。区间写法有三处小出入（次要）：
- C1 only-dim 写「2–5 pts」，实为 5.0/1.5，应写 1.5–5。
- C2 all6 dim→box5 写 0.94–0.97，实为 0.932–0.975。
- C3 写「margin 高 1.2–1.7」，这是两 seed 均值在三个 step 上的范围（1.20/1.70/1.50）；逐 seed 是 0.99–1.87。

**C1 成立，可以照写。**
- 18309、h5、k3：drop 45.6/43.9 vs 5.0/1.5 pts；excess +0.131/+0.105（z 4.4/2.7）vs −0.045/−0.039。
- k6 同向：+0.184/+0.141（z 4.7/3.0）vs −0.061/−0.030。strict 子集更强：drop 60/57 vs 2/1 pts，excess +0.24/+0.20。
- 按 code review 的标准，只对 torus 自己的 shuffled-label PC 加 random 做拟合，excess 仍是 +0.111/+0.118 vs +0.015/−0.021。anchor（cities/countries）在 k3 两 seed 都 ≤ +0.05。
- 适用范围：
  - (a) 只在 h5 成立（h4 k6 s16 也有 +0.15）。h3、h4 k3 ≈ 0，h6 为负（−0.095/−0.018）。
  - (b) k6 下 s16 的 countries 也有 +0.15（torus 是 +0.18）。所以 k6 的特异性只在 s17 干净，k3 两 seed 都干净。
- 「随训练增长」：
  - raw drop 28/10 → 40/24 → 46/44 属实，但 dose 本身也涨了 1.5–2 倍（0.105→0.159，0.083→0.172）。
  - 扣掉剂量后，s16 是 +0.09→+0.11→+0.13（对自身 shuffle 反而 0.19→0.11 下降），s17 是 −0.06→−0.02→+0.105。
  - 结论：「依赖增长」在 raw drop 和 dose 上成立；超出剂量的特异部分在 s16 的 7323 已经存在，s17 要到 18309 才出现。
- 只消融了 ap 一个位置（probe6_inner.py:13-16），所以 h3/h4 ≈ 0 只能说「ap 上没有」，不能说「该层没有」。

**C2 的 transfer 在 L6 成立；L5 和 pc overlap 的组间差距大多是 null 本身的组间差。**
- transfer：
  - L6：all6 0.94–0.975，rp95 0.31–0.56，超出 null 0.42–0.64；only-dim 0.70–0.83，超出 0.34–0.48。6 对里 all6 ≥ only-dim 6 次，其中 1 次持平。残差 1−R²：0.025–0.056 vs 0.17–0.30。
  - L5：all6 的 random-projection p95 本身就是 0.52–0.79，only-dim 只有 0.22–0.42（probe6_inner.py:533-541）。超出 null 的量 all6 0.17–0.42 vs only-dim 0.31–0.52，6 对里 5 对是 only-dim 更大。
- pc3 overlap：
  - raw 0.53–0.62 vs 0.33–0.46 复现。
  - row-shuffle null 均值 L5 0.37–0.51 vs 0.14–0.28，L6 0.31–0.42 vs 0.18–0.25（probe6_inner.py:489-498）。
  - 超出 row-shuffle 均值：L5 all6 0.12–0.20 vs only-dim 0.17–0.24，6/6 是 only-dim 更大；L6 3/6；z 值 all6 2.8–4.6 vs only-dim 4.3–8.5。
  - 只有对 cov-matched null，all6 的超出才更大（L5 0.42–0.47 vs 0.31–0.39）。
  - 解读：row-shuffle null 量的是两域主方差方向的重合。all6 里答案坐标主导方差，所以这个 null 会吃掉一部分真效应；它既不能证明组间有差，也不能证明没差。
- 最强可写：「all6 中 dim 解码器在 L6 读 box5 的 R² 为 0.94–0.975，only-dim 为 0.70–0.83，两组都高于各自的 random-projection null」。overlap 只能写成 dim 和 box5 共享更多主方差子空间，不能写成标签特异的共享编码。

**C3 数字复现；margin 部分是准确率差的另一种写法，「定位到末层 MLP」只在一半的 cell 成立。**
- mlp6 DLA 差（strict 子集，probe_v6_aggregate.py:91）：7323 +1.18/+0.19，12000 +1.05/+1.09，18309 +1.40/+0.65。在 all 子集上也是 6/6 为正（+0.26 到 +1.14）。
- L6 margin 等于 DLA total，也就是模型自己的最终 margin，跟 strict acc 差（0.97/0.82 vs 0.71/0.60）是同一件事。另外 dW 用的是各组自己的 best-other token（probe6_inner.py:567-571）。
- mlp6 占 margin 差的比例：s16 73–86%，s17 19–58%。s17 里 attn5 贡献相当（+0.72/+1.03/+0.76）。dim_seen 在 7323 的 mlp6 差 ≈ 0（+0.17/−0.01）。

**C4 复现，可以写得更强。**
- h5：box5 的 raw DiD +0.19/+0.16，但 excess DiD 为 −0.12/−0.04（k3）和 −0.13/−0.05（k6）。h4 ≈ 0。
- h6：all6 的 box5 excess 在 3 个 step × 2 个 seed 全为正，k3 +0.06 到 +0.19（z 1.7–4.6），k6 +0.14 到 +0.25（z 2.5–4.8）。only-dim 在 k3 ≈ 0（−0.09 到 +0.02），k6 +0.06 到 +0.09。对自身 shuffle 算时偏弱（DiD −0.04 到 +0.10）。
- 可写成：超出剂量的依赖，torus 在 h5，box5 在 h6，两 seed 一致。不要写成「小，符号随层变」。

**漏掉的效应（反向消融和 A1，均为两 seed）**
- 拿掉 dim 子空间伤 torus_held，h5 k8 18309：drop 0.44/0.45 vs 0.07/0.10，excess +0.16/+0.12（z 2.3/1.6）vs −0.01/+0.02。7323 起两 seed 一直为正。
- 拿掉 torus 子空间伤 box5_held，h5 k4：excess +0.20/+0.12（z 2.7/3.9）vs −0.04/−0.02。
- 这说明 dim–torus 的耦合是双向的，不是 dim 单向读 torus。但对自身 shuffle 只剩 +0.02 到 +0.06。
- A1 torus→dim 在 L6：all6 0.75–0.88，rp95 约 0.45–0.51；only-dim 0.38–0.57，不高于 rp95。注意 only-dim 选到的 torus pair 不同（B1），源坐标在两组间不同。
- A4：dim unembed 的 CKA 组间无差（−0.03 到 +0.035），这里没有漏掉的效应。

重要：C2 overlap 的组间比较受 null 的组间差主导（probe6_inner.py:489-498）；C2 在 L5 的 transfer 差距被 rp null 解释（:533-541）；C3 不能写成「末层 MLP 承载增益」，s17 一半来自 attn5（:567-571）；C1 k6 在 s16 不特异；C1「增长」中超出剂量的部分只在 s17 出现。
次要：三处区间写法见上；C1 的「dose-matched random 和 shuffled」实际拟合里还含 cov null（:737）；A3 用的 strict 子集 n=58/77，换 all 子集结论同号。

- **最可能测错的地方**：C2 的 pc3 overlap 组间差。row-shuffle null 本身在 all6 高很多（L5 0.37–0.51 vs 0.14–0.28），扣掉以后 all6 的标签特异 overlap 反而不高于 only-dim。
- **可能被测没或写没的真效应**：
  - C4 被写弱了：h6 的 box5 特异依赖在 all6 的 6 个 checkpoint 全为正（z 1.7–4.8），only-dim ≈ 0。
  - 反向消融（dim→torus_held，excess DiD +0.10 到 +0.17；torus→box5_held，+0.13 到 +0.23）和 L6 的 torus→dim transfer 不在结论里。
  - C1 在 h3/h4 读到的 ≈ 0 只是 ap 单位置的结果。
- **证据撑得住的最强结论**：训练结束时，在 ' is' 位置、第 5 层均值消融 torus 的 3 维类均值子空间，all6 的 held dim 准确率掉 44–46 pts，only-dim 掉 1.5–5 pts（两 seed）。超出剂量匹配 null 的部分为 +0.11 到 +0.13（z 2.7–4.4），only-dim 处于 null 或以下。耦合是双向的（拿掉 dim 子空间伤 torus 超出 null +0.12 到 +0.16）。all6 中 dim 解码器在 L6 读 box5 的 R² 为 0.94–0.975，only-dim 为 0.70–0.83。超出剂量的依赖，box5 在 h6，torus 在 h5。
