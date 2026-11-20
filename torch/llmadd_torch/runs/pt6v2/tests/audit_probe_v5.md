# probe_v5 代码 / 设计审查（exp-auditor，2026-11-20；主会话按原文存档，审查员只读不能写项目文件）

审查脚本与输出：/scratch/yh6210/tmp_audit/probe_v5/

**结论**：无阻断项，可照原样跑；要读出「all6 被冲刷得少」需先补重要 1–4、6（都只是 CPU 分析）。

**仪器核对通过**
- prompt / 位置 / 层号与 trainer 一致（ap = " is"，物品位置 506/508，store[L] = block L 输出）；E1–E4 只加 dump，不改 forward；eval context 由 numpy RNG 生成，与设备无关。
- 两套 CKA 同行同序；ridge 按 item 分组，alpha 只在训练 item 上选，无泄漏。
- selftest：SELFTEST PASS（4 条 WARN 在无 eval 的 step 5/10，预期）。
- 合成正对照（真实行布局）：植入几何 R² 0.983/0.981、lattice 1.000，shuffle ≤ −0.04；只有 identity 时 R² +0.10/−0.11，≈ 常数基线（约 −0.096）。

**重要**
1. 行池化（probe_v5.py:247、:268-273）：held+seen、A/B 槽、两种 op 混合打分，只存聚合分数；行为差只在 strict 题（s16 58/342、s17 77/342，约 642 个 answer 行的 9–12%）；登记 s16 在 18309 非 strict 准确率反向（all6 0.933 < only-dim 0.951）→ 池化读数可能读成无差别或反向。修：全部行拟合，保存逐行 OOF，按 strict / 非 strict held / seen / 答对答错 / 槽位分别打分（ans_hit、ans_set、item_slot 已存）。
2. answer 位置的输入搬运混淆（:247，目标 T[ans_c]）：E(c)=E(a)±E(b)，只搬运 A 就能 R²≈0.5。修：answer 位置也解码 E(a)、E(b)（ans_a/ans_b 已存），按 op 拆，报 E(c) 在 E(a)、E(b) 之外的增量 R²。
3. B 槽（probe_dump.py:143；508 在 op 之后）：深层可能编码答案而不是 E(b)，算得好的 all6 显得冲刷更多（假阴性方向）。修：按 item_slot 拆，A 槽（506）是干净的物品表征。
4. (a)/(b) 线性 CKA 不针对 dim 几何（:266、:289）：final norm 前的原始 residual，被 context 方差和 outlier 维度主导；只有 identity 时 (b) 地板 0.297。修：(a)/(b) 加 item 均值版（27×D）；加冻结解码器迁移（7323 拟合的 ridge 直接评 12000 / 18309）；主读数用 (c) 从 7323 到 18309 的 DiD。
5. seed 实际是重抽（HANDOFF_V5.md:73）：V5 多半在 A100/H100 copy 上算，登记 run 全在 B200，轨迹会发散；登记 8 seed 在 18309 的 strict 差 −6 到 +44（s16 +17.2、s17 +19.5、s23 −6）。修：每对先看 V5 自己的 strict 差和行为 DiD（登记 s16 +12、s17 +10）；差 < 约 10 pt 的那对回答不了问题（第 3 类，只影响该对），按常设授权补至多 2 个 seed；另用 run 内 342 题逐题 OOF 误差对比答对与否。
6. summary（:368-401）只打印每个 checkpoint 的 all6 − only-dim，没有 7323→18309 的 DiD 行。

**次要**：answer 位置 L0 退化（:120 cka_lin 返回 0.0，ridge = 常数，应打印 NaN）；零点不是 0（R² −0.096、lattice 0.087/0.072、(b) ≈ 0.30，要对照 shuffle 的 b_mean / r2_p95）；smoke 模型未训练（answer check 空检查，summary 按构造为 0）；CPU bf16 对 A100/H100 bf16，容差只打标记，strict 58–77 题上边界翻转约 1–3 pt，7323 对照的是 7250 的 eval；L6 是 final norm 前的状态，无归一化版本。

**最可能测错**：(a)/(b)/(c) 在池化的 1284/642 行上打分，读数被没有差甚至 only-dim 更好的行主导。
**可能漏检的真效应**：池化（1）和 B 槽（3）都偏向把 all6 的优势读没；只有 2 对重抽 seed、没有表征的 seed 方差，「无差别」不能当负结果写。
**证据撑得住的最强结论**：仪器复现了 trainer 记的 cka_dim（smoke |Δ| ≤ 3.9e-3），ridge 在真实行布局上能恢复植入的几何（R² 0.98）；要找的差异如果存在，应在 L5/L6 或 answer 位置的 strict 题上。
