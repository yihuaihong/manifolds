# pt6v2 Amendment V6：代码 / 设计审查（exp-auditor，2026-11-22，只读）
**判定：可以全量跑。** 没发现会让作业崩溃、或让位置 / 层 / 标签出错的 bug。smoke 作业 24062161（only-dim s16 @18309）的仪器检查全部通过：
- 运行：rc=0，817 s，MaxRSS 13.4G / 48G。
- 输入：args 和 eval sets 等于 run JSON；' is' 在 509、span 在 510–511，解码都对。
- 读出：末层 lens == logits（8/8 集）；hits 与 V5 CPU probe 0 题不同；strict 0.707 等于日志；k=0 逐位复现 clean；DLA 加和误差 1.6e-7；lens argmax 一致率 0.987–1.0。
- 阻断项：B1、B2 是第 3 类（个别 cell 回答不了它的问题），B3 是汇总里的方向问题。
- 建议：提交前先把 B2 的 null 加进 A2，小改，约 +12 min/任务；B1、B3 可以事后用已存文件修。核对脚本和输出都在 /scratch/yh6210/tmp_audit/pt6v2_v6/。

## 阻断
**B1 ［第 3 类］torus 读数按构造落在地板。** 受影响的 cell 是 torus 的 A1 R2 / nearest、A1 dec overlap、A1 transfer、A4 的 CKA(·, coords)。
- 位置：probe6_inner.py:223,404,419,460–482,536,546，目标都取 DTAB['torus']，即频率 1 的 (1,0)/(0,1) 两个字符。Z6×Z6 上任何一对生成字符都能编码答案，没有 torus prior 的臂没有理由恰好选这一对。
- smoke 实测：
  - DTAB 目标：torus R2 在 −0.05 到 −0.11 之间，等于 shuffled null；nearest 0.000。
  - 但 torus held acc 是 0.70，torus 自己的子空间被消融后降到 0.046，说明答案码存在。
  - 字符扫描（check_smoke_out.txt）：L4–6 排前面的字符是 (1,5) / (2,0) / (2,5) / (2,3)，CKA 0.52–0.61；DTAB 4 维的 CKA 只有 0.02–0.11。
  - 同一个 grouped ridge：top-2 字符作目标 R2 = 0.62 / 0.82 / 0.82（L4 / L5 / L6），DTAB 作目标 R2 = −0.94 / −1.48 / −1.31。
- 后果：
  - 如果 all6 的 prior 把答案码对齐到 DTAB，臂间差会被读成「only-dim 不共享 / 读出坏了」，实际测到的是约定是否对齐，这是假阳性。
  - 「两臂都共享」对 torus 会被假否定。
- 不受影响：pc overlap 和 A2，它们用 class-mean PCs，与约定无关；dim / box5 的 R2 是 0.94–0.98。
- 修法：在 seen 行上选 R2 最高的生成字符对作 torus 目标，两臂各自报（或者报所有生成字符对里的最大值）。ap 状态已存在 states_ans.npz，可以事后算；A4 的 unembedding 行要重载 ckpt，CPU 秒级。
- 置信度：读代码确认，并有实测。

**B2 ［第 3 类］A2 的 random 对照剂量不匹配。** 受影响的是所有 h、k 下「box5 / torus 比 random 更伤 dim」这一比较。
- 位置：probe6_inner.py:555,642。random 是各向同性的 D×k 子空间，只移除 dim_held 中心化方差的 0.7–1.6%；box5 的 class-mean PCs 移除 19–26%（h4–6，k3）。「比 random 更伤」对任何高方差方向都必然成立。
- 剂量匹配的两个候选（check_dose2_out.txt，h4–6，k3）：

  | 子空间 | 移除的 dim_held 方差 | 与 dim own PCs 的 overlap |
  |---|---|---|
  | box5 class-mean PCs | 19–26% | 0.41–0.44 |
  | 按 dim_held 协方差抽的 random | 21–25% | 0.28–0.39 |
  | box5 行打乱标签的 class-mean PCs | 15–19% | 0.20–0.32 |

  h3 上打乱标签的版本反而移除更多（0.34 vs 0.31）。
- 所以 smoke 里 dim_held h4 k3 的「box5 0.243 vs random 0.779 vs own 0.058」分不开两种解释：box5 与 dim 共享答案码，或者 box5 在 ' is' 处的高方差方向（模板 / 操作数）恰好也占 dim 的方差。
- 修法（约 10 行）：
  - 在 :637 的循环里加两个 null source，各抽 5 次：box5（和 torus）seen 行打乱 c 之后的 class-mean PCs；orth(H_E.T @ G)。
  - 每个条件记录移除方差的比例。
  - 判定改看「同剂量下 box5 是否比这两个 null 更伤」。
  - 不加的话，A2 要事后重跑一遍。
- 置信度：读代码确认。剂量是实测的；这两个 null 下的伤害还没测。

**B3 ［第 1 类］汇总里 A2 的臂间列可能读反方向。** 位置 probe_v6_aggregate.py:44–56,91–92。
- seed-mean 列算的是两臂消融后绝对准确率之差。spec 要的是降幅之差：DiD = (clean−abl)_all6 − (clean−abl)_only-dim。
- clean 差（strict 在 18309 约 +20 pt）会主导这一列。例：all6 从 0.90 降到 0.45，only-dim 从 0.70 降到 0.40。这一列 = +0.05，看起来像 all6 伤得更少；实际 all6 的降幅多 15 pt。
- 修法：事后在 aggregate 里加 drop 行和 DiD 列。不影响提交。
- 置信度：读代码确认。

## 重要
- **功效不足。** dim_held 的 A2 主指标是 strict（n=58，binomial SD 约 6 pt）；两 seed 均值的 DiD，SD 约 6 pt，低于约 12 pt 的臂间差读不出来。建议把 held 全集（n=342，SD 约 2.5 pt）并列为主读数，并给 random 补上 .held（aggregate:55）。置信度：推测，按 binomial 算。
- **dec overlap 不能当「不共享」的证据。** probe6_inner.py:417–441 用 ridge 系数的列空间，是 Σ⁻¹ 形状的读方向，即使写方向共享也可能很低。smoke 里 dim–box5 的 dec overlap 是 0.06–0.11，不高于 covmatched p95（0.14–0.30）；而 pc3 overlap 是 0.41–0.44，高于 row-shuffle p95（0.22–0.30）。以 pc overlap 为主读数。置信度：读代码确认。
- **anchors（cities / countries，ap 取 ':'）剂量很小。** 它们只移除 2–12% 的方差，所以「像 random」有一部分来自剂量低和模板不同。而且在 h5 k6，cities 已经把 dim held 降到 0.579、strict 降到 0.466，并不像 random。报 anchor 结果时要带上剂量。置信度：实测。
- **torus_held 在 h3 的正对照无效。** own 消融后 0.649，clean 是 0.702，这个 cell 回答不了问题；三个域在 h4–6 的正对照都有效。置信度：实测。
- **null 只抽 10 次，p95 很粗。** n_shuf=10 时，A1 shuffled-label、pc row-shuffle、transfer 的 p95 实际上是 10 次里第二大的值。改报 z 和 max。置信度：读代码确认。
- **只在 ap 编辑。** 后面的 block 能经 attention 从别的位置把删掉的成分补回来，所以 h3 的效应偏小。这是设计限度，不是 bug；结论只能说「ap 处、第 h 层的必要性」。

## 次要
缺 seed 时 aggregate:92 的 nanmean 会静默变成单 seed 均值；A4 的 class-mean vs coords CKA（:546）没有 null；L6 上 fp32 lens 与 bf16 模型的 argmax 不一致约 1.2%；GPU 日志里的 hits 与 probe 有 1 / 1 / 5 题不同（跨设备，只标记）；全量 n_rand=20 会让 A2 从 635 s 增加到约 30 min，加上 B2 之后每任务约 45–50 min，低于 2:30 的时限。

- **最可能测错的地方**：torus 的 A1 / A4 用 prior 的 (1,0)/(0,1) 字符作目标，没有 torus prior 的臂按构造落在地板（R2 ≈ −0.1，用 top-2 字符则是 0.62–0.82），臂间差测到的是约定是否对齐，不是是否共享。
- **可能被测没或写没的真效应**：only-dim 的 torus 共享（被 DTAB 约定压到地板）；把 dec overlap ≤ null 写成「不共享」（pc3 已经高于 null）；strict n=58 只能分辨约 12 pt 以上的 A2 臂间差；aggregate 的绝对值差可能把 DiD 方向读反。
- **证据撑得住的最强结论**：仅限 only-dim s16 @18309 单个 ckpt，null 次数按 smoke。
  - dim 的答案在 ' is' 处、L4–6 可线性解码（R2 0.97–0.98，strict nearest 1.0）；在 ap 处删掉 dim 自己的 3 维子空间，dim_held 从 0.79 降到 0.06（strict 从 0.71 降到 0.03）。
  - box5 子空间与 dim 的重叠高于标签打乱 null（pc3 0.41–0.44 vs p95 0.22–0.30），box5→dim transfer 0.50 vs 随机投影 p95 0.25–0.30。
  - 「共享答案码」在 A1 上初步成立；A2 的「比 random 更伤」在剂量匹配前不算证据；torus 是否共享目前无法判断。
