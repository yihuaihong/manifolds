# pt6v2 code / design audit (exp-auditor, 2026-11-19; make_jobs 961ef81fe230, readout e4beaf64417a, submit 23ff3c308576, refs d2ec97b41a71)
材料缺失：L40S smoke 23689162 仍 PENDING（无日志）；test 3 tiny2 23689529 仍 RUNNING（all6 rc 0、658 s，其余 3 臂未完）。

**阻断：未发现。** 已核对的事实如下。
(a) 用 render_v2 渲染 97 个作业（只渲染到 /scratch/yh6210/tmp_audit/pt6v2_code/）。逐个对过命令行：cell 是 12×8+canary；seed 16-23；lr；window 0.05,0.4；all6 和 only-numbers 用 `--target helix3`，其余 only-m 用 `--target none --prior_domains m`，none 臂只给 `--target none`。卡型按 seed 分配，有 guard，错卡 exit 3。CHECK_PY 会核 prior_on 窗口、核未启用的 prior 从没被应用、核 gnorm_prior/gnorm_true=λ。
(b) 判定规则：N_m 按每个 manifold 自己的 none 均值取 argmax，平局取高 lr（readout_pt6v2.py:473,479）。g/h/u（:481-483）和 PC.manifold_verdict 的单侧配对 t、wins≥6、干扰判据都和 AUDIT 一致。sf_test 的 df=n-1（readout_ptdom.py:961-970）。hcube/d6 走 twin-free primary（:144-146）。
(c) 现有测试：selftest 25/25，mutants 8/8 killed；known 14/14（复现 pt6comb readout.txt 5/6，ptdom2 455 对，ptdom3 21 个数）；regression 80/80。
(d) A100 smoke：step-0 LM 偏差 4.8e-6（门槛 1e-3）；gate 双路径偏差 1.4e-8（DP_TOL 1e-3）。我在 A100 smoke run 上只读跑了 `--canary`：6/6 个 manifold 都出数，hcube n_tf=66。

## 重要（每条一行）
1. 【Q2/Q3，假阳性方向，读代码+数据确认】make_jobs_pt6v2.py:69 的 none 网格去掉了 4e-3。pt6comb stage-1 里 cities/countries 的 none 最好 lr 正是 4e-3（86.8/90.2，2e-3 时 86.0/89.6；脚本 /scratch/yh6210/tmp_audit/pt6v2_code/none_lr.py）。所以 N_m 只能落到 2e-3，g 被抬高约 0.6-0.8 分（预期 g≈5，方向不变）。AUDIT.md:27 说「规则偏保守」，这只在网格内成立；另外 N_m 落在下沿 2e-3 时没有 flag（readout_pt6v2.py:87 只标上沿 3.2e-2）。检查：出数后 cities/countries 另报 g 减 0.7 的敏感性行，或加一个「N_m at bottom lr 2e-3」描述性 flag。
2. 【Q7，假阴性方向，推测+数据】P 固定在 1e-3（:68）。这是 pt6comb B 的 all6 选择，和 2e-3 几乎打平：76.4 vs 75.9。缺 lr 的 domain 会被低 lr 拖成 NOT SHOWN：pt6comb s4 的 all6 在 1e-3/2e-3/4e-3 = 36.6/40.0/43.0，结果 NOT SHOWN；ptdom3 hcube 的 none 从 55.5@2e-3 涨到 75.7@1.6e-2（seed 0）。hcube 很可能重演 s4。检查：NOT SHOWN 的 manifold 同时首报同 lr 的 g_same_lr 和 u（已算，:489）。
3. 【Q8，推测】GATE A2.1 只要任一半失败，就把整个 headline 换成 GATE_TXT（:507-516、:539-541）。新组合（compass、hcube 的 prior）没有任何全长 run 的 LM share。400 步 smoke/canary 的窗口在 20-160 步，看不出这个数：smoke 0.36，而 pt6comb 早期也是 0.39-0.69，之后才稳定在 0.76（tmp_audit/pt6v2_code/share_curve.py）。检查：第一个 B200 all6@1e-3 跑完（约 0.45 h）就用 grad_dp.py 读 share，<0.5 再报协调者；只打标记，不拦。
4. 【Q5，读代码确认】report()（:560-628）缺两段文字：一是 AUDIT.md:27 承诺要打印的「N_m 在同一批 seed 上选、偏向 none」；二是 pt6comb readout.txt:70 的 reading 行（Tier L：比较类 target 自带答案；tied embedding；NOT SHOWN≠无效应；not detected≠已证明不干扰）。数字不受影响，但两个方向都可能读错，写结论时要从 pt6comb 照抄这些限定。

## 次要（合并一行）
make_jobs sha 只打印不 assert（:563、:1063；SRC :51-54 里没有它）· `--allow_partial` 下 MIN_WINS 仍为 6（readout_pt6comb.py:93），7 seed 时要 6/7 · per-half gate 和 A3 都没有最少 seed 数，而且 A3 按半检，比 pt6comb 的单 cell 更严（pt6comb all6@2e-3 median 0.76，离 0.5 远）· twin-free primary 的 window_g 是 None · FLOOR 线按来源不同（cities/countries 76-77 是 2-way 启发式线，只是标签）· step-0 LM 1e-3 是硬检查，但余量约 200×；L40S 未测，submit_pt6v2.sh:48-50 在 smoke 没 PASS 时拒绝 `--go` · canary 读出单次超过 2 分钟（时限够）· none 臂带 lam 5.0（不起作用）· hcube 的 n_tf 在 seed 16 和 seed 0 都是 66（split 固定，配对不受影响）。

## 十问速答
Q1 未发现（同 seed 同卡，配对差在卡内）。Q2 见重要 1。Q3 hcube 已用 twin-free 防记忆；比较类 domain 的 Tier L 是解读问题，见重要 4。Q4 未发现（λ 比、窗口、未启用 prior 都有硬检查）。Q5 重要 1、4。Q6 lr 网格边缘问题在重要 1、2；其余未发现。Q7 重要 2。Q8 重要 3；双路径门槛实测余量 ≥10^4×。Q9 见末行。Q10 placeholder 里没有注定落地板或天花板的 cell。pick 时注意：若选 d6，ptdom3 的 none twin-free area 只有 1.2-4.8，低于 FLOOR 线 13.3，很可能打 FLOOR 标签。

- **最可能测错的地方**：cities/countries 的 none 最佳 lr（4e-3）不在网格里，N_m 被迫取 2e-3，g 约被抬高 0.7 分，「偏保守」的说法对这两个 domain 不成立。
- **可能被测没或写没的真效应**：P 固定在 1e-3，对缺 lr 的 domain（hcube，参照 s4 先例）会把真实的 prior 增益判成 NOT SHOWN；GATE 失败会撤掉整条 headline，而 canary 预览不到这个风险；NOT SHOWN 时应首报同 lr 的 g_same_lr 和 u，不写「无效应」。
- **证据撑得住的最强结论**：开跑前，作业和读出都按 AUDIT 测它声称的量：97 个作业逐个核对过；读出复现了 pt6comb 5/6 和 476 个已知数；A100 smoke 的 4/4 检查和 6/6 个 manifold 读数都通过。没有阻断项，可以按现配置提交（L40S 等 smoke PASS 后再用）。
