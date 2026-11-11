# ptsz 修订审查（对抗式，全新子 agent，只读）— 2026-11-06

审查对象：runs/ptsz/AUDIT.md 新节「Amendment after the fresh build audit (before any result)」（AUDIT.md:370-460）及其
落地改动：runs/ptsz/readout_ptsz.py（cf7a9a9d2626）、runs/ptsz/select_ptsz.py（da5833367c67）、runs/ptsz/submit_ptsz.sh
（38c1f51658fb）、runs/ptsz/tests/smoke_report.py（098a807f3764）、make_jobs_ptsz.py（63002a20b835，N1）。对照：
scale_work/ptsz/*.pre_amend（readout d66129bde8ac / select a9bca4a4b1b8 / AUDIT 65fc27eeca48 = 构建审查时的版本）、
runs/ptsz/tests/audit_build.md、冻结 trainer runs/ptsz/code7/tools（run_guard.py、pretrain_range.py a3bce3ab63c1）。

我自己的检查都在 /scratch/yh6210/tmp_audit_ptsz_amend/ 下，没有改项目文件（本文件除外），没有提交 / 取消任何训练作业
（唯一提交的是 cpu_run.sh 的一个 CPU 模拟作业，见 I-0 和文末「附」）：
- reconstruct_gen.py / reconstruct_other.py：把现文件只撤回 N1 / I4 那几行，sha256 正好回到修订前的值
  （make_jobs_ptsz.py ae1a75c91d3e、submit_ptsz.sh 68cea25b58f1、smoke_report.py 90c040c09ebd）→ 这三个文件除了这几行
  什么都没改。pyc_strings.py / pyc_consts.py：从 tests/__pycache__ 里 11-05 18:31 的旧 smoke_report.pyc 读出旧判据，交叉印证。
- builder.py + runner.py + compare.py → compare.out：pre / post 两版 readout 在同一批合成 root 上跑（10 个完整 root，真斜率
  -1..3，seed 噪声 SD 4、尺寸离散 SD 1.5；外加 1 个列了 own failure 的 root），逐字段比较。
- sel_equiv.py → sel_equiv.out：pre / post 两版 select 在 6 个随机 stage-A root 上跑（48 个 arm，X / C 都有），比较决定。
- robust_indep.py → robust_indep.out：自己写的 area / OLS / 中位数 / 截尾 / 留一 seed，不 import 任何项目模块，直接读 run JSON。
- render_check.py → render_check.out：用当前生成器（python -B，不调 main，不写项目）渲染 stage A / 合成决定下的 X、C /
  冒烟作业，和磁盘上的文件逐字节比较。
- dead_test.py + fakebin_ok / fakebin_err（假 sacct）：DEAD 告警在 sacct 正常、sacct 报错退出、作业没写 claim 三种情况下的行为。
- ka_rerun.out：用当前 select + readout 重跑 `readout --ka`（tests/readout_ka.out 写于 00:53:46，比 select_ptsz.py 最后一次
  修改 00:54:58 早 72 秒），输出与 tests/readout_ka.out 逐字相同。
- sim_qualifier.py → sim_qualifier.out（CPU 作业 22047344）：I1 新加的「读法」限定语在真增长 / 无增长下多常触发。

**总评：未发现阻断。** 所有 run 都正常时，主 / 次检验的判定规则和所用数字逐位不变（pre / post 等价测试：10 个完整
root 的全部判定字段相同，6 个 root 共 48 个 arm 的选 lr 决定相同）。新增的稳健性数字按名字实现，独立复算到 4 位小数一致。
nice 改动在作业文件、提交脚本、生成器（含 X / C / 冒烟渲染）里一致。问题有两类：
- I1 的「读法」规则只看符号。在构建审查 I1 自己拿来举例的四个复用格子上，它判「没问题」；模拟里它几乎从不触发
  （I-0）。
- I5 新加了一个「整格 NOT DETERMINABLE」的杠杆，但读出不核对被列入的 run 是否其实已经正常跑完（I-A）；在 M8 的
  情形下，DEAD 告警会原样重复旧作业号（I-B）；告警覆盖不全，查询失败时还静默（m1）。
这些都不会产生错的斜率，但前者会让「均值增益被个别 seed 主导」照样被报成稳健，后者会让一个正常的尺寸格子被判成
NOT DETERMINABLE，而读出里看不出来。

---

## 重要

### I-0. I1 的「读法」规则只看符号，在它自己的动机样例上判「没问题」；模拟中几乎从不触发
- 位置：AUDIT.md:193-196、396-398（「a verdict whose median / trimmed slopes or leave-one-seed-out range fail to agree in sign
  with it rests on a few late none seeds … and the FINDINGS entry says so」）；readout_ptsz.py:345-369（只打印数字，规则写在
  AUDIT 里）。
- 为什么会测错：构建审查 I1 要防的是「均值斜率主要来自个别晚学会的 none seed」，这是一个**幅度**问题。修订把它落成了
  **符号**检查：
  - 在 I1 的动机样例（四个复用 L40S 格子）上，均值斜率 +2.16，中位数 +1.01，截尾 +1.17，留一 seed 最小 +1.21，全部同号。
    规则判「不依赖晚学会的 seed」，可是去掉一个 seed（505M s5），斜率就少了 44%，中位数只有均值的 47%。也就是说，
    这条规则恰好放过了它要抓的情形。
  - 模拟（见文末「附：限定语模拟」）：所有情形下，判成 SHOWN 之后限定语的触发率都 ≤ 1.2%。真增长下判对的 SHOWN 不受
    影响；无增长下的假 SHOWN（次检验 0.09-0.19）也几乎从不触发。它只在「一个 seed 把均值从约 0 抬起来」这种很窄的机制下
    触发（自测植入的那种）。
  - 后果：「中位数 / 截尾同号」很可能被写进 FINDINGS 当作稳健性的证明，而这几乎是空的检验。
- 验证：robust_indep.out（四格数字）、sim_qualifier.out（触发率）。建议在任何结果之前改成幅度规则，例如「中位数斜率、
  截尾斜率或留一 seed 最小值有任何一个 < 均值斜率的 1/2 → FINDINGS 写『斜率有相当部分来自少数 seed』，并给出这些数字
  和 late 计数」；或者干脆不设规则，只要求 FINDINGS 把三个数和均值斜率并列。阈值要事先定好，并在复用四格上演示：按上面
  的例子它会触发（1.01 / 2.16 = 0.47）。
- 置信度：高（数据 + 模拟）。

### I-A. 读出不核对 OWN_FAILURES 里的 run 是否其实已经跑完、check 通过，就把整个尺寸判成 NOT DETERMINABLE；I5 把「列入」的后果放大了很多，又没有冻结时点
- 位置：readout_ptsz.py:154-157（`own_here` 非空就直接 `return c`，不看 run JSON / check）；select_ptsz.py:166-172、209-215
  （select 这边会打印 M6 的 WARNING，但仍按列入处理）；AUDIT.md:399-412（I5 的程序：只有「作业结束却没有完成的 run
  JSON、重投一次后同样失败」才该列入）。
- 为什么会测错 / 失效：
  - I5 规定只有「重投一次后复现的失败」才能列入，但代码不检查这一点。readout 里，一个已完成、CHECK7 通过的 run
    被列入后，没有任何警告，该尺寸直接 NOT DETERMINABLE；四个新尺寸都在主检验里，136M / 391M / 731M 还在次检验里，
    所以写一行就能让主检验（和次检验）都读不出判词。修订前 readout 走 new_run，会打印「listed as an own failure but
    finished with a passing check」；修订后这条在 readout 里没了。
  - 修订前，列入一个已完成的 run 几乎没有效果：按全部 75 次 eval 计分，就是它本来的 area，只有 bpb 记最差。修订后，
    列入一个 run 会让整个尺寸作废。OWN_FAILURES.txt 是手写文件，AUDIT 没规定它在读出前冻结，所以看过结果之后也能
    「抑制」一个不想要的判词。ND 不会把 NOT SHOWN 变成 SHOWN，但它是一种选择性报告的通道。
- 验证：compare.out 的 `own` root（列入的 psz-731m-lr6e-4-win-ctx-s2 已完成、check 通过）：修订前打印 WARNING，然后正常
  判 SHOWN；修订后只有一行「PROBLEM 731M: recurrent own failure(s) [...] (amendment I5)」，主、次检验都是 NOT DETERMINABLE，
  没有任何警告。建议在任何结果出来之前写一个 amendment：
  - readout 碰到列入的 run 如果 JSON 已完成、check 通过，就报 PROBLEM「I5 does not apply」并拒绝把它当成 own failure；
  - OWN_FAILURES 每一行写明两个作业号（原始作业 + 重投），select / readout 用 sacct 确认这两个作业都已结束，而且都没有
    产出完成的 JSON；
  - 读出时把 OWN_FAILURES.txt 的 sha256 打进输出和 AUDIT。
- 置信度：高（读代码 + 合成 root 实测）。

### I-B. M8 的情形下，DEAD 告警原样重复旧作业号，按 I5 程序可能误判成「复现」，进而误杀一个尺寸
- 位置：select_ptsz.py:150-155（`dead_job` 只看 `.claim` 里的作业号）、176-182（ALERT 文本：「resubmit once … if the same
  failure recurs, list it」）；code7/tools/run_guard.py:27-32（squeue 超时 → 当作还在跑）、41-46。
- 为什么会测错：
  1. 原作业 J1 在第 1000 步 OOM（FAILED），`.claim` = J1，告警显示「job J1 ended FAILED」。
  2. 协调者按 I5 手工重投，得到 J2。如果没有先删掉旧 claim，J2 开始时 squeue 又慢，run_guard 的 `_running(J1)` 超时，
     返回 True，J2 打印「being computed by running job J1」后退出。这时 trainer 正常返回，CHECK7 在半截 JSON 上失败，
     J2 也变成 FAILED。
  3. `.claim` 仍然是 J1，所以下一次 select 打印的告警和第 1 步一字不差。告警文本既不含 J2，也不看 J2 的日志，代码区分
     不了「J2 复现了 OOM」和「J2 根本没训练」。
  4. 按 I5 的字面程序，下一步就是列进 OWN_FAILURES，整个尺寸 NOT DETERMINABLE。731M stage A 一个尺寸就是 32 × ~9 h。
  AUDIT 的防线全是手工：先查 / 删 `.claim`，再 grep「run guard:」。而现在 Slurm 正慢，这条路径的前提正好成立。
- 验证：dead_test.py 的 `ok` 情形，说明告警只取决于 claim 里的作业号。修法：告警里同时打印该 run 名下最新的作业号
  （`sacct --name llmadd_<run>`，或者 SUBMITTED / 重投记录）及其状态；最新作业的日志里有「run guard:」就打印
  「NOT A RECURRENCE (run guard exit)」；只有两个不同的作业号都进入了训练（日志里有 `model:` 一行，或者 claim 被改写过）
  才提示「recurs」。
- 置信度：中。路径是读代码推出的，多常发生取决于 squeue 有多慢。

---

## 次要

- **m1 DEAD 告警覆盖不全，查询失败时静默**（select_ptsz.py:138-147、150-155；作业体 make_jobs_ptsz.py:350-366；
  pretrain_range.py:162-166）。AUDIT.md:405-407 写的是「任何作业结束却没有完成 run JSON 的 run」都会告警，代码做不到：
  - claim 只在 trainer 的训练调用里写。在它之前退出的作业都不会留下 claim：HOLD（exit 0）、trainer sha 不符（exit 3）、
    GPU guard（exit 4）、数据 sha256 不符（exit 3）、`--dry_inputs` 崩溃（例如 /scratch 被清理后 ModuleNotFound）。这些
    run 永远显示 missing，不告警。
  - sacct 返回非零但不抛异常（例如 slurmdbd 连不上，stdout 为空）时，STATES 为空，既没有 ALERT 也没有 WARNING。
    dead_test.py 的 `err` 情形实测：状态 running，输出为空。
  - 作业在 CHECK7 期间死掉（JSON 已完成、没有 check）时，永远显示 nocheck，不告警。
  - 这几种情况下的决定都不受影响（dead 和 running 都算等待），只是会等很久，而且是静默的。修法：用 SUBMITTED_<run>
    （里面有作业号）补充 claim；检查 `o.returncode` 和 stderr；对「请求了却没有返回行」的作业号打印 WARNING。
  - 置信度：高。
- **m2 I5 改了预注册的规则，但没记录用户批准**（AUDIT.md:373-374、399；AUDIT.md:144-147 的删除线）。
  - 原来 F87 rule 2 是「own failure 按 0 计分」，现在是「整格 NOT DETERMINABLE」，NOT DETERMINABLE 的条件因此扩大了。
    修订写在结果之前、有删除线、方向也是构建审查 I5 建议的，所以不算「悄悄改」。
  - 但 AUDIT.md:373 的「The registered PRIMARY and SECONDARY decision rules are unchanged」只对斜率判据成立；而且
    「The user's decisions」列表里，N1 / I1 / Secondary 都引了用户原话，I5 没有。
  - 建议补记用户是否同意 I5，并把 373 行改成「斜率判据不变；NOT DETERMINABLE 条件按 I5 扩大」。复用的 F87 格子当年
    没有 own failure（构建审查核对过 56 个日志），所以新旧格子之间没有规则不对称。
  - 置信度：确认（读文档）。
- **m3 「读法」的因果措辞**（AUDIT.md:196、397-398）。功效问题见 I-0；另外，「rests on a few late none seeds」是一个因果
  归因：中位数或截尾斜率异号，也可能来自先验臂的一个离群 seed，不一定是 none 学得晚。应同时看 late 计数，以及被留一
  去掉的那个 seed 在哪一臂偏离。
- **m4 `LATE = 45` 是绝对阈值，会随尺寸系统性升高**（readout_ptsz.py:77、361）。
  - 1.07B 的 none 有 2/8 个 seed 低于 45（43.9、44.0，robust_indep.out），看起来是这个格子正常分布的一部分，不像 505M s5
    （34.2）那样的离群点；
  - 731M 的期望 none 是 50-60。
  所以大尺寸的「late 计数」会随 none 均值下降而机械地变多，和尺寸趋势混在一起。只打印，不影响判定。可以同时打印一个
  相对阈值，例如 < 本格中位数 - 15。
- **m5 次检验的误判率行只引了 increase 的误判率**（readout_ptsz.py:78-80、479）。不管次检验判成什么，这一行都会打印；判成
  SHOWN (decrease) 时，相关的是 decrease 那一行（0.06 / 0.13 / 0.19 / 0.20，AUDIT.md:209）。
- **m6 回退方案的落地比 AUDIT 写的复杂**（AUDIT.md:408-412；select_ptsz.py:90、100）。integrity() 写死了 lm_chunk_tokens -1 /
  lm_chunk 0；CHECK7 的 intent 来自冻结的 pttpp intent7。所以「两臂同时用 --lm_chunk_tokens」也会让两臂都被 withheld，
  除非 select 的 integrity 和 intent 一起改，不是「a generator amendment」就够。另外，一个尺寸改用 chunked LM loss 后，和
  复用格子（lm_chunk 0）的路径就不同了。这是显式失败，不会静默。
- **m7 readout 把 dead 归为 PROBLEM，而不是 NOT FINAL**（readout_ptsz.py:172）。只是标签问题，结果同样是 NOT DETERMINABLE。
- **m8 文档残留**：
  - AUDIT.md:333（旧测试表的 args_diff 一行）仍写「--nice 20 -> 1000」，没有标注；
  - smoke/SUBMITTED_smoke731 仍是「Submitted batch job 21999324」，runs/ptsz/SUBMITTED_smoke731 写的是 22011938，两个标记
    不一致；
  - tests/testonly.out 的开始时间估计还是 nice 1000 时的；
  - tests/args_diff.py:31 把任何 `#SBATCH --nice=` 行都当作允许的差异，不断言具体值（submit_ptsz.sh:39 断言了，所以无害）。

---

## 逐题回答

### 1. 判定规则和判定所用的数有没有变（只允许新增打印）
- **斜率判据和数字：未发现改动。**
  - readout 的 diff 没碰 ols / ols_cov / boot_idx / interval / slope_block / verdict / EXPECTED / MARGIN，FINDINGS 核对也没变。
  - 合成 root 的等价测试（compare.out）：10 个完整 root 上 verdict / verdict2 的字符串、primary / secondary / h1 / h2 / s_h1 /
    s_h2 / criteria / criteria2 / ten / ten_b200 / loso / arms / swaps / nob11 / lnR50 / bridge294 全部逐位相同，判词覆盖了
    SHOWN、NOT SHOWN (a)(b)(c) 和 secondary SHOWN (decrease)。输出只差一行 sha，外加 5-6 行新增打印（robustness × 3、
    误判率、读法）。
  - select（sel_equiv.out）：6 个随机 root、48 个 arm（X / C 混合）的决定、lr、均值、bpb 逐项相同，打印行也相同。
- **会改变结果的只有 I5**：own failure 从「按 0 计分」变成「整个尺寸 NOT DETERMINABLE」（m2）。另外「What the verdicts may
  be called」加了两条限定语（I3 的读法、I1 的读法），只影响措辞，但 I1 那条只看符号，基本没有区分力（I-0、m3）。
- 生成器、提交脚本、冒烟判据：sha 重建证明只有 N1 / I4 那几行变了（见开头）。

### 2. 新增的稳健性输出：按名字实现、正确
- 实现（readout_ptsz.py:345-369）：
  - 中位数：np.median(8 个增益)，即第 4、5 个的平均；
  - 截尾：按增益排序，去掉最小、最大，取 6 个的均值，等于 8 个里截 20%；
  - 留一 seed：主检验 64 个、次检验 24 个、10 点 80 个，被去掉的那格取其余 7 个 seed 的均值；
  - late：每格 none area < 45 的 seed 数。
  打印位置：主检验 verdict 之后、次检验 verdict 之后、10 点描述性斜率之后（:459-461、479-482、501-503）；不进入任何判据。
  合成测试里植入一个晚学会的 seed，期望的数字我手算过（主检验 Δb = 3.125 × 1.567 / 11.161 = +0.439，次检验 +1.265），
  和自测一致。
- 我的独立复算（robust_indep.out，只读 run JSON；复用格子 = readout REUSED：ptlr 53m-lr8e-3 / 2e-3、190m-lr4.8e-3 / 1.2e-3、
  500m-lr9.6e-3 / 1.2e-3，pt300x p3x-93m-lr2.4e-3 / 1.2e-3）：

  | 四个复用 L40S 格子 | 我的值 | readout --ka | AUDIT / 协调者 |
  |---|---|---|---|
  | 均值斜率 | +2.1608 | +2.1608 | +2.1608 |
  | 中位数斜率 | +1.0059 | +1.0059 | +1.006 |
  | 截尾斜率 | +1.1673 | +1.1673 | +1.167 |
  | 留一 seed 范围 | [+1.2126 (-505M s5), +2.4741 (-53M s5)] | 最小值 -505M s5 +1.2126 | +1.213 |
  | late 计数 | 53M 0, 93M 0, 190M 0, 505M 1 | — | — |

  F99 六格：中位数斜率 +1.8228（= KA）；截尾 +2.4413；留一 seed [+2.4656 (-505M s5), +3.0441 (-1.07B s7)]；late 505M 1，
  1.07B 2。各格 8 seed 的 gain / none / prior 也复现了 pt300x audit_results（22.7927 / 28.8741 / 23.5299 / 29.1067 / 32.1885 /
  36.1292）。
- 「被复用的 4 个 L40S 旧格子」我按 readout 的 REUSED 理解为 53M / 93M / 190M / 505M：其中 93M 来自 pt300x，其余来自
  ptlr（F87）。

### 3. I5：误杀与漏判；Slurm 慢或查询失败时
- 可能误杀正常 run / 格子：
  - I-A：列入一个其实正常的 run，readout 不报警；
  - I-B：M8 情形下的重复告警被当成「复现」；
  - 一行写错尺寸，例如本该写 731m 写成 391m，会多杀一个尺寸（select 不报警，因为名字合法）；
  - 有 ckpt 的尺寸 TIMEOUT 后续跑：每段都在前进，不该算「复现」（告警里的「evals n/75」能看出进度）。
- 列入之后的逻辑本身没问题：
  - 列入任一臂的任一 run → select 两臂都 'nd'，不写 X / C / STOP（decide:244-245 在 C 检查之前）；
  - readout 整格 ND：主检验 ND；尺寸在 136M / 391M / 731M 时次检验也 ND；294M 只影响主检验；
  - C 已决定之后再列入，同样 ND；
  - 列入不会影响复用格子。
- 漏掉真的失败：m1 那几类（trainer claim 之前的失败、sacct 报错退出、CHECK7 期间死掉）。OOM 发生在 trainer 写 claim
  之后，所以构建审查 I5 关心的「第 1000 步 OOM 不再静默」是做到的：dead_test.py 的 `ok` 情形打印 ALERT，状态 dead，
  仍算等待。
- claim 命名和内容与 run_guard 一致：`<out>/<tag>/<run json>.claim`，内容是 `SLURM_JOB_ID\n`（run_guard.py:41-49；
  pretrain_range.py:162）。select 的 `d / (rf + ".claim")` 和 `isdigit()` 对得上。sacct 输出格式实测过：
  「21999324|CANCELLED by 4550886」解析成 CANCELLED。
- Slurm 慢 / 查询失败时：
  - < 180 s：正常，只是会阻塞；select 和 readout 每次各调一次；
  - 超时：TimeoutExpired 被捕获，打印 WARNING，没有告警；
  - 非零退出：静默，没有告警（m1）；
  - 部分作业号没有返回行：这些 run 不告警（安全方向）。
  这些情况都不改变任何决定，因为 dead 和 running 都算等待。trainer 那边的 run_guard 在 squeue 慢时会让重投作业空跑，
  见 I-B。

### 4. smoke_report.py 的显存判据
- 单位和字段都对：smoke_seg.py 里 `max_reserved = torch.cuda.max_memory_reserved()`、`total =
  get_device_properties(0).total_memory`，都是字节；`2 * 2**30` 是 2 GiB 的字节数（smoke_report.py:120）。判据作用于 731m 的
  A、B、C、D 段，和 AUDIT.md:385 一致；每段是一个独立进程，所以是各段自己的峰值。`if k in sm` 不会放宽判据，因为
  A ok、B ok、C D E ok 要求 A–E 的 SMOKE 行都在。
- 其余判据没有放宽：撤回这一行后 sha 正好回到 90c040c09ebd；旧 pyc 里的判据键也是同一组（A ok / B ok / C D E ok / resumed /
  step-0 lm within 1e-3，C 731m + E 136m，容差 1e-3）。
- 说明：CUDA context 不在 reserved 里，实际余量约 1.5 GiB。nvidia-smi 的最大 memory.used 只打印、不作判据，和 AUDIT 的
  说法一致（协调者自己读）。

### 5. nice = 100 的一致性
- 作业文件：128 个都是 `#SBATCH --nice=100`；我用当前生成器独立渲染，128/128 与磁盘逐字节相同（render_check.out）。
- X / C：在 tmp 里用合成 decisions 渲染 136m / 731m 的 X 和 C，都是 nice=100（header() 是所有 stage 共用的）。
- 提交脚本：submit_ptsz.sh:39 用 `grep -qx '#SBATCH --nice=100'` 精确匹配。
- 冒烟：当前渲染和已提交的 job_smoke731.sbatch 只差 nice 这一行；smoke/SUBMITTED_smoke731 存在，所以 `--smoke` 会走
  assert 显式拒绝，AUDIT 的说法成立。实际作业 22011938：PENDING，nice 100，priority 12,474；21999324 是「CANCELLED by
  4550886」。
- 优先级顺序：队列里 ptdom 是 nice 20（b200 分区），ptshare 是 nice 2000（l40s_public），所以 ptdom > ptsz > ptshare 仍然
  成立。
- select 自测已改成 100；args_diff 的输出写的是 20 -> 100。
- 遗漏：只有文档残留（m8）。

### 6. 其他副作用
- 冻结文件：
  - 导入当前生成器时，12 个 FROZEN 的 sha 全部通过断言；
  - ptsz_refs.json 831d6694cd6f、code7 pretrain_range.py a3bce3ab63c1、smoke 两个文件 08cf626a08f8 / c62eff092e5c 都未变；
  - regress_frozen.out PASS：FROZEN 26/26，pttpp 302、ptlr_1b 12、pt300x 84 个作业文件逐字节重渲染相同。
- 其他批次：修订窗口（11-05 20:20 起）内，llmadd_torch 里 runs/ptsz 之外被改的只有：
  - make_jobs_ptsz.py、scale_work/ptsz/*.pre_amend、watch/ 下的状态文件；
  - runs/ptdom 的作业文件、SUBMITTED 标记和 AUDIT.md（00:50–01:18）。这是 ptdom 自己的提交，ptsz 的代码不引用 ptdom。
  **未发现**本修订造成的副作用。
- 已提交作业：只有冒烟 22011938。没有 SUBMITTED_psz-* 标记，没有 runs/ptsz/out，logs 里没有 llmadd_psz-* 日志，所以确实
  是「任何结果之前」。

---

## 附：限定语模拟（m3；sim_qualifier.py）
自己实现的判定规则：bootstrap 2000 次（规则是 10000 次，用来估频率够了）。seed 噪声取四个复用 L40S 格子的 32 个格内
中心化增益，有放回抽样（含 505M s5）；尺寸离散 tau 服从正态；每行 2000 次模拟。

（CPU 作业 22047344，cs706）。池子：32 个增益，SD 6.61，最大 +23.6（505M s5），最小 -10.6。「增长」= 新旧八格都在 F99 的
线上（+2.79 / e-fold）；「无增长」= 常数。限定语 = 中位数斜率或截尾斜率与 b 异号，或者留一 seed 范围跨过 0。

| 情形 | tau | 主检验 P(SHOWN 增长) | 判 SHOWN 后限定语触发 | 次检验 P(SHOWN 增长) | 判 SHOWN 后限定语触发 |
|---|---|---|---|---|---|
| 增长 | 0 | 0.956 | 0.000 | 0.651 | 0.002 |
| 增长 | 1.85 | 0.870 | 0.000 | 0.617 | 0.002 |
| 增长 | 3.12 | 0.674 | 0.000 | 0.570 | 0.003 |
| 无增长 | 0 | 0.042 | 0.012 | 0.093 | 0.005 |
| 无增长 | 1.85 | 0.044 | 0.000 | 0.140 | 0.000 |
| 无增长 | 3.12 | 0.042 | 0.000 | 0.193 | 0.000 |

这个实现本身对得上构建的模拟：
- 功效 0.956 / 0.870 / 0.674 和 0.651 / 0.617 / 0.570，对应构建的 0.964 / 0.877 / 0.727 和 0.632 / 0.587 / 0.556；
- 次检验的假 SHOWN 0.093 / 0.140 / 0.193，对应 0.074 / 0.115 / 0.186。

结论：按符号的限定语对真、假 SHOWN 都几乎不触发，基本没有区分力（I-0）。

---

这次修订最可能出错的地方：I1 的稳健性数字算对了，但预注册的「读法」只比较符号。它在构建审查 I1 自己的例子上判
「不依赖晚学会的 seed」，模拟里也几乎从不触发。所以一个主要由 731M 一两个晚学会的 none seed 撑起来的 SHOWN，照样会被
报成「中位数 / 截尾同号、稳健」。其次是 I5 新加的整格 NOT DETERMINABLE 杠杆：读出不核对被列入的 run，加上 M8 情形下
告警重复旧作业号，可能误杀一个正常的尺寸。
