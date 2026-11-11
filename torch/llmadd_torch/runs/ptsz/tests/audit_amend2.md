# ptsz 修订 2 审查（对抗式，全新子 agent，只读）— 2026-11-06

审查对象：runs/ptsz/AUDIT.md「Amendment 2 after the audit of the amendment (before any result)」（AUDIT.md:467-531）及其落地：
runs/ptsz/select_ptsz.py（6b5bb0ea3c8f）、runs/ptsz/readout_ptsz.py（5825302449b8）。对照：scale_work/ptsz/*.amend1（select
da5833367c67 / readout cf7a9a9d2626 / AUDIT b899a4e8a1b4）、*.pre_amend（select a9bca4a4b1b8 / readout d66129bde8ac /
AUDIT 65fc27eeca48）、上一轮审查 tests/audit_amend.md、tests/audit_build.md、冻结 trainer runs/ptsz/code7/tools
（run_guard.py b944dc859c7a、pretrain_range.py a3bce3ab63c1）。

我的检查全部在 /scratch/yh6210/tmp_audit_ptsz_amend2/ 下（三版文件拷到 pre/、amend1/、cur/，合成 root 都是拷贝，sacct 都换成
假函数，没有调用 Slurm 写操作）。项目里只写了本文件；没有提交、取消任何作业；所有 python 都带 -B，项目 __pycache__ 未变
（runs/ptsz/__pycache__ 仍是 01:51 的）。脚本与输出：
- ast_compare.py → ast_compare.out：两版之间逐函数（含嵌套函数）和顶层语句的 AST 比较。
- runner3.py + compare3.py → compare3.out：三版 readout 在上一轮审查的 11 个合成 root（10 个完整 root，真斜率 -1..3；外加
  1 个 own root）的各自拷贝上跑，逐字段比较；同时检查 `res` 能否按 main() 的方式 json.dumps。
- sel3.py → sel3_compare.out：三版 select 在上一轮的 6 个随机 stage-A root（48 个 arm，X / C 都有）上跑。
- of_test.py → of_test.out：own_failures() 和 DEAD / ALERT 的边界情况（作者自测之外的 21 种）。
- fr_test.py → fr_test.out、fr_corrupt.py → fr_corrupt.out：FIRST_READOUT.json 冻结逻辑（经 readout 自己的 readout()）。
- json_main.py → json_main_{pre,cur}.out：readout main() 加 `--json`。
- ratio_indep.py → ratio_indep.out：I-0 比值的独立复算（不 import 任何项目模块，直接读 run JSON）。
- sim_label.py → sim_label.out：I-0 标签在真增长 / 无增长下的触发率（自写实现，每行 1000 次，bootstrap 1000）。
- render_check.py → render_check.out、names_check.py → names_check.out：用当前生成器渲染 A / 合成 X、C / 冒烟，与磁盘比较；
  作业名 / 日志路径 / --out 与 own_failures() 的假设一致。
- rerun_{select_selftest,readout_selftest,ka}.out：从拷贝重跑作者的自测和 KA（工作目录在 st/）。

**总评：未发现阻断。** 判定规则和判定所用的数在三版之间逐位不变（10 个完整 root 的全部判定字段、48 个 arm 的选 lr 决定
都相同）。integrity() 恢复得一字不差。I-0 的比值算对了，只打印。I-A / I-B 的校验大体按说明实现。关键的一点：有了 (a) 之后，
OWN_FAILURES 已经不可能改变任何判词（见第 3 题开头），所以这一块剩下的问题都只影响标签、进度和提示文字。

唯一的「重要」项与修订 2 的改动无关，但修订 2 没有发现它：`readout_ptsz.py --json` 在第一次有完整结果时必然崩溃（R1）。

---

## 阻断
未发现。

## 重要

### R1. `readout_ptsz.py --json OUT` 在最终读出时必然 TypeError（修订 1 引入，修订 2 未测、未修）
- 位置：readout_ptsz.py:363（`loso[(c["cid"], s)]`，键是元组）→ :508 / :529 / :550（`robust` / `robust2` / `robust10` 存进
  `res`）→ :964（main 里 `json.dumps(res, ..., default=...)`；`default` 只管值，不管键）。
- 为什么：主检验 8 格都 final 时才算 `robust`，3 个新格都 final 时才算 `robust2`。所以 readout_presubmit（没有 final 格）
  和作者的自测（直接调 readout()，不走 main）都暴露不出来。上一轮审查的 runner.py 自己把 loso 键转成了字符串，也把它掩盖
  了。pt300x 的惯例是用 `--json` 写 runs/pt300x/readout.json，它的结果审查 tests/audit_results.md 引用的就是这个文件。
  ptsz 照做的话，进程会在打印完判词、写完 FIRST_READOUT.json 之后 exit 1，JSON 不出。到时候再修，就成了「读过结果之后改读出
  代码」：虽然只是序列化，在预注册纪律下也很别扭。
- 验证：
  - json_main_cur.out：`TypeError: keys must be str, int, float, bool or None, not tuple`；
  - json_main_pre.out：同一个 root，修订前版本 OK；
  - compare3.out：10 个完整 root 上 amend1 和 cur 都是 TypeError，pre 都 OK。
  修法：把 loso 存成 `{f"{cid}|{s}": v}` 再放进 res（或者 main 里转换），然后加一条 `main --json` 的自测。
- 置信度：高（实测）。

---

## 次要

- **m1 I-A (a) 比 I5 自己的定义宽：已完成 JSON、但 CHECK7 失败或没有 check 文件的 run 也能被 honoured**
  - 位置：select_ptsz.py:210（只排除「完成且 check 通过」）。I5 的原文（AUDIT.md:404-406）是「a run whose job ended without a
    finished run JSON」。
  - 例子：构建审查 M5 说 391M 只有 CPU 的 step-0 参考。如果 GPU 和 CPU 差 > 1e-3，每个 391M run 都会 CHECK7 失败、exit 4、
    状态 FAILED；两次作业都进入了训练，满足代码的全部条件，就能被列入。规则 6 的处理是「withheld，协调者调查」，调查之后可能修好
    参考值；列入则变成 I5 的 ND，不会再回头看。判词上两者都没有结论（见第 3 题开头），所以只是标签和流程问题。
  - 验证：of_test.out 的 B（checkfail）和 C（nocheck）两种情况都 honoured。
  - 建议：(a) 改成「没有完成的 run JSON」，与 I5 一致；CHECK7 失败的 run 拒绝列入，并提示按规则 6 处理。
  - 置信度：高。
- **m2 FIRST_READOUT 冻结：时点和 I5 的程序冲突，冻结本身也很脆**
  - 位置：readout_ptsz.py:424-431；select_ptsz.py:179-181、196-198、250-256。
  - (i) 触发点是「readout 第一次看到某个 final 新格」，不是「第一次看到 ptsz 结果」。在这之前，select 已经打印过每个 lr 的
    stage-A 均值，run JSON 和 check 文件里也已经有 area。
  - (ii) 第一个 final 格（136M 或 731M）一出现，以后任何尺寸在 C 阶段出现的真复现都不能被 honoured。I5 的程序和 ALERT 的
    文字都说「列入 OWN_FAILURES」，但列入后会被忽略，那一格只能一直 NOT FINAL。见 fr_test.out 情形 2：只有 136M final 时
    冻结，之后 731M 的一个 C seed 两次 OOM，条目「added after the first read-out」被忽略。
  - (iii) 冻结只靠这一个文件：
    - 删掉它，下一次 readout 会用当时的条目重写，而且没有任何记录（of_test.out M2、fr_test.out 情形 3）；
    - 文件写坏（write_text 不是原子写，而 /scratch 在 94%）时，load_json 返回 None，冻结被静默关闭，没有 WARNING
      （fr_corrupt.out）。
  - 判词不受影响。
  - 建议：用 tmp + rename 原子写；写完 chmod a-w；sha 记进 AUDIT；文件存在却解析不了时报 WARNING；AUDIT 写明冻结之后出现的
    复现怎么办。
  - 置信度：高（实测）。
- **m3 时间戳只在有 FIRST_READOUT 时才比较，从不和「现在」比较**
  - 位置：select_ptsz.py:250-256；readout_ptsz.py:419-434。
  - 手误写成未来时间（2027-...）的有效条目，在第一次读出里被用来计算格子（PROBLEM 731M I5）。同一次输出的表头因为已经写了
    FIRST_READOUT，显示「honoured: none」。第二次读出翻成 NOT FINAL。
  - 验证：fr_test.out 情形 1。
  - 建议：拒绝晚于当前时间的时间戳。
  - 置信度：高。
- **m4 DEAD / ALERT 的优先级：按名字最新的作业压过仍然活着的 claim 作业（修订说明里没写的行为变化）**
  - 位置：select_ptsz.py:287（`j = latest or claim`）、290。
  - claim 作业 6001 RUNNING（正在训练），后提交的重复作业 6002 在 run guard 处退出，最终 FAILED（run guard 返回后 CHECK7
    在半截 JSON 上失败）。这时状态被判成 dead，ALERT 说「resubmit once by hand」，而原作业还在训练。修订 1 只看 claim，同样
    情形下是 running。
  - 另外两处：
    - claim 被删（按 M8）后 ALERT 写「the same job as the claim」，措辞不对（Q4）；
    - sacct 返回不全、claim 作业不在输出里时，状态静默为 running，没有提示（Q5）。上一轮 m1 的修复没有覆盖这一种。
  - 决定不受影响（dead 和 running 都算等待）。
  - 验证：of_test.out Q1、Q4、Q5。
  - 建议：claim 作业还活着就判 running；对 claim 作业号无记录的情况打印 WARNING。
  - 置信度：高。
- **m5 「复现」不比较失败原因；sacct 不可用时状态全靠手写**
  - 两个作业只要都进入过训练、都是 FAILED / OUT_OF_MEMORY 就算复现，比如一次 Xid、一次 ENOSPC。I5 写的是「the same failure
    recurs」。
  - sacct 超时、退出非零或抛异常时，条目照样 honoured，只打印「states not verified」（of_test.out E、F、H：实际是 CANCELLED
    也会被 honoured）。
  - 建议：honoured 时打印两份日志各自最后一行异常，方便协调者核对是不是同一失败。
  - 置信度：高（读代码 + 实测）。
- **m6 I-0 标签现在不是空检验了，但区分力有限**
  - 模拟结果（sim_label.out；池子是四个复用 L40S 格子的 32 个格内中心化增益）：
    - 真增长（+2.79）下，判 SHOWN 后带标签：主 0.000-0.005，次 0.024-0.047；
    - 无增长下的假 SHOWN 带标签：主 0.04-0.13，次 0.07-0.23。
  - 也就是说，至少 77% 的假 SHOWN 不带标签。FINDINGS 不能把「没有标签」写成稳健。
  - 另外，标签对 NOT SHOWN 也会打印。b 接近 0 时比值会爆炸，标签几乎必然触发（合成 r0 / r1 / r7），没有信息量。AUDIT 要求
    FINDINGS 带着标签报告判词，建议明确只对 SHOWN 适用。
  - 置信度：中（模拟模型与构建审查的 sim_null 一致，规模小）。
- **m7 b < 0 时，「哪一臂偏离」的信息打印给了错误的 seed**
  - 位置：readout_ptsz.py:383 与 390。
  - 标签用的是 loso_max（偏向 0 的那一端），none / prior 信息却只给 loso_min。合成 r5（b -0.856）里，信息给的是 -505M s6，
    而标签那一端是 -53M s2。只影响打印；SHOWN (decrease) 时 m3 的诊断会指向错误的 seed。
  - 置信度：高。
- **m8 文档和其他残留**
  - select_ptsz.py:6-10 的 docstring 仍写「one run name per line」，以及「job state from sacct via the run's .claim file」；
  - select_ptsz.py:330 的注释「prints M6's warning」已经是空操作：honoured 条目在 new_run 第一行就返回，不再打印；
  - ALIVE 新加了 SIGNALING / STAGE_OUT（无害，但没写进修订说明）；
  - readout 不再只读，第一次见到 final 格就会写 FIRST_READOUT.json，docstring 没提。以后任何「只读复跑」读出（包括审查
    agent）都可能触发冻结；
  - res["verdict"] 字符串不带标签，标签在两行之后单独一行；
  - I5 仍写「decided by the coordinator, put to the user」，用户是否批准仍未记录（上一轮 m2 的延续）。

---

## 逐题回答

### 1. 判定规则和判定所用的数有没有改？——未发现改动
- AST 逐函数比较（ast_compare.out）：
  - readout：pre → amend1 → cur 三版中，verdict、slope_block、ols、ols_cov、boot_idx、interval、sgn、lnR_block、swaps、
    grad_stats、bridge、finish、reused_cell、blank、build_root、fake_reused 完全相同；EXPECTED、MARGIN、TIE、N_BOOT、F99、
    KA3 等顶层常量没有变化（变的只有 LATE_REL、RATIO 两个新常量和 SECONDARY_RATES 的文字）。
  - select：integrity、select_arm、decide、all_names、unstable_ref_lr、say、fake_curve、fake_new_run 在 amend1 → cur 之间
    完全相同；顶层断言也没变（readout_rules e91ee8a8a846）。
- 合成 root（compare3.out）：
  - 10 个完整 root 上，verdict / verdict2 / primary / secondary / h1 / h2 / s_h1 / s_h2 / criteria / criteria2 / ten /
    ten_b200 / loso / arms / swaps / nob11 / lnR50 / bridge294 / investigate / grad 在 pre、amend1、cur 三版之间逐位相同。
    覆盖了 SHOWN (increase)、NOT SHOWN (a)(b)(c) 各种组合、b < 0；robust 的 mean / median / trimmed / loso / late 在 amend1
    和 cur 之间相同。
  - own root（列入一个已完成、check 通过的 run，旧格式）：
    - pre：打 WARNING，照常判 SHOWN；
    - amend1：731M 判 NOT DETERMINABLE；
    - cur：打出恢复的 WARNING，忽略该条目，判 SHOWN，判词与 pre 相同。
    pre 和 cur 只差 731M 均值 1e-14 量级的浮点误差：pre 是通过它的 own-failure 计分路径算的那个 run。
- select（sel3_compare.out）：6 个 root、48 个 arm 的决定、lr、均值、bpb、打印行，三版两两相同。
- AUDIT 文字：amend1 → 当前只有删除线、注释和新的「Amendment 2」节，预注册判据的原文没动。
- 新增打印（标签、比值、late_rel、seed 信息、OWN_FAILURES / FIRST_READOUT 表头）不进入任何判据（见第 4 题）。

### 2. 逐函数比较当前版和 amend1
- select：
  - 删掉的只有 dead_job（被 jobs_of 取代，有意为之），以及 FAKE_STATES、_alerted（自测用）；
  - 新增 _sacct、jobs_of、warn_once、own_entries、first_readout、own_failures；
  - 改动的是 load_states、new_run、parse_own（现在是 own_failures 的别名）、run（调用顺序）、selftest；
  - **integrity() 与 amend1、pre_amend 都 AST 相同，diff 里也没有它的 hunk**，恢复得一字不差。
- readout：新增 few_seeds；改动 ka、load_cells（调用顺序）、new_cell（只把 'dead' 归入 NOT FINAL，m7）、out_robust、
  readout（FIRST_READOUT、表头）、robust、selftest。
- 两个文件都能完整解析，没有函数被截断。
- 修订说明之外的行为变化：m4（dead 的优先级）、ALIVE 加了两个状态、select_arm 里那段 M6 打印成了空操作（m8）。
- 作者的测试我从拷贝重跑了一遍（rerun_*.out），去掉路径和时间戳后与 runs/ptsz/tests 下的输出逐行相同：SELECT SELFTEST PASS
  39/39、READOUT SELFTEST PASS 39/39、KA PASS 27/27。

### 3. own_failures() 和 FIRST_READOUT：能否被绕过、会不会误杀
**关键事实**：(a) 规定 honoured 的条目所指的 run 必须「没有完成且 check 通过」。而这样的 run 即使不列入，也会让格子 NOT FINAL
或 withheld，判词照样是 NOT DETERMINABLE；run 一旦完成并通过，条目自动失效（of_test.out、compare3 own root）。所以列入与否
**不会改变任何判词**，差别只有三处：
- 标签：「ND（I5）」还是「NOT FINAL」；
- select 是否停掉另一臂的 X / C（进度和成本）；
- 是否不再重投。
下面这些漏洞因此都只影响标签和流程。

- 格式与校验（of_test.out，实测）：
  - 不同作业号：同号拒绝 ✓。作业号顺序颠倒（L）可以；同一行写两遍（K）也可以，无害。
  - FAILED / OUT_OF_MEMORY：TIMEOUT 拒绝 ✓；sacct 状态与条目不符拒绝 ✓（G，CANCELLED）。
  - 与 sacct 一致：sacct 有输出但没有这两个作业号时只 WARNING，照样 honoured（D）。sacct 退出非零、超时或抛异常时也照样
    honoured，状态全靠手写（E、F、H），见 m5。
  - 日志里有训练调用：TRAIN_MARK `code a3bce3ab63c1; eval every` 与 trainer 第 468 行一致，只在 claim 之后、--dry_inputs
    之外打印。真实日志 llmadd_p3x-93m-lr2.4e-3-none-s5_21742352.out:16 就有这一行。GUARD_MARKS 的四个字符串与作业体
    （HOLD present / code_sha mismatch / not a L40S）和 run_guard.py 退出时打印的文字逐字一致；run_guard 正常运行时不打印
    「run guard:」，logs/ 里 145 个含「run guard:」的日志没有一个同时含训练行。作业名、日志路径（logs/%x_%j.out，Path(ROOT).parent/logs）
    和 --out 在 144 个渲染（A + 合成 X、C）里都与 own_failures / jobs_of 的假设一致（names_check.out）。所以真 OOM 两次的条目
    能被 honoured（A），M8 那种「重投在 run guard 处退出」会被拒绝（作者自测 + 上一轮 I-B）。
  - 早于 FIRST_READOUT：已有冻结文件时，冻结后新加的条目拒绝 ✓（M）；删掉的条目打 WARNING，不再 honoured ✓（N）；
    多一个词或用空格代替 T 都拒绝，有 WARNING ✓（O、P）；「Z」时区可以 ✓（I）。
- 能否被绕过：
  - 已完成且通过的 run 不可能被列入生效（这正是上一轮 I-A 的修复点）；
  - 写错尺寸的条目，因为必须有那个 run 名下两份真实训练日志，基本不可能生效（上一轮 I-A 的另一个误杀途径已被堵住）；
  - 剩下的绕过：CHECK7 失败或没有 check 的 run（m1）、不同原因的两次失败（m5）、sacct 不可用时手写状态（m5），以及冻结
    文件被删或写坏（m2）。
- 会不会误杀：正常完成的 run 不会。一个还在第三次重投中的 run，若已有两次真失败的有效条目，在它跑完之前该尺寸是 ND（I5），
  跑完通过后自动恢复；判词不受影响。
- sacct 慢、失败或返回不全：
  - 每次 select / readout 只调一次 sacct（readout 里 select_arm / new_run 都复用 STATES），timeout 180 s；
  - 超时、抛异常、退出非零都会打 WARNING，不再静默（上一轮 m1 修好了）；
  - 返回不全时，own_failures 逐个作业打 WARNING，但 DEAD 判定在 claim 作业缺失时静默为 running（m4 的 Q5）；
  - SACCT_START 固定为 11-05，要查的作业会随时间越来越多（现在 674 个、12 s），离 180 s 还远；
  - 这几种情况都不改变决定。

### 4. 「the slope rests on a few seeds」的比值：算对了，只打印
- 独立复算（ratio_indep.out，不 import 项目代码）：
  - 四个复用 L40S 格子：均值斜率 +2.1608，中位数 +1.0059，截尾 +1.1673，留一 seed 最小 +1.2126（去掉 505M s5）。比值
    0.4655 / 0.5402 / 0.5612，带标签，与 readout --ka 和 AUDIT 一致。被留一去掉的 seed：none 34.15（格内中位数 60.94），
    prior 89.91（中位数 89.74），偏离的是 none 臂，与读出打印一致；
  - F99 六格：0.6540 / 0.8759 / 0.8847，不带标签 ✓；
  - late：绝对阈值 505M 1、1.07B 2；相对阈值 505M 1、1.07B 0 ✓。
- 方向：b > 0 用最小值，b < 0 用最大值，b = 0 不打标签（readout_ptsz.py:376-385），与 AUDIT 一致。
- 只打印：`label` / `ratios` 只在 robust()、few_seeds()、out_robust() 和测试里被引用，verdict() 拿不到它们
  （readout_ptsz.py:376-403）；compare3 显示判定字段在三版之间相同。
- 区分力和打印问题见 m6、m7。

### 5. 其他副作用
- 冻结文件：
  - runs/pttpp/FROZEN.txt 26/26 sha 相符（我自己核对过）；
  - make_jobs_pttpp b4bfa6eb248d、make_jobs_pt300x e8fd563c335a、pt300x decisions 22532f929c7a、pttpp_refs d10d720c1242、
    ptsz_refs 831d6694cd6f、code7 pretrain_range a3bce3ab63c1 都没变；
  - regress_frozen.out（02:01）PASS。
- 作业文件和脚本：
  - 128 个 stage-A 文件与当前生成器的渲染逐字节相同，全部 `--nice=100`；合成的 X / C 也是 100（render_check.out）；
  - make_jobs_ptsz.py 63002a20b835、submit_ptsz.sh 38c1f51658fb、smoke_report.py 098a807f3764 与修订 1 后相同；
  - smoke/SUBMITTED_smoke731 多了一行（m8 的文档修复），没有任何脚本解析它的内容（生成器只看它存不存在）；
  - 冒烟作业文件和当前渲染只差 nice 一行（与修订 1 的说明一致）。
- 其他批次：01:41 以后，llmadd_torch 里 ptsz 之外被改的只有 runs/ptdom（它自己的作业输出、AUDIT 和读出）和 watch/ 下的状态
  文件。ptsz 的代码不引用 ptdom。**未发现**本修订造成的副作用。
- 训练：
  - 真实 root 里没有 runs/ptsz/out、decisions.json、OWN_FAILURES.txt、FIRST_READOUT.json，也没有 SUBMITTED_psz-* 标记；
  - logs/ 里没有 llmadd_psz-* 日志；
  - squeue 里本批次（psz）的作业只有冒烟 22011938（PENDING，priority 12474）。
  确实是「任何结果之前」。
- 修订 2 新增的副作用（有意为之）：readout_ptsz.py 现在会写 runs/ptsz/FIRST_READOUT.json（m2、m8）。

---

这次修订最可能出错的地方：判词本身不会错。最有把握会出事的是 pt300x 惯例的 `readout_ptsz.py --json` 一步：第一次有完整结果
时它一定崩溃（R1）。其次是 FIRST_READOUT 在第一个 final 格就冻结，之后真复现无法按 I5 列入（m2）；这只影响标签和流程。
