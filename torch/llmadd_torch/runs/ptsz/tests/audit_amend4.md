# ptsz 修订 4 审查（对抗式，全新子 agent，只读）— 2026-11-06

审查对象：runs/ptsz/AUDIT.md「Amendment 4」（AUDIT.md:590-639）及其落地：make_jobs_ptsz.py abee61db9ef3（对照
scale_work/ptsz/make_jobs_ptsz.py.amend3）、128 个 runs/ptsz/job_psz-*.sbatch（对照 scale_work/ptsz/jobs_all_amend3/）、
runs/ptsz/smoke/job_smoke731xs.sbatch 80764f3ea66e（已排队 22129757）、tests/smoke_report_xs.py 5e1f993f90f1、
tests/smoke_report_xs_test.py、submit_ptsz.sh 38c1f51658fb；第一次短测试 22011938 的真实日志、smoke_report_22011938.json、
SMOKE_VERDICT.json、gpu_22011938.csv；冻结 trainer code7/tools/pretrain_range.py a3bce3ab63c1、run_guard.py。

我的检查全部在 /scratch/yh6210/tmp_audit_ptsz_a4/ 下（CPU，当前节点 cs609，无 GPU）。项目里只写了本文件；没有提交、取消
任何作业，没写 HOLD；没调用生成器的 main()（它会 chmod code7），只在进程内 import 它；项目 __pycache__ 未被改写
（make_jobs_ptsz 的 pyc 仍是 19:32:51 的）；SMOKE_VERDICT.json 未变（md5 bab554ad…，仍是 22011938 的 FAIL）。
- envparse.py：在 torch 2.10 里往环境变量里埋非法值，看通用 allocator 配置是否解析它（Q3）。
- seg_compare.py：XS 三段 vs 第一次短测试的磁盘文件和 22011938 日志里 SMOKE 行记录的实际参数；XS 文件 vs 生成器当前渲染；
  XA/XC vs 真实 stage-A 作业的 ARGS。
- render_check.py：进程内渲染全部 128 个 stage-A 作业并逐字节比对；731m 的 X / C 两臂渲染；TRAIN_S / est_h / tlim / ckpt。
- xs_adversarial.py → xs_cases/：用 22011938 真实日志造 11 个对抗用例跑 smoke_report_xs.py（只写本目录）。
- curves.py：22011938 每段的 eval 曲线（判断 GPU 上是否逐位可复现）。
- queued_22129757.sh：`scontrol write batch_script` 取出的排队脚本（只读），与磁盘文件 cmp。

## 结论
**阻断：未发现。** 环境变量确实只加在 731M 的 32 个作业（两臂各 16）上，放在两个 trainer 调用之前，requeue 重跑脚本时会重新
export；其余 96 个文件逐字节不变；torch 2.10 确实读 PYTORCH_CUDA_ALLOC_CONF（而且优先于 PYTORCH_ALLOC_CONF），驱动 API 版
本已编译进来。XS 三段命令与第一次 A/B/C 实际跑的参数逐个 flag 相同（只差 --out，XC 的 --stop_after 500→260），XB 确实续跑
XA 的 step-199 ckpt。TRAIN_S / 时长上限计算正确，只影响 731M 的 --time。
**重要 1 条**：判据 (5) 是常量（读的是旧 JSON），XS 自己的数值完全没被检查，「只改 allocator、数值不变」在 GPU 上没有验证，
而 731M 是唯一带这个设置、又是两个斜率里杠杆最大的点；有一个免费且锐利的已知答案（step-0 lm 逐位相等）没用上。
**次要 7 条**：(4) 没有幅度、(1) 只看 shell echo；段内 run guard 提前返回会「通过」；真实 731M 作业事后无人核对 ALLOC 行；
提交闸门没绑定修订 4；文档过时；遗留测试 ckpt 占盘。

## 问题（按严重程度）

### 重要

**R1. 判据 (5) 是常量，XS 本身的数值（step-0 lm、续跑一致性）没有任何检查；「numerics unchanged」未在 GPU 上验证。**
- 位置：tests/smoke_report_xs.py:39-40、68（`f_crit` 来自 smoke_report_22011938.json，五项都已是 True，这一项此后永远不会因
  XS 失败）；AUDIT.md:619「(5) the first smoke's non-memory criteria stay true (… step-0 lm within 1e-3)」；AUDIT.md:600-602
  「numerics unchanged」。
- 为什么会测错：(5) 的名字读起来像「在 XS 上重验非显存判据」，实现是把上一个作业的结论抄过来（同名不同义）。XS 日志里的 eval
  曲线、smoke/XA、smoke/XC 的 run JSON 都没被读。对抗用例 `step0_lm_off_by_0.5`（XS 日志里所有 step-0 lm +0.5）→ **PASS**。
  这个设置只加在 731M 上，731M 在 PRIMARY 和 SECONDARY 里都是 ln N 最大（20.239）、杠杆最高的点；若 allocator 改变了数值，
  效应会和尺寸混在一起，正好落在斜率最敏感的位置。先验风险低（两种模式下块地址都 ≥512 B 对齐，cuBLAS 选的算法不该变），
  真实作业的 CHECK7 也会按 1e-3 查每个 run 的 step-0 lm；但 1e-3 容差测不出「有变化但很小」，逐位比较才是锐利的检测。
- 具体检查（XS 跑完、提交 128 个之前，coordinator 手工读即可，不必改判据）：
  1. XA 和 XC 的 step-0 `lm` 必须逐位等于 **11.74520206451416**——22011938 的 A、C、D 三段（同一 seed、不同命令）都给出这个值，
     这是训练前的纯前向，GPU 上可复现（curves.py 输出）；XC 的 step-0 lm 与 CPU 参考 11.7453 差 ≤ 2e-4。
  2. XB 的 step-200 `lm` 等于 XA 的 step-200 `lm`（第一次：A 与 B 都是 6.374513626098633，逐位相同）。
  3. step 100 / 200 / 300 / 250 只能宽松比较：训练本身在 GPU 上不逐位可复现（C 与 D 在 step 0 更新后的 val_loss 已差 1e-5，
     evaluate 用自己的 rng(7)，所以这是 GPU 非确定性，到 step 250 lm 差 0.019）。XA 的 7.6550 / 6.3745、XB 的 6.0608、XC 的
     6.1159 应在 ~0.03 以内；更大的差说明有东西变了。
- 置信度：判据为常量——读代码确认 + 对抗用例确认；「设置改变数值」本身——推测（低概率）。

### 次要

**m1. 「设置生效」的证据弱：(1) 只证明 shell 变量被 echo，(4) 只要求严格小于、没有幅度。**
- 位置：tests/smoke_report_xs.py:63（日志里的 ALLOC 行来自 bash 的 echo，不是 torch）、:67（`gap[k][0] < gap[k][1]`）。
- 为什么会测错：设置若无效，(4) 只能靠「同命令下 reserved 峰值完全确定」才会失败；1 字节的偶然下降就够通过。对抗用例
  `ineffective_noise_lower`（XA gap 少 1 字节，XB reserved 少 1.3 GB 而 allocated 不变）→ **PASS**。实际风险低：22011938 里
  C、D 两条不同命令的 max_reserved 完全相同（42,603,642,880），说明峰值近乎确定；而且设置无效时 XB 还得偶然少 1.28 GB 才能过
  (3)；加上 Q3 的 CPU 证据，联合误过概率很小。方向（XS 的 gap < 第一次的 gap）和单位（字节，GIB = 2**30）都对。
- 具体检查：读 XS 报告时看幅度——预期 XA / XB 的 gap ≪ 1 GiB（第一次 6.02 / 6.08 GiB）；第二条路径 nvidia-smi 最大 memory.used
  应明显低于第一次的 45,193 MiB（预期约 max_alloc + ~0.6 GB 上下文，≈ 39-40 GB）。若 gap 只小了几 MB 或 nvidia-smi 仍 ≥ 45 GB，
  就当设置没生效处理，不论 (4) 判了什么。（以后的短测试可在 smoke_seg 里记录 `torch.cuda.memory._snapshot()` 各段的
  `is_expandable`，那是直接证据；本次 smoke_seg.py 冻结、作业已排队，不要改。）
- 置信度：判据偏弱——读代码确认 + 对抗用例；设置真实无效——推测（证据指向有效）。

**m2. 段内「什么也没做」也算通过：run guard 提前返回时 smoke_seg 打印 ok=True、显存≈0；续跑只查有无 'resumed from'，不查步数。**
- 位置：smoke_seg.py:27-29（`pr.main()` 正常返回即 ok）、code7/tools/pretrain_range.py:163-166（`run_guard.claim` 为 False 时
  `return`）、tests/smoke_report_xs.py:64-67。
- 对抗用例：`xa_did_nothing`、`xc_did_nothing`、`xb_resumed_wrong_step`（"at step 300"）→ 都 **PASS**。XA→XB 这条链受 XB 的续跑
  检查保护（XA 没跑就没有 ckpt，XB 不会打印 resumed），XC 不受保护。
- 现实概率很低：smoke/XA、smoke/XC 现在不存在；只排了一个 XS 作业；Requeue=0（scontrol 确认，JobRequeue=0），不会因 requeue
  重跑而让 XA 从 step-299 ckpt 续跑（那种情况下 XA 会一直跑到 1 h 上限，结果是 FAIL，不是误过）。
- 具体检查：读报告时确认日志里没有 "run guard:"；每段都有 `stop_after 210 / 310 / 260: exiting (test)`；XB 的那一行是
  `at step 200 of 18310`。
- 置信度：读代码确认。

**m3. 真实 731M 作业事后没人核对设置是否真的在场。**
- 位置：make_jobs_ptsz.py:352-354（只在渲染时加）；run JSON 不记录；CHECK7、完整性复查、select_ptsz.py、readout_ptsz.py 都不读
  ALLOC 行。I5 里「两臂一致靠 coordinator 手工保证」现在由生成器按尺寸保证，但只在渲染 / `--check` / 提交时成立。
- 现状是对的：render_check.py 确认 32 个 731M stage-A 文件都带 export、另外 96 个都不带；731M 的 X / C 两臂渲染也都带 export、
  --time 23:30（它们走同一个 job_x）。风险只在「有人手工改 / 用旧生成器重渲染某个 731M 文件后提交」。
- 具体检查：结果审查时 grep 每个计入的 731M run 的每个日志段（含 requeue 追加段），要求 `ALLOC: PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True` 出现在该段 `code a3bce3ab63c1; eval every` 之前；其他尺寸的新 run 日志里不出现 ALLOC 行。
- 置信度：读代码确认。

**m4. 提交闸门没绑定修订 4。**
- 位置：submit_ptsz.sh:22 只要求 `pass is True`；不查 `amendment == 4`、job 22129757，也不查 731M 文件里有 export。:19 的注释
  仍写 smoke_report.py。两个报告脚本写同一个 SMOKE_VERDICT.json：之后谁再跑 `smoke_report.py 22011938` 会把它改回 FAIL
  （偏保守，不会误放行）。
- 具体检查：提交前 `python -c` 读 SMOKE_VERDICT.json，确认 job = 22129757、amendment = 4、pass = true；`grep -c "export
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" runs/ptsz/job_psz-731m-*.sbatch` 32 个都是 1。
- 置信度：读代码确认。

**m5. AUDIT 文档过时（不影响测量）。**
- Deviations 清单（AUDIT.md:79-95）最后写「Nothing else」，没列 731M 的 allocator 设置；deviation 5（:88）和 render 行（:337）
  仍写 731M 22:30；Cost 表（:304-305）仍是 8.89 h / 284.4 / 805.4。按生成器的 est_h 9.332 h：731M stage A 298.6、C 74.7、A+C
  373.3、加最坏 X 448.0；总计 A 658.5 + C 164.7 = 823.2 L40S-h（最坏 ~988）。
- 置信度：读代码确认。

**m6. 遗留测试 ckpt 占盘。**
- smoke/A 里还有 22011938 的 B 留下的 step-299 ckpt（8.2 GB；stop_after 在删除 ckpt 之前 return，pretrain_range.py:590-594）；
  XS 会在 smoke/XA 再留 ~8.8 GB。/scratch 在 94% 左右，M7 已登记的磁盘风险会因此多 17 GB。
- 具体检查：读完 XS 报告后删这两个 ckpt（由 coordinator 决定）。
- 置信度：读代码 + du 确认。

**m7.（备注，不是缺陷）XS 作业现在 PENDING，原因 QOSMaxGRESPerUser**（用户 gpu48 的 16 卡上限被 ptdom / ptshare 占满），要等
别的作业释放卡位才会开始。

## 逐题回答

### 1. 环境变量的作用范围；其他文件；两组条件
- 作用到真实训练：job_x 渲染出的 731M 文件第 24 行 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # …`、第 25 行 echo，
  两个 trainer 调用在第 36、37 行（`--dry_inputs` 和训练），CHECK7 在其后。`bash -n` 32 个 731M 文件 + XS 文件全部通过；把这两行
  抽出来在 `set -eu` 下 source，子进程 python 读到 `expandable_segments:True`（尾随注释不影响）。requeue：作业带 --requeue、
  --open-mode=append，Slurm 从头重跑脚本，export 会重新执行；trainer 里没有任何地方改写这个变量或调用 _set_allocator_settings /
  empty_cache（pretrain_range.py、pretrain_helix.py、numlib.py、digit_split_data.py、run_guard.py 都 grep 过）。
- 只作用到 731M：128 个文件和 jobs_all_amend3/ 逐个 cmp，96 个（136M / 294M / 391M）逐字节相同；32 个 731M 文件的 diff 只有
  `--time 22:30:00 → 23:30:00` 和 export / echo 两行（每个文件同样 3 处）。进程内重渲染 128 个 = 磁盘文件，0 个不同。
- 两组一致性：731M 两臂各 16 个都带 export（none 16、win-ctx 16）；X / C 两臂渲染也都带（render_check.py）；--time 两臂相同。
  提交环境（本节点 claude_host 的 env）里没有 PYTORCH_*ALLOC_CONF，sbatch 默认 --export=ALL，其他尺寸不会意外继承；就算继承了
  PYTORCH_ALLOC_CONF，731M 的 PYTORCH_CUDA_ALLOC_CONF 也优先（见 Q3）。**未发现**同尺寸两组条件不一致。

### 2. XS 是否测到了要测的东西
- 命令：seg_compare.py 解析两份磁盘文件，XA/XB/XC 与 A/B/C 的 python、smoke_seg.py、flag 顺序都相同，不同的 flag 只有 --out
  （A→XA、C→XC；XB 用 XA 目录）和 XC 的 --stop_after 500→260；第一次文件里 A/B/C 的参数又与 22011938 日志 SMOKE 行记录的实际
  参数逐个相同。生成器 smoke_xs_job() 的输出 = 磁盘 XS 文件 = Slurm 里排队的脚本（scontrol write batch_script 后 cmp 相同，
  80764f3ea66e；Nice=100，TimeLimit 01:00:00，Requeue=0）。
- XB 续跑 XA：同一 --out、同一 --tag → 同一个 ckpt 名；XA 在 s=99、199 写 ckpt，s=210 stop_after 直接 return（不删 ckpt），
  XB 从 step 200 开始，s=200 和 s=300 都是窗内 eval（--window 0,0.4），s=200 就是续跑后的第一个 grad_info eval——正是 22011938
  里没过余量的那一段；结构上也等于真实 731M 续跑（ckpt 在 1999 / 3999 / 5999，续跑后第一个 step 2000 / 4000 / 6000 正是窗内
  eval，st["model"] 的 2.9 GB 一直被引用）。XS 不像第一次那样以 A 成功为条件跑 B，但 XA 失败时 (2) 直接 FAIL，无影响。
- 判读五条：SMOKE 行缺失 → (2)(3)(4) 都失败（对抗用例 `xb_no_smoke_line` FAIL，正确）；单位一致（字节）；gap 方向正确；续跑
  只在 XB 段内找（`resume_only_in_XA` FAIL，正确）；`alloc_unset` FAIL，正确；旧 JSON 某项为 False 时 (5) 失败。会误过的情形
  见 R1（数值）、m1（幅度）、m2（空跑 / 续跑步数）。
- known-answer 测试：自己复跑（TMPDIR 指向审查目录）4/4 PASS。它咬得住登记的四种失败（无 ALLOC、无续跑、原始数字 → (3)(4)
  失败；方向写反会被 'good' 抓到，(3) 只查 XA 会被 'same' 抓到），但没有「设置无效 + 噪声」、「数值变了」、「段空跑」这几类用例，
  所以上面三个弱点它测不出来。

### 3. torch 2.10 是否仍识别 PYTORCH_CUDA_ALLOC_CONF；(4) 能否偶然满足
- **识别，且优先**（CPU 证据，envparse.py）：
  - 只设 `PYTORCH_CUDA_ALLOC_CONF=bogus_key_xyz:1` → `_accelerator_setAllocatorSettings` 触发通用配置初始化时抛
    `Unrecognized key 'bogus_key_xyz' in Accelerator allocator config`；设 `expandable_segments:Bogus` → `Expected 'True' or
    'False' … got 'Bogus'`（expandable_segments 自己的解析函数）；设 `expandable_segments:True` → 正常。说明通用
    AcceleratorAllocatorConfig 读的就是这个变量，并解析了 expandable_segments 这个键。
  - 两个都设：CUDA=True、通用=bogus → 不报错；CUDA=bogus、通用=True → 报错。所以 2.10 里 **PYTORCH_CUDA_ALLOC_CONF 优先**，
    PYTORCH_ALLOC_CONF 只在前者不存在时才读（libc10.so 里的字符串顺序 CUDA、HIP、ALLOC；CUDAAllocatorConfig.h:121-139 同样先读
    CUDA 变量）。头文件 AllocatorConfig.h:160-162 的注释说 CUDA 变量「lower priority」，与实际行为相反，但对本修订无影响。
  - 值必须是 `True`：小写 `true` 会报错（大声失败，不是静默失效）；作业里写的是 `True`。
  - 平台支持：`#ifndef PYTORCH_C10_DRIVER_API_SUPPORTED` 分支里的警告串「expandable_segments not supported on this platform」
    在 libc10.so / libc10_cuda.so / libtorch_cuda.so / libtorch_python.so / libtorch_cpu.so 里都不存在，cuMemCreate / cuMemMap /
    cuMemAddressReserve 符号在 libc10_cuda.so 里，说明驱动 API 路径已编译。没有找到弃用警告串。
  - 没有 GPU，无法在本节点直接看到「段是 expandable」；GPU 上的直接证据只能来自 XS 的 gap 幅度和 nvidia-smi 峰值（m1）。
- (4) 能否偶然满足：能，见 m1（严格小于、无幅度）；但同命令的 reserved 峰值近乎确定，且还需 (3) 同时偶然通过，联合误过概率很小。

### 4. TRAIN_S / 时长上限 / 闸门
- TRAIN_S['731m'] 31581 → 33177 = smoke_report_22011938.json est.win_s 33176.6 向上取整（t 1.7032 s/step、e 23.20 s/eval，由 C
  的 sec(500)-sec(250)=449.1 和 D 的 sec(250)-sec(0)=657.8 解出，我按日志重算一致），乘 F87 505M 的 win/none 比 1.0076。
  est_h = (33177 + 180 + 9 × 8.78 GB × 3 s/GB) / 3600 = 9.332 h → ceil(2.5 × 9.332 × 2) / 2 = 23.5 h → 23:30:00 < 48 h。生成器的
  ckpt 写入按 26 s/次估，比实测 11.4 s 保守。设置下的速度没测（XS 只打印），2.5 倍余量足够。
- 别的尺寸：只改了 731m 这一项；391m 的 TRAIN_S 是常数（不从 731m 推）；136m 05:00、294m 10:30、391m 13:30 不变（render_check）。
- ckpt_every：731m 仍 2000（smoke_job 里还 assert 了 `ckpt_every_of("731m") == 2000`）；n_ckpt 9。CHECK7 与 intent JSON 不含时长，
  diff 证实 ARGS / intent / CHECK7 行不变。
- 闸门：submit_ptsz.sh 读的就是 smoke/SMOKE_VERDICT.json，smoke_report_xs.py:73 写的也是它；现在仍是 22011938 的 FAIL，所以
  提交会被拒。弱点见 m4。

### 5. 没写出来的假设；与已登记规则的关系
- 「只改 allocator、数值不变」：见 R1。
- 「XS 一次、一个 seed、一个 lr、一个节点就够」：显存与 lr / seed 基本无关，同命令峰值近乎确定，可以接受；不同节点驱动版本可能
  不同，但 VMM API 在 L40S 的驱动上都有（推测，风险可忽略）。
- 「只有 731M 有显存风险」：294M / 391M / 505M 在 L40S 上续跑后的窗内 eval 从来没被测过（logs 里 F87 的 33 个 pt2s505 日志、
  pttpp 的 294M L40S 日志都没有 'resumed from'）。按 22011938 的数粗估（全 logits 路径 ≈ 固定 20+ GB；731M 与 391M 参数相关部分
  在续跑 grad_info 时差约 8 × 1.37 GB），391M 续跑窗内 eval 的 reserved 约 36 GB，余量 ~9 GB——检查过，不构成问题（推测）。
- 与 I5：I5 写明 fallback 由用户决定、按尺寸作用于两臂所有阶段、先写生成器修订——修订 4 完全照做（用户原话已记录，按尺寸、
  两臂、A/X/C 都走 job_x）。I5 说「run JSON 不记录、靠 coordinator 手工保证两臂」——现在由生成器保证，但事后核对缺位（m3）。
- 与 N1：XS 文件 nice 100（AUDIT 说明第一次文件留着 1000、实际以 --nice=100 提交；XS 从当前渲染派生，已是 100）。不冲突。
- 与 Smoke 节 / I4：Smoke 节原本把 expandable_segments 列为「A 失败时」的 fallback、要用户决定——用户已决定（选项 A）。I4 的余量
  判据原用于 A-D，XS 用于 XA/XB/XC（无 D），合理；在 expandable 模式下 reserved 更接近真实占用，余量判据反而更贴切。
  Smoke result 节要求在提交 731M 前更新 TRAIN_S——已做。未发现冲突。

### 6. 陷阱清单
- 训练 / 评测分布不一致：未发现（设置不碰数据、eval、ARGS）。
- 记忆代替泛化：不适用 / 未发现。
- 静默失效的设置 / 梯度：设置本身被 torch 读取（Q3），非法值会大声报错；但 XS 判读里证明「生效」的两条偏弱（m1），真实作业事后
  无人核对（m3）。
- 单 seed 偶然：XS 单次单 seed；显存峰值近乎确定，未发现实质问题；数值比较要考虑 GPU 训练本身不逐位可复现（R1 的第 3 点）。
- 读错位置：(5) 读的是上一个作业的 JSON 而不是本次运行（R1）；(4) 读的 A / B 数值与 22011938 日志的 SMOKE 行一致；续跑只在 XB
  段内查，正确。
- 同名不同义：(5)「第一次的非显存判据仍成立」实为常量（R1）；(1)「setting on」实为 shell echo（m1）；XS verdict 里的 `est` 是
  第一次的测速，不是 XS 的（读者可能误解，次要）。
- 静默过滤样本：未发现。
- 修 bug 的副作用：job_x 的改动对非 731M 渲染为空串，96 个文件逐字节不变；f-string 里 `${{…:-unset}}` 渲染正确；`bash -n` 全过；
  make_jobs_ptsz 只被 ptsz 自己的脚本 import（select_ptsz 自检只渲染 294m / 136m；args_diff / check7_ka / dry_digest 用的 ARGS /
  CHECK7 未变），其他批次不受影响。文档过时见 m5。

## 最可能测错的地方
XS 的 PASS 实际主要由 (3) 余量撑着：「设置生效」只有 shell echo 加一个无幅度的严格小于，「数值不变」完全没测（(5) 是抄来的常量）——
提交 128 个作业前应手工核对 XA / XC 的 step-0 lm 逐位等于 11.74520206451416、gap 降到 ≪ 1 GiB、nvidia-smi 峰值明显低于 45,193 MiB。
