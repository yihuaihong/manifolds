# pt6v2 V2 build: diff review (exp-auditor, 2026-11-19)

Scope: the V2 diff only. Files: make_jobs_pt6v2.py bb15d248afd8, readout_pt6v2.py dbd3654b6f28, submit_pt6v2.sh a1a06ded3a9d,
tests/run_refs_pt6v2.sh 355f0067bf62, auto_pt6v2.sh 95b1135cc607, tests/auto_read_pick.py 92726049c7f2. These shas were re-hashed now
and match the AUDIT "V2 build" table.

Read-only. My scripts and outputs are in /scratch/yh6210/tmp_audit/pt6v2_diffV2/. One side effect came from the permitted dry run:
runs/pt6v2/auto_20261119T112908Z.log.

Materials missing: the builder's dry-run log auto_20261119T110042Z.log has been deleted. I re-ran the dry chain with a stub instead
(stub_pick_extra.json → auto_dry_extra.out). Every step ran, 113 job texts parse, and step 5 refused "no COMPOSITION.json" as expected.

## Blockers
None found.

## Important
I1. `--go --pick_json` is not refused, and a non-final PICK is accepted.
- Where: auto_pt6v2.sh:26, 80-81; tests/auto_read_pick.py:18-22.
- Problem: auto_read_pick.py never checks `complete` or `mode`. I fed it the real partial file
  tests/pick_partial_E3src_20261119T1110.json (complete False, allow_partial). It returned COMP=months,dim,cities,countries,people
  and STATE=new.
- Consequence: with --go, the chain would write that fallback composition into COMPOSITION.json (write-once, 444) and submit it.
- Fix: refuse --go when --pick_json is given, or require `j["complete"] is True and j["mode"] == "strict"` under --go.
- Confidence: confirmed by reading the code and by a run.

I2. none@6.4e-2 is a core cell in verdict_v2, so one failed run drops its seed from all six verdicts.
- Where: readout_pt6v2.py:515-519.
- Problem: an invalid none@6.4e-2 run (crash, unfinished, or a disallowed check failure) removes that seed from all six verdicts,
  and strict mode refuses.
- Evidence: no 6.4e-2 run exists anywhere (runs/*/out/*6.4e-2* is empty). pt6comb none@3.2e-2 (n=4) trained with finite values;
  final val_loss was 4.385, against 4.068 at 2e-3.
- What does not trigger it: CHECK_PY_6 says "a diverged run is data, not a failed check", and the trainer has no NaN break. A
  diverged run therefore stays valid with a low area and cannot win N_m.
- What could trigger it: a crash or a check failure. The read-out of a run after it diverges (dom_metrics on NaN evals) is not
  covered by any pt6v2 test. This part is inferred.
- Check (pick one): run none@6.4e-2-s16 alongside the canary and read it with readout_pt6v2; or let the read-out drop only the
  invalid none@6.4e-2 run from that seed's N_m candidates instead of dropping the whole seed.

I3. A partial submission is not treated as a failure.
- Where: make_jobs_pt6v2.py:686-690; submit_pt6v2.sh:74; auto_pt6v2.sh:129-130.
- Problem: `--submit` keeps going after an sbatch rc≠0 and exits 0, so auto logs "done: submitted" on a partial batch.
- Mitigations: a rerun is safe (SUBMITTED flags), and the strict read-out refuses later on missing cells.
- Fix: count the failures and exit 1. The code predates V2, but it is now on the unattended chain.
- Confidence: confirmed by reading the code.

## Minor (one line each)
- run_refs_pt6v2.sh:32 writes refs_<m>.done even on FAIL. After a failed comp refs, a rerun polls the dead job id (auto:41-42) and
  stops each time. The message does not say to delete both tests/auto_refs_comp.jobid and tests/refs_comp.done. The failure is
  loud, not silent.
- auto:108-109 submits the refs job even when REFS=1. If refs_comp.done exists, step 3 stops with "no-op". Loud.
- refs_pt6v2.py table() (180-211) does not enforce the expected file list. If a dry run of all6, only-numbers or all6@1e-3 crashes in
  parse_dry before writing its JSON, the run is just absent and TABLE still PASSes. Only the none seeds and the canary are counted.
  Predates V2.
- readout_pt6v2.py:139-144: caesar gets the "real letter tokens" note but no tied-row caveat. Its items ' A'..' Z' are single tokens
  and the read-out is an argmax over the tied rows (ptdom3 AUDIT:92). Wording only, but the omission points toward a false positive.
- render_findings_pt6v2.py:281-282 hardcodes 「比较类与 s4 / dim 是 Tier L」. This is stale under V2's reading_line: toy3 domains are
  named, months is not Tier L, and s4 is out. Fix: quote the reading line from the read-out txt.
- none_grid (readout:93-100) does not cross-check CELLS.json's domains_arg against the composition. If CELLS.json is deleted:
  - the read-out falls back to the 5-lr grid, and the none@6.4e-2 dirs become problems, so strict mode refuses;
  - under allow_partial, N_m would be chosen over 5 lrs with only a printed problem.
  This needs a manual edit of a 444 file.
- HANDOFF_render.md:78's worry no longer applies: sample_json_pt6v2.py runs at the current shas (output in tmp_audit/.../sample_out,
  rc 0).
- The 90-minute sacct poll (auto:51-59) must not run inside a ≤10-minute foreground tool call. Resuming from the jobid works (AUDIT
  test).

## The delegation's six questions
Q1. Without --extra_none, renders and verdicts are unchanged, and the evidence is real.
- v2_body_diff: 105/105 byte-identical between the preV2 module and the new one (placeholder composition, real refs).
  - That test reimplements the job loop. main()'s own loop is covered by check_V2 and by my dry run (113 texts parse).
- Read-out: with no CELLS.json, or a default one, the grid is NONE_LRS. TOP_ and BOT_ are then 3.2e-2 and 2e-3, as before (TOP_FLAG
  is asserted).
- Test results: selftest 35/35 (mutants 13/13), known 15/15, regress 80/80. The logs postdate the last edit and print the current
  shas.
- The only intended change is the reading-line text.
- Scope: byte-identity was shown on the placeholder composition only. The render function is composition-generic.

Q2. none@6.4e-2 is rendered like the other none cells, and the read-out uses it for N_m and for the top flag.
- Rendering: I compared s16 (B200, slot 3) and s23 (A100) against none@3.2e-2. After substituting "6.4e-2"→"3.2e-2" and
  "0.064"→"0.032", the texts are identical (12 differing lines before substitution; diff_none_cells.py).
- N_m: verdict_v2:532 takes the argmax over the CELLS grid, and core includes 6.4e-2.
- Top flag: line 533 uses the grid max, giving "none at its top lr 6.4e-2". The selftest mutants for both are killed.
- Risk: see I2.

Q3. CELLS.json cannot disagree with what was submitted, except through a manual edit or a partial sbatch (I3).
- --write refuses to change it once any SUBMITTED_pt6v2-* exists (make_jobs:635-636).
- --submit asserts `same` (674), and submit_pt6v2.sh:61-62 greps for it.
- The read-out reads the same file, and a none dir outside the grid becomes a problem.

Q4. The Tier L table is correct for all 18 domains.
- months: no (ptdom AUDIT:150-152).
- The six Part R domains plus cities / countries / people: yes (ptdom2 AUDIT:81).
- toy3: yes (ptdom3 AUDIT:14-24). caesar: yes (byteid_23439548.log:17 "target encodes answer (Tier L): OK").
- s4 / dim: yes (pt6comb AUDIT:232).
- Only wording is affected (the caesar caveat above).

Q5. auto_pt6v2.sh stops at every failure except a partial sbatch (I3).
- Other compositions: it submits a composition other than PICK's only through --pick_json (I1). Otherwise a COMPOSITION.json that
  differs stops it at step 2.
- --canary_dev: a100 is passed consistently (auto:117/121/129/133; submit:57/74). The disk check catches any mismatch.
- Refs gate:
  - Slurm state COMPLETED carries no information, because refs_pt6v2.sbatch ends with `echo rc`. The grep for TABLE/SPLIT PASS is the
    real gate.
  - The refs file is written only on PASS (refs_pt6v2.py:202-209).
  - It catches subjobs with rc≠0 and missing none seeds or canary. It misses absent JSONs for the other arms (minor).
  - Run time: test job 23707298 took 4m38s against a 50-minute limit.
- Double submission: none. The SUBMITTED flags prevent it, and the refs jobid is kept so the refs job is not resubmitted.

Q6. The sha asserts are consistent.
- readout MJ2_SHA = bb15d248afd8 = the current make_jobs.
- The AUDIT table matches the files.
- The canary embeds the read-out sha at render time (make_jobs:384), so the read-out must not change after --write.

- **最可能测错的地方**：不是某个数，而是 auto 链会接受非最终 PICK（`--go --pick_json` 时 partial 或 stub 也能过），从而把错的组合一次性写死（COMPOSITION.json 444）并提交。
- **可能被测没或写没的真效应**：none@6.4e-2 是 core cell，它一旦 crash，会把该 seed 从全部六个判定里剔除，降低功效；同 seed 的 N_m argmax 多了第 6 个候选，winner's curse 对 none 的偏向会略增，使 HELPS 更难。这是已登记的设计取舍，不是 diff 错误。
- **证据撑得住的最强结论**：不带 --extra_none 时，105 个作业逐字节不变，判定规则不变（selftest 35/35、known 15/15、regress 80/80）。带 --extra_none 时只多出 8 个 none@6.4e-2 作业，渲染与 none@3.2e-2 只差 lr，读数器把它纳入 N_m 和 top flag。V2 build 可以照 AUDIT 的描述使用，前提是修掉 I1，或在操作上禁止 `--go --pick_json`。
