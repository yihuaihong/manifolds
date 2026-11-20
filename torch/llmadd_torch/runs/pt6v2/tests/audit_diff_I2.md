# Diff review: I2 (readout_pt6v2.py dbd3654b6f28 -> be0cc0ddcb49), 2026-11-19

Mode: diff review, one pass, read only. Material: H/readout_I2.diff (identical to `diff old new`), H/selftest_readout_I2.log, H/known_readout_I2.log.
Own checks (scripts and outputs in /scratch/yh6210/tmp_audit/pt6v2_I2/): cmp_old_new.py runs the old and new verdict_v2 / report on the same
planted worlds; its output is cmp_old_new.out. lm_line_order.py and its output lm_line_order.out cover the one text difference.

## Verdict: no blocker. All four asks hold, with the two caveats below.

1. Invalid none@6.4e-2 run: drops from that seed's N_m candidates only. Confirmed by reading the code (readout_pt6v2.py:525-553) and by a run.
   The selftest covers the pc route (cities). My run also covered the ro route (compass, P2.floor_tags with the per-seed "N_m" label) and the
   p3 route (hcube). Each keeps 8 seeds, gives g[s17] = GN against the fallback none@1.6e-2, and raises the I2 flag on that manifold only.
   No crash.
2. No other number changes. Confirmed by a run.
   - V2 off: the res dict and the report text match the old version exactly on the default world, on 60 random worlds, and on a world with a
     missing core cell (strict and --allow_partial).
   - V2 on with every 6.4e-2 run valid: the res dict matches on the selftest world and on 60 random worlds. The only text difference is the
     order of the report's "LM (val loss / bpb ...)" line, where none@6.4e-2 now prints after the only-* cells (readout_pt6v2.py:609). This
     is ordering only: the item sets match in 30 of 30 worlds.
   - KNOWN 15/15 shows that verdict_v2 still equals readout_pt6comb.verdict on the real pt6comb runs.
3. The new rules can pick a different N_m than a paired ranking would. This is important, not a blocker.
   - Where: readout_pt6v2.py:549 and 555. The mean of 6.4e-2 is taken over its valid seeds, while the core lrs average over all read seeds,
     so the ranking compares means over different seed sets.
   - Constructed case (cmp_old_new.out, section (3)): 6.4e-2 is invalid at s16-18, and none@3.2e-2 beats it by 2 at every one of s19-23.
     The seed means are 78 vs 72.5, so N_m = 6.4e-2 and mean g = 3.25. Against the paired-stronger none@3.2e-2, mean g = 2.00.
   - Given a flip, the error always runs toward HELPS: the picked none is the weaker one on the paired seeds. The flip needs the
     6.4e-2-vs-best-core gap to be smaller than (k/n)·|seed-effect gap between the invalid and the valid seeds|, where k is the number of
     invalid seeds. The g inflation is then at most (n-k)/n times that term, about 0.11·|Δe| at k=1 and 0.23·|Δe| at k=3.
   - The effect is small but goes against the read-out's stated conservatism (Q_SAME_SEEDS).
   - Fix (1 line): rank on the seeds where every grid lr is valid. With every run valid, this is the registered rule unchanged.
   - Other cases: the fallback lr is ranked by the all-seed mean, which equals the N_m the grid would give without 6.4e-2, so that rule is
     consistent. At "<= half valid", 6.4e-2 leaves the grid. If 6.4e-2 was in fact the strongest none, g is then measured against a weaker
     none (it can only rise). This needs at least 4 of 8 runs invalid, and the report shows it through its report line and the 3.2e-2 top
     flag.
4. Strict mode: an un-started run is still refused. A started run that is still running or requeued is not. This is important.
   - Confirmed by a run: no record gives `REFUSED ... none@6.4e-2 s20 (no run)`. The run directory is created inside the job body
     (`mkdir -p "$OUT"`), and Slurm output goes to logs/, so a pending job has no directory.
   - However, the job template has `#SBATCH --requeue` (pt6comb template line 12). A 6.4e-2 run that is preempted, requeued, or still
     running has a directory but an unfinished curve. Strict mode then reads it as invalid: no refusal, no PROVISIONAL, and the seed uses its
     fallback. If such runs total half or more of the seeds, 6.4e-2 leaves the grid.
   - It is visible in the output: "Amendment V2 (I2): none@6.4e-2 not valid at s20 (unfinished (31 evals))" plus the NOT VALID line.
   - Check / fix: run the strict read-out only once squeue shows no pt6v2 job. Alternatively, strict could count an EXTRA_NONE record whose
     why is "unfinished" or "no run JSON" as missing unless --allow_partial is set. That would refuse a crash too, which is a trade-off.

## Minor
- When 6.4e-2 leaves the grid, the renderer prints `MISSING m.<m>.none_means.6.4e-2` for every manifold. The cause is that res["none_lrs"]
  keeps the full list while none_means holds the grid only. A vs_lr entry with fewer than 2 valid seeds prints as "None"
  (render_findings_pt6v2.py:236-238). The RUNBOOK's `grep -c MISSING` would count these as false alarms. Cosmetic only.
- Side fix at readout_pt6v2.py:657-662 (a crashed record prints "-" instead of raising KeyError): this is a strict improvement. On valid
  core cells the output is unchanged.
