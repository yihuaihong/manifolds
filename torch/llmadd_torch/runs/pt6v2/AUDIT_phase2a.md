# pt6v2 phase 2a: read-out notes (for the coordinator to merge into AUDIT.md; AUDIT.md itself not edited)

Code: `runs/pt6v2/readout_pt6v2.py`. Tier: confirmatory. The verdict rule is AUDIT.md's, unchanged.

## What the read-out does
- **Sources.** The four source read-outs are imported read only. The module sha-asserts each one and checks its import path:
  - readout_pt6comb 05ea474d9c9b
  - readout_ptdom2 3f78507d5d47
  - readout_ptdom3 9abd89ad75a7
  - readout_ptdom 1e2f17522e47
- **Names, cells and composition** come from `make_jobs_pt6v2.py` (owned by 2b). Its sha is printed in the header, not asserted. Its constants (PREFIX, SEEDS, HALVES, P_LR, SAME_LR, NONE_LRS) are asserted.
- **Primary per domain.** Each domain is routed to its source's read-out:

| route | domains | primary | FLOOR rule |
|---|---|---|---|
| pc | numbers, s4, dim, cities, countries, people | `PC.series_of` | per-run `PC.floor_line`, averaged over none@N_m |
| ro | months, Part R | `metrics[RO.READ[d]]` | `P2.floor_tags` (max of the mean lines + 5) on adapter records |
| p3 | toy3 | `P3.dom_numbers` | the none runs' mean chance + 5 |

  - toy3 primary: the twin-free set for hcube and d6, the full set for the rest. The p3 FLOOR uses chance_u for the twin-free domains.
  - d6 prints AAE / noAAE.
  - caesar is printed in its own `[separate]` block with `P3.CAESAR_NOTE`.
- **Validity** is a port of `PC.load`: identity, check.json, finished, allowed failed checks, the A6 geo rule (geo_a6_v2) and `PC.integrity`. It also records the device from gpu.csv.
- **Verdict.**
  - N_m = argmax of the none mean over the 4 lrs; a tie goes to the higher lr.
  - g = all6@1e-3 − none@N_m.
  - HELPS / interference come from `PC.manifold_verdict`, unchanged.
  - Flags: TOP (N_m = 3.2e-2) and LOW (always).
  - Descriptive lines: g vs each none lr, vs the mean6-picked none, and all6@2e-3 − none@2e-3.
  - GATE (A2.1) is applied per half (16-19 and 20-23) at all6@1e-3; the A3 gate is applied per half at all6@2e-3.
  - Holm is a reference column. Strict refusal; `--allow_partial` marks the output PROVISIONAL.
- **Report.** Section 1 is the full table: area, f3, secondaries and LM. Section 2 is the verdicts, then the HEADLINE.

## Tests
- **Test 4 (`--selftest`).** 25 checks on planted worlds, plus 8 mutants, all killed. Log: `tests/selftest_pt6v2_*.log`.
  - The 'no refusal' mutant is killed by a crash (KeyError in the report), not by a clean check failure.
- **Test 5 (`--known`).** Four comparisons:
  - pt6comb: `load_v2(spec_pc)` + `verdict_v2` vs `PC.load` / `PC.verdict` / readout.txt;
  - ptdom2 Part R + months: `dom_metrics` vs `P2.load2`;
  - ptdom3 screen: vs `readout_partial_20261118T1405.txt` (21 numbers);
  - the canary path on a real pt6comb run and on the ptdom3 canary.

  Log: `tests/known_pt6v2_*.log`.

## Pending decisions (for the coordinator / user; the rule was NOT changed)
1. **caesar** counts in the "k of 6" headline but is printed separately. Should it be excluded from k instead? The AUDIT says "read out separately"; I kept it counted.
2. **FLOOR differs by source**, each inherited as is:
   - pt6comb: the mean of the per-run lines over the none runs; both arms ≤ line.
   - ptdom2: the max of the mean lines + 5.
   - ptdom3: chance + 5.

   A cross-family comparison of FLOOR tags is therefore not like for like. Tags only, never conditions.
3. **N_m with the 3.2e-2 top.** When N_m = 3.2e-2, the none optimum may lie above the ladder. The rule only flags this; g is then possibly an over-estimate. The flag is printed.
4. **The read-out depends on the mutable make_jobs_pt6v2.py.**
   - Its sha moved from 005fca70b2be to 961ef81fe230 during this build (2b is editing).
   - The coordinator should freeze it, or assert its sha in readout_pt6v2.py SRC, before the first real read.
5. **Strict refusal inherits every problem string:**
   - P3 marks (e.g. a missing twin set);
   - double-path mismatches > 1e-6;
   - `RO.run_subsets` problems.

   One such mark on one run refuses the whole read-out; `--allow_partial` reads through it.
6. **Window split** (in / out of the prior window) is not printed for the twin-free primaries (hcube, d6), because there is no per-eval twin-free series in dom_numbers' output.
7. **GATE per half with 3 or fewer valid seeds** in a half under `--allow_partial`: `RO.cell_gate` runs on whatever seeds are valid, with no minimum count.
8. **Device** comes from gpu.csv line 2, field 2 (falling back to check.json info.device). If neither exists it prints `None`. It is a note only.
