# ptdom2 Part R final: result audit (exp-auditor, 2026-11-19)

Scope: a confirmatory result audit of `tests/partR_final.{txt,json}` and `tests/partR_final_render.md`, checked against the planned F103 sentences.
Read only: no training rerun and no project file changed. The only exception is this notes file, which the delegation asked for.
Scripts and outputs: `/scratch/yh6210/tmp_audit/ptdom2_partR_final/{recompute.py,recompute.out,samelr.py,samelr.out}`.

## Verdict
No blocker. Every registered number reproduces exactly from the raw run outputs. Three wording fixes are needed, plus one wording that is too strong:
1. sentence 1: "≥ .55" should be ".546";
2. sentence 1: "beats … in none of them" reads as a null; per A5 (iii) it should read "not shown";
3. sentence 3: the FLOOR paraphrase is wrong;
4. sentence 3: the stars reason is wrong.

Sentence 2 also needs the "lr-confounded" label, which the pre-registration requires on that line.

## (a) Reproduction

### What was recomputed
`recompute.py` is independent: it does not import either read-out module. It covered all 5 family domains (not just 2), 3 arms each (true@P, shuf@P, none@N), seeds 8-15, which is 120 runs. Each run was taken from the highest `-rK` dir.

Area was computed four ways, and the four agree on every run (max |diff| = 0.0):
- (i) `metrics[<d>_readout].area`;
- (ii) 100 × mean of the 75-eval `<d>_held_2way` series;
- (iii) 100 × mean over evals of sum(hits2 digits) / (questions × contexts);
- (iv) the Slurm log's in-job `read_area`, from `logs/llmadd_<run>_<jobid>.out` with the job id taken from `SUBMITTED_<run>`.

### Integrity checks (all pass on all 120 runs)
- check.json has no failed checks.
- Each run has 75 evals.
- `args` matches the arm: seed, lr, prior_domains and prior_shuf.
- code_sha is d87a1502cee8.
- Within each seed, the three arms have the same per-eval data_sha and the same split_seed.

### Gains and statistics
For each domain I recomputed:
- per-seed gains for true−none and true−shuffled;
- the mean, bootstrap 90% interval (rng 0, 10000 resamples, 5/95 percentiles), both halves, and wins;
- the one-sided t p and Holm over 5, computed separately for each contrast.

Every one matches `partR_final.json` and the txt. The per-seed max diff is 0.

| | landmarks | compass | prefixes | ranks | scalar_adj |
|---|---|---|---|---|---|
| true−none, mean (wins) | −1.18 (3/8) | +2.76 (4/8) | −0.95 (3/8) | +3.73 (7/8) | +2.81 (6/8) |
| true−none, Holm p | 1.000 | .5456 | 1.000 | .5456 | .5456 |
| true−shuf, mean (wins) | +7.12 (7/8) | +7.12 (6/8) | +6.69 (7/8) | +15.74 (8/8) | +5.91 (7/8) |
| true−shuf, Holm p | .0109 | .0425 | .0218 | .0425 | .0168 |
| shuf−none, mean (wins) | −8.30 (0/8) | −4.37 (4/8) | −7.64 (0/8) | −12.01 (1/8) | −3.10 (1/8) |
| shuf−none, lr | lr-confounded | lr-confounded | same lr | lr-confounded | lr-confounded |

Holm by hand, true−none: the sorted raw p are .109, .125, .181, .684, .848. Multiplying by 5, 4, 3, … gives .546, .500, .544, …. Taking the running max gives .546 for all three small ones, and 1 for the other two.

**Number error (minor):**
- Sentence 1's "all Holm p ≥ .55" is wrong. The minimum is .5456, so write "≥ .54" or ".546".
- "Holm p ≤ .043" (max .0425), "+5.9 to +15.7", "6-8/8", "−1.2 to +3.7" and sentence 2's "−3.1 to −12.0" are all correct.

### FLOOR line
The registered rule (AUDIT.md:86-87) is line = max(chance, answer-frequency line, shorter-option line, shortcut baseline on every held-out question) + 5.

Inputs:
- chance, answer-frequency and shorter-option were recomputed from the raw run JSON `domains[d]` (true + none runs) and match the JSON `floor.lines`.
- The max is the shortcut baseline in all 5 domains:
  - shortcut baseline: 65.15 / 84.70 / 84.52 / 92.97 / 87.22;
  - line: 70.15 / 89.70 / 89.52 / 97.97 / 92.22.
- I did not re-derive the shortcut baseline (h3_held) independently. It comes from the frozen `split_info`, which `run_subsets` already checks against the trainer's masks and heuristic lines, and the read-out reports no problems.

Arm means (true / none / shuffled) against the line:

| Domain | true | none | shuffled | Line | Tag |
|---|---|---|---|---|---|
| landmarks | 76.62 | 77.80 | 69.50 | 70.15 | no tag |
| compass | 81.76 | 79.00 | 74.64 | 89.70 | FLOOR |
| prefixes | 83.35 | 84.30 | 76.66 | 89.52 | FLOOR |
| ranks | 93.86 | 90.13 | 78.12 | 97.97 | FLOOR |
| scalar_adj | 88.29 | 85.47 | 82.38 | 92.22 | FLOOR |

The tags are computed correctly, and floor_geo uses the same line.

**Description error (sentence 3):** "both arms at or below the shortcut line" does not match the definition.
- In ranks (true 93.86 > 92.97) and in scalar_adj (88.29 > 87.22), the true arm is above the shortcut baseline itself.
- The arms are only within 5 points of it.
- Better: "both arms no more than 5 points above the best shortcut line (ranks and scalar_adj's true arms are above the shortcut itself)".

### Stars reason (sentence 3)
"its none lr pick sits at the edge of the grid" is wrong.
- The none pick is 8e-3, which is not at the edge.
- What makes stars NOT FINAL is the none area argmax 1.6e-2, the outermost lr after the allowed extension (A4 rule: pick or argmax).
- Write: "the none arm's area peaks at the outermost lr, 1.6e-2, so none may be better beyond the grid".

### Other checks
- The partial read-out (`partR_partial_20261119T0524.txt`) has the same landmarks per-seed numbers as the final. Only Holm differs (family 1 → 5), as expected.
- The render table matches the txt.
- The render shows the shuffled−none boot90 as "MISSING (JSON 不存)", but the txt has it. This is only a display gap and not used by these sentences.

## (b) Strongest supported statement

### Registered results
- GEOMETRY-SPECIFIC in 5/5: all four conditions hold in every domain.
- PASS in 0/5.
- A5 (iii) fixes the wording as READING's: "true > shuffled, but no gain vs none shown (the shuffled prior may hurt)".

### Sentence 1 is too strong on the negative side
"beats the no-prior baseline … in none of them" reads as an established null, but NOT SHOWN means only "no reliable difference at 8 seeds".
- Raw means are positive in 3/5 domains.
- Ranks (+3.73, 7/8, both halves > 0) and scalar_adj (+2.81, 6/8, both halves > 0) meet 2 of the 4 registered conditions. They miss only the boot90 lower bound (−0.71 / −0.80) and Holm.
- Write: "a gain over the tuned no-prior baseline is not shown in any of them (−1.2 to +3.7, Holm p ≥ .546)".

### The other side, so this is not over-read
- Area is the primary metric; final3 is secondary and descriptive.
- On final3, the true−none differences in ranks (−3.22, 3/8) and scalar_adj (−0.53) disappear. So the ranks and scalar_adj area advantage over none is an early-training advantage, not a higher endpoint.
- No hidden PASS has been written away.

### Can be added to sentence 1 (descriptive, final3)
GEOMETRY-SPECIFIC also holds at the endpoint. On final3, the true−shuffled means are positive in all 5 domains:
- landmarks +5.73 (6/8)
- compass +6.93 (7/8)
- prefixes +4.98 (5/8)
- ranks +12.01 (5/8)
- scalar_adj +3.79 (5/8)

The LM gradient shares of the two arms are matched (0.92–0.98), so "dose-matched" holds.

### Sentence 2
"Below none in every domain (−3.1 to −12.0)" is correct as means.
- Compass is only 4/8 seeds, but its boot90 [−8.17, −0.64] excludes 0.
- **The pre-registration requires the label** (AUDIT.md:440; A3.2 at 549): this line is lr-confounded in 4/5 domains. Only prefixes has P = N.
- Add "lr-confounded in 4/5, except prefixes".

Same-lr check on existing runs (descriptive, not registered; all runs pass check, data_sha matches):
- The registered same-lr line on seeds 8-15 exists only for P = N. I extended it to the cells that exist:
  - landmarks: none@2e-3, from Part S;
  - compass and ranks: the B2 none@1e-3 cell, which is prefixes' none pick.
- Scalar_adj has no none@1e-3 cell.

Results at matched lr:
- shuffled−none:
  - landmarks −4.12 (2/8)
  - compass −1.70 (3/8, boot90 covers 0)
  - ranks −7.43 (2/8)
  - prefixes −7.64 (0/8)
- Share of the true−shuffled gap explained by the shuffled arm falling below none: 58% / 24% / 47% / 114%.

So "partly" is accurate:
- all of it in prefixes;
- about a quarter to a half in compass, ranks and landmarks.

The same check shows true − none at the prior's own lr:
- ranks +8.31 (8/8, boot90 [+5.17, +12.28], raw p .005);
- compass +5.42 (7/8, [+1.69, +8.80]);
- landmarks +3.00 (6/8, [+0.23, +5.83]).

This means that in these 3 domains the true geometry is above none at the same lr. "True > shuffled" there is not only the harm of a wrong target.

But none at 1e-3 / 2e-3 is not at its best lr. Tuning none's lr removes the difference, so this cannot be written as PASS. If it goes in, write it as: "at matched lr, true > none (descriptive); tuning the baseline's lr closes the gap".

### Tied-embedding note (not required)
Compass is 100% single-token and scalar_adj 94%, so the prior acts directly on the read-out candidates' own output rows. This mechanism belongs in the reading line of the read-out. One clause in F103 is optional.

## Planned sentences, rewritten (minimal changes)
1. "On fresh seeds 8-15, the true-geometry prior beats a dose-matched shuffled prior in all 5 comparison domains (landmarks, compass, prefixes, ranks, scalar_adj; area +5.9 to +15.7, 6-8/8 seeds, Holm p ≤ .043; final3 +3.8 to +12.0). A gain over the no-prior baseline (each domain's best none lr) is not shown in any of them (−1.2 to +3.7, Holm p ≥ .546): GEOMETRY-SPECIFIC without PASS in 5/5."
2. "The shuffled prior is below none in every domain (−3.1 to −12.0; lr-confounded except prefixes). So part of 'true > shuffled' is the harm of a wrong target: all of it in prefixes (same lr, −7.6, 0/8), and about a quarter to a half in landmarks, compass and ranks at matched lr (descriptive)."
3. "Compass, prefixes, ranks and scalar_adj carry a FLOOR tag (both arms no more than 5 points above the best shortcut line). Stars has no verdict: its none arm peaks at the outermost lr, 1.6e-2."

## file:line
- readout_ptdom2.py:559-593 floor_tags
- readout_ptdom.py:595-632 run_subsets (h3_held), :943-981 boot_ci / t / holm
- AUDIT.md:86-87 FLOOR line definition
- AUDIT.md:440, 549 the lr-confounded label
- AUDIT.md:728 A5 (iii) wording
- tests/partR_final.txt:4 (stars), :10/25/55/69 (shuffled−none, lr-confounded), :14/29/44/59/73 (FLOOR)
