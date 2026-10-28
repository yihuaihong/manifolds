# pttpp: remaining work per group (Amendment T8; 2026-10-27T20:52-04:00; structural fields only, no result read)

Run counts are the runs designed so far (stage A, the decided X / C stages); 'C to decide' = the confirm runs of the
arms without a C yet (8-seed cells: seeds 4-7; 1.07B: 2-7); an X may come first (open bracket, or T6: none's pick at
the upper grid edge). GPU-h per run = the generator's planning rate on the designed hardware (est_h).

| setting | size | tpp | group | hardware | done | running | pending | unsubmitted | lost / other | C to decide | GPU-h per run | lr decisions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2% | 190M | 0.745 | tpp2-190m-t0.745 | A100-80GB (520, a100_tandon) | 16 | 0 | 8 | 0 | 0 | 8 | 0.62 | none: open (X 1.92e-2 running / to finish, then C); win-ctx: open (X 2.4e-3 running / to finish, then C) |
| 2% | 294M | 0.745 | tpp2-294m-t0.745 | A100-80GB (520, a100_tandon) | 13 | 0 | 7 | 0 | 0 | 4 | 1.49 | none: C 4.8e-3 (closed); win-ctx: open (stage A not complete: C, or an X first if the pick is at an edge) |
| 7% | 190M | 0.318 | tpp7-190m-t0.318 | L40S (625, l40s_public) | 30 | 0 | 2 | 0 | 0 | 4 | 0.49 | none: open (X 3.84e-2 running / to finish, then C); win-ctx: C 4.8e-3 (closed) |
| 7% | 294M | 0.318 | tpp7-294m-t0.318 | L40S (625, l40s_public) | 31 | 0 | 1 | 0 | 0 | 0 | 1.08 | none: C 1.92e-2 (closed); win-ctx: C 1.2e-3 (closed) |
| 7% | 505M | 0.318 | tpp7-505m-t0.318 | L40S (625, l40s_public) | 24 | 0 | 0 | 0 | 0 | 0 | 2.99 | none: C 9.6e-3 (closed); win-ctx: C 1.2e-3 (closed) |
| 7% | 93M | 0.318 | tpp7-93m-t0.318 | L40S (625, l40s_public) | 32 | 0 | 0 | 0 | 0 | 0 | 0.18 | none: C 7.68e-2 (closed); win-ctx: C 9.6e-3 (closed) |
| 7% | 1.07B | 0.745 | tpp7-1b-t0.745 | B200 (520, preemptible) | 1 | 3 | 4 | 0 | 0 | 12 | 6.5 | none: open (stage A not complete: C, or an X first if the pick is at an edge); win-ctx: open (stage A not complete: C, or an X first if the pick is at an edge) |
| 7% | 190M | 0.745 | tpp7-190m-t0.745 | L40S (625, l40s_public) | 28 | 0 | 0 | 0 | 0 | 0 | 0.88 | none: C 9.6e-3 (closed); win-ctx: C 1.2e-3 (closed) |
| 7% | 294M | 0.745 | tpp7-294m-t0.745 | L40S (625, l40s_public) | 24 | 0 | 4 | 0 | 0 | 4 | 2.17 | none: open (X 3.84e-2 running / to finish, then C); win-ctx: C 2.4e-3 (closed) |
| 7% | 93M | 0.745 | tpp7-93m-t0.745 | L40S (625, l40s_public) | 28 | 0 | 0 | 0 | 0 | 0 | 0.25 | none: C 1.92e-2 (closed); win-ctx: C 2.4e-3 (closed) |
| 7% | 505M | 3e8 tokens | tppb-505m-3e8 | B200 (520, preemptible) | 0 | 0 | 8 | 0 | 0 | 0 | 1.49 | none (no selection) |

## Remaining work and a rough completion estimate (2026-10-27 21:00 EDT)
GPU-h on the designed hardware (planning rates; X = a possible extension before C):
- **B200** (1.07B@0.745 + bridge): stage A 7 runs not done (3 running, 4 pending) <= 45.5 h; bridge 8 x 1.49 = 11.9 h; C 12 x 6.5 = 78 h;
  X 0-2 arms x 2 x 6.5 = 0-26 h. About 130-160 B200-h. Critical path: stage A -> (X) -> C, 6.5 h per wave. Since T8: 8 singleton
  slots; 3 running, 5 eligible, the rest wait behind them in their slot.
- **L40S** (7%): 7 pending runs (190M@0.318 2 x 0.49, 294M@0.318 1 x 1.08, 294M@0.745 X 4 x 2.17) + C to decide 8 runs
  (190M@0.318 4 x 0.49, 294M@0.745 4 x 2.17) = about 21 L40S-h, two short stages.
- **A100** (2%): 15 pending runs (190M X 8 x 0.62, 294M 7 x 1.49) + C to decide 12 (190M 8 x 0.62, 294M win-ctx 4 x 1.49, maybe an
  X first) = about 26-32 A100-h, two or three stages; the jobs wait on the Tandon group's A100 cap (QOSGrpGRES), not on ours.

| part | optimistic | conservative |
|---|---|---|
| B200 | 8 GPUs from tonight: stage A 10-28 morning, an X wave, C in 2 waves -> 10-29 | about 3 GPUs (as now; sbatch --test-only estimates new starts 10-30 21:51; preemptions): stage A 10-30, X 10-31, C in 4 waves -> 11-02 |
| L40S | 10-28 | 10-30 (test-only estimate of a new start: 10-30 03:04) |
| A100 | 10-29 | 10-31 (about 1.1 runs / h so far through the group cap) |
| whole batch | 10-29 | 11-02 |

Note (T8 audit): the L40S conservative date can slip to 10-31 to 11-02 (an l40s_public probe at 20:15 estimated a new start at
11-01 17:53; 294M@0.745 and 190M@0.318 none still need X and then a newly submitted C), and the batch's conservative 11-02 is
not a hard bound (B200 preemption; a possible 1.07B X wave).
