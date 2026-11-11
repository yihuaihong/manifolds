# ptsize999 — size ladder in the unified 0..999 setup (2026-10-14)

Same code, data, split, evals, gate and metrics as runs/pt999 (intent spec runs/pt999/AUDIT.md); only the model size,
learning rate and the default prior layers change (1/3 and 2/3 of the depth: 190M 4,8; 505M 8,16; 53M 2,4).
Replaces runs/ptsize (10..99, pretrain_helix.py) after the user's decision to unify on 0..999.

Question: does the gated helix3 prior (operands in the lines, or natural-text numbers) still speed up learning of
held-out sums at 190M / 505M, and does the speed-up shrink or grow with size at a fixed 3e8 tokens?
Arms, seeds 0-3: 190M 7% none / gate-ctx / gate-text / shuf-ctx, 2% none / gate-ctx / gate-text; 505M 7% none /
gate-ctx / gate-text. Comparisons within this batch and against pt999 (all L40S).
Known limits: 4 seeds; one lr per size (6e-4, 3e-4, as ptsize); 3e8 tokens is far below compute-optimal for 505M.

2026-10-14: code snapshot refreshed to fdfadfd before any job ran (pt999 audit fixes). Relevant here: gradient
checkpointing turns on automatically above 150M parameters (the 10..99 505M job 19782846 ran out of memory on a 48 GB
L40S at bs 32; 190M also gets it, same math, ~1/3 slower); per-run code_sha in every result.

2026-10-15 (user): window arms added, 190M and 505M, 7%, win-ctx (0.05, 0.4), seeds 0-3, same snapshot fdfadfd (the
window schedule is unchanged code). Reason: the main recipe is moving to the 5% window (F79).
