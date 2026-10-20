# Standard for a "rescue" claim (pre-registered 2026-10-20, before any result of wall 1 / wall 2 / rb6 A7-A8)

User goal: find tasks that a language model cannot learn however it is trained, but learns with a hand-designed
geometric prior. Proposed by the writing session; adopted here as binding for any FINDINGS entry or paper sentence
that calls a result a "rescue" ("the prior makes an unlearnable task learnable"). A result that misses a criterion is
reported as what it is (acceleration, reliability, partial gain), never as a rescue.

- (a) none: a full lr x wd sweep with the optimum bracketed on both sides (no best cell at a grid edge by more than
  the experiment's tie margin), plus at least a 4x budget cell; none fails at every configuration by the experiment's
  pre-registered floor rule.
- (b) the prior arm gets its own lr sweep (best lr bracketed or the bracket state reported); the comparison is best vs
  best.
- (c) the target does not encode held-out answers: report what the target alone implies about held-out answers (e.g.
  matrix representations of a group encode the whole table; anchor coordinates in F81 encode the answer = fact
  injection). A target that encodes them can support "the model learns to read out a given structure", not a rescue.
- (d) a shuffled / same-spectrum random target in the same schedule fails.
- (e) confirm seeds (fresh, not used to form the hypothesis or select an lr) agree.
- (f) every arm runs on the same GPU type (or the device effect is measured).
Per experiment, its own AUDIT pre-registration still defines the floor rule, the metric and the tests; this file adds
the criteria a claim must also meet to be called a rescue.

Status at registration (2026-10-20):
- Wall 1 (runs/ptheld, Torch): (a) mostly (none lr 2e-3 / 8e-3 / 1.6e-2, wd 0.01 / 0.1 / 1, 4x tokens); (b) open: prior
  arms at one lr (2e-3); (c) met (helix encodes values, not sums); (f) met (L40S).
- Wall 2 (runs/toyshape, Torch): shape-only targets (c met by construction); (b) open for most prior arms (lr 1e-2).
- rb6 A7 / A8 (Burst, A100): S4 matrix targets encode the table, so (c) is NOT met [superseded by Amendment 1: at most Tier L]: no rescue claim is possible there
  whatever the result; A8 can support "reliable / early", not "rescue".

## Amendment 1 (2026-10-20, user decision, before any result of wall 1 / wall 2 / rb6 A8 was read): two tiers for criterion (c)
Criterion (c) was applied inconsistently above: a faithful matrix representation of a group determines its table, but
so do the numbers' values on the helix (given every value, the addition table is determined). The user asked to try both
readings; to avoid choosing one after the results, both are fixed now as two labelled tiers. Every rescue result is
reported with the highest tier it meets, and the paper names the tier.
- **Tier S (strict)**: the target does not determine the operation table (e.g. a permutation-invariant, shape-only
  target that gives the geometry but not which item sits where). Only shape-only priors (wall 2) can reach it.
- **Tier L (lenient, "structure given")**: the target does not place any question's answer, but it may give a
  structure from which the table follows once the model learns the operation (helix values for numbers, matrix
  representations of a group). A Tier-L rescue is written as "given the structure, the model learns to use it", never
  as "the model discovered the structure".
- **Neither**: targets that place the answer's coordinates directly (anchor targets, F81 fact injection).
- Criteria (a), (b), (d), (e), (f) apply to both tiers unchanged.
- Tier per current experiment: wall 1 (helix on held-out numbers) -> at most L; wall 2 shape / shapeprof arms -> at most
  S (its assignment controls at most L); rb6 S4 (rotation matrices) -> at most L, and (a) currently fails (none learns S4
  at lr 1.6e-2 / 3.2e-2 in some seeds, F88 A7); the new-manifold screen (notes/new_manifolds_screen_design.md) is to run
  both an assignment prior (L) and a shape-only prior (S) per candidate.
