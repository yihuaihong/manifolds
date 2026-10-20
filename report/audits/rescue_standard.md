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
- rb6 A7 / A8 (Burst, A100): S4 matrix targets encode the table, so (c) is NOT met: no rescue claim is possible there
  whatever the result; A8 can support "reliable / early", not "rescue".
