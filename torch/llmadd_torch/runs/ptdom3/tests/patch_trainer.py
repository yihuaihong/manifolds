"""ptdom3 phase 2 (2026-11-17): Rev H of tools/pretrain_range.py = Rev G (11f61b9) + the toy3 domains (tools/toy3.py), by
exact, counted text replacements (each old string must occur exactly once). Every new path is guarded by `d in TOY3` /
REVT (a toy3 domain in --domains); the default and every existing domain path are unchanged (tests/byteid.sh proves it)."""
from pathlib import Path

P = Path("/scratch/yh6210/llmadd-burst-ptdom3/tools/pretrain_range.py")
s = P.read_text()
R = [
# 1. the domain registry
('ALL_DOMAINS = ("numbers",) + EXTRA + ANCHOR + NEWDOM\n',
 '# Rev H (ptdom3): the toy-structure screen (tools/toy3.py, = toy3.TOY3, asserted at use): nonce / letter items, group-table and\n'
 '# box lines, exact-match read-outs as s4 / dim; their own SeedSequence(4400 + seed).spawn(3 x 6): 3 i split, 3 i + 1 shuffled\n'
 '# target, 3 i + 2 eval contexts of TOY3[i]; the training lines cycle over the batch\'s domains in the order EXTRA + ANCHOR +\n'
 '# NEWDOM + TOY3\n'
 'TOY3 = ("cyc12", "torus", "hcube", "d6", "box5", "caesar")\n'
 'ALL_DOMAINS = ("numbers",) + EXTRA + ANCHOR + NEWDOM + TOY3\n'),
# 2. the gate: Rev E's logging (gradient split, LM share, clip coefficient) also for a toy3-only batch
('    REVE = (DOM and bool(set(DOMS) & set(ANCHOR))) or REVF   # Rev E (ptdom): anchor domains, k-way / 2-way choice, gradient split\n',
 '    REVT = DOM and bool(set(DOMS) & set(TOY3))     # Rev H (ptdom3): a toy3 domain\n'
 '    REVE = (DOM and bool(set(DOMS) & set(ANCHOR))) or REVF or REVT   # Rev E (ptdom): anchor domains, k-way / 2-way choice, gradient split\n'),
# 3. EXT order
('    EXT += [d for d in NEWDOM if d in DOMS]     # Rev F (empty unless a Rev F domain is named)\n',
 '    EXT += [d for d in NEWDOM if d in DOMS]     # Rev F (empty unless a Rev F domain is named)\n'
 '    EXT += [d for d in TOY3 if d in DOMS]       # Rev H (empty unless a toy3 domain is named)\n'),
# 4. names: single distinct tokens, disjoint from s4 / dim
('        S4N, DIMN = list(S4_NAMES), list(DIM_NAMES)\n',
 '        S4N, DIMN = list(S4_NAMES), list(DIM_NAMES)\n'
 '        if REVT:   # Rev H: the toy3 items (nonce names from toy3_nonce.json; caesar: the capital letters), each one token after a space\n'
 '            import toy3\n'
 '            assert toy3.TOY3 == TOY3, (toy3.TOY3, TOY3)\n'
 '            T3N = {d: [nm.strip() for nm in toy3.item_names(d)] for d in EXT if d in TOY3}\n'
 '            t3all = [nm for d in T3N for nm in T3N[d]]\n'
 '            assert len(set(t3all)) == len(t3all) and not (set(t3all) & (set(S4_NAMES) | set(DIM_NAMES))), "toy3 names not distinct"\n'
 '            for d in T3N:\n'
 '                ids_ = [enc(" " + nm) for nm in T3N[d]]\n'
 '                assert all(len(t_) == 1 for t_ in ids_) and len({t_[0] for t_ in ids_}) == len(ids_), (d, "toy3 items: one distinct token each")\n'
 '                assert all(tok.decode(t_).strip() == nm for t_, nm in zip(ids_, T3N[d])), (d, "toy3 items decode")\n'),
# 5. the line of a question
('            Dd = DTASK[d]; x = Dd["X"][q]; c = int(Dd["Y"][q])\n            if d == "s4":\n',
 '            if d in TOY3:   # Rev H: toy3.pieces (the sites = both input items, the answer item never a site)\n'
 '                pcs_, astr_, ins_ = toy3.pieces(DTASK[d], q, [" " + nm for nm in T3N[d]])\n'
 '                return pieces_line(pcs_), astr_, ins_\n'
 '            Dd = DTASK[d]; x = Dd["X"][q]; c = int(Dd["Y"][q])\n            if d == "s4":\n'),
# 6. tables and the stream
('        DTASK ={d: ts.build_task(d) for d in ("s4", "dim") if d in EXT}\n',
 '        DTASK ={d: ts.build_task(d) for d in ("s4", "dim") if d in EXT}\n'
 '        if REVT:   # Rev H: the toy3 tables (deterministic) and their own stream (the 4100 / 4200 / 4300 streams untouched)\n'
 '            DTASK.update({d: toy3.build(d) for d in EXT if d in TOY3})\n'
 '            toy_ss = np.random.SeedSequence(toy3.TOY3_STREAM + args.seed).spawn(3 * len(TOY3))\n'),
('            if d in NEWDOM:     # Rev F: its own children of SeedSequence(4300 + seed) (the 4100 / 4200 streams untouched)\n',
 '            if d in TOY3:       # Rev H: its own children of SeedSequence(4400 + seed)\n'
 '                ix = TOY3.index(d)\n'
 '                ch_split, ch_shuf, ch_eval = toy_ss[3 * ix], toy_ss[3 * ix + 1], toy_ss[3 * ix + 2]\n'
 '            elif d in NEWDOM:   # Rev F: its own children of SeedSequence(4300 + seed) (the 4100 / 4200 streams untouched)\n'),
# 7. split, masks, read-out
('                DTAB[d] = circle(n)\n            else:\n                Dd = DTASK[d]\n',
 '                DTAB[d] = circle(n)\n'
 '            elif d in TOY3:   # Rev H: toy3.split (toy_struct.split\'s two modes, the s4 / dim retry rule) and toy3.masks\n'
 '                Dd = DTASK[d]\n'
 '                mode, frac = toy3.SPLIT[d]\n'
 '                base = int(sr.integers(2 ** 31))\n'
 '                for att in range(200):   # every item a trained input and a trained answer (toy3.split asserts it)\n'
 '                    try:\n'
 '                        tr, te = toy3.split(Dd, frac, base + att, mode)\n'
 '                        break\n'
 '                    except AssertionError:\n'
 '                        continue\n'
 '                else:\n'
 '                    raise RuntimeError(f"no valid {d} split")\n'
 '                train_q, held_s = [int(i) for i in tr], [int(i) for i in te]\n'
 '                rows_ = [tuple(r) for r in Dd["X"].tolist()]\n'
 '                assert not ({rows_[i] for i in train_q} & {rows_[i] for i in held_s}), "held-out question in training"\n'
 '                m_, ro_name, ro = toy3.masks(Dd, tr, te, ts.s4_closure)\n'
 '                hard_ = m_.pop("hard", None); m_.pop("hard_ro", None)   # descriptive only (info), never an eval mask\n'
 '                masks["held"] = m_\n'
 '                maj = int(np.bincount(Dd["Y"][tr], minlength=Dd["n_items"]).argmax())\n'
 '                info.update(n_items=Dd["n_items"], split_mode=mode, train_frac=frac, split_seed=base + att, tier=Dd["tier"],\n'
 '                            chance_uniform=1 / Dd["n_items"], chance_majority=ts.majority_chance(Dd, tr, te),\n'
 '                            n_copy_held=int((~m_["nonid"]).sum()), readout=ro_name)\n'
 '                if hard_ is not None:   # held cells not derivable by associativity + Latin cancellation (s4_hard\'s analogue)\n'
 '                    info.update(closure_latin=float(1 - hard_.mean()), n_hard_closure=int(hard_.sum()))\n'
 '                assert ro.sum() >= 50, (d, int(ro.sum()))\n'
 '                info.update(chance_readout=max(1 / Dd["n_items"], float(np.mean(Dd["Y"][te][ro] == maj))), n_readout=int(ro.sum()))\n'
 '                DTAB[d] = Dd["targets"]["true"]\n'
 '                sets = {"held": held_s}\n'
 '            else:\n                Dd = DTASK[d]\n'),
# 8. the nonce recount (toy3 nonce names; caesar's letters are real tokens: not counted)
('            print(f"nonce tokens: {len(want)} single tokens, occurrences in train.bin ({len(train)} tokens) as recorded: max {max(got.values())}")\n',
 '            print(f"nonce tokens: {len(want)} single tokens, occurrences in train.bin ({len(train)} tokens) as recorded: max {max(got.values())}")\n'
 '        if REVT and any(d in toy3.NONCE for d in T3N):   # Rev H: the toy3 nonce names re-counted (= toy3_nonce.json)\n'
 '            want = {nm: c for d in T3N if d in toy3.NONCE for nm, c in toy3.NONCE[d].items()}\n'
 '            ids_ = np.array([enc(" " + nm)[0] for nm in want])\n'
 '            cnt = np.zeros(len(ids_), dtype=np.int64)\n'
 '            for i in range(0, len(train), 50_000_000):\n'
 '                cnt += np.bincount(np.asarray(train[i:i + 50_000_000]), minlength=len(tok))[ids_]\n'
 '            got = dict(zip(want, cnt.tolist()))\n'
 '            assert got == want, {k: (got[k], want[k]) for k in want if got[k] != want[k]}\n'
 '            print(f"toy3 nonce tokens: {len(want)} single tokens, occurrences in train.bin as recorded: max {max(got.values())}"\n'
 '                  + ("; caesar: the capital letters (real tokens, not counted)" if "caesar" in T3N else ""))\n'),
# 9. item names (the step-0 site decode)
('    def item_name(d, v):\n        return {',
 '    def item_name(d, v):\n'
 '        if d in TOY3:   # Rev H\n'
 '            return T3N[d][v]\n'
 '        return {'),
# 10. step 0: every toy3 prior usable on the first batch with a non-zero gradient (before any compute is spent)
('            grad_info = dom_grads(lm_loss, ops, on, usable, gl, dgeo)\n',
 '            grad_info = dom_grads(lm_loss, ops, on, usable, gl, dgeo)\n'
 '            if s == 0 and REVT:   # Rev H: every toy3 domain\'s true prior has sites in the first batch and a non-zero gradient\n'
 '                for d in [d for d in EXT if d in TOY3]:\n'
 '                    assert grad_info.get(f"gnorm_true_{d}", 0.0) > 0, (d, "toy3 prior: no usable sites or a zero gradient at step 0")\n'
 '                print("step 0: toy3 true-prior gradient norms " + json.dumps({d: grad_info[f"gnorm_true_{d}"] for d in EXT if d in TOY3}), flush=True)\n'),
# 11. run JSON: the toy3 names and the module shas, only with a toy3 domain
('                res_.update(domains=DINFO, lam_domains=LAMD, lam_rule=args.lam_rule, nonce={"s4": S4N, "dim": DIMN},\n'
 '                            dep_sha=dep_, metrics=dom_metrics(curve))\n',
 '                res_.update(domains=DINFO, lam_domains=LAMD, lam_rule=args.lam_rule, nonce={"s4": S4N, "dim": DIMN},\n'
 '                            dep_sha=dep_, metrics=dom_metrics(curve))\n'
 '                if REVT:   # Rev H: the toy3 item names and modules\n'
 '                    res_["nonce"].update(T3N)\n'
 '                    dep_.update({f_: hashlib.sha256((Path(__file__).parent / f_).read_bytes()).hexdigest()[:12]\n'
 '                                 for f_ in ("toy3.py", "toy3_nonce.json")})\n'),
]
for a, b in R:
    n = s.count(a)
    assert n == 1, (n, a[:80])
    s = s.replace(a, b)
P.write_text(s)
print("patched", len(R), "sites")
