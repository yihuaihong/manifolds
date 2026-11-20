#!/usr/bin/env python3
"""Render a pt6v2 read-out JSON (readout_pt6v2.py --json OUT) as FINDINGS-style markdown (the F102 layout).

Pure formatting, read only: every number printed is a value read from the JSON (no means, CIs or tests are computed here; the only
arithmetic is counting list entries / seeds for the 'k of 6' lines and grouping seeds by device). A key the renderer expects but the
JSON lacks prints 'MISSING <key path>'; a null value prints '-'.

Usage:  render_findings_pt6v2.py IN.json [-o OUT.md]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse      # noqa: E402
import json          # noqa: E402
import math          # noqa: E402
from pathlib import Path   # noqa: E402

# Chinese names used in FINDINGS (F100-F102); any other key prints as is.
ZH = {"numbers": "数字", "s4": "S4（s4_hard）", "dim": "量纲格", "cities": "城市", "countries": "国家", "people": "人物", "months": "月份"}
MINUS = "−"


class Missing:
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return f"MISSING {self.path}"

    __repr__ = __str__


def _path(keys):
    s = ""
    for k in keys:
        s += f"[{k}]" if isinstance(k, int) else (("." if s else "") + str(k))
    return s


def get(d, *keys):
    """d[k1][k2]... or Missing('k1.k2...') when a key / index is absent (a present null stays None)."""
    cur = d
    for i, k in enumerate(keys):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and isinstance(k, int) and -len(cur) <= k < len(cur):
            cur = cur[k]
        else:
            return Missing(_path(keys[:i + 1]))
    return cur


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def sg(v, p=1):
    """Signed number, U+2212 for minus (FINDINGS style); Missing -> 'MISSING ...'; None -> '-'."""
    if isinstance(v, Missing):
        return str(v)
    if v is None:
        return "-"
    if not _num(v):
        return f"NONNUMERIC({v!r})"
    if math.isnan(v):
        return "nan"
    return f"{v:+.{p}f}".replace("-", MINUS)


def nf(v, p=1):
    if isinstance(v, Missing):
        return str(v)
    if v is None:
        return "-"
    if not _num(v):
        return f"NONNUMERIC({v!r})"
    return f"{v:.{p}f}".replace("-", MINUS)


def pf(v):
    """p value in FINDINGS style: '.006', '< .001'."""
    if isinstance(v, Missing):
        return str(v)
    if v is None:
        return "-"
    if not _num(v):
        return f"NONNUMERIC({v!r})"
    return "< .001" if v < 0.001 else f"{v:.3f}".lstrip("0")


def ci(lst, p=1):
    if isinstance(lst, Missing):
        return str(lst)
    if lst is None:
        return "-"
    if not isinstance(lst, list) or len(lst) != 2:
        return f"MALFORMED({lst!r})"
    return f"[{sg(lst[0], p)}, {sg(lst[1], p)}]"


def name(m):
    return ZH.get(m, m)


def int_zh(w):
    if isinstance(w, Missing) or w is None:
        return str(w) if isinstance(w, Missing) else "-"
    if w.startswith("interference not detected"):
        return "没测出干扰"
    if w == "INTERFERENCE DETECTED":
        return "测出干扰"
    if w.startswith("interference WITHHELD"):
        return "干扰判定暂缓（同 lr 对不全）"
    if w.startswith("AUX GATE FAILED"):
        return f"干扰判定暂缓（{w}）"
    return w


def render_v3(x, mani):
    """AUDIT.md Amendment V3 (post-hoc, user-approved 2026-11-19): the compact extended-grid table (readout_pt6v2.py --posthoc_none,
    JSON key posthoc_V3), printed after the registered table; read only, like the rest."""
    ss = get(x, "res", "seeds")
    n = len(ss) if isinstance(ss, list) else ss
    pv = get(x, "provisional")
    L = ["", f"**扩展网格（post-hoc V3，用户 2026-11-19 批准；登记表不变，扩展判词并列报）**：none 网格 {get(x, 'grid')}，读到的 seed {ss}"]
    if pv:
        L.append("> **扩展 PROVISIONAL**：" + ("；".join(str(y) for y in pv) if isinstance(pv, list) else str(pv)))
    L += ["", "| 流形 | N_m（登记 → 扩展） | g（登记 → 扩展） | 扩展 g 90% boot，赢 | p（Holm） | final3 的 g | 判词（登记 → 扩展） | 变化 |",
          "|---|---|---|---|---|---|---|---|"]
    for m in mani:
        K = lambda *k: get(x, "res", "m", m, *k)   # noqa: E731
        C = lambda k: get(x, "change", m, k)      # noqa: E731
        chg = [w for w, k in (("g 变号", "sign_flip"), ("失去 HELPS", "lost_helps"), ("新得 HELPS", "gained_helps")) if C(k) is True]
        L.append(f"| {name(m)} | {C('N_reg')} → {C('N_ext')} | {sg(C('g_reg'))} → {sg(C('g_ext'))}（{sg(C('dg'))}） | "
                 f"{ci(K('boot90_g'))}，{K('wins_g')}/{n} | {pf(K('p_help'))}（{pf(K('p_help_holm'))}） | "
                 f"{sg(get(x, 'res', 'm', m, 'final3', 'g'))}（{get(x, 'res', 'm', m, 'final3', 'wins_g')}/{n}） | "
                 f"{C('word_reg')} → {C('word_ext')} | {'、'.join(chg) + '：两个都报，论文措辞跟扩展' if chg else '无'} |")
    L.append(f"\n- 扩展 HEADLINE 原文：{get(x, 'res', 'headline', 'text')}")
    L.append(f"- 读法：{get(x, 'rule')}")
    return L


def render(res, src=""):
    L = []
    mani = get(res, "mani")
    P, same = get(res, "P"), get(res, "same")
    seeds, wanted = get(res, "seeds"), get(res, "seeds_wanted")
    n_read = len(seeds) if isinstance(seeds, list) else seeds
    L.append(f"<!-- render_findings_pt6v2.py from {src}; every number is read from that JSON -->")
    L.append(f"**pt6v2 结果**（曲线面积，%；读到的 seed {seeds}，共登记 {wanted}；all6@{P} = P；同 lr {same}；"
             f"none 网格 {get(res, 'none_lrs')}）：")
    prov = get(res, "provisional")
    if isinstance(prov, list) and prov:
        L.append("")
        L.append("> **PROVISIONAL**（--allow_partial）：" + "；".join(str(x) for x in prov))
    elif isinstance(prov, Missing):
        L.append("")
        L.append(f"> {prov}（JSON 没有 provisional 字段：可能是 strict 读数前就停了）")
    if get(res, "missing"):
        L.append(f"> 缺 / 无效的核心 run：{get(res, 'missing')}")
    if res.get("missing_pair"):
        L.append(f"> 同 lr 对不全（干扰判定暂缓）：{res['missing_pair']}")
    M = get(res, "m")
    if res.get("empty") or not isinstance(M, dict) or not M or not isinstance(mani, list):
        L.append("")
        L.append(f"没有逐流形判词：{'empty（完整 seed 少于 2 个）' if res.get('empty') else (str(M) if isinstance(M, Missing) else str(mani) if isinstance(mani, Missing) else 'm 为空')}")
        return "\n".join(L) + "\n"
    gate_failed = get(res, "gate_failed")

    # ------------------------------------------------------------------ 1. the result table (F102 layout, N_m per manifold)
    L += ["", f"| 流形 | all6@{P} | none@N_m | N_m | g [95% CI]，赢 | final3 的 g | 同 lr：h [95% CI] | 同 lr：u | 判词 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in mani:
        v = get(M, m)
        if isinstance(v, Missing):
            L.append(f"| {name(m)} | MISSING m.{m} | | | | | | | |")
            continue
        K = lambda *k: get(res, "m", m, *k)   # noqa: E731
        word = K("word")
        helps = K("helps")
        gtxt = sg(K("mean_g"))
        if helps is True and gate_failed is not True:
            gtxt = f"**{gtxt}**"
        wins = f"{K('wins_g')}/{n_read}"
        f3 = K("final3")
        f3txt = sg(get(res, "m", m, "final3", "g")) + (f"（{get(res, 'm', m, 'final3', 'wins_g')}/{n_read}）" if isinstance(f3, dict) else "")
        verdict = []
        if word == "NOT SHOWN":   # V1 item 2: the same-lr gain at 2e-3 and u first
            gs = K("g_same_lr")
            if isinstance(gs, dict):
                verdict.append(f"同 lr {same} 增益 {sg(gs.get('mean', Missing(f'm.{m}.g_same_lr.mean')))}"
                               f"（{gs.get('wins', Missing(f'm.{m}.g_same_lr.wins'))}/{gs.get('n', Missing(f'm.{m}.g_same_lr.n'))}，"
                               f"p {pf(gs.get('p', Missing(f'm.{m}.g_same_lr.p')))}），u {sg(K('mean_u'))}；")
            else:
                verdict.append(f"同 lr {same} 增益 {'-（同 lr 对不全）' if gs is None else gs}，u {sg(K('mean_u'))}；")
        if gate_failed is True:
            verdict.append(f"GATE FAILED：不给判词（规则读数 {word}）")
        else:
            verdict.append(f"{word}")
        verdict.append(int_zh(K("int_word")))
        tags = K("tags")
        if isinstance(tags, list):
            verdict += [str(t) for t in tags]
        else:
            verdict.append(str(tags))
        lev = lambda k: get(res, "m", m, "levels", k)   # noqa: E731
        L.append(f"| {name(m)} | {nf(lev('all6@P'))} | {nf(lev('none@N'))} | {K('N')} | {gtxt} {ci(K('ci95_g'))}，{wins} | {f3txt} | "
                 f"{(sg(K('mean_h')) + ' ' + ci(K('ci95_h'))) if K('mean_h') is not None else '-'} | {sg(K('mean_u'))} | {'，'.join(verdict).replace('；，', '；')} |")

    if "posthoc_V3" in res:   # Amendment V3: the extended table right after the registered one (absent: output unchanged)
        L += render_v3(res["posthoc_V3"], mani)

    # ------------------------------------------------------------------ 2. verdict counts
    helps_l = [m for m in mani if get(res, "m", m, "helps") is True]
    noint_l = [m for m in mani if get(res, "m", m, "noint") is True]
    hl = get(res, "headline")
    L += ["", f"- 判词计数（{len(mani)} 个流形）："]
    L.append(f"  - HELPS（all6@{P} 对各自的 none@N_m）：{len(helps_l)} / {len(mani)}（{'、'.join(name(m) for m in helps_l) or '无'}）"
             + ("；**all6 GATE FAILED，不给 HELPS 判词**" if gate_failed is True else ""))
    L.append(f"  - 同 lr {same} 没测出干扰：{len(noint_l)} / {len(mani)}（{'、'.join(name(m) for m in noint_l) or '无'}）")
    L.append(f"  - HELPS 且没测出干扰（读数器 headline k）：{get(res, 'headline', 'k')} / {len(mani)}"
             f"（{'、'.join(name(m) for m in (get(res, 'headline', 'list') or [])) if isinstance(get(res, 'headline', 'list'), list) else get(res, 'headline', 'list')}）")
    L.append(f"  - 读数器 HEADLINE 原文：{get(res, 'headline', 'text') if isinstance(hl, dict) else hl}")
    L.append("  - Holm p（参考列）：" + "；".join(f"{name(m)} {pf(get(res, 'm', m, 'p_help_holm'))}" for m in mani)
             + "；未校正单侧 p：" + "；".join(f"{name(m)} {pf(get(res, 'm', m, 'p_help'))}" for m in mani))
    mg = get(res, "mean6_gain")
    L.append(f"- 六项平均增益 {sg(get(res, 'mean6_gain', 'mean'))} [95% CI {sg(get(res, 'mean6_gain', 'ci95', 0))}, "
             f"{sg(get(res, 'mean6_gain', 'ci95', 1))}]（描述性）" if not isinstance(mg, Missing) else f"- 六项平均增益：{mg}")
    L.append("- 先验窗口内 / 窗口外的 g（窗口外 = 先验关闭后）：" + "；".join(
        f"{name(m)} {sg(get(res, 'm', m, 'window_g', 'in'))} / {sg(get(res, 'm', m, 'window_g', 'out'))}"
        if get(res, "m", m, "window_g") is not None else f"{name(m)} -（无逐 eval 曲线）" for m in mani))
    L.append("- 干扰界：h 的 95% CI 下界 对 预登记界 −0.5u（margin），判定依据：" + "；".join(
        f"{name(m)} {sg(get(res, 'm', m, 'ci95_h', 0))} 对 {sg(get(res, 'm', m, 'margin'))}（{get(res, 'm', m, 'noint_by') or '-'}）"
        if get(res, "m", m, "ci95_h") is not None else f"{name(m)} -" for m in mani))

    # NOT SHOWN: the same-lr gain at 2e-3 and u first (V1 item 2), then the descriptive g vs every none lr
    for m in mani:
        if get(res, "m", m, "word") != "NOT SHOWN":
            continue
        gs = get(res, "m", m, "g_same_lr")
        lv = lambda k: get(res, "m", m, "levels", k)   # noqa: E731
        if isinstance(gs, dict):
            first = (f"同 lr {same} 下 all6 {nf(lv('all6@same'))} 对 none {nf(get(res, 'm', m, 'none_means', same))}"
                     f"（{sg(get(res, 'm', m, 'g_same_lr', 'mean'))}，{get(res, 'm', m, 'g_same_lr', 'wins')}/{get(res, 'm', m, 'g_same_lr', 'n')}，"
                     f"p {pf(get(res, 'm', m, 'g_same_lr', 'p'))}，描述性），u {sg(get(res, 'm', m, 'mean_u'))}")
        else:
            first = f"同 lr {same} 增益 -（同 lr 对不全），u {sg(get(res, 'm', m, 'mean_u'))}"
        L.append(f"- {name(m)}（NOT SHOWN）：{first}；对选过 lr 的 none@{get(res, 'm', m, 'N')}：g {sg(get(res, 'm', m, 'mean_g'))}，"
                 f"{get(res, 'm', m, 'wins_g')}/{n_read}，p {pf(get(res, 'm', m, 'p_help'))}。")

    # descriptive g vs every none lr
    NL = get(res, "none_lrs")
    if isinstance(NL, list):
        L += ["", f"g 对每个 none lr（描述性；均值，赢 / n）与对 mean6 选的 none@{get(res, 'N6')}：", "",
              "| 流形 | " + " | ".join(f"none@{lr}" for lr in NL) + " | mean6 的 none | none 各 lr 均值 |",
              "|---|" + "---|" * (len(NL) + 2)]
        # I2 (review fix 3): an Amendment V2 lr that left the N_m grid has no none_means entry: one line below the table, not
        # MISSING per manifold; a vs_lr entry stored as null (< 2 valid seeds) prints n/a with its count. Absent keys stay MISSING.
        grid = res.get("none_grid") if isinstance(res.get("none_grid"), list) else NL
        xn = res.get("extra_none") if isinstance(res.get("extra_none"), dict) else {}
        for m in mani:
            cells = []
            for lr in NL:
                d = get(res, "m", m, "vs_lr", lr)
                cells.append(f"n/a ({(xn.get(lr) or {}).get('n_valid', '< 2')} valid seeds)" if d is None else
                             f"{sg(d.get('mean', Missing(f'm.{m}.vs_lr.{lr}.mean')))}，{d.get('wins', '?')}/{d.get('n', '?')}"
                             + ("（N_m）" if lr == get(res, "m", m, "N") else "") if isinstance(d, dict) else str(d))
            vm = get(res, "m", m, "vs_mean6")
            c6 = (f"{sg(vm.get('mean', Missing(f'm.{m}.vs_mean6.mean')))}，{vm.get('wins', '?')}/{vm.get('n', '?')}，"
                  f"{'HELPS' if vm.get('helps') else 'NOT SHOWN'}（描述性）") if isinstance(vm, dict) else str(vm)
            nm = get(res, "m", m, "none_means")
            cm = "，".join(f"{lr} {nf(get(res, 'm', m, 'none_means', lr))}" for lr in NL if lr in grid) if isinstance(nm, dict) else str(nm)
            L.append(f"| {name(m)} | " + " | ".join(cells) + f" | {c6} | {cm} |")
        for lr in NL:
            if lr not in grid:
                L.append(f"\n{lr} not in the N_m grid (valid at <= half the seeds: {(xn.get(lr) or {}).get('n_valid', '?')} of "
                         f"{len(get(res, 'seeds')) if isinstance(get(res, 'seeds'), list) else '?'})")

    # ------------------------------------------------------------------ 3. flags
    L += ["", "**标记**（只标记，不是条件）："]
    for m in mani:
        tags, flags = get(res, "m", m, "tags"), get(res, "m", m, "flags")
        L.append(f"- {name(m)}：FLOOR / CEILING：{('、'.join(tags) or '无') if isinstance(tags, list) else tags}"
                 f"（FLOOR 线 {nf(get(res, 'm', m, 'floor_line'))}）；lr 标记：{('、'.join(flags) or '无') if isinstance(flags, list) else flags}")
    L.append(f"- 设备：{get(res, 'device_note')}")

    # ------------------------------------------------------------------ 4. devices per seed
    devs = get(res, "devices")
    if isinstance(devs, dict):
        by = {}
        for s, dl in devs.items():
            for x in (dl if isinstance(dl, list) else [dl]):
                by.setdefault(str(x), []).append(str(s))
        L.append("- 每个设备上的 seed：" + "；".join(f"{d} {', '.join(ss)}" for d, ss in sorted(by.items()))
                 + "（逐 seed：" + "；".join(f"s{s} {'/'.join(dl) if isinstance(dl, list) else dl}" for s, dl in devs.items()) + "）")
    else:
        L.append(f"- 每个设备上的 seed：{devs}")

    # ------------------------------------------------------------------ 5. LM share
    A = get(res, "arms")
    pk, sk = f"all6@{P}", f"all6@{same}"
    L.append(f"- 辅助损失：{pk} 的 LM 梯度占比（范数 a/(a+b)，cell 中位）{nf(get(res, 'arms', pk, 'lm_share'), 2)}；"
             "GATE 每半：" + ("；".join(f"{k} {nf(g.get('median'), 3)}（{'ok' if g.get('ok') else 'FAILED'}）"
                                     for k, g in get(res, "gate_all6").items()) if isinstance(get(res, "gate_all6"), dict) else str(get(res, "gate_all6")))
             + f"；{sk} {nf(get(res, 'arms', sk, 'lm_share'), 2)}，每半："
             + ("；".join(f"{k} {nf(g.get('median'), 3)}（{'ok' if g.get('ok') else 'FAILED'}）" for k, g in res["gate_all6_same_lr"].items())
                if isinstance(res.get("gate_all6_same_lr"), dict) else "-（同 lr 对不全）"))
    if isinstance(A, dict):
        L.append("  - only-m@" + f"{same} 的 LM 占比中位：" + "；".join(
            f"{name(m)} {nf(get(res, 'arms', f'only-{m}@{same}', 'lm_share'), 2)}" for m in mani))
        L.append(f"  - LM（训练结束的 val loss / bpb，seed 均值）：" + "；".join(
            f"{k} {nf(v.get('val_loss'), 4)} / {nf(v.get('bpb'), 4)}" for k, v in A.items() if isinstance(v, dict)))
    L.append("  - 最低 run 的占比、裁剪系数：JSON 里没有（MISSING min lm_share / clip_coef；读数器只存 cell 中位）")
    L.append("- Tier L：JSON 里没有逐流形字段（读数器 reading 行：比较类与 s4 / dim 是 Tier L）；两条限定行（N_m 与 g 用同一批 seed 选、"
             "reading 行）也只在读数 txt 里。")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("json")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    res = json.loads(Path(a.json).read_text())
    md = render(res, a.json)
    if a.out:
        Path(a.out).write_text(md)
    print(md, end="")


if __name__ == "__main__":
    main()
