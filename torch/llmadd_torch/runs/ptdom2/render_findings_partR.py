#!/usr/bin/env python3
"""Render a ptdom2 Part R read-out JSON (readout_ptdom2.py --part R --json OUT) as FINDINGS-style markdown (F101 / F102 house style).

Pure formatting, read only. Every scalar printed is read from the JSON, with one exception marked with a dagger (†): the JSON keeps
some descriptive lines only as per-seed lists (final3 gains, shuffled - none, the window split, the same-lr gain on seeds 8-15), and
for those the renderer prints the plain mean and the count of seeds > 0 of that list, the same definition as readout_ptdom2's report
(mn / wn). Nothing else is computed: no bootstrap, CI or test (the shuffled - none boot90, which the read-out computes only at print
time, prints as MISSING). A key the renderer expects but the JSON lacks prints 'MISSING <key path>'; a null value prints '-'.

Usage:  render_findings_partR.py IN.json [-o OUT.md] [--part R|S] [--dp 1]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse      # noqa: E402
import json          # noqa: E402
import math          # noqa: E402
from pathlib import Path   # noqa: E402

ZH = {"countries": "国家", "cities": "城市", "people": "人物", "months": "月份"}   # names used in F100 / F101; others print as keys
MINUS = "−"
DP = 1


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


def _chk(v):
    if isinstance(v, Missing):
        return str(v)
    if v is None:
        return "-"
    if not _num(v):
        return f"NONNUMERIC({v!r})"
    if math.isnan(v):
        return "nan"
    return None


def sg(v, p=None):
    return _chk(v) or f"{v:+.{DP if p is None else p}f}".replace("-", MINUS)


def nf(v, p=None):
    return _chk(v) or f"{v:.{DP if p is None else p}f}".replace("-", MINUS)


def pf(v):
    return _chk(v) or ("< .001" if v < 0.001 else f"{v:.3f}".lstrip("0"))


def ci(lst, p=None):
    if isinstance(lst, Missing):
        return str(lst)
    if lst is None:
        return "-"
    if not isinstance(lst, list) or len(lst) != 2:
        return f"MALFORMED({lst!r})"
    return f"[{sg(lst[0], p)}, {sg(lst[1], p)}]"


def mean_d(lst, p=None):
    """† the plain mean of a JSON per-seed list (readout_ptdom2.mn); Missing / None pass through."""
    if isinstance(lst, Missing):
        return str(lst)
    if lst is None:
        return "-"
    if not isinstance(lst, list) or not lst:
        return f"MALFORMED({lst!r})"
    if any(not _num(v) for v in lst):
        return f"NONNUMERIC-IN-LIST({lst!r})"
    return sg(sum(lst) / len(lst), p) + "†"


def wins_d(lst):
    """† seeds > 0 of a JSON per-seed list (readout_ptdom2.wn)."""
    if isinstance(lst, Missing):
        return str(lst)
    if not isinstance(lst, list) or any(not _num(v) for v in lst):
        return "-"
    return f"{sum(v > 0 for v in lst)}/{len(lst)}†"


def name(d):
    return ZH.get(d, d)


def row(cells):
    """One markdown table row; a '|' inside a cell is escaped so it cannot split the cell."""
    return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"


def n_of(lst):
    return len(lst) if isinstance(lst, list) else "?"


def rule_cell(det, n, bold, full=True, pre=""):
    if isinstance(det, Missing):
        return str(det)
    if det is None:
        return "-（缺 run）"
    m = sg(det.get("mean", Missing(f"{pre}.mean")))
    if bold:
        m = f"**{m}**"
    b = ci(det.get("boot90", Missing(f"{pre}.boot90")))
    wv = det.get("wins", Missing(f"{pre}.wins"))
    w = f"{wv}/{n}"
    hv = det.get("halves", Missing(f"{pre}.halves"))
    h = (f"{sg(hv[0])} / {sg(hv[1])}" if isinstance(hv, list) and len(hv) == 2 else str(hv))
    ph = pf(det.get("p_holm", Missing(f"{pre}.p_holm")))
    return f"{m} {b}，两半 {h}，{w}，Holm p {ph}" if full else f"{m} {b}，{w}，Holm p {ph}"


def render(res, src="", part="R"):
    L = []
    X = get(res, part)
    L.append(f"<!-- render_findings_partR.py from {src} (part {part}); every number is read from that JSON; † = mean / count of a JSON "
             f"per-seed list (readout_ptdom2 mn / wn) -->")
    if not isinstance(X, dict):
        L.append(f"Part {part}：{X if isinstance(X, Missing) else 'null（JSON 里没有这一部分）'}")
        return "\n".join(L) + "\n"
    fam, final = get(X, "family"), get(X, "final")
    L.append(f"**ptdom2 Part {part} 结果**（曲线面积，%；Holm 族 = {fam}（{get(X, 'holm_n')} 个）；final = {final}）：")
    probs = get(X, "problems")
    if final is not True:
        L.append("")
        L.append(f"> **PROVISIONAL**（final = {final}）")
    if isinstance(probs, list) and probs:
        uniq = list(dict.fromkeys(probs))
        L.append(f"> problems（{len(probs)} 条，去重后 {len(uniq)} 条）：" + "；".join(uniq))
    elif isinstance(probs, Missing):
        L.append(f"> {probs}")
    for k in ("consistency", "integrity", "sub_problems"):
        v = get(res, k)
        if v:
            L.append(f"> {k}：{v if isinstance(v, Missing) else '；'.join(map(str, v))}")
    rows = get(X, "rows")
    if not isinstance(rows, dict):
        L.append(str(rows))
        return "\n".join(L) + "\n"

    # ------------------------------------------------------------------ the per-domain table
    L += ["", "| 领域 | 批 | 先验@lr 对 none@lr（面积） | 真 − none：均值 [boot90]，两半，赢，Holm p | 真 − 打乱：均值 [boot90]，赢，Holm p | "
              "final3：真 − none / 真 − 打乱 | 同 lr 增益 | 判词 | Tier 与标签 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d in rows:
        K = lambda *k: get(res, part, "rows", d, *k)   # noqa: E731
        st = K("status")
        if st != "FINAL":
            why = K("why")
            L.append(row([name(d), K("batch") if "batch" in rows[d] else "-", "-", "-", "-", "-", "-",
                          f"{st}：{'；'.join(why) if isinstance(why, list) else why}",
                          ("Tier " + str(K("tier"))) if "tier" in rows[d] else "-"]))
            continue
        P, N = K("pick_prior"), K("pick_none")
        n = n_of(K("gain"))
        verdict, geom = K("verdict"), K("geometry")
        fl = K("floor")
        if fl is None:
            lv = f"only-{d}@{P} 对 none@{N}：-（缺 run）"
        else:
            lv = (f"only-{d}@{P} {nf(K('floor', 'means', f'only-{d}@{P}'))} 对 none@{N} {nf(K('floor', 'means', f'none@{N}'))}"
                  f"（打乱 {nf(K('floor_geo', 'means', f'only-{d}-shuf@{P}')) if K('floor_geo') is not None else '-'}）")
        g_cell = rule_cell(K("rule_gain"), n, isinstance(verdict, str) and verdict.startswith("PASS"), pre=f"{part}.rows.{d}.rule_gain") if "rule_gain" in rows[d] else (
            "-（缺 run）" if rows[d].get("gain") is None else str(K("rule_gain")))
        geo_cell = rule_cell(K("rule_geo"), n, isinstance(geom, str) and geom.startswith("GEOMETRY-SPECIFIC"), full=False, pre=f"{part}.rows.{d}.rule_geo") if "rule_geo" in rows[d] else (
            "-（缺 run）" if rows[d].get("geo_gain") is None else str(K("rule_geo")))
        f3 = f"{mean_d(K('gain_final3'))}（{wins_d(K('gain_final3'))}） / {mean_d(K('geo_final3'))}（{wins_d(K('geo_final3'))}）"
        sp = K("same_lr_pass")
        if isinstance(sp, dict):
            if "missing" in sp:
                same = f"same-lr PASS 行：缺 run（{'；'.join(sp['missing'][:3])}）"
            else:
                same = (f"same-lr PASS 行（{sp.get('label', '-')}）：{sp.get('would_be')}，{sg(sp.get('mean'))} {ci(sp.get('boot90'))}，"
                        f"{sp.get('wins')}/{n}，p（未校正）{pf(sp.get('p_one_sided_raw'))}")
        else:
            s03 = K("same_lr_2e3_seeds0_3")
            same = ("@2e-3（ptdom seed 0–3）[" + ", ".join(sg(v) for v in s03) + "]") if isinstance(s03, list) else (
                f"@2e-3（ptdom seed 0–3）{s03 if isinstance(s03, Missing) else '-'}")
            s815 = K("same_lr_seeds8_15")
            if isinstance(s815, list):
                same += f"；@{P}（seed 8–15）{mean_d([v for v in s815 if v is not None])}"
        vtxt = f"{verdict}；几何：{geom}"
        if isinstance(K("reading"), str):
            vtxt += "；reading 见下"
        tags = [f"Tier {K('tier')}"]
        ft = K("floor", "tags") if fl is not None else None
        if isinstance(ft, list):
            tags += [str(t) for t in ft]
        elif isinstance(ft, Missing):
            tags.append(str(ft))
        fg = K("floor_geo")
        if isinstance(fg, dict) and fg.get("tags"):
            tags += [f"真 vs 打乱：{t}" for t in fg["tags"]]
        notes = K("notes")
        if isinstance(notes, list):
            if any(str(x).startswith("PRIOR AT ITS LOWEST LR") for x in notes):
                tags.append("先验在最低 lr")
            if any("share their site token" in str(x) for x in notes):
                tags.append("共享位点（见注）")
        if K("diverged"):
            tags.append(f"DIVERGED {K('diverged')}")
        if rows[d].get("gate_failed"):
            tags.append("AUX GATE FAILED")
        L.append(row([name(d), K("batch"), lv, g_cell, geo_cell, f3, same, vtxt, "，".join(tags)]))

    # ------------------------------------------------------------------ descriptive lines per FINAL domain
    L += ["", "描述性（不是条件；† = JSON 逐 seed 列表的均值 / 计数）："]
    for d in rows:
        K = lambda *k: get(res, part, "rows", d, *k)   # noqa: E731
        if K("status") != "FINAL":
            continue
        P = K("pick_prior")
        L.append(f"- {name(d)}：")
        sn, lr_sn = K("shuf_none"), K("shuf_none_lr")
        conf = "；lr 混淆：P ≠ N" if K("lr_confounded_shuf_none") is True else ""
        nsm = K("ns_missing")
        if sn is None:
            L.append(f"  - 打乱 − none@{lr_sn}（描述性{conf}）：-（{'缺 / 无效：' + '；'.join(nsm) if nsm else '缺 run'}）")
        else:
            L.append(f"  - 打乱 − none@{lr_sn}（描述性{conf}）：面积 {mean_d(sn, 2)}（{wins_d(sn)}），boot90 MISSING（JSON 不存；读数器只在打印时算，"
                     f"见读数 txt），final3 {mean_d(K('shuf_none_final3'), 2)}")
        spl = K("split")
        if isinstance(spl, dict):
            parts = []
            for k, v in spl.items():
                kz = {"vs none": "对 none", "vs shuffled": "对打乱", "shuffled vs none": "打乱对 none"}.get(k, k)
                if isinstance(v, dict):
                    parts.append(f"{kz} {mean_d(v.get('in', Missing(f'split.{k}.in')), 2)} / {mean_d(v.get('out', Missing(f'split.{k}.out')), 2)}"
                                 f"（窗口内 eval {v.get('n_window_evals', '?')} 次）")
                else:
                    parts.append(f"{kz} -")
            L.append("  - 窗口划分（窗口内 / 窗口外；两者之和 = 面积增益）：" + "；".join(parts))
        else:
            L.append(f"  - 窗口划分：{spl}")
        ls = K("lm_share")
        if isinstance(ls, dict):
            L.append("  - LM 占比 a/(a+b)（cell 中位，seed 8–15）：" + "；".join(f"{k} {nf(v, 3)}" for k, v in ls.items()))
        else:
            L.append(f"  - LM 占比：{ls}")
        gt = K("gate")
        if isinstance(gt, dict):
            L.append("  - aux gate（每 cell 中位 ≥ 0.5）：" + "；".join(
                f"{k} {nf(g.get('median'), 3)}（最低 run {nf(g.get('min'), 3)}，裁剪 < 1 的比例 {nf(g.get('frac_clip_lt1'), 3)}）"
                f"{'' if g.get('ok') else ' FAIL'}" for k, g in gt.items()))
        if rows[d].get("gate_failed"):
            L.append(f"  - AUX GATE FAILED：{K('gate_failed')}；不看门槛时：{K('would_be_without_gate')}")
        L.append(f"  - 难题子集 final3：对 none {mean_d(K('gain_hard3'), 2)}，对打乱 {mean_d(K('geo_hard3'), 2)}")
        for rk, lab in (("rule_gain", "真 − none"), ("rule_geo", "真 − 打乱")):
            r = K(rk)
            if isinstance(r, dict):
                cd = r.get("conditions") or {}
                L.append(f"  - {lab} 规则细项：单侧 t p {pf(r.get('p_one_sided'))}，Holm {pf(r.get('p_holm'))}，双侧 p "
                         f"{pf(get(r, 't_two_sided', 'p_two_sided'))}，t95 {ci(get(r, 't_two_sided', 't95'))}；条件 "
                         + "，".join(f"{k} {'ok' if v else 'NO'}" for k, v in cd.items()))
        fl = K("floor")
        if isinstance(fl, dict):
            L.append(f"  - FLOOR 线 {nf(fl.get('line'))}（" + "，".join(f"{a} {nf(b)}" for a, b in zip(fl.get("line_names") or [], fl.get("lines") or []))
                     + f"）；真 vs none：{fl.get('tags') or '无'}；真 vs 打乱：{(K('floor_geo') or {}).get('tags') or '无'}")
        sc = K("shortcut")
        if isinstance(sc, dict):
            L.append("  - 难题子集捷径基线（描述性）：" + "；".join(
                f"{a} {nf(v.get('prior'))} 对基线 {nf(v.get('base'))}（n {v.get('n_pooled')}）" if isinstance(v, dict) else f"{a} -"
                for a, v in sc.items()))
        sct = K("shuf_cka_to_true")
        if isinstance(sct, list) and len(sct) == 2:
            L.append(f"  - 打乱目标对真目标的 CKA（8 seed 范围）：{nf(sct[0], 3)} .. {nf(sct[1], 3)}")
        ck = K("cka_check")
        if isinstance(ck, dict):
            L.append(f"  - 双路径检查（A2.4，描述性）：{ck.get('n_ok')}/{ck.get('n')} seed 满足")
        if isinstance(K("would_be"), list):
            L.append(f"  - final 时会是：{K('would_be')}（{K('would_be_holm')}）")
        if isinstance(K("reading"), str):
            L.append("  - reading（读数器原文）：" + K("reading"))
        rf = rows[d].get("reading_failed")
        if rf:
            L.append("  - reading 没印出的原因：" + "；".join(rf))
        if isinstance(notes := K("notes"), list) and notes:
            L.append("  - 注：" + "；".join(notes))
    return "\n".join(L) + "\n"


def main():
    global DP
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("json")
    ap.add_argument("-o", "--out")
    ap.add_argument("--part", default="R", choices=("R", "S"))
    ap.add_argument("--dp", type=int, default=1, help="decimals for areas / gains in the table (default 1)")
    a = ap.parse_args()
    DP = a.dp
    res = json.loads(Path(a.json).read_text())
    md = render(res, a.json, a.part)
    if a.out:
        Path(a.out).write_text(md)
    print(md, end="")


if __name__ == "__main__":
    main()
