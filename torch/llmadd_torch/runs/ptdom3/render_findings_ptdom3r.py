"""Render a readout_ptdom3r.py JSON as a FINDINGS draft (F101 / F103 layout: heading, 设置, 结果 + table, 限定, 审查状态).

Every number is read from the JSON: an absent key or a value of the wrong type prints `MISSING <key path>` (never a TypeError), a
null prints '—'. No number is computed here except formatting (rounding, sign) and two level differences (final3 - in-window, per arm,
from the JSON's window.level_means) used only to word the window qualifier; that qualifier's wording is derived from the signs of the
JSON fields it names. Exploratory tier: the words used are the JSON's labels ('preliminary (1 seed)', 'replicated at k seeds',
'supports') and E1-E5.

    python runs/ptdom3/render_findings_ptdom3r.py <readout_rep_*.json> [--out X.md] [--fid F104]
    python runs/ptdom3/render_findings_ptdom3r.py --selftest [<readout_rep_*.json>]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
import tempfile
from pathlib import Path

REP = ("torus", "hcube", "box5")
SCREEN_ONLY = ("cyc12", "d6", "caesar")
E_KEYS = ("E1", "E2", "E3", "E4", "E5")
WIN_TXT = "step 916–7323"
_MISS = object()


def isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class Doc:
    def __init__(self, obj):
        self.o, self.missing = obj, []

    def raw(self, path):
        """dotted path; a dict key may itself contain dots (file names: the longest matching key wins); a digit indexes a list."""
        cur, parts, i = self.o, path.split("."), 0
        while i < len(parts):
            for j in range(len(parts), i, -1):
                k = ".".join(parts[i:j])
                if isinstance(cur, dict) and k in cur:
                    cur, i = cur[k], j
                    break
                if isinstance(cur, list) and j == i + 1 and k.isdigit() and int(k) < len(cur):
                    cur, i = cur[int(k)], j
                    break
            else:
                self.missing.append(path)
                return _MISS
        return cur

    def odd(self, path):
        self.missing.append(path)
        return f"MISSING {path}"

    def __call__(self, path, f="{}"):
        """formatted value; a number format on a non-number (or any format error) prints MISSING <path> (odd type)."""
        v = self.raw(path)
        if v is _MISS:
            return f"MISSING {path}"
        if v is None:
            return "—"
        if isinstance(v, str) and v in ("inf", "-inf"):
            return "∞" if v == "inf" else "−∞"
        if f != "{}" and not isnum(v):
            return self.odd(path)
        try:
            return f.format(v)
        except (ValueError, TypeError, IndexError, KeyError):
            return self.odd(path)

    def num(self, path):
        """a finite number or None (None / absent / odd -> None; absent and odd are recorded as MISSING)."""
        v = self.raw(path)
        if v is _MISS or v is None:
            return None
        if not isnum(v):
            self.missing.append(path)
            return None
        return v

    def dict(self, path):
        v = self.raw(path)
        if v is _MISS or v is None:
            return {}
        if not isinstance(v, dict):
            self.missing.append(path)
            return {}
        return v

    def strs(self, path):
        """a list rendered as strings ([] for absent / null; a non-list is recorded as MISSING)."""
        v = self.raw(path)
        if v is _MISS or v is None:
            return []
        if not isinstance(v, list):
            self.missing.append(path)
            return [f"MISSING {path}"]
        return [str(x) for x in v]


def seeds_of(D, path):
    return sorted(D.dict(path), key=lambda s: (not str(s).isdigit(), int(s) if str(s).isdigit() else 0, str(s)))


def per_seed(D, path, f="{:+.1f}"):
    return ", ".join(f"s{s} {D(f'{path}.{s}', f)}" for s in seeds_of(D, path)) or "—"


def sgn(x):
    return (x > 0) - (x < 0)


def side(x):
    return "prior 臂更高" if x > 0 else ("none 更高" if x < 0 else "持平")


def arm_move(x):
    return f"升 {x:.1f}" if x > 0 else (f"降 {-x:.1f}" if x < 0 else "不变")


def window_qualifier(D, d):
    """-> the window sentence for domain d, or None when the computed signs of in-window g, out-window g, final3 g and area g agree
    (or there are < 2 of them). Each clause is worded from the sign of the field it prints."""
    p = f"domains.{d}.primary"
    w = f"{p}.window"
    vals = [("prior window 内", D.num(f"{w}.in.mean_g"), f"{w}.in"), ("window 外", D.num(f"{w}.out.mean_g"), f"{w}.out"),
            ("最后 3 次 eval", D.num(f"{p}.final3.mean_g"), f"{p}.final3"), ("area（全部 75 次 eval）", D.num(f"{p}.mean_g"), p)]
    have = [(lab, v, src) for lab, v, src in vals if v is not None]
    signs = {sgn(v) for _, v, _ in have} - {0}
    if len(have) < 2 or len(signs) < 2:
        return None
    parts = [f"{lab} {v:+.1f}（{side(v)}，wins {D(f'{src}.wins')}/{D(f'{src}.k')}）" for lab, v, src in have]
    txt = f"{d}：g 的符号不一致（字段 {w}.in / .out.mean_g、{p}.final3.mean_g、{p}.mean_g）：" + "；".join(parts) + "。"
    lm = f"{w}.level_means"
    pi, pf, ni, nf = (D.num(f"{lm}.prior.in"), D.num(f"{lm}.prior.final3"), D.num(f"{lm}.none.in"), D.num(f"{lm}.none.final3"))
    if None not in (pi, pf, ni, nf):
        dp, dn = pf - pi, nf - ni
        txt += (f"水平（各 seed 均值，window 内 → final3）：only-{d}@{D(f'{p}.p')} {pi:.1f} → {pf:.1f}（{arm_move(dp)}），"
                f"none@{D(f'{p}.N')} {ni:.1f} → {nf:.1f}（{arm_move(dn)}）。")
        if sgn(dp) != sgn(dn):
            big = "prior 臂" if abs(dp) > abs(dn) else f"none@{D(f'{p}.N')}"
            txt += (f"即 window 之后 prior 臂{arm_move(dp)[0] if dp else '不变'}、none@{D(f'{p}.N')} {arm_move(dn)[0] if dn else '不变'}，"
                    f"变化主要来自 {big}（|Δ| {max(abs(dp), abs(dn)):.1f} vs {min(abs(dp), abs(dn)):.1f}）。")
        elif dp != dn:
            more = "prior 臂" if abs(dp) > abs(dn) else f"none@{D(f'{p}.N')}"
            txt += f"即两臂都{arm_move(dp)[0]}，{more}{arm_move(dp)[0]}得更多。"
    return txt


def render(obj, fid="F104"):
    D = Doc(obj)
    utc = D("inputs.utc")
    date = utc[:10] if not utc.startswith("MISSING") else utc
    complete = D.raw("complete")
    prov = complete is not True
    probs = D.strs("problems")
    L = []
    # headline: the replicated domains labelled 'supports', with their mean g (from the JSON)
    sup = [d for d in REP if D.raw(f"domains.{d}.primary.grade.supports") is True]
    if sup:
        head = "toy3 复现：" + "、".join(f"{d} 的 prior 增益 {D(f'domains.{d}.primary.mean_g', '{:+.1f}')} pp（area，{D(f'domains.{d}.primary.grade.label')}，supports）"
                                       for d in sup)
    else:
        head = "toy3 复现：没有领域达到 supports（见表）"
    L.append(f"## [{date}] {fid}{'（PROVISIONAL 草稿：输入不全）' if prov else ''} {head}")
    L.append("")
    if prov:
        L.append(f"> **PROVISIONAL**：`complete` = {D('complete')}；缺 {len(probs)} 项（见限定 1）。下面的数是部分读数，不是终读。")
        L.append("")
    # 设置
    L.append("**设置**")
    L.append(f"- ptdom3 toy3（53M、75 次 eval、prior window 为 {WIN_TXT}），seed 0–2。复现的 prior 格子："
             + "、".join(f"only-{d}@{D(f'domains.{d}.rep_lr')}" for d in REP) + "；none 网格：screen 的 2e-3…1.6e-2（s0）、1.6e-2（s0–2）和 3.2e-2（s0–2）。")
    L.append("- 主指标（PRIMARY）：" + "；".join(f"{d} {D(f'domains.{d}.primary.metric')}" for d in REP)
             + "。hcube / d6 用 twin-free 子集，即 read-out 题目中三元组不在任何训练 fact 里的那部分。")
    L.append("- 增益 g = only-d@P − none@N，按 seed 配对。P 和 N 都按 pick_pt6v2 的规则选：在证据 seed 上，"
             "一个 lr 的 seed 覆盖数超过同臂最好覆盖数的一半才算 present；然后取均值最高的 lr，打平取较大的 lr。"
             f"Cohen d = mean / sd（ddof 1）。boot90：{D('boot.method')}（B {D('boot.B')}）。")
    L.append(f"- 判级：{D('grading')}。判定词只用 E1–E5（pt6v2 pick rule）和标签 preliminary (1 seed) / replicated at k seeds。")
    L.append(f"- 输入：readout_ptdom3r.py {D('inputs.readout_ptdom3r.py')}、readout_ptdom3.py {D('inputs.readout_ptdom3.py')}、"
             f"pick_pt6v2.py {D('inputs.pick_pt6v2.py')}；读于 {utc}；共 {D('inputs.n_runs')} 个 run。")
    L.append("")
    # 结果
    L.append("**结果**（单位：百分点。area = read-out 准确率曲线在全部 75 次 eval 上的均值 ×100，不只是 prior window；"
             f"final3 = 最后 3 次 eval 的均值；窗口拆分：window 内 = prior window（{WIN_TXT}）内各次 eval 的均值，"
             "window 外 = 其余各次 eval 的均值（含 step < 916 的早期 eval），与 readout_ptdom3 的 in / out 同一定义）")
    L.append("")
    L.append("| 领域 | 指标 | P / N | 每 seed g（area） | 均值 g [boot90] | wins | d | final3 g（均值） | 标签 | E1–E5 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in REP + SCREEN_ONLY:
        p = f"domains.{d}.primary"
        b = D.raw(f"{p}.boot90")
        btxt = f" [{b[0]:+.1f}, {b[1]:+.1f}]" if isinstance(b, list) and len(b) == 2 and all(isnum(x) for x in b) else (
            "" if b is _MISS or b is None or b == [] else f" [{D.odd(p + '.boot90')}]")
        f3s = seeds_of(D, f"{p}.final3.g")
        f3 = f"{per_seed(D, f'{p}.final3.g')}（{D(f'{p}.final3.mean_g', '{:+.1f}')}）" if f3s else "—"
        lab = D(f"{p}.grade.label")
        if D.raw(f"{p}.grade.supports") is True:
            lab += f"，supports（{D(f'{p}.grade.direction')}）"
        tags = D.strs(f"{p}.tags")
        if tags:
            lab += "；" + "；".join(t.split(" (")[0] for t in tags)
        E = " ".join(f"{e}{'✓' if D.raw(f'{p}.E.{e}') is True else '✗'}" for e in E_KEYS)
        name = d + ("（只有 s0）" if d in SCREEN_ONLY else "")
        L.append(f"| {name} | {D(f'{p}.metric')} | {D(f'{p}.p')} / {D(f'{p}.N')} | {per_seed(D, f'{p}.g')} | {D(f'{p}.mean_g', '{:+.1f}')}{btxt} | "
                 f"{D(f'{p}.wins')}/{D(f'{p}.k')} | {D(f'{p}.d', '{:.2f}')} | {f3} | {lab} | {E} |")
    L.append("")
    # window split (computed by readout_ptdom3r.window_split; same in / out definition as readout_ptdom3)
    L.append(f"窗口拆分（同一组 P / N 格子、同一主指标；window 内 = {WIN_TXT}）：")
    L.append("")
    L.append("| 领域 | window 内 g（均值，wins） | window 外 g（均值，wins） | final3 g（均值，wins） | prior 臂水平 内 / 外 / final3 | none 臂水平 内 / 外 / final3 |")
    L.append("|---|---|---|---|---|---|")
    for d in REP:
        p = f"domains.{d}.primary"
        w = f"{p}.window"
        if D.raw(w) is None:
            L.append(f"| {d} | — | — | — | — | — |")
            continue
        cols = []
        for src in (f"{w}.in", f"{w}.out", f"{p}.final3"):
            cols.append(f"{per_seed(D, src + '.g')}（{D(src + '.mean_g', '{:+.1f}')}，{D(src + '.wins')}/{D(src + '.k')}）")
        for arm in ("prior", "none"):
            lm = f"{w}.level_means.{arm}"
            cols.append(f"{D(lm + '.in', '{:.1f}')} / {D(lm + '.out', '{:.1f}')} / {D(lm + '.final3', '{:.1f}')}")
        L.append(f"| {d} | " + " | ".join(cols) + " |")
    L.append("")
    for d in REP:
        p = f"domains.{d}.primary"
        cells = D.dict(f"domains.{d}._cells")
        nc = D(f"{p}.N")
        pk, nk = f"only-{d}@{D(f'{p}.p')}", f"none@{nc}"
        L.append(f"- **{d}**（{D(f'{p}.metric')}，chance {D(f'{p}.chance', '{:.1f}')}）："
                 f"{pk} 各 seed area " + (per_seed(D, f"domains.{d}._cells.{pk}", "{:.1f}") if pk in cells else "—")
                 + f"；{nk} 各 seed area " + (per_seed(D, f"domains.{d}._cells.{nk}", "{:.1f}") if nk in cells else "—")
                 + f"。配对均值 prior {D(f'{p}.pn_means_paired.0', '{:.1f}')} / none {D(f'{p}.pn_means_paired.1', '{:.1f}')}。"
                 + f"判级依据：{D(f'{p}.grade.why')}。")
        lv = f"{p}.window.levels"
        if D.raw(lv) not in (_MISS, None):
            for arm, nm in (("prior", pk), ("none", nk)):
                L.append(f"  - {nm} 各 seed 水平（window 内 / window 外 / final3）："
                         + (", ".join(f"s{s} {D(f'{lv}.{arm}.{s}.in', '{:.1f}')} / {D(f'{lv}.{arm}.{s}.out', '{:.1f}')} / {D(f'{lv}.{arm}.{s}.final3', '{:.1f}')}"
                                      for s in seeds_of(D, f"{lv}.{arm}")) or "—") + "。")
        sl = D.dict(f"{p}.same_lr")
        if sl:
            L.append(f"  - 同 lr 增益（描述性，只有 s0 有 none 同 lr）：" + "；".join(
                f"{lr}: {per_seed(D, f'{p}.same_lr.{lr}.g')}" for lr in sl) + "。")
        sc = f"domains.{d}.secondary"
        L.append(f"  - 次指标 {D(f'{sc}.metric')}（同一组格子）：g {per_seed(D, f'{sc}.g')}"
                 + f"，均值 {D(f'{sc}.mean_g', '{:+.1f}')}，wins {D(f'{sc}.wins')}/{D(f'{sc}.k')}，d {D(f'{sc}.d', '{:.2f}')}；final3 均值 {D(f'{sc}.final3_mean_g', '{:+.1f}')}。")
        L.append(f"  - LM 梯度占比（only-{d}@{D(f'{p}.p')}，prior 开启期间的中位数）："
                 + (", ".join(f"s{s} {D(f'domains.{d}.lm.{s}.lm_share', '{:.2f}')}" for s in seeds_of(D, f"domains.{d}.lm")) or "—")
                 + (f"；LM<1/2 的 seed：{D(f'domains.{d}.lm_flag')}" if D.raw(f"domains.{d}.lm_flag") not in (_MISS, None, [], False) else "") + "。")
        L.append(f"  - read-out 规模（twin-free / twin / 合计）：" + (", ".join(
            f"s{s} {D(f'domains.{d}.seed_info.{s}.n_tf')} / {D(f'domains.{d}.seed_info.{s}.n_twin')} / {D(f'domains.{d}.seed_info.{s}.n_readout')}"
            for s in seeds_of(D, f"domains.{d}.seed_info")) or "—") + "。")
    L.append(f"- cyc12 / d6 / caesar 只有 s0：seed 0 上是 FLOOR（标签见表），没有复现。caesar 用的是真实字母 token，单独看。")
    L.append("")
    # 限定
    L.append("**限定**")
    q = []
    if prov:
        q.append(f"PROVISIONAL：缺 " + "；".join(probs) + "。")
    for x in D.strs("excused"):
        q.append(f"EXCUSED：{x}")
    for d in REP:
        p = f"domains.{d}.primary"
        wq = window_qualifier(D, d)
        if wq:
            q.append(wq)
        k = D.raw(f"{p}.k")
        if k == 1:
            q.append(f"{d}：只有 1 个配对 seed（{D(f'{p}.grade.label')}），没有 d 和 boot90。")
        elif isinstance(k, int) and not isinstance(k, bool) and 1 < k <= 3 and D.raw(f"{p}.boot90") not in (_MISS, None, []):
            q.append(f"{d}：k = {k}，boot90 只有 {math.comb(2 * k - 1, k)} 种不同的重采样均值，区间粗糙（k = 2 时就是 [min g, max g]）。")
        if D.raw(f"{p}.v2_flag") is True:
            q.append(f"{d}：N = 3.2e-2 是网格顶端，入选的话会触发 Amendment V2（none@6.4e-2）。")
        tags = D.strs(f"{p}.tags")
        if tags:
            q.append(f"{d}：{'；'.join(tags)}。")
        for m in D.strs(f"domains.{d}.marks"):
            q.append(f"{d} MARK：{m}")
    q.append("P 和 N 都是在同一批 seed 上按均值挑出来的（两边都有 winner's curse）：P 一侧让 g 偏高，N 一侧让 g 偏低。")
    q.append(f"seed 1–2 没有 auditor 计数，twin-free 掩码只做了能做的断言；各 seed 的 twin-free n 已在上面列出。")
    cc = D("cross_check.status")
    sdiff = D.raw("screen_consistency.diffs")
    q.append(f"与 pick_pt6v2.py --allow_partial 交叉核对：{cc}（{D('cross_check.n_fields')} 个字段）"
             + ("：" + "；".join(D.strs("cross_check.diffs")) if cc != "EQUAL" else "")
             + f"；seed 0 与 screen read-out 一致性：{'EQUAL' if sdiff in (_MISS, None, []) and 'screen_consistency.diffs' not in D.missing else '有差异或缺失 ' + str(sdiff if sdiff is not _MISS else 'MISSING screen_consistency.diffs')}"
             + f"（{D('screen_consistency.n_cells')} 行）。")
    for i, x in enumerate(q, 1):
        L.append(f"{i}. {x}")
    L.append("")
    L.append(f"**审查状态**：readout_ptdom3r.py {D('inputs.readout_ptdom3r.py')} 待新开代码审查；探索档，没有预登记判定规则。")
    if D.missing:
        L += ["", f"<!-- MISSING keys ({len(set(D.missing))}): {sorted(set(D.missing))} -->"]
    return "\n".join(L) + "\n", sorted(set(D.missing))


def latest_json(folder):
    """the newest readout_rep_partial_*.json in folder (the name carries the UTC stamp), never a *.pick.json side file."""
    c = [p for p in Path(folder).glob("readout_rep_partial_*.json") if not p.name.endswith(".pick.json")]
    return max(c) if c else None


def load_input(path):
    """-> (obj, None) or (None, 'MISSING <path> (...)'); never raises on a missing / unreadable / non-object file."""
    p = Path(path)
    try:
        obj = json.loads(p.read_text())
    except FileNotFoundError:
        return None, f"MISSING {p} (no such file)"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as ex:
        return None, f"MISSING {p} (unreadable / not JSON: {ex})"
    if not isinstance(obj, dict):
        return None, f"MISSING {p} (JSON is a {type(obj).__name__}, not an object)"
    return obj, None


def selftest(path=None):
    n, ok_ = 0, 0

    def ok(name, c, got=""):
        nonlocal n, ok_
        n += 1
        ok_ += bool(c)
        print(("PASS " if c else "FAIL ") + name + ("" if c else f"   got: {str(got)[:300]}"))
    src = Path(path) if path else latest_json(Path(__file__).parent / "tests")
    obj, err = load_input(src) if src else (None, "MISSING tests/readout_rep_partial_*.json")
    if obj is None:
        print(err)
        return 1
    md, miss = render(obj)
    ok(f"T1 the real JSON ({src.name}) renders with no MISSING key", not miss and not re.search(r"MISSING [A-Za-z0-9_]+\.", md), miss)
    ok("T2 the torus mean g printed equals the JSON value", f"{obj['domains']['torus']['primary']['mean_g']:+.1f}" in md, md[:300])
    o2 = copy.deepcopy(obj)
    del o2["domains"]["hcube"]["primary"]["mean_g"]
    md2, miss2 = render(o2)
    ok("T3 a deleted key prints 'MISSING domains.hcube.primary.mean_g'", "MISSING domains.hcube.primary.mean_g" in md2
       and "domains.hcube.primary.mean_g" in miss2, miss2)
    o3 = copy.deepcopy(obj)
    o3["domains"]["torus"]["primary"]["d"] = None
    ok("T4 a null prints '—'", "| — |" in render(o3)[0])
    o4 = copy.deepcopy(obj)
    o4["complete"] = True
    o4["problems"] = []
    md4 = render(o4)[0]
    ok("T5 complete true: no PROVISIONAL banner", "PROVISIONAL" not in md4, md4[:200])
    o5 = copy.deepcopy(obj)
    o5["cross_check"]["status"], o5["cross_check"]["diffs"] = "MISMATCH", ["torus: k 2 (here) != 3 (pick)"]
    ok("T6 a cross-check MISMATCH is printed with its diff", "MISMATCH" in render(o5)[0] and "torus: k 2" in render(o5)[0])
    # T7 the unit line says area = all 75 evals, not the prior window
    ok("T7 unit text: area over all 75 evals; window split defined", "全部 75 次 eval 上的均值" in md and "不只是 prior window" in md
       and "window 外 = 其余各次 eval" in md, re.findall(r"\*\*结果\*\*.*", md))
    # T8 the window split table and levels come from the JSON
    wt = obj["domains"]["torus"]["primary"].get("window")
    ok("T8 window table + per-seed levels printed from the JSON (torus in-window mean g, prior s0 final3 level)",
       wt is not None and f"{wt['in']['mean_g']:+.1f}" in md and f"{wt['levels']['prior']['0']['final3']:.1f}" in md
       and "各 seed 水平（window 内 / window 外 / final3）" in md, wt)

    # T9 the window qualifier is worded from the computed signs: both directions, and none when the signs agree
    def planted(in_g, out_g, f3_g, g, lv):
        o = copy.deepcopy(obj)
        pr = o["domains"]["torus"]["primary"]
        pr["mean_g"], pr["final3"]["mean_g"] = g, f3_g
        pr["window"]["in"]["mean_g"], pr["window"]["out"]["mean_g"] = in_g, out_g
        pr["window"]["level_means"] = {"prior": {"area": 0, "in": lv[0], "out": 0, "final3": lv[1]},
                                       "none": {"area": 0, "in": lv[2], "out": 0, "final3": lv[3]}}
        for dd in ("hcube", "box5"):
            o["domains"][dd]["primary"]["window"] = None
        return [x for x in render(o)[0].splitlines() if x.split(". ", 1)[-1].startswith("torus：g 的符号")]
    a = planted(+9.0, -4.0, -12.0, +5.0, (80.0, 70.0, 60.0, 90.0))
    ok("T9a g > 0 > final3, prior falls / none rises: qualifier says so", len(a) == 1 and "prior window 内 +9.0（prior 臂更高" in a[0]
       and "最后 3 次 eval -12.0（none 更高" in a[0] and "prior 臂降、none@" in a[0] and "升" in a[0], a)
    b = planted(-6.0, +3.0, +8.0, -2.0, (60.0, 85.0, 75.0, 70.0))
    ok("T9b g < 0 < final3, prior rises / none falls: qualifier worded the other way", len(b) == 1
       and "prior window 内 -6.0（none 更高" in b[0] and "最后 3 次 eval +8.0（prior 臂更高" in b[0] and "prior 臂升、none@" in b[0], b)
    c = planted(+6.0, +3.0, +8.0, +5.0, (60.0, 85.0, 75.0, 70.0))
    ok("T9c all signs agree: no window qualifier", c == [], c)
    e = planted(+9.0, +1.0, -2.0, +5.0, (80.0, 87.0, 70.0, 94.0))
    ok("T9d both arms rise, none more: 'two arms rise, none@N more'", len(e) == 1 and "两臂都升" in e[0] and "升得更多" in e[0]
       and "none@" in e[0].split("两臂都升")[1], e)
    # T10 the default glob never picks a *.pick.json; odd / missing input prints MISSING, never a TypeError
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "readout_rep_partial_20990101T0000.json").write_text("{}")
        (Path(td) / "readout_rep_partial_20990101T0000.pick.json").write_text("{}")
        (Path(td) / "readout_rep_partial_20990101T0001.pick.json").write_text("{}")
        lj = latest_json(td)
        ok("T10a default glob skips *.pick.json", lj is not None and lj.name == "readout_rep_partial_20990101T0000.json", lj)
        (Path(td) / "list.json").write_text("[1, 2]")
        (Path(td) / "bad.json").write_text("{not json")
        r1, r2, r3 = load_input(Path(td) / "nope.json"), load_input(Path(td) / "list.json"), load_input(Path(td) / "bad.json")
        ok("T10b missing / non-object / non-JSON input -> MISSING message", all(x[0] is None and x[1].startswith("MISSING ") for x in (r1, r2, r3)),
           (r1, r2, r3))
    try:
        mde, misse = render({})
        o6 = copy.deepcopy(obj)
        pr = o6["domains"]["torus"]["primary"]
        pr.update({"mean_g": "abc", "boot90": ["x", 1], "g": [1, 2], "tags": [3, None], "window": "odd", "same_lr": [1],
                   "d": {"x": 1}, "final3": {"g": {"0": "y"}, "mean_g": [1]}})
        o6["domains"]["torus"].update({"_cells": [1], "lm": 5, "seed_info": "s", "marks": "m"})
        o6["domains"]["hcube"]["primary"]["window"] = {"in": 1, "levels": [1], "level_means": {"prior": "z"}}
        o6.update({"problems": [1, 2], "excused": "e", "cross_check": {"status": None, "diffs": 7}})
        md6, miss6 = render(o6)
        raised = None
    except Exception as ex:  # noqa: BLE001 (the test is that nothing raises)
        raised, md6, miss6, mde, misse = repr(ex), "", [], "", []
    ok("T10c an empty object and odd-typed values render without raising; odd fields print MISSING",
       raised is None and "MISSING inputs.utc" in mde and "MISSING domains.torus.primary.mean_g" in md6
       and "domains.torus.primary.boot90" in miss6 and "domains.torus.primary.window" in " ".join(miss6), raised or miss6)
    print(f"SELFTEST {'PASS' if ok_ == n else 'FAIL'} {ok_}/{n}")
    return 0 if ok_ == n else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--fid", default="F104")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest(a.json))
    if not a.json:
        print("MISSING input: give a readout_rep_*.json (not a *.pick.json)")
        sys.exit(1)
    src = Path(a.json)
    obj, err = load_input(src)
    if obj is None:
        print(err)
        sys.exit(1)
    md, miss = render(obj, a.fid)
    md += f"\n<!-- rendered from {src} (sha {hashlib.sha256(src.read_bytes()).hexdigest()[:12]}) by render_findings_ptdom3r.py "
    md += f"{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]} -->\n"
    if a.out:
        Path(a.out).write_text(md)
        print(f"wrote {a.out}" + (f"; MISSING {len(miss)} key(s): {miss}" if miss else ""))
    else:
        print(md)
    sys.exit(1 if miss else 0)


if __name__ == "__main__":
    main()
