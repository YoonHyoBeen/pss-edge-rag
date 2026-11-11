"""결과 분석 — 유형별 정확도, 판정(수치·단위 + 식별자 2점), 오답 3분류, 지연 예측 (10-20, B · E0′ · E · F 공용).

판정 (추가실험계획 v3.1)
  같은 입력 길이의 무압축(L{L}_nocomp, E0′ 는 nocomp) 대비 **수치·단위 + 식별자 EM 이 2점 넘게 떨어지지 않으면 통과**.
  통과한 설정 중 응답 시간이 가장 짧은 것 = 권장 설정. 같은 문항끼리 짝지어 McNemar(정확 이항) · 부트스트랩 95% CI.
  절차·조건은 같이 보고하지만 판정에는 안 쓴다.
오답 3분류 (틀린 답만)
  모름   빈 답, 「not mentioned / unknown / not specified ...」
  부분   F1 ≥ 0.5, 또는 정답을 품거나 정답에 품김
  다른값 그 밖 (EM 0 · F1 < 0.5 · 포함 아님) — 다른 수치·다른 규격을 말함 (안전에 가장 나쁜 오답)
  (E0′ 결과 문서 10-20 의 규칙과 같다)
지연 예측 (--fit, latency_curve.py 의 보드 곡선)
  예측 = 프리필(프롬프트 토큰 중앙값) + 출력 토큰 × 디코드(토큰당) + 질문 시점 압축 비용(--comp-ms, 없으면 0)

사용:
  python src/analyze_results.py --results data/processed/results_b_qa_s1000_fs4-v1-q4km.jsonl \\
         --fit jetson/results/latency_fit_board_llama_c6144_<시각>.json
  python src/analyze_results.py --results data/processed/pilot_ctx_results_qa_s1000_k5_fs4-v1-q4km.jsonl   (E0′)
  python src/analyze_results.py --results <ft 결과> --by-doc-split           (E: 학습한 문서 / 처음 보는 문서)
출력: 화면 표 + <결과 이름>_analysis.json + <결과 이름>_analysis.csv (엑셀로 열기)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import config as C
from scoring import normalize

TYPES = ["numeric_unit", "identifier", "procedure_step", "conditional"]
NI = ("numeric_unit", "identifier")
MARGIN = 0.02
RE_DONTKNOW = re.compile(r"^(?:|n/?a|none|unknown|not (?:mentioned|specified|stated|provided|given|found|available|"
                         r"listed|in the context)|no (?:information|answer)|cannot (?:be )?(?:determined|answer(?:ed)?)|"
                         r"i (?:don'?t|do not) know|the context does not.*|not enough information.*)\.?$", re.I)


def baseline_of(cond: str) -> str:
    m = re.match(r"^(L\d+)_", cond)
    return f"{m.group(1)}_nocomp" if m else "nocomp"


def wrong_class(r: dict) -> str | None:
    if r["em"]:
        return None
    p, g = normalize(r.get("pred", "")), normalize(r["answer"])
    if not p or RE_DONTKNOW.match(r.get("pred", "").strip()) or RE_DONTKNOW.match(p):
        return "모름"
    if (g and g in p) or (p and p in g) or r.get("f1", 0) >= 0.5:      # E0′ 결과 문서(10-20)와 같은 규칙
        return "부분"
    return "다른값"


def mcnemar_p(b: int, c: int) -> float:
    """정확 이항 검정 (양측). b = 기준만 맞음, c = 비교만 맞음."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    s = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * s)


def paired(base: dict, cmp: dict, qids: list[str], n_boot: int = 2000, seed: int = 0) -> dict:
    xs = [cmp[q] - base[q] for q in qids]
    if not xs:
        return {}
    d = sum(xs) / len(xs)
    rng = random.Random(seed)
    bs = sorted(sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs) for _ in range(n_boot))
    b = sum(1 for q in qids if base[q] and not cmp[q])
    c = sum(1 for q in qids if cmp[q] and not base[q])
    return {"n": len(xs), "delta": d, "ci_lo": bs[int(0.025 * n_boot)], "ci_hi": bs[int(0.975 * n_boot) - 1],
            "mcnemar_p": mcnemar_p(b, c), "only_base": b, "only_cmp": c}


def load_fit(p: str | None) -> dict | None:
    if not p:
        return None
    j = json.loads(Path(p).read_text(encoding="utf-8"))
    return j.get("fit") or j


def predict_ms(fit: dict, P: float, N: float) -> float:
    p, d = fit["prefill_ms"], fit["decode_ms_per_tok"]
    return p["a"] + p["b"] * P + p["c"] * P * P + N * (d["d"] + d["e"] * P)


def method_of(cond: str) -> str:
    m = re.match(r"^L\d+_(?:B\d+_)?(.+)$", cond)
    return m.group(1) if m else cond


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, nargs="+")
    ap.add_argument("--qa", default=str(C.ROOT / "qa" / "qa_s1000.jsonl"))
    ap.add_argument("--qids", default=None, help="이 QA 파일의 문항만 (예: sub300 — 요약 조건과 같은 문항끼리 비교할 때)")
    ap.add_argument("--fit", default=None, help="latency_fit_*.json (보드 곡선)")
    ap.add_argument("--comp-ms", default=None, help='질문 시점 압축 비용 JSON, 예: {"prov": 180, "bge": 3, "csp": 3}')
    ap.add_argument("--by-doc-split", action="store_true", help="실험 E: 학습한 문서 / 처음 보는 문서로 나눠 본다")
    ap.add_argument("--margin", type=float, default=MARGIN)
    a = ap.parse_args()
    qa = {q["qid"]: q for q in map(json.loads, open(a.qa, encoding="utf-8"))}
    keep = {json.loads(l)["qid"] for l in open(a.qids, encoding="utf-8")} if a.qids else None
    rows = []
    for p in a.results:
        for r in map(json.loads, open(p, encoding="utf-8")):
            if r["qid"] in qa and (keep is None or r["qid"] in keep):
                q = qa[r["qid"]]
                r.setdefault("answer_type", q["answer_type"])
                r.setdefault("source", q.get("source", "base"))
                r.setdefault("doc_id", q.get("doc_id"))
                rows.append(r)
    fit = load_fit(a.fit)
    comp = json.loads(a.comp_ms) if a.comp_ms and a.comp_ms.strip().startswith("{") else \
        (json.loads(Path(a.comp_ms).read_text(encoding="utf-8")) if a.comp_ms else {})
    tok = None
    if fit:
        try:
            import toktools
            tok = toktools.load_tokenizer()
        except Exception as e:
            print(f"[주의] 토크나이저 없음 ({type(e).__name__}) — 출력 토큰을 단어 수 × 1.3 으로 어림")
    split = None
    if a.by_doc_split:
        ds = json.loads((C.OUT / "doc_split.json").read_text(encoding="utf-8"))
        hold = set(ds["holdout_docs"])
        split = lambda r: "학습한문서" if (r.get("source", "base") == "base" and r.get("doc_id") not in hold) else "처음보는문서"

    by = defaultdict(dict)                                   # cond -> qid -> row
    for r in rows:
        by[r["condition"]][r["qid"]] = r
    conds = sorted(by, key=lambda c: (int(re.match(r"L(\d+)", c).group(1)) if re.match(r"L\d+", c) else 0,
                                      0 if c.endswith("nocomp") else 1, c))
    report, table = {}, []
    groups = [("전체", lambda r: True)] + [(t, (lambda t: lambda r: r["answer_type"] == t)(t)) for t in TYPES] + \
             [("수치+식별", lambda r: r["answer_type"] in NI), ("기존", lambda r: r["source"] == "base"),
              ("새문서", lambda r: r["source"] == "new")]
    if split:
        groups += [("학습한문서", lambda r: split(r) == "학습한문서"), ("처음보는문서", lambda r: split(r) == "처음보는문서")]

    def outn(r: dict) -> int:
        if tok is not None:
            return len(tok(r.get("pred", ""), add_special_tokens=False)["input_ids"]) + 1
        return int(len(r.get("pred", "").split()) * 1.3) + 1

    for c in conds:
        cur = by[c]
        bname = baseline_of(c)
        base = by.get(bname, {})
        s = {"n": len(cur), "baseline": bname, "method": method_of(c)}
        for gname, f in groups:
            rs = [r for r in cur.values() if f(r)]
            s[f"em_{gname}"] = (sum(r["em"] for r in rs) / len(rs)) if rs else None
            s[f"n_{gname}"] = len(rs)
        ak = [r["answer_kept"] for r in cur.values() if r.get("answer_kept") is not None]
        s["answer_kept"] = sum(ak) / len(ak) if ak else None
        ct = sorted(r.get("ctx_tokens", 0) for r in cur.values())
        pt = sorted(r.get("prompt_tokens", 0) for r in cur.values())
        s["ctx_tokens_median"], s["prompt_tokens_median"] = ct[len(ct) // 2], pt[len(pt) // 2]
        wr = defaultdict(int)
        ni_rows = [r for r in cur.values() if r["answer_type"] in NI]
        for r in ni_rows:
            w = wrong_class(r)
            if w:
                wr[w] += 1
        s["wrong_ni"] = {k: wr[k] / max(1, len(ni_rows)) for k in ("다른값", "모름", "부분")}
        if base and c != bname:
            common = [q for q in cur if q in base]
            ni = [q for q in common if cur[q]["answer_type"] in NI]
            s["vs_base_all"] = paired({q: base[q]["em"] for q in common}, {q: cur[q]["em"] for q in common}, common)
            s["vs_base_ni"] = paired({q: base[q]["em"] for q in ni}, {q: cur[q]["em"] for q in ni}, ni)
            for t in ("procedure_step", "conditional"):
                tq = [q for q in common if cur[q]["answer_type"] == t]
                s[f"vs_base_{t}"] = paired({q: base[q]["em"] for q in tq}, {q: cur[q]["em"] for q in tq}, tq)
            s["pass"] = (s["vs_base_ni"]["delta"] >= -a.margin - 1e-12) if s["vs_base_ni"] else None
        if fit:
            on = sorted(outn(r) for r in cur.values())
            s["out_tokens_median"] = on[len(on) // 2]
            llm = sum(predict_ms(fit, r.get("prompt_tokens", 0), outn(r)) for r in cur.values()) / len(cur)
            m = s["method"]
            cm = comp.get(m, 0.0)
            if isinstance(cm, dict):                         # {"per_1k_ctx": x} — 압축 전 문맥 길이에 비례
                cm = cm.get("per_1k_ctx", 0) * (sum(r.get("orig_tokens", 0) for r in cur.values()) / len(cur)) / 1000
            s["pred_llm_ms"], s["pred_comp_ms"], s["pred_total_ms"] = round(llm, 1), cm, round(llm + cm, 1)
        report[c] = s

    # ---- 표
    f2 = lambda x: "  -  " if x is None else f"{x:.3f}"
    print(f"문항 {len({r['qid'] for r in rows})} · 조건 {len(conds)} · 행 {len(rows):,}   (판정: 수치+식별 EM 이 기준 대비 -{a.margin * 100:.0f}점 이내)")
    hdr = f"{'조건':<20}{'n':>5}{'EM':>7}{'수치':>7}{'식별':>7}{'절차':>7}{'조건':>7}{'수치+식별':>9}{'Δ수치+식별 [95%CI]':>24}{'p':>7}{'판정':>5}{'정답보존':>8}{'문맥':>6}"
    if fit:
        hdr += f"{'예측ms':>8}"
    print(hdr)
    for c in conds:
        s = report[c]
        v = s.get("vs_base_ni") or {}
        dd = f"{v['delta'] * 100:+.1f} [{v['ci_lo'] * 100:+.1f},{v['ci_hi'] * 100:+.1f}]" if v else \
            ("(기준)" if c == s["baseline"] else "-")
        pp = f"{v['mcnemar_p']:.3f}" if v else ""
        ps = {True: "통과", False: "탈락"}.get(s.get("pass"), "")
        line = (f"{c:<20}{s['n']:>5}{f2(s['em_전체']):>7}{f2(s['em_numeric_unit']):>7}{f2(s['em_identifier']):>7}"
                f"{f2(s['em_procedure_step']):>7}{f2(s['em_conditional']):>7}{f2(s['em_수치+식별']):>9}{dd:>24}{pp:>7}{ps:>5}"
                f"{f2(s['answer_kept']):>8}{s['ctx_tokens_median']:>6}")
        if fit:
            line += f"{s['pred_total_ms']:>8.0f}"
        print(line)
        table.append({"condition": c, **{k: v for k, v in s.items() if not isinstance(v, dict)},
                      **{f"ni_{k}": v for k, v in (s.get("vs_base_ni") or {}).items()},
                      **{f"wrong_ni_{k}": v for k, v in s["wrong_ni"].items()}})
    print(f"\n출처별 EM (기존 / 새 문서)" + ("   ·   학습한 문서 / 처음 보는 문서" if split else ""))
    for c in conds:
        s = report[c]
        extra = f"   {f2(s.get('em_학습한문서'))} / {f2(s.get('em_처음보는문서'))}" if split else ""
        print(f"  {c:<20}{f2(s['em_기존'])} / {f2(s['em_새문서'])}{extra}")
    print(f"\n수치+식별 오답 3분류 (그 조건 수치+식별 문항 중 비율)   다른값 / 모름 / 부분")
    for c in conds:
        w = report[c]["wrong_ni"]
        print(f"  {c:<20}{w['다른값']:.3f} / {w['모름']:.3f} / {w['부분']:.3f}")

    # ---- 권장 설정
    print("\n권장 설정 (통과한 것 중 " + ("예측 응답 시간" if fit else "문맥 토큰") + "이 가장 짧은 것, 입력 길이별)")
    recs = {}
    for c in conds:
        bname = report[c]["baseline"]
        if report[c].get("pass") and not report[c]["method"].startswith(("oracle", "rand")) \
                and report[c]["method"] not in ("csp05", "csp20"):    # 참조 조건 · CSP 민감도(10-21)는 권장에서 뺀다
            key = report[c].get("pred_total_ms") if fit else report[c]["ctx_tokens_median"]
            if bname not in recs or key < recs[bname][1]:
                recs[bname] = (c, key)
    for bname in sorted(recs, key=lambda b: (int(re.match(r"L(\d+)", b).group(1)) if re.match(r"L\d+", b) else 0)):
        c, key = recs[bname]
        b = report.get(bname, {})
        bk = b.get("pred_total_ms") if fit else b.get("ctx_tokens_median")
        print(f"  {bname:<14} -> {c:<20} {key:.0f}" + (f"  (무압축 {bk:.0f})" if bk else ""))
    if not recs:
        print("  통과한 압축 설정이 없다")

    stem = Path(a.results[0]).stem + ("_sub" if keep else "")
    jp = C.OUT / f"{stem}_analysis.json"
    jp.write_text(json.dumps({"margin": a.margin, "fit": a.fit, "comp_ms": comp, "by_condition": report,
                              "recommended": {k: v[0] for k, v in recs.items()}}, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    cp = C.OUT / f"{stem}_analysis.csv"
    keys = sorted({k for t in table for k in t}, key=lambda k: (k != "condition", k))
    with cp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    print(f"\n-> {jp}\n-> {cp}")


if __name__ == "__main__":
    main()
