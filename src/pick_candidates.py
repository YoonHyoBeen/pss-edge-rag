"""문서 배분(평가 먼저) + QA 후보 추출.

  - 배분 단위는 문서. 후보 N_CANDIDATES 개가 찰 때까지 평가 문서를 담고
    나머지 전부를 학습용으로 돌린다
  - 답 유형 4종(수치·단위 / 고유명사·식별자 / 절차 단계 / 조건값)으로 후보를 고른다
  - 한 청크에서 한 문항, 한 문서에서 최대 MAX_QA_PER_DOC 문항
  - 후보는 근거 문장을 하나 이상 가져야 한다 (evidence 필드로 같이 저장한다)

여기서 뽑는 것은 **후보**다. 실제 QA 작성·검수는 사람이 한다.

사용:
    python pick_candidates.py
"""

from __future__ import annotations

import json
import random
import re
import collections
from collections import defaultdict
from pathlib import Path

import config as C
from chunking import _sentences          # 분할기는 하나만 쓴다

# ---------------------------------------------------------------- 근거 문장

# 영어 기능어. 뭉개진 표에는 거의 없고 산문에는 반드시 있다.
STOPWORDS = set("""
a an the of to in on for with and or but is are was were be been being as at by from
that this these those it its which shall must not no any all each other such same than then so
if when unless where while after before during until per into onto over under between within
may can will would should have has had do does did use used using
""".split())

_WORD = re.compile(r"[A-Za-z]+")


def _stop_ratio(sent: str) -> float:
    toks = sent.split()
    if not toks:
        return 0.0
    n = sum(1 for w in toks
            if (m := _WORD.findall(w.lower())) and m[0] in STOPWORDS)
    return n / len(toks)


# 도장 매트릭스 셀. "SEE NOTE (15A)" 같은 참조가 한 문장에 여럿이면 표 칸이
# 줄줄이 이어 붙은 것이다. 문장처럼 보이지만 출제할 내용이 없다.
SEE_NOTE = re.compile(r"SEE\s+NOTES?\b", re.I)


def evidence_sentences(text: str) -> list[str]:
    """QA의 근거로 쓸 수 있는 문장만 남긴다.

    여기를 통과하지 못한 문장은 답이 들어 있어도 쓰지 않는다. 근거 문장이 하나도
    없는 청크는 후보에서 빠진다 — QA를 못 쓰고 근거 보존율도 못 잰다.
    """
    out = []
    for s in _sentences(text):
        n = len(s.split())
        if not (C.EVIDENCE_MIN_TOKENS <= n <= C.EVIDENCE_MAX_TOKENS):
            continue
        if _stop_ratio(s) < C.EVIDENCE_MIN_STOP_RATIO:
            continue
        if len(SEE_NOTE.findall(s)) >= 2:
            continue
        out.append(s)
    return out


# ---------------------------------------------------------------- 답 유형 신호

# 수치 + 단위. NAVSEA 에 흔한 단위를 모았다.
UNIT = (r"(?:N·m|N-m|Nm|newton-meters?|psi|psig|kPa|MPa|bar|"
        r"mm|cm|m|in\.?|inches|ft|lbs?|kg|g|"
        r"°F|°C|degrees?\s*[FC]|V|A|Hz|rpm|%|hours?|minutes?|days?)")
RE_NUMERIC = re.compile(rf"\b\d+(?:\.\d+)?\s*{UNIT}\b", re.I)

# 규격 코드·식별자. "009-021", "MIL-STD-2041", "NAVSEA 0900-LP-001"
RE_IDENTIFIER = re.compile(
    r"\b(?:\d{3}-\d{2,3}|[A-Z]{2,}-[A-Z]*-?\d{2,}|[A-Z]{3,}\s+\d{4}-[A-Z]{2}-\d{3})\b")

# 절차 단계. 명령형 또는 shall 구문.
RE_PROCEDURE = re.compile(
    r"\b(?:shall|must)\s+(?:be\s+)?\w+|"
    r"^\s*(?:Remove|Install|Inspect|Verify|Clean|Apply|Record|Ensure|Replace|Torque)\b",
    re.I | re.M)

# 조건값. if / when / unless 구문.
RE_CONDITIONAL = re.compile(
    r"\b(?:if|when|unless|whenever|where|except|provided that|in the event|"
    r"in case of|as applicable|if required|when required|prior to|subject to)\b[^.]{10,}",
    re.I)

DETECTORS = {
    "numeric_unit": RE_NUMERIC,
    "identifier": RE_IDENTIFIER,
    "procedure_step": RE_PROCEDURE,
    "conditional": RE_CONDITIONAL,
}


def score_types(sents: list[str], doc_id: str = "") -> dict[str, int]:
    """유형 신호를 **근거 문장 안에서만** 센다.

    청크 전체에서 세면 뭉개진 표가 수치 신호 1위로 올라온다. 근거가 되지 못하는
    자리에 있는 신호는 세지 않는다.

    식별자는 자기 문서번호를 빼고 센다. PDF 머리글의 문서번호("009-053")가
    본문에 섞여 들어와 있어서, 그대로 두면 식별자 후보의 3분의 1이
    "이 문서의 번호는?"짜리가 된다.
    """
    out = {}
    for k, rx in DETECTORS.items():
        if k == "identifier":
            out[k] = sum(1 for s in sents
                         if any(m != doc_id for m in rx.findall(s)))
        else:
            out[k] = sum(1 for s in sents if rx.search(s))
    return out


# ---------------------------------------------------------------- 챕터 배분


_PAIR = re.compile(r"(?:^|;)\s*(.+?)\s+is\s+(.+?)(?=;|\.$|$)")


def broken_table_row(sent: str) -> bool:
    """표 행이 열 어긋남으로 깨졌는가.

    표 행은 chunking 에서 "{캡션} — {머리글} is {값}; ..." 로 만든다. 열이
    어긋나면 머리글이 한두 글자로 잘리거나("N is (STAINLESS)"), 머리글과 값이
    같아지거나("MIL-STD-2035, is MIL-STD-2035,"), 같은 덩어리가 반복된다.
    검색 코퍼스에는 그대로 두되(실제 노이즈다) QA 출제에는 쓰지 않는다.
    """
    body = re.sub(r"^.*? — ", "", sent)
    pairs = _PAIR.findall(body)
    if not pairs:
        return True
    for h, v in pairs:
        h, v = h.strip(" .,"), v.strip(" .,")
        if len(h) < 3 or h.lower() == v.lower():
            return True
        if not re.search(r"[A-Za-z]{3,}", h):       # 머리글이 숫자·코드 조각
            return True
        if re.search(r"\w-(?:\s|$)", v):            # 값이 하이픈에서 잘렸다
            return True
    toks = body.split()
    for n in (3, 4, 5):
        for i in range(len(toks) - 2 * n + 1):
            if toks[i:i + n] == toks[i + n:i + 2 * n]:
                return True
    return False


def candidate_view(c: dict) -> dict | None:
    """후보가 될 수 있는 청크면 근거 문장과 유형 점수를 붙여 돌려준다.

    세 가지를 본다.
      1. 450토큰 초과 — 청킹이 실패한 조각이다.
      2. 챕터 1(SCOPE)·2(REFERENCES) — 문서마다 같은 정형구라 QA 거리가 안 된다.
      3. 근거 문장이 하나도 없음 — 뭉개진 표가 여기서 걸린다.
    """
    if c["n_tokens"] > C.MAX_CHUNK_TOKENS:
        return None
    if c["chapter"] in C.EXCLUDE_CHAPTERS:
        return None
    ev = evidence_sentences(c["text"])
    if c["is_table_row"]:
        ev = [s for s in ev if not broken_table_row(s)]
    if not ev:
        return None
    sc = score_types(ev, c["doc_id"])
    if max(sc.values()) == 0:
        return None
    return {"evidence": ev, "type_scores": sc}


def usable_as_candidate(c: dict) -> bool:
    return candidate_view(c) is not None


def pick_holdout(chunks: list[dict], seed: int) -> set[str]:
    """학습에 전혀 쓰지 않을 문서를 고른다.

    후보의 HOLDOUT_FRACTION 만큼을 담을 때까지만 담는다. 거대 문서는
    제외한다 — 하나가 들어오면 학습 청크가 수백 개씩 날아간다.
    """
    usable: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for c in chunks:
        total[c["doc_id"]] += 1
        if usable_as_candidate(c):
            usable[c["doc_id"]] += 1

    target = int(C.N_CANDIDATES * C.HOLDOUT_FRACTION) + 10   # 상한은 낙관적이라 여유
    pool = [d for d in sorted(total)
            if usable[d] > 0 and total[d] <= C.HOLDOUT_MAX_DOC_CHUNKS]
    random.Random(seed).shuffle(pool)

    hold: set[str] = set()
    ceiling = 0
    for d in pool:
        if ceiling >= target:
            break
        hold.add(d)
        ceiling += min(usable[d], C.MAX_QA_PER_DOC)

    hc = sum(total[d] for d in hold)
    print(f"[홀드아웃] {len(hold):3d}문서 / {hc:,}청크 / 후보 상한 {ceiling} (목표 {target})")
    print(f"[나머지]   {len(total) - len(hold):3d}문서 / {len(chunks) - hc:,}청크")
    return hold


# ---------------------------------------------------------------- 후보 선정


def pick(chunks: list[dict], hold_docs: set[str]) -> list[dict]:
    """홀드아웃 문서에서 먼저 뽑고, 나머지 문서에서 채운다.

    한 청크에서 한 문항, 한 문서에서 MAX_QA_PER_DOC 문항까지.
    """
    scored: list[dict] = []
    for c in chunks:
        v = candidate_view(c)
        if v is None:
            continue
        sc = v["type_scores"]
        best = max(sc, key=lambda k: sc[k])
        scored.append({**c, "evidence": v["evidence"], "type_scores": sc,
                       "answer_type": best, "signal": sc[best],
                       "split": "holdout" if c["doc_id"] in hold_docs else "pool"})

    per_doc: dict[str, int] = defaultdict(int)
    used: set[str] = set()
    out: list[dict] = []

    def take(cands: list[dict], atype: str, quota: int) -> int:
        n = 0
        for c in cands:
            if n >= quota:
                break
            if c["chunk_id"] in used:
                continue
            if per_doc[c["doc_id"]] >= C.MAX_QA_PER_DOC:
                continue
            # 그 유형의 신호를 실제로 가진 문장을 앞으로 올린다. QA를 쓸 때
            # evidence[0]이 곧 답이 있는 문장이어야 한다.
            rx = DETECTORS[atype]
            hit = [s for s in c["evidence"]
                   if (any(m != c["doc_id"] for m in rx.findall(s))
                       if atype == "identifier" else rx.search(s))]
            rest = [s for s in c["evidence"] if s not in hit]
            out.append({**c, "answer_type": atype, "evidence": hit + rest,
                        "n_signal_sents": len(hit)})
            used.add(c["chunk_id"])
            per_doc[c["doc_id"]] += 1
            n += 1
        return n

    def round_of(subset: list[dict], quota: int, label: str) -> None:
        # 희소한 유형부터 처리한다. 고정 순서로 돌리면 앞 유형이 청크를 선점해
        # 뒤 유형이 굶는다 — 실제로 105개 목표에 50개밖에 못 채웠다.
        avail = {t: sum(1 for c in subset if c["type_scores"][t] > 0)
                 for t in C.ANSWER_TYPES}
        order = sorted(C.ANSWER_TYPES, key=lambda t: avail[t])
        print(f"\n[{label}] 가용량 적은 유형부터: "
              + ", ".join(f"{t}({avail[t]})" for t in order))
        for atype in order:
            primary = sorted((c for c in subset if c["answer_type"] == atype),
                             key=lambda c: (-c["signal"], c["chunk_id"]))
            taken = take(primary, atype, quota)
            extra = 0
            if taken < quota:
                # argmax 로만 배정하면 신호가 섞인 청크가 한 유형에 몰린다.
                secondary = sorted(
                    (c for c in subset
                     if c["answer_type"] != atype and c["type_scores"][atype] > 0),
                    key=lambda c: (-c["type_scores"][atype], c["chunk_id"]))
                extra = take(secondary, atype, quota - taken)
            got = taken + extra
            tail = f" (+보충 {extra})" if extra else ""
            flag = "" if got >= quota else f"   부족 {quota - got}"
            print(f"  {atype:16s} {got:4d} / {quota}{tail}{flag}")

    q_hold = int(C.PER_TYPE_CANDIDATES * C.HOLDOUT_FRACTION)
    round_of([c for c in scored if c["split"] == "holdout"], q_hold, "홀드아웃")
    round_of([c for c in scored if c["split"] == "pool"],
             C.PER_TYPE_CANDIDATES - q_hold, "나머지")
    return out


# ---------------------------------------------------------------- 출력 순서


def interleave(cands: list[dict]) -> list[dict]:
    """앞에서 몇 개를 잘라도 균형이 맞도록 섞는다.

    QA 작성은 사람이 하고 시간이 모자랄 수 있다. 유형별·구간별로 뭉쳐 있으면
    중간에 멈춘 세트는 쓸 수 없다 — 어떤 유형은 다 차고 어떤 유형은 비어버린다.

    유형 4종을 번갈아 내고, 각 유형 안에서는 홀드아웃을 HOLDOUT_FRACTION 비율로
    끼워 넣는다. 그러면 **몇 번째에서 끊든** 네 유형이 고르고 홀드아웃 비율도
    유지된 평가 세트가 된다. 어디까지 쓸지는 작성 속도를 보고 정하면 된다.

    유형 안의 순서는 **무작위(시드 고정)** 다. 신호 문장 수 내림차순으로 두면
    품질이 나쁜 쪽이 앞으로 몰린다 — 뭉개진 표는 같은 꼴의 문장이 많아 신호가
    높게 잡히기 때문이다. 실제로 배치 1의 출제 불가 비율이 28%였는데 풀 전체로는
    6% 수준이었다. 무작위로 두어야 각 배치가 풀의 대표 표본이 되고, 배치 1에서
    잰 작성 속도와 불량률을 나머지에 그대로 적용할 수 있다.

    이미 검수에 넘긴 후보는 REVIEWED_ORDER 에 적힌 순서 그대로 앞에 고정한다.
    파이프라인을 다시 돌려도 배치 번호가 흔들리지 않아야 검수가 안 헛돈다.
    """
    # 근거 문장이 같은 후보는 하나만 남긴다. NAVSEA 표준 항목은 제출·통보
    # 정형구를 문서끼리 그대로 공유해서, 두면 같은 문항이 여러 번 나온다.
    seen_ev: set[str] = set()
    deduped: list[dict] = []
    for c in cands:
        k = re.sub(r"[^a-z0-9]+", " ", c["evidence"][0].lower()).strip()[:140]
        if k in seen_ev:
            continue
        seen_ev.add(k)
        deduped.append(c)
    if len(deduped) < len(cands):
        print(f"[중복] 근거 문장이 같은 후보 {len(cands) - len(deduped)}건 제거")
    cands = deduped

    pinned_ids: list[str] = []
    pin_path = C.OUT / "reviewed_order.json"
    if pin_path.exists():
        pinned_ids = json.loads(pin_path.read_text(encoding="utf-8"))
    by_id = {c["chunk_id"]: c for c in cands}
    pinned = [by_id[i] for i in pinned_ids if i in by_id]
    rest = [c for c in cands if c["chunk_id"] not in set(pinned_ids)]
    if pinned:
        missing = len(pinned_ids) - len(pinned)
        print(f"[고정] 검수 넘긴 후보 {len(pinned)}건을 앞에 고정"
              + (f" (재선정에서 빠진 것 {missing}건)" if missing else ""))

    rng = random.Random(C.SPLIT_SEED)
    queues: dict[str, list[dict]] = {}
    step = max(1, round(1 / C.HOLDOUT_FRACTION))     # 0.25 -> 4개마다 홀드아웃 1개
    for t in C.ANSWER_TYPES:
        hold = [c for c in rest if c["answer_type"] == t and c["split"] == "holdout"]
        pool = [c for c in rest if c["answer_type"] == t and c["split"] == "pool"]
        rng.shuffle(hold)
        rng.shuffle(pool)
        q: list[dict] = []
        while hold or pool:
            if hold:
                q.append(hold.pop(0))
            for _ in range(step - 1):
                if pool:
                    q.append(pool.pop(0))
        queues[t] = q

    out: list[dict] = list(pinned)
    while any(queues.values()):
        for t in C.ANSWER_TYPES:                      # 유형 번갈아
            if queues[t]:
                out.append(queues[t].pop(0))
    for i, c in enumerate(out):
        c["order"] = i
        c["batch"] = i // C.QA_BATCH + 1
    return out


# ---------------------------------------------------------------- 학습 풀


def train_chunk_ids(chunks: list[dict], hold_docs: set[str],
                    cand_ids: set[str]) -> list[str]:
    """학습에 쓸 청크. 홀드아웃 문서 전체와 평가 후보 청크를 뺀다."""
    return [c["chunk_id"] for c in chunks
            if c["doc_id"] not in hold_docs and c["chunk_id"] not in cand_ids]


def main() -> None:
    if not C.CHUNKS_JSONL.exists():
        raise SystemExit(f"{C.CHUNKS_JSONL} 가 없다. 먼저 chunking.py 를 실행하라.")

    chunks = [json.loads(l) for l in C.CHUNKS_JSONL.open(encoding="utf-8")]
    print(f"청크 {len(chunks):,}개 로드\n")

    hold_docs = pick_holdout(chunks, C.SPLIT_SEED)
    cands = interleave(pick(chunks, hold_docs))

    cand_ids = {c["chunk_id"] for c in cands}
    train_ids = train_chunk_ids(chunks, hold_docs, cand_ids)

    C.CANDIDATES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with C.CANDIDATES_JSONL.open("w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    split_path = C.OUT / "doc_split.json"
    split_path.write_text(json.dumps({
        "seed": C.SPLIT_SEED,
        "holdout_docs": sorted(hold_docs),
        "train_chunk_ids": train_ids,
        "eval_candidate_chunk_ids": sorted(cand_ids),
    }, indent=2), encoding="utf-8")

    n_hold = sum(1 for c in cands if c["split"] == "holdout")
    print(f"\n후보 {len(cands):,}개 (홀드아웃 {n_hold} / 나머지 {len(cands) - n_hold})"
          f"  -> {C.CANDIDATES_JSONL}")
    print(f"학습 청크 {len(train_ids):,}개 -> 청크당 {C.TRAIN_QA_PER_CHUNK}문항이면 "
          f"약 {len(train_ids) * C.TRAIN_QA_PER_CHUNK:,}건")
    print(f"배분 기록 -> {split_path}")

    # 중간에 멈춰도 쓸 수 있는지 확인한다. 배치 경계마다 유형이 고른지 본다.
    print(f"\n배치 {C.QA_BATCH}개 단위 — 앞에서 자른 세트의 균형")
    for n in (C.QA_BATCH, C.QA_BATCH * 2, 420, len(cands)):
        if n > len(cands):
            continue
        head = cands[:n]
        cnt = collections.Counter(c["answer_type"] for c in head)
        nh = sum(1 for c in head if c["split"] == "holdout")
        print(f"  앞 {n:3d}건: " + " ".join(f"{t[:4]} {cnt[t]:3d}" for t in C.ANSWER_TYPES)
              + f"  | 홀드아웃 {nh/n:.0%}")
    print("\n다음 — 후보를 50개씩 배치로 나눠 QA를 작성한다. 배치 1~2를 먼저 쓰고 파일럿을 돌린다.")


if __name__ == "__main__":
    main()
