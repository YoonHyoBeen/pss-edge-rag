"""채점 정규화와 EM/F1. **채점 규칙은 여기 하나뿐이다.**

계획서 7절 「정답 정규화 세 겹」:
  1겹  소문자화 · 관사 제거 · 공백 정리 (SQuAD 표준)
       단, SQuAD 처럼 ASCII 구두점만 지우면 32 N-m 은 "32 nm", 32 N·m 은 "32 n·m" 이 되어 같은 값이
       다르게 판정된다(계획서 근거). 그래서 **유니코드 구두점 전부**를 같은 방식으로 공백으로 바꾼다.
       숫자 사이 소수점은 지킨다. % 는 " percent" 로 푼다. 숫자 뒤 단위 "a"(ampere) 는 관사로 지우지 않는다.
  2겹  단위 표기 정규화 — 정답마다 만든 **허용 변형 목록**과 대조 (answer_variants)
       표기 차이만 허용하고 의미 추가는 금지한다. "no higher than 8 feet" 와 "8 feet" 는 다른 답이다.
  3겹  추출형 압축기 전제 — 압축기가 전부 원문 문장을 그대로 남기므로 생성 답을 원문 정답과 직접
       대조할 수 있다. 코드로 할 일은 없고, 정답 보존율로 전제가 지켜지는지 확인한다.

EM 은 변형 중 하나와 정규화 후 완전 일치, F1 은 변형들 중 최댓값 (SQuAD 의 다중 정답 방식).
변형 없이 1겹만 적용한 값(em_strict)도 같이 남긴다.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from collections import Counter

_TRANS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                        "—": "-", "−": "-", " ": " ", "⁄": "/", "·": " "})
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_ARTICLES = re.compile(r"(?<!\d )\b(a|an|the)\b")       # "400 a"(ampere) 는 남긴다
_DIGIT_EDGE = re.compile(r"(?<=\d)(?=[^\d\s.])|(?<=[^\d\s.])(?=\d)")


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def normalize(s: str) -> str:
    """1겹."""
    s = unicodedata.normalize("NFKC", s).translate(_TRANS).lower()
    s = s.replace("%", " percent ")
    s = _DECIMAL.sub("", s)                     # 1.5 를 지키려고 잠시 다른 점으로
    s = "".join(" " if _is_punct(ch) else ch for ch in s)
    s = s.replace("", ".")
    s = _DIGIT_EDGE.sub(" ", s)                       # 450v -> 450 v, 27°f -> 27 °f
    s = " ".join(s.split())
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


# ---------------------------------------------------------------- 2겹: 허용 변형
# 같은 단위의 표기만 묶는다. 숫자 바로 뒤에서만 바꾼다 ("in accordance" 의 in 을 건드리지 않게).
UNIT_GROUPS = [
    ["degrees fahrenheit f", "degrees fahrenheit", "degrees f", "°f", "deg f"],
    ["degrees celsius c", "degrees celsius", "degrees c", "°c", "deg c"],
    ["ft lbs", "ft lb", "foot pounds", "foot pound", "lb ft"],
    ["n m", "nm", "newton meters", "newton meter"],
    ["inches", "inch", "in"],
    ["feet", "foot", "ft"],
    ["pounds", "pound", "lbs", "lb"],
    ["hours", "hour", "hrs", "hr"],
    ["minutes", "minute", "mins", "min"],
    ["seconds", "second", "secs", "sec"],
    ["gallons", "gallon", "gal"],
    ["volts", "volt", "v"],
    ["amperes", "ampere", "amps", "amp", "a"],
    ["millimeters", "millimeter", "mm"],
    ["kilometers", "kilometer", "km"],
    ["meters", "meter", "m"],
    ["hertz", "hz"],
    ["kilowatts", "kilowatt", "kw"],
]
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
             "eighteen", "nineteen", "twenty"]
RANGE_WORDS = ["through", "to", "thru"]
MAX_VARIANTS = 32


def _is_num(tok: str) -> bool:
    return bool(re.fullmatch(r"\d+(\.\d+)?", tok)) or tok in NUM_WORDS


def _standalone_numbers(gold: str) -> set[str]:
    """정답 원문에서 '단독 수량'으로 쓰인 0~20 숫자/숫자어. 식별자 일부(009-024, 4E, B36)는 제외."""
    raw = unicodedata.normalize("NFKC", gold).lower()
    # 규격 코드 바로 뒤 번호(SSPC-SP 11, Grade 5 같은 등급 표기 포함)는 식별자의 일부다
    raw = re.sub(r"\b([a-z]*[a-z]{2,}[-/][a-z0-9-]*|grade|class|type|level|section|paragraph|table|figure|note|item)\s+(\d+)",
                 lambda m: m.group(1) + "_" + m.group(2), raw)
    nums = set(re.findall(r"(?<![\w.\-/])([1-9]\d?|0)(?![\w./])(?!-\d)", raw))   # "3-phase" 는 허용, "009-024" 는 제외
    words = set(re.findall(r"(?<![\w\-])(" + "|".join(NUM_WORDS) + r")(?!\w)(?!-\d)", raw))
    return {n for n in nums if int(n) <= 20} | words


def _slots(tokens: list[str], free_nums: set[str] = frozenset()) -> list[tuple[str, list[str]]]:
    """토큰열을 (묶음 키, 대안 목록) 칸의 열로 바꾼다. 같은 키의 칸은 한꺼번에 같은 대안으로 바꾼다."""
    groups = [[m.split() for m in g] for g in UNIT_GROUPS]
    slots, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        # 숫자어 + 같은 값 숫자: "ten 10 percent" -> 하나만 남긴 형태도 허용
        if t in NUM_WORDS and t in free_nums and i + 1 < len(tokens) and tokens[i + 1] == str(NUM_WORDS.index(t)):
            slots.append((f"num:{t}", [f"{t} {tokens[i+1]}", t, tokens[i + 1]])); i += 2; continue
        if t in NUM_WORDS and t in free_nums:
            slots.append((f"num:{t}", [t, str(NUM_WORDS.index(t))])); i += 1; continue
        if t in free_nums and re.fullmatch(r"\d+", t):
            slots.append((f"num:{t}", [t, NUM_WORDS[int(t)]])); i += 1; continue
        # 범위 접속어는 숫자 사이(또는 숫자 옆)에서만: "PDR-10 through PDR-30", "27 °f to 110 °f"
        if t in RANGE_WORDS and 0 < i < len(tokens) - 1 and (_is_num(tokens[i - 1]) or _is_num(tokens[i + 1])):
            slots.append(("range", [t] + [w for w in RANGE_WORDS if w != t])); i += 1; continue
        hit = None
        if i > 0 and _is_num(tokens[i - 1]):
            for gi, g in enumerate(groups):
                for m in sorted(g, key=len, reverse=True):
                    if tokens[i:i + len(m)] == m:
                        hit = (gi, g, len(m)); break
                if hit:
                    break
        if hit:
            slots.append((f"unit:{hit[0]}", [" ".join(m) for m in hit[1]])); i += hit[2]; continue
        slots.append(("", [t])); i += 1
    return slots


def answer_variants(gold: str) -> list[str]:
    """정규화된 허용 변형 목록. 첫 원소는 정답 자체.
    같은 묶음 키(예: 화씨 표기 둘)는 한꺼번에 같은 대안으로 바꾼다 — 조합 폭발 없이 '표기만 다른' 답을 덮는다."""
    base = normalize(gold)
    slots = _slots(base.split(), _standalone_numbers(gold))
    keys = sorted({k for k, _ in slots if k})
    width = {k: max(len(a) for kk, a in slots if kk == k) for k in keys}
    out = [base]
    for pick in itertools.product(*[range(width[k]) for k in keys]):
        choice = dict(zip(keys, pick))
        toks = [alts[0] if not k else alts[min(choice[k], len(alts) - 1)] for k, alts in slots]
        v = " ".join(" ".join(toks).split())
        if v not in out:
            out.append(v)
        if len(out) >= MAX_VARIANTS:
            break
    return out


def _f1_tokens(p: list[str], g: list[str]) -> float:
    if not p or not g:
        return float(p == g)
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def score(pred: str, gold: str, variants: list[str] | None = None) -> dict:
    """세 겹 적용 점수. variants 가 없으면 즉석에서 만든다."""
    vs = variants or answer_variants(gold)
    p = normalize(pred)
    return {
        "em": float(p in vs),
        "f1": round(max(_f1_tokens(p.split(), v.split()) for v in vs), 4),
        "contains": float(any(v and f" {v} " in f" {p} " for v in vs)),
        "em_strict": float(p == normalize(gold)),        # 1겹만 (참고)
    }


# 이전 이름 호환 (무문서 필터 등)
def exact_match(pred: str, gold: str) -> float:
    return score(pred, gold)["em"]


def f1(pred: str, gold: str) -> float:
    return score(pred, gold)["f1"]


def contains(pred: str, gold: str) -> float:
    return score(pred, gold)["contains"]
