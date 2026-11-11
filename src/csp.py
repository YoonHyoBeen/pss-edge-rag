"""CSP (Critical Sentence Preservation) — 핵심 요소가 든 문장에 가산점 (10-20, 실험 B·C·E).

선박 정비 문서에서 틀리면 위험한 정보 = 수치·단위, 규격·문서 번호, 조건, 금지. 이런 문장은 bge 점수가
조금 낮아도 먼저 남긴다.

  점수' = bge 코사인 + LAMBDA × [문장에 핵심 요소가 하나라도 있으면 1]

  * 검출 규칙은 QA 후보를 뽑을 때 쓴 정규식(pick_candidates.py)을 그대로 쓰고, 금지 표현만 더했다.
    QA 를 보고 만든 규칙이 아니라 문서 유형 규칙이다 (정답·질문은 안 본다)
  * LAMBDA = 0.1 은 **미리 정한 판단값** (S1000 으로 맞추지 않는다, 점수 분포를 실측해서 정한 값도 아니다).
    의도: 「관련도가 비슷한 문장끼리만 순서를 바꾸고, 관련 없는 문장을 끌어올리지는 않는 크기」.
    S1000 밖 문항(새 문서 146)은 수치 20 · 식별자 0 이라 판정 유형으로 값을 고를 수도 없다
  * 다른 값(0.05 · 0.2)은 민감도로만 (SENSITIVITY, 판정에는 0.1 만)
"""

from __future__ import annotations

import re

from pick_candidates import RE_CONDITIONAL, RE_IDENTIFIER, RE_NUMERIC

LAMBDA = 0.1
# 10-21 결정: 판정·권장 설정은 LAMBDA 0.1 로만 (결과를 보기 전에 고정, 결과가 나빠도 안 바꾼다).
# 아래 값은 「다른 값이면 결론이 뒤집히나」를 보는 민감도 전용 — L 2,000 × B 250·500 에서만, 결과와 상관없이 셋 다 보고
SENSITIVITY = {"csp05": 0.05, "csp20": 0.2}

RE_PROHIBIT = re.compile(
    r"\b(?:shall\s+not|must\s+not|will\s+not|do\s+not|does\s+not|cannot|can\s+not|never|"
    r"prohibited|not\s+permitted|not\s+allowed|not\s+authorized|forbidden|no\s+\w+\s+shall)\b", re.I)

DETECT = {"num": RE_NUMERIC, "id": RE_IDENTIFIER, "cond": RE_CONDITIONAL, "prohib": RE_PROHIBIT}


def flags(text: str) -> list[str]:
    return [k for k, r in DETECT.items() if r.search(text)]


def is_critical(text: str) -> bool:
    return any(r.search(text) for r in DETECT.values())


def adjust(scores: list[float], texts: list[str], lam: float = LAMBDA) -> list[float]:
    return [s + (lam if is_critical(t) else 0.0) for s, t in zip(scores, texts)]
