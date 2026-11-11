"""리더 토크나이저 불러오기 + 프롬프트 토큰 ID 만들기 (10-20, 실험 A·B·C·F 공용).

reader.py 와 같은 순서로 HF 저장소(공식 -> ungated 미러)에서 받는다.
환경변수 SHIP_TOKENIZER_DIR 가 있으면 그 폴더에서 바로 읽는다 (HF 가 막힌 곳에서 시험할 때 —
GGUF 안의 어휘·병합·chat template 으로 만든 폴더. 데스크탑·보드에서는 쓰지 않는다).
"""

from __future__ import annotations

import os

import config as C


def load_tokenizer(prefer_fallback: bool = False):
    from transformers import AutoTokenizer
    d = os.environ.get("SHIP_TOKENIZER_DIR")
    if d:
        tok = AutoTokenizer.from_pretrained(d)
        print(f"[토크나이저] {d} (SHIP_TOKENIZER_DIR — 시험용)")
        return tok
    m = C.READER_MODELS[C.READER_MODEL]
    order = (m["fallback"], m["hf"]) if prefer_fallback else (m["hf"], m["fallback"])
    last = None
    for mid in order:
        try:
            return AutoTokenizer.from_pretrained(mid)
        except Exception as e:                   # gated / 미로그인 / 오프라인
            print(f"[{mid}] 토크나이저 실패: {type(e).__name__}")
            last = e
    raise SystemExit(f"리더 토크나이저를 못 받았다: {last}")


def chat_ids(tok, messages: list[dict]) -> list[int]:
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "keys"):                      # 일부 버전은 BatchEncoding 을 돌려준다
        ids = ids["input_ids"]
    return [int(x) for x in ids]


def ids_of(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def n_tokens(tok, text: str) -> int:
    return len(ids_of(tok, text))


def cut_to(tok, text: str, n: int) -> str:
    """앞에서 n 토큰이 되게 자른다 (넘지 않는 가장 긴 앞부분).
    디코드한 앞부분을 다시 세면 토큰이 합쳐지거나(점선 「.....」 등) 갈라져 k 개 ID 가 k 토큰이 아닐 수 있다
    -> 「다시 셌을 때 n 이하」인 가장 큰 k 를 이분 탐색으로 찾는다."""
    ids = ids_of(tok, text)
    if len(ids) <= n:
        return text
    cnt = lambda k: n_tokens(tok, tok.decode(ids[:k]))
    lo, hi = 0, len(ids)                       # cnt(lo) <= n 을 지키며 hi 를 줄인다
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cnt(mid) <= n:
            lo = mid
        else:
            hi = mid
    return tok.decode(ids[:lo]) if lo else ""


PROBE = "Within how many days must the report be submitted to the SUPERVISOR? 27°F to 110°F, MIL-PRF-32584."


def check_against_server(tok, post) -> None:
    """서버가 같은 문자열을 같은 토큰으로 자르는가 (BOS 빼고). 다르면 다른 모델 파일이다."""
    ours = ids_of(tok, PROBE)
    theirs = post("/tokenize", {"content": PROBE, "add_special": False})["tokens"]
    if ours != theirs:
        raise SystemExit(f"토크나이저 불일치: HF {ours[:10]}... 서버 {theirs[:10]}... — 서버에 다른 모델이 떠 있다")
