"""압축기 — 계획서 1절의 사다리.

  요약카드 (LLMLingua-2) : 질문 안 봄, 청크 단위로 전날 미리 압축 (precompute_cards.py)
  bge 필터              : 질문 봄, 비학습. 문장 임베딩과 질의 임베딩의 코사인
  Provence              : 질문 봄, 학습됨. 토큰별 '남김' 확률의 문장 평균
  무작위                : 참조 하한. 같은 예산을 아무 문장으로
  오라클                : 참조 상한. 정답 근거 문장만

공통 규칙 (계획서 4·9·11절)
  * 문장 분할기는 하나 — chunking._sentences. bge 필터와 Provence 가 같은 문장 경계를 쓴다
  * 예산 방식 — 예산 = 요청 압축률 × 원문(top-k, 각 450토큰 자름) 리더 토큰 수.
    점수 높은 문장부터 담고, 넘치는 문장은 건너뛰고 다음 문장을 본다. 요청·실제 압축률을 둘 다 기록
  * 청크 순서는 검색 순위 그대로, 청크 안 문장 순서도 원문 그대로 (Lost in the Middle — 9절)
  * Provence 는 원 논문의 threshold 대신 점수 상위부터 예산까지 (11절, iso-token 비교)
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import config as C
from chunking import _sentences


@dataclass
class Sent:
    chunk_rank: int       # 검색 순위 (0부터)
    pos: int              # 청크 안 순서
    text: str
    n_tokens: int         # 리더 토큰


def split_passages(tok, passages: list[str]) -> list[Sent]:
    out = []
    for r, p in enumerate(passages):
        for i, s in enumerate(_sentences(p)):
            out.append(Sent(r, i, s, len(tok(s, add_special_tokens=False)["input_ids"])))
    return out


def count_tokens(tok, passages: list[str]) -> int:
    return sum(len(tok(p, add_special_tokens=False)["input_ids"]) for p in passages)


def select_by_score(sents: list[Sent], scores: list[float], budget: int) -> list[Sent]:
    """점수 높은 순으로 예산 안에서 담는다. 최소 한 문장은 담는다."""
    order = sorted(range(len(sents)), key=lambda i: -scores[i])
    kept, used = [], 0
    for i in order:
        if used + sents[i].n_tokens <= budget:
            kept.append(i)
            used += sents[i].n_tokens
    if not kept and order:
        kept = [order[0]]
    return [sents[i] for i in sorted(kept, key=lambda i: (sents[i].chunk_rank, sents[i].pos))]


def assemble(kept: list[Sent]) -> list[str]:
    """청크 순위대로, 청크 안은 원문 순서로 다시 묶는다. 빈 청크는 뺀다."""
    by = {}
    for s in kept:
        by.setdefault(s.chunk_rank, []).append(s.text)
    return [" ".join(by[r]) for r in sorted(by)]


# ---------------------------------------------------------------- 점수기
class BgeScorer:
    """검색용으로 이미 올라와 있는 bge-base 를 재사용 — 추가 메모리 0 (6절)."""

    def __init__(self, model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            model = SentenceTransformer(C.EMBED_MODEL_ID, device="cuda" if torch.cuda.is_available() else "cpu")
        self.m = model

    def __call__(self, question: str, sents: list[Sent]) -> list[float]:
        q = self.m.encode([C.BGE_QUERY_PREFIX + question], normalize_embeddings=True)
        e = self.m.encode([s.text for s in sents], batch_size=64, normalize_embeddings=True)
        return [float(x) for x in (e @ q[0])]


def _provence_normalize(q: str) -> str:
    """Provence 원 코드의 질의 정규화 — 소문자, 구두점 제거, 공백 정리."""
    q = re.sub(r"[^\w\s]", " ", q.lower())
    return " ".join(q.split())


class ProvenceScorer:
    """Provence forward 를 우리 문장에 적용한다.

    입력 형식은 원 코드와 같다: [CLS] 질의 [SEP] 문장들 [EOS], 길이 512 를 넘으면 같은 질의로 새 블록.
    문장 점수 = 그 문장 토큰들의 '남김' 확률(softmax 의 1번) 평균. 원 코드는 이 확률을 threshold 로 0/1 로 만든 뒤
    평균을 내는데, 예산 방식에서는 순위가 필요하므로 연속값을 쓴다 (11절).
    """

    def __init__(self, model=None, tokenizer=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = model or AutoModel.from_pretrained(C.PROVENCE_ID, trust_remote_code=True).to(self.dev).eval()
        self.t = tokenizer or AutoTokenizer.from_pretrained(C.PROVENCE_ID)
        self.max_len = min(C.PROVENCE_MAX_LEN, getattr(self.m.config, "max_position_embeddings", 512))
        eos = self.t.eos_token_id if self.t.eos_token_id is not None else self.t.sep_token_id
        self.cls, self.sep, self.eos = self.t.cls_token_id, self.t.sep_token_id, eos

    def _blocks(self, question: str, sents: list[Sent]):
        qids = self.t(_provence_normalize(question), add_special_tokens=False)["input_ids"]
        prefix = [self.cls] + qids + [self.sep]
        blocks, cur, spans = [], list(prefix), []
        for si, s in enumerate(sents):
            text = s.text if s.pos == 0 else " " + s.text
            ids = self.t(text, add_special_tokens=False)["input_ids"][: self.max_len - len(prefix) - 1]
            if len(cur) + len(ids) + 1 > self.max_len and len(cur) > len(prefix):
                blocks.append((cur + [self.eos], spans))
                cur, spans = list(prefix), []
            spans.append((si, len(cur), len(cur) + len(ids)))
            cur += ids
        if len(cur) > len(prefix):
            blocks.append((cur + [self.eos], spans))
        return blocks

    def __call__(self, question: str, sents: list[Sent]) -> list[float]:
        import torch
        blocks = self._blocks(question, sents)
        scores = [0.0] * len(sents)
        for b in range(0, len(blocks), C.PROVENCE_BATCH):
            part = blocks[b: b + C.PROVENCE_BATCH]
            L = max(len(ids) for ids, _ in part)
            ids = torch.tensor([x + [0] * (L - len(x)) for x, _ in part], device=self.dev)      # 원 코드도 0 으로 채운다
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in part], device=self.dev)
            with torch.inference_mode():
                out = self.m(input_ids=ids, attention_mask=att)
            p = torch.softmax(out.compression_logits.float(), dim=-1)[..., 1].cpu()
            for row, (_, spans) in enumerate(part):
                for si, a, z in spans:
                    scores[si] = float(p[row, a:z].mean()) if z > a else 0.0
        return scores


def random_scores(qid: str, condition: str, sents: list[Sent]) -> list[float]:
    rng = random.Random(f"{C.RANDOM_SEED}:{qid}:{condition}")
    return [rng.random() for _ in sents]
