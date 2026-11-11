"""bge-base-en-v1.5 로 청크 임베딩을 만들고 저장한다.

Jetson 으로 넘길 인덱스가 여기서 나온다. 데스크탑에서 만들어 보드로 복사한다.

사용:
    python build_index.py
    python build_index.py --query "What is the torque for the flange bolt?"   # 점검
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config as C


def load_chunks() -> list[dict]:
    if not C.CHUNKS_JSONL.exists():
        raise SystemExit(f"{C.CHUNKS_JSONL} 가 없다. 먼저 chunking.py 를 실행하라.")
    return [json.loads(l) for l in C.CHUNKS_JSONL.open(encoding="utf-8")]


def get_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("pip install sentence-transformers")
    return SentenceTransformer(C.EMBED_MODEL_ID)


def build() -> None:
    chunks = load_chunks()
    model = get_model()
    texts = [c["text"] for c in chunks]

    # 문서 쪽은 접두어를 붙이지 않는다 (bge-en 공식 권장).
    emb = model.encode(
        texts, batch_size=32, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype("float32")

    C.EMB_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(C.EMB_NPY, emb)
    C.EMB_IDS.write_text(
        json.dumps([c["chunk_id"] for c in chunks], ensure_ascii=False),
        encoding="utf-8")
    print(f"\n{C.EMB_NPY}  shape={emb.shape}")
    print(f"{C.EMB_IDS}  {len(chunks):,}개")


def search(query: str, k: int = C.TOP_K) -> None:
    chunks = load_chunks()
    emb = np.load(C.EMB_NPY)
    ids = json.loads(C.EMB_IDS.read_text(encoding="utf-8"))
    by_id = {c["chunk_id"]: c for c in chunks}

    model = get_model()
    # 질의에는 접두어를 붙인다.
    q = model.encode([C.BGE_QUERY_PREFIX + query],
                     normalize_embeddings=True, convert_to_numpy=True)[0]
    sims = emb @ q
    top = np.argsort(-sims)[:k]

    print(f"\n질의: {query}\n")
    for rank, i in enumerate(top, 1):
        c = by_id[ids[i]]
        print(f"[{rank}] sim={sims[i]:.4f}  {c['chunk_id']}  ({c['n_tokens']}토큰)")
        print(f"    {c['text'][:160]}...\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="인덱스를 만들지 않고 검색만 점검")
    a = ap.parse_args()
    if a.query:
        search(a.query)
    else:
        build()


if __name__ == "__main__":
    main()
