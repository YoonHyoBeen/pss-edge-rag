"""실험 D — bge 문장 임베딩을 미리 계산한다 (질문 시점에는 질의 임베딩 + 내적만 남는다).

문장은 본 실험과 똑같이 만든다: 조각을 450 리더 토큰에서 자른 뒤(reader.truncate_passage) chunking._sentences 로 나눈다
(= compressors.split_passages). 그래서 bge4pre 와 bge4 가 같은 문장 · 같은 점수(부동소수 오차 안)가 된다.

출력 (data/processed/)
  sent_emb_<코퍼스>.npy      float16 [문장 수, 768]  (정규화된 bge-base 임베딩)
  sent_meta_<코퍼스>.jsonl   조각마다 {chunk_id, start, sents, n_tokens}
  sent_emb_<코퍼스>_cost.json  사전 비용: 걸린 시간 · 문장 수 · 파일 크기 · 기기   <- 「사전 비용」으로 따로 보고

사용:  보드  source jetson/board_env.sh && python3 src/precompute_sent_emb.py
       데스크탑  set SHIP_CORPUS=v2 && python src\\precompute_sent_emb.py
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime

import numpy as np

import compressors as Z
import config as C
import devmon as M
import reader
import toktools as T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="시험용: 앞 N 조각만")
    a = ap.parse_args()
    out = C.OUT / f"sent_emb_{C.CORPUS}.npy"
    meta_p = C.OUT / f"sent_meta_{C.CORPUS}.jsonl"
    chunks = [json.loads(l) for l in C.CHUNKS_JSONL.open(encoding="utf-8")]
    if a.limit:
        chunks = chunks[: a.limit]
    tok = T.load_tokenizer(prefer_fallback=M.is_jetson())

    t0 = time.perf_counter()
    meta, texts = [], []
    for c in chunks:
        p, _ = reader.truncate_passage(tok, c["text"])
        ss = Z.split_passages(tok, [p])
        meta.append({"chunk_id": c["chunk_id"], "start": len(texts), "sents": [s.text for s in ss],
                     "n_tokens": [s.n_tokens for s in ss]})
        texts += [s.text for s in ss]
    t_split = time.perf_counter() - t0
    print(f"조각 {len(chunks):,}개 -> 문장 {len(texts):,}개  (자르기·나누기 {t_split:.0f}s)")

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(C.EMBED_MODEL_ID, device=dev)
    if dev == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    e = m.encode(texts, batch_size=a.batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    if dev == "cuda":
        torch.cuda.synchronize()
    t_emb = time.perf_counter() - t1
    np.save(out, e.astype("float16"))
    with meta_p.open("w", encoding="utf-8") as f:
        for r in meta:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cost = {"time": datetime.now().isoformat(timespec="seconds"), "host": platform.node(), "jetson": M.is_jetson(),
            "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "corpus": C.CORPUS,
            "n_chunks": len(chunks), "n_sentences": len(texts), "split_s": round(t_split, 1), "embed_s": round(t_emb, 1),
            "sentences_per_s": round(len(texts) / max(t_emb, 1e-9), 1), "batch": a.batch,
            "npy_mb": round(out.stat().st_size / 2**20, 1), "meta_mb": round(meta_p.stat().st_size / 2**20, 1),
            "nvpmodel": M.sh(["nvpmodel", "-q"]) if M.is_jetson() else None, "temp_max_c_end": M.temp_max_c()}
    (C.OUT / f"sent_emb_{C.CORPUS}_cost.json").write_text(json.dumps(cost, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(cost, ensure_ascii=False, indent=2))
    print(f"-> {out}\n-> {meta_p}")


if __name__ == "__main__":
    main()
