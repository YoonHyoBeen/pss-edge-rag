"""수정본(11-11)의 추가 보드 실험 E6 · E3. 측정 방식은 board_paper_1103.py · board_guide_1106.py 와 같다 (그 코드를 가져다 쓴다).

조건은 돌리기 전에 고정했다. 결과가 어떻게 나와도 바꾸지 않고 그대로 보고한다.

작업 (--job)
  sent    E6 문장 직접 검색. S1000 1,000문항 (배치 2 = source "new" 500문항을 먼저), 문항마다 아래 2조건을 연달아
            sent_B250   청크 검색 없이 문장 벡터 107,114개 전체에서 바로 검색 → 250토큰까지 담고 → 문서 순서로 복원
                          1) 질문 벡터와 모든 문장 벡터의 내적 → 상위 200문장 (후보)
                          2) 글자가 같은 문장은 점수 높은 것 하나만 남김
                          3) PSS 와 같은 규칙(select_exact)으로 250토큰까지: 점수 높은 순, 넘치는 문장은 건너뜀
                          4) 문서 순서(청크 파일 순서 → 청크 안 문장 순서)로 다시 이어 붙임
            k10_B250    PSS (10청크 검색 → 사전 계산 문장 선택 250토큰). 같은 실행 안의 비교 기준
          시간: 검색 = 질문 임베딩 + 내적 + 상위 고르기, 압축 = 담기 + 순서 복원. 문장 벡터 적재는 사전 비용
  pubce   E3 크로스인코더 문장 선택을 공개 QA 에. HotpotQA 400 (보드 bge 검색 10청크) · NQ 500 (10문서), 250토큰
            k10_B250_ce   11-06 선박 측정과 같은 크로스인코더(MiniLM-L6) · 같은 문장 · 같은 예산 · 같은 순서 복원
            k10_B250_bge  bge 문장 선택 (11-03 공개 QA 측정과 같은 조건). 같은 실행 안의 비교 기준
          11-03 공개 QA 측정과 같이 정확도만 잰다 (응답 시간은 기록하지 않음). 점수 계산 시간만 score_ms 로 남긴다
  all     sent → pubce 를 차례로 (같은 Llama 서버)

측정 규칙 (11-03 · 11-06 과 같음): 응답 = 검색 + 압축 + 토큰화 + 프리필 + 디코드, cache_prompt=false, 문항 단위로 조건 연달아,
몸풀기 1문항, 80°C 쉼 · 65°C 재개, MemAvailable 1.5GB 아래면 멈춤 (같은 명령으로 이어서), VDD_IN 에너지.

사용:  source ~/board_kit/jetson/board_env.sh && cd ~/board_kit
       nohup python3 jetson/board_prof_1111.py --job all > jetson/results/prof_1111_all.log 2>&1 &
결과:  jetson/results/prof_<sent|hotpot|nq>_1111.jsonl · prof_<...>_1111_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import board_paper_1103 as BP   # noqa: E402
import board_guide_1106 as BG   # noqa: E402

C, Z, M, T, reader = BP.C, BP.Z, BP.M, BP.T, BP.reader
SENT_CONDS = ["sent_B250", "k10_B250"]
PUB_CONDS = ["k10_B250_ce", "k10_B250_bge"]
POOL = 200          # 문장 직접 검색의 후보 문장 수 (PSS 가 10청크에서 보는 문장은 약 135개)
BUDGET = 250


def _key(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


class Prof(BG.Guide):
    def __init__(self, a):
        super().__init__(a)
        self.env.update(guide="11-11 prof", pool=POOL, budget=BUDGET)

    def loop(self, job, qa, conds_of, do_question):                     # 파일 이름만 다르다 (prof_*_1111)
        BP.RES.mkdir(exist_ok=True)
        tag = f"_test{self.a.limit}" if self.a.limit else ""
        out = BP.RES / f"prof_{job}_1111{tag}.jsonl"
        done = set()
        if out.exists():
            for r in map(json.loads, out.open(encoding="utf-8")):
                if not r.get("_env"):
                    done.add((r["qid"], r["condition"]))
        todo = [q for q in qa if any((q["qid"], c) not in done for c in conds_of(q))]
        print(f"\n===== [{job}] 문항 {len(qa)} — 남은 문항 {len(todo)} -> {out.name}")
        if not todo:
            return None
        print("몸풀기 1문항 (기록 안 함)")
        do_question(todo[0], set(), None)
        t_start, stopped = time.time(), None
        with out.open("a", encoding="utf-8") as f:
            if not done:
                f.write(json.dumps({"_env": dict(self.env, job=job, n_questions=len(qa))}, ensure_ascii=False) + "\n")
            for i, q in enumerate(todo, 1):
                waited = self.guard.wait_if_hot()
                skip = {c for c in conds_of(q) if (q["qid"], c) in done}
                for r in do_question(q, skip, waited):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                el = (time.time() - t_start) / 60
                if i == 1 or i % 20 == 0 or i == len(todo):
                    print(f"  [{job}] {i}/{len(todo)} 문항  {el:.1f}분 (남은 예상 {el / i * (len(todo) - i):.0f}분)  "
                          f"온도 {M.temp_max_c()}C  MemAvailable {M.mem_available_mb()}MB", flush=True)
                ma = M.mem_available_mb()
                if ma is not None and ma < 1500:
                    stopped = f"MemAvailable {ma:.0f}MB"
                    break
        BP.summarize(out)
        return stopped

    # ------------------------------------------------------------------ sent (E6)
    def job_sent(self):
        import numpy as np
        t_prep = time.perf_counter()
        D = self._ship_data()
        qa = sorted(D["qa"], key=lambda q: 0 if q.get("source") == "new" else 1)      # 배치 2 먼저 (같은 묶음 안은 파일 순서)
        order = [json.loads(l)["chunk_id"] for l in C.CHUNKS_JSONL.open(encoding="utf-8")]   # 문서 순서
        pos_of = {cid: i for i, cid in enumerate(order)}
        S = D["semb"].astype("float32")                                               # 문장 색인 (사전 비용)
        n = S.shape[0]
        r_chunk = np.zeros(n, dtype="int32")
        r_pos = np.zeros(n, dtype="int32")
        r_tok = np.zeros(n, dtype="int32")
        r_text = [""] * n
        filled = 0
        for cid, m in D["meta"].items():
            for i, (s, nt) in enumerate(zip(m["sents"], m["n_tokens"])):
                j = m["start"] + i
                r_chunk[j], r_pos[j], r_tok[j], r_text[j] = pos_of[cid], i, nt, s
                filled += 1
        if filled != n:
            sys.exit(f"[멈춤] 문장 벡터 {n}개와 문장 목록 {filled}개가 맞지 않는다")
        self.env.update(sent_prep_s=round(time.perf_counter() - t_prep, 1), n_sent_vectors=n, conds=SENT_CONDS,
                        n_batch2=sum(1 for q in qa if q.get("source") == "new"))
        print(f"[sent] 사전 준비 {self.env['sent_prep_s']:.0f}초 · 문장 벡터 {n}개 · 배치 2 {self.env['n_batch2']}문항 먼저 · "
              f"MemAvailable {M.mem_available_mb()}MB")

        if not self.a.limit and self.env["n_batch2"] != 500:
            print(f"[주의] 배치 2 문항이 500개가 아니다 ({self.env['n_batch2']}개). 측정은 1,000문항 모두 하므로 계속한다 — 담당자에게 알릴 것")

        def do_question(q, skip, waited):
            out, first = [], True
            for c in SENT_CONDS:
                if c in skip:
                    continue
                if c == "sent_B250":
                    self.sync()
                    t0 = time.perf_counter()
                    qv = self.qvec(q["question"])
                    sc = S @ qv
                    top = np.argpartition(-sc, POOL)[:POOL] if n > POOL else np.arange(n)
                    top = top[np.argsort(-sc[top], kind="stable")]
                    retr = (time.perf_counter() - t0) * 1000
                    t1 = time.perf_counter()
                    seen, sents, scores, rows = set(), [], [], []
                    for j in top.tolist():
                        k = _key(r_text[j])
                        if k in seen:
                            continue
                        seen.add(k)
                        sents.append(Z.Sent(int(r_chunk[j]), int(r_pos[j]), r_text[j], int(r_tok[j])))
                        scores.append(float(sc[j]))
                        rows.append(j)
                    kept = BP.select_exact(self.ltok, sents, scores, BUDGET)
                    passages = Z.assemble(kept)                                       # chunk_rank = 문서 순서
                    comp = (time.perf_counter() - t1) * 1000
                    sel_chunks = {order[s.chunk_rank] for s in kept}
                    extra = {"gold_chunk_in_sel": q["chunk_id"] in sel_chunks,
                             "gold_chunk_in_pool": q["chunk_id"] in {order[int(r_chunk[j])] for j in rows},
                             "n_sel_sents": len(kept), "n_sel_chunks": len(sel_chunks), "n_dup_dropped": len(top) - len(sents),
                             "sel": [[order[s.chunk_rank], s.pos] for s in kept]}
                else:
                    t0, qv, ids, retr = self._retrieve(D, q["question"], 10)
                    self.sync()
                    t1 = time.perf_counter()
                    sents, rows = self._sents(D, ids)
                    sc = (D["semb"][rows].astype("float32") @ qv).tolist()
                    kept = BP.select_exact(self.ltok, sents, sc, BUDGET)
                    passages = Z.assemble(kept)
                    comp = (time.perf_counter() - t1) * 1000
                    extra = {"retrieval_hit": q["chunk_id"] in ids, "k": 10, "n_sel_sents": len(kept),
                             "n_sel_chunks": len({s.chunk_rank for s in kept}),
                             "sel": [[ids[s.chunk_rank], s.pos] for s in kept]}
                g = self.generate(passages, q["question"])
                if waited is None:
                    continue
                r = self.row("ship", q, c, passages, g, t0, retr, comp, True, dict(extra, batch=q.get("source")))
                r["job"] = "sent"
                r["paused_s_before"] = round(waited, 1) if first else 0.0
                first = False
                out.append(r)
            return out

        return self.loop("sent", qa, lambda q: SENT_CONDS, do_question)

    # ------------------------------------------------------------------ pubce (E3)
    def job_pubce(self):
        import numpy as np
        stopped = None
        self._ce()
        for set_ in ("hotpot", "nq"):
            qa = [json.loads(l) for l in (BP.ROOT / "qa" / ("qa_hotpot_v1.jsonl" if set_ == "hotpot" else "qa_nq_litm500.jsonl"))
                  .open(encoding="utf-8")]
            if self.a.limit:
                qa = qa[: self.a.limit]
            if set_ == "hotpot":                               # 11-03 job_pub 과 같은 검색
                chunks = [json.loads(l) for l in (C.OUT / "chunks_hotpot.jsonl").open(encoding="utf-8")]
                emb_p = C.OUT / "chunk_emb_hotpot_board.npy"
                if emb_p.exists():
                    cemb = np.load(emb_p)
                else:
                    cemb = self.bge.encode([c["text"] for c in chunks], batch_size=32, normalize_embeddings=True,
                                           convert_to_numpy=True).astype("float32")
                    np.save(emb_p, cemb)
                ctext = {c["chunk_id"]: c["text"] for c in chunks}
                cids = [c["chunk_id"] for c in chunks]
            else:
                docs = json.loads((C.OUT / "nq_litm500_docs.json").read_text(encoding="utf-8"))

            def do_question(q, skip, waited, set_=set_):
                out = []
                extra = {"set": set_}
                if set_ == "hotpot":
                    qv = self.qvec(q["question"])
                    ids = [cids[j] for j in np.argsort(-(cemb @ qv))[:10]]
                    raw = [ctext[c] for c in ids]
                    extra["chunk_ids"] = ids
                else:
                    qv = None
                    raw = docs[q["qid"]][:10]
                base = [reader.truncate_passage(self.ltok, p)[0] for p in raw]
                sents = Z.split_passages(self.ltok, base)
                for c in PUB_CONDS:
                    if c in skip:
                        continue
                    self.sync()
                    t1 = time.perf_counter()
                    sc = self.ce_scores(q["question"], sents) if c.endswith("_ce") else self.sent_scores(q["question"], sents, qv)
                    self.sync()
                    score_ms = (time.perf_counter() - t1) * 1000
                    kept = BP.select_exact(self.ltok, sents, sc, BUDGET)
                    passages = Z.assemble(kept)
                    g = self.generate(passages, q["question"])
                    if waited is not None:
                        out.append(self.row(set_, q, c, passages, g, None, None, None, False,
                                            dict(extra, score_ms=round(score_ms, 2), n_sents=len(sents), n_sel_sents=len(kept),
                                                 sel=[[s.chunk_rank, s.pos] for s in kept])))
                return out

            stopped = self.loop(set_, qa, lambda q: PUB_CONDS, do_question)
            if stopped:
                return stopped
        self.free("ce")
        return stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=["sent", "pubce", "all"])
    ap.add_argument("--limit", type=int, default=0, help="앞 N 문항만 (시험)")
    ap.add_argument("--ce", default=BG.CE_DEFAULT, help="크로스인코더 모델 (기본: 이미 받은 MiniLM-L6)")
    ap.add_argument("--block-minutes", type=float, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--skip-mode-check", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if C.READER_MODEL != "llama":
        sys.exit("[멈춤] 이 측정은 Llama 서버에서 (source jetson/board_env.sh 를 모델 이름 없이)")
    run = Prof(a)
    if run.power.src:
        run.env["idle_power_w"] = run.power.idle_w(10)
        print(f"[전력] 대기 전력 {run.env['idle_power_w']} W (10초 평균)")
    for j in (["sent", "pubce"] if a.job == "all" else [a.job]):
        stopped = getattr(run, f"job_{j}")()
        if stopped:
            print(f"\n[{stopped}] 에서 멈춤. 서버를 다시 띄우고 같은 명령으로 이어서")
            return
    print("\n[끝] 작업 완료")


if __name__ == "__main__":
    main()
