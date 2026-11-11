"""논문 가이드 중 아직 보드에서 안 잰 실험 (11-06). 측정 방식은 board_paper_1103.py 와 같다 (그 코드를 가져다 쓴다).

조건은 돌리기 전에 고정했다. 결과가 어떻게 나와도 바꾸지 않고 그대로 보고한다.

작업 (--job)
  ship   S1000 1,000문항, 보드 bge 검색, 문항마다 아래 11조건을 연달아 (가이드 5.2 · 5.3 · 5.4)
           5.2  k{5,10} × B{무압축, 100, 250, 500}  (8조건, PSS = 사전 계산 bge 문장 선택, 가산점 없음)
                k5_nocomp k5_B100 k5_B250 k5_B500 k10_nocomp k10_B100 k10_B250 k10_B500
           5.3  모든 행에 정답 청크 검색 여부(retrieval_hit) 기록 → 1,000문항 분해 분석 (새 실행 없이 이 파일로)
           5.4  k10_B250_chunk   청크 단위 선택: 검색 순위대로 청크를 통째로 담고, 넘치는 청크는 남은 예산만큼 잘라 넣음 (비용 0)
                k10_B250_ce      크로스인코더 문장 선택: 같은 문장 · 같은 예산 · 같은 순서 복원, 점수기만 크로스인코더 (질문 시점 계산)
                k10_B250_online  사전 계산 없는 PSS: 같은 bge 점수를 질문마다 문장 임베딩해서 계산
  gemma  두 번째 비교 리더 Gemma-2-2B (가이드 5.5): k5_nocomp, k10_B250 — 서버를 gemma2-2b 로 바꿔 띄운 뒤.
         문맥(텍스트)은 Llama 토크나이저로 만들고 프롬프트만 Gemma 토크나이저로 (실험 F · 11-03 Qwen 과 같은 규칙)

측정 규칙 (11-03 과 같음): 응답 = 검색 + 압축 + 토큰화 + 프리필 + 디코드, cache_prompt=false, 문항 단위로 조건 연달아,
몸풀기 1문항, 80°C 쉼 · 65°C 재개, MemAvailable 1.5GB 아래면 멈춤 (같은 명령으로 이어서), VDD_IN 에너지.

사용:  source ~/board_kit/jetson/board_env.sh && cd ~/board_kit
       nohup python3 jetson/board_guide_1106.py --job ship > jetson/results/guide_1106_ship.log 2>&1 &
       (Gemma 서버로 바꾼 뒤) source ~/board_kit/jetson/board_env.sh gemma2-2b
       nohup python3 jetson/board_guide_1106.py --job gemma > jetson/results/guide_1106_gemma.log 2>&1 &
결과:  jetson/results/guide_<작업>_1106.jsonl · guide_<작업>_1106_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import board_paper_1103 as BP   # noqa: E402  (측정 코드 · 전력 · 요약을 그대로 쓴다)

C, Z, M, T, reader = BP.C, BP.Z, BP.M, BP.T, BP.reader
CE_DEFAULT = "cross-encoder/ms-marco-MiniLM-L-6-v2"     # 10-29 보드 측정 때 받은 모델 (새 다운로드 없음)
SHIP_CONDS = ["k5_nocomp", "k5_B100", "k5_B250", "k5_B500", "k10_nocomp", "k10_B100", "k10_B250", "k10_B500",
              "k10_B250_chunk", "k10_B250_ce", "k10_B250_online"]
GEMMA_CONDS = ["k5_nocomp", "k10_B250"]


def select_chunks(tok, passages: list[str], budget: int) -> list[str]:
    """청크 단위 선택 (데스크탑 build_b_contexts.select_chunks 와 같은 규칙, 순서 = 검색 순위)."""
    kept, used = [], 0
    for p in passages:
        n = T.n_tokens(tok, p)
        if used + n <= budget:
            kept.append(p)
            used += n
        else:
            if budget - used > 0:
                s = T.cut_to(tok, p, budget - used)
                if s:
                    kept.append(s)
            break
    return kept


class Guide(BP.Runner):
    def __init__(self, a):
        super().__init__(a)
        self.env.update(guide="11-06", ce_model=a.ce)

    def loop(self, job, qa, conds_of, do_question):                     # 파일 이름만 다르다 (guide_*_1106)
        BP.RES.mkdir(exist_ok=True)
        tag = f"_test{self.a.limit}" if self.a.limit else ""
        out = BP.RES / f"guide_{job}_1106{tag}.jsonl"
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

    def _ship_data(self):
        import numpy as np
        qa = [json.loads(l) for l in (BP.ROOT / "qa" / "qa_s1000.jsonl").open(encoding="utf-8")]
        if self.a.limit:
            qa = qa[: self.a.limit]
        chunks = {c["chunk_id"]: c for c in map(json.loads, C.CHUNKS_JSONL.open(encoding="utf-8"))}
        d = dict(qa=qa, cemb=np.load(C.EMB_NPY).astype("float32"), cids=json.loads(C.EMB_IDS.read_text(encoding="utf-8")),
                 semb=np.load(C.OUT / f"sent_emb_{C.CORPUS}.npy"),
                 meta={m["chunk_id"]: m for m in map(json.loads, (C.OUT / f"sent_meta_{C.CORPUS}.jsonl").open(encoding="utf-8"))})
        d["trunc"] = {cid: reader.truncate_passage(self.ltok, chunks[cid]["text"])[0] for cid in d["cids"]}
        return d

    def _retrieve(self, D, question, k):
        import numpy as np
        self.sync()
        t0 = time.perf_counter()
        qv = self.qvec(question)
        ids = [D["cids"][j] for j in np.argsort(-(D["cemb"] @ qv))[:k]]
        return t0, qv, ids, (time.perf_counter() - t0) * 1000

    @staticmethod
    def _sents(D, ids):
        sents, rows = [], []
        for r, cid in enumerate(ids):
            m = D["meta"][cid]
            for i, (s, n) in enumerate(zip(m["sents"], m["n_tokens"])):
                sents.append(Z.Sent(r, i, s, n))
                rows.append(m["start"] + i)
        return sents, rows

    def _ce(self):
        if "ce" not in self.models:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            t = AutoTokenizer.from_pretrained(self.a.ce)
            m = AutoModelForSequenceClassification.from_pretrained(self.a.ce).to(self.dev).eval()
            self.models["ce"] = (t, m)
        return self.models["ce"]

    def ce_scores(self, question, sents):
        t, m = self._ce()
        out = []
        for a in range(0, len(sents), 64):
            part = sents[a:a + 64]
            enc = t([question] * len(part), [s.text for s in part], padding=True, truncation=True, max_length=512,
                    return_tensors="pt").to(self.dev)
            with self.torch.inference_mode():
                out += m(**enc).logits[:, 0].float().cpu().tolist()
        return out

    # ------------------------------------------------------------------ ship (5.2 · 5.3 · 5.4)
    def job_ship(self):
        t_prep = time.perf_counter()
        D = self._ship_data()
        self._ce()
        self.env.update(ship_prep_s=round(time.perf_counter() - t_prep, 1), conds=SHIP_CONDS)
        print(f"[ship] 사전 준비 {self.env['ship_prep_s']:.0f}초 · 크로스인코더 {self.a.ce} · MemAvailable {M.mem_available_mb()}MB")

        def do_question(q, skip, waited):
            out, first = [], True
            for c in SHIP_CONDS:
                if c in skip:
                    continue
                parts = c.split("_")
                k = int(parts[0][1:])
                t0, qv, ids, retr = self._retrieve(D, q["question"], k)
                self.sync()
                t1 = time.perf_counter()
                extra = {"retrieval_hit": q["chunk_id"] in ids, "k": k}
                if parts[1] == "nocomp":
                    passages = [D["trunc"][x] for x in ids]
                elif len(parts) == 2:                                      # PSS (사전 계산)
                    sents, rows = self._sents(D, ids)
                    sc = (D["semb"][rows].astype("float32") @ qv).tolist()
                    passages = Z.assemble(BP.select_exact(self.ltok, sents, sc, int(parts[1][1:])))
                elif parts[2] == "chunk":
                    passages = select_chunks(self.ltok, [D["trunc"][x] for x in ids], 250)
                elif parts[2] == "ce":
                    sents, _ = self._sents(D, ids)
                    sc = self.ce_scores(q["question"], sents)
                    self.sync()
                    passages = Z.assemble(BP.select_exact(self.ltok, sents, sc, 250))
                else:                                                      # online: 질문마다 문장 임베딩
                    sents, _ = self._sents(D, ids)
                    sc = self.sent_scores(q["question"], sents, qv)
                    self.sync()
                    passages = Z.assemble(BP.select_exact(self.ltok, sents, sc, 250))
                comp = (time.perf_counter() - t1) * 1000 if parts[1] != "nocomp" else 0.0
                g = self.generate(passages, q["question"])
                if waited is None:
                    continue
                r = self.row("ship", q, c, passages, g, t0, retr, comp, True, extra)
                r["paused_s_before"] = round(waited, 1) if first else 0.0
                first = False
                out.append(r)
            return out

        stopped = self.loop("ship", D["qa"], lambda q: SHIP_CONDS, do_question)
        self.free("ce")
        return stopped

    # ------------------------------------------------------------------ gemma (5.5)
    def job_gemma(self):
        if C.READER_MODEL != "gemma2-2b":
            sys.exit("[멈춤] --job gemma 는 `source jetson/board_env.sh gemma2-2b` 와 Gemma 서버에서")
        D = self._ship_data()

        def do_question(q, skip, waited):
            out = []
            for c in GEMMA_CONDS:
                if c in skip:
                    continue
                k = 5 if c.startswith("k5") else 10
                t0, qv, ids, retr = self._retrieve(D, q["question"], k)
                t1 = time.perf_counter()
                if c.endswith("nocomp"):
                    passages = [D["trunc"][x] for x in ids]
                else:
                    sents, rows = self._sents(D, ids)
                    sc = (D["semb"][rows].astype("float32") @ qv).tolist()
                    passages = Z.assemble(BP.select_exact(self.ltok, sents, sc, 250))
                comp = (time.perf_counter() - t1) * 1000 if not c.endswith("nocomp") else 0.0
                g = self.generate(passages, q["question"])
                if waited is not None:
                    out.append(self.row("gemma", q, c, passages, g, t0, retr, comp, True,
                                        {"retrieval_hit": q["chunk_id"] in ids, "k": k}))
            return out

        return self.loop("gemma", D["qa"], lambda q: GEMMA_CONDS, do_question)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=["ship", "gemma"])
    ap.add_argument("--limit", type=int, default=0, help="앞 N 문항만 (시험)")
    ap.add_argument("--ce", default=CE_DEFAULT, help="크로스인코더 모델 (기본: 이미 받은 MiniLM-L6)")
    ap.add_argument("--block-minutes", type=float, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--skip-mode-check", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.job == "ship" and C.READER_MODEL != "llama":
        sys.exit("[멈춤] --job ship 은 Llama 서버에서 (source jetson/board_env.sh)")
    run = Guide(a)
    if run.power.src:
        run.env["idle_power_w"] = run.power.idle_w(10)
        print(f"[전력] 대기 전력 {run.env['idle_power_w']} W (10초 평균)")
    stopped = getattr(run, f"job_{a.job}")()
    if stopped:
        print(f"\n[{stopped}] 에서 멈춤. 서버를 다시 띄우고 같은 명령으로 이어서")
        return
    print("\n[끝] 작업 완료")


if __name__ == "__main__":
    main()
