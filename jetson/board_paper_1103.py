"""WWW 2027 논문 수치 전부를 보드에서 다시 잰다 (11-03 결정: 논문에 쓰는 결과는 모두 보드 결과로).

조건은 돌리기 전에 고정했다. 결과가 어떻게 나와도 바꾸지 않고 그대로 보고한다.
모든 리더 호출은 llama-server(Llama-3.2-3B Q4_K_M, ctx 6144, greedy, 출력 32, cache_prompt=false)로,
압축 모델(bge · Provence · RECOMP · LLMLingua-2)도 모두 이 보드의 GPU 에서 돈다.

작업 (--job)
  ship   S1000 1,000문항, 보드 bge 검색 (표 2 정비 열 · 5.3절 · 5.4절 방해 값 · E1 · E2 · E4 · P4 · P5)
           k10_B{100,250,500}_{csp,prov,recomp,llml2}   250 은 처음부터 끝까지 시간 측정, 100·500 은 같은 압축 점수로 정확도만
           k10_B250_bge      최종 설정 재측정 (P4 — 10-29 종단 실측과 비교해 재실행 변동)
           k10_B250_bge_pc   같은 설정 + 공통 지시문·예시 KV 재사용(cache_prompt=true) (E4) — 문항마다 맨 처음에 돈다
           k1_nocomp k2_nocomp  검색 청크 수만 줄인 기준선 (E1)
           k5_nocomp k10_nocomp 층화 300문항(sub300)에서만 — 무압축 응답의 에너지 (E2)
           시작할 때 검색 Recall@5 · @10 (P5)
  pub    HotpotQA 400 (보드 bge 검색 10청크) · NQ 500 (Lost in the Middle 10문서), 250토큰 (표 2 공개 QA 열)
           k10_nocomp, k10_B250_{bge,csp,prov,recomp,llml2}.  RECOMP 는 원 논문 설정 (HotpotQA → hotpotqa 가중치, NQ → nq 가중치)
  ctrl   통제 문맥 (정답 청크 + hard negative, 정확히 L 토큰 — 데스크탑 실험 B 와 같은 입력 텍스트) (5.4절)
           L{500,1000,2000,4000}_B{100,250,500}_{bge,csp} (B < L, 11칸 × 2) · L4000_nocomp · L2000_B250_csp20 (λ=0.2)
  all    ship → pub → ctrl 을 차례로 (같은 Llama 서버)
  qwen   두 번째 리더 Qwen2.5-3B (E3): k5_nocomp, k10_B250_bge — 서버를 qwen2.5-3b 로 바꿔 띄운 뒤.
         문맥(텍스트)은 Llama 토크나이저로 만들고 (실험 F 와 같은 규칙) 프롬프트만 Qwen 토크나이저로

측정 규칙 (10-29 종단 실측과 같음)
  * 응답 시간 = 검색(질문 임베딩 + 내적 + 상위 k) + 압축(점수 + 고르기) + 프롬프트 토큰화 + 프리필 + 디코드.
    청크 450토큰 자르기 · 문장 나누기 · 문장 임베딩 · CSP 표시 · 모델 적재는 사전 비용 (시간에서 뺀다)
  * 문항 단위 인터리빙, 시작할 때 몸풀기 1문항 (기록 안 함), 80°C 넘으면 쉬고 65°C 아래에서 재개,
    MemAvailable 1.5GB 아래면 멈춤 (서버 다시 띄우고 같은 명령 → 이어서)
  * 에너지 (E2): 보드 입력 전력 VDD_IN 을 20Hz 로 읽어 요청 구간에서 적분한다 (INA3221 sysfs, 없으면 tegrastats)
  * 점검: 보드 압축 문맥 · 답이 데스크탑 결과와 같은 비율 (paper_ref_1103.json)

사용:  source ~/board_kit/jetson/board_env.sh && cd ~/board_kit
       python3 jetson/board_paper_1103.py --job all --limit 3      (시험, 결과는 _test 파일)
       nohup python3 jetson/board_paper_1103.py --job all > jetson/results/paper_1103_all.log 2>&1 &
결과:  jetson/results/paper_<작업>_1103.jsonl (요청마다) · paper_<작업>_1103_summary.json
"""
from __future__ import annotations

import argparse
import bisect
import glob
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
import config as C            # noqa: E402
import compressors as Z       # noqa: E402
import csp                    # noqa: E402
import devmon as M            # noqa: E402
import reader                 # noqa: E402
import toktools as T          # noqa: E402
from scoring import normalize, score   # noqa: E402

RES = HERE / "results"
REF = HERE / "paper_ref_1103.json"
NI = ("numeric_unit", "identifier")
METHODS = ("csp", "prov", "recomp", "llml2")
RECOMP_ID = {"ship": "fangyuan/nq_extractive_compressor", "nq": "fangyuan/nq_extractive_compressor",
             "hotpot": "fangyuan/hotpotqa_extractive_compressor"}
LLAMA_TOK = ("unsloth/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct")
LLML2_ID = getattr(C, "LLMLINGUA2_ID", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
CTRL_LS, CTRL_BS = (500, 1000, 2000, 4000), (100, 250, 500)


def h12(ps: list[str]) -> str:
    return hashlib.md5("\x1e".join(ps).encode("utf-8")).hexdigest()[:12]


def evidence_recall(evidence: str, passages: list[str]) -> float:          # compress_contexts.evidence_recall 와 같다
    from collections import Counter
    e = Counter(normalize(evidence).split())
    c = Counter(normalize(" ".join(passages)).split())
    return sum((e & c).values()) / max(1, sum(e.values()))


def answer_kept(q: dict, passages: list[str]) -> bool:
    vs = q.get("answer_variants") or [normalize(q["answer"])]
    joined = f" {normalize(' '.join(passages))} "
    return any(v and f" {v} " in joined for v in vs)


# ----------------------------------------------------------------------------- 전력 (E2)
class Power:
    """VDD_IN(보드 입력 전력)을 20Hz 로 읽는다. 요청 구간 [t0, t1] 의 에너지(J) = 평균 전력 × 시간."""

    def __init__(self, hz: float = 20.0):
        self.ts, self.ws, self.src, self.dt = [], [], None, 1.0 / hz
        self.lock = threading.Lock()
        v = c = None
        for lab in sorted(glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label")):
            try:
                if "VDD_IN" in Path(lab).read_text():
                    i = re.search(r"in(\d+)_label$", lab).group(1)
                    v, c = lab.replace(f"in{i}_label", f"in{i}_input"), lab.replace(f"in{i}_label", f"curr{i}_input")
                    break
            except OSError:
                pass
        if v and Path(v).exists() and Path(c).exists():
            self.src = f"sysfs {Path(v).parent}"
            self._v, self._c = v, c
            threading.Thread(target=self._sysfs, daemon=True).start()
        else:
            try:
                self.p = subprocess.Popen(["tegrastats", "--interval", str(int(self.dt * 1000))], stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, text=True)
                self.src = "tegrastats"
                threading.Thread(target=self._tegra, daemon=True).start()
            except (OSError, FileNotFoundError):
                self.src = None

    def _add(self, w: float) -> None:
        with self.lock:
            self.ts.append(time.perf_counter())
            self.ws.append(w)
            if len(self.ts) > 20000:
                del self.ts[:10000], self.ws[:10000]

    def _sysfs(self) -> None:
        while True:
            try:
                mv = int(open(self._v).read())
                ma = int(open(self._c).read())
                self._add(mv * ma / 1e6)
            except (OSError, ValueError):
                pass
            time.sleep(self.dt)

    def _tegra(self) -> None:
        for line in self.p.stdout:
            m = re.search(r"VDD_IN (\d+)mW", line)
            if m:
                self._add(int(m.group(1)) / 1000)

    def energy(self, t0: float, t1: float) -> tuple[float | None, float | None]:
        if not self.src:
            return None, None
        time.sleep(self.dt * 1.5)                              # 구간 끝 표본이 들어올 때까지
        with self.lock:
            a, b = bisect.bisect_left(self.ts, t0), bisect.bisect_right(self.ts, t1)
            ws = self.ws[a:b] or self.ws[max(0, a - 1):a + 1]
        if not ws:
            return None, None
        w = sum(ws) / len(ws)
        return round(w * (t1 - t0), 3), round(w, 2)

    def idle_w(self, seconds: float = 10.0) -> float | None:
        if not self.src:
            return None
        t0 = time.perf_counter()
        time.sleep(seconds)
        return self.energy(t0, time.perf_counter())[1]


# ----------------------------------------------------------------------------- 비교 압축기 (exp11_baselines 와 같은 구현)
class Lingua2:
    def __init__(self, torch, dev):
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.torch, self.dev = torch, dev
        self.t = AutoTokenizer.from_pretrained(LLML2_ID)
        self.m = AutoModelForTokenClassification.from_pretrained(LLML2_ID).to(dev).eval()

    def word_probs(self, text: str):
        words = text.split()
        probs = [[] for _ in words]
        for a in range(0, len(words), 200):
            enc = self.t(words[a:a + 200], is_split_into_words=True, truncation=True, max_length=512,
                         return_tensors="pt").to(self.dev)
            with self.torch.inference_mode():
                p = self.torch.softmax(self.m(**enc).logits.float(), dim=-1)[0, :, 1].cpu()
            for ti, wi in enumerate(enc.word_ids(0)):
                if wi is not None:
                    probs[a + wi].append(float(p[ti]))
        return words, [sum(x) / len(x) if x else 0.0 for x in probs]

    def prepare(self, passages):
        return [self.word_probs(p) for p in passages]

    @staticmethod
    def select(per, budget: int, rtok) -> list[str]:
        cost = [[len(x) for x in rtok([" " + w for w in ws], add_special_tokens=False)["input_ids"]] if ws else []
                for ws, _ in per]
        items = sorted(((sc, pi, wi) for pi, (_, scs) in enumerate(per) for wi, sc in enumerate(scs)), key=lambda x: -x[0])
        b, out = budget, []
        for _ in range(8):
            keep, used = set(), 0
            for _, pi, wi in items:
                if used + cost[pi][wi] <= b:
                    keep.add((pi, wi))
                    used += cost[pi][wi]
            out = [" ".join(w for wi, w in enumerate(ws) if (pi, wi) in keep) for pi, (ws, _) in enumerate(per)]
            out = [p for p in out if p]
            n = sum(len(rtok(p, add_special_tokens=False)["input_ids"]) for p in out)
            if n <= budget:
                return out
            b -= max(1, n - budget)
        return out


class Recomp:
    def __init__(self, torch, dev, mid: str):
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.dev, self.mid = torch, dev, mid
        self.t = AutoTokenizer.from_pretrained(mid)
        self.m = AutoModel.from_pretrained(mid).to(dev).eval()

    def _enc(self, texts):
        out = []
        for a in range(0, len(texts), 64):
            b = self.t(texts[a:a + 64], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.dev)
            with self.torch.inference_mode():
                h = self.m(**b).last_hidden_state
            msk = b["attention_mask"].unsqueeze(-1).float()
            out.append((h * msk).sum(1) / msk.sum(1))
        return self.torch.cat(out)

    def __call__(self, question, sents):
        q = self._enc([question])[0]
        e = self._enc([s.text for s in sents])
        return [float(x) for x in (e @ q).float().cpu()]


def select_exact(tok, sents, scores, budget):
    """데스크탑 build_b_contexts.select_exact 와 같다."""
    b = budget
    kept = []
    for _ in range(8):
        kept = Z.select_by_score(sents, scores, b)
        n = sum(T.n_tokens(tok, p) for p in Z.assemble(kept))
        if n <= budget or len(kept) <= 1:
            return kept
        b -= max(1, n - budget)
    return kept


def load_llama_tok():
    from transformers import AutoTokenizer
    for mid in LLAMA_TOK:
        try:
            return AutoTokenizer.from_pretrained(mid)
        except Exception as e:  # noqa: BLE001
            print(f"[{mid}] 토크나이저 실패: {type(e).__name__}")
    raise SystemExit("Llama 토크나이저를 못 받았다")


# ----------------------------------------------------------------------------- 실행기
class Runner:
    def __init__(self, a):
        import torch
        self.a, self.torch = a, torch
        self.env = M.board_env()
        if self.env["jetson"] and not a.skip_mode_check:
            M.check_power_mode(self.env)
        self.url = f"http://127.0.0.1:{C.LLAMA_PORT}"
        self.env["server"] = M.check_server(self.url, C.LLAMA_CTX, reader.MODEL["gguf"])
        self.ptok = T.load_tokenizer(prefer_fallback=self.env["jetson"])       # 프롬프트 = 리더 토크나이저
        T.check_against_server(self.ptok, lambda p, b: M.post_json(self.url, p, b))
        self.ltok = self.ptok if C.READER_MODEL == "llama" else load_llama_tok()  # 문맥 = Llama 토크나이저 (실험 F 규칙)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dev != "cuda" and not a.allow_cpu:
            sys.exit("[멈춤] GPU(cuda) 가 안 보인다")
        self.power = Power()
        self.env.update(reader=C.READER_MODEL, power_source=self.power.src, cache_prompt_default=False)
        print(f"[전력] {self.power.src or '읽을 수 없음 — 에너지는 빈칸으로 남는다'}")
        self.guard = M.ThermalGuard(80, 65)
        self.models: dict = {}
        from sentence_transformers import SentenceTransformer
        self.bge = SentenceTransformer(C.EMBED_MODEL_ID, device=self.dev)
        self.ref = json.loads(REF.read_text(encoding="utf-8")) if REF.exists() else {}

    def sync(self):
        if self.dev == "cuda":
            self.torch.cuda.synchronize()

    def free(self, *names):
        for n in names:
            self.models.pop(n, None)
        import gc
        gc.collect()
        if self.dev == "cuda":
            self.torch.cuda.empty_cache()

    def model(self, name: str, set_: str = "ship"):
        key = name if name != "recomp" else f"recomp:{RECOMP_ID[set_]}"
        if key not in self.models:
            if name == "prov":
                self.models[key] = Z.ProvenceScorer()
            elif name == "recomp":
                self.free(*[k for k in self.models if k.startswith("recomp:")])
                self.models[key] = Recomp(self.torch, self.dev, RECOMP_ID[set_])
            elif name == "llml2":
                self.models[key] = Lingua2(self.torch, self.dev)
        return self.models[key]

    def qvec(self, question: str):
        return self.bge.encode([C.BGE_QUERY_PREFIX + question], normalize_embeddings=True,
                               convert_to_numpy=True)[0].astype("float32")

    def sent_scores(self, question: str, sents, qv=None):
        """즉석 bge 문장 점수 (공개 QA · 통제 문맥 — 데스크탑 BgeScorer 와 같은 방식)."""
        qv = self.qvec(question) if qv is None else qv
        e = self.bge.encode([s.text for s in sents], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        return (e.astype("float32") @ qv).tolist()

    def generate(self, passages: list[str], question: str, cache_prompt: bool = False) -> dict:
        t2 = time.perf_counter()
        prompt = T.chat_ids(self.ptok, reader.build_messages(question, passages))
        t3 = time.perf_counter()
        r = M.post_json(self.url, "/completion", {"prompt": prompt, "n_predict": reader.MAX_NEW_TOKENS, "temperature": 0,
                                                  "top_k": 1, "cache_prompt": cache_prompt, "seed": 0})
        t4 = time.perf_counter()
        tm = r.get("timings") or {}
        return {"pred": r["content"].strip(), "tok_ms": (t3 - t2) * 1000, "prefill_ms": tm.get("prompt_ms") or 0.0,
                "decode_ms": tm.get("predicted_ms") or 0.0, "decode_n": tm.get("predicted_n"), "prompt_n": tm.get("prompt_n"),
                "tokens_cached": r.get("tokens_cached"), "prompt_tokens": len(prompt), "http_ms": (t4 - t3) * 1000, "t_end": t4}

    def row(self, job: str, q: dict, cond: str, passages: list[str], g: dict, t0: float | None, retr_ms: float | None,
            comp_ms: float | None, timed: bool, extra: dict | None = None) -> dict:
        sc = score(g["pred"], q["answer"], q.get("answer_variants"))
        total = None
        if timed:
            total = round((retr_ms or 0) + (comp_ms or 0) + g["tok_ms"] + g["prefill_ms"] + g["decode_ms"], 2)
        ej, w = self.power.energy(t0, g["t_end"]) if (timed and t0 is not None) else (None, None)
        refset = "ship" if job in ("ship", "qwen") else job
        rf = self.ref.get(refset, {}).get(q["qid"], {}).get(cond, {}) if job != "qwen" else {}
        r = {"qid": q["qid"], "job": job, "condition": cond, "answer_type": q.get("answer_type"), "pred": g["pred"], **sc,
             "timed": timed, "retr_ms": None if retr_ms is None else round(retr_ms, 2),
             "comp_ms": None if comp_ms is None else round(comp_ms, 2), "tok_ms": round(g["tok_ms"], 2),
             "prefill_ms": round(g["prefill_ms"], 2), "decode_ms": round(g["decode_ms"], 2), "decode_n": g["decode_n"],
             "prompt_n": g["prompt_n"], "tokens_cached": g["tokens_cached"], "total_ms": total,
             "http_ms": round(g["http_ms"], 2), "energy_j": ej, "power_w": w, "prompt_tokens": g["prompt_tokens"],
             "ctx_tokens": sum(T.n_tokens(self.ltok, p) for p in passages), "passages": passages,
             "answer_kept": answer_kept(q, passages),
             "ctx_same_desktop": (h12(passages) == rf["ctx"]) if "ctx" in rf else None,
             "pred_same_desktop": (g["pred"] == rf["pred"]) if "pred" in rf else None, "em_desktop": rf.get("em"),
             "temp_max_c": M.temp_max_c(), "mem_available_mb": M.mem_available_mb()}
        if q.get("evidence"):
            rec = evidence_recall(q["evidence"], passages)
            r.update(evidence_recall=round(rec, 3), evidence_kept=rec >= 0.8)
        if extra:
            r.update(extra)
        return r

    # ------------------------------------------------------------------ 공통 반복 (이어 돌리기 · 몸풀기 · 과열 · 메모리)
    def loop(self, job: str, qa: list[dict], conds_of, do_question) -> str | None:
        RES.mkdir(exist_ok=True)
        tag = f"_test{self.a.limit}" if self.a.limit else ""
        out = RES / f"paper_{job}_1103{tag}.jsonl"
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
        t_start = time.time()
        stopped = None
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
                if self.a.block_minutes and el >= self.a.block_minutes and i < len(todo):
                    stopped = f"블록 {self.a.block_minutes:.0f}분"
                    break
        summarize(out)
        return stopped

    # ------------------------------------------------------------------ ship
    def job_ship(self) -> str | None:
        import numpy as np
        t_prep = time.perf_counter()
        qa = [json.loads(l) for l in (ROOT / "qa" / "qa_s1000.jsonl").open(encoding="utf-8")]
        sub300 = {json.loads(l)["qid"] for l in (ROOT / "qa" / "qa_s1000_sub300.jsonl").open(encoding="utf-8")}
        if self.a.limit:
            qa = qa[: self.a.limit]
            sub300 |= {q["qid"] for q in qa}
        chunks = {c["chunk_id"]: c for c in map(json.loads, C.CHUNKS_JSONL.open(encoding="utf-8"))}
        cemb = np.load(C.EMB_NPY).astype("float32")
        cids = json.loads(C.EMB_IDS.read_text(encoding="utf-8"))
        semb = np.load(C.OUT / f"sent_emb_{C.CORPUS}.npy")
        meta = {m["chunk_id"]: m for m in map(json.loads, (C.OUT / f"sent_meta_{C.CORPUS}.jsonl").open(encoding="utf-8"))}
        crit = np.zeros(semb.shape[0], dtype="float32")                       # CSP 표시 (질문과 무관 → 사전 계산)
        for m in meta.values():
            for i, s in enumerate(m["sents"]):
                crit[m["start"] + i] = 1.0 if csp.is_critical(s) else 0.0
        trunc = {cid: reader.truncate_passage(self.ltok, chunks[cid]["text"])[0] for cid in cids}
        for name in METHODS[1:]:
            self.model(name, "ship")
        prep_s = time.perf_counter() - t_prep
        # P5 — 보드 검색 Recall@5 · @10 (정답 청크가 상위 k 안에 든 비율)
        qv_all = self.bge.encode([C.BGE_QUERY_PREFIX + q["question"] for q in qa], batch_size=64, normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")
        top = np.argsort(-(qv_all @ cemb.T), axis=1)[:, :10]
        rec = {k: round(float(np.mean([q["chunk_id"] in {cids[j] for j in top[i, :k]} for i, q in enumerate(qa)])), 4)
               for k in (1, 2, 5, 10)}
        self.env.update(ship_prep_s=round(prep_s, 1), recall_at=rec, n_crit_sents=int(crit.sum()))
        print(f"[ship] 사전 준비 {prep_s:.0f}초 · 보드 검색 Recall@k {rec} · MemAvailable {M.mem_available_mb()}MB")

        def conds_of(q):
            cs = ["k10_B250_bge_pc", "k10_B250_bge"] + [f"k10_B{b}_{m}" for m in METHODS for b in (250, 100, 500)]
            cs += ["k1_nocomp", "k2_nocomp"] + (["k5_nocomp", "k10_nocomp"] if q["qid"] in sub300 else [])
            return cs

        def retrieve(qtext, k):
            self.sync()
            t0 = time.perf_counter()
            qv = self.qvec(qtext)
            ids = [cids[j] for j in np.argsort(-(cemb @ qv))[:k]]
            return t0, qv, ids, (time.perf_counter() - t0) * 1000

        def sents_of(ids):
            sents, rows = [], []
            for r, cid in enumerate(ids):
                m = meta[cid]
                for i, (s, n) in enumerate(zip(m["sents"], m["n_tokens"])):
                    sents.append(Z.Sent(r, i, s, n))
                    rows.append(m["start"] + i)
            return sents, rows

        def do_question(q, skip, waited):
            out = []
            conds = [c for c in conds_of(q) if c not in skip]
            first = True

            def emit(c, passages, g, t0, retr, comp, timed, extra=None):
                nonlocal first
                if waited is None:                       # 몸풀기
                    return
                r = self.row("ship", q, c, passages, g, t0, retr, comp, timed, extra)
                r["paused_s_before"] = round(waited, 1) if first else 0.0
                first = False
                out.append(r)

            # E4 (문항마다 맨 처음 — 앞 요청은 다른 문항이라 공통 지시문·예시까지만 겹친다) · P4
            for c, pc in (("k10_B250_bge_pc", True), ("k10_B250_bge", False)):
                if c not in conds:
                    continue
                t0, qv, ids, retr = retrieve(q["question"], 10)
                t1 = time.perf_counter()
                sents, rows = sents_of(ids)
                sc = (semb[rows].astype("float32") @ qv).tolist()
                kept = select_exact(self.ltok, sents, sc, 250)
                passages = Z.assemble(kept)
                comp = (time.perf_counter() - t1) * 1000
                emit(c, passages, self.generate(passages, q["question"], cache_prompt=pc), t0, retr, comp, True,
                     {"cache_prompt": pc})
            # 표 2: 250 은 시간 측정, 100 · 500 은 같은 점수로 정확도만
            for m in METHODS:
                if not any(f"k10_B{b}_{m}" in conds for b in (250, 100, 500)):
                    continue
                t0, qv, ids, retr = retrieve(q["question"], 10)
                passages10 = [trunc[c] for c in ids]
                sents, rows = sents_of(ids)
                self.sync()
                t1 = time.perf_counter()
                if m == "csp":
                    sc = ((semb[rows].astype("float32") @ qv) + csp.LAMBDA * crit[rows]).tolist()
                elif m == "llml2":
                    per = self.model("llml2").prepare(passages10)
                elif m == "prov":
                    sc = self.model("prov")(q["question"], sents)
                else:
                    sc = self.model("recomp", "ship")(q["question"], sents)
                self.sync()
                score_ms = (time.perf_counter() - t1) * 1000
                for b in (250, 100, 500):
                    c = f"k10_B{b}_{m}"
                    if c not in conds:
                        continue
                    ts = time.perf_counter()
                    passages = Lingua2.select(per, b, self.ltok) if m == "llml2" else Z.assemble(select_exact(self.ltok, sents, sc, b))
                    sel_ms = (time.perf_counter() - ts) * 1000
                    timed = b == 250
                    tstart = t0 if timed else time.perf_counter()
                    g = self.generate(passages, q["question"])
                    emit(c, passages, g, tstart if timed else None, retr if timed else None, score_ms + sel_ms if timed else None,
                         timed, {"score_ms": round(score_ms, 2), "select_ms": round(sel_ms, 2)})
            # E1 · E2: 무압축 k 청크
            for k in (1, 2, 5, 10):
                c = f"k{k}_nocomp"
                if c not in conds:
                    continue
                t0, qv, ids, retr = retrieve(q["question"], k)
                passages = [trunc[x] for x in ids]
                emit(c, passages, self.generate(passages, q["question"]), t0, retr, 0.0, True,
                     {"retrieval_hit": q["chunk_id"] in ids})
            return out

        stopped = self.loop("ship", qa, conds_of, do_question)
        self.free("prov", "llml2", *[k for k in self.models if k.startswith("recomp:")])
        return stopped

    # ------------------------------------------------------------------ pub
    def job_pub(self) -> str | None:
        import numpy as np
        stopped = None
        for set_ in ("hotpot", "nq"):
            qa = [json.loads(l) for l in (ROOT / "qa" / ("qa_hotpot_v1.jsonl" if set_ == "hotpot" else "qa_nq_litm500.jsonl"))
                  .open(encoding="utf-8")]
            if self.a.limit:
                qa = qa[: self.a.limit]
            t_prep = time.perf_counter()
            if set_ == "hotpot":
                chunks = [json.loads(l) for l in (C.OUT / "chunks_hotpot.jsonl").open(encoding="utf-8")]
                emb_p = C.OUT / "chunk_emb_hotpot_board.npy"
                if emb_p.exists():
                    cemb = np.load(emb_p)
                else:                                       # 색인 = build_index.py 와 같은 방식 (문서 쪽은 접두어 없음)
                    cemb = self.bge.encode([c["text"] for c in chunks], batch_size=32, normalize_embeddings=True,
                                           convert_to_numpy=True).astype("float32")
                    np.save(emb_p, cemb)
                ctext = {c["chunk_id"]: c["text"] for c in chunks}
                cids = [c["chunk_id"] for c in chunks]
                dref = json.loads((C.OUT / "retrieval_top10_qa_hotpot_400.json").read_text(encoding="utf-8"))
            else:
                docs = json.loads((C.OUT / "nq_litm500_docs.json").read_text(encoding="utf-8"))
            for name in ("prov", "llml2"):
                self.model(name)
            self.model("recomp", set_)
            self.env[f"{set_}_prep_s"] = round(time.perf_counter() - t_prep, 1)
            conds = ["k10_nocomp"] + [f"k10_B250_{m}" for m in ("bge", "csp", "prov", "recomp", "llml2")]

            def do_question(q, skip, waited, set_=set_):
                out = []
                extra = {}
                if set_ == "hotpot":
                    qv = self.qvec(q["question"])
                    ids = [cids[j] for j in np.argsort(-(cemb @ qv))[:10]]
                    raw = [ctext[c] for c in ids]
                    extra["retrieval_same_desktop"] = ids == [t["chunk_id"] for t in dref.get(q["qid"], {}).get("top", [])[:10]]
                else:
                    qv = None
                    raw = docs[q["qid"]][:10]
                base = [reader.truncate_passage(self.ltok, p)[0] for p in raw]
                sents = Z.split_passages(self.ltok, base)
                scores = {}
                for c in conds:
                    if c in skip:
                        continue
                    if c.endswith("nocomp"):
                        passages = base
                    else:
                        m = c.split("_")[2]
                        if m == "llml2":
                            passages = Lingua2.select(self.model("llml2").prepare(base), 250, self.ltok)
                        else:
                            key = "bge" if m in ("bge", "csp") else m
                            if key not in scores:
                                if key == "bge":
                                    scores[key] = self.sent_scores(q["question"], sents, qv)
                                elif key == "prov":
                                    scores[key] = self.model("prov")(q["question"], sents)
                                else:
                                    scores[key] = self.model("recomp", set_)(q["question"], sents)
                            sc = csp.adjust(scores["bge"], [s.text for s in sents]) if m == "csp" else scores[key]
                            passages = Z.assemble(select_exact(self.ltok, sents, sc, 250))
                    g = self.generate(passages, q["question"])
                    if waited is not None:
                        out.append(self.row(set_, q, c, passages, g, None, None, None, False, dict(extra, set=set_)))
                return out

            stopped = self.loop(set_, qa, lambda q: conds, do_question)
            if stopped:
                return stopped
        self.free("prov", "llml2", *[k for k in self.models if k.startswith("recomp:")])
        return stopped

    # ------------------------------------------------------------------ ctrl
    def job_ctrl(self) -> str | None:
        qa = [json.loads(l) for l in (ROOT / "qa" / "qa_s1000.jsonl").open(encoding="utf-8")]
        if self.a.limit:
            qa = qa[: self.a.limit]
        inp = defaultdict(dict)
        for r in map(json.loads, (C.OUT / "ctrl_inputs_1103.jsonl").open(encoding="utf-8")):
            inp[r["qid"]][r["L"]] = r
        cells = [f"L{L}_B{B}_{m}" for L in CTRL_LS for B in CTRL_BS if B < L for m in ("bge", "csp")]
        conds = ["L4000_nocomp"] + cells + ["L2000_B250_csp20"]

        def do_question(q, skip, waited):
            out = []
            qv = self.qvec(q["question"])
            for L in CTRL_LS:
                want = [c for c in conds if c.startswith(f"L{L}_") and c not in skip]
                if not want:
                    continue
                x = inp[q["qid"]][L]
                ps = x["passages"]
                ex = {"L": L, "gold_pos": x["gold_pos"], "orig_tokens": x["orig_tokens"]}
                sents = Z.split_passages(self.ltok, ps)
                texts = [s.text for s in sents]
                bge_sc = self.sent_scores(q["question"], sents, qv)
                for c in want:
                    if c.endswith("nocomp"):
                        passages, ex2 = ps, dict(ex)
                    else:
                        _, b, m = c.split("_")
                        lam = {"bge": 0.0, "csp": csp.LAMBDA, "csp20": 0.2}[m]
                        sc = csp.adjust(bge_sc, texts, lam) if lam else bge_sc
                        kept = select_exact(self.ltok, sents, sc, int(b[1:]))
                        passages = Z.assemble(kept)
                        ex2 = dict(ex, B=int(b[1:]), method=m, n_critical_kept=sum(csp.is_critical(s.text) for s in kept),
                                   gold_sent_kept=any(s.chunk_rank == x["gold_pos"] for s in kept))
                    g = self.generate(passages, q["question"])
                    if waited is not None:
                        out.append(self.row("ctrl", q, c, passages, g, None, None, None, False, ex2))
            return out

        return self.loop("ctrl", qa, lambda q: conds, do_question)

    # ------------------------------------------------------------------ qwen (E3)
    def job_qwen(self) -> str | None:
        import numpy as np
        if C.READER_MODEL != "qwen2.5-3b":
            sys.exit("[멈춤] --job qwen 은 `source jetson/board_env.sh qwen2.5-3b` 와 Qwen 서버에서")
        qa = [json.loads(l) for l in (ROOT / "qa" / "qa_s1000.jsonl").open(encoding="utf-8")]
        if self.a.limit:
            qa = qa[: self.a.limit]
        chunks = {c["chunk_id"]: c for c in map(json.loads, C.CHUNKS_JSONL.open(encoding="utf-8"))}
        cemb = np.load(C.EMB_NPY).astype("float32")
        cids = json.loads(C.EMB_IDS.read_text(encoding="utf-8"))
        semb = np.load(C.OUT / f"sent_emb_{C.CORPUS}.npy")
        meta = {m["chunk_id"]: m for m in map(json.loads, (C.OUT / f"sent_meta_{C.CORPUS}.jsonl").open(encoding="utf-8"))}
        trunc = {cid: reader.truncate_passage(self.ltok, chunks[cid]["text"])[0] for cid in cids}
        conds = ["k5_nocomp", "k10_B250_bge"]

        def do_question(q, skip, waited):
            out = []
            for c in conds:
                if c in skip:
                    continue
                k = 5 if c.startswith("k5") else 10
                self.sync()
                t0 = time.perf_counter()
                qv = self.qvec(q["question"])
                ids = [cids[j] for j in np.argsort(-(cemb @ qv))[:k]]
                retr = (time.perf_counter() - t0) * 1000
                t1 = time.perf_counter()
                if c.endswith("nocomp"):
                    passages = [trunc[x] for x in ids]
                else:
                    sents, rows = [], []
                    for r, cid in enumerate(ids):
                        m = meta[cid]
                        for i, (s, n) in enumerate(zip(m["sents"], m["n_tokens"])):
                            sents.append(Z.Sent(r, i, s, n))
                            rows.append(m["start"] + i)
                    sc = (semb[rows].astype("float32") @ qv).tolist()
                    passages = Z.assemble(select_exact(self.ltok, sents, sc, 250))
                comp = (time.perf_counter() - t1) * 1000
                g = self.generate(passages, q["question"])
                if waited is not None:
                    out.append(self.row("qwen", q, c, passages, g, t0, retr, comp, True))
            return out

        return self.loop("qwen", qa, lambda q: conds, do_question)


# ----------------------------------------------------------------------------- 요약
def summarize(path: Path) -> None:
    rows = [r for r in map(json.loads, path.open(encoding="utf-8")) if not r.get("_env")]
    g = defaultdict(list)
    for r in rows:
        g[r["condition"]].append(r)
    med = lambda xs: round(statistics.median(xs), 1) if xs else None
    p90 = lambda xs: round(sorted(xs)[int(0.9 * (len(xs) - 1))], 1) if xs else None
    rate = lambda xs: round(sum(1 for x in xs if x) / len(xs), 3) if xs else None
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    summ = {}
    print(f"\n{path.name}")
    print(f"{'조건':<18}{'n':>5}{'EM':>7}{'수치식별':>9}{'F1':>7}{'응답ms':>8}{'p90':>8}{'압축ms':>8}{'에너지J':>8}"
          f"{'문맥같음':>9}{'답같음':>8}{'데탑EM':>8}")
    for c in sorted(g):
        rs = g[c]
        ni = [r for r in rs if r["answer_type"] in NI]
        tm = [r["total_ms"] for r in rs if r.get("total_ms") is not None]
        s = {"n": len(rs), "em": mean([r["em"] for r in rs]), "em_ni": mean([r["em"] for r in ni]) if ni else None,
             "f1": mean([r["f1"] for r in rs]), "contains": mean([r["contains"] for r in rs]),
             "total_ms_median": med(tm), "total_ms_p90": p90(tm),
             "comp_ms_median": med([r["comp_ms"] for r in rs if r.get("comp_ms") is not None]),
             "prefill_ms_median": med([r["prefill_ms"] for r in rs]),
             "energy_j_median": med([r["energy_j"] for r in rs if r.get("energy_j") is not None]),
             "ctx_tokens_median": med([r["ctx_tokens"] for r in rs]),
             "answer_kept": rate([r["answer_kept"] for r in rs]),
             "evidence_kept": rate([r["evidence_kept"] for r in rs if "evidence_kept" in r]),
             "ctx_same_desktop": rate([r["ctx_same_desktop"] for r in rs if r.get("ctx_same_desktop") is not None]),
             "pred_same_desktop": rate([r["pred_same_desktop"] for r in rs if r.get("pred_same_desktop") is not None]),
             "em_desktop": mean([r["em_desktop"] for r in rs if r.get("em_desktop") is not None]),
             "prompt_n_median": med([r["prompt_n"] for r in rs if r.get("prompt_n") is not None]),
             "temp_max_c": max((r["temp_max_c"] or 0) for r in rs),
             "mem_available_min_mb": min((r["mem_available_mb"] or 1e9) for r in rs)}
        summ[c] = s
        fm = lambda x, f="": "-" if x is None else format(x, f)
        print(f"{c:<18}{s['n']:>5}{fm(s['em'], '.3f'):>7}{fm(s['em_ni'], '.3f'):>9}{fm(s['f1'], '.3f'):>7}"
              f"{fm(s['total_ms_median']):>8}{fm(s['total_ms_p90']):>8}{fm(s['comp_ms_median']):>8}{fm(s['energy_j_median']):>8}"
              f"{fm(s['ctx_same_desktop']):>9}{fm(s['pred_same_desktop']):>8}{fm(s['em_desktop'], '.3f'):>8}")
    sp = path.with_name(path.stem + "_summary.json")
    sp.write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {sp}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=["ship", "pub", "ctrl", "all", "qwen"])
    ap.add_argument("--limit", type=int, default=0, help="앞 N 문항만 (시험)")
    ap.add_argument("--block-minutes", type=float, default=0, help="0 이면 멈추지 않고 끝까지 (과열 · 메모리 보호는 그대로)")
    ap.add_argument("--skip-mode-check", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    run = Runner(a)
    if a.job == "qwen":
        jobs = ["qwen"]
    else:
        jobs = ["ship", "pub", "ctrl"] if a.job == "all" else [a.job]
        if C.READER_MODEL != "llama":
            sys.exit("[멈춤] ship · pub · ctrl 은 Llama 서버에서 (source jetson/board_env.sh)")
    if run.power.src:
        run.env["idle_power_w"] = run.power.idle_w(10)
        print(f"[전력] 대기 전력 {run.env['idle_power_w']} W (서버 · 모델 적재 상태, 10초 평균)")
    for j in jobs:
        stopped = getattr(run, f"job_{j}")()
        if stopped:
            print(f"\n[{stopped}] 에서 멈춤. 서버를 다시 띄우고 같은 명령으로 이어서")
            return
    print("\n[끝] 모든 작업 완료")


if __name__ == "__main__":
    main()
