"""리더(Llama-3.2-3B-Instruct) 프롬프트와 생성. **프롬프트는 여기 하나뿐이다.**

백엔드 (10-16 결정): 기본은 **llamacpp** — 보드와 같은 GGUF Q4_K_M 을 llama-server 로 돌린다.
프롬프트는 HF 토크나이저의 chat template 으로 만들고 **토큰 ID 를 그대로** 서버에 보낸다.
서버 쪽 템플릿·토크나이저 차이가 끼어들 틈을 없애고, 보드도 같은 ID 를 보내면 같은 입력이 된다.
"hf" 백엔드(fp16 transformers)는 파인튜닝 점검용으로만 남긴다.

무문서 필터와 무압축·압축·오라클 조건이 전부 이 파일의 build_messages 를 쓴다.
무문서 조건 = 무압축 프롬프트에서 Context 블록만 뺀 것 (계획서 정의).

데스크탑은 fp16 (정확도), 보드는 같은 모델의 GGUF Q4_K_M (지연).
"""

from __future__ import annotations

import platform

import config as C

MODEL = C.READER_MODELS[C.READER_MODEL]
MODEL_ID = MODEL["hf"]
MODEL_FALLBACK = MODEL["fallback"]     # gated 접근이 안 되면 같은 가중치의 ungated 미러
import os as _os0
MAX_NEW_TOKENS = int(_os0.environ.get("SHIP_MAX_NEW", "32"))   # 계획서 9절 max_tokens 32. 실험 C(출력 길이)만 바꾼다
PROMPT_VERSION = "fs4-v1"    # 프롬프트를 바꾸면 올린다. 결과 캐시가 이 값으로 갈린다

# 10-15 무문서 필터 500건이 이 문자열로 돌았다(그때는 few-shot 없음, prompt_version 없음).
SYSTEM = ("You are a ship maintenance assistant. Answer the question with a short answer "
          "only (a number with unit, an identifier, or a short phrase). Do not explain.")

# 실험 C 선택지 (10-20, 결정 전 — 기본은 안 쓴다): S2 긴 답은 「짧게 답하라」 지시와 맞지 않아 출력 상한이
# 걸리기 전에 답을 줄일 수 있다. SHIP_PROMPT=long 이면 지시문만 바꾼다 (예시 4개는 그대로, 결과 파일 꼬리표 fs4-long-v1)
SYSTEM_LONG = ("You are a ship maintenance assistant. Answer the question using only the context. Give the complete "
               "answer, keeping every step, condition, value, unit, and identifier the question asks for, in the "
               "words of the context. Do not explain.")
import os as _os_p
if _os_p.environ.get("SHIP_PROMPT") == "long":
    SYSTEM, PROMPT_VERSION = SYSTEM_LONG, "fs4-long-v1"


# 계획서 6절 — few-shot 4개, 카테고리별 1개, 평가 세트에 들어가지 않는 문항(학습 청크)에서.
# 베이스·파인튜닝 모델 모두 같은 4개를 쓰고, 학습 입력에도 넣는다(학습·추론 불일치 방지).
# 정답은 근거 문장에서 그대로 옮긴 구절이다 — 평가 정답과 같은 작성 규칙.
FEWSHOT = [
    {"chunk_id": "009-032::3.10.6.5-3.10.8::0034", "type": "numeric_unit",
     "context": "For submarines, the start of coating application must be a maximum of 4 hours from the final surface preparation acceptance time.",
     "question": "When painting submarines, what is the maximum time from the final surface preparation acceptance time to the start of coating application?",
     "answer": "a maximum of 4 hours"},
    {"chunk_id": "009-033::3.11-3.15::0007", "type": "identifier",
     "context": "Burnish the commutator with a very fine commercial burnishing stone conforming to A-A-58052.",
     "question": "When repairing rotating electrical equipment, to which specification must the burnishing stone used on the commutator conform?",
     "answer": "A-A-58052"},
    {"chunk_id": "009-008::3.46.15-3.46.18::0026", "type": "procedure_step",
     "context": "Disconnect each inert gas/oxygen depleting (OD), fuel gas and oxygen hose line at the supply manifold at the end of each shift.",
     "question": "Under shipboard fire prevention requirements, where must each fuel gas and oxygen hose line be disconnected at the end of each shift?",
     "answer": "at the supply manifold"},
    {"chunk_id": "NSTM-583::26-26::0270", "type": "conditional",
     "context": "If a hull registry number which is only painted on the hull is removed during repairs, it shall be replaced immediately after repairs to that area are completed.",
     "question": "For Navy boats and small craft, when must a hull registry number that was removed during repairs be replaced?",
     "answer": "immediately after repairs to that area are completed"},
]


def _user(question: str, passages: list[str] | None) -> str:
    if passages is None:
        return question
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    return f"Context:\n{ctx}\n\nQuestion: {question}"


def build_messages(question: str, passages: list[str] | None = None) -> list[dict]:
    """passages=None 이면 무문서 조건. few-shot 4개는 두 경우 모두 같다 — 대상 질문의 Context 만 빠진다."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in FEWSHOT:
        msgs.append({"role": "user", "content": _user(ex["question"], [ex["context"]])})
        msgs.append({"role": "assistant", "content": ex["answer"]})
    msgs.append({"role": "user", "content": _user(question, passages)})
    if not MODEL["system_role"]:                     # Gemma: system 을 못 받는다 -> 첫 user 턴 앞에 붙인다
        msgs = [{"role": "user", "content": SYSTEM + "\n\n" + msgs[1]["content"]}] + msgs[2:]
    return msgs


BACKEND = "llamacpp"
# 결과 파일 꼬리표. 프롬프트나 백엔드가 바뀌면 다른 파일에 쌓인다 (fs4-v1 = fp16 결과, 10-16 오후 이전)
READER_TAG = f"{PROMPT_VERSION}-q4km" if BACKEND == "llamacpp" else PROMPT_VERSION
if C.READER_MODEL != "llama":                       # 실험 F — 다른 리더 결과는 다른 파일에
    READER_TAG = f"{READER_TAG}-{C.READER_MODEL}"
if MAX_NEW_TOKENS != 32:                             # 실험 C — 출력 상한이 다르면 다른 파일에
    READER_TAG = f"{READER_TAG}-out{MAX_NEW_TOKENS}"
# 파인튜닝 리더: 환경변수 SHIP_READER_FT=<이름> 이면 models/ft/<이름>-Q4_K_M.gguf 를 8092 포트에 띄우고
# 결과 파일 꼬리표에 이름을 붙인다 (베이스 결과와 섞이지 않게). 프롬프트·few-shot 은 베이스와 글자까지 같다
import os as _os
FT_NAME = _os.environ.get("SHIP_READER_FT") or None
FT_PORT = C.LLAMA_PORT + 3      # 기본 8092 (ctx 가 다르면 같이 바뀐다)
if FT_NAME:
    READER_TAG = f"{READER_TAG}-{FT_NAME}"


def load_reader(backend: str | None = None):
    if (backend or BACKEND) == "llamacpp":
        return _load_llamacpp()
    return _load_hf()


def _load_hf():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 가 안 잡힌다. fp16 3B 는 GPU에서 돌린다.")
    last = None
    for mid in (MODEL_ID, MODEL_FALLBACK):
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16, device_map="cuda")
            print(f"모델: {mid}")
            break
        except Exception as e:                       # gated / 미로그인
            print(f"[{mid}] 로드 실패: {type(e).__name__}: {str(e)[:200]}")
            last = e
    else:
        raise SystemExit(f"두 저장소 모두 실패: {last}")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    model.generation_config.max_length = None        # max_new_tokens 와 겹친다는 경고 제거
    env = {"model": mid, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "dtype": "float16", "python": platform.python_version()}
    return tok, model, env


# ---------------------------------------------------------------- llama.cpp 백엔드
class LlamaServer:
    """llama-server 를 띄우고 /completion 으로 토큰 ID 를 보낸다."""

    def __init__(self, gguf=None, port: int | None = None, ctx: int | None = None, log_name: str = "llama-server.log",
                 parallel: int = 1):
        """기본값은 평가 리더(베이스 Q4_K_M, 8089). 파인튜닝 리더·생성기는 gguf/port 를 바꿔 띄운다."""
        import atexit
        import subprocess
        import time
        import urllib.request
        gguf = gguf or (C.MODELS_DIR / MODEL["gguf"])
        port = port or C.LLAMA_PORT
        ctx = ctx or C.LLAMA_CTX
        self.gguf = gguf
        self.url = f"http://127.0.0.1:{port}"
        self.proc = None
        if not self._alive():
            exe = C.LLAMA_CPP_DIR / "llama-server.exe"
            if not exe.exists():
                exe = C.LLAMA_CPP_DIR / "llama-server"
            if not exe.exists() or not gguf.exists():
                raise SystemExit("llama-server 또는 GGUF 가 없다. setup_llamacpp.py -> make_gguf.py 를 먼저 돌려라")
            log = open(C.ROOT / "logs" / log_name, "a", encoding="utf-8")
            extra = [] if ctx == 3072 else ["--cache-ram", "0"]   # 10-20: 긴 문맥 실험은 보드처럼 호스트 프롬프트 캐시를 끈다
            self.proc = subprocess.Popen(
                [str(exe), "-m", str(gguf), "-ngl", "99", "-c", str(ctx), "-np", str(parallel),
                 "--host", "127.0.0.1", "--port", str(port), *extra], stdout=log, stderr=subprocess.STDOUT)
            atexit.register(self.close)
            for _ in range(180):
                if self._alive():
                    break
                if self.proc.poll() is not None:
                    raise SystemExit(f"llama-server 가 바로 죽었다. logs/{log_name} 를 보라")
                time.sleep(1)
            else:
                raise SystemExit(f"llama-server 가 3분 안에 안 떴다. logs/{log_name}")
        self._check_props(gguf, ctx, parallel)

    def _check_props(self, gguf, ctx: int, parallel: int) -> None:
        """이미 떠 있던 서버가 다른 모델·다른 ctx 로 떠 있으면 멈춘다 (10-20: ctx 6144 실험이 3072 서버에 붙는 사고 방지)."""
        import pathlib
        try:
            p = self.get("/props")
        except Exception:
            return                                         # 옛 빌드라 /props 가 없으면 넘어간다
        n_ctx = (p.get("default_generation_settings") or {}).get("n_ctx")
        mp = p.get("model_path")
        if n_ctx is not None and int(n_ctx) * max(1, int(p.get("total_slots") or parallel)) < ctx and int(n_ctx) != ctx:
            raise SystemExit(f"{self.url} 서버 ctx {n_ctx} != 원하는 {ctx}. 그 서버를 끄고 다시 돌려라")
        if mp and pathlib.Path(mp).name != pathlib.Path(gguf).name:
            raise SystemExit(f"{self.url} 서버 모델 {pathlib.Path(mp).name} != 원하는 {pathlib.Path(gguf).name}")
        self.props = p

    def get(self, path: str) -> dict:
        import json
        import urllib.request
        with urllib.request.urlopen(self.url + path, timeout=10) as r:
            return json.loads(r.read())

    def _alive(self) -> bool:
        import urllib.request
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def post(self, path: str, body: dict) -> dict:
        import json
        import urllib.request
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def _load_llamacpp():
    import hashlib
    import json
    import toktools
    tok = toktools.load_tokenizer()                  # 공식 -> ungated 미러 (SHIP_TOKENIZER_DIR 이면 그 폴더)
    if FT_NAME:
        gguf = C.MODELS_DIR / "ft" / f"{FT_NAME}-Q4_K_M.gguf"
        srv = LlamaServer(gguf=gguf, port=FT_PORT, log_name="llama-server-ft.log")
    else:
        gguf = C.MODELS_DIR / MODEL["gguf"]
        srv = LlamaServer(gguf=gguf, log_name=f"llama-server-{C.READER_MODEL}-c{C.LLAMA_CTX}.log"
                          if (C.READER_MODEL != "llama" or C.LLAMA_CTX != 3072) else "llama-server.log")
    # 정합성 점검: 서버가 같은 문자열을 같은 토큰으로 자르는가 (BOS 빼고)
    probe = "Within how many days must the report be submitted to the SUPERVISOR? 27°F to 110°F, MIL-PRF-32584."
    ours = tok(probe, add_special_tokens=False)["input_ids"]
    theirs = srv.post("/tokenize", {"content": probe, "add_special": False})["tokens"]
    if ours != theirs:
        raise SystemExit(f"토크나이저 불일치: HF {ours[:12]}... 서버 {theirs[:12]}...")
    info_p = (C.MODELS_DIR / "ft" / f"{FT_NAME}_INFO.json") if FT_NAME else (C.MODELS_DIR / MODEL["info"])
    info = json.loads(info_p.read_text(encoding="utf-8")) if info_p.exists() else {}
    ver_p = C.LLAMA_CPP_DIR / "VERSION"
    env = {"backend": "llamacpp", "model": gguf.name, "gguf_sha256": info.get("q4_k_m_sha256") or info.get("sha256"),
           "ft": FT_NAME,
           "llama_cpp": ver_p.read_text(encoding="utf-8").split()[0] if ver_p.exists() else None,
           "ctx": C.LLAMA_CTX, "tokenizer": tok.name_or_path, "python": platform.python_version()}
    print(f"리더: llama-server {env['llama_cpp']} / {env['model']} / ctx {C.LLAMA_CTX}")
    return tok, srv, env


def truncate_passage(tok, text: str, max_tokens: int = C.CHUNK_PROMPT_MAX_TOKENS) -> tuple[str, bool]:
    """청크를 450 실토큰에서 자른다 (계획서 5절). 잘렸는지도 돌려준다."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text, False
    return tok.decode(ids[:max_tokens]), True


def prompt_text(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(tok, model, batch_messages: list[list[dict]]) -> tuple[list[str], list[int]]:
    """greedy. (답, 프롬프트 토큰 수) 를 돌려준다. model 이 LlamaServer 면 llama.cpp 로."""
    if isinstance(model, LlamaServer):
        preds, ns = [], []
        for m in batch_messages:
            ids = tok.apply_chat_template(m, tokenize=True, add_generation_prompt=True)
            if hasattr(ids, "keys"):                      # 일부 버전은 BatchEncoding 을 돌려준다
                ids = ids["input_ids"]
            if len(ids) + MAX_NEW_TOKENS > C.LLAMA_CTX:
                raise RuntimeError(f"프롬프트 {len(ids)}토큰이 ctx {C.LLAMA_CTX} 를 넘는다")
            r = model.post("/completion", {"prompt": ids, "n_predict": MAX_NEW_TOKENS, "temperature": 0,
                                           "top_k": 1, "cache_prompt": True, "seed": 0})
            preds.append(r["content"].strip())
            ns.append(len(ids))
            te = r.get("tokens_evaluated")
            if te is not None and te != len(ids) and not getattr(model, "_warned", False):
                print(f"  [주의] 서버가 센 프롬프트 {te}토큰 != 보낸 {len(ids)}토큰 (BOS 중복?)")
                model._warned = True
        return preds, ns
    import torch
    prompts = [prompt_text(tok, m) for m in batch_messages]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    n_prompt = enc["attention_mask"].sum(dim=1).tolist()
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             temperature=None, top_p=None, pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)], n_prompt
