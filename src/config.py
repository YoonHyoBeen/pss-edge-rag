"""실험 전역 상수. 설계 문서와 1:1로 대응한다 — 여기 값을 바꾸면 설계가 바뀌는 것이다."""

from pathlib import Path

# ---------------------------------------------------------------- 경로
ROOT = Path(__file__).resolve().parent.parent   # src/ 의 한 단계 위 = 저장소 루트
DATA = ROOT / "data"
RAW = ROOT / "data" / "corpus" / "txt"   # NAVSEA Standard Item 추출 텍스트
OUT = ROOT / "data" / "processed"

# 코퍼스 전환 — 환경변수 SHIP_CORPUS=hotpot 이면 HotpotQA 코퍼스의 청크·인덱스·요약카드를 쓴다 (run_hotpot.py)
import os as _os
CORPUS = _os.environ.get("SHIP_CORPUS", "ship")
_SFX = "" if CORPUS == "ship" else f"_{CORPUS}"
CHUNKS_JSONL = OUT / f"chunks{_SFX}.jsonl"
EMB_NPY = OUT / f"chunk_emb{_SFX}.npy"
EMB_IDS = OUT / f"chunk_emb_ids{_SFX}.json"
CANDIDATES_JSONL = OUT / "qa_candidates.jsonl"

# ---------------------------------------------------------------- 청킹
# 450 "실제" 토큰. 단어수 x 1.3 근사는 규격 코드 구간에서 최대 10.8배까지 빗나갔다.
# 반드시 리더의 토크나이저로 센다.
MAX_CHUNK_TOKENS = 450
MIN_CHUNK_TOKENS = 40       # 이보다 짧은 조각은 앞 청크에 붙인다

# 리더 모델. gated 저장소라 HF 로그인이 필요하면 UNGATED_FALLBACK을 쓴다.
TOKENIZER_ID = "meta-llama/Llama-3.2-3B-Instruct"
TOKENIZER_FALLBACK = "unsloth/Llama-3.2-3B-Instruct"

# ---------------------------------------------------------------- 검색
EMBED_MODEL_ID = "BAAI/bge-base-en-v1.5"
TOP_K = 5
# bge-en 계열은 질의에 접두어를 붙이는 것이 공식 권장이다. 문서 쪽은 붙이지 않는다.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------- 평가·학습 배분
# 분할 단위는 문서(Standard Item)다. NAVSEA의 "챕터"는 문서마다 1/2/3/4라서
# 문서끼리 번호가 겹쳐 분할 단위로 쓸 수 없다.
#
# 코퍼스가 작다 (2,141청크, 107문서, 상위 소수 문서에 집중). 평가와 학습을
# 문서 단위로 완전히 가르면 둘 중 하나가 못 쓸 만큼 작아진다 — 실측으로
# 평가 420후보를 채우면 학습이 207~271청크밖에 안 남았다.
#
# 그래서 두 단계로 나눈다.
#   1. 홀드아웃 문서 — 학습에 단 한 글자도 쓰지 않는다. 여기서 뽑은 후보가
#      오염 없는 평가 구간이 된다.
#   2. 나머지 문서 — 후보로 뽑힌 청크는 평가 전용, 나머지 청크는 학습용.
#      같은 문서의 다른 절이 학습에 들어가지만 평가 청크 자체는 들어가지 않는다.
# 결과를 홀드아웃 구간과 전체로 각각 내서 두 값이 같으면 오염이 결과를
# 끌고 가지 않았다는 근거가 된다.
SPLIT_SEED = 20261014
HOLDOUT_FRACTION = 0.25      # 후보의 25%를 홀드아웃 문서에서 뽑는다
HOLDOUT_MAX_DOC_CHUNKS = 60  # 거대 문서가 홀드아웃에 들어가면 학습 청크가 통째로 날아간다

# ---------------------------------------------------------------- 후보 적격 조건
# 후보는 "근거 문장"을 하나 이상 가져야 한다. 근거 문장이 없으면 QA를 쓸 수 없고,
# 근거 보존율 지표도 잴 수 없다.
#
# PDF 추출에서 표가 산문으로 뭉개진 청크가 있다. 예: NSTM-075 토크표가
#   "UNF 50 ft-lbs 109 ft-lbs 143 ft-lbs ... .875 9 UNC 73 ft-lbs"
# 로 평평해진다. 숫자 신호는 가장 강하게 잡히지만 근거 문장이 없다.
# 기능어 비율로 거른다 — 뭉개진 표는 기능어가 거의 없다.
EVIDENCE_MIN_TOKENS = 8      # 이보다 짧으면 답만 있고 맥락이 없다
EVIDENCE_MAX_TOKENS = 80     # 이보다 길면 문장 분할이 실패한 것이다
EVIDENCE_MIN_STOP_RATIO = 0.15

# 챕터 1(SCOPE)·2(REFERENCES)는 제외한다. 모든 Standard Item 이 같은 형식의
# 제목 한 줄과 참고문헌 목록을 가지고 있어 QA 거리가 안 된다.
EXCLUDE_CHAPTERS = ("1", "2")

# ---------------------------------------------------------------- QA 후보
# QA 규모는 코퍼스가 정했다 (2026-10-15 실측).
#   위 적격 조건을 적용하면 적격 청크 1,042개. 문서당 중앙값은 3개뿐이고
#   상위 소수 문서에 몰려 있어 문서당 상한 16으로 전체 후보 상한이 581이다.
#   500건(후보 700)은 불가능하고, 전부 평가에 쓰면 학습용 문서가 0이 된다.
N_QA_FINAL = 500            # 답 유형 4종 x 125건
N_CANDIDATES = 800          # 500건을 추린다. 중복·표 제외 손실을 감안한 여유분.
QA_BATCH = 50               # 작성 배치 크기. 후보는 배치 단위로 균형이 맞게 나온다
MAX_QA_PER_DOC = 36         # 상한 924. 상위 3문서 집중도는 12%로 여전히 낮다.
                            # 한 문서에서 너무 많이 뽑으면 문항끼리 어휘·구조가 겹쳐
                            # 독립 표본이 아니게 되고, 부트스트랩·McNemar 신뢰구간이
                            # 실제보다 좁아진다. 20에서 700개가 정확히 채워지고
                            # 문서 110개에 퍼진다(상위 3문서 집중도 9%).
ANSWER_TYPES = ["numeric_unit", "identifier", "procedure_step", "conditional"]
PER_TYPE_CANDIDATES = N_CANDIDATES // len(ANSWER_TYPES)

# 학습 데이터 — 평가와 규칙이 다르다
# "한 청크 한 문항"은 평가 세트의 규칙이다(문항 간 상관을 막기 위해).
# 학습 데이터는 검수하지 않으므로 한 청크에서 여러 문항을 뽑아도 된다.
# 학습 풀이 약 900청크라 2문항이면 1,800건 — QLoRA에 충분하다.
TRAIN_QA_PER_CHUNK = 2

# ---------------------------------------------------------------- 리더 실행 (10-16 결정)
# 데스크탑 정확도도 보드와 같은 GGUF Q4_K_M 을 llama.cpp 로 돌린다 (계획서 6절: 전 조건 Q4_K_M 하나).
# GGUF 는 내려받지 않고 HF 가중치에서 직접 변환한다 — 파인튜닝 모델과 같은 변환 경로여야 비교가 공정하다.
# 같은 llama.cpp 빌드 번호를 보드에서도 써야 답이 맞는다 (tools/llama.cpp/VERSION).
LLAMA_CPP_DIR = ROOT / "tools" / "llama.cpp"            # Windows CUDA 바이너리 (llama-server.exe 등)
LLAMA_CPP_SRC = ROOT / "tools" / "llama.cpp-src"        # 같은 태그의 소스 (convert_hf_to_gguf.py)
MODELS_DIR = ROOT / "models"
GGUF_F16 = MODELS_DIR / "Llama-3.2-3B-Instruct-f16.gguf"
GGUF_Q4 = MODELS_DIR / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
LLAMA_CTX = int(_os.environ.get("SHIP_LLAMA_CTX", "3072"))   # 계획서 9절 -c 3072. 실험 A·B 는 6144 (입력 4,000까지, 10-20)

# 리더 모델 (10-20 실험 F). 환경변수 SHIP_READER_MODEL 로 고른다 — 기본은 지금까지 쓴 Llama.
# 문맥(텍스트)은 모델과 무관하게 같은 것을 쓰고, 프롬프트 토큰만 각 모델 토크나이저로 만든다.
# system_role=False 인 모델(Gemma)은 chat template 이 system 을 안 받아서 SYSTEM 을 첫 user 턴 앞에 붙인다.
READER_MODELS = {
    "llama": {"hf": TOKENIZER_ID, "fallback": TOKENIZER_FALLBACK,
              "gguf": "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "f16": "Llama-3.2-3B-Instruct-f16.gguf",
              "info": "MODEL_INFO.json", "system_role": True, "port_offset": 0},
    "qwen2.5-3b": {"hf": "Qwen/Qwen2.5-3B-Instruct", "fallback": "unsloth/Qwen2.5-3B-Instruct",
                   "gguf": "Qwen2.5-3B-Instruct-Q4_K_M.gguf", "f16": "Qwen2.5-3B-Instruct-f16.gguf",
                   "info": "Qwen2.5-3B-Instruct_INFO.json", "system_role": True, "port_offset": 20},
    "gemma2-2b": {"hf": "google/gemma-2-2b-it", "fallback": "unsloth/gemma-2-2b-it",
                  "gguf": "gemma-2-2b-it-Q4_K_M.gguf", "f16": "gemma-2-2b-it-f16.gguf",
                  "info": "gemma-2-2b-it_INFO.json", "system_role": False, "port_offset": 40},
}
READER_MODEL = _os.environ.get("SHIP_READER_MODEL", "llama")
if READER_MODEL not in READER_MODELS:
    raise SystemExit(f"SHIP_READER_MODEL={READER_MODEL} — 가능한 값: {', '.join(READER_MODELS)}")
# ctx·모델이 다르면 다른 포트 — 이미 떠 있는 다른 설정의 서버에 붙지 않게 (기본 Llama·3072 는 8089 그대로)
LLAMA_PORT = (8089 if LLAMA_CTX == 3072 else 8090 + LLAMA_CTX // 1024) + READER_MODELS[READER_MODEL]["port_offset"]
if _os.environ.get("SHIP_LLAMA_PORT"):                 # 보드 등에서 포트를 직접 정할 때
    LLAMA_PORT = int(_os.environ["SHIP_LLAMA_PORT"])
CHUNK_PROMPT_MAX_TOKENS = MAX_CHUNK_TOKENS   # 프롬프트에 넣을 때 청크를 450 실토큰에서 자른다
                                             # (목차 페이지 등 450 초과 청크 152개 — 청크 ID 보존 위해 재청킹 안 함)

# ---------------------------------------------------------------- 압축기 (계획서 1·3·4절)
COMPRESS_RATE = 0.25            # ×4 고정 (4절). Provence 는 ×8 = 0.125 도 (곡선용, 뺄 순서 1순위)
PROVENCE_ID = "naver/provence-reranker-debertav3-v1"
PROVENCE_BATCH = 4              # 계획서 6절
PROVENCE_MAX_LEN = 512
LLMLINGUA2_ID = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
RANDOM_SEED = 20261016
CARDS_JSONL = OUT / f"cards_x4{_SFX}.jsonl"
