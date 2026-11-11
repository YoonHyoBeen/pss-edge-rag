# 10-20 측정 공통 환경 — 측정 명령 전에 매번:  source ~/board_kit/jetson/board_env.sh [llama|qwen2.5-3b|gemma2-2b]
export SHIP_CORPUS=v2              # 합친 코퍼스 7,913조각
export SHIP_LLAMA_CTX=6144         # 입력 4,000 + 예시·질문 + 출력 256 이 들어가는 크기
export SHIP_READER_MODEL=${1:-llama}
unset SHIP_LLAMA_PORT
cd ~/board_kit
export SHIP_LLAMA_PORT=$(python3 -c "import sys; sys.path.insert(0, 'src'); import config as C; print(C.LLAMA_PORT)")
export SHIP_GGUF=$HOME/board_kit/models/$(python3 -c "import sys; sys.path.insert(0, 'src'); import config as C; print(C.READER_MODELS[C.READER_MODEL]['gguf'])")
echo "[환경] 코퍼스 $SHIP_CORPUS · ctx $SHIP_LLAMA_CTX · 리더 $SHIP_READER_MODEL · 포트 $SHIP_LLAMA_PORT"
echo "       모델 파일 $SHIP_GGUF"
