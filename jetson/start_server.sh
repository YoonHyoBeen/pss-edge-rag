#!/bin/bash
# llama-server 를 10-20 측정 설정으로 띄운다 (터미널 A 에서, 측정 끝날 때까지 켜 둔다)
#   bash ~/board_kit/jetson/start_server.sh            # Llama (기본)
#   bash ~/board_kit/jetson/start_server.sh qwen2.5-3b # 실험 F
# 옵션을 손으로 바꾸지 말 것: -c 6144 -np 1 -ngl 99 --cache-ram 0 (측정 스크립트가 확인하고 다르면 멈춘다)
source ~/board_kit/jetson/board_env.sh ${1:-llama}
if [ ! -f "$SHIP_GGUF" ]; then echo "모델 파일이 없다: $SHIP_GGUF"; exit 1; fi
if pgrep -a llama-server; then echo "이미 llama-server 가 떠 있다 (위 줄). 먼저 끄고(Ctrl+C 또는 kill) 다시"; exit 1; fi
cd ~/llama.cpp
exec ./build/bin/llama-server -m "$SHIP_GGUF" -ngl 99 -c 6144 -np 1 --cache-ram 0 --host 127.0.0.1 --port $SHIP_LLAMA_PORT
