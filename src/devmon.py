"""측정 중 장비 상태 — 온도 · 메모리 · 서버 설정 확인 · 과열 일시정지 (10-20, 데스크탑·보드 공용).

  temp_max_c()        보드: /sys/class/thermal 최고값. 데스크탑(Windows): nvidia-smi GPU 온도
  mem_available_mb()  보드: /proc/meminfo MemAvailable. Windows 는 None
  ThermalGuard        80°C 를 넘으면 멈추고 65°C 아래로 내려오면 다시 시작 (기다린 시간은 기록만, 지연에 안 들어간다)
  check_server()      /props 로 ctx·모델 확인, 보드에서는 프로세스 인자에 --cache-ram 0 이 있는지 확인
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
import urllib.request
from pathlib import Path

IS_LINUX = platform.system() == "Linux"


def sh(cmd: list[str], timeout: int = 20) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return None


def mem_available_mb() -> float | None:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


def temp_max_c() -> float | None:
    best = None
    for z in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            t = int(z.read_text().strip()) / 1000
            if t < 150:                                # 고장난 센서(예: 107°C 고정값이 아닌 비정상 큰 값) 제외
                best = t if best is None else max(best, t)
        except Exception:
            pass
    if best is None and not IS_LINUX:
        out = sh(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], timeout=10)
        try:
            best = float(out.splitlines()[0]) if out else None
        except ValueError:
            best = None
    return round(best, 1) if best is not None else None


def is_jetson() -> bool:
    return Path("/etc/nv_tegra_release").exists()


class ThermalGuard:
    """질문(또는 칸) 사이에서만 부른다 — 요청 도중에는 멈추지 않는다."""

    def __init__(self, pause_c: float = 80.0, resume_c: float = 65.0, log=print, check_every_s: int = 15):
        self.pause_c, self.resume_c, self.log, self.every = pause_c, resume_c, log, check_every_s
        self.paused_s = 0.0
        self.n_pauses = 0

    def wait_if_hot(self) -> float:
        t = temp_max_c()
        if t is None or t < self.pause_c:
            return 0.0
        self.n_pauses += 1
        self.log(f"  [과열] {t}°C >= {self.pause_c}°C — {self.resume_c}°C 아래로 내려갈 때까지 쉰다")
        t0 = time.time()
        while True:
            time.sleep(self.every)
            t = temp_max_c()
            if t is None or t < self.resume_c:
                break
        waited = time.time() - t0
        self.paused_s += waited
        self.log(f"  [재개] {t}°C, {waited / 60:.1f}분 쉼")
        return waited


def get_json(url: str, path: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url + path, timeout=timeout) as r:
        return json.loads(r.read())


def post_json(url: str, path: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def server_cmdline(port: int) -> str | None:
    """같은 포트로 떠 있는 llama-server 의 실행 인자 (리눅스만)."""
    if not IS_LINUX:
        return None
    out = sh(["pgrep", "-a", "llama-server"]) or ""
    lines = [l for l in out.splitlines() if "llama-server" in l]
    for l in lines:
        if f"--port {port}" in l or f"--port={port}" in l:
            return l
    return lines[0] if len(lines) == 1 else None


def check_server(url: str, want_ctx: int, want_model: str | None = None, need_cache_ram0: bool | None = None) -> dict:
    """서버 설정이 실험 설정과 같은지. 다르면 멈춘다. 돌려준 dict 는 결과 파일 env 에 넣는다."""
    try:
        get_json(url, "/health", timeout=5)
    except Exception as e:
        raise SystemExit(f"llama-server 가 {url} 에 없다 ({e}). 먼저 서버를 띄울 것")
    info: dict = {"url": url}
    try:
        p = get_json(url, "/props")
        info.update(build_info=p.get("build_info"), model_path=p.get("model_path"),
                    n_ctx=(p.get("default_generation_settings") or {}).get("n_ctx"), total_slots=p.get("total_slots"))
    except Exception:
        p = {}
    if info.get("n_ctx") is not None and int(info["n_ctx"]) != want_ctx:
        raise SystemExit(f"서버 ctx {info['n_ctx']} != 실험 ctx {want_ctx}. 서버를 -c {want_ctx} 로 다시 띄울 것")
    if want_model and info.get("model_path") and Path(info["model_path"]).name != want_model:
        raise SystemExit(f"서버 모델 {Path(info['model_path']).name} != 실험 모델 {want_model}")
    port = int(url.rsplit(":", 1)[1].split("/")[0])
    cmd = server_cmdline(port)
    info["cmdline"] = cmd
    if need_cache_ram0 is None:
        need_cache_ram0 = is_jetson()
    if need_cache_ram0:
        if not cmd:
            raise SystemExit("llama-server 실행 인자를 못 읽었다 (pgrep). jetson/start_server.sh 로 띄웠는지 확인")
        if "--cache-ram 0" not in cmd and "--cache-ram=0" not in cmd:
            raise SystemExit("llama-server 가 --cache-ram 0 없이 떠 있다 (10-17 OOM 원인). 끄고 start_server.sh 로 다시")
    return info


def board_env() -> dict:
    return {"host": platform.node(), "python": platform.python_version(), "jetson": is_jetson(),
            "nvpmodel": sh(["nvpmodel", "-q"]) if is_jetson() else None,
            "l4t": sh(["cat", "/etc/nv_tegra_release"]) if is_jetson() else None,
            "mem_available_mb": mem_available_mb(), "temp_max_c": temp_max_c()}


def check_power_mode(env: dict, required: str = "MAXN") -> None:
    nv = env.get("nvpmodel")
    if not nv or required not in nv.splitlines()[0]:
        raise SystemExit(f"전력 모드가 {required} 가 아니다: {nv!r}\nsudo nvpmodel -m <MAXN 의 ID> && sudo jetson_clocks 후 다시")
