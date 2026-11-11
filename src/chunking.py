"""NAVSEA Standard Item 텍스트 → 절 단위 450 실토큰 청크.

실제 코퍼스(data/corpus/txt/*.txt) 형식에 맞춰 작성했다.

  1.   SCOPE:
     1.1   Title:   General Criteria; accomplish
     3.1      Identify materials that may contain toxic or hazardous substances as listed
  in Subpart ... (본문이 헤더 줄에서 시작해 다음 줄로 이어진다)

표는 파이프 표가 아니라 공백으로 열을 맞춘 형태다.

                 Marine Coating Removal Per Ship Class
  Ship Class                                      Square Foot of Removal
  Aircraft Carriers (CVN)                                          200

사용:
    python chunking.py                        # data/corpus/txt -> data/processed/chunks.jsonl
    python chunking.py --tokenizer naive      # 파싱만 점검 (실험 금지)
    python chunking.py --inspect 009-032      # 한 문서의 파싱 결과를 눈으로 확인
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import config as C

# ---------------------------------------------------------------- 토크나이저


class Tokenizer:
    """리더 토크나이저 래퍼. `naive`는 점검 전용이며 실험에 쓰면 안 된다."""

    def __init__(self, mode: str = "real"):
        self.mode = mode
        self._tok = None
        if mode == "real":
            self._tok = self._load_real()
            if self._tok is None:
                raise SystemExit(
                    "토크나이저를 불러오지 못했다.\n"
                    f"  1) huggingface-cli login 후 {C.TOKENIZER_ID} 접근 권한 확인\n"
                    f"  2) 또는 ungated 미러 {C.TOKENIZER_FALLBACK}\n"
                    "  3) 파싱만 점검하려면 --tokenizer naive (실험 금지)"
                )

    @staticmethod
    def _load_real():
        try:
            from transformers import AutoTokenizer
        except ImportError:
            print("[경고] transformers 미설치", file=sys.stderr)
            return None
        for mid in (C.TOKENIZER_ID, C.TOKENIZER_FALLBACK):
            try:
                tok = AutoTokenizer.from_pretrained(mid)
                print(f"[토크나이저] {mid}")
                return tok
            except Exception as e:  # noqa: BLE001
                print(f"[토크나이저] {mid} 실패: {type(e).__name__}", file=sys.stderr)
        return None

    def count(self, text: str) -> int:
        if self.mode == "naive":
            return len(re.findall(r"\w+|[^\w\s]", text))
        return len(self._tok.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------- 문서 로딩


def load_documents(raw_dir: Path) -> Iterator[tuple[str, str]]:
    files = sorted(raw_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"{raw_dir} 에 .txt 가 없다.")
    for path in files:
        yield path.stem, path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- 줄바꿈 하이픈

# PDF가 줄 끝에서 단어를 자른 자리다. 줄을 이어 붙이면 "Fahren- heit" 가 남는다.
# 그냥 붙이면 "flame- retardant" 가 "flameretardant" 가 되므로 가를 기준이 필요하다.
# 외부 사전 없이 **코퍼스 자체를 사전으로** 쓴다 — 붙인 형태가 코퍼스 다른 곳에
# 단어로 두 번 이상 나오고 하이픈 형태보다 흔하면 붙이고, 아니면 하이픈만 남긴다.
HYPHEN_BREAK = re.compile(r"\b([A-Za-z]{2,})- ([A-Za-z]{2,})\b")
_VOCAB: dict[str, int] = {}


def build_vocab(raw_dir: Path) -> None:
    import collections
    v: collections.Counter[str] = collections.Counter()
    for path in sorted(raw_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for w in re.findall(r"[A-Za-z]{2,}(?:-[A-Za-z]{2,})?", text):
            v[w.lower()] += 1
    _VOCAB.clear()
    _VOCAB.update(v)


def dehyphenate(text: str) -> str:
    if not _VOCAB:
        return text

    def fix(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        merged, hyph = (a + b).lower(), f"{a}-{b}".lower()
        if _VOCAB.get(merged, 0) >= 2 and _VOCAB.get(merged, 0) > _VOCAB.get(hyph, 0):
            return a + b
        return f"{a}-{b}"

    return HYPHEN_BREAK.sub(fix, text)


# ---------------------------------------------------------------- 절 분리

# "1.   SCOPE:" / "   1.1   Title: ..." / "        3.1.2.2   If the total costs..."
# 번호 뒤 마침표는 최상위에만 붙고, 본문이 같은 줄에서 시작한다.
SECTION_RE = re.compile(
    r"^(\s*)(\d{1,2}(?:\.\d{1,3}){0,3})\.?"
    r"(?:[ \t]{2,}|[ \t](?=[A-Z(\"]))"      # 2칸 이상, 또는 1칸 + 대문자
    r"(\S.*)$")

PAGE_NOISE_RE = re.compile(
    r"^\s*(?:\d+\s+of\s+\d+|ITEM NO:|DATE:|CATEGORY:|FY-\d+|NAVSEA|STANDARD ITEM)\s*$",
    re.I)


@dataclass
class Section:
    doc_id: str
    chapter: str
    path: str
    body: str


def split_sections(doc_id: str, text: str) -> list[Section]:
    lines = [l for l in text.splitlines() if not PAGE_NOISE_RE.match(l)]

    marks: list[tuple[int, str, str]] = []   # (줄번호, 번호, 같은 줄의 본문 시작)
    for i, line in enumerate(lines):
        m = SECTION_RE.match(line)
        if m:
            marks.append((i, m.group(2), m.group(3)))

    if not marks:
        return [Section(doc_id, "0", "0", text.strip())]

    out: list[Section] = []
    for idx, (ln, num, head_rest) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(lines)
        tail = "\n".join(lines[ln + 1:end])
        body = (head_rest + "\n" + tail).strip()
        if not body:
            continue
        out.append(Section(doc_id, num.split(".")[0], num, body))
    return out


# ---------------------------------------------------------------- 표 처리

# 왼쪽 셀 + 3칸 이상 공백 + 오른쪽 셀
COL_RE = re.compile(r"^\s*(\S.*?\S|\S)[ \t]{3,}(\S.*?)\s*$")
MAX_LEFT, MAX_RIGHT = 90, 70
MIN_TABLE_ROWS = 3


def _is_col_row(line: str) -> tuple[str, str] | None:
    if SECTION_RE.match(line):
        return None
    m = COL_RE.match(line)
    if not m:
        return None
    left, right = m.group(1).strip(), m.group(2).strip()
    if len(left) > MAX_LEFT or len(right) > MAX_RIGHT:
        return None          # 이 길이를 넘으면 표가 아니라 줄바꿈된 산문이다
    return left, right


def extract_table_rows(section: Section) -> tuple[list[str], str]:
    """공백 정렬 표를 행 단위 문장으로 바꾸고, 표를 뺀 본문을 돌려준다."""
    lines = section.body.splitlines()
    rows_out: list[str] = []
    kept: list[str] = []

    i = 0
    while i < len(lines):
        cols = _is_col_row(lines[i])
        if cols is None:
            kept.append(lines[i])
            i += 1
            continue

        # 표 블록 수집 (사이의 빈 줄 한 개까지는 허용)
        start = i
        block: list[tuple[str, str]] = []
        blanks = 0
        while i < len(lines):
            c = _is_col_row(lines[i])
            if c:
                block.append(c)
                blanks = 0
            elif not lines[i].strip() and blanks == 0 and block:
                blanks = 1
            else:
                break
            i += 1

        if len(block) < MIN_TABLE_ROWS:
            kept.extend(lines[start:i])
            continue

        # 캡션 = 표 바로 위의 비어 있지 않은 줄 (열 구분이 없는 줄)
        caption = ""
        for j in range(start - 1, max(-1, start - 4), -1):
            if j < 0:
                break
            s = lines[j].strip()
            if s and _is_col_row(lines[j]) is None:
                caption = s
                break

        (h_left, h_right), body_rows = block[0], block[1:]
        for left, right in body_rows:
            sent = (f"{caption} — " if caption else "") + \
                   f"{h_left} is {left}; {h_right} is {right}"
            if not sent.endswith("."):
                sent += "."
            rows_out.append(sent)

    return rows_out, "\n".join(kept).strip()


# ---------------------------------------------------------------- 청크 조립


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chapter: str
    section_path: str
    text: str
    n_tokens: int
    is_table_row: bool


def _sentences(text: str) -> list[str]:
    """문장 분할. Provence·bge 필터와 같은 분할기를 써야 하므로 여기서 한 번만 정의한다."""
    flat = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?:])\s+(?=[A-Z0-9(\"'])", flat)
    return [p.strip() for p in parts if p.strip()]


def pack_document(doc_id: str, secs: list[Section], tok: Tokenizer) -> list[Chunk]:
    """형제 절을 450토큰까지 묶는다.

    NAVSEA 절은 한 단락짜리가 많아 절마다 청크를 만들면 20~30토큰 조각이 쏟아진다.
    같은 최상위 챕터 안에서는 연속한 절을 이어 붙이고, 챕터가 바뀌면 끊는다.
    표 행과 산문은 서로 섞지 않는다(표 행이 근거 단위이기 때문).
    """
    chunks: list[Chunk] = []
    seq = 0

    buf: dict[bool, dict] = {
        True: {"txt": [], "path": [], "tok": 0},    # 표 행
        False: {"txt": [], "path": [], "tok": 0},   # 산문
    }

    def flush(is_row: bool):
        nonlocal seq
        b = buf[is_row]
        if not b["txt"]:
            return
        text = dehyphenate(" ".join(b["txt"]))
        paths = b["path"]
        span = paths[0] if len(paths) == 1 else f"{paths[0]}-{paths[-1]}"
        chunks.append(Chunk(
            chunk_id=f"{doc_id}::{span}::{seq:04d}",
            doc_id=doc_id,
            chapter=paths[0].split(".")[0],
            section_path=span,
            text=text,
            n_tokens=tok.count(text),
            is_table_row=is_row,
        ))
        seq += 1
        b["txt"], b["path"], b["tok"] = [], [], 0

    def add(piece: str, path: str, is_row: bool):
        t = tok.count(piece)
        b = buf[is_row]
        if b["txt"] and b["tok"] + t > C.MAX_CHUNK_TOKENS:
            flush(is_row)
        if t > C.MAX_CHUNK_TOKENS:          # 한 조각이 예산을 넘으면 단독 청크
            b["txt"], b["path"], b["tok"] = [piece], [path], t
            flush(is_row)
            return
        b["txt"].append(piece)
        b["path"].append(path)
        b["tok"] += t

    cur_chapter = None
    for s_ in secs:
        if cur_chapter is not None and s_.chapter != cur_chapter:
            flush(True)
            flush(False)
        cur_chapter = s_.chapter

        rows, prose = extract_table_rows(s_)
        for r in rows:
            add(r, s_.path, True)
        for sent in _sentences(prose):
            add(sent, s_.path, False)

    flush(True)
    flush(False)

    # 남은 아주 짧은 청크는 같은 종류의 앞 청크에 붙인다
    merged: list[Chunk] = []
    for c in chunks:
        if (merged and c.n_tokens < C.MIN_CHUNK_TOKENS
                and merged[-1].is_table_row == c.is_table_row
                and merged[-1].n_tokens + c.n_tokens <= C.MAX_CHUNK_TOKENS):
            prev = merged[-1]
            prev.text += " " + c.text
            prev.n_tokens = tok.count(prev.text)
            if not prev.section_path.endswith(c.section_path.split("-")[-1]):
                prev.section_path = f"{prev.section_path.split('-')[0]}-{c.section_path.split('-')[-1]}"
        else:
            merged.append(c)
    return merged


def build_chunks(raw_dir: Path, tok: Tokenizer) -> list[Chunk]:
    out: list[Chunk] = []
    n_doc = 0
    for doc_id, text in load_documents(raw_dir):
        out.extend(pack_document(doc_id, split_sections(doc_id, text), tok))
        n_doc += 1
    print(f"문서 {n_doc}개 처리")
    return out


# ---------------------------------------------------------------- 점검·보고


def inspect(raw_dir: Path, doc_id: str) -> None:
    path = raw_dir / f"{doc_id}.txt"
    if not path.exists():
        raise SystemExit(f"{path} 없음")
    secs = split_sections(doc_id, path.read_text(encoding="utf-8", errors="replace"))
    print(f"=== {doc_id} — 절 {len(secs)}개 ===\n")
    for s in secs[:40]:
        rows, prose = extract_table_rows(s)
        tag = f" [표 행 {len(rows)}]" if rows else ""
        print(f"  {s.path:10s}{tag}  {prose[:88]!r}")
        for r in rows[:2]:
            print(f"      표> {r[:100]}")


def report(chunks: list[Chunk]) -> None:
    if not chunks:
        print("청크가 0개다.")
        return
    toks = sorted(c.n_tokens for c in chunks)
    n = len(toks)
    rows = sum(c.is_table_row for c in chunks)
    chapters = sorted({c.chapter for c in chunks}, key=lambda x: (len(x), x))

    print("\n=== 청킹 통계 ===")
    print(f"총 청크      {n:,}")
    print(f"표 행 청크   {rows:,}  ({rows / n:.1%})")
    print(f"문서 수      {len({c.doc_id for c in chunks})}")
    print(f"챕터         {len(chapters)}개  {chapters[:12]}")
    print(f"토큰 중앙값  {toks[n // 2]}")
    print(f"토큰 p10/p90 {toks[n // 10]} / {toks[min(n - 1, 9 * n // 10)]}")
    print(f"토큰 최대    {toks[-1]}")
    print(f"40토큰 미만  {sum(t < 40 for t in toks):,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=C.RAW)
    ap.add_argument("--out", type=Path, default=C.CHUNKS_JSONL)
    ap.add_argument("--tokenizer", choices=["real", "naive"], default="real")
    ap.add_argument("--inspect", metavar="DOC_ID")
    a = ap.parse_args()

    if a.inspect:
        inspect(a.raw, a.inspect)
        return

    if a.tokenizer == "naive":
        print("[경고] naive 토크나이저는 점검 전용이다. 실험에 쓰지 마라.", file=sys.stderr)

    build_vocab(a.raw)          # 줄바꿈 하이픈 판정용 사전. 청킹보다 먼저.
    print(f"[사전] 코퍼스 어휘 {len(_VOCAB):,}종")

    chunks = build_chunks(a.raw, Tokenizer(a.tokenizer))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"{a.out} 에 {len(chunks):,}개 기록")
    report(chunks)


if __name__ == "__main__":
    main()
