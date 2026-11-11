# Precomputed Sentence Selection (PSS) for On-Device RAG

Anonymous code and data for the submission "Query-Aware Context Compression with
Precomputed Sentence Embeddings for On-Device Retrieval-Augmented Generation".

PSS embeds every sentence of the corpus once with the retrieval encoder
(bge-base-en-v1.5). At query time the device scores the sentences of the retrieved
chunks by inner product with the question vector, keeps the best ones under a token
budget, and restores their original order before the reader sees them.

Code comments and log messages are in Korean.

## Contents

| Path | What it is |
|---|---|
| `qa/qa_s1000.jsonl` | S1000, the 1,000 evaluation questions. Fields: `question`, `answer`, `answer_variants`, `evidence` (the gold sentence), `answer_type`, `chunk_id`, `doc_id`, `source` (`base` = batch 1, `new` = batch 2) |
| `qa/qa_s1000_sub300.jsonl` | 300-question stratified subset used by one job of the first run |
| `data/processed/chunks_v2.jsonl` | The corpus: 7,913 chunks from publicly released NAVSEA maintenance documents |
| `data/processed/chunk_emb_v2.npy`, `chunk_emb_ids_v2.json` | Chunk vectors for retrieval and their chunk ids |
| `data/processed/ctrl_inputs_1103.jsonl` | Controlled contexts (gold chunk among similar chunks, fixed length) |
| `src/precompute_sent_emb.py` | Offline step: splits chunks into sentences and stores one vector per sentence |
| `src/compressors.py` | Sentence splitting, budgeted selection, order restoration |
| `src/reader.py`, `src/toktools.py` | Prompt (instruction and four examples) and tokenization |
| `src/scoring.py`, `src/analyze_results.py` | Answer normalization, exact match, error classes |
| `src/build_index.py`, `src/chunking.py`, `src/config.py`, `src/devmon.py` | Chunk index, chunking, paths and settings, temperature and memory guards |
| `src/csp.py`, `src/pick_candidates.py` | Imported by the measurement scripts; not part of the method in the paper |
| `jetson/board_paper_1103.py` | Board run: trained compressors, public QA, controlled contexts, second reader |
| `jetson/board_guide_1106.py` | Board run: retrieval depth by token budget, chunk truncation, cross-encoder, PSS without precomputation, third reader |
| `jetson/board_prof_1111.py` | Board run: direct sentence retrieval, cross-encoder on public QA |
| `jetson/board_env.sh`, `jetson/start_server.sh` | Environment variables and the llama-server command |

The sentence vectors (157 MB) are not included because of the file size limit.
`src/precompute_sent_emb.py` rebuilds them in about 18 minutes on the board.
The HotpotQA and Natural Questions subsets are not included; the jobs that use them
(`--job pub`, `--job pubce`) need them prepared separately from the original datasets.

## Setup

Hardware: NVIDIA Jetson Orin NX 16 GB, JetPack 6.2.2, MAXN power mode with `jetson_clocks`.

1. Place this repository at `~/board_kit` (the shell scripts assume this path).
2. Install the Python packages in `requirements.txt` (use the PyTorch build for your JetPack).
3. Build llama.cpp (build b11680) at `~/llama.cpp`.
4. Put the 4-bit GGUF reader in `~/board_kit/models/`:
   `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (and, for the other readers,
   `Qwen2.5-3B-Instruct-Q4_K_M.gguf`, `gemma-2-2b-it-Q4_K_M.gguf`).

## Run

```bash
# 1. Offline: sentence vectors (once)
source ~/board_kit/jetson/board_env.sh
cd ~/board_kit && python3 src/precompute_sent_emb.py

# 2. Terminal A: reader server (keep it running)
bash ~/board_kit/jetson/start_server.sh

# 3. Terminal B: measurements
source ~/board_kit/jetson/board_env.sh && cd ~/board_kit
python3 jetson/board_guide_1106.py --job ship     # k x B grid and added baselines
python3 jetson/board_paper_1103.py --job ship     # trained compressors
python3 jetson/board_paper_1103.py --job ctrl     # controlled contexts
python3 jetson/board_prof_1111.py  --job sent     # direct sentence retrieval
```

Each job writes one JSON line per request to `jetson/results/` (prediction, exact match,
retrieval, compression, prefill and decoding time, energy, and the compressed context)
and a `_summary.json`. A job stopped midway resumes from where it stopped when the same
command is run again.

## Measurement rules

- Response time = retrieval + compression + prompt tokenization + prefill + decoding.
- Prompt caching is off. All conditions of a job run back to back for each question.
- One warm-up question is not recorded. The run pauses at 80 °C and resumes at 65 °C.
- Energy is the board input power (VDD_IN, INA3221) sampled at 20 Hz over each request.

## Data source

The corpus and the evidence sentences come from publicly released NAVSEA maintenance
documents (NSTM, JFMM, Standard Items, Tag-Out User's Manual).
