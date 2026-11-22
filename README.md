*This project has been created as part of the 42 curriculum by skhachat.*

# RAG against the machine

A Retrieval-Augmented Generation (RAG) system that answers questions about the
[vLLM](https://github.com/vllm-project/vllm) 0.10.1 repository. Given a natural
language question, it retrieves the most relevant passages of the vLLM source
code and documentation, then uses a small local language model
(`Qwen/Qwen3-0.6B`) to write an answer grounded in those passages.

---

## Description

vLLM is a large codebase (about 2,900 files). Searching it by hand is slow and a
language model alone does not know the details of this specific version. This
project builds the whole pipeline around that problem:

1. **Ingest** the vLLM repository (`.py`, `.md` and `.txt` files).
2. **Chunk** every file into pieces of at most 2,000 characters, along the
   natural boundaries of the file (functions and classes for Python, headings for
   Markdown).
3. **Index** the chunks with an inverted index and rank them with BM25.
4. **Retrieve** the top-*k* sources for a question. A source is a
   `(file_path, first_character_index, last_character_index)` triple.
5. **Answer** the question with Qwen3-0.6B, using the retrieved passages as the
   only context.

Everything is exposed through a command-line interface built with
[Python Fire](https://github.com/google/python-fire), and every input and output
is validated with [Pydantic](https://docs.pydantic.dev/) models.

Retrieval quality is measured with Recall@k, computed by the provided
`moulinette` binary (a retrieved source counts as correct when it lies in the
same file and its character range overlaps the reference range with an
IoU ≥ 0.05). On the public datasets this implementation reaches
**Recall@5 = 86.0 % on docs** and **82.8 % on code** (required: 80 % and 50 %).
See [Performance analysis](#performance-analysis).

---

## Instructions

### Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management
- Runtime dependencies (see `pyproject.toml`): `fire`, `pydantic`, `tqdm`,
  `torch`, `transformers`
- Development dependencies: `flake8`, `mypy`, `pytest`

`torch` is a large download (several hundred MB). It is only imported when an
answer is generated, so `index`, `search`, `search_dataset` and `evaluate` do not
load it.

### Installation

```bash
make install        # same as: uv sync
```

### Data layout

The program expects the following layout, relative to the directory it is run
from (the repository root):

```
data/
├── raw/vllm-0.10.1/...                 # the vLLM 0.10.1 repository to index
├── processed/                          # created by `index` (bm25_index.pkl)
├── datasets/
│   ├── UnansweredQuestions/            # questions only
│   └── AnsweredQuestions/              # questions with reference sources
└── output/
    ├── search_results/<DatasetScope>/
    └── search_results_and_answer/<DatasetScope>/
```

Put the vLLM repository under `data/raw/` and the public datasets under
`data/datasets/`. File paths stored in the results are relative to the repository
root (for example `data/raw/vllm-0.10.1/vllm/config.py`), so always run the
commands from the repository root.

The model `Qwen/Qwen3-0.6B` is downloaded from the Hugging Face Hub the first
time `answer` or `answer_dataset` runs (internet access required for that first
run).

### Commands

All commands are run as `uv run python -m src <command> [options]`.

| Command | Purpose | Main options (defaults) |
|---|---|---|
| `index` | Chunk `data/raw/` and build the BM25 index | `--max_chunk_size` (2000), `--raw_dir` (`data/raw`), `--output_dir` (`data/processed`), `--overlap` (200) |
| `search` | Top-*k* sources for one query | `<query>`, `--k` (5), `--index_dir` (`data/processed`) |
| `search_dataset` | Search a whole dataset, write a `StudentSearchResults` JSON | `--dataset_path`, `--k` (10), `--save_directory` (`data/output/search_results`), `--index_dir` |
| `answer` | Answer one query from the retrieved context | `<query>`, `--k` (5), `--index_dir`, `--model_name` (`Qwen/Qwen3-0.6B`) |
| `answer_dataset` | Generate answers for a search-results file, write a `StudentSearchResultsAndAnswer` JSON | `--student_search_results_path`, `--save_directory` (`data/output/search_results_and_answer`), `--model_name` |
| `evaluate` | Local Recall@1/3/5/10 against a ground-truth dataset (for your own iteration) | `--student_search_results_path`, `--dataset_path` |

Every path is a command-line argument; nothing is hard-coded.

### Full pipeline

```bash
# 1. Index the corpus once
uv run python -m src index --max_chunk_size 2000

# 2. Search a dataset (scope the output directory by dataset)
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

# 3. Score with the moulinette (ground-truth dataset second)
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000

# 4. Generate answers from the search results
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Repeat steps 2–4 with `dataset_code_public.json` for the code questions.

### Development

```bash
make lint     # flake8 + mypy (--warn-return-any --warn-unused-ignores
              #   --ignore-missing-imports --disallow-untyped-defs
              #   --check-untyped-defs)
make run      # uv run python -m src
make debug    # uv run python -m pdb -m src
make clean    # remove __pycache__ and tool caches
```

---

## Example usage

Single query:

```console
$ uv run python -m src search "How does vLLM handle tensor parallelism?" --k 3
data/raw/vllm-0.10.1/docs/configuration/optimization.md [4411:5719]
data/raw/vllm-0.10.1/docs/serving/parallelism_scaling.md [2554:4287]
data/raw/vllm-0.10.1/docs/design/arch_overview.md [3245:4365]
```

Each line is `file_path [first_character_index:last_character_index]`.

Dataset search and local evaluation:

```console
$ uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions
Searching: 100%|██████████| 99/99 [00:00<00:00, ...q/s]
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_code_public.json

$ uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json
Recall@1: 0.586 (58.6%) Recall@3: 0.778 (77.8%) Recall@5: 0.828 (82.8%) Recall@10: 0.889 (88.9%)
```

Degenerate inputs are reported as messages, never as tracebacks:

```console
$ uv run python -m src search "" --k 3
No results (empty query).
$ uv run python -m src search "hello" --k 0
No results (k must be a positive integer).
$ uv run python -m src search "zzzxqj qqqwww"
No matching sources found.
$ uv run python -m src search "hello" --index_dir /nonexistent
Error during search: Index not found at /nonexistent/bm25_index.pkl. Run 'index' first.
$ uv run python -m src index --max_chunk_size 2001
Error during indexing: max_chunk_size must be between 1 and 2000, got 2001
$ uv run python -m src search_dataset --dataset_path nope.json --k 3
Error during search_dataset: [Errno 2] No such file or directory: 'nope.json'
```

Malformed JSON and JSON with the wrong schema are caught the same way (the
latter through Pydantic validation errors).

---

## System architecture

```
                         ┌───────────────────────────── offline (index) ─────────────────────────────┐
 data/raw/vllm-0.10.1 ──►│  chunker.py            indexer.py                                          │
 (.py .md .txt)          │  AST / heading   ──►   tokenize + BM25 statistics ──► data/processed/       │
                         │  chunks ≤ 2000 chars   (inverted index)               bm25_index.pkl      │
                         └────────────────────────────────────────────────────────────────────────────┘
                                                                                      │
                         ┌───────────────────────────── online (query) ───────────────▼──────────────┐
 question ──────────────►│  retriever.py: same tokenizer ─► BM25 scoring ─► top-k MinimalSource       │
                         └───────────────┬──────────────────────────────────────┬─────────────────────┘
                                         │                                      │
                         search_dataset  ▼                       generator.py   ▼
                      StudentSearchResults JSON        read spans from disk ─► prompt ─► Qwen3-0.6B
                                         │                                      │
                          moulinette / `evaluate`                  answer_dataset ▼
                                                           StudentSearchResultsAndAnswer JSON
```

### Source layout

```
src/
├── __main__.py          # RagCLI (Python Fire): index, search, search_dataset,
│                        #   answer, answer_dataset, evaluate
├── models.py            # Pydantic models
└── rag/
    ├── chunker.py       # file → chunks (AST for Python, headings for Markdown)
    ├── indexer.py       # tokenizer + inverted BM25 index, persisted with pickle
    ├── retriever.py     # BM25Retriever: query → top-k MinimalSource
    ├── generator.py     # context building + Qwen3-0.6B answer generation
    └── evaluator.py     # local Recall@k (IoU ≥ 0.05)
```

### Data models (`src/models.py`)

| Model | Fields |
|---|---|
| `MinimalSource` | `file_path`, `first_character_index`, `last_character_index` (validated: `first ≥ 0`, `last ≥ first`) |
| `UnansweredQuestion` | `question_id` (UUID by default), `question` |
| `AnsweredQuestion` | `UnansweredQuestion` + `sources: list[MinimalSource]`, `answer` |
| `RagDataset` | `rag_questions: list[AnsweredQuestion \| UnansweredQuestion]` |
| `MinimalSearchResults` | `question_id`, `question`, `retrieved_sources` |
| `MinimalAnswer` | `MinimalSearchResults` + `answer` |
| `StudentSearchResults` | `search_results: list[MinimalSearchResults]`, `k` |
| `StudentSearchResultsAndAnswer` | `search_results: list[MinimalAnswer]`, `k` |

---

## Chunking strategy

Retrieval is only as good as its units: a chunk that is too large dilutes the
signal, one that cuts a function in half loses it. The chunker (`src/rag/chunker.py`)
therefore follows the structure of each file type, under one hard rule: **a chunk
is never longer than `max_chunk_size` characters (at most 2,000)**, because the
moulinette rejects longer sources. The size is validated (1 to 2,000).

- **Which files:** `.py`, `.md`, `.txt` (extension matched case-insensitively).
  Everything else in the repository (C++/CUDA sources, JSON, images, ...) is
  skipped, since the questions are about the Python code and the documentation.
  Files that cannot be read are skipped instead of aborting the run.
- **Python:** the file is parsed with `ast`. Each top-level class and function
  (decorators included) becomes a unit; the code between them (imports,
  constants, module docstring) is kept as its own unit, and whitespace-only gaps
  are absorbed by a neighbouring unit when the result still fits. A file that does not parse falls back to plain
  line-aware windows.
- **Markdown:** the file is split on ATX headings (`#` to `######`). `#` lines
  inside fenced code blocks are not treated as headings. Small neighbouring
  sections are merged up to the size limit.
- **Plain text:** handled like Markdown (it simply has no headings).
- **Oversized units:** a unit longer than the limit is cut into windows that end
  on line breaks where possible. Consecutive windows overlap by `--overlap`
  characters (default 200, capped at a quarter of the chunk size and snapped to
  the start of a line), so a sentence or statement cut at a boundary still
  appears whole in one of the two chunks.
- **Offsets:** `first_character_index` / `last_character_index` are character
  offsets (not bytes) into the file as Python reads it in text mode (UTF-8,
  universal newlines), with `last_character_index` exclusive. The generator reads
  the same spans back the same way. Line terminators are handled explicitly
  (`\n`, `\r\n`, `\r`) so offsets stay exact.
- **Paths:** `file_path` is stored relative to the working directory in POSIX
  form, which is what the moulinette matches against.

Result on vLLM 0.10.1 with the default settings:

| Type | Files with content | Chunks | Mean chunk size | Max |
|---|---:|---:|---:|---:|
| `.py` | 1,641 | 17,890 | 1,120 chars | 2,000 |
| `.md` | 178 | 806 | 1,249 chars | 2,000 |
| `.txt` | 28 | 105 | 1,305 chars | 2,000 |
| **Total** | **1,847** | **18,801** | **1,127 chars** | **2,000** |

(The corpus has 1,969 `.py`/`.md`/`.txt` files; 122 of them are empty and
produce no chunk.)

Known limitation: only *top-level* definitions are used as boundaries. A class
longer than 2,000 characters is cut into line-aware windows, not method by
method.

---

## Retrieval method

Retrieval uses **Okapi BM25**, implemented from scratch on top of an inverted
index (`src/rag/indexer.py`, `src/rag/retriever.py`), with no search library.

**Scoring.** For a query with terms *q*, a chunk *d* scores

```
score(d) = Σ_q  idf(q) · tf(q,d)·(k1+1) / ( tf(q,d) + k1·(1 − b + b·|d|/avgdl) )
idf(q)   = ln( (N − df(q) + 0.5) / (df(q) + 0.5) + 1 )
```

with `k1 = 1.5` and `b = 0.65`. At query time only the posting lists of the
query terms are visited, so a query takes about 8 ms on average (measured over
the 199 public questions, with `k = 10`).

**Tokenizer (shared by indexing and querying).**

- Tokens are `[A-Za-z0-9_]+`, lower-cased; one-character tokens and a short list
  of stop words (`the`, `how`, `what`, `which`, `is`, ...) are dropped.
- Identifiers are kept whole **and** split into sub-tokens: `max_num_seqs` →
  `max_num_seqs`, `max`, `num`, `seqs`; `SamplingParams` → `samplingparams`,
  `sampling`, `params`. A natural-language question ("which parameter limits the
  number of sequences?") and the code that implements it share vocabulary this
  way.
- **Path tokens.** The words of the file path (without `data/raw/` and the
  top-level `vllm-0.10.1` directory) are appended twice to every chunk of that
  file. The file name is often the best hint for a question, and many chunks of a
  large file do not repeat it in their own text.

**Index.** Chunk descriptors, document lengths, average length, posting lists
(`term → [(doc_id, tf)]`) and IDF values are stored in one pickle file,
`data/processed/bm25_index.pkl` (about 11 MB, 57,169 distinct terms, 159.7 tokens
per chunk on average, path tokens included). Loading it takes about 0.4 s. The pickle is produced and consumed locally by this
program; do not load an index file from an untrusted source.

**Answer generation** (`src/rag/generator.py`).

1. The text of each retrieved span is read back from disk (at most 5 sources).
2. The snippets are assembled into a context, each under a
   `[Source i: <path>]` header, within a 6,000-character budget.
3. The prompt is built with the model's chat template: a system message
   ("answer concisely and accurately using ONLY the provided context snippets;
   do not invent facts") and a user message containing the context and the
   question. Qwen3's "thinking" mode is disabled.
4. `Qwen/Qwen3-0.6B` generates greedily (`do_sample=False`, at most 256 new
   tokens); a leftover `<think>...</think>` block is stripped. The model runs on
   a GPU in float16 when CUDA is available, otherwise on CPU in float32.
5. Degenerate cases return an explicit message instead of an error: empty
   question, no retrievable context, empty model output.

---

## Performance analysis

### Setup

All figures below were measured on the public datasets (100 docs questions and
99 code questions) with the default parameters (`--max_chunk_size 2000`,
overlap 200, `k1 = 1.5`, `b = 0.65`), on a 2-vCPU Linux machine with Python
3.10, no GPU. The recall values come from the provided moulinette and are
identical to the project's own `evaluate` command.

### Recall@k

| Dataset | Questions | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required Recall@5 |
|---|---:|---:|---:|---:|---:|---:|
| Docs | 100 | 66.0 % | 78.0 % | **86.0 %** | 90.0 % | 80 % |
| Code | 99 | 58.6 % | 77.8 % | **82.8 %** | 88.9 % | 50 % |

### System performance

| Metric | Measured | Limit |
|---|---:|---:|
| Indexing the whole corpus (1,969 files → 18,801 chunks) | 10.8 s | 5 min |
| `search_dataset`, 100 docs questions, `k = 10` | 1.4 s | 90 s per 200 questions |
| `search_dataset`, 99 code questions, `k = 10` | 1.3 s | 90 s per 200 questions |
| Index size on disk | 11 MB | – |
| `make lint` (flake8 + mypy with the subject's flags) | clean | – |

The search timings are wall-clock times of the whole command, including Python
start-up and loading the index. Generation speed with Qwen3-0.6B was not part of
these measurements; it depends on the hardware (GPU or CPU) and on the 256-token
generation cap.

### Where the remaining errors come from

Looking at the questions missed at *k* = 5 (each question has a single
reference source):

| Dataset | Missed at *k* = 5 | Right file in the top 5, wrong span | Right file not in the top 5 |
|---|---:|---:|---:|
| Docs | 14 | 7 | 7 |
| Code | 17 | 12 | 5 |

Most code misses are therefore not "wrong file" errors: the file is found but
the returned chunk is another part of it, typically in large files with many
similar functions. Recall@10 (88.9 %) being clearly above Recall@5 (82.8 %)
confirms that the right region is often just below the cut.

### What mattered (ablation)

Each variant below was re-indexed (or re-scored) and evaluated with the same
IoU ≥ 0.05 rule on both public datasets.

| Variant | Docs R@5 | Code R@1 | Code R@5 |
|---|---:|---:|---:|
| **No path tokens** | 86.0 % | 45.5 % | 74.7 % |
| Path tokens ×1 | 86.0 % | 54.5 % | 80.8 % |
| **Path tokens ×2 (shipped)** | 86.0 % | 58.6 % | 82.8 % |
| Path tokens ×3 | 83.0 % | 59.6 % | 82.8 % |

| Chunk size (overlap 200) | Chunks | Docs R@5 | Code R@5 |
|---|---:|---:|---:|
| 500 | 57,134 | 83.0 % | 82.8 % |
| 1000 | 30,463 | 82.0 % | 84.8 % |
| 1500 | 22,330 | 83.0 % | 83.8 % |
| **2000 (shipped)** | 18,801 | 86.0 % | 82.8 % |

- Adding the path to the indexed text is the one change with a clear effect on
  code questions (+8 points Recall@5, +13 points Recall@1).
- Chunk size, overlap (0, 100, 200, 400) and the BM25 parameters
  (`k1 ∈ {1.2, 1.5, 2.0}`, `b ∈ {0.5, 0.65, 0.75}`) only move Recall@5 by a few
  points. With about 100 questions per dataset, one question is one point, so
  differences of 1–3 points are within noise; the shipped defaults were kept
  rather than tuned further to the public sets.

---

## Design decisions

- **BM25 instead of dense embeddings.** The questions are full of exact
  identifiers (class, function and parameter names), which lexical matching
  handles very well. BM25 needs no model and no GPU for retrieval, builds the
  index in about 10 seconds, answers a query in milliseconds, and its behaviour
  can be inspected term by term, which made the failures above easy to analyse.
  It is far below the 5-minute and 90-second limits.
- **Own implementation of BM25.** It is a few dozen lines, avoids an extra
  dependency, and made it possible to add code-specific tokenization and path
  tokens at index time.
- **Structure-aware chunking.** AST boundaries for Python and heading boundaries
  for Markdown keep a function or a documentation section in one piece, which is
  what a question is usually about. Chunks are capped at the maximum allowed
  size (2,000 characters) because it gave the best docs recall in the ablation
  while staying valid for the moulinette.
- **Character offsets handled explicitly.** Evaluation compares character ranges,
  so the chunker computes offsets itself from the decoded text and treats every
  line terminator explicitly instead of relying on `str.splitlines`, which
  also splits on characters such as form feed or U+2028 and would shift offsets.
- **Code-aware tokenizer.** Keeping `snake_case`/`camelCase` identifiers whole
  and split bridges the vocabulary gap between questions and code.
- **Persisted pickle index.** `index` is run once; `search` and
  `search_dataset` only load it. Pickle keeps loading fast and the code simple.
- **Lazy import of `torch`/`transformers`.** They are imported only when a
  model is needed, so retrieval commands start quickly and work without loading
  the model.
- **Pydantic everywhere at the I/O boundary.** Datasets and result files are
  validated on read and serialised from the same models on write, so the output
  always matches the schema the moulinette expects.
- **Errors reported, not raised.** Every command catches the expected failure
  modes (missing or malformed files, invalid `k` or chunk size, missing index)
  and prints a one-line message, as required for degenerate inputs.
- **Small grounded prompt for a small model.** Qwen3-0.6B has limited capacity
  and context: at most five sources, a 6,000-character context budget, an
  instruction to use only the context, thinking mode off and greedy decoding
  for reproducible, concise answers.

---

## Challenges faced

- **Exact character offsets.** The metric is computed on character ranges, so an
  off-by-one in an offset, a source of 2,001 characters (rejected; 2,000 is the
  limit, measured as `last − first`) or a different treatment of line endings
  silently costs recall. Solved by deriving every offset from one decoded string,
  handling `\n`, `\r\n` and `\r` explicitly, and validating the size limit.
- **Choosing chunk boundaries.** Naive fixed-size windows cut functions in half;
  pure AST chunks can exceed the limit. The final strategy combines structural
  units with a line-aware, overlapping fallback for oversized ones.
- **Vocabulary mismatch between questions and code.** A question says "number of
  sequences", the code says `max_num_seqs`. Identifier splitting and path tokens
  closed most of that gap; path tokens alone improved code Recall@5 from 74.7 %
  to 82.8 %.
- **Right file, wrong chunk.** In large files with many similar functions,
  BM25 ranks the right file but not always the right region (12 of the 17 code
  misses at *k* = 5). A reranking step or finer chunking of long classes would
  be the next improvement.
- **Never crashing on bad input.** Empty queries, `k = 0`, nonsensical queries,
  missing or malformed files and invalid options each needed their own handling
  path and a clear message.
- **Heavy model dependencies.** `torch` is large, so it is imported lazily; the
  retrieval pipeline stays fast and light.
- **Answer quality is bounded by retrieval.** The 0.6B model only sees the
  passages it is given and cannot compensate for a missed retrieval; this is why
  most of the effort went into retrieval quality, which is also the part that
  can be measured objectively with the moulinette.

---

## Resources

### References

- S. Robertson, H. Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond*, Foundations and Trends in Information Retrieval, 2009.
- C. D. Manning, P. Raghavan, H. Schütze, *Introduction to Information
  Retrieval*, Cambridge University Press, 2008 (inverted indexes, tf–idf, term
  weighting).
- P. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks*, NeurIPS 2020.
- Qwen Team, *Qwen3 Technical Report*, 2025, and the
  [Qwen/Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B).
- [vLLM project and documentation](https://github.com/vllm-project/vllm).
- [Python `ast` module](https://docs.python.org/3/library/ast.html) and the
  [CommonMark specification](https://spec.commonmark.org/) (ATX headings and
  fenced code blocks).
- [Python Fire](https://github.com/google/python-fire),
  [Pydantic v2](https://docs.pydantic.dev/latest/),
  [tqdm](https://tqdm.github.io/),
  [Hugging Face Transformers](https://huggingface.co/docs/transformers),
  [uv](https://docs.astral.sh/uv/).

### Use of AI

Claude (Anthropic's AI assistant) was used as a tutor and reviewer during this
project, for these tasks:

- **Planning and explanation:** breaking the subject into steps, and explaining
  concepts along the way (what the task requires, how Python Fire turns class
  methods into commands, how Pydantic validates and constructs models).
- **Project scaffolding:** help writing `pyproject.toml` and setting up the
  `uv` project.
- **Code review of the chunker:** auditing `chunker.py` for bugs and edge cases
  (line-terminator handling and character offsets, chunk-size limits, fenced code
  blocks in Markdown, fallback behaviour) and suggesting fixes.
- **This README:** drafted by Claude from the final source code and from results
  obtained by actually running the pipeline (indexing, dataset search, the
  moulinette, the ablations, the edge cases and the linters); the numbers above
  are measured values, not estimates.

AI suggestions were checked by running them (tests on real corpus files, the
moulinette, flake8 and mypy) before being kept. Retrieval quality is measured
only by the moulinette and the project's own `evaluate` command, never by AI
judgement.
