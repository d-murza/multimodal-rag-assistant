# Applied Machine Learning — Project (Assignments 8–10)

A multimodal retrieval + RAG system built up over three assignments:

- **Assignment 8 — Preprocessing:** embed WikiHow articles and HowTo100M video clips into SQLite + FAISS.
- **Assignment 9 — Chatbot:** a local TinyLlama chat loop with a structured prompt, token management, and output validation.
- **Assignment 10 — RAG:** retrieval components (`ArticleRetriever`, `VideoRetriever`) wired to the chatbot into an end-to-end pipeline, with an optional Gradio UI.

All models are used frozen (inference only) and download automatically from Hugging Face on first use. Datasets, model files, the virtual environment, and `processed-data/` are git-ignored.

---

## 1. Repository structure

```
Project/
├── Preprocessing/                      # Assignment 8
│   ├── wikihow_text_preprocessing.py   #   Task 1: WikiHow -> SQLite + 2 FAISS text indices
│   ├── howTo100M_preprocessing.py      #   Task 2: HowTo100M -> SQLite + FAISS CLIP index
│   ├── data/                           #   input datasets (git-ignored, you place them here)
│   └── processed-data/                 #   generated outputs (git-ignored, created automatically)
├── Chatbot/                            # Assignment 9
│   └── model_setup.py                  #   prompt construction + chat loop + output validation
├── Retrieval/                          # Assignment 10
│   ├── article_retriever.py            #   Task 1: ArticleRetriever
│   ├── video_retriever.py              #   Task 2: VideoRetriever
│   ├── rag_pipeline.py                 #   Task 3: end-to-end RAG pipeline (+ ChatBot wrapper)
│   └── app.py                          #   Task 3 (optional): Gradio interface
├── Assignments/                        # the assignment PDFs
├── Plan_of_interface.png               # interface sketch (Assignment 8, Task 3)
├── requirements.txt
└── README.md
```

---

## 2. Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**GPU (recommended).** The pinned `torch==2.6.0+cu124` is a CUDA build (tested on an RTX 3070).
Because the `+cu124` wheels are not on plain PyPI, install torch from the PyTorch index:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The scripts pick the GPU automatically (`cuda` if available, else CPU) and print the device on startup.
CPU also works but is much slower for the video preprocessing.

---

## 3. Datasets — where to put them

Nothing is committed; download the datasets from the links in the assignment PDFs and place them as follows:

- **WikiHow (Task 1):** `Preprocessing/data/wikihow/wikihow-cleaned.csv`
- **HowTo100M (Task 2):** under `Preprocessing/data/howto100m_bundle/`
  - `task_ids.csv`
  - `available_videos.csv`
  - `videos/<video_id>.mp4`  (the 342 video files)

Both preprocessing scripts use **relative paths**, so always run them **from inside `Preprocessing/`**.
That way outputs land in `Preprocessing/processed-data/`, which is exactly where the Assignment 10
retrievers look for them.

---

## 4. Running the pipeline

Run the steps in order the first time. Steps 1–2 (Assignment 8) create the artifacts that Assignment 10 reuses.

### Step 1 — WikiHow text preprocessing (Assignment 8, Task 1)

```powershell
cd Preprocessing
python wikihow_text_preprocessing.py
```

Reads `wikihow-cleaned.csv` in chunks, stores `title`, `summary`, `text` in SQLite, embeds with
`all-MiniLM-L6-v2` (384-d), and builds two FAISS indices (cosine via normalized inner product).
Finishes with a query-based retrieval test.

Outputs in `Preprocessing/processed-data/`:
- `wikihow.db` — table `articles` (the row `id` matches the FAISS position)
- `title_index.faiss` — index over titles only
- `combined_index.faiss` — index over title + summary

### Step 2 — HowTo100M video preprocessing (Assignment 8, Task 2)

Test on a few videos first, then run the full set by dropping `--limit-videos`:

```powershell
cd Preprocessing
python howTo100M_preprocessing.py --limit-videos 5 --query "how to boil water" # for testing purposes
python howTo100M_preprocessing.py # full videos dataset
```

Outputs in `Preprocessing/processed-data/`:
- `howto100m_clips.sqlite3`
- `howto100m_clips.faiss`

Useful flags: `--window-sec`, `--stride-sec`, `--min-clip-sec`, `--frames-per-clip`, `--batch-size`,
`--limit-videos`, `--query`, `--top-k`.

### Step 3 — Chatbot (Assignment 9)

```powershell
cd Chatbot
python model_setup.py
```

Starts the interactive TinyLlama chat loop (type `quit` to exit). See Section 6 for details.

### Step 4 — Retrieval and RAG (Assignment 10)

Run from inside `Retrieval/`. Each component can be tested on its own; the pipeline and UI combine them.

```powershell
cd Retrieval
python article_retriever.py --query "how to clean a saddle?"      # Task 1
python video_retriever.py   --query "how to cut an onion"         # Task 2
python rag_pipeline.py      --query "How to change a tire?"        # Task 3 (end-to-end)
python app.py                                                     # Task 3 (optional Gradio UI)
```

See Section 7 for the architecture.

---

## 5. Interface design (Assignment 8, Task 3)

The conceptual design is sketched in [`Plan_of_interface.png`](Plan_of_interface.png) and implemented
in `Retrieval/app.py` (Assignment 10). It has two columns plus an inspection panel:

![Interface sketch](Plan_of_interface.png)

---

## 6. Chatbot core (Assignment 9)

`Chatbot/model_setup.py` implements the three tasks and is written so retrieval can be plugged in later.

- **Task 1 — model setup and prompt structure.** TinyLlama-1.1B-Chat via Transformers. `prompt_construction()`
  assembles the prompt from a **system prompt**, a **context block**, and the **conversation history**,
  using explicit delimiters `<|system|>`, `<|context|>`, `<|user|>`, `<|assistant|>`. The context block is
  empty in Assignment 9 and is where Assignment 10 injects retrieved articles.
- **Task 2 — chat loop and conversation management.** Reads input, builds the prompt, generates, and appends
  the reply to the history (list of `{"role", "content"}` entries). Token usage is tracked with the tokenizer;
  when the prompt exceeds the budget, the oldest turns are dropped. Stopping criteria: EOS token, a configurable
  `max_new_tokens`, and role-marker handling in validation. Inference, prompt construction, and loop control are
  separate functions.
- **Task 3 — output validation and RAG preparation.** `output_validation()` strips leaked role markers and
  rejects incomplete, degenerate (low unique-word ratio), or malformed (no list structure) responses. The
  system prompt constrains the assistant to concise, structured, comment-free answers.

The reusable pieces (`system_prompt`, `prompt_construction`, `output_validation`) are importable without
side effects; running the file directly starts the chat loop. Assignment 10 imports and reuses them.

---

## 7. RAG architecture (Assignment 10)

The `Retrieval/` package reuses the Assignment 8 artifacts and the Assignment 9 chatbot core — no
re-preprocessing or re-embedding.

**Components:**
- **`ArticleRetriever`** (Task 1) — loads `all-MiniLM-L6-v2`, `wikihow.db`, and both FAISS indices. Encodes a
  query, searches the title-only and title+summary indices, and merges them with a configurable weight
  (`score = w·title + (1−w)·combined`). Resolves FAISS positions to articles via the SQLite `id`, offers a
  keyword `LIKE` fallback for weak results, and formats results into a context snippet.
- **`VideoRetriever`** (Task 2) — loads CLIP ViT-B/32 and the clip FAISS index + SQLite mapping. Encodes queries
  with CLIP's text encoder, runs inner-product search over the normalized clip embeddings, and returns video id,
  temporal segment, and score. The text encoder and metadata helper are imported from the Assignment 8 script.
- **`ChatBot`** (in `rag_pipeline.py`) — wraps TinyLlama and reuses the Assignment 9 prompt construction,
  validation, and system prompt.
- **`RAGPipeline`** (Task 3) — orchestrates the flow.

**Data flow (Figure 1 in the assignment):**

```
user query
  -> ArticleRetriever.retrieve_with_fallback()          # semantic (+ keyword fallback)
  -> ArticleRetriever.format_context()                  # titles + summaries
  -> prompt_construction(system, <|context|>, history)  # Assignment 9 prompt
  -> TinyLlama generation -> output_validation()        # Assignment 9 chatbot
  -> parse_steps(response)                              # numbered instruction steps
  -> VideoRetriever.retrieve_for_steps()                # one CLIP-retrieved clip per step
  -> RAGResult(response, articles, clips, + metadata)
```

`RAGResult` also carries the transparency metadata the assignment asks for: per-article and per-clip
similarity scores, clip timestamps and source video ids, the injected context block, and the prompt token
count. `app.py` renders these in the panels from Section 5.

---

## 8. Notes

- All models are frozen (inference only); no fine-tuning.
- Models download from Hugging Face on first run: `all-MiniLM-L6-v2`, `openai/clip-vit-base-patch32`,
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
- `gradio` is only needed for `app.py`.
- On Windows, `sentence_transformers` can segfault on import when it loads after torch/transformers; the
  Retrieval scripts import `pyarrow.dataset` first to avoid this.
- Large artifacts (datasets, `venv/`, model files, `processed-data/`) are git-ignored per the assignment guidelines.
