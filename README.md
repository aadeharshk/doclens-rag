# 🔎 DocLens RAG: Chat With Your Documents, With Citations

Upload PDFs, Word files or text, ask questions in plain English, and get answers that **cite the exact passages** they came from. If the answer isn't in the documents, it says so instead of guessing.

**Live demo:** [aadeharshk-doclens-rag-app-amj0er.streamlit.app](https://aadeharshk-doclens-rag-app-amj0er.streamlit.app) · **Stack:** Python, ChromaDB, all-MiniLM-L6-v2 embeddings (ONNX), BM25, LangChain text splitters, Groq (Llama 3.3 70B) with Gemini fallback, Streamlit

## Results

Retrieval tested on **25 paraphrased questions** (worded differently from the source text), each labelled with the section that holds the answer, over a 16-section sample handbook:

| Chunking | Search | Hit@1 | Hit@3 | MRR |
|---|---|---|---|---|
| **Section-aware (used in app), 17 chunks** | **Vector** | **92%** | **100%** | **0.96** |
| Section-aware | Hybrid (vector + BM25, RRF) | 80% | 92% | 0.87 |
| Section-aware | Keyword (BM25) | 76% | 88% | 0.81 |
| Naive size-only, 10 chunks | Hybrid | 92% | 100% | 0.96 |

What I learned from this:
- **Vector search beats keyword search on paraphrased questions** (MRR 0.96 vs 0.81). BM25 misses "vacation days" when the document says "annual leave".
- **Hybrid isn't automatically better.** Adding BM25 pulled the ranking down here, so the app defaults to vector, with hybrid available for documents full of exact codes, IDs or names.
- **Section-aware chunking matches the accuracy of naive chunking with chunks ~40% smaller**, so each passage sent to the LLM is about one topic and carries less noise.
- The test set is small (25 questions, 1 document), so treat these as directional numbers. Reproduce them with `python -m eval.evaluate_retrieval`.

## Features

- **Multi-format upload:** PDF (with page numbers), DOCX, TXT, Markdown, several files at once
- **Grounded answers with citations:** the LLM must cite `[1]`, `[2]`… and refuse when the context doesn't hold the answer
- **Source panel:** see every retrieved passage, which ones were cited, and retrieval and total latency
- **Three search modes:** vector, BM25 keyword, hybrid with Reciprocal Rank Fusion
- **LLM fallback chain:** Groq first; if it fails or is rate-limited, Gemini answers
- **Follow-up questions:** the last few turns are sent as conversation history
- **Built-in evaluation tab** and a chunk inspector

## Architecture

```
Upload ─► load_file() ─► chunk_pages() ─► HybridIndex ─────────► top-k passages ─► build_prompt() ─► Groq ─(fail)─► Gemini
          pypdf/docx      section-aware      ChromaDB (cosine)                    numbered context                   │
                          + 700-char split   + BM25, fused by RRF                 + citation rules                   ▼
                                                                                                        answer with [n] citations
```

All the core logic is in `rag/core.py` (~200 lines). `app.py` is only the UI.

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.example.toml .streamlit/secrets.toml   # add a free Groq or Gemini key
streamlit run app.py
pytest -q                                                    # 3 tests
python -m eval.evaluate_retrieval                            # regenerates eval/results.json
```

Free keys: [Groq console](https://console.groq.com/keys) · [Google AI Studio](https://aistudio.google.com/apikey)

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub, then go to share.streamlit.io → **Create app** → choose the repo and `app.py`.
2. Under **Advanced settings → Secrets**, paste `GROQ_API_KEY = "..."` (and optionally `GEMINI_API_KEY`).
3. If the build fails with an SQLite version error from ChromaDB, add `pysqlite3-binary` to `requirements.txt` and put this at the top of `app.py`:
   `__import__("pysqlite3"); import sys; sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")`

---
Built by **Harsh Aade** · [LinkedIn](https://linkedin.com/in/harsh-aade) · [GitHub](https://github.com/aadeharshk)
