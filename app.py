"""DocLens RAG: chat with your documents, with citations (Streamlit app)."""
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from rag.core import HybridIndex, build_prompt, chunk_pages, cited_numbers, generate, load_file

ROOT = Path(__file__).parent
SAMPLE = ROOT / "sample_docs" / "northwind_employee_handbook.md"

st.set_page_config(page_title="DocLens RAG", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Indexing documents…")
def build_index(files: tuple[tuple[str, bytes], ...]):
    pages = [p for name, data in files for p in load_file(name, data)]
    chunks = chunk_pages(pages)
    return HybridIndex(chunks), len(pages)


# ---------------- Sidebar: documents and settings ----------------
with st.sidebar:
    st.header("📄 Documents")
    uploads = st.file_uploader("Upload PDF, DOCX, TXT or MD", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if uploads:
        files = tuple((u.name, u.getvalue()) for u in uploads)
    else:
        st.info("No upload yet, so using the sample: a fictional company's employee handbook.")
        files = ((SAMPLE.name, SAMPLE.read_bytes()),)
    try:
        index, n_pages = build_index(files)
    except Exception as exc:
        st.error(f"Could not read the files: {exc}")
        st.stop()
    st.success(f"{len(files)} file(s) · {n_pages} page(s) · {len(index.chunks)} chunks indexed")

    st.header("⚙️ Retrieval")
    mode = st.radio("Search mode", ["vector", "hybrid", "keyword"], index=0,
                    help="vector = semantic embeddings · keyword = BM25 · hybrid = both, merged with Reciprocal Rank Fusion")
    k = st.slider("Passages sent to the LLM", 2, 8, 4)
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []

st.title("🔎 DocLens RAG")
st.caption("Ask questions about your documents. Every answer is grounded in retrieved passages and cites them; if the answer isn't in the documents, it says so.")

tab_chat, tab_eval, tab_chunks = st.tabs(["💬 Chat", "📊 Retrieval evaluation", "🧩 Chunks"])

# ---------------- Chat ----------------
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if not uploads and not st.session_state.messages:
        st.markdown("**Try:** _How many vacation days do I get?_ · _Can I paste client code into ChatGPT?_ · _What's the CEO's name?_ (not in the document)")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(msg["meta"]):
                    for i, (label, text) in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{i}] {label}**")
                        st.caption(text[:600] + ("…" if len(text) > 600 else ""))

    if question := st.chat_input("Ask about your documents…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            t0 = time.perf_counter()
            hits = index.search(question, k=k, mode=mode)
            t_ret = time.perf_counter() - t0
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            try:
                with st.spinner("Thinking…"):
                    answer, provider = generate(build_prompt(question, hits, history))
            except RuntimeError as exc:
                answer, provider = f"⚠️ {exc}", "none"
            t_total = time.perf_counter() - t0
            used = cited_numbers(answer)
            st.markdown(answer)
            meta = f"Sources · {provider} · retrieval {t_ret * 1000:.0f} ms · total {t_total:.1f} s · cited {sorted(used) or 'none'}"
            sources = [(c.label(), c.text) for c in hits]
            with st.expander(meta):
                for i, (label, text) in enumerate(sources, 1):
                    st.markdown(f"**[{i}] {label}**" + ("  ✅ cited" if i in used else ""))
                    st.caption(text[:600] + ("…" if len(text) > 600 else ""))
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "meta": meta})

# ---------------- Evaluation ----------------
with tab_eval:
    results_path = ROOT / "eval" / "results.json"
    if results_path.exists():
        report = json.loads(results_path.read_text())
        st.subheader(f"Retrieval quality on {report['n_questions']} labelled questions")
        st.write("Each question is paraphrased (different words from the document) and labelled with the section that holds the answer. "
                 "**Hit@k** = share of questions where the right section is in the top k results. **MRR** rewards ranking it first.")
        rows = []
        for strategy, res in report["results"].items():
            for m in ("keyword", "vector", "hybrid"):
                rows.append({"chunking": strategy.replace("_", " "), "chunks": res["n_chunks"], "search": m, **res[m]})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Reproduce with: python -m eval.evaluate_retrieval")
    else:
        st.info("Run `python -m eval.evaluate_retrieval` to generate results.")

# ---------------- Chunks ----------------
with tab_chunks:
    st.write("How the documents were split before indexing (section-aware: a chunk never mixes two headings).")
    st.dataframe(pd.DataFrame([{"id": c.id, "source": c.label(), "chars": len(c.text), "text": c.text[:200]} for c in index.chunks]),
                 width="stretch", hide_index=True)

st.divider()
st.caption("Built by Harsh Aade · Python, LangChain text splitters, ChromaDB, BM25, Groq / Gemini, Streamlit")
