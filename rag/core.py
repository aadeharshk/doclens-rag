"""DocLens RAG core: loading, chunking, hybrid retrieval and grounded answering."""
from __future__ import annotations

import io
import os
import re
import uuid
from dataclasses import dataclass, field

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


# ---------- 1. Loading ----------------------------------------------------------

@dataclass
class Page:
    text: str
    source: str
    page: int | None = None  # page number for PDFs, None otherwise


def load_file(name: str, data: bytes) -> list[Page]:
    """Turn an uploaded file into text pages. Supports PDF, DOCX, TXT and MD."""
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [Page(p.extract_text() or "", name, i + 1) for i, p in enumerate(reader.pages)]
    if ext == "docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        return [Page("\n".join(p.text for p in doc.paragraphs), name)]
    if ext in {"txt", "md"}:
        return [Page(data.decode("utf-8", errors="ignore"), name)]
    raise ValueError(f"Unsupported file type: .{ext}")


# ---------- 2. Chunking ---------------------------------------------------------

@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None
    section: str = ""

    def label(self) -> str:
        where = f"p.{self.page}" if self.page else self.section or "text"
        return f"{self.source} · {where}"


_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.M)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split text at Markdown-style headings into (section_title, section_text) pairs."""
    marks = list(_HEADING.finditer(text))
    if not marks:
        return [("", text)]
    sections = [("", text[: marks[0].start()])] if text[: marks[0].start()].strip() else []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        sections.append((m.group(1).strip(), text[m.start(): end]))
    return sections


def chunk_pages(pages: list[Page], size: int = 700, overlap: int = 100,
                structure_aware: bool = True) -> list[Chunk]:
    """Split pages into overlapping chunks.

    structure_aware=True first splits at headings so a chunk never mixes two sections,
    then size-splits long sections. False is the naive baseline (size only).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " "])
    chunks: list[Chunk] = []
    for page in pages:
        if structure_aware:
            sections = split_sections(page.text)
        else:
            sections = [("", page.text)]
        spans = [(page.text.find(body), title) for title, body in split_sections(page.text)]
        for title, body in sections:
            for piece in splitter.split_text(body):
                section = title
                if not structure_aware:
                    # Baseline chunks can straddle sections: record every section they overlap.
                    start = page.text.find(piece[:80]); end = start + len(piece)
                    bounds = spans + [(len(page.text), "")]
                    section = "|".join(t for (b, t), (nb, _) in zip(bounds, bounds[1:]) if b < end and nb > start and t)
                chunks.append(Chunk(f"c{len(chunks)}", piece.strip(), page.source, page.page, section))
    return [c for c in chunks if len(c.text) > 30]


# ---------- 3. Hybrid retrieval (vector + BM25, fused with RRF) ----------------

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class HybridIndex:
    chunks: list[Chunk]
    collection: object = field(init=False, repr=False)
    bm25: BM25Okapi = field(init=False, repr=False)

    def __post_init__(self):
        client = chromadb.EphemeralClient()
        self.collection = client.create_collection(
            name=f"docs_{uuid.uuid4().hex[:8]}",
            embedding_function=embedding_functions.DefaultEmbeddingFunction(),  # all-MiniLM-L6-v2 (ONNX)
            metadata={"hnsw:space": "cosine"},
        )
        self.collection.add(ids=[c.id for c in self.chunks], documents=[c.text for c in self.chunks])
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        self._by_id = {c.id: c for c in self.chunks}

    def vector_search(self, query: str, k: int) -> list[str]:
        res = self.collection.query(query_texts=[query], n_results=min(k, len(self.chunks)))
        return res["ids"][0]

    def keyword_search(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self.chunks[i].id for i in order]

    def search(self, query: str, k: int = 4, mode: str = "hybrid", pool: int = 10) -> list[Chunk]:
        if mode == "vector":
            ids = self.vector_search(query, k)
        elif mode == "keyword":
            ids = self.keyword_search(query, k)
        else:
            # Reciprocal Rank Fusion: score = sum(1 / (60 + rank)) across both result lists.
            fused: dict[str, float] = {}
            for ranked in (self.vector_search(query, pool), self.keyword_search(query, pool)):
                for rank, cid in enumerate(ranked):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank + 1)
            ids = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self._by_id[i] for i in ids]


# ---------- 4. Grounded generation ----------------------------------------------

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.
Rules:
- Cite every fact with the passage number in square brackets, e.g. [1] or [2][3].
- If the context does not contain the answer, reply exactly: "I couldn't find that in the uploaded documents."
- Do not use outside knowledge. Be concise: 1-4 sentences or a short list."""


def build_prompt(question: str, chunks: list[Chunk], history: list[dict] | None = None) -> list[dict]:
    context = "\n\n".join(f"[{i + 1}] ({c.label()})\n{c.text}" for i, c in enumerate(chunks))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-4:]:  # short memory for follow-up questions
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    return messages


def _key(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def generate(messages: list[dict]) -> tuple[str, str]:
    """Call Groq first, fall back to Gemini. Returns (answer, provider_used)."""
    errors = []
    if _key("GROQ_API_KEY"):
        try:
            from groq import Groq
            client = Groq(api_key=_key("GROQ_API_KEY"))
            out = client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=messages, temperature=0.1, max_tokens=500)
            return out.choices[0].message.content, "Groq · Llama 3.3 70B"
        except Exception as exc:  # rate limit, outage, bad key
            errors.append(f"Groq: {exc}")
    if _key("GEMINI_API_KEY"):
        try:
            from google import genai
            client = genai.Client(api_key=_key("GEMINI_API_KEY"))
            system = messages[0]["content"]
            convo = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[1:])
            out = client.models.generate_content(
                model="gemini-2.5-flash", contents=convo,
                config={"system_instruction": system, "temperature": 0.1})
            return out.text, "Gemini 2.5 Flash"
        except Exception as exc:
            errors.append(f"Gemini: {exc}")
    raise RuntimeError("No LLM available. Add GROQ_API_KEY or GEMINI_API_KEY. " + " | ".join(errors))


def cited_numbers(answer: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
