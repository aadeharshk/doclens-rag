"""Compare keyword (BM25), vector and hybrid retrieval on a labelled question set.

Each question is labelled with the handbook section that contains the answer.
A retrieval "hits" if a chunk from that section appears in the top k.

Run:  python -m eval.evaluate_retrieval
"""
import json
from pathlib import Path

from rag.core import HybridIndex, chunk_pages, load_file

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "sample_docs" / "northwind_employee_handbook.md"
QUESTIONS = json.loads((ROOT / "eval" / "questions.json").read_text())
K_VALUES = (1, 3)


def evaluate(index: HybridIndex, mode: str) -> dict:
    hits = {k: 0 for k in K_VALUES}
    rr_total = 0.0
    for item in QUESTIONS:
        results = index.search(item["q"], k=5, mode=mode)
        ranks = [i for i, c in enumerate(results) if item["section"] in c.section.split("|")]
        if ranks:
            rr_total += 1.0 / (ranks[0] + 1)
            for k in K_VALUES:
                hits[k] += ranks[0] < k
    n = len(QUESTIONS)
    return {**{f"hit@{k}": round(hits[k] / n, 3) for k in K_VALUES}, "mrr": round(rr_total / n, 3)}


def main():
    pages = load_file(DOC.name, DOC.read_bytes())
    report = {"n_questions": len(QUESTIONS), "results": {}}
    for strategy, aware in (("naive_chunking", False), ("structure_aware_chunking", True)):
        chunks = chunk_pages(pages, structure_aware=aware)
        index = HybridIndex(chunks)
        report["results"][strategy] = {
            "n_chunks": len(chunks),
            **{m: evaluate(index, m) for m in ("keyword", "vector", "hybrid")},
        }
    (ROOT / "eval" / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
