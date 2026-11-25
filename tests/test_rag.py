from pathlib import Path

from rag.core import HybridIndex, build_prompt, chunk_pages, cited_numbers, load_file

DOC = Path(__file__).resolve().parents[1] / "sample_docs" / "northwind_employee_handbook.md"


def _index():
    return HybridIndex(chunk_pages(load_file(DOC.name, DOC.read_bytes())))


def test_structure_aware_chunks_never_mix_sections():
    chunks = chunk_pages(load_file(DOC.name, DOC.read_bytes()))
    for c in chunks:
        assert c.text.count("\n## ") == 0


def test_each_search_mode_finds_the_leave_section():
    index = _index()
    for mode in ("vector", "keyword", "hybrid"):
        top = index.search("How many days of paid annual leave do employees get?", k=3, mode=mode)
        assert any(c.section == "3. Leave Policy" for c in top), mode


def test_prompt_numbers_passages_and_citations_parse():
    index = _index()
    hits = index.search("referral bonus", k=2)
    msgs = build_prompt("What is the referral bonus?", hits)
    assert "[1]" in msgs[-1]["content"] and "[2]" in msgs[-1]["content"]
    assert cited_numbers("It is INR 50,000 [1][2].") == {1, 2}
