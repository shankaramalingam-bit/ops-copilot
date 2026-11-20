from ops_copilot.chunk import CHUNK_CHARS, chunk_doc, chunk_docs, parse_heading
from ops_copilot.ingest import clean, load_docs
from ops_copilot.types import Doc

from .conftest import FIXTURE_DOCS


def test_load_docs_reads_front_matter_ids():
    docs = load_docs(FIXTURE_DOCS)
    assert [d.doc_id for d in docs] == ["PR-001", "QA-001", "WH-003"]
    wh = next(d for d in docs if d.doc_id == "WH-003")
    assert wh.title == "Overages, Shortages & Damages (OS&D)"
    assert wh.metadata["owner"] == "Warehouse Operations"


def test_clean_strips_boilerplate_but_keeps_headings():
    raw = (FIXTURE_DOCS / "WH-003.md").read_text(encoding="utf-8")
    assert "CONFIDENTIAL" in raw and "<!--" in raw and "Uncontrolled" in raw
    out = clean(raw)
    for gone in (
        "CONFIDENTIAL",
        "<!--",
        "-->",
        "internal draft note",
        "multi-line comment",
        "Document control",
        "Uncontrolled when printed",
        "Page 1 of 1",
        "doc_id:",
        "---",
        "**",
        "](",
        "*OS&D report*",
    ):
        assert gone not in out
    for kept in ("# WH-003 ", "## 2. Discrepancy handling", "### 2.1 Shortage reporting"):
        assert kept in out
    assert "receiving staff" in out and "carrier policy" in out and "OS&D report" in out
    assert "\n\n\n" not in out


def test_clean_whitespace_normalised():
    assert clean("# T\n\n\n\n  a   b \t c  \n") == "# T\n\na b c\n"


def test_parse_heading():
    assert parse_heading("2.1 Shortage reporting") == ("2.1", "Shortage reporting")
    assert parse_heading("2. Discrepancy handling") == ("2", "Discrepancy handling")
    assert parse_heading("WH-003 Title") is None


def test_chunk_metadata_and_ids():
    docs = {d.doc_id: d for d in load_docs(FIXTURE_DOCS)}
    chunks = chunk_doc(docs["WH-003"])
    by_section = {c.section: c for c in chunks}
    assert set(by_section) == {"1", "2.1", "2.2", "3"}  # "2" has no body of its own
    c = by_section["2.1"]
    assert c.id == "WH-003#2.1#0"
    assert c.doc_id == "WH-003"
    assert c.section_title == "Shortage reporting"
    assert c.text.startswith(
        "WH-003 Overages, Shortages & Damages (OS&D) > 2.1 Shortage reporting\n"
    )
    assert "2% of the ordered quantity" in c.text
    assert len({c.id for c in chunk_docs(list(docs.values()))}) == len(
        chunk_docs(list(docs.values()))
    )


def test_chunk_size_limit_and_unique_ids_on_long_section():
    para = "Pallets are counted and checked against the packing list. " * 6
    body = "\n\n".join(para for _ in range(20))
    doc = Doc("WH-001", "Receiving", f"# WH-001 Receiving\n\n## 4. Long\n\n{body}\n", {})
    chunks = chunk_doc(doc)
    assert len(chunks) > 3
    assert all(len(c.text) <= CHUNK_CHARS for c in chunks)
    assert [c.id for c in chunks] == [f"WH-001#4#{i}" for i in range(len(chunks))]
    assert all(c.section == "4" for c in chunks)


def test_preamble_without_number_gets_section_zero():
    doc = Doc("WH-001", "Receiving", "# WH-001 Receiving\n\nIntro text.\n\n## 1. A\n\nBody.\n", {})
    assert [c.section for c in chunk_doc(doc)] == ["0", "1"]
