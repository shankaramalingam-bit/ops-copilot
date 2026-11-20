"""Offline structural and fact checks for the 10-document corpus."""

import re
from pathlib import Path

import pytest
import yaml

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"
DOC_IDS = [
    "WH-001",
    "WH-002",
    "WH-003",
    "WH-004",
    "PR-001",
    "PR-002",
    "QA-001",
    "FIN-001",
    "WH-005",
    "CS-001",
]

# doc_id -> (section number that must contain the facts, strings that must appear in it)
FACTS: dict[str, list[tuple[str, list[str]]]] = {
    "WH-001": [
        ("3.1", ["ASN", "PO"]),
        ("5.1", ["4 hours"]),
        ("6.1", ["hazmat", "cage zone H"]),
        ("2.1", ["bill of lading"]),
    ],
    "WH-002": [
        ("2.2", ["Monthly", "Quarterly", "Twice a year", "semi-annually"]),
        ("4.1", ["±2%", "$250"]),
        ("4.2", ["different counter"]),
    ],
    "WH-003": [
        ("2.1", ["bill of lading"]),
        ("2.2", ["photograph"]),
        ("2.3", ["24 hours", "Procurement"]),
        ("3.1", ["5%", "held"]),
        ("5.1", ["quarantined"]),
        ("6.1", ["30 days"]),
    ],
    "WH-004": [
        ("2.1", ["No RMA number, no return"]),
        ("2.2", ["30 days"]),
        ("3.2", ["2 business days"]),
        ("4.1", ["Restock", "Refurbish", "Scrap"]),
        ("5.1", ["15%", "not defective"]),
    ],
    "PR-001": [
        ("2.1", ["$5,000", "$25,000", "$100,000", "Buyer", "Procurement Manager"]),
        ("2.1", ["Director of Operations", "CFO"]),
        ("3.1", ["$10,000", "sole-source justification form"]),
        ("4.1", ["re-approved", "band of the new total"]),
    ],
    "PR-002": [
        ("2.1", ["W-9", "$2M", "quality questionnaire", "credit check"]),
        ("3.1", ["90-day", "provisional"]),
        ("3.2", ["Approved Supplier List"]),
    ],
    "QA-001": [
        ("2.1", ["1 business day"]),
        ("2.2", ["QC hold"]),
        ("3.1", ["10 business days", "SCAR"]),
        ("4.1", ["3 NCRs in 6 months", "probation"]),
    ],
    "FIN-001": [
        ("2.1", ["exactly"]),
        ("2.2", ["±2%", "$50", "whichever is lower"]),
        ("3.1", ["buyer", "held"]),
        ("4.1", ["net 45"]),
    ],
    "WH-005": [
        ("2.1", ["$500", "$5,000", "Warehouse Supervisor", "Inventory Control Manager"]),
        ("2.2", ["Director of Operations", "Finance"]),
        ("3.1", ["reason code", "mandatory"]),
    ],
    "CS-001": [
        ("2.1", ["Harbor Point Grocers", "Cascade Medical Supply"]),
        ("3.1", ["3 pm"]),
        ("3.2", ["98%"]),
        ("4.1", ["4 business hours"]),
        ("4.2", ["5 days"]),
    ],
}

FORBIDDEN = ["forklift", "401(k)", "pto"]

HEADING_RE = re.compile(r"^(#{2,3}) (\d+(?:\.\d+)?)[. ]\s*(.+)$", re.M)


def _read(doc_id: str) -> str:
    return (DOCS_DIR / f"{doc_id}.md").read_text(encoding="utf-8")


def _front_matter(text: str) -> dict:
    assert text.startswith("---\n")
    end = text.index("\n---\n", 4)
    return yaml.safe_load(text[4:end])


def _section(text: str, number: str) -> str:
    """Body of the subsection (###) with the given number, up to the next heading."""
    m = re.search(rf"^### {re.escape(number)} .*?$(.*?)(?=^#{{2,3}} |\Z)", text, re.M | re.S)
    assert m, f"subsection {number} not found"
    return m.group(0)


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_file_exists_and_front_matter(doc_id: str) -> None:
    assert (DOCS_DIR / f"{doc_id}.md").is_file()
    fm = _front_matter(_read(doc_id))
    assert fm["doc_id"] == doc_id
    for key in ("title", "owner_role", "version", "effective_date", "review_cycle"):
        assert fm.get(key), f"{doc_id} missing {key}"
    assert str(fm["effective_date"]).startswith("2026-")


def test_no_unexpected_files() -> None:
    assert sorted(p.stem for p in DOCS_DIR.glob("*.md")) == sorted(DOC_IDS)


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_headings_numbered_and_unique(doc_id: str) -> None:
    text = _read(doc_id)
    assert re.search(rf"^# {doc_id} \S", text, re.M)
    headings = HEADING_RE.findall(text)
    numbers = [n for _, n, _ in headings]
    assert numbers, "no numbered headings"
    assert len(numbers) == len(set(numbers)), f"duplicate section numbers in {doc_id}"
    top = [n for h, n, _ in headings if h == "##"]
    assert top[0] == "1" and top == [str(i) for i in range(1, len(top) + 1)]
    assert "Purpose" in [t for h, n, t in headings if n == "1"][0]
    # every ## heading must be numbered
    all_h2 = re.findall(r"^## .*$", text, re.M)
    assert len(all_h2) == len(top)
    # every subsection n.m sits under top-level n
    for h, n, _ in headings:
        if h == "###":
            assert n.split(".")[0] in top


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_related_documents_section(doc_id: str) -> None:
    text = _read(doc_id)
    m = re.search(r"^## \d+\. Related documents$(.*?)(?=^---$|\Z)", text, re.M | re.S)
    assert m, "no Related documents section"
    refs = set(re.findall(r"\b(?:WH|PR|QA|FIN|CS)-\d{3}\b", m.group(1)))
    assert refs and doc_id not in refs
    assert refs <= set(DOC_IDS)


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_boilerplate_present(doc_id: str) -> None:
    text = _read(doc_id)
    assert "CONFIDENTIAL — Meridian Supply Co. internal use only" in text
    assert re.search(r"<!--.*?-->", text, re.S)
    assert re.search(
        r"^Document control: .* \| Page 1 of 1 \| Uncontrolled when printed$", text, re.M
    )


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_word_count_in_range(doc_id: str) -> None:
    assert 600 <= len(_read(doc_id).split()) <= 1200


@pytest.mark.parametrize(
    ("doc_id", "number", "needles"),
    [(d, n, needles) for d, items in FACTS.items() for n, needles in items],
)
def test_mandated_facts(doc_id: str, number: str, needles: list[str]) -> None:
    section = _section(_read(doc_id), number)
    lowered = section.lower()
    for needle in needles:
        assert needle.lower() in lowered, f"{doc_id} §{number} missing {needle!r}"


@pytest.mark.parametrize("doc_id", DOC_IDS)
def test_forbidden_topics_absent(doc_id: str) -> None:
    text = _read(doc_id).lower()
    for term in FORBIDDEN:
        assert term not in text, f"{doc_id} contains forbidden topic {term!r}"
    for term in ("recertification", "benefits", "vacation"):
        assert term not in text


def test_no_sku_prices_in_corpus() -> None:
    # Dollar amounts allowed only as policy thresholds, not per-unit prices.
    for doc_id in DOC_IDS:
        assert not re.search(r"\$\d+(\.\d{2})\s*(per|/)\s*(unit|each|ea)", _read(doc_id), re.I)
        assert "unit price" not in _read(doc_id).lower().replace("invoiced unit price", "")


def test_ambiguity_preserved() -> None:
    # Two documents carry approval limits; two carry reporting deadlines.
    assert "Approval Limits" in _read("PR-001") and "Approval Limits" in _read("WH-005")
    assert "within 24 hours" in _read("WH-003") and "within 1 business day" in _read("QA-001")
