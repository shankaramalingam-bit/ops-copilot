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
        ("3.1", ["advance ship notice (ASN)", "purchase order (PO)"]),
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
        if needle.isupper() or needle.endswith(")"):
            # acronyms / exact phrases: case-sensitive
            assert needle in section, f"{doc_id} §{number} missing {needle!r}"
        else:
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


# ---------------------------------------------------------------- pairing tests


def _table_rows(section: str) -> list[tuple[str, str]]:
    rows = re.findall(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$", section, re.M)
    body = [(a, b) for a, b in rows if not set(a) <= set("-")]
    return body[1:]  # drop the header row


def test_pr001_band_rows_pair_band_with_approver() -> None:
    rows = _table_rows(_section(_read("PR-001"), "2.1"))
    assert rows == [
        ("Up to $5,000", "Buyer"),
        ("$5,000 to $25,000", "Procurement Manager"),
        ("$25,000 to $100,000", "Director of Operations"),
        ("Above $100,000", "CFO"),
    ]


def test_wh005_band_rows_pair_band_with_approver() -> None:
    rows = _table_rows(_section(_read("WH-005"), "2.1"))
    assert rows == [
        ("Up to $500", "Warehouse Supervisor"),
        ("$500 to $5,000", "Inventory Control Manager"),
        ("Above $5,000", "Director of Operations plus Finance"),
    ]


def test_wh005_section_2_2_threshold_and_both_approvers() -> None:
    s = _section(_read("WH-005"), "2.2")
    assert re.search(
        r"adjustment above \$5,000 needs the approval of both the Director of Operations "
        r"and a member of the Finance team",
        s,
    )


def test_fin001_exceptions_routed_to_buyer() -> None:
    assert re.search(r"Every exception is routed to the buyer named on the PO\.", _read("FIN-001"))
    assert "routed to the buyer" in _section(_read("FIN-001"), "3.1")


def test_boundary_sentences() -> None:
    pr = _section(_read("PR-001"), "2.1")
    assert re.search(
        r"A PO of exactly \$5,000, \$25,000 or \$100,000 is approved by the lower band\.", pr
    )
    wh = _section(_read("WH-005"), "2.1")
    assert re.search(
        r"An adjustment of exactly \$500 is approved by the Warehouse Supervisor, and an "
        r"adjustment of exactly \$5,000 by the Inventory Control Manager\.",
        wh,
    )


def test_probation_has_no_po_approval_rule() -> None:
    s = _section(_read("QA-001"), "4.2")
    assert not re.search(r"\bPOs?\b|sign-off|approv", s, re.I)


def test_approval_authority_only_in_pr001_and_wh005() -> None:
    # Other docs may refer to the approval limits but must not name an approver role.
    roles = r"(Buyer|Procurement Manager|Director of Operations|CFO|Supervisor|Control Manager)"
    for doc_id in DOC_IDS:
        if doc_id in ("PR-001", "WH-005"):
            continue
        for raw in _read(doc_id).splitlines():
            line = raw.replace("Approved Supplier List", "ASL")
            assert not (re.search(r"approv|sign-off", line, re.I) and re.search(roles, line)), (
                f"{doc_id}: competing approval statement: {line}"
            )


# ------------------------------------------------- conflicting-number guard


def _corpus() -> str:
    return "\n".join(_read(d) for d in DOC_IDS)


# (regex with one capture group, expected value). Every match in the corpus must equal it,
# and there must be at least one match (positive control).
NUMBER_RULES: dict[str, tuple[str, str]] = {
    "putaway hours": (r"put\s+away\s+within\s+(\d+)\s+hours?", "4"),
    "cycle-count tolerance pct": (r"variance is within ±(\d+)%", "2"),
    "cycle-count tolerance usd": (r"±\d+% of the system quantity or \$(\d+)", "250"),
    "OS&D notify hours": (r"notified of every OS&D exception within (\d+) hours", "24"),
    "overage pct": (r"overages? (?:up to and including|above) (\d+)%", "5"),
    "carrier claim days": (r"claim against the carrier must be filed within (\d+) days", "30"),
    "carrier claim days (heading)": (r"Carrier claim within (\d+) days", "30"),
    "RMA validity days": (r"RMA number is valid for (\d+) days", "30"),
    "inspect business days": (r"inspected within (\d+) business days", "2"),
    "restocking pct": (r"(\d+)% restocking fee", "15"),
    "insurance usd": (r"\$(\d+)M general liability", "2"),
    "provisional days": (r"[Pp]rovisional status lasts (\d+) days", "90"),
    "NCR business days": (r"raised within (\d+) business day", "1"),
    "SCAR business days": (r"respond within (\d+) business days", "10"),
    "NCR count": (r"receives (\d+) NCRs in \d+ months", "3"),
    "NCR months": (r"receives \d+ NCRs in (\d+) months", "6"),
    "price tolerance pct": (r"within ±(\d+)% or \$\d+", "2"),
    "price tolerance usd": (r"within ±\d+% or \$(\d+)", "50"),
    "net terms": (r"\b[Nn]et (\d+)\b", "45"),
    "cut-off time": (r"received (?:by|after) (\d+) pm", "3"),
    "fill rate": (r"fill-rate target for Tier-1 accounts is (\d+)%", "98"),
    "damage-claim hours": (r"acknowledged within (\d+) business hours", "4"),
    "RMA-credit days": (r"credit is issued within (\d+) days", "5"),
}


@pytest.mark.parametrize("name", sorted(NUMBER_RULES))
def test_governing_number_consistent_across_corpus(name: str) -> None:
    pattern, expected = NUMBER_RULES[name]
    found = re.findall(pattern, _corpus())
    assert found, f"positive control empty for {name}"
    assert set(found) == {expected}, f"{name}: {found}"


def test_no_conflicting_sentence_level_numbers() -> None:
    """Sentence-level guard: hours near 'putaway' must be 4; days near 'carrier' must be 30."""
    seen_putaway = seen_carrier = False
    for sentence in re.split(r"(?<=[.!?])\s+", _corpus()):
        if re.search(r"put\s*away", sentence, re.I):
            seen_putaway |= bool(re.search(r"\d+\s+hours", sentence))
            assert set(re.findall(r"(\d+)\s+hours", sentence)) <= {"4"}, sentence
        if re.search(r"carrier", sentence, re.I):
            seen_carrier |= bool(re.search(r"\d+\s+days", sentence))
            assert set(re.findall(r"(\d+)\s+days", sentence)) <= {"30"}, sentence
    assert seen_putaway and seen_carrier  # positive controls


# ------------------------------------------- no invented approval-routing rules

_PO_WORDS = re.compile(r"\bPOs?\b|purchase orders?|\btrial orders?\b|\borders?\b", re.I)


def _sentences() -> list[tuple[str, str]]:
    out = []
    for d in DOC_IDS:
        for s in re.split(r"(?<=[.!?])\s+|\n+", _read(d)):
            out.append((d, s))
    return out


def test_sentence_splitter_positive_control() -> None:
    assert any(d == "PR-001" and "Approvers" in s for d, s in _sentences())


def test_no_po_escalation_to_another_band_or_approver() -> None:
    pat = re.compile(
        r"escalat\w*|next band|higher band|next approver|next level|passes? to the|"
        r"reassign\w*|auto-?approv\w*",
        re.I,
    )
    for d, s in _sentences():
        if _PO_WORDS.search(s):
            assert not pat.search(s), f"{d}: PO routing to another approver: {s}"


def test_no_orders_permitted_to_suppliers_off_the_asl() -> None:
    off_asl = re.compile(
        r"not (?:yet )?(?:completed )?onboard\w*|"
        r"not (?:yet )?on the (?:Approved Supplier List|ASL)|"
        r"(?:before|without) (?:the supplier is |being )?onboard\w*|"
        r"not (?:yet )?(?:been )?(?:approved|qualified)|trial orders?|"
        r"unapproved supplier|non-ASL",
        re.I,
    )
    prohibition = re.compile(
        r"\bare rejected\b|\bis rejected\b|\bmust not be (?:issued|placed|approved)", re.I
    )
    checked = False
    for d, s in _sentences():
        if _PO_WORDS.search(s) and off_asl.search(s):
            checked = True
            assert prohibition.search(s), f"{d}: order to non-ASL supplier permitted: {s}"
    assert checked  # positive control: PR-001 §5.1 rejection sentence is seen
