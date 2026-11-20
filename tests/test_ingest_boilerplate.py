import pytest

from ops_copilot.ingest import clean


@pytest.mark.parametrize(
    "banner",
    [
        "CONFIDENTIAL — Meridian Supply Co. internal use only",
        "Confidential - For internal use",
        "CONFIDENTIAL – Meridian internal",
        "  confidential — anything else here",
    ],
)
def test_banner_variants_stripped(banner):
    out = clean(f"# T\n\n{banner}\n\nBody text.\n")
    assert "onfidential" not in out.lower()
    assert "Body text." in out and "# T" in out


def test_body_line_mentioning_confidential_kept():
    out = clean("# T\n\nCustomer data is confidential — handle with care.\n")
    assert "confidential" in out


@pytest.mark.parametrize(
    "footer",
    [
        "Document control: WH-003 v1.2 | Meridian Supply Co. Document Control"
        " | Page 1 of 1 | Uncontrolled when printed",
        "  **Document control:** WH-003 v1.2",
        "_Document control: QA-001 v2_",
        "Meridian Document Control | Page 2 of 5 | Uncontrolled when printed",
    ],
)
def test_footer_stripped(footer):
    out = clean(f"# T\n\nBody text.\n\n{footer}\n")
    assert "ocument control" not in out and "ncontrolled" not in out
    assert "Body text." in out


@pytest.mark.parametrize(
    "body",
    [
        "Contact Document Control for copies; documents are uncontrolled when printed.",
        "See the Document control: section of the handbook for details.",
        "Uncontrolled when printed copies must be destroyed.",
    ],
)
def test_body_lines_mentioning_document_control_kept(body):
    out = clean(f"# T\n\n{body}\n")
    assert body in out


def test_headings_kept():
    out = clean("# Document control\n\n## Uncontrolled copies\n\nText.\n")
    assert "# Document control" in out and "## Uncontrolled copies" in out
