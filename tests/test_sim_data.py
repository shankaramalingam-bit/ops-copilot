"""Offline consistency checks: simulated dashboard data vs. the corpus rules."""

import json
import re
from datetime import date
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parent.parent / "data" / "sim"


def load(name: str) -> dict:
    return json.loads((SIM / name).read_text(encoding="utf-8"))


def pr001_approver(total_usd: float) -> str:
    """PR-001 §2.1 bands; boundary values belong to the lower band."""
    if total_usd <= 5_000:
        return "Buyer"
    if total_usd <= 25_000:
        return "Procurement Manager"
    if total_usd <= 100_000:
        return "Director of Operations"
    return "CFO"


@pytest.fixture(scope="module")
def pos() -> list[dict]:
    return load("purchase_orders.json")["purchase_orders"]


@pytest.fixture(scope="module")
def receipts() -> list[dict]:
    return load("receipts.json")["receipts"]


@pytest.fixture(scope="module")
def suppliers() -> list[dict]:
    return load("suppliers.json")["suppliers"]


@pytest.fixture(scope="module")
def kpis() -> dict:
    return load("kpis.json")


@pytest.mark.parametrize("name", ["kpis", "receipts", "purchase_orders", "suppliers"])
def test_json_parses(name: str) -> None:
    assert load(f"{name}.json")


def test_band_function_boundaries() -> None:
    assert pr001_approver(5_000) == "Buyer"
    assert pr001_approver(5_000.01) == "Procurement Manager"
    assert pr001_approver(25_000) == "Procurement Manager"
    assert pr001_approver(40_000) == "Director of Operations"
    assert pr001_approver(100_000) == "Director of Operations"
    assert pr001_approver(100_001) == "CFO"


def test_awaiting_pos_exist_for_every_band(pos: list[dict]) -> None:
    awaiting = {p["approver"] for p in pos if p["status"].startswith("awaiting")}
    assert awaiting == {"Buyer", "Procurement Manager", "Director of Operations", "CFO"}


def test_awaiting_po_approver_matches_band(pos: list[dict]) -> None:
    awaiting = [p for p in pos if p["status"].startswith("awaiting")]
    assert awaiting
    for p in awaiting:
        assert p["approver"] == pr001_approver(p["total_usd"]), p["po"]
        assert p["status"] == f"awaiting {p['approver']}", p["po"]


def test_all_po_approvers_match_band(pos: list[dict]) -> None:
    for p in pos:
        assert p["approver"] == pr001_approver(p["total_usd"]), p["po"]


def test_forty_thousand_po_awaiting_director(pos: list[dict]) -> None:
    hits = [p for p in pos if p["total_usd"] == 40_000]
    assert hits
    assert any(p["status"] == "awaiting Director of Operations" for p in hits)


def test_amended_po_reapproved_at_new_total_band(pos: list[dict]) -> None:
    amended = [p for p in pos if "original_total_usd" in p]
    assert amended
    for p in amended:
        assert p["total_usd"] > p["original_total_usd"]
        assert pr001_approver(p["total_usd"]) != pr001_approver(p["original_total_usd"])
        assert p["approver"] == pr001_approver(p["total_usd"])


def test_sole_source_above_10k_flagged(pos: list[dict]) -> None:
    sole = [p for p in pos if p.get("sole_source") and p["total_usd"] > 10_000]
    assert sole
    assert all("sole_source_form" in p for p in sole)


def test_probation_supplier_has_three_ncrs_in_six_months(suppliers: list[dict]) -> None:
    prob = [s for s in suppliers if s["status"] == "probation"]
    assert prob
    for s in prob:
        ds = sorted(date.fromisoformat(d) for d in s["ncr_dates"])
        assert len(ds) >= 3
        # some window of 3 consecutive NCRs spans at most 6 months (183 days)
        assert any((ds[i + 2] - ds[i]).days <= 183 for i in range(len(ds) - 2))
        assert s["probation_since"] == ds[-1].isoformat()


def test_non_probation_suppliers_do_not_meet_trigger(suppliers: list[dict]) -> None:
    for s in suppliers:
        if s["status"] != "probation":
            assert len(s["ncr_dates"]) < 3


def test_over_shipment_above_5pct_is_held(receipts: list[dict]) -> None:
    over = [r for r in receipts if r["exception"] and r["exception"]["type"] == "over_shipment"]
    assert over
    for r in over:
        pct = (r["received_qty"] - r["ordered_qty"]) / r["ordered_qty"] * 100
        assert pct == pytest.approx(r["exception"]["over_pct"], abs=0.01)
        if pct > 5:
            assert r["status"] == "held"


def test_damaged_receipt_has_osd_exception(receipts: list[dict]) -> None:
    dmg = [r for r in receipts if r["exception"] and r["exception"]["type"] == "damaged"]
    assert dmg
    assert all(r["exception"]["policy"] == "WH-003" and r["status"] == "quarantined" for r in dmg)


def test_short_shipment_present(receipts: list[dict]) -> None:
    short = [r for r in receipts if r["exception"] and r["exception"]["type"] == "short_shipment"]
    assert short
    for r in short:
        assert r["received_qty"] < r["ordered_qty"]
        assert r["exception"]["short_qty"] == r["ordered_qty"] - r["received_qty"]


def test_price_exception_exceeds_tolerance(receipts: list[dict]) -> None:
    px = [r for r in receipts if r["exception"] and r["exception"]["type"] == "price_mismatch"]
    assert px
    for r in px:
        e = r["exception"]
        pct = (e["invoice_unit_price"] - e["po_unit_price"]) / e["po_unit_price"] * 100
        usd = (e["invoice_unit_price"] - e["po_unit_price"]) * r["received_qty"]
        assert pct > 2
        assert usd > 50  # outside the FIN-001 tolerance (lower of 2% / $50 is exceeded)
        assert e["invoice_status"] == "held" and e["routed_to"] == "Buyer"


def test_every_exception_has_ask(
    pos: list[dict], receipts: list[dict], suppliers: list[dict]
) -> None:
    rows = [r for r in receipts if r["exception"]]
    rows += [p for p in pos if p["status"].startswith("awaiting")]
    rows += [s for s in suppliers if s["status"] == "probation"]
    assert rows
    for row in rows:
        assert isinstance(row.get("ask"), str) and row["ask"].strip(), row
    assert all(r.get("exception") is None or r["ask"] for r in receipts)


def test_asks_are_unique(pos: list[dict], receipts: list[dict], suppliers: list[dict]) -> None:
    asks = [x["ask"] for x in (*pos, *receipts, *suppliers) if x.get("ask")]
    assert len(asks) == len(set(asks))


def test_references_resolve(pos: list[dict], receipts: list[dict], suppliers: list[dict]) -> None:
    po_ids = {p["po"] for p in pos}
    names = {s["name"] for s in suppliers}
    assert len(po_ids) == len(pos)
    for r in receipts:
        assert r["po"] in po_ids
        assert r["supplier"] in names
    for p in pos:
        assert p["supplier"] in names


def test_kpis_consistent_with_data(
    kpis: dict, pos: list[dict], receipts: list[dict], suppliers: list[dict]
) -> None:
    assert kpis["open_pos"] == sum(p["status"] not in ("closed", "received") for p in pos)
    assert kpis["receipts_today"] == sum(r["date"] == kpis["as_of"] for r in receipts)
    assert kpis["open_discrepancies"] == sum(r["exception"] is not None for r in receipts)
    rec = sum(s["units_received_90d"] for s in suppliers)
    rej = sum(s["units_rejected_90d"] for s in suppliers)
    assert kpis["supplier_quality_pct"] == round((1 - rej / rec) * 100, 1)
    for s in suppliers:
        expect = round((1 - s["units_rejected_90d"] / s["units_received_90d"]) * 100, 1)
        assert s["quality_pct"] == expect, s["name"]
    fr = kpis["key_account_fill_rate"]
    accts = {a["account"] for a in fr["accounts"]}
    assert accts == {"Harbor Point Grocers", "Cascade Medical Supply"}
    mean = sum(a["fill_rate_pct"] for a in fr["accounts"]) / len(fr["accounts"])
    assert fr["overall_pct"] == pytest.approx(mean, abs=0.05)
    assert fr["target_pct"] == 98.0


def test_dates_are_iso_around_2026_11(pos: list[dict], receipts: list[dict]) -> None:
    for r in receipts:
        assert date.fromisoformat(r["date"]).strftime("%Y-%m") == "2026-11"
    for p in pos:
        assert date.fromisoformat(p["created"]).year == 2026


def _nums(text: str) -> set[str]:
    """Numbers, amounts and percents in a string, thousands separators removed."""
    return {m.replace(",", "") for m in re.findall(r"\$?\d[\d,]*(?:\.\d+)?%?", text)}


def _usd(v: float) -> str:
    return f"${v:,.0f}".replace(",", "")


def _ids(text: str) -> set[str]:
    return set(re.findall(r"\b(?:PO|RCV)-\d+\b", text))


def test_po_asks_match_record_fields(pos: list[dict]) -> None:
    checked = 0
    for p in pos:
        if "ask" not in p:
            continue
        ask = p["ask"]
        assert _ids(ask) == {p["po"]}, p["po"]
        expected = {_usd(p["total_usd"])}
        if "original_total_usd" in p:
            expected.add(_usd(p["original_total_usd"]))
        assert _nums(re.sub(r"(?:PO|RCV)-\d+", "", ask)) == expected, p["po"]
        checked += 1
    assert checked == 6


def test_receipt_asks_match_record_fields(receipts: list[dict]) -> None:
    checked = 0
    for r in receipts:
        if "ask" not in r:
            continue
        e = r["exception"]
        ask = r["ask"]
        rest = re.sub(r"(?:PO|RCV)-\d+", "", ask)
        if e["type"] == "damaged":
            assert _ids(ask) == {r["receipt_id"]}
            assert _nums(rest) == set()
        elif e["type"] == "over_shipment":
            assert _ids(ask) == {r["receipt_id"]}
            assert _nums(rest) == {f"{e['over_pct']:g}%"}
        elif e["type"] == "short_shipment":
            assert _ids(ask) == {r["receipt_id"]}
            assert _nums(rest) == {str(e["short_qty"])}
        elif e["type"] == "price_mismatch":
            assert _ids(ask) == {r["po"]}
            assert _nums(rest.replace("3-way", "")) == {f"{e['variance_pct']:g}%"}
        else:
            raise AssertionError(e["type"])
        checked += 1
    assert checked == 4


def test_supplier_asks_name_their_supplier(suppliers: list[dict]) -> None:
    probation = [s for s in suppliers if s["status"] == "probation"]
    assert probation
    for s in probation:
        assert s["name"] in s["ask"], s["name"]
