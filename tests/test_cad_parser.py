"""
Tests for the CAD company register downloader and parser (Task 02 — TDD).

All parser tests use in-memory synthetic CSV data — no network calls.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from b3_pipeline.cad_parser import parse_cad_file, extract_cad_summary


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _cad_csv(rows: list[dict]) -> io.StringIO:
    """Build an in-memory StringIO with semicolon-delimited latin-1 CAD CSV."""
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, sep=";", index=False)
    buf.seek(0)
    return buf


def _base_row(
    cnpj="00000000000191",
    denom="EMPRESA TESTE SA",
    cd_cvm="001023",
    dt_reg="2000-01-15",
    dt_cancel=None,
    sit="ATIVO",
) -> dict:
    return {
        "CNPJ_CIA": cnpj,
        "DENOM_SOCIAL": denom,
        "CD_CVM": cd_cvm,
        "DT_REG": dt_reg,
        "DT_CANCEL": dt_cancel if dt_cancel is not None else "",
        "SIT": sit,
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_parse_cad_file_active_company():
    """Active company: is_active=True, delisting_date=None, 14-digit CNPJ, YYYY-MM-DD dates."""
    rows = [_base_row(cnpj="00.000.000/0001-91", sit="ATIVO", dt_reg="2000-03-10")]
    df = parse_cad_file(_cad_csv(rows))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["is_active"] is True or row["is_active"] == True
    assert row["delisting_date"] is None or pd.isna(row["delisting_date"])
    assert row["cnpj"] == "00000000000191"
    assert row["listing_date"] == "2000-03-10"


def test_parse_cad_file_cancelled_company():
    """Cancelled company: is_active=False, delisting_date is a valid YYYY-MM-DD string."""
    rows = [_base_row(sit="CANCELADO", dt_cancel="2015-06-30")]
    df = parse_cad_file(_cad_csv(rows))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["is_active"] is False or row["is_active"] == False
    assert row["delisting_date"] == "2015-06-30"


def test_parse_cad_file_drops_invalid_cnpj():
    """Rows with empty or malformed CNPJ are dropped."""
    rows = [
        _base_row(cnpj="00.000.000/0001-91"),  # valid
        _base_row(cnpj=""),                      # empty — should be dropped
        _base_row(cnpj="INVALID"),               # malformed — should be dropped
    ]
    df = parse_cad_file(_cad_csv(rows))
    assert len(df) == 1
    assert df.iloc[0]["cnpj"] == "00000000000191"


def test_parse_cad_file_parses_dd_mm_yyyy_dates():
    """DD/MM/YYYY dates are converted to YYYY-MM-DD."""
    rows = [_base_row(cnpj="00.000.000/0001-91", dt_reg="15/03/2001", dt_cancel="30/06/2015", sit="CANCELADO")]
    df = parse_cad_file(_cad_csv(rows))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["listing_date"] == "2001-03-15"
    assert row["delisting_date"] == "2015-06-30"


def test_parse_cad_file_multiple_status_values():
    """is_active is True only for ATIVO rows."""
    rows = [
        _base_row(cnpj="00000000000191", sit="ATIVO"),
        _base_row(cnpj="00000000000282", sit="CANCELADO"),
        _base_row(cnpj="00000000000373", sit="SUSPENSO"),
    ]
    df = parse_cad_file(_cad_csv(rows))
    assert len(df) == 3
    active = df[df["cnpj"] == "00000000000191"].iloc[0]
    cancelled = df[df["cnpj"] == "00000000000282"].iloc[0]
    suspended = df[df["cnpj"] == "00000000000373"].iloc[0]
    assert active["is_active"] == True
    assert cancelled["is_active"] == False
    assert suspended["is_active"] == False


def test_extract_cad_summary_counts():
    """extract_cad_summary returns correct total/active/delisted/with_listing_date counts."""
    data = pd.DataFrame({
        "cnpj":          ["00000000000191", "00000000000282", "00000000000373"],
        "cvm_code":      ["001", "002", "003"],
        "company_name":  ["A", "B", "C"],
        "listing_date":  ["2000-01-01", "2001-06-15", None],
        "delisting_date":[None, "2015-06-30", None],
        "is_active":     [True, False, True],
    })
    summary = extract_cad_summary(data)
    assert summary["total"] == 3
    assert summary["active"] == 2
    assert summary["delisted"] == 1
    assert summary["with_listing_date"] == 2


def test_parse_cad_file_returns_required_columns():
    """Output DataFrame has all 6 required columns."""
    rows = [_base_row()]
    df = parse_cad_file(_cad_csv(rows))
    for col in ["cnpj", "cvm_code", "company_name", "listing_date", "delisting_date", "is_active"]:
        assert col in df.columns, f"Missing column: {col}"


def test_parse_cad_file_null_listing_date():
    """Rows with empty DT_REG produce None listing_date (not crash)."""
    rows = [_base_row(dt_reg="")]
    df = parse_cad_file(_cad_csv(rows))
    assert len(df) == 1
    assert df.iloc[0]["listing_date"] is None or pd.isna(df.iloc[0]["listing_date"])
