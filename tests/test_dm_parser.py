"""
Tests for the Download Múltiplo (CVMWIN legacy DBF) parser.

All tests build synthetic in-memory DBF/ZIP fixtures -- no real CVM data,
network calls, or credentials required (matches test_cvm_parser.py's style).
"""
from __future__ import annotations

import io
import struct
import zipfile

import pandas as pd
import pytest

from b3_pipeline.dm_parser import (
    parse_dfp_dbf_zip,
    parse_itr_dbf_zip,
    parse_ian_dbf_zip,
    read_dbf,
)


# ── DBF fixture writer (test-only; mirrors the dBase III format read_dbf parses) ──

def _write_dbf(records: list[dict]) -> bytes:
    """Build a minimal valid dBase III file from a list of same-shaped dicts.

    Field length is the max string length observed for that column (min 4),
    which is all read_dbf needs -- it doesn't care about field *type*.
    """
    if not records:
        raise ValueError("need at least one record to infer field widths")
    field_names = list(records[0].keys())
    field_lens = {
        name: max(4, max(len(str(r[name])) for r in records))
        for name in field_names
    }

    header_size = 32 + 32 * len(field_names) + 1
    record_size = 1 + sum(field_lens[n] for n in field_names)

    buf = bytearray()
    buf += bytes([0x03])              # dBase III, no memo
    buf += bytes([0, 1, 1])           # last-update date YY MM DD (unused by reader)
    buf += struct.pack("<I", len(records))
    buf += struct.pack("<H", header_size)
    buf += struct.pack("<H", record_size)
    buf += bytes(20)                  # reserved

    for name in field_names:
        name_bytes = name.encode("latin-1")[:10].ljust(11, b"\x00")
        buf += name_bytes
        buf += b"C"                   # type: character (reader ignores type)
        buf += bytes(4)               # data address (unused)
        buf += bytes([field_lens[name]])
        buf += bytes([0])             # decimal count
        buf += bytes(14)              # reserved

    buf += bytes([0x0D])              # header terminator

    for r in records:
        buf += b" "                   # not deleted
        for name in field_names:
            val = str(r[name]).encode("latin-1")[: field_lens[name]]
            buf += val.ljust(field_lens[name], b" ")

    buf += bytes([0x1A])              # EOF marker
    return bytes(buf)


def _make_zip(members: dict) -> io.BytesIO:
    """members: {filename: list[dict]} -> in-memory ZIP of DBF files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for filename, records in members.items():
            zf.writestr(filename, _write_dbf(records))
    buf.seek(0)
    return buf


CNPJ_FMT = "33.000.167/0001-01"
CNPJ_CLEAN = "33000167000101"


def _ctr_row(tipo_doc="2", data_emiss="20070315", data_dfp="20061231"):
    return {
        "CODCVM": "1234",
        "DATADFP": data_dfp,
        "TIPO_DOC": tipo_doc,
        "CGC": CNPJ_FMT,
        "RAZAO_SOC": "COMPANHIA TESTE SA",
        "DATA_EMISS": data_emiss,
    }


def _config_row(escala="02", escala_qtd="01", data_dfp="20061231"):
    return {"CODCVM": "1234", "DATADFP": data_dfp, "ESCALA": escala, "ESCALA_QTD": escala_qtd}


def _stmt_row(codconta, desconta, valor3, data_dfp="20061231", valor1="0", valor2="0"):
    return {
        "CODCVM": "1234", "DATADFP": data_dfp,
        "CODCONTA": codconta, "DESCONTA": desconta,
        "VALOR1": valor1, "VALOR2": valor2, "VALOR3": valor3,
    }


def _basic_dfp_zip(escala="02", tipo_doc="2"):
    return _make_zip({
        "CVM.CTR": [_ctr_row(tipo_doc=tipo_doc)],
        "CONFIG.001": [_config_row(escala=escala)],
        "DFPCBPAE.001": [_stmt_row("1", "ATIVO TOTAL", "5000000")],
        "DFPCBPPE.001": [_stmt_row("2.03", "PATRIMONIO LIQUIDO", "2000000")],
        "DFPCDERE.001": [
            _stmt_row("3.01", "RECEITA LIQUIDA DE VENDAS E/OU SERVICOS", "1000000"),
            _stmt_row("3.11", "LUCRO LIQUIDO DO EXERCICIO", "150000"),
        ],
    })


# ──────────────────────────────────────────────────────────────────────────────
# 0. DBF reader roundtrip (foundation for everything else)
# ──────────────────────────────────────────────────────────────────────────────

def test_read_dbf_roundtrip():
    """read_dbf should recover exactly what _write_dbf encoded."""
    records = [
        {"CODCVM": "1234", "DESCONTA": "ATIVO TOTAL", "VALOR3": "5000000"},
        {"CODCVM": "5678", "DESCONTA": "PASSIVO", "VALOR3": "1234"},
    ]
    raw = _write_dbf(records)
    df = read_dbf(io.BytesIO(raw))
    assert list(df["CODCVM"]) == ["1234", "5678"]
    assert list(df["DESCONTA"]) == ["ATIVO TOTAL", "PASSIVO"]
    assert list(df["VALOR3"]) == ["5000000", "1234"]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Core metric extraction + scale (mil -- already thousands)
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_dfp_dbf_zip_extracts_metrics_scale_thousand():
    fake_zip = _basic_dfp_zip(escala="02")
    filings_df, fundamentals_df = parse_dfp_dbf_zip(fake_zip, {CNPJ_CLEAN: "PETR"})

    assert not fundamentals_df.empty
    row = fundamentals_df.iloc[0]
    assert row["total_assets"] == pytest.approx(5_000_000.0)
    assert row["equity"] == pytest.approx(2_000_000.0)
    assert row["revenue"] == pytest.approx(1_000_000.0)
    assert row["net_income"] == pytest.approx(150_000.0)
    assert row["ticker"] == "PETR"
    assert row["doc_type"] == "DFP"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Scale conversion: ESCALA='01' (unidade) must be divided by 1000
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_dfp_dbf_zip_scale_unit_converts_to_thousands():
    fake_zip = _basic_dfp_zip(escala="01")
    _, fundamentals_df = parse_dfp_dbf_zip(fake_zip, {CNPJ_CLEAN: "PETR"})

    row = fundamentals_df.iloc[0]
    # Raw VALOR3=5,000,000 in whole BRL -> 5,000 thousand BRL
    assert row["total_assets"] == pytest.approx(5_000.0)
    assert row["equity"] == pytest.approx(2_000.0)
    assert row["revenue"] == pytest.approx(1_000.0)
    assert row["net_income"] == pytest.approx(150.0)


# ──────────────────────────────────────────────────────────────────────────────
# 3. period_end / filing_date / filing_id
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_dfp_dbf_zip_dates_and_filing_id():
    fake_zip = _basic_dfp_zip()
    filings_df, fundamentals_df = parse_dfp_dbf_zip(fake_zip, {CNPJ_CLEAN: "PETR"})

    assert filings_df.iloc[0]["period_end"] == "2006-12-31"
    assert filings_df.iloc[0]["filing_date"] == "2007-03-15"
    assert filings_df.iloc[0]["cnpj"] == CNPJ_CLEAN
    assert filings_df.iloc[0]["fiscal_year"] == 2006
    # filing_version is derived from filing_date (YYYYMMDD) so it is stable
    # across ZIPs (one ZIP per submission day).
    expected_id = f"{CNPJ_CLEAN}_DFP_2006-12-31_20070315"
    assert fundamentals_df.iloc[0]["filing_id"] == expected_id


# ──────────────────────────────────────────────────────────────────────────────
# 4. TIPO_DOC filter: constant-currency variant must be excluded
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_dfp_dbf_zip_excludes_constant_currency_variant():
    """CVM.CTR rows with TIPO_DOC='1' (moeda constante) should be ignored."""
    fake_zip = _basic_dfp_zip(tipo_doc="1")  # only the constant-currency row exists
    filings_df, fundamentals_df = parse_dfp_dbf_zip(fake_zip, {CNPJ_CLEAN: "PETR"})

    assert filings_df.empty
    assert fundamentals_df.empty


# ──────────────────────────────────────────────────────────────────────────────
# 5. Consolidated statement preferred over individual
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_dfp_dbf_zip_prefers_consolidated():
    fake_zip = _make_zip({
        "CVM.CTR": [_ctr_row()],
        "CONFIG.001": [_config_row()],
        "DFPCBPAE.001": [_stmt_row("1", "ATIVO TOTAL", "9000000")],   # consolidated
        "DFPBPAE.001": [_stmt_row("1", "ATIVO TOTAL", "1000000")],    # individual (ignored)
        "DFPCBPPE.001": [_stmt_row("2.03", "PATRIMONIO LIQUIDO", "2000000")],
        "DFPCDERE.001": [_stmt_row("3.01", "RECEITA LIQUIDA DE VENDAS E/OU SERVICOS", "1000000")],
    })
    _, fundamentals_df = parse_dfp_dbf_zip(fake_zip, {})
    assert fundamentals_df.iloc[0]["total_assets"] == pytest.approx(9_000_000.0)


def test_parse_dfp_dbf_zip_nonconsolidated_income_fallback():
    """Companies with no consolidated statements fall back to the
    non-consolidated files -- income stem is DFPDERE (layout: DFP{C}DERE)."""
    fake_zip = _make_zip({
        "CVM.CTR": [_ctr_row()],
        "CONFIG.001": [_config_row()],
        "DFPBPAE.001": [_stmt_row("1", "ATIVO TOTAL", "5000000")],
        "DFPBPPE.001": [_stmt_row("2.03", "PATRIMONIO LIQUIDO", "2000000")],
        "DFPDERE.001": [
            _stmt_row("3.01", "RECEITA LIQUIDA DE VENDAS E/OU SERVICOS", "1000000"),
            _stmt_row("3.11", "LUCRO LIQUIDO DO EXERCICIO", "150000"),
        ],
    })
    _, fundamentals_df = parse_dfp_dbf_zip(fake_zip, {})
    row = fundamentals_df.iloc[0]
    assert row["revenue"] == pytest.approx(1_000_000.0)
    assert row["net_income"] == pytest.approx(150_000.0)


# ──────────────────────────────────────────────────────────────────────────────
# 6. filing_version increments on restatement
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_dfp_dbf_zip_filing_version_increments_on_restatement():
    fake_zip = _make_zip({
        "CVM.CTR": [
            _ctr_row(data_emiss="20070315"),
            {**_ctr_row(data_emiss="20070601"), "CODCVM": "1234"},  # restatement, same period
        ],
        "CONFIG.001": [_config_row()],
        "DFPCBPAE.001": [_stmt_row("1", "ATIVO TOTAL", "5000000")],
        "DFPCBPPE.001": [_stmt_row("2.03", "PATRIMONIO LIQUIDO", "2000000")],
        "DFPCDERE.001": [_stmt_row("3.01", "RECEITA LIQUIDA DE VENDAS E/OU SERVICOS", "1000000")],
    })
    filings_df, fundamentals_df = parse_dfp_dbf_zip(fake_zip, {})
    versions = sorted(filings_df["filing_version"].tolist())
    assert versions == [20070315, 20070601]  # date-derived, later filing -> higher
    ids = set(fundamentals_df["filing_id"])
    assert f"{CNPJ_CLEAN}_DFP_2006-12-31_20070315" in ids
    assert f"{CNPJ_CLEAN}_DFP_2006-12-31_20070601" in ids


def test_filing_version_distinct_across_zips():
    """Download Múltiplo delivers one ZIP per (company, submission day): an
    original filing and a restatement in SEPARATE ZIPs must not collide on
    the same filing_id (which would let INSERT OR REPLACE overwrite the
    earlier point-in-time row)."""
    def _zip_for(data_emiss):
        return _make_zip({
            "CVM.CTR": [_ctr_row(data_emiss=data_emiss)],
            "CONFIG.001": [_config_row()],
            "DFPCBPAE.001": [_stmt_row("1", "ATIVO TOTAL", "5000000")],
            "DFPCBPPE.001": [_stmt_row("2.03", "PATRIMONIO LIQUIDO", "2000000")],
            "DFPCDERE.001": [_stmt_row("3.01", "RECEITA LIQUIDA DE VENDAS E/OU SERVICOS", "1000000")],
        })

    filings_a, _ = parse_dfp_dbf_zip(_zip_for("20070315"), {})
    filings_b, _ = parse_dfp_dbf_zip(_zip_for("20070601"), {})

    id_a, id_b = filings_a.iloc[0]["filing_id"], filings_b.iloc[0]["filing_id"]
    assert id_a != id_b
    assert filings_b.iloc[0]["filing_version"] > filings_a.iloc[0]["filing_version"]


# ──────────────────────────────────────────────────────────────────────────────
# 7. ITR quarter inference
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_itr_dbf_zip_sets_quarter():
    fake_zip = _make_zip({
        "CVM.CTR": [_ctr_row(tipo_doc="4", data_dfp="20070331", data_emiss="20070515")],
        "CONFIG.001": [_config_row(data_dfp="20070331")],
        "ITRCBPAE.001": [_stmt_row("1", "ATIVO TOTAL", "5000000", data_dfp="20070331")],
        "ITRCBPPE.001": [_stmt_row("2.03", "PATRIMONIO LIQUIDO", "2000000", data_dfp="20070331")],
        "ITRCDERE.001": [_stmt_row("3.01", "RECEITA LIQUIDA DE VENDAS E/OU SERVICOS", "250000", data_dfp="20070331")],
    })
    filings_df, fundamentals_df = parse_itr_dbf_zip(fake_zip, {})
    assert filings_df.iloc[0]["quarter"] == 1
    assert filings_df.iloc[0]["doc_type"] == "ITR"
    assert fundamentals_df.iloc[0]["revenue"] == pytest.approx(250_000.0)


# ──────────────────────────────────────────────────────────────────────────────
# 8. IAN shares_outstanding: sum across share classes + ESCALA_QTD scaling
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_ian_dbf_zip_sums_shares_with_scale():
    fake_zip = _make_zip({
        "CVM.CTR": [_ctr_row(tipo_doc="5", data_dfp="20061231", data_emiss="20070401")],
        "CONFIG.001": [_config_row(escala_qtd="02", data_dfp="20061231")],  # mil -> x1000
        "IANCAPSO.001": [
            {"CODCVM": "1234", "DATAIAN": "20061231", "ITEM": "1", "DESCRICAO": "ORDINARIA", "QTDEACOES": "1000"},
            {"CODCVM": "1234", "DATAIAN": "20061231", "ITEM": "2", "DESCRICAO": "PREFERENCIAL", "QTDEACOES": "2000"},
        ],
    })
    filings_df, fundamentals_df = parse_ian_dbf_zip(fake_zip, {CNPJ_CLEAN: "PETR"})

    assert not fundamentals_df.empty
    # (1000 + 2000) shares, scale 'mil' -> x1000 = 3,000,000 shares
    assert fundamentals_df.iloc[0]["shares_outstanding"] == pytest.approx(3_000_000.0)
    assert fundamentals_df.iloc[0]["doc_type"] == "IAN"
    assert filings_df.iloc[0]["quarter"] is None


# ──────────────────────────────────────────────────────────────────────────────
# 9. Unmapped CNPJ -> ticker is None/NaN
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_dfp_dbf_zip_ticker_none_for_unmapped():
    fake_zip = _basic_dfp_zip()
    _, fundamentals_df = parse_dfp_dbf_zip(fake_zip, {})
    ticker_val = fundamentals_df["ticker"].iloc[0]
    assert ticker_val is None or pd.isna(ticker_val)


# ──────────────────────────────────────────────────────────────────────────────
# 10. Missing CVM.CTR -> empty result, no crash
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_dfp_dbf_zip_missing_ctr_returns_empty():
    fake_zip = _make_zip({
        "DFPCBPAE.001": [_stmt_row("1", "ATIVO TOTAL", "5000000")],
    })
    filings_df, fundamentals_df = parse_dfp_dbf_zip(fake_zip, {})
    assert filings_df.empty
    assert fundamentals_df.empty
