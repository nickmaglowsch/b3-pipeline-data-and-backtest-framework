"""
CVM IPE document index parser.

IMPORTANT — READ BEFORE MODIFYING:
As documented in tasks/ipe-structure-report.md, the CVM IPE dataset
(/DOC/IPE/DADOS/) is a DOCUMENT FILING INDEX, not structured financial
statement data. Each yearly ZIP contains a single CSV with these columns:

    CNPJ_Companhia, Nome_Companhia, Codigo_CVM, Data_Referencia, Categoria,
    Tipo, Especie, Assunto, Data_Entrega, Tipo_Apresentacao, Protocolo_Entrega,
    Versao, Link_Download

There are NO financial values (no CD_CONTA / VL_CONTA / ORDEM_EXERC equivalents).
The CSV contains only metadata and links to PDF/HTML documents.

Consequently:
- parse_ipe_zip() returns (filings_df, fundamentals_df) where fundamentals_df
  is ALWAYS EMPTY — no financial metrics can be extracted from IPE data.
- extract_ipe_company_index() extracts company CNPJ/name/CVM code for
  populating cvm_companies (company registry).

# NOTE (Task 05 — Path B):
# IPE does NOT contain share count data. Shares outstanding remain NULL for
# pre-2010 rows sourced from IPE. See tasks/ipe-structure-report.md for details.
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# IPE_ACCOUNT_MAP is intentionally empty — IPE has no account codes.
# Kept for structural parity with cvm_parser.py.
IPE_ACCOUNT_MAP: dict = {}

# IPE_SHARES_COLUMN: does not exist — IPE has no capital structure data.
# Path B confirmed by tasks/ipe-structure-report.md.
IPE_SHARES_COLUMN = None


# ── Private helpers ────────────────────────────────────────────────────────────

def _clean_cnpj(raw: str) -> Optional[str]:
    """Strip CNPJ formatting and return 14-digit string or None."""
    if not raw:
        return None
    digits = re.sub(r"\D", "", str(raw))
    return digits if len(digits) == 14 else None


def _parse_date(val) -> Optional[str]:
    """Parse date string, trying YYYY-MM-DD then DD/MM/YYYY. Returns YYYY-MM-DD or None."""
    if val is None:
        return None
    s = str(val).strip()
    if s in ("", "nan", "NaT", "None"):
        return None
    from datetime import datetime
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _open_zip(zip_path) -> Optional[zipfile.ZipFile]:
    """Open a ZipFile from Path or BytesIO. Returns None on error."""
    try:
        if isinstance(zip_path, (str, Path)):
            return zipfile.ZipFile(zip_path, "r")
        else:
            zip_path.seek(0)
            return zipfile.ZipFile(zip_path, "r")
    except Exception as e:
        logger.warning(f"Failed to open ZIP: {e}")
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_ipe_zip(
    zip_path, cnpj_ticker_map: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse an IPE document index ZIP file.

    Returns (filings_df, fundamentals_df).

    filings_df contains document filing metadata rows with doc_type='IPE'.
    fundamentals_df is ALWAYS EMPTY — IPE contains no financial statement data.
    See module docstring for details.
    """
    zf = _open_zip(zip_path)
    if zf is None:
        return pd.DataFrame(), pd.DataFrame()

    with zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            logger.warning("No CSV found in IPE ZIP")
            return pd.DataFrame(), pd.DataFrame()

        try:
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(f, sep=";", encoding="latin-1", dtype=str, low_memory=False)
        except Exception as e:
            logger.warning(f"Failed to read IPE CSV {csv_names[0]}: {e}")
            return pd.DataFrame(), pd.DataFrame()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Clean CNPJ
    cnpj_col = "CNPJ_Companhia" if "CNPJ_Companhia" in df.columns else None
    if cnpj_col is None:
        logger.warning("No CNPJ column found in IPE CSV")
        return pd.DataFrame(), pd.DataFrame()

    df["cnpj_clean"] = df[cnpj_col].apply(lambda x: _clean_cnpj(str(x) if x else ""))
    df = df[df["cnpj_clean"].notna() & (df["cnpj_clean"] != "")]

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Parse dates
    date_ref_col = "Data_Referencia" if "Data_Referencia" in df.columns else None
    date_entry_col = "Data_Entrega" if "Data_Entrega" in df.columns else None
    versao_col = "Versao" if "Versao" in df.columns else None
    cvm_code_col = "Codigo_CVM" if "Codigo_CVM" in df.columns else None
    nome_col = "Nome_Companhia" if "Nome_Companhia" in df.columns else None

    df["period_end"] = df[date_ref_col].apply(_parse_date) if date_ref_col else None
    df["filing_date"] = df[date_entry_col].apply(_parse_date) if date_entry_col else None
    # Use period_end as filing_date fallback
    df["filing_date"] = df.apply(
        lambda r: r["filing_date"] if r["filing_date"] else r["period_end"],
        axis=1
    )

    # Version: use 1 as default since many rows have empty Versao
    df["version"] = df[versao_col].apply(
        lambda v: int(float(v)) if v and str(v).strip() not in ("", "nan") else 1
    ) if versao_col else 1

    # Fiscal year from period_end
    df["fiscal_year"] = df["period_end"].apply(
        lambda d: int(d[:4]) if d else None
    )

    # Build filing_id: {cnpj}_IPE_{period_end}_{version}
    df["filing_id"] = df.apply(
        lambda r: f"{r['cnpj_clean']}_IPE_{r['period_end'] or 'unknown'}_{r['version']}",
        axis=1,
    )

    # Ticker from map
    df["ticker"] = df["cnpj_clean"].map(cnpj_ticker_map).where(
        df["cnpj_clean"].isin(cnpj_ticker_map), None
    )

    # Build filings_df — drop duplicate filing_ids, keep last
    filings_rows = []
    seen_ids = set()
    for _, row in df.iterrows():
        fid = row["filing_id"]
        if fid in seen_ids:
            continue
        seen_ids.add(fid)
        filings_rows.append({
            "filing_id": fid,
            "cnpj": row["cnpj_clean"],
            "ticker": row.get("ticker"),
            "doc_type": "IPE",
            "period_end": row["period_end"],
            "filing_date": row["filing_date"],
            "filing_version": int(row["version"]),
            "fiscal_year": row.get("fiscal_year"),
            "quarter": None,
            "source_file": csv_names[0],
        })
    filings_df = pd.DataFrame(filings_rows)

    # fundamentals_df is always empty for IPE — no financial data available
    # Ratio columns (pe_ratio, pb_ratio, ev_ebitda) intentionally excluded — computed dynamically
    fundamentals_df = pd.DataFrame(columns=[
        "filing_id", "cnpj", "ticker", "period_end", "filing_date", "filing_version",
        "doc_type", "fiscal_year", "quarter",
        "revenue", "net_income", "ebitda", "total_assets", "equity", "net_debt",
        "shares_outstanding",
    ])

    return filings_df, fundamentals_df


def extract_ipe_company_index(zip_path) -> list:
    """Extract unique (cnpj, cvm_code, company_name) tuples from an IPE ZIP.

    Mirrors extract_company_index() in cvm_parser.py.
    Used by the orchestrator to populate cvm_companies before processing.

    Returns a list of (cnpj_14digits, cvm_code_str, company_name) tuples.
    """
    result = {}
    zf = _open_zip(zip_path)
    if zf is None:
        return []

    with zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            return []
        try:
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(
                    f, sep=";", encoding="latin-1", dtype=str,
                    usecols=lambda c: c in (
                        "CNPJ_Companhia", "Codigo_CVM", "Nome_Companhia"
                    ),
                    low_memory=False,
                )
        except Exception as e:
            logger.warning(f"Failed to read IPE CSV for company index: {e}")
            return []

    cnpj_col = "CNPJ_Companhia"
    cvm_col = "Codigo_CVM" if "Codigo_CVM" in df.columns else None
    nome_col = "Nome_Companhia" if "Nome_Companhia" in df.columns else None

    for _, row in df.drop_duplicates(subset=[cnpj_col]).iterrows():
        cnpj = _clean_cnpj(str(row.get(cnpj_col, "") or ""))
        if cnpj:
            cvm_code = str(row.get(cvm_col, "") or "").strip() if cvm_col else ""
            company_name = str(row.get(nome_col, "") or "").strip() if nome_col else ""
            result[cnpj] = (cvm_code, company_name)

    return [(cnpj, cvm_code, name) for cnpj, (cvm_code, name) in result.items()]
