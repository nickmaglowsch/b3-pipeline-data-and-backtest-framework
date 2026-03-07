"""
CVM CAD (company register) parser.

Parses the bulk CSV file cad_cia_aberta.csv from CVM's open data portal.
This file contains registration metadata for every company that has ever
filed with CVM, including listing/delisting dates for survivorship bias removal.

Output columns: cnpj, cvm_code, company_name, listing_date, delisting_date, is_active
"""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


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


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_cad_file(file_path: Union[Path, io.StringIO]) -> pd.DataFrame:
    """Parse the CVM CAD company register CSV.

    Accepts either a filesystem Path or an io.StringIO (for testing).

    Returns a DataFrame with columns:
        cnpj, cvm_code, company_name, listing_date, delisting_date, is_active

    - cnpj: 14-digit string (rows with invalid/empty CNPJ are dropped)
    - listing_date: YYYY-MM-DD string or None
    - delisting_date: YYYY-MM-DD string or None (None for active companies)
    - is_active: bool (True when SIT == 'ATIVO')
    """
    # Read the CSV — supports Path or StringIO
    try:
        df = pd.read_csv(
            file_path,
            sep=";",
            encoding="latin-1",
            dtype=str,
            low_memory=False,
            usecols=lambda c: c in (
                "CNPJ_CIA", "CD_CVM", "DENOM_SOCIAL", "DT_REG", "DT_CANCEL", "SIT"
            ),
        )
    except Exception as e:
        logger.error(f"Failed to read CAD CSV: {e}")
        return pd.DataFrame(columns=["cnpj", "cvm_code", "company_name",
                                     "listing_date", "delisting_date", "is_active"])

    # Ensure required columns exist (handle missing gracefully)
    for col in ("CNPJ_CIA", "CD_CVM", "DENOM_SOCIAL", "DT_REG", "DT_CANCEL", "SIT"):
        if col not in df.columns:
            df[col] = None

    # Clean CNPJ
    df["cnpj"] = df["CNPJ_CIA"].apply(lambda x: _clean_cnpj(str(x) if x else ""))
    df = df.dropna(subset=["cnpj"])
    df = df[df["cnpj"] != ""]

    # Parse dates
    df["listing_date"] = df["DT_REG"].apply(_parse_date)
    df["delisting_date"] = df["DT_CANCEL"].apply(_parse_date)

    # is_active: True only when SIT == 'ATIVO'
    df["is_active"] = df["SIT"].apply(
        lambda s: str(s).strip().upper() == "ATIVO" if s and str(s) not in ("nan", "None") else False
    )

    # Rename and select output columns
    df = df.rename(columns={
        "CD_CVM": "cvm_code",
        "DENOM_SOCIAL": "company_name",
    })

    result = df[["cnpj", "cvm_code", "company_name", "listing_date", "delisting_date", "is_active"]].copy()

    # Normalise: replace pandas NaT/NaN strings with None for date columns
    for col in ("listing_date", "delisting_date"):
        result[col] = result[col].where(result[col].notna(), None)
        result[col] = result[col].apply(
            lambda v: None if v in (None, "nan", "NaT", "None", "") else v
        )

    logger.info(f"Parsed CAD file: {len(result):,} companies")
    return result.reset_index(drop=True)


def extract_cad_summary(df: pd.DataFrame) -> dict:
    """Return summary statistics for the CAD DataFrame.

    Returns:
        {'total': int, 'active': int, 'delisted': int, 'with_listing_date': int}
    """
    total = len(df)
    active = int(df["is_active"].sum())
    delisted = int(df["delisting_date"].notna().sum())
    with_listing_date = int(df["listing_date"].notna().sum())
    return {
        "total": total,
        "active": active,
        "delisted": delisted,
        "with_listing_date": with_listing_date,
    }
