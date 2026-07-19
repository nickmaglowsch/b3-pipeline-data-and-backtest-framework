"""
CVM FCA (Formulário Cadastral) parser.

Parses the valor_mobiliario CSV inside FCA yearly ZIPs from CVM's open data
portal. This table maps CNPJ -> negotiated ticker (Codigo_Negociacao) with
trading start/end dates. Unlike the B3 API (which only lists currently listed
companies), FCA covers companies that have since delisted — closing the
survivorship hole in CNPJ -> ticker mapping.
"""
from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

import pandas as pd

from .cvm_parser import _clean_cnpj, _parse_date

logger = logging.getLogger(__name__)

# Case-insensitive candidate names per output column (CVM has renamed columns
# across file versions; look up defensively).
_COL_CANDIDATES = {
    "cnpj": ("cnpj_companhia", "cnpj_cia"),
    "ticker": ("codigo_negociacao",),
    "market": ("mercado",),
    "start_date": ("data_inicio_negociacao", "data_inicio_listagem"),
    "end_date": ("data_fim_negociacao", "data_fim_listagem"),
}

# Standard B3 equity ticker: 4-letter root + 1-2 digit suffix (e.g. PETR4, BOVA11)
_TICKER_RE = re.compile(r"^[A-Z]{4}\d{1,2}$")


def parse_fca_sectors(zip_path) -> pd.DataFrame:
    """Parse the fca_cia_aberta_geral_{year}.csv inside an FCA ZIP.

    Returns point-in-time CVM sector classification (Setor_Atividade) per company,
    survivorship-free (FCA covers companies that have since delisted).

    Accepts a filesystem path or a BytesIO (for testing).

    Returns a DataFrame with columns: cnpj, ref_date, sector.
    Rows without a valid 14-digit CNPJ or a sector are dropped.
    """
    empty = pd.DataFrame(columns=["cnpj", "ref_date", "sector"])

    if not isinstance(zip_path, (str, Path)):
        zip_path.seek(0)
    with zipfile.ZipFile(zip_path, "r") as zf:
        matches = [
            n for n in zf.namelist()
            if "geral" in n.lower() and n.lower().endswith(".csv")
        ]
        if not matches:
            logger.warning(f"No geral CSV found in FCA zip {zip_path}")
            return empty
        with zf.open(matches[0]) as f:
            df = pd.read_csv(f, sep=";", encoding="latin-1", dtype=str, low_memory=False)

    cnpj_col = _find_col(df, ("cnpj_companhia", "cnpj_cia"))
    if cnpj_col is None or "Setor_Atividade" not in df.columns:
        raise ValueError(
            f"FCA geral CSV missing expected columns (need CNPJ + Setor_Atividade); "
            f"found: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["cnpj"] = df[cnpj_col].apply(lambda x: _clean_cnpj(str(x)))
    out["ref_date"] = df["Data_Referencia"].apply(_parse_date)
    out["sector"] = df["Setor_Atividade"].astype(str).str.strip()

    out = out[out["cnpj"].notna() & out["sector"].notna() & (out["sector"] != "")]
    out = out[out["sector"].str.lower() != "nan"]
    if out.empty:
        return empty
    out = out.drop_duplicates(subset=["cnpj", "ref_date"], keep="last")
    return out[["cnpj", "ref_date", "sector"]].reset_index(drop=True)


def _find_col(df: pd.DataFrame, names: tuple) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    for name in names:
        if name in lower:
            return lower[name]
    return None


def parse_fca_zip(zip_path) -> pd.DataFrame:
    """Parse the fca_cia_aberta_valor_mobiliario_{year}.csv inside an FCA ZIP.

    Accepts a filesystem path or a BytesIO (for testing).

    Returns a DataFrame with columns:
        cnpj, ticker, ticker_root, market, start_date, end_date, source
    Rows without a valid 14-digit CNPJ or a standard B3 ticker are dropped.
    """
    empty = pd.DataFrame(
        columns=["cnpj", "ticker", "ticker_root", "market", "start_date", "end_date", "source"]
    )

    if not isinstance(zip_path, (str, Path)):
        zip_path.seek(0)
    with zipfile.ZipFile(zip_path, "r") as zf:
        matches = [
            n for n in zf.namelist()
            if "valor_mobiliario" in n.lower() and n.lower().endswith(".csv")
        ]
        if not matches:
            logger.warning(f"No valor_mobiliario CSV found in FCA zip {zip_path}")
            return empty
        with zf.open(matches[0]) as f:
            df = pd.read_csv(f, sep=";", encoding="latin-1", dtype=str, low_memory=False)

    cols = {out: _find_col(df, names) for out, names in _COL_CANDIDATES.items()}
    if cols["cnpj"] is None or cols["ticker"] is None:
        raise ValueError(
            f"FCA valor_mobiliario CSV missing expected columns "
            f"(need CNPJ_Companhia and Codigo_Negociacao); found: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["cnpj"] = df[cols["cnpj"]].apply(lambda x: _clean_cnpj(str(x)))
    out["ticker"] = df[cols["ticker"]].fillna("").astype(str).str.strip().str.upper()
    out["market"] = df[cols["market"]].astype(str).str.strip() if cols["market"] else None
    out["start_date"] = df[cols["start_date"]].apply(_parse_date) if cols["start_date"] else None
    out["end_date"] = df[cols["end_date"]].apply(_parse_date) if cols["end_date"] else None

    out = out[out["cnpj"].notna() & out["ticker"].str.match(_TICKER_RE)]
    if out.empty:
        return empty

    out["ticker_root"] = out["ticker"].str[:4]
    out["source"] = "FCA"
    out = out.drop_duplicates(subset=["cnpj", "ticker", "start_date"], keep="last")

    logger.info(f"Parsed FCA valor_mobiliario: {len(out):,} ticker mappings")
    return out[["cnpj", "ticker", "ticker_root", "market", "start_date", "end_date", "source"]].reset_index(drop=True)
