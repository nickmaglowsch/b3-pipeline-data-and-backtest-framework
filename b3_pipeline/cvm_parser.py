"""
CVM bulk CSV parser.

Parses DFP, ITR, and FRE ZIP files downloaded from CVM's open data portal.
Each ZIP contains semicolon-delimited CSV files (latin-1 encoded) with
financial statement data in long format.

All values in VL_CONTA are in thousands of BRL — stored as-is; do not multiply.
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

# ── Account code → metric mapping ─────────────────────────────────────────────

# Income statement (DRE files — *DRE*con*)
INCOME_ACCOUNT_MAP = {
    "3.01": "revenue",       # Receita Líquida
    "3.05": "ebitda",        # EBIT (used as EBITDA proxy — no D&A subtracted)
    "3.11": "net_income",    # Lucro/Prejuízo do Período
}

# Balance sheet assets (BPA files — *BPA*con*)
ASSET_ACCOUNT_MAP = {
    "1":       "total_assets",  # Total do Ativo
    "1.01.01": "_cash",          # Caixa e Equivalentes (intermediate for net_debt)
}

# Balance sheet liabilities (BPP files — *BPP*con*)
LIABILITY_ACCOUNT_MAP = {
    "2.01.04": "_short_debt",  # Empréstimos e Financiamentos (CP)
    "2.02.01": "_long_debt",   # Empréstimos e Financiamentos (LP)
    "2.03":    "equity",       # Patrimônio Líquido
}

ALL_ACCOUNT_MAP = {**INCOME_ACCOUNT_MAP, **ASSET_ACCOUNT_MAP, **LIABILITY_ACCOUNT_MAP}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_cnpj(raw: str) -> Optional[str]:
    """Strip CNPJ formatting and return 14-digit string or None."""
    if not raw:
        return None
    digits = re.sub(r"\D", "", str(raw))
    return digits if len(digits) == 14 else None


def _parse_date(val: str) -> Optional[str]:
    """Parse date string, trying YYYY-MM-DD first then DD/MM/YYYY. Returns YYYY-MM-DD or None."""
    if not val or str(val).strip() in ("", "nan", "NaT"):
        return None
    val = str(val).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            from datetime import datetime
            return datetime.strptime(val, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _infer_quarter(period_end: str) -> Optional[int]:
    """Infer quarter from period_end date string (YYYY-MM-DD). Returns 1/2/3 or None."""
    if not period_end:
        return None
    try:
        month = int(period_end[5:7])
        if month == 3:
            return 1
        elif month == 6:
            return 2
        elif month == 9:
            return 3
    except (ValueError, IndexError):
        pass
    return None


def extract_company_index(zip_path) -> list:
    """Extract unique (cnpj, cvm_code, company_name) tuples from a CVM ZIP file.

    Reads the first available CSV in the ZIP — all CVM files share the same
    CNPJ_CIA / CD_CVM / DENOM_CIA columns, so one CSV is enough.

    Returns a list of (cnpj_14digits, cvm_code_str, company_name) tuples.
    """
    result = {}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                return []
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(
                    f, sep=";", encoding="latin-1", dtype=str,
                    usecols=["CNPJ_CIA", "CD_CVM", "DENOM_CIA"],
                    low_memory=False,
                )
            for _, row in df.drop_duplicates(subset=["CNPJ_CIA"]).iterrows():
                cnpj = _clean_cnpj(str(row.get("CNPJ_CIA", "") or ""))
                if cnpj:
                    result[cnpj] = (
                        str(row.get("CD_CVM", "") or "").strip(),
                        str(row.get("DENOM_CIA", "") or "").strip(),
                    )
    except Exception as e:
        logger.warning(f"Failed to extract company index from {zip_path}: {e}")
    return [(cnpj, cvm_code, name) for cnpj, (cvm_code, name) in result.items()]


def _load_csv_from_zip(zf: zipfile.ZipFile, name_pattern: str) -> Optional[pd.DataFrame]:
    """Find a CSV matching the pattern (case-insensitive) inside the ZIP and load it."""
    matches = [
        n for n in zf.namelist()
        if name_pattern.lower() in n.lower() and n.lower().endswith(".csv")
    ]
    if not matches:
        return None
    try:
        with zf.open(matches[0]) as f:
            return pd.read_csv(f, sep=";", encoding="latin-1", dtype=str, low_memory=False)
    except Exception as e:
        logger.warning(f"Failed to load CSV {matches[0]}: {e}")
        return None


def _filter_ultimo(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to ORDEM_EXERC == 'ÚLTIMO' (current period, not prior-year comparison)."""
    if "ORDEM_EXERC" not in df.columns:
        return df
    return df[df["ORDEM_EXERC"].str.strip().str.upper() == "ÚLTIMO"].copy()


def _pivot_accounts(df: pd.DataFrame, account_map: dict) -> pd.DataFrame:
    """Filter to relevant account codes and pivot CD_CONTA -> VL_CONTA."""
    if df is None or df.empty:
        return pd.DataFrame()
    codes = list(account_map.keys())
    df = df[df["CD_CONTA"].isin(codes)].copy()
    if df.empty:
        return pd.DataFrame()
    df["VL_CONTA"] = pd.to_numeric(df["VL_CONTA"], errors="coerce")
    df["metric"] = df["CD_CONTA"].map(account_map)
    return df


def _build_group_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add a cleaned CNPJ column and parsed date columns."""
    df = df.copy()
    df["cnpj_clean"] = df["CNPJ_CIA"].apply(lambda x: _clean_cnpj(str(x)))
    df["period_end_parsed"] = df["DT_REFER"].apply(_parse_date)
    # DT_RECEB (filing receipt date) may not be present in all CSV files —
    # fall back to DT_REFER (period-end date) which is always available.
    receb_col = "DT_RECEB" if "DT_RECEB" in df.columns else "DT_REFER"
    df["filing_date_parsed"] = df[receb_col].apply(_parse_date)
    df["version"] = pd.to_numeric(df["VERSAO"], errors="coerce").fillna(1).astype(int)
    return df


def _extract_metrics_from_csvs(
    zip_path, doc_type: str, cnpj_ticker_map: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Core extraction logic shared by DFP and ITR parsers.

    Opens the ZIP, loads DRE / BPA / BPP CSVs, filters to current-period rows,
    pivots account codes to metric columns, merges them, and builds the two
    output DataFrames.

    Returns (filings_df, fundamentals_df).
    """
    # Support both Path and BytesIO inputs (for testing with in-memory ZIPs)
    if isinstance(zip_path, (str, Path)):
        source_file = Path(zip_path).name
        zf_ctx = zipfile.ZipFile(zip_path, "r")
    else:
        source_file = "in-memory"
        zip_path.seek(0)
        zf_ctx = zipfile.ZipFile(zip_path, "r")

    with zf_ctx as zf:
        dre = _load_csv_from_zip(zf, "DRE_con")
        bpa = _load_csv_from_zip(zf, "BPA_con")
        bpp = _load_csv_from_zip(zf, "BPP_con")
        # Main metadata CSV (e.g. dfp_cia_aberta_2023.csv / itr_cia_aberta_2023.csv)
        # is the only file that contains DT_RECEB (filing receipt date).
        # It has fewer underscores than sub-tables and matches the zip's own base name.
        all_names = zf.namelist()
        meta_name = next(
            (n for n in all_names
             if n.lower().endswith(".csv") and n.count("_") <= 4),
            None,
        )
        meta_receb = None
        if meta_name:
            try:
                with zf.open(meta_name) as f:
                    _m = pd.read_csv(f, sep=";", encoding="latin-1", dtype=str, low_memory=False)
                if {"CNPJ_CIA", "DT_REFER", "VERSAO", "DT_RECEB"}.issubset(_m.columns):
                    meta_receb = _m[["CNPJ_CIA", "DT_REFER", "VERSAO", "DT_RECEB"]].drop_duplicates()
            except Exception:
                pass

    # Filter + pivot each statement type
    frames = {}

    def _inject_receb(df: pd.DataFrame) -> pd.DataFrame:
        """Merge DT_RECEB from the metadata CSV into a statement DataFrame."""
        if meta_receb is not None and "DT_RECEB" not in df.columns:
            df = df.merge(meta_receb, on=["CNPJ_CIA", "DT_REFER", "VERSAO"], how="left")
        return df

    if dre is not None and not dre.empty:
        dre = _inject_receb(_filter_ultimo(dre))
        dre = _build_group_key(dre)
        dre_pivot = _pivot_accounts(dre, INCOME_ACCOUNT_MAP)
        if not dre_pivot.empty:
            frames["dre"] = (dre, dre_pivot)

    if bpa is not None and not bpa.empty:
        bpa = _inject_receb(_filter_ultimo(bpa))
        bpa = _build_group_key(bpa)
        bpa_pivot = _pivot_accounts(bpa, ASSET_ACCOUNT_MAP)
        if not bpa_pivot.empty:
            frames["bpa"] = (bpa, bpa_pivot)

    if bpp is not None and not bpp.empty:
        bpp = _inject_receb(_filter_ultimo(bpp))
        bpp = _build_group_key(bpp)
        bpp_pivot = _pivot_accounts(bpp, LIABILITY_ACCOUNT_MAP)
        if not bpp_pivot.empty:
            frames["bpp"] = (bpp, bpp_pivot)

    if not frames:
        logger.warning(f"No usable CSV data found in {source_file}")
        return pd.DataFrame(), pd.DataFrame()

    # Collect all group keys from any available frame
    group_keys = ["cnpj_clean", "period_end_parsed", "filing_date_parsed", "version"]

    def _wide_metrics(raw_df, pivot_df, account_map):
        """Return a wide DataFrame with one row per group key and metric columns."""
        merged = raw_df[group_keys].drop_duplicates().merge(
            pivot_df[group_keys + ["metric", "VL_CONTA"]].drop_duplicates(),
            on=group_keys, how="inner"
        )
        wide = merged.pivot_table(
            index=group_keys,
            columns="metric",
            values="VL_CONTA",
            aggfunc="last",
        ).reset_index()
        wide.columns.name = None
        return wide

    # Build wide tables per statement
    wide_parts = []
    for key, (raw_df, pivot_df) in frames.items():
        if key == "dre":
            account_map = INCOME_ACCOUNT_MAP
        elif key == "bpa":
            account_map = ASSET_ACCOUNT_MAP
        else:
            account_map = LIABILITY_ACCOUNT_MAP
        wide_parts.append(_wide_metrics(raw_df, pivot_df, account_map))

    # Start from the first wide part and merge the rest
    merged = wide_parts[0]
    for part in wide_parts[1:]:
        merged = merged.merge(part, on=group_keys, how="outer")

    # Compute derived metrics
    if "_cash" in merged.columns and "_short_debt" in merged.columns and "_long_debt" in merged.columns:
        merged["net_debt"] = (
            merged.get("_short_debt", 0).fillna(0) +
            merged.get("_long_debt", 0).fillna(0) -
            merged.get("_cash", 0).fillna(0)
        )
    elif "_short_debt" in merged.columns or "_long_debt" in merged.columns:
        merged["net_debt"] = (
            merged.get("_short_debt", pd.Series(0)).fillna(0) +
            merged.get("_long_debt", pd.Series(0)).fillna(0) -
            merged.get("_cash", pd.Series(0)).fillna(0)
        )
    else:
        merged["net_debt"] = None

    # Drop intermediate helper columns
    for col in ["_cash", "_short_debt", "_long_debt"]:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)

    # Build filing_id
    merged["filing_id"] = merged.apply(
        lambda r: f"{r['cnpj_clean']}_{doc_type}_{r['period_end_parsed']}_{r['version']}",
        axis=1,
    )

    # Resolve ticker from map
    merged["ticker"] = merged["cnpj_clean"].map(cnpj_ticker_map)

    # Infer fiscal_year and quarter
    merged["fiscal_year"] = merged["period_end_parsed"].apply(
        lambda d: int(d[:4]) if d else None
    )
    if doc_type == "ITR":
        merged["quarter"] = merged["period_end_parsed"].apply(_infer_quarter)
    else:
        merged["quarter"] = None

    # ── Build filings_df ──────────────────────────────────────────────────────
    filings_rows = []
    for _, row in merged.iterrows():
        filings_rows.append({
            "filing_id": row["filing_id"],
            "cnpj": row["cnpj_clean"],
            "doc_type": doc_type,
            "period_end": row["period_end_parsed"],
            "filing_date": row["filing_date_parsed"],
            "filing_version": int(row["version"]),
            "fiscal_year": row.get("fiscal_year"),
            "quarter": row.get("quarter"),
            "source_file": source_file,
        })
    filings_df = pd.DataFrame(filings_rows)

    # ── Build fundamentals_df ─────────────────────────────────────────────────
    metric_cols = [
        c for c in ["revenue", "net_income", "ebitda", "total_assets",
                    "equity", "net_debt", "shares_outstanding"]
        if c in merged.columns
    ]
    fund_cols = [
        "filing_id", "cnpj_clean", "ticker", "period_end_parsed",
        "filing_date_parsed", "version", "fiscal_year", "quarter",
    ] + metric_cols
    fundamentals_df = merged[[c for c in fund_cols if c in merged.columns]].copy()
    fundamentals_df.rename(columns={
        "cnpj_clean": "cnpj",
        "period_end_parsed": "period_end",
        "filing_date_parsed": "filing_date",
        "version": "filing_version",
    }, inplace=True)
    fundamentals_df["doc_type"] = doc_type
    if "shares_outstanding" not in fundamentals_df.columns:
        fundamentals_df["shares_outstanding"] = None

    return filings_df, fundamentals_df


# ── Public API ────────────────────────────────────────────────────────────────

def parse_dfp_zip(
    zip_path, cnpj_ticker_map: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a DFP (annual) ZIP file. Returns (filings_df, fundamentals_df)."""
    logger.info(f"Parsing DFP ZIP: {zip_path}")
    return _extract_metrics_from_csvs(zip_path, "DFP", cnpj_ticker_map)


def parse_itr_zip(
    zip_path, cnpj_ticker_map: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse an ITR (quarterly) ZIP file. Returns (filings_df, fundamentals_df)."""
    logger.info(f"Parsing ITR ZIP: {zip_path}")
    return _extract_metrics_from_csvs(zip_path, "ITR", cnpj_ticker_map)


def parse_fre_zip(
    zip_path, cnpj_ticker_map: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a FRE ZIP file to extract shares outstanding from capital_social CSV.

    Returns (filings_df, fundamentals_df).
    The fundamentals_df only contains shares_outstanding — other metric columns are None.
    """
    logger.info(f"Parsing FRE ZIP: {zip_path}")
    doc_type = "FRE"

    if isinstance(zip_path, (str, Path)):
        source_file = Path(zip_path).name
        zf_ctx = zipfile.ZipFile(zip_path, "r")
    else:
        source_file = "in-memory"
        zip_path.seek(0)
        zf_ctx = zipfile.ZipFile(zip_path, "r")

    with zf_ctx as zf:
        # Main FRE CSV has the filing receipt date (DT_RECEB) — standard columns
        main = _load_csv_from_zip(zf, f"fre_cia_aberta_{source_file[-8:-4]}.csv" if source_file != "in-memory" else "fre_cia_aberta")
        # capital_social has shares — the plain capital_social_YYYY.csv appears first
        # alphabetically, so this returns the right file over sub-tables like capital_social_aumento
        cap = _load_csv_from_zip(zf, "capital_social")

    # Prefer the most specific capital_social file (classe_acao not needed; skip sub-tables)
    # The plain capital_social_YYYY.csv has Quantidade_Total_Acoes
    if cap is None or cap.empty:
        logger.warning(f"No capital_social CSV found in {source_file}")
        return pd.DataFrame(), pd.DataFrame()

    # FRE capital_social CSV uses different column names than DFP/ITR CSVs:
    # CNPJ_Companhia, Data_Referencia, Versao, Quantidade_Total_Acoes (etc.)
    # Normalise to the standard names expected by _build_group_key and downstream code.
    col_rename = {
        "CNPJ_Companhia": "CNPJ_CIA",
        "Data_Referencia": "DT_REFER",
        "Versao": "VERSAO",
    }
    cap = cap.rename(columns={k: v for k, v in col_rename.items() if k in cap.columns})

    # Inject DT_RECEB from the main FRE index CSV when available.
    # Only merge when cap doesn't already have DT_RECEB and main has all required columns.
    required_main_cols = {"CNPJ_CIA", "DT_REFER", "VERSAO", "DT_RECEB"}
    if (
        "DT_RECEB" not in cap.columns
        and main is not None
        and not main.empty
        and required_main_cols.issubset(main.columns)
    ):
        receipt_map = main[list(required_main_cols)].drop_duplicates()
        cap = cap.merge(receipt_map, on=["CNPJ_CIA", "DT_REFER", "VERSAO"], how="left")
    if "DT_RECEB" not in cap.columns:
        # Fall back to period-end date as filing date (conservative)
        cap["DT_RECEB"] = cap["DT_REFER"]

    cap = _build_group_key(cap)

    required_cols = ["cnpj_clean", "period_end_parsed", "filing_date_parsed", "version"]
    for col in required_cols:
        if col not in cap.columns:
            logger.warning(f"Missing column {col} in FRE capital_social CSV")
            return pd.DataFrame(), pd.DataFrame()

    # Determine total shares. CVM uses different column names across file versions:
    # - New format: Quantidade_Total_Acoes (may be 0; fall back to Ord + Pref sum)
    # - Legacy format: QTDE_TOTAL_ACOES
    # Any value <= 0 is replaced with NaN so that downstream propagation and ratio
    # computation treat it as missing data rather than dividing by zero.
    if "Quantidade_Total_Acoes" in cap.columns:
        total = pd.to_numeric(cap["Quantidade_Total_Acoes"], errors="coerce").fillna(0)
        ord_ = pd.to_numeric(cap.get("Quantidade_Acoes_Ordinarias", pd.Series(0, index=cap.index)), errors="coerce").fillna(0)
        pref = pd.to_numeric(cap.get("Quantidade_Acoes_Preferenciais", pd.Series(0, index=cap.index)), errors="coerce").fillna(0)
        shares = total.where(total > 0, ord_ + pref)
        cap["shares_outstanding"] = shares.where(shares > 0, other=float("nan"))
    elif "QTDE_TOTAL_ACOES" in cap.columns:
        shares_raw = pd.to_numeric(cap["QTDE_TOTAL_ACOES"], errors="coerce")
        cap["shares_outstanding"] = shares_raw.where(shares_raw > 0, other=float("nan"))
    else:
        logger.warning("No shares outstanding column found in FRE capital_social CSV")
        return pd.DataFrame(), pd.DataFrame()
    cap = cap.dropna(subset=["cnpj_clean", "period_end_parsed"])

    # Keep latest version per (cnpj, period_end)
    cap = (
        cap.sort_values(["cnpj_clean", "period_end_parsed", "version"])
        .drop_duplicates(subset=["cnpj_clean", "period_end_parsed"], keep="last")
    )

    cap["filing_id"] = cap.apply(
        lambda r: f"{r['cnpj_clean']}_{doc_type}_{r['period_end_parsed']}_{r['version']}",
        axis=1,
    )
    cap["ticker"] = cap["cnpj_clean"].map(cnpj_ticker_map)
    cap["fiscal_year"] = cap["period_end_parsed"].apply(
        lambda d: int(d[:4]) if d else None
    )
    cap["quarter"] = None

    filings_rows = []
    for _, row in cap.iterrows():
        filings_rows.append({
            "filing_id": row["filing_id"],
            "cnpj": row["cnpj_clean"],
            "doc_type": doc_type,
            "period_end": row["period_end_parsed"],
            "filing_date": row["filing_date_parsed"],
            "filing_version": int(row["version"]),
            "fiscal_year": row.get("fiscal_year"),
            "quarter": None,
            "source_file": source_file,
        })
    filings_df = pd.DataFrame(filings_rows)

    fundamentals_df = cap[[
        "filing_id", "cnpj_clean", "ticker",
        "period_end_parsed", "filing_date_parsed", "version",
        "fiscal_year", "quarter", "shares_outstanding"
    ]].copy()
    fundamentals_df.rename(columns={
        "cnpj_clean": "cnpj",
        "period_end_parsed": "period_end",
        "filing_date_parsed": "filing_date",
        "version": "filing_version",
    }, inplace=True)
    fundamentals_df["doc_type"] = doc_type
    for col in ["revenue", "net_income", "ebitda", "total_assets", "equity", "net_debt"]:
        fundamentals_df[col] = None

    return filings_df, fundamentals_df
