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
# NOTE: net_income and ebitda are intentionally NOT here — the EBIT line
# ('Resultado Antes do Resultado Financeiro e dos Tributos') sits at 3.05 for
# corporates but 3.07 for insurer-chart filers (BBSE), and bank-chart filers
# (ITUB, BBAS) have no EBIT line at all — their 3.05 is pre-tax income.
# Both are selected by description in _desc_account_rows, like equity.
# 3.01 is fine as a code: it's Receita Líquida for corporates and
# Receitas da Intermediação Financeira for banks — the standard bank revenue.
INCOME_ACCOUNT_MAP = {
    "3.01": "revenue",       # Receita Líquida
}

# Balance sheet assets (BPA files — *BPA*con*)
ASSET_ACCOUNT_MAP = {
    "1":       "total_assets",  # Total do Ativo
    "1.01.01": "_cash",          # Caixa e Equivalentes (intermediate for net_debt)
}

# Balance sheet liabilities (BPP files — *BPP*con*)
# NOTE: equity is intentionally NOT here — corporates report PL at 2.03 but
# financial institutions use a different chart where 2.03 is 'Provisões' and
# PL sits at 2.07/2.08. Equity is selected by description in _equity_rows.
LIABILITY_ACCOUNT_MAP = {
    "2.01.04": "_short_debt",  # Empréstimos e Financiamentos (CP)
    "2.02.01": "_long_debt",   # Empréstimos e Financiamentos (LP)
}

ALL_ACCOUNT_MAP = {**INCOME_ACCOUNT_MAP, **ASSET_ACCOUNT_MAP, **LIABILITY_ACCOUNT_MAP}

# Legal filing deadlines (days after period end), used as a conservative
# filing_date fallback when DT_RECEB is unavailable. Using period_end directly
# would leak results into backtests months before they were public.
FILING_DEADLINE_DAYS = {"ITR": 45, "DFP": 90, "FRE": 90}


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


def _add_days(date_str: Optional[str], days: int) -> Optional[str]:
    """Add N days to a YYYY-MM-DD string. Returns YYYY-MM-DD or None."""
    if not date_str:
        return None
    from datetime import datetime, timedelta
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


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


def _load_statement_with_ind_fallback(
    zf: zipfile.ZipFile, base: str
) -> Optional[pd.DataFrame]:
    """Load a statement preferring consolidated, falling back to individual.

    ~77 listed companies (ASAI3, ABCB4, BRAP4, CGAS5, BAZA3, ...) file only
    individual (_ind) statements — holdings and companies without subsidiaries
    never appear in the _con CSVs, so con-only parsing silently drops them.
    Per company: use _con when present, else its _ind rows.
    """
    con = _load_csv_from_zip(zf, f"{base}_con")
    ind = _load_csv_from_zip(zf, f"{base}_ind")
    if ind is None or ind.empty:
        return con
    if con is None or con.empty:
        return ind
    ind_only = ind[~ind["CNPJ_CIA"].isin(set(con["CNPJ_CIA"]))]
    return pd.concat([con, ind_only], ignore_index=True)


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


def _desc_account_rows(
    df: pd.DataFrame, code_re: str, ds_prefix: str, metric: str,
    ds_contains: Optional[str] = None,
) -> pd.DataFrame:
    """Select top-level account rows by description instead of a fixed code.

    Corporates and financial institutions use different charts of accounts:
    PL is 2.03 for corporates but 2.07/2.08 for banks (whose 2.03 is
    'Provisões'); net income is 3.11 for corporates but 3.09 for bank-chart
    filers like ITUB. A fixed code silently stores the wrong line for banks.
    """
    if "DS_CONTA" not in df.columns:
        return pd.DataFrame()
    m = (
        df["CD_CONTA"].str.fullmatch(code_re)
        & df["DS_CONTA"].str.strip().str.startswith(ds_prefix)
    )
    if ds_contains:
        m &= df["DS_CONTA"].str.contains(ds_contains)
    out = df[m].copy()
    if out.empty:
        return pd.DataFrame()
    out["VL_CONTA"] = pd.to_numeric(out["VL_CONTA"], errors="coerce")
    out["metric"] = metric
    return out


def _build_group_key(df: pd.DataFrame, fallback_days: int = 90) -> pd.DataFrame:
    """Add a cleaned CNPJ column and parsed date columns.

    filing_date is DT_RECEB (receipt date) when present. When DT_RECEB is
    missing, fall back to period_end + fallback_days (the legal filing
    deadline) — NOT period_end itself, which would create look-ahead bias.
    """
    df = df.copy()
    df["cnpj_clean"] = df["CNPJ_CIA"].apply(lambda x: _clean_cnpj(str(x)))
    df["period_end_parsed"] = df["DT_REFER"].apply(_parse_date)
    if "DT_RECEB" in df.columns:
        df["filing_date_parsed"] = df["DT_RECEB"].apply(_parse_date)
    else:
        df["filing_date_parsed"] = None
    missing = df["filing_date_parsed"].isna()
    if missing.any():
        df.loc[missing, "filing_date_parsed"] = df.loc[missing, "period_end_parsed"].apply(
            lambda d: _add_days(d, fallback_days)
        )
    df["version"] = pd.to_numeric(df["VERSAO"], errors="coerce").fillna(1).astype(int)
    return df


def _select_ytd_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only year-to-date DRE rows.

    ITR DRE files contain BOTH quarter-only rows (DT_INI_EXERC = quarter start)
    and year-to-date rows (DT_INI_EXERC = fiscal-year start) for the same
    DT_REFER. Keep the rows whose DT_INI_EXERC is the earliest per
    (company, period_end, version) — the YTD rows. For DFP (annual) files this
    is a no-op since the only rows are already full-year.
    """
    if "DT_INI_EXERC" not in df.columns:
        return df
    df = df.copy()
    df["_ini"] = df["DT_INI_EXERC"].apply(_parse_date)
    min_ini = df.groupby(["CNPJ_CIA", "DT_REFER", "VERSAO"])["_ini"].transform("min")
    # Keep YTD rows; rows without a parseable DT_INI_EXERC are kept as-is.
    df = df[(df["_ini"] == min_ini) | df["_ini"].isna()]
    return df.drop(columns=["_ini"])


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
        dre = _load_statement_with_ind_fallback(zf, "DRE")
        bpa = _load_statement_with_ind_fallback(zf, "BPA")
        bpp = _load_statement_with_ind_fallback(zf, "BPP")
        # Main metadata CSV (e.g. dfp_cia_aberta_2023.csv / itr_cia_aberta_2023.csv)
        # is the only file that contains DT_RECEB (filing receipt date).
        # Match it exactly -- sub-tables like dfp_cia_aberta_parecer_2023.csv
        # can otherwise win the old underscore-count heuristic (kept as fallback).
        all_names = zf.namelist()
        meta_re = re.compile(rf"(?:^|/){doc_type.lower()}_cia_aberta_\d{{4}}\.csv$")
        meta_name = next((n for n in all_names if meta_re.search(n.lower())), None)
        if meta_name is None:
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
    fallback_days = FILING_DEADLINE_DAYS.get(doc_type, 90)

    def _inject_receb(df: pd.DataFrame) -> pd.DataFrame:
        """Merge DT_RECEB from the metadata CSV into a statement DataFrame."""
        if meta_receb is not None and "DT_RECEB" not in df.columns:
            df = df.merge(meta_receb, on=["CNPJ_CIA", "DT_REFER", "VERSAO"], how="left")
        return df

    if dre is not None and not dre.empty:
        dre = _inject_receb(_filter_ultimo(dre))
        # Income-statement metrics are stored as YTD values (annual for DFP).
        dre = _select_ytd_rows(dre)
        dre = _build_group_key(dre, fallback_days)
        _dre_parts = [
            p for p in (
                _pivot_accounts(dre, INCOME_ACCOUNT_MAP),
                # 'Lucro/Prejuízo ... do Período' (corporate) and 'Lucro ou
                # Prejuízo Líquido ... do Período' (bank chart). 'Lucro por
                # Ação' and 'Lucro ou Prejuízo antes das Participações' have
                # no 'Período' and are excluded.
                _desc_account_rows(dre, r"3\.\d{2}", "Lucro",
                                   "net_income", ds_contains="Período"),
                # EBIT (used as EBITDA proxy — no D&A subtracted). Description
                # is stable 2010-2026 but the code drifts (3.05 corporate,
                # 3.07 insurer chart). Bank-chart filers have no such line →
                # NULL, which is right: their 3.05 is pre-tax income, and
                # storing that as ebitda poisons EV/EBITDA screens.
                _desc_account_rows(dre, r"3\.\d{2}",
                                   "Resultado Antes do Resultado Financeiro",
                                   "ebitda"),
            )
            if not p.empty
        ]
        dre_pivot = pd.concat(_dre_parts, ignore_index=True) if _dre_parts else pd.DataFrame()
        if not dre_pivot.empty:
            frames["dre"] = (dre, dre_pivot)

    if bpa is not None and not bpa.empty:
        bpa = _inject_receb(_filter_ultimo(bpa))
        bpa = _build_group_key(bpa, fallback_days)
        bpa_pivot = _pivot_accounts(bpa, ASSET_ACCOUNT_MAP)
        if not bpa_pivot.empty:
            frames["bpa"] = (bpa, bpa_pivot)

    if bpp is not None and not bpp.empty:
        bpp = _inject_receb(_filter_ultimo(bpp))
        bpp = _build_group_key(bpp, fallback_days)
        _bpp_parts = [
            p for p in (
                _pivot_accounts(bpp, LIABILITY_ACCOUNT_MAP),
                _desc_account_rows(bpp, r"2\.\d{2}", "Patrimônio Líquido", "equity"),
            )
            if not p.empty
        ]
        bpp_pivot = pd.concat(_bpp_parts, ignore_index=True) if _bpp_parts else pd.DataFrame()
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

    # Compute derived metrics. Rows where none of the components exist stay
    # NULL — bank-chart filers report no 1.01.01/2.01.04/2.02.01, and a fake
    # net_debt of 0 for a bank is a wrong value, not a neutral one.
    parts = merged.reindex(columns=["_short_debt", "_long_debt", "_cash"])
    net = parts["_short_debt"].fillna(0) + parts["_long_debt"].fillna(0) - parts["_cash"].fillna(0)
    merged["net_debt"] = net.where(parts.notna().any(axis=1))

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

    # When DT_RECEB is unavailable, _build_group_key falls back to
    # period_end + legal deadline (90 days for FRE) to avoid look-ahead bias.
    cap = _build_group_key(cap, FILING_DEADLINE_DAYS["FRE"])

    # A filing carries one capital_social row per Tipo_Capital (Emitido /
    # Subscrito / Integralizado / Autorizado). "Capital Autorizado" is an
    # issuance CEILING, not shares that exist (COCE: 300bn authorized vs 78M
    # real; ITUB: 2.15bn authorized vs 11bn real) — the old arbitrary dedup
    # sometimes stored it. Prefer Integralizado (paid-in = actually issued).
    if "Tipo_Capital" in cap.columns:
        rank = {"Capital Emitido": 0, "Capital Subscrito": 1, "Capital Integralizado": 2}
        typed = cap[cap["Tipo_Capital"].isin(rank)]
        if not typed.empty:
            cap = typed.copy()
            cap["_tipo_rank"] = cap["Tipo_Capital"].map(rank)
        else:
            cap = cap.copy()
            cap["_tipo_rank"] = 0
    else:
        cap = cap.copy()
        cap["_tipo_rank"] = 0
    # Among duplicate rows of the same type (capital-history entries — TELB
    # filed the same capital twice, one entry fat-fingered x10000), the row
    # with the LATEST authorization date is the current capital.
    cap["_aut_date"] = pd.to_datetime(
        cap.get("Data_Autorizacao_Aprovacao"), errors="coerce"
    )

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
        # Per-class counts (ON / PN incl. all preferred classes) so market cap
        # can price each class at its own quote instead of total x one close.
        valid = shares > 0
        cap["shares_on"] = ord_.where(valid, other=float("nan"))
        cap["shares_pn"] = pref.where(valid, other=float("nan"))
    elif "QTDE_TOTAL_ACOES" in cap.columns:
        shares_raw = pd.to_numeric(cap["QTDE_TOTAL_ACOES"], errors="coerce")
        cap["shares_outstanding"] = shares_raw.where(shares_raw > 0, other=float("nan"))
        cap["shares_on"] = float("nan")
        cap["shares_pn"] = float("nan")
    else:
        logger.warning("No shares outstanding column found in FRE capital_social CSV")
        return pd.DataFrame(), pd.DataFrame()
    cap = cap.dropna(subset=["cnpj_clean", "period_end_parsed"])

    # Keep one row per (cnpj, period_end): latest version, then most reliable
    # Tipo_Capital, then latest authorization date (NaT loses).
    cap = (
        cap.sort_values(
            ["cnpj_clean", "period_end_parsed", "version", "_tipo_rank", "_aut_date"],
            na_position="first",
        )
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
        "fiscal_year", "quarter", "shares_outstanding", "shares_on", "shares_pn"
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
