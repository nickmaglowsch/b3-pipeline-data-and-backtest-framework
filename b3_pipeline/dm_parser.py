"""
Parser for CVM's "Download Múltiplo" legacy CVMWIN-format files (ITR/DFP/IAN,
fiscal years ~2005-2009, pre-dating the dados.cvm.gov.br bulk CSV portal).

── Format ───────────────────────────────────────────────────────────────────
Per the technical manual (https://conteudo.cvm.gov.br/menu/regulados/companhias/
download_multiplo/manual_tecnico.html) and the published layout spec
(http://sistemas.cvm.gov.br/port/ciasabertas/Layout_Compactado.zip), each
CVMWIN filing is delivered as a ZIP of several files, one per "quadro" (form
section), keyed by CODCVM + DATADFP (reference date):

  CVM.CTR      control/index file: CODCVM, DATADFP (period end), TIPO_DOC
               (1/2=DFP moeda-constante/legislação-societária,
               3/4=ITR idem, 5=IAN), CGC (CNPJ), RAZAO_SOC, DATA_EMISS
               (protocol/emission date -- our point-in-time filing_date).
  CONFIG.001   ESCALA (currency scale: 01=unidade, 02=mil) and ESCALA_QTD
               (share-count scale, same codes) per CODCVM+DATADFP.
  DFPHDR.001 / ITRHDR.001 / IANHDR.001   header/company data.
  DFP*BPAE / DFP*BPPE / DFP*DERE (and ITR equivalents)   balance sheet
               (assets/liabilities+equity) and income statement, long format:
               CODCVM, DATADFP, CODCONTA, DESCONTA (account code/description),
               VALOR1 (2 years back), VALOR2 (1 year back), VALOR3 (current
               "último exercício" -- the value we want, analogous to the
               modern format's ORDEM_EXERC == 'ÚLTIMO'). Consolidated variants
               are C-prefixed (DFPCBPAE, DFPCBPPE, DFPCDERE, ITRC...) and are
               preferred, mirroring cvm_parser.py's use of *_con files;
               non-consolidated ("TIPO 1" -- comercial/industrial) files are
               the fallback. Bank/insurer layouts (TIPO 2/3) are out of scope,
               same scope limitation as the modern cvm_parser.py.
  IANCAPSO.001 "Composição do Capital Social": one row per share class
               (ON/PN/...) with QTDEACOES (quantity, scaled by ESCALA_QTD).
               shares_outstanding = sum of QTDEACOES across classes for the
               same CODCVM + DATAIAN, mirroring cvm_parser.parse_fre_zip's
               ordinary+preferred summation.

The three format variants documented in the layout zip (TXT/DBF/XML) are not
fully disambiguated by the manual for what Download Múltiplo actually ships;
this parser targets the **DBF** variant (dBase III), because DBF is
self-describing (field name/type/length live in the file header) so no
external byte-offset spec is required to parse it correctly -- unlike the
fixed-width TXT variant, whose exact column widths could not be recovered
from the (garbled) layout documentation. DBF field names match 1:1 with the
names documented in the TXT layout spec (both are ≤10 chars, classic DBF
naming). See docs/download_multiplo.md for the full research writeup and
this format assumption's limitations.

── Scale handling ───────────────────────────────────────────────────────────
This pipeline's existing convention (cvm_storage.py) stores all financial
values in THOUSANDS of BRL. CONFIG.ESCALA tells us the scale of the raw
VALOR* figures: '01' (unidade) means raw values are in whole BRL and must be
divided by 1000; '02' (mil) means they are already in thousands (no
conversion). Missing CONFIG rows default to '02' (thousands), the dominant
convention for listed-company filings, with a logged warning. ESCALA_QTD
scales IANCAPSO.QTDEACOES the same way, but toward an absolute share count
(mil -> multiply by 1000).

── filing_version ───────────────────────────────────────────────────────────
CVMWIN's CONFIG.STATUS ("apresentação"/"reapresentação") is categorical, not
a monotonic version counter like the modern VERSAO field. We derive
filing_version from filing_date itself (YYYYMMDD as an integer). A
rank-within-ZIP would restart at 1 in every ZIP -- Download Múltiplo delivers
one ZIP per (company, submission day), so an original filing and a later
restatement in different ZIPs would both get version 1, collide on the same
filing_id, and INSERT OR REPLACE would overwrite the earlier point-in-time
row. The date-derived version is unique per submission day, monotonic
("later version -> new row" semantics preserved), and stable across ZIPs.

filing_date = CVM.CTR.DATA_EMISS (the protocol/emission date) -- the
point-in-time date, never period_end.
"""
from __future__ import annotations

import logging
import re
import struct
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants (self-contained; do not depend on other modules' constants) ───

SCALE_UNIT = "01"       # unidade -- raw values are whole BRL / whole shares
SCALE_THOUSAND = "02"   # mil -- raw values already in thousands
DEFAULT_CURRENCY_SCALE = SCALE_THOUSAND  # dominant convention; logged when assumed

# CVM.CTR.TIPO_DOC codes. Only the "legislação societária" (nominal BRL,
# standard corporate-law) variants are kept -- NOT the inflation-adjusted
# "moeda de capacidade aquisitiva constante" variants, which would corrupt
# comparisons against the modern (post-2010) nominal-BRL dataset.
TIPO_DOC_DFP_LEGISLACAO = "2"
TIPO_DOC_ITR_LEGISLACAO = "4"
TIPO_DOC_IAN = "5"

DOC_TYPE_TIPO_DOC = {
    "DFP": TIPO_DOC_DFP_LEGISLACAO,
    "ITR": TIPO_DOC_ITR_LEGISLACAO,
    "IAN": TIPO_DOC_IAN,
}

# Statement file basenames (DBF stems, extension-agnostic), consolidated
# variant first (preferred), non-consolidated ("TIPO 1") fallback.
_STATEMENT_FILES = {
    "DFP": {
        "assets": ["DFPCBPAE", "DFPBPAE"],
        "liab_equity": ["DFPCBPPE", "DFPBPPE"],
        "income": ["DFPCDERE", "DFPDERE"],
    },
    "ITR": {
        "assets": ["ITRCBPAE", "ITRBPAE"],
        "liab_equity": ["ITRCBPPE", "ITRBPPE"],
        "income": ["ITRCDERE", "ITRDERE"],
    },
}
_CTR_FILE = "CVM"          # CVM.CTR
_CONFIG_FILE = "CONFIG"    # CONFIG.001
_IAN_CAPSO_FILE = "IANCAPSO"

# Account description (DESCONTA) matching -- normalized (upper, accent-free,
# whitespace-collapsed) substring patterns. DESCONTA is used instead of
# CODCONTA because the numeric chart-of-accounts drifted across the layout
# versions (5-8, spanning the Lei 11.638/07 accounting reform); the account
# *labels* below are stable across that period.
_REVENUE_PATTERNS = ["RECEITA LIQUIDA"]
_NET_INCOME_PATTERNS = [
    "LUCRO LIQUIDO DO EXERCICIO",
    "PREJUIZO LIQUIDO DO EXERCICIO",
    "LUCRO/PREJUIZO DO EXERCICIO",
    "RESULTADO LIQUIDO DO EXERCICIO",
    "RESULTADO LIQUIDO DO PERIODO",
]
_TOTAL_ASSETS_PATTERNS = ["ATIVO TOTAL"]
_EQUITY_PATTERNS = ["PATRIMONIO LIQUIDO"]
_EQUITY_EXCLUDE_PATTERNS = ["MINORIT", "NAO CONTROLADOR"]
_NET_INCOME_EXCLUDE_PATTERNS = ["POR ACAO", "POR LOTE"]

# net_debt is intentionally NOT computed for Download Múltiplo filings: the
# CVMWIN chart-of-accounts does not reliably separate short-term vs.
# long-term "Empréstimos e Financiamentos" by DESCONTA alone across versions
# 5-8, and a wrong money-path value is worse than a NULL. Left None; the
# fundamentals_pit schema allows it.


# ── Minimal dBase III reader (self-describing -- no external spec needed) ──

def read_dbf(source) -> pd.DataFrame:
    """Read a dBase III/IV (.dbf) file into a DataFrame of raw string fields.

    Field names/types/lengths are declared in the file's own header, so this
    needs no external byte-offset documentation (unlike the fixed-width TXT
    layout variant). Memo ('M') fields are read as their raw 10-byte pointer
    text, not dereferenced -- no .dbt file support -- since none of the
    fields this pipeline needs (CODCVM, DATADFP, CODCONTA, DESCONTA, VALOR*,
    CGC, DATA_EMISS, QTDEACOES, ESCALA...) are memo fields per the layout spec.
    """
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            data = f.read()
    else:
        source.seek(0)
        data = source.read()

    if len(data) < 32:
        raise ValueError("File too small to be a valid DBF")

    n_records, header_size, record_size = struct.unpack_from("<IHH", data, 4)

    fields = []
    pos = 32
    while pos < header_size - 1 and pos + 32 <= len(data):
        if data[pos] == 0x0D:
            break
        raw_name = data[pos:pos + 11]
        name = raw_name.split(b"\x00")[0].decode("latin-1").strip()
        flen = data[pos + 16]
        fields.append((name, flen))
        pos += 32

    if not fields:
        raise ValueError("No field descriptors found -- not a valid DBF")

    rows = []
    offset = header_size
    for _ in range(n_records):
        rec = data[offset:offset + record_size]
        offset += record_size
        if len(rec) < record_size or rec[0:1] == b"*":
            continue  # short read or deleted record
        cursor = 1  # skip deletion-flag byte
        row = {}
        for name, flen in fields:
            raw = rec[cursor:cursor + flen]
            cursor += flen
            row[name] = raw.decode("latin-1", errors="replace").strip()
        rows.append(row)

    return pd.DataFrame(rows, columns=[f[0] for f in fields])


# ── Small self-contained helpers (duplicated rather than importing another
#    module's private helpers -- see FILE OWNERSHIP note in the task spec) ──

def _clean_cnpj(raw) -> Optional[str]:
    if not raw:
        return None
    digits = re.sub(r"\D", "", str(raw))
    return digits if len(digits) == 14 else None


def _parse_yyyymmdd(val) -> Optional[str]:
    """Parse an AAAAMMDD string (the layout's date format) to YYYY-MM-DD."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s in ("00000000", "nan", "NaT"):
        return None
    try:
        return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _infer_quarter(period_end: Optional[str]) -> Optional[int]:
    if not period_end:
        return None
    try:
        month = int(period_end[5:7])
    except (ValueError, IndexError):
        return None
    return {3: 1, 6: 2, 9: 3}.get(month)


def _norm_desc(s) -> str:
    """Normalize an account description for matching: upper, accent-free, collapsed whitespace."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.upper().split())


def _matches(desc_norm: str, patterns: list, exclude: Optional[list] = None) -> bool:
    if any(p in desc_norm for p in (exclude or [])):
        return False
    return any(p in desc_norm for p in patterns)


def _classify_account(desc_norm: str) -> Optional[str]:
    if _matches(desc_norm, _TOTAL_ASSETS_PATTERNS):
        return "total_assets"
    if _matches(desc_norm, _EQUITY_PATTERNS, _EQUITY_EXCLUDE_PATTERNS):
        return "equity"
    if _matches(desc_norm, _REVENUE_PATTERNS):
        return "revenue"
    if _matches(desc_norm, _NET_INCOME_PATTERNS, _NET_INCOME_EXCLUDE_PATTERNS):
        return "net_income"
    return None


# ── ZIP member lookup ────────────────────────────────────────────────────────

def _find_member(zf: zipfile.ZipFile, stem: str) -> Optional[str]:
    """Return the first ZIP member whose filename stem matches (case-insensitive)."""
    stem_upper = stem.upper()
    for name in zf.namelist():
        base = name.replace("\\", "/").rsplit("/", 1)[-1]
        file_stem = base.rsplit(".", 1)[0].upper()
        if file_stem == stem_upper:
            return name
    return None


def _load_dbf_member(zf: zipfile.ZipFile, stem: str) -> Optional[pd.DataFrame]:
    name = _find_member(zf, stem)
    if name is None:
        return None
    try:
        with zf.open(name) as f:
            import io
            buf = io.BytesIO(f.read())
        return read_dbf(buf)
    except Exception as e:
        logger.warning(f"Failed to read DBF member {name}: {e}")
        return None


def _load_first_available(zf: zipfile.ZipFile, stems: list) -> Optional[pd.DataFrame]:
    for stem in stems:
        df = _load_dbf_member(zf, stem)
        if df is not None and not df.empty:
            return df
    return None


# ── Core extraction ──────────────────────────────────────────────────────────

def _build_ctr_index(zf: zipfile.ZipFile, doc_type: str) -> Optional[pd.DataFrame]:
    """Load CVM.CTR, filter to the 'legislação societária' TIPO_DOC for doc_type,
    and return one row per (cod_cvm, period_end) with cnpj/company/filing_date.
    """
    ctr = _load_dbf_member(zf, _CTR_FILE)
    if ctr is None or ctr.empty:
        logger.warning("CVM.CTR not found in ZIP -- cannot determine filing metadata")
        return None
    if "TIPO_DOC" in ctr.columns:
        wanted = DOC_TYPE_TIPO_DOC[doc_type]
        ctr = ctr[ctr["TIPO_DOC"].astype(str).str.strip() == wanted].copy()
    if ctr.empty:
        return None

    ctr["cod_cvm"] = ctr.get("CODCVM", "").astype(str).str.strip()
    ctr["cnpj"] = ctr.get("CGC", "").apply(_clean_cnpj)
    ctr["period_end"] = ctr.get("DATADFP", "").apply(_parse_yyyymmdd)
    ctr["filing_date"] = ctr.get("DATA_EMISS", "").apply(_parse_yyyymmdd)
    # Fall back to period_end when DATA_EMISS is unparseable (never leave filing_date null).
    ctr["filing_date"] = ctr["filing_date"].fillna(ctr["period_end"])

    ctr = ctr.dropna(subset=["cnpj", "period_end"])
    return ctr[["cod_cvm", "cnpj", "period_end", "filing_date"]].drop_duplicates()


def _build_scale_map(zf: zipfile.ZipFile) -> pd.DataFrame:
    """Return a (cod_cvm, period_end) -> (escala, escala_qtd) lookup DataFrame."""
    config = _load_dbf_member(zf, _CONFIG_FILE)
    if config is None or config.empty:
        logger.warning(
            f"CONFIG not found in ZIP -- assuming default currency scale "
            f"'{DEFAULT_CURRENCY_SCALE}' (mil/thousands) for all rows"
        )
        return pd.DataFrame(columns=["cod_cvm", "period_end", "escala", "escala_qtd"])
    config = config.copy()
    config["cod_cvm"] = config.get("CODCVM", "").astype(str).str.strip()
    config["period_end"] = config.get("DATADFP", "").apply(_parse_yyyymmdd)
    config["escala"] = config.get("ESCALA", DEFAULT_CURRENCY_SCALE).astype(str).str.strip()
    config["escala_qtd"] = config.get("ESCALA_QTD", SCALE_UNIT).astype(str).str.strip()
    return config[["cod_cvm", "period_end", "escala", "escala_qtd"]].drop_duplicates(
        subset=["cod_cvm", "period_end"]
    )


def _scale_currency(raw_series: pd.Series, escala_series: pd.Series) -> pd.Series:
    """Normalize raw VALOR* figures to thousands of BRL given ESCALA codes."""
    values = pd.to_numeric(raw_series, errors="coerce")
    escala = escala_series.fillna(DEFAULT_CURRENCY_SCALE)
    divisor = escala.map({SCALE_UNIT: 1000.0, SCALE_THOUSAND: 1.0}).fillna(1.0)
    return values / divisor


def _extract_statement_metrics(
    zf: zipfile.ZipFile, doc_type: str, scale_map: pd.DataFrame
) -> pd.DataFrame:
    """Load assets/liab_equity/income statement DBFs and pivot to metric columns.

    Returns a DataFrame indexed by (cod_cvm, period_end) with revenue,
    net_income, total_assets, equity columns (any not found are absent).
    """
    files = _STATEMENT_FILES[doc_type]
    parts = []
    for kind, stems in files.items():
        df = _load_first_available(zf, stems)
        if df is None:
            continue
        if not {"CODCVM", "DATADFP", "CODCONTA", "DESCONTA", "VALOR3"}.issubset(df.columns):
            logger.warning(f"{doc_type} {kind} DBF missing expected columns -- skipping")
            continue
        df = df.copy()
        df["cod_cvm"] = df["CODCVM"].astype(str).str.strip()
        df["period_end"] = df["DATADFP"].apply(_parse_yyyymmdd)
        df["metric"] = df["DESCONTA"].apply(_norm_desc).apply(_classify_account)
        df = df.dropna(subset=["period_end", "metric"])
        if df.empty:
            continue
        df = df.merge(scale_map, on=["cod_cvm", "period_end"], how="left")
        df["value"] = _scale_currency(df["VALOR3"], df["escala"])
        parts.append(df[["cod_cvm", "period_end", "metric", "value"]])

    if not parts:
        return pd.DataFrame(columns=["cod_cvm", "period_end"])

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["cod_cvm", "period_end", "metric"], keep="last")
    wide = combined.pivot_table(
        index=["cod_cvm", "period_end"], columns="metric", values="value", aggfunc="last"
    ).reset_index()
    wide.columns.name = None
    return wide


def _assign_filing_version(ctr: pd.DataFrame) -> pd.DataFrame:
    """Derive filing_version from filing_date (YYYYMMDD as int).

    Must be stable ACROSS ZIPs: each ZIP holds one submission, so a
    rank-within-ZIP would give an original and a restatement (different ZIPs)
    the same version -> same filing_id -> the earlier PIT row gets overwritten.
    See the module docstring's "filing_version" section.
    """
    # ponytail: two same-day resubmissions collide (last parsed wins) -- switch
    # to a DB max-version lookup if that ever happens in real data.
    ctr = ctr.copy()
    ctr["filing_version"] = (
        ctr["filing_date"].astype(str).str.replace("-", "", regex=False).astype(int)
    )
    return ctr


def _build_filings_and_fundamentals(
    merged: pd.DataFrame, doc_type: str, cnpj_ticker_map: dict, source_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged = merged.copy()
    merged["fiscal_year"] = merged["period_end"].apply(lambda d: int(d[:4]) if d else None)
    merged["quarter"] = merged["period_end"].apply(_infer_quarter) if doc_type == "ITR" else None
    merged["filing_id"] = merged.apply(
        lambda r: f"{r['cnpj']}_{doc_type}_{r['period_end']}_{int(r['filing_version'])}", axis=1
    )
    merged["ticker"] = merged["cnpj"].map(cnpj_ticker_map)

    filings_rows = [
        {
            "filing_id": r["filing_id"],
            "cnpj": r["cnpj"],
            "doc_type": doc_type,
            "period_end": r["period_end"],
            "filing_date": r["filing_date"],
            "filing_version": int(r["filing_version"]),
            "fiscal_year": r.get("fiscal_year"),
            "quarter": r.get("quarter"),
            "source_file": source_file,
        }
        for _, r in merged.iterrows()
    ]
    filings_df = pd.DataFrame(filings_rows)

    metric_cols = [c for c in ["revenue", "net_income", "total_assets", "equity", "shares_outstanding"] if c in merged.columns]
    fund_cols = ["filing_id", "cnpj", "ticker", "period_end", "filing_date", "filing_version", "fiscal_year", "quarter"] + metric_cols
    fundamentals_df = merged[[c for c in fund_cols if c in merged.columns]].copy()
    fundamentals_df["doc_type"] = doc_type
    for col in ["revenue", "net_income", "ebitda", "total_assets", "equity", "net_debt", "shares_outstanding"]:
        if col not in fundamentals_df.columns:
            fundamentals_df[col] = None

    return filings_df, fundamentals_df


def _open_zip(zip_path):
    if isinstance(zip_path, (str, Path)):
        source_file = Path(zip_path).name
        return zipfile.ZipFile(zip_path, "r"), source_file
    zip_path.seek(0)
    return zipfile.ZipFile(zip_path, "r"), "in-memory"


# ── Public API ────────────────────────────────────────────────────────────────

def _parse_statement_zip(zip_path, doc_type: str, cnpj_ticker_map: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Shared DFP/ITR parsing path (balance sheet + income statement)."""
    zf, source_file = _open_zip(zip_path)
    with zf:
        ctr = _build_ctr_index(zf, doc_type)
        if ctr is None or ctr.empty:
            logger.warning(f"No {doc_type} filings found in {source_file} (missing/empty CVM.CTR)")
            return pd.DataFrame(), pd.DataFrame()
        scale_map = _build_scale_map(zf)
        metrics = _extract_statement_metrics(zf, doc_type, scale_map)

    if metrics.empty:
        logger.warning(f"No usable statement data found in {source_file}")
        return pd.DataFrame(), pd.DataFrame()

    merged = ctr.merge(metrics, on=["cod_cvm", "period_end"], how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = _assign_filing_version(merged)
    return _build_filings_and_fundamentals(merged, doc_type, cnpj_ticker_map, source_file)


def parse_dfp_dbf_zip(zip_path, cnpj_ticker_map: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a Download Múltiplo DFP (annual) filing ZIP. Returns (filings_df, fundamentals_df)."""
    logger.info(f"Parsing Download Múltiplo DFP ZIP: {zip_path}")
    return _parse_statement_zip(zip_path, "DFP", cnpj_ticker_map)


def parse_itr_dbf_zip(zip_path, cnpj_ticker_map: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a Download Múltiplo ITR (quarterly) filing ZIP. Returns (filings_df, fundamentals_df)."""
    logger.info(f"Parsing Download Múltiplo ITR ZIP: {zip_path}")
    return _parse_statement_zip(zip_path, "ITR", cnpj_ticker_map)


def parse_ian_dbf_zip(zip_path, cnpj_ticker_map: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a Download Múltiplo IAN filing ZIP to extract shares_outstanding.

    Returns (filings_df, fundamentals_df); fundamentals_df only has
    shares_outstanding populated -- other metric columns are None.
    """
    logger.info(f"Parsing Download Múltiplo IAN ZIP: {zip_path}")
    doc_type = "IAN"
    zf, source_file = _open_zip(zip_path)
    with zf:
        ctr = _build_ctr_index(zf, doc_type)
        if ctr is None or ctr.empty:
            logger.warning(f"No IAN filings found in {source_file} (missing/empty CVM.CTR)")
            return pd.DataFrame(), pd.DataFrame()
        scale_map = _build_scale_map(zf)
        capso = _load_dbf_member(zf, _IAN_CAPSO_FILE)

    if capso is None or capso.empty:
        logger.warning(f"No {_IAN_CAPSO_FILE} DBF found in {source_file}")
        return pd.DataFrame(), pd.DataFrame()
    if not {"CODCVM", "DATAIAN", "QTDEACOES"}.issubset(capso.columns):
        logger.warning(f"IANCAPSO DBF missing expected columns in {source_file}")
        return pd.DataFrame(), pd.DataFrame()

    capso = capso.copy()
    capso["cod_cvm"] = capso["CODCVM"].astype(str).str.strip()
    capso["period_end"] = capso["DATAIAN"].apply(_parse_yyyymmdd)
    capso = capso.dropna(subset=["period_end"])
    capso = capso.merge(scale_map, on=["cod_cvm", "period_end"], how="left")
    qty = pd.to_numeric(capso["QTDEACOES"], errors="coerce").fillna(0)
    qty_scale = capso["escala_qtd"].fillna(SCALE_UNIT).map({SCALE_UNIT: 1.0, SCALE_THOUSAND: 1000.0}).fillna(1.0)
    capso["shares"] = qty * qty_scale

    totals = capso.groupby(["cod_cvm", "period_end"], as_index=False)["shares"].sum()
    totals = totals.rename(columns={"shares": "shares_outstanding"})
    totals.loc[totals["shares_outstanding"] <= 0, "shares_outstanding"] = None

    merged = ctr.merge(totals, on=["cod_cvm", "period_end"], how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = _assign_filing_version(merged)
    return _build_filings_and_fundamentals(merged, doc_type, cnpj_ticker_map, source_file)
