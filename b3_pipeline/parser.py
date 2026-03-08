"""
Parse COTAHIST fixed-width files from B3.

Delegates to the compiled Rust extension `cotahist_rs` for performance when available.
Falls back to a pure-Python implementation if the extension is not compiled.
Run `make dev-rust` (or `cd b3_pipeline_rs && maturin develop`) to build the extension.
"""

import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rust fast-path helpers
# ---------------------------------------------------------------------------

def _get_cotahist_rs():
    try:
        import cotahist_rs
        return cotahist_rs
    except ModuleNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Pure-Python fallback helpers
# ---------------------------------------------------------------------------

def _parse_price(value: str) -> float:
    """Parse price string (integer with 2 implied decimals) to float."""
    try:
        return int(value.strip() or "0") / 100.0
    except ValueError:
        return 0.0


def _parse_date(value: str) -> Optional[datetime]:
    """Parse date string in YYYYMMDD format."""
    try:
        return datetime.strptime(value.strip(), "%Y%m%d")
    except ValueError:
        return None


def _parse_int(value: str) -> int:
    """Parse integer string."""
    try:
        return int(value.strip() or "0")
    except ValueError:
        return 0


def _parse_cotahist_file_python(zip_path: Path) -> pd.DataFrame:
    """Pure-Python implementation of COTAHIST ZIP parsing."""
    records = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            txt_files = [f for f in zf.namelist() if not f.endswith("/")]
            if not txt_files:
                logger.error(f"No files found in {zip_path}")
                return pd.DataFrame()

            with zf.open(txt_files[0]) as f:
                for line_bytes in f:
                    try:
                        line = line_bytes.decode("latin-1")
                    except UnicodeDecodeError:
                        line = line_bytes.decode("utf-8", errors="replace")

                    if len(line) < 245:
                        continue

                    tipo_registro = line[0:2]
                    if tipo_registro != "01":
                        continue

                    cod_bdi = line[10:12].strip()
                    if cod_bdi not in config.EQUITY_BDI_CODES:
                        continue

                    tipo_mercado = line[24:27].strip()
                    if tipo_mercado not in ("010", "011", "012", "013", "014", "015"):
                        continue

                    ticker = line[12:24].strip().upper()
                    if not ticker:
                        continue

                    date = _parse_date(line[2:10])
                    if date is None:
                        continue

                    open_price = _parse_price(line[56:69])
                    high_price = _parse_price(line[69:82])
                    low_price = _parse_price(line[82:95])
                    close_price = _parse_price(line[108:121])
                    volume = _parse_price(line[170:188])
                    isin_code = line[230:242].strip()

                    quotation_factor = _parse_int(line[210:217])
                    if quotation_factor <= 0:
                        quotation_factor = 1

                    if quotation_factor != 1:
                        open_price = open_price / quotation_factor
                        high_price = high_price / quotation_factor
                        low_price = low_price / quotation_factor
                        close_price = close_price / quotation_factor

                    records.append({
                        "date": date,
                        "ticker": ticker,
                        "isin_code": isin_code,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume,
                        "quotation_factor": quotation_factor,
                    })

    except zipfile.BadZipFile as e:
        logger.error(f"Bad ZIP file {zip_path}: {e}")
        return pd.DataFrame()

    if records:
        df = pd.DataFrame(records)
        logger.debug(f"Parsed {len(df)} records from {zip_path.name}")
        return df

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_cotahist_file(zip_path: Path) -> pd.DataFrame:
    """
    Parse a single COTAHIST ZIP file.

    Tries the compiled Rust extension first; falls back to pure Python if
    the extension is not available.

    Args:
        zip_path: Path to COTAHIST ZIP file

    Returns:
        DataFrame with columns: [date, ticker, isin_code, open, high, low, close, volume, quotation_factor]
        Prices (open/high/low/close) are normalized to per-share basis by dividing by quotation_factor.
        Returns an empty DataFrame if the file does not exist or cannot be parsed.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        logger.error(f"File not found: {zip_path}")
        return pd.DataFrame()

    cotahist_rs = _get_cotahist_rs()
    if cotahist_rs is not None:
        try:
            batch = cotahist_rs.parse_zip(str(zip_path))
            if batch.num_rows == 0:
                return pd.DataFrame()
            df = batch.to_pandas()
            df["date"] = pd.to_datetime(df["date"])
            logger.debug(f"Parsed {len(df)} records from {zip_path.name}")
            return df
        except Exception as e:
            logger.warning(f"Rust parser failed for {zip_path}, falling back to Python: {e}")

    return _parse_cotahist_file_python(zip_path)


def parse_all_cotahist_files(directory: Path) -> pd.DataFrame:
    """
    Parse all COTAHIST ZIP files in a directory.

    Tries the compiled Rust extension first (processes all files concurrently via rayon);
    falls back to pure-Python serial processing if the extension is not available.

    Args:
        directory: Path to directory containing COTAHIST_*.ZIP files

    Returns:
        Combined DataFrame with all price records, deduplicated and sorted.
    """
    zip_files = sorted(directory.glob("COTAHIST_*.ZIP"))

    if not zip_files:
        logger.warning(f"No COTAHIST ZIP files found in {directory}")
        return pd.DataFrame()

    logger.info(f"Found {len(zip_files)} COTAHIST files to parse")

    cotahist_rs = _get_cotahist_rs()
    if cotahist_rs is not None:
        try:
            batch = cotahist_rs.parse_multiple_zips([str(p) for p in zip_files])
            if batch.num_rows == 0:
                return pd.DataFrame()
            combined = batch.to_pandas()
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
            combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
            logger.info(f"Total parsed records: {len(combined):,}")
            logger.info(f"Unique tickers: {combined['ticker'].nunique():,}")
            logger.info(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
            return combined
        except Exception as e:
            logger.warning(f"Rust parser failed, falling back to Python: {e}")

    # Python fallback: serial per-file parsing
    all_dfs = []
    for i, zip_path in enumerate(zip_files, 1):
        logger.info(f"Parsing {zip_path.name} ({i}/{len(zip_files)})")
        df = _parse_cotahist_file_python(zip_path)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    logger.info(f"Total parsed records: {len(combined):,}")
    logger.info(f"Unique tickers: {combined['ticker'].nunique():,}")
    logger.info(f"Date range: {combined['date'].min()} to {combined['date'].max()}")

    return combined


def get_tickers_from_prices(prices_df: pd.DataFrame) -> List[str]:
    """
    Extract unique tickers from prices DataFrame.

    Args:
        prices_df: DataFrame with price data

    Returns:
        Sorted list of unique tickers
    """
    if prices_df.empty:
        return []
    return sorted(prices_df["ticker"].unique().tolist())
