"""
Parse COTAHIST fixed-width files from B3.
"""

import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


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


def parse_cotahist_file(zip_path: Path) -> pd.DataFrame:
    """
    Parse a single COTAHIST ZIP file.

    The ZIP contains a single .TXT file with fixed-width records:
    - Header: tipo_registro = "00"
    - Data: tipo_registro = "01"
    - Trailer: tipo_registro = "99"

    Args:
        zip_path: Path to COTAHIST ZIP file

    Returns:
        DataFrame with columns: [date, ticker, open, high, low, close, volume]
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        logger.error(f"File not found: {zip_path}")
        return pd.DataFrame()

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
                    volume = _parse_int(line[170:188])
                    isin_code = line[230:242].strip()

                    records.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "isin_code": isin_code,
                            "open": open_price,
                            "high": high_price,
                            "low": low_price,
                            "close": close_price,
                            "volume": volume,
                        }
                    )

    except zipfile.BadZipFile as e:
        logger.error(f"Bad ZIP file {zip_path}: {e}")
        return pd.DataFrame()

    if records:
        df = pd.DataFrame(records)
        logger.debug(f"Parsed {len(df)} records from {zip_path.name}")
        return df

    return pd.DataFrame()


def parse_all_cotahist_files(directory: Path) -> pd.DataFrame:
    """
    Parse all COTAHIST ZIP files in a directory.

    Args:
        directory: Path to directory containing COTAHIST_*.ZIP files

    Returns:
        Combined DataFrame with all price records
    """
    zip_files = sorted(directory.glob("COTAHIST_*.ZIP"))

    if not zip_files:
        logger.warning(f"No COTAHIST ZIP files found in {directory}")
        return pd.DataFrame()

    logger.info(f"Found {len(zip_files)} COTAHIST files to parse")

    all_dfs = []
    for i, zip_path in enumerate(zip_files, 1):
        logger.info(f"Parsing {zip_path.name} ({i}/{len(zip_files)})")
        df = parse_cotahist_file(zip_path)
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
