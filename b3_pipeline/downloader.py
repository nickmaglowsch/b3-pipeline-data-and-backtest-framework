"""
Download COTAHIST files and corporate actions data.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

from . import config

logger = logging.getLogger(__name__)


def detect_available_years() -> List[int]:
    """
    Probe B3 server to detect which years have COTAHIST files available.
    Returns list of years from START_YEAR to current year that exist.
    """
    available = []
    current_year = config.CURRENT_YEAR

    for year in range(config.START_YEAR, current_year + 1):
        url = config.BASE_COTAHIST_URL.format(year=year)
        try:
            resp = requests.head(
                url, headers=config.B3_HEADERS, timeout=10, allow_redirects=True
            )
            if resp.status_code == 200:
                available.append(year)
        except requests.RequestException:
            pass

    return available


def download_annual_file(year: int, force: bool = False) -> Optional[Path]:
    """
    Download annual COTAHIST ZIP file for a given year.

    Args:
        year: Year to download
        force: If True, re-download even if file exists

    Returns:
        Path to downloaded file, or None if download failed
    """
    filename = f"COTAHIST_A{year}.ZIP"
    filepath = config.DATA_DIR / filename

    if filepath.exists() and not force:
        logger.debug(f"File already exists: {filepath}")
        return filepath

    url = config.BASE_COTAHIST_URL.format(year=year)
    logger.info(f"Downloading {url}...")

    try:
        resp = requests.get(url, headers=config.B3_HEADERS, timeout=300, stream=True)
        resp.raise_for_status()

        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        if downloaded % (10 * 1024 * 1024) < 8192:
                            logger.debug(
                                f"Downloaded {downloaded:,}/{total_size:,} bytes ({pct:.1f}%)"
                            )

        logger.info(f"Downloaded {filepath} ({downloaded:,} bytes)")
        return filepath

    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        if filepath.exists():
            filepath.unlink()
        return None


def download_current_year_daily(force: bool = False) -> List[Path]:
    """
    Download daily COTAHIST files for the current in-progress year.

    Returns:
        List of paths to downloaded files
    """
    downloaded_files = []
    today = datetime.now()
    start_date = datetime(config.CURRENT_YEAR, 1, 1)

    current = start_date
    while current <= today:
        if current.weekday() < 5:
            filepath = config.DATA_DIR / f"COTAHIST_D{current:%d%m%Y}.ZIP"

            if filepath.exists() and not force:
                downloaded_files.append(filepath)
            else:
                url = config.DAILY_COTAHIST_URL.format(date=current)
                try:
                    resp = requests.head(
                        url, headers=config.B3_HEADERS, timeout=10, allow_redirects=True
                    )
                    if resp.status_code == 200:
                        logger.info(f"Downloading daily file for {current:%Y-%m-%d}...")
                        resp = requests.get(url, headers=config.B3_HEADERS, timeout=60)
                        resp.raise_for_status()
                        with open(filepath, "wb") as f:
                            f.write(resp.content)
                        downloaded_files.append(filepath)
                except requests.RequestException:
                    pass

        current += timedelta(days=1)

    return downloaded_files


def fetch_corporate_actions(ticker: str) -> pd.DataFrame:
    """
    Fetch dividend and JCP data from StatusInvest for a single ticker.

    Args:
        ticker: Stock ticker (e.g., 'PETR4')

    Returns:
        DataFrame with columns: [ticker, event_date, event_type, value]
    """
    url = config.STATUSINVEST_PROVENTS_URL.format(ticker=ticker)
    headers = {
        **config.STATUSINVEST_HEADERS,
        "Referer": f"https://statusinvest.com.br/acoes/{ticker.lower()}",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data is None:
            return pd.DataFrame(columns=["ticker", "event_date", "event_type", "value"])

        events = []
        models = data.get("assetEarningsModels", [])

        for event in models:
            event_type = event.get("et", "")
            if event_type in ("Dividendo", "JCP", "Rend. Tributado"):
                ed_str = event.get("ed", "")
                value = event.get("v", 0)

                if ed_str and value > 0:
                    try:
                        event_date = datetime.strptime(ed_str, "%d/%m/%Y")
                        events.append(
                            {
                                "ticker": ticker,
                                "event_date": event_date.date(),
                                "event_type": event_type,
                                "value": float(value),
                            }
                        )
                    except ValueError:
                        continue

        if events:
            return pd.DataFrame(events)
        return pd.DataFrame(columns=["ticker", "event_date", "event_type", "value"])

    except requests.RequestException as e:
        logger.warning(f"Failed to fetch corporate actions for {ticker}: {e}")
        return pd.DataFrame(columns=["ticker", "event_date", "event_type", "value"])
    except (KeyError, ValueError) as e:
        logger.warning(f"Failed to parse corporate actions for {ticker}: {e}")
        return pd.DataFrame(columns=["ticker", "event_date", "event_type", "value"])


def fetch_all_corporate_actions(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch corporate actions for all tickers with rate limiting and progress logging.

    Args:
        tickers: List of ticker symbols

    Returns:
        DataFrame with all corporate actions
    """
    all_actions = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        if i % 10 == 0 or i == total:
            logger.info(f"Fetching corporate actions: {i}/{total} ({ticker})")

        actions = fetch_corporate_actions(ticker)
        if not actions.empty:
            all_actions.append(actions)

        time.sleep(config.RATE_LIMIT_DELAY)

    if all_actions:
        result = pd.concat(all_actions, ignore_index=True)
        logger.info(
            f"Fetched {len(result)} corporate action events for {len(tickers)} tickers"
        )
        return result

    return pd.DataFrame(columns=["ticker", "event_date", "event_type", "value"])
