"""
Download COTAHIST files from B3.

Note: Corporate actions are now fetched via b3_corporate_actions module,
which queries B3 directly as the single source of truth.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

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
