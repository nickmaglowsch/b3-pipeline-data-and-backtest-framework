"""
Download CVM IPE document index ZIP files.

IMPORTANT: As documented in tasks/ipe-structure-report.md, the IPE dataset
is a DOCUMENT FILING INDEX — not structured financial statement data.
Each yearly ZIP contains a single CSV with metadata (company ID, document
category, dates, download links to PDFs/HTML) but no financial values.

URL pattern: https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/ipe_cia_aberta_{year}.zip
Year range available: 2003-present
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import requests

from . import config

logger = logging.getLogger(__name__)


def download_ipe_file(year: int, force: bool = False) -> Optional[Path]:
    """Download an IPE document index ZIP file for the given year.

    Skips download if the file already exists and force=False.
    On failure, cleans up any partial file and returns None.
    """
    filename = f"ipe_cia_aberta_{year}.zip"
    filepath = config.CVM_DATA_DIR / filename
    url = f"{config.CVM_IPE_BASE_URL}{filename}"

    if filepath.exists() and not force:
        logger.debug(f"IPE file already exists: {filepath}")
        return filepath

    logger.info(f"Downloading IPE file for {year}: {url}")

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
                        if downloaded % (2 * 1024 * 1024) < 8192:
                            logger.debug(
                                f"Downloaded {downloaded:,}/{total_size:,} bytes ({pct:.1f}%)"
                            )

        logger.info(f"Downloaded IPE {year}: {filepath} ({downloaded:,} bytes)")
        return filepath

    except requests.RequestException as e:
        logger.error(f"Failed to download IPE {year} from {url}: {e}")
        if filepath.exists():
            filepath.unlink()
        return None


def detect_available_ipe_years(start_year: int = 2003, end_year: int = 2009) -> List[int]:
    """Probe CVM server to find which IPE years are available.

    Returns list of year integers where the server returns HTTP 200.
    Default range is 2003-2009 (pre-DFP era).
    """
    available = []
    for year in range(start_year, end_year + 1):
        filename = f"ipe_cia_aberta_{year}.zip"
        url = config.CVM_IPE_BASE_URL + filename
        try:
            resp = requests.head(url, headers=config.B3_HEADERS, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                available.append(year)
        except requests.RequestException:
            pass
        time.sleep(0.2)
    return available
