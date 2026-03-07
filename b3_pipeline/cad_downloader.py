"""
Download the CVM CAD (company register) bulk CSV file.

The CAD dataset is a single bulk file (not yearly ZIPs) covering all companies
that have ever filed with CVM, including delisted companies.

URL: https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import requests

from . import config

logger = logging.getLogger(__name__)


def download_cad_file(force: bool = False) -> Optional[Path]:
    """Download the CVM CAD company register CSV.

    Skips download if the file already exists and force=False.
    On failure, cleans up any partial file and returns None.

    Returns the local Path on success, None on failure.
    """
    filepath = config.CVM_DATA_DIR / config.CVM_CAD_FILENAME
    url = config.CVM_CAD_BASE_URL + config.CVM_CAD_FILENAME

    if filepath.exists() and not force:
        logger.debug(f"CAD file already exists: {filepath}")
        return filepath

    logger.info(f"Downloading CAD company register from {url}...")

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
                        if downloaded % (5 * 1024 * 1024) < 8192:
                            logger.debug(
                                f"Downloaded {downloaded:,}/{total_size:,} bytes ({pct:.1f}%)"
                            )

        logger.info(f"Downloaded CAD file: {filepath} ({downloaded:,} bytes)")
        return filepath

    except requests.RequestException as e:
        logger.error(f"Failed to download CAD file from {url}: {e}")
        if filepath.exists():
            filepath.unlink()
        return None
