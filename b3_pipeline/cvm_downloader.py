"""
Download CVM bulk data files (DFP, ITR, FRE) from CVM's open data portal.

Mirrors the patterns established in b3_pipeline/downloader.py:
- Streaming downloads, file existence checks, cleanup on failure.
- Returns Optional[Path].
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

from . import config

logger = logging.getLogger(__name__)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _download_file(url: str, filepath: Path, force: bool = False) -> Optional[Path]:
    """
    Generic streaming file download.

    Skips the download if the file already exists and force=False.
    On failure, cleans up any partial file and returns None.
    """
    if filepath.exists() and not force:
        logger.debug(f"File already exists: {filepath}")
        return filepath

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


def _detect_years(base_url: str, filename_fn) -> List[int]:
    """
    Probe the CVM server with HEAD requests to find available years.

    Args:
        base_url: Base URL for the document type.
        filename_fn: Callable(year) -> filename string.

    Returns:
        List of years (ints) that return HTTP 200.
    """
    available = []
    current_year = datetime.now().year

    for year in range(config.CVM_START_YEAR, current_year + 1):
        url = base_url + filename_fn(year)
        try:
            resp = requests.head(url, headers=config.B3_HEADERS, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                available.append(year)
        except requests.RequestException:
            pass
        time.sleep(0.2)

    return available


# ── Public API ─────────────────────────────────────────────────────────────────

def download_dfp_file(year: int, force: bool = False) -> Optional[Path]:
    """Download a DFP annual ZIP file for the given year."""
    filename = f"dfp_cia_aberta_{year}.zip"
    filepath = config.CVM_DATA_DIR / filename
    url = f"{config.CVM_DFP_BASE_URL}{filename}"
    return _download_file(url, filepath, force=force)


def download_itr_file(year: int, force: bool = False) -> Optional[Path]:
    """Download an ITR quarterly ZIP file for the given year."""
    filename = f"itr_cia_aberta_{year}.zip"
    filepath = config.CVM_DATA_DIR / filename
    url = f"{config.CVM_ITR_BASE_URL}{filename}"
    return _download_file(url, filepath, force=force)


def download_fre_file(year: int, force: bool = False) -> Optional[Path]:
    """Download a FRE ZIP file for the given year."""
    filename = f"fre_cia_aberta_{year}.zip"
    filepath = config.CVM_DATA_DIR / filename
    url = f"{config.CVM_FRE_BASE_URL}{filename}"
    return _download_file(url, filepath, force=force)


def download_all_cvm_files(
    start_year: int = None,
    end_year: int = None,
    force: bool = False,
) -> Dict[str, List[Optional[Path]]]:
    """
    Download all DFP, ITR, and FRE files for the given year range.

    Args:
        start_year: First year to download (default: config.CVM_START_YEAR)
        end_year: Last year to download (default: current year)
        force: Re-download even if files already exist

    Returns:
        dict with keys "dfp", "itr", "fre", each a list of Paths (or None on failure)
    """
    if start_year is None:
        start_year = config.CVM_START_YEAR
    if end_year is None:
        end_year = datetime.now().year

    dfp_paths: List[Optional[Path]] = []
    itr_paths: List[Optional[Path]] = []
    fre_paths: List[Optional[Path]] = []

    for year in range(start_year, end_year + 1):
        logger.info(f"Downloading CVM files for year {year}...")

        dfp_paths.append(download_dfp_file(year, force=force))
        time.sleep(0.2)

        itr_paths.append(download_itr_file(year, force=force))
        time.sleep(0.2)

        fre_paths.append(download_fre_file(year, force=force))
        time.sleep(0.2)

    return {"dfp": dfp_paths, "itr": itr_paths, "fre": fre_paths}


def detect_available_cvm_years(doc_type: str) -> List[int]:
    """
    Probe CVM server to find which years are available for the given doc_type.

    Args:
        doc_type: One of "DFP", "ITR", or "FRE" (case-insensitive)

    Returns:
        List of year integers where the server returns HTTP 200.
    """
    doc_type = doc_type.upper()
    if doc_type == "DFP":
        base_url = config.CVM_DFP_BASE_URL
        filename_fn = lambda y: f"dfp_cia_aberta_{y}.zip"
    elif doc_type == "ITR":
        base_url = config.CVM_ITR_BASE_URL
        filename_fn = lambda y: f"itr_cia_aberta_{y}.zip"
    elif doc_type == "FRE":
        base_url = config.CVM_FRE_BASE_URL
        filename_fn = lambda y: f"fre_cia_aberta_{y}.zip"
    else:
        raise ValueError(f"Unknown doc_type: {doc_type!r}. Expected DFP, ITR, or FRE.")

    return _detect_years(base_url, filename_fn)
