"""
Tests for the CVM bulk data downloader (Task 03 — TDD).

All tests mock requests.get / requests.head — no real network calls.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import requests

from b3_pipeline import cvm_downloader


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_cvm_dir(tmp_path, monkeypatch):
    """Point CVM_DATA_DIR at a temporary directory for each test."""
    import b3_pipeline.config as _cfg
    monkeypatch.setattr(_cfg, "CVM_DATA_DIR", tmp_path)
    return tmp_path


# ── 1. Skip existing file ──────────────────────────────────────────────────────

def test_download_dfp_skips_existing_file(tmp_cvm_dir):
    """If the file already exists, requests.get should NOT be called."""
    # Pre-create the file
    expected = tmp_cvm_dir / "dfp_cia_aberta_2023.zip"
    expected.write_bytes(b"dummy")

    with patch("requests.get") as mock_get:
        result = cvm_downloader.download_dfp_file(2023)
        mock_get.assert_not_called()

    assert result == expected


# ── 2. Force re-download ───────────────────────────────────────────────────────

def test_download_dfp_force_redownloads(tmp_cvm_dir):
    """force=True should call requests.get even if file exists."""
    expected = tmp_cvm_dir / "dfp_cia_aberta_2023.zip"
    expected.write_bytes(b"old content")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-length": "5"}
    mock_resp.iter_content = MagicMock(return_value=[b"hello"])
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp) as mock_get:
        result = cvm_downloader.download_dfp_file(2023, force=True)
        mock_get.assert_called_once()

    assert result == expected


# ── 3. Cleanup on failure ─────────────────────────────────────────────────────

def test_download_dfp_cleans_up_on_failure(tmp_cvm_dir):
    """On RequestException, return None and do not leave a partial file."""
    with patch("requests.get", side_effect=requests.RequestException("timeout")):
        result = cvm_downloader.download_dfp_file(2023)

    assert result is None

    partial = tmp_cvm_dir / "dfp_cia_aberta_2023.zip"
    assert not partial.exists(), "Partial file should be cleaned up after failure"


# ── 4. download_all returns dict with three keys ───────────────────────────────

def test_download_all_returns_dict_with_three_keys(tmp_cvm_dir, monkeypatch):
    """download_all_cvm_files should return {'dfp': [...], 'itr': [...], 'fre': [...]}."""
    dummy_path = tmp_cvm_dir / "dummy.zip"
    dummy_path.write_bytes(b"x")

    monkeypatch.setattr(cvm_downloader, "download_dfp_file", lambda year, force=False: dummy_path)
    monkeypatch.setattr(cvm_downloader, "download_itr_file", lambda year, force=False: dummy_path)
    monkeypatch.setattr(cvm_downloader, "download_fre_file", lambda year, force=False: dummy_path)

    result = cvm_downloader.download_all_cvm_files(start_year=2023, end_year=2023)

    assert isinstance(result, dict)
    assert "dfp" in result
    assert "itr" in result
    assert "fre" in result
    assert isinstance(result["dfp"], list)
    assert isinstance(result["itr"], list)
    assert isinstance(result["fre"], list)


# ── 5. detect_available_years filters 200 only ────────────────────────────────

def test_detect_available_years_filters_200_only(monkeypatch):
    """Only years where HEAD returns 200 should be included."""
    import b3_pipeline.config as _cfg
    monkeypatch.setattr(_cfg, "CVM_START_YEAR", 2020)

    def fake_head(url, **kwargs):
        resp = MagicMock()
        # Return 200 only for 2022 and 2023
        if "2022" in url or "2023" in url:
            resp.status_code = 200
        else:
            resp.status_code = 404
        return resp

    with patch("requests.head", side_effect=fake_head):
        # CVM_START_YEAR=2020, current year >= 2026, so probes 2020..2026
        result = cvm_downloader.detect_available_cvm_years("DFP")

    assert 2022 in result
    assert 2023 in result
    # Other years should NOT be in result
    for y in [2020, 2021, 2024, 2025]:
        assert y not in result, f"Year {y} should not be available (404)"
