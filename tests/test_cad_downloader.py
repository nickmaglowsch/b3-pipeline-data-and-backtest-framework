"""
Tests for the CAD downloader (Task 02 — TDD).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def test_download_cad_file_skips_when_exists(tmp_path):
    """If file already exists (and force=False), requests.get is never called."""
    from b3_pipeline import cad_downloader, config

    # Place a dummy file at the expected download path
    dummy_file = tmp_path / config.CVM_CAD_FILENAME
    dummy_file.write_bytes(b"dummy content")

    with patch("b3_pipeline.cad_downloader.config") as mock_cfg, \
         patch("b3_pipeline.cad_downloader.requests.get") as mock_get:
        mock_cfg.CVM_DATA_DIR = tmp_path
        mock_cfg.CVM_CAD_BASE_URL = "https://example.com/"
        mock_cfg.CVM_CAD_FILENAME = config.CVM_CAD_FILENAME
        mock_cfg.B3_HEADERS = config.B3_HEADERS

        result = cad_downloader.download_cad_file(force=False)
        mock_get.assert_not_called()
        assert result is not None


def test_download_cad_file_downloads_when_missing(tmp_path):
    """When file does not exist, requests.get is called once."""
    from b3_pipeline import cad_downloader, config

    with patch("b3_pipeline.cad_downloader.config") as mock_cfg, \
         patch("b3_pipeline.cad_downloader.requests.get") as mock_get:
        mock_cfg.CVM_DATA_DIR = tmp_path
        mock_cfg.CVM_CAD_BASE_URL = "https://example.com/"
        mock_cfg.CVM_CAD_FILENAME = config.CVM_CAD_FILENAME
        mock_cfg.B3_HEADERS = config.B3_HEADERS

        # Mock a successful response
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {"content-length": "10"}
        mock_resp.iter_content.return_value = [b"hello data"]
        mock_get.return_value = mock_resp

        result = cad_downloader.download_cad_file(force=False)
        mock_get.assert_called_once()


def test_download_cad_file_force_redownloads(tmp_path):
    """When force=True, existing file is re-downloaded."""
    from b3_pipeline import cad_downloader, config

    dummy_file = tmp_path / config.CVM_CAD_FILENAME
    dummy_file.write_bytes(b"old content")

    with patch("b3_pipeline.cad_downloader.config") as mock_cfg, \
         patch("b3_pipeline.cad_downloader.requests.get") as mock_get:
        mock_cfg.CVM_DATA_DIR = tmp_path
        mock_cfg.CVM_CAD_BASE_URL = "https://example.com/"
        mock_cfg.CVM_CAD_FILENAME = config.CVM_CAD_FILENAME
        mock_cfg.B3_HEADERS = config.B3_HEADERS

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {"content-length": "10"}
        mock_resp.iter_content.return_value = [b"new data"]
        mock_get.return_value = mock_resp

        result = cad_downloader.download_cad_file(force=True)
        mock_get.assert_called_once()
