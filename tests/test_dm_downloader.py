"""
Tests for the Download Múltiplo downloader's local filename scheme.

No network: files are pre-created so download_document returns early via
its filepath.exists() check.
"""
from __future__ import annotations

from datetime import date

from b3_pipeline import dm_downloader


def test_download_filename_includes_submission_date(tmp_path, monkeypatch):
    """DataRef is the PERIOD reference date: a restatement of the same period
    submitted on a later day must map to a DIFFERENT file, not be skipped as
    already-downloaded."""
    monkeypatch.setattr(dm_downloader, "DM_DATA_DIR", tmp_path)
    link = {
        "url": "http://example.com/pkg.zip",
        "Documento": "DFP",
        "ccvm": "1234",
        "DataRef": "31/12/2006",
    }
    original = tmp_path / "DFP_1234_31122006_20070315.zip"
    restatement = tmp_path / "DFP_1234_31122006_20070601.zip"
    original.write_bytes(b"original")
    restatement.write_bytes(b"restatement")

    assert dm_downloader.download_document(link, submission_date=date(2007, 3, 15)) == original
    assert dm_downloader.download_document(link, submission_date=date(2007, 6, 1)) == restatement
