"""
Unit tests for the fetch layer of b3_pipeline/b3_corporate_actions.py.

Tests cover:
- fetch_company_data()
- fetch_cash_dividends_paginated()
- fetch_all_corporate_actions()

All tests use unittest.mock.patch to stub requests.get.
No real HTTP calls are made. No real SQLite file is used.
"""
import json
import sqlite3
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from b3_pipeline import storage
from b3_pipeline.b3_corporate_actions import (
    fetch_company_data,
    fetch_cash_dividends_paginated,
    fetch_all_corporate_actions,
)


def _in_memory_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    storage.init_db(conn, rebuild=False)
    return conn


def _make_response(json_data=None, raise_exc=None, json_raise=None, status_code=200):
    """Create a mock requests.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status = MagicMock()
    if raise_exc is not None:
        mock_resp.raise_for_status.side_effect = raise_exc
    if json_raise is not None:
        mock_resp.json.side_effect = json_raise
    else:
        mock_resp.json.return_value = json_data
    return mock_resp


# ---------------------------------------------------------------------------
# TestFetchCompanyData
# ---------------------------------------------------------------------------

class TestFetchCompanyData:
    """Tests for fetch_company_data()."""

    def test_returns_dict_on_success(self):
        """Mock a 200 JSON dict response; result should be a dict with 'companyName'."""
        payload = {
            "companyName": "PETRO",
            "tradingName": "PETROBRAS",
            "cnpj": "33.000.167/0001-01",
            "stockDividends": [],
            "cashDividends": [],
        }
        with patch(
            "b3_pipeline.b3_corporate_actions.requests.get",
            return_value=_make_response(json_data=payload),
        ):
            result = fetch_company_data("PETR")

        assert isinstance(result, dict)
        assert "companyName" in result

    def test_returns_first_element_when_list(self):
        """When B3 returns a list, fetch_company_data should return index 0."""
        payload = [{"companyName": "PETRO"}, {"companyName": "OTHER"}]
        with patch(
            "b3_pipeline.b3_corporate_actions.requests.get",
            return_value=_make_response(json_data=payload),
        ):
            result = fetch_company_data("PETR")

        assert result == {"companyName": "PETRO"}

    def test_returns_none_on_request_exception(self):
        """A RequestException should be caught and None returned."""
        import requests as _requests

        with patch(
            "b3_pipeline.b3_corporate_actions.requests.get",
            side_effect=_requests.RequestException("timeout"),
        ):
            result = fetch_company_data("FAIL")

        assert result is None

    def test_records_failure_when_conn_provided(self):
        """On RequestException with a conn, a row should appear in fetch_failures."""
        import requests as _requests

        conn = _in_memory_conn()
        with patch(
            "b3_pipeline.b3_corporate_actions.requests.get",
            side_effect=_requests.RequestException("timeout"),
        ):
            result = fetch_company_data("FAIL", conn=conn)

        assert result is None
        cursor = conn.cursor()
        cursor.execute(
            "SELECT company_code FROM fetch_failures WHERE company_code='FAIL'"
        )
        rows = cursor.fetchall()
        conn.close()
        assert len(rows) == 1

    def test_returns_none_on_json_decode_error(self):
        """A JSONDecodeError from .json() should be caught and None returned."""
        with patch(
            "b3_pipeline.b3_corporate_actions.requests.get",
            return_value=_make_response(
                json_raise=json.JSONDecodeError("bad", "", 0)
            ),
        ):
            result = fetch_company_data("FAIL")

        assert result is None

    def test_returns_none_when_response_is_empty_list(self):
        """An empty list response should result in None."""
        with patch(
            "b3_pipeline.b3_corporate_actions.requests.get",
            return_value=_make_response(json_data=[]),
        ):
            result = fetch_company_data("PETR")

        assert result is None


# ---------------------------------------------------------------------------
# TestFetchCashDividendsPaginated
# ---------------------------------------------------------------------------

class TestFetchCashDividendsPaginated:
    """Tests for fetch_cash_dividends_paginated()."""

    def test_returns_all_records_single_page(self):
        """Single page response: should return a list of length 2."""
        payload = {
            "page": {"totalPages": 1, "totalRecords": 2},
            "results": [{"isinCode": "BRPETR"}, {"isinCode": "BRPETR"}],
        }
        with patch(
            "b3_pipeline.b3_corporate_actions.requests.get",
            return_value=_make_response(json_data=payload),
        ):
            result = fetch_cash_dividends_paginated("PETROBRAS")

        assert isinstance(result, list)
        assert len(result) == 2

    def test_paginates_across_multiple_pages(self):
        """Two-page response: both pages fetched and combined."""
        page1 = {
            "page": {"totalPages": 2, "totalRecords": 2},
            "results": [{"isinCode": "BRPETR1"}],
        }
        page2 = {
            "page": {"totalPages": 2, "totalRecords": 2},
            "results": [{"isinCode": "BRPETR2"}],
        }
        responses = [
            _make_response(json_data=page1),
            _make_response(json_data=page2),
        ]
        with patch(
            "b3_pipeline.b3_corporate_actions.requests.get",
            side_effect=responses,
        ) as mock_get:
            result = fetch_cash_dividends_paginated("PETROBRAS")

        assert len(result) == 2
        assert mock_get.call_count == 2

    def test_returns_empty_list_on_request_exception(self):
        """A RequestException should be caught and [] returned."""
        import requests as _requests

        with patch(
            "b3_pipeline.b3_corporate_actions.requests.get",
            side_effect=_requests.RequestException("timeout"),
        ):
            result = fetch_cash_dividends_paginated("FAIL")

        assert result == []

    def test_returns_empty_list_on_json_error(self):
        """A JSONDecodeError should be caught and [] returned."""
        with patch(
            "b3_pipeline.b3_corporate_actions.requests.get",
            return_value=_make_response(
                json_raise=json.JSONDecodeError("bad", "", 0)
            ),
        ):
            result = fetch_cash_dividends_paginated("FAIL")

        assert result == []


# ---------------------------------------------------------------------------
# TestFetchAllCorporateActions
# ---------------------------------------------------------------------------

class TestFetchAllCorporateActions:
    """Tests for fetch_all_corporate_actions() with patched sub-functions."""

    def test_returns_three_dataframes(self):
        """Should always return a 3-tuple of DataFrames even when all data is None."""
        with patch(
            "b3_pipeline.b3_corporate_actions.fetch_company_data",
            return_value=None,
        ):
            result = fetch_all_corporate_actions(["PETR", "VALE"], ticker_to_isin={})

        assert len(result) == 3
        corp_df, stock_df, skipped_df = result
        assert isinstance(corp_df, pd.DataFrame)
        assert isinstance(stock_df, pd.DataFrame)
        assert isinstance(skipped_df, pd.DataFrame)

    def test_processes_stock_dividends(self):
        """A DESDOBRAMENTO record should land in stock_actions_df as STOCK_SPLIT."""
        company_data = {
            "tradingName": "PETROBRAS",
            "cnpj": "33.000.167/0001-01",
            "stockDividends": [
                {
                    "label": "DESDOBRAMENTO",
                    "factor": "2",
                    "isinCode": "BRPETR",
                    "lastDatePrior": "01/01/2020",
                }
            ],
        }
        with patch(
            "b3_pipeline.b3_corporate_actions.fetch_company_data",
            return_value=company_data,
        ), patch(
            "b3_pipeline.b3_corporate_actions.fetch_cash_dividends_paginated",
            return_value=[],
        ):
            corp_df, stock_df, skipped_df = fetch_all_corporate_actions(
                ["PETR"], ticker_to_isin={}
            )

        assert len(stock_df) == 1
        assert stock_df.iloc[0]["action_type"] == "STOCK_SPLIT"

    def test_processes_cash_dividends(self):
        """A DIVIDENDO cash record should land in corp_actions_df as CASH_DIVIDEND."""
        company_data = {
            "tradingName": "PETROBRAS",
            "cnpj": "33.000.167/0001-01",
            "stockDividends": [],
        }
        cash_records = [
            {
                "isinCode": "BRPETR3",
                "label": "DIVIDENDO",
                "rate": "1,50",
                "lastDatePrior": "01/06/2020",
            }
        ]
        with patch(
            "b3_pipeline.b3_corporate_actions.fetch_company_data",
            return_value=company_data,
        ), patch(
            "b3_pipeline.b3_corporate_actions.fetch_cash_dividends_paginated",
            return_value=cash_records,
        ):
            corp_df, stock_df, skipped_df = fetch_all_corporate_actions(
                ["PETR"], ticker_to_isin={"PETR3": "BRPETR3"}
            )

        assert len(corp_df) >= 1
        assert corp_df.iloc[0]["event_type"] == "CASH_DIVIDEND"

    def test_skips_company_when_fetch_returns_none(self):
        """When fetch_company_data returns None, no crash and empty DataFrames."""
        with patch(
            "b3_pipeline.b3_corporate_actions.fetch_company_data",
            return_value=None,
        ):
            corp_df, stock_df, skipped_df = fetch_all_corporate_actions(
                ["PETR"], ticker_to_isin={}
            )

        assert corp_df.empty
        assert stock_df.empty

    def test_deduplicates_results(self):
        """Duplicate (isin_code, event_date, event_type) rows should be dropped."""
        # Same DESDOBRAMENTO for BRPETR returned by two company names
        company_data = {
            "tradingName": "PETROBRAS",
            "cnpj": "33.000.167/0001-01",
            "stockDividends": [
                {
                    "label": "DESDOBRAMENTO",
                    "factor": "2",
                    "isinCode": "BRPETR",
                    "lastDatePrior": "01/01/2020",
                }
            ],
        }
        with patch(
            "b3_pipeline.b3_corporate_actions.fetch_company_data",
            return_value=company_data,
        ), patch(
            "b3_pipeline.b3_corporate_actions.fetch_cash_dividends_paginated",
            return_value=[],
        ):
            corp_df, stock_df, skipped_df = fetch_all_corporate_actions(
                ["PETR", "PETR"],  # same company name twice to force duplicate
                ticker_to_isin={},
            )

        # After dedup the (isin_code, event_date, event_type) tuple must be unique
        if not corp_df.empty:
            dup_mask = corp_df.duplicated(
                subset=["isin_code", "event_date", "event_type"]
            )
            assert not dup_mask.any(), "Duplicate corp actions found after dedup"
