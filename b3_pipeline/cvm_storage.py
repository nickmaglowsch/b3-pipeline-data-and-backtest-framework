"""
CVM fundamentals storage operations.

All financial values are stored in thousands of BRL as reported by CVM.
Do NOT multiply by 1000 when storing or reading.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def upsert_cvm_company(
    conn: sqlite3.Connection,
    cnpj: str,
    ticker: Optional[str],
    company_name: Optional[str],
    cvm_code: Optional[str],
    b3_trading_name: Optional[str],
) -> None:
    """Insert or update a CVM company mapping row.

    Uses ON CONFLICT(cnpj) DO UPDATE so calling twice with the same CNPJ
    updates the existing row rather than raising an IntegrityError.
    """
    sql = """
        INSERT INTO cvm_companies (cnpj, ticker, company_name, cvm_code, b3_trading_name, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(cnpj) DO UPDATE SET
            ticker = excluded.ticker,
            company_name = excluded.company_name,
            cvm_code = excluded.cvm_code,
            b3_trading_name = excluded.b3_trading_name,
            updated_at = CURRENT_TIMESTAMP
    """
    conn.execute(sql, (cnpj, ticker, company_name, cvm_code, b3_trading_name))
    conn.commit()
    logger.debug(f"Upserted cvm_company: cnpj={cnpj} ticker={ticker}")


def upsert_cvm_filing(
    conn: sqlite3.Connection,
    filing_id: str,
    cnpj: str,
    doc_type: str,
    period_end: str,
    filing_date: str,
    filing_version: int,
    fiscal_year: Optional[int],
    quarter: Optional[int],
    source_file: Optional[str],
) -> None:
    """Insert or replace a CVM filing row.

    The primary key is filing_id. A restatement (new version) produces a new
    filing_id, so it is a new row rather than an update.
    """
    sql = """
        INSERT OR REPLACE INTO cvm_filings
            (filing_id, cnpj, doc_type, period_end, filing_date, filing_version, fiscal_year, quarter, source_file)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    conn.execute(
        sql,
        (filing_id, cnpj, doc_type, period_end, filing_date, filing_version, fiscal_year, quarter, source_file),
    )
    conn.commit()
    logger.debug(f"Upserted cvm_filing: filing_id={filing_id}")


def upsert_fundamentals_pit(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Batch upsert fundamentals_pit rows from a DataFrame.

    Uses INSERT OR REPLACE keyed on filing_id (primary key).
    Returns the number of rows processed.
    """
    if df.empty:
        logger.warning("upsert_fundamentals_pit: empty DataFrame, nothing to insert")
        return 0

    sql = """
        INSERT OR REPLACE INTO fundamentals_pit (
            filing_id, cnpj, ticker, period_end, filing_date, filing_version,
            doc_type, fiscal_year, quarter,
            revenue, net_income, ebitda, total_assets, equity, net_debt,
            shares_outstanding, pe_ratio, pb_ratio, ev_ebitda
        ) VALUES (
            :filing_id, :cnpj, :ticker, :period_end, :filing_date, :filing_version,
            :doc_type, :fiscal_year, :quarter,
            :revenue, :net_income, :ebitda, :total_assets, :equity, :net_debt,
            :shares_outstanding, :pe_ratio, :pb_ratio, :ev_ebitda
        )
    """

    # Fill optional columns not present in df with None
    optional_cols = [
        "ticker", "fiscal_year", "quarter",
        "revenue", "net_income", "ebitda", "total_assets", "equity", "net_debt",
        "shares_outstanding", "pe_ratio", "pb_ratio", "ev_ebitda",
    ]
    for col in optional_cols:
        if col not in df.columns:
            df = df.copy()
            df[col] = None

    records = df.to_dict("records")
    conn.executemany(sql, records)
    conn.commit()

    logger.info(f"Upserted {len(records):,} fundamentals_pit rows")
    return len(records)


def get_fundamentals_stats(conn: sqlite3.Connection) -> dict:
    """Return counts for the fundamentals stats panel."""
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM cvm_companies")
    total_cvm_companies = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM cvm_filings")
    total_cvm_filings = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM fundamentals_pit")
    total_fundamentals_pit = cursor.fetchone()[0]

    return {
        "total_cvm_companies": total_cvm_companies,
        "total_cvm_filings": total_cvm_filings,
        "total_fundamentals_pit": total_fundamentals_pit,
    }


def get_cvm_company_map(conn: sqlite3.Connection) -> dict:
    """Return {cnpj: ticker} for all mapped companies."""
    cursor = conn.cursor()
    cursor.execute("SELECT cnpj, ticker FROM cvm_companies WHERE ticker IS NOT NULL")
    return {row[0]: row[1] for row in cursor.fetchall()}


def get_ticker_to_cnpj_map(conn: sqlite3.Connection) -> dict:
    """Return {ticker_root: cnpj} reverse lookup."""
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, cnpj FROM cvm_companies WHERE ticker IS NOT NULL")
    return {row[0]: row[1] for row in cursor.fetchall()}
