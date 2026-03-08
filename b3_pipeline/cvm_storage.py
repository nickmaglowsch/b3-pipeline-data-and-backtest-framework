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
    ticker_root = ticker[:4] if ticker else None
    sql = """
        INSERT INTO cvm_companies (cnpj, ticker, ticker_root, company_name, cvm_code, b3_trading_name, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(cnpj) DO UPDATE SET
            ticker = excluded.ticker,
            ticker_root = excluded.ticker_root,
            company_name = excluded.company_name,
            cvm_code = excluded.cvm_code,
            b3_trading_name = excluded.b3_trading_name,
            updated_at = CURRENT_TIMESTAMP
    """
    conn.execute(sql, (cnpj, ticker, ticker_root, company_name, cvm_code, b3_trading_name))
    conn.commit()
    logger.debug(f"Upserted cvm_company: cnpj={cnpj} ticker={ticker}")


def bulk_upsert_companies_index(conn: sqlite3.Connection, companies: list) -> int:
    """Upsert company rows from CVM file data (cnpj + cvm_code + company_name).

    Does NOT overwrite an existing ticker — only fills in missing company
    metadata. Call this from the CVM pipeline before reading the ticker map.

    companies: list of (cnpj, cvm_code, company_name) tuples.
    Returns number of rows inserted/updated.
    """
    sql = """
        INSERT INTO cvm_companies (cnpj, cvm_code, company_name, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(cnpj) DO UPDATE SET
            cvm_code    = COALESCE(cvm_companies.cvm_code, excluded.cvm_code),
            company_name = COALESCE(cvm_companies.company_name, excluded.company_name),
            updated_at  = CURRENT_TIMESTAMP
    """
    conn.executemany(sql, companies)
    conn.commit()
    return len(companies)


def update_ticker_by_cvm_code(
    conn: sqlite3.Connection, cvm_code: str, ticker: str, b3_trading_name: str = None
) -> bool:
    """Update ticker (and b3_trading_name) for all cvm_companies rows matching cvm_code.

    Called from the B3 corporate-actions pipeline after fetching company data.
    Returns True if at least one row was updated.
    """
    # B3 API returns codeCVM as a bare integer; CVM CSVs store 6-digit zero-padded strings.
    if cvm_code.isdigit():
        cvm_code = cvm_code.zfill(6)
    ticker_root = ticker[:4] if ticker else None
    result = conn.execute(
        """
        UPDATE cvm_companies
        SET ticker = ?, ticker_root = ?, b3_trading_name = ?, updated_at = CURRENT_TIMESTAMP
        WHERE cvm_code = ?
        """,
        (ticker, ticker_root, b3_trading_name, cvm_code),
    )
    conn.commit()
    return result.rowcount > 0


def populate_tickers_from_cvm_companies(conn: sqlite3.Connection) -> int:
    """Set fundamentals_pit.ticker from cvm_companies by joining on cnpj.

    Run this at the end of the CVM pipeline (after B3 pipeline has populated
    cvm_companies.ticker) so valuation ratios can be computed.
    Returns number of rows updated.
    """
    result = conn.execute(
        """
        UPDATE fundamentals_pit
        SET ticker = (
            SELECT ticker FROM cvm_companies
            WHERE cvm_companies.cnpj = fundamentals_pit.cnpj
              AND cvm_companies.ticker IS NOT NULL
        )
        WHERE fundamentals_pit.ticker IS NULL
          AND EXISTS (
              SELECT 1 FROM cvm_companies
              WHERE cvm_companies.cnpj = fundamentals_pit.cnpj
                AND cvm_companies.ticker IS NOT NULL
          )
        """
    )
    conn.commit()
    return result.rowcount


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
            shares_outstanding
        ) VALUES (
            :filing_id, :cnpj, :ticker, :period_end, :filing_date, :filing_version,
            :doc_type, :fiscal_year, :quarter,
            :revenue, :net_income, :ebitda, :total_assets, :equity, :net_debt,
            :shares_outstanding
        )
    """

    # Fill optional columns not present in df with None
    optional_cols = [
        "ticker", "fiscal_year", "quarter",
        "revenue", "net_income", "ebitda", "total_assets", "equity", "net_debt",
        "shares_outstanding",
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


def upsert_cad_company_dates(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Upsert company listing/delisting dates from the CVM CAD dataset.

    Does NOT overwrite existing ticker or b3_trading_name values.
    listing_date uses COALESCE (first known date is canonical).
    delisting_date is always overwritten (cancellations can be updated).
    Returns number of rows processed.
    """
    sql = """
        INSERT INTO cvm_companies (cnpj, cvm_code, company_name, listing_date, delisting_date, updated_at)
        VALUES (:cnpj, :cvm_code, :company_name, :listing_date, :delisting_date, CURRENT_TIMESTAMP)
        ON CONFLICT(cnpj) DO UPDATE SET
            cvm_code       = COALESCE(cvm_companies.cvm_code, excluded.cvm_code),
            company_name   = COALESCE(cvm_companies.company_name, excluded.company_name),
            listing_date   = COALESCE(cvm_companies.listing_date, excluded.listing_date),
            delisting_date = excluded.delisting_date,
            updated_at     = CURRENT_TIMESTAMP
    """
    # Normalise None/NaN to Python None for SQLite
    records = []
    for rec in df.to_dict("records"):
        records.append({
            "cnpj": rec["cnpj"],
            "cvm_code": rec.get("cvm_code"),
            "company_name": rec.get("company_name"),
            "listing_date": rec.get("listing_date") or None,
            "delisting_date": rec.get("delisting_date") or None,
        })
    conn.executemany(sql, records)
    conn.commit()
    logger.info(f"Upserted {len(records):,} CAD company date rows")
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

    cursor.execute("SELECT COUNT(*) FROM cvm_companies WHERE listing_date IS NOT NULL")
    companies_with_listing_date = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM cvm_companies WHERE delisting_date IS NOT NULL")
    companies_with_delisting_date = cursor.fetchone()[0]

    return {
        "total_cvm_companies": total_cvm_companies,
        "total_cvm_filings": total_cvm_filings,
        "total_fundamentals_pit": total_fundamentals_pit,
        "companies_with_listing_date": companies_with_listing_date,
        "companies_with_delisting_date": companies_with_delisting_date,
    }


def populate_company_isin_map(conn: sqlite3.Connection) -> int:
    """Populate (or refresh) company_isin_map by joining cvm_companies with prices.

    Links each (cnpj, isin_code) pair to the ticker and share_class derived from
    the ticker suffix. Sets is_primary=1 for the ON share class (suffix '3').

    Uses an in-memory Python dict to avoid a SUBSTR JOIN on the large prices table.
    Safe to call multiple times -- uses INSERT OR REPLACE to handle updates.
    Returns the number of rows inserted or replaced.
    """
    cursor = conn.cursor()

    # Build root -> cnpj map from cvm_companies (fast: small table)
    cursor.execute("SELECT ticker_root, cnpj FROM cvm_companies WHERE ticker_root IS NOT NULL")
    root_to_cnpj = {row[0]: row[1] for row in cursor.fetchall()}
    if not root_to_cnpj:
        return 0

    # Load prices grouped by (ticker, isin_code) — no JOIN needed
    cursor.execute("""
        SELECT ticker, isin_code, MIN(date), MAX(date)
        FROM prices
        WHERE isin_code != 'UNKNOWN'
        GROUP BY ticker, isin_code
    """)
    price_rows = cursor.fetchall()

    records = []
    for ticker, isin, first, last in price_rows:
        root = ticker[:4]
        cnpj = root_to_cnpj.get(root)
        if cnpj is None:
            continue
        last_char = ticker[-1] if ticker else ""
        last_two = ticker[-2:] if len(ticker) >= 2 else ""
        if len(ticker) >= 6 and last_two == "11":
            share_class = "UNT"
            is_primary = 0
        elif last_char == "3":
            share_class = "ON"
            is_primary = 1
        elif last_char == "4":
            share_class = "PN"
            is_primary = 0
        elif last_char == "5":
            share_class = "PNA"
            is_primary = 0
        elif last_char == "6":
            share_class = "PNB"
            is_primary = 0
        else:
            share_class = "OTHER"
            is_primary = 0
        records.append((cnpj, isin, ticker, share_class, is_primary, first, last))

    if not records:
        return 0

    cursor.executemany("""
        INSERT OR REPLACE INTO company_isin_map
            (cnpj, isin_code, ticker, share_class, is_primary, first_seen, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()
    count = len(records)
    logger.info(f"Populated company_isin_map: {count} rows inserted/replaced")
    return count


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


def upsert_fundamentals_monthly(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Batch upsert fundamentals_monthly rows from a DataFrame.

    Uses INSERT OR REPLACE keyed on (month_end, ticker).
    Returns the number of rows processed.
    """
    if df.empty:
        logger.warning("upsert_fundamentals_monthly: empty DataFrame, nothing to insert")
        return 0

    sql = """
        INSERT OR REPLACE INTO fundamentals_monthly (
            month_end, ticker,
            revenue, net_income, ebitda, total_assets, equity, net_debt,
            shares_outstanding
        ) VALUES (
            :month_end, :ticker,
            :revenue, :net_income, :ebitda, :total_assets, :equity, :net_debt,
            :shares_outstanding
        )
    """

    # Fill optional columns not present in df with None
    optional_cols = [
        "revenue", "net_income", "ebitda", "total_assets", "equity", "net_debt",
        "shares_outstanding",
    ]
    for col in optional_cols:
        if col not in df.columns:
            df = df.copy()
            df[col] = None

    records = df.to_dict("records")
    conn.executemany(sql, records)
    conn.commit()

    n = len(records)
    logger.info(f"Upserted {n} fundamentals_monthly rows")
    return n


def truncate_fundamentals_monthly(conn: sqlite3.Connection) -> None:
    """Delete all rows from fundamentals_monthly. Used before a full rebuild."""
    conn.execute("DELETE FROM fundamentals_monthly")
    conn.commit()
    logger.info("Truncated fundamentals_monthly table")
