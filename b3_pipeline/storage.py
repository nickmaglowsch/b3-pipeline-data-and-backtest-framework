"""
SQLite storage operations for B3 market data.
"""

import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


SCHEMA_PRICES = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    isin_code TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    quotation_factor INTEGER DEFAULT 1,
    split_adj_open REAL,
    split_adj_high REAL,
    split_adj_low REAL,
    split_adj_close REAL,
    adj_close REAL,
    PRIMARY KEY (ticker, date)
);
"""

SCHEMA_CORPORATE_ACTIONS = """
CREATE TABLE IF NOT EXISTS corporate_actions (
    isin_code TEXT NOT NULL,
    event_date DATE NOT NULL,
    event_type TEXT NOT NULL,
    value REAL,
    factor REAL,
    source TEXT DEFAULT 'B3',
    PRIMARY KEY (isin_code, event_date, event_type)
);
"""

SCHEMA_STOCK_ACTIONS = """
CREATE TABLE IF NOT EXISTS stock_actions (
    isin_code TEXT NOT NULL,
    ex_date DATE NOT NULL,
    action_type TEXT NOT NULL,
    factor REAL NOT NULL,
    source TEXT DEFAULT 'B3',
    PRIMARY KEY (isin_code, ex_date, action_type)
);
"""

SCHEMA_DETECTED_SPLITS = """
CREATE TABLE IF NOT EXISTS detected_splits (
    isin_code TEXT NOT NULL,
    ex_date DATE NOT NULL,
    split_factor REAL NOT NULL,
    description TEXT,
    PRIMARY KEY (isin_code, ex_date)
);
"""

SCHEMA_SKIPPED_EVENTS = """
CREATE TABLE IF NOT EXISTS skipped_events (
    isin_code TEXT NOT NULL,
    event_date DATE NOT NULL,
    label TEXT NOT NULL,
    factor REAL,
    source TEXT DEFAULT 'B3',
    reason TEXT,
    PRIMARY KEY (isin_code, event_date, label)
);
"""

SCHEMA_FETCH_FAILURES = """
CREATE TABLE IF NOT EXISTS fetch_failures (
    company_code TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    error_message TEXT,
    failed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    retry_count INTEGER DEFAULT 0,
    resolved INTEGER DEFAULT 0,
    PRIMARY KEY (company_code, endpoint)
);
"""

INDEX_PRICES_DATE = "CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);"
INDEX_PRICES_TICKER = "CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);"
INDEX_PRICES_ISIN = "CREATE INDEX IF NOT EXISTS idx_prices_isin ON prices(isin_code);"

# ── CVM Fundamentals Tables ────────────────────────────────────────────────────

SCHEMA_CVM_COMPANIES = """
CREATE TABLE IF NOT EXISTS cvm_companies (
    cnpj TEXT NOT NULL PRIMARY KEY,
    ticker TEXT,
    company_name TEXT,
    cvm_code TEXT,
    b3_trading_name TEXT,
    listing_date DATE,
    delisting_date DATE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

SCHEMA_CVM_FILINGS = """
CREATE TABLE IF NOT EXISTS cvm_filings (
    filing_id TEXT NOT NULL PRIMARY KEY,
    cnpj TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    period_end DATE NOT NULL,
    filing_date DATE NOT NULL,
    filing_version INTEGER NOT NULL DEFAULT 1,
    fiscal_year INTEGER,
    quarter INTEGER,
    source_file TEXT
);
"""

SCHEMA_FUNDAMENTALS_PIT = """
CREATE TABLE IF NOT EXISTS fundamentals_pit (
    filing_id TEXT NOT NULL PRIMARY KEY,
    cnpj TEXT NOT NULL,
    ticker TEXT,
    period_end DATE NOT NULL,
    filing_date DATE NOT NULL,
    filing_version INTEGER NOT NULL DEFAULT 1,
    doc_type TEXT NOT NULL,
    fiscal_year INTEGER,
    quarter INTEGER,
    revenue REAL,
    net_income REAL,
    ebitda REAL,
    total_assets REAL,
    equity REAL,
    net_debt REAL,
    shares_outstanding REAL,
    pe_ratio REAL,
    pb_ratio REAL,
    ev_ebitda REAL,
    FOREIGN KEY (filing_id) REFERENCES cvm_filings(filing_id)
);
"""

# Note: all financial values are stored in thousands of BRL as reported by CVM.
# Do NOT multiply by 1000 when storing or reading.

SCHEMA_COMPANY_ISIN_MAP = """
CREATE TABLE IF NOT EXISTS company_isin_map (
    cnpj TEXT NOT NULL,
    isin_code TEXT NOT NULL,
    ticker TEXT NOT NULL,
    share_class TEXT,
    is_primary INTEGER DEFAULT 0,
    first_seen DATE,
    last_seen DATE,
    PRIMARY KEY (cnpj, isin_code)
);
"""

INDEX_COMPANY_ISIN_MAP_CNPJ = "CREATE INDEX IF NOT EXISTS idx_company_isin_map_cnpj ON company_isin_map(cnpj);"
INDEX_COMPANY_ISIN_MAP_ISIN = "CREATE INDEX IF NOT EXISTS idx_company_isin_map_isin ON company_isin_map(isin_code);"
INDEX_COMPANY_ISIN_MAP_TICKER = "CREATE INDEX IF NOT EXISTS idx_company_isin_map_ticker ON company_isin_map(ticker);"
INDEX_COMPANY_ISIN_MAP_PRIMARY = "CREATE INDEX IF NOT EXISTS idx_company_isin_map_primary ON company_isin_map(cnpj, is_primary);"

SCHEMA_FUNDAMENTALS_MONTHLY = """
CREATE TABLE IF NOT EXISTS fundamentals_monthly (
    month_end DATE NOT NULL,
    ticker TEXT NOT NULL,
    revenue REAL,
    net_income REAL,
    ebitda REAL,
    total_assets REAL,
    equity REAL,
    net_debt REAL,
    shares_outstanding REAL,
    pe_ratio REAL,
    pb_ratio REAL,
    ev_ebitda REAL,
    PRIMARY KEY (month_end, ticker)
);
"""

INDEX_FUNDAMENTALS_MONTHLY_TICKER = "CREATE INDEX IF NOT EXISTS idx_fundamentals_monthly_ticker ON fundamentals_monthly(ticker);"
INDEX_FUNDAMENTALS_MONTHLY_MONTH_END = "CREATE INDEX IF NOT EXISTS idx_fundamentals_monthly_month_end ON fundamentals_monthly(month_end);"

INDEX_CVM_COMPANIES_TICKER = "CREATE INDEX IF NOT EXISTS idx_cvm_companies_ticker ON cvm_companies(ticker);"
INDEX_CVM_FILINGS_CNPJ = "CREATE INDEX IF NOT EXISTS idx_cvm_filings_cnpj ON cvm_filings(cnpj);"
INDEX_CVM_FILINGS_PERIOD = "CREATE INDEX IF NOT EXISTS idx_cvm_filings_period ON cvm_filings(period_end);"
INDEX_FUNDAMENTALS_PIT_CNPJ = "CREATE INDEX IF NOT EXISTS idx_fundamentals_pit_cnpj ON fundamentals_pit(cnpj);"
INDEX_FUNDAMENTALS_PIT_TICKER = "CREATE INDEX IF NOT EXISTS idx_fundamentals_pit_ticker ON fundamentals_pit(ticker);"
INDEX_FUNDAMENTALS_PIT_FILING_DATE = "CREATE INDEX IF NOT EXISTS idx_fundamentals_pit_filing_date ON fundamentals_pit(filing_date);"
INDEX_FUNDAMENTALS_PIT_PERIOD = "CREATE INDEX IF NOT EXISTS idx_fundamentals_pit_period ON fundamentals_pit(period_end);"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a SQLite database connection."""
    if db_path is None:
        db_path = config.DB_PATH

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    return conn


def init_db(conn: sqlite3.Connection, rebuild: bool = False) -> None:
    """Initialize the database schema."""
    cursor = conn.cursor()

    if rebuild:
        logger.info("Dropping existing tables...")
        # Drop fundamentals_monthly first (depends on nothing but should be cleaned)
        cursor.execute("DROP TABLE IF EXISTS fundamentals_monthly")
        # Drop CVM fundamentals tables first (before existing tables)
        cursor.execute("DROP TABLE IF EXISTS fundamentals_pit")
        cursor.execute("DROP TABLE IF EXISTS cvm_filings")
        cursor.execute("DROP TABLE IF EXISTS company_isin_map")
        cursor.execute("DROP TABLE IF EXISTS cvm_companies")
        # Drop original tables
        cursor.execute("DROP TABLE IF EXISTS prices")
        cursor.execute("DROP TABLE IF EXISTS corporate_actions")
        cursor.execute("DROP TABLE IF EXISTS stock_actions")
        cursor.execute("DROP TABLE IF EXISTS detected_splits")
        cursor.execute("DROP TABLE IF EXISTS skipped_events")
        cursor.execute("DROP TABLE IF EXISTS fetch_failures")

    logger.info("Creating database schema...")
    cursor.execute(SCHEMA_PRICES)
    cursor.execute(SCHEMA_CORPORATE_ACTIONS)
    cursor.execute(SCHEMA_STOCK_ACTIONS)
    cursor.execute(SCHEMA_DETECTED_SPLITS)
    cursor.execute(SCHEMA_SKIPPED_EVENTS)
    cursor.execute(SCHEMA_FETCH_FAILURES)
    cursor.execute(INDEX_PRICES_DATE)
    cursor.execute(INDEX_PRICES_TICKER)
    cursor.execute(INDEX_PRICES_ISIN)
    # CVM fundamentals tables (added after existing tables — do not reorder above)
    cursor.execute(SCHEMA_CVM_COMPANIES)
    cursor.execute(SCHEMA_CVM_FILINGS)
    cursor.execute(SCHEMA_FUNDAMENTALS_PIT)
    cursor.execute(SCHEMA_COMPANY_ISIN_MAP)
    cursor.execute(INDEX_CVM_COMPANIES_TICKER)
    cursor.execute(INDEX_CVM_FILINGS_CNPJ)
    cursor.execute(INDEX_CVM_FILINGS_PERIOD)
    cursor.execute(INDEX_FUNDAMENTALS_PIT_CNPJ)
    cursor.execute(INDEX_FUNDAMENTALS_PIT_TICKER)
    cursor.execute(INDEX_FUNDAMENTALS_PIT_FILING_DATE)
    cursor.execute(INDEX_FUNDAMENTALS_PIT_PERIOD)
    cursor.execute(INDEX_COMPANY_ISIN_MAP_CNPJ)
    cursor.execute(INDEX_COMPANY_ISIN_MAP_ISIN)
    cursor.execute(INDEX_COMPANY_ISIN_MAP_TICKER)
    cursor.execute(INDEX_COMPANY_ISIN_MAP_PRIMARY)
    # fundamentals_monthly snapshot table (added after CVM tables)
    cursor.execute(SCHEMA_FUNDAMENTALS_MONTHLY)
    cursor.execute(INDEX_FUNDAMENTALS_MONTHLY_TICKER)
    cursor.execute(INDEX_FUNDAMENTALS_MONTHLY_MONTH_END)

    # Migrate existing databases: add new columns if they don't exist
    _migrate_schema(conn)

    conn.commit()
    logger.info("Database schema initialized")


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Apply non-destructive schema migrations for existing databases."""
    cursor = conn.cursor()

    # Add quotation_factor to prices if missing
    cursor.execute("PRAGMA table_info(prices)")
    prices_cols = {row[1] for row in cursor.fetchall()}
    if "quotation_factor" not in prices_cols:
        logger.info("Migrating: adding quotation_factor column to prices table")
        cursor.execute(
            "ALTER TABLE prices ADD COLUMN quotation_factor INTEGER DEFAULT 1"
        )

    # CVM fundamentals tables are created by CREATE TABLE IF NOT EXISTS above.
    # Add listing_date / delisting_date to cvm_companies if missing (Task 03 migration).
    cursor.execute("PRAGMA table_info(cvm_companies)")
    cvm_companies_cols = {row[1] for row in cursor.fetchall()}
    if "listing_date" not in cvm_companies_cols:
        logger.info("Migrating: adding listing_date column to cvm_companies")
        cursor.execute("ALTER TABLE cvm_companies ADD COLUMN listing_date DATE")
    if "delisting_date" not in cvm_companies_cols:
        logger.info("Migrating: adding delisting_date column to cvm_companies")
        cursor.execute("ALTER TABLE cvm_companies ADD COLUMN delisting_date DATE")

    # company_isin_map: created via SCHEMA_COMPANY_ISIN_MAP above (CREATE TABLE IF NOT EXISTS).
    # Ensure indexes exist for existing DBs that were created before this table was added.
    cursor.execute(INDEX_COMPANY_ISIN_MAP_CNPJ)
    cursor.execute(INDEX_COMPANY_ISIN_MAP_ISIN)
    cursor.execute(INDEX_COMPANY_ISIN_MAP_TICKER)
    cursor.execute(INDEX_COMPANY_ISIN_MAP_PRIMARY)

    # fundamentals_monthly: created via SCHEMA_FUNDAMENTALS_MONTHLY (CREATE TABLE IF NOT EXISTS).
    # Safe to call on existing DBs that already have the table.
    cursor.execute(SCHEMA_FUNDAMENTALS_MONTHLY)
    cursor.execute(INDEX_FUNDAMENTALS_MONTHLY_TICKER)
    cursor.execute(INDEX_FUNDAMENTALS_MONTHLY_MONTH_END)


def _prepare_date(val) -> Optional[str]:
    """Convert date to standard string format."""
    if pd.isna(val):
        return None
    if isinstance(val, (date, datetime)):
        return val.strftime("%Y-%m-%d")
    return str(val)[:10]


def upsert_prices(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Upsert price records into the database."""
    if df.empty:
        logger.warning("No price records to upsert")
        return 0

    cursor = conn.cursor()

    sql = """
        INSERT INTO prices (ticker, isin_code, date, open, high, low, close, volume, quotation_factor)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date) DO UPDATE SET
            isin_code = excluded.isin_code,
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            quotation_factor = excluded.quotation_factor
    """

    records = []
    for _, row in df.iterrows():
        records.append(
            (
                row["ticker"],
                row.get("isin_code", "UNKNOWN"),
                _prepare_date(row["date"]),
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
                int(row.get("quotation_factor", 1) or 1),
            )
        )

    cursor.executemany(sql, records)
    conn.commit()

    logger.info(f"Upserted {len(records):,} price records")
    return len(records)


def upsert_corporate_actions(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Upsert corporate action records into the database."""
    if df.empty:
        logger.warning("No corporate action records to upsert")
        return 0

    cursor = conn.cursor()

    sql = """
        INSERT OR REPLACE INTO corporate_actions (isin_code, event_date, event_type, value, factor, source)
        VALUES (?, ?, ?, ?, ?, ?)
    """

    records = []
    for _, row in df.iterrows():
        records.append(
            (
                row["isin_code"],
                _prepare_date(row["event_date"]),
                row["event_type"],
                row.get("value"),
                row.get("factor"),
                row.get("source", "B3"),
            )
        )

    cursor.executemany(sql, records)
    conn.commit()

    logger.info(f"Upserted {len(records):,} corporate action records")
    return len(records)


def upsert_detected_splits(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Upsert detected split records into the database."""
    if df.empty:
        logger.warning("No detected split records to upsert")
        return 0

    cursor = conn.cursor()

    sql = """
        INSERT OR REPLACE INTO detected_splits (isin_code, ex_date, split_factor, description)
        VALUES (?, ?, ?, ?)
    """

    records = []
    for _, row in df.iterrows():
        records.append(
            (
                row["isin_code"],
                _prepare_date(row["ex_date"]),
                row["split_factor"],
                row.get("description"),
            )
        )

    cursor.executemany(sql, records)
    conn.commit()

    logger.info(f"Upserted {len(records):,} detected split records")
    return len(records)


def upsert_stock_actions(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Upsert stock action records (splits, reverse splits, bonuses) into the database."""
    if df.empty:
        logger.warning("No stock action records to upsert")
        return 0

    cursor = conn.cursor()

    sql = """
        INSERT OR REPLACE INTO stock_actions (isin_code, ex_date, action_type, factor, source)
        VALUES (?, ?, ?, ?, ?)
    """

    records = []
    for _, row in df.iterrows():
        records.append(
            (
                row["isin_code"],
                _prepare_date(row["ex_date"]),
                row["action_type"],
                row["factor"],
                row.get("source", "B3"),
            )
        )

    cursor.executemany(sql, records)
    conn.commit()

    logger.info(f"Upserted {len(records):,} stock action records")
    return len(records)


def update_adjusted_columns(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Update adjusted columns for existing price records."""
    if df.empty:
        return 0

    cursor = conn.cursor()

    sql = """
        UPDATE prices 
        SET split_adj_open = ?,
            split_adj_high = ?,
            split_adj_low = ?,
            split_adj_close = ?,
            adj_close = ?,
            volume = ?
        WHERE ticker = ? AND isin_code = ? AND date = ?
    """

    records = []
    for _, row in df.iterrows():
        records.append(
            (
                row.get("split_adj_open"),
                row.get("split_adj_high"),
                row.get("split_adj_low"),
                row.get("split_adj_close"),
                row.get("adj_close"),
                row.get("volume"),
                row["ticker"],
                row.get("isin_code", "UNKNOWN"),
                _prepare_date(row["date"]),
            )
        )

    cursor.executemany(sql, records)
    conn.commit()

    logger.info(f"Updated adjusted columns for {len(records):,} records")
    return len(records)


def get_all_tickers(conn: sqlite3.Connection) -> List[str]:
    """Get all unique ticker symbols from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker")
    return [row[0] for row in cursor.fetchall()]


def get_ticker_isin_map(conn: sqlite3.Connection) -> dict:
    """Get mapping of ticker to most recent ISIN code."""
    cursor = conn.cursor()
    # Find the most recent ISIN for each ticker
    cursor.execute("""
        SELECT p.ticker, p.isin_code
        FROM prices p
        INNER JOIN (
            SELECT ticker, MAX(date) as max_date
            FROM prices
            WHERE isin_code != 'UNKNOWN'
            GROUP BY ticker
        ) latest ON p.ticker = latest.ticker AND p.date = latest.max_date
        WHERE p.isin_code != 'UNKNOWN'
    """)
    return {row[0]: row[1] for row in cursor.fetchall()}


def get_tickers_by_company_code(
    conn: sqlite3.Connection, company_code: str
) -> List[str]:
    """Get all tickers that start with a given company code."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT ticker FROM prices WHERE ticker LIKE ? ORDER BY ticker",
        (f"{company_code}%",),
    )
    return [row[0] for row in cursor.fetchall()]


def get_trading_names(conn: sqlite3.Connection) -> List[str]:
    """Get all unique ticker root codes (first 4 characters) from prices."""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker")
    tickers = [row[0] for row in cursor.fetchall()]
    roots = sorted(set(t[:4] for t in tickers if len(t) >= 4))
    return roots


def get_prices_for_ticker(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """Get all price records for a specific ticker."""
    query = """
        SELECT ticker, isin_code, date, open, high, low, close, volume,
               quotation_factor,
               split_adj_open, split_adj_high, split_adj_low, split_adj_close, adj_close
        FROM prices
        WHERE ticker = ?
        ORDER BY date
    """
    return pd.read_sql_query(query, conn, params=(ticker,))


def get_prices_for_isin(conn: sqlite3.Connection, isin_code: str) -> pd.DataFrame:
    """Get all price records for a specific ISIN code."""
    query = """
        SELECT ticker, isin_code, date, open, high, low, close, volume,
               quotation_factor,
               split_adj_open, split_adj_high, split_adj_low, split_adj_close, adj_close
        FROM prices
        WHERE isin_code = ?
        ORDER BY date
    """
    return pd.read_sql_query(query, conn, params=(isin_code,))


def get_all_prices(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get all price records from the database."""
    query = """
        SELECT ticker, isin_code, date, open, high, low, close, volume,
               quotation_factor,
               split_adj_open, split_adj_high, split_adj_low, split_adj_close, adj_close
        FROM prices
        ORDER BY isin_code, date
    """
    return pd.read_sql_query(query, conn)


def get_all_corporate_actions(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get all corporate action records from the database."""
    query = """
        SELECT isin_code, event_date, event_type, value, factor, source
        FROM corporate_actions
        ORDER BY isin_code, event_date
    """
    return pd.read_sql_query(query, conn)


def get_all_detected_splits(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get all detected split records from the database."""
    query = """
        SELECT isin_code, ex_date, split_factor, description
        FROM detected_splits
        ORDER BY isin_code, ex_date
    """
    return pd.read_sql_query(query, conn)


def get_all_stock_actions(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get all stock action records from the database."""
    query = """
        SELECT isin_code, ex_date, action_type, factor, source
        FROM stock_actions
        ORDER BY isin_code, ex_date
    """
    return pd.read_sql_query(query, conn)


def upsert_skipped_events(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Upsert skipped event records (unrecognized B3 labels) into the database."""
    if df.empty:
        return 0

    cursor = conn.cursor()

    sql = """
        INSERT OR REPLACE INTO skipped_events (isin_code, event_date, label, factor, source, reason)
        VALUES (?, ?, ?, ?, ?, ?)
    """

    records = []
    for _, row in df.iterrows():
        records.append(
            (
                row["isin_code"],
                _prepare_date(row["event_date"]),
                row["label"],
                row.get("factor"),
                row.get("source", "B3"),
                row.get("reason"),
            )
        )

    cursor.executemany(sql, records)
    conn.commit()

    logger.info(f"Upserted {len(records):,} skipped event records")
    return len(records)


def record_fetch_failure(
    conn: sqlite3.Connection,
    company_code: str,
    endpoint: str,
    error_message: str,
) -> None:
    """Record an API fetch failure, incrementing retry_count if already exists."""
    cursor = conn.cursor()
    sql = """
        INSERT INTO fetch_failures (company_code, endpoint, error_message, retry_count, resolved)
        VALUES (?, ?, ?, 0, 0)
        ON CONFLICT(company_code, endpoint) DO UPDATE SET
            error_message = excluded.error_message,
            failed_at = CURRENT_TIMESTAMP,
            retry_count = fetch_failures.retry_count + 1,
            resolved = 0
    """
    cursor.execute(sql, (company_code, endpoint, error_message))
    conn.commit()


def resolve_fetch_failure(
    conn: sqlite3.Connection, company_code: str, endpoint: str
) -> None:
    """Mark a fetch failure as resolved."""
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE fetch_failures SET resolved = 1 WHERE company_code = ? AND endpoint = ?",
        (company_code, endpoint),
    )
    conn.commit()


def get_unresolved_failures(conn: sqlite3.Connection) -> list:
    """Return all unresolved fetch failures as a list of dicts."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT company_code, endpoint, error_message, failed_at, retry_count
        FROM fetch_failures
        WHERE resolved = 0
        ORDER BY company_code
        """
    )
    cols = ["company_code", "endpoint", "error_message", "failed_at", "retry_count"]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def get_summary_stats(conn: sqlite3.Connection) -> dict:
    """Get summary statistics from the database."""
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM prices")
    total_prices = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT ticker) FROM prices")
    total_tickers = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT isin_code) FROM prices")
    total_isins = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(date), MAX(date) FROM prices")
    date_range = cursor.fetchone()

    cursor.execute("SELECT COUNT(*) FROM corporate_actions")
    total_actions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM detected_splits")
    total_splits = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM stock_actions")
    total_stock_actions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM skipped_events")
    total_skipped = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM fetch_failures WHERE resolved = 0")
    unresolved_failures = cursor.fetchone()[0]

    # CVM fundamentals counts (tables may not exist in very old DBs — default to 0)
    try:
        cursor.execute("SELECT COUNT(*) FROM cvm_companies")
        total_cvm_companies = cursor.fetchone()[0]
    except Exception:
        total_cvm_companies = 0

    try:
        cursor.execute("SELECT COUNT(*) FROM cvm_filings")
        total_cvm_filings = cursor.fetchone()[0]
    except Exception:
        total_cvm_filings = 0

    try:
        cursor.execute("SELECT COUNT(*) FROM fundamentals_pit")
        total_fundamentals_pit = cursor.fetchone()[0]
    except Exception:
        total_fundamentals_pit = 0

    return {
        "total_prices": total_prices,
        "total_tickers": total_tickers,
        "total_isins": total_isins,
        "date_range": (date_range[0], date_range[1]),
        "total_corporate_actions": total_actions,
        "total_detected_splits": total_splits,
        "total_stock_actions": total_stock_actions,
        "total_skipped_events": total_skipped,
        "unresolved_fetch_failures": unresolved_failures,
        "total_cvm_companies": total_cvm_companies,
        "total_cvm_filings": total_cvm_filings,
        "total_fundamentals_pit": total_fundamentals_pit,
    }
