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
    date DATE NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
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
    ticker TEXT NOT NULL,
    event_date DATE NOT NULL,
    event_type TEXT NOT NULL,
    value REAL,
    isin_code TEXT,
    factor REAL,
    source TEXT DEFAULT 'B3',
    PRIMARY KEY (ticker, event_date, event_type)
);
"""

SCHEMA_STOCK_ACTIONS = """
CREATE TABLE IF NOT EXISTS stock_actions (
    ticker TEXT NOT NULL,
    ex_date DATE NOT NULL,
    action_type TEXT NOT NULL,
    factor REAL NOT NULL,
    isin_code TEXT,
    source TEXT DEFAULT 'B3',
    PRIMARY KEY (ticker, ex_date, action_type)
);
"""

SCHEMA_DETECTED_SPLITS = """
CREATE TABLE IF NOT EXISTS detected_splits (
    ticker TEXT NOT NULL,
    ex_date DATE NOT NULL,
    split_factor REAL NOT NULL,
    description TEXT,
    PRIMARY KEY (ticker, ex_date)
);
"""

INDEX_PRICES_DATE = "CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);"
INDEX_PRICES_TICKER = "CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Get a SQLite database connection.

    Args:
        db_path: Path to database file (default: config.DB_PATH)

    Returns:
        SQLite connection object
    """
    if db_path is None:
        db_path = config.DB_PATH

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    return conn


def init_db(conn: sqlite3.Connection, rebuild: bool = False) -> None:
    """
    Initialize the database schema.

    Args:
        conn: SQLite connection
        rebuild: If True, drop all tables before creating
    """
    cursor = conn.cursor()

    if rebuild:
        logger.info("Dropping existing tables...")
        cursor.execute("DROP TABLE IF EXISTS prices")
        cursor.execute("DROP TABLE IF EXISTS corporate_actions")
        cursor.execute("DROP TABLE IF EXISTS stock_actions")
        cursor.execute("DROP TABLE IF EXISTS detected_splits")

    logger.info("Creating database schema...")
    cursor.execute(SCHEMA_PRICES)
    cursor.execute(SCHEMA_CORPORATE_ACTIONS)
    cursor.execute(SCHEMA_STOCK_ACTIONS)
    cursor.execute(SCHEMA_DETECTED_SPLITS)
    cursor.execute(INDEX_PRICES_DATE)
    cursor.execute(INDEX_PRICES_TICKER)

    conn.commit()
    logger.info("Database schema initialized")


def _prepare_date(value):
    """Convert date value to ISO string for SQLite."""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return (
            value.isoformat() if isinstance(value, date) else value.date().isoformat()
        )
    return str(value)


def upsert_prices(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """
    Upsert price records into the database.

    Uses INSERT OR REPLACE for idempotency.

    Args:
        conn: SQLite connection
        df: DataFrame with price data

    Returns:
        Number of records upserted
    """
    if df.empty:
        logger.warning("No price records to upsert")
        return 0

    cursor = conn.cursor()

    columns = ["ticker", "date", "open", "high", "low", "close", "volume"]
    placeholders = ", ".join(["?" for _ in columns])
    sql = f"""
        INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, volume)
        VALUES ({placeholders})
    """

    records = []
    for _, row in df.iterrows():
        records.append(
            (
                row["ticker"],
                _prepare_date(row["date"]),
                row.get("open"),
                row.get("high"),
                row.get("low"),
                row.get("close"),
                row.get("volume"),
            )
        )

    cursor.executemany(sql, records)
    conn.commit()

    logger.info(f"Upserted {len(records):,} price records")
    return len(records)


def upsert_corporate_actions(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """
    Upsert corporate action records into the database.

    Args:
        conn: SQLite connection
        df: DataFrame with corporate action data

    Returns:
        Number of records upserted
    """
    if df.empty:
        logger.warning("No corporate action records to upsert")
        return 0

    cursor = conn.cursor()

    sql = """
        INSERT OR REPLACE INTO corporate_actions (ticker, event_date, event_type, value, isin_code, factor, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    records = []
    for _, row in df.iterrows():
        records.append(
            (
                row["ticker"],
                _prepare_date(row["event_date"]),
                row["event_type"],
                row.get("value"),
                row.get("isin_code"),
                row.get("factor"),
                row.get("source", "B3"),
            )
        )

    cursor.executemany(sql, records)
    conn.commit()

    logger.info(f"Upserted {len(records):,} corporate action records")
    return len(records)


def upsert_detected_splits(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """
    Upsert detected split records into the database.

    Args:
        conn: SQLite connection
        df: DataFrame with detected split data

    Returns:
        Number of records upserted
    """
    if df.empty:
        logger.warning("No detected split records to upsert")
        return 0

    cursor = conn.cursor()

    sql = """
        INSERT OR REPLACE INTO detected_splits (ticker, ex_date, split_factor, description)
        VALUES (?, ?, ?, ?)
    """

    records = []
    for _, row in df.iterrows():
        records.append(
            (
                row["ticker"],
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
    """
    Upsert stock action records (splits, reverse splits, bonuses) into the database.

    Args:
        conn: SQLite connection
        df: DataFrame with stock action data

    Returns:
        Number of records upserted
    """
    if df.empty:
        logger.warning("No stock action records to upsert")
        return 0

    cursor = conn.cursor()

    sql = """
        INSERT OR REPLACE INTO stock_actions (ticker, ex_date, action_type, factor, isin_code, source)
        VALUES (?, ?, ?, ?, ?, ?)
    """

    records = []
    for _, row in df.iterrows():
        records.append(
            (
                row["ticker"],
                _prepare_date(row["ex_date"]),
                row["action_type"],
                row["factor"],
                row.get("isin_code"),
                row.get("source", "B3"),
            )
        )

    cursor.executemany(sql, records)
    conn.commit()

    logger.info(f"Upserted {len(records):,} stock action records")
    return len(records)


def update_adjusted_columns(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """
    Update adjusted price columns in the database.

    Args:
        conn: SQLite connection
        df: DataFrame with adjusted price data

    Returns:
        Number of records updated
    """
    if df.empty:
        logger.warning("No adjusted records to update")
        return 0

    cursor = conn.cursor()

    sql = """
        UPDATE prices
        SET split_adj_open = ?,
            split_adj_high = ?,
            split_adj_low = ?,
            split_adj_close = ?,
            adj_close = ?
        WHERE ticker = ? AND date = ?
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
                row["ticker"],
                _prepare_date(row["date"]),
            )
        )

    cursor.executemany(sql, records)
    conn.commit()

    logger.info(f"Updated {len(records):,} adjusted price records")
    return len(records)


def get_all_tickers(conn: sqlite3.Connection) -> List[str]:
    """
    Get all unique tickers from the prices table.

    Args:
        conn: SQLite connection

    Returns:
        Sorted list of ticker symbols
    """
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker")
    return [row[0] for row in cursor.fetchall()]


def get_prices_for_ticker(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """
    Get all price records for a single ticker.

    Args:
        conn: SQLite connection
        ticker: Ticker symbol

    Returns:
        DataFrame with price data
    """
    query = """
        SELECT ticker, date, open, high, low, close, volume,
               split_adj_open, split_adj_high, split_adj_low, split_adj_close, adj_close
        FROM prices
        WHERE ticker = ?
        ORDER BY date
    """
    return pd.read_sql_query(query, conn, params=(ticker,))


def get_all_prices(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Get all price records from the database.

    Args:
        conn: SQLite connection

    Returns:
        DataFrame with all price data
    """
    query = """
        SELECT ticker, date, open, high, low, close, volume,
               split_adj_open, split_adj_high, split_adj_low, split_adj_close, adj_close
        FROM prices
        ORDER BY ticker, date
    """
    return pd.read_sql_query(query, conn)


def get_all_corporate_actions(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Get all corporate action records from the database.

    Args:
        conn: SQLite connection

    Returns:
        DataFrame with corporate action data
    """
    query = """
        SELECT ticker, event_date, event_type, value
        FROM corporate_actions
        ORDER BY ticker, event_date
    """
    return pd.read_sql_query(query, conn)


def get_all_detected_splits(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Get all detected split records from the database.

    Args:
        conn: SQLite connection

    Returns:
        DataFrame with detected split data
    """
    query = """
        SELECT ticker, ex_date, split_factor, description
        FROM detected_splits
        ORDER BY ticker, ex_date
    """
    return pd.read_sql_query(query, conn)


def get_all_stock_actions(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Get all stock action records from the database.

    Args:
        conn: SQLite connection

    Returns:
        DataFrame with stock action data
    """
    query = """
        SELECT ticker, ex_date, action_type, factor, isin_code, source
        FROM stock_actions
        ORDER BY ticker, ex_date
    """
    return pd.read_sql_query(query, conn)


def get_tickers_by_company_code(
    conn: sqlite3.Connection, company_code: str
) -> List[str]:
    """
    Get all tickers that start with a given company code.

    Args:
        conn: SQLite connection
        company_code: The company code (e.g., 'PETR' from B3 response)

    Returns:
        List of ticker symbols matching the company code
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT ticker FROM prices WHERE ticker LIKE ? ORDER BY ticker",
        (f"{company_code}%",),
    )
    return [row[0] for row in cursor.fetchall()]


def get_trading_names(conn: sqlite3.Connection) -> List[str]:
    """
    Get all unique ticker root codes (first 4 characters) from prices.

    Args:
        conn: SQLite connection

    Returns:
        Sorted list of unique 4-char ticker roots
    """
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker")
    tickers = [row[0] for row in cursor.fetchall()]
    roots = sorted(set(t[:4] for t in tickers if len(t) >= 4))
    return roots


def get_summary_stats(conn: sqlite3.Connection) -> dict:
    """
    Get summary statistics from the database.

    Args:
        conn: SQLite connection

    Returns:
        Dictionary with summary statistics
    """
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM prices")
    total_prices = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT ticker) FROM prices")
    total_tickers = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(date), MAX(date) FROM prices")
    date_range = cursor.fetchone()

    cursor.execute("SELECT COUNT(*) FROM corporate_actions")
    total_actions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM detected_splits")
    total_splits = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM stock_actions")
    total_stock_actions = cursor.fetchone()[0]

    return {
        "total_prices": total_prices,
        "total_tickers": total_tickers,
        "date_range": (date_range[0], date_range[1]),
        "total_corporate_actions": total_actions,
        "total_detected_splits": total_splits,
        "total_stock_actions": total_stock_actions,
    }
