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

INDEX_PRICES_DATE = "CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);"
INDEX_PRICES_TICKER = "CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);"
INDEX_PRICES_ISIN = "CREATE INDEX IF NOT EXISTS idx_prices_isin ON prices(isin_code);"


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
    cursor.execute(INDEX_PRICES_ISIN)

    conn.commit()
    logger.info("Database schema initialized")


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
        INSERT INTO prices (ticker, isin_code, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date) DO UPDATE SET
            isin_code = excluded.isin_code,
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume
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

    return {
        "total_prices": total_prices,
        "total_tickers": total_tickers,
        "total_isins": total_isins,
        "date_range": (date_range[0], date_range[1]),
        "total_corporate_actions": total_actions,
        "total_detected_splits": total_splits,
        "total_stock_actions": total_stock_actions,
    }
