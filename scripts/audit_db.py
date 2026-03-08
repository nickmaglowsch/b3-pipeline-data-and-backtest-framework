#!/usr/bin/env python
"""
B3 Database Audit Script
========================
Scans b3_market_data.sqlite and reports data quality metrics, anomalies,
and gaps across all tables. Outputs a human-readable report to stdout.

Usage:
    python scripts/audit_db.py [--db path/to/db.sqlite]
"""

import argparse
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

OK   = f"{GREEN}OK{RESET}"
WARN = f"{YELLOW}WARN{RESET}"
ERR  = f"{RED}ERROR{RESET}"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _status(val: int, warn_threshold: int = 1, error_threshold: Optional[int] = None) -> str:
    if error_threshold is not None and val >= error_threshold:
        return ERR
    if val >= warn_threshold:
        return WARN
    return OK


def section(title: str) -> None:
    width = 72
    print(f"\n{BOLD}{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")


def row(label: str, value, status: str = "") -> None:
    label_width = 50
    status_str = f"  [{status}]" if status else ""
    print(f"  {label:<{label_width}} {BOLD}{value}{RESET}{status_str}")


def subheader(title: str) -> None:
    print(f"\n  {DIM}{title}{RESET}")


def scalar(conn: sqlite3.Connection, sql: str, *params) -> int:
    cur = conn.execute(sql, params)
    r = cur.fetchone()
    return r[0] if r and r[0] is not None else 0


# ── Check functions ───────────────────────────────────────────────────────────

def audit_overview(conn: sqlite3.Connection) -> None:
    section("DATABASE OVERVIEW")

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )]
    row("Tables found", ", ".join(tables))
    print()

    for t in tables:
        count = scalar(conn, f"SELECT COUNT(*) FROM {t}")
        status = WARN if count == 0 else OK
        row(f"  {t}", f"{count:,} rows", status)


def audit_prices(conn: sqlite3.Connection) -> None:
    section("PRICES TABLE")

    total = scalar(conn, "SELECT COUNT(*) FROM prices")
    row("Total rows", f"{total:,}", WARN if total == 0 else OK)
    if total == 0:
        print(f"  {DIM}(table empty — run the pipeline first){RESET}")
        return

    # Basic coverage
    subheader("Coverage")
    tickers   = scalar(conn, "SELECT COUNT(DISTINCT ticker) FROM prices")
    isins     = scalar(conn, "SELECT COUNT(DISTINCT isin_code) FROM prices")
    min_date  = conn.execute("SELECT MIN(date) FROM prices").fetchone()[0]
    max_date  = conn.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    row("Unique tickers", f"{tickers:,}")
    row("Unique ISINs", f"{isins:,}")
    row("Date range", f"{min_date}  →  {max_date}")

    # Adjusted columns populated?
    subheader("Adjustment column coverage")
    null_split_adj  = scalar(conn, "SELECT COUNT(*) FROM prices WHERE split_adj_close IS NULL")
    null_adj_close  = scalar(conn, "SELECT COUNT(*) FROM prices WHERE adj_close IS NULL")
    pct_split = null_split_adj / total * 100
    pct_adj   = null_adj_close / total * 100
    row("Rows with NULL split_adj_close", f"{null_split_adj:,}  ({pct_split:.1f}%)",
        _status(null_split_adj, 1))
    row("Rows with NULL adj_close",       f"{null_adj_close:,}  ({pct_adj:.1f}%)",
        _status(null_adj_close, 1))

    # OHLC sanity
    subheader("OHLC sanity")
    zero_close   = scalar(conn, "SELECT COUNT(*) FROM prices WHERE close <= 0")
    zero_vol     = scalar(conn, "SELECT COUNT(*) FROM prices WHERE volume <= 0")
    high_lt_low  = scalar(conn, "SELECT COUNT(*) FROM prices WHERE high < low")
    close_gt_hi  = scalar(conn, "SELECT COUNT(*) FROM prices WHERE close > high * 1.001")
    close_lt_low = scalar(conn, "SELECT COUNT(*) FROM prices WHERE close < low  * 0.999")
    open_zero    = scalar(conn, "SELECT COUNT(*) FROM prices WHERE open <= 0")
    row("Rows with close ≤ 0",    f"{zero_close:,}",   _status(zero_close, 1))
    row("Rows with volume ≤ 0",   f"{zero_vol:,}",     _status(zero_vol, 1))
    row("Rows with high < low",   f"{high_lt_low:,}",  _status(high_lt_low, 1))
    row("Rows with close > high", f"{close_gt_hi:,}",  _status(close_gt_hi, 1))
    row("Rows with close < low",  f"{close_lt_low:,}", _status(close_lt_low, 1))
    row("Rows with open ≤ 0",     f"{open_zero:,}",    _status(open_zero, 1))

    # adj_close extreme ratios
    subheader("adj_close extremes (adj_close / close ratio)")
    extreme_high = scalar(conn, """
        SELECT COUNT(*) FROM prices
        WHERE adj_close IS NOT NULL AND close > 0
          AND (adj_close / close) > 100000
    """)
    extreme_low = scalar(conn, """
        SELECT COUNT(*) FROM prices
        WHERE adj_close IS NOT NULL AND close > 0
          AND (adj_close / close) < 0.00001
    """)
    negative_adj = scalar(conn, "SELECT COUNT(*) FROM prices WHERE adj_close IS NOT NULL AND adj_close < 0")
    row("Rows with adj_close/close > 100,000×", f"{extreme_high:,}", _status(extreme_high, 1))
    row("Rows with adj_close/close < 0.00001×", f"{extreme_low:,}",  _status(extreme_low, 1))
    row("Rows with adj_close < 0",              f"{negative_adj:,}", _status(negative_adj, 1))

    # Detect potential missing splits (>80% overnight drop not explained by known split)
    subheader("Potential missing splits (overnight price drop > 60%)")
    suspicious = conn.execute("""
        WITH ranked AS (
            SELECT ticker, date, close,
                   LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS prev_close
            FROM prices
            WHERE close > 0
        )
        SELECT ticker, date, prev_close, close,
               ROUND(close * 1.0 / prev_close, 4) AS ratio
        FROM ranked
        WHERE prev_close > 0
          AND (close / prev_close) < 0.40
          AND (close / prev_close) > 0
        ORDER BY ratio
        LIMIT 20
    """).fetchall()
    row("Suspicious overnight drops (top 20 shown)", f"{len(suspicious):,}",
        _status(len(suspicious), 1))
    if suspicious:
        print()
        print(f"  {'Ticker':<8} {'Date':<12} {'Prev Close':>12} {'Close':>10} {'Ratio':>8}")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")
        for ticker, dt, prev, curr, ratio in suspicious:
            flag = f"{RED}▲{RESET}" if ratio < 0.10 else f"{YELLOW}▲{RESET}"
            print(f"  {ticker:<8} {dt:<12} {prev:>12.4f} {curr:>10.4f} {ratio:>8.4f} {flag}")

    # Duplicate (ticker, date) rows
    subheader("Duplicates")
    dupes = scalar(conn, """
        SELECT COUNT(*) FROM (
            SELECT ticker, date FROM prices
            GROUP BY ticker, date HAVING COUNT(*) > 1
        )
    """)
    row("Duplicate (ticker, date) pairs", f"{dupes:,}", _status(dupes, 1))

    # Unknown ISINs
    unknown_isin = scalar(conn, "SELECT COUNT(*) FROM prices WHERE isin_code = 'UNKNOWN' OR isin_code IS NULL")
    row("Rows with UNKNOWN/NULL ISIN", f"{unknown_isin:,}", _status(unknown_isin, 1))

    # Tickers with very few trading days
    subheader("Ticker coverage quality")
    thin_tickers = conn.execute("""
        SELECT ticker, COUNT(*) AS days FROM prices
        GROUP BY ticker HAVING days < 10
        ORDER BY days
    """).fetchall()
    row("Tickers with < 10 trading days", f"{len(thin_tickers):,}", _status(len(thin_tickers), 1))
    if thin_tickers[:10]:
        print(f"    {DIM}" + ", ".join(f"{t}({d})" for t, d in thin_tickers[:10]) + RESET)

    # Per-year trading day counts (flag years with unusually few days)
    subheader("Trading days per calendar year (sanity check)")
    year_counts = conn.execute("""
        SELECT SUBSTR(date, 1, 4) AS yr,
               COUNT(DISTINCT date) AS trading_days,
               COUNT(DISTINCT ticker) AS active_tickers
        FROM prices
        GROUP BY yr ORDER BY yr
    """).fetchall()
    print(f"  {'Year':<6} {'Trading Days':>14} {'Active Tickers':>16}")
    print(f"  {'-'*6} {'-'*14} {'-'*16}")
    for yr, td, at in year_counts:
        flag = f"  {YELLOW}⚠ low{RESET}" if td < 200 else ""
        print(f"  {yr:<6} {td:>14,} {at:>16,}{flag}")


def audit_stock_actions(conn: sqlite3.Connection) -> None:
    section("STOCK ACTIONS TABLE")

    total = scalar(conn, "SELECT COUNT(*) FROM stock_actions")
    row("Total rows", f"{total:,}", WARN if total == 0 else OK)
    if total == 0:
        return

    by_type = conn.execute("""
        SELECT action_type, source, COUNT(*) AS n
        FROM stock_actions
        GROUP BY action_type, source ORDER BY action_type, source
    """).fetchall()
    subheader("Breakdown by action_type / source")
    for atype, src, n in by_type:
        row(f"  {atype} / {src}", f"{n:,}")

    bad_factors = scalar(conn, "SELECT COUNT(*) FROM stock_actions WHERE factor <= 0 OR factor IS NULL")
    extreme_factors = scalar(conn, "SELECT COUNT(*) FROM stock_actions WHERE factor > 10000 OR factor < 0.0001")
    row("Rows with factor ≤ 0 or NULL", f"{bad_factors:,}", _status(bad_factors, 1))
    row("Rows with extreme factor (>10000 or <0.0001)", f"{extreme_factors:,}", _status(extreme_factors, 1))

    dupes = scalar(conn, """
        SELECT COUNT(*) FROM (
            SELECT isin_code, ex_date, action_type FROM stock_actions
            GROUP BY isin_code, ex_date, action_type HAVING COUNT(*) > 1
        )
    """)
    row("Duplicate (isin, ex_date, action_type)", f"{dupes:,}", _status(dupes, 1))


def audit_corporate_actions(conn: sqlite3.Connection) -> None:
    section("CORPORATE ACTIONS TABLE")

    total = scalar(conn, "SELECT COUNT(*) FROM corporate_actions")
    row("Total rows", f"{total:,}", WARN if total == 0 else OK)
    if total == 0:
        return

    by_type = conn.execute("""
        SELECT event_type, COUNT(*) AS n
        FROM corporate_actions
        GROUP BY event_type ORDER BY n DESC
    """).fetchall()
    subheader("Breakdown by event_type")
    for etype, n in by_type:
        row(f"  {etype}", f"{n:,}")

    bad_val = scalar(conn, "SELECT COUNT(*) FROM corporate_actions WHERE value <= 0 AND value IS NOT NULL")
    null_val = scalar(conn, "SELECT COUNT(*) FROM corporate_actions WHERE value IS NULL AND factor IS NULL")
    row("Rows with value ≤ 0 (non-NULL)", f"{bad_val:,}", _status(bad_val, 1))
    row("Rows with both value and factor NULL", f"{null_val:,}", _status(null_val, 1))

    # Orphaned ISINs (corporate action for ISIN not in prices)
    orphaned = scalar(conn, """
        SELECT COUNT(DISTINCT ca.isin_code) FROM corporate_actions ca
        WHERE NOT EXISTS (SELECT 1 FROM prices p WHERE p.isin_code = ca.isin_code)
    """)
    row("Distinct ISINs in corp_actions but not in prices", f"{orphaned:,}", _status(orphaned, 10))


def audit_fundamentals(conn: sqlite3.Connection) -> None:
    section("FUNDAMENTALS (fundamentals_pit)")

    total = scalar(conn, "SELECT COUNT(*) FROM fundamentals_pit")
    row("Total filings", f"{total:,}", WARN if total == 0 else OK)
    if total == 0:
        return

    tickers = scalar(conn, "SELECT COUNT(DISTINCT ticker) FROM fundamentals_pit")
    row("Distinct tickers with filings", f"{tickers:,}")

    by_doctype = conn.execute("""
        SELECT doc_type, COUNT(*) AS n FROM fundamentals_pit
        GROUP BY doc_type ORDER BY doc_type
    """).fetchall()
    subheader("Breakdown by doc_type")
    for dt, n in by_doctype:
        row(f"  {dt}", f"{n:,}")

    subheader("Filing coverage by fiscal year")
    by_year = conn.execute("""
        SELECT fiscal_year, COUNT(*) AS n, COUNT(DISTINCT ticker) AS tickers
        FROM fundamentals_pit
        GROUP BY fiscal_year ORDER BY fiscal_year
    """).fetchall()
    print(f"  {'Year':<6} {'Filings':>10} {'Tickers':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10}")
    for yr, n, t in by_year:
        print(f"  {str(yr) if yr else 'NULL':<6} {n:>10,} {t:>10,}")

    subheader("NULL fundamental values")
    for col in ["revenue", "net_income", "ebitda", "total_assets", "equity", "shares_outstanding"]:
        null_count = scalar(conn, f"SELECT COUNT(*) FROM fundamentals_pit WHERE {col} IS NULL")
        pct = null_count / total * 100
        row(f"  NULL {col}", f"{null_count:,}  ({pct:.1f}%)", _status(null_count, int(total * 0.5), int(total * 0.9)))

    subheader("Suspicious fundamental values")
    neg_assets = scalar(conn, "SELECT COUNT(*) FROM fundamentals_pit WHERE total_assets IS NOT NULL AND total_assets < 0")
    zero_shares = scalar(conn, "SELECT COUNT(*) FROM fundamentals_pit WHERE shares_outstanding IS NOT NULL AND shares_outstanding <= 0")
    extreme_rev = scalar(conn, """
        SELECT COUNT(*) FROM fundamentals_pit
        WHERE revenue IS NOT NULL AND ABS(revenue) > 1e12
    """)
    row("Rows with negative total_assets", f"{neg_assets:,}", _status(neg_assets, 1))
    row("Rows with shares_outstanding ≤ 0", f"{zero_shares:,}", _status(zero_shares, 1))
    row("Rows with |revenue| > 1 trillion BRL", f"{extreme_rev:,}", _status(extreme_rev, 1))

    subheader("Duplicate filings (same ticker + period_end + doc_type + version)")
    dupes = scalar(conn, """
        SELECT COUNT(*) FROM (
            SELECT ticker, period_end, doc_type, filing_version FROM fundamentals_pit
            GROUP BY ticker, period_end, doc_type, filing_version HAVING COUNT(*) > 1
        )
    """)
    row("Duplicate (ticker, period, doc_type, version)", f"{dupes:,}", _status(dupes, 1))


def audit_cvm_companies(conn: sqlite3.Connection) -> None:
    section("CVM COMPANIES")

    total = scalar(conn, "SELECT COUNT(*) FROM cvm_companies")
    row("Total companies", f"{total:,}", WARN if total == 0 else OK)
    if total == 0:
        return

    with_ticker     = scalar(conn, "SELECT COUNT(*) FROM cvm_companies WHERE ticker IS NOT NULL AND ticker != ''")
    delisted        = scalar(conn, "SELECT COUNT(*) FROM cvm_companies WHERE delisting_date IS NOT NULL")
    no_listing_date = scalar(conn, "SELECT COUNT(*) FROM cvm_companies WHERE listing_date IS NULL")
    row("Companies with a ticker",            f"{with_ticker:,}")
    row("Delisted companies",                 f"{delisted:,}")
    row("Companies missing listing_date",     f"{no_listing_date:,}", _status(no_listing_date, 1))

    # Tickers in prices but not in cvm_companies (no survivorship info)
    prices_total = scalar(conn, "SELECT COUNT(*) FROM prices")
    if prices_total > 0:
        subheader("Price tickers without CVM company record")
        unmapped = conn.execute("""
            SELECT DISTINCT p.ticker FROM prices p
            WHERE NOT EXISTS (
                SELECT 1 FROM cvm_companies c WHERE c.ticker = p.ticker
            )
            ORDER BY p.ticker
            LIMIT 30
        """).fetchall()
        count_unmapped = scalar(conn, """
            SELECT COUNT(DISTINCT p.ticker) FROM prices p
            WHERE NOT EXISTS (
                SELECT 1 FROM cvm_companies c WHERE c.ticker = p.ticker
            )
        """)
        row("Tickers in prices with no CVM record", f"{count_unmapped:,}",
            _status(count_unmapped, 10, 100))
        if unmapped:
            print(f"    {DIM}" + " ".join(r[0] for r in unmapped) + RESET)

    # Tickers in fundamentals_pit but not in cvm_companies
    fund_total = scalar(conn, "SELECT COUNT(*) FROM fundamentals_pit")
    if fund_total > 0:
        subheader("Fundamentals tickers without CVM company record")
        unmapped_fund = scalar(conn, """
            SELECT COUNT(DISTINCT f.ticker) FROM fundamentals_pit f
            WHERE f.ticker IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM cvm_companies c WHERE c.ticker = f.ticker
              )
        """)
        row("Tickers in fundamentals_pit with no CVM record", f"{unmapped_fund:,}",
            _status(unmapped_fund, 1))


def audit_fetch_failures(conn: sqlite3.Connection) -> None:
    section("FETCH FAILURES")

    total = scalar(conn, "SELECT COUNT(*) FROM fetch_failures")
    unresolved = scalar(conn, "SELECT COUNT(*) FROM fetch_failures WHERE resolved = 0")
    row("Total fetch failures logged", f"{total:,}")
    row("Unresolved failures", f"{unresolved:,}", _status(unresolved, 1))

    if unresolved > 0:
        by_endpoint = conn.execute("""
            SELECT endpoint, COUNT(*) AS n, MAX(retry_count) AS max_retries
            FROM fetch_failures WHERE resolved = 0
            GROUP BY endpoint ORDER BY n DESC LIMIT 10
        """).fetchall()
        subheader("Unresolved failures by endpoint (top 10)")
        for ep, n, retries in by_endpoint:
            row(f"  {ep}", f"{n:,}  (max retries: {retries})", _status(n, 1))


def audit_cross_table(conn: sqlite3.Connection) -> None:
    section("CROSS-TABLE CONSISTENCY")

    prices_total = scalar(conn, "SELECT COUNT(*) FROM prices")
    sa_total     = scalar(conn, "SELECT COUNT(*) FROM stock_actions")
    fund_total   = scalar(conn, "SELECT COUNT(*) FROM fundamentals_pit")

    if prices_total == 0 or sa_total == 0:
        print(f"  {DIM}(skipping — prices or stock_actions empty){RESET}")
        return

    # ISINs in stock_actions with no prices
    orphan_sa = scalar(conn, """
        SELECT COUNT(DISTINCT sa.isin_code) FROM stock_actions sa
        WHERE NOT EXISTS (SELECT 1 FROM prices p WHERE p.isin_code = sa.isin_code)
    """)
    row("ISINs in stock_actions with no price rows", f"{orphan_sa:,}", _status(orphan_sa, 5))

    # Stock action dates before the earliest price date for that ISIN
    future_sa = scalar(conn, """
        SELECT COUNT(*) FROM stock_actions sa
        WHERE sa.ex_date < (
            SELECT MIN(p.date) FROM prices p WHERE p.isin_code = sa.isin_code
        )
    """)
    row("Stock actions before earliest price date (ISIN)", f"{future_sa:,}", _status(future_sa, 1))

    # Stock action dates after the latest price date for that ISIN
    after_sa = scalar(conn, """
        SELECT COUNT(*) FROM stock_actions sa
        WHERE sa.ex_date > (
            SELECT MAX(p.date) FROM prices p WHERE p.isin_code = sa.isin_code
        )
    """)
    row("Stock actions after latest price date (ISIN)", f"{after_sa:,}", _status(after_sa, 1))

    if fund_total > 0 and prices_total > 0:
        # Tickers in both prices and fundamentals_pit
        overlap = scalar(conn, """
            SELECT COUNT(DISTINCT f.ticker) FROM fundamentals_pit f
            WHERE EXISTS (SELECT 1 FROM prices p WHERE p.ticker = f.ticker)
        """)
        fund_tickers = scalar(conn, "SELECT COUNT(DISTINCT ticker) FROM fundamentals_pit WHERE ticker IS NOT NULL")
        price_tickers = scalar(conn, "SELECT COUNT(DISTINCT ticker) FROM prices")
        row("Tickers present in both prices and fundamentals",
            f"{overlap:,}  ({overlap/max(fund_tickers,1)*100:.0f}% of fund, "
            f"{overlap/max(price_tickers,1)*100:.0f}% of prices)")


def audit_summary(issues: list) -> None:
    section("SUMMARY")
    if not issues:
        print(f"  {GREEN}{BOLD}No issues detected.{RESET}")
    else:
        print(f"  {YELLOW}Issues detected ({len(issues)}):{RESET}\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i:>3}. {issue}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit B3 market data SQLite database")
    parser.add_argument(
        "--db",
        default=str(Path(__file__).parent.parent / "b3_market_data.sqlite"),
        help="Path to SQLite database",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"{RED}Database not found: {db_path}{RESET}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    print(f"\n{BOLD}B3 Market Data — Database Audit{RESET}")
    print(f"  Database : {db_path}")
    print(f"  Size     : {db_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Run at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    audit_overview(conn)
    audit_prices(conn)
    audit_stock_actions(conn)
    audit_corporate_actions(conn)
    audit_fundamentals(conn)
    audit_cvm_companies(conn)
    audit_fetch_failures(conn)
    audit_cross_table(conn)

    print(f"\n{DIM}Legend: [{GREEN}OK{RESET}{DIM}] within expected range  "
          f"[{YELLOW}WARN{RESET}{DIM}] worth investigating  "
          f"[{RED}ERROR{RESET}{DIM}] likely corrupts results{RESET}\n")

    conn.close()


if __name__ == "__main__":
    main()
