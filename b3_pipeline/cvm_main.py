#!/usr/bin/env python3
"""
CVM Fundamentals Data Pipeline

Downloads and processes CVM financial statements (DFP, ITR, FRE) for
point-in-time fundamental analysis.

Usage:
    python -m b3_pipeline.cvm_main                              # Run pipeline (default years)
    python -m b3_pipeline.cvm_main --start-year 2020            # From 2020
    python -m b3_pipeline.cvm_main --start-year 2023 --end-year 2023
    python -m b3_pipeline.cvm_main --rebuild                    # Drop and recreate tables
    python -m b3_pipeline.cvm_main --skip-ratios                # Skip valuation ratio computation
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

from . import config, cvm_downloader, cvm_parser, cvm_storage, storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Valuation ratio materialization ───────────────────────────────────────────

def materialize_valuation_ratios(conn: sqlite3.Connection) -> int:
    """
    Join fundamentals_pit with prices to compute P/E, P/B, and EV/EBITDA.

    Uses the closing price on filing_date (or the nearest prior trading day).
    Updates pe_ratio, pb_ratio, ev_ebitda columns in-place.
    Returns the number of rows updated.

    Annualization factor for ITR:
        Q1 -> 4x, Q2 -> 2x, Q3 -> 1.33x (12 / (quarter * 3))
    DFP: factor = 1 (already annual).
    """
    cursor = conn.cursor()

    # Load all rows that have the required data but missing ratios
    cursor.execute("""
        SELECT
            filing_id, cnpj, ticker, period_end, filing_date, filing_version,
            doc_type, quarter,
            net_income, equity, ebitda, net_debt, shares_outstanding
        FROM fundamentals_pit
        WHERE shares_outstanding IS NOT NULL
          AND ticker IS NOT NULL
          AND (pe_ratio IS NULL OR pb_ratio IS NULL OR ev_ebitda IS NULL)
    """)
    rows = cursor.fetchall()
    cols = [
        "filing_id", "cnpj", "ticker", "period_end", "filing_date", "filing_version",
        "doc_type", "quarter",
        "net_income", "equity", "ebitda", "net_debt", "shares_outstanding",
    ]
    df = pd.DataFrame(rows, columns=cols)

    if df.empty:
        logger.info("No fundamentals rows eligible for valuation ratio computation")
        return 0

    updates = []
    for _, row in df.iterrows():
        ticker_root = str(row["ticker"])
        filing_date = str(row["filing_date"])

        # Find close price on or before filing_date for the common share (suffix 3/4/5/6)
        cursor.execute(
            """
            SELECT close FROM prices
            WHERE ticker LIKE ?
              AND date <= ?
              AND LENGTH(ticker) = 5
              AND SUBSTR(ticker, 5, 1) IN ('3', '4', '5', '6')
            ORDER BY date DESC
            LIMIT 1
            """,
            (f"{ticker_root}%", filing_date),
        )
        price_row = cursor.fetchone()
        if price_row is None:
            continue
        price = price_row[0]
        if not price or price <= 0:
            continue

        shares = float(row["shares_outstanding"])
        if shares <= 0:
            continue

        market_cap = price * shares

        # CVM financial values are reported in thousands of BRL; convert to BRL
        # for ratio computation so units match market_cap (price × shares).
        THOUSANDS = 1_000.0

        # Annualization factor
        if row["doc_type"] == "ITR" and row["quarter"] is not None:
            q = int(row["quarter"])
            ann_factor = 12.0 / (q * 3) if q > 0 else 1.0
        else:
            ann_factor = 1.0

        # P/E
        pe = None
        if row["net_income"] is not None:
            net_income = float(row["net_income"]) * THOUSANDS
            ann_ni = net_income * ann_factor
            if ann_ni > 0:
                pe = market_cap / ann_ni

        # P/B
        pb = None
        if row["equity"] is not None:
            equity = float(row["equity"]) * THOUSANDS
            if equity > 0:
                pb = market_cap / equity

        # EV/EBITDA
        ev_ebitda = None
        if row["ebitda"] is not None:
            ebitda = float(row["ebitda"]) * THOUSANDS
            ann_ebitda = ebitda * ann_factor
            if ann_ebitda > 0:
                net_debt_val = float(row["net_debt"]) * THOUSANDS if row["net_debt"] is not None else 0.0
                ev = market_cap + net_debt_val
                ev_ebitda = ev / ann_ebitda

        updates.append((pe, pb, ev_ebitda, row["filing_id"]))

    if updates:
        cursor.executemany(
            "UPDATE fundamentals_pit SET pe_ratio = ?, pb_ratio = ?, ev_ebitda = ? WHERE filing_id = ?",
            updates,
        )
        conn.commit()
        logger.info(f"Updated valuation ratios for {len(updates):,} rows")

    return len(updates)


def _propagate_fre_shares(conn: sqlite3.Connection) -> None:
    """
    Propagate shares_outstanding from FRE filings into DFP/ITR rows.

    For each DFP/ITR row where shares_outstanding is NULL, find the most
    recent FRE filing for the same company on or before the DFP/ITR filing_date.
    """
    conn.execute("""
        UPDATE fundamentals_pit
        SET shares_outstanding = (
            SELECT fre.shares_outstanding
            FROM fundamentals_pit fre
            WHERE fre.cnpj = fundamentals_pit.cnpj
              AND fre.doc_type = 'FRE'
              AND fre.filing_date <= fundamentals_pit.filing_date
              AND fre.shares_outstanding IS NOT NULL
            ORDER BY fre.filing_date DESC
            LIMIT 1
        )
        WHERE fundamentals_pit.doc_type IN ('DFP', 'ITR')
          AND fundamentals_pit.shares_outstanding IS NULL
    """)
    conn.commit()


# ── Main orchestration ─────────────────────────────────────────────────────────

def run_fundamentals_pipeline(
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    rebuild: bool = False,
    force_download: bool = False,
    skip_ratios: bool = False,
) -> None:
    """
    Execute the complete CVM fundamentals data pipeline.

    Steps:
     1. Initialize DB (non-destructively unless rebuild=True)
     2. Determine year range
     3. Download DFP ZIPs
     4. Download ITR ZIPs
     5. Download FRE ZIPs
     6. Parse and upsert DFP filings
     7. Parse and upsert ITR filings
     8. Parse and upsert FRE filings + propagate shares_outstanding
     9. Materialize valuation ratios (unless skip_ratios)
    10. Log summary statistics
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("CVM Fundamentals Data Pipeline")
    logger.info("=" * 60)
    logger.info(f"Started at: {start_time}")
    logger.info(f"Rebuild mode: {rebuild}")
    logger.info(f"Skip ratios: {skip_ratios}")

    conn = storage.get_connection()

    try:
        # Step 1
        logger.info("")
        logger.info("Step 1/10: Initializing database...")
        storage.init_db(conn, rebuild=rebuild)

        # Step 2
        logger.info("")
        logger.info("Step 2/10: Determining year range...")
        if start_year is None:
            start_year = config.CVM_START_YEAR
        if end_year is None:
            end_year = config.get_current_year()
        logger.info(f"Year range: {start_year} – {end_year}")

        # Step 3
        logger.info("")
        logger.info("Step 3/10: Downloading DFP files...")
        dfp_paths = []
        for year in range(start_year, end_year + 1):
            path = cvm_downloader.download_dfp_file(year, force=force_download)
            if path:
                dfp_paths.append(path)
        logger.info(f"DFP files available: {len(dfp_paths)}")

        # Step 4
        logger.info("")
        logger.info("Step 4/10: Downloading ITR files...")
        itr_paths = []
        for year in range(start_year, end_year + 1):
            path = cvm_downloader.download_itr_file(year, force=force_download)
            if path:
                itr_paths.append(path)
        logger.info(f"ITR files available: {len(itr_paths)}")

        # Step 5
        logger.info("")
        logger.info("Step 5/10: Downloading FRE files...")
        fre_paths = []
        for year in range(start_year, end_year + 1):
            path = cvm_downloader.download_fre_file(year, force=force_download)
            if path:
                fre_paths.append(path)
        logger.info(f"FRE files available: {len(fre_paths)}")

        # Step 5b: Populate cvm_companies from CVM files (CNPJ + CVM code + company name).
        # This must run before reading the ticker map so the B3 pipeline can link tickers.
        logger.info("")
        logger.info("Step 5b/10: Indexing companies from CVM files...")
        all_zip_paths = dfp_paths + itr_paths + fre_paths
        seen_cnpjs: set = set()
        for zip_path in all_zip_paths:
            try:
                companies = cvm_parser.extract_company_index(zip_path)
                new_companies = [(c, v, n) for c, v, n in companies if c not in seen_cnpjs]
                if new_companies:
                    cvm_storage.bulk_upsert_companies_index(conn, new_companies)
                    seen_cnpjs.update(c for c, _, _ in new_companies)
            except Exception as e:
                logger.warning(f"Failed to index companies from {zip_path}: {e}")
        logger.info(f"Company index: {len(seen_cnpjs):,} unique CNPJs in cvm_companies")

        # Load CNPJ → ticker map (built from cvm_companies table populated by B3 pipeline)
        cnpj_ticker_map = cvm_storage.get_cvm_company_map(conn)
        logger.info(f"CNPJ → ticker map: {len(cnpj_ticker_map):,} companies")

        # Step 6
        logger.info("")
        logger.info("Step 6/10: Parsing and upserting DFP filings...")
        for zip_path in dfp_paths:
            try:
                filings_df, fund_df = cvm_parser.parse_dfp_zip(zip_path, cnpj_ticker_map)
                if not filings_df.empty:
                    for _, row in filings_df.iterrows():
                        cvm_storage.upsert_cvm_filing(
                            conn, row["filing_id"], row["cnpj"], row["doc_type"],
                            row["period_end"], row["filing_date"],
                            int(row["filing_version"]),
                            row.get("fiscal_year"), row.get("quarter"),
                            row.get("source_file"),
                        )
                if not fund_df.empty:
                    cvm_storage.upsert_fundamentals_pit(conn, fund_df)
            except Exception as e:
                logger.error(f"Failed to parse DFP file {zip_path}: {e}")

        # Step 7
        logger.info("")
        logger.info("Step 7/10: Parsing and upserting ITR filings...")
        for zip_path in itr_paths:
            try:
                filings_df, fund_df = cvm_parser.parse_itr_zip(zip_path, cnpj_ticker_map)
                if not filings_df.empty:
                    for _, row in filings_df.iterrows():
                        cvm_storage.upsert_cvm_filing(
                            conn, row["filing_id"], row["cnpj"], row["doc_type"],
                            row["period_end"], row["filing_date"],
                            int(row["filing_version"]),
                            row.get("fiscal_year"), row.get("quarter"),
                            row.get("source_file"),
                        )
                if not fund_df.empty:
                    cvm_storage.upsert_fundamentals_pit(conn, fund_df)
            except Exception as e:
                logger.error(f"Failed to parse ITR file {zip_path}: {e}")

        # Step 8
        logger.info("")
        logger.info("Step 8/10: Parsing and upserting FRE filings (shares outstanding)...")
        for zip_path in fre_paths:
            try:
                filings_df, fund_df = cvm_parser.parse_fre_zip(zip_path, cnpj_ticker_map)
                if not filings_df.empty:
                    for _, row in filings_df.iterrows():
                        cvm_storage.upsert_cvm_filing(
                            conn, row["filing_id"], row["cnpj"], row["doc_type"],
                            row["period_end"], row["filing_date"],
                            int(row["filing_version"]),
                            row.get("fiscal_year"), row.get("quarter"),
                            row.get("source_file"),
                        )
                if not fund_df.empty:
                    cvm_storage.upsert_fundamentals_pit(conn, fund_df)
            except Exception as e:
                logger.error(f"Failed to parse FRE file {zip_path}: {e}")

        logger.info("Propagating FRE shares_outstanding into DFP/ITR rows...")
        _propagate_fre_shares(conn)

        logger.info("Propagating tickers from cvm_companies into fundamentals_pit...")
        ticker_rows = cvm_storage.populate_tickers_from_cvm_companies(conn)
        logger.info(f"Ticker propagation updated {ticker_rows:,} fundamentals_pit rows")

        # Step 9
        if not skip_ratios:
            logger.info("")
            logger.info("Step 9/10: Materializing valuation ratios...")
            updated = materialize_valuation_ratios(conn)
            logger.info(f"Valuation ratios updated for {updated:,} rows")
        else:
            logger.info("")
            logger.info("Step 9/10: Skipping valuation ratio computation (--skip-ratios)")

        # Step 10
        logger.info("")
        logger.info("Step 10/10: Summary statistics...")
        stats = cvm_storage.get_fundamentals_stats(conn)
        logger.info(f"Total filings: {stats['total_cvm_filings']:,}")
        logger.info(f"Companies with CVM data: {stats['total_cvm_companies']:,}")
        logger.info(f"Fundamentals rows: {stats['total_fundamentals_pit']:,}")

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info("CVM PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total duration: {duration}")
        logger.info(f"Database: {config.DB_PATH}")

    except Exception as e:
        logger.exception(f"CVM pipeline failed with error: {e}")
        raise
    finally:
        conn.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point."""
    arg_parser = argparse.ArgumentParser(
        description="CVM Fundamentals Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m b3_pipeline.cvm_main                       # Run all years
    python -m b3_pipeline.cvm_main --start-year 2020     # From 2020 to current
    python -m b3_pipeline.cvm_main --start-year 2023 --end-year 2023
    python -m b3_pipeline.cvm_main --rebuild              # Rebuild from scratch
    python -m b3_pipeline.cvm_main --skip-ratios          # Skip ratio computation
        """,
    )
    arg_parser.add_argument("--start-year", type=int, default=None)
    arg_parser.add_argument("--end-year", type=int, default=None)
    arg_parser.add_argument(
        "--rebuild", action="store_true",
        help="Drop and recreate CVM fundamentals tables before processing",
    )
    arg_parser.add_argument(
        "--force-download", action="store_true",
        help="Re-download ZIP files even if they already exist",
    )
    arg_parser.add_argument(
        "--skip-ratios", action="store_true",
        help="Skip valuation ratio computation (P/E, P/B, EV/EBITDA)",
    )
    arg_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = arg_parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_fundamentals_pipeline(
        start_year=args.start_year,
        end_year=args.end_year,
        rebuild=args.rebuild,
        force_download=args.force_download,
        skip_ratios=args.skip_ratios,
    )


if __name__ == "__main__":
    main()
