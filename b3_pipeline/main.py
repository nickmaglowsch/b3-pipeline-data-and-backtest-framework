#!/usr/bin/env python3
"""
B3 Historical Market Data Pipeline

Downloads, parses, and adjusts historical equity data from B3 (Brazilian Stock Exchange).

Usage:
    python main.py              # Run pipeline (downloads missing data)
    python main.py --rebuild    # Rebuild entire database from scratch
    python main.py --year 2024  # Process specific year only
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Optional

from . import adjustments, config, downloader, parser, storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def run_pipeline(
    rebuild: bool = False,
    year: Optional[int] = None,
    skip_corporate_actions: bool = False,
) -> None:
    """
    Execute the complete B3 data pipeline.

    Steps:
    1. Initialize database (drop + recreate if rebuild=True)
    2. Detect and download COTAHIST files
    3. Parse COTAHIST files into standardized DataFrame
    4. Upsert raw prices to SQLite
    5. Fetch corporate actions from StatusInvest
    6. Detect splits from price gaps
    7. Compute split adjustment factors
    8. Compute dividend adjustment factors
    9. Update adjusted columns in database
    10. Log summary statistics

    Args:
        rebuild: If True, drop and recreate all tables
        year: If specified, only process this year
        skip_corporate_actions: If True, skip fetching corporate actions
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("B3 Historical Market Data Pipeline")
    logger.info("=" * 60)
    logger.info(f"Started at: {start_time}")
    logger.info(f"Rebuild mode: {rebuild}")
    logger.info(f"Target year: {year or 'all'}")

    conn = storage.get_connection()

    try:
        logger.info("")
        logger.info("Step 1/9: Initializing database...")
        storage.init_db(conn, rebuild=rebuild)

        logger.info("")
        logger.info("Step 2/9: Detecting available years...")
        available_years = downloader.detect_available_years()
        logger.info(
            f"Found {len(available_years)} years with data: {available_years[0]} to {available_years[-1]}"
        )

        if year:
            if year not in available_years:
                logger.error(f"Year {year} not available on B3")
                return
            years_to_process = [year]
        else:
            years_to_process = available_years

        logger.info("")
        logger.info("Step 3/9: Downloading COTAHIST files...")
        downloaded_files = []
        for y in years_to_process:
            filepath = downloader.download_annual_file(y)
            if filepath:
                downloaded_files.append(filepath)

        logger.info("")
        logger.info("Step 4/9: Parsing COTAHIST files...")
        prices = parser.parse_all_cotahist_files(config.DATA_DIR)

        if prices.empty:
            logger.error("No price data parsed. Exiting.")
            return

        logger.info("")
        logger.info("Step 5/9: Upserting raw prices to database...")
        storage.upsert_prices(conn, prices)

        corporate_actions = None
        if not skip_corporate_actions:
            logger.info("")
            logger.info("Step 6/9: Fetching corporate actions...")
            tickers = storage.get_all_tickers(conn)
            logger.info(f"Found {len(tickers)} unique tickers")

            if tickers:
                corporate_actions = downloader.fetch_all_corporate_actions(tickers)
                if not corporate_actions.empty:
                    storage.upsert_corporate_actions(conn, corporate_actions)
        else:
            logger.info("")
            logger.info("Step 6/9: Skipping corporate actions fetch...")
            corporate_actions = storage.get_all_corporate_actions(conn)

        logger.info("")
        logger.info("Step 7/9: Computing adjustments...")

        prices_from_db = storage.get_all_prices(conn)

        if corporate_actions is None or corporate_actions.empty:
            corporate_actions = storage.get_all_corporate_actions(conn)

        adjusted_prices, detected_splits = adjustments.compute_all_adjustments(
            prices_from_db, corporate_actions
        )

        if not detected_splits.empty:
            storage.upsert_detected_splits(conn, detected_splits)

        logger.info("")
        logger.info("Step 8/9: Updating adjusted columns in database...")
        storage.update_adjusted_columns(conn, adjusted_prices)

        logger.info("")
        logger.info("Step 9/9: Summary statistics...")
        stats = storage.get_summary_stats(conn)

        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total price records: {stats['total_prices']:,}")
        logger.info(f"Unique tickers: {stats['total_tickers']:,}")
        logger.info(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        logger.info(f"Corporate actions: {stats['total_corporate_actions']:,}")
        logger.info(f"Detected splits: {stats['total_detected_splits']:,}")

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Total duration: {duration}")
        logger.info(f"Database: {config.DB_PATH}")

    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        raise
    finally:
        conn.close()


def main():
    """CLI entry point."""
    arg_parser = argparse.ArgumentParser(
        description="B3 Historical Market Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m b3_pipeline.main              # Run pipeline
    python -m b3_pipeline.main --rebuild    # Rebuild from scratch
    python -m b3_pipeline.main --year 2024  # Process 2024 only
        """,
    )

    arg_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate database tables before processing",
    )

    arg_parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Process only the specified year",
    )

    arg_parser.add_argument(
        "--skip-corporate-actions",
        action="store_true",
        help="Skip fetching corporate actions (use existing data)",
    )

    arg_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = arg_parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_pipeline(
        rebuild=args.rebuild,
        year=args.year,
        skip_corporate_actions=args.skip_corporate_actions,
    )


if __name__ == "__main__":
    main()
