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
    python -m b3_pipeline.cvm_main --include-historical         # Also load CAD company registry

Data coverage:
    CVM's open data portal provides structured bulk financial data only from 2010 onward:
    - DFP (annual):    2010+
    - ITR (quarterly): 2011+
    - FRE (shares):    2010+

    Pre-2010 financial data (revenue, net income, equity) does NOT exist as structured
    bulk CSV on dados.cvm.gov.br. The IPE dataset (2003-2009) contains individual PDF
    documents per company, not parseable structured data.

    --include-historical adds:
    - CAD company register: listing/delisting dates for survivorship-bias correction
    - IPE document index: company metadata and filing history (NOT financial statements)
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from . import b3_corporate_actions, config, cvm_downloader, cvm_parser, cvm_storage, storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Shared helper: ADTV ticker map ────────────────────────────────────────────

def _build_adtv_ticker_map(conn: sqlite3.Connection) -> dict:
    """Build a {cnpj: best_ticker} map using highest average daily trading volume.

    Only considers standard-lot tickers:
    - Length 5 ending in '3','4','5','6'
    - Length 6 ending in '11' (units/FII tickers)
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            c.cnpj,
            p.ticker,
            AVG(p.volume / 100.0) AS adtv
        FROM cvm_companies c
        JOIN prices p ON SUBSTR(p.ticker, 1, 4) = c.ticker
        WHERE c.ticker IS NOT NULL
          AND (
              (LENGTH(p.ticker) = 5 AND SUBSTR(p.ticker, 5, 1) IN ('3','4','5','6'))
              OR (LENGTH(p.ticker) = 6 AND SUBSTR(p.ticker, 5, 2) = '11')
          )
        GROUP BY c.cnpj, p.ticker
        ORDER BY c.cnpj, adtv DESC
    """)
    rows = cursor.fetchall()

    adtv_map: dict = {}
    for cnpj, ticker, adtv in rows:
        if cnpj not in adtv_map:
            adtv_map[cnpj] = ticker
    return adtv_map


def materialize_fundamentals_monthly(conn: sqlite3.Connection) -> int:
    """
    Build (or rebuild) the fundamentals_monthly snapshot table.

    For each (month_end, ticker), stores the most recently known fundamental
    values as of that date. Valuation ratios (P/E, P/B, EV/EBITDA) are no
    longer stored — they are computed dynamically at query time using the
    dynamic ratio helpers in backtests.core.data.

    Returns the number of rows upserted.
    """
    import datetime as dt

    cursor = conn.cursor()

    # Check if fundamentals_pit has any rows with a ticker
    cursor.execute(
        "SELECT MIN(filing_date) FROM fundamentals_pit WHERE ticker IS NOT NULL"
    )
    row = cursor.fetchone()
    if row is None or row[0] is None:
        logger.info("fundamentals_pit is empty — nothing to materialize")
        return 0

    earliest_filing_date = row[0]

    # Build month-end grid from earliest filing to today
    today = dt.date.today()
    month_ends = pd.date_range(
        start=earliest_filing_date, end=today.strftime("%Y-%m-%d"), freq="ME"
    )
    if len(month_ends) == 0:
        return 0

    # Build ADTV ticker map: {cnpj: best_ticker}
    adtv_map = _build_adtv_ticker_map(conn)

    # Load all fundamentals_pit rows with a matched ticker
    raw_metrics = [
        "revenue", "net_income", "ebitda", "total_assets", "equity",
        "net_debt", "shares_outstanding",
    ]
    metrics_sql = ", ".join(f"f.{m}" for m in raw_metrics)
    cursor.execute(f"""
        SELECT f.ticker, f.filing_date, {metrics_sql}
        FROM fundamentals_pit f
        WHERE f.ticker IS NOT NULL
        ORDER BY f.ticker, f.filing_date, f.filing_version
    """)
    pit_rows = cursor.fetchall()
    pit_cols = ["ticker"] + ["filing_date"] + raw_metrics
    pit_df = pd.DataFrame(pit_rows, columns=pit_cols)

    if pit_df.empty:
        return 0

    pit_df["filing_date"] = pd.to_datetime(pit_df["filing_date"])

    # Pivot each metric to wide format (filing_date x ticker), then forward-fill to month-ends
    metric_dfs = {}
    for metric in raw_metrics:
        sub = pit_df[["filing_date", "ticker", metric]].dropna(subset=[metric])
        if sub.empty:
            metric_dfs[metric] = pd.DataFrame(index=month_ends, dtype=float)
            continue
        wide = sub.pivot_table(
            index="filing_date", columns="ticker", values=metric, aggfunc="last"
        )
        wide.index = pd.to_datetime(wide.index)
        wide.columns.name = None
        extended_idx = wide.index.union(month_ends).sort_values()
        wide = wide.reindex(extended_idx).ffill()
        wide = wide.reindex(month_ends)
        metric_dfs[metric] = wide

    # Build ticker_root -> best_ticker map via cvm_companies cnpj -> root -> adtv_map
    cursor.execute("SELECT cnpj, ticker FROM cvm_companies WHERE ticker IS NOT NULL")
    cnpj_to_root = {row[0]: row[1] for row in cursor.fetchall()}
    root_to_best: dict = {}
    for cnpj, best_tkr in adtv_map.items():
        root = cnpj_to_root.get(cnpj)
        if root:
            root_to_best[root] = best_tkr

    if not metric_dfs:
        cvm_storage.truncate_fundamentals_monthly(conn)
        return 0

    # ── Vectorized build: stack each metric into (month_end, ticker_root) series ──
    stacked = []
    for metric, mdf in metric_dfs.items():
        s = mdf.stack()  # drops NaN; index=(month_end, ticker_root)
        s.name = metric
        stacked.append(s)

    combined = pd.concat(stacked, axis=1)  # outer join keeps any pair with ≥1 metric
    combined.index.names = ["month_end", "ticker_root"]
    combined = combined.reset_index()

    # Ensure all raw metric columns exist (some may be absent if no data)
    for metric in ["revenue", "net_income", "ebitda", "total_assets", "equity", "net_debt", "shares_outstanding"]:
        if metric not in combined.columns:
            combined[metric] = float("nan")

    combined["ticker"] = combined["ticker_root"].map(root_to_best).fillna(combined["ticker_root"])
    combined["month_end"] = combined["month_end"].dt.strftime("%Y-%m-%d")

    # Ratio columns (pe_ratio, pb_ratio, ev_ebitda) intentionally excluded — compute dynamically at query time
    output_cols = [
        "month_end", "ticker", "revenue", "net_income", "ebitda", "total_assets",
        "equity", "net_debt", "shares_outstanding",
    ]
    result_df = combined[output_cols].copy()

    cvm_storage.truncate_fundamentals_monthly(conn)
    n = cvm_storage.upsert_fundamentals_monthly(conn, result_df)
    logger.info(f"Materialized {n:,} fundamentals_monthly rows")
    return n


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
              AND fre.shares_outstanding > 0
            ORDER BY fre.filing_date DESC
            LIMIT 1
        )
        WHERE fundamentals_pit.doc_type IN ('DFP', 'ITR')
          AND fundamentals_pit.shares_outstanding IS NULL
    """)
    conn.commit()


# ── Ticker mapping helper ───────────────────────────────────────────────────────

def _fetch_ticker_mappings(conn: sqlite3.Connection) -> int:
    """Fetch ticker -> cvm_code mappings from the B3 API and update cvm_companies.

    Primary path: fetches the full B3 company list via GetInitialCompanies
    (one paginated call for ~3,329 companies) and matches each company to an
    existing cvm_companies row using codeCVM. This is much faster than the
    previous per-company approach and correctly uses codeCVM for matching since
    GetListedSupplementCompany never returns a CNPJ.

    Fallback path: for ticker roots NOT found in the bulk list, falls back to
    the per-company GetListedSupplementCompany call and still uses codeCVM to
    match the cvm_companies row (NOT cnpj, which is always None on that endpoint).

    Fix 4: Warns if match rate is suspiciously low (< 10%) so silent failures
    are immediately visible.

    Returns the number of cvm_companies rows that had their ticker updated.
    """
    tickers = storage.get_all_tickers(conn)
    if not tickers:
        logger.info("No tickers in prices table -- skipping B3 ticker fetch")
        return 0

    ticker_roots = sorted(set(t[:4] for t in tickers if len(t) >= 4))
    logger.info(
        f"Fetching ticker mappings for {len(ticker_roots)} company roots from B3 API..."
    )

    updated = 0

    # ── Primary path: GetInitialCompanies bulk fetch ───────────────────────────
    # Build a lookup from issuingCompany (4-char root) -> company record
    logger.info("Fetching full B3 company list via GetInitialCompanies...")
    bulk_companies = b3_corporate_actions.fetch_all_b3_listed_companies()

    bulk_by_root: dict = {}
    for rec in bulk_companies:
        issuing = str(rec.get("issuingCompany", "") or "").strip().upper()
        if issuing:
            bulk_by_root[issuing] = rec

    covered_roots: set = set()

    for root in ticker_roots:
        rec = bulk_by_root.get(root.upper())
        if rec is None:
            continue
        cvm_code = str(rec.get("codeCVM", "") or "").strip()
        if not cvm_code:
            continue
        matched = cvm_storage.update_ticker_by_cvm_code(
            conn,
            cvm_code=cvm_code,
            ticker=root,
            b3_trading_name=rec.get("tradingName", ""),
        )
        if matched:
            updated += 1
        else:
            logger.debug(
                f"No cvm_companies row for codeCVM={cvm_code} ({root}) from bulk list"
            )
        covered_roots.add(root)

    # ── Fallback path: per-company GetListedSupplementCompany (concurrent) ──────
    fallback_roots = [r for r in ticker_roots if r not in covered_roots]
    if fallback_roots:
        logger.info(
            f"Falling back to per-company API for {len(fallback_roots)} roots "
            f"not found in bulk list (concurrent, {config.MAX_WORKERS} workers)..."
        )
        completed = 0

        def _fetch_one(root: str):
            return root, b3_corporate_actions.fetch_company_data(root, conn=None)

        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = {executor.submit(_fetch_one, root): root for root in fallback_roots}
            for future in as_completed(futures):
                completed += 1
                if completed % 50 == 0 or completed == len(fallback_roots):
                    logger.info(f"  Fallback {completed}/{len(fallback_roots)}")
                try:
                    root, company_data = future.result()
                except Exception as e:
                    logger.debug(f"Fallback fetch failed for {futures[future]}: {e}")
                    continue
                if company_data is None:
                    continue
                cvm_code = str(company_data.get("codeCVM", "") or "").strip()
                if not cvm_code:
                    continue
                # DB write is sequential (one at a time via GIL + SQLite serialization)
                matched = cvm_storage.update_ticker_by_cvm_code(
                    conn,
                    cvm_code=cvm_code,
                    ticker=root,
                    b3_trading_name=company_data.get("tradingName", ""),
                )
                if matched:
                    updated += 1
                else:
                    logger.debug(
                        f"No cvm_companies row for codeCVM={cvm_code} ({root}) from fallback"
                    )

    # ── Fix 4: Match rate warning ──────────────────────────────────────────────
    match_rate = updated / len(ticker_roots) if ticker_roots else 0
    if updated == 0:
        logger.warning(
            f"No tickers were mapped! match rate = 0/{len(ticker_roots)}. "
            f"Check B3 API responses and cvm_companies population."
        )
    elif match_rate < 0.1:
        logger.warning(
            f"Ticker mapping match rate is very low ({match_rate:.1%}). "
            f"Only {updated}/{len(ticker_roots)} roots matched cvm_companies. "
            f"Check whether B3 API response format has changed."
        )

    logger.info(f"B3 ticker fetch complete: {updated}/{len(ticker_roots)} companies updated")
    return updated


# ── Main orchestration ─────────────────────────────────────────────────────────

def run_fundamentals_pipeline(
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    rebuild: bool = False,
    force_download: bool = False,
    skip_ratios: bool = False,
    skip_ticker_fetch: bool = False,
    skip_monthly: bool = False,
    include_historical: bool = False,
) -> None:
    """
    Execute the complete CVM fundamentals data pipeline.

    Steps:
     1. Initialize DB (non-destructively unless rebuild=True)
     2. Determine year range
     3. Download DFP ZIPs
     4. Download ITR ZIPs
     5. Download FRE ZIPs
     5b. Index companies from CVM files (CNPJ + CVM code)
     5c. Fetch ticker mappings from B3 API (unless skip_ticker_fetch)
     6. Parse and upsert DFP filings
     7. Parse and upsert ITR filings
     8. Parse and upsert FRE filings + propagate shares_outstanding
     9. [No-op] Valuation ratios are computed dynamically at query time
    10. Log summary statistics
    [When include_historical=True:]
    11. Download and parse CAD company register (listing/delisting dates)
    12. Download and process IPE document index (pre-2010 company data)
    13. Re-propagate tickers and re-run ratio/monthly materialization for IPE rows
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("CVM Fundamentals Data Pipeline")
    logger.info("=" * 60)
    logger.info(f"Started at: {start_time}")
    logger.info(f"Rebuild mode: {rebuild}")
    logger.info(f"Skip ratios: {skip_ratios}")
    logger.info(f"Skip ticker fetch: {skip_ticker_fetch}")
    logger.info(f"Skip monthly: {skip_monthly}")

    conn = storage.get_connection()

    try:
        # Step 1
        logger.info("")
        logger.info("Step 1/10: Initializing database...")
        storage.init_db(conn, rebuild=rebuild)

        # Step 2
        logger.info("")
        logger.info("Step 2/10: Determining year range...")
        _orig_start_year = start_year  # preserve before default override (needed for IPE range)
        if start_year is None:
            start_year = config.CVM_START_YEAR
        if end_year is None:
            end_year = config.get_current_year()
        logger.info(f"Year range: {start_year} – {end_year}")

        # Steps 3-5: Download DFP, ITR, FRE files concurrently
        logger.info("")
        logger.info("Steps 3-5/10: Downloading DFP, ITR, FRE files concurrently...")

        download_fns = {
            "DFP": cvm_downloader.download_dfp_file,
            "ITR": cvm_downloader.download_itr_file,
            "FRE": cvm_downloader.download_fre_file,
        }
        years = list(range(start_year, end_year + 1))
        tasks = [(doc_type, year, fn) for doc_type, fn in download_fns.items() for year in years]

        dfp_paths, itr_paths, fre_paths = [], [], []
        path_buckets = {"DFP": dfp_paths, "ITR": itr_paths, "FRE": fre_paths}

        MAX_DOWNLOAD_WORKERS = 6  # be respectful to CVM servers

        with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
            future_to_meta = {
                executor.submit(fn, year, force_download): (doc_type, year)
                for doc_type, year, fn in tasks
            }
            for future in as_completed(future_to_meta):
                doc_type, year = future_to_meta[future]
                try:
                    path = future.result()
                    if path:
                        path_buckets[doc_type].append(path)
                except Exception as e:
                    logger.warning(f"Download failed for {doc_type} {year}: {e}")

        dfp_paths.sort()
        itr_paths.sort()
        fre_paths.sort()
        logger.info(f"DFP: {len(dfp_paths)} files, ITR: {len(itr_paths)} files, FRE: {len(fre_paths)} files")

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

        # Step 5c: Fetch ticker mappings from B3 API so cvm_companies.ticker is
        # populated before we build the cnpj_ticker_map used during parsing.
        if not skip_ticker_fetch:
            logger.info("")
            logger.info("Step 5c/10: Fetching ticker mappings from B3 API...")
            ticker_count = _fetch_ticker_mappings(conn)
            logger.info(f"Ticker mappings: {ticker_count:,} companies updated")
        else:
            logger.info("Step 5c/10: Skipping ticker fetch (--skip-ticker-fetch)")

        # Load CNPJ → ticker map (built from cvm_companies table populated above
        # or by a prior B3 pipeline run).
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

        logger.info("Populating company_isin_map for precise ISIN-based price lookup...")
        isin_map_rows = cvm_storage.populate_company_isin_map(conn)
        logger.info(f"company_isin_map: {isin_map_rows:,} rows inserted/replaced")

        # Step 9
        logger.info("")
        logger.info("Step 9/10: Valuation ratio columns removed — ratios computed dynamically at query time.")

        # Step 9b
        if not skip_monthly:
            logger.info("")
            logger.info("Step 9b/10: Materializing monthly fundamentals snapshot...")
            monthly_rows = materialize_fundamentals_monthly(conn)
            logger.info(f"Monthly snapshot: {monthly_rows:,} rows materialized")
        else:
            logger.info("")
            logger.info("Step 9b/10: Skipping monthly snapshot (--skip-monthly)")

        # Step 10
        logger.info("")
        logger.info("Step 10/10: Summary statistics...")
        stats = cvm_storage.get_fundamentals_stats(conn)
        logger.info(f"Total filings: {stats['total_cvm_filings']:,}")
        logger.info(f"Companies with CVM data: {stats['total_cvm_companies']:,}")
        logger.info(f"Fundamentals rows: {stats['total_fundamentals_pit']:,}")

        # ── Historical extension (Steps 11-13) ────────────────────────────────
        if include_historical:
            from . import cad_downloader, cad_parser, ipe_downloader, ipe_parser  # noqa: PLC0415

            IPE_END_YEAR = 2009  # IPE predates DFP/ITR which starts 2010
            # Use the original user-supplied start_year (not the CVM_START_YEAR default)
            # so that --include-historical with no --start-year defaults to 2003, not 2010.
            user_start = _orig_start_year  # captured before Step 2 override
            ipe_start = max(2003, user_start or 2003)
            ipe_end = min(IPE_END_YEAR, end_year or IPE_END_YEAR)

            # Step 11: CAD download and company date upsert
            logger.info("")
            logger.info("Step 11/13: Downloading and parsing CVM CAD company register...")
            cad_path = cad_downloader.download_cad_file(force=force_download)
            if cad_path is None:
                logger.warning("CAD download failed — skipping company date upsert")
            else:
                cad_df = cad_parser.parse_cad_file(cad_path)
                summary = cad_parser.extract_cad_summary(cad_df)
                logger.info(
                    f"CAD: {summary['total']:,} companies "
                    f"({summary['active']:,} active, {summary['delisted']:,} delisted, "
                    f"{summary['with_listing_date']:,} with listing date)"
                )
                cad_rows = cvm_storage.upsert_cad_company_dates(conn, cad_df)
                logger.info(f"Upserted {cad_rows:,} CAD company date rows")
                # Refresh ticker map after CAD adds new CNPJ rows
                if not skip_ticker_fetch:
                    logger.info("Re-fetching ticker mappings after CAD upsert...")
                    _fetch_ticker_mappings(conn)
                cnpj_ticker_map = cvm_storage.get_cvm_company_map(conn)
                logger.info(f"Refreshed CNPJ → ticker map: {len(cnpj_ticker_map):,} companies")

            # Step 12: IPE download and index
            # IPE (2003-2009) is a document filing index — each entry is a link to a PDF.
            # It does NOT contain structured financial data (revenue, net income, etc.).
            # We process it for two purposes only:
            #   1. Index company CNPJs seen before 2010 (helps ticker mapping)
            #   2. Store filing metadata in cvm_filings for audit/completeness
            # Financial coverage remains 2010+ from DFP/ITR/FRE.
            logger.info("")
            logger.info(f"Step 12/13: Downloading and indexing IPE filing metadata ({ipe_start}-{ipe_end})...")
            total_ipe_filings = 0
            for year in range(ipe_start, ipe_end + 1):
                try:
                    ipe_path = ipe_downloader.download_ipe_file(year, force=force_download)
                    if ipe_path is None:
                        logger.warning(f"IPE {year}: download failed — skipping")
                        continue
                    # Index companies from IPE
                    ipe_companies = ipe_parser.extract_ipe_company_index(ipe_path)
                    if ipe_companies:
                        cvm_storage.bulk_upsert_companies_index(conn, ipe_companies)
                        logger.debug(f"IPE {year}: indexed {len(ipe_companies)} companies")
                    # Parse IPE filings and fundamentals (fundamentals_df will be empty)
                    filings_df, fund_df = ipe_parser.parse_ipe_zip(ipe_path, cnpj_ticker_map)
                    if not filings_df.empty:
                        for _, row in filings_df.iterrows():
                            try:
                                cvm_storage.upsert_cvm_filing(
                                    conn, row["filing_id"], row["cnpj"], row["doc_type"],
                                    row["period_end"], row["filing_date"],
                                    int(row["filing_version"]),
                                    row.get("fiscal_year"), row.get("quarter"),
                                    row.get("source_file"),
                                )
                            except Exception as upsert_err:
                                logger.warning(f"Failed to upsert IPE filing {row.get('filing_id', '?')}: {upsert_err}")
                        total_ipe_filings += len(filings_df)
                    if not fund_df.empty:
                        cvm_storage.upsert_fundamentals_pit(conn, fund_df)
                except Exception as e:
                    logger.error(f"IPE {year}: failed to process — {e}")
            logger.info(f"IPE total filings processed: {total_ipe_filings:,}")

            # Propagate tickers into any new IPE fundamentals_pit rows
            ticker_rows = cvm_storage.populate_tickers_from_cvm_companies(conn)
            logger.info(f"Ticker propagation (post-IPE): {ticker_rows:,} rows updated")

            # Step 13: Re-run monthly snapshot for IPE rows
            logger.info("")
            logger.info("Step 13/13: Rebuilding monthly snapshot...")
            # IPE adds no financial rows to fundamentals_pit, so monthly snapshot is
            # unchanged by IPE processing. This step is kept for completeness in case
            # future datasets add pre-2010 structured data.
            if not skip_monthly:
                monthly_rows = materialize_fundamentals_monthly(conn)
                logger.info(f"Monthly snapshot (post-IPE): {monthly_rows:,} rows materialized")


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
        help="[DEPRECATED — no-op] Valuation ratios are computed dynamically at query time and are no longer materialized.",
    )
    arg_parser.add_argument(
        "--skip-ticker-fetch", action="store_true",
        help="Skip fetching ticker mappings from B3 API (use when tickers are already populated)",
    )
    arg_parser.add_argument(
        "--skip-monthly", action="store_true",
        help="Skip materializing the fundamentals_monthly snapshot table",
    )
    arg_parser.add_argument(
        "--include-historical", action="store_true",
        help=(
            "Download and process CAD company register (listing/delisting dates) and "
            "IPE document index (2003-2009 company metadata). "
            "NOTE: does NOT provide pre-2010 financial data — CVM's bulk structured "
            "financials start at 2010. IPE documents are individual PDFs, not parseable CSVs."
        ),
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
        skip_ticker_fetch=args.skip_ticker_fetch,
        skip_monthly=args.skip_monthly,
        include_historical=args.include_historical,
    )


if __name__ == "__main__":
    main()
