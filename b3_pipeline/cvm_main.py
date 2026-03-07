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

    # Build {cnpj: best_ticker} map using highest-ADTV standard-lot ticker per company.
    # This replaces the old is_primary=1 / LIKE fallback logic.
    adtv_map = _build_adtv_ticker_map(conn)

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
        cnpj = str(row["cnpj"])
        filing_date_str = str(row["filing_date"])

        # Lookup best ticker for this company using the ADTV map.
        best_ticker = adtv_map.get(cnpj)
        if best_ticker is None:
            continue

        # ±10 calendar days window (guarantees ≥5 trading days in either direction).
        filing_date_obj = datetime.strptime(filing_date_str, "%Y-%m-%d").date()
        window_start = (filing_date_obj - timedelta(days=10)).strftime("%Y-%m-%d")
        window_end = (filing_date_obj + timedelta(days=10)).strftime("%Y-%m-%d")

        cursor.execute(
            """
            SELECT date, close FROM prices
            WHERE ticker = ?
              AND date >= ?
              AND date <= ?
            ORDER BY date
            """,
            (best_ticker, window_start, window_end),
        )
        price_candidates = cursor.fetchall()

        if not price_candidates:
            continue

        # Pick closest date to filing_date; tie-break: prefer earlier date (backward preference).
        def sort_key(row_pair):
            date_str, _ = row_pair
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            distance = abs((d - filing_date_obj).days)
            return (distance, date_str)  # earlier date sorts first on tie

        best = sorted(price_candidates, key=sort_key)[0]
        price = best[1]

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
        quarter_val = row["quarter"]
        if row["doc_type"] == "ITR" and quarter_val is not None and quarter_val == quarter_val:
            q = int(quarter_val)
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


def materialize_fundamentals_monthly(conn: sqlite3.Connection) -> int:
    """
    Build (or rebuild) the fundamentals_monthly snapshot table.

    For each (month_end, ticker), stores the most recently known fundamental
    values as of that date, with valuation ratios recomputed using the actual
    month-end closing price from prices.

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

    # Get all best_tickers from adtv_map
    best_tickers = list(set(adtv_map.values()))

    # Load all prices for best_tickers up to last month_end, pivot to wide
    max_month_end = month_ends[-1].strftime("%Y-%m-%d")
    if best_tickers:
        placeholders = ",".join("?" * len(best_tickers))
        cursor.execute(f"""
            SELECT ticker, date, close
            FROM prices
            WHERE ticker IN ({placeholders})
              AND date <= ?
            ORDER BY ticker, date
        """, best_tickers + [max_month_end])
        price_rows = cursor.fetchall()
        price_df = pd.DataFrame(price_rows, columns=["ticker", "date", "close"])
        if not price_df.empty:
            price_df["date"] = pd.to_datetime(price_df["date"])
            price_wide = price_df.pivot_table(
                index="date", columns="ticker", values="close", aggfunc="last"
            )
            price_wide.columns.name = None
            # Resample to month-end, use last available price on or before each month-end
            extended_price_idx = price_wide.index.union(month_ends).sort_values()
            price_wide = price_wide.reindex(extended_price_idx).ffill()
            price_wide = price_wide.reindex(month_ends)
        else:
            price_wide = pd.DataFrame(index=month_ends, dtype=float)
    else:
        price_wide = pd.DataFrame(index=month_ends, dtype=float)

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

    # ── Map month-end prices from price_wide (best_ticker columns) to ticker_root ──
    if not price_wide.empty:
        best_to_root = {v: k for k, v in root_to_best.items()}
        price_by_root = price_wide.rename(
            columns={bt: best_to_root.get(bt, bt) for bt in price_wide.columns}
        )
        price_by_root.index.name = "month_end"
        price_stacked = price_by_root.stack()
        price_stacked.name = "price"
        price_stacked.index.names = ["month_end", "ticker_root"]
        combined = combined.merge(price_stacked.reset_index(), on=["month_end", "ticker_root"], how="left")
    else:
        combined["price"] = float("nan")

    # ── Vectorized ratio computation ──────────────────────────────────────────────
    THOUSANDS = 1_000.0
    price = combined["price"]
    shares = combined["shares_outstanding"]
    valid = (price > 0) & (shares > 0) & price.notna() & shares.notna()
    market_cap = price * shares

    ni_brl = combined["net_income"] * THOUSANDS
    combined["pe_ratio"] = (market_cap / ni_brl).where(valid & (ni_brl > 0))

    eq_brl = combined["equity"] * THOUSANDS
    combined["pb_ratio"] = (market_cap / eq_brl).where(valid & (eq_brl > 0))

    ebitda_brl = combined["ebitda"] * THOUSANDS
    nd_brl = combined["net_debt"].fillna(0) * THOUSANDS
    ev = market_cap + nd_brl
    combined["ev_ebitda"] = (ev / ebitda_brl).where(valid & (ebitda_brl > 0))

    combined["month_end"] = combined["month_end"].dt.strftime("%Y-%m-%d")

    output_cols = [
        "month_end", "ticker", "revenue", "net_income", "ebitda", "total_assets",
        "equity", "net_debt", "shares_outstanding", "pe_ratio", "pb_ratio", "ev_ebitda",
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

    # ── Fallback path: per-company GetListedSupplementCompany ─────────────────
    fallback_roots = [r for r in ticker_roots if r not in covered_roots]
    if fallback_roots:
        logger.info(
            f"Falling back to per-company API for {len(fallback_roots)} roots "
            f"not found in bulk list..."
        )
        for i, root in enumerate(fallback_roots, 1):
            if i % 50 == 0 or i == len(fallback_roots):
                logger.info(f"  Fallback {i}/{len(fallback_roots)} ({root})")
            company_data = b3_corporate_actions.fetch_company_data(root, conn=conn)
            if company_data is None:
                continue
            cvm_code = str(company_data.get("codeCVM", "") or "").strip()
            if not cvm_code:
                continue
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
     9. Materialize valuation ratios (unless skip_ratios)
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
        if not skip_ratios:
            logger.info("")
            logger.info("Step 9/10: Materializing valuation ratios...")
            updated = materialize_valuation_ratios(conn)
            logger.info(f"Valuation ratios updated for {updated:,} rows")
        else:
            logger.info("")
            logger.info("Step 9/10: Skipping valuation ratio computation (--skip-ratios)")

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

            # Step 12: IPE download, index, and parse
            logger.info("")
            logger.info(f"Step 12/13: Downloading and processing IPE historical filings ({ipe_start}-{ipe_end})...")
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

            # Step 13: Re-run ratio materialization and monthly snapshot for IPE rows
            logger.info("")
            logger.info("Step 13/13: Re-propagating shares and recomputing ratios for IPE rows...")
            # Note: _propagate_ipe_shares() is not implemented (Path B from Task 05).
            # IPE has no capital structure data — shares_outstanding stays NULL for IPE rows.
            if not skip_ratios:
                updated = materialize_valuation_ratios(conn)
                logger.info(f"Valuation ratios (post-IPE): {updated:,} rows updated")
            if not skip_monthly:
                monthly_rows = materialize_fundamentals_monthly(conn)
                logger.info(f"Monthly snapshot (post-IPE): {monthly_rows:,} rows materialized")

            # Log IPE coverage stats
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_ipe,
                        COUNT(shares_outstanding) as ipe_with_shares,
                        COUNT(pe_ratio) as ipe_with_pe
                    FROM fundamentals_pit
                    WHERE doc_type = 'IPE'
                """)
                ipe_row = cursor.fetchone()
                if ipe_row:
                    logger.info(
                        f"IPE coverage: {ipe_row[0]:,} total rows, "
                        f"{ipe_row[1]:,} with shares, {ipe_row[2]:,} with P/E"
                    )
            except Exception:
                pass

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
        "--skip-ticker-fetch", action="store_true",
        help="Skip fetching ticker mappings from B3 API (use when tickers are already populated)",
    )
    arg_parser.add_argument(
        "--skip-monthly", action="store_true",
        help="Skip materializing the fundamentals_monthly snapshot table",
    )
    arg_parser.add_argument(
        "--include-historical", action="store_true",
        help="Download and process IPE (pre-2010) and CAD company register data",
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
