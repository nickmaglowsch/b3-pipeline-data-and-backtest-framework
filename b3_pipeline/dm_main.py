#!/usr/bin/env python3
"""
CVM Download Múltiplo Pipeline (pre-2010 legacy CVMWIN fundamentals).

Downloads (or, with --parse-only, just parses files already saved under
data/dm/), parses, and upserts ITR/DFP/IAN filings from CVM's "Download
Múltiplo" service into the same cvm_filings / fundamentals_pit tables used
by the modern b3_pipeline.cvm_main pipeline. See docs/download_multiplo.md
for the registration process, protocol, and file-layout research this is
built from.

Usage:
    python -m b3_pipeline.dm_main --start 2006-01-01 --end 2010-12-31 --types ITR,DFP,IAN
    python -m b3_pipeline.dm_main --parse-only                          # just parse data/dm/
    python -m b3_pipeline.dm_main --start 2006-01-01 --end 2006-12-31 --types DFP

Requires CVM_DM_USER / CVM_DM_PASS environment variables (unless
--parse-only) -- see docs/download_multiplo.md for how to register and
obtain credentials.

LIMITATION: this pipeline cannot be integration-tested without real CVM
Download Múltiplo credentials. The parser (b3_pipeline/dm_parser.py) is
verified against synthetic fixtures built from the published layout spec
(tests/test_dm_parser.py); real-file behavior should be spot-checked once
credentials are available.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List

from . import cvm_storage, dm_downloader, dm_parser, storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PARSE_FNS = {
    "DFP": dm_parser.parse_dfp_dbf_zip,
    "ITR": dm_parser.parse_itr_dbf_zip,
    "IAN": dm_parser.parse_ian_dbf_zip,
}

RECOMPUTE_REMINDER = (
    "Re-run `python -m b3_pipeline.cvm_main --skip-ticker-fetch` to recompute "
    "net_income_ttm and rebuild the fundamentals_monthly snapshot with the newly "
    "upserted Download Múltiplo rows."
)


def _find_local_files(doc_type: str) -> List[Path]:
    """Find previously downloaded files for doc_type under data/dm/ (see
    dm_downloader.download_document's f"{doc}_{ccvm}_{dataref}_{subdate}{suffix}" naming)."""
    if not dm_downloader.DM_DATA_DIR.exists():
        return []
    return sorted(
        p for p in dm_downloader.DM_DATA_DIR.iterdir()
        if p.is_file() and p.name.upper().startswith(f"{doc_type.upper()}_")
    )


def _parse_and_upsert(conn, doc_type: str, paths: List[Path], cnpj_ticker_map: dict) -> int:
    parse_fn = PARSE_FNS[doc_type]
    total_rows = 0
    for path in paths:
        try:
            filings_df, fund_df = parse_fn(path, cnpj_ticker_map)
        except Exception as e:
            logger.error(f"Failed to parse {doc_type} file {path}: {e}")
            continue

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
            n = cvm_storage.upsert_fundamentals_pit(conn, fund_df)
            total_rows += n
            logger.info(f"{doc_type} {path.name}: upserted {n} fundamentals_pit rows")
        else:
            logger.info(f"{doc_type} {path.name}: no usable data")

    return total_rows


def run(
    start: date = None,
    end: date = None,
    doc_types: List[str] = None,
    parse_only: bool = False,
    force_download: bool = False,
) -> None:
    doc_types = doc_types or ["ITR", "DFP", "IAN"]
    unknown = [d for d in doc_types if d not in PARSE_FNS]
    if unknown:
        raise ValueError(f"Unsupported doc types: {unknown}. Expected any of {list(PARSE_FNS)}")

    logger.info("=" * 60)
    logger.info("CVM Download Múltiplo Pipeline")
    logger.info("=" * 60)
    logger.info(f"Doc types: {doc_types}")
    logger.info(f"Parse-only: {parse_only}")

    if not parse_only:
        if start is None or end is None:
            raise ValueError("--start and --end are required unless --parse-only is set")
        try:
            logger.info(f"Downloading {start} .. {end} for {doc_types} into {dm_downloader.DM_DATA_DIR}")
            paths = dm_downloader.download_range(start, end, doc_types, force=force_download)
            logger.info(f"Downloaded {len(paths)} files")
        except dm_downloader.CredentialsMissingError as e:
            logger.error(str(e))
            sys.exit(1)

    conn = storage.get_connection()
    try:
        storage.init_db(conn, rebuild=False, cvm_only=True)
        cnpj_ticker_map = cvm_storage.get_cvm_company_map(conn)
        logger.info(f"CNPJ -> ticker map: {len(cnpj_ticker_map):,} companies")

        total = 0
        for doc_type in doc_types:
            paths = _find_local_files(doc_type)
            logger.info(f"{doc_type}: {len(paths)} local file(s) under {dm_downloader.DM_DATA_DIR}")
            if not paths:
                continue
            total += _parse_and_upsert(conn, doc_type, paths, cnpj_ticker_map)

        logger.info(f"Total fundamentals_pit rows upserted: {total:,}")
    finally:
        conn.close()

    logger.info("")
    logger.info(RECOMPUTE_REMINDER)
    print(RECOMPUTE_REMINDER)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main():
    parser = argparse.ArgumentParser(
        description="CVM Download Múltiplo Pipeline (pre-2010 legacy fundamentals)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m b3_pipeline.dm_main --start 2006-01-01 --end 2010-12-31 --types ITR,DFP,IAN
    python -m b3_pipeline.dm_main --parse-only
        """,
    )
    parser.add_argument("--start", type=_parse_date, default=None, help="YYYY-MM-DD (required unless --parse-only)")
    parser.add_argument("--end", type=_parse_date, default=None, help="YYYY-MM-DD (required unless --parse-only)")
    parser.add_argument("--types", type=str, default="ITR,DFP,IAN", help="Comma-separated: ITR,DFP,IAN")
    parser.add_argument("--parse-only", action="store_true", help="Skip download; parse files already in data/dm/")
    parser.add_argument("--force-download", action="store_true", help="Re-download files even if they already exist")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    doc_types = [t.strip().upper() for t in args.types.split(",") if t.strip()]

    try:
        run(
            start=args.start,
            end=args.end,
            doc_types=doc_types,
            parse_only=args.parse_only,
            force_download=args.force_download,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
