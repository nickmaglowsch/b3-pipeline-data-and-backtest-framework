# TODO: COTAHIST Quotation Factor (FATCOT) Support

## Problem

B3's COTAHIST format includes a field called **FATCOT** (fator de cotacao) that indicates how a stock's price is quoted — per share, per lot of 1000, per lot of 1M, etc. Our parser ignores this field, which causes incorrect price adjustments for stocks that changed their quotation basis over time.

### How it manifests

When a stock like VSPT3 (Ferrovia Centro-Atlantica) had a grupamento (reverse split) of 3,333,333:1 in 2017, B3's API correctly reports `factor: "0,00000030000"`. The adjustment code multiplies all pre-grupamento prices by 3,333,333x. But the pre-grupamento prices in COTAHIST were quoted **per lot**, not per share, so the multiplication is wrong:

- Raw close in 2010: R$2.80 (per lot, NOT per share)
- After adjustment: R$2.80 * 3,333,333 = R$9,333,333 (nonsensical)
- B3 API confirms: `"quotedPerSharSince": "23/10/2017"`

This affects many stocks that went through quotation basis changes, especially:
- Hyperinflation-era stocks (1980s-1990s) with massive bonus shares (99,900%)
- Stocks that transitioned from per-lot to per-share quotation
- Penny stocks that underwent extreme grupamentos

### Current workaround

Sanity capping in `b3_pipeline/adjustments.py`:
- Individual split factors capped to 1000:1 max
- Cumulative factors capped to 100,000x
- Final adj_close/close ratio check with fallback to split_adj_close

This prevents extreme values from reaching backtests but silently discards legitimate adjustments for affected stocks.

### Affected stocks (examples)

| Ticker | Event | B3 Factor | Result |
|--------|-------|-----------|--------|
| VSPT3 | Grupamento 2017 | 0.0000003 | 3,333,333x adjustment on lot-priced data |
| BEES3/4 | Desdobramento 2012 | 99,900 | 99,900x split on old quotation basis |
| MERC3/4 | Desdobramento 2023 | 99,900 | Combined with 0.001 grupamento same day |
| Various | Hyperinflation bonuses | 99,900-351,900 | 1980s/90s bonus shares |

## Proposed Fix

### 1. Parse FATCOT from COTAHIST

The COTAHIST fixed-width format has a quotation factor field. Add it to the parser:

```
Position 210-217 (0-indexed): FATCOT - Fator de Cotacao (quotation factor)
Format: integer
Value: number of shares per quoted unit (e.g., 1 = per share, 1000 = per lot of 1000)
```

In `b3_pipeline/parser.py`, add:
```python
quotation_factor = _parse_int(line[210:217])
```

### 2. Normalize prices to per-share

Before storing or adjusting prices, divide by the quotation factor:
```python
per_share_price = raw_price / quotation_factor
```

### 3. Track quotation basis changes

Store the quotation factor per ticker per date in the database. When applying split/grupamento adjustments, only adjust prices that are already on the same quotation basis, or normalize first.

### 4. Alternative approach: only adjust post-quotation-change data

For stocks with `quotedPerSharSince` dates from B3's API, only apply corporate action adjustments to data after that date. For earlier data, use the FATCOT field to normalize.

## References

- B3 COTAHIST layout specification (positions and field formats)
- B3 API field: `quotedPerSharSince` in GetListedSupplementCompany response
- FATCOT field documentation in B3's historical data manual

---

# TODO: Rust + Polars Performance Overhaul

## Motivation

The pipeline is well-architected but has two structural performance ceilings that Python cannot overcome without algorithmic workarounds:

1. **COTAHIST parsing** is fixed-width binary record parsing over 2.5M rows per year — pure CPU-bound string slicing where Python is 50-100x slower than a compiled parser.
2. **Backtest signal computation** builds wide sparse DataFrames (dates x tickers) and runs rolling window operations, where pandas' copy-heavy memory model becomes the bottleneck at 1000+ tickers.

## Part 1: Rust COTAHIST Parser

### Target
Replace `b3_pipeline/parser.py` record parsing loop with a Rust extension compiled to a Python `.so` via `pyo3`.

### What to rewrite
Only the inner parsing loop — the fixed-width record extraction from raw bytes. Everything else (orchestration, DB upserts, corporate actions) stays in Python.

### Interface contract
```python
# Current (Python)
prices_df = parser.parse_cotahist_zip(zip_path)

# Target (Rust via pyo3)
import b3_pipeline.cotahist_rs as cotahist_rs
records = cotahist_rs.parse_zip(zip_path)  # -> list[dict] or PyArrow RecordBatch
prices_df = pd.DataFrame(records)
```

### Expected gains
- Parse time: from ~1-2s per year to ~10-50ms per year
- Memory: avoid intermediate Python string allocations (parse directly to typed arrays)
- Parallelism: use rayon to parse multiple year files concurrently

### Implementation notes
- Use `pyo3` + `maturin` for the Python extension
- Crate lives at `b3_pipeline_rs/` at project root
- Output Arrow RecordBatch (via `arrow2`) to skip the Python list→DataFrame conversion overhead
- COTAHIST layout documented in B3's spec: fixed-width ASCII, record type `01` = header, `99` = trailer, `02` = quote record
- Key fields: positions 12-24 (ticker), 27-39 (ISIN), 99-108 (close price), 170-188 (total volume), 210-217 (FATCOT)
- Prices are stored as integers (divide by 100.0 for BRL)

### Files affected
- New: `b3_pipeline_rs/src/lib.rs`, `b3_pipeline_rs/Cargo.toml`, `b3_pipeline_rs/pyproject.toml`
- Modified: `b3_pipeline/parser.py` (call Rust extension, fall back to Python if not compiled)
- Modified: `pyproject.toml` or `setup.py` (add maturin build step)

---

## Part 2: Polars Backtest Engine

### Target
Replace `backtests/core/data.py` (data loading) and the rolling/rank operations in `backtests/core/shared_data.py` with Polars lazy queries + Arrow-native computation.

### Why Polars instead of just optimizing pandas
- `join_asof` in Polars is the correct primitive for PIT forward-fill (vs. manual pivot + ffill)
- Lazy evaluation means signal chains don't materialize intermediate DataFrames
- Arrow memory layout eliminates the copy-on-write overhead of pandas operations
- `rolling().rank()` on wide frames avoids the intermediate allocations pandas creates

### Key Polars primitives to use
```python
import polars as pl

# PIT forward-fill fundamentals (replaces pivot + ffill + reindex)
prices = pl.scan_parquet("prices.parquet")  # or read from SQLite
fundamentals = pl.scan_parquet("fundamentals.parquet")
merged = prices.join_asof(
    fundamentals,
    on="date",
    by="ticker",
    strategy="backward"
)

# Rolling signals (replaces pandas rolling().rank())
signals = merged.with_columns([
    pl.col("adj_close").pct_change().over("ticker").alias("ret"),
    pl.col("ret").rolling_mean(12).over("ticker").alias("mom_12m"),
])
ranked = signals.with_columns([
    pl.col("mom_12m").rank("dense").over("date").alias("mom_rank")
])
```

### Migration strategy
- Phase 1: Replace `load_b3_data()` and `load_fundamentals_pit()` with Polars reads (drop-in: convert to pandas at the end for strategy compatibility)
- Phase 2: Rewrite `build_shared_data()` to return a Polars LazyFrame dict instead of wide DataFrames
- Phase 3: Port strategies one-by-one to use Polars expressions natively

### Files affected
- Modified: `backtests/core/data.py`
- Modified: `backtests/core/shared_data.py`
- Modified: Strategies in `backtests/strategies/` (Phase 3 only)
- New: `backtests/core/data_polars.py` (parallel implementation during migration)

### Expected gains
- Backtest initialization: from 2-5s to ~200-500ms
- Memory: from 400-600MB to ~80-120MB for 532 tickers over 20 years (Arrow layout + lazy eval)
- PIT fundamentals join: from 5s (pivot + ffill + reindex) to ~50ms (join_asof)
- Scales to 2000+ tickers without memory pressure

---

## Dependency
- Rust parser (Part 1) can be done independently of Polars (Part 2)
- Part 2 Phase 1 can be done before Part 1 — both are independent
- Part 2 Phase 3 (strategy migration) depends on Phase 2

---

# Known Limitation: No Pre-2010 Fundamental Data

## Problem

CVM's open data portal (`dados.cvm.gov.br`) provides structured bulk financial data only from
2010 onward. The IPE dataset (2003-2009) contains per-company PDF documents — not structured
CSV data — so automated financial extraction from it is not viable.

Current coverage:
```
prices:        1994+  (COTAHIST)
DFP (annual):  2010+
ITR (quarterly):2011+
FRE (shares):  2010+
fundamentals_pit: 2010+ only
```

## What --include-historical actually provides

- **CAD company register**: listing/delisting dates for survivorship-bias correction
- **IPE document index**: company CNPJs and filing metadata (2003-2009), helps ticker mapping
- **NOT** pre-2010 financial statements

## Verified findings

- `dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IAN/` → 404 (IAN dataset does not exist as bulk data)
- IPE ZIP files contain `ipe_cia_aberta_{year}.csv` — an index of document links
- Each document link (`rad.cvm.gov.br/ENET/frmDownloadDocumento.aspx?...`) returns a **PDF file**
- ~530 PDF documents per year for "Demonstrações Financeiras Anuais Completas" in 2008 alone
- No structured parsing path exists without a PDF extraction library + company-specific layout handling

## Options if pre-2010 data is needed

1. **Commercial data providers**: Economatica, Bloomberg, Refinitiv — all have structured
   Brazilian fundamentals going back to 2000+ with clean corporate action handling.
2. **Academic datasets**: NEFIN (USP) provides risk factors but not individual fundamentals.
3. **PDF extraction (future)**: If `pdfplumber` or similar is added, IPE PDFs for
   "Demonstrações Financeiras Anuais Completas" could be parsed. High engineering effort,
   variable accuracy across companies. ~3,700 PDF downloads for 2003-2009.
