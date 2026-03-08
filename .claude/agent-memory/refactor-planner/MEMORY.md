# Refactor Planner Memory — B3 Data Pipeline

## Test Framework
- pytest; run with `pytest tests/ -v`
- ~198+ tests in `tests/` directory (count grows with each task)
- Test style: class-based (`class TestFoo`), helper functions (not pytest fixtures)
- DB fixture pattern: `_setup_db(tmp_path)` calls `storage.init_db(conn)` on a temp file
- Mock pattern: `unittest.mock.patch("b3_pipeline.<module>.<symbol>")` — patch at point of use

## Project Structure
- Pipeline modules: `b3_pipeline/` — `main.py`, `b3_corporate_actions.py`, `config.py`, `storage.py`, `cvm_storage.py`
- Config constants all in `b3_pipeline/config.py` — single source of truth for URLs, delays, labels
- No ORM; raw `sqlite3` everywhere; connections opened via `storage.get_connection()` (sets WAL mode + PRAGMAs)
- SQLite connections are NOT thread-safe — always open per-thread connections in concurrent workers

## Key Conventions
- Private helpers named with `_` prefix (`_fetch_one_company`, `_encode_payload`)
- Module-level `logger = logging.getLogger(__name__)` in every module
- `Optional[X]` return types used for functions that can return `None` on failure
- Imports of sibling modules done lazily inside functions (`from . import storage as _storage`) to avoid circular imports

## Concurrency Pattern Established (task-03)
- `ThreadPoolExecutor` with `config.MAX_WORKERS` workers
- Workers return plain data (DataFrames); no shared mutable state written from workers
- Each worker opens + closes its own `storage.get_connection()` for SQLite side-effects
- Main thread collects via `as_completed`, appends to lists, does final `pd.concat` + dedup
- Progress counter uses `threading.Lock` + shared int; logged every 10 completions

## Rust Extension (cotahist_rs)
- Crate at `b3_pipeline_rs/`, PyO3 module name `cotahist_rs`, crate name `cotahist_rs`
- Build: `make dev-rust` (debug) / `make build-rust` (release wheel) / `make test-rust`
- Boundary pattern: Arrow RecordBatch in, Arrow RecordBatch out (pyo3-arrow 0.5 + arrow 53)
- rayon already available for parallelism; DO NOT create extra ThreadPoolBuilders
- Python lazy-import: `try: import cotahist_rs` with graceful None fallback
- New Rust modules added to `b3_pipeline_rs/src/` and declared in `lib.rs` with `mod name;`
- Log messages: Rust collects `Vec<String>` and returns to Python; Python calls logger
- Float precision: always `f64`, never `f32`; use `MAX_CUMULATIVE_FACTOR = 100_000.0`
- Date boundary: convert Python dates to `Date32` (days since epoch) before Arrow conversion
- Arrow null vs NaN: use `append_null()` in Float64Builder for missing cells (not NaN sentinel)

## B3 Corporate Actions API
- Two endpoints per company: `GetListedSupplementCompany` (stock splits/bonuses) + `GetListedCashDividends` (paginated)
- Both at `sistemaswebb3-listados.b3.com.br` — no documented rate limit; conservative `RATE_LIMIT_DELAY = 0.05`
- Failures recorded to `fetch_failures` table; recovery via `--retry-failures` CLI (not inline)
- `fetch_company_data` returns `None` on any exception (RequestException, JSONDecodeError, AttributeError, TypeError)
