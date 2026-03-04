# Refactor Planner Memory — B3 Data Pipeline

## Test Framework
- pytest; run with `pytest tests/ -v`
- 60 tests, all in `tests/` directory
- Test style: class-based (`class TestFoo`), in-memory SQLite via `sqlite3.connect(":memory:")`
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

## B3 Corporate Actions API
- Two endpoints per company: `GetListedSupplementCompany` (stock splits/bonuses) + `GetListedCashDividends` (paginated)
- Both at `sistemaswebb3-listados.b3.com.br` — no documented rate limit; conservative `RATE_LIMIT_DELAY = 0.05`
- Failures recorded to `fetch_failures` table; recovery via `--retry-failures` CLI (not inline)
- `fetch_company_data` returns `None` on any exception (RequestException, JSONDecodeError, AttributeError, TypeError)
