"""
B3 Historical Market Data Pipeline

A complete Python data pipeline for downloading and processing historical
equity data from B3 (Brazilian Stock Exchange).

B3 is the single source of truth for:
- Price data (COTAHIST)
- Cash dividends and JCP
- Stock splits, reverse splits, and bonus shares

Modules:
    - config: Configuration constants
    - downloader: Download COTAHIST files from B3
    - b3_corporate_actions: Fetch corporate actions from B3 API
    - parser: Parse fixed-width COTAHIST format
    - adjustments: Split and dividend adjustment calculations
    - storage: SQLite database operations
    - main: CLI orchestrator
"""

__version__ = "2.0.0"

from . import adjustments, b3_corporate_actions, config, downloader, parser, storage

__all__ = [
    "adjustments",
    "b3_corporate_actions",
    "config",
    "downloader",
    "parser",
    "storage",
]
