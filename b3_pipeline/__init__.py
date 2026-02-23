"""
B3 Historical Market Data Pipeline

A complete Python data pipeline for downloading and processing historical
equity data from B3 (Brazilian Stock Exchange).

Modules:
    - config: Configuration constants
    - downloader: Download COTAHIST files and corporate actions
    - parser: Parse fixed-width COTAHIST format
    - adjustments: Split and dividend adjustment calculations
    - storage: SQLite database operations
    - main: CLI orchestrator
"""

__version__ = "1.0.0"

from . import adjustments, config, downloader, parser, storage

__all__ = [
    "adjustments",
    "config",
    "downloader",
    "parser",
    "storage",
]
