"""
Discovery Service
=================
Loads feature discovery results and runs the discovery pipeline.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
CATALOG_PATH = PROJECT_ROOT / "research" / "output" / "feature_catalog.json"
REPORT_PATH = PROJECT_ROOT / "research" / "output" / "discovery_report.txt"
REGISTRY_PATH = PROJECT_ROOT / "research" / "feature_store" / "registry.json"
FEATURE_STORE_DIR = PROJECT_ROOT / "research" / "feature_store"


@st.cache_data(ttl=300)
def get_discovery_catalog(catalog_mtime: float = 0.0) -> Optional[dict]:
    """
    Load and parse research/output/feature_catalog.json.
    Returns the full parsed dict, or None if file does not exist.

    The _catalog_mtime parameter (underscore prefix: not hashed by Streamlit)
    is passed as the file's mtime so the cache refreshes when the file changes.
    """
    if not CATALOG_PATH.exists():
        return None
    try:
        with open(CATALOG_PATH) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_discovery_report() -> Optional[str]:
    """
    Load research/output/discovery_report.txt as a string.
    Returns None if the file does not exist.
    """
    if not REPORT_PATH.exists():
        return None
    try:
        return REPORT_PATH.read_text()
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_discovery_registry_summary() -> Optional[dict]:
    """
    Load research/feature_store/registry.json and return a summary:
    {total_features, categories: {name: count}, data_hash, updated_at}
    """
    if not REGISTRY_PATH.exists():
        return None
    try:
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)

        features = registry.get("features", {})
        categories: dict[str, int] = {}
        for feature_data in features.values():
            cat = feature_data.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_features": len(features),
            "categories": categories,
            "data_hash": registry.get("data_hash"),
            "updated_at": registry.get("updated_at"),
        }
    except Exception:
        return None


def load_feature_data(feature_id: str) -> Optional[pd.DataFrame]:
    """
    Load a single feature Parquet from the feature store.
    Returns a long-format DataFrame (columns: date, ticker, value) or None on error.
    NOT cached -- Parquet files can be large.
    """
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from research.discovery.store import FeatureStore
        store = FeatureStore(store_dir=FEATURE_STORE_DIR)
        return store.load_feature(feature_id)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Failed to load feature %s: %s", feature_id, e)
        return None


def run_discovery_pipeline(
    incremental: bool = False,
    force_recompute: bool = False,
    pruning_overrides: Optional[dict] = None,
) -> None:
    """
    Run the discovery pipeline inside a JobRunner thread.
    Prints go to the log stream.

    Args:
        incremental: Skip already-computed features.
        force_recompute: Invalidate the feature store and recompute everything.
        pruning_overrides: Optional dict with keys min_ic_threshold, max_correlation,
                           max_features to temporarily override config values.
    """
    print("=" * 60)
    print("Starting Feature Discovery Pipeline")
    print("=" * 60)

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # Apply pruning overrides by monkey-patching the config module
    original_values: dict = {}
    if pruning_overrides:
        import research.discovery.config as discovery_config
        override_map = {
            "min_ic_threshold": "MIN_IC_THRESHOLD",
            "max_correlation": "MAX_CORRELATION",
            "max_features": "MAX_FEATURES",
        }
        for key, attr in override_map.items():
            if key in pruning_overrides:
                original_values[attr] = getattr(discovery_config, attr)
                setattr(discovery_config, attr, pruning_overrides[key])
                print(f"  Override: {attr} = {pruning_overrides[key]} (was {original_values[attr]})")

    try:
        # Build sys.argv to pass flags to main()
        argv = ["discovery"]
        if incremental:
            argv.append("--incremental")
        if force_recompute:
            argv.append("--force-recompute")

        old_argv = sys.argv
        sys.argv = argv
        try:
            from research.discovery.main import main as discovery_main
            discovery_main()
        finally:
            sys.argv = old_argv
    finally:
        # Restore original config values
        if original_values:
            import research.discovery.config as discovery_config
            for attr, value in original_values.items():
                setattr(discovery_config, attr, value)
                print(f"  Restored: {attr} = {value}")

    # Clear cached functions so the UI picks up fresh results
    get_discovery_catalog.clear()
    get_discovery_report.clear()
    get_discovery_registry_summary.clear()

    print("=" * 60)
    print("Feature Discovery Pipeline Complete")
    print("=" * 60)
