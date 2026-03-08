"""
Feature store for persistence of computed features and evaluations.
Uses Parquet files for feature values and JSON registry for metadata.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from research.discovery import config


def sanitize_feature_id(name: str) -> str:
    """
    Convert feature name to filesystem-safe string.
    Replaces special characters that could cause filesystem issues.
    """
    name = name.replace("/", "_div_")
    name = name.replace("*", "_mul_")
    name = name.replace(" ", "_")
    return name


def compute_data_hash(data: dict) -> str:
    """
    Compute a fingerprint of the source data.
    Uses shape + date range of adj_close DataFrame plus fundamentals count.
    Returns a hex string.
    """
    adj = data["adj_close"]
    # Fundamentals fingerprint: count of non-NaN cells in f_net_income
    f_ni = data.get("f_net_income")
    fund_count = int(f_ni.count().sum()) if f_ni is not None and not f_ni.empty else 0
    key = f"{adj.shape}|{adj.index.min()}|{adj.index.max()}|{adj.columns.size}|fund:{fund_count}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class FeatureStore:
    """
    Persistent storage for computed features and evaluation results.
    
    Manages:
    1. JSON registry of feature metadata
    2. Parquet files of feature values (long format)
    3. Parquet files of evaluation results
    """
    
    def __init__(self, store_dir: Path = None):
        """
        Initialize store. Load existing registry or create new.
        
        Args:
            store_dir: Override the default feature store directory
        """
        self.store_dir = store_dir or config.FEATURE_STORE_DIR
        self.features_dir = self.store_dir / "features"
        self.evaluations_dir = self.store_dir / "evaluations"
        self.registry_path = self.store_dir / "registry.json"
        
        # Create directories
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize registry
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                self._registry = json.load(f)
        else:
            self._registry = {
                "version": 1,
                "data_hash": None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "features": {},
            }
    
    def compute_data_hash(self, data: dict) -> str:
        """Compute fingerprint of source data."""
        return compute_data_hash(data)
    
    def is_valid(self, data_hash: str) -> bool:
        """
        Check if store is valid for this data hash.
        Returns False if hash does not match (data has changed).
        """
        current_hash = self._registry.get("data_hash")
        if current_hash is None:
            return True  # First run
        return current_hash == data_hash
    
    def invalidate(self) -> None:
        """Clear all cached features and evaluations. Reset registry."""
        # Delete all Parquet files
        for f in self.features_dir.glob("*.parquet"):
            f.unlink()
        for f in self.evaluations_dir.glob("*.parquet"):
            f.unlink()
        
        # Reset registry
        self._registry = {
            "version": 1,
            "data_hash": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "features": {},
        }
        self.save_registry()
    
    def has_feature(self, feature_id: str) -> bool:
        """Check if a feature is already computed and in the registry."""
        return feature_id in self._registry["features"]
    
    def save_feature(
        self,
        feature_id: str,
        df: pd.DataFrame,
        metadata: dict,
    ) -> None:
        """
        Save a computed feature to the store.
        
        Args:
            feature_id: Unique feature identifier
            df: Long-format DataFrame with columns [date, ticker, value]
            metadata: dict with keys: category, level, formula, params
        """
        # Sanitize feature ID for filesystem
        safe_id = sanitize_feature_id(feature_id)
        parquet_path = self.features_dir / f"{safe_id}.parquet"
        
        # Ensure value column is float32
        if "value" in df.columns:
            df["value"] = df["value"].astype("float32")
        
        # Write Parquet
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        
        # Update registry
        self._registry["features"][feature_id] = {
            "id": feature_id,
            "category": metadata.get("category", "unknown"),
            "level": metadata.get("level", 0),
            "formula": metadata.get("formula", ""),
            "params": metadata.get("params", {}),
            "computed_at": datetime.now().isoformat(),
            "parquet_file": f"features/{safe_id}.parquet",
            "evaluation": {},
        }
        self._registry["updated_at"] = datetime.now().isoformat()
    
    def load_feature(self, feature_id: str) -> pd.DataFrame:
        """Load a feature's values from Parquet. Returns long-format DataFrame."""
        if feature_id not in self._registry["features"]:
            raise ValueError(f"Feature {feature_id} not found in store")
        
        parquet_file = self._registry["features"][feature_id]["parquet_file"]
        path = self.store_dir / parquet_file
        
        df = pd.read_parquet(path, engine="pyarrow")
        return df
    
    def load_features_batch(self, feature_ids: list) -> pd.DataFrame:
        """
        Load multiple features and return wide DataFrame (date x ticker x features).
        
        Actually returns a long-format combined DataFrame with a feature_id column.
        This is more practical than true wide format with many feature columns.
        """
        frames = []
        for fid in feature_ids:
            df = self.load_feature(fid)
            df["feature_id"] = fid
            frames.append(df)
        
        return pd.concat(frames, ignore_index=True)
    
    def save_evaluation(self, feature_id: str, evaluation: dict) -> None:
        """Save IC evaluation results for a feature into the registry."""
        if feature_id not in self._registry["features"]:
            raise ValueError(f"Feature {feature_id} not found in store")
        
        self._registry["features"][feature_id]["evaluation"] = evaluation
        self._registry["updated_at"] = datetime.now().isoformat()
    
    def has_evaluation(self, feature_id: str, horizon: str = None) -> bool:
        """
        Check if evaluation results exist for a feature.
        If horizon is provided, check for that specific horizon.
        """
        if feature_id not in self._registry["features"]:
            return False
        
        eval_dict = self._registry["features"][feature_id].get("evaluation", {})
        if not eval_dict:
            return False
        
        if horizon is None:
            return True
        return horizon in eval_dict
    
    def get_all_evaluations(self) -> pd.DataFrame:
        """
        Return a DataFrame of all feature evaluations.
        Columns: feature_id, category, level, formula, mean_ic_5d, ic_ir_5d,
                 mean_ic_20d, ic_ir_20d, ... (one column per metric per horizon)
        """
        rows = []
        for feature_id, feature_data in self._registry["features"].items():
            row = {
                "feature_id": feature_id,
                "category": feature_data.get("category"),
                "level": feature_data.get("level"),
                "formula": feature_data.get("formula"),
            }
            
            # Flatten evaluation metrics
            evaluation = feature_data.get("evaluation", {})
            for horizon, metrics in evaluation.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        col = f"{metric_name}_{horizon}"
                        row[col] = value
                else:
                    # Scalar metric (e.g., turnover)
                    row[horizon] = metrics
            
            rows.append(row)
        
        if not rows:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "feature_id", "category", "level", "formula"
            ])
        
        return pd.DataFrame(rows)

    def save_ic_timeseries_batch(self, ic_records: list[dict]) -> None:
        """
        Save a batch of IC time series records to consolidated Parquet.

        Args:
            ic_records: list of dicts with keys: feature_id, horizon, date, ic
        """
        if not ic_records:
            return

        # Convert to DataFrame
        new_df = pd.DataFrame(ic_records)
        ic_path = self.evaluations_dir / config.IC_TIMESERIES_FILE

        # If file exists, read, concatenate, deduplicate
        if ic_path.exists():
            existing_df = pd.read_parquet(ic_path, engine="pyarrow")
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        # Deduplicate on (feature_id, horizon, date), keep last
        combined_df = combined_df.drop_duplicates(
            subset=["feature_id", "horizon", "date"], keep="last"
        )

        # Ensure correct dtypes
        combined_df["ic"] = combined_df["ic"].astype("float32")
        combined_df["date"] = pd.to_datetime(combined_df["date"])

        # Write to Parquet
        combined_df.to_parquet(ic_path, engine="pyarrow", index=False)

    def load_ic_timeseries(
        self, feature_ids: list[str] = None, horizon: str = None
    ) -> pd.DataFrame:
        """
        Load IC time series from consolidated Parquet.

        Args:
            feature_ids: optional list of feature IDs to filter
            horizon: optional horizon string (e.g., "fwd_20d") to filter

        Returns:
            DataFrame with columns [feature_id, horizon, date, ic]
        """
        ic_path = self.evaluations_dir / config.IC_TIMESERIES_FILE

        if not ic_path.exists():
            return pd.DataFrame(columns=["feature_id", "horizon", "date", "ic"])

        df = pd.read_parquet(ic_path, engine="pyarrow")

        # Filter by feature_ids
        if feature_ids is not None:
            df = df[df["feature_id"].isin(feature_ids)]

        # Filter by horizon
        if horizon is not None:
            df = df[df["horizon"] == horizon]

        return df

    def get_registry(self) -> dict:
        """Return the full registry dict."""
        return self._registry
    
    def save_registry(self) -> None:
        """Write registry to JSON file."""
        self._registry["updated_at"] = datetime.now().isoformat()
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)
    
    def feature_count(self) -> int:
        """Return number of features in the store."""
        return len(self._registry["features"])
