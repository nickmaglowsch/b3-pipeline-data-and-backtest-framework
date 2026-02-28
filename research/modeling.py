"""
Modeling and evaluation for B3 feature importance study.
Trains RF and XGB classifiers, evaluates on test set, extracts feature importance.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

from research import config
from research.targets import get_dataset_for_target


def time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    train_fraction: float = None,
) -> tuple:
    """
    Split data chronologically by date.

    Args:
        X: feature DataFrame (n_samples x n_features)
        y: target Series (n_samples,)
        meta: DataFrame with 'date' and 'ticker' columns (n_samples,)
        train_fraction: fraction of unique dates for training (default from config)

    Returns:
        (X_train, X_test, y_train, y_test, meta_train, meta_test)
    """
    if train_fraction is None:
        train_fraction = config.TRAIN_FRACTION

    dates = meta["date"].sort_values().unique()
    n_train_dates = int(len(dates) * train_fraction)
    cutoff_date = dates[n_train_dates - 1]

    train_mask = meta["date"] <= cutoff_date
    test_mask = meta["date"] > cutoff_date

    print(f"  Train period: {dates[0].strftime('%Y-%m-%d')} to {cutoff_date.strftime('%Y-%m-%d')} ({train_mask.sum():,} samples)")
    print(f"  Test period:  {dates[n_train_dates].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')} ({test_mask.sum():,} samples)")
    print(f"  Train class balance: {y[train_mask].mean():.3f}")
    print(f"  Test class balance:  {y[test_mask].mean():.3f}")

    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
        meta[train_mask], meta[test_mask],
    )


def train_model(model, X_train, y_train, model_name: str):
    """
    Train a model and print timing info.
    Returns the fitted model.
    """
    print(f"  Training {model_name}...")
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  {model_name} trained in {elapsed:.1f}s")
    return model


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a trained model on test data.

    Returns dict with:
        "accuracy", "roc_auc", "precision", "recall", "f1",
        "confusion_matrix" (as list of lists for JSON serialization)
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\n  {model_name} Test Results:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")

    return metrics


def extract_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list,
    model_name: str,
    importance_type: str,  # "gini" or "gain"
) -> pd.DataFrame:
    """
    Extract both built-in and permutation feature importance.

    Returns DataFrame with columns:
        feature, builtin_importance, permutation_importance_mean, permutation_importance_std

    Sorted by builtin_importance descending.
    """
    # Built-in importance
    builtin = model.feature_importances_

    # Permutation importance on test set
    print(f"  Computing permutation importance for {model_name}...")
    perm_result = permutation_importance(
        model, X_test, y_test,
        n_repeats=config.PERM_N_REPEATS,
        random_state=config.PERM_RANDOM_STATE,
        n_jobs=2,  # -1 duplicates data per worker, blowing memory
    )

    col_name = f"{importance_type}_importance"
    df = pd.DataFrame({
        "feature": feature_names,
        col_name: builtin,
        "permutation_importance_mean": perm_result.importances_mean,
        "permutation_importance_std": perm_result.importances_std,
    })

    df = df.sort_values(col_name, ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    return df


def run_experiment(
    full_df: pd.DataFrame,
    target_col: str,
    feature_names: list,
    experiment_name: str,
) -> dict:
    """
    Run a single experiment: split, train both models, evaluate, extract importance.

    Returns dict with:
        "metrics": {model_name: metrics_dict}
        "importance": {model_name: importance_df}
        "split_info": info about train/test dates and sizes
    """
    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Target: {target_col}")
    print(f"{'='*60}")

    # 1. Extract X, y, meta for this target
    X, y, meta = get_dataset_for_target(full_df, target_col, feature_names)
    print(f"  Total samples: {len(X):,}")
    print(f"  Class balance: {y.mean():.3f} (fraction positive)")

    # 2. Time-series split
    X_train, X_test, y_train, y_test, meta_train, meta_test = time_series_split(X, y, meta)

    # 3. Train RF + XGB concurrently (both release GIL during .fit())
    rf_model = RandomForestClassifier(**config.RF_PARAMS)
    xgb_model = XGBClassifier(**config.XGB_PARAMS)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rf_future = pool.submit(train_model, rf_model, X_train, y_train, "RandomForest")
        xgb_future = pool.submit(train_model, xgb_model, X_train, y_train, "XGBoost")
        rf = rf_future.result()
        xgb = xgb_future.result()

    # 4. Evaluate (fast, no need to parallelize)
    rf_metrics = evaluate_model(rf, X_test, y_test, "RandomForest")
    xgb_metrics = evaluate_model(xgb, X_test, y_test, "XGBoost")

    # 5. Feature importance -- run RF and XGB permutation importance concurrently
    with ThreadPoolExecutor(max_workers=2) as pool:
        rf_imp_future = pool.submit(
            extract_importance, rf, X_test, y_test, feature_names, "RandomForest", "gini"
        )
        xgb_imp_future = pool.submit(
            extract_importance, xgb, X_test, y_test, feature_names, "XGBoost", "gain"
        )
        rf_importance = rf_imp_future.result()
        xgb_importance = xgb_imp_future.result()

    # 6. Split info for reporting
    train_dates = meta_train["date"].sort_values()
    test_dates = meta_test["date"].sort_values()
    split_info = {
        "train_start": str(train_dates.min().date()),
        "train_end": str(train_dates.max().date()),
        "test_start": str(test_dates.min().date()),
        "test_end": str(test_dates.max().date()),
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    return {
        "metrics": {"RandomForest": rf_metrics, "XGBoost": xgb_metrics},
        "importance": {"RandomForest": rf_importance, "XGBoost": xgb_importance},
        "split_info": split_info,
    }


def run_all_experiments(full_df: pd.DataFrame, feature_names: list) -> dict:
    """
    Run experiments for all three targets:
    1. target_20d (primary)
    2. target_60d (robustness A)
    3. target_20d_median (robustness B)

    Returns dict keyed by experiment name.
    """
    experiments = [
        ("target_20d", "target_20d", "20-Day Forward Return > 0"),
        ("target_60d", "target_60d", "60-Day Forward Return > 0"),
        ("target_20d_median", "target_20d_median", "20-Day Forward Return > Median"),
    ]

    all_results = {}
    for exp_key, target_col, exp_name in experiments:
        result = run_experiment(full_df, target_col, feature_names, exp_name)
        all_results[exp_key] = result

    return all_results


def save_results(all_results: dict, feature_names: list) -> None:
    """
    Save results to output directory.

    Creates:
    - importance_results.csv: combined importance rankings across all experiments
    - model_metrics.json: all metrics for all experiments
    """
    # Build importance CSV
    rows = []
    for exp_name, exp_data in all_results.items():
        for model_name, imp_df in exp_data["importance"].items():
            imp_copy = imp_df.copy()
            imp_copy["experiment"] = exp_name
            imp_copy["model"] = model_name
            rows.append(imp_copy)

    importance_df = pd.concat(rows, ignore_index=True)

    # Normalize: rename gini_importance/gain_importance to builtin_importance
    if "gini_importance" in importance_df.columns and "gain_importance" in importance_df.columns:
        importance_df["builtin_importance"] = importance_df["gini_importance"].fillna(
            importance_df["gain_importance"]
        )
    elif "gini_importance" in importance_df.columns:
        importance_df["builtin_importance"] = importance_df["gini_importance"]
    elif "gain_importance" in importance_df.columns:
        importance_df["builtin_importance"] = importance_df["gain_importance"]

    csv_path = config.OUTPUT_DIR / config.IMPORTANCE_CSV
    importance_df.to_csv(csv_path, index=False)
    print(f"  Saved importance results to {csv_path}")

    # Build metrics JSON
    metrics_dict = {}
    for exp_name, exp_data in all_results.items():
        metrics_dict[exp_name] = exp_data["metrics"]

    # Add split info from primary experiment
    if "target_20d" in all_results:
        metrics_dict["split_info"] = all_results["target_20d"]["split_info"]

    json_path = config.OUTPUT_DIR / config.METRICS_JSON
    with open(json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Saved model metrics to {json_path}")
