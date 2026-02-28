"""
ML Research Viewer Page -- implemented fully.
"""
from __future__ import annotations

import sys
import os
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Research", page_icon="microscope", layout="wide")

st.title("ML Research -- Feature Importance Discovery")
st.caption("Explore ML-driven feature importance for B3 return prediction.")

# ── Session state ─────────────────────────────────────────────────────────────
if "job_runner" not in st.session_state:
    from ui.services.job_runner import JobRunner
    st.session_state.job_runner = JobRunner()

try:
    from ui.services.research_service import get_research_results, run_research_pipeline
    from ui.components.log_stream import render_log_stream
    from ui.services.job_runner import JobStatus
    from ui.components.charts import PALETTE, _apply_dark_theme
    import plotly.graph_objects as go
    _LOADED = True
except ImportError as e:
    _LOADED = False
    st.warning(f"Research services not yet available: {e}")

if not _LOADED:
    st.info("This page will be fully functional after all task implementations are complete.")
    st.stop()


def plot_feature_importance(importance_df: pd.DataFrame, model: str, target: str, top_n: int = 15) -> go.Figure:
    """Create an interactive horizontal bar chart of feature importance."""
    model_data = importance_df[
        (importance_df["model"] == model) &
        (importance_df["target"] == target)
    ].copy()

    if model_data.empty:
        model_data = importance_df[importance_df["model"] == model].copy()

    if "permutation_importance_mean" in model_data.columns:
        sort_col = "permutation_importance_mean"
        err_col = "permutation_importance_std"
    elif "importance" in model_data.columns:
        sort_col = "importance"
        err_col = None
    else:
        sort_col = model_data.columns[-1]
        err_col = None

    model_data = model_data.nlargest(top_n, sort_col)

    error_x = None
    if err_col and err_col in model_data.columns:
        error_x = dict(type="data", array=model_data[err_col].fillna(0).tolist())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=model_data[sort_col],
        y=model_data["feature"],
        orientation="h",
        marker_color=PALETTE["pretax"],
        error_x=error_x,
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Top {top_n} Features -- {model} ({target})",
        xaxis_title="Importance",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
    )
    return _apply_dark_theme(fig)


runner = st.session_state.job_runner

# ── Status Bar ────────────────────────────────────────────────────────────────
results = get_research_results()

if results:
    ts = datetime.fromtimestamp(results["last_run"]).strftime("%Y-%m-%d %H:%M:%S")
    st.success(f"Last run: {ts}")
else:
    st.warning("No results found. Run the research pipeline to generate results.")

st.divider()

# ── Run Pipeline ──────────────────────────────────────────────────────────────
with st.expander("Run Research Pipeline", expanded=(results is None)):
    st.markdown("""
    The research pipeline runs a feature importance study using **RandomForest** and **XGBoost**
    on 19 engineered features with 3 binary classification targets (20d, 60d, 20d_median returns).
    It takes approximately 5-10 minutes on a full dataset.
    """)

    active_job = runner.get_active_job("research")
    if active_job and active_job.status == JobStatus.RUNNING:
        st.info("Research pipeline is running...")
        render_log_stream("research", runner)
    else:
        if st.button("Run Research Pipeline", type="primary"):
            job_id = runner.submit("research", run_research_pipeline)
            st.success(f"Research job started (ID: {job_id})")
            st.rerun()

        if active_job and active_job.status == JobStatus.COMPLETED:
            st.success("Last run completed successfully.")
            render_log_stream("research", runner)
        elif active_job and active_job.status == JobStatus.FAILED:
            st.error(f"Last run failed: {active_job.error}")
            render_log_stream("research", runner)

# ── Results Tabs ──────────────────────────────────────────────────────────────
if results:
    st.divider()
    st.subheader("Research Results")

    tab_imp, tab_perf, tab_summary, tab_raw = st.tabs([
        "Feature Importance", "Model Performance", "Summary Report", "Raw Data",
    ])

    # ── Feature Importance ────────────────────────────────────────────────────
    with tab_imp:
        importance_df = results.get("importance")
        if importance_df is not None:
            col1, col2 = st.columns(2)
            models = importance_df["model"].unique().tolist() if "model" in importance_df.columns else ["RandomForest"]
            targets = importance_df["target"].unique().tolist() if "target" in importance_df.columns else ["target_20d"]

            with col1:
                selected_model = st.selectbox("Model", options=models, key="imp_model")
            with col2:
                selected_target = st.selectbox("Target", options=targets, key="imp_target")

            try:
                fig = plot_feature_importance(importance_df, selected_model, selected_target)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")

            # Show existing PNG as comparison
            png_key = "png_feature_importance_top15.png"
            if png_key in results:
                with st.expander("Original PNG (comparison)", expanded=False):
                    st.image(results[png_key])
        else:
            st.info("No importance data available.")

    # ── Model Performance ─────────────────────────────────────────────────────
    with tab_perf:
        metrics_data = results.get("metrics")
        if metrics_data:
            rows = []
            for model_name, targets_data in metrics_data.items():
                if isinstance(targets_data, dict):
                    for target_name, metric_vals in targets_data.items():
                        if isinstance(metric_vals, dict):
                            row = {"Model": model_name, "Target": target_name}
                            row.update(metric_vals)
                            rows.append(row)
            if rows:
                metrics_df = pd.DataFrame(rows)
                st.dataframe(
                    metrics_df,
                    column_config={
                        "Accuracy": st.column_config.NumberColumn(format="%.3f"),
                        "ROC AUC": st.column_config.NumberColumn(format="%.3f"),
                        "Precision": st.column_config.NumberColumn(format="%.3f"),
                        "Recall": st.column_config.NumberColumn(format="%.3f"),
                        "F1": st.column_config.NumberColumn(format="%.3f"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.json(metrics_data)
        else:
            st.info("No model metrics available.")

    # ── Summary Report ────────────────────────────────────────────────────────
    with tab_summary:
        summary = results.get("summary")
        if summary:
            st.text(summary)
        else:
            st.info("No summary report found.")

    # ── Raw Data ──────────────────────────────────────────────────────────────
    with tab_raw:
        importance_df = results.get("importance")
        if importance_df is not None:
            st.caption("importance_results.csv")
            st.dataframe(importance_df, use_container_width=True)
        else:
            st.info("No raw importance CSV available.")

        # Robustness PNG
        png_key = "png_robustness_comparison.png"
        if png_key in results:
            st.subheader("Robustness Comparison")
            st.image(results[png_key])
