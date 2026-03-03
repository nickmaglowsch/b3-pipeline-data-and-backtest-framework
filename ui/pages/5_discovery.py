"""
Feature Discovery Engine Page
"""
from __future__ import annotations

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(page_title="Discovery", page_icon="mag", layout="wide")

st.title("Feature Discovery Engine")
st.caption("Automatic feature generation, IC evaluation, and ranking for alpha research.")

# ── Session state ──────────────────────────────────────────────────────────────
if "job_runner" not in st.session_state:
    from ui.services.job_runner import JobRunner
    st.session_state.job_runner = JobRunner()

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from ui.services.discovery_service import (
        get_discovery_catalog,
        get_discovery_report,
        get_discovery_registry_summary,
        load_feature_data,
        run_discovery_pipeline,
        CATALOG_PATH,
    )
    from ui.components.discovery_charts import (
        plot_ic_bar_chart,
        plot_ic_decay_chart,
        plot_turnover_vs_ic,
        plot_train_test_scatter,
        plot_category_breakdown,
        plot_pruning_funnel,
        plot_feature_timeseries,
        plot_feature_histogram,
    )
    from ui.components.log_stream import render_log_stream
    from ui.services.job_runner import JobStatus
    import pandas as pd
    _LOADED = True
except ImportError as e:
    _LOADED = False
    st.warning(f"Discovery services not yet available: {e}")

if not _LOADED:
    st.info("This page will be fully functional after the discovery service is implemented.")
    st.stop()

runner = st.session_state.job_runner

# ── Status Bar ────────────────────────────────────────────────────────────────
_catalog_mtime = os.path.getmtime(str(CATALOG_PATH)) if CATALOG_PATH.exists() else 0.0
catalog = get_discovery_catalog(catalog_mtime=_catalog_mtime)

if catalog:
    st.success(
        f"Last run: {catalog['generated_at']} | "
        f"{catalog['total_after_pruning']} features from {catalog['total_generated']} generated"
    )
else:
    st.warning("No discovery results found. Run the pipeline to generate features.")

# ── Run Pipeline Expander ─────────────────────────────────────────────────────
with st.expander("Run Discovery Pipeline", expanded=(catalog is None)):
    st.markdown("""
    The **Feature Discovery Engine** automatically generates and evaluates hundreds of alpha signals
    from price, volume, and market data. It computes Information Coefficients (IC) for each signal
    across multiple horizons, then runs a multi-stage pruning pipeline to identify the best
    non-redundant features. The final ranked catalog is used for alpha research and model building.
    """)

    active_job = runner.get_active_job("discovery")

    if active_job and active_job.status == JobStatus.RUNNING:
        st.info("Discovery pipeline is running...")
        render_log_stream("discovery", runner)
    else:
        mode = st.radio(
            "Mode",
            ["Incremental (skip computed)", "Force Recompute (fresh start)"],
            key="discovery_mode",
        )

        col_ic, col_corr, col_feat = st.columns(3)
        with col_ic:
            min_ic = st.number_input(
                "Min IC Threshold",
                value=0.005,
                min_value=0.0,
                max_value=0.1,
                step=0.001,
                format="%.4f",
                key="discovery_min_ic",
            )
        with col_corr:
            max_corr = st.number_input(
                "Max Correlation",
                value=0.90,
                min_value=0.5,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                key="discovery_max_corr",
            )
        with col_feat:
            max_feats = st.number_input(
                "Max Features",
                value=500,
                min_value=10,
                max_value=2000,
                step=10,
                key="discovery_max_feats",
            )

        if st.button("Run Discovery Pipeline", type="primary", key="discovery_run_btn"):
            incremental = "Incremental" in mode
            force_recompute = "Force Recompute" in mode
            pruning_overrides = {
                "min_ic_threshold": min_ic,
                "max_correlation": max_corr,
                "max_features": int(max_feats),
            }
            job_id = runner.submit(
                "discovery",
                run_discovery_pipeline,
                incremental=incremental,
                force_recompute=force_recompute,
                pruning_overrides=pruning_overrides,
            )
            st.success(f"Discovery job started (ID: {job_id})")
            st.rerun()

        if active_job and active_job.status == JobStatus.COMPLETED:
            st.success("Last run completed successfully.")
            render_log_stream("discovery", runner)
        elif active_job and active_job.status == JobStatus.FAILED:
            st.error(f"Last run failed: {active_job.error}")
            render_log_stream("discovery", runner)

# ── Results Tabs ──────────────────────────────────────────────────────────────
if catalog is not None:
    st.divider()
    st.subheader("Discovery Results")

    tab_overview, tab_features, tab_analysis, tab_detail, tab_report = st.tabs([
        "Overview", "Top Features", "Analysis", "Feature Detail", "Report",
    ])

    # ── Overview Tab ──────────────────────────────────────────────────────────
    with tab_overview:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Features Generated", catalog["total_generated"])
        col2.metric("After Pruning", catalog["total_after_pruning"])
        col3.metric("Primary Horizon", catalog["primary_horizon"])
        date_range = catalog.get("evaluation_date_range", {})
        col4.metric(
            "Date Range",
            f"{date_range.get('start', '?')} to {date_range.get('end', '?')}",
        )

        col_funnel, col_cats = st.columns(2)
        with col_funnel:
            try:
                report_text = get_discovery_report()
                fig = plot_pruning_funnel(catalog, report_text=report_text)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")

        with col_cats:
            try:
                fig = plot_category_breakdown(catalog)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")

        if date_range:
            st.info(
                f"Train period: {date_range.get('start')} to {date_range.get('train_end')} | "
                f"Test period: {date_range.get('test_start')} to {date_range.get('end')}"
            )

    # ── Top Features Tab ──────────────────────────────────────────────────────
    with tab_features:
        col_hz, col_topn = st.columns(2)
        with col_hz:
            feat_horizon = st.selectbox(
                "Horizon",
                options=["fwd_5d", "fwd_10d", "fwd_20d", "fwd_60d"],
                index=2,
                key="feat_horizon",
            )
        with col_topn:
            feat_top_n = st.slider(
                "Top N",
                min_value=10,
                max_value=100,
                value=30,
                step=5,
                key="feat_top_n",
            )

        try:
            fig = plot_ic_bar_chart(
                catalog["features"],
                horizon=feat_horizon,
                top_n=feat_top_n,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

        # Category filter
        cat_filter = st.multiselect(
            "Filter by Category",
            options=list(catalog.get("category_summary", {}).keys()),
            default=[],
            key="cat_filter",
        )

        # Build feature table
        table_rows = []
        for f in catalog["features"]:
            metrics = f.get("metrics", {}).get(feat_horizon, {})
            if not metrics:
                continue
            if cat_filter and f.get("category") not in cat_filter:
                continue
            table_rows.append({
                "Rank": f.get("rank"),
                "Feature ID": f["id"],
                "Category": f.get("category", ""),
                "Level": f.get("level", 0),
                "Formula": f.get("formula_human", ""),
                "IC_IR": metrics.get("ic_ir", 0.0),
                "Mean IC": metrics.get("mean_ic", 0.0),
                "Pct Positive IC": metrics.get("pct_positive_ic", 0.0) * 100,
                "Turnover": metrics.get("turnover", f.get("turnover", 0.0)),
            })

        if table_rows:
            feat_df = pd.DataFrame(table_rows)
            feat_df = feat_df.reindex(
                feat_df["IC_IR"].abs().sort_values(ascending=False).index
            )
            st.dataframe(
                feat_df,
                column_config={
                    "IC_IR": st.column_config.NumberColumn(format="%.4f"),
                    "Mean IC": st.column_config.NumberColumn(format="%.6f"),
                    "Pct Positive IC": st.column_config.NumberColumn(
                        "% Positive IC", format="%.1f"
                    ),
                    "Turnover": st.column_config.NumberColumn(format="%.4f"),
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No features match the current filter.")

    # ── Analysis Tab ──────────────────────────────────────────────────────────
    with tab_analysis:
        analysis_horizon = st.selectbox(
            "Horizon",
            options=["fwd_5d", "fwd_10d", "fwd_20d", "fwd_60d"],
            index=2,
            key="analysis_horizon",
        )

        col_scatter1, col_scatter2 = st.columns(2)
        with col_scatter1:
            try:
                fig = plot_turnover_vs_ic(catalog["features"], horizon=analysis_horizon)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")

        with col_scatter2:
            try:
                fig = plot_train_test_scatter(catalog["features"], horizon=analysis_horizon)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")

        try:
            fig = plot_ic_decay_chart(
                catalog["features"],
                horizon=analysis_horizon,
                top_n=15,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    # ── Feature Detail Tab ────────────────────────────────────────────────────
    with tab_detail:
        features = catalog["features"]

        if not features:
            st.info("No features in catalog.")
        else:
            # Build options: display formula_human, keyed by id
            feature_options = {
                f["id"]: f"{f['rank']}. {f['formula_human']} ({f['category']})"
                for f in features
            }

            selected_feature_id = st.selectbox(
                "Select a Feature",
                options=list(feature_options.keys()),
                format_func=lambda x: feature_options[x],
                key="detail_feature_select",
            )

            selected_feature = next(
                (f for f in features if f["id"] == selected_feature_id), None
            )

            if selected_feature is None:
                st.warning("Feature not found in catalog.")
            else:
                # Row 1 -- Feature identity
                col_info, col_rank = st.columns([3, 1])
                with col_info:
                    st.markdown(f"**Feature ID:** `{selected_feature['id']}`")
                    st.markdown(f"**Formula:** {selected_feature['formula_human']}")
                    st.markdown(
                        f"**Category:** {selected_feature['category']} | "
                        f"**Level:** {selected_feature['level']}"
                    )
                with col_rank:
                    st.metric("Rank", f"#{selected_feature['rank']}")

                # Row 2 -- Horizon selector
                available_horizons = list(selected_feature.get("metrics", {}).keys())
                if not available_horizons:
                    st.warning("No metrics available for this feature.")
                else:
                    default_idx = (
                        available_horizons.index("fwd_20d")
                        if "fwd_20d" in available_horizons
                        else 0
                    )
                    selected_horizon = st.selectbox(
                        "Evaluation Horizon",
                        options=available_horizons,
                        index=default_idx,
                        key="detail_horizon",
                    )
                    horizon_metrics = selected_feature["metrics"].get(selected_horizon, {})

                    # Row 3 -- Metric cards (two rows of 5)
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Mean IC", f"{horizon_metrics.get('mean_ic', 0):.6f}")
                    c2.metric("IC_IR", f"{horizon_metrics.get('ic_ir', 0):.4f}")
                    c3.metric("IC t-stat", f"{horizon_metrics.get('ic_t_stat', 0):.2f}")
                    c4.metric(
                        "% Positive IC",
                        f"{horizon_metrics.get('pct_positive_ic', 0) * 100:.1f}%",
                    )
                    c5.metric(
                        "Turnover",
                        f"{horizon_metrics.get('turnover', 0):.4f}",
                    )

                    c6, c7, c8, c9, c10 = st.columns(5)
                    train_ic = horizon_metrics.get("mean_ic_train")
                    test_ic = horizon_metrics.get("mean_ic_test")
                    c6.metric(
                        "Train IC",
                        f"{train_ic:.6f}" if train_ic is not None else "N/A",
                    )
                    c7.metric(
                        "Train IC_IR",
                        f"{horizon_metrics.get('ic_ir_train', 0):.4f}",
                    )
                    c8.metric(
                        "Test IC",
                        f"{test_ic:.6f}" if test_ic is not None else "N/A",
                    )
                    c9.metric(
                        "Test IC_IR",
                        f"{horizon_metrics.get('ic_ir_test', 0):.4f}",
                    )
                    c10.metric(
                        "Obs. Dates",
                        f"{horizon_metrics.get('n_dates', 0):,}",
                    )

                    # Row 4 -- Decay table
                    decay = horizon_metrics.get("decay", {})
                    if decay:
                        st.markdown("**IC Decay (lagged feature):**")
                        decay_cols = st.columns(len(decay) + 1)
                        decay_cols[0].metric(
                            "Lag 0",
                            f"{horizon_metrics.get('mean_ic', 0):.6f}",
                        )
                        for i, (lag, ic_val) in enumerate(decay.items()):
                            decay_cols[i + 1].metric(f"Lag {lag}", f"{ic_val:.6f}")

                    # Row 5 -- Load Parquet and show charts
                    st.divider()
                    st.subheader("Feature Data Exploration")

                    with st.spinner(f"Loading feature data for {selected_feature_id}..."):
                        feature_df = load_feature_data(selected_feature_id)

                    if feature_df is not None and not feature_df.empty:
                        col_ts, col_hist = st.columns(2)

                        with col_ts:
                            try:
                                fig = plot_feature_timeseries(feature_df, selected_feature_id)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Time series chart error: {e}")

                        with col_hist:
                            try:
                                fig = plot_feature_histogram(feature_df, selected_feature_id)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Histogram chart error: {e}")

                        # Data summary
                        with st.expander("Raw Data Summary", expanded=False):
                            st.markdown(
                                f"**Rows:** {len(feature_df):,} | "
                                f"**Tickers:** {feature_df['ticker'].nunique()} | "
                                f"**Dates:** {feature_df['date'].nunique()}"
                            )
                            st.markdown(
                                f"**Date Range:** {feature_df['date'].min()} "
                                f"to {feature_df['date'].max()}"
                            )
                            st.markdown(
                                f"**Value Stats:** "
                                f"mean={feature_df['value'].mean():.4f}, "
                                f"std={feature_df['value'].std():.4f}, "
                                f"min={feature_df['value'].min():.4f}, "
                                f"max={feature_df['value'].max():.4f}"
                            )
                    else:
                        st.warning(
                            f"Feature data not available for '{selected_feature_id}'. "
                            "The Parquet file may not exist in the feature store."
                        )

    # ── Report Tab ────────────────────────────────────────────────────────────
    with tab_report:
        report = get_discovery_report()
        if report:
            st.code(report, language="text")
        else:
            st.info("No discovery report found.")
