"""
B3 Data Pipeline & Quantitative Research -- Streamlit Application Entry Point

Launch with:
    streamlit run ui/app.py

from the project root directory. Access at http://localhost:8501
"""
from __future__ import annotations

import sys
import os

# Ensure project root is on the path so all imports work
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="B3 Data Pipeline",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialization ─────────────────────────────────────────────
if "job_runner" not in st.session_state:
    from ui.services.job_runner import JobRunner
    st.session_state.job_runner = JobRunner()

if "last_backtest_result" not in st.session_state:
    st.session_state.last_backtest_result = None

# ── Main landing page ─────────────────────────────────────────────────────────
st.title("B3 Data Pipeline & Quantitative Research")

st.markdown("""
Welcome to the B3 data pipeline management UI. Use the sidebar to navigate:

- **Pipeline**: Manage the data pipeline, view database stats, trigger updates
- **Backtest Runner**: Configure and run backtest strategies with custom parameters
- **Dashboard**: Browse and compare backtest results
- **Research**: View ML feature importance research results
""")

st.divider()

# ── Quick stats ───────────────────────────────────────────────────────────────
st.subheader("Database Overview")

try:
    from ui.services.pipeline_service import get_db_stats
    with st.spinner("Loading database stats..."):
        stats = get_db_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Price Records", f"{stats['total_prices']:,}")
    col2.metric("Tickers", f"{stats['total_tickers']:,}")
    date_min, date_max = stats["date_range"]
    col3.metric("Date Range", f"{date_min} to {date_max}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Corporate Actions", f"{stats['total_corporate_actions']:,}")
    col5.metric("Stock Actions", f"{stats['total_stock_actions']:,}")
    col6.metric("Detected Splits", f"{stats.get('total_detected_splits', 0):,}")
except FileNotFoundError:
    st.info("No database found. Go to the **Pipeline** page to set up the data.")
except Exception as e:
    st.info(f"No database available. Go to the **Pipeline** page to set up the data.")

st.divider()

st.sidebar.markdown("### Navigation")
st.sidebar.markdown("""
Use the pages above to navigate between modules:

1. **Pipeline** -- Data management
2. **Backtest Runner** -- Run strategies
3. **Dashboard** -- Browse results
4. **Research** -- ML insights
""")

st.sidebar.divider()
st.sidebar.caption("B3 Data Pipeline UI | Python 3.9 | Streamlit")
