"""
Dynamic Parameter Form Component
==================================
Generates Streamlit form inputs dynamically from ParameterSpec lists.
Groups parameters into Common and Strategy-specific sections.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

import pandas as pd
import streamlit as st


# Parameter names that are considered "common" and grouped together at top.
COMMON_PARAM_NAMES = {
    "start_date", "end_date", "initial_capital",
    "tax_rate", "slippage", "rebalance_freq", "min_adtv",
}


def render_parameter_form(
    specs: list,  # list[ParameterSpec]
    strategy_name: str,
    defaults: Optional[dict] = None,
) -> dict:
    """
    Render a parameter form from a list of ParameterSpec objects.

    Args:
        specs:         List of ParameterSpec from the strategy.
        strategy_name: Used as a key prefix to avoid Streamlit key collisions.
        defaults:      Optional override dict (e.g., from a previous run).

    Returns:
        Dict of {param_name: value} ready to pass to generate_signals().
    """
    if defaults is None:
        defaults = {}

    params: dict[str, Any] = {}

    common_specs = [s for s in specs if s.name in COMMON_PARAM_NAMES]
    specific_specs = [s for s in specs if s.name not in COMMON_PARAM_NAMES]

    # ── Common parameters ─────────────────────────────────────────────────────
    if common_specs:
        st.subheader("Common Parameters")

        date_specs = [s for s in common_specs if s.param_type == "date"]
        other_common = [s for s in common_specs if s.param_type != "date"]

        if date_specs:
            cols = st.columns(len(date_specs))
            for col, spec in zip(cols, date_specs):
                with col:
                    params[spec.name] = _render_single_param(
                        spec, strategy_name, defaults.get(spec.name)
                    )

        if other_common:
            cols = st.columns(min(len(other_common), 3))
            for i, spec in enumerate(other_common):
                with cols[i % len(cols)]:
                    params[spec.name] = _render_single_param(
                        spec, strategy_name, defaults.get(spec.name)
                    )

    # ── Strategy-specific parameters ──────────────────────────────────────────
    if specific_specs:
        st.subheader("Strategy Parameters")
        n_cols = min(len(specific_specs), 2)
        cols = st.columns(n_cols) if n_cols > 1 else [st.container()]
        for i, spec in enumerate(specific_specs):
            with cols[i % n_cols]:
                params[spec.name] = _render_single_param(
                    spec, strategy_name, defaults.get(spec.name)
                )

    return params


def _render_single_param(spec: Any, strategy_name: str, override_default: Any = None) -> Any:
    """Render a single parameter input and return its current value."""
    key = f"param_{strategy_name}_{spec.name}"
    default = override_default if override_default is not None else spec.default

    if spec.param_type == "int":
        return st.number_input(
            spec.label,
            value=int(default),
            min_value=int(spec.min_value) if spec.min_value is not None else None,
            max_value=int(spec.max_value) if spec.max_value is not None else None,
            step=int(spec.step) if spec.step else 1,
            help=spec.description or None,
            key=key,
        )

    elif spec.param_type == "float":
        step = float(spec.step) if spec.step else 0.01
        if step < 0.001:
            fmt = "%.4f"
        elif step < 0.1:
            fmt = "%.3f"
        else:
            fmt = "%.2f"
        return st.number_input(
            spec.label,
            value=float(default),
            min_value=float(spec.min_value) if spec.min_value is not None else None,
            max_value=float(spec.max_value) if spec.max_value is not None else None,
            step=step,
            format=fmt,
            help=spec.description or None,
            key=key,
        )

    elif spec.param_type == "date":
        if default == "today" or default is None:
            default_val = date.today()
        elif isinstance(default, str):
            try:
                default_val = pd.to_datetime(default).date()
            except Exception:
                default_val = date.today()
        elif isinstance(default, datetime):
            default_val = default.date()
        elif isinstance(default, date):
            default_val = default
        else:
            default_val = date.today()

        selected = st.date_input(
            spec.label,
            value=default_val,
            help=spec.description or None,
            key=key,
        )
        return selected.isoformat()

    elif spec.param_type == "choice":
        choices = spec.choices or []
        idx = choices.index(default) if default in choices else 0
        return st.selectbox(
            spec.label,
            options=choices,
            index=idx,
            help=spec.description or None,
            key=key,
        )

    elif spec.param_type == "str":
        return st.text_input(
            spec.label,
            value=str(default),
            help=spec.description or None,
            key=key,
        )

    else:
        st.warning(f"Unknown parameter type '{spec.param_type}' for {spec.name!r}")
        return default


def render_reset_button(strategy_name: str, specs: list) -> bool:
    """
    Render a 'Reset to Defaults' button.
    When clicked, deletes all parameter session state keys for this strategy.

    Returns:
        True if the button was clicked.
    """
    if st.button("Reset to Defaults", key=f"reset_{strategy_name}"):
        for spec in specs:
            key = f"param_{strategy_name}_{spec.name}"
            if key in st.session_state:
                del st.session_state[key]
        return True
    return False
