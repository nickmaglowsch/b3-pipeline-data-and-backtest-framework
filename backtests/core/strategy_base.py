"""
Strategy Base Class and Parameter Specification
================================================
Defines the abstract base class for all backtest strategies and the
ParameterSpec descriptor used to build dynamic UI forms.

All strategy plugins live in backtests/strategies/ and extend StrategyBase.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd


# ── Parameter Specification ───────────────────────────────────────────────────

class ParameterSpec:
    """Describes a single strategy parameter for UI form generation."""

    def __init__(
        self,
        name: str,
        label: str,
        param_type: str,  # "int", "float", "str", "date", "choice"
        default: Any,
        description: str = "",
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None,
        step: Optional[Any] = None,
        choices: Optional[list] = None,
    ):
        self.name = name
        self.label = label
        self.param_type = param_type
        self.default = default
        self.description = description
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.choices = choices

    def __repr__(self) -> str:
        return (
            f"ParameterSpec(name={self.name!r}, type={self.param_type!r}, "
            f"default={self.default!r})"
        )


# ── Selection helpers ─────────────────────────────────────────────────────────

def keep_most_liquid_per_root(scores: pd.Series, adtv_row: pd.Series) -> pd.Series:
    """Collapse multi-share-class companies to a single ticker before ranking.

    For each 4-char company root, keep only the ticker with the highest ADTV
    (e.g. PETR4 over PETR3). Company fundamentals are now broadcast onto every
    share class, so without this a fundamentals ranker could select PETR3 AND
    PETR4 as two "separate" names — double exposure to one company. Ties and
    missing-ADTV rows fall back to first-seen. Returns ``scores`` unchanged when
    empty; the returned order is arbitrary (callers re-sort via nlargest/nsmallest).
    """
    if scores.empty:
        return scores
    roots = [str(t)[:4] for t in scores.index]
    liq = adtv_row.reindex(scores.index).fillna(-1.0)   # unknown ADTV -> lowest
    keep = liq.groupby(roots).idxmax()                  # one ticker per root
    return scores.loc[keep.values]


def dedup_target_weights(tw: pd.DataFrame, adtv: pd.DataFrame) -> pd.DataFrame:
    """Global safety net applied to EVERY strategy's target weights before the sim.

    For each rebalance row, weights on tickers sharing a 4-char company root
    (e.g. PETR3 + PETR4) are merged onto that row's highest-ADTV ticker, so no
    company is ever held via two share classes. Merging (not dropping) preserves
    the row's weight sum — no vaporized weight. Special assets like CDI_ASSET /
    IBOV have unique roots and are never touched. No-op when a strategy already
    holds one ticker per company (e.g. rankers that pre-dedup their pool).
    """
    groups: dict[str, list] = {}
    for c in tw.columns:
        groups.setdefault(str(c)[:4], []).append(c)
    dup_groups = {r: cols for r, cols in groups.items() if len(cols) > 1}
    if not dup_groups:
        return tw

    out = tw.copy()
    for cols in dup_groups.values():
        sub = out[cols]
        multi = (sub != 0).sum(axis=1) > 1          # rows holding >1 class of this company
        for dt in out.index[multi]:
            held = sub.loc[dt]
            held = held[held != 0]
            liq = adtv.loc[dt, held.index] if dt in adtv.index else pd.Series(dtype=float)
            keep = liq.reindex(held.index).fillna(-1.0).idxmax()
            out.loc[dt, held.index] = 0.0
            out.loc[dt, keep] = held.sum()          # consolidate exposure, keep sum
    return out


# ── Common Parameter Specs ────────────────────────────────────────────────────
# Reusable specs that most strategies include.

COMMON_START_DATE = ParameterSpec(
    "start_date", "Start Date", "date", "2005-01-01",
    description="Backtest start date",
)
COMMON_END_DATE = ParameterSpec(
    "end_date", "End Date", "date", "today",
    description="Backtest end date ('today' for current date)",
)
COMMON_INITIAL_CAPITAL = ParameterSpec(
    "initial_capital", "Initial Capital (BRL)", "float", 100_000.0,
    description="Starting portfolio value in BRL. 0 = pure DCA, the book is "
                "built entirely from the monthly buy-ins.",
    min_value=0.0, step=10_000.0,
)
COMMON_TAX_RATE = ParameterSpec(
    "tax_rate", "Tax Rate", "float", 0.15,
    description="Capital gains tax rate applied to realised profits",
    min_value=0.0, max_value=0.30, step=0.01,
)
COMMON_SLIPPAGE = ParameterSpec(
    "slippage", "Slippage", "float", 0.001,
    description="Round-trip transaction cost fraction per trade",
    min_value=0.0, max_value=0.01, step=0.0005,
)
COMMON_MIN_ADTV = ParameterSpec(
    "min_adtv", "Min ADTV (BRL)", "float", 1_000_000.0,
    description="Minimum average daily trading volume (BRL) for a stock to be eligible",
    min_value=0.0, step=100_000.0,
)
COMMON_REBALANCE_FREQ = ParameterSpec(
    "rebalance_freq", "Rebalance Frequency", "choice", "ME",
    description="Portfolio rebalancing frequency",
    choices=["ME", "QE", "W-FRI"],
)
COMMON_CONTRIBUTION = ParameterSpec(
    "contribution", "Monthly Buy-In (BRL)", "float", 0.0,
    description="Periodic buy-in (aporte) added every calendar month and "
                "allocated by the rebalance — the holding farthest below its "
                "target gets the most. Negative = withdrawal. 0 = lump sum only.",
    step=500.0,
)
COMMON_MONTHLY_SALES_EXEMPTION = ParameterSpec(
    "monthly_sales_exemption", "Monthly Sales Exemption (BRL)", "float", 20_000.0,
    description="Brazilian tax exemption threshold for monthly sales (R$20K)",
    min_value=0.0, step=5_000.0,
)


# ── Abstract Base Class ───────────────────────────────────────────────────────

class StrategyBase(ABC):
    """Abstract base class for all backtest strategy plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique name for the strategy (e.g. 'LowVol')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description shown in the UI."""
        ...

    @abstractmethod
    def get_parameter_specs(self) -> list[ParameterSpec]:
        """
        Return the list of configurable parameters with their UI specs.
        Used by the dynamic parameter form component.
        """
        ...

    def get_default_parameters(self) -> dict:
        """Return {param_name: default_value} for all parameters."""
        return {spec.name: spec.default for spec in self.get_parameter_specs()}

    @abstractmethod
    def generate_signals(
        self, shared_data: dict, params: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate trading signals and target weights.

        Args:
            shared_data: Dict of precomputed DataFrames from build_shared_data().
            params: Parameter dict (from get_default_parameters or UI form).

        Returns:
            Tuple of (returns_matrix, target_weights).
            Both are DataFrames with DatetimeIndex rows and ticker columns.
            target_weights rows should sum to <= 1.0.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
