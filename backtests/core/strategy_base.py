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
    description="Starting portfolio value in BRL",
    min_value=1_000.0, step=10_000.0,
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
