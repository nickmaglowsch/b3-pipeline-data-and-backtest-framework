"""
Strategy: QDL-Equity — QDL with the CDI vol-target overlay removed.
===================================================================
Ablation of QDL: the exact same Quality-Dividend Low-Vol stock sleeve (regular
dividend payers + positive PIT earnings + the 20 lowest-vol names, inverse-vol
weighted), but ALWAYS fully invested — no CDI de-risking dial.

Mechanically it is QDL with target_vol -> infinity, so the overlay
`eq_frac = clip(target_vol / realized_sleeve_vol, 0, 1)` is always 1.0 and no
weight is ever parked in CDI by the overlay. (The <5-qualifying-names fallback
still holds CDI in thin quarters — you can't invest in a basket that doesn't
exist — but that is data availability, not the overlay.)

Purpose: isolate how much of QDL's edge comes from the stock sleeve vs the CDI
overlay. Compare QDL-Equity ("sleeve only") against QDL ("sleeve + overlay") and
the defensive-income peers via backtests/validate_qdl.py.
"""
from __future__ import annotations

from backtests.strategies.qdl import QDLStrategy


class QDLFullEquityStrategy(QDLStrategy):
    """QDL stock sleeve, always fully invested (CDI overlay disabled)."""

    @property
    def name(self) -> str:
        return "QDL-Equity"

    @property
    def description(self) -> str:
        return (
            "QDL with the CDI vol-target overlay removed — the identical "
            "quality-dividend low-vol sleeve held always-fully-invested "
            "(no de-risking into cash). Ablation to isolate the sleeve's own "
            "contribution vs the overlay."
        )

    def get_parameter_specs(self):
        # target_vol is a dead knob here (forced to infinity), so hide it.
        return [s for s in super().get_parameter_specs() if s.name != "target_vol"]

    def generate_signals(self, shared_data: dict, params: dict):
        params = {**params, "target_vol": float("inf")}  # overlay off => eq_frac == 1
        return super().generate_signals(shared_data, params)
