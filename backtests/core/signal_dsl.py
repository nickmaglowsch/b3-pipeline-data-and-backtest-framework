"""
Signal expression interpreter (the signal DSL)
=============================================
Turns a signal from *code* into *data*: a factor is a string expression over a
fixed, safe vocabulary of primitives, evaluated to a wide (date x ticker) frame
where **higher = better**. Named signals live in
``backtests/strategies/signals.yaml``; a strategy spec may also inline an
``expr:`` directly. Adding a new factor is a line of YAML — no Python.

Safety: the expression is parsed with the ``ast`` module and only a whitelist of
node types is allowed (arithmetic, numeric literals, whitelisted names, and
calls to whitelisted functions). There is **no** ``eval()`` of raw source, no
attribute access, no subscripting, no lambdas/comprehensions — so a malicious or
typo'd spec cannot execute arbitrary code.

Namespace = every ``shared_data`` frame (``ret``, ``log_ret``, ``adtv``,
``vol_60d``, ``f_pe_ratio_dyn`` …) plus any numeric params (e.g. ``lookback``).

Parity note: rolling reductions default ``min_periods = window`` (plain
``.rolling(w)``), matching the original hand-written signals — NOT the
``max(k, w//2)`` used by research/discovery base_signals.
"""
from __future__ import annotations

import ast
import operator

import numpy as np
import pandas as pd

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow,
}
_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}

_ALLOWED = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Call,
    ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)


def _zscore(x: pd.DataFrame) -> pd.DataFrame:
    m = x.mean(axis=1)
    s = x.std(axis=1).replace(0, np.nan)
    return x.sub(m, axis=0).div(s, axis=0)


def _build_funcs(ns: dict) -> dict:
    glitch = ns.get("has_glitch")

    def mask_glitch(x):
        return x.where(glitch != 1) if glitch is not None else x

    def _roll(x, w, agg):
        return getattr(x.rolling(int(w)), agg)()   # min_periods = window (parity)

    return {
        # time-series (operate per column, along time)
        "shift": lambda x, n: x.shift(int(n)),
        "roll_sum": lambda x, w: _roll(x, w, "sum"),
        "roll_mean": lambda x, w: _roll(x, w, "mean"),
        "roll_std": lambda x, w: _roll(x, w, "std"),
        "roll_var": lambda x, w: _roll(x, w, "var"),
        "roll_min": lambda x, w: _roll(x, w, "min"),
        "roll_max": lambda x, w: _roll(x, w, "max"),
        "roll_median": lambda x, w: _roll(x, w, "median"),
        "roll_skew": lambda x, w: _roll(x, w, "skew"),
        "roll_kurt": lambda x, w: _roll(x, w, "kurt"),
        "pct_change": lambda x, w=1: x.pct_change(int(w)),
        "diff": lambda x, n=1: x - x.shift(int(n)),
        "ewm_mean": lambda x, span: x.ewm(span=int(span), min_periods=int(span)).mean(),
        "ewm_std": lambda x, span: x.ewm(span=int(span), min_periods=int(span)).std(),
        # elementwise
        "log": lambda x: np.log(x),
        "sign": lambda x: np.sign(x),
        "abs": lambda x: x.abs() if hasattr(x, "abs") else abs(x),
        "clip": lambda x, lo, hi: x.clip(lo, hi),
        # cross-sectional (operate per row, across tickers)
        "rank": lambda x: x.rank(axis=1, pct=True),   # pandas semantics (parity)
        "zscore": _zscore,
        "demean": lambda x: x.sub(x.mean(axis=1), axis=0),
        # data-quality
        "mask_glitch": mask_glitch,
    }


def evaluate(expr: str, namespace: dict) -> pd.DataFrame:
    """Evaluate a signal expression string against ``namespace`` (name -> frame or
    scalar). Returns a wide DataFrame. Raises ValueError on disallowed syntax or
    unknown names/functions."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"signal expr: disallowed syntax {type(node).__name__!r} in {expr!r}")
    funcs = _build_funcs(namespace)
    return _ev(tree.body, namespace, funcs)


def _ev(node, ns, funcs):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"signal expr: only numeric constants allowed, got {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in ns:
            return ns[node.id]
        raise ValueError(f"signal expr: unknown name {node.id!r}")
    if isinstance(node, ast.UnaryOp):
        return _UNARYOPS[type(node.op)](_ev(node.operand, ns, funcs))
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_ev(node.left, ns, funcs), _ev(node.right, ns, funcs))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("signal expr: only positional calls to named functions are allowed")
        fn = funcs.get(node.func.id)
        if fn is None:
            raise ValueError(f"signal expr: unknown function {node.func.id!r}")
        return fn(*[_ev(a, ns, funcs) for a in node.args])
    raise ValueError(f"signal expr: disallowed node {type(node).__name__!r}")
