"""Discovery contract for the strategy registry.

Guards the failure mode the registry's own comment warns about: discovery that
silently registers the Python plugins but drops every YAML spec (or vice versa).
"""
from __future__ import annotations

import pytest

from backtests.core.strategy_registry import StrategyRegistry, get_registry
from backtests.core.strategy_base import StrategyBase


def test_discover_registers_plugin_classes_and_yaml_specs():
    reg = StrategyRegistry()
    reg.discover()
    names = reg.names()

    assert "QDL" in names, "plugin classes not discovered"
    assert "LowVol" in names, "specs/*.yaml not discovered"
    assert len(names) > 20, f"suspiciously few strategies discovered: {len(names)}"
    assert all(isinstance(s, StrategyBase) for s in (reg.get(n) for n in names))


def test_abstract_bases_are_not_registered():
    """Base classes carry a sentinel empty NAME; registering one puts a blank
    row in the UI strategy picker and a '' key in the registry."""
    reg = StrategyRegistry()
    reg.discover()

    blank = [n for n in reg.names() if not str(n).strip()]
    assert blank == [], f"blank-named strategies registered: {blank!r}"


def test_registry_singleton_is_cached_and_fully_built():
    a, b = get_registry(), get_registry()
    assert a is b
    # a half-built registry (plugins but no specs) must never be cached
    assert "LowVol" in a.names() and "QDL" in a.names()


def test_register_then_get_round_trips():
    class _Stub(StrategyBase):
        @property
        def name(self): return "_Stub"
        @property
        def description(self): return "stub"
        def get_parameter_specs(self): return []
        def generate_signals(self, shared_data, params): return None, None

    reg = StrategyRegistry()
    reg.register(_Stub())
    assert reg.get("_Stub").name == "_Stub"
    assert reg.names() == ["_Stub"]
    with pytest.raises(KeyError):
        reg.get("nope")
