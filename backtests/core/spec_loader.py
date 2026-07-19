"""
YAML spec loader
================
Loads config-driven strategies from ``backtests/strategies/specs/*.yaml`` and
turns each into a registered ``StrategyBase`` instance. Called from
``StrategyRegistry.discover()`` after the normal package scan.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

from backtests.core.config_strategy import RankAndHold, FixedWeight

logger = logging.getLogger(__name__)

_KINDS = {"rank_and_hold": RankAndHold, "fixed_weight": FixedWeight}

SPECS_DIR = Path(__file__).resolve().parent.parent / "strategies" / "specs"


def load_specs(registry, specs_dir: Path = SPECS_DIR) -> None:
    """Discover every *.yaml spec and register a strategy instance per file."""
    if not specs_dir.is_dir():
        return
    for path in sorted(specs_dir.glob("*.yaml")):
        try:
            spec = yaml.safe_load(path.read_text())
            cls = _KINDS[spec["kind"]]
            registry.register(cls(spec))
        except Exception as e:
            logger.warning("Failed to load strategy spec %s: %s", path.name, e)
