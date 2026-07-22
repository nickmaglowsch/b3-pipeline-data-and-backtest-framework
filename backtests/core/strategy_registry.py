"""
Strategy Registry
=================
Discovers and manages strategy plugin classes from backtests/strategies/.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

from backtests.core.strategy_base import StrategyBase

SPECS_DIR = Path(__file__).resolve().parent.parent / "strategies" / "specs"


class StrategyRegistry:
    """Discovers and manages strategy plugins."""

    def __init__(self) -> None:
        self._strategies: dict[str, StrategyBase] = {}

    def register(self, strategy: StrategyBase) -> None:
        """Register a strategy instance."""
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> StrategyBase:
        """Retrieve a strategy by name. Raises KeyError if not found."""
        return self._strategies[name]

    def names(self) -> list[str]:
        """Return list of strategy names."""
        return list(self._strategies.keys())

    def discover(self, package_path: str = "backtests.strategies") -> None:
        """
        Auto-discover strategy classes from the strategies package.
        Imports all modules in the package and registers any StrategyBase subclass found.
        """
        try:
            pkg = importlib.import_module(package_path)
        except ImportError:
            return

        pkg_path = getattr(pkg, "__path__", None)
        if pkg_path is None:
            return

        for _importer, modname, _ispkg in pkgutil.iter_modules(pkg_path):
            try:
                module = importlib.import_module(f"{package_path}.{modname}")
            except Exception as e:
                logger.warning("Failed to import strategy module %s.%s: %s", package_path, modname, e)
                continue

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, StrategyBase)
                    and attr is not StrategyBase
                ):
                    try:
                        instance = attr()
                        # Abstract bases (e.g. _SMATiltBase) satisfy the ABC but
                        # carry a sentinel empty name — registering one puts a
                        # blank row in the UI picker and a '' key in the registry.
                        if not str(instance.name).strip():
                            continue
                        self.register(instance)
                    except Exception as e:
                        logger.warning("Failed to instantiate strategy %s.%s: %s", modname, attr_name, e)

        # Config-driven strategies from backtests/strategies/specs/*.yaml
        from backtests.core.config_strategy import RankAndHold, FixedWeight

        kinds = {"rank_and_hold": RankAndHold, "fixed_weight": FixedWeight}
        for path in sorted(SPECS_DIR.glob("*.yaml")) if SPECS_DIR.is_dir() else []:
            try:
                spec = yaml.safe_load(path.read_text())
                self.register(kinds[spec["kind"]](spec))
            except Exception as e:
                logger.warning("Failed to load strategy spec %s: %s", path.name, e)


# ── Global singleton ──────────────────────────────────────────────────────────

_registry: Optional[StrategyRegistry] = None


def get_registry() -> StrategyRegistry:
    """Return the global strategy registry, discovering strategies on first call."""
    global _registry
    if _registry is None:
        # Build into a local and publish only after discover() fully succeeds.
        # If discovery raises partway (e.g. spec loading), we must NOT cache a
        # half-built registry (plugins-but-no-specs) — that would stick for the
        # whole process and silently drop every YAML strategy. Leaving _registry
        # None lets the next call retry cleanly.
        reg = StrategyRegistry()
        reg.discover()
        _registry = reg
    return _registry
