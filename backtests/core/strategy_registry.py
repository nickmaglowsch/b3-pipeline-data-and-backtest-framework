"""
Strategy Registry
=================
Discovers and manages strategy plugin classes from backtests/strategies/.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Optional

logger = logging.getLogger(__name__)

from backtests.core.strategy_base import StrategyBase


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

    def list_all(self) -> list[StrategyBase]:
        """Return all registered strategies."""
        return list(self._strategies.values())

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
                        self.register(instance)
                    except Exception as e:
                        logger.warning("Failed to instantiate strategy %s.%s: %s", modname, attr_name, e)


# ── Global singleton ──────────────────────────────────────────────────────────

_registry: Optional[StrategyRegistry] = None


def get_registry() -> StrategyRegistry:
    """Return the global strategy registry, discovering strategies on first call."""
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
        _registry.discover()
    return _registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _registry
    _registry = None
