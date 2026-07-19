"""Self-check for the benchmark/CDI disk+memory cache in backtests/core/data.py."""

import pandas as pd
import pytest

from backtests.core import data


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the cache at a temp dir and start each test with an empty memory cache."""
    monkeypatch.setattr(data, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(data, "_MEM_CACHE", {})
    monkeypatch.delenv("B3_DISABLE_CACHE", raising=False)


def _counting_fetch():
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))

    return calls, fetch


def test_memory_then_disk_hit_avoids_refetch():
    calls, fetch = _counting_fetch()

    first = data._fetch_cached("t", ("A", "2020", "2021"), fetch)
    second = data._fetch_cached("t", ("A", "2020", "2021"), fetch)  # memory hit
    assert calls["n"] == 1
    pd.testing.assert_series_equal(first, second)

    data._MEM_CACHE.clear()  # force the disk path
    third = data._fetch_cached("t", ("A", "2020", "2021"), fetch)
    assert calls["n"] == 1  # served from pickle on disk, no refetch
    pd.testing.assert_series_equal(first, third)


def test_different_key_refetches():
    calls, fetch = _counting_fetch()
    data._fetch_cached("t", ("A", "2020", "2021"), fetch)
    data._fetch_cached("t", ("A", "2020", "2022"), fetch)  # different window
    assert calls["n"] == 2


def test_empty_result_is_not_cached():
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return pd.Series(dtype=float)

    data._fetch_cached("t", ("A", "2020", "2021"), fetch)
    data._fetch_cached("t", ("A", "2020", "2021"), fetch)
    assert calls["n"] == 2  # a failed/empty fetch must never be cached


def test_disable_env_bypasses_cache(monkeypatch):
    monkeypatch.setenv("B3_DISABLE_CACHE", "1")
    calls, fetch = _counting_fetch()
    data._fetch_cached("t", ("A", "2020", "2021"), fetch)
    data._fetch_cached("t", ("A", "2020", "2021"), fetch)
    assert calls["n"] == 2


def test_mutating_result_does_not_corrupt_cache():
    _, fetch = _counting_fetch()
    got = data._fetch_cached("t", ("A", "2020", "2021"), fetch)
    got.iloc[0] = 999.0  # caller mutates its copy
    again = data._fetch_cached("t", ("A", "2020", "2021"), fetch)
    assert again.iloc[0] == 1.0
