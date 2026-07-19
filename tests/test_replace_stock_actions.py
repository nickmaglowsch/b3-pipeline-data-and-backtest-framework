"""
Tests for storage.replace_stock_actions transactional behavior: the DELETE
and the re-insert must commit atomically, so a failure mid-replace can never
leave stock_actions permanently empty.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from b3_pipeline import storage


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    storage.init_db(c)
    yield c
    c.close()


def _action(isin="BRTEST000001", ex_date="2020-01-02", factor=2.0):
    return {"isin_code": isin, "ex_date": ex_date, "action_type": "SPLIT",
            "factor": factor, "source": "B3"}


def test_replace_swaps_full_table(conn):
    storage.upsert_stock_actions(conn, pd.DataFrame([_action("BROLD0000001")]))
    n = storage.replace_stock_actions(conn, pd.DataFrame([_action("BRNEW0000001")]))
    assert n == 1
    rows = conn.execute("SELECT isin_code FROM stock_actions").fetchall()
    assert rows == [("BRNEW0000001",)]


def test_replace_failure_rolls_back_delete(conn):
    """If the insert fails after the DELETE, the old rows must survive."""
    storage.upsert_stock_actions(conn, pd.DataFrame([_action("BROLD0000001")]))
    bad = pd.DataFrame([{"isin_code": "BRNEW0000001", "ex_date": "2021-01-02"}])
    with pytest.raises(Exception):
        storage.replace_stock_actions(conn, bad)  # missing action_type/factor
    rows = conn.execute("SELECT isin_code FROM stock_actions").fetchall()
    assert rows == [("BROLD0000001",)], "DELETE must have been rolled back"
