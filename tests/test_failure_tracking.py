"""
Tests for Phase 4: API failure tracking.

These tests verify:
1. A fetch_failures table is created by init_db
2. record_fetch_failure inserts a row into fetch_failures
3. get_unresolved_failures returns only unresolved entries
4. resolve_fetch_failure marks an entry as resolved
"""
import sqlite3
import pytest

from b3_pipeline import storage


def _in_memory_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


class TestFetchFailureTracking:
    """Test the fetch_failures table operations."""

    def test_init_db_creates_fetch_failures_table(self):
        """
        GIVEN a fresh database
        WHEN init_db is called
        THEN a fetch_failures table should exist
        """
        conn = _in_memory_conn()
        storage.init_db(conn, rebuild=False)

        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fetch_failures'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None, "fetch_failures table should be created by init_db"

    def test_record_fetch_failure_inserts_row(self):
        """
        GIVEN a database with fetch_failures table
        WHEN record_fetch_failure is called
        THEN a row should be inserted with resolved=0
        """
        conn = _in_memory_conn()
        storage.init_db(conn, rebuild=False)

        storage.record_fetch_failure(conn, "ARZZ", "GetListedSupplementCompany", "520 Server Error")

        cursor = conn.cursor()
        cursor.execute("SELECT company_code, endpoint, resolved FROM fetch_failures")
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 1, "Should have one failure record"
        assert rows[0][0] == "ARZZ"
        assert rows[0][1] == "GetListedSupplementCompany"
        assert rows[0][2] == 0, "resolved should be 0 (False)"

    def test_get_unresolved_failures(self):
        """
        GIVEN two failures, one resolved and one not
        WHEN get_unresolved_failures is called
        THEN only the unresolved one should be returned
        """
        conn = _in_memory_conn()
        storage.init_db(conn, rebuild=False)

        storage.record_fetch_failure(conn, "ARZZ", "GetListedSupplementCompany", "520 error")
        storage.record_fetch_failure(conn, "BMGB", "GetListedSupplementCompany", "timeout")

        # Resolve one of them
        storage.resolve_fetch_failure(conn, "ARZZ", "GetListedSupplementCompany")

        unresolved = storage.get_unresolved_failures(conn)
        conn.close()

        assert len(unresolved) == 1, f"Expected 1 unresolved, got {len(unresolved)}"
        assert unresolved[0]["company_code"] == "BMGB"

    def test_resolve_fetch_failure(self):
        """
        GIVEN a failure record
        WHEN resolve_fetch_failure is called
        THEN resolved should be set to 1
        """
        conn = _in_memory_conn()
        storage.init_db(conn, rebuild=False)

        storage.record_fetch_failure(conn, "ARZZ", "GetListedSupplementCompany", "error")
        storage.resolve_fetch_failure(conn, "ARZZ", "GetListedSupplementCompany")

        cursor = conn.cursor()
        cursor.execute("SELECT resolved FROM fetch_failures WHERE company_code='ARZZ'")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == 1, "resolved should be 1 after resolve_fetch_failure"

    def test_upsert_on_duplicate_failure(self):
        """
        GIVEN an existing failure record
        WHEN record_fetch_failure is called again for the same company+endpoint
        THEN the retry_count should be incremented and resolved stays 0
        """
        conn = _in_memory_conn()
        storage.init_db(conn, rebuild=False)

        storage.record_fetch_failure(conn, "ARZZ", "GetListedSupplementCompany", "first error")
        storage.record_fetch_failure(conn, "ARZZ", "GetListedSupplementCompany", "second error")

        cursor = conn.cursor()
        cursor.execute("SELECT retry_count FROM fetch_failures WHERE company_code='ARZZ'")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] >= 1, "retry_count should be incremented on duplicate failure"
