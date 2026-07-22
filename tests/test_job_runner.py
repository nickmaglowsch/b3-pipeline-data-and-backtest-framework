"""Characterization tests for the UI background job runner.

These pin the contract the Streamlit UI relies on (ui/components/log_stream.py and
ui/pages/*.py): job status transitions, per-job timestamped log capture, pass-through
to the real stdout, and per-job-type isolation.
"""

import io
import re
import sys
import threading
import time

import pytest

from ui.services import job_runner
from ui.services.job_runner import JobRunner, JobStatus

TIMESTAMP = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] ")


class _FakeStd(io.StringIO):
    """Stand-in for the real terminal stream (needs a fileno for sys.stdout duck-typing)."""

    def fileno(self) -> int:
        return 1


def _wait(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for condition")


@pytest.fixture
def make_runner(monkeypatch):
    """Builds a JobRunner whose stdout/stderr redirectors write to fakes, not the terminal.

    Must be called from the test body, not fixture setup: pytest re-installs its own
    sys.stdout capture between setup and the call phase, which would undo the runner's
    global redirectors.
    """
    fake_out, fake_err = _FakeStd(), _FakeStd()
    monkeypatch.setattr(sys, "__stdout__", fake_out)
    monkeypatch.setattr(sys, "__stderr__", fake_err)
    monkeypatch.setattr(job_runner, "_redirect_ref_count", 0)
    monkeypatch.setattr(job_runner, "_stdout_redirector", None)
    monkeypatch.setattr(job_runner, "_stderr_redirector", None)
    saved: list = []

    def factory() -> JobRunner:
        saved.append((sys.stdout, sys.stderr))
        r = JobRunner()
        r.fake_out, r.fake_err = fake_out, fake_err  # type: ignore[attr-defined]
        return r

    try:
        yield factory
    finally:
        if saved:
            sys.stdout, sys.stderr = saved[0]


# ── Completion ────────────────────────────────────────────────────────────────

def test_job_runs_and_completes(make_runner):
    runner = make_runner()
    job_id = runner.submit("backtest", lambda a, b=0: a + b, 40, b=2)
    job = runner.get_job(job_id)

    assert job is not None
    assert job.id == job_id
    assert job.job_type == "backtest"
    _wait(lambda: job.status is not JobStatus.RUNNING)

    assert job.status is JobStatus.COMPLETED
    assert job.result == 42
    assert job.error is None
    assert job.started_at is not None and job.completed_at is not None
    assert job.completed_at >= job.started_at


def test_snapshot_reports_completed_job(make_runner):
    runner = make_runner()
    job = runner.get_job(runner.submit("backtest", lambda: {"ok": True}))
    _wait(lambda: job.status is not JobStatus.RUNNING)

    snap = job.get_snapshot()
    assert snap["status"] is JobStatus.COMPLETED
    assert snap["result"] == {"ok": True}
    assert snap["error"] is None
    assert snap["started_at"] is not None and snap["completed_at"] is not None


def test_get_job_unknown_id_returns_none(make_runner):
    runner = make_runner()
    assert runner.get_job("nope") is None
    assert runner.get_active_job("nothing-ran") is None
    assert runner.is_running("nothing-ran") is False


# ── Log capture ───────────────────────────────────────────────────────────────

def test_stdout_is_captured_timestamped_and_drained_in_order(make_runner):
    runner = make_runner()

    def job_fn():
        print("first line")
        print("second line")
        print("third line")

    job_id = runner.submit("pipeline", job_fn)
    _wait(lambda: runner.get_job(job_id).status is not JobStatus.RUNNING)

    lines = runner.drain_logs(job_id)
    assert [TIMESTAMP.sub("", ln) for ln in lines] == [
        "first line",
        "second line",
        "third line",
    ]
    assert all(TIMESTAMP.match(ln) for ln in lines)
    # drained lines are accumulated on the job and not handed out twice
    assert runner.get_job(job_id).log_lines == lines
    assert runner.drain_logs(job_id) == []
    assert runner.drain_logs("nope") == []


def test_real_stdout_still_receives_the_text(make_runner):
    runner = make_runner()
    runner_out = runner.fake_out
    job_id = runner.submit("pipeline", lambda: print("to the terminal"))
    _wait(lambda: runner.get_job(job_id).status is not JobStatus.RUNNING)

    assert "to the terminal\n" in runner_out.getvalue()


def test_main_thread_print_bypasses_job_capture(make_runner):
    """The global redirectors must not steal output from non-job threads."""
    runner = make_runner()
    job_id = runner.submit("pipeline", lambda: print("in job"))
    _wait(lambda: runner.get_job(job_id).status is not JobStatus.RUNNING)

    print("in main thread")

    lines = runner.drain_logs(job_id)
    assert [TIMESTAMP.sub("", ln) for ln in lines] == ["in job"]
    assert "in main thread\n" in runner.fake_out.getvalue()


def test_concurrent_jobs_do_not_cross_contaminate_logs(make_runner):
    runner = make_runner()
    started = {"pipeline": threading.Event(), "research": threading.Event()}
    release = threading.Event()

    def job_fn(tag):
        print(f"{tag} output")
        started[tag].set()
        release.wait(5.0)

    pipeline_id = runner.submit("pipeline", job_fn, "pipeline")
    research_id = runner.submit("research", job_fn, "research")
    assert started["pipeline"].wait(5.0) and started["research"].wait(5.0)

    try:
        pipeline_lines = runner.drain_logs(pipeline_id)
        research_lines = runner.drain_logs(research_id)
    finally:
        release.set()

    assert [TIMESTAMP.sub("", ln) for ln in pipeline_lines] == ["pipeline output"]
    assert [TIMESTAMP.sub("", ln) for ln in research_lines] == ["research output"]
    _wait(lambda: runner.get_job(pipeline_id).status is JobStatus.COMPLETED)
    _wait(lambda: runner.get_job(research_id).status is JobStatus.COMPLETED)


# ── Failure ───────────────────────────────────────────────────────────────────

def test_failing_job_is_failed_with_traceback(make_runner):
    runner = make_runner()

    def boom():
        raise ValueError("kaboom")

    job = runner.get_job(runner.submit("research", boom))
    _wait(lambda: job.status is not JobStatus.RUNNING)

    assert job.status is JobStatus.FAILED
    assert job.result is None
    assert "Traceback (most recent call last):" in job.error
    assert "ValueError: kaboom" in job.error
    assert job.completed_at is not None

    error_lines = [ln for ln in runner.drain_logs(job.id) if ln.startswith("[ERROR] ")]
    assert any("ValueError: kaboom" in ln for ln in error_lines)

    snap = job.get_snapshot()
    assert snap["status"] is JobStatus.FAILED
    assert "ValueError: kaboom" in snap["error"]
    assert runner.is_running("research") is False


# ── Active job / is_running ───────────────────────────────────────────────────

def test_is_running_and_active_job_during_and_after(make_runner):
    runner = make_runner()
    started, release = threading.Event(), threading.Event()

    def blocker():
        started.set()
        release.wait(5.0)
        return "done"

    job_id = runner.submit("fundamentals", blocker)
    assert started.wait(5.0)

    try:
        assert runner.is_running("fundamentals") is True
        active = runner.get_active_job("fundamentals")
        assert active is not None and active.id == job_id
        assert active.status is JobStatus.RUNNING
        assert active.result is None
        assert active.error is None
        assert active.started_at is not None
        assert runner.is_running("pipeline") is False
    finally:
        release.set()

    _wait(lambda: runner.is_running("fundamentals") is False)
    job = runner.get_active_job("fundamentals")
    assert job.status is JobStatus.COMPLETED
    assert job.result == "done"


def test_submit_while_same_type_running_returns_existing_job(make_runner):
    runner = make_runner()
    started, release = threading.Event(), threading.Event()

    def blocker():
        started.set()
        release.wait(5.0)

    first = runner.submit("discovery", blocker)
    assert started.wait(5.0)

    try:
        assert runner.submit("discovery", blocker) == first
        other = runner.submit("pipeline", lambda: None)
        assert other != first
    finally:
        release.set()

    _wait(lambda: runner.get_job(first).status is JobStatus.COMPLETED)

    second = runner.submit("discovery", lambda: None)
    assert second != first
    _wait(lambda: runner.get_job(second).status is JobStatus.COMPLETED)
    assert runner.get_active_job("discovery").id == second
