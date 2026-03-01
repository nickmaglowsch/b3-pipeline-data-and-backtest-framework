"""
Background Job Runner with Log Streaming
=========================================
Runs Python callables in background threads, captures stdout/stderr in real-time
via a queue, and exposes job status + log lines to the Streamlit UI.
"""
from __future__ import annotations

import io
import queue
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


# ── Job Status ────────────────────────────────────────────────────────────────

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Thread-local stdout capture ───────────────────────────────────────────────

_thread_local = threading.local()


class _CapturingStream(io.TextIOBase):
    """
    A file-like object that captures writes and pushes them to a queue.
    Also writes to the original stream so terminal output is preserved.
    """

    def __init__(self, log_queue: "queue.Queue[str]", original_stream: Any) -> None:
        self._queue = log_queue
        self._original = original_stream
        self._buf = ""

    def write(self, text: str) -> int:
        if text:
            ts = datetime.now().strftime("%H:%M:%S")
            for line in text.splitlines(keepends=True):
                stripped = line.rstrip("\n\r")
                if stripped:
                    self._queue.put(f"[{ts}] {stripped}")
            try:
                self._original.write(text)
                self._original.flush()
            except Exception:
                pass
        return len(text)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8")


class _RedirectingStream(io.TextIOBase):
    """
    Proxy for sys.stdout / sys.stderr that routes writes to a per-thread capturing
    stream if one is installed, or to the global original stream otherwise.
    """

    def __init__(self, name: str, original: Any) -> None:
        self._name = name
        self._original = original

    def _target(self) -> Any:
        capturing = getattr(_thread_local, "capturing_stream", None)
        return capturing if capturing is not None else self._original

    def write(self, text: str) -> int:
        return self._target().write(text)

    def flush(self) -> None:
        self._target().flush()

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8")

    def fileno(self) -> int:
        return self._original.fileno()


# ── Job dataclass ─────────────────────────────────────────────────────────────

@dataclass
class Job:
    id: str
    job_type: str
    status: JobStatus = JobStatus.PENDING
    log_lines: list[str] = field(default_factory=list)
    log_queue: "queue.Queue[str]" = field(default_factory=queue.Queue)
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    thread: Optional[threading.Thread] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_running(self) -> None:
        """Atomically mark the job as running with a start timestamp."""
        with self._lock:
            self.started_at = datetime.now()
            self.status = JobStatus.RUNNING

    def set_completed(self, result: Any = None) -> None:
        """Atomically mark the job as completed.

        Sets ``completed_at`` *before* ``status`` so that any reader that
        observes ``COMPLETED`` will always see a non-None ``completed_at``.
        """
        with self._lock:
            self.result = result
            self.completed_at = datetime.now()
            self.status = JobStatus.COMPLETED

    def set_failed(self, error: str) -> None:
        """Atomically mark the job as failed.

        Sets ``completed_at`` *before* ``status`` so that any reader that
        observes ``FAILED`` will always see a non-None ``completed_at``.
        """
        with self._lock:
            self.error = error
            self.completed_at = datetime.now()
            self.status = JobStatus.FAILED

    def get_snapshot(self) -> dict[str, Any]:
        """Return a consistent snapshot of status, timestamps, and result."""
        with self._lock:
            return {
                "status": self.status,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "result": self.result,
                "error": self.error,
            }


# ── JobRunner ─────────────────────────────────────────────────────────────────

_stdout_redirector: Optional[_RedirectingStream] = None
_stderr_redirector: Optional[_RedirectingStream] = None
_redirect_lock = threading.Lock()
_redirect_ref_count = 0


def _install_redirectors() -> None:
    """Install global stdout/stderr redirectors (idempotent)."""
    global _stdout_redirector, _stderr_redirector, _redirect_ref_count
    with _redirect_lock:
        _redirect_ref_count += 1
        if _redirect_ref_count == 1:
            _stdout_redirector = _RedirectingStream("stdout", sys.__stdout__)
            _stderr_redirector = _RedirectingStream("stderr", sys.__stderr__)
            sys.stdout = _stdout_redirector  # type: ignore[assignment]
            sys.stderr = _stderr_redirector  # type: ignore[assignment]


class JobRunner:
    """
    Singleton-style job runner stored in st.session_state.
    Executes callables in background threads and captures their output.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._active_by_type: dict[str, str] = {}  # job_type -> job_id
        self._lock = threading.Lock()
        _install_redirectors()

    def submit(
        self, job_type: str, fn: Callable, *args: Any, **kwargs: Any
    ) -> str:
        """
        Submit a callable to run in a background thread.

        Args:
            job_type: Tag (e.g. 'pipeline', 'backtest', 'research').
            fn:       The callable to execute.
            *args, **kwargs: Forwarded to fn.

        Returns:
            The job ID string.

        If a job of the same type is already RUNNING, returns the existing job ID.
        """
        with self._lock:
            existing_id = self._active_by_type.get(job_type)
            if existing_id:
                existing = self._jobs.get(existing_id)
                if existing and existing.status == JobStatus.RUNNING:
                    return existing_id

            job_id = str(uuid.uuid4())[:8]
            job = Job(id=job_id, job_type=job_type)
            self._jobs[job_id] = job
            self._active_by_type[job_type] = job_id

        thread = threading.Thread(
            target=self._run,
            args=(job, fn, args, kwargs),
            daemon=True,
            name=f"job-{job_type}-{job_id}",
        )
        job.thread = thread
        job.set_running()
        thread.start()
        return job_id

    def _run(
        self, job: Job, fn: Callable, args: tuple, kwargs: dict
    ) -> None:
        """Thread target: runs fn and captures output."""
        capturing = _CapturingStream(job.log_queue, _stdout_redirector._original if _stdout_redirector else sys.__stdout__)
        _thread_local.capturing_stream = capturing
        try:
            result = fn(*args, **kwargs)
            job.set_completed(result)
        except Exception:
            error_tb = traceback.format_exc()
            # Push error to log queue before changing status so the UI
            # sees the error lines as soon as the status flips to FAILED.
            for line in error_tb.splitlines():
                job.log_queue.put(f"[ERROR] {line}")
            job.set_failed(error_tb)
        finally:
            _thread_local.capturing_stream = None

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def get_active_job(self, job_type: str) -> Optional[Job]:
        job_id = self._active_by_type.get(job_type)
        if job_id:
            return self._jobs.get(job_id)
        return None

    def is_running(self, job_type: str) -> bool:
        job = self.get_active_job(job_type)
        return job is not None and job.status == JobStatus.RUNNING

    def drain_logs(self, job_id: str) -> list[str]:
        """
        Drain any new log lines from the job queue into job.log_lines.
        Returns only the newly drained lines.
        """
        job = self._jobs.get(job_id)
        if not job:
            return []
        new_lines: list[str] = []
        try:
            while True:
                line = job.log_queue.get_nowait()
                new_lines.append(line)
        except queue.Empty:
            pass
        job.log_lines.extend(new_lines)
        return new_lines
