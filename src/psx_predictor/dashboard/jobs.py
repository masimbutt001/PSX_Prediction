"""Background jobs the dashboard may start: a fixed list of `psx` commands, never free text.

Each job runs `python -m psx_predictor.cli ...` in a child process. Its output goes
to data/jobs/<id>.log and its status to data/jobs/<id>.json, so the page can show live
progress and the history survives a page reload. Only one job runs at a time.
"""

import datetime
import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from psx_predictor.scheduler.jobs import DailyPipelineStep

STEP_LABELS = {
    "update_prices": "Prices",
    "collect_news": "News and announcements",
    "update_macro": "Macro (USD/PKR, Brent, KSE-100)",
    "build_features": "Features",
    "generate_predictions": "Predictions",
    "audit_predictions": "Audit of past predictions",
}

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class JobError(ValueError):
    """Raised for a request outside the allowed actions, or while another job runs."""


@dataclass
class Job:
    """One background command and its outcome."""

    id: str
    action: str
    label: str
    argv: list[str]
    started_at: str
    status: str = "running"  # running | succeeded | failed | lost
    ended_at: Optional[str] = None
    returncode: Optional[int] = None
    pid: Optional[int] = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def running(self) -> bool:
        return self.status == "running"

    def elapsed_seconds(self, now: Optional[datetime.datetime] = None) -> float:
        start = datetime.datetime.fromisoformat(self.started_at)
        end = (
            datetime.datetime.fromisoformat(self.ended_at)
            if self.ended_at
            else (now or datetime.datetime.now(datetime.timezone.utc))
        )
        return max(0.0, (end - start).total_seconds())


def build_command(
    action: str,
    symbols: Optional[list[str]] = None,
    step: Optional[str] = None,
    allowed_symbols: Optional[set[str]] = None,
) -> tuple[list[str], str]:
    """psx arguments and a readable label for an allowed action.

    Raises JobError for anything not on the list: an unknown action, step or symbol.
    """
    if action == "update_prices":
        if symbols:
            syms = [s.strip().upper() for s in symbols if s.strip()]
            for s in syms:
                if not s.isalnum() or (allowed_symbols is not None and s not in allowed_symbols):
                    raise JobError(f"Unknown stock: {s}")
            return (
                ["data", "update", "--symbols", ",".join(syms)],
                f"Update prices: {', '.join(syms)}",
            )
        return ["data", "update"], "Update prices: all stocks"
    if action == "run_pipeline":
        return ["schedule", "run-daily", "--continue-on-error"], "Daily pipeline"
    if action == "run_step":
        valid = {s.value for s in DailyPipelineStep}
        if step not in valid:
            raise JobError(f"Unknown pipeline step: {step}")
        return (
            ["schedule", "run-daily", "--continue-on-error", "--step", str(step)],
            f"Rerun step: {STEP_LABELS.get(str(step), str(step))}",
        )
    if action == "audit_predictions":
        return ["predict", "audit"], "Score past predictions"
    raise JobError(f"Action not allowed: {action}")


class JobManager:
    """Starts allowed jobs and reports their status from data/jobs/."""

    def __init__(self, jobs_dir: Path, cwd: Optional[Path] = None) -> None:
        self.jobs_dir = jobs_dir
        self.cwd = cwd
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._procs: dict[str, subprocess.Popen[bytes]] = {}

    # ---- files ----
    def _meta_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def log_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.log"

    def _save(self, job: Job) -> None:
        tmp = self._meta_path(job.id).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")
        os.replace(tmp, self._meta_path(job.id))

    def _load(self, path: Path) -> Optional[Job]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return Job(**raw)
        except (OSError, ValueError, TypeError):
            return None

    # ---- queries ----
    def get(self, job_id: str) -> Optional[Job]:
        job = self._load(self._meta_path(job_id))
        if job and job.running and job.id not in self._procs:
            # Started by an earlier dashboard process that is gone; we can't follow it.
            job.status = "lost"
            job.ended_at = job.ended_at or datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._save(job)
        return job

    def recent(self, limit: int = 10) -> list[Job]:
        jobs = [j for p in self.jobs_dir.glob("*.json") if (j := self.get(p.stem)) is not None]
        jobs.sort(key=lambda j: j.started_at, reverse=True)
        return jobs[:limit]

    def current(self) -> Optional[Job]:
        """The job that is running now, if any."""
        for job_id in list(self._procs):
            job = self.get(job_id)
            if job and job.running:
                return job
        return None

    def read_log(self, job_id: str, tail: int = 40) -> str:
        path = self.log_path(job_id)
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail:])

    # ---- start ----
    def start(
        self,
        action: str,
        symbols: Optional[list[str]] = None,
        step: Optional[str] = None,
        allowed_symbols: Optional[set[str]] = None,
        env: Optional[dict[str, str]] = None,
    ) -> Job:
        args, label = build_command(action, symbols, step, allowed_symbols)
        with self._lock:
            running = self.current()
            if running:
                raise JobError(f"'{running.label}' is still running. Wait for it to finish.")
            argv = [sys.executable, "-m", "psx_predictor.cli", *args]
            job = Job(
                id=datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
                action=action,
                label=label,
                argv=argv,
                started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                params={"symbols": symbols or [], "step": step},
            )
            child_env = {
                **os.environ,
                **(env or {}),
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "COLUMNS": "110",
                "NO_COLOR": "1",
            }
            log = open(self.log_path(job.id), "wb")  # noqa: SIM115 - closed by the watcher
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.Popen(
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=str(self.cwd) if self.cwd else None,
                env=child_env,
                creationflags=creationflags,
            )
            job.pid = proc.pid
            self._procs[job.id] = proc
            self._save(job)
            threading.Thread(
                target=self._watch, args=(job, proc, log), daemon=True, name=f"job-{job.id}"
            ).start()
            return job

    def _watch(self, job: Job, proc: "subprocess.Popen[bytes]", log: Any) -> None:
        code = proc.wait()
        log.close()
        job.returncode = code
        job.status = "succeeded" if code == 0 else "failed"
        job.ended_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._save(job)
        self._procs.pop(job.id, None)

    def wait(self, job_id: str, timeout: Optional[float] = None) -> Optional[Job]:
        """Block until the job ends (used by tests)."""
        proc = self._procs.get(job_id)
        if proc is not None:
            proc.wait(timeout=timeout)
            for t in threading.enumerate():
                if t.name == f"job-{job_id}":
                    t.join(timeout=timeout)
        return self.get(job_id)


def is_local_request(server_address: Optional[str], client_ip: Optional[str]) -> bool:
    """Commands may only be started from this computer.

    True when the server listens on loopback only, or the browser connects from loopback.
    Streamlit reports a loopback client as None (`st.context.ip_address`).
    """
    if server_address and server_address in LOOPBACK:
        return True
    return client_ip is None or client_ip in LOOPBACK
