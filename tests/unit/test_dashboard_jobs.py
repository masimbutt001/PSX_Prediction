"""Tests for running allowed commands from the dashboard and partial pipeline reruns."""

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app as cli_app
from psx_predictor.dashboard import jobs
from psx_predictor.dashboard.jobs import JobError, JobManager, build_command, is_local_request
from psx_predictor.scheduler.jobs import DailyPipelineStep
from psx_predictor.scheduler.runner import DailyPipelineRunner


def test_build_command_allowlist() -> None:
    args, label = build_command("update_prices", ["ogdc", "PPL"], allowed_symbols={"OGDC", "PPL"})
    assert args == ["data", "update", "--symbols", "OGDC,PPL"]
    assert "OGDC" in label
    assert build_command("update_prices")[0] == ["data", "update"]
    assert build_command("run_pipeline")[0] == ["schedule", "run-daily", "--continue-on-error"]
    assert build_command("run_step", step="collect_news")[0][-2:] == ["--step", "collect_news"]
    assert build_command("audit_predictions")[0] == ["predict", "audit"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action": "rm -rf /"},
        {"action": "update_prices", "symbols": ["OGDC;del"]},
        {"action": "update_prices", "symbols": ["XYZ"], "allowed_symbols": {"OGDC"}},
        {"action": "run_step", "step": "format_disk"},
        {"action": "run_step"},
    ],
)
def test_build_command_rejects_anything_else(kwargs: dict) -> None:
    with pytest.raises(JobError):
        build_command(**kwargs)


def test_is_local_request() -> None:
    assert is_local_request("127.0.0.1", "203.0.113.5")
    assert is_local_request(None, None)  # Streamlit reports a loopback browser as None
    assert is_local_request("0.0.0.0", "127.0.0.1")
    assert not is_local_request("0.0.0.0", "192.168.1.20")
    assert not is_local_request(None, "203.0.113.5")


def test_job_manager_runs_one_job_and_records_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs, "build_command", lambda *a, **k: (["--help"], "Show help"))
    mgr = JobManager(jobs_dir=tmp_path / "jobs", cwd=tmp_path)
    job = mgr.start("anything")
    assert job.running and job.argv[:3] == [sys.executable, "-m", "psx_predictor.cli"]
    with pytest.raises(JobError, match="still running"):
        mgr.start("anything")
    done = mgr.wait(job.id, timeout=120)
    assert done is not None and done.status == "succeeded" and done.returncode == 0
    assert "schedule" in mgr.read_log(job.id, tail=200)
    assert mgr.current() is None
    assert [j.id for j in mgr.recent()] == [job.id]

    # A job left "running" by a dashboard that has since stopped is reported as lost.
    meta = json.loads((tmp_path / "jobs" / f"{job.id}.json").read_text())
    meta.update(id="old", status="running", ended_at=None)
    (tmp_path / "jobs" / "old.json").write_text(json.dumps(meta))
    lost = JobManager(jobs_dir=tmp_path / "jobs").get("old")
    assert lost is not None and lost.status == "lost"


def test_job_failure_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "build_command", lambda *a, **k: (["no-such-command"], "Bad"))
    mgr = JobManager(jobs_dir=tmp_path / "jobs", cwd=tmp_path)
    done = mgr.wait(mgr.start("anything").id, timeout=120)
    assert done is not None and done.status == "failed" and done.returncode != 0


def test_partial_run_is_logged_and_ignored_by_catch_up(tmp_path: Path) -> None:
    log_file = tmp_path / "pipeline.log"
    runner = DailyPipelineRunner(data_dir=tmp_path / "data", log_path=log_file)
    report = runner.run_daily_pipeline(
        dry_run=True, only_steps=[DailyPipelineStep.AUDIT_PREDICTIONS]
    )
    assert report is not None and report.partial
    assert [t.step for t in report.tasks] == [DailyPipelineStep.AUDIT_PREDICTIONS]

    # Same partial run, logged live, must not count as the day's completed run.
    entry = json.loads(log_file.read_text().splitlines()[-1])
    entry.update(dry_run=False, success=True, started_at="2026-11-23T12:00:00+00:00")
    log_file.write_text(json.dumps(entry) + "\n")
    import datetime
    from zoneinfo import ZoneInfo

    now = datetime.datetime(2026, 11, 23, 18, tzinfo=ZoneInfo("Asia/Karachi"))
    assert runner.catch_up_decision(now)[0] is True

    full = runner.run_daily_pipeline(dry_run=True, only_steps=list(DailyPipelineStep))
    assert full is not None and not full.partial


def test_cli_step_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PSX_DATA_DIR", str(tmp_path))
    runner = CliRunner()
    bad = runner.invoke(cli_app, ["schedule", "run-daily", "--dry-run", "--step", "nope"])
    assert bad.exit_code == 2 and "Unknown step" in bad.output
    ok = runner.invoke(
        cli_app, ["schedule", "run-daily", "--dry-run", "--step", "audit_predictions"]
    )
    assert ok.exit_code == 0, ok.output


def test_latest_steps_overlays_reruns() -> None:
    from psx_predictor.dashboard.data import latest_steps

    full = {
        "started_at": "2026-11-21T12:00:00+00:00",
        "success": False,
        "tasks": [
            {"step": "update_prices", "status": "FAILED"},
            {"step": "collect_news", "status": "SUCCESS"},
        ],
    }
    rerun = {
        "started_at": "2026-11-21T13:00:00+00:00",
        "success": True,
        "partial": True,
        "tasks": [{"step": "update_prices", "status": "SUCCESS"}],
    }
    base, tasks = latest_steps([rerun, full])
    assert base is full
    assert [(t["step"], t["status"], bool(t.get("rerun"))) for t in tasks] == [
        ("update_prices", "SUCCESS", True),
        ("collect_news", "SUCCESS", False),
    ]
    assert latest_steps([]) == (None, [])
