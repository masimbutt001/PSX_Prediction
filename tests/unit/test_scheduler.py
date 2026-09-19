"""Unit tests for Phase 19: Daily Scheduled Automation and EOD Pipeline Orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from psx_predictor.cli.main import app as cli_app
from psx_predictor.scheduler.jobs import (
    DailyPipelineStep,
    PipelineRunReport,
    PipelineTaskResult,
    create_cron_triggers,
)
from psx_predictor.scheduler.runner import DailyPipelineRunner


def test_pipeline_step_definitions_and_ordering() -> None:
    """Verify DailyPipelineStep enum members and topological execution ordering."""
    expected_order = [
        DailyPipelineStep.UPDATE_PRICES,
        DailyPipelineStep.COLLECT_NEWS,
        DailyPipelineStep.UPDATE_MACRO,
        DailyPipelineStep.BUILD_FEATURES,
        DailyPipelineStep.GENERATE_PREDICTIONS,
        DailyPipelineStep.AUDIT_PREDICTIONS,
    ]

    steps = list(DailyPipelineStep)
    assert steps == expected_order
    assert len(steps) == 6
    assert steps[0].value == "update_prices"
    assert steps[-1].value == "audit_predictions"


def test_cron_triggers_creation() -> None:
    """Verify cron schedule triggers for Mon-Thu 16:00 and Fri 17:00 Karachi time."""
    triggers = create_cron_triggers(timezone="Asia/Karachi")
    assert len(triggers) == 2

    # Check that both day_of_week and hour match PSX post-market timings
    trigger_strs = [str(t) for t in triggers]
    assert any("mon-thu" in t and "16" in t for t in trigger_strs)
    assert any("fri" in t and "17" in t for t in trigger_strs)


def test_pipeline_task_result_and_report_serialization() -> None:
    """Test data class serialization and properties."""
    res = PipelineTaskResult(
        step=DailyPipelineStep.UPDATE_PRICES,
        status="SUCCESS",
        duration_seconds=1.234,
        message="Updated prices for 5 stocks",
    )
    d = res.to_dict()
    assert d["step"] == "update_prices"
    assert d["status"] == "SUCCESS"
    assert d["duration_seconds"] == 1.23
    assert d["message"] == "Updated prices for 5 stocks"
    assert d["error"] is None

    report = PipelineRunReport(
        started_at="2026-09-19T16:00:00+05:00",
        ended_at="2026-09-19T16:02:00+05:00",
        duration_seconds=120.0,
        success=True,
        tasks=[res],
    )
    report_dict = report.to_dict()
    assert report_dict["started_at"] == "2026-09-19T16:00:00+05:00"
    assert report_dict["success"] is True
    assert len(report_dict["tasks"]) == 1


def test_pipeline_dry_run_execution(tmp_path: Path) -> None:
    """Verify DailyPipelineRunner executes all 6 steps smoothly in dry_run mode."""
    log_file = tmp_path / "pipeline.log"
    runner = DailyPipelineRunner(log_path=log_file)

    report = runner.run_daily_pipeline(
        dry_run=True,
        continue_on_error=False,
        symbols=["OGDC", "HBL"],
    )

    assert report.success is True
    assert len(report.tasks) == 6
    for task in report.tasks:
        assert task.status == "SUCCESS"
        assert task.error is None
        assert task.duration_seconds >= 0.0

    # Ensure pipeline log was created and appended
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert '"success": true' in content


def test_pipeline_failure_halts_remaining_steps(tmp_path: Path) -> None:
    """Verify that when a step fails and continue_on_error=False, downstream steps are skipped."""
    log_file = tmp_path / "pipeline.log"
    runner = DailyPipelineRunner(log_path=log_file, max_retries=1)

    # Monkeypatch the macro update step to raise an exception
    def failing_macro(dry_run: bool) -> str:
        raise ConnectionError("SBP portal timed out")

    runner._step_update_macro = failing_macro  # type: ignore[assignment]

    report = runner.run_daily_pipeline(dry_run=True, continue_on_error=False, symbols=["OGDC"])

    assert report.success is False
    task_map = {t.step: t for t in report.tasks}

    # First two steps succeed
    assert task_map[DailyPipelineStep.UPDATE_PRICES].status == "SUCCESS"
    assert task_map[DailyPipelineStep.COLLECT_NEWS].status == "SUCCESS"

    # Third step fails
    assert task_map[DailyPipelineStep.UPDATE_MACRO].status == "FAILED"
    assert "SBP portal timed out" in str(task_map[DailyPipelineStep.UPDATE_MACRO].error)

    # Subsequent steps are skipped
    assert task_map[DailyPipelineStep.BUILD_FEATURES].status == "SKIPPED"
    assert task_map[DailyPipelineStep.GENERATE_PREDICTIONS].status == "SKIPPED"
    assert task_map[DailyPipelineStep.AUDIT_PREDICTIONS].status == "SKIPPED"


def test_pipeline_retry_mechanism_recovers() -> None:
    """Verify exponential backoff retry succeeds after transient failures."""
    runner = DailyPipelineRunner(max_retries=3, retry_delay=0.01)

    mock_action = MagicMock(side_effect=[ValueError("Transient lock error"), {"recovered": True}])
    result = runner._execute_with_retry(
        mock_action,
        step_name="test_retry_step",
    )

    assert result == {"recovered": True}
    assert mock_action.call_count == 2


def test_cli_schedule_status() -> None:
    """Test 'psx schedule status' CLI command outputs timing table cleanly."""
    runner = CliRunner()
    result = runner.invoke(cli_app, ["schedule", "status"])
    assert result.exit_code == 0
    assert "PSX Predictor Automated Pipeline Status" in result.output
    assert "Monday - Thursday" in result.output
    assert "16:00 PKT" in result.output
    assert "Friday" in result.output
    assert "17:00 PKT" in result.output


def test_cli_schedule_run_daily_dry_run() -> None:
    """Test 'psx schedule run-daily --dry-run' executes full pipeline cleanly via CLI."""
    runner = CliRunner()
    result = runner.invoke(cli_app, ["schedule", "run-daily", "--dry-run", "--symbols", "OGDC,HBL"])
    assert result.exit_code == 0
    assert "Daily EOD Pipeline" in result.output
    assert "update_prices" in result.output
    assert "generate_predictions" in result.output
    assert "SUCCESS" in result.output
