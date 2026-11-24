"""Tests for stock research files: schema, store, point-in-time features, and wiring."""

import copy
import datetime
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.dashboard.components.research_view import (
    create_score_history_chart,
    files_table,
    overview_table,
    score_history,
)
from psx_predictor.predictions.predictor import LivePredictor
from psx_predictor.research.features import (
    RESEARCH_FEATURE_COLUMNS,
    build_research_features,
    recency_weight,
    research_context_for_date,
)
from psx_predictor.research.store import ResearchStore, validate_research_data
from psx_predictor.storage.parquet_io import write_parquet_atomic

FIXTURE = Path(__file__).parent.parent / "fixtures" / "research" / "OGDC_2026-11-20.json"


@pytest.fixture
def sample() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _write(dir_: Path, data: dict[str, Any], name: str | None = None) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / (name or f"{data['symbol']}_{data['research_date']}.json")
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestValidation:
    def test_fixture_is_valid(self, sample: dict[str, Any]) -> None:
        report, errors, warnings = validate_research_data(
            sample, "OGDC", datetime.date(2026, 11, 20)
        )
        assert report is not None
        assert errors == []
        assert report.stance_value == 0.5
        assert not any("source_ids" in w for w in warnings)

    def test_bad_enum_and_range_rejected(self, sample: dict[str, Any]) -> None:
        bad = copy.deepcopy(sample)
        bad["recommendation"]["stance"] = "ACCUMULATE"
        bad["scores"]["risk_score"] = 150
        report, errors, _ = validate_research_data(bad)
        assert report is None
        assert any("stance" in e for e in errors)
        assert any("risk_score" in e for e in errors)

    def test_lookahead_news_is_error(self, sample: dict[str, Any]) -> None:
        bad = copy.deepcopy(sample)
        bad["news"][0]["date"] = "2026-11-25"
        _, errors, _ = validate_research_data(bad)
        assert any("look-ahead" in e for e in errors)

    def test_filename_mismatch_and_unknown_sources(self, sample: dict[str, Any]) -> None:
        data = copy.deepcopy(sample)
        data["risks"][0]["source_ids"] = ["S99"]
        _, errors, warnings = validate_research_data(data, "PPL", datetime.date(2026, 11, 21))
        assert len(errors) == 2
        assert any("S99" in w for w in warnings)

    def test_missing_optional_sections_default(self, sample: dict[str, Any]) -> None:
        minimal = {k: sample[k] for k in ("schema_version", "symbol", "research_date", "scores")}
        report, errors, warnings = validate_research_data(minimal)
        assert report is not None and errors == []
        assert report.news == []
        assert any("Missing keys" in w for w in warnings)


class TestStore:
    def test_scan_sorts_and_flags_invalid(self, tmp_path: Path, sample: dict[str, Any]) -> None:
        rdir = tmp_path / "research"
        _write(rdir, sample)
        newer = copy.deepcopy(sample)
        newer["research_date"] = "2026-11-22"
        _write(rdir, newer)
        (rdir / "OGDC_2026-11-23.json").write_text("{not json", encoding="utf-8")
        (rdir / "notes.json").write_text("{}", encoding="utf-8")

        store = ResearchStore(research_dir=rdir)
        files = store.scan("OGDC")
        assert [str(f.research_date) for f in files][:2] == ["2026-11-23", "2026-11-22"]
        assert not files[0].is_valid
        assert len(store.load_reports("OGDC")) == 2
        latest = store.latest("OGDC", as_of=datetime.date(2026, 11, 21))
        assert latest is not None and str(latest.research_date) == "2026-11-20"
        assert any(not f.is_valid for f in store.scan())  # notes.json flagged

    def test_utf8_bom_is_accepted(self, tmp_path: Path, sample: dict[str, Any]) -> None:
        rdir = tmp_path / "research"
        rdir.mkdir()
        path = rdir / "OGDC_2026-11-20.json"
        path.write_text(json.dumps(sample), encoding="utf-8-sig")
        assert ResearchStore(research_dir=rdir).load_file(path).is_valid

    def test_storage_paths_without_root(self, tmp_path: Path) -> None:
        store = ResearchStore(storage_paths={"predictions": tmp_path})
        assert store.scan() == []


class TestFeatures:
    def test_point_in_time_and_decay(self, sample: dict[str, Any]) -> None:
        report, _, _ = validate_research_data(sample)
        assert report is not None
        dates = ["2026-11-19", "2026-11-20", "2026-12-20", "2027-04-01"]
        df = build_research_features(dates, [report])

        assert list(df.columns) == RESEARCH_FEATURE_COLUMNS
        # Before the research date: neutral (no look-ahead)
        assert df.loc[0, "research_available"] == 0.0
        assert df.loc[0, "research_sentiment"] == 0.0
        # On the research date: full weight times confidence
        assert df.loc[1, "research_weight"] == pytest.approx(0.8)
        assert df.loc[1, "research_sentiment"] == pytest.approx(0.4 * 0.8)
        assert df.loc[1, "research_valuation"] == pytest.approx(0.5 * 0.8)
        assert df.loc[1, "research_upside"] == pytest.approx(0.2 * 0.8)
        assert df.loc[1, "research_days_to_event"] == 4.0
        assert df.loc[1, "research_event_soon"] == 1.0
        # 30 days later: half weight; event has passed
        assert df.loc[2, "research_weight"] == pytest.approx(0.4)
        assert df.loc[2, "research_event_soon"] == 0.0
        # Beyond max age: neutral again
        assert df.loc[3, "research_available"] == 0.0

    def test_newer_research_replaces_older(self, sample: dict[str, Any]) -> None:
        old, _, _ = validate_research_data(sample)
        newer_data = copy.deepcopy(sample)
        newer_data["research_date"] = "2026-11-25"
        newer_data["scores"]["overall_sentiment"] = -0.5
        new, _, _ = validate_research_data(newer_data)
        assert old is not None and new is not None
        df = build_research_features(["2026-11-24", "2026-11-25"], [new, old])
        assert df.loc[0, "research_sentiment"] > 0
        assert df.loc[1, "research_sentiment"] == pytest.approx(-0.5 * 0.8)

    def test_recency_weight_bounds(self) -> None:
        assert recency_weight(0, None) == pytest.approx(0.5)
        assert recency_weight(-1, 1.0) == 0.0
        assert recency_weight(121, 1.0) == 0.0

    def test_context(self, sample: dict[str, Any]) -> None:
        report, _, _ = validate_research_data(sample)
        assert report is not None
        assert research_context_for_date([report], "2026-11-19") == {}
        ctx = research_context_for_date([report], "2026-11-21")
        assert ctx["stance"] == "BUY"
        assert ctx["next_event"]["type"] == "POLICY_RATE"
        assert ctx["top_catalysts"] == ["Circular debt settlement (positive)"]


class TestDashboardHelpers:
    def test_tables_and_chart(self, tmp_path: Path, sample: dict[str, Any]) -> None:
        rdir = tmp_path / "research"
        _write(rdir, sample)
        newer = copy.deepcopy(sample)
        newer["research_date"] = "2026-11-22"
        _write(rdir, newer)
        files = ResearchStore(research_dir=rdir).scan()

        assert len(files_table(files)) == 2
        ov = overview_table(files, today=datetime.date(2026, 11, 23))
        assert ov.loc[0, "Research date"] == "2026-11-22"
        assert ov.loc[0, "Age (days)"] == 1
        hist = score_history(files)
        assert len(hist) == 2
        fig = create_score_history_chart(hist, "OGDC")
        assert len(fig.data) == 4


class TestPredictionWiring:
    def _paths(self, tmp_path: Path) -> dict[str, Path]:
        paths = {
            "root": tmp_path,
            "features_technical": tmp_path / "features" / "technical",
            "predictions": tmp_path / "predictions",
            "research": tmp_path / "research",
        }
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)
        return paths

    def _features(self, start: str) -> pd.DataFrame:
        n = 60
        dates = pd.date_range(start=start, periods=n, freq="B")
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "trade_date": [d.strftime("%Y-%m-%d") for d in dates],
                "returns_1d": rng.normal(0, 0.01, n),
                "rsi_14": rng.uniform(30, 70, n),
                "target_next_day_dir": [i % 2 for i in range(n)],
            }
        )
        df.loc[n - 1, "target_next_day_dir"] = np.nan
        return df

    def test_prediction_uses_research(self, tmp_path: Path, sample: dict[str, Any]) -> None:
        paths = self._paths(tmp_path)
        feats = self._features("2026-09-01")
        write_parquet_atomic(feats, paths["features_technical"] / "OGDC_tech_features.parquet")
        shutil.copy(FIXTURE, paths["research"] / FIXTURE.name)

        rec = LivePredictor(storage_paths=paths).generate_prediction(
            "OGDC", model_name="logistic", log_to_registry=False
        )
        ctx = json.loads(rec.research_json)
        assert ctx["research_date"] == "2026-11-20"
        assert ctx["stance"] == "BUY"

        rec_off = LivePredictor(storage_paths=paths).generate_prediction(
            "OGDC", model_name="logistic", log_to_registry=False, use_research=False
        )
        assert rec_off.research_json == "{}"

    def test_prediction_without_research(self, tmp_path: Path) -> None:
        paths = self._paths(tmp_path)
        write_parquet_atomic(
            self._features("2024-01-01"),
            paths["features_technical"] / "OGDC_tech_features.parquet",
        )
        rec = LivePredictor(storage_paths=paths).generate_prediction(
            "OGDC", model_name="logistic", log_to_registry=False
        )
        assert rec.research_json == "{}"


class TestCli:
    def test_list_validate_show(
        self, tmp_path: Path, sample: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rdir = tmp_path / "research"
        _write(rdir, sample)
        store = ResearchStore(research_dir=rdir)
        monkeypatch.setattr("psx_predictor.cli.main._research_store", lambda: store)
        runner = CliRunner()

        res = runner.invoke(app, ["research", "list"])
        assert res.exit_code == 0 and "OGDC" in res.stdout
        res = runner.invoke(app, ["research", "validate"])
        assert res.exit_code == 0 and "1 valid, 0 invalid" in res.stdout
        res = runner.invoke(app, ["research", "show", "--symbol", "OGDC"])
        assert res.exit_code == 0 and "BUY" in res.stdout

        (rdir / "PPL_2026-11-20.json").write_text("[]", encoding="utf-8")
        res = runner.invoke(app, ["research", "validate"])
        assert res.exit_code == 1


class TestResearchVideos:
    def test_list_and_find_videos(self, tmp_path: Path, sample: dict[str, Any]) -> None:
        from psx_predictor.research.videos import find_research_video, list_research_videos

        rdir = tmp_path / "research"
        vdir = rdir / "videos"
        vdir.mkdir(parents=True)
        (vdir / "OGDC_2026-11-20.mp4").write_bytes(b"\x00")
        (vdir / "OGDC_2026-11-20.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")
        (vdir / "PPL_2026-11-21.mp4").write_bytes(b"\x00")
        (vdir / "notes.mp4").write_bytes(b"\x00")
        (vdir / "index.json").write_text(
            json.dumps(
                [
                    {
                        "symbol": "OGDC",
                        "research_date": "2026-11-20",
                        "video": "OGDC_2026-11-20.mp4",
                        "subtitles": "OGDC_2026-11-20.srt",
                        "duration_sec": 214.0,
                        "style": "Broker research note",
                        "language": "English",
                    },
                    {"video": "GONE_2026-01-01.mp4"},
                ]
            ),
            encoding="utf-8",
        )

        videos = list_research_videos(rdir)
        assert [v.symbol for v in videos] == ["PPL", "OGDC"]
        ogdc = find_research_video(rdir, "ogdc", datetime.date(2026, 11, 20))
        assert ogdc is not None
        assert ogdc.duration_sec == 214.0 and ogdc.style == "Broker research note"
        assert ogdc.subtitles_path is not None
        ppl = find_research_video(rdir, "PPL", datetime.date(2026, 11, 21))
        assert ppl is not None and ppl.subtitles_path is None and ppl.duration_sec is None
        assert find_research_video(rdir, "OGDC", datetime.date(2026, 11, 22)) is None

        _write(rdir, sample)
        table = files_table(ResearchStore(research_dir=rdir).scan(), {v.key for v in videos})
        assert table.loc[0, "Video"] == "Yes"

    def test_bad_index_and_missing_folder(self, tmp_path: Path) -> None:
        from psx_predictor.research.videos import list_research_videos

        assert list_research_videos(tmp_path / "research") == []
        vdir = tmp_path / "research" / "videos"
        vdir.mkdir(parents=True)
        (vdir / "OGDC_2026-11-20.mp4").write_bytes(b"\x00")
        (vdir / "index.json").write_text("{broken", encoding="utf-8")
        assert len(list_research_videos(tmp_path / "research")) == 1


class TestVideoBrief:
    def test_video_brief_parsed_and_optional(self, sample: dict[str, Any]) -> None:
        report, errors, warnings = validate_research_data(sample)
        assert report is not None and errors == []
        assert report.video_brief is not None
        assert report.video_brief.tone == "BALANCED"
        assert report.video_brief.pronunciations[0].say == "O G D C"

        older = copy.deepcopy(sample)
        del older["video_brief"]
        report, errors, warnings = validate_research_data(older)
        assert report is not None and report.video_brief is None
        assert not any("Missing keys" in w for w in warnings)

    def test_bad_tone_rejected(self, sample: dict[str, Any]) -> None:
        bad = copy.deepcopy(sample)
        bad["video_brief"]["tone"] = "EXCITED"
        report, errors, _ = validate_research_data(bad)
        assert report is None and any("tone" in e for e in errors)
