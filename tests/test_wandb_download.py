from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


def _load_wandb_download_module():
    module_path = Path(__file__).resolve().parents[1] / "sumo_rl" / "wandb_download.py"
    spec = importlib.util.spec_from_file_location("sumo_rl_wandb_download", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


wandb_download = _load_wandb_download_module()


def test_load_wandb_credentials_reads_repo_env(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "WANDB_API_KEY=test-key\nWANDB_ENTITY=test-entity\nWANDB_PROJECT=test-project\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)

    credentials = wandb_download.load_wandb_credentials(explicit_env_file=str(env_path))

    assert credentials["api_key"] == "test-key"
    assert credentials["entity"] == "test-entity"
    assert credentials["project"] == "test-project"
    assert credentials["env_path"] == str(env_path)


def test_require_all_tags_accepts_full_match_and_rejects_missing_tag():
    assert wandb_download.require_all_tags(["thesis", "resco_grid4x4"], ["thesis"]) is True
    assert wandb_download.require_all_tags(["thesis", "resco_grid4x4"], ["thesis", "resco_grid4x4"]) is True
    assert wandb_download.require_all_tags(["thesis"], ["thesis", "resco_grid4x4"]) is False


def test_require_run_name_accepts_empty_filter_and_exact_matches():
    assert wandb_download.require_run_name("Run One", []) is True
    assert wandb_download.require_run_name("Run One", ["Run One", "Run Two"]) is True
    assert wandb_download.require_run_name("Run Three", ["Run One", "Run Two"]) is False


class DummyRun:
    def __init__(self, run_id, name, tags, history_rows):
        self.id = run_id
        self.name = name
        self.tags = tags
        self.config = {"learning_rate": 0.001}
        self.summary = {"validation/resco_delay_mean": 7.5}
        self.group = "demo-group"
        self.job_type = "train"
        self.state = "finished"
        self.url = f"https://wandb.example/{run_id}"
        self.path = ["entity", "project", run_id]
        self.created_at = "2026-06-23T00:00:00"
        self.updated_at = "2026-06-23T01:00:00"
        self.entity = type("Entity", (), {"name": "entity"})()
        self.project = type("Project", (), {"name": "project"})()
        self._history_rows = history_rows

    def scan_history(self):
        for row in self._history_rows:
            yield row


class DummyApi:
    def __init__(self, runs):
        self._runs = runs
        self.calls = []

    def runs(self, path):
        self.calls.append(path)
        return list(self._runs)


def test_download_runs_exports_only_matching_runs(tmp_path):
    matching_run = DummyRun(
        "run-1",
        "Run One",
        ["thesis", "resco_grid4x4"],
        [{"train/episode_index": 1, "train/reward_mean": 2.0}],
    )
    non_matching_run = DummyRun(
        "run-2",
        "Run Two",
        ["other"],
        [{"train/episode_index": 1, "train/reward_mean": 1.0}],
    )
    api = DummyApi([matching_run, non_matching_run])

    manifest = wandb_download.download_runs(
        api=api,
        entity="entity",
        project="project",
        required_tags=["thesis", "resco_grid4x4"],
        required_names=None,
        output_dir=tmp_path,
    )

    export_dir = tmp_path / "entity" / "project" / "run-1__Run-One"
    assert api.calls == ["entity/project"]
    assert manifest["matched_runs"] == 1
    assert manifest["exported_runs"] == 1
    assert manifest["skipped_runs"] == 0
    assert export_dir.exists()
    assert json.loads((export_dir / "run.json").read_text(encoding="utf-8"))["id"] == "run-1"
    assert json.loads((export_dir / "config.json").read_text(encoding="utf-8"))["learning_rate"] == 0.001
    assert json.loads((export_dir / "summary.json").read_text(encoding="utf-8"))["validation/resco_delay_mean"] == 7.5
    history_lines = (export_dir / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(history_lines) == 1
    assert json.loads(history_lines[0])["train/reward_mean"] == 2.0

    manifest_path = tmp_path / "entity" / "project" / "download_manifest.json"
    written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written_manifest["matched_runs"] == 1
    assert written_manifest["runs"][0]["run_id"] == "run-1"


def test_download_runs_can_filter_by_exact_run_names(tmp_path):
    matching_run = DummyRun(
        "run-1",
        "Run One",
        ["thesis", "resco_grid4x4"],
        [{"train/episode_index": 1, "train/reward_mean": 2.0}],
    )
    other_tag_match = DummyRun(
        "run-2",
        "Run Two",
        ["thesis", "resco_grid4x4"],
        [{"train/episode_index": 1, "train/reward_mean": 1.0}],
    )
    api = DummyApi([matching_run, other_tag_match])

    manifest = wandb_download.download_runs(
        api=api,
        entity="entity",
        project="project",
        required_tags=["thesis", "resco_grid4x4"],
        required_names=["Run One"],
        output_dir=tmp_path,
        dry_run=True,
    )

    assert manifest["matched_runs"] == 1
    assert manifest["runs"][0]["run_name"] == "Run One"
    assert manifest["requested_run_names"] == ["Run One"]


def test_main_prefers_cli_entity_and_project_over_env(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "WANDB_API_KEY=test-key\nWANDB_ENTITY=env-entity\nWANDB_PROJECT=env-project\n",
        encoding="utf-8",
    )

    captured = {}

    class DummyWandb:
        @staticmethod
        def Api(api_key):
            captured["api_key"] = api_key
            return "dummy-api"

    def fake_download_runs(**kwargs):
        captured.update(kwargs)
        return {
            "entity": kwargs["entity"],
            "project": kwargs["project"],
            "requested_tags": kwargs["required_tags"],
            "requested_run_names": kwargs["required_names"],
            "matched_runs": 0,
            "exported_runs": 0,
            "skipped_runs": 0,
            "dry_run": kwargs["dry_run"],
        }

    monkeypatch.setitem(os.sys.modules, "wandb", DummyWandb)
    monkeypatch.setattr(wandb_download, "download_runs", fake_download_runs)

    result = wandb_download.main(
        [
            "--env-file",
            str(env_path),
            "--entity",
            "cli-entity",
            "--project",
            "cli-project",
            "--tag",
            "thesis",
            "--run-name",
            "Run One",
            "--dry-run",
        ]
    )

    assert result == 0
    assert captured["api_key"] == "test-key"
    assert captured["entity"] == "cli-entity"
    assert captured["project"] == "cli-project"
    assert captured["required_tags"] == ["thesis"]
    assert captured["required_names"] == ["Run One"]
    assert captured["dry_run"] is True
