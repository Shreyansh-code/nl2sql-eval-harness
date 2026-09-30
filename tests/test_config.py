"""Settings must never leak the API key into a committed run artifact."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.config import Settings, load_settings


def test_dotenv_populates_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERATOR_MODEL", "gpt6-luna")
    monkeypatch.setenv("JUDGE_MODEL", "gpt6-luna")
    settings = load_settings()
    assert settings.generator_model == "gpt6-luna"
    assert settings.judge_model == "gpt6-luna"


def test_real_environment_wins_over_the_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """override=False is what keeps CI from depending on a developer's local .env."""
    monkeypatch.setenv("MAX_ROWS", "7")
    assert load_settings().max_rows == 7


def test_redacted_snapshot_excludes_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")
    snapshot = load_settings().redacted()
    assert "openai_api_key" not in snapshot
    assert "sk-super-secret-value" not in repr(snapshot)


def test_redacted_snapshot_keeps_what_reproduces_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")
    snapshot = load_settings().redacted()
    for key in ("generator_model", "judge_model", "generator_mode", "judge_mode", "max_rows"):
        assert key in snapshot


def test_invalid_mode_is_rejected_at_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERATOR_MODE", "carrier-pigeon")
    with pytest.raises(ValueError, match="GENERATOR_MODE"):
        load_settings()


def test_relative_paths_resolve_against_the_repo_root() -> None:
    settings = load_settings()
    assert settings.subset_path.is_absolute()
    assert settings.data_dir.is_absolute()


def test_databases_dir_derives_from_data_dir() -> None:
    settings = Settings(
        generator_model=None,
        judge_model=None,
        generator_temperature=None,
        generator_mode="standard",
        judge_mode="batch",
        openai_base_url=None,
        openai_api_key=None,
        data_dir=Path("/tmp/bird"),
        subset_path=Path("/tmp/subset.yaml"),
        max_concurrency=8,
        sql_timeout_seconds=5.0,
        max_rows=1000,
        label_sample_size=50,
        langsmith_tracing=False,
    )
    assert settings.databases_dir.name == "dev_databases"


class TestPublishablePaths:
    def test_paths_inside_the_repo_become_relative(self) -> None:
        from harness.config import REPO_ROOT, _publishable_path

        assert _publishable_path(REPO_ROOT / "data" / "bird") == "data/bird"

    def test_paths_outside_the_repo_are_reduced_to_a_name(self) -> None:
        from harness.config import _publishable_path

        assert _publishable_path(Path("/Users/someone/secret/data")) == "data"

    def test_redacted_snapshot_has_no_home_directory(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        snapshot = load_settings().redacted()
        assert "/Users/" not in repr(snapshot)
        assert "sk-secret" not in repr(snapshot)
