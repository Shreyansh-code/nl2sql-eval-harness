"""Tracing must be invisible when off, and must never carry a credential."""

from __future__ import annotations

import pytest

from harness.tracing import langsmith


class TestDisabledByDefault:
    def test_starts_off(self) -> None:
        langsmith.configure(False)
        assert not langsmith.is_enabled()

    def test_passthrough_yields_an_empty_outputs_dict(self) -> None:
        with langsmith.trace("noop", inputs={"a": 1}) as outputs:
            outputs["result"] = "value"
        # Nothing is sent anywhere, and the block still yields a usable dict.

    def test_configure_false_clears_a_previous_enable(self) -> None:
        langsmith.configure(False)
        langsmith.configure(False)
        assert not langsmith.is_enabled()

    def test_enabling_without_an_api_key_stays_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tracing must not be switchable-on into a broken state by accident."""
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        assert langsmith.configure(True) is False
        assert not langsmith.is_enabled()

    def test_exceptions_propagate_even_with_tracing_wired(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        langsmith.configure(False)
        with pytest.raises(ValueError, match="boom"), langsmith.trace("failing"):
            raise ValueError("boom")


class TestRedaction:
    def test_secret_looking_keys_are_masked(self) -> None:
        redacted = langsmith._redact(
            {
                "openai_api_key": "sk-live-abc",
                "api_key": "sk-live-abc",
                "auth_token": "t",
                "password": "p",
                "authorization": "Bearer x",
                "question": "how many?",
            }
        )
        assert redacted["openai_api_key"] == "[redacted]"
        assert redacted["auth_token"] == "[redacted]"
        assert redacted["password"] == "[redacted]"
        assert redacted["authorization"] == "[redacted]"

    def test_ordinary_payload_fields_survive(self) -> None:
        redacted = langsmith._redact({"question_id": "717", "db_id": "superhero"})
        assert redacted == {"question_id": "717", "db_id": "superhero"}

    def test_no_secret_survives_redaction(self) -> None:
        payload = {"api_key": "sk-live-abc", "nested": "not-masked"}
        assert "sk-live-abc" not in repr(langsmith._redact(payload))


def test_redaction_covers_the_key_the_project_actually_uses(tmp_path) -> None:
    """The generator traces inputs; the key must never be one of them."""
    from pathlib import Path

    from harness.config import Settings

    settings = Settings(
        generator_model="gpt-6-luna",
        judge_model="gpt-6-luna",
        generator_temperature=None,
        generator_mode="standard",
        judge_mode="batch",
        openai_base_url=None,
        openai_api_key="sk-live-abc",
        data_dir=Path(tmp_path),
        subset_path=Path(tmp_path / "subset.yaml"),
        max_concurrency=8,
        sql_timeout_seconds=5.0,
        max_rows=1000,
        label_sample_size=50,
        langsmith_tracing=False,
    )
    assert "sk-live-abc" not in repr(settings.redacted())
