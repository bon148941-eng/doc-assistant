import pytest

from assistant.config import DEFAULT_MIN_SCORE, DEFAULT_MODEL, ConfigError, load_config


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Настройки только из тестового окружения: .env проекта не читается."""
    monkeypatch.setattr("assistant.config.load_dotenv", lambda *_a, **_k: None)
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_MODEL", "DOCS_DIR", "TOP_K", "MIN_SCORE"):
        monkeypatch.delenv(name, raising=False)


def test_defaults():
    config = load_config()

    assert config.api_key is None
    assert config.model == DEFAULT_MODEL
    assert config.min_score == DEFAULT_MIN_SCORE
    assert config.docs_dir.name == "sample_docs"


def test_empty_values_mean_defaults(monkeypatch):
    monkeypatch.setenv("TOP_K", "")
    monkeypatch.setenv("MIN_SCORE", " ")

    assert load_config().min_score == DEFAULT_MIN_SCORE


@pytest.mark.parametrize("value", ["15", "-0.1", "1.5"])
def test_min_score_out_of_range_raises(monkeypatch, value):
    monkeypatch.setenv("MIN_SCORE", value)

    with pytest.raises(ConfigError, match="MIN_SCORE"):
        load_config()


def test_min_score_not_a_number_raises(monkeypatch):
    monkeypatch.setenv("MIN_SCORE", "много")

    with pytest.raises(ConfigError, match="MIN_SCORE"):
        load_config()


def test_top_k_must_be_positive(monkeypatch):
    monkeypatch.setenv("TOP_K", "0")

    with pytest.raises(ConfigError, match="TOP_K"):
        load_config()
