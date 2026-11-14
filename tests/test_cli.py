from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from assistant.__main__ import answer_question, main, run_chat
from assistant.config import Config
from assistant.llm import LLMError
from assistant.loader import Chunk
from assistant.prompts import NO_INFO_MESSAGE
from assistant.search import Retriever


@pytest.fixture
def config():
    return Config(api_key="test-key", model="test-model", docs_dir=Path("."), top_k=2, min_score=0.15)


@pytest.fixture
def retriever():
    return Retriever([
        Chunk("delivery.md", 1, "Курьерская доставка стоит 300 рублей."),
        Chunk("returns.md", 1, "Вернуть товар можно в течение 30 дней."),
    ])


def client_replying(text):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
    return client


def test_relevant_question_calls_api_and_lists_sources(retriever, config):
    client = client_replying("Доставка стоит 300 рублей.")

    answer = answer_question("Сколько стоит курьерская доставка?", retriever, config, client)

    client.messages.create.assert_called_once()
    assert client.messages.create.call_args.kwargs["model"] == "test-model"
    assert "Доставка стоит 300 рублей." in answer
    assert "Источники:" in answer
    assert "delivery.md, фрагмент 1" in answer


def test_irrelevant_question_does_not_call_api(retriever, config):
    client = client_replying("не должно быть вызвано")

    answer = answer_question("Какая погода на Марсе?", retriever, config, client)

    assert answer == NO_INFO_MESSAGE
    client.messages.create.assert_not_called()


def test_score_below_threshold_does_not_call_api(retriever, config):
    client = client_replying("не должно быть вызвано")
    strict = Config(**{**config.__dict__, "min_score": 0.99})

    answer = answer_question("доставка", retriever, strict, client)

    assert answer == NO_INFO_MESSAGE
    client.messages.create.assert_not_called()


def test_model_no_info_answer_has_no_sources(retriever, config):
    client = client_replying(NO_INFO_MESSAGE)

    answer = answer_question("Сколько стоит курьерская доставка?", retriever, config, client)

    assert answer == NO_INFO_MESSAGE
    assert "Источники" not in answer


def test_weak_fragments_are_not_listed_as_sources(config):
    retriever = Retriever([
        Chunk("delivery.md", 1, "Курьерская доставка стоит 300 рублей."),
        Chunk("vacation.md", 1, "Отпуск составляет 28 дней. Доставка отпуска не предусмотрена."),
        Chunk("returns.md", 1, "Вернуть товар можно в течение 30 дней."),
    ])
    client = client_replying("Доставка стоит 300 рублей.")
    results = retriever.search("курьерская доставка стоит", config.top_k)
    assert len(results) >= 2  # второй фрагмент найден, но слабее первого

    answer = answer_question("курьерская доставка стоит", retriever, config, client)

    sources = answer.split("Источники:")[1]
    assert "delivery.md, фрагмент 1" in sources
    assert "vacation.md" not in sources


def test_model_sees_only_fragments_listed_as_sources(config):
    retriever = Retriever([
        Chunk("delivery.md", 1, "Курьерская доставка стоит 300 рублей."),
        Chunk("vacation.md", 1, "Отпуск составляет 28 дней. Доставка отпуска не предусмотрена."),
        Chunk("returns.md", 1, "Вернуть товар можно в течение 30 дней."),
    ])
    client = client_replying("Доставка стоит 300 рублей.")

    answer = answer_question("курьерская доставка стоит", retriever, config, client)

    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    sources = answer.split("Источники:")[1]
    assert 'source="delivery.md"' in prompt
    assert 'source="vacation.md"' not in prompt
    assert "vacation.md" not in sources


@pytest.mark.parametrize("reply", [
    "Доставка стоит 300 рублей. В документах нет информации о сроках.",
    "Отпуск 28 дней\nВ документах нет информации о переносе",
    "- Отпуск 28 дней\n- в документах нет информации о переносе",
    "Отпуск 28 дней; в документах нет информации о переносе!",
    "Срок возврата 14 дней, про обмен в документах нет информации",
    "Срок возврата 14 дней, про обмен в документах нет информации.",
    "Обмен возможен, в документах нет информации о возврате.",
    "Доставка бесплатная, в документах нет информации о самовывозе.",
    "Да, но в документах нет информации о сроках.",
    "Нет, в документах нет информации об этом.",
    "Только курьером: в документах нет информации о самовывозе.",
])
def test_partial_answer_with_disclaimer_keeps_sources(retriever, config, reply):
    client = client_replying(reply)

    answer = answer_question("Сколько стоит курьерская доставка?", retriever, config, client)

    assert answer.startswith(reply)
    assert "Источники:\n- delivery.md, фрагмент 1" in answer


@pytest.mark.parametrize("reply", [
    "В документах нет информации по этому вопросу",
    "в документах нет информации об этом, увы!",
    "«В документах нет информации по этому вопросу.»",
    "**В документах нет информации по этому вопросу.**",
    "К сожалению, в документах нет информации по этому вопросу.",
    "Ответ: в документах нет информации по этому вопросу.",
])
def test_no_info_reply_without_exact_punctuation_has_no_sources(retriever, config, reply):
    client = client_replying(reply)

    answer = answer_question("Сколько стоит курьерская доставка?", retriever, config, client)

    assert answer == reply
    assert "Источники" not in answer


def test_chat_exits_on_ctrl_c(retriever, config, monkeypatch, capsys):
    def interrupt(_prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)

    assert run_chat(retriever, config, MagicMock()) == 0


def test_chat_survives_api_error_and_exits(retriever, config, monkeypatch, capsys):
    questions = iter(["Сколько стоит доставка?", "", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(questions))
    client = MagicMock()
    client.messages.create.side_effect = LLMError("API недоступен")

    assert run_chat(retriever, config, client) == 0

    assert "API недоступен" in capsys.readouterr().err
    assert client.messages.create.call_count == 1


def test_main_without_api_key_prints_clear_error(monkeypatch, capsys):
    monkeypatch.setattr("assistant.config.load_dotenv", lambda *_a, **_k: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert main(["ask", "вопрос"]) == 1

    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
