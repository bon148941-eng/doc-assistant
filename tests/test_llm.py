from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx2 as httpx  # HTTP-библиотека, которую использует SDK anthropic
import pytest

from assistant.llm import MAX_RETRIES, REQUEST_TIMEOUT, LLMError, ask_claude, create_client


def make_response(*blocks):
    return SimpleNamespace(content=list(blocks))


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def make_client(**kwargs):
    client = MagicMock()
    client.messages.create.return_value = kwargs.get("response")
    if "error" in kwargs:
        client.messages.create.side_effect = kwargs["error"]
    return client


def api_error(cls, status):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    return cls("ошибка", response=response, body=None)


def test_create_client_without_key_raises():
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        create_client(None)


def test_create_client_sets_timeout_and_retries():
    client = create_client("test-key")

    assert client.timeout == REQUEST_TIMEOUT
    assert client.max_retries == MAX_RETRIES


def test_ask_claude_returns_text_and_passes_arguments():
    client = make_client(response=make_response(text_block("Ответ")))

    answer = ask_claude(client, "test-model", "система", "вопрос")

    assert answer == "Ответ"
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["system"] == "система"
    assert kwargs["messages"] == [{"role": "user", "content": "вопрос"}]


def test_empty_response_raises():
    client = make_client(response=make_response())

    with pytest.raises(LLMError, match="пустой"):
        ask_claude(client, "m", "s", "q")


@pytest.mark.parametrize(
    "error, message",
    [
        (api_error(anthropic.AuthenticationError, 401), "ключ"),
        (api_error(anthropic.RateLimitError, 429), "лимит"),
        (api_error(anthropic.InternalServerError, 500), "500"),
        (
            anthropic.APIResponseValidationError(
                response=httpx.Response(200, request=httpx.Request("POST", "https://api.anthropic.com")),
                body=None,
            ),
            "корректный ответ",
        ),
        (
            anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com")),
            "сеть",
        ),
    ],
)
def test_api_errors_become_llm_error(error, message):
    client = make_client(error=error)

    with pytest.raises(LLMError, match=message):
        ask_claude(client, "m", "s", "q")
