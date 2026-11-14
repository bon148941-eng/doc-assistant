"""Обращение к Claude API через официальный SDK anthropic."""

import anthropic

REQUEST_TIMEOUT = 30.0  # секунд на один запрос
MAX_RETRIES = 2  # повторы при сетевых сбоях, лимитах и 5xx делает сам SDK
MAX_TOKENS = 1024


class LLMError(Exception):
    """Ошибка обращения к API с текстом, понятным пользователю."""


def create_client(api_key: str | None) -> anthropic.Anthropic:
    """Создаёт клиент. Без ключа сразу даёт понятную ошибку."""
    if not api_key:
        raise LLMError(
            "Не задан ANTHROPIC_API_KEY. Скопируйте .env.example в .env "
            "и впишите ключ (см. README.md)."
        )
    return anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES)


def ask_claude(client: anthropic.Anthropic, model: str, system: str, user_message: str) -> str:
    """Отправляет запрос и возвращает текст ответа."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.AuthenticationError:
        raise LLMError("API отклонил ключ. Проверьте ANTHROPIC_API_KEY.") from None
    except anthropic.RateLimitError:
        raise LLMError("Превышен лимит запросов к API. Повторите попытку позже.") from None
    except (anthropic.APITimeoutError, anthropic.APIConnectionError):
        raise LLMError("Не удалось связаться с API (сеть или таймаут). Повторите попытку позже.") from None
    except anthropic.APIStatusError as exc:
        raise LLMError(f"API вернул ошибку (код {exc.status_code}). Повторите попытку позже.") from None
    except anthropic.APIError:
        # Остальные ошибки SDK (например, неожиданный формат ответа).
        raise LLMError("Не удалось получить корректный ответ от API. Повторите попытку позже.") from None

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise LLMError("API вернул пустой ответ.")
    return text
