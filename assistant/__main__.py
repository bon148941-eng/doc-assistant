"""Консольный интерфейс: `python -m assistant ask "вопрос"` и `python -m assistant chat`."""

import argparse
import re
import sys

from .config import Config, ConfigError, load_config
from .llm import LLMError, ask_claude, create_client
from .loader import DocsError, load_chunks
from .prompts import NO_INFO_MESSAGE, NO_INFO_PHRASE, SYSTEM_PROMPT, build_user_prompt, format_source
from .search import Retriever

EXIT_COMMAND = "exit"
# В источники попадают фрагменты не слабее этой доли от лучшего результата.
SOURCE_SCORE_RATIO = 0.5
# Вводные слова, после которых фраза об отсутствии информации всё ещё считается отказом.
REFUSAL_INTROS = ("к сожалению", "увы", "извините", "извините, но", "простите", "но", "ответ")


def is_refusal(answer: str) -> bool:
    """Отказ: ответ начинается с фразы или перед ней стоит только вводное из списка."""
    text = answer.strip().strip("*_`#>«»\"' ").lower()
    text = re.sub(r"[ \t]+", " ", text)
    position = text.find(NO_INFO_PHRASE)
    if position < 0:
        return False
    prefix = text[:position].rstrip(" ,:-–—")
    return not prefix or prefix in REFUSAL_INTROS


def answer_question(question: str, retriever: Retriever, config: Config, client) -> str:
    """Полный цикл: поиск, проверка порога, запрос к модели, добавление источников."""
    results = retriever.search(question, config.top_k)
    # Низкая релевантность: документы вопроса не касаются, API не вызываем.
    if not results or results[0].score < config.min_score:
        return NO_INFO_MESSAGE

    # Модели передаём только сильные совпадения: они же становятся источниками,
    # поэтому ответ не может опираться на фрагмент, которого нет в списке.
    source_threshold = max(config.min_score, results[0].score * SOURCE_SCORE_RATIO)
    strong = [result for result in results if result.score >= source_threshold]

    answer = ask_claude(client, config.model, SYSTEM_PROMPT, build_user_prompt(question, strong))
    # Отказ (см. is_refusal) источников не получает; частичный ответ с оговоркой
    # в конце остаётся ответом и получает источники.
    if is_refusal(answer):
        return answer

    sources = "\n".join(f"- {format_source(result)}" for result in strong)
    return f"{answer}\n\nИсточники:\n{sources}"


def run_ask(question: str, retriever: Retriever, config: Config, client) -> int:
    try:
        print(answer_question(question, retriever, config, client))
    except LLMError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


def run_chat(retriever: Retriever, config: Config, client) -> int:
    print(f"Задайте вопрос по документам. Для выхода введите {EXIT_COMMAND}.")
    while True:
        try:
            question = input("Вопрос> ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+Z / Ctrl+D / Ctrl+C
            print()
            break
        if question.lower() == EXIT_COMMAND:
            break
        if not question:
            continue
        try:
            print(answer_question(question, retriever, config, client), end="\n\n")
        except LLMError as exc:
            # В режиме chat ошибка API не должна завершать программу.
            print(f"Ошибка: {exc}\n", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="assistant", description="Ассистент по документам компании")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask = subparsers.add_parser("ask", help="задать один вопрос")
    ask.add_argument("question", help="текст вопроса")
    subparsers.add_parser("chat", help="интерактивный режим")
    return parser


def main(argv: list[str] | None = None) -> int:
    # На Windows консоль может быть не в UTF-8, а в ответах русский текст.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)
    try:
        config = load_config()
        client = create_client(config.api_key)
        retriever = Retriever(load_chunks(config.docs_dir))
    except (ConfigError, LLMError, DocsError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    if args.command == "ask":
        return run_ask(args.question, retriever, config, client)
    return run_chat(retriever, config, client)


if __name__ == "__main__":
    sys.exit(main())
