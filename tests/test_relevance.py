"""Наборы вопросов «по теме / не по теме» на реальных sample_docs.

Проверяют, что порог MIN_SCORE по умолчанию не отсекает вопросы по документам
и отсекает заведомо посторонние вопросы без обращения к API. Сеть не нужна.

Приоритет: вопрос по теме отсекаться не должен никогда (ложный пропуск стоит
одного запроса к API, и модель сама ответит «нет информации»). Поэтому порог
выбран по вопросам по теме, а часть вопросов не по теме проходит порог:
они вынесены в LEXICALLY_CLOSE и описаны в README как ограничение лексического поиска.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from assistant.__main__ import answer_question
from assistant.config import DEFAULT_MIN_SCORE, DEFAULT_TOP_K, Config
from assistant.loader import load_chunks
from assistant.prompts import NO_INFO_MESSAGE
from assistant.search import Retriever

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "sample_docs"

# Набор для подбора порога.
ON_TOPIC_TUNING = [
    "Сколько дней отпуска положено?",
    "Сколько стоит курьерская доставка?",
    "Как вернуть товар?",
    "Как оформить отпуск?",
    "За сколько дней подавать заявление на отпуск?",
    "Сколько хранится заказ в пункте выдачи?",
    "Когда вернут деньги за возврат?",
    "Можно ли разделить отпуск на части?",
    "Доставляете ли вы за границу?",
    "Как отследить заказ?",
    "Какие товары нельзя вернуть?",
    "Сколько стоит доставка?",
    "Что делать, если меня не было дома при доставке?",
    "Сколько дней отпуска за стаж?",
    "Можно ли обменять товар на другой размер?",
    "Где мои заказы?",
    "Сколько денег вернут?",
    "Сроки возврата товаров",
]
OFF_TOPIC_TUNING = [
    "Как достать билеты в театр?",
    "Какая погода на Марсе?",
    "Кто выиграл чемпионат мира по футболу?",
    "Сколько дней в високосном году?",
]

# Отдельный набор: вопросы написаны после подбора порога и на него не влияли.
ON_TOPIC_HELD_OUT = [
    "Через сколько дней вернут деньги?",
    "Можно ли перенести отпуск?",
    "Когда приедет курьер, если заказ оформлен вечером?",
    "Сколько дней хранится посылка в пункте выдачи?",
    "Надо ли платить за обратную доставку при возврате?",
    "За какой срок до отпуска нужно подавать заявление?",
    "Что будет с неиспользованными днями отпуска?",
    "Как продлить отпуск из-за больничного?",
]
OFF_TOPIC_HELD_OUT = [
    "Как настроить роутер дома?",
    "Какая столица Австралии?",
    "Сколько калорий в банане?",
    "Как выучить английский язык?",
    "Где купить зимние шины?",
    "Как посадить помидоры на балконе?",
    "Кто написал роман Война и мир?",
]

# Не по теме, но порог их пропускает (честное ограничение лексического поиска):
# общие слова и совпадающие куски слов («курьер», «выдачи», «отпуск», «заказ», «компании»).
# Отсечь их, не отсекая вопросы по теме, не удалось. Тест фиксирует это, чтобы
# изменение поведения не прошло незамеченным: модель в этом случае сама ответит
# «в документах нет информации».
LEXICALLY_CLOSE = [
    "Кто президент компании?",
    "Отпустите меня домой пораньше",
    "Сколько стоит iPhone?",
    "Как приготовить борщ?",
    "Как оформить кредит в банке?",
    "Как заказать такси?",
    "Расскажи анекдот про программистов",
    "Как стать курьером?",
    "Как работает пункт выдачи Почты России?",
    "Сколько стоит доставка пиццы?",
    "Сколько отпуска у депутатов?",
]

ON_TOPIC = ON_TOPIC_TUNING + ON_TOPIC_HELD_OUT
OFF_TOPIC = OFF_TOPIC_TUNING + OFF_TOPIC_HELD_OUT


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_chunks(SAMPLE_DOCS))


@pytest.fixture
def config():
    return Config(
        api_key="test-key", model="test-model", docs_dir=SAMPLE_DOCS,
        top_k=DEFAULT_TOP_K, min_score=DEFAULT_MIN_SCORE,
    )


def best_score(retriever, question):
    results = retriever.search(question, DEFAULT_TOP_K)
    return results[0].score if results else 0.0


@pytest.mark.parametrize("question", ON_TOPIC)
def test_on_topic_question_passes_threshold(retriever, question):
    assert best_score(retriever, question) >= DEFAULT_MIN_SCORE


@pytest.mark.parametrize("question", OFF_TOPIC)
def test_off_topic_question_is_rejected_without_api(retriever, config, question):
    client = MagicMock()

    assert answer_question(question, retriever, config, client) == NO_INFO_MESSAGE
    client.messages.create.assert_not_called()


@pytest.mark.parametrize("question", LEXICALLY_CLOSE)
def test_lexically_close_question_reaches_the_model(retriever, question):
    assert best_score(retriever, question) >= DEFAULT_MIN_SCORE
