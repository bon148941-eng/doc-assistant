import pytest

from assistant.loader import Chunk
from assistant.search import Retriever, preprocess


@pytest.fixture
def retriever():
    chunks = [
        Chunk("vacation.md", 1, "Ежегодный отпуск составляет 28 календарных дней."),
        Chunk("delivery.md", 1, "Курьерская доставка стоит 300 рублей."),
        Chunk("returns.md", 1, "Вернуть товар можно в течение 30 дней."),
    ]
    return Retriever(chunks)


def test_preprocess_drops_only_generic_stop_words():
    assert preprocess("Сколько СТОИТ доставка компании?") == "стоит доставка компании"


def test_best_chunk_comes_first(retriever):
    results = retriever.search("Сколько стоит доставка курьером?", top_k=3)

    assert results[0].chunk.source == "delivery.md"
    assert results[0].score > 0


@pytest.mark.parametrize("query", ["отпуска", "отпуске", "отпусков"])
def test_inflected_words_still_match(retriever, query):
    assert retriever.search(query, top_k=1)[0].chunk.source == "vacation.md"


def test_irregular_forms_share_part_of_the_word(retriever):
    # «рубли» и «рублей» отличаются окончанием, но делят начало слова.
    assert retriever.search("рубли", top_k=1)[0].chunk.source == "delivery.md"


def test_top_k_limits_results(retriever):
    assert len(retriever.search("дней рублей", top_k=1)) == 1


def test_query_without_common_text_gives_no_results(retriever):
    assert retriever.search("xyz qwerty", top_k=3) == []


def test_frequent_terms_are_dropped_on_large_collections():
    # «альфа» есть в каждом фрагменте, остальные слова уникальны для своего фрагмента.
    chunks = [Chunk("d.md", i, f"альфа слово{chr(0x430 + i)}{chr(0x430 + i)}{chr(0x430 + i)}") for i in range(1, 7)]

    assert Retriever(chunks).search("альфа", top_k=3) == []
