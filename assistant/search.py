"""Поиск релевантных фрагментов методом TF-IDF."""

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer

from .loader import Chunk

# Служебные слова русского языка: не несут темы и дают ложные совпадения.
STOP_WORDS = frozenset(
    "и в во не на с со к ко у о об от до по за из или а но же ли бы то это как что кто где когда "
    "какой какая какое какие каков сколько почему зачем есть быть можно нужно для при про над под "
    "мне меня мой моя мои вы ваш нас нам он она оно они мы я".split()
)

# Русские слова склоняются, поэтому сравниваем не слова целиком, а их куски:
# символьные n-граммы длиной 3-5 внутри слов («возврат» и «возврата» делят почти все n-граммы).
NGRAM_RANGE = (3, 5)
# Куски, встречающиеся более чем в 80% фрагментов, отбрасываются по данным:
# это общие для всей коллекции слова, а не признак темы.
MAX_DF = 0.8
# На малом числе фрагментов статистика частот ненадёжна, там MAX_DF не применяется.
MIN_CHUNKS_FOR_MAX_DF = 5


def preprocess(text: str) -> str:
    """Нижний регистр и удаление служебных слов."""
    words = re.findall(r"\w+", text.lower())
    return " ".join(word for word in words if word not in STOP_WORDS)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float  # косинусная близость к вопросу, 0..1


class Retriever:
    """Индекс TF-IDF по фрагментам документов."""

    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=NGRAM_RANGE,
            preprocessor=preprocess,
            sublinear_tf=True,
            max_df=MAX_DF if len(chunks) >= MIN_CHUNKS_FOR_MAX_DF else 1.0,
        )
        self._matrix = self._vectorizer.fit_transform(chunk.text for chunk in chunks)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Возвращает до top_k фрагментов с ненулевой близостью, лучшие первыми."""
        query_vector = self._vectorizer.transform([query])
        # Векторы TF-IDF нормированы, поэтому скалярное произведение = косинус.
        scores = (self._matrix @ query_vector.T).toarray().ravel()
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            SearchResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in ranked[:top_k]
            if scores[i] > 0
        ]
