"""Чтение документов из папки и нарезка на фрагменты."""

import re
from dataclasses import dataclass
from pathlib import Path

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SUPPORTED_SUFFIXES = {".md", ".txt"}


class DocsError(Exception):
    """Папка с документами недоступна или в ней нечего индексировать."""


@dataclass(frozen=True)
class Chunk:
    source: str  # имя файла
    number: int  # номер фрагмента в файле, с 1
    text: str


def _split_long_paragraph(paragraph: str, size: int) -> list[str]:
    """Режет слишком длинный абзац на части не длиннее size, по пробелам."""
    parts = []
    start = 0
    while start < len(paragraph):
        end = min(start + size, len(paragraph))
        if end < len(paragraph):
            cut = paragraph.rfind(" ", start, end)
            if cut > start:
                end = cut
        part = paragraph[start:end].strip()
        if part:
            parts.append(part)
        start = end
    return parts


def _overlap_tail(text: str, overlap: int) -> str:
    """Хвост предыдущего фрагмента для перекрытия, начиная с границы слова."""
    if overlap <= 0:
        return ""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    space = tail.find(" ")
    if space != -1:
        tail = tail[space + 1:]
    return tail.strip()


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Нарезает текст на фрагменты ~size символов по границам абзацев.

    Следующий фрагмент начинается с хвоста предыдущего (overlap символов),
    чтобы смысл на стыке не терялся.
    """
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if paragraph:
            pieces.extend(_split_long_paragraph(paragraph, size))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= size or not current:
            current = candidate
            continue
        chunks.append(current)
        tail = _overlap_tail(current, overlap)
        # Если с перекрытием фрагмент не помещается в лимит, начинаем без него.
        if tail and len(tail) + 2 + len(piece) <= size:
            current = f"{tail}\n\n{piece}"
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def load_chunks(docs_dir: Path) -> list[Chunk]:
    """Читает все .md и .txt из папки и возвращает их фрагменты."""
    if not docs_dir.is_dir():
        raise DocsError(f"Папка с документами не найдена: {docs_dir}")

    files = sorted(
        path for path in docs_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise DocsError(f"В папке {docs_dir} нет файлов .md или .txt")

    chunks: list[Chunk] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise DocsError(f"Не удалось прочитать файл {path.name}: {exc}") from exc
        for number, chunk_text in enumerate(split_text(text), start=1):
            chunks.append(Chunk(source=path.name, number=number, text=chunk_text))

    if not chunks:
        raise DocsError(f"Файлы в папке {docs_dir} пусты")
    return chunks
