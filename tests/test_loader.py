import pytest

from assistant.loader import DocsError, load_chunks, split_text


def test_short_text_is_single_chunk():
    assert split_text("Первый абзац.\n\nВторой абзац.", size=800) == ["Первый абзац.\n\nВторой абзац."]


def test_chunks_respect_size_and_paragraph_boundaries():
    paragraphs = [f"Абзац номер {i}. " + "слово " * 20 for i in range(10)]
    chunks = split_text("\n\n".join(paragraphs), size=300, overlap=50)

    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)
    # Абзацы не рвутся посередине: каждый абзац целиком лежит хотя бы в одном фрагменте.
    for paragraph in paragraphs:
        assert any(paragraph.strip() in chunk for chunk in chunks)


def test_consecutive_chunks_overlap():
    text = "\n\n".join(f"Абзац {i}. " + "текст " * 25 for i in range(6))
    chunks = split_text(text, size=300, overlap=100)

    for previous, following in zip(chunks, chunks[1:]):
        head = following.split("\n\n")[0]
        assert head and head in previous


def test_long_paragraph_is_split():
    chunks = split_text("слово " * 400, size=200, overlap=0)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_empty_text_gives_no_chunks():
    assert split_text("  \n\n  ") == []


def test_load_chunks_reads_supported_files_only(tmp_path):
    (tmp_path / "a.md").write_text("Текст А", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Текст Б", encoding="utf-8")
    (tmp_path / "c.pdf").write_text("игнорируется", encoding="utf-8")

    chunks = load_chunks(tmp_path)

    assert [(c.source, c.number) for c in chunks] == [("a.md", 1), ("b.txt", 1)]


def test_missing_folder_raises(tmp_path):
    with pytest.raises(DocsError, match="не найдена"):
        load_chunks(tmp_path / "нет_такой")


def test_folder_without_supported_files_raises(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    with pytest.raises(DocsError, match="нет файлов"):
        load_chunks(tmp_path)
