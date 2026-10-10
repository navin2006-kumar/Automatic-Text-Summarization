from io import BytesIO

import pytest
from docx import Document
from pypdf import PdfWriter

from preprocessing import DocumentProcessingError, extract_text, validate_text, word_count


def test_validate_and_word_count():
    assert validate_text("  two words here ") == "two words here"
    assert word_count("two words\nhere") == 3
    with pytest.raises(DocumentProcessingError):
        validate_text(" \n ")


def test_extract_txt():
    assert extract_text("article.TXT", b"\xef\xbb\xbfHello news") == "Hello news"


def test_reject_unsupported_and_empty_files():
    with pytest.raises(DocumentProcessingError, match="Unsupported"):
        extract_text("article.rtf", b"some text")
    with pytest.raises(DocumentProcessingError, match="empty"):
        extract_text("article.txt", b"")


def test_extract_docx():
    document = Document()
    document.add_paragraph("Research result")
    stream = BytesIO()
    document.save(stream)
    assert extract_text("paper.docx", stream.getvalue()) == "Research result"


def test_extract_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    stream = BytesIO()
    writer.write(stream)
    with pytest.raises(DocumentProcessingError, match="containing text"):
        extract_text("paper.pdf", stream.getvalue())
