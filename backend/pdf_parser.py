"""
PDF parser utility for extracting clean textual passages from uploaded PDF files.
Uses pypdf to parse pages and chunk text into meaningful paragraph blocks.
"""

import io
import re
import logging
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_passages_from_pdf(file_bytes: bytes) -> tuple[list[str], int]:
    """
    Reads PDF file bytes and extracts text blocks/paragraphs page by page.

    Returns:
        tuple[list[str], int]: (list of extracted passage strings, total page count)
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    total_pages = len(reader.pages)
    extracted_passages: list[str] = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue

        # Split page text by double newlines or paragraph breaks
        raw_paragraphs = re.split(r'\n\s*\n', page_text)

        for paragraph in raw_paragraphs:
            # Clean up internal whitespace and newlines
            clean_p = " ".join(paragraph.split()).strip()

            # Filter out tiny strings like page numbers or header fragments (< 15 chars)
            if len(clean_p) >= 15:
                extracted_passages.append(clean_p)

    logger.info(f"Extracted {len(extracted_passages)} passages from {total_pages} PDF pages.")
    return extracted_passages, total_pages
