"""
Loads PDF pages directly into memory.

No images are saved to disk.
"""

import fitz  # PyMuPDF
import cv2
import numpy as np

from config import PDF_DPI


def render_page(page):
    """
    Convert a PyMuPDF page into an OpenCV image.
    """

    zoom = PDF_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    image = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(
        pix.height,
        pix.width,
        pix.n
    )

    # Convert RGB → BGR (OpenCV format)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def load_pdf(pdf_path):
    """
    Generator that yields one page image at a time.

    Example:
        for page_number, image in load_pdf("student1.pdf"):
            ...
    """

    document = fitz.open(pdf_path)

    try:

        for page_number in range(len(document)):

            page = document.load_page(page_number)

            image = render_page(page)

            yield page_number + 1, image

    finally:
        document.close()


def get_page_count(pdf_path):
    """
    Returns the number of pages in a PDF.
    """

    document = fitz.open(pdf_path)

    count = len(document)

    document.close()

    return count