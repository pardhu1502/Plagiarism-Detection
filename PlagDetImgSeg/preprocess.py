"""
Image preprocessing for plagiarism detection.

The goal is to improve image quality while preserving
handwriting, diagrams, and page layout.
"""

import cv2

from config import (
    MAX_PAGE_SIZE,
    USE_GRAYSCALE,
    HIST_EQUALIZATION,
    BLUR_KERNEL
)

from utils import resize_keep_aspect


def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess(image):
    """
    Complete preprocessing pipeline.
    """

    image = resize_keep_aspect(
        image,
        MAX_PAGE_SIZE
    )

    if USE_GRAYSCALE:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    image = cv2.GaussianBlur(
        image,
        BLUR_KERNEL,
        0
    )

    if HIST_EQUALIZATION:
        image = apply_clahe(image)

    return image