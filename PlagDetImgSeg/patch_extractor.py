"""
Sliding window patch extraction.

Each patch stores:
- image
- page number
- location
- unique id
"""

from models import Patch

import numpy as np

from config import PATCH_SIZE, STRIDE


#@dataclass
class Patch:
    id: int
    page: int

    x: int
    y: int

    width: int
    height: int

    image: np.ndarray


import cv2
import numpy as np


def is_blank_patch(
    patch,
    white_threshold=245,
    blank_ratio=0.90
):
    """
    Returns True if the patch is mostly white.
    """

    gray = cv2.cvtColor(
        patch,
        cv2.COLOR_BGR2GRAY
    )

    white_pixels = np.sum(gray >= white_threshold)

    ratio = white_pixels / gray.size

    return ratio >= blank_ratio


def has_enough_information(
    patch,
    edge_threshold=150
):
    """
    Reject patches with almost no edges.
    """

    gray = cv2.cvtColor(
        patch,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        75,
        150
    )

    edge_pixels = np.count_nonzero(edges)

    return edge_pixels >= edge_threshold

def extract_patches(
    image: np.ndarray,
    page_number: int,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE
):
    """
    Extract overlapping patches from a page.

    Returns
    -------
    List[Patch]
    """

    patches = []

    height, width = image.shape[:2]

    patch_id = 0

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):

            patch = image[
                y:y + patch_size,
                x:x + patch_size
            ]

            # Skip mostly blank patches
            if is_blank_patch(patch):
                continue

            # Skip low-information patches
            if not has_enough_information(patch):
                continue

            patches.append(
                Patch(
                    id=patch_id,
                    page=page_number,
                    x=x,
                    y=y,
                    width=patch_size,
                    height=patch_size,
                    image=patch
                )
            )

            patch_id += 1

    return patches