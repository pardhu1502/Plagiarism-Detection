"""
Common utility functions used across the project.
"""

import time
import numpy as np
import torch


def get_device():
    """
    Returns the available device.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    vec1 = np.asarray(vec1, dtype=np.float32)
    vec2 = np.asarray(vec2, dtype=np.float32)

    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if denom == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / denom)


def normalize_embedding(embedding):
    """
    L2 normalize an embedding.
    """
    embedding = np.asarray(embedding, dtype=np.float32)

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


def resize_keep_aspect(image, max_size):
    """
    Resize image while maintaining aspect ratio.
    """
    import cv2

    h, w = image.shape[:2]

    longest = max(h, w)

    if longest <= max_size:
        return image

    scale = max_size / longest

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


def timer(func):
    """
    Measure execution time of a function.
    """

    def wrapper(*args, **kwargs):
        start = time.time()

        result = func(*args, **kwargs)

        end = time.time()

        print(f"{func.__name__} completed in {end-start:.2f}s")

        return result

    return wrapper


def chunk_list(items, batch_size):
    """
    Yield batches from a list.
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def clamp(value, minimum, maximum):
    """
    Restrict value within a range.
    """
    return max(minimum, min(value, maximum))


def create_empty_heatmap(height, width):
    """
    Create an empty float heatmap.
    """
    return np.zeros((height, width), dtype=np.float32)