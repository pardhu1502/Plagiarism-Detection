"""
Heatmap generation for plagiarism visualization.
"""

import cv2
import numpy as np

from config import (
    ALPHA,
    HEATMAP_THRESHOLD
)


class HeatmapGenerator:

    def __init__(self):
        pass

    def create(self, image, matches):
        """
        Generate heatmap overlay.

        Parameters
        ----------
        image : np.ndarray
            Original page image

        matches : list
            Matches returned by SimilarityEngine

        Returns
        -------
        overlay_image
        """

        heatmap = np.zeros(
            image.shape[:2],
            dtype=np.float32
        )

        # -----------------------------------
        # Paint suspicious regions
        # -----------------------------------

        for match in matches:

            similarity = match["similarity"]

            if similarity < HEATMAP_THRESHOLD:
                continue

            patch = match["query_patch"]

            x = patch.x
            y = patch.y

            w = patch.width
            h = patch.height

            heatmap[
                y:y+h,
                x:x+w
            ] += similarity

        # Normalize

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        heatmap_uint8 = np.uint8(
            heatmap * 255
        )

        colored = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            image,
            1 - ALPHA,
            colored,
            ALPHA,
            0
        )

        return overlay