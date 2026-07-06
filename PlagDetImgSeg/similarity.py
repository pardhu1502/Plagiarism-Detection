"""
similarity.py

Patch similarity computation using FAISS.
"""

from typing import List

import faiss
import numpy as np

from config import (
    TOP_K,
    SIMILARITY_THRESHOLD
)

from models import (
    Patch,
    PatchMatch
)


class SimilarityEngine:
    """
    Computes similarity between two pages using FAISS.
    """

    def __init__(self):

        self.index = None
        self.metadata = []

    # --------------------------------------------------
    # Build FAISS Index
    # --------------------------------------------------

    def build(
        self,
        embeddings: np.ndarray,
        metadata: List[Patch]
    ):
        """
        Build a fresh FAISS index.

        Parameters
        ----------
        embeddings : ndarray (N,D)

        metadata : List[Patch]
        """

        if len(embeddings) == 0:
            self.index = None
            self.metadata = []
            return

        embeddings = embeddings.astype(np.float32)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.metadata = metadata

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(
        self,
        query_embeddings: np.ndarray
    ):

        if self.index is None:

            return (
                np.empty((0, TOP_K)),
                np.empty((0, TOP_K), dtype=int)
            )

        scores, indices = self.index.search(

            query_embeddings.astype(np.float32),

            TOP_K

        )

        return scores, indices

    # --------------------------------------------------
    # Find Matches
    # --------------------------------------------------

    def find_matches(
        self,
        query_embeddings: np.ndarray,
        query_metadata: List[Patch]
    ) -> List[PatchMatch]:

        scores, indices = self.search(query_embeddings)

        matches = []

        for query_index in range(len(query_metadata)):

            for rank in range(TOP_K):

                similarity = float(
                    scores[query_index][rank]
                )

                if similarity < SIMILARITY_THRESHOLD:
                    continue

                database_index = indices[query_index][rank]

                if database_index < 0:
                    continue

                matches.append(

                    PatchMatch(

                        source_patch=query_metadata[
                            query_index
                        ],

                        target_patch=self.metadata[
                            database_index
                        ],

                        similarity=similarity

                    )

                )

        return matches

    # --------------------------------------------------
    # Page Similarity
    # --------------------------------------------------

    def page_similarity(
        self,
        matches: List[PatchMatch],
        total_patches: int
    ) -> float:
        """
        Percentage of unique matched patches.
        """

        if total_patches == 0:
            return 0.0

        matched = {

            m.source_patch.id

            for m in matches

        }

        return (

            len(matched)

            / total_patches

        ) * 100

    # --------------------------------------------------
    # Best Match
    # --------------------------------------------------

    def best_match(
        self,
        matches: List[PatchMatch]
    ):

        if not matches:
            return None

        return max(

            matches,

            key=lambda m: m.similarity

        )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    def statistics(
        self,
        matches: List[PatchMatch]
    ):

        if not matches:

            return {

                "count": 0,

                "average": 0,

                "maximum": 0

            }

        similarities = [

            m.similarity

            for m in matches

        ]

        return {

            "count": len(matches),

            "average": float(

                np.mean(similarities)

            ),

            "maximum": float(

                np.max(similarities)

            )

        }