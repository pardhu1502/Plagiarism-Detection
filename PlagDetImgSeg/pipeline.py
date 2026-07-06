"""
Main plagiarism detection pipeline.
"""

from itertools import combinations

import numpy as np

from pdf_loader import load_pdf
from preprocess import preprocess
from patch_extractor import extract_patches

from embedding import EmbeddingExtractor
from similarity import SimilarityEngine
from heatmap import HeatmapGenerator

from models import (
    Document,
    Page,
    PageResult,
    DocumentResult,
    DetectionResult
)


class PlagiarismPipeline:

    def __init__(self):

        self.extractor = EmbeddingExtractor()

        self.heatmap = HeatmapGenerator()

    # --------------------------------------------------
    # Process one PDF
    # --------------------------------------------------

    def process_document(self, pdf_path):

        document = Document(name=pdf_path)

        total_patches = 0

        for page_number, image in load_pdf(pdf_path):

            image = preprocess(image)

            patches = extract_patches(
                image,
                page_number
            )

            embeddings = self.extractor.extract(
                patches
            )

            page = Page(
                number=page_number,
                image=image,
                patches=patches,
                embeddings=embeddings
            )

            document.pages.append(page)

            total_patches += len(patches)

        document.total_patches = total_patches

        return document

    # --------------------------------------------------
    # Compare two documents
    # --------------------------------------------------

    def compare_documents(
        self,
        document_a,
        document_b
    ):

        engine = SimilarityEngine()

        page_results = []

        total_matches = 0

        total_patches = 0

        for page_a in document_a.pages:

            best_page = None
            best_similarity = 0
            best_matches = []

            for page_b in document_b.pages:

                engine.build(
                    page_b.embeddings,
                    page_b.patches
                )

                matches = engine.find_matches(
                    page_a.embeddings,
                    page_a.patches
                )

                similarity = engine.document_similarity(
                    matches,
                    len(page_a.patches)
                )

                if similarity > best_similarity:

                    best_similarity = similarity
                    best_matches = matches
                    best_page = page_b

            heatmap = self.heatmap.create(
                page_a.image,
                best_matches
            )

            page_results.append(

                PageResult(
                    page_number=page_a.number,
                    similarity=best_similarity,
                    matches=best_matches,
                    heatmap=heatmap
                )

            )

            total_matches += len(best_matches)

            total_patches += len(page_a.patches)

        overall_similarity = (
            (total_matches / total_patches) * 100
            if total_patches > 0 else 0
        )

        return DocumentResult(

            source_document=document_a.name,

            target_document=document_b.name,

            similarity=overall_similarity,

            pages=page_results
        )

    # --------------------------------------------------
    # Run Entire System
    # --------------------------------------------------

    def run(self, pdf_files):

        documents = []

        for pdf in pdf_files:

            print(f"Processing {pdf}")

            document = self.process_document(pdf)

            documents.append(document)

        detection = DetectionResult()

        for doc_a, doc_b in combinations(
            documents,
            2
        ):

            print(
                f"Comparing {doc_a.name} ↔ {doc_b.name}"
            )

            result = self.compare_documents(
                doc_a,
                doc_b
            )

            detection.comparisons.append(result)

        detection.comparisons.sort(
            key=lambda x: x.similarity,
            reverse=True
        )

        return detection