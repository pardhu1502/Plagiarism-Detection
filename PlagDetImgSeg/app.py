"""
Main application.
"""

from pdf_loader import load_pdf
from preprocess import preprocess
from patch_extractor import extract_patches

from embedding import EmbeddingExtractor
from similarity import SimilarityEngine
from heatmap import HeatmapGenerator
from report import ReportGenerator


def process_document(pdf_path, extractor):

    all_patches = []
    all_embeddings = []

    for page_no, image in load_pdf(pdf_path):

        image = preprocess(image)

        patches = extract_patches(
            image,
            page_no
        )

        embeddings = extractor.extract(
            patches
        )

        all_patches.extend(patches)
        all_embeddings.extend(embeddings)

    return all_patches, all_embeddings


def main():

    document_a = "student1.pdf"
    document_b = "student2.pdf"

    extractor = EmbeddingExtractor()

    patches_a, embeddings_a = process_document(
        document_a,
        extractor
    )

    patches_b, embeddings_b = process_document(
        document_b,
        extractor
    )

    similarity = SimilarityEngine()

    similarity.build(
        embeddings_b,
        patches_b
    )

    matches = similarity.find_matches(
        embeddings_a,
        patches_a
    )

    score = similarity.document_similarity(
        matches,
        len(patches_a)
    )

    print(f"Similarity : {score:.2f}%")

    # Heatmaps

    generator = HeatmapGenerator()

    heatmaps = []

    report = ReportGenerator()

    report.generate(
        document_a,
        document_b,
        score,
        heatmaps
    )


if __name__ == "__main__":
    main()