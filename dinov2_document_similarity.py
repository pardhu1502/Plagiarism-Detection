import fitz
import torch
import numpy as np
import time

from PIL import Image
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity



PDF_1 = r"C:/Users/pardh/Downloads/PDP/24-25 Assignment 1/Please upload your assignment file (in .pdf format) (File responses)/22bcs002_assignment2 - ABHIJAY IIIT Dharwad.pdf"

PDF_2 = r"C:/Users/pardh/Downloads/PDP/24-25 Assignment 1/Please upload your assignment file (in .pdf format) (File responses)/22BCS001 - ABHIGYAN NIRANJAN IIIT Dharwad.pdf"

DPI = 150


print("=" * 60)

if torch.cuda.is_available():
    device = 0
    print("GPU DETECTED")
    print(torch.cuda.get_device_name(0))
else:
    device = -1
    print("CPU MODE")

print("=" * 60)



print("Loading DINOv2 Base...")

pipe = pipeline(
    task="image-feature-extraction",
    model="facebook/dinov2-base",
    device=device
)

print("Model Loaded")



def pdf_to_images(pdf_path):

    doc = fitz.open(pdf_path)

    images = []

    for page in doc:

        mat = fitz.Matrix(
            DPI / 72,
            DPI / 72
        )

        pix = page.get_pixmap(matrix=mat)

        img = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        images.append(img)

    doc.close()

    return images



def image_embedding(image):

    features = pipe(image)

    features = np.array(features)

    print("Raw Feature Shape:", features.shape)

    #
    # Expected:
    # (1, 257, 768)
    #
    # CLS token = features[0,0]
    #

    if len(features.shape) == 3:

        embedding = features[0, 0]

    elif len(features.shape) == 2:

        embedding = features[0]

    else:

        embedding = features.flatten()

    embedding = embedding.astype(np.float32)

    return embedding



def document_embedding(pdf_path):

    print("\n" + "=" * 60)
    print(f"Processing: {pdf_path}")
    print("=" * 60)

    pages = pdf_to_images(pdf_path)

    print(f"Pages: {len(pages)}")

    page_embeddings = []

    for idx, page_image in enumerate(pages):

        print(
            f"Embedding page "
            f"{idx + 1}/{len(pages)}"
        )

        emb = image_embedding(page_image)

        page_embeddings.append(emb)

    page_embeddings = np.array(
        page_embeddings,
        dtype=np.float32
    )

    print(
        "Page Embeddings Shape:",
        page_embeddings.shape
    )


    doc_embedding = np.mean(
        page_embeddings,
        axis=0
    )

    print(
        "Document Embedding Shape:",
        doc_embedding.shape
    )

    return doc_embedding



start = time.time()

doc1_emb = document_embedding(PDF_1)
doc2_emb = document_embedding(PDF_2)

score = cosine_similarity(
    doc1_emb.reshape(1, -1),
    doc2_emb.reshape(1, -1)
)[0][0]

end = time.time()

print("\n" + "=" * 60)

print(
    f"Document Similarity: "
    f"{score:.4f}"
)

print(
    f"Percentage: "
    f"{score * 100:.2f}%"
)

print(
    f"Execution Time: "
    f"{end - start:.2f} seconds"
)

print("=" * 60)