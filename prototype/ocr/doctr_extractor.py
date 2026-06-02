import fitz

from doctr.io import DocumentFile
from doctr.models import ocr_predictor


print("Loading docTR OCR Model...")

model = ocr_predictor(
    pretrained=True
)

print("docTR Model Loaded")


def validate_pdf(pdf_path):
    """
    Validate PDF before OCR
    """

    try:

        doc = fitz.open(pdf_path)

        if doc.page_count == 0:
            raise Exception("PDF contains no pages")

        doc.close()

        return True

    except Exception as e:

        print("\n" + "=" * 60)
        print("INVALID PDF DETECTED")
        print(pdf_path)
        print("Reason:", str(e))
        print("=" * 60)

        return False


def extract_text(pdf_path):

    print("\n" + "=" * 60)
    print("OCR Processing:")
    print(pdf_path)
    print("=" * 60)

    if not validate_pdf(pdf_path):

        return ""

    try:

        doc = DocumentFile.from_pdf(pdf_path)

        result = model(doc)

        pages_text = []

        for page in result.pages:

            page_lines = []

            for block in page.blocks:

                for line in block.lines:

                    words = []

                    for word in line.words:
                        words.append(word.value)

                    page_lines.append(
                        " ".join(words)
                    )

            pages_text.append(
                "\n".join(page_lines)
            )

        extracted_text = "\n\n".join(
            pages_text
        )

        return extracted_text

    except Exception as e:

        print("\n" + "=" * 60)
        print("OCR FAILED")
        print(pdf_path)
        print("Reason:", str(e))
        print("=" * 60)

        return ""