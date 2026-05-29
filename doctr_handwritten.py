from doctr.io import DocumentFile
from doctr.models import ocr_predictor

pdf_path = r"C:/Users/pardh/Downloads/PDP/24-25 Assignment 1/Please upload your assignment file (in .pdf format) (File responses)/22bcs002_assignment2 - ABHIJAY IIIT Dharwad.pdf"

print("Loading docTR...")
model = ocr_predictor(pretrained=True)

print("Reading PDF...")
doc = DocumentFile.from_pdf(pdf_path)

print("Running OCR...")
result = model(doc)

data = result.export()

full_text = ""

for i, page in enumerate(data["pages"]):
    full_text += f"\n===== PAGE {i+1} =====\n"

    for block in page["blocks"]:
        for line in block["lines"]:
            line_text = " ".join(word["value"] for word in line["words"])
            full_text += line_text + "\n"

print(full_text)

with open("doctr_handwritten_output.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("\nSaved to doctr_handwritten_output.txt")