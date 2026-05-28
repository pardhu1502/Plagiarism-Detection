
import os
import time
import fitz
import torch
import easyocr
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)


PDF_PATH = r"C:/Users/pardh/Downloads/PDP/24-25 Assignment 1/Please upload your assignment file (in .pdf format) (File responses)/22BCS001 - ABHIGYAN NIRANJAN IIIT Dharwad.pdf"
OUTPUT_FILE = "hybrid_output.txt"

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

DPI = 200


print("=" * 60)

if torch.cuda.is_available():
    DEVICE = "cuda"
    print("GPU DETECTED")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    print("GPU NOT FOUND")
    print("Using CPU")

print("=" * 60)


print("Loading EasyOCR...")
easyocr_reader = easyocr.Reader(
    ['en'],
    gpu=torch.cuda.is_available()
)

print("Loading Qwen2.5-VL model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

print("Models Loaded Successfully")
print("=" * 60)


def render_page(page, dpi=200):
    """
    Convert PDF page to image
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)

    img = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    return img


def is_typed_page(page):
    """
    Detect if PDF page contains embedded selectable text
    """
    text = page.get_text().strip()

    return len(text) > 100


def extract_pdf_text(page):
    """
    Direct extraction from PDF
    """
    return page.get_text()


def easyocr_extract(image):
    """
    Fast OCR using EasyOCR
    """
    image_np = np.array(image)

    results = easyocr_reader.readtext(
        image_np,
        detail=0,
        paragraph=True
    )

    return "\n".join(results)


def qwen_extract(image):
    """
    Qwen extraction for difficult handwritten pages
    """
    prompt = """
You are an OCR system.

Extract ALL visible text carefully.

Rules:
1. Preserve equations and symbols
2. Preserve formatting
3. Do not summarize
4. Extract handwritten text accurately
5. Extract diagrams and resolution graphs if present
6. Output plain text only
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


def needs_qwen(text):
    """
    Detect poor OCR quality
    """

    if len(text.strip()) < 40:
        return True

    bad_patterns = [
        "(123,",
        "),( ",
        "),(1",
        "],[",
        "???"
    ]

    for pattern in bad_patterns:
        if pattern in text:
            return True

    words = text.split()

    if len(words) == 0:
        return True

    weird = 0

    for w in words:
        if len(w) > 20:
            weird += 1

    if weird > 10:
        return True

    return False


def print_gpu_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)

        print(f"GPU Allocated Memory : {allocated:.2f} GB")
        print(f"GPU Reserved Memory  : {reserved:.2f} GB")



start_total = time.time()

doc = fitz.open(PDF_PATH)

all_text = []

for page_num in range(len(doc)):

    page_start = time.time()

    print("\n" + "=" * 60)
    print(f"Processing Page {page_num + 1}")
    print("=" * 60)

    page = doc[page_num]


    if is_typed_page(page):

        print("Typed page detected")
        print("Using direct PDF extraction...")

        extracted_text = extract_pdf_text(page)

    else:


        print("Scanned/Handwritten page detected")

        image = render_page(page, dpi=DPI)

        print("Running EasyOCR")

        extracted_text = easyocr_extract(image)

        if needs_qwen(extracted_text):

            print("EasyOCR quality poor")
            print("Switching to Qwen2.5-VL")

            extracted_text = qwen_extract(image)

        else:
            print("EasyOCR extraction accepted")


    page_text = f"\n===== PAGE {page_num + 1} =====\n"
    page_text += extracted_text + "\n"

    all_text.append(page_text)

    print("\nExtracted Text:\n")
    print(extracted_text[:3000])


    page_end = time.time()

    print("\n" + "-" * 60)
    print(
        f"Page {page_num + 1} completed in "
        f"{page_end - page_start:.2f} seconds"
    )
    print("-" * 60)

    print_gpu_stats()


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(all_text))

end_total = time.time()

print("\n" + "=" * 60)
print("OCR COMPLETED")
print("=" * 60)

print(f"\nSaved extracted text to: {OUTPUT_FILE}")

print("\n" + "=" * 60)
print(
    f"TOTAL EXECUTION TIME: "
    f"{end_total - start_total:.2f} seconds"
)
print("=" * 60)