import time
import fitz
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)

PDF_PATH = r"C:/Users/pardh/Downloads/PDP/24-25 Assignment 1/Please upload your assignment file (in .pdf format) (File responses)/22bcs002_assignment2 - ABHIJAY IIIT Dharwad.pdf"

OUTPUT_FILE = "qwen_only_output.txt"

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

DPI = 220


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


print("Loading Qwen2.5-VL model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

model.eval()

print("Model Loaded Successfully")

print("=" * 60)


def render_page(page, dpi=220):

    mat = fitz.Matrix(dpi / 72, dpi / 72)

    pix = page.get_pixmap(
        matrix=mat,
        alpha=False
    )

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    return image



def qwen_extract(image):

    prompt = """
You are an advanced OCR system.

Extract ALL text from the image accurately.

Important instructions:
1. Extract handwritten text carefully
2. Extract typed text carefully
3. Preserve equations and symbols
4. Preserve formatting and line breaks
5. Extract resolution graphs and diagrams
6. Do NOT summarize
7. Do NOT explain
8. Return ONLY extracted text
9. If some text is unclear, still try your best
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
            max_new_tokens=4096,
            do_sample=False,
            temperature=0.1,
            repetition_penalty=1.05
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs.input_ids,
            generated_ids
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text



def print_gpu_stats():

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated() / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved() / (1024 ** 3)
        )

        print(f"GPU Allocated Memory : {allocated:.2f} GB")
        print(f"GPU Reserved Memory  : {reserved:.2f} GB")



start_total = time.time()

print("Opening PDF...")

doc = fitz.open(PDF_PATH)

print(f"Total Pages: {len(doc)}")

all_text = []


for page_num in range(len(doc)):

    page_start = time.time()

    print("\n" + "=" * 60)
    print(f"Processing Page {page_num + 1}")
    print("=" * 60)


    page = doc[page_num]

    print("Rendering page image...")

    image = render_page(page, dpi=DPI)


    print("Running Qwen OCR...")

    extracted_text = qwen_extract(image)


    page_text = (
        f"\n===== PAGE {page_num + 1} =====\n"
    )

    page_text += extracted_text + "\n"

    all_text.append(page_text)


    print("\nExtracted Text:\n")

    print(extracted_text[:5000])

    page_end = time.time()

    print("\n" + "-" * 60)

    print(
        f"Page {page_num + 1} completed in "
        f"{page_end - page_start:.2f} seconds"
    )

    print("-" * 60)

    print_gpu_stats()


    if torch.cuda.is_available():
        torch.cuda.empty_cache()


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

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