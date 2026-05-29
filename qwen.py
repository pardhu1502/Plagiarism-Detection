import os
import torch
import fitz

from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration
)

from qwen_vl_utils import process_vision_info

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

PDF_PATH = r"C:/Users/pardh/Downloads/PDP/24-25 Assignment 1/Please upload your assignment file (in .pdf format) (File responses)/22bcs057-3 - KRISHNA JOSHI IIIT Dharwad.pdf"

OUTPUT_FILE = "qwen_output.txt"

IMAGE_DIR = "temp_pages"

os.makedirs(IMAGE_DIR, exist_ok=True)


print("=" * 60)

if torch.cuda.is_available():
    device = "cuda"
    print("GPU DETECTED")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("GPU NOT FOUND")
    print("Using CPU")

print("=" * 60)


print("Loading model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

print("Model loaded successfully!")


print("Reading PDF...")

doc = fitz.open(PDF_PATH)

all_text = ""



for page_num in range(len(doc)):

    print("\n" + "=" * 60)
    print(f"Processing page {page_num + 1}")
    print("=" * 60)

    

    page = doc.load_page(page_num)

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

    img = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    image_path = os.path.join(
        IMAGE_DIR,
        f"page_{page_num + 1}.png"
    )

    img.save(image_path)


    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": """
Extract ALL text from this assignment page accurately.

Important instructions:
1. Preserve handwritten text.
2. Preserve typed text.
3. Preserve equations and symbols.
4. Preserve line breaks.
5. Preserve formatting as much as possible.
6. Do NOT summarize.
7. Output only extracted text.
8. Keep question numbers and steps aligned properly.
"""
                }
            ]
        }
    ]


    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = inputs.to(device)


    print("Running OCR...")

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False
        )


    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids
        )
    ]


    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]


    print("\nExtracted Text:\n")
    print(output_text[:1000])

    all_text += f"\n===== PAGE {page_num + 1} =====\n\n"
    all_text += output_text
    all_text += "\n"


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(all_text)

print("\n" + "=" * 60)
print("OCR COMPLETED")
print("=" * 60)

print(f"\nSaved extracted text to: {OUTPUT_FILE}")