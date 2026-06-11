import os
import csv
import time
import gc
import torch

from PIL import Image
from datasets import load_dataset

from jiwer import cer
from jiwer import wer

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

SAVE_CSV = "qwen_gnhk_results.csv"

MAX_SAMPLES = 10

# Image resize limit
MAX_IMAGE_SIZE = 1024

# OCR generation length
MAX_NEW_TOKENS = 64


print("=" * 60)

if torch.cuda.is_available():
    print("GPU DETECTED")
    print(torch.cuda.get_device_name(0))
else:
    print("CPU MODE")

print("=" * 60)


print("Loading Qwen2.5-VL...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    max_pixels=512 * 28 * 28
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
    if torch.cuda.is_available()
    else torch.float32,
    device_map="auto"
)

model.eval()

print("Model Loaded")


print("Loading Dataset...")

dataset = load_dataset(
    "bhavya777/GNHK-dataset"
)

samples = dataset["train"]

print(
    "Total samples:",
    len(samples)
)


def qwen_ocr(image):

    image = image.copy()

    image.thumbnail(
        (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE),
        Image.Resampling.LANCZOS
    )

    prompt = """
Extract all visible text exactly.

Rules:
- OCR only
- No explanation
- No summary
- Preserve text order
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
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids
        in zip(
            inputs.input_ids,
            generated_ids
        )
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    del generated_ids
    del generated_ids_trimmed
    del inputs

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output.strip()


all_gt = []
all_pred = []

if not os.path.exists(SAVE_CSV):

    with open(
        SAVE_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "sample",
            "cer",
            "wer",
            "ground_truth",
            "prediction"
        ])

start_total = time.time()

for idx in range(
    min(
        MAX_SAMPLES,
        len(samples)
    )
):

    sample = samples[idx]

    image = sample["image"]

    gt = sample["text"]

    print("\n" + "=" * 60)
    print(
        f"Processing {idx + 1}/{min(MAX_SAMPLES, len(samples))}"
    )
    print("=" * 60)

    start = time.time()

    try:

        pred = qwen_ocr(image)

        sample_cer = cer(
            gt,
            pred
        )

        sample_wer = wer(
            gt,
            pred
        )

        all_gt.append(gt)
        all_pred.append(pred)

        with open(
            SAVE_CSV,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                idx,
                sample_cer,
                sample_wer,
                gt,
                pred
            ])

        elapsed = time.time() - start

        print(
            f"CER: {sample_cer:.4f}"
        )

        print(
            f"WER: {sample_wer:.4f}"
        )

        print(
            f"Time: {elapsed:.2f}s"
        )

        if len(all_gt) > 0:

            overall_cer = cer(
                " ".join(all_gt),
                " ".join(all_pred)
            )

            overall_wer = wer(
                " ".join(all_gt),
                " ".join(all_pred)
            )

            print(
                f"Running CER: {overall_cer:.4f}"
            )

            print(
                f"Running WER: {overall_wer:.4f}"
            )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:

        print(
            "FAILED:",
            idx,
            str(e)
        )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


end_total = time.time()

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

if len(all_gt) > 0:

    final_cer = cer(
        " ".join(all_gt),
        " ".join(all_pred)
    )

    final_wer = wer(
        " ".join(all_gt),
        " ".join(all_pred)
    )

    print(
        f"CER: {final_cer:.4f}"
    )

    print(
        f"WER: {final_wer:.4f}"
    )

    print(
        f"CER %: {final_cer * 100:.2f}%"
    )

    print(
        f"WER %: {final_wer * 100:.2f}%"
    )

else:

    print(
        "No successful predictions."
    )

print(
    f"Total Time: "
    f"{end_total - start_total:.2f}s"
)