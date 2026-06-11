import time
import re

from datasets import load_dataset

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

from jiwer import cer, wer

NUM_SAMPLES = 50


def clean_text(text):
    text = re.sub(r"%[A-Za-z]+%", " ", text)
    text = text.lower()
    text = " ".join(text.split())
    return text


print("Loading Dataset...")

ds = load_dataset(
    "bhavya777/GNHK-dataset"
)

subset = ds["train"].select(
    range(NUM_SAMPLES)
)

print("Loading TrOCR...")

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

predictions = []
references = []

start = time.time()

for idx, sample in enumerate(subset):

    image = sample["image"]

    pixel_values = processor(
        image,
        return_tensors="pt"
    ).pixel_values

    generated_ids = model.generate(
        pixel_values
    )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    predictions.append(
        clean_text(text)
    )

    references.append(
        clean_text(sample["text"])
    )

    print(
        f"Processed {idx + 1}/{NUM_SAMPLES}"
    )

end = time.time()

print("\n TrOCR RESULTS ")

print(
    "CER:",
    cer(references, predictions)
)

print(
    "WER:",
    wer(references, predictions)
)

print(
    "Average Time/Image:",
    (end - start) / NUM_SAMPLES
)