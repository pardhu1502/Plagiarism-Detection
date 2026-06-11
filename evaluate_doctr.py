import time
import re

from datasets import load_dataset
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

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

print("Loading docTR...")

model = ocr_predictor(
    pretrained=True
)

predictions = []
references = []

start = time.time()

for idx, sample in enumerate(subset):

    image = sample["image"]

    image.save("temp.png")

    doc = DocumentFile.from_images(
        "temp.png"
    )

    result = model(doc)

    extracted = ""

    for page in result.pages:

        for block in page.blocks:

            for line in block.lines:

                extracted += (
                    " ".join(
                        word.value
                        for word in line.words
                    )
                    + " "
                )

    predictions.append(
        clean_text(extracted)
    )

    references.append(
        clean_text(sample["text"])
    )

    print(
        f"Processed {idx + 1}/{NUM_SAMPLES}"
    )

end = time.time()

print("\n docTR RESULTS ")

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