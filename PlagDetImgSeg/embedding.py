"""
DINOv2 Embedding Extraction

Loads the model once and generates embeddings
for image patches.
"""

import cv2
import torch
import numpy as np

from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from config import (
    MODEL_NAME,
    DEVICE,
    BATCH_SIZE
)

from utils import normalize_embedding


class EmbeddingExtractor:

    def __init__(self):

        print("Loading DINOv2...")

        self.device = torch.device(
            DEVICE if torch.cuda.is_available() else "cpu"
        )

        self.processor = AutoImageProcessor.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModel.from_pretrained(
            MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"Running on {self.device}")

    def _prepare_patch(self, patch):

        image = cv2.cvtColor(
            patch.image,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(image)

        return image

    @torch.no_grad()
    def extract(self, patches):

        images = [
            self._prepare_patch(p)
            for p in patches
        ]

        embeddings = []

        for start in range(
            0,
            len(images),
            BATCH_SIZE
        ):

            batch = images[start:start+BATCH_SIZE]

            inputs = self.processor(
                images=batch,
                return_tensors="pt"
            )

            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }

            outputs = self.model(**inputs)

            cls_tokens = outputs.last_hidden_state[:, 0]

            cls_tokens = cls_tokens.cpu().numpy()

            for vector in cls_tokens:

                embeddings.append(
                    normalize_embedding(vector)
                )

        return np.array(
            embeddings,
            dtype=np.float32
        )