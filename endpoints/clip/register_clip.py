# Databricks notebook source
# MAGIC %md
# MAGIC # Register CLIP to Unity Catalog
# MAGIC
# MAGIC CLIP (Contrastive Language-Image Pre-training) is OpenAI's model that embeds text
# MAGIC AND images into the same 768-dimensional vector space. That shared space lets you
# MAGIC compute cosine similarity between text and images: a photo of a tractor will be
# MAGIC closer to the text "a green tractor" than to "an industrial pipeline".
# MAGIC
# MAGIC This notebook wraps the HuggingFace `CLIPModel` + `CLIPProcessor` classes in a
# MAGIC custom MLflow PyFunc with a single unified input shape (mix text and images in any
# MAGIC order), and registers it to Unity Catalog so Databricks Model Serving can serve it
# MAGIC on a GPU endpoint.
# MAGIC
# MAGIC ### Prerequisites
# MAGIC
# MAGIC 1. **UC catalog + schema**. The constants `UC_CATALOG` and `UC_SCHEMA` below should
# MAGIC    point at a catalog/schema you can create tables in. CLIP is not gated, so no HF
# MAGIC    token is required.
# MAGIC
# MAGIC ### Input contract (what the deployed endpoint accepts)
# MAGIC ```
# MAGIC {"dataframe_records": [
# MAGIC     {"type": "text",  "value": "a yellow circle"},
# MAGIC     {"type": "image", "value": "<base64 png>"}
# MAGIC ]}
# MAGIC ```
# MAGIC Each row is one input. `type` is either `"text"` or `"image"`. The wrapper
# MAGIC batches text and image inputs separately under the hood and re-assembles
# MAGIC results in the original order.
# MAGIC
# MAGIC ### Output contract
# MAGIC ```
# MAGIC {"predictions": {
# MAGIC     "embeddings": [[...768 floats...], [...768 floats...]],
# MAGIC     "dim": 768
# MAGIC }}
# MAGIC ```
# MAGIC All vectors are L2-normalized, so the raw dot product equals cosine similarity.

# COMMAND ----------

# Pin transformers to a version that works with CLIPModel/CLIPProcessor and is
# stable in Databricks Model Serving. 4.46.3 is conservative and well-tested.
# MAGIC %pip install -q -U mlflow transformers==4.46.3 pillow torch
# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import base64
import io
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.models.signature import infer_signature
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# --------------------------------------------------------------------------------------
# Configuration — change these to match your workspace.
# --------------------------------------------------------------------------------------
HF_MODEL = "openai/clip-vit-large-patch14"   # Source weights on HuggingFace (not gated).
UC_CATALOG = "<YOUR_CATALOG>"                 # Unity Catalog catalog name.
UC_SCHEMA = "model_workbench"                 # Unity Catalog schema name.
UC_MODEL = "clip_vit_large_patch14"           # Unity Catalog registered model name.
# --------------------------------------------------------------------------------------

UC_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL}"

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")  # noqa: F821

# COMMAND ----------

class CLIPEmbeddingModel(mlflow.pyfunc.PythonModel):
    """Custom PyFunc wrapping CLIP for unified text+image embedding inference."""

    def load_context(self, context: Any) -> None:
        self.processor = CLIPProcessor.from_pretrained(HF_MODEL)
        self.model = CLIPModel.from_pretrained(HF_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()

    def _decode_image(self, value: str) -> Image.Image:
        # Accept either raw base64 or a "data:image/...;base64,..." data URL.
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> list[list[float]]:
        # MLflow serving normalizes input to a DataFrame with the columns from the signature.
        # We return one embedding per input row so the response is `{"predictions": [[...], [...]]}`
        # — the MLflow-idiomatic shape that ai_query and other batch tooling expect.
        records = model_input.to_dict(orient="records") if isinstance(model_input, pd.DataFrame) else list(model_input)

        # Split inputs by modality, preserving original order so we can interleave outputs.
        texts: list[tuple[int, str]] = []
        images: list[tuple[int, Image.Image]] = []
        for idx, r in enumerate(records):
            t = str(r.get("type", "")).lower()
            v = r.get("value", "")
            if t == "text":
                texts.append((idx, str(v)))
            elif t == "image":
                images.append((idx, self._decode_image(str(v))))
            else:
                raise ValueError(f"Unsupported input type at index {idx}: {t!r} (expected 'text' or 'image')")

        # Pre-allocate the output array sized to the input.
        embeddings: list[list[float]] = [[] for _ in records]

        with torch.inference_mode():
            if texts:
                idxs, values = zip(*texts)
                inputs = self.processor(text=list(values), return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = self.model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats = feats.cpu().numpy()
                for j, idx in enumerate(idxs):
                    embeddings[idx] = feats[j].tolist()

            if images:
                idxs, values = zip(*images)
                inputs = self.processor(images=list(values), return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = self.model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats = feats.cpu().numpy()
                for j, idx in enumerate(idxs):
                    embeddings[idx] = feats[j].tolist()

        return embeddings

# COMMAND ----------

# Build a 1x1 PNG as a base64 example so the signature includes both modalities.
_tiny_png_buf = io.BytesIO()
Image.new("RGB", (4, 4), color=(255, 0, 0)).save(_tiny_png_buf, format="PNG")
_tiny_png_b64 = base64.b64encode(_tiny_png_buf.getvalue()).decode("ascii")

input_example = pd.DataFrame(
    [
        {"type": "text", "value": "a small red square"},
        {"type": "image", "value": _tiny_png_b64},
    ]
)
output_example = [[0.0] * 768, [0.0] * 768]  # one 768-d vector per input row
signature = infer_signature(input_example, output_example)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="clip-vit-large-patch14-register"):
    mlflow.pyfunc.log_model(
        artifact_path="clip-pyfunc",
        python_model=CLIPEmbeddingModel(),
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "mlflow",
            "transformers==4.46.3",
            "pillow",
            "torch",
        ],
        registered_model_name=UC_NAME,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Done. Note the model URI logged above — Phase 3a step 2 (create-endpoint) will deploy this version on a GPU serving endpoint with scale-to-zero.
