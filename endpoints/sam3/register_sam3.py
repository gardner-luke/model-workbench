# Databricks notebook source
# MAGIC %md
# MAGIC # Register SAM 3 to Unity Catalog
# MAGIC
# MAGIC SAM 3 (Segment Anything Model 3) is Meta's promptable concept segmentation model.
# MAGIC Given an image and a short noun phrase (e.g. "corn kernel", "tractor"), it returns a
# MAGIC binary mask, bounding box, and confidence score for every matching instance in the
# MAGIC image.
# MAGIC
# MAGIC This notebook wraps the HuggingFace `Sam3Model` + `Sam3Processor` classes in a
# MAGIC custom MLflow PyFunc and registers it to Unity Catalog so Databricks Model Serving
# MAGIC can serve it on a GPU endpoint.
# MAGIC
# MAGIC ### Prerequisites
# MAGIC
# MAGIC 1. **HuggingFace access**. `facebook/sam3` is a gated repo — request access at
# MAGIC    https://huggingface.co/facebook/sam3 (approval is usually quick).
# MAGIC 2. **HF token in a Databricks secret**. Save your HF read token (from
# MAGIC    https://huggingface.co/settings/tokens) into a Databricks secret. The constants
# MAGIC    `HF_TOKEN_SCOPE` and `HF_TOKEN_KEY` below tell this notebook where to look.
# MAGIC 3. **UC catalog + schema**. The constants `UC_CATALOG` and `UC_SCHEMA` below should
# MAGIC    point at a catalog/schema you can create tables in.
# MAGIC
# MAGIC ### Input contract (what the deployed endpoint accepts)
# MAGIC ```
# MAGIC {"dataframe_records": [{
# MAGIC     "image": "<base64 png/jpg>",
# MAGIC     "text_prompt": "corn kernel",
# MAGIC     "threshold": 0.5,         # optional, presence threshold (filters low-confidence)
# MAGIC     "mask_threshold": 0.5     # optional, per-pixel binarization threshold
# MAGIC }]}
# MAGIC ```
# MAGIC
# MAGIC ### Output contract
# MAGIC ```
# MAGIC {"predictions": {
# MAGIC     "masks": ["<base64 PNG>", ...],     # 1-bit PNG per detected instance, image-sized
# MAGIC     "boxes": [[x1, y1, x2, y2], ...],   # absolute pixel xyxy
# MAGIC     "scores": [0.91, 0.87, ...],
# MAGIC     "count": 12,
# MAGIC     "image_size": [width, height]
# MAGIC }}
# MAGIC ```

# COMMAND ----------

# Install the transformers version that includes the Sam3 classes plus their deps.
# transformers 5.0+ ships Sam3Model and Sam3Processor; we pin to a known-good 5.x
# release that matches the huggingface_hub 1.x API. accelerate is required by the
# transformers 5.x model loader.
# MAGIC %pip install -q -U mlflow "transformers==5.8.0" "huggingface_hub>=1.0,<2.0" "accelerate>=0.34.0" pillow torch
# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import base64
import io
import os
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.models.signature import infer_signature
from PIL import Image

# --------------------------------------------------------------------------------------
# Configuration — change these to match your workspace.
# --------------------------------------------------------------------------------------
HF_MODEL = "facebook/sam3"            # Source weights on HuggingFace.
UC_CATALOG = "<YOUR_CATALOG>"          # Unity Catalog catalog name.
UC_SCHEMA = "model_workbench"          # Unity Catalog schema name.
UC_MODEL = "sam3"                      # Unity Catalog registered model name.
HF_TOKEN_SCOPE = "<YOUR_SCOPE>"        # Databricks secret scope holding your HF token.
HF_TOKEN_KEY = "<YOUR_KEY>"            # Key under HF_TOKEN_SCOPE holding the token.
# --------------------------------------------------------------------------------------

UC_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL}"

# HF model gating requires a valid token; pull it from the Databricks secret scope.
# Surface a clear error early instead of failing deep inside transformers at serving time.
HF_TOKEN_RAW = dbutils.secrets.get(scope=HF_TOKEN_SCOPE, key=HF_TOKEN_KEY)  # noqa: F821
HF_TOKEN = (HF_TOKEN_RAW or "").strip()
if not HF_TOKEN:
    raise RuntimeError(
        f"HF_TOKEN is empty. Put your HuggingFace read token in scope '{HF_TOKEN_SCOPE}' "
        f"key '{HF_TOKEN_KEY}' before running this notebook."
    )
# Reject non-ASCII characters here so we don't crash deep inside httpx at serving time
# when the token gets passed as an Authorization header.
if not HF_TOKEN.isascii():
    raise RuntimeError("HF_TOKEN contains non-ASCII characters — regenerate the token on HF.")
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")  # noqa: F821

# COMMAND ----------

class Sam3SegmenterModel(mlflow.pyfunc.PythonModel):
    """Custom PyFunc wrapping SAM 3.1 for promptable concept segmentation.

    Each input row is one (image, text_prompt) pair. The output is structured per-row so
    the serving endpoint returns a list of detections for each input image.
    """

    def load_context(self, context: Any) -> None:
        from transformers import Sam3Model, Sam3Processor

        token = (os.environ.get("HF_TOKEN") or "").strip()
        if not token or not token.isascii():
            raise RuntimeError(
                "HF_TOKEN missing/non-ASCII at serving startup. Verify the secret value."
            )
        self.processor = Sam3Processor.from_pretrained(HF_MODEL, token=token)
        self.model = Sam3Model.from_pretrained(HF_MODEL, token=token)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()

    def _decode_image(self, value: str) -> Image.Image:
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")

    def _encode_mask(self, mask: np.ndarray) -> str:
        # mask is HxW bool / 0-1; render as a 1-bit PNG (alpha channel) for compact transport.
        mask_uint = (mask.astype(np.uint8) * 255)
        img = Image.fromarray(mask_uint, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> list[dict]:
        records = (
            model_input.to_dict(orient="records")
            if isinstance(model_input, pd.DataFrame)
            else list(model_input)
        )

        outputs: list[dict] = []
        with torch.inference_mode():
            for r in records:
                img_b64 = r.get("image")
                if not img_b64:
                    raise ValueError("`image` (base64) is required on every record")
                image = self._decode_image(str(img_b64))
                width, height = image.size

                text_prompt = r.get("text_prompt") or None
                threshold = float(r.get("threshold") or 0.5)
                mask_threshold = float(r.get("mask_threshold") or 0.5)

                proc_inputs = self.processor(
                    images=image,
                    text=text_prompt,
                    return_tensors="pt",
                )
                proc_inputs = {k: v.to(self.device) for k, v in proc_inputs.items()}

                raw = self.model(**proc_inputs)

                # post_process_instance_segmentation returns a list (one per image).
                result = self.processor.post_process_instance_segmentation(
                    raw,
                    threshold=threshold,
                    mask_threshold=mask_threshold,
                    target_sizes=[(height, width)],
                )[0]

                masks = result.get("masks")
                boxes = result.get("boxes")
                scores = result.get("scores")

                mask_list: list[str] = []
                box_list: list[list[float]] = []
                score_list: list[float] = []

                if masks is not None and len(masks) > 0:
                    masks_np = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
                    boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
                    scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
                    for m, b, s in zip(masks_np, boxes_np, scores_np):
                        # SAM 3 sometimes returns masks shaped (1, H, W); squeeze leading singleton dims.
                        m2 = np.squeeze(m)
                        mask_list.append(self._encode_mask(m2 > 0))
                        box_list.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                        score_list.append(float(s))

                outputs.append(
                    {
                        "masks": mask_list,
                        "boxes": box_list,
                        "scores": score_list,
                        "count": len(mask_list),
                        "image_size": [width, height],
                    }
                )

        return outputs

# COMMAND ----------

# Build a 4x4 png example for signature inference.
_tiny_png_buf = io.BytesIO()
Image.new("RGB", (4, 4), color=(255, 0, 0)).save(_tiny_png_buf, format="PNG")
_tiny_png_b64 = base64.b64encode(_tiny_png_buf.getvalue()).decode("ascii")

input_example = pd.DataFrame(
    [
        {"image": _tiny_png_b64, "text_prompt": "red square", "threshold": 0.5, "mask_threshold": 0.5},
    ]
)
output_example = [
    {"masks": ["<base64>"], "boxes": [[0.0, 0.0, 4.0, 4.0]], "scores": [0.9], "count": 1, "image_size": [4, 4]}
]
signature = infer_signature(input_example, output_example)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="sam3-1-register"):
    mlflow.pyfunc.log_model(
        artifact_path="sam3-pyfunc",
        python_model=Sam3SegmenterModel(),
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "mlflow",
            "transformers==5.8.0",
            "huggingface_hub>=1.0,<2.0",
            "accelerate>=0.34.0",
            "pillow",
            "torch",
        ],
        registered_model_name=UC_NAME,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Done. Phase 4a step 2 (create-endpoint) will deploy this version on a GPU serving
# MAGIC endpoint with scale-to-zero, passing `HF_TOKEN` as an environment variable from the
# MAGIC same secret so SAM 3 weights download at cold-start.
