# Databricks notebook source
# MAGIC %md
# MAGIC # Register Depth Anything V2 to Unity Catalog
# MAGIC
# MAGIC Depth Anything V2 (`depth-anything/Depth-Anything-V2-Large-hf`) is Meta-collaborator
# MAGIC TikTok Research's monocular depth estimation model. Single image in, per-pixel
# MAGIC depth out. Useful for robotics, AR, 3D scene reconstruction, and any pipeline
# MAGIC that needs spatial context from a single camera.
# MAGIC
# MAGIC ### Input contract (what the deployed endpoint accepts)
# MAGIC ```
# MAGIC {"dataframe_records": [{"image": "<base64 png/jpg>"}]}
# MAGIC ```
# MAGIC
# MAGIC ### Output contract
# MAGIC ```
# MAGIC {"predictions": [{
# MAGIC     "depth_png": "<base64 grayscale PNG>",   # same dims as input, brighter = closer
# MAGIC     "min_depth": 1.23,                       # raw depth values before normalization
# MAGIC     "max_depth": 87.65,
# MAGIC     "image_size": [width, height]
# MAGIC }]}
# MAGIC ```
# MAGIC
# MAGIC The depth map is encoded as an 8-bit grayscale PNG (small, fast to transport,
# MAGIC easy to render). The wrapper normalizes raw depth to 0–255 per image, so if you
# MAGIC need the original scale, multiply: `raw = (pixel/255) * (max_depth - min_depth) + min_depth`.

# COMMAND ----------

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
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# --------------------------------------------------------------------------------------
# Configuration — change these to match your workspace.
# --------------------------------------------------------------------------------------
HF_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"   # Apache 2.0, not gated.
UC_CATALOG = "<YOUR_CATALOG>"          # Unity Catalog catalog name.
UC_SCHEMA = "model_workbench"          # Unity Catalog schema name.
UC_MODEL = "depth_anything"            # Unity Catalog registered model name.
# --------------------------------------------------------------------------------------

UC_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL}"

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")  # noqa: F821

# COMMAND ----------

class DepthAnythingModel(mlflow.pyfunc.PythonModel):
    """Custom PyFunc wrapping Depth Anything V2 for monocular depth estimation.

    Returns one prediction record per input row, with the depth map encoded as a
    base64 grayscale PNG (small over the wire, easy to overlay on a canvas).
    """

    def load_context(self, context: Any) -> None:
        self.processor = AutoImageProcessor.from_pretrained(HF_MODEL)
        self.model = AutoModelForDepthEstimation.from_pretrained(HF_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()

    def _decode_image(self, value: str) -> Image.Image:
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")

    def _encode_depth(self, depth: np.ndarray) -> str:
        # depth is HxW float — normalize to 0-255 (brighter = closer) and write as
        # a single-channel PNG. We preserve the raw min/max separately so callers
        # can recover the original scale if needed.
        img = Image.fromarray(depth.astype(np.uint8), mode="L")
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

                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                raw = self.model(**inputs)
                pred = raw.predicted_depth  # shape (1, H', W')

                # Upsample back to source dimensions.
                resized = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()

                d_min = float(resized.min())
                d_max = float(resized.max())
                if d_max - d_min > 1e-9:
                    norm = (resized - d_min) / (d_max - d_min) * 255.0
                else:
                    norm = np.zeros_like(resized)

                outputs.append({
                    "depth_png": self._encode_depth(norm),
                    "min_depth": d_min,
                    "max_depth": d_max,
                    "image_size": [width, height],
                })

        return outputs

# COMMAND ----------

_tiny_buf = io.BytesIO()
Image.new("RGB", (4, 4), color=(255, 0, 0)).save(_tiny_buf, format="PNG")
_tiny_b64 = base64.b64encode(_tiny_buf.getvalue()).decode("ascii")

input_example = pd.DataFrame([{"image": _tiny_b64}])
output_example = [
    {
        "depth_png": "<base64>",
        "min_depth": 0.0,
        "max_depth": 1.0,
        "image_size": [4, 4],
    }
]
signature = infer_signature(input_example, output_example)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="depth-anything-register"):
    mlflow.pyfunc.log_model(
        artifact_path="depth-anything-pyfunc",
        python_model=DepthAnythingModel(),
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
