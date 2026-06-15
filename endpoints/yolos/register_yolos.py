# Databricks notebook source
# MAGIC %md
# MAGIC # Register YOLOS (transformer YOLO) to Unity Catalog
# MAGIC
# MAGIC YOLOS (`hustvl/yolos-small`) is a transformer-based YOLO trained on the COCO
# MAGIC 80-class dataset. **Closed vocabulary** — there's no prompt, it just returns
# MAGIC boxes for every person, car, bottle, etc. it recognizes. Fast and small
# MAGIC compared to Grounding DINO, which makes it the right pick when you don't need
# MAGIC open-vocab detection.
# MAGIC
# MAGIC Note: this is not Ultralytics YOLOv8 — that one is AGPL-licensed. YOLOS is
# MAGIC Apache 2.0 and lives natively in the `transformers` library, so the wrapper
# MAGIC pattern matches everything else in this repo.
# MAGIC
# MAGIC ### Input contract (what the deployed endpoint accepts)
# MAGIC ```
# MAGIC {"dataframe_records": [{
# MAGIC     "image": "<base64 png/jpg>",
# MAGIC     "threshold": 0.3            # optional, box confidence threshold
# MAGIC }]}
# MAGIC ```
# MAGIC (No text_prompt — YOLOS detects from a fixed class list.)
# MAGIC
# MAGIC ### Output contract
# MAGIC ```
# MAGIC {"predictions": [{
# MAGIC     "boxes": [[x1, y1, x2, y2], ...],   # absolute pixel xyxy
# MAGIC     "scores": [0.91, 0.87, ...],
# MAGIC     "labels": ["person", "car", ...],   # COCO class names
# MAGIC     "count": 3,
# MAGIC     "image_size": [width, height]
# MAGIC }]}
# MAGIC ```

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
from transformers import AutoImageProcessor, YolosForObjectDetection

# --------------------------------------------------------------------------------------
# Configuration — change these to match your workspace.
# --------------------------------------------------------------------------------------
HF_MODEL = "hustvl/yolos-small"        # Apache 2.0, not gated. COCO 80 classes.
UC_CATALOG = "<YOUR_CATALOG>"          # Unity Catalog catalog name.
UC_SCHEMA = "model_workbench"          # Unity Catalog schema name.
UC_MODEL = "yolos"                     # Unity Catalog registered model name.
# --------------------------------------------------------------------------------------

UC_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL}"

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")  # noqa: F821

# COMMAND ----------

class YolosModel(mlflow.pyfunc.PythonModel):
    """Custom PyFunc wrapping YOLOS for closed-vocab object detection.

    Returns one prediction record per input row so the response is
    `{"predictions": [{...}, {...}]}` — ai_query-compatible.
    """

    def load_context(self, context: Any) -> None:
        self.processor = AutoImageProcessor.from_pretrained(HF_MODEL)
        self.model = YolosForObjectDetection.from_pretrained(HF_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()
        # `id2label` maps COCO class index to human-readable name.
        self.id2label = self.model.config.id2label

    def _decode_image(self, value: str) -> Image.Image:
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")

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

                threshold = float(r.get("threshold") or 0.3)

                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                raw = self.model(**inputs)

                results = self.processor.post_process_object_detection(
                    raw,
                    threshold=threshold,
                    target_sizes=torch.tensor([(height, width)]),
                )[0]

                boxes_np = results["boxes"].detach().cpu().numpy() if len(results.get("boxes", [])) else np.empty((0, 4))
                scores_np = results["scores"].detach().cpu().numpy() if len(results.get("scores", [])) else np.empty((0,))
                label_ids = results["labels"].detach().cpu().numpy() if len(results.get("labels", [])) else np.empty((0,), dtype=int)

                box_list: list[list[float]] = []
                score_list: list[float] = []
                label_list: list[str] = []
                for b, s, lid in zip(boxes_np, scores_np, label_ids):
                    box_list.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                    score_list.append(float(s))
                    label_list.append(str(self.id2label.get(int(lid), str(int(lid)))))

                outputs.append({
                    "boxes": box_list,
                    "scores": score_list,
                    "labels": label_list,
                    "count": len(box_list),
                    "image_size": [width, height],
                })

        return outputs

# COMMAND ----------

_tiny_buf = io.BytesIO()
Image.new("RGB", (4, 4), color=(255, 0, 0)).save(_tiny_buf, format="PNG")
_tiny_b64 = base64.b64encode(_tiny_buf.getvalue()).decode("ascii")

input_example = pd.DataFrame([{"image": _tiny_b64, "threshold": 0.3}])
output_example = [
    {
        "boxes": [[0.0, 0.0, 4.0, 4.0]],
        "scores": [0.9],
        "labels": ["person"],
        "count": 1,
        "image_size": [4, 4],
    }
]
signature = infer_signature(input_example, output_example)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="yolos-register"):
    mlflow.pyfunc.log_model(
        artifact_path="yolos-pyfunc",
        python_model=YolosModel(),
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
