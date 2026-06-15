# Databricks notebook source
# MAGIC %md
# MAGIC # Register Grounding DINO to Unity Catalog
# MAGIC
# MAGIC Grounding DINO (IDEA Research) is an **open-vocabulary** object detection model.
# MAGIC Unlike YOLO/DETR which are limited to a fixed set of training classes, you give
# MAGIC Grounding DINO any noun phrase ("tractor", "yellow circle", "industrial pipe")
# MAGIC and it returns bounding boxes for every matching instance.
# MAGIC
# MAGIC Versus SAM 3 (which also takes text prompts): Grounding DINO returns boxes only,
# MAGIC no masks. Faster + cheaper when "where is the thing" is enough. The two pair
# MAGIC well: Grounding DINO → SAM 3 is the classic "Grounded SAM" auto-labeling pipeline.
# MAGIC
# MAGIC ### Input contract (what the deployed endpoint accepts)
# MAGIC ```
# MAGIC {"dataframe_records": [{
# MAGIC     "image": "<base64 png/jpg>",
# MAGIC     "text_prompt": "tractor. person. wheel.",   # noun phrases separated by periods
# MAGIC     "threshold": 0.3            # optional, box confidence threshold
# MAGIC }]}
# MAGIC ```
# MAGIC Concepts are separated by periods (Grounding DINO's training convention).
# MAGIC
# MAGIC ### Output contract
# MAGIC ```
# MAGIC {"predictions": [{
# MAGIC     "boxes": [[x1, y1, x2, y2], ...],   # absolute pixel xyxy
# MAGIC     "scores": [0.91, 0.87, ...],
# MAGIC     "labels": ["tractor", "person", ...], # matched concept per box
# MAGIC     "count": 3,
# MAGIC     "image_size": [width, height]
# MAGIC }]}
# MAGIC ```

# COMMAND ----------

# transformers 4.46+ ships GroundingDinoForObjectDetection.
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
from transformers import AutoProcessor, GroundingDinoForObjectDetection

# --------------------------------------------------------------------------------------
# Configuration — change these to match your workspace.
# --------------------------------------------------------------------------------------
HF_MODEL = "IDEA-Research/grounding-dino-base"   # Apache 2.0, not gated.
UC_CATALOG = "<YOUR_CATALOG>"                  # Unity Catalog catalog name.
UC_SCHEMA = "model_workbench"                  # Unity Catalog schema name.
UC_MODEL = "grounding_dino"                    # Unity Catalog registered model name.
# --------------------------------------------------------------------------------------

UC_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL}"

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")  # noqa: F821

# COMMAND ----------

class GroundingDinoModel(mlflow.pyfunc.PythonModel):
    """Custom PyFunc wrapping Grounding DINO for open-vocab object detection.

    Returns one prediction record per input row so the MLflow response is
    `{"predictions": [{...}, {...}]}` — the shape ai_query and downstream batch
    tooling expect.
    """

    def load_context(self, context: Any) -> None:
        self.processor = AutoProcessor.from_pretrained(HF_MODEL)
        self.model = GroundingDinoForObjectDetection.from_pretrained(HF_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()

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

                text_prompt = (r.get("text_prompt") or "object .").strip()
                # Grounding DINO expects concepts separated by ". " — normalize whatever
                # the caller sent into that form.
                concepts = [c.strip() for c in text_prompt.replace(",", ".").split(".") if c.strip()]
                normalized = ". ".join(concepts) + "."

                threshold = float(r.get("threshold") or 0.3)
                text_threshold = float(r.get("text_threshold") or 0.25)

                inputs = self.processor(
                    images=image,
                    text=normalized,
                    return_tensors="pt",
                ).to(self.device)
                raw = self.model(**inputs)

                results = self.processor.post_process_grounded_object_detection(
                    raw,
                    inputs.input_ids,
                    box_threshold=threshold,
                    text_threshold=text_threshold,
                    target_sizes=[(height, width)],
                )[0]

                boxes_t = results.get("boxes")
                scores_t = results.get("scores")
                # `text_labels` (newer transformers) or `labels` — both can appear.
                labels = results.get("text_labels") or results.get("labels") or []
                if isinstance(labels, torch.Tensor):
                    labels = [str(int(x)) for x in labels.tolist()]

                box_list: list[list[float]] = []
                score_list: list[float] = []
                label_list: list[str] = []

                if boxes_t is not None and len(boxes_t) > 0:
                    boxes_np = boxes_t.detach().cpu().numpy()
                    scores_np = scores_t.detach().cpu().numpy() if scores_t is not None else np.zeros(len(boxes_np))
                    for b, s, lab in zip(boxes_np, scores_np, labels):
                        box_list.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                        score_list.append(float(s))
                        label_list.append(str(lab))

                outputs.append({
                    "boxes": box_list,
                    "scores": score_list,
                    "labels": label_list,
                    "count": len(box_list),
                    "image_size": [width, height],
                })

        return outputs

# COMMAND ----------

# Build a tiny example for signature inference.
_tiny_buf = io.BytesIO()
Image.new("RGB", (4, 4), color=(255, 0, 0)).save(_tiny_buf, format="PNG")
_tiny_b64 = base64.b64encode(_tiny_buf.getvalue()).decode("ascii")

input_example = pd.DataFrame(
    [{"image": _tiny_b64, "text_prompt": "red square.", "threshold": 0.3}]
)
output_example = [
    {
        "boxes": [[0.0, 0.0, 4.0, 4.0]],
        "scores": [0.9],
        "labels": ["red square"],
        "count": 1,
        "image_size": [4, 4],
    }
]
signature = infer_signature(input_example, output_example)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="grounding-dino-register"):
    mlflow.pyfunc.log_model(
        artifact_path="grounding-dino-pyfunc",
        python_model=GroundingDinoModel(),
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
