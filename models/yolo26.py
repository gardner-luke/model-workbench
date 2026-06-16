# Databricks notebook source
# MAGIC %md
# MAGIC # YOLO26 — Real-Time Object Detection
# MAGIC
# MAGIC Registers a PyFunc wrapper around Ultralytics YOLO26 (small variant) for detecting
# MAGIC 80 COCO object classes. NMS-free, one-to-one detection head.

# COMMAND ----------

# MAGIC %pip install -q -U mlflow ultralytics pillow torch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import base64
import io
from typing import Any

import mlflow
import pandas as pd
import torch
from mlflow.models.signature import infer_signature
from PIL import Image

UC_CATALOG = dbutils.widgets.get("uc_catalog").strip()
UC_SCHEMA = "model_workbench"
UC_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.yolo26"
YOLO_MODEL = "yolo26s.pt"

mlflow.set_registry_uri("databricks-uc")


class Yolo26Model(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        from ultralytics import YOLO
        self.model = YOLO(YOLO_MODEL)

    def _decode_image(self, value: str) -> Image.Image:
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> list[dict]:
        records = model_input.to_dict(orient="records")
        outputs = []
        for r in records:
            image = self._decode_image(str(r["image"]))
            w, h = image.size
            threshold = float(r.get("threshold") or 0.3)
            results = self.model(image, conf=threshold, verbose=False)
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy().tolist() if len(result.boxes) > 0 else []
            scores = result.boxes.conf.cpu().numpy().tolist() if len(result.boxes) > 0 else []
            labels = [result.names[int(c)] for c in result.boxes.cls.cpu().numpy()] if len(result.boxes) > 0 else []
            outputs.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "count": len(boxes),
                "image_size": [w, h],
            })
        return outputs


# Register
_buf = io.BytesIO()
Image.new("RGB", (4, 4), color=(255, 0, 0)).save(_buf, format="PNG")
_tiny = base64.b64encode(_buf.getvalue()).decode("ascii")

_in = pd.DataFrame([{"image": _tiny, "threshold": 0.3}])
_sig = infer_signature(_in, [{"boxes": [[0.0]*4], "scores": [0.9], "labels": ["person"], "count": 1, "image_size": [4, 4]}])
with mlflow.start_run(run_name="yolo26-register"):
    mlflow.pyfunc.log_model("model", python_model=Yolo26Model(), signature=_sig, input_example=_in,
                            pip_requirements=["mlflow", "ultralytics", "pillow", "torch"],
                            registered_model_name=UC_NAME)
print(f"✓ Model registered: {UC_NAME}")

# COMMAND ----------

# DBTITLE 1,Create serving endpoint
import requests

HOST = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

payload = {
    "name": "yolo26",
    "config": {
        "served_entities": [{
            "name": "yolo26",
            "entity_name": UC_NAME,
            "entity_version": "1",
            "workload_type": "GPU_SMALL",
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
        }],
        "traffic_config": {"routes": [{"served_model_name": "yolo26", "traffic_percentage": 100}]},
    },
}
resp = requests.post(f"{HOST}/api/2.0/serving-endpoints", headers=HEADERS, json=payload)
if resp.status_code == 200:
    print(f"✓ Endpoint 'yolo26' created")
elif "already exists" in resp.text.lower():
    print(f"⏭ Endpoint 'yolo26' already exists")
else:
    print(f"✗ Failed ({resp.status_code}): {resp.text[:200]}")
