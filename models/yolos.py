# Databricks notebook source
# MAGIC %md
# MAGIC # YOLOS — Closed-Vocab Object Detection
# MAGIC
# MAGIC Registers a PyFunc wrapper around `hustvl/yolos-small` for detecting
# MAGIC 80 COCO object classes in images.

# COMMAND ----------

# MAGIC %pip install -q -U mlflow transformers==4.46.3 pillow torch

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
from transformers import AutoImageProcessor, YolosForObjectDetection

UC_CATALOG = dbutils.widgets.get("uc_catalog").strip()
UC_SCHEMA = "model_workbench"
UC_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.yolos"
HF_MODEL = "hustvl/yolos-small"

mlflow.set_registry_uri("databricks-uc")


class YolosModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        self.processor = AutoImageProcessor.from_pretrained(HF_MODEL)
        self.model = YolosForObjectDetection.from_pretrained(HF_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()
        self.id2label = self.model.config.id2label

    def _decode_image(self, value: str) -> Image.Image:
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> list[dict]:
        records = model_input.to_dict(orient="records")
        outputs = []
        with torch.inference_mode():
            for r in records:
                image = self._decode_image(str(r["image"]))
                w, h = image.size
                threshold = float(r.get("threshold") or 0.3)
                inp = self.processor(images=image, return_tensors="pt").to(self.device)
                raw = self.model(**inp)
                res = self.processor.post_process_object_detection(raw, threshold=threshold, target_sizes=torch.tensor([(h, w)]))[0]
                outputs.append({
                    "boxes": res["boxes"].cpu().numpy().tolist(),
                    "scores": res["scores"].cpu().numpy().tolist(),
                    "labels": [self.id2label.get(int(l), str(int(l))) for l in res["labels"].cpu().numpy()],
                    "count": len(res["boxes"]), "image_size": [w, h]
                })
        return outputs


# Register
_buf = io.BytesIO()
Image.new("RGB", (4, 4), color=(255, 0, 0)).save(_buf, format="PNG")
_tiny = base64.b64encode(_buf.getvalue()).decode("ascii")

_in = pd.DataFrame([{"image": _tiny, "threshold": 0.3}])
_sig = infer_signature(_in, [{"boxes": [[0.0]*4], "scores": [0.9], "labels": ["person"], "count": 1, "image_size": [4, 4]}])
with mlflow.start_run(run_name="yolos-register"):
    mlflow.pyfunc.log_model("model", python_model=YolosModel(), signature=_sig, input_example=_in,
                            pip_requirements=["mlflow", "transformers==4.46.3", "pillow", "torch"],
                            registered_model_name=UC_NAME)
print(f"✓ Model registered: {UC_NAME}")

# COMMAND ----------

# DBTITLE 1,Create serving endpoint
import requests

HOST = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

payload = {
    "name": "yolos",
    "config": {
        "served_entities": [{
            "name": "yolos",
            "entity_name": UC_NAME,
            "entity_version": "1",
            "workload_type": "GPU_SMALL",
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
        }],
        "traffic_config": {"routes": [{"served_model_name": "yolos", "traffic_percentage": 100}]},
    },
}
resp = requests.post(f"{HOST}/api/2.0/serving-endpoints", headers=HEADERS, json=payload)
if resp.status_code == 200:
    print(f"✓ Endpoint 'yolos' created")
elif "already exists" in resp.text.lower():
    print(f"⏭ Endpoint 'yolos' already exists")
else:
    print(f"✗ Failed ({resp.status_code}): {resp.text[:200]}")
