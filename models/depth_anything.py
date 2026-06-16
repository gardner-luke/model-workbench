# Databricks notebook source
# MAGIC %md
# MAGIC # Depth Anything V2 — Monocular Depth Estimation
# MAGIC
# MAGIC Registers a PyFunc wrapper around `depth-anything/Depth-Anything-V2-Large-hf`
# MAGIC that returns a grayscale depth map PNG for any input image.

# COMMAND ----------

# MAGIC %pip install -q -U mlflow transformers==4.46.3 pillow torch

# COMMAND ----------

dbutils.library.restartPython()

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

UC_CATALOG = dbutils.widgets.get("uc_catalog").strip()
UC_SCHEMA = "model_workbench"
UC_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.depth_anything"
HF_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"

mlflow.set_registry_uri("databricks-uc")


class DepthAnythingModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        self.processor = AutoImageProcessor.from_pretrained(HF_MODEL)
        self.model = AutoModelForDepthEstimation.from_pretrained(HF_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()

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
                inp = self.processor(images=image, return_tensors="pt").to(self.device)
                raw = self.model(**inp)
                depth = torch.nn.functional.interpolate(
                    raw.predicted_depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
                ).squeeze().cpu().numpy()
                d_min, d_max = float(depth.min()), float(depth.max())
                norm = (depth - d_min) / (d_max - d_min) * 255.0 if d_max - d_min > 1e-9 else np.zeros_like(depth)
                buf = io.BytesIO()
                Image.fromarray(norm.astype(np.uint8), mode="L").save(buf, format="PNG", optimize=True)
                outputs.append({
                    "depth_png": base64.b64encode(buf.getvalue()).decode("ascii"),
                    "min_depth": d_min, "max_depth": d_max, "image_size": [w, h]
                })
        return outputs


# Register
_buf = io.BytesIO()
Image.new("RGB", (4, 4), color=(255, 0, 0)).save(_buf, format="PNG")
_tiny = base64.b64encode(_buf.getvalue()).decode("ascii")

_in = pd.DataFrame([{"image": _tiny}])
_sig = infer_signature(_in, [{"depth_png": "x", "min_depth": 0.0, "max_depth": 1.0, "image_size": [4, 4]}])
with mlflow.start_run(run_name="depth-anything-register"):
    mlflow.pyfunc.log_model("model", python_model=DepthAnythingModel(), signature=_sig, input_example=_in,
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
    "name": "depth-anything",
    "config": {
        "served_entities": [{
            "name": "depth-anything",
            "entity_name": UC_NAME,
            "entity_version": "1",
            "workload_type": "GPU_SMALL",
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
        }],
        "traffic_config": {"routes": [{"served_model_name": "depth-anything", "traffic_percentage": 100}]},
    },
}
resp = requests.post(f"{HOST}/api/2.0/serving-endpoints", headers=HEADERS, json=payload)
if resp.status_code == 200:
    print(f"✓ Endpoint 'depth-anything' created")
elif "already exists" in resp.text.lower():
    print(f"⏭ Endpoint 'depth-anything' already exists")
else:
    print(f"✗ Failed ({resp.status_code}): {resp.text[:200]}")
