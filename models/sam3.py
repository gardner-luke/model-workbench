# Databricks notebook source
# MAGIC %md
# MAGIC # SAM 3 — Promptable Segmentation
# MAGIC
# MAGIC Registers a PyFunc wrapper around `facebook/sam3` for text-prompted instance
# MAGIC segmentation. This is a gated HuggingFace model — requires a token with access.
# MAGIC
# MAGIC **Requires widgets**: `uc_catalog`, `hf_token_scope`, `hf_token_key`

# COMMAND ----------

# MAGIC %pip install -q -U mlflow "transformers==5.8.0" "huggingface_hub>=1.0,<2.0" "accelerate>=0.34.0" pillow torch

# COMMAND ----------

dbutils.library.restartPython()

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
from transformers import Sam3Model, Sam3Processor

UC_CATALOG = dbutils.widgets.get("uc_catalog").strip()
UC_SCHEMA = "model_workbench"
UC_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.sam3"
HF_MODEL = "facebook/sam3"

HF_TOKEN_SCOPE = dbutils.widgets.get("hf_token_scope").strip()
HF_TOKEN_KEY = dbutils.widgets.get("hf_token_key").strip()
HF_TOKEN = dbutils.secrets.get(scope=HF_TOKEN_SCOPE, key=HF_TOKEN_KEY).strip()
assert HF_TOKEN and HF_TOKEN.isascii(), "HF token is empty or invalid — check your secret"
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

mlflow.set_registry_uri("databricks-uc")


class Sam3SegmenterModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        from transformers import Sam3Model, Sam3Processor
        token = (os.environ.get("HF_TOKEN") or "").strip()
        self.processor = Sam3Processor.from_pretrained("facebook/sam3", token=token)
        self.model = Sam3Model.from_pretrained("facebook/sam3", token=token)
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
                proc_inputs = self.processor(images=image, text=r.get("text_prompt"), return_tensors="pt")
                proc_inputs = {k: v.to(self.device) for k, v in proc_inputs.items()}
                raw = self.model(**proc_inputs)
                result = self.processor.post_process_instance_segmentation(
                    raw, threshold=float(r.get("threshold") or 0.5),
                    mask_threshold=float(r.get("mask_threshold") or 0.5), target_sizes=[(h, w)])[0]
                masks = result.get("masks")
                boxes = result.get("boxes")
                scores = result.get("scores")
                mask_list, box_list, score_list = [], [], []
                if masks is not None and len(masks) > 0:
                    masks_np = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
                    boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
                    scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
                    for m, b, s in zip(masks_np, boxes_np, scores_np):
                        buf = io.BytesIO()
                        Image.fromarray((np.squeeze(m) > 0).astype(np.uint8) * 255, mode="L").save(buf, format="PNG")
                        mask_list.append(base64.b64encode(buf.getvalue()).decode("ascii"))
                        box_list.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                        score_list.append(float(s))
                outputs.append({"masks": mask_list, "boxes": box_list, "scores": score_list,
                                "count": len(mask_list), "image_size": [w, h]})
        return outputs


# Register
_buf = io.BytesIO()
Image.new("RGB", (4, 4), color=(255, 0, 0)).save(_buf, format="PNG")
_tiny = base64.b64encode(_buf.getvalue()).decode("ascii")

_in = pd.DataFrame([{"image": _tiny, "text_prompt": "object", "threshold": 0.5, "mask_threshold": 0.5}])
_sig = infer_signature(_in, [{"masks": ["x"], "boxes": [[0.0]*4], "scores": [0.9], "count": 1, "image_size": [4, 4]}])
with mlflow.start_run(run_name="sam3-register"):
    mlflow.pyfunc.log_model("model", python_model=Sam3SegmenterModel(), signature=_sig, input_example=_in,
                            pip_requirements=["mlflow", "transformers==5.8.0", "huggingface_hub>=1.0,<2.0", "accelerate>=0.34.0", "pillow", "torch"],
                            registered_model_name=UC_NAME)
print(f"✓ Model registered: {UC_NAME}")

# COMMAND ----------

# DBTITLE 1,Create serving endpoint
import requests

HOST = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

payload = {
    "name": "sam3",
    "config": {
        "served_entities": [{
            "name": "sam3",
            "entity_name": UC_NAME,
            "entity_version": "1",
            "workload_type": "GPU_MEDIUM",
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
            "environment_vars": {
                "HF_TOKEN": f"{{{{secrets/{HF_TOKEN_SCOPE}/{HF_TOKEN_KEY}}}}}",
                "HUGGINGFACE_HUB_TOKEN": f"{{{{secrets/{HF_TOKEN_SCOPE}/{HF_TOKEN_KEY}}}}}",
            },
        }],
        "traffic_config": {"routes": [{"served_model_name": "sam3", "traffic_percentage": 100}]},
    },
}
resp = requests.post(f"{HOST}/api/2.0/serving-endpoints", headers=HEADERS, json=payload)
if resp.status_code == 200:
    print(f"✓ Endpoint 'sam3' created")
elif "already exists" in resp.text.lower():
    print(f"⏭ Endpoint 'sam3' already exists")
else:
    print(f"✗ Failed ({resp.status_code}): {resp.text[:200]}")
