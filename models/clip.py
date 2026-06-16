# Databricks notebook source
# MAGIC %md
# MAGIC # CLIP ViT-L/14 — Text + Image Embeddings
# MAGIC
# MAGIC Registers a PyFunc wrapper around `openai/clip-vit-large-patch14` that produces
# MAGIC 768-d normalized embeddings for both text and images in a shared vector space.

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
from transformers import CLIPModel, CLIPProcessor

UC_CATALOG = dbutils.widgets.get("uc_catalog").strip()
UC_SCHEMA = "model_workbench"
UC_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.clip_vit_large_patch14"
HF_MODEL = "openai/clip-vit-large-patch14"

mlflow.set_registry_uri("databricks-uc")


class CLIPEmbeddingModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        self.processor = CLIPProcessor.from_pretrained(HF_MODEL)
        self.model = CLIPModel.from_pretrained(HF_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()

    def _decode_image(self, value: str) -> Image.Image:
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> list[list[float]]:
        records = model_input.to_dict(orient="records")
        texts, images = [], []
        for idx, r in enumerate(records):
            if str(r.get("type", "")).lower() == "text":
                texts.append((idx, str(r["value"])))
            else:
                images.append((idx, self._decode_image(str(r["value"]))))
        embeddings: list[list[float]] = [[] for _ in records]
        with torch.inference_mode():
            if texts:
                idxs, vals = zip(*texts)
                inp = self.processor(text=list(vals), return_tensors="pt", padding=True, truncation=True)
                inp = {k: v.to(self.device) for k, v in inp.items()}
                feats = self.model.get_text_features(**inp)
                feats = (feats / feats.norm(dim=-1, keepdim=True)).cpu().numpy()
                for j, i in enumerate(idxs):
                    embeddings[i] = feats[j].tolist()
            if images:
                idxs, vals = zip(*images)
                inp = self.processor(images=list(vals), return_tensors="pt")
                inp = {k: v.to(self.device) for k, v in inp.items()}
                feats = self.model.get_image_features(**inp)
                feats = (feats / feats.norm(dim=-1, keepdim=True)).cpu().numpy()
                for j, i in enumerate(idxs):
                    embeddings[i] = feats[j].tolist()
        return embeddings


# Register
_buf = io.BytesIO()
Image.new("RGB", (4, 4), color=(255, 0, 0)).save(_buf, format="PNG")
_tiny = base64.b64encode(_buf.getvalue()).decode("ascii")

_in = pd.DataFrame([{"type": "text", "value": "hello"}, {"type": "image", "value": _tiny}])
_sig = infer_signature(_in, [[0.0] * 768, [0.0] * 768])
with mlflow.start_run(run_name="clip-register"):
    mlflow.pyfunc.log_model("model", python_model=CLIPEmbeddingModel(), signature=_sig, input_example=_in,
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
    "name": "clip-vit-large-patch14",
    "config": {
        "served_entities": [{
            "name": "clip-vit-large-patch14",
            "entity_name": UC_NAME,
            "entity_version": "1",
            "workload_type": "GPU_MEDIUM",
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
        }],
        "traffic_config": {"routes": [{"served_model_name": "clip-vit-large-patch14", "traffic_percentage": 100}]},
    },
}
resp = requests.post(f"{HOST}/api/2.0/serving-endpoints", headers=HEADERS, json=payload)
if resp.status_code == 200:
    print(f"✓ Endpoint 'clip-vit-large-patch14' created")
elif "already exists" in resp.text.lower():
    print(f"⏭ Endpoint 'clip-vit-large-patch14' already exists")
else:
    print(f"✗ Failed ({resp.status_code}): {resp.text[:200]}")
