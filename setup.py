# Databricks notebook source
# MAGIC %md
# MAGIC # Model Workbench — Setup
# MAGIC
# MAGIC This notebook deploys everything you need to run Model Workbench:
# MAGIC 1. Creates a Unity Catalog schema for the custom models
# MAGIC 2. Registers each model to UC (CLIP, YOLOS, Grounding DINO, Depth Anything, and optionally SAM 3)
# MAGIC 3. Creates GPU model serving endpoints with scale-to-zero
# MAGIC 4. Deploys the Databricks App
# MAGIC
# MAGIC **Time to complete**: ~15–20 minutes (most of that is endpoint provisioning).
# MAGIC
# MAGIC ### Before you start
# MAGIC - You need a Unity Catalog catalog you can create schemas in
# MAGIC - GPU model serving must be available in your workspace
# MAGIC - (Optional) For SAM 3: a HuggingFace token with access to `facebook/sam3`
# MAGIC   saved as a Databricks secret

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Edit the values below, then **Run All**.

# COMMAND ----------

# DBTITLE 1,Configuration — edit these values
UC_CATALOG = "<YOUR_CATALOG>"          # Your Unity Catalog catalog name
UC_SCHEMA = "model_workbench"          # Schema name (will be created if it doesn't exist)

# SAM 3 is a gated HuggingFace model — set these if you want to deploy it.
# Leave as empty strings to skip SAM 3 deployment.
HF_TOKEN_SCOPE = ""                    # Databricks secret scope with your HF token
HF_TOKEN_KEY = ""                      # Key within that scope

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create UC Schema

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")  # noqa: F821
print(f"✓ Schema {UC_CATALOG}.{UC_SCHEMA} ready")

# Store config so it survives %pip restartPython
spark.conf.set("spark.databricks.mw.catalog", UC_CATALOG)  # noqa: F821
spark.conf.set("spark.databricks.mw.hfScope", HF_TOKEN_SCOPE)  # noqa: F821
spark.conf.set("spark.databricks.mw.hfKey", HF_TOKEN_KEY)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Register Models

# COMMAND ----------

# MAGIC %pip install -q -U mlflow transformers==4.46.3 pillow torch

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# Recover config after restart
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

UC_CATALOG = spark.conf.get("spark.databricks.mw.catalog")  # noqa: F821
UC_SCHEMA = "model_workbench"
HF_TOKEN_SCOPE = spark.conf.get("spark.databricks.mw.hfScope", "")  # noqa: F821
HF_TOKEN_KEY = spark.conf.get("spark.databricks.mw.hfKey", "")  # noqa: F821

# Tiny test image reused across all registrations
_tiny_buf = io.BytesIO()
Image.new("RGB", (4, 4), color=(255, 0, 0)).save(_tiny_buf, format="PNG")
_tiny_b64 = base64.b64encode(_tiny_buf.getvalue()).decode("ascii")

mlflow.set_registry_uri("databricks-uc")
print(f"Config: catalog={UC_CATALOG}, schema={UC_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CLIP (text + image embeddings)

# COMMAND ----------

from transformers import CLIPModel, CLIPProcessor

HF_MODEL_CLIP = "openai/clip-vit-large-patch14"
UC_NAME_CLIP = f"{UC_CATALOG}.{UC_SCHEMA}.clip_vit_large_patch14"


class CLIPEmbeddingModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        self.processor = CLIPProcessor.from_pretrained(HF_MODEL_CLIP)
        self.model = CLIPModel.from_pretrained(HF_MODEL_CLIP)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()

    def _decode_image(self, value: str) -> Image.Image:
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> list[list[float]]:
        records = model_input.to_dict(orient="records") if isinstance(model_input, pd.DataFrame) else list(model_input)
        texts, images = [], []
        for idx, r in enumerate(records):
            t = str(r.get("type", "")).lower()
            v = r.get("value", "")
            if t == "text":
                texts.append((idx, str(v)))
            elif t == "image":
                images.append((idx, self._decode_image(str(v))))
        embeddings: list[list[float]] = [[] for _ in records]
        with torch.inference_mode():
            if texts:
                idxs, values = zip(*texts)
                inp = self.processor(text=list(values), return_tensors="pt", padding=True, truncation=True)
                inp = {k: v.to(self.device) for k, v in inp.items()}
                feats = self.model.get_text_features(**inp)
                feats = (feats / feats.norm(dim=-1, keepdim=True)).cpu().numpy()
                for j, idx in enumerate(idxs):
                    embeddings[idx] = feats[j].tolist()
            if images:
                idxs, values = zip(*images)
                inp = self.processor(images=list(values), return_tensors="pt")
                inp = {k: v.to(self.device) for k, v in inp.items()}
                feats = self.model.get_image_features(**inp)
                feats = (feats / feats.norm(dim=-1, keepdim=True)).cpu().numpy()
                for j, idx in enumerate(idxs):
                    embeddings[idx] = feats[j].tolist()
        return embeddings


input_ex = pd.DataFrame([{"type": "text", "value": "hello"}, {"type": "image", "value": _tiny_b64}])
sig = infer_signature(input_ex, [[0.0] * 768, [0.0] * 768])

with mlflow.start_run(run_name="clip-register"):
    mlflow.pyfunc.log_model("clip-pyfunc", python_model=CLIPEmbeddingModel(), signature=sig, input_example=input_ex,
                            pip_requirements=["mlflow", "transformers==4.46.3", "pillow", "torch"], registered_model_name=UC_NAME_CLIP)
print(f"✓ CLIP registered to {UC_NAME_CLIP}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### YOLOS (closed-vocab object detection)

# COMMAND ----------

from transformers import AutoImageProcessor, YolosForObjectDetection

HF_MODEL_YOLOS = "hustvl/yolos-small"
UC_NAME_YOLOS = f"{UC_CATALOG}.{UC_SCHEMA}.yolos"


class YolosModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        self.processor = AutoImageProcessor.from_pretrained(HF_MODEL_YOLOS)
        self.model = YolosForObjectDetection.from_pretrained(HF_MODEL_YOLOS)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()
        self.id2label = self.model.config.id2label

    def _decode_image(self, value: str) -> Image.Image:
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> list[dict]:
        records = model_input.to_dict(orient="records") if isinstance(model_input, pd.DataFrame) else list(model_input)
        outputs = []
        with torch.inference_mode():
            for r in records:
                image = self._decode_image(str(r["image"]))
                w, h = image.size
                threshold = float(r.get("threshold") or 0.3)
                inp = self.processor(images=image, return_tensors="pt").to(self.device)
                raw = self.model(**inp)
                res = self.processor.post_process_object_detection(raw, threshold=threshold, target_sizes=torch.tensor([(h, w)]))[0]
                boxes = res["boxes"].detach().cpu().numpy()
                scores = res["scores"].detach().cpu().numpy()
                labels = res["labels"].detach().cpu().numpy()
                outputs.append({"boxes": boxes.tolist(), "scores": scores.tolist(),
                                "labels": [self.id2label.get(int(l), str(int(l))) for l in labels],
                                "count": len(boxes), "image_size": [w, h]})
        return outputs


input_ex = pd.DataFrame([{"image": _tiny_b64, "threshold": 0.3}])
sig = infer_signature(input_ex, [{"boxes": [[0.0]*4], "scores": [0.9], "labels": ["person"], "count": 1, "image_size": [4, 4]}])

with mlflow.start_run(run_name="yolos-register"):
    mlflow.pyfunc.log_model("yolos-pyfunc", python_model=YolosModel(), signature=sig, input_example=input_ex,
                            pip_requirements=["mlflow", "transformers==4.46.3", "pillow", "torch"], registered_model_name=UC_NAME_YOLOS)
print(f"✓ YOLOS registered to {UC_NAME_YOLOS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grounding DINO (open-vocab object detection)

# COMMAND ----------

from transformers import AutoProcessor, GroundingDinoForObjectDetection

HF_MODEL_GDINO = "IDEA-Research/grounding-dino-base"
UC_NAME_GDINO = f"{UC_CATALOG}.{UC_SCHEMA}.grounding_dino"


class GroundingDinoModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        self.processor = AutoProcessor.from_pretrained(HF_MODEL_GDINO)
        self.model = GroundingDinoForObjectDetection.from_pretrained(HF_MODEL_GDINO)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()

    def _decode_image(self, value: str) -> Image.Image:
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> list[dict]:
        records = model_input.to_dict(orient="records") if isinstance(model_input, pd.DataFrame) else list(model_input)
        outputs = []
        with torch.inference_mode():
            for r in records:
                image = self._decode_image(str(r["image"]))
                w, h = image.size
                text = (r.get("text_prompt") or "object .").strip()
                concepts = [c.strip() for c in text.replace(",", ".").split(".") if c.strip()]
                normalized = ". ".join(concepts) + "."
                threshold = float(r.get("threshold") or 0.3)
                inp = self.processor(images=image, text=normalized, return_tensors="pt").to(self.device)
                raw = self.model(**inp)
                res = self.processor.post_process_grounded_object_detection(
                    raw, inp.input_ids, box_threshold=threshold, text_threshold=0.25, target_sizes=[(h, w)])[0]
                boxes = res["boxes"].detach().cpu().numpy() if res.get("boxes") is not None and len(res["boxes"]) else np.empty((0, 4))
                scores = res["scores"].detach().cpu().numpy() if res.get("scores") is not None and len(res["scores"]) else np.empty((0,))
                labels = res.get("text_labels") or res.get("labels") or []
                outputs.append({"boxes": boxes.tolist(), "scores": scores.tolist(),
                                "labels": [str(l) for l in labels], "count": len(boxes), "image_size": [w, h]})
        return outputs


input_ex = pd.DataFrame([{"image": _tiny_b64, "text_prompt": "red square.", "threshold": 0.3}])
sig = infer_signature(input_ex, [{"boxes": [[0.0]*4], "scores": [0.9], "labels": ["red square"], "count": 1, "image_size": [4, 4]}])

with mlflow.start_run(run_name="grounding-dino-register"):
    mlflow.pyfunc.log_model("grounding-dino-pyfunc", python_model=GroundingDinoModel(), signature=sig, input_example=input_ex,
                            pip_requirements=["mlflow", "transformers==4.46.3", "pillow", "torch"], registered_model_name=UC_NAME_GDINO)
print(f"✓ Grounding DINO registered to {UC_NAME_GDINO}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Depth Anything V2 (monocular depth estimation)

# COMMAND ----------

from transformers import AutoModelForDepthEstimation

HF_MODEL_DEPTH = "depth-anything/Depth-Anything-V2-Large-hf"
UC_NAME_DEPTH = f"{UC_CATALOG}.{UC_SCHEMA}.depth_anything"


class DepthAnythingModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        self.processor = AutoImageProcessor.from_pretrained(HF_MODEL_DEPTH)
        self.model = AutoModelForDepthEstimation.from_pretrained(HF_MODEL_DEPTH)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()

    def _decode_image(self, value: str) -> Image.Image:
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> list[dict]:
        records = model_input.to_dict(orient="records") if isinstance(model_input, pd.DataFrame) else list(model_input)
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
                img = Image.fromarray(norm.astype(np.uint8), mode="L")
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                outputs.append({"depth_png": base64.b64encode(buf.getvalue()).decode("ascii"),
                                "min_depth": d_min, "max_depth": d_max, "image_size": [w, h]})
        return outputs


input_ex = pd.DataFrame([{"image": _tiny_b64}])
sig = infer_signature(input_ex, [{"depth_png": "<b64>", "min_depth": 0.0, "max_depth": 1.0, "image_size": [4, 4]}])

with mlflow.start_run(run_name="depth-anything-register"):
    mlflow.pyfunc.log_model("depth-anything-pyfunc", python_model=DepthAnythingModel(), signature=sig, input_example=input_ex,
                            pip_requirements=["mlflow", "transformers==4.46.3", "pillow", "torch"], registered_model_name=UC_NAME_DEPTH)
print(f"✓ Depth Anything registered to {UC_NAME_DEPTH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### SAM 3 (optional — promptable segmentation)
# MAGIC
# MAGIC Skipped if `HF_TOKEN_SCOPE` is empty. To enable, set it in the Configuration cell above.

# COMMAND ----------

if HF_TOKEN_SCOPE and HF_TOKEN_KEY:
    HF_TOKEN = dbutils.secrets.get(scope=HF_TOKEN_SCOPE, key=HF_TOKEN_KEY).strip()  # noqa: F821
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

    import subprocess
    subprocess.check_call(["pip", "install", "-q", "transformers==5.8.0", "huggingface_hub>=1.0,<2.0", "accelerate>=0.34.0"])

    from transformers import Sam3Model, Sam3Processor

    HF_MODEL_SAM3 = "facebook/sam3"
    UC_NAME_SAM3 = f"{UC_CATALOG}.{UC_SCHEMA}.sam3"

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
            records = model_input.to_dict(orient="records") if isinstance(model_input, pd.DataFrame) else list(model_input)
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
                    masks, boxes, scores = result.get("masks"), result.get("boxes"), result.get("scores")
                    mask_list, box_list, score_list = [], [], []
                    if masks is not None and len(masks) > 0:
                        for m, b, s in zip(masks.cpu().numpy(), boxes.cpu().numpy(), scores.cpu().numpy()):
                            m2 = np.squeeze(m)
                            img = Image.fromarray((m2 > 0).astype(np.uint8) * 255, mode="L")
                            buf = io.BytesIO()
                            img.save(buf, format="PNG", optimize=True)
                            mask_list.append(base64.b64encode(buf.getvalue()).decode("ascii"))
                            box_list.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                            score_list.append(float(s))
                    outputs.append({"masks": mask_list, "boxes": box_list, "scores": score_list,
                                    "count": len(mask_list), "image_size": [w, h]})
            return outputs

    input_ex = pd.DataFrame([{"image": _tiny_b64, "text_prompt": "object", "threshold": 0.5, "mask_threshold": 0.5}])
    sig = infer_signature(input_ex, [{"masks": ["<b64>"], "boxes": [[0.0]*4], "scores": [0.9], "count": 1, "image_size": [4, 4]}])

    with mlflow.start_run(run_name="sam3-register"):
        mlflow.pyfunc.log_model("sam3-pyfunc", python_model=Sam3SegmenterModel(), signature=sig, input_example=input_ex,
                                pip_requirements=["mlflow", "transformers==5.8.0", "huggingface_hub>=1.0,<2.0", "accelerate>=0.34.0", "pillow", "torch"],
                                registered_model_name=UC_NAME_SAM3)
    print(f"✓ SAM 3 registered to {UC_NAME_SAM3}")
else:
    print("⏭ SAM 3 skipped (no HF token configured)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Serving Endpoints

# COMMAND ----------

import requests

host = spark.conf.get("spark.databricks.workspaceUrl")  # noqa: F821
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()  # noqa: F821
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

ENDPOINTS = [
    {"name": "clip-vit-large-patch14", "entity": f"{UC_CATALOG}.{UC_SCHEMA}.clip_vit_large_patch14", "gpu": "GPU_MEDIUM"},
    {"name": "yolos", "entity": f"{UC_CATALOG}.{UC_SCHEMA}.yolos", "gpu": "GPU_SMALL"},
    {"name": "grounding-dino", "entity": f"{UC_CATALOG}.{UC_SCHEMA}.grounding_dino", "gpu": "GPU_SMALL"},
    {"name": "depth-anything", "entity": f"{UC_CATALOG}.{UC_SCHEMA}.depth_anything", "gpu": "GPU_SMALL"},
]

if HF_TOKEN_SCOPE and HF_TOKEN_KEY:
    ENDPOINTS.append({
        "name": "sam3", "entity": f"{UC_CATALOG}.{UC_SCHEMA}.sam3", "gpu": "GPU_MEDIUM",
        "env_vars": {"HF_TOKEN": f"{{{{secrets/{HF_TOKEN_SCOPE}/{HF_TOKEN_KEY}}}}}",
                     "HUGGINGFACE_HUB_TOKEN": f"{{{{secrets/{HF_TOKEN_SCOPE}/{HF_TOKEN_KEY}}}}}"}
    })

for ep in ENDPOINTS:
    served_entity = {
        "name": ep["name"], "entity_name": ep["entity"], "entity_version": "1",
        "workload_type": ep["gpu"], "workload_size": "Small", "scale_to_zero_enabled": True,
    }
    if "env_vars" in ep:
        served_entity["environment_vars"] = ep["env_vars"]

    payload = {
        "name": ep["name"],
        "config": {
            "served_entities": [served_entity],
            "traffic_config": {"routes": [{"served_model_name": ep["name"], "traffic_percentage": 100}]},
        },
    }
    resp = requests.post(f"https://{host}/api/2.0/serving-endpoints", headers=headers, json=payload)
    if resp.status_code == 200:
        print(f"✓ Endpoint '{ep['name']}' created (will be ready in 3–10 min)")
    elif resp.status_code == 409 or "already exists" in resp.text.lower():
        print(f"⏭ Endpoint '{ep['name']}' already exists")
    else:
        print(f"✗ Endpoint '{ep['name']}' failed: {resp.status_code} — {resp.text[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Deploy the App
# MAGIC
# MAGIC From your local terminal:
# MAGIC ```
# MAGIC cd model-workbench
# MAGIC databricks apps deploy model-workbench
# MAGIC ```
# MAGIC
# MAGIC The app auto-discovers all serving endpoints in your workspace. Custom endpoints
# MAGIC will show as "Not Ready" until cold start completes. Foundation Model APIs appear immediately.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✓ Done!
# MAGIC
# MAGIC - **Models registered** to `{UC_CATALOG}.{UC_SCHEMA}`
# MAGIC - **Endpoints** provisioning with scale-to-zero (3–10 min cold start)
# MAGIC - **Next**: run `databricks apps deploy model-workbench` locally to deploy the UI
