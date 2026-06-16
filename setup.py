# Databricks notebook source
# MAGIC %md
# MAGIC # Model Workbench — Setup
# MAGIC
# MAGIC This notebook deploys everything end-to-end:
# MAGIC 1. Creates a Unity Catalog schema for custom models
# MAGIC 2. Registers 5 models (CLIP, YOLOS, Grounding DINO, Depth Anything, and optionally SAM 3)
# MAGIC 3. Creates GPU serving endpoints with scale-to-zero
# MAGIC 4. Creates and deploys the Databricks App
# MAGIC 5. Grants the app's service principal access to query all endpoints
# MAGIC
# MAGIC **Fill in the widgets at the top, then Run All.** Total time: ~15–20 min.

# COMMAND ----------

# DBTITLE 1,Configuration (edit these widgets then Run All)
dbutils.widgets.text("uc_catalog", "", "Unity Catalog Name")
dbutils.widgets.text("hf_token_scope", "", "HF Secret Scope (for SAM 3, leave blank to skip)")
dbutils.widgets.text("hf_token_key", "", "HF Secret Key (for SAM 3, leave blank to skip)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Validate & Create Schema

# COMMAND ----------

UC_CATALOG = dbutils.widgets.get("uc_catalog").strip()
UC_SCHEMA = "model_workbench"
HF_TOKEN_SCOPE = dbutils.widgets.get("hf_token_scope").strip()
HF_TOKEN_KEY = dbutils.widgets.get("hf_token_key").strip()

assert UC_CATALOG, "Set the 'uc_catalog' widget to your Unity Catalog name before running"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")
print(f"✓ Schema ready: {UC_CATALOG}.{UC_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Install Dependencies & Register Models

# COMMAND ----------

# MAGIC %pip install -q -U mlflow transformers==4.46.3 pillow torch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Recover config from widgets after restartPython
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
from transformers import CLIPModel, CLIPProcessor, AutoImageProcessor, YolosForObjectDetection, AutoProcessor, GroundingDinoForObjectDetection, AutoModelForDepthEstimation

UC_CATALOG = dbutils.widgets.get("uc_catalog").strip()
UC_SCHEMA = "model_workbench"
HF_TOKEN_SCOPE = dbutils.widgets.get("hf_token_scope").strip()
HF_TOKEN_KEY = dbutils.widgets.get("hf_token_key").strip()

mlflow.set_registry_uri("databricks-uc")

_buf = io.BytesIO()
Image.new("RGB", (4, 4), color=(255, 0, 0)).save(_buf, format="PNG")
TINY_IMG = base64.b64encode(_buf.getvalue()).decode("ascii")

print(f"Config: {UC_CATALOG}.{UC_SCHEMA} | SAM 3: {'enabled' if HF_TOKEN_SCOPE else 'skipped'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### CLIP (text + image embeddings)

# COMMAND ----------

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


_in = pd.DataFrame([{"type": "text", "value": "hello"}, {"type": "image", "value": TINY_IMG}])
_sig = infer_signature(_in, [[0.0] * 768, [0.0] * 768])
with mlflow.start_run(run_name="clip-register"):
    mlflow.pyfunc.log_model("model", python_model=CLIPEmbeddingModel(), signature=_sig, input_example=_in,
                            pip_requirements=["mlflow", "transformers==4.46.3", "pillow", "torch"],
                            registered_model_name=UC_NAME_CLIP)
print(f"✓ CLIP → {UC_NAME_CLIP}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### YOLOS (closed-vocab object detection)

# COMMAND ----------

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


_in = pd.DataFrame([{"image": TINY_IMG, "threshold": 0.3}])
_sig = infer_signature(_in, [{"boxes": [[0.0]*4], "scores": [0.9], "labels": ["person"], "count": 1, "image_size": [4, 4]}])
with mlflow.start_run(run_name="yolos-register"):
    mlflow.pyfunc.log_model("model", python_model=YolosModel(), signature=_sig, input_example=_in,
                            pip_requirements=["mlflow", "transformers==4.46.3", "pillow", "torch"],
                            registered_model_name=UC_NAME_YOLOS)
print(f"✓ YOLOS → {UC_NAME_YOLOS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grounding DINO (open-vocab object detection)

# COMMAND ----------

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
        records = model_input.to_dict(orient="records")
        outputs = []
        with torch.inference_mode():
            for r in records:
                image = self._decode_image(str(r["image"]))
                w, h = image.size
                text = (r.get("text_prompt") or "object.").strip()
                concepts = [c.strip() for c in text.replace(",", ".").split(".") if c.strip()]
                normalized = ". ".join(concepts) + "."
                threshold = float(r.get("threshold") or 0.3)
                inp = self.processor(images=image, text=normalized, return_tensors="pt").to(self.device)
                raw = self.model(**inp)
                res = self.processor.post_process_grounded_object_detection(
                    raw, inp.input_ids, box_threshold=threshold, text_threshold=0.25, target_sizes=[(h, w)])[0]
                boxes = res["boxes"].cpu().numpy() if len(res.get("boxes", [])) else np.empty((0, 4))
                scores = res["scores"].cpu().numpy() if len(res.get("scores", [])) else np.empty((0,))
                labels = res.get("text_labels") or res.get("labels") or []
                outputs.append({
                    "boxes": boxes.tolist(), "scores": scores.tolist(),
                    "labels": [str(l) for l in labels], "count": len(boxes), "image_size": [w, h]
                })
        return outputs


_in = pd.DataFrame([{"image": TINY_IMG, "text_prompt": "red square.", "threshold": 0.3}])
_sig = infer_signature(_in, [{"boxes": [[0.0]*4], "scores": [0.9], "labels": ["red square"], "count": 1, "image_size": [4, 4]}])
with mlflow.start_run(run_name="grounding-dino-register"):
    mlflow.pyfunc.log_model("model", python_model=GroundingDinoModel(), signature=_sig, input_example=_in,
                            pip_requirements=["mlflow", "transformers==4.46.3", "pillow", "torch"],
                            registered_model_name=UC_NAME_GDINO)
print(f"✓ Grounding DINO → {UC_NAME_GDINO}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Depth Anything V2 (monocular depth estimation)

# COMMAND ----------

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


_in = pd.DataFrame([{"image": TINY_IMG}])
_sig = infer_signature(_in, [{"depth_png": "x", "min_depth": 0.0, "max_depth": 1.0, "image_size": [4, 4]}])
with mlflow.start_run(run_name="depth-anything-register"):
    mlflow.pyfunc.log_model("model", python_model=DepthAnythingModel(), signature=_sig, input_example=_in,
                            pip_requirements=["mlflow", "transformers==4.46.3", "pillow", "torch"],
                            registered_model_name=UC_NAME_DEPTH)
print(f"✓ Depth Anything → {UC_NAME_DEPTH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### SAM 3 (optional — requires gated HuggingFace access)
# MAGIC
# MAGIC SAM 3 needs `transformers>=5.8.0` which conflicts with 4.46.3 used above.
# MAGIC The pip install and restart always run (they're fast); the actual registration
# MAGIC is skipped if no HF token is configured.

# COMMAND ----------

# MAGIC %pip install -q -U "transformers==5.8.0" "huggingface_hub>=1.0,<2.0" "accelerate>=0.34.0" mlflow pillow torch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Register SAM 3 (skipped if no HF token)
import os
HF_TOKEN_SCOPE = dbutils.widgets.get("hf_token_scope").strip()
HF_TOKEN_KEY = dbutils.widgets.get("hf_token_key").strip()

if HF_TOKEN_SCOPE and HF_TOKEN_KEY:
    import base64
    import io
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
    UC_NAME_SAM3 = f"{UC_CATALOG}.{UC_SCHEMA}.sam3"

    HF_TOKEN = dbutils.secrets.get(scope=HF_TOKEN_SCOPE, key=HF_TOKEN_KEY).strip()
    assert HF_TOKEN and HF_TOKEN.isascii(), "HF token is empty or invalid — check your secret"
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

    HF_MODEL_SAM3 = "facebook/sam3"

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

    mlflow.set_registry_uri("databricks-uc")
    _buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(_buf, format="PNG")
    TINY_IMG = base64.b64encode(_buf.getvalue()).decode("ascii")

    _in = pd.DataFrame([{"image": TINY_IMG, "text_prompt": "object", "threshold": 0.5, "mask_threshold": 0.5}])
    _sig = infer_signature(_in, [{"masks": ["x"], "boxes": [[0.0]*4], "scores": [0.9], "count": 1, "image_size": [4, 4]}])
    with mlflow.start_run(run_name="sam3-register"):
        mlflow.pyfunc.log_model("model", python_model=Sam3SegmenterModel(), signature=_sig, input_example=_in,
                                pip_requirements=["mlflow", "transformers==5.8.0", "huggingface_hub>=1.0,<2.0", "accelerate>=0.34.0", "pillow", "torch"],
                                registered_model_name=UC_NAME_SAM3)
    print(f"✓ SAM 3 → {UC_NAME_SAM3}")
else:
    print("⏭ SAM 3 skipped (no HF token scope/key provided)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Serving Endpoints

# COMMAND ----------

import requests
import time

UC_CATALOG = dbutils.widgets.get("uc_catalog").strip()
UC_SCHEMA = "model_workbench"
HF_TOKEN_SCOPE = dbutils.widgets.get("hf_token_scope").strip()
HF_TOKEN_KEY = dbutils.widgets.get("hf_token_key").strip()

HOST = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

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

endpoint_ids = {}
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
    resp = requests.post(f"{HOST}/api/2.0/serving-endpoints", headers=HEADERS, json=payload)
    if resp.status_code == 200:
        endpoint_ids[ep["name"]] = resp.json().get("id")
        print(f"✓ Endpoint '{ep['name']}' created")
    elif resp.status_code == 409 or "already exists" in resp.text.lower():
        r2 = requests.get(f"{HOST}/api/2.0/serving-endpoints/{ep['name']}", headers=HEADERS)
        if r2.status_code == 200:
            endpoint_ids[ep["name"]] = r2.json().get("id")
        print(f"⏭ Endpoint '{ep['name']}' already exists")
    else:
        print(f"✗ '{ep['name']}' failed ({resp.status_code}): {resp.text[:200]}")

print(f"\n{len(endpoint_ids)} endpoints ready or creating")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create & Deploy the App

# COMMAND ----------

# DBTITLE 1,Create the app
import json

APP_NAME = "model-workbench"
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
source_path = "/".join(notebook_path.split("/")[:-1])

resp = requests.get(f"{HOST}/api/2.0/apps/{APP_NAME}", headers=HEADERS)
if resp.status_code == 200:
    app = resp.json()
    print(f"⏭ App '{APP_NAME}' already exists")
else:
    payload = {"name": APP_NAME, "description": "Model Workbench — explore every model deployed in your Databricks workspace"}
    resp = requests.post(f"{HOST}/api/2.0/apps", headers=HEADERS, json=payload)
    resp.raise_for_status()
    app = resp.json()
    print(f"✓ App '{APP_NAME}' created")
    time.sleep(5)
    resp = requests.get(f"{HOST}/api/2.0/apps/{APP_NAME}", headers=HEADERS)
    app = resp.json()

sp_client_id = app.get("service_principal_client_id")
print(f"  App SP: {app.get('service_principal_name', sp_client_id)}")

# COMMAND ----------

# DBTITLE 1,Grant CAN_QUERY on each serving endpoint to the app SP
for ep_name, ep_id in endpoint_ids.items():
    if not ep_id or not sp_client_id:
        continue
    perm_payload = {"access_control_list": [
        {"service_principal_name": sp_client_id, "permission_level": "CAN_QUERY"}
    ]}
    resp = requests.patch(f"{HOST}/api/2.0/permissions/serving-endpoints/{ep_id}", headers=HEADERS, json=perm_payload)
    if resp.status_code == 200:
        print(f"✓ CAN_QUERY on '{ep_name}'")
    else:
        print(f"⚠ Permission for '{ep_name}': {resp.text[:120]}")

# COMMAND ----------

# DBTITLE 1,Deploy the app from workspace files
deploy_payload = {"source_code_path": source_path}
resp = requests.post(f"{HOST}/api/2.0/apps/{APP_NAME}/deployments", headers=HEADERS, json=deploy_payload)
if resp.status_code == 200:
    deployment = resp.json()
    print(f"✓ Deployment started: {deployment.get('deployment_id', '')}")
    print(f"  Source: {source_path}")
else:
    print(f"Deployment response ({resp.status_code}): {resp.text[:300]}")

print("\nWaiting for app to start...")
for i in range(60):
    time.sleep(10)
    resp = requests.get(f"{HOST}/api/2.0/apps/{APP_NAME}", headers=HEADERS)
    if resp.status_code == 200:
        status = resp.json().get("app_status", {})
        state = status.get("state", "")
        if state == "RUNNING":
            print(f"\n✓ App is live: {resp.json().get('url', '')}")
            break
        elif state in ("FAILED", "CRASHED"):
            print(f"\n✗ App failed: {status.get('message', '')}")
            break
        elif i % 3 == 0:
            print(f"  ... {state}")
else:
    print("\n⚠ Timed out — check app status in workspace UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC Your Model Workbench is deployed. The app auto-discovers all serving endpoints
# MAGIC in your workspace — the custom models above plus any Foundation Model APIs.
# MAGIC
# MAGIC Custom endpoints use scale-to-zero. First request after cold start takes 3–10 min to warm up.
