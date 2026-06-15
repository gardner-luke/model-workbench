# CLIP endpoint

OpenAI's [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14)
deployed as a custom Databricks Model Serving endpoint. CLIP embeds text and
images into the same 768-dimensional vector space — useful for cross-modal
search ("find frames matching 'tractor at field edge'"), zero-shot
classification, and content tagging.

## What this directory contains

| File | Purpose |
|---|---|
| `register_clip.py` | Databricks notebook. Defines an MLflow PyFunc wrapper, downloads CLIP weights from HuggingFace, and registers the model to Unity Catalog. |
| `README.md` | This file — the contract the endpoint speaks. |

## Endpoint contract

### Request

```json
POST /serving-endpoints/clip-vit-large-patch14/invocations
{
  "dataframe_records": [
    {"type": "text",  "value": "a yellow circle on a blue square"},
    {"type": "image", "value": "<base64-encoded JPEG or PNG>"}
  ]
}
```

Each row is one input. `type` is either `"text"` or `"image"`. You can mix them
freely in one call. The wrapper splits them by modality, batches each through
CLIP's text or vision tower, and reassembles the result in the original input
order.

### Response

```json
{
  "predictions": {
    "embeddings": [
      [0.012, -0.045, 0.078, ...],   // 768 floats
      [0.023, -0.011, 0.094, ...]
    ],
    "dim": 768
  }
}
```

All vectors are **L2-normalized**, so the raw dot product equals cosine
similarity. Values in CLIP's shared space typically run lower than pure
text-text or image-image — a confident match between an image and a matching
caption is around 0.25–0.35, not 0.9+.

## Setup

1. Edit the configuration block at the top of `register_clip.py` to point at
   your UC catalog/schema.
2. Run the notebook in your Databricks workspace. It will:
   - Download CLIP weights from HuggingFace
   - Wrap them in a custom `mlflow.pyfunc.PythonModel`
   - Log + register to `${UC_CATALOG}.${UC_SCHEMA}.${UC_MODEL}`
3. Create the serving endpoint pointing at the registered model:
   ```sh
   databricks serving-endpoints create --json '{
     "name": "clip-vit-large-patch14",
     "config": {
       "served_entities": [{
         "name": "clip",
         "entity_name": "<your_catalog>.<your_schema>.clip_vit_large_patch14",
         "entity_version": "1",
         "workload_type": "GPU_MEDIUM",
         "workload_size": "Small",
         "scale_to_zero_enabled": true
       }],
       "traffic_config": {
         "routes": [{"served_model_name": "clip", "traffic_percentage": 100}]
       }
     }
   }'
   ```
4. (Optional) Enable inference tables to log every request/response to UC:
   ```sh
   databricks serving-endpoints put-ai-gateway clip-vit-large-patch14 --json '{
     "usage_tracking_config": {"enabled": true},
     "inference_table_config": {
       "enabled": true,
       "catalog_name": "<your_catalog>",
       "schema_name": "<your_schema>",
       "table_name_prefix": "clip_inference"
     }
   }'
   ```

## Compute notes

- **`GPU_MEDIUM` (A10G 24GB)** is generous for CLIP ViT-L/14 (~1.7GB on disk).
  `GPU_SMALL` (T4 16GB) also works.
- **Scale-to-zero** is safe — cold start is ~3–5 minutes (image download + weight
  load). Subsequent calls are sub-second.
- The Model Workbench UI downsamples images to ≤768px on the longest edge before
  sending, since CLIP processes at 224×224 internally and you don't need more.

## How the Model Workbench uses this endpoint

The app's `Embeddings` page lets the user mix text and image inputs, compute
embeddings for each, and visualize the NxN cosine similarity matrix. Because
the vectors live in a shared space, you can directly compare text↔image and
image↔image with the same metric.
