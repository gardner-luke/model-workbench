# SAM 3 endpoint

Meta's [Segment Anything Model 3](https://huggingface.co/facebook/sam3) deployed
as a custom Databricks Model Serving endpoint. SAM 3 does *promptable concept
segmentation*: you give it an image and a noun phrase ("corn kernel", "tractor",
"person"), and it returns a mask, bounding box, and confidence score for every
matching instance in the image.

This is what makes SAM 3 different from SAM 2 — you don't need to click points
or draw boxes. Text alone is enough.

## What this directory contains

| File | Purpose |
|---|---|
| `register_sam3.py` | Databricks notebook. Defines an MLflow PyFunc wrapper, downloads SAM 3 weights from HuggingFace, and registers the model to Unity Catalog. |
| `README.md` | This file — the contract the endpoint speaks. |

## Endpoint contract

### Request

```json
POST /serving-endpoints/sam3/invocations
{
  "dataframe_records": [{
    "image": "<base64-encoded JPEG or PNG>",
    "text_prompt": "corn kernel",
    "threshold": 0.5,        // optional, default 0.5 — presence threshold
    "mask_threshold": 0.5    // optional, default 0.5 — per-pixel binarization
  }]
}
```

`text_prompt` is a short noun phrase describing the concept to find. The two
thresholds tune precision-recall:
- `threshold` filters detections by confidence. Lower = more detections, more
  false positives.
- `mask_threshold` controls how strictly each pixel is included in the mask.

### Response

```json
{
  "predictions": {
    "masks": ["<base64 PNG>", "<base64 PNG>", ...],
    "boxes": [[x1, y1, x2, y2], ...],
    "scores": [0.94, 0.87, 0.71, ...],
    "count": 3,
    "image_size": [width, height]
  }
}
```

Each mask is a **1-bit PNG** at the same dimensions as the input image —
small over the wire, easy to overlay on a canvas. Boxes are `xyxy` in absolute
pixel coordinates. Scores are 0–1.

## Setup

1. **Request HuggingFace access** to `facebook/sam3`. Visit the model card and
   click "Agree and access repository". Approval is usually quick.
2. **Save your HF token** as a Databricks secret. Generate a read token at
   https://huggingface.co/settings/tokens then:
   ```sh
   databricks secrets put-secret <your-scope> <your-key> --string-value 'hf_xxx...'
   ```
   Use **single quotes** so the shell doesn't normalize special characters in
   the token.
3. Edit the configuration block at the top of `register_sam3.py` to point at
   your UC catalog/schema and your HF secret scope/key.
4. Run the notebook. It will:
   - Read the HF token from your secret
   - Download SAM 3 weights from HuggingFace
   - Wrap them in a custom `mlflow.pyfunc.PythonModel`
   - Log + register to `${UC_CATALOG}.${UC_SCHEMA}.${UC_MODEL}`
5. Create the serving endpoint. The endpoint must receive the HF token via
   `environment_vars` so it can re-download the weights at cold start:
   ```sh
   databricks serving-endpoints create --json '{
     "name": "sam3",
     "config": {
       "served_entities": [{
         "name": "sam3",
         "entity_name": "<your_catalog>.<your_schema>.sam3",
         "entity_version": "1",
         "workload_type": "GPU_MEDIUM",
         "workload_size": "Small",
         "scale_to_zero_enabled": true,
         "environment_vars": {
           "HF_TOKEN": "{{secrets/<your-scope>/<your-key>}}",
           "HUGGINGFACE_HUB_TOKEN": "{{secrets/<your-scope>/<your-key>}}"
         }
       }],
       "traffic_config": {
         "routes": [{"served_model_name": "sam3", "traffic_percentage": 100}]
       }
     }
   }'
   ```
6. (Optional) Enable inference tables to log every request/response to UC:
   ```sh
   databricks serving-endpoints put-ai-gateway sam3 --json '{
     "usage_tracking_config": {"enabled": true},
     "inference_table_config": {
       "enabled": true,
       "catalog_name": "<your_catalog>",
       "schema_name": "<your_schema>",
       "table_name_prefix": "sam3_inference"
     }
   }'
   ```

## Compute notes

- **`GPU_MEDIUM` (A10G 24GB)** is required. SAM 3 plus its image encoder needs
  more than the 16GB on T4.
- **Cold start** is 10–25 minutes the first time (image build + weight download
  for a gated repo). After scale-to-zero kicks down, subsequent cold starts are
  faster (~5 min) because the container image is cached.
- The Model Workbench UI downsamples images to ≤1536px on the longest edge.
  SAM 3 internally processes at ~1024px patches, so 1536px gives it headroom
  without bloating the payload.

## How the Model Workbench uses this endpoint

The app's `Segmentation` page lets the user upload an image, type a concept
prompt, and see each detected instance rendered as a colored mask overlay with
its bounding box and confidence score.

## A few things I learned the hard way deploying this

1. **Pin transformers exactly.** `transformers>=4.55` resolved to the 5.x major
   version, which mostly works but has surprises. Pinning to a known-good 5.x
   (e.g. `5.8.0`) prevents weeks-of-debugging-in-the-future.
2. **Validate the HF token early.** Strip whitespace, check `isascii()`, fail
   loudly at notebook time. Otherwise you'll see an inscrutable
   `UnicodeEncodeError` from httpx 30 minutes into a deploy.
3. **`facebook/sam3.1` doesn't have root-level safetensors.** Use
   `facebook/sam3`. Both are covered by the same gating group; one just
   loads cleanly with the standard MLflow PyFunc download flow.
4. **Custom endpoint payloads are capped at 16 MiB.** A 4K-resolution image is
   close to the limit base64-encoded. Resize client-side.
