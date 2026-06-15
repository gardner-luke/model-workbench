# Grounding DINO endpoint

[IDEA Research's Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base)
deployed as a custom Databricks Model Serving endpoint. Grounding DINO does
**open-vocabulary object detection**: you give it any noun phrase ("tractor",
"yellow circle", "industrial pipe") and it returns bounding boxes for every
matching instance. No retraining, no fixed class list.

Versus SAM 3 (which also takes text prompts): Grounding DINO returns boxes
only, no masks. Faster and cheaper when "where is the thing" is enough. The
two pair well — Grounding DINO → SAM 3 is the standard "Grounded SAM" auto-
labeling pipeline.

## What this directory contains

| File | Purpose |
|---|---|
| `register_grounding_dino.py` | Databricks notebook that logs the PyFunc to Unity Catalog. |
| `README.md` | This file — the endpoint contract. |

## Endpoint contract

### Request

```json
POST /serving-endpoints/grounding-dino/invocations
{
  "dataframe_records": [{
    "image": "<base64-encoded JPEG or PNG>",
    "text_prompt": "tractor. person. wheel.",
    "threshold": 0.3
  }]
}
```

Concepts must be separated by periods — this is Grounding DINO's training
convention, not arbitrary. The wrapper normalizes commas and other separators
to periods if you forget.

### Response

```json
{
  "predictions": [{
    "boxes": [[x1, y1, x2, y2], ...],
    "scores": [0.91, 0.87, ...],
    "labels": ["tractor", "person", ...],
    "count": 3,
    "image_size": [width, height]
  }]
}
```

One prediction record per input row (MLflow-idiomatic shape — works with
`ai_query` for batch inference). `labels` records which concept from your
prompt matched each box.

## Setup

1. Edit the configuration block at the top of `register_grounding_dino.py` to
   point at your UC catalog/schema. No HF token needed — Grounding DINO is
   not gated.
2. Run the notebook in your workspace.
3. Create the serving endpoint:
   ```sh
   databricks serving-endpoints create --json '{
     "name": "grounding-dino",
     "config": {
       "served_entities": [{
         "name": "grounding-dino",
         "entity_name": "<your_catalog>.<your_schema>.grounding_dino",
         "entity_version": "1",
         "workload_type": "GPU_SMALL",
         "workload_size": "Small",
         "scale_to_zero_enabled": true
       }],
       "traffic_config": {
         "routes": [{"served_model_name": "grounding-dino", "traffic_percentage": 100}]
       }
     }
   }'
   ```
4. (Optional) Enable inference tables to log every request/response to UC.

## Compute notes

- **`GPU_SMALL` (T4 16GB)** is enough — Grounding DINO base is ~700 MB.
- Cold start ~5 min after scale-to-zero. Subsequent calls sub-second.

## When to use this vs other detection models in this repo

| Need | Pick |
|---|---|
| "Where is this specific concept (in plain English)?" | **Grounding DINO** |
| "Tell me everything you see from COCO's 80 classes." | YOLOS (closed vocab, faster) |
| "Where is it AND give me a precise pixel mask." | SAM 3 |
