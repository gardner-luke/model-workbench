# YOLOS endpoint

[`hustvl/yolos-small`](https://huggingface.co/hustvl/yolos-small) — a
transformer-based YOLO trained on the COCO 80-class dataset, deployed as a
custom Databricks Model Serving endpoint.

**Closed vocabulary**: there's no text prompt. You give it an image, it
returns bounding boxes for every person, car, bottle, etc. it recognizes from
its training classes. This is the "fast detector for known objects" baseline.

> Note: this is not Ultralytics YOLOv8. That model is AGPL-licensed which is
> a problem for customer-shareable examples. YOLOS is Apache 2.0 and lives
> natively in the `transformers` library, which keeps the wrapper pattern
> consistent with everything else in this repo. If you specifically need
> Ultralytics weights, swap them in here with the same wrapper structure.

## What this directory contains

| File | Purpose |
|---|---|
| `register_yolos.py` | Databricks notebook that logs the PyFunc to Unity Catalog. |
| `README.md` | This file — the endpoint contract. |

## Endpoint contract

### Request

```json
POST /serving-endpoints/yolos/invocations
{
  "dataframe_records": [{
    "image": "<base64-encoded JPEG or PNG>",
    "threshold": 0.3
  }]
}
```

No `text_prompt`. The model detects from a fixed list of 80 COCO classes.

### Response

```json
{
  "predictions": [{
    "boxes": [[x1, y1, x2, y2], ...],
    "scores": [0.91, 0.87, ...],
    "labels": ["person", "car", ...],
    "count": 3,
    "image_size": [width, height]
  }]
}
```

`labels` are human-readable COCO class names mapped from the model's
`config.id2label` table.

## Setup

1. Edit the configuration block at the top of `register_yolos.py`.
2. Run the notebook.
3. Create the serving endpoint:
   ```sh
   databricks serving-endpoints create --json '{
     "name": "yolos",
     "config": {
       "served_entities": [{
         "name": "yolos",
         "entity_name": "<your_catalog>.<your_schema>.yolos",
         "entity_version": "1",
         "workload_type": "GPU_SMALL",
         "workload_size": "Small",
         "scale_to_zero_enabled": true
       }],
       "traffic_config": {
         "routes": [{"served_model_name": "yolos", "traffic_percentage": 100}]
       }
     }
   }'
   ```

## Compute notes

- **`GPU_SMALL` (T4 16GB)** is overkill — yolos-small is ~120 MB. CPU also
  works, but GPU inference is sub-100ms per image.
- Cold start ~3 min.

## Quirk to expect when demoing this

YOLOS can only label what it was trained on. If you feed it images of things
outside COCO's 80 classes, expect "creative" predictions. Our synthetic test
image (a yellow circle on a blue square) confidently came back as a "frisbee"
inside a "tv". This is the right outcome to show a customer — it explains
*why* open-vocab models like Grounding DINO matter for non-COCO domains.

## When to use this vs other detection models in this repo

| Need | Pick |
|---|---|
| Speed and simplicity, COCO-domain images | **YOLOS** |
| Arbitrary concepts described in text | Grounding DINO |
| Precise pixel masks | SAM 3 |
