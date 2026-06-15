# Depth Anything V2 endpoint

[`depth-anything/Depth-Anything-V2-Large-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf)
deployed as a custom Databricks Model Serving endpoint.

Monocular depth estimation — single image in, per-pixel depth map out. No
stereo cameras, no LIDAR, no calibration. Strong for:
- Robotics perception (autonomous fleet, sensor fusion)
- 3D scene reconstruction
- AR effects (background blur, depth-aware compositing)
- Annotating field-imagery datasets with spatial context

## What this directory contains

| File | Purpose |
|---|---|
| `register_depth_anything.py` | Databricks notebook that logs the PyFunc to Unity Catalog. |
| `README.md` | This file — the endpoint contract. |

## Endpoint contract

### Request

```json
POST /serving-endpoints/depth-anything/invocations
{
  "dataframe_records": [{"image": "<base64-encoded JPEG or PNG>"}]
}
```

### Response

```json
{
  "predictions": [{
    "depth_png": "<base64 grayscale PNG>",
    "min_depth": 1.23,
    "max_depth": 87.65,
    "image_size": [width, height]
  }]
}
```

The depth map is an **8-bit grayscale PNG** at the same dimensions as the input
image. Brighter pixels = closer to the camera (after the wrapper's per-image
normalization). If you need the raw scale, recover with:

```python
raw = (pixel / 255.0) * (max_depth - min_depth) + min_depth
```

## Setup

1. Edit the configuration block at the top of `register_depth_anything.py` to
   point at your UC catalog/schema. No HF token needed — Depth Anything V2 is
   not gated.
2. Run the notebook.
3. Create the serving endpoint:
   ```sh
   databricks serving-endpoints create --json '{
     "name": "depth-anything",
     "config": {
       "served_entities": [{
         "name": "depth-anything",
         "entity_name": "<your_catalog>.<your_schema>.depth_anything",
         "entity_version": "1",
         "workload_type": "GPU_SMALL",
         "workload_size": "Small",
         "scale_to_zero_enabled": true
       }],
       "traffic_config": {
         "routes": [{"served_model_name": "depth-anything", "traffic_percentage": 100}]
       }
     }
   }'
   ```

## Compute notes

- **`GPU_SMALL` (T4 16GB)** is enough for the Large variant (~1.4 GB on disk).
- Cold start ~5 min. Inference ~1-2 seconds for a 1024×768 image.

## How the Model Workbench uses this endpoint

The `Depth` playground page shows two views of the result:
- **Heatmap** — the raw depth rendered with a turbo colormap. Best for
  inspecting depth structure on its own.
- **Overlay** — the colormapped depth alpha-blended over the source image at
  55% opacity. Best for "where in this scene is depth coming from".

Both views are rendered client-side from the single base64 PNG returned by the
endpoint — no extra calls.
