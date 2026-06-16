# Model Workbench

A Databricks App for exploring custom vision models and Foundation Model APIs in one playground — chat, embeddings, object detection, segmentation, and depth estimation.

## What you get

- **Scoped discovery** — shows only the custom models you deploy (registered to the `model_workbench` schema) plus active Databricks Foundation Model APIs
- **Per-modality playgrounds** — chat, multimodal chat, embeddings with cosine matrix, object detection, segmentation, depth maps
- **Custom GPU models** — CLIP, YOLO26, YOLOS, Grounding DINO, Depth Anything, and optionally SAM 3
- **Scale-to-zero** — custom endpoints only run (and cost) when in use

## Setup

1. Import this repo into your Databricks workspace (Repos or Workspace Files)
2. Open `setup` as a notebook
3. Edit the configuration cell at the top (catalog, app name, which models to deploy)
4. **Run All**

The notebook handles everything: schema creation, model registration, endpoint creation, app deployment, and permissions. Takes ~15–20 minutes.

## Prerequisites

- A Databricks workspace with GPU model serving and Unity Catalog
- A catalog you have `CREATE SCHEMA` privileges on
- (Optional) A HuggingFace token with access to `facebook/sam3`, stored as a Databricks secret

## Repo structure

```
setup.py              ← Orchestrator notebook (run this)
models/
  clip.py             ← CLIP ViT-L/14 embeddings
  yolo26.py           ← YOLO26 real-time detection (Ultralytics)
  yolos.py            ← YOLOS transformer detection
  grounding_dino.py   ← Grounding DINO open-vocab detection
  depth_anything.py   ← Depth Anything V2 depth estimation
  sam3.py             ← SAM 3 segmentation (optional, gated)
app/                  ← Databricks App source (React + Node)
dashboard/            ← Lakeview usage dashboard (optional)
```

## Adding new models

1. Create a new notebook in `models/` (use any existing one as a template)
2. Add the notebook path and endpoint name to the `MODELS` dict in setup.py
3. Re-run `setup.py`

The app auto-discovers any endpoint whose registered model is in the `model_workbench` schema — no app code changes needed.

## Custom models included

| Model | What it does | GPU |
|---|---|---|
| [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) | Text + image embeddings in a shared vector space | A10G |
| [YOLO26](https://docs.ultralytics.com/models/yolo26/) | Real-time object detection (80 COCO classes, NMS-free) | T4 |
| [YOLOS](https://huggingface.co/hustvl/yolos-small) | Transformer-based detection (80 COCO classes) | T4 |
| [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) | Open-vocab object detection (any text prompt) | T4 |
| [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) | Monocular depth estimation | T4 |
| [SAM 3](https://huggingface.co/facebook/sam3) | Promptable segmentation (requires HF token) | A10G |

## License

MIT
