# Model Workbench

A Databricks App that auto-discovers every model serving endpoint in your workspace and gives you a playground for each one — chat, embeddings, object detection, segmentation, and depth estimation.

## What you get

- **Auto-discovery** — every serving endpoint in your workspace appears automatically
- **Per-modality playgrounds** — chat, multimodal chat, embeddings with cosine matrix, object detection, segmentation, depth maps
- **5 custom GPU models** — CLIP, YOLOS, Grounding DINO, Depth Anything, and SAM 3
- **Scale-to-zero** — custom endpoints only run (and cost) when in use

## Setup

1. Import this repo into your Databricks workspace (Repos or Workspace Files)
2. Open `setup.py` as a notebook
3. Fill in the widgets at the top (catalog name, optional HF token for SAM 3)
4. **Run All**

The notebook handles everything: model registration, endpoint creation, app deployment, and permissions. Takes ~15–20 minutes.

## Prerequisites

- A Databricks workspace with GPU model serving and Unity Catalog
- A catalog you have `CREATE SCHEMA` privileges on
- (Optional) A HuggingFace token with access to `facebook/sam3`, stored as a Databricks secret

## Custom models included

| Model | What it does | GPU |
|---|---|---|
| [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) | Text + image embeddings in a shared vector space | A10G |
| [YOLOS](https://huggingface.co/hustvl/yolos-small) | Closed-vocab object detection (80 COCO classes) | T4 |
| [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) | Open-vocab object detection (any text prompt) | T4 |
| [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) | Monocular depth estimation | T4 |
| [SAM 3](https://huggingface.co/facebook/sam3) | Promptable segmentation (requires HF token) | A10G |

## License

MIT
