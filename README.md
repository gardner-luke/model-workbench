# Model Workbench

A Databricks App that auto-discovers every model serving endpoint in your workspace and gives you a playground for each one — chat, embeddings, object detection, segmentation, and depth estimation.

![Architecture](https://img.shields.io/badge/stack-React_+_Node_+_AppKit-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## What you get

- **Auto-discovery** — every serving endpoint in your workspace appears automatically
- **Per-modality playgrounds** — chat, multimodal chat, embeddings with cosine matrix, object detection, segmentation, depth maps
- **5 custom GPU models** — CLIP, YOLOS, Grounding DINO, Depth Anything, and SAM 3
- **Scale-to-zero** — custom endpoints only run (and cost) when in use

## Setup

### Prerequisites

- A Databricks workspace with GPU model serving and Unity Catalog
- Databricks CLI installed locally (`pip install databricks-cli` or `brew install databricks`)

### 1. Clone this repo into your workspace

Import this repository into your Databricks workspace, or clone it locally and upload the `setup` notebook.

### 2. Run the setup notebook

Open `setup.py` as a notebook in your workspace. Edit the two configuration values at the top:

```python
UC_CATALOG = "<YOUR_CATALOG>"    # A catalog you can create schemas in
UC_SCHEMA = "model_workbench"    # Will be created for you
```

Then **Run All**. The notebook registers all models and creates the serving endpoints (~15 min).

### 3. Deploy the app

From your local terminal:

```sh
# Edit databricks.yml — set your workspace host
databricks apps deploy model-workbench
```

Open the URL printed by the CLI. Done.

## Custom models included

| Model | What it does | GPU |
|---|---|---|
| [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) | Text + image embeddings in a shared vector space | A10G |
| [YOLOS](https://huggingface.co/hustvl/yolos-small) | Closed-vocab object detection (80 COCO classes) | T4 |
| [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) | Open-vocab object detection (any text prompt) | T4 |
| [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) | Monocular depth estimation | T4 |
| [SAM 3](https://huggingface.co/facebook/sam3) | Promptable segmentation (requires HF token) | A10G |

## Adding your own models

1. Copy any `endpoints/<model>/` folder as a template
2. Replace the HuggingFace model and `predict()` logic
3. Run the notebook to register → the app picks it up automatically

## License

MIT
