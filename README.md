# Model Workbench

A single Databricks App that auto-discovers every model deployed in your
workspace and gives you a task-appropriate playground for each one — chat,
multimodal chat, embeddings, segmentation. Two custom GPU model serving
endpoints are deployed alongside it: **CLIP** for shared text+image embeddings
and **SAM 3** for promptable concept segmentation.

The repo is intentionally compact. It exists as an example you can read,
modify, and use as a starting point — not as a framework.

## What it demonstrates

| Concept | Where to look |
|---|---|
| **A full-stack Databricks App** (React + Node) | `client/`, `server/`, `app.yaml`, `databricks.yml` |
| **Auto-discovery of serving endpoints** | `server/server.ts` — `/api/endpoints` calls the Databricks SDK and classifies endpoints by modality |
| **Foundation Model API invocation** | `server/server.ts` — `/api/invoke/:name` and `/api/embed/:name` |
| **Custom GPU model serving** | `endpoints/clip/`, `endpoints/sam3/` — MLflow PyFunc wrappers, UC registration, endpoint create |
| **Secret-backed gated model access** | `endpoints/sam3/register_sam3.py` — reads HF token from a Databricks secret, injects it as a serving endpoint env var |
| **Inference tables for governance** | AI Gateway config in the per-endpoint READMEs — every request/response logged to UC |
| **Per-modality UIs** | `client/src/pages/Playground.tsx`, `Embeddings.tsx`, `Segmentation.tsx` |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Browser (React app)                         │
│   • Auto-detects modality per endpoint                       │
│   • Routes user to the right playground page                 │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTPS (with Databricks OAuth)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           Databricks App (Node + AppKit, this repo)          │
│   • Lists serving endpoints via Databricks SDK               │
│   • Proxies invocations with the app's service-principal    │
│   • Normalizes responses (chat / embed / segment)            │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────┐         ┌──────────────────────────────┐
│   Foundation Model   │         │   Custom Model Serving       │
│   API endpoints      │         │   endpoints (GPU)            │
│   (Claude, Llama,    │         │                              │
│    GPT-OSS, BGE,     │         │   • clip-vit-large-patch14   │
│    GTE, Qwen, …)     │         │   • sam3                     │
└──────────────────────┘         └──────────────────────────────┘
                                            ▲
                                            │  (MLflow + UC)
                                            │
                                  ┌─────────┴────────────────┐
                                  │   endpoints/<model>/     │
                                  │   register notebooks     │
                                  └──────────────────────────┘
```

## Repo layout

```
model-workbench/
├── README.md                      ← this file
├── app.yaml                       ← Databricks Apps runtime entrypoint
├── databricks.yml                 ← DABs bundle definition (deploys the app)
│
├── client/                        ← React frontend
│   └── src/
│       ├── App.tsx                ← Router + Databricks-branded layout
│       ├── pages/
│       │   ├── Registry.tsx       ← Home page: searchable, filterable card grid
│       │   ├── Playground.tsx     ← Chat + multimodal chat playground
│       │   ├── Embeddings.tsx     ← Text + multimodal embeddings + cosine matrix
│       │   └── Segmentation.tsx   ← Image + prompt → mask overlay
│       ├── components/
│       │   └── DatabricksLogo.tsx
│       ├── lib/
│       │   └── image.ts           ← Client-side image downscaling
│       └── types.ts               ← Shared types between client + server
│
├── server/
│   └── server.ts                  ← Node + AppKit. All /api/* routes live here.
│
└── endpoints/                     ← One subdirectory per custom serving endpoint
    ├── clip/
    │   ├── README.md              ← Endpoint contract + how to deploy
    │   └── register_clip.py       ← Databricks notebook: log to UC
    └── sam3/
        ├── README.md
        └── register_sam3.py
```

## Walking through this with someone

The order I'd open files when explaining it to a teammate:

1. **`README.md` → "What it demonstrates" table** to set context.
2. **The deployed app** (live URL or screenshot) — show the registry, click a
   chat model, click an embedding model, click SAM 3. Emphasize that *one* app
   adapts the UI to each modality.
3. **`server/server.ts`** — start from the top. The `classifyModality()`
   function is the core trick: tasks like `llm/v1/chat` get a chat UI,
   `llm/v1/embeddings` gets an embedding UI, model names matching `/sam-?\d?/`
   get a segmentation UI. Anyone can fork this and add patterns.
4. **`client/src/pages/Registry.tsx`** — show how the card grid groups by
   `kind` (custom vs foundation) and filters by `modality`. Click a card →
   navigate to the right playground.
5. **One playground file** (`Segmentation.tsx` is the most visual) — show how
   the UI for one modality is just a focused React page that calls one server
   route.
6. **`endpoints/sam3/register_sam3.py`** — the per-model recipe.
   - Top-of-file constants are the only things to change per workspace.
   - HF token comes from a Databricks secret (no key in source).
   - The `Sam3SegmenterModel` class is a standard MLflow PyFunc — anyone
     familiar with MLflow can add another model the same way.
   - `mlflow.pyfunc.log_model(..., registered_model_name=UC_NAME)` does the UC
     registration in one call.
7. **`endpoints/sam3/README.md`** — the public contract of the endpoint.
   Anything calling the endpoint only needs this.

## Quickstart

### Prerequisites

- Node 22+ and npm 11+
- Databricks CLI 0.295+
- A Databricks workspace with:
  - Foundation Model APIs enabled (most workspaces have this by default)
  - GPU model serving available (for the CLIP and SAM 3 endpoints)
  - A Unity Catalog catalog you can create schemas/tables in

### 1. Deploy the app

```sh
git clone <this-repo>
cd model-workbench

# Configure databricks.yml workspace host
# Then:
databricks apps deploy --profile <your-profile>
```

Open the URL the CLI prints. The registry will already show every foundation
model in your workspace.

### 2. Deploy CLIP

```sh
# Open endpoints/clip/register_clip.py
# Edit the top-of-file constants (catalog/schema)
# Upload + run the notebook in your workspace:
databricks workspace import /Workspace/Users/YOU/clip-register \
  --format SOURCE --language PYTHON \
  --file endpoints/clip/register_clip.py --overwrite \
  --profile <your-profile>

databricks jobs submit --json '{
  "run_name": "register-clip",
  "tasks": [{
    "task_key": "register",
    "notebook_task": {"notebook_path": "/Workspace/Users/YOU/clip-register"},
    "environment_key": "default"
  }],
  "environments": [{"environment_key": "default", "spec": {"environment_version": "3", "dependencies": []}}]
}' --profile <your-profile>
```

Then follow `endpoints/clip/README.md` to create the serving endpoint and
grant the app's service principal `CAN_QUERY` on it.

### 3. Deploy SAM 3

See `endpoints/sam3/README.md`. There's one extra step: SAM 3 is a gated
HuggingFace repo, so you need an HF token saved as a Databricks secret first.

## Conventions used throughout

### Secrets

Anything sensitive (HF tokens, third-party API keys) is read from Databricks
secrets at registration time AND passed as a serving endpoint
`environment_vars` referencing the same secret. Source code never contains a
token.

### Catalog / schema

Custom models all register to the same UC schema so they're easy to find:
`${UC_CATALOG}.model_workbench.${UC_MODEL}`. Inference tables follow the same
pattern: `${UC_CATALOG}.model_workbench.${endpoint}_inference_payload`.

### Modality classification

The server classifies each endpoint into one of:

| Modality | Detected when | UI page |
|---|---|---|
| `text` | task is `llm/v1/chat` or `llm/v1/completions`, model name not in multimodal list | `/playground/:name` |
| `multimodal` | model name matches Claude/Llama-4/Gemma-3/GPT-5/Gemini-2.5/etc. | `/playground/:name` (with image attach) |
| `text_embedding` | task is `llm/v1/embeddings` | `/embeddings/:name` |
| `multimodal_embedding` | endpoint or model name matches `/clip|siglip|blip|imagebind/` | `/embeddings/:name` (text + image inputs) |
| `segmentation` | endpoint or model name matches `/sam-?\d?\|grounding-?dino\|grounded-?sam/` | `/segmentation/:name` |

To add a new modality, extend `classifyModality()` in `server/server.ts` and
add a new page under `client/src/pages/`. Then add a route in `App.tsx` and a
link from the registry card.

### Image handling

The client downscales images to ≤1536px (≤768px for CLIP) using a shared
helper at `client/src/lib/image.ts` before encoding to base64. Two reasons:

1. Databricks Model Serving caps request payloads at 16 MiB. A phone photo
   easily exceeds that base64-encoded.
2. The models themselves resize internally (CLIP: 224×224, SAM 3: ~1024px) —
   sending 4K does nothing useful.

### Body parsing

AppKit's default JSON parser caps request bodies at 100 KB and runs before
custom routes. Heavy routes (`/api/segment`, `/api/embed`, `/api/invoke`) send
`Content-Type: application/octet-stream` from the client, which makes AppKit's
parser skip them, then a per-route `express.raw({ limit: '64mb' })` reads the
body. See the comment block at the top of `server/server.ts`.

## Adding a new model

The pattern is intentionally simple:

1. Pick a model on HuggingFace (or anywhere). Decide its modality.
2. Copy `endpoints/clip/` or `endpoints/sam3/` to `endpoints/<your-model>/`.
3. Replace the `HF_MODEL` constant and the `predict()` logic inside the
   PyFunc class.
4. Update the input/output contract in the README.
5. Run the notebook → create the serving endpoint → grant the app's SP
   `CAN_QUERY`.
6. If it's a brand-new modality the workbench doesn't know about yet, add a
   pattern to `classifyModality()` in `server/server.ts` and a corresponding
   page in `client/src/pages/`.

That's the whole loop. New models become available in the UI without touching
existing model code.

## Things this app does *not* do (and where they'd go if added)

- **No run history persistence.** Every invocation is one-shot, no log of past
  prompts. The natural place to add this is a Lakebase Postgres or a Delta
  table written from the server after each call. Inference tables already
  capture the wire-level data — a run-history feature would be a
  human-readable view on top.
- **No UC Volume picker.** Uploads are direct from the user's machine. AppKit
  ships a Files plugin for browsing UC volumes — wiring it in would let users
  pick frames from a video-log volume instead of uploading.
- **No agent layer.** Each playground calls one endpoint. The architecture
  intentionally separates "raw model access" from "agent that decides which
  model to call". Agent Bricks or a custom agent in the server is the natural
  next step.

## Stack

- **Frontend**: React 19, TypeScript, Tailwind, Vite, React Router 7
- **UI primitives**: `@databricks/appkit-ui` (shadcn/ui under the hood)
- **Backend**: Node 22+, Express via `@databricks/appkit`
- **Databricks**: AppKit, Databricks SDK for Node, MLflow PyFunc
- **Models**: HuggingFace transformers (CLIP, SAM 3), Foundation Model APIs
