# Databricks notebook source
# MAGIC %md
# MAGIC # Model Workbench — Setup
# MAGIC
# MAGIC Full end-to-end deployment in two steps:
# MAGIC 1. **Models** — creates schema, registers models, creates serving endpoints
# MAGIC 2. **App** — creates the Databricks App, deploys dashboard, deploys app
# MAGIC
# MAGIC **Edit the configuration cell below, then Run All.**
# MAGIC
# MAGIC You can also run each step independently:
# MAGIC - `setup/1_models` — re-register models or add new ones without redeploying the app
# MAGIC - `setup/2_app` — redeploy the app without re-registering models

# COMMAND ----------

# DBTITLE 1,Configuration — edit these values
UC_CATALOG = ""                        # Your Unity Catalog catalog name
UC_SCHEMA = "model_workbench"          # Schema name (will be created if needed)
APP_NAME = "model-workbench"           # Databricks App name (becomes the URL slug)
WORKSPACE_ID = ""                      # Numeric workspace ID (number after o= in your URL) — needed for dashboard

# Models to deploy. Each entry maps a notebook path to its serving endpoint name.
# Comment out any model you don't want to deploy.
MODELS = {
    "models/clip": "clip-vit-large-patch14",
    "models/grounding_dino": "grounding-dino",
    "models/depth_anything": "depth-anything",
    "models/yolo26": "yolo26",
    # "models/sam3": "sam3",             # Requires HF_TOKEN_SCOPE/KEY (gated model)
}

# HuggingFace token — only needed if sam3 is in the MODELS list above.
# The token must have access granted to the gated model on huggingface.co.
HF_TOKEN_SCOPE = ""                    # Databricks secret scope containing your HF token
HF_TOKEN_KEY = ""                      # Key within that scope

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Register Models & Create Endpoints

# COMMAND ----------

import json

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
base_path = "/".join(notebook_path.split("/")[:-1])

models_json = json.dumps(MODELS)

print("Running setup/1_models...")
dbutils.notebook.run(f"{base_path}/setup/1_models", timeout_seconds=0, arguments={
    "uc_catalog": UC_CATALOG,
    "uc_schema": UC_SCHEMA,
    "models_json": models_json,
    "hf_token_scope": HF_TOKEN_SCOPE,
    "hf_token_key": HF_TOKEN_KEY,
})
print("✓ Models step complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Deploy App & Dashboard

# COMMAND ----------

print("Running setup/2_app...")
dbutils.notebook.run(f"{base_path}/setup/2_app", timeout_seconds=0, arguments={
    "uc_catalog": UC_CATALOG,
    "uc_schema": UC_SCHEMA,
    "app_name": APP_NAME,
    "models_json": models_json,
    "workspace_id": WORKSPACE_ID,
})
print("✓ App step complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC Your Model Workbench is deployed. The app displays:
# MAGIC - Custom models registered to `model_workbench` schema
# MAGIC - Databricks Foundation Model APIs (auto-discovered)
# MAGIC
# MAGIC Custom endpoints use scale-to-zero. First request after cold start takes 3–10 min.
# MAGIC
# MAGIC ### Adding new models
# MAGIC
# MAGIC 1. Create a new notebook in `models/` (copy an existing one as a template)
# MAGIC 2. Add the notebook and endpoint name to the `MODELS` dict above
# MAGIC 3. Re-run this notebook (or just `setup/1_models` + `setup/2_app`)
