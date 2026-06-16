# Databricks notebook source
# MAGIC %md
# MAGIC # Model Workbench — Setup
# MAGIC
# MAGIC This notebook orchestrates the full deployment:
# MAGIC 1. Creates a Unity Catalog schema
# MAGIC 2. Runs each model notebook in parallel (register model + create endpoint)
# MAGIC 3. Creates and deploys the Databricks App
# MAGIC 4. Grants the app's service principal access to all model endpoints
# MAGIC
# MAGIC The app shows only the custom models registered by this project plus
# MAGIC Databricks Foundation Model APIs. New models added to the schema appear automatically.
# MAGIC
# MAGIC **Edit the configuration cell below, then Run All.**

# COMMAND ----------

# DBTITLE 1,Configuration — edit these values
UC_CATALOG = ""                        # Your Unity Catalog catalog name
UC_SCHEMA = "model_workbench"          # Schema name (will be created if needed)
APP_NAME = "model-workbench"           # Databricks App name (becomes the URL slug)

# Models to deploy. Each entry maps a notebook path to its serving endpoint name.
# Comment out any model you don't want to deploy.
MODELS = {
    "models/clip": "clip-vit-large-patch14",
    "models/yolos": "yolos",
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
# MAGIC ## Step 1: Validate & Create Schema

# COMMAND ----------

assert UC_CATALOG, "Set UC_CATALOG in the configuration cell above before running"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")
print(f"✓ Schema ready: {UC_CATALOG}.{UC_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Register Models & Create Endpoints
# MAGIC
# MAGIC Each model has its own notebook in `models/`. They run in parallel for faster setup.
# MAGIC Add new models by creating a notebook there and adding it to the `MODELS` dict above.

# COMMAND ----------

# DBTITLE 1,Run model notebooks in parallel
from concurrent.futures import ThreadPoolExecutor, as_completed

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
base_path = "/".join(notebook_path.split("/")[:-1])

def run_model(notebook_name):
    full_path = f"{base_path}/{notebook_name}"
    dbutils.notebook.run(full_path, timeout_seconds=1800, arguments={
        "uc_catalog": UC_CATALOG,
        "hf_token_scope": HF_TOKEN_SCOPE,
        "hf_token_key": HF_TOKEN_KEY,
    })
    return notebook_name

print(f"Deploying {len(MODELS)} models in parallel...")
results = {}
with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
    futures = {pool.submit(run_model, nb): nb for nb in MODELS.keys()}
    for future in as_completed(futures):
        nb = futures[future]
        try:
            future.result()
            results[nb] = "✓"
            print(f"  ✓ {nb}")
        except Exception as e:
            results[nb] = f"✗ {e}"
            print(f"  ✗ {nb}: {e}")

print(f"\n{sum(1 for v in results.values() if v == '✓')}/{len(MODELS)} models deployed successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create & Deploy the App

# COMMAND ----------

# DBTITLE 1,Create app and grant permissions
import requests
import time

HOST = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# Create or get the app
resp = requests.get(f"{HOST}/api/2.0/apps/{APP_NAME}", headers=HEADERS)
if resp.status_code == 200:
    app = resp.json()
    print(f"⏭ App '{APP_NAME}' already exists")
else:
    payload = {"name": APP_NAME, "description": "Model Workbench — custom vision models and Foundation Model APIs in one playground"}
    resp = requests.post(f"{HOST}/api/2.0/apps", headers=HEADERS, json=payload)
    resp.raise_for_status()
    app = resp.json()
    print(f"✓ App '{APP_NAME}' created")
    time.sleep(5)
    resp = requests.get(f"{HOST}/api/2.0/apps/{APP_NAME}", headers=HEADERS)
    app = resp.json()

sp_client_id = app.get("service_principal_client_id")
print(f"  App SP: {app.get('service_principal_name', sp_client_id)}")

# Grant CAN_QUERY on each model endpoint
for ep_name in MODELS.values():
    r = requests.get(f"{HOST}/api/2.0/serving-endpoints/{ep_name}", headers=HEADERS)
    if r.status_code != 200:
        continue
    ep_id = r.json().get("id")
    if not ep_id or not sp_client_id:
        continue
    perm_payload = {"access_control_list": [
        {"service_principal_name": sp_client_id, "permission_level": "CAN_QUERY"}
    ]}
    resp = requests.patch(f"{HOST}/api/2.0/permissions/serving-endpoints/{ep_id}", headers=HEADERS, json=perm_payload)
    if resp.status_code == 200:
        print(f"  ✓ CAN_QUERY on '{ep_name}'")
    else:
        print(f"  ⚠ Permission for '{ep_name}': {resp.text[:120]}")

# COMMAND ----------

# DBTITLE 1,Deploy the app
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
base_path = "/".join(notebook_path.split("/")[:-1])

# The apps API requires an absolute workspace path starting with /Workspace
app_source_path = f"{base_path}/app"
if not app_source_path.startswith("/Workspace"):
    app_source_path = f"/Workspace{app_source_path}"

print(f"Deploying from: {app_source_path}")
deploy_payload = {"source_code_path": app_source_path}
resp = requests.post(f"{HOST}/api/2.0/apps/{APP_NAME}/deployments", headers=HEADERS, json=deploy_payload)
if resp.status_code == 200:
    deployment = resp.json()
    print(f"✓ Deployment started: {deployment.get('deployment_id', '')}")
else:
    print(f"✗ Deployment failed ({resp.status_code}): {resp.text[:300]}")
    dbutils.notebook.exit(f"Deployment failed: {resp.text[:200]}")

print("\nWaiting for app to start...")
for i in range(60):
    time.sleep(10)
    resp = requests.get(f"{HOST}/api/2.0/apps/{APP_NAME}", headers=HEADERS)
    if resp.status_code == 200:
        status = resp.json().get("app_status", {})
        state = status.get("state", "")
        if state == "RUNNING":
            print(f"\n✓ App is live: {resp.json().get('url', '')}")
            break
        elif state in ("FAILED", "CRASHED"):
            print(f"\n✗ App failed: {status.get('message', '')}")
            break
        elif i % 3 == 0:
            print(f"  ... {state}")
else:
    print("\n⚠ Timed out — check app status in workspace UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC Your Model Workbench is deployed. The app displays:
# MAGIC - The custom models registered to `{UC_CATALOG}.model_workbench`
# MAGIC - Databricks Foundation Model APIs (auto-discovered)
# MAGIC
# MAGIC Custom endpoints use scale-to-zero. First request after cold start takes 3–10 min.
# MAGIC
# MAGIC ### Adding new models
# MAGIC
# MAGIC 1. Create a new notebook in `models/` (copy an existing one as a template)
# MAGIC 2. Add the notebook and endpoint name to the `MODELS` dict in the config cell
# MAGIC 3. Re-run this notebook
