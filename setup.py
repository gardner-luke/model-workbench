# Databricks notebook source
# MAGIC %md
# MAGIC # Model Workbench — Setup
# MAGIC
# MAGIC This notebook orchestrates the full deployment:
# MAGIC 1. Creates a Unity Catalog schema
# MAGIC 2. Runs each model notebook (register + create endpoint)
# MAGIC 3. Creates and deploys the Databricks App
# MAGIC 4. Grants the app's service principal access to all endpoints
# MAGIC
# MAGIC **Fill in the widgets at the top, then Run All.**

# COMMAND ----------

# DBTITLE 1,Configuration
dbutils.widgets.text("uc_catalog", "", "Unity Catalog Name")
dbutils.widgets.text("hf_token_scope", "", "HF Secret Scope (for SAM 3, leave blank to skip)")
dbutils.widgets.text("hf_token_key", "", "HF Secret Key (for SAM 3, leave blank to skip)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Validate & Create Schema

# COMMAND ----------

UC_CATALOG = dbutils.widgets.get("uc_catalog").strip()
UC_SCHEMA = "model_workbench"
HF_TOKEN_SCOPE = dbutils.widgets.get("hf_token_scope").strip()
HF_TOKEN_KEY = dbutils.widgets.get("hf_token_key").strip()

assert UC_CATALOG, "Set the 'uc_catalog' widget to your Unity Catalog name before running"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")
print(f"✓ Schema ready: {UC_CATALOG}.{UC_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Register Models & Create Endpoints
# MAGIC
# MAGIC Each model has its own notebook in `models/`. Add new models by creating a new
# MAGIC notebook there and adding it to the list below.

# COMMAND ----------

# DBTITLE 1,Run model notebooks
import os

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
base_path = "/".join(notebook_path.split("/")[:-1])

models = [
    "models/clip",
    "models/yolos",
    "models/grounding_dino",
    "models/depth_anything",
]

if HF_TOKEN_SCOPE and HF_TOKEN_KEY:
    models.append("models/sam3")
else:
    print("⏭ SAM 3 skipped (no HF token configured)")

for model in models:
    full_path = f"{base_path}/{model}"
    print(f"Running {model}...")
    try:
        dbutils.notebook.run(full_path, timeout_seconds=1800, arguments={
            "uc_catalog": UC_CATALOG,
            "hf_token_scope": HF_TOKEN_SCOPE,
            "hf_token_key": HF_TOKEN_KEY,
        })
        print(f"  ✓ {model} done")
    except Exception as e:
        print(f"  ✗ {model} failed: {e}")

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

APP_NAME = "model-workbench"

# Create or get the app
resp = requests.get(f"{HOST}/api/2.0/apps/{APP_NAME}", headers=HEADERS)
if resp.status_code == 200:
    app = resp.json()
    print(f"⏭ App '{APP_NAME}' already exists")
else:
    payload = {"name": APP_NAME, "description": "Model Workbench — explore every model deployed in your Databricks workspace"}
    resp = requests.post(f"{HOST}/api/2.0/apps", headers=HEADERS, json=payload)
    resp.raise_for_status()
    app = resp.json()
    print(f"✓ App '{APP_NAME}' created")
    time.sleep(5)
    resp = requests.get(f"{HOST}/api/2.0/apps/{APP_NAME}", headers=HEADERS)
    app = resp.json()

sp_client_id = app.get("service_principal_client_id")
print(f"  App SP: {app.get('service_principal_name', sp_client_id)}")

# Grant CAN_QUERY on all model endpoints
ENDPOINT_NAMES = ["clip-vit-large-patch14", "yolos", "grounding-dino", "depth-anything"]
if HF_TOKEN_SCOPE and HF_TOKEN_KEY:
    ENDPOINT_NAMES.append("sam3")

for ep_name in ENDPOINT_NAMES:
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
app_source_path = f"{base_path}/app"

deploy_payload = {"source_code_path": app_source_path}
resp = requests.post(f"{HOST}/api/2.0/apps/{APP_NAME}/deployments", headers=HEADERS, json=deploy_payload)
if resp.status_code == 200:
    deployment = resp.json()
    print(f"✓ Deployment started")
    print(f"  Source: {app_source_path}")
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
# MAGIC Your Model Workbench is deployed. The app auto-discovers all serving endpoints
# MAGIC in your workspace — the custom models above plus any Foundation Model APIs.
# MAGIC
# MAGIC Custom endpoints use scale-to-zero. First request after cold start takes 3–10 min.
# MAGIC
# MAGIC ### Adding new models
# MAGIC
# MAGIC 1. Create a new notebook in `models/` (copy an existing one as a template)
# MAGIC 2. Add the notebook name to the `models` list in Step 2 above
# MAGIC 3. Re-run this notebook
