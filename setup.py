# Databricks notebook source
# MAGIC %md
# MAGIC # Model Workbench — Setup
# MAGIC
# MAGIC This notebook orchestrates the full deployment:
# MAGIC 1. Creates a Unity Catalog schema
# MAGIC 2. Runs each model notebook (register model + create endpoint)
# MAGIC 3. Creates the Databricks App and grants permissions
# MAGIC 4. Deploys an analytics dashboard (optional — links from the app's nav)
# MAGIC 5. Deploys the app
# MAGIC
# MAGIC The app shows only the custom models registered to the `model_workbench` schema
# MAGIC plus Databricks Foundation Model APIs. New models added to the schema appear automatically.
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
# MAGIC Each model has its own notebook in `models/`. Add new models by creating a
# MAGIC notebook there and adding it to the `MODELS` dict above.

# COMMAND ----------

# DBTITLE 1,Run model notebooks
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
base_path = "/".join(notebook_path.split("/")[:-1])

for notebook_name, endpoint_name in MODELS.items():
    full_path = f"{base_path}/{notebook_name}"
    print(f"Running {notebook_name}...")
    try:
        dbutils.notebook.run(full_path, timeout_seconds=1200, arguments={
            "uc_catalog": UC_CATALOG,
            "hf_token_scope": HF_TOKEN_SCOPE,
            "hf_token_key": HF_TOKEN_KEY,
        })
        print(f"  ✓ {notebook_name} done")
    except Exception as e:
        print(f"  ✗ {notebook_name} failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create App & Grant Permissions

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
    print(f"✓ App '{APP_NAME}' created — waiting for RUNNING state...")
    for _wait in range(30):
        time.sleep(10)
        resp = requests.get(f"{HOST}/api/2.0/apps/{APP_NAME}", headers=HEADERS)
        app = resp.json()
        state = app.get("app_status", {}).get("state", "")
        if state == "RUNNING":
            print(f"  ✓ App is RUNNING")
            break
        elif state in ("FAILED", "CRASHED"):
            print(f"  ✗ App failed to start: {app.get('app_status', {}).get('message', '')}")
            break
    else:
        print(f"  ⚠ App not yet RUNNING (state: {state}) — deployment may fail")

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

# MAGIC %md
# MAGIC ## Step 4: Deploy Analytics Dashboard

# COMMAND ----------

# DBTITLE 1,Create Lakeview dashboard
import json

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
base_path = "/".join(notebook_path.split("/")[:-1])

try:
    workspace_id = spark.conf.get("spark.databricks.clusterUsageTags.orgId")
except Exception:
    workspace_id = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply("orgId")

# Read and fill placeholders in the dashboard spec
dashboard_ws_path = f"{base_path}/dashboard/usage_dashboard.lvdash.json"
if not dashboard_ws_path.startswith("/Workspace"):
    dashboard_ws_path = f"/Workspace{dashboard_ws_path}"

dashboard_url = ""

# Read dashboard spec from workspace file via REST API
import base64
export_path = dashboard_ws_path.replace("/Workspace", "", 1) if dashboard_ws_path.startswith("/Workspace") else dashboard_ws_path
resp = requests.get(f"{HOST}/api/2.0/workspace/export", headers=HEADERS, params={"path": dashboard_ws_path, "format": "AUTO"})
dash_spec = None
if resp.status_code == 200:
    content_b64 = resp.json().get("content", "")
    dash_spec = base64.b64decode(content_b64).decode("utf-8")
    dash_spec = dash_spec.replace("<YOUR_CATALOG>", UC_CATALOG)
    dash_spec = dash_spec.replace("<YOUR_WORKSPACE_ID>", workspace_id)
else:
    print(f"⚠ Could not read dashboard JSON ({resp.status_code}): {resp.text[:120]}")

if dash_spec:

    # Check if dashboard already exists
    resp = requests.get(f"{HOST}/api/2.0/lakeview/dashboards", headers=HEADERS, params={"page_size": 100})
    existing_id = None
    if resp.status_code == 200:
        for d in resp.json().get("dashboards", []):
            if d.get("display_name") == "Model Workbench — Usage & Cost":
                existing_id = d["dashboard_id"]
                break

    dash_payload = {
        "display_name": "Model Workbench — Usage & Cost",
        "serialized_dashboard": dash_spec,
        "parent_path": "/".join(dashboard_ws_path.rsplit("/", 2)[:-2]),
    }

    if existing_id:
        resp = requests.patch(f"{HOST}/api/2.0/lakeview/dashboards/{existing_id}", headers=HEADERS, json=dash_payload)
        dash_id = existing_id
        print(f"⏭ Dashboard updated: {dash_id}")
    else:
        resp = requests.post(f"{HOST}/api/2.0/lakeview/dashboards", headers=HEADERS, json=dash_payload)
        if resp.status_code == 200:
            dash_id = resp.json().get("dashboard_id", "")
            print(f"✓ Dashboard created: {dash_id}")
        else:
            dash_id = ""
            print(f"⚠ Dashboard creation failed ({resp.status_code}): {resp.text[:200]}")

    if dash_id:
        # Publish dashboard
        pub_resp = requests.post(
            f"{HOST}/api/2.0/lakeview/dashboards/{dash_id}/published",
            headers=HEADERS,
            json={"embed_credentials": True},
        )
        if pub_resp.status_code == 200:
            print(f"  ✓ Dashboard published")
        else:
            print(f"  ⚠ Publish failed: {pub_resp.text[:120]}")
        dashboard_url = f"{HOST}/dashboardsv3/{dash_id}/published"
        print(f"  URL: {dashboard_url}")
else:
    print("⚠ Dashboard JSON not found — skipping analytics dashboard")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Deploy the App

# COMMAND ----------

# DBTITLE 1,Deploy the app
# The apps API requires an absolute workspace path starting with /Workspace
app_source_path = f"{base_path}/app"
if not app_source_path.startswith("/Workspace"):
    app_source_path = f"/Workspace{app_source_path}"

# Set DASHBOARD_URL env var on the app so the Analytics link shows in the nav
if dashboard_url:
    env_payload = {"name": APP_NAME, "env": [{"name": "DASHBOARD_URL", "value": dashboard_url}]}
    resp = requests.patch(f"{HOST}/api/2.0/apps/{APP_NAME}", headers=HEADERS, json=env_payload)
    if resp.status_code == 200:
        print(f"✓ DASHBOARD_URL set on app")
    else:
        print(f"⚠ Failed to set DASHBOARD_URL: {resp.text[:120]}")

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
