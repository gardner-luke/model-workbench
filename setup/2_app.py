# Databricks notebook source
# MAGIC %md
# MAGIC # Model Workbench — Deploy App & Dashboard
# MAGIC
# MAGIC Creates the Databricks App, grants endpoint permissions to its service principal,
# MAGIC deploys the analytics dashboard, and deploys the app.
# MAGIC
# MAGIC **Normally called by `setup.py`. Can also be run standalone — fill in the widgets when prompted.**

# COMMAND ----------

# DBTITLE 1,Read configuration
import json
import requests
import time

uc_catalog = dbutils.widgets.get("uc_catalog").strip()
uc_schema = dbutils.widgets.get("uc_schema").strip() or "model_workbench"
app_name = dbutils.widgets.get("app_name").strip()
models_json = dbutils.widgets.get("models_json").strip()
workspace_id = dbutils.widgets.get("workspace_id").strip()

MODELS = json.loads(models_json)

assert uc_catalog, "uc_catalog widget is required"
assert app_name, "app_name widget is required"

HOST = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

print(f"App: {app_name}")
print(f"Endpoints: {list(MODELS.values())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create App & Grant Permissions

# COMMAND ----------

# DBTITLE 1,Create or get the app
resp = requests.get(f"{HOST}/api/2.0/apps/{app_name}", headers=HEADERS)
if resp.status_code == 200:
    app = resp.json()
    print(f"⏭ App '{app_name}' already exists")
else:
    payload = {"name": app_name, "description": "Model Workbench — custom vision models and Foundation Model APIs in one playground"}
    resp = requests.post(f"{HOST}/api/2.0/apps", headers=HEADERS, json=payload)
    resp.raise_for_status()
    app = resp.json()
    print(f"✓ App '{app_name}' created — waiting for RUNNING state...")
    for _wait in range(30):
        time.sleep(10)
        resp = requests.get(f"{HOST}/api/2.0/apps/{app_name}", headers=HEADERS)
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

# COMMAND ----------

# DBTITLE 1,Grant CAN_QUERY on each model endpoint
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
# MAGIC ## Deploy Analytics Dashboard

# COMMAND ----------

# DBTITLE 1,Create Lakeview dashboard
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
base_path = "/".join(notebook_path.split("/")[:-2])

dashboard_url = ""

if not workspace_id:
    print("⚠ WORKSPACE_ID not set — skipping dashboard deployment")
else:
    dashboard_file = f"/Workspace{base_path}/dashboard/usage_dashboard.lvdash.json"
    dash_spec = None
    try:
        with open(dashboard_file) as f:
            dash_spec = f.read()
        dash_spec = dash_spec.replace("<YOUR_CATALOG>", uc_catalog)
        dash_spec = dash_spec.replace("<YOUR_WORKSPACE_ID>", workspace_id)
    except FileNotFoundError:
        print(f"⚠ Dashboard file not found: {dashboard_file}")

    if dash_spec:
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the App

# COMMAND ----------

# DBTITLE 1,Deploy the app
app_source_path = f"{base_path}/app"
if not app_source_path.startswith("/Workspace"):
    app_source_path = f"/Workspace{app_source_path}"

# Set DASHBOARD_URL env var on the app so the Analytics link shows in the nav
if dashboard_url:
    env_payload = {"name": app_name, "env": [{"name": "DASHBOARD_URL", "value": dashboard_url}]}
    resp = requests.patch(f"{HOST}/api/2.0/apps/{app_name}", headers=HEADERS, json=env_payload)
    if resp.status_code == 200:
        print(f"✓ DASHBOARD_URL set on app")
    else:
        print(f"⚠ Failed to set DASHBOARD_URL: {resp.text[:120]}")

print(f"Deploying from: {app_source_path}")
deploy_payload = {"source_code_path": app_source_path}
resp = requests.post(f"{HOST}/api/2.0/apps/{app_name}/deployments", headers=HEADERS, json=deploy_payload)
if resp.status_code == 200:
    deployment = resp.json()
    print(f"✓ Deployment started: {deployment.get('deployment_id', '')}")
else:
    print(f"✗ Deployment failed ({resp.status_code}): {resp.text[:300]}")
    dbutils.notebook.exit(f"Deployment failed: {resp.text[:200]}")

print("\nWaiting for app to start...")
for i in range(60):
    time.sleep(10)
    resp = requests.get(f"{HOST}/api/2.0/apps/{app_name}", headers=HEADERS)
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
# MAGIC The app is deployed and the analytics dashboard is linked.
# MAGIC Custom endpoints use scale-to-zero — first request after cold start takes 3–10 min.
