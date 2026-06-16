# Databricks notebook source
# MAGIC %md
# MAGIC # Model Workbench — Register Models & Create Endpoints
# MAGIC
# MAGIC Creates the Unity Catalog schema and runs each model notebook to register
# MAGIC the model and create its serving endpoint.
# MAGIC
# MAGIC **Normally called by `setup.py`. Can also be run standalone — fill in the widgets when prompted.**

# COMMAND ----------

# DBTITLE 1,Read configuration
uc_catalog = dbutils.widgets.get("uc_catalog").strip()
uc_schema = dbutils.widgets.get("uc_schema").strip() or "model_workbench"
models_json = dbutils.widgets.get("models_json").strip()
hf_token_scope = dbutils.widgets.get("hf_token_scope").strip()
hf_token_key = dbutils.widgets.get("hf_token_key").strip()

import json
MODELS = json.loads(models_json)

assert uc_catalog, "uc_catalog widget is required"
print(f"Catalog: {uc_catalog}")
print(f"Schema: {uc_schema}")
print(f"Models: {list(MODELS.values())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Schema

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {uc_catalog}.{uc_schema}")
print(f"✓ Schema ready: {uc_catalog}.{uc_schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Model Notebooks
# MAGIC
# MAGIC Each model has its own notebook in `models/`. Runs sequentially to avoid
# MAGIC pip lock contention.

# COMMAND ----------

# DBTITLE 1,Run model notebooks
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
base_path = "/".join(notebook_path.split("/")[:-2])

for notebook_name, endpoint_name in MODELS.items():
    full_path = f"{base_path}/{notebook_name}"
    print(f"Running {notebook_name}...")
    try:
        dbutils.notebook.run(full_path, timeout_seconds=1200, arguments={
            "uc_catalog": uc_catalog,
            "hf_token_scope": hf_token_scope,
            "hf_token_key": hf_token_key,
        })
        print(f"  ✓ {notebook_name} done")
    except Exception as e:
        print(f"  ✗ {notebook_name} failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC All model notebooks have been executed. Endpoints may still be provisioning
# MAGIC (takes 5–15 min for first-time GPU endpoints).
