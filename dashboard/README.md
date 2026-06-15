# Usage & cost dashboard

A 2-page Databricks Lakeview dashboard published in the lab workspace and
linked from the Model Workbench app via the **Analytics** nav link.

## What it shows

**Overview**
- KPI cards (last 7 days): invocations, USD cost, error rate
- Stacked bar charts (last 30 days): invocations per day per endpoint, cost per
  day per endpoint

**Endpoints**
- Per-endpoint table for the last 7 days: invocations, cost, errors, error
  rate, last call — sorted by cost descending

## Data sources

| Source | Used for |
|---|---|
| `system.serving.endpoint_usage` | per-request log (invocations, status code) |
| `system.serving.served_entities` | endpoint name lookup (joins on `served_entity_id`), filter to current workspace |
| `system.billing.usage` | DBU consumption per endpoint per day |
| `system.billing.list_prices` | $ per DBU at the time the usage was recorded |

All queries filter on `workspace_id` — replace `<YOUR_WORKSPACE_ID>` in the
dashboard JSON with your actual workspace ID before deploying.

## How to deploy

```sh
python3 -c "
import json
spec = open('usage_dashboard.lvdash.json').read()
print(json.dumps({
  'display_name': 'Model Workbench — Usage & Cost',
  'parent_path': '/Workspace/Users/<you>',
  'warehouse_id': '<your_warehouse_id>',
  'serialized_dashboard': spec
}))
" > /tmp/dash.json
databricks lakeview create --json @/tmp/dash.json --profile <your_profile>

# Then publish (so it's shareable beyond yourself):
databricks lakeview publish <DASHBOARD_ID> \
  --json '{"embed_credentials": true, "warehouse_id": "<your_warehouse_id>"}' \
  --profile <your_profile>
```

The dashboard URL is `https://<workspace-host>/dashboardsv3/<DASHBOARD_ID>/published`.
Drop it into `app.yaml` as `DASHBOARD_URL` and the workbench's Analytics nav
link picks it up automatically.

## Caveats

- **Billing data is delayed 2–24h.** The cost KPI and cost chart trail real-time.
- **Endpoint usage is near-real-time (~5 min)** but the join through
  `served_entities` excludes the most-recently-created endpoints if they
  haven't yet propagated to the system tables.
- The dashboard scopes to **serving endpoints hosted in this workspace**. Calls
  to shared Foundation Model APIs hosted centrally won't appear in the
  invocations chart (they don't carry this workspace's `workspace_id` in
  `endpoint_usage`). They do appear in `system.billing.usage` if billed to
  this workspace.

## Things to add later (out of scope for v1)

- Latency distributions (need to source from inference payload tables)
- Per-user attribution (`endpoint_usage.requester` is available — group by it)
- Cold-start counts (would need to mine the serving event log)
- "Featured candidates" / "archive candidates" rule-of-thumb sections
- Genie space over the same data for ad-hoc Q&A
