import { useEffect, useState } from 'react';
import {
  Button,
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@databricks/appkit-ui/react';
import { Check, ChevronDown, Code, Copy } from 'lucide-react';
import type { Modality } from '../types';

interface NotebookSnippetProps {
  endpointName: string;
  modality: Modality;
}

function pythonSnippet(host: string, endpointName: string, modality: Modality): string {
  const url = `${host || 'https://<your-workspace-url>'}/serving-endpoints/${endpointName}/invocations`;

  if (modality === 'text' || modality === 'multimodal') {
    return `# Call '${endpointName}' from a Databricks notebook
# Uses the OpenAI-compatible Foundation Model API.
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ.get("DATABRICKS_TOKEN"),  # in notebooks this is auto-injected
    base_url="${host || '<your-workspace-url>'}/serving-endpoints",
)

response = client.chat.completions.create(
    model="${endpointName}",
    messages=[
${
  modality === 'multimodal'
    ? `        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<your-base64-image>"}},
            ],
        },`
    : `        {"role": "user", "content": "Hello, how are you?"},`
}
    ],
    max_tokens=512,
)
print(response.choices[0].message.content)`;
  }

  if (modality === 'text_embedding') {
    return `# Call '${endpointName}' from a Databricks notebook
# Uses the OpenAI-compatible embeddings API.
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ.get("DATABRICKS_TOKEN"),
    base_url="${host || '<your-workspace-url>'}/serving-endpoints",
)

response = client.embeddings.create(
    model="${endpointName}",
    input=["a yellow circle on a blue square", "an industrial pipeline at sunset"],
)
for i, item in enumerate(response.data):
    print(f"vector {i}: dim={len(item.embedding)}  preview={item.embedding[:4]}")`;
  }

  if (modality === 'multimodal_embedding') {
    return `# Call '${endpointName}' from a Databricks notebook
# Custom CLIP endpoint — text and images mix in one request.
import base64, json, os, requests

with open("/dbfs/path/to/image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = {
    "dataframe_records": [
        {"type": "text",  "value": "a yellow circle on a blue square"},
        {"type": "image", "value": img_b64},
    ]
}

resp = requests.post(
    "${url}",
    headers={
        "Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}",
        "Content-Type": "application/json",
    },
    data=json.dumps(payload),
)
resp.raise_for_status()
preds = resp.json()["predictions"]
print(f"got {len(preds['embeddings'])} vectors of dim {preds['dim']}")`;
  }

  if (modality === 'depth_estimation') {
    return `# Call '${endpointName}' from a Databricks notebook
# Custom Depth Anything endpoint — image in, per-pixel depth out as a base64 PNG.
import base64, io, json, os, requests
from PIL import Image

with open("/dbfs/path/to/image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "${url}",
    headers={
        "Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}",
        "Content-Type": "application/json",
    },
    data=json.dumps({"dataframe_records": [{"image": img_b64}]}),
)
resp.raise_for_status()
preds = resp.json()["predictions"]
if isinstance(preds, list):
    preds = preds[0]

print(f"depth range: {preds['min_depth']:.3f} → {preds['max_depth']:.3f}")
print(f"image_size: {preds['image_size']}")

# The 'depth_png' field is a base64 grayscale PNG, same dimensions as the input.
# Brighter pixels = closer to the camera (after the wrapper's normalization).
depth = Image.open(io.BytesIO(base64.b64decode(preds["depth_png"])))
depth.save("/tmp/depth.png")`;
  }

  if (modality === 'object_detection') {
    return `# Call '${endpointName}' from a Databricks notebook
# Custom detection endpoint — image (+ optional text prompt) → boxes, scores, labels.
import base64, json, os, requests

with open("/dbfs/path/to/image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = {
    "dataframe_records": [{
        "image": img_b64,
        # Open-vocab models (e.g. Grounding DINO) accept a text_prompt with concepts
        # separated by periods. Closed-vocab models (YOLOS, DETR) ignore this field.
        "text_prompt": "person. car. tractor.",
        "threshold": 0.3,
    }]
}

resp = requests.post(
    "${url}",
    headers={
        "Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}",
        "Content-Type": "application/json",
    },
    data=json.dumps(payload),
)
resp.raise_for_status()
preds = resp.json()["predictions"]
if isinstance(preds, list):
    preds = preds[0]

print(f"Found {preds['count']} objects at {preds['image_size']}")
for i, (box, score, label) in enumerate(zip(preds["boxes"], preds["scores"], preds["labels"])):
    print(f"  #{i+1}: {label} score={score:.3f} box={box}")`;
  }

  if (modality === 'segmentation') {
    return `# Call '${endpointName}' from a Databricks notebook
# Custom SAM 3 endpoint — image + text prompt → masks, boxes, scores.
import base64, io, json, os, requests
from PIL import Image

with open("/dbfs/path/to/image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = {
    "dataframe_records": [{
        "image": img_b64,
        "text_prompt": "corn kernel",
        "threshold": 0.3,
        "mask_threshold": 0.5,
    }]
}

resp = requests.post(
    "${url}",
    headers={
        "Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}",
        "Content-Type": "application/json",
    },
    data=json.dumps(payload),
)
resp.raise_for_status()
result = resp.json()["predictions"]
if isinstance(result, list):
    result = result[0]

print(f"Found {result['count']} instances at image_size {result['image_size']}")
for i, (box, score) in enumerate(zip(result["boxes"], result["scores"])):
    print(f"  #{i+1}: score={score:.3f}  box={box}")

# Each entry in result['masks'] is a base64-encoded 1-bit PNG, same size as the input image.
mask_png = base64.b64decode(result["masks"][0])
mask = Image.open(io.BytesIO(mask_png))
mask.save("/tmp/mask0.png")`;
  }

  // Fallback — generic invocation.
  return `# Call '${endpointName}' from a Databricks notebook
import json, os, requests

resp = requests.post(
    "${url}",
    headers={
        "Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}",
        "Content-Type": "application/json",
    },
    data=json.dumps({"inputs": []}),
)
resp.raise_for_status()
print(resp.json())`;
}

function sqlSnippet(endpointName: string, modality: Modality): string {
  if (modality === 'text') {
    return `-- Call '${endpointName}' from SQL with ai_query.
-- For a one-off prompt:
SELECT ai_query('${endpointName}', 'Hello, how are you?') AS response;

-- For batch inference over a Delta table:
-- SELECT prompt, ai_query('${endpointName}', prompt) AS response
-- FROM my_prompts_table;`;
  }

  if (modality === 'multimodal') {
    return `-- Call '${endpointName}' from SQL with ai_query.
-- SQL is great for text prompts in batch — for image inputs, use Python (the
-- multimodal payload shape doesn't compose cleanly in SQL).
SELECT ai_query('${endpointName}', 'Describe a green tractor.') AS response;`;
  }

  if (modality === 'text_embedding') {
    return `-- Embed a single string from SQL:
SELECT ai_query('${endpointName}', 'a yellow circle on a blue square') AS embedding;

-- Or run over a Delta table:
-- SELECT id, ai_query('${endpointName}', text_col) AS embedding
-- FROM my_docs_table;`;
  }

  if (modality === 'multimodal_embedding') {
    return `-- Embed a single text input. Pass a named_struct matching the endpoint's
-- input signature (columns: type, value).
SELECT ai_query(
  '${endpointName}',
  named_struct('type', 'text', 'value', 'a yellow circle on a blue square')
) AS embedding;

-- Batch over a Delta table:
-- SELECT id, ai_query('${endpointName}', named_struct('type', type_col, 'value', value_col)) AS embedding
-- FROM my_inputs_table;
-- For image rows, 'value' should hold the base64-encoded image string.`;
  }

  if (modality === 'depth_estimation') {
    return `-- One-off depth call. 'image' is a base64-encoded JPEG/PNG.
-- The result struct has 'depth_png' (base64 grayscale PNG), 'min_depth',
-- 'max_depth', and 'image_size'.
SELECT ai_query(
  '${endpointName}',
  named_struct('image', '<base64-jpeg-here>')
) AS result;

-- Batch over a Delta table of pre-encoded images:
-- SELECT image_id,
--   ai_query('${endpointName}', named_struct('image', image_base64)) AS result
-- FROM my_images_table;`;
  }

  if (modality === 'object_detection') {
    return `-- One-off detection call. 'image' is a base64-encoded JPEG/PNG.
-- For open-vocab models (Grounding DINO) pass a text_prompt with concepts
-- separated by periods; closed-vocab models (YOLOS) ignore that field.
SELECT ai_query(
  '${endpointName}',
  named_struct(
    'image', '<base64-jpeg-here>',
    'text_prompt', 'person. car. tractor.',
    'threshold', 0.3
  )
) AS result;

-- Batch over a Delta table of pre-encoded images:
-- SELECT image_id,
--   ai_query('${endpointName}', named_struct(
--     'image', image_base64,
--     'text_prompt', 'person. car. tractor.',
--     'threshold', 0.3
--   )) AS result
-- FROM my_images_table;`;
  }

  if (modality === 'segmentation') {
    return `-- One-off call. 'image' is a base64-encoded JPEG/PNG. The named_struct
-- matches the endpoint's input signature (image, text_prompt, threshold,
-- mask_threshold).
SELECT ai_query(
  '${endpointName}',
  named_struct(
    'image', '<base64-jpeg-here>',
    'text_prompt', 'corn kernel',
    'threshold', 0.3,
    'mask_threshold', 0.5
  )
) AS result;

-- Batch over a Delta table of pre-encoded images:
-- SELECT image_id,
--   ai_query('${endpointName}', named_struct(
--     'image', image_base64,
--     'text_prompt', 'corn kernel',
--     'threshold', 0.3,
--     'mask_threshold', 0.5
--   )) AS result
-- FROM my_images_table;`;
  }

  return `-- ai_query support depends on the endpoint shape — see the Python tab for
-- the canonical call against this endpoint.`;
}

function curlSnippet(host: string, endpointName: string, modality: Modality): string {
  const url = `${host || '<your-workspace-url>'}/serving-endpoints/${endpointName}/invocations`;
  let body = '{"inputs": []}';
  if (modality === 'text' || modality === 'multimodal') {
    body = `{"messages": [{"role":"user","content":"Hello"}]}`;
  } else if (modality === 'text_embedding') {
    body = `{"input": ["text 1", "text 2"]}`;
  } else if (modality === 'multimodal_embedding') {
    body = `{"dataframe_records": [{"type":"text","value":"a yellow circle"}]}`;
  } else if (modality === 'segmentation') {
    body = `{"dataframe_records": [{"image":"<base64>","text_prompt":"corn kernel"}]}`;
  } else if (modality === 'object_detection') {
    body = `{"dataframe_records": [{"image":"<base64>","text_prompt":"person. car. tractor.","threshold":0.3}]}`;
  } else if (modality === 'depth_estimation') {
    body = `{"dataframe_records": [{"image":"<base64>"}]}`;
  }
  return `curl -X POST \\
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '${body}' \\
  "${url}"`;
}

export function NotebookSnippet({ endpointName, modality }: NotebookSnippetProps) {
  const [host, setHost] = useState<string>('');
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/workspace')
      .then((r) => r.json() as Promise<{ host?: string }>)
      .then((data) => setHost(data.host ?? ''))
      .catch(() => {
        // Non-blocking — snippet falls back to placeholder host.
      });
  }, []);

  const python = pythonSnippet(host, endpointName, modality);
  const sql = sqlSnippet(endpointName, modality);
  const curl = curlSnippet(host, endpointName, modality);

  const copy = async (text: string, tab: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(tab);
      setTimeout(() => setCopied((c) => (c === tab ? null : c)), 1500);
    } catch {
      // ignore
    }
  };

  return (
    <Collapsible className="border-t pt-4">
      <CollapsibleTrigger asChild>
        <Button variant="outline" size="sm">
          <Code className="h-4 w-4" />
          Use in a Databricks notebook
          <ChevronDown className="h-4 w-4 ml-1" />
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-3">
        <Tabs defaultValue="python">
          <TabsList>
            <TabsTrigger value="python">Python</TabsTrigger>
            <TabsTrigger value="sql">SQL</TabsTrigger>
            <TabsTrigger value="curl">cURL</TabsTrigger>
          </TabsList>
          <TabsContent value="python">
            <div className="relative">
              <Button
                variant="ghost"
                size="sm"
                className="absolute top-2 right-2 z-10"
                onClick={() => void copy(python, 'python')}
              >
                {copied === 'python' ? (
                  <>
                    <Check className="h-3 w-3" /> copied
                  </>
                ) : (
                  <>
                    <Copy className="h-3 w-3" /> copy
                  </>
                )}
              </Button>
              <pre className="text-xs bg-muted p-3 pr-20 rounded-md overflow-x-auto font-mono leading-relaxed">
                {python}
              </pre>
            </div>
          </TabsContent>
          <TabsContent value="sql">
            <div className="relative">
              <Button
                variant="ghost"
                size="sm"
                className="absolute top-2 right-2 z-10"
                onClick={() => void copy(sql, 'sql')}
              >
                {copied === 'sql' ? (
                  <>
                    <Check className="h-3 w-3" /> copied
                  </>
                ) : (
                  <>
                    <Copy className="h-3 w-3" /> copy
                  </>
                )}
              </Button>
              <pre className="text-xs bg-muted p-3 pr-20 rounded-md overflow-x-auto font-mono leading-relaxed">
                {sql}
              </pre>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              <code className="font-mono">ai_query</code> runs from any Databricks SQL warehouse
              or notebook SQL cell. Useful for batch inference over a Delta table.
            </p>
          </TabsContent>
          <TabsContent value="curl">
            <div className="relative">
              <Button
                variant="ghost"
                size="sm"
                className="absolute top-2 right-2 z-10"
                onClick={() => void copy(curl, 'curl')}
              >
                {copied === 'curl' ? (
                  <>
                    <Check className="h-3 w-3" /> copied
                  </>
                ) : (
                  <>
                    <Copy className="h-3 w-3" /> copy
                  </>
                )}
              </Button>
              <pre className="text-xs bg-muted p-3 pr-20 rounded-md overflow-x-auto font-mono leading-relaxed">
                {curl}
              </pre>
            </div>
          </TabsContent>
        </Tabs>
        <p className="text-xs text-muted-foreground mt-2">
          In a Databricks notebook, <code className="font-mono">DATABRICKS_TOKEN</code> is
          auto-injected — no manual auth needed.
        </p>
      </CollapsibleContent>
    </Collapsible>
  );
}
