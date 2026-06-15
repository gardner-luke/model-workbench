import { createApp, server } from '@databricks/appkit';
import { WorkspaceClient } from '@databricks/sdk-experimental';
import type { Request, Response } from 'express';
import express from 'express';
import { getEndpointMetadata } from './endpoint-metadata.js';

type ContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } };

interface InvokeBody {
  messages?: Array<{
    role: 'system' | 'user' | 'assistant';
    content: string | ContentPart[];
  }>;
  max_tokens?: number;
  temperature?: number;
}

type EmbedInput = { type: 'text' | 'image'; value: string };
interface EmbedBody {
  inputs?: EmbedInput[];
}

interface SegmentBody {
  image?: string;
  text_prompt?: string;
  threshold?: number;
  mask_threshold?: number;
}

interface DetectBody {
  image?: string;
  text_prompt?: string;
  threshold?: number;
}

interface DepthBody {
  image?: string;
}

type EndpointKind = 'foundation_model' | 'custom' | 'external_model' | 'unknown';
type Modality =
  | 'text'
  | 'multimodal'
  | 'text_embedding'
  | 'multimodal_embedding'
  | 'segmentation'
  | 'object_detection'
  | 'depth_estimation'
  | 'unknown';

interface EndpointInfo {
  name: string;
  task: string | null;
  ready: boolean;
  state: string | null;
  configUpdate: string | null;
  kind: EndpointKind;
  modality: Modality;
  modelName: string | null;
  description: string | null;
  creator: string | null;
  // Curated metadata from endpoint-metadata.ts. Defaults to empty/zeros if no
  // pattern matches the endpoint name.
  speed: number;
  cost: number;
  quality: number;
  recommendedFor: string | null;
  curatedDescription: string | null;
  modelCardUrl: string | null;
}

function classifyKind(entity: {
  foundation_model?: unknown;
  external_model?: unknown;
} | undefined): EndpointKind {
  if (!entity) return 'unknown';
  if (entity.foundation_model) return 'foundation_model';
  if (entity.external_model) return 'external_model';
  return 'custom';
}

const MULTIMODAL_CHAT_PATTERNS = [
  /\bclaude\b/, // Claude family is multimodal
  /\bllama-?4\b/,
  /\bgemma-?3\b/,
  /\bgemini-?[12]\.[05]\b/, // Gemini 1.5+, 2.x
  /\bgpt-?4o\b/,
  /\bgpt-?4\.1\b/,
  /\bgpt-?5\b/,
  /\bqwen.*vl\b/,
  /\bllava\b/,
  /\bpixtral\b/,
];

// Embedding models whose vector space accepts BOTH text and images
// (text and image embeddings live in the same shared space → comparable via cosine).
const MULTIMODAL_EMBEDDING_PATTERNS = [/\bclip\b/, /\bsiglip\b/, /\bblip\b/, /\bimagebind\b/];

// Image segmentation models with promptable concept output (masks + boxes + scores).
// SAM 3 / 2 / 1 family — text-promptable concept segmentation that returns masks.
const SEGMENTATION_PATTERNS = [/\bsam-?\d?\b/, /grounded-?sam/];

// Object detection models — returns boxes + scores + labels, no masks.
// Includes both open-vocab (Grounding DINO, OWL-ViT, etc.) and closed-vocab (YOLO, YOLOS, DETR).
const DETECTION_PATTERNS = [
  /grounding-?dino/,
  /\byolo/,
  /\bdetr\b/,
  /\bowl(-?v?\d+)?\b/,
];

// Monocular depth estimation — image in, per-pixel depth map out.
const DEPTH_PATTERNS = [/depth-?anything/, /\bdepth\b/, /\bmidas\b/, /\bzoedepth\b/];

function classifyModality(
  endpointName: string | null,
  task: string | null,
  modelName: string | null,
): Modality {
  const t = (task ?? '').toLowerCase();
  const n = (modelName ?? '').toLowerCase();
  const e = (endpointName ?? '').toLowerCase();
  if (DETECTION_PATTERNS.some((re) => re.test(n) || re.test(e))) return 'object_detection';
  if (DEPTH_PATTERNS.some((re) => re.test(n) || re.test(e))) return 'depth_estimation';
  if (SEGMENTATION_PATTERNS.some((re) => re.test(n) || re.test(e))) return 'segmentation';
  const isMultimodalEmbedder =
    MULTIMODAL_EMBEDDING_PATTERNS.some((re) => re.test(n) || re.test(e));
  if (isMultimodalEmbedder) return 'multimodal_embedding';
  if (t.includes('embed')) return 'text_embedding';
  if (MULTIMODAL_CHAT_PATTERNS.some((re) => re.test(n))) return 'multimodal';
  if (t.includes('chat') || t.includes('completion')) return 'text';
  return 'unknown';
}

const KIND_ORDER: Record<EndpointKind, number> = {
  foundation_model: 0,
  custom: 1,
  external_model: 2,
  unknown: 3,
};

const appkit = await createApp({
  plugins: [server({ autoStart: false })],
});

// AppKit's server plugin registers a default express.json() in start() with the
// body-parser default 100kb limit. It only parses when the request's Content-Type
// includes the substring "json" — so heavy routes use Content-Type:
// application/octet-stream from the client, AppKit's parser skips them, and we
// parse the raw body here with a 64 MiB cap.
const HEAVY_BODY_LIMIT = '64mb';
const rawJsonParser = express.raw({ type: 'application/octet-stream', limit: HEAVY_BODY_LIMIT });

// Tag every outbound invocation so endpoint_usage.usage_context attributes the call
// back to this app. The cost dashboard filters on this to exclude FMAPI traffic
// from other apps/users in the same workspace.
const APP_USAGE_CONTEXT = { app: 'model-workbench' };

function parseRawJsonBody<T>(req: Request): T {
  const buf = req.body as unknown;
  if (Buffer.isBuffer(buf) && buf.length > 0) {
    return JSON.parse(buf.toString('utf-8')) as T;
  }
  return {} as T;
}

appkit.server.extend((app) => {

  app.get('/api/workspace', (_req: Request, res: Response) => {
    // Expose the workspace host (used by code snippets) and any companion
    // dashboard URLs (used by the top-nav Analytics link).
    const host = process.env.DATABRICKS_HOST ?? '';
    const dashboardUrl = process.env.DASHBOARD_URL ?? '';
    res.json({ host, dashboardUrl });
  });

  app.get('/api/endpoints', async (_req: Request, res: Response) => {
    try {
      const ws = new WorkspaceClient({});
      const endpoints: EndpointInfo[] = [];

      for await (const ep of ws.servingEndpoints.list()) {
        const entity = ep.config?.served_entities?.[0];
        const modelName =
          entity?.foundation_model?.name ??
          entity?.external_model?.name ??
          entity?.entity_name ??
          entity?.name ??
          null;

        const task = ep.task ?? null;
        const meta = getEndpointMetadata(ep.name ?? null);
        endpoints.push({
          name: ep.name ?? '',
          task,
          ready: ep.state?.ready === 'READY',
          state: ep.state?.ready ?? null,
          configUpdate: ep.state?.config_update ?? null,
          kind: classifyKind(entity),
          modality: classifyModality(ep.name ?? null, task, modelName),
          modelName,
          description: ep.description ?? null,
          creator: ep.creator ?? null,
          speed: meta.speed,
          cost: meta.cost,
          quality: meta.quality,
          recommendedFor: meta.recommendedFor ?? null,
          curatedDescription: meta.description || null,
          modelCardUrl: meta.modelCardUrl ?? null,
        });
      }

      endpoints.sort((a, b) => {
        const byKind = KIND_ORDER[a.kind] - KIND_ORDER[b.kind];
        return byKind !== 0 ? byKind : a.name.localeCompare(b.name);
      });

      res.json({ endpoints });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error('list endpoints error', err);
      res.status(500).json({ error: message });
    }
  });

  app.post('/api/embed/:name', rawJsonParser, async (req: Request<{ name: string }>, res: Response) => {
    const { name } = req.params;
    const body = parseRawJsonBody<EmbedBody>(req);
    const inputs = body.inputs ?? [];

    if (inputs.length === 0) {
      res.status(400).json({ error: 'inputs[] is required' });
      return;
    }

    // Detect multimodal-embedding endpoints by name (CLIP/SigLIP/etc.).
    // Standard text-embedding endpoints (GTE/BGE/Qwen3) use the OpenAI-style `input` field.
    const isMultimodal = MULTIMODAL_EMBEDDING_PATTERNS.some((re) => re.test(name.toLowerCase()));
    const hasImages = inputs.some((i) => i.type === 'image');

    if (hasImages && !isMultimodal) {
      res
        .status(400)
        .json({ error: `Endpoint "${name}" does not accept image inputs.` });
      return;
    }

    try {
      const ws = new WorkspaceClient({});

      const payload: Record<string, unknown> = isMultimodal
        ? { dataframe_records: inputs, usage_context: APP_USAGE_CONTEXT }
        : { input: inputs.map((i) => i.value), usage_context: APP_USAGE_CONTEXT };

      const raw = await ws.apiClient.request({
        path: `/serving-endpoints/${encodeURIComponent(name)}/invocations`,
        method: 'POST',
        headers: new Headers({ 'Content-Type': 'application/json', Accept: 'application/json' }),
        raw: false,
        payload,
      });

      // Normalize response across the two endpoint shapes.
      let embeddings: number[][] = [];
      let dim = 0;
      const r = raw as Record<string, unknown>;
      if (isMultimodal) {
        // CLIP wrapper returns one vector per input row, MLflow wraps as
        // `{"predictions": [[...], [...]]}`. We also accept legacy shapes:
        //   `{"predictions": {"embeddings": [...], "dim": N}}`
        //   `{"embeddings": [...], "dim": N}`
        const preds = r.predictions;
        if (Array.isArray(preds)) {
          embeddings = preds as number[][];
        } else if (preds && typeof preds === 'object') {
          embeddings = (preds as { embeddings?: number[][] }).embeddings ?? [];
        } else if (Array.isArray(r.embeddings)) {
          embeddings = r.embeddings as number[][];
        }
        dim = embeddings[0]?.length ?? 0;
      } else {
        const data = (r.data as Array<{ embedding?: number[] }> | undefined) ?? [];
        embeddings = data.map((d) => d.embedding ?? []);
        dim = embeddings[0]?.length ?? 0;
      }

      res.json({ embeddings, dim, raw });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`embed ${name} error`, err);
      res.status(500).json({ error: message });
    }
  });

  app.post('/api/segment/:name', rawJsonParser, async (req: Request<{ name: string }>, res: Response) => {
    const { name } = req.params;
    const body = parseRawJsonBody<SegmentBody>(req);

    if (!body.image) {
      res.status(400).json({ error: '`image` (base64) is required' });
      return;
    }

    try {
      const ws = new WorkspaceClient({});
      const record: Record<string, unknown> = { image: body.image };
      if (body.text_prompt) record.text_prompt = body.text_prompt;
      if (typeof body.threshold === 'number') record.threshold = body.threshold;
      if (typeof body.mask_threshold === 'number') record.mask_threshold = body.mask_threshold;

      const raw = await ws.apiClient.request({
        path: `/serving-endpoints/${encodeURIComponent(name)}/invocations`,
        method: 'POST',
        headers: new Headers({ 'Content-Type': 'application/json', Accept: 'application/json' }),
        raw: false,
        payload: { dataframe_records: [record], usage_context: APP_USAGE_CONTEXT },
      });

      // Endpoint returns either {predictions: [{...}]} (most common for pyfunc) or {predictions: {...}}.
      const r = raw as Record<string, unknown>;
      let pred: Record<string, unknown> | null = null;
      if (Array.isArray(r.predictions) && r.predictions.length > 0) {
        pred = r.predictions[0] as Record<string, unknown>;
      } else if (r.predictions && typeof r.predictions === 'object') {
        pred = r.predictions as Record<string, unknown>;
      } else {
        pred = r;
      }

      const masks = (pred?.masks as string[] | undefined) ?? [];
      const boxes = (pred?.boxes as number[][] | undefined) ?? [];
      const scores = (pred?.scores as number[] | undefined) ?? [];
      const count = (pred?.count as number | undefined) ?? masks.length;
      const imageSize = (pred?.image_size as [number, number] | undefined) ?? [0, 0];

      res.json({ masks, boxes, scores, count, image_size: imageSize, raw });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`segment ${name} error`, err);
      res.status(500).json({ error: message });
    }
  });

  app.post('/api/depth/:name', rawJsonParser, async (req: Request<{ name: string }>, res: Response) => {
    const { name } = req.params;
    const body = parseRawJsonBody<DepthBody>(req);

    if (!body.image) {
      res.status(400).json({ error: '`image` (base64) is required' });
      return;
    }

    try {
      const ws = new WorkspaceClient({});
      const raw = await ws.apiClient.request({
        path: `/serving-endpoints/${encodeURIComponent(name)}/invocations`,
        method: 'POST',
        headers: new Headers({ 'Content-Type': 'application/json', Accept: 'application/json' }),
        raw: false,
        payload: { dataframe_records: [{ image: body.image }], usage_context: APP_USAGE_CONTEXT },
      });

      const r = raw as Record<string, unknown>;
      let pred: Record<string, unknown> | null = null;
      if (Array.isArray(r.predictions) && r.predictions.length > 0) {
        pred = r.predictions[0] as Record<string, unknown>;
      } else if (r.predictions && typeof r.predictions === 'object') {
        pred = r.predictions as Record<string, unknown>;
      } else {
        pred = r;
      }

      const depthPng = (pred?.depth_png as string | undefined) ?? '';
      const minDepth = (pred?.min_depth as number | undefined) ?? 0;
      const maxDepth = (pred?.max_depth as number | undefined) ?? 0;
      const imageSize = (pred?.image_size as [number, number] | undefined) ?? [0, 0];

      res.json({
        depth_png: depthPng,
        min_depth: minDepth,
        max_depth: maxDepth,
        image_size: imageSize,
        raw,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`depth ${name} error`, err);
      res.status(500).json({ error: message });
    }
  });

  app.post('/api/detect/:name', rawJsonParser, async (req: Request<{ name: string }>, res: Response) => {
    const { name } = req.params;
    const body = parseRawJsonBody<DetectBody>(req);

    if (!body.image) {
      res.status(400).json({ error: '`image` (base64) is required' });
      return;
    }

    try {
      const ws = new WorkspaceClient({});
      const record: Record<string, unknown> = { image: body.image };
      if (body.text_prompt) record.text_prompt = body.text_prompt;
      if (typeof body.threshold === 'number') record.threshold = body.threshold;

      const raw = await ws.apiClient.request({
        path: `/serving-endpoints/${encodeURIComponent(name)}/invocations`,
        method: 'POST',
        headers: new Headers({ 'Content-Type': 'application/json', Accept: 'application/json' }),
        raw: false,
        payload: { dataframe_records: [record], usage_context: APP_USAGE_CONTEXT },
      });

      // The wrapper returns one prediction record per input row, so the live
      // response shape is `{"predictions": [{...detection result...}]}`.
      const r = raw as Record<string, unknown>;
      let pred: Record<string, unknown> | null = null;
      if (Array.isArray(r.predictions) && r.predictions.length > 0) {
        pred = r.predictions[0] as Record<string, unknown>;
      } else if (r.predictions && typeof r.predictions === 'object') {
        pred = r.predictions as Record<string, unknown>;
      } else {
        pred = r;
      }

      const boxes = (pred?.boxes as number[][] | undefined) ?? [];
      const scores = (pred?.scores as number[] | undefined) ?? [];
      const labels = (pred?.labels as string[] | undefined) ?? [];
      const count = (pred?.count as number | undefined) ?? boxes.length;
      const imageSize = (pred?.image_size as [number, number] | undefined) ?? [0, 0];

      res.json({ boxes, scores, labels, count, image_size: imageSize, raw });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`detect ${name} error`, err);
      res.status(500).json({ error: message });
    }
  });

  app.post('/api/invoke/:name', rawJsonParser, async (req: Request<{ name: string }>, res: Response) => {
    const { name } = req.params;
    const body = parseRawJsonBody<InvokeBody>(req);

    if (!body.messages || body.messages.length === 0) {
      res.status(400).json({ error: 'messages[] is required' });
      return;
    }

    try {
      const ws = new WorkspaceClient({});
      // Use apiClient.request directly so we can send the OpenAI-style multimodal
      // content array (`[{type:'text',...}, {type:'image_url',...}]`). The SDK's
      // typed query() method restricts content to string and would reject images.
      const payload: Record<string, unknown> = {
        messages: body.messages,
        usage_context: APP_USAGE_CONTEXT,
      };
      if (body.max_tokens) payload.max_tokens = body.max_tokens;
      if (typeof body.temperature === 'number') payload.temperature = body.temperature;

      const raw = await ws.apiClient.request({
        path: `/serving-endpoints/${encodeURIComponent(name)}/invocations`,
        method: 'POST',
        headers: new Headers({ 'Content-Type': 'application/json', Accept: 'application/json' }),
        raw: false,
        payload,
      });

      const result = raw as {
        choices?: Array<{
          message?: { content?: string };
          finish_reason?: string;
          finishReason?: string;
        }>;
      };
      const choice = result.choices?.[0];
      const content = choice?.message?.content ?? '';
      const finishReason = choice?.finishReason ?? choice?.finish_reason ?? null;

      res.json({ content, finishReason, raw });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`invoke ${name} error`, err);
      res.status(500).json({ error: message });
    }
  });
});

await appkit.server.start();
