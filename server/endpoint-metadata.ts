// Static metadata layer for serving endpoints. The server matches each discovered
// endpoint name against the patterns below and decorates the API response with
// curated description, speed/cost/quality indicators, and recommended use cases.
//
// Speed: 1 (slow) → 4 (very fast)
// Cost:  1 (cheap) → 4 (premium)
// Quality: 1 (weak) → 4 (best-in-class)
//
// Editing tip: if you add a new endpoint and want it to show up with proper
// indicators, add an entry to METADATA_PATTERNS. Order matters — first match wins.

export interface EndpointMetadata {
  description: string;
  speed: number;
  cost: number;
  quality: number;
  recommendedFor?: string;
  modelCardUrl?: string;
}

interface MetadataPattern extends EndpointMetadata {
  match: RegExp;
}

const METADATA_PATTERNS: MetadataPattern[] = [
  // ─── Custom models we built ────────────────────────────────────────────────
  {
    match: /^clip-vit-large-patch14$/,
    description:
      'OpenAI CLIP ViT-L/14. Embeds text and images into a shared 768-d vector space — cosine similarity is comparable across modalities. Great for cross-modal search, zero-shot classification, content tagging.',
    speed: 4,
    cost: 1,
    quality: 4,
    recommendedFor: 'Cross-modal search, image clustering, zero-shot classification',
    modelCardUrl: 'https://huggingface.co/openai/clip-vit-large-patch14',
  },
  {
    match: /^sam3$/,
    description:
      "Meta's SAM 3 — promptable concept segmentation. Give it a noun phrase and an image, get a precise mask, bounding box, and confidence for every matching instance. No clicks needed.",
    speed: 2,
    cost: 2,
    quality: 4,
    recommendedFor: 'Auto-labeling, content moderation, robotics perception',
    modelCardUrl: 'https://huggingface.co/facebook/sam3',
  },
  {
    match: /^grounding-dino$/,
    description:
      "IDEA Research Grounding DINO. Open-vocabulary object detection — type any noun phrase and get bounding boxes for every matching instance. Faster + cheaper than SAM 3 when you just need 'where is the thing' without precise masks.",
    speed: 3,
    cost: 1,
    quality: 4,
    recommendedFor: 'Open-vocab detection, pre-filtering for downstream segmentation',
    modelCardUrl: 'https://huggingface.co/IDEA-Research/grounding-dino-base',
  },
  {
    match: /^yolos$/,
    description:
      'YOLOS (transformer-based YOLO) trained on COCO 80 classes. No prompt — image in, boxes out for every person/car/object it recognizes. The closed-vocab, fixed-class baseline. Fast and small.',
    speed: 4,
    cost: 1,
    quality: 3,
    recommendedFor: 'Real-time inference, fixed-class detection, simple object counting',
    modelCardUrl: 'https://huggingface.co/hustvl/yolos-small',
  },
  {
    match: /^depth-anything$/,
    description:
      "Meta's Depth Anything V2 — monocular depth estimation. Single image in, dense per-pixel depth out. Useful for 3D scene understanding, robotics, AR, and any pipeline that needs spatial context from a single camera.",
    speed: 3,
    cost: 1,
    quality: 4,
    recommendedFor: 'Robotics perception, 3D reconstruction, AR effects, scene understanding',
    modelCardUrl: 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf',
  },

  // ─── Foundation Model APIs — chat ──────────────────────────────────────────
  // GPT-5 family
  {
    match: /^databricks-gpt-5-5-pro$/,
    description:
      "OpenAI GPT-5.5 Pro. Flagship reasoning model — best for hard multi-step problems where quality matters more than latency.",
    speed: 1,
    cost: 4,
    quality: 4,
    recommendedFor: 'Complex reasoning, code generation, deep research',
    modelCardUrl: 'https://openai.com/index/introducing-gpt-5/',
  },
  {
    match: /^databricks-gpt-5-(5|4|3-codex|2-codex)$/,
    description:
      'OpenAI GPT-5 family. Strong general-purpose multimodal model. Good balance of quality and speed.',
    speed: 2,
    cost: 3,
    quality: 4,
    recommendedFor: 'General assistant, code, multimodal Q&A',
    modelCardUrl: 'https://openai.com/index/introducing-gpt-5/',
  },
  {
    match: /^databricks-gpt-5(-1|-2)?$/,
    description: 'OpenAI GPT-5 baseline. Solid multimodal model for most tasks.',
    speed: 2,
    cost: 3,
    quality: 4,
    modelCardUrl: 'https://openai.com/index/introducing-gpt-5/',
  },
  {
    match: /^databricks-gpt-5-mini$/,
    description: 'OpenAI GPT-5 Mini. Cheaper and faster than full GPT-5, still strong multimodal.',
    speed: 3,
    cost: 2,
    quality: 3,
    modelCardUrl: 'https://openai.com/index/introducing-gpt-5/',
  },
  {
    match: /^databricks-gpt-5-4-mini$/,
    description: 'OpenAI GPT-5.4 Mini. Cost-effective multimodal model for everyday workloads.',
    speed: 3,
    cost: 2,
    quality: 3,
    modelCardUrl: 'https://openai.com/index/introducing-gpt-5/',
  },
  {
    match: /^databricks-gpt-5(-4)?-nano$/,
    description: 'OpenAI GPT-5 Nano. Smallest, fastest, cheapest GPT-5 variant.',
    speed: 4,
    cost: 1,
    quality: 2,
    recommendedFor: 'High-volume light tasks, classification',
    modelCardUrl: 'https://openai.com/index/introducing-gpt-5/',
  },

  // Claude family
  {
    match: /^databricks-claude-opus-4-7$/,
    description:
      "Anthropic's Claude Opus 4.7. Top-tier reasoning and tool use, especially strong for coding and long-context tasks. Multimodal.",
    speed: 1,
    cost: 4,
    quality: 4,
    recommendedFor: 'Code generation, complex reasoning, long-context analysis',
    modelCardUrl: 'https://www.anthropic.com/claude',
  },
  {
    match: /^databricks-claude-opus-4-[1-6]$/,
    description: 'Anthropic Claude Opus 4.x. Premium reasoning + multimodal capability.',
    speed: 1,
    cost: 4,
    quality: 4,
    modelCardUrl: 'https://www.anthropic.com/claude',
  },
  {
    match: /^databricks-claude-sonnet-4-6$/,
    description:
      "Anthropic's Claude Sonnet 4.6. Excellent balance of quality, speed, and cost. Multimodal — works great with images.",
    speed: 2,
    cost: 3,
    quality: 4,
    recommendedFor: 'General assistant, vision tasks, agent backbone',
    modelCardUrl: 'https://www.anthropic.com/claude',
  },
  {
    match: /^databricks-claude-sonnet-4(-5)?$/,
    description: 'Anthropic Claude Sonnet 4.x. Reliable multimodal mid-tier model.',
    speed: 2,
    cost: 3,
    quality: 4,
    modelCardUrl: 'https://www.anthropic.com/claude',
  },
  {
    match: /^databricks-claude-haiku-4-5$/,
    description: 'Anthropic Claude Haiku 4.5. Fast, cheap multimodal model for high-volume use.',
    speed: 3,
    cost: 2,
    quality: 3,
    recommendedFor: 'Real-time chat, classification, light multimodal',
    modelCardUrl: 'https://www.anthropic.com/claude',
  },

  // Gemini
  {
    match: /^databricks-gemini-2-5-pro$/,
    description: "Google Gemini 2.5 Pro. Long-context (~1M tokens) and strong reasoning.",
    speed: 1,
    cost: 4,
    quality: 4,
    recommendedFor: 'Very long-context, complex reasoning',
    modelCardUrl: 'https://deepmind.google/technologies/gemini/',
  },
  {
    match: /^databricks-gemini-2-5-flash$/,
    description: 'Google Gemini 2.5 Flash. Cheap and fast multimodal model.',
    speed: 3,
    cost: 2,
    quality: 3,
    modelCardUrl: 'https://deepmind.google/technologies/gemini/',
  },

  // Gemma
  {
    match: /^databricks-gemma-3-12b$/,
    description:
      "Google Gemma 3 12B. Open-weight multimodal model. Cost-effective for in-house workloads.",
    speed: 3,
    cost: 1,
    quality: 3,
    recommendedFor: 'Cost-sensitive multimodal tasks, batch inference',
    modelCardUrl: 'https://huggingface.co/google/gemma-3-12b-it',
  },

  // Llama family
  {
    match: /^databricks-llama-4-maverick$/,
    description:
      "Meta's Llama 4 Maverick. Open-weight multimodal model with mixture-of-experts architecture. Strong on vision + text.",
    speed: 2,
    cost: 2,
    quality: 3,
    recommendedFor: 'Multimodal Q&A, open-weight compliance requirements',
    modelCardUrl: 'https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct',
  },
  {
    match: /^databricks-meta-llama-3\.1-405b/,
    description: "Meta Llama 3.1 405B. Largest open Llama model. Strong text-only reasoning.",
    speed: 1,
    cost: 3,
    quality: 4,
    modelCardUrl: 'https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct',
  },
  {
    match: /^databricks-meta-llama-3-3-70b/,
    description:
      "Meta Llama 3.3 70B. Open-weight text model. Good for fine-tuning and on-prem workloads.",
    speed: 3,
    cost: 1,
    quality: 3,
    modelCardUrl: 'https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct',
  },
  {
    match: /^databricks-meta-llama-3-1-8b/,
    description: 'Meta Llama 3.1 8B Instruct. Small open-weight model for cheap workloads.',
    speed: 4,
    cost: 1,
    quality: 2,
    modelCardUrl: 'https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct',
  },

  // GPT-OSS (open-source ChatGPT-class)
  {
    match: /^databricks-gpt-oss-120b$/,
    description:
      "OpenAI's open-weight GPT-OSS 120B. Free to run, strong baseline quality.",
    speed: 2,
    cost: 0,
    quality: 3,
    recommendedFor: 'Compliance-sensitive workloads, fine-tuning starting point',
    modelCardUrl: 'https://huggingface.co/openai/gpt-oss-120b',
  },
  {
    match: /^databricks-gpt-oss-20b$/,
    description: "OpenAI's open-weight GPT-OSS 20B. Smaller, faster, cheaper sibling.",
    speed: 4,
    cost: 0,
    quality: 2,
    modelCardUrl: 'https://huggingface.co/openai/gpt-oss-20b',
  },

  // Qwen
  {
    match: /^databricks-qwen35-122b/,
    description: 'Alibaba Qwen 3.5 122B. Strong open-weight multilingual model.',
    speed: 2,
    cost: 1,
    quality: 3,
    modelCardUrl: 'https://huggingface.co/Qwen',
  },
  {
    match: /^databricks-qwen3-next-80b/,
    description:
      'Alibaba Qwen 3 Next 80B Instruct. Open-weight, large-context model.',
    speed: 3,
    cost: 1,
    quality: 3,
    modelCardUrl: 'https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct',
  },

  // ─── Foundation Model APIs — embeddings ────────────────────────────────────
  {
    match: /^databricks-bge-large-en$/,
    description:
      'BAAI BGE Large EN v1.5. Industry-standard English text embedding model (1024-d). Strong on semantic search.',
    speed: 4,
    cost: 1,
    quality: 3,
    recommendedFor: 'RAG retrieval, semantic search, clustering',
    modelCardUrl: 'https://huggingface.co/BAAI/bge-large-en-v1.5',
  },
  {
    match: /^databricks-gte-large-en$/,
    description: 'GTE Large EN v1.5. Alibaba general-text embedding model (1024-d).',
    speed: 4,
    cost: 1,
    quality: 3,
    modelCardUrl: 'https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5',
  },
  {
    match: /^databricks-qwen3-embedding-0-6b$/,
    description:
      'Qwen3 Embedding 0.6B. Multilingual embedding model from Alibaba.',
    speed: 4,
    cost: 1,
    quality: 3,
    modelCardUrl: 'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B',
  },
];

const DEFAULT_METADATA: EndpointMetadata = {
  description: '',
  speed: 0,
  cost: 0,
  quality: 0,
};

export function getEndpointMetadata(name: string | null): EndpointMetadata {
  if (!name) return DEFAULT_METADATA;
  for (const p of METADATA_PATTERNS) {
    if (p.match.test(name)) {
      const { match: _match, ...rest } = p;
      return rest;
    }
  }
  return DEFAULT_METADATA;
}
