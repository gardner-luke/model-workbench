export type EndpointKind = 'foundation_model' | 'custom' | 'external_model' | 'unknown';
export type Modality =
  | 'text'
  | 'multimodal'
  | 'text_embedding'
  | 'multimodal_embedding'
  | 'segmentation'
  | 'object_detection'
  | 'depth_estimation'
  | 'unknown';

export function isChatModality(m: Modality): boolean {
  return m === 'text' || m === 'multimodal';
}

export function isEmbeddingModality(m: Modality): boolean {
  return m === 'text_embedding' || m === 'multimodal_embedding';
}

export function isSegmentationModality(m: Modality): boolean {
  return m === 'segmentation';
}

export function isDetectionModality(m: Modality): boolean {
  return m === 'object_detection';
}

export function isDepthModality(m: Modality): boolean {
  return m === 'depth_estimation';
}

export interface EndpointInfo {
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
  // Curated metadata. 0 means "unrated / unknown".
  speed: number;
  cost: number;
  quality: number;
  recommendedFor: string | null;
  curatedDescription: string | null;
  modelCardUrl: string | null;
}

export type ContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } };

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | ContentPart[];
}

export interface AttachedImage {
  id: string;
  name: string;
  dataUrl: string;
}

export interface InvokeRequest {
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
}

export interface InvokeResponse {
  content: string;
  finishReason: string | null;
  raw: unknown;
}

export function isChatEndpoint(ep: { task: string | null; modality: Modality }): boolean {
  return isChatModality(ep.modality);
}

export interface EmbedRequest {
  inputs: Array<{ type: 'text' | 'image'; value: string }>;
}

export interface EmbedResponse {
  embeddings: number[][];
  dim: number;
  raw: unknown;
}

export interface SegmentRequest {
  image: string;
  text_prompt?: string;
  threshold?: number;
  mask_threshold?: number;
}

export interface SegmentResponse {
  masks: string[]; // base64 PNGs
  boxes: number[][]; // [[x1, y1, x2, y2], ...]
  scores: number[];
  count: number;
  image_size: [number, number]; // [width, height]
  raw: unknown;
}

export interface DetectRequest {
  image: string;
  text_prompt?: string;
  threshold?: number;
}

export interface DetectResponse {
  boxes: number[][];
  scores: number[];
  labels: string[];
  count: number;
  image_size: [number, number];
  raw: unknown;
}

export interface DepthRequest {
  image: string;
}

export interface DepthResponse {
  depth_png: string; // base64 grayscale PNG, image-sized
  min_depth: number;
  max_depth: number;
  image_size: [number, number];
  raw: unknown;
}
