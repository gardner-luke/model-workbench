import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router';
import {
  Alert,
  AlertDescription,
  Badge,
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Input,
  Label,
  Slider,
  Spinner,
} from '@databricks/appkit-ui/react';
import { ArrowLeft, ImagePlus, Sparkles, X } from 'lucide-react';
import type { DetectResponse, EndpointInfo } from '../types';
import { dataUrlToBase64, prepareImage } from '../lib/image';
import { IndicatorRow } from '../components/ModelIndicators';
import { NotebookSnippet } from '../components/NotebookSnippet';
import { ScaleToZeroCallout } from '../components/ScaleToZeroCallout';
import { ModelCardLink } from '../components/ModelCardLink';
import { SamplePicker } from '../components/SamplePicker';

const MAX_INPUT_BYTES = 32 * 1024 * 1024;

const INSTANCE_COLORS = [
  [251, 113, 133],
  [56, 189, 248],
  [251, 191, 36],
  [134, 239, 172],
  [196, 181, 253],
  [252, 165, 165],
  [125, 211, 252],
  [253, 224, 71],
] as const;

interface LoadedImage {
  name: string;
  dataUrl: string;
  width: number;
  height: number;
  resized: boolean;
  originalWidth: number;
  originalHeight: number;
}

// Heuristic — Grounding DINO, OWL etc. accept text concept prompts; YOLOS/DETR don't.
function isOpenVocab(endpointName: string, modelName: string | null): boolean {
  const hay = `${endpointName} ${modelName ?? ''}`.toLowerCase();
  return /grounding-?dino|owl(-?v?\d+)?/.test(hay);
}

export function DetectionPage() {
  const { name = '' } = useParams<{ name: string }>();
  const navigate = useNavigate();

  const [endpoint, setEndpoint] = useState<EndpointInfo | null>(null);
  const [endpointError, setEndpointError] = useState<string | null>(null);

  const [image, setImage] = useState<LoadedImage | null>(null);
  const [textPrompt, setTextPrompt] = useState('');
  const [threshold, setThreshold] = useState(0.3);

  const [result, setResult] = useState<DetectResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/endpoints')
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<{ endpoints: EndpointInfo[] }>;
      })
      .then((data) => {
        if (cancelled) return;
        const match = data.endpoints.find((e) => e.name === name);
        if (!match) {
          setEndpointError(`Endpoint "${name}" not found.`);
          return;
        }
        setEndpoint(match);
      })
      .catch((err: unknown) => {
        if (!cancelled) setEndpointError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [name]);

  const openVocab = endpoint ? isOpenVocab(endpoint.name, endpoint.modelName) : false;

  const setImageFromFile = useCallback(async (file: File) => {
    setSubmitError(null);
    if (!file.type.startsWith('image/')) {
      setSubmitError(`"${file.name}" is not an image.`);
      return;
    }
    if (file.size > MAX_INPUT_BYTES) {
      setSubmitError(`"${file.name}" exceeds ${MAX_INPUT_BYTES / 1024 / 1024} MB.`);
      return;
    }
    try {
      const prepared = await prepareImage(file);
      setImage({
        name: file.name,
        dataUrl: prepared.dataUrl,
        width: prepared.width,
        height: prepared.height,
        resized: prepared.resized,
        originalWidth: prepared.originalWidth,
        originalHeight: prepared.originalHeight,
      });
      setResult(null);
    } catch (err: unknown) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  const submit = useCallback(async () => {
    if (!image || submitting) return;
    setSubmitting(true);
    setSubmitError(null);
    setResult(null);
    try {
      const resp = await fetch(`/api/detect/${encodeURIComponent(name)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: JSON.stringify({
          image: dataUrlToBase64(image.dataUrl),
          text_prompt: openVocab && textPrompt.trim() ? textPrompt.trim() : undefined,
          threshold,
        }),
      });
      if (!resp.ok) {
        const body = (await resp.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? `HTTP ${resp.status}`);
      }
      const data = (await resp.json()) as DetectResponse;
      setResult(data);
    } catch (err: unknown) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }, [image, name, submitting, textPrompt, threshold, openVocab]);

  if (endpointError) {
    return (
      <div className="max-w-4xl mx-auto space-y-4">
        <Button variant="ghost" size="sm" onClick={() => void navigate('/')}>
          <ArrowLeft className="h-4 w-4" /> Registry
        </Button>
        <Alert variant="destructive">
          <AlertDescription>{endpointError}</AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!endpoint) {
    return (
      <div className="max-w-4xl mx-auto flex items-center gap-2 text-sm text-muted-foreground py-8 justify-center">
        <Spinner /> Loading endpoint…
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto space-y-4">
      <Button variant="ghost" size="sm" onClick={() => void navigate('/')}>
        <ArrowLeft className="h-4 w-4" /> Registry
      </Button>

      <Card>
        <CardHeader>
          <CardTitle>{endpoint.name}</CardTitle>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            {endpoint.modelName && endpoint.modelName !== endpoint.name && (
              <span className="font-mono">{endpoint.modelName}</span>
            )}
            <Badge>Object detection</Badge>
            <Badge variant="outline">{openVocab ? 'Open-vocab' : 'Closed-vocab'}</Badge>
            <ModelCardLink url={endpoint.modelCardUrl} />
          </div>
          {endpoint.curatedDescription && (
            <p className="text-sm text-muted-foreground mt-2 leading-relaxed">
              {endpoint.curatedDescription}
            </p>
          )}
          {(endpoint.speed > 0 || endpoint.cost > 0 || endpoint.quality > 0) && (
            <div className="mt-3 max-w-md">
              <IndicatorRow speed={endpoint.speed} cost={endpoint.cost} quality={endpoint.quality} />
            </div>
          )}
          {endpoint.recommendedFor && (
            <p className="text-xs text-muted-foreground mt-2">
              <span className="font-medium text-foreground">Good for:</span> {endpoint.recommendedFor}
            </p>
          )}
        </CardHeader>
        <CardContent className="space-y-4">
          {endpoint.kind === 'custom' && <ScaleToZeroCallout />}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void setImageFromFile(f);
              e.target.value = '';
            }}
          />
          {!image ? (
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="outline" onClick={() => fileInputRef.current?.click()}>
                <ImagePlus className="h-4 w-4" /> Upload image
              </Button>
              <SamplePicker
                modality="detection"
                onPick={async (file, prompt) => {
                  await setImageFromFile(file);
                  if (openVocab && prompt && !textPrompt.trim()) setTextPrompt(prompt);
                }}
                disabled={submitting}
              />
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <img
                src={image.dataUrl}
                alt={image.name}
                className="h-12 w-12 object-cover rounded border"
              />
              <div className="flex flex-col min-w-0">
                <span className="text-sm truncate" title={image.name}>
                  {image.name}
                </span>
                <span className="text-xs text-muted-foreground">
                  {image.width}×{image.height}
                  {image.resized && ` (resized from ${image.originalWidth}×${image.originalHeight})`}
                </span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setImage(null);
                  setResult(null);
                }}
              >
                <X className="h-4 w-4" /> Replace
              </Button>
            </div>
          )}

          {openVocab && (
            <div className="space-y-2">
              <Label htmlFor="prompt">Concept prompt</Label>
              <Input
                id="prompt"
                value={textPrompt}
                onChange={(e) => setTextPrompt(e.target.value)}
                placeholder='e.g. "bottle. pallet. forklift. person."'
                disabled={submitting}
              />
              <p className="text-xs text-muted-foreground">
                Grounding DINO style: list concepts separated by periods.
              </p>
            </div>
          )}

          <div className="space-y-1 max-w-md">
            <Label className="flex justify-between">
              <span>Score threshold</span>
              <span className="text-muted-foreground font-mono">{threshold.toFixed(2)}</span>
            </Label>
            <Slider
              value={[threshold]}
              min={0}
              max={1}
              step={0.05}
              onValueChange={(v) => setThreshold(v[0])}
              disabled={submitting}
            />
          </div>

          <div className="flex items-center gap-2">
            <Button onClick={() => void submit()} disabled={!image || submitting}>
              {submitting ? <Spinner /> : <Sparkles className="h-4 w-4" />}
              Detect
            </Button>
            {result && (
              <span className="text-sm text-muted-foreground">
                {result.count} {result.count === 1 ? 'detection' : 'detections'} ·{' '}
                {result.image_size[0]}×{result.image_size[1]}
              </span>
            )}
          </div>

          {submitError && (
            <Alert variant="destructive">
              <AlertDescription>{submitError}</AlertDescription>
            </Alert>
          )}

          {image && result && (
            <BoxOverlay
              image={image.dataUrl}
              imageSize={result.image_size}
              boxes={result.boxes}
              scores={result.scores}
              labels={result.labels}
            />
          )}

          {result && result.count > 0 && (
            <div className="border-t pt-3 space-y-2">
              <div className="text-sm font-medium">Detections</div>
              <div className="flex flex-wrap gap-2">
                {result.scores.map((s, i) => {
                  const box = result.boxes[i] ?? [0, 0, 0, 0];
                  const key = `${i}-${box[0]}-${box[1]}-${box[2]}-${box[3]}`;
                  const color = INSTANCE_COLORS[i % INSTANCE_COLORS.length];
                  const label = result.labels[i] ?? `obj ${i + 1}`;
                  return (
                    <Badge
                      key={key}
                      variant="outline"
                      style={{ borderLeft: `4px solid rgb(${color.join(',')})` }}
                    >
                      {label}: {s.toFixed(3)}
                    </Badge>
                  );
                })}
              </div>
            </div>
          )}

          <NotebookSnippet endpointName={endpoint.name} modality={endpoint.modality} />
        </CardContent>
      </Card>
    </div>
  );
}

interface BoxOverlayProps {
  image: string;
  imageSize: [number, number];
  boxes: number[][];
  scores: number[];
  labels: string[];
}

function BoxOverlay({ image, imageSize, boxes, scores, labels }: BoxOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const canvas = canvasRef.current;
    if (!canvas) return;

    async function render() {
      try {
        const src = new Image();
        await new Promise<void>((resolve, reject) => {
          src.onload = () => resolve();
          src.onerror = () => reject(new Error('Failed to load image'));
          src.src = image;
        });

        const [serverW, serverH] = imageSize;
        const width = serverW || src.naturalWidth;
        const height = serverH || src.naturalHeight;
        if (!canvas || cancelled) return;
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.drawImage(src, 0, 0, width, height);

        ctx.lineWidth = Math.max(2, width / 400);
        const fontSize = Math.max(12, width / 60);
        ctx.font = `${fontSize}px sans-serif`;

        for (let i = 0; i < boxes.length; i++) {
          const [x1, y1, x2, y2] = boxes[i];
          const [r, g, b] = INSTANCE_COLORS[i % INSTANCE_COLORS.length];
          ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

          const label = `${labels[i] ?? `#${i + 1}`} ${(scores[i] ?? 0).toFixed(2)}`;
          const textWidth = ctx.measureText(label).width + 8;
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
          ctx.fillRect(x1, Math.max(0, y1 - fontSize - 4), textWidth, fontSize + 4);
          ctx.fillStyle = '#fff';
          ctx.fillText(label, x1 + 4, Math.max(fontSize, y1 - 4));
        }
      } catch (err: unknown) {
        if (!cancelled) setRenderError(err instanceof Error ? err.message : String(err));
      }
    }

    void render();
    return () => {
      cancelled = true;
    };
  }, [image, imageSize, boxes, scores, labels]);

  const aspectStyle = useMemo(() => {
    const [w, h] = imageSize;
    if (!w || !h) return {};
    return { aspectRatio: `${w} / ${h}` };
  }, [imageSize]);

  return (
    <div className="space-y-2">
      <div className="border rounded-md overflow-hidden bg-muted/30 max-w-full" style={aspectStyle}>
        <canvas
          ref={canvasRef}
          className="block max-w-full h-auto"
          style={{ width: '100%', height: 'auto' }}
        />
      </div>
      {renderError && (
        <Alert variant="destructive">
          <AlertDescription>{renderError}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}
