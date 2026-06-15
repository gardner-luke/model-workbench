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
import type { EndpointInfo, SegmentResponse } from '../types';
import { dataUrlToBase64, prepareImage } from '../lib/image';
import { IndicatorRow } from '../components/ModelIndicators';
import { NotebookSnippet } from '../components/NotebookSnippet';
import { ScaleToZeroCallout } from '../components/ScaleToZeroCallout';
import { ModelCardLink } from '../components/ModelCardLink';
import { SamplePicker } from '../components/SamplePicker';

// Source file cap before downscaling. We resize anyway, so this is just a
// sanity bound — 32 MB allows large iPhone HEIC originals.
const MAX_INPUT_BYTES = 32 * 1024 * 1024;

// Distinct hues for instance masks. 8 colors then cycle.
const INSTANCE_COLORS = [
  [251, 113, 133], // rose
  [56, 189, 248], // sky
  [251, 191, 36], // amber
  [134, 239, 172], // green
  [196, 181, 253], // violet
  [252, 165, 165], // red-300
  [125, 211, 252], // sky-300
  [253, 224, 71], // yellow
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

export function SegmentationPage() {
  const { name = '' } = useParams<{ name: string }>();
  const navigate = useNavigate();

  const [endpoint, setEndpoint] = useState<EndpointInfo | null>(null);
  const [endpointError, setEndpointError] = useState<string | null>(null);

  const [image, setImage] = useState<LoadedImage | null>(null);
  const [textPrompt, setTextPrompt] = useState('');
  const [threshold, setThreshold] = useState(0.3);
  const [maskThreshold, setMaskThreshold] = useState(0.5);
  const [showBoxes, setShowBoxes] = useState(true);

  const [result, setResult] = useState<SegmentResponse | null>(null);
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
      const resp = await fetch(`/api/segment/${encodeURIComponent(name)}`, {
        method: 'POST',
        // Non-json Content-Type bypasses AppKit's default 100kb json parser; the
        // server reads the raw body and parses with a 64 MiB limit.
        headers: { 'Content-Type': 'application/octet-stream' },
        body: JSON.stringify({
          image: dataUrlToBase64(image.dataUrl),
          text_prompt: textPrompt.trim() || undefined,
          threshold,
          mask_threshold: maskThreshold,
        }),
      });
      if (!resp.ok) {
        const body = (await resp.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? `HTTP ${resp.status}`);
      }
      const data = (await resp.json()) as SegmentResponse;
      setResult(data);
    } catch (err: unknown) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }, [image, name, submitting, textPrompt, threshold, maskThreshold]);

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
          <div className="flex items-center gap-2 text-sm text-muted-foreground flex-wrap">
            {endpoint.modelName && endpoint.modelName !== endpoint.name && (
              <span className="font-mono">{endpoint.modelName}</span>
            )}
            <Badge>Segmentation</Badge>
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
                modality="segmentation"
                onPick={async (file, prompt) => {
                  await setImageFromFile(file);
                  if (prompt && !textPrompt.trim()) setTextPrompt(prompt);
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

          <div className="space-y-2">
            <Label htmlFor="prompt">Concept prompt (noun phrase)</Label>
            <Input
              id="prompt"
              value={textPrompt}
              onChange={(e) => setTextPrompt(e.target.value)}
              placeholder='e.g. "bottle", "pallet", "scratch", "person on conveyor"…'
              disabled={submitting}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <Label className="flex justify-between">
                <span>Presence threshold</span>
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
            <div className="space-y-1">
              <Label className="flex justify-between">
                <span>Mask threshold</span>
                <span className="text-muted-foreground font-mono">{maskThreshold.toFixed(2)}</span>
              </Label>
              <Slider
                value={[maskThreshold]}
                min={0}
                max={1}
                step={0.05}
                onValueChange={(v) => setMaskThreshold(v[0])}
                disabled={submitting}
              />
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button onClick={() => void submit()} disabled={!image || submitting}>
              {submitting ? <Spinner /> : <Sparkles className="h-4 w-4" />}
              Segment
            </Button>
            {result && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowBoxes((b) => !b)}
                disabled={submitting}
              >
                {showBoxes ? 'Hide' : 'Show'} boxes
              </Button>
            )}
            {result && (
              <span className="text-sm text-muted-foreground">
                {result.count} {result.count === 1 ? 'instance' : 'instances'} ·{' '}
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
            <MaskOverlay
              image={image.dataUrl}
              imageSize={result.image_size}
              masks={result.masks}
              boxes={result.boxes}
              scores={result.scores}
              showBoxes={showBoxes}
            />
          )}

          {result && result.count > 0 && (
            <div className="border-t pt-3 space-y-1">
              <div className="text-sm font-medium mb-2">Detections</div>
              <div className="flex flex-wrap gap-2">
                {result.scores.map((s, i) => {
                  const box = result.boxes[i] ?? [0, 0, 0, 0];
                  const boxKey = `${i}-${box[0]}-${box[1]}-${box[2]}-${box[3]}`;
                  return (
                    <Badge
                      key={boxKey}
                      variant="outline"
                      style={{
                        borderLeft: `4px solid rgb(${INSTANCE_COLORS[i % INSTANCE_COLORS.length].join(',')})`,
                      }}
                    >
                      #{i + 1}: {s.toFixed(3)}
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

interface MaskOverlayProps {
  image: string;
  imageSize: [number, number];
  masks: string[];
  boxes: number[][];
  scores: number[];
  showBoxes: boolean;
}

function MaskOverlay({ image, imageSize, masks, boxes, scores, showBoxes }: MaskOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const canvas = canvasRef.current;
    if (!canvas) return;

    async function render() {
      try {
        const src = new Image();
        src.crossOrigin = 'anonymous';
        await new Promise<void>((resolve, reject) => {
          src.onload = () => resolve();
          src.onerror = () => reject(new Error('Failed to load source image'));
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

        for (let i = 0; i < masks.length; i++) {
          if (cancelled) return;
          const maskB64 = masks[i];
          if (!maskB64) continue;
          const maskImg = new Image();
          maskImg.crossOrigin = 'anonymous';
          await new Promise<void>((resolve, reject) => {
            maskImg.onload = () => resolve();
            maskImg.onerror = () => reject(new Error(`Failed to decode mask ${i}`));
            maskImg.src = `data:image/png;base64,${maskB64}`;
          });

          // Render mask as a colored translucent overlay by drawing it to an offscreen
          // canvas, then using source-in to recolor white pixels.
          const off = document.createElement('canvas');
          off.width = width;
          off.height = height;
          const offCtx = off.getContext('2d');
          if (!offCtx) continue;
          offCtx.drawImage(maskImg, 0, 0, width, height);
          const [r, g, b] = INSTANCE_COLORS[i % INSTANCE_COLORS.length];
          offCtx.globalCompositeOperation = 'source-in';
          offCtx.fillStyle = `rgba(${r}, ${g}, ${b}, 1)`;
          offCtx.fillRect(0, 0, width, height);

          ctx.globalAlpha = 0.45;
          ctx.drawImage(off, 0, 0);
          ctx.globalAlpha = 1.0;
        }

        if (showBoxes) {
          ctx.lineWidth = Math.max(2, width / 400);
          ctx.font = `${Math.max(12, width / 60)}px sans-serif`;
          for (let i = 0; i < boxes.length; i++) {
            const [x1, y1, x2, y2] = boxes[i];
            const [r, g, b] = INSTANCE_COLORS[i % INSTANCE_COLORS.length];
            ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            const label = `#${i + 1} ${scores[i]?.toFixed(2) ?? ''}`;
            const textWidth = ctx.measureText(label).width + 6;
            const fontSize = Math.max(12, width / 60);
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.85)`;
            ctx.fillRect(x1, Math.max(0, y1 - fontSize - 4), textWidth, fontSize + 4);
            ctx.fillStyle = '#fff';
            ctx.fillText(label, x1 + 3, Math.max(fontSize, y1 - 4));
          }
        }
      } catch (err: unknown) {
        if (!cancelled) setRenderError(err instanceof Error ? err.message : String(err));
      }
    }

    void render();
    return () => {
      cancelled = true;
    };
  }, [image, imageSize, masks, boxes, scores, showBoxes]);

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
