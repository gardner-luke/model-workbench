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
  Spinner,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@databricks/appkit-ui/react';
import { ArrowLeft, ImagePlus, Sparkles, X } from 'lucide-react';
import type { DepthResponse, EndpointInfo } from '../types';
import { dataUrlToBase64, prepareImage } from '../lib/image';
import { IndicatorRow } from '../components/ModelIndicators';
import { NotebookSnippet } from '../components/NotebookSnippet';
import { ScaleToZeroCallout } from '../components/ScaleToZeroCallout';
import { ModelCardLink } from '../components/ModelCardLink';
import { SamplePicker } from '../components/SamplePicker';

const MAX_INPUT_BYTES = 32 * 1024 * 1024;

interface LoadedImage {
  name: string;
  dataUrl: string;
  width: number;
  height: number;
  resized: boolean;
  originalWidth: number;
  originalHeight: number;
}

export function DepthPage() {
  const { name = '' } = useParams<{ name: string }>();
  const navigate = useNavigate();

  const [endpoint, setEndpoint] = useState<EndpointInfo | null>(null);
  const [endpointError, setEndpointError] = useState<string | null>(null);

  const [image, setImage] = useState<LoadedImage | null>(null);
  const [result, setResult] = useState<DepthResponse | null>(null);
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
      const resp = await fetch(`/api/depth/${encodeURIComponent(name)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: JSON.stringify({ image: dataUrlToBase64(image.dataUrl) }),
      });
      if (!resp.ok) {
        const body = (await resp.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? `HTTP ${resp.status}`);
      }
      const data = (await resp.json()) as DepthResponse;
      setResult(data);
    } catch (err: unknown) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }, [image, name, submitting]);

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
            <Badge>Depth estimation</Badge>
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
                onPick={async (file) => {
                  await setImageFromFile(file);
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

          <div>
            <Button onClick={() => void submit()} disabled={!image || submitting}>
              {submitting ? <Spinner /> : <Sparkles className="h-4 w-4" />}
              Estimate depth
            </Button>
          </div>

          {submitError && (
            <Alert variant="destructive">
              <AlertDescription>{submitError}</AlertDescription>
            </Alert>
          )}

          {image && result && (
            <DepthVisualizer image={image.dataUrl} result={result} />
          )}

          <NotebookSnippet endpointName={endpoint.name} modality={endpoint.modality} />
        </CardContent>
      </Card>
    </div>
  );
}

interface DepthVisualizerProps {
  image: string;
  result: DepthResponse;
}

// Turbo colormap stops — perceptually uniform, good for depth.
// Source: https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html (8-stop approximation).
const TURBO_STOPS: Array<[number, number, number]> = [
  [48, 18, 59],
  [70, 107, 227],
  [27, 208, 213],
  [101, 254, 124],
  [221, 220, 49],
  [253, 137, 38],
  [220, 47, 14],
  [122, 4, 3],
];

function turboColor(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  const idx = x * (TURBO_STOPS.length - 1);
  const i = Math.floor(idx);
  const f = idx - i;
  const a = TURBO_STOPS[i];
  const b = TURBO_STOPS[Math.min(i + 1, TURBO_STOPS.length - 1)];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

function DepthVisualizer({ image, result }: DepthVisualizerProps) {
  const heatmapRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function render() {
      try {
        if (!result.depth_png) return;

        const depthImg = new Image();
        await new Promise<void>((resolve, reject) => {
          depthImg.onload = () => resolve();
          depthImg.onerror = () => reject(new Error('Failed to decode depth map'));
          depthImg.src = `data:image/png;base64,${result.depth_png}`;
        });

        if (cancelled) return;

        const [serverW, serverH] = result.image_size;
        const width = serverW || depthImg.naturalWidth;
        const height = serverH || depthImg.naturalHeight;

        // Read grayscale pixels from the depth PNG once.
        const probe = document.createElement('canvas');
        probe.width = width;
        probe.height = height;
        const probeCtx = probe.getContext('2d');
        if (!probeCtx) return;
        probeCtx.drawImage(depthImg, 0, 0, width, height);
        const pixels = probeCtx.getImageData(0, 0, width, height).data;

        // Build a turbo-colormapped version for the heatmap canvas.
        const heatmap = heatmapRef.current;
        if (heatmap && !cancelled) {
          heatmap.width = width;
          heatmap.height = height;
          const ctx = heatmap.getContext('2d');
          if (ctx) {
            const out = ctx.createImageData(width, height);
            for (let i = 0; i < pixels.length; i += 4) {
              const v = pixels[i] / 255; // grayscale already
              const [r, g, b] = turboColor(v);
              out.data[i] = r;
              out.data[i + 1] = g;
              out.data[i + 2] = b;
              out.data[i + 3] = 255;
            }
            ctx.putImageData(out, 0, 0);
          }
        }

        // Build an alpha-blended overlay of the depth heatmap over the source image.
        const overlay = overlayRef.current;
        if (overlay && !cancelled) {
          const src = new Image();
          await new Promise<void>((resolve, reject) => {
            src.onload = () => resolve();
            src.onerror = () => reject(new Error('Failed to load source image'));
            src.src = image;
          });
          if (cancelled) return;
          overlay.width = width;
          overlay.height = height;
          const ctx = overlay.getContext('2d');
          if (ctx) {
            ctx.drawImage(src, 0, 0, width, height);
            // Color-map the depth and draw it at 55% opacity on top.
            const tint = document.createElement('canvas');
            tint.width = width;
            tint.height = height;
            const tctx = tint.getContext('2d');
            if (tctx) {
              const out = tctx.createImageData(width, height);
              for (let i = 0; i < pixels.length; i += 4) {
                const v = pixels[i] / 255;
                const [r, g, b] = turboColor(v);
                out.data[i] = r;
                out.data[i + 1] = g;
                out.data[i + 2] = b;
                out.data[i + 3] = 255;
              }
              tctx.putImageData(out, 0, 0);
              ctx.globalAlpha = 0.55;
              ctx.drawImage(tint, 0, 0);
              ctx.globalAlpha = 1.0;
            }
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
  }, [image, result]);

  const aspectStyle = useMemo(() => {
    const [w, h] = result.image_size;
    if (!w || !h) return {};
    return { aspectRatio: `${w} / ${h}` };
  }, [result.image_size]);

  return (
    <div className="space-y-3">
      <div className="text-sm text-muted-foreground">
        Depth range:{' '}
        <span className="font-mono">
          {result.min_depth.toFixed(3)} → {result.max_depth.toFixed(3)}
        </span>{' '}
        · {result.image_size[0]}×{result.image_size[1]} · brighter = closer
      </div>
      <Tabs defaultValue="heatmap">
        <TabsList>
          <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
          <TabsTrigger value="overlay">Overlay</TabsTrigger>
        </TabsList>
        <TabsContent value="heatmap">
          <div className="border rounded-md overflow-hidden bg-muted/30 max-w-full" style={aspectStyle}>
            <canvas
              ref={heatmapRef}
              className="block max-w-full h-auto"
              style={{ width: '100%', height: 'auto' }}
            />
          </div>
        </TabsContent>
        <TabsContent value="overlay">
          <div className="border rounded-md overflow-hidden bg-muted/30 max-w-full" style={aspectStyle}>
            <canvas
              ref={overlayRef}
              className="block max-w-full h-auto"
              style={{ width: '100%', height: 'auto' }}
            />
          </div>
        </TabsContent>
      </Tabs>
      {renderError && (
        <Alert variant="destructive">
          <AlertDescription>{renderError}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}
