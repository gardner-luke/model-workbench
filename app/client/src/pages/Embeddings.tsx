import { useCallback, useEffect, useRef, useState } from 'react';
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
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
  Input,
  Spinner,
} from '@databricks/appkit-ui/react';
import { ArrowLeft, ChevronDown, ImagePlus, Plus, Sparkles, Trash2 } from 'lucide-react';
import type { EmbedResponse, EndpointInfo } from '../types';
import { prepareImage } from '../lib/image';
import { IndicatorRow } from '../components/ModelIndicators';
import { NotebookSnippet } from '../components/NotebookSnippet';
import { ScaleToZeroCallout } from '../components/ScaleToZeroCallout';
import { ModelCardLink } from '../components/ModelCardLink';
import { SamplePicker } from '../components/SamplePicker';

const MAX_INPUT_BYTES = 32 * 1024 * 1024;
const MAX_INPUTS = 16;

type EmbedInput =
  | { id: string; kind: 'text'; value: string }
  | { id: string; kind: 'image'; value: string; name: string };

function makeId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // CLIP/GTE/BGE vectors are L2-normalized; raw dot = cosine.
}

function similarityColor(value: number): string {
  // value in [-1, 1]. Map to a Databricks-ish blue gradient.
  const clamped = Math.max(-1, Math.min(1, value));
  const t = (clamped + 1) / 2; // 0..1
  const r = Math.round(255 - 130 * t);
  const g = Math.round(255 - 100 * t);
  const b = Math.round(255 - 30 * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function shortLabel(inp: EmbedInput, idx: number): string {
  if (inp.kind === 'image') return inp.name;
  const trimmed = inp.value.trim();
  if (!trimmed) return `text ${idx + 1}`;
  return trimmed.length > 32 ? `${trimmed.slice(0, 30)}…` : trimmed;
}

export function EmbeddingsPage() {
  const { name = '' } = useParams<{ name: string }>();
  const navigate = useNavigate();

  const [endpoint, setEndpoint] = useState<EndpointInfo | null>(null);
  const [endpointError, setEndpointError] = useState<string | null>(null);

  const [inputs, setInputs] = useState<EmbedInput[]>([
    { id: makeId(), kind: 'text', value: '' },
    { id: makeId(), kind: 'text', value: '' },
  ]);
  const [result, setResult] = useState<EmbedResponse | null>(null);
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

  const supportsImages = endpoint?.modality === 'multimodal_embedding';

  const updateInput = useCallback((id: string, value: string) => {
    setInputs((prev) =>
      prev.map((i) => (i.id === id && i.kind === 'text' ? { ...i, value } : i)),
    );
  }, []);

  const removeInput = useCallback((id: string) => {
    setInputs((prev) => prev.filter((i) => i.id !== id));
  }, []);

  const addTextInput = useCallback(() => {
    setInputs((prev) => {
      if (prev.length >= MAX_INPUTS) return prev;
      return [...prev, { id: makeId(), kind: 'text', value: '' }];
    });
  }, []);

  const addImageInputs = useCallback(async (files: FileList | File[]) => {
    const fileArr = Array.from(files);
    setSubmitError(null);
    const added: EmbedInput[] = [];
    for (const f of fileArr) {
      if (!f.type.startsWith('image/')) {
        setSubmitError(`"${f.name}" is not an image.`);
        continue;
      }
      if (f.size > MAX_INPUT_BYTES) {
        setSubmitError(`"${f.name}" exceeds ${MAX_INPUT_BYTES / 1024 / 1024} MB.`);
        continue;
      }
      try {
        // CLIP downsamples to 224 internally — 768px is more than enough.
        const prepared = await prepareImage(f, 768, 0.9);
        added.push({ id: makeId(), kind: 'image', value: prepared.dataUrl, name: f.name });
      } catch (err: unknown) {
        setSubmitError(err instanceof Error ? err.message : String(err));
      }
    }
    if (added.length > 0) {
      setInputs((prev) => [...prev, ...added].slice(0, MAX_INPUTS));
    }
  }, []);

  const compute = useCallback(async () => {
    const nonEmpty = inputs.filter((i) => (i.kind === 'image' ? !!i.value : !!i.value.trim()));
    if (nonEmpty.length < 1 || submitting) return;

    setSubmitError(null);
    setSubmitting(true);
    setResult(null);

    const payload = {
      inputs: nonEmpty.map((i) =>
        i.kind === 'text'
          ? { type: 'text' as const, value: i.value }
          : { type: 'image' as const, value: i.value },
      ),
    };

    try {
      const resp = await fetch(`/api/embed/${encodeURIComponent(name)}`, {
        method: 'POST',
        // Non-json Content-Type bypasses AppKit's default 100kb json parser; the
        // server reads the raw body and parses with a 64 MiB limit.
        headers: { 'Content-Type': 'application/octet-stream' },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const body = (await resp.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? `HTTP ${resp.status}`);
      }
      const data = (await resp.json()) as EmbedResponse;
      setResult(data);
    } catch (err: unknown) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }, [inputs, name, submitting]);

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

  const nonEmptyInputs = inputs.filter((i) => (i.kind === 'image' ? !!i.value : !!i.value.trim()));

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
            <Badge variant="outline">{endpoint.task ?? 'embedding'}</Badge>
            <Badge variant={supportsImages ? 'default' : 'secondary'}>
              {supportsImages ? 'Multimodal embedding' : 'Text embedding'}
            </Badge>
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
          <div className="space-y-2">
            {inputs.map((inp, idx) => (
              <div key={inp.id} className="flex gap-2 items-start">
                <div className="text-xs text-muted-foreground font-mono pt-2.5 w-8 text-right">
                  {idx + 1}
                </div>
                {inp.kind === 'text' ? (
                  <Input
                    value={inp.value}
                    onChange={(e) => updateInput(inp.id, e.target.value)}
                    placeholder="Text input…"
                    disabled={submitting}
                  />
                ) : (
                  <div className="flex-1 flex items-center gap-2 border rounded-md p-2 bg-muted/30">
                    <img
                      src={inp.value}
                      alt={inp.name}
                      className="h-10 w-10 object-cover rounded"
                    />
                    <span className="text-sm truncate" title={inp.name}>
                      {inp.name}
                    </span>
                  </div>
                )}
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => removeInput(inp.id)}
                  disabled={submitting || inputs.length <= 1}
                  aria-label={`Remove input ${idx + 1}`}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>

          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={addTextInput}
              disabled={submitting || inputs.length >= MAX_INPUTS}
            >
              <Plus className="h-4 w-4" /> Add text
            </Button>
            {supportsImages && (
              <>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) {
                      void addImageInputs(e.target.files);
                    }
                    e.target.value = '';
                  }}
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={submitting || inputs.length >= MAX_INPUTS}
                >
                  <ImagePlus className="h-4 w-4" /> Add image
                </Button>
                <SamplePicker
                  modality="embedding"
                  onPick={async (file, prompt) => {
                    await addImageInputs([file]);
                    if (prompt) {
                      setInputs((prev) => {
                        if (prev.length >= MAX_INPUTS) return prev;
                        const blankIdx = prev.findIndex(
                          (i) => i.kind === 'text' && !i.value.trim(),
                        );
                        if (blankIdx >= 0) {
                          const next = [...prev];
                          const item = next[blankIdx];
                          if (item.kind === 'text') {
                            next[blankIdx] = { ...item, value: prompt };
                          }
                          return next;
                        }
                        return [...prev, { id: makeId(), kind: 'text', value: prompt }];
                      });
                    }
                  }}
                  disabled={submitting || inputs.length >= MAX_INPUTS}
                />
              </>
            )}
            <div className="flex-1" />
            <Button onClick={() => void compute()} disabled={submitting || nonEmptyInputs.length < 1}>
              {submitting ? <Spinner /> : <Sparkles className="h-4 w-4" />}
              Compute embeddings
            </Button>
          </div>

          {submitError && (
            <Alert variant="destructive">
              <AlertDescription>{submitError}</AlertDescription>
            </Alert>
          )}

          {result && (
            <ResultPanel
              result={result}
              labelInputs={nonEmptyInputs.map((i, idx) => ({ inp: i, label: shortLabel(i, idx) }))}
            />
          )}

          <NotebookSnippet endpointName={endpoint.name} modality={endpoint.modality} />
        </CardContent>
      </Card>
    </div>
  );
}

interface LabeledInput {
  inp: EmbedInput;
  label: string;
}

function ResultPanel({
  result,
  labelInputs,
}: {
  result: EmbedResponse;
  labelInputs: LabeledInput[];
}) {
  const n = result.embeddings.length;
  const showMatrix = n >= 2;
  return (
    <div className="space-y-4 border-t pt-4">
      <div className="text-sm text-muted-foreground">
        Returned {n} embeddings · {result.dim} dimensions each
      </div>

      {showMatrix && (
        <div>
          <div className="text-sm font-medium mb-2">Cosine similarity</div>
          <div className="overflow-x-auto">
            <table className="text-xs border-collapse">
              <thead>
                <tr>
                  <th className="p-1" />
                  {labelInputs.map((l, j) => (
                    <th key={l.inp.id} className="p-1 text-left max-w-[120px]">
                      <div className="text-[10px] text-muted-foreground">{j + 1}</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {labelInputs.map((rowLabel, i) => (
                  <tr key={rowLabel.inp.id}>
                    <td className="p-1 pr-2 text-right max-w-[260px] truncate" title={rowLabel.label}>
                      <span className="text-[10px] text-muted-foreground mr-1">{i + 1}</span>
                      {rowLabel.inp.kind === 'image' ? (
                        <span className="inline-flex items-center gap-1">
                          <img
                            src={rowLabel.inp.value}
                            alt={rowLabel.inp.name}
                            className="h-4 w-4 inline object-cover rounded"
                          />
                          <span className="truncate">{rowLabel.label}</span>
                        </span>
                      ) : (
                        <span className="truncate">{rowLabel.label}</span>
                      )}
                    </td>
                    {labelInputs.map((colLabel, j) => {
                      const sim = cosineSimilarity(result.embeddings[i], result.embeddings[j]);
                      return (
                        <td
                          key={`${rowLabel.inp.id}-${colLabel.inp.id}`}
                          className="p-1 text-center min-w-[56px] font-mono"
                          style={{ backgroundColor: similarityColor(sim) }}
                          title={`${rowLabel.label} ↔ ${colLabel.label}`}
                        >
                          {sim.toFixed(3)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <Collapsible>
        <CollapsibleTrigger asChild>
          <Button variant="outline" size="sm">
            <ChevronDown className="h-4 w-4" /> Raw vectors
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-2 space-y-2">
          {result.embeddings.map((emb, i) => {
            const labeled = labelInputs[i];
            const key = labeled ? labeled.inp.id : `vec-${i}`;
            const label = labeled ? labeled.label : `vector ${i + 1}`;
            return (
              <div key={key} className="text-xs">
                <div className="text-muted-foreground mb-1">
                  {i + 1}. {label} (showing first 8 of {emb.length})
                </div>
                <pre className="font-mono bg-muted p-2 rounded overflow-x-auto">
                  [{emb.slice(0, 8).map((v) => v.toFixed(4)).join(', ')}…]
                </pre>
              </div>
            );
          })}
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
