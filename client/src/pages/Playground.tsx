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
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
  Input,
  Label,
  Spinner,
  Textarea,
} from '@databricks/appkit-ui/react';
import { ArrowLeft, ChevronDown, Send, User, Bot, Paperclip, X } from 'lucide-react';
import type { AttachedImage, ChatMessage, ContentPart, EndpointInfo, InvokeResponse } from '../types';
import { prepareImage } from '../lib/image';
import { IndicatorRow } from '../components/ModelIndicators';
import { NotebookSnippet } from '../components/NotebookSnippet';
import { ModelCardLink } from '../components/ModelCardLink';

const DEFAULT_MAX_TOKENS = 1024;
const DEFAULT_TEMPERATURE = 0.7;

interface ChatTurn {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  images?: AttachedImage[];
  raw?: unknown;
  finishReason?: string | null;
}

const MAX_INPUT_BYTES = 32 * 1024 * 1024; // 32 MB per source file (we downscale)
const MAX_IMAGES_PER_TURN = 4;

function makeTurnId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function PlaygroundPage() {
  const { name = '' } = useParams<{ name: string }>();
  const navigate = useNavigate();

  const [endpoint, setEndpoint] = useState<EndpointInfo | null>(null);
  const [endpointError, setEndpointError] = useState<string | null>(null);

  const [systemPrompt, setSystemPrompt] = useState('');
  const [input, setInput] = useState('');
  const [maxTokens, setMaxTokens] = useState<number>(DEFAULT_MAX_TOKENS);
  const [temperature, setTemperature] = useState<number>(DEFAULT_TEMPERATURE);

  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [sending, setSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const [pendingImages, setPendingImages] = useState<AttachedImage[]>([]);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const supportsImages = endpoint?.modality === 'multimodal';

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

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [turns.length]);

  const addImages = useCallback(
    async (files: FileList | File[]) => {
      const fileArr = Array.from(files);
      setSendError(null);
      const added: AttachedImage[] = [];
      for (const f of fileArr) {
        if (pendingImages.length + added.length >= MAX_IMAGES_PER_TURN) {
          setSendError(`Max ${MAX_IMAGES_PER_TURN} images per message.`);
          break;
        }
        if (!f.type.startsWith('image/')) {
          setSendError(`"${f.name}" is not an image.`);
          continue;
        }
        if (f.size > MAX_INPUT_BYTES) {
          setSendError(`"${f.name}" exceeds ${MAX_INPUT_BYTES / 1024 / 1024} MB.`);
          continue;
        }
        try {
          const prepared = await prepareImage(f);
          added.push({ id: makeTurnId(), name: f.name, dataUrl: prepared.dataUrl });
        } catch (err: unknown) {
          setSendError(err instanceof Error ? err.message : String(err));
        }
      }
      if (added.length > 0) {
        setPendingImages((prev) => [...prev, ...added]);
      }
    },
    [pendingImages.length],
  );

  const removePendingImage = useCallback((id: string) => {
    setPendingImages((prev) => prev.filter((img) => img.id !== id));
  }, []);

  const send = useCallback(async () => {
    const trimmed = input.trim();
    if ((!trimmed && pendingImages.length === 0) || sending) return;

    const turnImages = pendingImages;
    const userTurn: ChatTurn = {
      id: makeTurnId(),
      role: 'user',
      content: trimmed,
      images: turnImages.length > 0 ? turnImages : undefined,
    };
    const nextTurns: ChatTurn[] = [...turns, userTurn];
    setTurns(nextTurns);
    setInput('');
    setPendingImages([]);
    setSendError(null);
    setSending(true);

    const messages: ChatMessage[] = [];
    if (systemPrompt.trim()) {
      messages.push({ role: 'system', content: systemPrompt.trim() });
    }
    for (const t of nextTurns) {
      if (t.images && t.images.length > 0) {
        const parts: ContentPart[] = [];
        if (t.content) {
          parts.push({ type: 'text', text: t.content });
        }
        for (const img of t.images) {
          parts.push({ type: 'image_url', image_url: { url: img.dataUrl } });
        }
        messages.push({ role: t.role, content: parts });
      } else {
        messages.push({ role: t.role, content: t.content });
      }
    }

    try {
      const resp = await fetch(`/api/invoke/${encodeURIComponent(name)}`, {
        method: 'POST',
        // Non-json Content-Type bypasses AppKit's default 100kb json parser; the
        // server reads the raw body and parses with a 64 MiB limit.
        headers: { 'Content-Type': 'application/octet-stream' },
        body: JSON.stringify({
          messages,
          max_tokens: maxTokens || undefined,
          temperature: typeof temperature === 'number' ? temperature : undefined,
        }),
      });
      if (!resp.ok) {
        const body = (await resp.json().catch(() => ({}))) as { error?: string };
        throw new Error(body.error ?? `HTTP ${resp.status}`);
      }
      const data = (await resp.json()) as InvokeResponse;
      setTurns((prev) => [
        ...prev,
        {
          id: makeTurnId(),
          role: 'assistant',
          content: data.content,
          raw: data.raw,
          finishReason: data.finishReason,
        },
      ]);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setSendError(message);
      // Roll back the optimistic user turn so the box is editable
      setTurns(turns);
      setInput(trimmed);
      setPendingImages(turnImages);
    } finally {
      setSending(false);
    }
  }, [input, sending, turns, systemPrompt, maxTokens, temperature, name, pendingImages]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        void send();
      }
    },
    [send],
  );

  const headerMeta = useMemo(() => {
    if (!endpoint) return null;
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground flex-wrap">
        {endpoint.modelName && endpoint.modelName !== endpoint.name && (
          <span className="font-mono">{endpoint.modelName}</span>
        )}
        <Badge variant="outline">{endpoint.task ?? 'unknown'}</Badge>
        <Badge variant={endpoint.modality === 'multimodal' ? 'default' : 'outline'}>
          {endpoint.modality}
        </Badge>
        <ModelCardLink url={endpoint.modelCardUrl} />
      </div>
    );
  }, [endpoint]);

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
    <div className="max-w-4xl mx-auto space-y-4">
      <Button variant="ghost" size="sm" onClick={() => void navigate('/')}>
        <ArrowLeft className="h-4 w-4" /> Registry
      </Button>

      <Card>
        <CardHeader>
          <CardTitle>{endpoint.name}</CardTitle>
          {headerMeta}
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
          <Collapsible>
            <CollapsibleTrigger asChild>
              <Button variant="outline" size="sm">
                <ChevronDown className="h-4 w-4" /> System prompt & parameters
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3 space-y-3">
              <div>
                <Label htmlFor="system">System prompt</Label>
                <Textarea
                  id="system"
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  placeholder="Optional. e.g. 'You are a helpful assistant.'"
                  rows={2}
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label htmlFor="max">Max tokens</Label>
                  <Input
                    id="max"
                    type="number"
                    min={1}
                    max={32000}
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(Number(e.target.value))}
                  />
                </div>
                <div>
                  <Label htmlFor="temp">Temperature</Label>
                  <Input
                    id="temp"
                    type="number"
                    min={0}
                    max={2}
                    step={0.1}
                    value={temperature}
                    onChange={(e) => setTemperature(Number(e.target.value))}
                  />
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>

          <div
            ref={scrollRef}
            className="border rounded-md p-4 space-y-4 max-h-[420px] overflow-y-auto bg-muted/20"
          >
            {turns.length === 0 && (
              <p className="text-sm text-muted-foreground text-center py-6">
                Send a message to start. Cmd/Ctrl+Enter sends.
              </p>
            )}
            {turns.map((t) => (
              <ChatBubble key={t.id} turn={t} />
            ))}
            {sending && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Spinner /> Calling endpoint…
              </div>
            )}
          </div>

          {sendError && (
            <Alert variant="destructive">
              <AlertDescription>{sendError}</AlertDescription>
            </Alert>
          )}

          {pendingImages.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {pendingImages.map((img) => (
                <div
                  key={img.id}
                  className="relative border rounded-md overflow-hidden bg-muted"
                  title={img.name}
                >
                  <img src={img.dataUrl} alt={img.name} className="h-16 w-16 object-cover" />
                  <button
                    type="button"
                    onClick={() => removePendingImage(img.id)}
                    className="absolute top-0.5 right-0.5 bg-background/90 rounded-full p-0.5 hover:bg-background"
                    aria-label={`Remove ${img.name}`}
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <div className="flex gap-2">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder={supportsImages ? 'Type a message or attach images…' : 'Type a message…'}
              rows={3}
              disabled={sending}
            />
            <div className="flex flex-col gap-2">
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
                        void addImages(e.target.files);
                      }
                      e.target.value = '';
                    }}
                  />
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={sending || pendingImages.length >= MAX_IMAGES_PER_TURN}
                    title="Attach images"
                  >
                    <Paperclip className="h-4 w-4" />
                  </Button>
                </>
              )}
              <Button
                onClick={() => void send()}
                disabled={sending || (!input.trim() && pendingImages.length === 0)}
                className="flex-1"
              >
                <Send className="h-4 w-4" /> Send
              </Button>
            </div>
          </div>

          <NotebookSnippet endpointName={endpoint.name} modality={endpoint.modality} />
        </CardContent>
      </Card>
    </div>
  );
}

function ChatBubble({ turn }: { turn: ChatTurn }) {
  const isUser = turn.role === 'user';
  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className="flex-shrink-0 mt-1">
        {isUser ? <User className="h-5 w-5" /> : <Bot className="h-5 w-5" />}
      </div>
      <div className={`flex-1 ${isUser ? 'text-right' : ''}`}>
        {turn.images && turn.images.length > 0 && (
          <div className={`flex flex-wrap gap-2 mb-2 ${isUser ? 'justify-end' : ''}`}>
            {turn.images.map((img) => (
              <img
                key={img.id}
                src={img.dataUrl}
                alt={img.name}
                className="h-24 w-24 object-cover rounded-md border"
              />
            ))}
          </div>
        )}
        {turn.content && (
          <div
            className={`inline-block max-w-full px-3 py-2 rounded-lg whitespace-pre-wrap text-sm text-left ${
              isUser ? 'bg-primary text-primary-foreground' : 'bg-background border'
            }`}
          >
            {turn.content}
          </div>
        )}
        {turn.finishReason && turn.finishReason !== 'stop' && (
          <div className="text-xs text-muted-foreground mt-1">finish: {turn.finishReason}</div>
        )}
        {!isUser && turn.raw !== undefined && (
          <Collapsible>
            <CollapsibleTrigger asChild>
              <button className="text-xs text-muted-foreground mt-1 hover:underline">
                raw response
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <pre className="text-xs bg-muted p-2 rounded mt-1 overflow-x-auto max-w-full">
                {JSON.stringify(turn.raw, null, 2)}
              </pre>
            </CollapsibleContent>
          </Collapsible>
        )}
      </div>
    </div>
  );
}
