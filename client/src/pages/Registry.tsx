import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';
import {
  Alert,
  AlertDescription,
  Badge,
  Button,
  Card,
  CardContent,
  Input,
  Spinner,
  Toggle,
} from '@databricks/appkit-ui/react';
import {
  Brush,
  ExternalLink,
  ImageIcon,
  Layers,
  Mountain,
  MessageSquareText,
  Search,
  Sparkles,
  Target,
  Wand2,
  X,
} from 'lucide-react';
import type { EndpointInfo, EndpointKind, Modality } from '../types';
import {
  isChatEndpoint,
  isDepthModality,
  isDetectionModality,
  isEmbeddingModality,
  isSegmentationModality,
} from '../types';
import { IndicatorRow } from '../components/ModelIndicators';
import { FirstVisitHint } from '../components/FirstVisitHint';

const KIND_SHORT: Record<EndpointKind, string> = {
  foundation_model: 'FMAPI',
  custom: 'Custom',
  external_model: 'External',
  unknown: '—',
};

const MODALITY_LABEL: Record<Modality, string> = {
  text: 'Text',
  multimodal: 'Multimodal',
  text_embedding: 'Text embedding',
  multimodal_embedding: 'Multimodal embedding',
  segmentation: 'Segmentation',
  object_detection: 'Object detection',
  depth_estimation: 'Depth estimation',
  unknown: '—',
};

const MODALITY_ICON: Record<Modality, React.ComponentType<{ className?: string }>> = {
  text: MessageSquareText,
  multimodal: ImageIcon,
  text_embedding: Sparkles,
  multimodal_embedding: Wand2,
  segmentation: Brush,
  object_detection: Target,
  depth_estimation: Mountain,
  unknown: Layers,
};

const MODALITY_ACCENT: Record<Modality, string> = {
  text: 'text-slate-600 bg-slate-50',
  multimodal: 'text-violet-700 bg-violet-50',
  text_embedding: 'text-sky-700 bg-sky-50',
  multimodal_embedding: 'text-fuchsia-700 bg-fuchsia-50',
  segmentation: 'text-rose-700 bg-rose-50',
  object_detection: 'text-orange-700 bg-orange-50',
  depth_estimation: 'text-emerald-700 bg-emerald-50',
  unknown: 'text-muted-foreground bg-muted',
};

const KIND_SECTION_ORDER: EndpointKind[] = [
  'custom',
  'foundation_model',
  'external_model',
  'unknown',
];

const KIND_SECTION_TITLE: Record<EndpointKind, string> = {
  custom: 'Custom models',
  foundation_model: 'Foundation Model APIs',
  external_model: 'External models',
  unknown: 'Other',
};

// User-facing filter chips. Order matters for the UI.
const FILTER_MODALITIES: Modality[] = [
  'text',
  'multimodal',
  'text_embedding',
  'multimodal_embedding',
  'segmentation',
  'object_detection',
  'depth_estimation',
];

export function RegistryPage() {
  const navigate = useNavigate();
  const [endpoints, setEndpoints] = useState<EndpointInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [search, setSearch] = useState('');
  const [activeModalities, setActiveModalities] = useState<Set<Modality>>(new Set());
  const [readyOnly, setReadyOnly] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/endpoints')
      .then(async (r) => {
        if (!r.ok) {
          const body = (await r.json().catch(() => ({}))) as { error?: string };
          throw new Error(body.error ?? `HTTP ${r.status}`);
        }
        return r.json() as Promise<{ endpoints: EndpointInfo[] }>;
      })
      .then((data) => {
        if (!cancelled) setEndpoints(data.endpoints);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo(() => {
    if (!endpoints) return null;
    const q = search.trim().toLowerCase();
    return endpoints.filter((ep) => {
      if (activeModalities.size > 0 && !activeModalities.has(ep.modality)) return false;
      if (readyOnly && !ep.ready) return false;
      if (q) {
        const hay = `${ep.name} ${ep.modelName ?? ''} ${ep.task ?? ''}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [endpoints, search, activeModalities, readyOnly]);

  const grouped = useMemo(() => {
    if (!filtered) return null;
    const buckets: Record<EndpointKind, EndpointInfo[]> = {
      foundation_model: [],
      custom: [],
      external_model: [],
      unknown: [],
    };
    for (const ep of filtered) buckets[ep.kind].push(ep);
    return buckets;
  }, [filtered]);

  const toggleModality = (m: Modality) => {
    setActiveModalities((prev) => {
      const next = new Set(prev);
      if (next.has(m)) next.delete(m);
      else next.add(m);
      return next;
    });
  };

  const totalCount = endpoints?.length ?? 0;
  const visibleCount = filtered?.length ?? 0;

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="space-y-2">
        <h2 className="text-2xl font-semibold tracking-tight">Endpoints</h2>
        <p className="text-sm text-muted-foreground">
          Every model deployed in this workspace. Click a card to open its playground.
        </p>
      </div>

      <FirstVisitHint />

      <div className="flex flex-col gap-3">
        <div className="relative">
          <Search className="h-4 w-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by name, model, or task…"
            className="pl-9"
          />
          {search && (
            <button
              type="button"
              onClick={() => setSearch('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              aria-label="Clear search"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs uppercase tracking-wider text-muted-foreground mr-1">
            Modality
          </span>
          {FILTER_MODALITIES.map((m) => {
            const Icon = MODALITY_ICON[m];
            const active = activeModalities.has(m);
            return (
              <Toggle
                key={m}
                pressed={active}
                onPressedChange={() => toggleModality(m)}
                size="sm"
                className="data-[state=on]:bg-[#FF3621] data-[state=on]:text-white"
              >
                <Icon className="h-3 w-3" /> {MODALITY_LABEL[m]}
              </Toggle>
            );
          })}
          <div className="flex-1" />
          <Toggle
            pressed={readyOnly}
            onPressedChange={() => setReadyOnly((v) => !v)}
            size="sm"
            className="data-[state=on]:bg-emerald-600 data-[state=on]:text-white"
          >
            Ready only
          </Toggle>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>Failed to load endpoints: {error}</AlertDescription>
        </Alert>
      )}

      {!error && !endpoints && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground py-16 justify-center">
          <Spinner /> Loading endpoints…
        </div>
      )}

      {endpoints && grouped && (
        <>
          <div className="flex items-center text-xs text-muted-foreground">
            Showing {visibleCount} of {totalCount}
            {(activeModalities.size > 0 || readyOnly || search) && (
              <Button
                variant="ghost"
                size="sm"
                className="ml-2 h-6 px-2 text-xs"
                onClick={() => {
                  setActiveModalities(new Set());
                  setReadyOnly(false);
                  setSearch('');
                }}
              >
                Reset filters
              </Button>
            )}
          </div>

          {visibleCount === 0 && (
            <div className="text-center text-sm text-muted-foreground py-12">
              No endpoints match the current filters.
            </div>
          )}

          {KIND_SECTION_ORDER.map((kind) => {
            const items = grouped[kind];
            if (items.length === 0) return null;
            return (
              <section key={kind} className="space-y-3">
                <div className="flex items-baseline gap-2">
                  <h3 className="text-lg font-semibold">{KIND_SECTION_TITLE[kind]}</h3>
                  <span className="text-xs text-muted-foreground">({items.length})</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {items.map((ep) => (
                    <EndpointCard
                      key={ep.name}
                      ep={ep}
                      onOpen={() => {
                        const path = isDepthModality(ep.modality)
                          ? `/depth/${encodeURIComponent(ep.name)}`
                          : isDetectionModality(ep.modality)
                            ? `/detection/${encodeURIComponent(ep.name)}`
                            : isSegmentationModality(ep.modality)
                              ? `/segmentation/${encodeURIComponent(ep.name)}`
                              : isEmbeddingModality(ep.modality)
                                ? `/embeddings/${encodeURIComponent(ep.name)}`
                                : `/playground/${encodeURIComponent(ep.name)}`;
                        void navigate(path);
                      }}
                    />
                  ))}
                </div>
              </section>
            );
          })}
        </>
      )}
    </div>
  );
}

interface EndpointCardProps {
  ep: EndpointInfo;
  onOpen: () => void;
}

function EndpointCard({ ep, onOpen }: EndpointCardProps) {
  const Icon = MODALITY_ICON[ep.modality];
  const accent = MODALITY_ACCENT[ep.modality];
  const playable =
    (isChatEndpoint(ep) ||
      isEmbeddingModality(ep.modality) ||
      isSegmentationModality(ep.modality) ||
      isDetectionModality(ep.modality) ||
      isDepthModality(ep.modality)) &&
    ep.ready;

  return (
    <Card
      onClick={playable ? onOpen : undefined}
      className={`transition-all ${
        playable
          ? 'cursor-pointer hover:shadow-md hover:border-[#FF3621]/40'
          : 'opacity-70 cursor-default'
      }`}
    >
      <CardContent className="p-4 space-y-3">
        <div className="flex items-start justify-between gap-2">
          <div className={`rounded-md p-2 ${accent}`}>
            <Icon className="h-4 w-4" />
          </div>
          <Badge
            variant={ep.ready ? 'default' : 'secondary'}
            className={ep.ready ? 'bg-emerald-600 hover:bg-emerald-600' : ''}
          >
            {ep.state ?? 'unknown'}
          </Badge>
        </div>

        <div className="min-w-0">
          <div className="font-semibold truncate" title={ep.name}>
            {ep.name}
          </div>
          {ep.modelName && ep.modelName !== ep.name && (
            <div className="text-xs text-muted-foreground font-mono truncate" title={ep.modelName}>
              {ep.modelName}
            </div>
          )}
        </div>

        {ep.curatedDescription && (
          <p className="text-xs text-muted-foreground leading-snug line-clamp-3" title={ep.curatedDescription}>
            {ep.curatedDescription}
          </p>
        )}

        {(ep.speed > 0 || ep.cost > 0 || ep.quality > 0) && (
          <IndicatorRow speed={ep.speed} cost={ep.cost} quality={ep.quality} />
        )}

        <div className="flex flex-wrap gap-1.5 text-xs">
          <Badge variant="outline" className="font-normal">
            {MODALITY_LABEL[ep.modality]}
          </Badge>
          <Badge variant="outline" className="font-normal">
            {KIND_SHORT[ep.kind]}
          </Badge>
          {ep.task && (
            <Badge variant="outline" className="font-mono font-normal text-[10px]">
              {ep.task}
            </Badge>
          )}
        </div>

        {ep.modelCardUrl && (
          <a
            href={ep.modelCardUrl}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Model card <ExternalLink className="h-3 w-3" />
          </a>
        )}
      </CardContent>
    </Card>
  );
}
