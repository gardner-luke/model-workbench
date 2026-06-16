import { ChevronUp, DollarSign, Zap } from 'lucide-react';

interface ScaleProps {
  value: number; // 0–4
  max?: number;
  Icon: React.ComponentType<{ className?: string }>;
  activeClass: string;
  inactiveClass?: string;
}

function Scale({ value, max = 4, Icon, activeClass, inactiveClass = 'text-muted-foreground/30' }: ScaleProps) {
  if (!value || value < 1) return <span className="text-xs text-muted-foreground">—</span>;
  return (
    <span className="inline-flex">
      {Array.from({ length: max }, (_, i) => (
        <Icon key={i} className={`h-3 w-3 ${i < value ? activeClass : inactiveClass}`} />
      ))}
    </span>
  );
}

export function SpeedIndicator({ value }: { value: number }) {
  return <Scale value={value} Icon={Zap} activeClass="fill-amber-500 text-amber-500" />;
}

export function CostIndicator({ value }: { value: number }) {
  // Cost 0 (free / open-weight) is meaningful — show "free" badge.
  if (value === 0) {
    return (
      <span className="text-[10px] uppercase tracking-wider font-medium text-emerald-600">free</span>
    );
  }
  return <Scale value={value} Icon={DollarSign} activeClass="text-emerald-700" />;
}

export function QualityIndicator({ value }: { value: number }) {
  return <Scale value={value} Icon={ChevronUp} activeClass="fill-sky-600 text-sky-600" />;
}

interface IndicatorRowProps {
  speed: number;
  cost: number;
  quality: number;
  compact?: boolean;
}

export function IndicatorRow({ speed, cost, quality, compact = false }: IndicatorRowProps) {
  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <SpeedIndicator value={speed} />
        <CostIndicator value={cost} />
        <QualityIndicator value={quality} />
      </div>
    );
  }
  return (
    <div className="grid grid-cols-3 gap-2 text-xs">
      <div className="space-y-0.5">
        <div className="text-muted-foreground uppercase tracking-wider text-[10px]">Speed</div>
        <SpeedIndicator value={speed} />
      </div>
      <div className="space-y-0.5">
        <div className="text-muted-foreground uppercase tracking-wider text-[10px]">Cost</div>
        <CostIndicator value={cost} />
      </div>
      <div className="space-y-0.5">
        <div className="text-muted-foreground uppercase tracking-wider text-[10px]">Quality</div>
        <QualityIndicator value={quality} />
      </div>
    </div>
  );
}
