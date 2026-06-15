import { useState } from 'react';
import {
  Button,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  Spinner,
} from '@databricks/appkit-ui/react';
import { ChevronDown, Images } from 'lucide-react';
import type { SampleImage } from '../lib/samples';
import { SAMPLES } from '../lib/samples';

interface SamplePickerProps {
  /** Called with a File (and optional pre-filled prompt) once the sample is fetched. */
  onPick: (file: File, prompt?: string) => void | Promise<void>;
  /** Which prompt field from the sample to forward (if available). */
  modality?: 'segmentation' | 'detection' | 'embedding';
  disabled?: boolean;
}

export function SamplePicker({ onPick, modality, disabled }: SamplePickerProps) {
  const [busy, setBusy] = useState(false);

  const handlePick = async (sample: SampleImage) => {
    if (busy) return;
    setBusy(true);
    try {
      const resp = await fetch(sample.src);
      if (!resp.ok) throw new Error(`Failed to load ${sample.src} (HTTP ${resp.status})`);
      const blob = await resp.blob();
      const file = new File([blob], `${sample.name.toLowerCase().replace(/\s+/g, '-')}.jpg`, {
        type: blob.type || 'image/jpeg',
      });
      const prompt = modality ? sample.prompts?.[modality] : undefined;
      await onPick(file, prompt);
    } finally {
      setBusy(false);
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" disabled={disabled || busy}>
          {busy ? <Spinner /> : <Images className="h-4 w-4" />}
          Try a sample <ChevronDown className="h-3 w-3" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-72">
        {SAMPLES.map((s) => (
          <DropdownMenuItem
            key={s.src}
            onClick={() => void handlePick(s)}
            className="gap-3"
          >
            <img
              src={s.src}
              alt={s.name}
              className="h-10 w-10 object-cover rounded border shrink-0"
            />
            <div className="flex flex-col min-w-0">
              <span className="text-sm font-medium truncate">{s.name}</span>
              <span className="text-xs text-muted-foreground truncate">{s.caption}</span>
            </div>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
