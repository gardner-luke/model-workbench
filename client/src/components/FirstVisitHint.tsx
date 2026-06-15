import { useState } from 'react';
import { BarChart3, MousePointerClick, SlidersHorizontal, X } from 'lucide-react';

const STORAGE_KEY = 'model-workbench.registry.hint-dismissed';

function readDismissed(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    return localStorage.getItem(STORAGE_KEY) === '1';
  } catch {
    return false;
  }
}

/**
 * Small one-time orientation banner shown on the registry home. Persists its
 * dismissed state in localStorage. Designed to be skipped fast — three short
 * pointers, no walkthrough, no modal.
 */
export function FirstVisitHint() {
  const [visible, setVisible] = useState(() => !readDismissed());

  const dismiss = () => {
    setVisible(false);
    try {
      localStorage.setItem(STORAGE_KEY, '1');
    } catch {
      // ignore
    }
  };

  if (!visible) return null;

  return (
    <div className="relative rounded-md border border-[#FF3621]/20 bg-[#FF3621]/5 px-4 py-3 text-sm">
      <button
        type="button"
        onClick={dismiss}
        className="absolute top-2 right-2 text-muted-foreground hover:text-foreground"
        aria-label="Dismiss"
      >
        <X className="h-4 w-4" />
      </button>
      <div className="font-medium mb-2 text-foreground">New here?</div>
      <ul className="space-y-1.5 text-muted-foreground">
        <li className="flex items-start gap-2">
          <MousePointerClick className="h-4 w-4 mt-0.5 text-[#FF3621] shrink-0" />
          <span>Click any card to open its playground — chat, embeddings, segmentation, detection, or depth.</span>
        </li>
        <li className="flex items-start gap-2">
          <SlidersHorizontal className="h-4 w-4 mt-0.5 text-[#FF3621] shrink-0" />
          <span>Use the modality chips to narrow the list, or search by name.</span>
        </li>
        <li className="flex items-start gap-2">
          <BarChart3 className="h-4 w-4 mt-0.5 text-[#FF3621] shrink-0" />
          <span>Open <b>Analytics</b> in the top nav to see what this app is actually costing.</span>
        </li>
      </ul>
    </div>
  );
}
