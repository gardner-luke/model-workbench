import { ExternalLink } from 'lucide-react';

/**
 * Small "Model card →" link rendered in playground headers. Hidden when the
 * endpoint has no curated model card URL.
 */
export function ModelCardLink({ url }: { url: string | null }) {
  if (!url) return null;
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
    >
      Model card <ExternalLink className="h-3 w-3" />
    </a>
  );
}
