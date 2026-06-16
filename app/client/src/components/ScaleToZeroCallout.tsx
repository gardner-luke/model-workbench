import { Alert, AlertDescription } from '@databricks/appkit-ui/react';
import { Zap } from 'lucide-react';

/**
 * Banner that appears on custom-endpoint playgrounds to set expectations about
 * scale-to-zero cold starts. FMAPI endpoints don't need this — they're hosted
 * by Databricks and respond instantly.
 */
export function ScaleToZeroCallout() {
  return (
    <Alert className="border-amber-300 bg-amber-50 text-amber-900">
      <Zap className="h-4 w-4 text-amber-700" />
      <AlertDescription className="text-amber-900">
        <span className="font-medium">Scale-to-zero is on for this endpoint.</span> Custom GPU
        endpoints sleep when idle to save cost. The first request after a quiet period takes
        1–3 minutes to warm up; subsequent requests respond in well under a second.
      </AlertDescription>
    </Alert>
  );
}
