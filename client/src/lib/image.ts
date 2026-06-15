// Image preprocessing helpers.
// Databricks Model Serving has a 16 MiB per-request payload limit. Raw camera
// images (especially modern phone HEIC/JPG at 4-12 MP) easily blow past it once
// base64-encoded. Downscale aggressively before sending — CV models like CLIP
// resize to 224px internally, and SAM 3 typically processes ≤1024px patches.

export interface PreparedImage {
  dataUrl: string; // data:image/jpeg;base64,...
  width: number;
  height: number;
  originalWidth: number;
  originalHeight: number;
  approxBytes: number; // payload size of the base64 portion
  resized: boolean;
}

const DEFAULT_MAX_DIMENSION = 1536;
const DEFAULT_QUALITY = 0.88;

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

/**
 * Reads a user-selected image file and returns a JPEG data URL downscaled so the
 * longest edge is at most `maxDimension` pixels. Already-small images pass through
 * as their original encoding to preserve quality + transparency.
 */
export async function prepareImage(
  file: File,
  maxDimension = DEFAULT_MAX_DIMENSION,
  quality = DEFAULT_QUALITY,
): Promise<PreparedImage> {
  const bitmap = await createImageBitmap(file);
  const originalWidth = bitmap.width;
  const originalHeight = bitmap.height;
  const longest = Math.max(originalWidth, originalHeight);

  if (longest <= maxDimension) {
    const dataUrl = await readFileAsDataUrl(file);
    const base64 = dataUrl.split(',', 2)[1] ?? '';
    return {
      dataUrl,
      width: originalWidth,
      height: originalHeight,
      originalWidth,
      originalHeight,
      approxBytes: Math.floor((base64.length * 3) / 4),
      resized: false,
    };
  }

  const scale = maxDimension / longest;
  const width = Math.round(originalWidth * scale);
  const height = Math.round(originalHeight * scale);

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D context unavailable');
  ctx.drawImage(bitmap, 0, 0, width, height);
  const dataUrl = canvas.toDataURL('image/jpeg', quality);
  const base64 = dataUrl.split(',', 2)[1] ?? '';

  return {
    dataUrl,
    width,
    height,
    originalWidth,
    originalHeight,
    approxBytes: Math.floor((base64.length * 3) / 4),
    resized: true,
  };
}

export function dataUrlToBase64(dataUrl: string): string {
  const idx = dataUrl.indexOf(',');
  return idx >= 0 ? dataUrl.slice(idx + 1) : dataUrl;
}
