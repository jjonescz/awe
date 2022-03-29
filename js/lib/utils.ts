import { existsSync, renameSync, writeFileSync } from 'fs';
import { readFile } from 'fs/promises';
import https from 'https';
import path from 'path';
import { URL } from 'url';
import { logger } from './logging';

export type Writable<T> = { -readonly [P in keyof T]: T[P] };

export function nameOf<T>(name: Extract<keyof T, string>): string {
  return name;
}

export function replaceExtension(fullPath: string, ext: string) {
  return path.format({
    ...path.parse(fullPath),
    base: undefined,
    ext: ext,
  });
}

export function replacePrefix(
  fullPath: string,
  prefix: string,
  replacement: string
) {
  const parsed = path.parse(fullPath);
  if (!parsed.name.startsWith(prefix)) return fullPath;
  return path.format({
    ...parsed,
    base: undefined,
    name: `${replacement}${parsed.name.substring(prefix.length)}`,
  });
}

export function addSuffix(fullPath: string, suffix: string) {
  const parsed = path.parse(fullPath);
  return path.format({
    ...parsed,
    base: undefined,
    name: `${parsed.name}${suffix}`,
  });
}

export async function tryReadFile(fullPath: string, defaultContents: string) {
  if (!existsSync(fullPath)) return defaultContents;
  return await readFile(fullPath, { encoding: 'utf-8' });
}

/**
 * Saves {@link data} through a temporary file, so that this operation can
 * survive an interruption without losing the original {@link file}.
 */
export function writeFileSafe(file: string, data: string) {
  const tempFile = `${file}.temp`;
  // Note that this must NOT be asynchronous, so it's "atomic" (in non-threaded
  // Node.js world, at least).
  writeFileSync(tempFile, data, { encoding: 'utf-8' });
  renameSync(tempFile, file);
}

export function escapeFilePath(filePath: string) {
  return filePath.replace(/[`$^*+?()[\]]/g, '\\$&');
}

/** Inspired by https://stackoverflow.com/a/25279399. */
export function secondsToTimeString(seconds: number) {
  // Get time in format HH:MM:SS.
  const date = new Date(0);
  date.setSeconds(seconds);
  const time = date.toISOString().substr(11, 8);

  // Append number of days if necessary.
  const days = Math.floor(seconds / 86_400);
  if (days > 0) return `${days}d ${time}`;
  return time;
}

/** Performs HTTPS request and returns response `string`. */
export function getHttps(url: string, query: Record<string, string>) {
  const urlObject = new URL(url);
  for (const [key, value] of Object.entries(query)) {
    urlObject.searchParams.append(key, value);
  }

  return new Promise<string>((resolve) => {
    const req = https.request(urlObject, (res) => {
      let data = '';
      res.on('data', (chunk) => (data += chunk));
      res.on('end', () => resolve(data));
    });
    req.end();
  });
}

export function normalizeUrl(url: string) {
  // This removes superfluous port numbers.
  try {
    return new URL(url).toString();
  } catch (e) {
    // URL is not valid.
    logger.error('invalid URL', { url, error: (e as Error)?.stack });
    return url;
  }
}

export function urlsEqual(a: string, b: string) {
  return normalizeUrl(a) === normalizeUrl(b);
}

export function enumerate<T>(array: T[]) {
  return array.map((v, i) => [i, v] as const);
}

export function removeWhere<T>(
  array: T[],
  predicate: (item: T, index: number) => boolean
) {
  let i = array.length;
  while (i--) {
    if (predicate(array[i], i)) {
      array.splice(i, 1);
    }
  }
}
