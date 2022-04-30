import crypto from 'crypto';
import { existsSync, renameSync, writeFileSync } from 'fs';
import { mkdir, readFile } from 'fs/promises';
import https from 'https';
import path from 'path';
import { URL } from 'url';
import { TEMPORARY_DIR } from './constants';
import { logger } from './logging';

export type Writable<T> = { -readonly [P in keyof T]: T[P] };

export function nameOf<T>(name: Extract<keyof T, string>): string {
  return name;
}

/** Replaces file extension of {@link fullPath} with {@link ext}. */
export function replaceExtension(fullPath: string, ext: string) {
  return path.format({
    ...path.parse(fullPath),
    base: undefined,
    ext: ext,
  });
}

/**
 * Replaces {@link prefix} inside file name of {@link fullPath} with
 * {@link replacement}.
 */
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

/** Adds {@link suffix} to file name (before extension) of {@link fullPath}. */
export function addSuffix(fullPath: string, suffix: string) {
  const parsed = path.parse(fullPath);
  return path.format({
    ...parsed,
    base: undefined,
    name: `${parsed.name}${suffix}`,
  });
}

/** Reads file at {@link fullPath} if it exists. */
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

/** Escapes {@link filePath} to be safe for use in a glob pattern. */
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

/** Normalizes {@link url} so it can be compared for equality. */
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

/** Compares two URLs for equality. */
export function urlsEqual(a: string, b: string) {
  return normalizeUrl(a) === normalizeUrl(b);
}

/** Iterates over elements of {@link array} along their indices. */
export function enumerate<T>(array: T[]) {
  return array.map((v, i) => [i, v] as const);
}

/** Removes elements from {@link array} where {@link predicate} holds. */
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

/** Parses {@link input} into an integer if possible. */
export function tryParseInt(input: any, defaultValue: number): number {
  const result = parseInt(input?.toString());
  if (isNaN(result)) return defaultValue;
  return result;
}

/** Constructs temporary file path with the given file {@link extension}. */
export async function temporaryFilePath(extension: string) {
  await mkdir(TEMPORARY_DIR, { recursive: true });
  const num = crypto.randomBytes(8).readBigUInt64LE(0);
  return `${TEMPORARY_DIR}/${num}${extension}`;
}
