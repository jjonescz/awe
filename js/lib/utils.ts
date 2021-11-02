import { existsSync } from 'fs';
import { readFile } from 'fs/promises';
import https from 'https';
import path from 'path';
import { URL } from 'url';

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
  return new URL(url).toString();
}
