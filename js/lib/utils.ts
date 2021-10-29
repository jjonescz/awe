import path from 'path';
import https from 'https';
import { URLSearchParams } from 'url';
import { existsSync } from 'fs';
import { readFile } from 'fs/promises';

export function replaceExtension(fullPath: string, ext: string) {
  return path.format({
    ...path.parse(fullPath),
    base: undefined,
    ext: ext,
  });
}

export async function tryReadFile(fullPath: string, defaultContents: string) {
  if (!existsSync(fullPath)) return defaultContents;
  return await readFile(fullPath, { encoding: 'utf-8' });
}

/** Performs HTTPS request and returns response `string`. */
export function getHttps(
  host: string,
  pathname: string,
  query: Record<string, string>
) {
  return new Promise<string>((resolve) => {
    const req = https.request(
      {
        host,
        pathname,
        search: new URLSearchParams(query).toString(),
      },
      (res) => {
        let data = '';
        res.on('data', (chunk) => (data += chunk));
        res.on('end', () => resolve(data));
      }
    );
    req.end();
  });
}
