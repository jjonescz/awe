import path from 'path';
import https from 'https';
import { URLSearchParams } from 'url';

export function replaceExtension(fullPath: string, ext: string) {
  return path.format({
    ...path.parse(fullPath),
    base: undefined,
    ext: ext,
  });
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
