import { readdir, readFile } from 'fs/promises';
import path from 'path';
import { LOG_DIR } from '../constants';
import naturalCompare from 'natural-compare-lite';

export interface ModelInfo {
  labels: string[];
  vertical: string;
  description: string | undefined;
  websites: string[];
  examples: string[] | undefined;
}

export async function loadModel() {
  const versions = await readdir(LOG_DIR);

  // Sort versions like a human would.
  versions.sort(naturalCompare);

  // Choose only the last version for now.
  const versionName = versions.pop()!;

  const infoPath = path.join(LOG_DIR, versionName, 'info.json');
  const infoText = await readFile(infoPath, { encoding: 'utf-8' });
  return JSON.parse(infoText) as ModelInfo;
}
