import { readdir, readFile } from 'fs/promises';
import naturalCompare from 'natural-compare-lite';
import path from 'path';
import { LOG_DIR } from '../constants';

export class Model {
  constructor(
    public readonly versionDir: string,
    public readonly info: ModelInfo,
    public readonly params: Record<string, any>
  ) {}
}

export interface ModelInfo {
  labels: string[];
  vertical: string;
  description: string | undefined;
  websites: string[];
  examples: string[] | undefined;
  timestamp: string | undefined;
}

export async function loadModel() {
  const versions = await readdir(LOG_DIR);

  // Sort versions like a human would.
  versions.sort(naturalCompare);

  // Choose only the last version for now.
  const versionName = versions.pop()!;
  const versionDir = path.join(LOG_DIR, versionName);

  // Load JSON files.
  const loadJson = async <T>(filename: string) => {
    const filePath = path.join(versionDir, filename);
    const text = await readFile(filePath, { encoding: 'utf-8' });
    return JSON.parse(text) as T;
  };
  const info = await loadJson<ModelInfo>('info.json');
  const params = await loadJson<Record<string, any>>('params.json');

  return new Model(versionDir, info, params);
}
