import { createHash } from 'crypto';
import { mkdir, readFile, writeFile } from 'fs/promises';
import path from 'path';
import { ResponseForRequest } from 'puppeteer-core';
import { ARCHIVE_FOLDER } from './constants';

const ARCHIVE_FILES_FOLDER = path.join(ARCHIVE_FOLDER, 'files');
const ARCHIVE_MAP_PATH = path.join(ARCHIVE_FOLDER, 'map.json');

/** In-memory variant of {@link ResponseForRequest}. */
export interface FileResponse {
  status?: number;
  headers?: Record<string, unknown>;
  contentType?: string;
  hash: string;
}

export class Archive {
  private constructor(
    /** Map from URL to {@link FileResponse}. */
    private readonly map: Map<string, FileResponse>
  ) {}

  public static async create() {
    // Prepare storage.
    await mkdir(ARCHIVE_FILES_FOLDER, { recursive: true });
    const mapJson = await readFile(ARCHIVE_MAP_PATH, { encoding: 'utf-8' });
    const map = new Map<string, FileResponse>(JSON.parse(mapJson));
    return new Archive(map);
  }

  public async getOrAdd(
    url: string,
    fileFactory: (url: string) => Promise<ResponseForRequest>
  ): Promise<Partial<ResponseForRequest>> {
    const file = this.map.get(url);

    // If this hash doesn't exist, use `fileFactory` and store it.
    if (file === undefined) {
      const response = await fileFactory(url);

      // Create hash of file contents.
      const hasher = createHash('sha256');
      hasher.update(response.body);
      const hash = hasher.digest('hex');

      // Store file contents under the hash.
      const filePath = this.getPath(hash);
      await writeFile(filePath, response.body, { encoding: 'utf-8' });

      // Add file into the map.
      const { body, ...fileResponse } = response;
      this.map.set(url, { ...fileResponse, hash });

      return response;
    }

    // Otherwise, read file contents.
    const filePath = this.getPath(file.hash);
    const body = await readFile(filePath);
    const { hash, ...response } = file;
    return { ...response, body };
  }

  /** Saves file map. */
  public async save() {
    const mapJson = JSON.stringify(this.map);
    await writeFile(ARCHIVE_MAP_PATH, mapJson, { encoding: 'utf-8' });
  }

  private getPath(hash: string) {
    return path.join(ARCHIVE_FILES_FOLDER, hash);
  }
}
