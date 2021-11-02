import { BinaryLike, createHash } from 'crypto';
import { existsSync } from 'fs';
import { mkdir, readFile, writeFile } from 'fs/promises';
import path from 'path';
import { ResponseForRequest } from 'puppeteer-core';
import { ARCHIVE_DIR } from './constants';
import { logger } from './logging';
import { tryReadFile } from './utils';

const ARCHIVE_FILES_FOLDER = path.join(ARCHIVE_DIR, 'files');
const ARCHIVE_MAP_PATH = path.join(ARCHIVE_DIR, 'map.json');

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
    private readonly map: Record<string, FileResponse | null>
  ) {}

  public static async create() {
    // Prepare storage.
    await mkdir(ARCHIVE_FILES_FOLDER, { recursive: true });
    const mapJson = await tryReadFile(ARCHIVE_MAP_PATH, '{}');
    return new Archive(JSON.parse(mapJson));
  }

  public async get(
    url: string,
    timestamp: string
  ): Promise<Partial<ResponseForRequest> | null | undefined> {
    const key = this.stringifyKey(url, timestamp);
    const file = this.map[key];
    if (file === undefined || file === null) return file;

    // Read file contents.
    const filePath = this.getPath(file.hash);
    const body = await readFile(filePath);
    const { hash, ...response } = file;
    return { ...response, body };
  }

  public async add(
    url: string,
    timestamp: string,
    value: ResponseForRequest | null,
    { force = false } = {}
  ) {
    const key = this.stringifyKey(url, timestamp);
    if (this.map[key] && !force) {
      // Note that we don't want this error to be thrown when `this.map[url] ===
      // null` (then, we want to overwrite it).
      throw new Error(`URL already exists in the map: ${url} (${timestamp})`);
    }

    // Store `null` to indicate this request is "in progress".
    if (value === null) {
      this.map[key] = null;
      return;
    }

    // Create hash of file contents.
    const hash = this.computeHash(value.body);

    // Store file contents under the hash.
    const filePath = this.getPath(hash);
    if (existsSync(filePath)) {
      // If the file already exists, check that it has the same contents.
      const contents = value.body.toString('utf-8');
      const existing = await readFile(filePath, { encoding: 'utf-8' });
      if (this.computeHash(existing) !== hash) {
        // Ignore if the file has wrong hash. It will be overwritten.
        logger.debug('file invalid', { hash });
      } else if (contents !== existing) {
        throw new Error(`Hash collision (${hash}): ${timestamp}:${url}`);
      }
    } else {
      await writeFile(filePath, value.body, { encoding: 'utf-8' });
    }

    // Add file into the map.
    const { body, ...fileResponse } = value;
    this.map[key] = { ...fileResponse, hash };
  }

  public getHash(url: string, timestamp: string) {
    const key = this.stringifyKey(url, timestamp);
    return this.map[key]?.hash;
  }

  private computeHash(contents: BinaryLike) {
    const hasher = createHash('sha256');
    hasher.update(contents);
    return hasher.digest('hex');
  }

  /** Saves file map. */
  public async save() {
    const mapJson = JSON.stringify(this.map);
    await writeFile(ARCHIVE_MAP_PATH, mapJson, { encoding: 'utf-8' });
  }

  private getPath(hash: string) {
    return path.join(ARCHIVE_FILES_FOLDER, hash);
  }

  private stringifyKey(url: string, timestamp: string) {
    return `${timestamp}:${url}`;
  }
}
