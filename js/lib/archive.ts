import { BinaryLike, createHash } from 'crypto';
import { mkdir, readFile, writeFile } from 'fs/promises';
import path from 'path';
import { ARCHIVE_FOLDER } from './constants';

const ARCHIVE_FILES_FOLDER = path.join(ARCHIVE_FOLDER, 'files');
const ARCHIVE_MAP_PATH = path.join(ARCHIVE_FOLDER, 'map.json');

export class Archive {
  private constructor(
    /** Map from URL to file hash. */
    private readonly map: Map<string, string>
  ) {}

  public async create() {
    await mkdir(ARCHIVE_FILES_FOLDER, { recursive: true });
    const mapJson = await readFile(ARCHIVE_MAP_PATH, { encoding: 'utf-8' });
    const map = new Map<string, string>(JSON.parse(mapJson));
    return new Archive(map);
  }

  public async getOrAdd(url: string, contentsFactory: () => BinaryLike) {
    let hash = this.map.get(url);

    // If this hash doesn't exist, use `contentsFactory` and store it.
    if (hash === undefined) {
      const contents = contentsFactory();

      // Create hash of file contents.
      const hasher = createHash('sha256');
      hasher.update(contents);
      hash = hasher.digest('hex');

      // Store file contents under the hash.
      const filePath = this.getPath(hash);
      await writeFile(filePath, contents, { encoding: 'utf-8' });

      // Add file into the map.
      this.map.set(url, hash);

      // Return contents of the file.
      return contents;
    }

    // Otherwise, read file contents.
    const filePath = this.getPath(hash);
    return await readFile(filePath);
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
