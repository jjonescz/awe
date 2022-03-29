import { readFile, writeFile } from 'fs/promises';
import path from 'path';
import { pathToFileURL } from 'url';
import { DATA_DIR } from './constants';
import { Writable } from './utils';

// First character is UTF-8 BOM marker.
export const BASE_TAG_REGEX = /^\uFEFF?<base href="([^\n]*)"\/>\w*\n(.*)/s;

/** Page from the SWDE dataset. */
export class PageInfo {
  /** Timestamp used to scrape this page. */
  public timestamp: string | null = null;

  public constructor(
    public readonly fullPath: string,
    public readonly url: string,
    public readonly html: string,
    public readonly isSwde: boolean
  ) {}

  public get id() {
    return path.relative(DATA_DIR, this.fullPath);
  }

  public static async parse(fullPath: string) {
    const contents = await readFile(fullPath, { encoding: 'utf-8' });
    // Extract original page URL from a `<base>` tag that is at the beginning of
    // every HTML file in SWDE.
    const match = contents.match(BASE_TAG_REGEX);
    if (match !== null) {
      const [_, url, html] = match;
      return new PageInfo(fullPath, url, html, /* isSwde */ true);
    }

    // If it's not an SWDE page, use `file://` URL.
    const url = pathToFileURL(fullPath).toString();
    return new PageInfo(fullPath, url, contents, /* isSwde */ false);
  }

  public withHtml(html: string) {
    const clone = Object.create(this) as Writable<PageInfo>;
    clone.html = html;
    return clone as PageInfo;
  }

  public stringify() {
    if (this.isSwde) {
      // First character is UTF-8 BOM marker.
      return `\uFEFF<base href="${this.url}"/>\n${this.html}`;
    }
    return this.html;
  }

  public async saveAs(fullPath: string) {
    const contents = this.stringify();
    await writeFile(fullPath, contents, { encoding: 'utf-8' });
  }
}
