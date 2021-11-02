import { readFile, writeFile } from 'fs/promises';
import path from 'path/posix';
import { SWDE_DIR } from './constants';
import { Writable } from './utils';

// First character is UTF-8 BOM marker.
export const BASE_TAG_REGEX = /^\uFEFF?<base href="([^\n]*)"\/>\w*\n(.*)/s;

/** Page from the SWDE dataset. */
export class SwdePage {
  /** Timestamp used to scrape this page. */
  public timestamp: string | null = null;

  public constructor(
    public readonly fullPath: string,
    public readonly url: string,
    public readonly html: string
  ) {}

  public get id() {
    return path.relative(SWDE_DIR, this.fullPath);
  }

  public static async parse(fullPath: string) {
    const contents = await readFile(fullPath, { encoding: 'utf-8' });
    // Extract original page URL from a `<base>` tag that is at the beginning of
    // every HTML file in SWDE.
    const [_, url, html] = contents.match(BASE_TAG_REGEX)!;
    return new SwdePage(fullPath, url, html);
  }

  public withHtml(html: string) {
    const clone = Object.create(this) as Writable<SwdePage>;
    clone.html = html;
    return clone as SwdePage;
  }

  public stringify() {
    // First character is UTF-8 BOM marker.
    return `\uFEFF<base href="${this.url}"/>\n${this.html}`;
  }

  public async saveAs(fullPath: string) {
    const contents = this.stringify();
    await writeFile(fullPath, contents, { encoding: 'utf-8' });
  }
}