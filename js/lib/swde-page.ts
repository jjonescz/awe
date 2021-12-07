import glob from 'fast-glob';
import { readFile, writeFile } from 'fs/promises';
import path from 'path';
import { SWDE_DIR } from './constants';
import { enumerate, Writable } from './utils';

// First character is UTF-8 BOM marker.
export const BASE_TAG_REGEX = /^\uFEFF?<base href="([^\n]*)"\/>\w*\n(.*)/s;

const WEBSITE_REGEX = /^(\w+)-(\w+)\((\d+)\)$/;
const GROUND_TRUTH_REGEX = /^(\w+)-(\w+)-(\w+)\.txt$/;

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

  public get vertical() {
    return path.basename(path.dirname(path.dirname(this.fullPath)));
  }

  public get website() {
    const websiteDir = path.basename(path.dirname(this.fullPath));
    const [_, _vertical, name, _num] = websiteDir.match(WEBSITE_REGEX)!;
    return name;
  }

  public get index() {
    return parseInt(path.parse(path.basename(this.fullPath)).name);
  }

  public get groundTruthPrefix() {
    return path.resolve(
      SWDE_DIR,
      'groundtruth',
      this.vertical,
      `${this.vertical}-${this.website}`
    );
  }

  private *iterateLabels() {
    const files = glob.sync(`${this.groundTruthPrefix}-*.txt`);
    for (const file of files) {
      const [_, _vertical, _website, label] = path
        .basename(file)
        .match(GROUND_TRUTH_REGEX)!;
      yield label;
    }
  }

  /** Gets groundtruth labels available for this page's website. */
  public get labels() {
    return [...this.iterateLabels()];
  }

  public groundTruthPath(label: string) {
    return `${this.groundTruthPrefix}-${label}.txt`;
  }

  public getGroundTruth(label: string) {
    return GroundTruthFile.getOrParse(this.groundTruthPath(label));
  }
}

/** Parsed `groundtruth/<vertical>/<website>-<label>.txt` file from SWDE. */
class GroundTruthFile {
  private static cache: Map<string, GroundTruthFile> = new Map();

  public constructor(
    public readonly fullPath: string,
    public readonly entries: string[][]
  ) {}

  public static async getOrParse(fullPath: string) {
    let file = this.cache.get(fullPath);
    if (file === undefined) {
      // Read file lines.
      const content = await readFile(fullPath, { encoding: 'utf-8' });
      const lines = content.split(/\r?\n/);

      // First two lines are header and total count, respectively.
      const totalCount = parseInt(lines[1].split('\t')[0]);

      // Other lines are ground truth values.
      const pages = new Array<string[]>(totalCount);
      for (const [num, line] of enumerate(lines.slice(2))) {
        // Skip empty line at the end.
        if (num === totalCount && line.length === 0) continue;

        const [indexStr, countStr, ...values] = line.split('\t');

        // If line is `<NULL>`, it means no values.
        if (values.length === 1 && values[0] === '<NULL>') {
          values.pop();
        }

        // Verify count.
        const count = parseInt(countStr);
        if (values.length !== count) {
          throw new Error(
            `Expected line #${num} to have ${count} values, but got ` +
              `${JSON.stringify(values)} in file ${fullPath} (${line}).`
          );
        }

        // Check no values for this index were saved, yet.
        const index = parseInt(indexStr);
        if (pages[index] !== undefined) {
          throw new Error(
            `Duplicate value for ${index} in file ${fullPath} ` +
              `(now ${JSON.stringify(values)} at line #${num}, previously ` +
              `${JSON.stringify(pages[index])}).`
          );
        }

        // Save values.
        pages[index] = values;
      }

      // Save to cache.
      file = new GroundTruthFile(fullPath, pages);
      this.cache.set(fullPath, file);
    }
    return file;
  }
}
