import { writeFile } from 'fs/promises';
import { Page } from 'puppeteer-core';
import { logger } from './logging';
import { SwdePage } from './scraper';
import { replaceExtension } from './utils';

/**
 * Structure in which extracted visual attributes are stored for an element and
 * its descendants.
 */
type DomData = {
  /**
   * @param key {@link key} starting with `/` denotes child with rest of the key
   * being its tag name and value being {@link DomData} for that element.
   *
   * Other {@link key}s denote extracted visual attributes.
   */
  [key: string]: string | DomData;
};

/** Can extract visual attributes from a Puppeteer-controlled page. */
export class Extractor {
  public readonly data: DomData = {};

  constructor(public readonly page: Page, public readonly swdePage: SwdePage) {}

  /** Extracts visual attributes for all DOM nodes in the {@link page}. */
  public async extract() {
    // Start a queue with root elements.
    const rootElements = await this.page.$x('*');
    const queue = rootElements.map((e) => ({
      element: e,
      parent: this.data,
    }));
    while (queue.length !== 0) {
      const { element, parent } = queue.pop()!;

      // Evaluate info for an element.
      const info = await element.evaluate((e) => {
        return { tagName: e.tagName.toLowerCase() };
      });

      // Append this element's data to parent `DomData`.
      const container: DomData = {};
      parent[`/${info.tagName}`] = container;
      const children = await element.$x('*');
      queue.push(...children.map((e) => ({ element: e, parent: container })));
    }
  }

  public get filePath() {
    return replaceExtension(this.swdePage.fullPath, '.json');
  }

  public async save() {
    logger.info('data', { path: this.filePath });
    const json = JSON.stringify(this.data, null, 1);
    await writeFile(this.filePath, json, { encoding: 'utf-8' });
  }
}
