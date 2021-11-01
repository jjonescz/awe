import { Page } from 'puppeteer-core';
import { logger } from './logging';

type DomData = {
  [key: string]: string | DomData;
};

/** Can extract visual attributes from a Puppeteer-controlled page. */
export class Extractor {
  public readonly data: DomData = {};

  constructor(public readonly page: Page) {}

  public async extract() {
    const rootElements = await this.page.$x('*');
    const queue = rootElements.map((e) => ({
      element: e,
      parent: this.data,
    }));
    while (queue.length !== 0) {
      const { element, parent } = queue.pop()!;
      const info = await element.evaluate((e) => {
        return { tagName: e.tagName.toLowerCase() };
      });
      const container: DomData = {};
      parent[`/${info.tagName}`] = container;
      const children = await element.$x('*');
      queue.push(...children.map((e) => ({ element: e, parent: container })));
    }
    logger.info('data', this.data);
  }
}
