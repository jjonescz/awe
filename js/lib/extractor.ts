import { Page } from 'puppeteer-core';
import { logger } from './logging';

/** Can extract visual attributes from a Puppeteer-controlled page. */
export class Extractor {
  constructor(public readonly page: Page) {}

  public async extract() {
    const rootElements = await this.page.$x('/*');
    for (const element of rootElements) {
      const tagName = await element.evaluate((e) => e.tagName);
      logger.info('root', { tagName });
    }
  }
}
