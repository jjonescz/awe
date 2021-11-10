import genericPool from 'generic-pool';
import { logger } from './logging';
import { Scraper } from './scraper';
import puppeteer from 'puppeteer-core';

export function createPagePool(scraper: Scraper, poolSize: number) {
  const factory = <genericPool.Factory<puppeteer.Page>>{
    create: async () => {
      const page = await scraper.browser.newPage();

      // Intercept requests.
      await page.setRequestInterception(true);

      // Ignore some errors that would prevent WaybackMachine redirection.
      await page.setBypassCSP(true);
      page.setDefaultTimeout(0); // disable timeout

      return page;
    },
    destroy: async (page) => {
      try {
        await page.close();
      } catch (e: any) {
        logger.error('closing failed', {
          error: (e as Error)?.stack,
        });
      }
    },
  };

  return genericPool.createPool(factory, { max: poolSize });
}
