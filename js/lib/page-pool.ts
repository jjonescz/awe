import genericPool from 'generic-pool';
import { logger } from './logging';
import { Scraper } from './scraper';
import puppeteer from 'puppeteer-core';

export interface PagePoolOptions {
  poolSize: number;
  timeout: number;
  disableJavaScript: boolean;
}

/** Creates a pool of reusable Puppeteer pages. */
export function createPagePool(scraper: Scraper, opts: PagePoolOptions) {
  const factory = <genericPool.Factory<puppeteer.Page>>{
    create: async () => {
      const page = await scraper.browser.newPage();

      // Disable JavaScript.
      if (opts.disableJavaScript) await page.setJavaScriptEnabled(false);

      // Intercept requests.
      await page.setRequestInterception(true);

      // Ignore some errors that would prevent WaybackMachine redirection.
      await page.setBypassCSP(true);

      page.setDefaultTimeout(0); // disable default timeout
      page.setDefaultNavigationTimeout(opts.timeout);

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

  return genericPool.createPool(factory, { max: opts.poolSize });
}
