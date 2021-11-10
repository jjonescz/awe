import { existsSync } from 'fs';
import { Controller } from './controller';
import { Extractor } from './extractor';
import { PageScraper, SwdeHandling } from './page-scraper';
import { SwdePage } from './swde-page';
import { addSuffix, replaceExtension } from './utils';

export const enum ScrapeVersion {
  /** Scrapes version that preserves HTML from the SWDE dataset. */
  Exact,
  /**
   * Scrapes latest available version from the WaybackMachine. This doesn't
   * consider HTML from the SWDE dataset and therefore gold labels might be
   * inconsistent.
   */
  Latest,
}

export function scrapeVersionToSwdeHandling(version: ScrapeVersion) {
  switch (version) {
    case ScrapeVersion.Exact:
      return SwdeHandling.Offline;
    case ScrapeVersion.Latest:
      return SwdeHandling.Wayback;
  }
}

export function scrapeVersionToString(version: ScrapeVersion) {
  switch (version) {
    case ScrapeVersion.Exact:
      return 'exact';
    case ScrapeVersion.Latest:
      return 'latest';
  }
}

/** {@link PageScraper} controller to scrape one {@link SwdePage}. */
export class PageController {
  private constructor(
    private readonly controller: Controller,
    private readonly page: SwdePage,
    private readonly pageScraper: PageScraper
  ) {}

  public static async create(controller: Controller, fullPath: string) {
    const page = await SwdePage.parse(fullPath);
    const pageScraper = await controller.scraper.for(page);
    return new PageController(controller, page, pageScraper);
  }

  /** Scrapes {@link SwdePage} determined by {@link fullPath}. */
  public async scrape(fullPath: string, version: ScrapeVersion) {
    // Prepare dependencies.
    const suffix = `-${scrapeVersionToString(version)}`;
    const htmlPath = addSuffix(fullPath, suffix);
    const extractor = new Extractor(
      this.pageScraper.page,
      this.page,
      this.pageScraper.logger
    );

    // Check if not already scraped.
    if (this.controller.skipExisting) {
      const jsonPath = addSuffix(extractor.filePath, suffix);
      const jsonExists = existsSync(jsonPath);
      const htmlExists = existsSync(htmlPath);
      const metadata = {
        jsonExists,
        jsonPath,
        htmlExists,
        htmlPath,
      };
      if (jsonExists && htmlExists) {
        this.pageScraper.logger.verbose('skipping', metadata);
        return;
      } else {
        this.pageScraper.logger.debug('not skipping', metadata);
      }
    }

    // Configure page scraper.
    this.pageScraper.swdeHandling = scrapeVersionToSwdeHandling(version);

    // Navigate to the page.
    this.pageScraper.logger.verbose('goto', { fullPath, suffix });
    await this.pageScraper.start();

    // Abort remaining requests.
    await this.pageScraper.stop();

    // Report stats.
    this.pageScraper.logger.verbose('stats', {
      stats: this.controller.scraper.stats,
    });

    // Save local cache.
    await this.controller.scraper.save();

    if (this.page.timestamp === null) {
      // Couldn't find snapshot of this page in the WaybackMachine, abort early.
      this.pageScraper.logger.error('page not reached');
      return;
    }

    if (!this.controller.skipExtraction) {
      // Save page HTML (can be different from original due to JavaScript
      // dynamically updating the DOM).
      const html = await this.pageScraper.page.content();
      await this.page.withHtml(html).saveAs(htmlPath);

      // Extract visual attributes.
      await extractor.extract();
      await extractor.save({ suffix });
    }

    // Take screenshot.
    if (this.controller.takeScreenshot) {
      const screenshotPath = replaceExtension(fullPath, `${suffix}.png`);
      await this.screenshot(screenshotPath, { fullPage: false });
      await this.screenshot(screenshotPath, { fullPage: true });
    }
  }

  private async screenshot(fullPath: string, { fullPage = true } = {}) {
    const suffix = fullPage ? '-full' : '-preview';
    const screenshotPath = addSuffix(fullPath, suffix);
    this.pageScraper.logger.verbose('screenshot', { screenshotPath });
    await this.pageScraper.page.screenshot({
      path: screenshotPath,
      fullPage: fullPage,
    });
  }

  public async close() {
    try {
      await this.pageScraper.page.close();
    } catch (e: any) {
      this.pageScraper.logger.error('closing failed', {
        error: (e as Error)?.stack,
      });
    }
  }
}
