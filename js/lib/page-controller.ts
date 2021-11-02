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
  public async scrape(
    fullPath: string,
    { version = ScrapeVersion.Exact } = {}
  ) {
    // Configure page scraper.
    this.pageScraper.swdeHandling = scrapeVersionToSwdeHandling(version);

    // Navigate to the page.
    this.pageScraper.logger.verbose('goto', { fullPath });
    await this.pageScraper.start();

    // Abort remaining requests.
    await this.pageScraper.stop();

    // Save page HTML (can be different from original due to JavaScript
    // dynamically updating the DOM).
    const suffix =
      version === ScrapeVersion.Latest ? `-${this.page.timestamp}` : '-exact';
    const html = await this.pageScraper.page.content();
    const htmlPath = addSuffix(fullPath, suffix);
    await this.page.withHtml(html).saveAs(htmlPath);

    // Report stats.
    this.pageScraper.logger.verbose('stats', {
      stats: this.controller.scraper.stats,
    });

    // Save local archive.
    await this.controller.scraper.save();

    if (version === ScrapeVersion.Latest && this.page.timestamp === null) {
      // Couldn't find snapshot of this page in the archive, abort early.
      return;
    }

    // Extract visual attributes.
    const extractor = new Extractor(this.pageScraper.page, this.page);
    await extractor.extract();
    await extractor.save({ suffix });

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

  /** Scrapes old and new versions of {@link SwdePage} at {@link fullPath}. */
  public async scrapeBoth(fullPath: string) {
    await this.scrape(fullPath, { version: ScrapeVersion.Exact });
    //await this.scrape(fullPath, { version: ScrapeVersion.Latest });
  }

  public async close() {
    await this.pageScraper.page.close();
  }
}
