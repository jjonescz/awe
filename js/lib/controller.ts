import { writeFile } from 'fs/promises';
import { Extractor } from './extractor';
import { logger } from './logging';
import { Scraper, SwdeHandling, SwdePage } from './scraper';
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

/** {@link Scraper} controller to scrape one {@link SwdePage}. */
export class Controller {
  public constructor(public readonly scraper: Scraper) {}

  /** Scrapes {@link SwdePage} determined by {@link fullPath}. */
  public async scrape(
    fullPath: string,
    { version = ScrapeVersion.Exact } = {}
  ) {
    // Configure scraper.
    this.scraper.swdeHandling = scrapeVersionToSwdeHandling(version);
    // this.scraper.wayback.variant =
    //   // If we are getting the HTML from archive.org, we can let them rewrite
    //   // URLs (that's what `if_` variant does).
    //   version === ScrapeVersion.Latest ? 'if_' : 'id_';

    // Navigate to the page.
    const page = await SwdePage.parse(fullPath);
    logger.info('goto', { fullPath });
    await this.scraper.go(page);

    // Abort remaining requests.
    await this.scraper.stop();

    // Save page HTML (can be different from original due to JavaScript
    // dynamically updating the DOM).
    const suffix =
      version === ScrapeVersion.Latest ? `-${page.timestamp}` : '-exact';
    const html = await this.scraper.page.content();
    const htmlPath = addSuffix(fullPath, suffix);
    await page.withHtml(html).saveAs(htmlPath);

    // Report stats.
    logger.info('stats', { stats: this.scraper.stats });

    if (version === ScrapeVersion.Latest && page.timestamp === null) {
      // Couldn't find snapshot of this page in the archive, abort early.
      return;
    }

    // Save local archive.
    await this.scraper.save();

    // Extract visual attributes.
    const extractor = new Extractor(this.scraper.page, page);
    await extractor.extract();
    await extractor.save({ suffix });

    // Take screenshot.
    const screenshotPath = replaceExtension(fullPath, `${suffix}.png`);
    await this.screenshot(screenshotPath, { fullPage: false });
    await this.screenshot(screenshotPath, { fullPage: true });
  }

  private async screenshot(fullPath: string, { fullPage = true } = {}) {
    const suffix = fullPage ? '-full' : '-preview';
    const screenshotPath = addSuffix(fullPath, suffix);
    logger.info('screenshot', { screenshotPath });
    await this.scraper.page.screenshot({
      path: screenshotPath,
      fullPage: fullPage,
    });
  }

  /** Scrapes old and new versions of {@link SwdePage} at {@link fullPath}. */
  public async scrapeBoth(fullPath: string) {
    await this.scrape(fullPath, { version: ScrapeVersion.Exact });
    //await this.scrape(fullPath, { version: ScrapeVersion.Latest });
  }
}
