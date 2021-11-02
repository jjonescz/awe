import progress from 'cli-progress';
import { Extractor } from './extractor';
import { PageScraper, Scraper, SwdeHandling, SwdePage } from './scraper';
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
  /** Take screenshot of each page. */
  public takeScreenshot = true;

  public constructor(public readonly scraper: Scraper) {}

  /** Scrapes {@link SwdePage} determined by {@link fullPath}. */
  public async scrape(
    fullPath: string,
    { version = ScrapeVersion.Exact } = {}
  ) {
    // Create page scraper.
    const page = await SwdePage.parse(fullPath);
    const pageScraper = await this.scraper.for(page);
    pageScraper.swdeHandling = scrapeVersionToSwdeHandling(version);

    // Navigate to the page.
    pageScraper.logger.verbose('goto', { fullPath });
    await pageScraper.start();

    // Abort remaining requests.
    await pageScraper.stop();

    // Save page HTML (can be different from original due to JavaScript
    // dynamically updating the DOM).
    const suffix =
      version === ScrapeVersion.Latest ? `-${page.timestamp}` : '-exact';
    const html = await pageScraper.page.content();
    const htmlPath = addSuffix(fullPath, suffix);
    await page.withHtml(html).saveAs(htmlPath);

    // Report stats.
    pageScraper.logger.verbose('stats', { stats: this.scraper.stats });

    // Save local archive.
    await this.scraper.save();

    if (version === ScrapeVersion.Latest && page.timestamp === null) {
      // Couldn't find snapshot of this page in the archive, abort early.
      return;
    }

    // Extract visual attributes.
    const extractor = new Extractor(pageScraper.page, page);
    await extractor.extract();
    await extractor.save({ suffix });

    // Take screenshot.
    if (this.takeScreenshot) {
      const screenshotPath = replaceExtension(fullPath, `${suffix}.png`);
      await this.screenshot(pageScraper, screenshotPath, { fullPage: false });
      await this.screenshot(pageScraper, screenshotPath, { fullPage: true });
    }

    // Release memory.
    await pageScraper.page.close();
  }

  private async screenshot(
    pageScraper: PageScraper,
    fullPath: string,
    { fullPage = true } = {}
  ) {
    const suffix = fullPage ? '-full' : '-preview';
    const screenshotPath = addSuffix(fullPath, suffix);
    pageScraper.logger.verbose('screenshot', { screenshotPath });
    await pageScraper.page.screenshot({
      path: screenshotPath,
      fullPage: fullPage,
    });
  }

  /** Scrapes old and new versions of {@link SwdePage} at {@link fullPath}. */
  public async scrapeBoth(fullPath: string) {
    await this.scrape(fullPath, { version: ScrapeVersion.Exact });
    //await this.scrape(fullPath, { version: ScrapeVersion.Latest });
  }

  /** Scrapes all SWDE page {@link files}. */
  public async scrapeAll(files: string[], { showProgressBar = true } = {}) {
    // Prepare progress bar.
    const bar = showProgressBar
      ? new progress.SingleBar({
          format:
            'progress [{bar}] {percentage}% | ETA: {eta_formatted} | ' +
            '{value}/{total} | {stats} | {file}',
        })
      : null;
    bar?.start(files.length, 0);

    // Scrape every page.
    for (const file of files) {
      // Show stats.
      bar?.update({ file, stats: this.scraper.stats.toString() });

      await this.scrapeBoth(file);
      bar?.increment();
    }
    bar?.stop();
  }
}
