import { Scraper, SwdeHandling, SwdePage } from './scraper';
import { replaceExtension } from './utils';

/** {@link Scraper} controller to scrape one {@link SwdePage}. */
export class Controller {
  public constructor(public readonly scraper: Scraper) {}

  /** Scrapes {@link SwdePage} determined by {@link fullPath}. */
  public async scrape(fullPath: string, { latest = false } = {}) {
    this.scraper.swdeHandling = latest
      ? SwdeHandling.Wayback
      : SwdeHandling.Offline;

    // Navigate to the page.
    const page = await SwdePage.parse(fullPath);
    console.log('goto:', fullPath);
    await this.scraper.go(page);

    // Abort remaining requests.
    this.scraper.stop();

    // Report stats.
    console.log('stats:', this.scraper.stats);

    if (latest && page.timestamp === null) {
      // Couldn't find snapshot of this page in the archive, abort early.
      return;
    }

    // Take screenshot.
    const suffix = latest ? `.${page.timestamp}` : '';
    const screenshotPath = replaceExtension(`${fullPath}${suffix}`, '.png');
    console.log('screenshot:', screenshotPath);
    await this.scraper.page.screenshot({
      path: screenshotPath,
      fullPage: true,
    });

    // Save local archive.
    await this.scraper.save();
  }

  /** Scrapes old and new versions of {@link SwdePage} at {@link fullPath}. */
  public async scrapeBoth(fullPath: string) {
    await this.scrape(fullPath, { latest: false });
    await this.scrape(fullPath, { latest: true });
  }
}
