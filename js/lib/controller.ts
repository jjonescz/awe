import { Scraper, SwdePage } from './scraper';
import { replaceExtension } from './utils';

/** {@link Scraper} controller to scrape one {@link SwdePage}. */
export class Controller {
  public constructor(public readonly scraper: Scraper) {}

  /** Scrapes {@link SwdePage} determined by {@link fullPath}. */
  public async scrape(fullPath: string) {
    // Navigate to the page.
    const page = await SwdePage.parse(fullPath);
    console.log('goto:', fullPath);
    await this.scraper.go(page);

    // Abort remaining requests.
    this.scraper.stop();

    // Report stats.
    console.log('stats:', this.scraper.stats);

    // Take screenshot.
    const screenshotPath = replaceExtension(fullPath, '.png');
    console.log('screenshot:', screenshotPath);
    await this.scraper.page.screenshot({
      path: screenshotPath,
      fullPage: true,
    });

    // Save local archive.
    await this.scraper.save();
  }
}
