import progress from 'cli-progress';
import path from 'path';
import { PageController } from './page-controller';
import { Scraper } from './scraper';

/** Container for multiple {@link PageController}s. */
export class Controller {
  /** Take screenshot of each page. */
  public takeScreenshot = true;

  public constructor(public readonly scraper: Scraper) {}

  public async for(fullPath: string) {
    return await PageController.create(this, fullPath);
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

      // Execute `PageController`.
      const fullPath = path.resolve(file);
      const pageController = await this.for(fullPath);
      await pageController.scrapeBoth(file);

      // Update progress bar.
      bar?.increment();
    }
    bar?.stop();
  }
}
