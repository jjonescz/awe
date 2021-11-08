import progress from 'cli-progress';
import path from 'path';
import { from, lastValueFrom, mergeMap } from 'rxjs';
import { SWDE_DIR } from './constants';
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
  public async scrapeAll(
    files: string[],
    { showProgressBar = true, jobs = 1 } = {}
  ) {
    // Prepare progress bar.
    const bar = showProgressBar
      ? new progress.SingleBar({
          format:
            '[{bar}] {percentage}% | ETA: {eta_formatted} | ' +
            '{value}/{total} | {details}',
        })
      : null;
    bar?.start(files.length, 0);

    // Scrape every page.
    const observable = from(files).pipe(
      mergeMap(
        async (file) => {
          // Show stats.
          bar?.update({
            details: [
              this.scraper.stats.toString(),
              path.relative(SWDE_DIR, file),
            ]
              .filter((s) => s?.length)
              .join(' | '),
          });

          // Execute `PageController`.
          const fullPath = path.resolve(file);
          const pageController = await this.for(fullPath);
          try {
            await pageController.scrapeBoth(file);
          } finally {
            await pageController.close();
          }

          // Update progress bar.
          bar?.increment();
        },
        // Run this number of scrapers in parallel.
        jobs
      )
    );
    await lastValueFrom(observable, { defaultValue: null });
    bar?.stop();
  }
}
