import progress from 'cli-progress';
import path from 'path';
import { from, lastValueFrom, mergeMap } from 'rxjs';
import { SWDE_DIR } from './constants';
import { logger } from './logging';
import { PageController } from './page-controller';
import { Scraper } from './scraper';
import { secondsToTimeString } from './utils';

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
    { showProgressBar = true, jobs = 1, continueOnError = false } = {}
  ) {
    // Prepare progress bar.
    const bar = showProgressBar
      ? new progress.SingleBar({
          format:
            '[{bar}] {percentage}% | ETA: {eta_formatted} | ' +
            '{value}/{total} | {details}',
        })
      : null;
    bar?.start(files.length, 0, { details: 'starting...' });
    const showStats = (...stats: string[]) => {
      bar?.update({
        details: stats.filter((s) => s?.length).join(' | '),
      });
    };

    // Scrape every page.
    const startTime = process.hrtime();
    const observable = from(files).pipe(
      mergeMap(
        async (file) => {
          // Show stats.
          showStats(
            this.scraper.stats.toString(),
            path.relative(SWDE_DIR, file)
          );

          // Execute `PageController`.
          try {
            const fullPath = path.resolve(file);
            const pageController = await this.for(fullPath);
            try {
              await pageController.scrapeBoth(file);
            } finally {
              await pageController.close();
            }
          } catch (e) {
            if (continueOnError) {
              logger.error('ignoring page controller error', {
                file,
                error: (e as Error)?.stack,
              });
            } else {
              throw e;
            }
          }

          // Update progress bar.
          bar?.increment();
        },
        // Run this number of scrapers in parallel.
        jobs
      )
    );
    await lastValueFrom(observable, { defaultValue: null });

    // Show final stats.
    const duration = process.hrtime(startTime);
    showStats(this.scraper.stats.toString(), secondsToTimeString(duration[0]));

    bar?.stop();
  }
}
