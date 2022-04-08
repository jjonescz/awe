import progress from 'cli-progress';
import { existsSync } from 'fs';
import path from 'path';
import { from, lastValueFrom, mergeMap } from 'rxjs';
import { Blender } from './blender';
import { DATA_DIR } from './constants';
import { logger } from './logging';
import { PageController } from './page-controller';
import { PageInfo } from './page-info';
import { PageRecipe } from './page-recipe';
import { ScrapeVersion } from './scrape-version';
import { Scraper } from './scraper';
import { secondsToTimeString } from './utils';
import { ValidationResultType, Validator } from './validator';

/** Container for multiple {@link PageController}s. */
export class Controller {
  /** Take screenshot of every n-th page. */
  public takeScreenshot = 0;
  /** Skip already-scraped pages. */
  public skipExisting = false;
  /** Skip HTML page saving. */
  public skipSave = false;
  /** Skip extraction (perform only page loading). */
  public skipExtraction = false;
  /** Only validate existing extraction outcomes. */
  public validateOnly = false;
  /** Only blend existing JSON and HTML into XML. */
  public blendOnly = false;
  /** Extract HTML and visuals together as XML instead of just visuals as JSON. */
  public extractXml = false;
  /** Freeze HTML before extracting visuals. */
  public freeze = false;

  public constructor(public readonly scraper: Scraper) {}

  public async for(page: PageInfo) {
    return await PageController.create(this, page);
  }

  /** Scrapes all SWDE page {@link files}. */
  public async scrapeAll(
    files: string[],
    {
      showProgressBar = true,
      jobs = 1,
      versions = [ScrapeVersion.Exact, ScrapeVersion.Latest],
    } = {}
  ) {
    // Prepare progress bar.
    const bar = showProgressBar
      ? new progress.SingleBar({
          format:
            '[{bar}] {percentage}% | ETA: {eta_formatted} | ' +
            '{value}/{total} | {details}',
          // In parallel mode, jobs can complete quickly after each other, so we
          // must compute ETA from large enough previous jobs for it to be
          // representative.
          etaBuffer: jobs * 2,
        })
      : null;
    bar?.start(files.length, 0, { details: 'starting...' });
    const showStats = (...stats: string[]) => {
      // Add timestamp of last progress update (useful to detect stuck
      // scraping).
      stats.unshift(new Date().toLocaleString());

      bar?.update({
        details: stats.filter((s) => s?.length).join(' | '),
      });
    };

    // Scrape every page.
    const startTime = process.hrtime();
    const observable = from(files).pipe(
      mergeMap(
        async (file, index) => {
          // Show stats.
          showStats(
            this.scraper.stats.toString(),
            path.relative(DATA_DIR, file)
          );

          // Scrape different versions of the same page.
          try {
            const page = await PageInfo.parse(path.resolve(file));
            for (const version of versions) {
              // Validate existing extraction of this version.
              if (this.validateOnly) {
                const validator = new Validator(new PageRecipe(page, version));
                const result = await validator.validate();
                if (result.type !== ValidationResultType.Valid) {
                  logger.warn('invalid', {
                    file,
                    suffix: validator.recipe.suffix,
                    result: result.toString(),
                  });
                }
                continue;
              }

              // Blend JSON + HTML = XML.
              if (this.blendOnly) {
                const recipe = new PageRecipe(page, version);
                const useExact = existsSync(recipe.htmlPath);
                const log = logger.child({
                  file: useExact ? recipe.htmlPath : file,
                  suffix: recipe.suffix,
                });
                const blender = new Blender(recipe, log);
                await blender.loadJsonData();
                await blender.loadHtmlDom({ useExact });
                blender.blend();
                await blender.save();
                continue;
              }

              // Execute `PageController`.
              for (let retries = 0; retries < 2; retries++) {
                const pageController = await this.for(page);
                try {
                  if (!(await pageController.scrape(version, index))) {
                    logger.verbose('retrying', { file, version, retries });
                    continue;
                  } else {
                    break;
                  }
                } finally {
                  await pageController.dispose();
                }
              }
            }
          } catch (e) {
            logger.error('page controller error', {
              file,
              error: (e as Error)?.stack,
            });
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
