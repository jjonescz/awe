import { existsSync } from 'fs';
import { Controller } from './controller';
import { Extractor } from './extractor';
import { PageInfo } from './page-info';
import { PageRecipe } from './page-recipe';
import { PageScraper } from './page-scraper';
import { ScrapeVersion, scrapeVersionToSwdeHandling } from './scrape-version';

/** {@link PageScraper} controller to scrape one {@link SwdePage}. */
export class PageController {
  private constructor(
    private readonly controller: Controller,
    private readonly page: PageInfo,
    private readonly pageScraper: PageScraper
  ) {}

  public static async create(controller: Controller, page: PageInfo) {
    const pageScraper = await controller.scraper.for(page);
    return new PageController(controller, page, pageScraper);
  }

  /** Scrapes the specified {@link version}.
   *
   * @param index absolute index among all scraped pages.
   */
  public async scrape(version: ScrapeVersion, index: number) {
    const recipe = new PageRecipe(this.page, version);

    // Prepare dependencies.
    const extractor = new Extractor(
      this.pageScraper.page,
      recipe,
      this.pageScraper.logger,
      { extractXml: this.controller.extractXml }
    );

    // Check if not already scraped.
    const shouldTakeScreenshot =
      this.controller.takeScreenshot > 0 &&
      index % this.controller.takeScreenshot == 0;
    if (this.controller.skipExisting) {
      const jsonExists = existsSync(recipe.jsonPath);
      const jsonNeeded = !this.controller.skipExtraction && !jsonExists;
      const htmlExists = existsSync(recipe.htmlPath);
      const htmlNeeded = !this.controller.skipSave && !htmlExists;
      const screenshotExists = existsSync(recipe.screenshotFullPath);
      const screenshotNeeded = shouldTakeScreenshot && !screenshotExists;
      const metadata = {
        jsonNeeded,
        jsonPath: recipe.jsonPath,
        htmlNeeded,
        htmlPath: recipe.htmlPath,
        screenshotNeeded,
        screenshotPath: recipe.screenshotFullPath,
      };
      if (!jsonNeeded && !htmlNeeded && !screenshotNeeded) {
        this.pageScraper.logger.verbose('skipping', metadata);
        return;
      } else {
        this.pageScraper.logger.debug('not skipping', metadata);
      }
    }

    // Configure page scraper.
    this.pageScraper.swdeHandling = scrapeVersionToSwdeHandling(recipe.version);

    // Navigate to the page.
    this.pageScraper.logger.verbose('goto', {
      path: this.page.fullPath,
      suffix: recipe.suffix,
    });
    if (!(await this.pageScraper.start())) {
      return false;
    }

    // Freeze if desired.
    if (this.controller.freeze) {
      await this.pageScraper.freeze(recipe.mhtmlPath);
    }

    // Abort remaining requests.
    await this.pageScraper.stop();

    // Report stats.
    this.pageScraper.logger.verbose('stats', {
      stats: this.controller.scraper.stats,
    });

    // Save local cache.
    this.controller.scraper.save();

    if (this.page.timestamp === null) {
      // Couldn't find snapshot of this page in the WaybackMachine, abort early.
      this.pageScraper.logger.error('page not reached');
      return;
    }

    if (!this.controller.skipSave) {
      // Save page HTML (can be different from original due to JavaScript
      // dynamically updating the DOM).
      const html = await this.pageScraper.page.content();
      await this.page.withHtml(html).saveAs(recipe.htmlPath);
    }

    if (!this.controller.skipExtraction) {
      // Extract visual attributes.
      await extractor.extract();
      await extractor.save();
    }

    // Take screenshot.
    if (shouldTakeScreenshot) {
      await this.screenshot(recipe.screenshotPreviewPath, { fullPage: false });
      await this.screenshot(recipe.screenshotFullPath, { fullPage: true });
    }

    return true;
  }

  private async screenshot(fullPath: string, { fullPage = true } = {}) {
    this.pageScraper.logger.verbose('screenshot', { fullPath });
    await this.pageScraper.page.screenshot({
      path: fullPath,
      fullPage: fullPage,
    });
  }

  public async dispose() {
    await this.pageScraper.dispose();
  }
}
