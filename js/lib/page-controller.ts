import { existsSync } from 'fs';
import { Controller } from './controller';
import { Extractor } from './extractor';
import { PageRecipe } from './page-recipe';
import { PageScraper } from './page-scraper';
import { ScrapeVersion, scrapeVersionToSwdeHandling } from './scrape-version';
import { SwdePage } from './swde-page';
import { addSuffix } from './utils';

/** {@link PageScraper} controller to scrape one {@link SwdePage}. */
export class PageController {
  private constructor(
    private readonly controller: Controller,
    private readonly page: SwdePage,
    private readonly pageScraper: PageScraper
  ) {}

  public static async create(controller: Controller, page: SwdePage) {
    const pageScraper = await controller.scraper.for(page);
    return new PageController(controller, page, pageScraper);
  }

  /** Scrapes the specified {@link version}. */
  public async scrape(version: ScrapeVersion) {
    const recipe = new PageRecipe(this.page, version);

    // Prepare dependencies.
    const extractor = new Extractor(
      this.pageScraper.page,
      recipe,
      this.pageScraper.logger
    );

    // Check if not already scraped.
    if (this.controller.skipExisting) {
      const jsonExists = existsSync(recipe.jsonPath);
      const htmlExists = existsSync(recipe.htmlPath);
      const metadata = {
        jsonExists,
        jsonPath: recipe.jsonPath,
        htmlExists,
        htmlPath: recipe.htmlPath,
      };
      if (jsonExists && htmlExists) {
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
    await this.pageScraper.start();

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

    if (!this.controller.skipExtraction) {
      // Save page HTML (can be different from original due to JavaScript
      // dynamically updating the DOM).
      const html = await this.pageScraper.page.content();
      await this.page.withHtml(html).saveAs(recipe.htmlPath);

      // Extract visual attributes.
      await extractor.extract();
      await extractor.save();
    }

    // Take screenshot.
    if (this.controller.takeScreenshot) {
      await this.screenshot(recipe.screenshotPath, { fullPage: false });
      await this.screenshot(recipe.screenshotPath, { fullPage: true });
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

  public async dispose() {
    await this.pageScraper.dispose();
  }
}
