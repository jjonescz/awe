import { ScrapeVersion, scrapeVersionToString } from './scrape-version';
import { SwdePage } from './swde-page';
import { replaceExtension } from './utils';

/** Input for scraping one {@link version} of {@link SwdePage}. */
export class PageRecipe {
  public readonly suffix: string;

  public constructor(
    public readonly page: SwdePage,
    public readonly version: ScrapeVersion
  ) {
    this.suffix = `-${scrapeVersionToString(version)}`;
  }

  public get jsonPath() {
    return replaceExtension(this.page.fullPath, `${this.suffix}.json`);
  }

  public get htmlPath() {
    return replaceExtension(this.jsonPath, '.htm');
  }

  public get screenshotPath() {
    return replaceExtension(this.jsonPath, '.png');
  }
}
