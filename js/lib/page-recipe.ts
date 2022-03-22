import { PageInfo } from './page-info';
import { ScrapeVersion, scrapeVersionToString } from './scrape-version';
import { replaceExtension } from './utils';

/** Input for scraping one {@link version} of {@link SwdePage}. */
export class PageRecipe {
  public readonly suffix: string;

  public constructor(
    public readonly page: PageInfo,
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

  public get screenshotFullPath() {
    return replaceExtension(this.jsonPath, '-full.png');
  }

  public get screenshotPreviewPath() {
    return replaceExtension(this.jsonPath, '-preview.png');
  }
}
