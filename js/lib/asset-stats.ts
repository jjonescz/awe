import puppeteer from 'puppeteer-core';

export class AssetStats {
  /*** Maps page ID to its stats. */
  public readonly pages: Map<string, AssetPageStats> = new Map();

  public getPage(id: string) {
    let page = this.pages.get(id);
    if (page === undefined) {
      page = new AssetPageStats();
      this.pages.set(id, page);
    }
    return page;
  }
}

/**
 * How many CSS files were scraped. Can be used to determine which pages contain
 * valid visuals.
 */
export class AssetPageStats {
  public cssSuccess = 0;
  public cssError = 0;

  public success(
    request: puppeteer.HTTPRequest,
    response: Partial<puppeteer.ResponseForRequest>
  ) {
    this.cssSuccess++;
  }

  public error(request: puppeteer.HTTPRequest) {
    this.cssError++;
  }
}
