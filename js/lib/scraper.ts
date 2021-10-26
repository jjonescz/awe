import { readFile } from 'fs/promises';
import puppeteer from 'puppeteer-core';

// First character is BOM marker.
const BASE_TAG_REGEX = /^\uFEFF?<base href="([^\n]*)"\/>\w*\n(.*)/s;

export class Scraper {
  private swdePage: SwdePage | null = null;

  private constructor(
    public readonly browser: puppeteer.Browser,
    public readonly page: puppeteer.Page
  ) {
    // Intercept requests.
    page.setRequestInterception(true);
    page.on('request', this.onRequest.bind(this));
  }

  public static async create() {
    // Open browser.
    const browser = await puppeteer.launch({
      args: [
        // Allow running as root.
        '--no-sandbox',
      ],
      executablePath: 'google-chrome-stable',
    });
    const page = await browser.newPage();

    return new Scraper(browser, page);
  }

  private onRequest(request: puppeteer.HTTPRequest) {
    if (this.swdePage !== null && request.url() === this.swdePage.url) {
      // Replace request to SWDE page with its HTML content.
      console.log(
        'request page: ',
        request.url(),
        ' replaced with: ',
        this.swdePage.fullPath
      );
      request.respond({
        body: this.swdePage.html,
      });
    } else {
      // Abort other requests for now.
      console.log('request aborted: ', request.url());
      request.abort();
    }
  }

  public go(page: SwdePage) {
    // Navigate to page's URL. This will be intercepted in `onRequest`.
    this.swdePage = page;
    return this.page.goto(page.url, {
      waitUntil: 'networkidle2',
    });
  }
}

export class SwdePage {
  public constructor(
    public readonly fullPath: string,
    public readonly url: string,
    public readonly html: string
  ) {}

  public static async parse(fullPath: string) {
    const contents = await readFile(fullPath, { encoding: 'utf-8' });
    // Extract original page URL from a `<base>` tag that is at the beginning of
    // every HTML file in SWDE.
    const [_, url, html] = contents.match(BASE_TAG_REGEX)!;
    return new SwdePage(fullPath, url, html);
  }
}
