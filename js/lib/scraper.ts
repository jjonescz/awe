import { readFile } from 'fs/promises';
import puppeteer from 'puppeteer-core';
import { Archive } from './archive';

// First character is BOM marker.
const BASE_TAG_REGEX = /^\uFEFF?<base href="([^\n]*)"\/>\w*\n(.*)/s;

/** Browser navigator intercepting requests. */
export class Scraper {
  private swdePage: SwdePage | null = null;

  private constructor(
    public readonly browser: puppeteer.Browser,
    public readonly page: puppeteer.Page,
    public readonly archive: Archive
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

    // Open file archive.
    const archive = await Archive.create();

    return new Scraper(browser, page, archive);
  }

  private async onRequest(request: puppeteer.HTTPRequest) {
    if (this.swdePage !== null && request.url() === this.swdePage.url) {
      // Replace request to SWDE page with its HTML content.
      console.log(
        'request page:',
        request.url(),
        'replaced with:',
        this.swdePage.fullPath
      );
      request.respond({
        body: this.swdePage.html,
      });
    } else {
      // Handle other requests from local archive if available or request them
      // from WaybackMachine if they are not stored yet.
      const response = await this.archive.getOrAdd(
        request.url(),
        async (url) => {
          console.log('request resume:', url);
          await request.continue();
          const response = await this.page.waitForResponse(url);
          const body = await response.buffer();
          const headers = response.headers();
          return {
            url,
            status: response.status(),
            headers,
            contentType: headers['Content-Type'],
            body,
          };
        }
      );
      console.log('request handled:', request.url());
      request.respond(response);
    }
  }

  public go(page: SwdePage) {
    // Navigate to page's URL. This will be intercepted in `onRequest`.
    this.swdePage = page;
    return this.page.goto(page.url, {
      waitUntil: 'networkidle2',
    });
  }

  public async dispose() {
    console.log('saving archive');
    await this.archive.save();
    await this.browser.close();
  }
}

/** Page from the SWDE dataset. */
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
