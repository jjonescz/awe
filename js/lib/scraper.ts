import { readFile } from 'fs/promises';
import puppeteer, { HTTPRequest } from 'puppeteer-core';
import { Archive } from './archive';
import { SWDE_TIMESTAMP } from './constants';

// First character is BOM marker.
const BASE_TAG_REGEX = /^\uFEFF?<base href="([^\n]*)"\/>\w*\n(.*)/s;

const ARCHIVE_URL_REGEX = /^https:\/\/web.archive.org\/web\/(\d{14})id_\/(.*)$/;

/** Browser navigator intercepting requests. */
export class Scraper {
  private swdePage: SwdePage | null = null;
  private readonly inProgress: Set<string> = new Set();

  private constructor(
    public readonly browser: puppeteer.Browser,
    public readonly page: puppeteer.Page,
    public readonly archive: Archive
  ) {
    // Intercept requests.
    page.setRequestInterception(true);
    page.on('request', this.onRequest.bind(this));
  }

  public get numWaiting() {
    return this.inProgress.size;
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
      // Pass WaybackMachine redirects through.
      const redirectUrl = this.isArchiveRedirect(request);
      if (redirectUrl !== null) {
        console.log('redirected:', request.url());
        request.continue();
        return;
      }

      // Handle other requests from local archive if available or request them
      // from WaybackMachine if they are not stored yet.
      const offline = await this.archive.get(request.url());
      if (offline) {
        console.log(
          'offline request:',
          request.url(),
          'hash:',
          this.archive.getHash(request.url())
        );
        request.respond(offline);
      } else {
        if (offline === null) {
          // This request was aborted last time.
          console.log('aborted:', request.url());
          request.abort();
          return;
        }

        // Save this request as "in progress" if not already.
        if (!this.inProgress.has(request.url())) {
          this.inProgress.add(request.url());

          // Redirect to `web.archive.org`.
          const archiveUrl = this.getArchiveUrl(request.url());
          console.log('live request:', archiveUrl);
          await request.continue({ url: archiveUrl });
        } else {
          console.log('waiting:', request.url());
        }

        // Note that if WaybackMachine doesn't have the page archived at
        // exactly the provided timestamp, it will redirect. That's detected
        // by `isArchiveRedirect`.
        const response = await this.page.waitForResponse((res) => {
          if (res.url() === request.url() && res.status() !== 302) return true;
          const redirectUrl = this.isArchiveRedirect(res.request());
          if (redirectUrl === request.url()) return true;
          return false;
        });

        // Handle response.
        console.log('response for:', response.url());
        const body = await response.buffer();
        const headers = response.headers();
        const archived = {
          status: response.status(),
          headers,
          contentType: headers['Content-Type'],
          body,
        };
        this.inProgress.delete(request.url());
        this.archive.add(request.url(), archived);
      }
    }
  }

  private getArchiveUrl(url: string) {
    // For URL scheme, see
    // https://en.wikipedia.org/wiki/Help:Using_the_Wayback_Machine#Specific_archive_copy.
    return `https://web.archive.org/web/${SWDE_TIMESTAMP}id_/${url}`;
  }

  private isArchiveRedirect(request: HTTPRequest) {
    const match = request.url().match(ARCHIVE_URL_REGEX);
    if (match === null) return null;
    const [_full, _date, url] = match;
    const chain = request.redirectChain();
    if (chain.length !== 1) return null;
    const prev = chain[0];
    if (prev.url() !== url) return null;
    return url;
  }

  public go(page: SwdePage) {
    // Navigate to page's URL. This will be intercepted in `onRequest`.
    this.swdePage = page;
    return this.page.goto(page.url, {
      waitUntil: 'networkidle2',
    });
  }

  public stop() {
    for (const url of this.inProgress) {
      console.log('unhandled:', url);

      // Save as "aborted" for the next time.
      this.archive.add(url, null);
    }
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
