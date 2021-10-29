import { readFile } from 'fs/promises';
import puppeteer from 'puppeteer-core';
import { Archive } from './archive';
import { Wayback } from './wayback';

// First character is BOM marker.
const BASE_TAG_REGEX = /^\uFEFF?<base href="([^\n]*)"\/>\w*\n(.*)/s;

export class ScrapingStats {
  /** Map from status code to number of occurrences. */
  public readonly status: Record<number, number> = {};
  public undef = 0;
  public aborted = 0;
  public offline = 0;
  public live = 0;

  public increment(statusCode: number) {
    this.status[statusCode] = (this.status[statusCode] ?? 0) + 1;
  }
}

/** Browser navigator intercepting requests. */
export class Scraper {
  private readonly wayback = new Wayback();
  private swdePage: SwdePage | null = null;
  private readonly inProgress: Set<string> = new Set();
  /** Allow live (online) requests if needed? */
  public allowLive = true;
  /** Load and save requests locally (in {@link Archive})? */
  public allowOffline = true;
  /** Force retry of all live (online) requests. */
  public forceLive = false;
  public readonly stats = new ScrapingStats();

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
      this.handleSwdePage(request, this.swdePage);
    } else {
      // Pass WaybackMachine redirects through.
      const redirectUrl = this.wayback.isArchiveRedirect(request);
      if (redirectUrl !== null) {
        console.log('redirected:', request.url());
        request.continue();
        return;
      }

      // Handle other requests from local archive if available or request them
      // from WaybackMachine if they are not stored yet.
      const offline =
        this.allowOffline && !this.forceLive
          ? await this.archive.get(request.url())
          : undefined;
      if (offline) {
        this.handleOfflineRequest(request, offline);
      } else {
        if (offline === null && !this.forceLive) {
          // This request didn't complete last time, abort it.
          console.log('aborted:', request.url());
          request.abort();
          this.stats.aborted++;
          return;
        }

        if (!this.allowLive) {
          // In offline mode, act as if this endpoint was not available.
          console.log('disabled:', request.url());
          request.respond({ status: 404 });
          this.stats.increment(404);
          return;
        }

        await this.handleLiveRequest(request);
      }
    }
  }

  /** Replaces request to SWDE page with its HTML content. */
  private handleSwdePage(request: puppeteer.HTTPRequest, page: SwdePage) {
    console.log(
      'request page:',
      request.url(),
      'replaced with:',
      page.fullPath
    );
    request.respond({
      body: page.html,
    });
  }

  /** Handles request from local archive. */
  private handleOfflineRequest(
    request: puppeteer.HTTPRequest,
    offline: Partial<puppeteer.ResponseForRequest>
  ) {
    console.log(
      'offline request:',
      request.url(),
      'hash:',
      this.archive.getHash(request.url())
    );
    request.respond(offline);
    this.addToStats(offline);
    this.stats.offline++;
  }

  /** Redirects request to WaybackMachine. */
  private async handleLiveRequest(request: puppeteer.HTTPRequest) {
    // Save this request as "in progress".
    this.inProgress.add(request.url());

    // Redirect to `web.archive.org`.
    const archiveUrl = this.wayback.getArchiveUrl(request.url());
    console.log('live request:', archiveUrl);
    await request.continue({ url: archiveUrl });

    // Note that if WaybackMachine doesn't have the page archived at
    // exactly the provided timestamp, it will redirect. That's detected
    // by `isArchiveRedirect`.
    const response = await this.page.waitForResponse((res) => {
      if (res.url() === request.url() && res.status() !== 302) return true;
      const redirectUrl = this.wayback.isArchiveRedirect(res.request());
      if (redirectUrl === request.url()) return true;
      return false;
    });

    // Handle response.
    console.log('response for:', response.url());
    const body = await response.buffer();
    const headers = response.headers();
    const archived: puppeteer.ResponseForRequest = {
      status: response.status(),
      headers,
      contentType: headers['Content-Type'],
      body,
    };
    this.inProgress.delete(request.url());
    if (this.allowOffline)
      this.archive.add(request.url(), archived, { force: this.forceLive });
    this.addToStats(archived);
    this.stats.live++;
  }

  private addToStats(response: Partial<puppeteer.ResponseForRequest>) {
    if (response.status === undefined) {
      this.stats.undef++;
    } else {
      this.stats.increment(response.status);
    }
  }

  public async go(page: SwdePage) {
    // Navigate to page's URL. This will be intercepted in `onRequest`.
    this.swdePage = page;
    try {
      await this.page.goto(page.url, {
        waitUntil: 'networkidle2',
      });
    } catch (e: any) {
      if (e.name === 'TimeoutError') {
        // Ignore timeouts.
        console.log('timeout');
      } else {
        throw e;
      }
    }
  }

  public stop() {
    for (const url of this.inProgress) {
      console.log('unhandled:', url);

      // Save as "aborted" for the next time.
      this.archive.add(url, null);
      this.stats.aborted++;
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
