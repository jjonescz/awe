import { readFile, writeFile } from 'fs/promises';
import puppeteer from 'puppeteer-core';
import { Archive } from './archive';
import { SWDE_TIMESTAMP } from './constants';
import { ignoreUrl } from './ignore';
import { logger } from './logging';
import { nameOf, Writable } from './utils';
import { Wayback } from './wayback';

// First character is UTF-8 BOM marker.
const BASE_TAG_REGEX = /^\uFEFF?<base href="([^\n]*)"\/>\w*\n(.*)/s;

export class ScrapingStats {
  /** Map from status code to number of occurrences. */
  public readonly status: Record<number, number> = {};
  public undef = 0;
  public aborted = 0;
  public offline = 0;
  public live = 0;
  public ignored = 0;

  public increment(statusCode: number) {
    this.status[statusCode] = (this.status[statusCode] ?? 0) + 1;
  }

  public *iterateStrings() {
    for (const key in this) {
      if (key !== nameOf<ScrapingStats>('status')) {
        const value = this[key] as unknown as number;
        if (value !== 0) {
          yield `${key}: ${value}`;
        }
      }
    }

    for (const [code, count] of Object.entries(this.status)) {
      if (count !== 0) {
        yield `${code}: ${count}`;
      }
    }
  }

  public toString() {
    return [...this.iterateStrings()].join(', ');
  }
}

/** Method of handling request to SWDE page. */
export const enum SwdeHandling {
  /** Serves HTML content from the dataset. */
  Offline,
  /** Redirects to latest available version in the WaybackMachine. */
  Wayback,
}

/** Browser navigator intercepting requests. */
export class Scraper {
  private swdePage: SwdePage | null = null;
  private readonly inProgress: Set<readonly [string, string]> = new Set();
  /** Allow live (online) requests if needed? */
  public allowLive = true;
  /** Load and save requests locally (in {@link Archive})? */
  public allowOffline = true;
  /** Force retry of all live (online) requests. */
  public forceLive = false;
  public swdeHandling = SwdeHandling.Offline;
  public readonly stats = new ScrapingStats();

  private constructor(
    public readonly wayback: Wayback,
    public readonly browser: puppeteer.Browser,
    public readonly page: puppeteer.Page,
    public readonly archive: Archive
  ) {
    // Handle events..
    page.on('request', this.onRequest.bind(this));
    page.on('error', (e) => logger.error('page error', { e }));
    page.on('console', (m) => logger.debug('page console', { text: m.text() }));
  }

  public get numWaiting() {
    return this.inProgress.size;
  }

  private async initialize() {
    // Intercept requests.
    await this.page.setRequestInterception(true);

    // Ignore some errors that would prevent WaybackMachine redirection.
    await this.page.setBypassCSP(true);
    this.page.setDefaultTimeout(0); // disable timeout
  }

  public static async create() {
    // Open browser.
    const browser = await puppeteer.launch({
      args: [
        // Allow running as root.
        '--no-sandbox',
        // Improve headless running.
        '--ignore-certificate-errors',
        '--disable-setuid-sandbox',
        '--disable-accelerated-2d-canvas',
        '--disable-gpu',
        '--disable-web-security',
        '--disable-features=IsolateOrigins',
        '--disable-site-isolation-trials',
        '--disable-features=BlockInsecurePrivateNetworkRequests',
      ],
      executablePath: 'google-chrome-stable',
    });
    const page = await browser.newPage();

    // Open file archive.
    const archive = await Archive.create();

    // Load WaybackMachine API cache.
    const wayback = new Wayback();
    await wayback.loadResponses();

    const scraper = new Scraper(wayback, browser, page, archive);
    await scraper.initialize();
    return scraper;
  }

  private async onRequest(request: puppeteer.HTTPRequest) {
    try {
      await this.handleRequest(request);
    } catch (e) {
      logger.error('request error', {
        url: request.url(),
        error: (e as puppeteer.CustomError)?.message,
      });
    }
  }

  private async handleRequest(request: puppeteer.HTTPRequest) {
    if (this.swdePage !== null && request.url() === this.swdePage.url) {
      await this.handleSwdePage(request, this.swdePage);
    } else {
      // Pass WaybackMachine redirects through.
      const redirectUrl = this.wayback.isArchiveRedirect(request);
      if (redirectUrl !== null) {
        logger.debug('redirected:', { url: request.url() });
        await request.continue();
        return;
      }

      await this.handleExternalRequest(request);
    }
  }

  /** Handles request to SWDE page. */
  private async handleSwdePage(request: puppeteer.HTTPRequest, page: SwdePage) {
    switch (this.swdeHandling) {
      case SwdeHandling.Offline:
        page.timestamp = null;
        logger.verbose('request page replaced', {
          url: request.url(),
          path: page.fullPath,
        });
        await request.respond({
          body: page.html,
        });
        break;

      case SwdeHandling.Wayback:
        const timestamp = await this.wayback.closestTimestamp(request.url());
        page.timestamp = timestamp;
        if (timestamp === null) {
          logger.error('redirect not found', { url: request.url() });
        } else {
          logger.verbose('redirect page', { url: request.url(), timestamp });
          await this.handleExternalRequest(request, timestamp);
        }
        break;
    }
  }

  /**
   * Serves request either from local archive or redirects it to WaybackMachine.
   */
  private async handleExternalRequest(
    request: puppeteer.HTTPRequest,
    timestamp = SWDE_TIMESTAMP
  ) {
    const offline =
      this.allowOffline && !this.forceLive
        ? await this.archive.get(request.url(), timestamp)
        : undefined;
    if (offline) {
      await this.handleOfflineRequest(request, timestamp, offline);
    } else {
      if (offline === null && !this.forceLive) {
        // This request didn't complete last time, abort it.
        logger.debug('aborted', { url: request.url() });
        await request.abort();
        this.stats.aborted++;
        return;
      }

      if (!this.allowLive) {
        // In offline mode, act as if this endpoint was not available.
        logger.debug('disabled', { url: request.url() });
        await request.respond({ status: 404 });
        this.stats.increment(404);
        return;
      }

      // Ignore requests matching specified patterns.
      if (ignoreUrl(request.url())) {
        logger.debug('ignored', { url: request.url() });
        await request.abort();
        this.stats.ignored++;
        return;
      }

      await this.handleLiveRequest(request, timestamp);
    }
  }

  /** Handles request from local archive. */
  private async handleOfflineRequest(
    request: puppeteer.HTTPRequest,
    timestamp: string,
    offline: Partial<puppeteer.ResponseForRequest>
  ) {
    logger.debug('offline request', {
      url: request.url(),
      hash: this.archive.getHash(request.url(), timestamp),
    });
    await request.respond(offline);
    this.addToStats(offline);
    this.stats.offline++;
  }

  /** Redirects request to WaybackMachine. */
  private async handleLiveRequest(
    request: puppeteer.HTTPRequest,
    timestamp: string
  ) {
    // Save this request as "in progress".
    const inProgressEntry = [request.url(), timestamp] as const;
    this.inProgress.add(inProgressEntry);

    // Redirect to `web.archive.org` unless it's already directed there.
    const archiveUrl =
      this.wayback.parseArchiveUrl(request.url()) !== null
        ? request.url()
        : this.wayback.getArchiveUrl(request.url(), timestamp);
    logger.debug('live request', { archiveUrl });
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
    logger.debug('response', { url: response.url() });
    const body = await response.buffer();
    const headers = response.headers();
    const archived: puppeteer.ResponseForRequest = {
      status: response.status(),
      headers,
      contentType: headers['Content-Type'],
      body,
    };
    if (!this.inProgress.delete(inProgressEntry))
      throw new Error(`Failed to delete ${request.url()} (${timestamp})`);
    if (this.allowOffline)
      await this.archive.add(request.url(), timestamp, archived, {
        force: this.forceLive,
      });
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
    // Undo effect of method `stop`.
    await this.page.setOfflineMode(false);

    // Navigate to page's URL. This will be intercepted in `onRequest`.
    this.swdePage = page;
    try {
      await this.page.goto(page.url, {
        waitUntil: 'networkidle0',
      });
    } catch (e: any) {
      if (e.name === 'TimeoutError') {
        // Ignore timeouts.
        logger.error('timeout');
      } else {
        throw e;
      }
    }
  }

  public async stop() {
    // Go offline.
    await this.page.setOfflineMode(true);

    for (const [url, timestamp] of this.inProgress) {
      logger.debug('unhandled', { url, timestamp });

      // Save as "aborted" for the next time.
      await this.archive.add(url, timestamp, null, { force: this.forceLive });
      this.stats.aborted++;
    }
  }

  public async save() {
    logger.debug('saving data');
    await this.archive.save();
    await this.wayback.saveResponses();
  }

  public async dispose() {
    await this.save();
    await this.browser.close();
  }
}

/** Page from the SWDE dataset. */
export class SwdePage {
  /** Timestamp used to scrape this page. */
  public timestamp: string | null = null;

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

  public withHtml(html: string) {
    const clone = Object.create(this) as Writable<SwdePage>;
    clone.html = html;
    return clone as SwdePage;
  }

  public stringify() {
    // First character is UTF-8 BOM marker.
    return `\uFEFF<base href="${this.url}"/>\n${this.html}`;
  }

  public async saveAs(fullPath: string) {
    const contents = this.stringify();
    await writeFile(fullPath, contents, { encoding: 'utf-8' });
  }
}
