import puppeteer from 'puppeteer-core';
import winston from 'winston';
import { SWDE_TIMESTAMP } from './constants';
import { ignoreUrl } from './ignore';
import { logger } from './logging';
import { Scraper } from './scraper';
import { SwdePage } from './swde-page';

/** Method of handling request to SWDE page. */
export const enum SwdeHandling {
  /** Serves HTML content from the dataset. */
  Offline,
  /** Redirects to latest available version in the WaybackMachine. */
  Wayback,
}

/** Browser navigator intercepting requests. */
export class PageScraper {
  private stopped = false;
  private readonly inProgress: Set<readonly [string, string]> = new Set();
  public swdeHandling = SwdeHandling.Offline;

  constructor(
    public readonly logger: winston.Logger,
    private readonly scraper: Scraper,
    private readonly swdePage: SwdePage,
    public readonly page: puppeteer.Page
  ) {
    // Handle events.
    page.on('request', this.onRequest.bind(this));
    page.on('error', (e) => logger.error('page error', { e }));
    page.on('console', (m) => logger.debug('page console', { text: m.text() }));
  }

  public get numWaiting() {
    return this.inProgress.size;
  }

  public static async create(scraper: Scraper, swdePage: SwdePage) {
    const page = await scraper.browser.newPage();
    const pageLogger = logger.child({ page: swdePage.id });
    const pageScraper = new PageScraper(pageLogger, scraper, swdePage, page);
    await pageScraper.initialize();
    return pageScraper;
  }

  private async initialize() {
    // Intercept requests.
    await this.page.setRequestInterception(true);

    // Ignore some errors that would prevent WaybackMachine redirection.
    await this.page.setBypassCSP(true);
    this.page.setDefaultTimeout(0); // disable timeout
  }

  private async onRequest(request: puppeteer.HTTPRequest) {
    try {
      await this.handleRequest(request);
    } catch (e) {
      const error = e as Error;
      // Ignore aborted requests after `stop()` has been called.
      const ignore = error?.message === 'Target closed' && this.stopped;
      this.logger.log(ignore ? 'debug' : 'error', 'request error', {
        url: request.url(),
        error: error?.stack,
      });
    }
  }

  private async handleRequest(request: puppeteer.HTTPRequest) {
    if (this.swdePage !== null && request.url() === this.swdePage.url) {
      await this.handleSwdePage(request, this.swdePage);
    } else {
      // Pass WaybackMachine redirects through.
      const redirectUrl = this.scraper.wayback.isArchiveRedirect(request);
      if (redirectUrl !== null) {
        this.logger.debug('redirected:', { url: request.url() });
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
        page.timestamp = 'offline';
        this.logger.verbose('request page replaced', {
          url: request.url(),
          path: page.fullPath,
        });
        await request.respond({
          body: page.html,
        });
        break;

      case SwdeHandling.Wayback:
        const timestamp = await this.scraper.wayback.closestTimestamp(
          request.url()
        );
        page.timestamp = timestamp;
        if (timestamp === null) {
          this.logger.error('redirect not found', { url: request.url() });
        } else {
          this.logger.verbose('redirect page', {
            url: request.url(),
            timestamp,
          });
          await this.handleExternalRequest(request, timestamp);
        }
        break;
    }
  }

  /**
   * Serves request either from local cache or redirects it to WaybackMachine.
   */
  private async handleExternalRequest(
    request: puppeteer.HTTPRequest,
    timestamp = SWDE_TIMESTAMP
  ) {
    const offline =
      this.scraper.allowOffline && !this.scraper.forceLive
        ? await this.scraper.cache.get(request.url(), timestamp)
        : undefined;
    if (offline) {
      await this.handleOfflineRequest(request, timestamp, offline);
    } else {
      if (offline === null && !this.scraper.forceLive) {
        // This request didn't complete last time, abort it.
        this.logger.debug('aborted', { url: request.url() });
        await request.abort();
        this.scraper.stats.aborted++;
        return;
      }

      if (!this.scraper.allowLive) {
        // In offline mode, act as if this endpoint was not available.
        this.logger.debug('disabled', { url: request.url() });
        await request.respond({ status: 404 });
        this.scraper.stats.increment(404);
        return;
      }

      // Ignore requests matching specified patterns.
      if (ignoreUrl(request.url())) {
        this.logger.debug('ignored', { url: request.url() });
        await request.abort();
        this.scraper.stats.ignored++;
        return;
      }

      await this.handleLiveRequest(request, timestamp);
    }
  }

  /** Handles request from local cache. */
  private async handleOfflineRequest(
    request: puppeteer.HTTPRequest,
    timestamp: string,
    offline: Partial<puppeteer.ResponseForRequest>
  ) {
    this.logger.debug('offline request', {
      url: request.url(),
      hash: this.scraper.cache.getHash(request.url(), timestamp),
    });
    await request.respond(offline);
    this.addToStats(offline);
    this.scraper.stats.offline++;
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
      this.scraper.wayback.parseArchiveUrl(request.url()) !== null
        ? request.url()
        : this.scraper.wayback.getArchiveUrl(request.url(), timestamp);
    this.logger.debug('live request', { archiveUrl });
    await request.continue({ url: archiveUrl });

    // Note that if WaybackMachine doesn't have the page archived at
    // exactly the provided timestamp, it will redirect. That's detected
    // by `isArchiveRedirect`.
    const response = await this.page.waitForResponse((res) => {
      if (res.url() === request.url() && res.status() !== 302) return true;
      const redirectUrl = this.scraper.wayback.isArchiveRedirect(res.request());
      if (redirectUrl === request.url()) return true;
      return false;
    });

    // Handle response.
    this.logger.debug('response', { url: response.url() });
    const body = await response.buffer();
    const headers = response.headers();
    const cached: puppeteer.ResponseForRequest = {
      status: response.status(),
      headers,
      contentType: headers['Content-Type'],
      body,
    };
    if (!this.inProgress.delete(inProgressEntry))
      throw new Error(`Failed to delete ${request.url()} (${timestamp})`);
    if (this.scraper.allowOffline) {
      // HACK: This header causes error in Puppeteer, remove it.
      delete cached.headers['x-archive-orig-set-cookie'];

      await this.scraper.cache.add(request.url(), timestamp, cached);
    }
    this.addToStats(cached);
    this.scraper.stats.live++;
  }

  private addToStats(response: Partial<puppeteer.ResponseForRequest>) {
    if (response.status === undefined) {
      this.scraper.stats.undef++;
    } else {
      this.scraper.stats.increment(response.status);
    }
  }

  public async start() {
    if (this.stopped) {
      throw new Error('Cannot start page scraper once stopped.');
    }

    // Navigate to page's URL. This will be intercepted in `onRequest`.
    try {
      await this.page.goto(this.swdePage.url, {
        waitUntil: 'networkidle0',
      });
    } catch (e: any) {
      if (e.name === 'TimeoutError') {
        // Ignore timeouts.
        this.logger.error('timeout');
      } else {
        throw e;
      }
    }
  }

  public async stop() {
    this.stopped = true;

    // Go offline.
    await this.page.setOfflineMode(true);

    for (const [url, timestamp] of this.inProgress) {
      this.logger.debug('unhandled', { url, timestamp });

      // Save as "aborted" for the next time.
      await this.scraper.cache.add(url, timestamp, null);
      this.scraper.stats.aborted++;
    }
  }
}
