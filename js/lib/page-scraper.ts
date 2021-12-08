import puppeteer from 'puppeteer-core';
import winston from 'winston';
import { SWDE_TIMESTAMP } from './constants';
import { cleanHeaders, ignoreUrl } from './ignore';
import { logger } from './logging';
import { Scraper } from './scraper';
import { SwdePage } from './swde-page';
import { normalizeUrl, urlsEqual } from './utils';

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
  private destroyed = false;
  private readonly inProgress: Set<readonly [string, string]> = new Set();
  private readonly handled: Map<string, number> = new Map();
  private totalHandled = 0;
  public swdeHandling = SwdeHandling.Offline;

  constructor(
    public readonly logger: winston.Logger,
    private readonly scraper: Scraper,
    private readonly swdePage: SwdePage,
    public readonly page: puppeteer.Page
  ) {
    // Handle events.
    this.page.on('request', this.onRequest);
    this.page.on('error', this.onError);
    this.page.on('console', this.onConsole);
  }

  public get numWaiting() {
    return this.inProgress.size;
  }

  public static async create(scraper: Scraper, swdePage: SwdePage) {
    const page = await scraper.pagePool.acquire();
    const pageLogger = logger.child({ page: swdePage.id });
    return new PageScraper(pageLogger, scraper, swdePage, page);
  }

  private onError = (e: Error) => {
    this.logger.error('page error', { error: e?.stack });
  };

  private onConsole = (m: puppeteer.ConsoleMessage) => {
    this.logger.debug('page console', { text: m.text() });
  };

  private onRequest = async (request: puppeteer.HTTPRequest) => {
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
  };

  private async handleRequest(request: puppeteer.HTTPRequest) {
    // Check for infinite loops.
    const url = normalizeUrl(request.url());
    const numHandled = this.handled.get(url) ?? 0;
    if (numHandled >= 10) {
      await this.page.close();
      throw new Error(
        `One URL handled too many times (${numHandled}): ${request.url()}`
      );
    }
    this.handled.set(url, numHandled + 1);
    if (this.totalHandled >= 1000) {
      await this.page.close();
      throw new Error(
        `Too many requests (${this.totalHandled}) for one page: ` +
          `${this.swdePage.url}`
      );
    }
    this.totalHandled++;

    if (this.swdePage !== null && urlsEqual(request.url(), this.swdePage.url)) {
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
      // Ignore requests matching specified patterns.
      if (ignoreUrl(request.url())) {
        this.logger.debug('ignored', { url: request.url() });
        await request.abort();
        this.scraper.stats.ignored++;
        return;
      }

      if (this.scraper.rememberAborted && offline === null) {
        // This request didn't complete last time, abort it.
        this.logger.debug('aborted', { url: request.url() });
        await request.abort();
        this.scraper.stats.aborted++;
        return;
      }

      if (!this.scraper.allowLive) {
        // In offline mode, act as if this endpoint was not available.
        this.logger.verbose('disabled', { url: request.url() });
        await request.abort();
        this.scraper.stats.disabled++;
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
    // Header filter might have changed since last run, so we remove potentially
    // invalid headers here although we remove them also when saving the
    // request.
    cleanHeaders(offline);
    await request.respond(offline);
    this.addToStats(offline);
    this.scraper.stats.offline++;
  }

  /** Redirects request to WaybackMachine. */
  private async handleLiveRequest(
    request: puppeteer.HTTPRequest,
    timestamp: string
  ) {
    const url = normalizeUrl(request.url());

    // Save this request as "in progress".
    const inProgressEntry = [url, timestamp] as const;
    this.inProgress.add(inProgressEntry);

    // Redirect to `web.archive.org` unless it's already directed there.
    const archiveUrl =
      this.scraper.wayback.parseArchiveUrl(url) !== null
        ? url
        : this.scraper.wayback.getArchiveUrl(url, timestamp);
    this.logger.debug('live request', { archiveUrl });
    await request.continue({ url: archiveUrl });

    // Note that if WaybackMachine doesn't have the page archived at
    // exactly the provided timestamp, it will redirect. That's detected
    // by `isArchiveRedirect`.
    const response = await this.page.waitForResponse((res) => {
      if (urlsEqual(res.url(), url) && res.status() !== 302) return true;
      const redirectUrl = this.scraper.wayback.isArchiveRedirect(res.request());
      if (redirectUrl !== null && urlsEqual(redirectUrl, url)) return true;
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
      throw new Error(`Failed to delete ${url} (${timestamp})`);
    if (this.scraper.allowOffline) {
      cleanHeaders(cached);
      await this.scraper.cache.add(url, timestamp, cached);
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

    // Go online (revert offline mode set in `stop` which would be otherwise
    // preserved when using page pool).
    try {
      await this.page.setOfflineMode(false);
    } catch (e) {
      const error = e as Error;
      this.logger.error('cannot start page scraper', { error: error?.stack });
      await this.scraper.pagePool.destroy(this.page);
      this.destroyed = true;
      return false;
    }

    // Navigate to page's URL. This will be intercepted in `onRequest`.
    try {
      await this.page.goto(this.swdePage.url, {
        waitUntil: 'networkidle0',
      });
    } catch (e) {
      const error = e as Error;
      if (error?.name === 'TimeoutError') {
        // Handle timeouts.
        this.logger.verbose('timeout', { error: error?.stack });
        this.scraper.stats.timeout++;
      } else {
        this.logger.error('goto failed', { error: error?.stack });
      }
    }

    return true;
  }

  public async stop() {
    this.stopped = true;

    // Go offline.
    try {
      await this.page.setOfflineMode(true);
    } catch (e) {
      this.logger.error('stopping failed', { error: (e as Error)?.stack });
    }

    for (const [url, timestamp] of this.inProgress) {
      this.logger.debug('unhandled', { url, timestamp });

      // Save as "aborted" for the next time.
      await this.scraper.cache.add(url, timestamp, null);
      this.scraper.stats.unhandled++;
    }
  }

  public async dispose() {
    this.page.off('request', this.onRequest);
    this.page.off('error', this.onError);
    this.page.off('console', this.onConsole);
    if (!this.destroyed) await this.scraper.pagePool.release(this.page);
  }
}
