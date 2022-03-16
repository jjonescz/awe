import puppeteer from 'puppeteer-core';
import { AssetStats } from './asset-stats';
import { Cache } from './cache';
import { logger } from './logging';
import { PageInfo } from './page-info';
import { createPagePool, PagePoolOptions } from './page-pool';
import { PageScraper } from './page-scraper';
import { ScrapingStats } from './scraping-stats';
import { Wayback } from './wayback';

/** Common dependencies of all {@link PageScraper}s. */
export class Scraper {
  /** Allow live (online) requests if needed? */
  public allowLive = true;
  /** Load and save requests locally (in {@link Cache})? */
  public allowOffline = true;
  /** Force retry of all live (online) requests. */
  public forceLive = false;
  /** Avoid retrying requests aborted in previous runs. */
  public rememberAborted = false;
  public readonly stats = new ScrapingStats();
  public readonly assetStats = new AssetStats();
  public readonly pagePool;

  private constructor(
    public readonly wayback: Wayback,
    public readonly browser: puppeteer.Browser,
    public readonly cache: Cache,
    opts: PagePoolOptions
  ) {
    this.pagePool = createPagePool(this, opts);
  }

  public static async create(
    opts: PagePoolOptions & {
      executablePath: string;
      devtools: boolean;
    }
  ) {
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
      executablePath: opts.executablePath,
      devtools: opts.devtools,
    });

    // Open local cache.
    const cache = await Cache.create();

    // Load WaybackMachine API cache.
    const wayback = new Wayback();
    await wayback.loadResponses();

    return new Scraper(wayback, browser, cache, opts);
  }

  public async for(swdePage: PageInfo) {
    return await PageScraper.create(this, swdePage);
  }

  public save() {
    logger.debug('saving data');
    this.cache.save();
    this.wayback.saveResponses();
  }

  public async dispose() {
    this.save();
    await this.browser.close();
  }
}
