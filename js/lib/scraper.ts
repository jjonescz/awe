import puppeteer from 'puppeteer-core';
import { Archive } from './archive';
import { logger } from './logging';
import { PageScraper } from './page-scraper';
import { ScrapingStats } from './scraping-stats';
import { SwdePage } from './swde-page';
import { Wayback } from './wayback';

/** Common dependencies of all {@link PageScraper}s. */
export class Scraper {
  /** Allow live (online) requests if needed? */
  public allowLive = true;
  /** Load and save requests locally (in {@link Archive})? */
  public allowOffline = true;
  /** Force retry of all live (online) requests. */
  public forceLive = false;
  public readonly stats = new ScrapingStats();

  private constructor(
    public readonly wayback: Wayback,
    public readonly browser: puppeteer.Browser,
    public readonly archive: Archive
  ) {}

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

    // Open file archive.
    const archive = await Archive.create();

    // Load WaybackMachine API cache.
    const wayback = new Wayback();
    await wayback.loadResponses();

    return new Scraper(wayback, browser, archive);
  }

  public async for(swdePage: SwdePage) {
    return await PageScraper.create(this, swdePage);
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
