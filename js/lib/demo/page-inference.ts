import { Request, Response } from 'express';
import { unlink, writeFile } from 'fs/promises';
import puppeteer from 'puppeteer-core';
import { pathToFileURL } from 'url';
import { Logger } from 'winston';
import { Extractor } from '../extractor';
import { PageInfo } from '../page-info';
import { PageRecipe } from '../page-recipe';
import { ScrapeVersion } from '../scrape-version';
import { temporaryFilePath, tryParseInt } from '../utils';
import { DemoApp } from './app';
import { InferenceOutput } from './python';
import * as views from './views';

/** Inference of a page. */
export class PageInference {
  private readonly log: Logger;
  public readonly url: string;
  public readonly timeout: number;
  private page: puppeteer.Page | null = null;
  private snapshotPath: string | null = null;

  public constructor(
    public readonly app: DemoApp,
    private readonly req: Request,
    private readonly res: Response
  ) {
    // Parse parameters.
    this.url = this.req.query['url']?.toString() ?? '';
    this.timeout = tryParseInt(
      this.req.query['timeout'],
      this.app.options.timeout
    );
    this.log = this.app.log.child({ url: this.url });
  }

  public async run() {
    this.log.debug('run');

    // Start writing response so it's asynchronous.
    this.res.setHeader('Transfer-Encoding', 'chunked');
    this.res.flushHeaders();
    this.res.write(views.layoutStart());
    this.res.write(views.info(this.app.model.info, this.app.options));
    this.res.write(views.form(this));
    if (this.url === '') {
      // Return empty form if no URL was provided.
      this.res.write(views.layoutEnd(this.app.options));
      this.res.end();
      return;
    }
    this.res.write(views.logStart());

    // Wait for Puppeteer.
    if (this.app.browser instanceof Promise) {
      this.log.debug('wait for Puppeteer');
      this.res.write(
        views.logEntry('⏳ Waiting for Puppeteer one-time init...')
      );
      this.flushChunk();
      this.app.browser = await this.app.browser;
    }

    // Run through Puppeteer.
    this.log.debug('load page');
    this.res.write(views.logEntry('Loading page...'));
    this.flushChunk();
    this.page = await this.app.browser.newPage();
    this.page.setDefaultTimeout(this.timeout * 1000);
    const nav1 = await this.wrapNavigation((o) => this.page!.goto(this.url, o));
    if (nav1 === 'fail') return;

    // Capture snapshot. This can fail if navigation timeouts at a bad time.
    this.log.debug('snapshot');
    this.res.write(views.logEntry('Freezing page...'));
    this.flushChunk();
    this.snapshotPath = await temporaryFilePath('.mhtml');
    try {
      const cdp = await this.page.target().createCDPSession();
      const { data } = await cdp.send('Page.captureSnapshot', {
        format: 'mhtml',
      });
      await writeFile(this.snapshotPath, data, { encoding: 'utf-8' });
    } catch (e) {
      const error = e as Error;
      this.log.error('snapshot fail', { error: error?.stack });
      this.res.write(views.logEntry('⛔ Capturing snapshot failed.'));
      this.flushChunk();
    }

    // Page must be recreated if timeout occurred.
    if (nav1 === 'timeout') {
      this.log.debug('recreating page');
      await this.closePage();
      this.page = await this.app.browser.newPage();
      this.page.setDefaultTimeout(this.timeout * 1000);
    }

    // Freeze the page.
    this.log.debug('freeze page');
    try {
      this.page.setJavaScriptEnabled(false);
    } catch (e) {
      const error = e as Error;
      this.log.debug('freeze fail', { error: error?.stack });
      this.res.write(views.logEntry('⛔ Reloading failed.'));
      this.flushChunk();
    }
    const nav2 = await this.wrapNavigation((o) =>
      this.page!.goto(pathToFileURL(this.snapshotPath!).toString(), o)
    );
    if (nav2 === 'fail') return;

    // Extract HTML. This can fail if navigation timeouts at a bad time.
    this.log.debug('extract content');
    let html: string;
    try {
      html = await this.page.content();
    } catch (e) {
      const error = e as Error;
      this.log.error('content fail', { error: error?.stack });
      this.res.write(views.logEntry('⛔ HTML extraction failed.'));
      this.res.end();
      return;
    }

    // Extract visuals.
    this.log.debug('extract visuals');
    this.res.write(views.logEntry('Extracting visuals...'));
    this.flushChunk();
    const pageInfo = new PageInfo(this.url, this.url, html, /* isSwde */ false);
    const recipe = new PageRecipe(pageInfo, ScrapeVersion.Exact);
    const extractor = new Extractor(
      this.page,
      recipe,
      this.log,
      this.app.extractorOptions
    );
    const stats = await extractor.extract();
    this.log.debug('extracted', { stats });
    const visuals = extractor.data;

    // Take screenshot.
    this.log.debug('screenshot');
    this.res.write(views.logEntry('Taking screenshot...'));
    this.flushChunk();
    const screenshot = (await this.page.screenshot({
      fullPage: true,
      encoding: 'base64',
    })) as string;

    // Log full inputs if desired.
    if (this.app.options.logInputs) {
      this.log.silly('inputs', { html, visuals, screenshot });
    }

    // Pass HTML and visuals to Python.
    let response: InferenceOutput;
    if (this.app.python !== null) {
      // Wait for Python inference to fully load.
      const pythonLoading = this.app.python.loading;
      if (pythonLoading !== null) {
        this.log.debug('wait for Python');
        this.res.write(
          views.logEntry('⏳ Waiting for inference one-time init...')
        );
        this.flushChunk();
        await pythonLoading;
      }

      // Call Python inference.
      this.log.debug('inference');
      this.res.write(views.logEntry('Running inference...'));
      this.flushChunk();
      response = await this.app.python.send({
        url: this.url,
        html,
        visuals,
        screenshot,
      });
    } else {
      // Mock inference.
      const defaults = { url: null, box: null };
      response = {
        screenshot,
        pages: [
          {
            engine: [],
            fuel_economy: [
              {
                ...defaults,
                confidence: 4.112531661987305,
                probability: 0.9950769543647766,
                text: '22 / 29 mpg',
                xpath:
                  '/html/body/div[1]/main/div[1]/section[2]/div[1]/div/div[5]/div/div[1]/div[1]/div[2]/div[2]/text()',
              },
              {
                ...defaults,
                confidence: 1.1698349714279175,
                probability: 0.6030704975128174,
                text: '5/5',
                xpath:
                  '/html/body/div[1]/main/div[1]/section[2]/div[1]/div/div[5]/div/div[1]/div[3]/div[2]/div[2]/text()',
              },
            ],
            model: [],
            price: [
              {
                ...defaults,
                confidence: 1.6609370708465576,
                probability: 0.6562321186065674,
                text: '$25,377',
                xpath:
                  '/html/body/div[1]/main/div[1]/section[2]/div[1]/div/div[3]/div/table/tbody/tr[1]/td[3]/text()',
              },
            ],
          },
        ],
      };
    }
    this.res.write(views.logEntry('Rendering response...'));
    this.flushChunk();

    // Log full inputs if they haven't been logged already and there was an
    // error during inference.
    let rows = [];
    if ('error' in response) {
      if (!this.app.options.logInputs)
        this.log.silly('inputs', { html, visuals });
    } else {
      // Render table with results.
      const pagePred = response.pages[0];
      if (pagePred !== undefined) {
        for (const [labelKey, nodePreds] of Object.entries(pagePred)) {
          for (const nodePred of nodePreds) {
            rows.push({ labelKey, ...nodePred });
          }
        }
      }

      // Display most probable rows first.
      rows.sort((x, y) => y.probability - x.probability);
    }

    this.res.write(views.logEntry('Done.'));
    this.res.write(views.logEnd());
    this.res.write(views.results(rows, response.screenshot, stats));
    this.res.write(views.layoutEnd(this.app.options));
    this.res.end();
  }

  private async closePage() {
    if (this.page !== null) {
      try {
        await this.page.close();
        this.page = null;
      } catch (e) {
        const error = e as Error;
        this.log.error('closing failed', { error: error?.stack });
      }
    }
  }

  public async close() {
    this.log.debug('closing');

    await this.closePage();

    if (this.snapshotPath !== null) {
      try {
        await unlink(this.snapshotPath);
        this.snapshotPath = null;
      } catch (e) {
        const error = e as Error;
        this.log.error('unlink failed', { error: error?.stack });
      }
    }
  }

  private async wrapNavigation(
    navigator: (options: puppeteer.WaitForOptions) => Promise<unknown>
  ) {
    try {
      await navigator({ waitUntil: 'networkidle0' });
    } catch (e) {
      const error = e as Error;
      if (error?.name === 'TimeoutError') {
        this.log.debug('goto timeout', { error: error?.stack });
        this.res.write(views.logEntry('⚠️ Navigation timeout.'));
        return 'timeout';
      } else {
        this.log.debug('goto failed', { error: error?.stack });
        this.res.write(views.logEntry(`⛔ Navigation failed: ${error}`));
        this.res.end();
        return 'fail';
      }
    }
  }

  private flushChunk() {
    if (this.app.options.largeChunk !== null) {
      this.res.write(this.app.options.largeChunk);
    }
  }
}
