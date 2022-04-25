import express, { Request, Response } from 'express';
import puppeteer from 'puppeteer-core';
import { Logger } from 'winston';
import { Extractor } from '../extractor';
import { logFile } from '../logging';
import { PageInfo } from '../page-info';
import { PageRecipe } from '../page-recipe';
import { ScrapeVersion } from '../scrape-version';
import { tryParseInt } from '../utils';
import { loadModel, ModelInfo } from './model-info';
import { Inference, InferenceOutput, NodePrediction } from './python';
import * as views from './views';

export class DemoOptions {
  public readonly debug: boolean;
  /** More verbose logging. */
  public readonly port: number;
  /** Log full HTML and visuals to `scraping_logs`. */
  public readonly logInputs: boolean;
  /** Set when developing server UI to avoid waiting for Python. */
  public readonly mockInference: boolean;
  /** Puppeteer page loading timeout in seconds. */
  public readonly timeout: number;
  /** Send artificially large response chunks to bypass network buffering. */
  public readonly largeChunks: number;

  public constructor() {
    this.debug = !!process.env.DEBUG;
    this.port = tryParseInt(process.env.PORT, 3000);
    this.logInputs = !!process.env.LOG_INPUTS;
    this.mockInference = !!process.env.MOCK_INFERENCE;
    this.timeout = tryParseInt(process.env.TIMEOUT, 15);
    this.largeChunks = tryParseInt(process.env.LARGE_CHUNKS, 0);
  }

  public get largeChunk() {
    if (this.largeChunks > 0) {
      return '<!--' + 'x'.repeat(this.largeChunks) + '-->';
    } else {
      return null;
    }
  }
}

export class DemoApp {
  private readonly python: Inference | null = null;
  private browser: Promise<puppeteer.Browser> | puppeteer.Browser;

  private constructor(
    private readonly options: DemoOptions,
    private readonly log: Logger,
    private readonly model: ModelInfo
  ) {
    // Create Express HTTP server.
    const app = express();
    app.get('/', this.mainPage);

    // Start the server.
    log.verbose('starting demo server');
    const server = app.listen(options.port, async () => {
      console.log(`Listening on http://localhost:${options.port}/`);
    });

    // Create Puppeteer.
    this.browser = new Promise<puppeteer.Browser>(async (resolve) => {
      log.verbose('opening Puppeteer');
      const browser = await puppeteer.launch({
        args: [
          // Allow running as root.
          '--no-sandbox',
        ],
        executablePath: 'google-chrome-stable',
      });
      log.verbose('opened Puppeteer');
      resolve(browser);
      this.browser = browser;
    });

    if (!options.mockInference) {
      // Start Python inference shell.
      this.python = new Inference(options, log);

      // Close the server when Python closes.
      this.python.shell.on('close', () => {
        log.verbose('closing server');
        setTimeout(() => {
          log.error('closing timeout');
          process.exit(2);
        }, 5000);
        server.close((err) => {
          log.verbose('closed server', { err });
          process.exit(1);
        });
      });
    }
  }

  public static async start(options: DemoOptions, logger: Logger) {
    const log = logger.child({});
    log.info('start', { options, logLevel: log.level, logFile });

    // Load model info.
    const model = await loadModel(log);
    log.verbose('loaded model info', { model });

    return new DemoApp(options, log, model);
  }

  private mainPage = async (req: Request, res: Response) => {
    // Parse parameters.
    const url = req.query['url']?.toString() ?? '';
    const timeout = tryParseInt(req.query['timeout'], this.options.timeout);
    const log = this.log.child({ url: url });
    log.debug('run');

    // Start writing response so it's asynchronous.
    res.setHeader('Transfer-Encoding', 'chunked');
    res.flushHeaders();
    res.write(views.layoutStart());
    res.write(views.info(this.model));
    res.write(views.form(this.model, { url }));
    if (url === '') {
      // Return empty form if no URL was provided.
      res.write(views.layoutEnd());
      res.end();
      return;
    }
    res.write(views.logStart());

    // Wait for Puppeteer.
    if (this.browser instanceof Promise) {
      log.debug('wait for Puppeteer');
      res.write(views.logEntry('Waiting for Puppeteer one-time init...'));
      this.flushChunk(res);
      this.browser = await this.browser;
    }

    // Run through Puppeteer.
    log.debug('load page');
    res.write(views.logEntry('Loading page...'));
    this.flushChunk(res);
    let page = await this.browser.newPage();
    page.setDefaultTimeout(timeout * 1000);
    const nav1 = await DemoApp.wrapNavigation(
      (o) => page.goto(url, o),
      res,
      log
    );
    if (nav1 === 'fail') return;

    // Capture snapshot. This can fail if navigation timeouts at a bad time.
    log.debug('snapshot');
    res.write(views.logEntry('Freezing page...'));
    this.flushChunk(res);
    let snapshot: string;
    try {
      const cdp = await page.target().createCDPSession();
      const { data } = await cdp.send('Page.captureSnapshot', {
        format: 'mhtml',
      });
      snapshot = data;
    } catch (e) {
      const error = e as Error;
      log.error('snapshot fail', { error: error?.stack });
      res.write(views.logEntry('Capturing snapshot failed.'));
      this.flushChunk(res);
    }

    // Page must be recreated if it timeout occurred.
    if (nav1 === 'timeout') {
      log.debug('recreating page');
      try {
        page.close();
      } catch (e) {
        log.error('page closing failed', { error: (e as Error)?.stack });
      }
      page = await this.browser.newPage();
      page.setDefaultTimeout(timeout * 1000);
    }

    // Freeze the page.
    log.debug('freeze page');
    try {
      page.setJavaScriptEnabled(false);
    } catch (e) {
      const error = e as Error;
      log.debug('freeze fail', { error: error?.stack });
      res.write(views.logEntry('Reloading failed.'));
      this.flushChunk(res);
    }
    const nav2 = await DemoApp.wrapNavigation(
      (o) => page.setContent(snapshot, o),
      res,
      log
    );
    if (nav2 === 'fail') return;

    // Extract HTML. This can fail if navigation timeouts at a bad time.
    log.debug('extract content');
    let html: string;
    try {
      html = await page.content();
    } catch (e) {
      const error = e as Error;
      log.error('content fail', { error: error?.stack });
      res.write(views.logEntry('HTML extraction failed.'));
      res.end();
      return;
    }

    // Extract visuals.
    log.debug('extract visuals');
    res.write(views.logEntry('Extracting visuals...'));
    this.flushChunk(res);
    const pageInfo = new PageInfo(url, url, html, /* isSwde */ false);
    const recipe = new PageRecipe(pageInfo, ScrapeVersion.Exact);
    const extractor = new Extractor(page, recipe, log, /* extractXml */ false);
    await extractor.extract();
    const visuals = extractor.data;

    // Take screenshot.
    log.debug('screenshot');
    res.write(views.logEntry('Taking screenshot...'));
    this.flushChunk(res);
    const screenshot = (await page.screenshot({
      fullPage: true,
      encoding: 'base64',
    })) as string;

    // Log full inputs if desired.
    if (this.options.logInputs) {
      log.silly('inputs', { html, visuals, screenshot });
    }

    // Pass HTML and visuals to Python.
    let response: InferenceOutput;
    if (this.python !== null) {
      // Wait for Python inference to fully load.
      const pythonLoading = this.python.loading;
      if (pythonLoading !== null) {
        log.debug('wait for Python');
        res.write(views.logEntry('Waiting for inference one-time init...'));
        this.flushChunk(res);
        await pythonLoading;
      }

      // Call Python inference.
      log.debug('inference');
      res.write(views.logEntry('Running inference...'));
      this.flushChunk(res);
      response = await this.python.send({ url, html, visuals, screenshot });
    } else {
      // Mock inference.
      response = {
        screenshot,
        pages: [
          {
            engine: [],
            fuel_economy: [
              {
                url: null,
                confidence: 4.112531661987305,
                probability: 0.9950769543647766,
                text: '22 / 29 mpg',
                xpath:
                  '/html/body/div[1]/main/div[1]/section[2]/div[1]/div/div[5]/div/div[1]/div[1]/div[2]/div[2]/text()',
              },
              {
                url: null,
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
                url: null,
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
    res.write(views.logEntry('Rendering response...'));
    this.flushChunk(res);

    // Log full inputs if they haven't been logged already and there was an
    // error during inference.
    let rows = [];
    if ('error' in response) {
      if (!this.options.logInputs) log.silly('inputs', { html, visuals });
    } else {
      // Render table with results.
      const pagePred = response.pages[0];
      for (const [labelKey, nodePreds] of Object.entries<NodePrediction[]>(
        pagePred
      )) {
        for (const nodePred of nodePreds) {
          rows.push({ labelKey, ...nodePred });
        }
      }
      rows.sort((x, y) => y.confidence - x.confidence);
    }

    res.write(views.logEntry('Done.'));
    res.write(views.logEnd());
    res.write(views.results(rows, response.screenshot));
    res.write(views.layoutEnd());
    res.end();
  };

  private static async wrapNavigation(
    navigator: (options: puppeteer.WaitForOptions) => Promise<unknown>,
    res: Response,
    log: Logger
  ) {
    try {
      await navigator({ waitUntil: 'networkidle0' });
    } catch (e) {
      const error = e as Error;
      if (error?.name === 'TimeoutError') {
        log.debug('goto timeout', { error: error?.stack });
        res.write(views.logEntry('Navigation timeout.'));
        return 'timeout';
      } else {
        log.debug('goto failed', { error: error?.stack });
        res.write(views.logEntry(`Navigation failed: ${error}`));
        res.end();
        return 'fail';
      }
    }
  }

  private flushChunk(res: Response) {
    if (this.options.largeChunk !== null) {
      res.write(this.options.largeChunk);
    }
  }
}
