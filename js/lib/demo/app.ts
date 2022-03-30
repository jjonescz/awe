import express, { Request, Response } from 'express';
import puppeteer from 'puppeteer-core';
import { Logger } from 'winston';
import { Extractor } from '../extractor';
import { logFile } from '../logging';
import { PageInfo } from '../page-info';
import { PageRecipe } from '../page-recipe';
import { ScrapeVersion } from '../scrape-version';
import { loadModel, ModelInfo } from './model-info';
import { Inference, NodePrediction } from './python';
import * as views from './views';

export class DemoOptions {
  public readonly debug: boolean;
  /** More verbose logging. */
  public readonly port: number;
  /** Log full HTML and visuals to `scraping_logs`. */
  public readonly logInputs: boolean;
  /** Set when developing server UI to avoid waiting for Python. */
  public readonly mockInference: boolean;

  public constructor() {
    this.debug = !!process.env.DEBUG;
    this.port = parseInt(process.env.PORT || '3000');
    this.logInputs = !!process.env.LOG_INPUTS;
    this.mockInference = !!process.env.MOCK_INFERENCE;
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
    const model = await loadModel();
    log.verbose('loaded model info', { model });

    return new DemoApp(options, log, model);
  }

  private mainPage = async (req: Request, res: Response) => {
    // Parse parameters.
    const url = req.query['url']?.toString() ?? '';
    const log = this.log.child({ url: url });
    log.debug('run');

    // Start writing response so it's asynchronous.
    res.setHeader('transfer-encoding', 'chunked');
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
      this.browser = await this.browser;
    }

    // Run through Puppeteer.
    log.debug('load page');
    res.write(views.logEntry('Loading page...'));
    const page = await this.browser.newPage();
    page.setDefaultTimeout(0); // disable default timeout
    page.setDefaultNavigationTimeout(10_000); // 10 seconds
    const nav1 = await DemoApp.wrapNavigation(
      (o) => page.goto(url, o),
      res,
      log
    );
    if (nav1 === 'fail') return;
    const html = await page.content();

    // Freeze the page.
    if (nav1 !== 'timeout') {
      log.debug('freeze page');
      res.write(views.logEntry('Freezing page...'));
      page.setJavaScriptEnabled(false);
      const nav2 = await DemoApp.wrapNavigation(
        (o) => page.setContent(html, o),
        res,
        log
      );
      if (nav2 === 'fail') return;
    }

    // Extract visuals.
    log.debug('extract visuals');
    res.write(views.logEntry('Extracting visuals...'));
    const pageInfo = new PageInfo(url, url, html, /* isSwde */ false);
    const recipe = new PageRecipe(pageInfo, ScrapeVersion.Exact);
    const extractor = new Extractor(page, recipe, log, /* extractXml */ false);
    await extractor.extract();
    const visuals = extractor.data;

    // Log full inputs if desired.
    if (this.options.logInputs) {
      log.silly('inputs', { html, visuals });
    }

    // Pass HTML and visuals to Python.
    let response: any;
    if (this.python !== null) {
      // Wait for Python inference to fully load.
      const pythonLoading = this.python.loading;
      if (pythonLoading !== null) {
        log.debug('wait for Python');
        res.write(views.logEntry('Waiting for inference one-time init...'));
        await pythonLoading;
      }

      // Call Python inference.
      log.debug('inference');
      res.write(views.logEntry('Running inference...'));
      response = await this.python.send({ url, html, visuals });
    } else {
      // Mock inference.
      response = [
        {
          engine: [],
          fuel_economy: [
            {
              confidence: 4.112531661987305,
              probability: 0.9950769543647766,
              text: '22 / 29 mpg',
              xpath:
                '/html/body/div[1]/main/div[1]/section[2]/div[1]/div/div[5]/div/div[1]/div[1]/div[2]/div[2]/text()',
            },
            {
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
              confidence: 1.6609370708465576,
              probability: 0.6562321186065674,
              text: '$25,377',
              xpath:
                '/html/body/div[1]/main/div[1]/section[2]/div[1]/div/div[3]/div/table/tbody/tr[1]/td[3]/text()',
            },
          ],
        },
      ];
    }
    log.debug('response', { response });
    res.write(views.logEntry('Rendering screenshot...'));

    // Log full inputs if they haven't been logged already and there was an
    // error during inference.
    let rows = [];
    if ('error' in response) {
      if (!this.options.logInputs) log.silly('inputs', { html, visuals });
    } else {
      // Render table with results.
      const pagePred = response[0];
      for (const [labelKey, nodePreds] of Object.entries<NodePrediction[]>(
        pagePred
      )) {
        for (const nodePred of nodePreds) {
          rows.push({ labelKey, ...nodePred });
        }
      }
      rows.sort((x, y) => y.confidence - x.confidence);
    }

    // Take screenshot.
    const screenshot = (await page.screenshot({
      fullPage: true,
      encoding: 'base64',
    })) as string;

    res.write(views.logEntry('Done.'));
    res.write(views.logEnd());
    res.write(views.results(rows, screenshot));
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
}
