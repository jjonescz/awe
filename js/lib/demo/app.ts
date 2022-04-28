import express, { Request, Response } from 'express';
import puppeteer from 'puppeteer-core';
import { Logger } from 'winston';
import { ExtractorOptions } from '../extractor';
import { logFile } from '../logging';
import { tryParseInt } from '../utils';
import { loadModel, Model } from './model-info';
import { PageInference } from './page-inference';
import { Inference } from './python';

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
  /** Base GitHub URL. */
  public readonly githubUrl: string;
  /** Git commit hash. */
  public readonly commitHash: string;
  /** Git commit timestamp. */
  public readonly commitTimestamp: Date | null;
  /** If `true`, examples from model `info.json` won't be used. */
  public readonly resetExamples: boolean;
  /** Additional example URLs to show. */
  public readonly moreExamples: string[];

  public constructor() {
    this.debug = !!process.env.DEBUG;
    this.port = tryParseInt(process.env.PORT, 3000);
    this.logInputs = !!process.env.LOG_INPUTS;
    this.mockInference = !!process.env.MOCK_INFERENCE;
    this.timeout = tryParseInt(process.env.TIMEOUT, 15);
    this.largeChunks = tryParseInt(process.env.LARGE_CHUNKS, 0);
    this.githubUrl =
      process.env.GITHUB_URL || 'https://github.com/jjonescz/awe';
    this.commitHash = process.env.GIT_COMMIT_HASH ?? '';
    // The timestamp should be in ISO 8601 format.
    const timestamp = process.env.GIT_COMMIT_TIMESTAMP ?? '';
    this.commitTimestamp = timestamp === '' ? null : new Date(timestamp);
    this.resetExamples = !!process.env.RESET_EXAMPLES;
    // The list of more examples should be a comma-separated list of URLs.
    const moreExamples = process.env.MORE_EXAMPLES ?? '';
    this.moreExamples = moreExamples === '' ? [] : moreExamples.split(',');
  }

  public get largeChunk() {
    if (this.largeChunks > 0) {
      return '<!--' + 'x'.repeat(this.largeChunks) + '-->';
    } else {
      return null;
    }
  }

  public get githubRepoShortName() {
    return new URL(this.githubUrl).pathname.substring(1);
  }

  public get githubInfo() {
    if (this.commitHash !== '') {
      const shortHash = this.commitHash.substring(0, 7);
      return {
        url: `${this.githubUrl}/tree/${this.commitHash}`,
        display: `${this.githubRepoShortName}@${shortHash}`,
      };
    }
    return { url: this.githubUrl, display: this.githubRepoShortName };
  }
}

export class DemoApp {
  public readonly extractorOptions: ExtractorOptions;
  public readonly python: Inference | null = null;
  public browser: Promise<puppeteer.Browser> | puppeteer.Browser;

  private constructor(
    public readonly options: DemoOptions,
    public readonly log: Logger,
    public readonly model: Model
  ) {
    this.extractorOptions = ExtractorOptions.fromModelParams(model.params);
    this.log.info('extractor options', { options: this.extractorOptions });

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

    // Load model.
    const model = await loadModel();
    log.verbose('loaded model', {
      versionDir: model.versionDir,
      info: model.info,
    });

    return new DemoApp(options, log, model);
  }

  public getExamples() {
    return (
      this.options.resetExamples ? [] : this.model.info.examples ?? []
    ).concat(this.options.moreExamples);
  }

  private mainPage = async (req: Request, res: Response) => {
    const inference = new PageInference(this, req, res);
    try {
      await inference.run();
    } finally {
      await inference.close();
    }
  };
}
