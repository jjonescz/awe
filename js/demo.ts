import express from 'express';
import puppeteer from 'puppeteer-core';
import { PythonShell } from 'python-shell';
import { Extractor } from './lib/extractor';
import { logger } from './lib/logging';
import { PageInfo } from './lib/page-info';
import { PageRecipe } from './lib/page-recipe';
import { ScrapeVersion } from './lib/scrape-version';

logger.level = process.env.DEBUG ? 'debug' : 'verbose';

(async () => {
  // Parse arguments.
  const port = process.env.PORT || 3000;
  const log = logger.child({ server: port });
  log.info('start', { logLevel: logger.level });

  // Create server.
  const app = express();

  // Open browser.
  log.verbose('opening Puppeteer');
  const browser = await puppeteer.launch({
    args: [
      // Allow running as root.
      '--no-sandbox',
    ],
    executablePath: 'google-chrome-stable',
  });
  log.verbose('opened Puppeteer');

  // Open Python inference.
  log.verbose('opening Python shell');
  const python = new PythonShell('awe.inference', {
    pythonOptions: ['-u', '-m'],
    cwd: '..',
  });

  // Wait for Python inference to start.
  log.verbose('waiting for Python');
  await new Promise<void>((resolve, reject) => {
    const messageListener = (data: string) => {
      console.log(`PYTHON: ${data}`);
      if (data === 'Inference started.') {
        python.off('message', messageListener);
        resolve();
      }
    };
    python.on('message', messageListener);
    python.on('stderr', (data) => {
      console.error(`PYTERR: ${data}`);
    });
    python.on('close', () => {
      log.verbose('python closed');
      reject();
    });
    python.on('pythonError', (error) => {
      log.error('python killed', { error });
    });
    python.on('error', (error) => {
      log.error('python failure', { error });
      reject();
    });
  });

  // Configure demo server routes.
  app.get(['/'], (req, res) => {
    res.send('<h1>Hello From Node</h1>');
  });

  app.get('/run', async (req, res) => {
    // Parse parameters.
    const url = req.query['url']?.toString() ?? '';
    const runLog = log.child({ url: url });
    runLog.debug('run');

    // Run through Puppeteer.
    const page = await browser.newPage();
    await page.goto(url, { waitUntil: 'networkidle0' });
    const html = await page.content();

    // Extract visuals.
    runLog.debug('extract visuals');
    const pageInfo = new PageInfo(url, url, html);
    const recipe = new PageRecipe(pageInfo, ScrapeVersion.Exact);
    const extractor = new Extractor(page, recipe, runLog);
    await extractor.extract();
    const visuals = extractor.data;

    // Pass HTML and visuals to Python.
    runLog.debug('inference');
    python.send(JSON.stringify({ url, html, visuals }));
    const responseStr = await new Promise<string>((resolve) =>
      python.once('message', resolve)
    );
    const response = JSON.parse(responseStr);
    runLog.debug('response', { response });

    // Take screenshot.
    const screenshot = await page.screenshot({
      fullPage: true,
      encoding: 'base64',
    });

    res.send(
      `<h1>${url}</h1>
      <code>${JSON.stringify(response)}</code>
      <img src="data:image/png;base64,${screenshot}" />`
    );
  });

  // Start demo server.
  log.verbose('starting demo server');
  const server = app.listen(port, async () => {
    console.log(`Listening on http://localhost:${port}/`);
  });

  // Close server when Python closes.
  python.on('close', () => {
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
})();
