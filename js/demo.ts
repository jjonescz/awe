import express from 'express';
import puppeteer from 'puppeteer-core';
import { PythonShell } from 'python-shell';
import { Extractor } from './lib/extractor';
import { logFile, logger } from './lib/logging';
import { PageInfo } from './lib/page-info';
import { PageRecipe } from './lib/page-recipe';
import { ScrapeVersion } from './lib/scrape-version';
import h from 'html-template-tag';

logger.level = process.env.DEBUG ? 'debug' : 'verbose';

// Set this to log full HTML and visuals to `scraping_logs`.
const logInputs = !!process.env.LOG_INPUTS;

// Set this when developing server UI to avoid waiting for Python.
const mockInference = !!process.env.MOCK_INFERENCE;

interface NodePrediction {
  text: string;
  xpath: string;
  confidence: number;
}

(async () => {
  // Parse arguments.
  const port = process.env.PORT || 3000;
  const log = logger.child({ server: port });
  log.info('start', { logLevel: logger.level, logFile });

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

  let python: PythonShell;
  if (!mockInference) {
    // Open Python inference.
    log.verbose('opening Python shell');
    python = new PythonShell('awe.inference', {
      pythonOptions: ['-u', '-m'],
      cwd: '..',
    });

    // Wait for Python inference to start.
    log.verbose('waiting for Python');
    await new Promise<void>((resolve, reject) => {
      const messageListener = (data: string) => {
        console.log(`PYTHON: ${data}`);
        log.silly('python stdout', { data });
        if (data === 'Inference started.') {
          python.off('message', messageListener);
          resolve();
        }
      };
      python.on('message', messageListener);
      python.on('stderr', (data) => {
        console.error(`PYTERR: ${data}`);
        log.silly('python stderr', { data });
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
  }

  // Configure demo server routes.
  app.get(['/'], (req, res) => {
    res.send('<h1>AWE</h1>');
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

    // Log full inputs if desired.
    if (logInputs) {
      runLog.silly('inputs', { html, visuals });
    }

    // Pass HTML and visuals to Python.
    runLog.debug('inference');
    let response: any;
    if (!mockInference) {
      python.send(JSON.stringify({ url, html, visuals }));
      const responseStr = await new Promise<string>((resolve) =>
        python.once('message', resolve)
      );
      response = JSON.parse(responseStr);
    } else {
      response = [
        {
          engine: [],
          fuel_economy: [
            {
              confidence: 4.112531661987305,
              text: '22 / 29 mpg',
              xpath:
                '/html/body/div[1]/main/div[1]/section[2]/div[1]/div/div[5]/div/div[1]/div[1]/div[2]/div[2]/text()',
            },
            {
              confidence: 1.1698349714279175,
              text: '5/5',
              xpath:
                '/html/body/div[1]/main/div[1]/section[2]/div[1]/div/div[5]/div/div[1]/div[3]/div[2]/div[2]/text()',
            },
          ],
          model: [],
          price: [
            {
              confidence: 1.6609370708465576,
              text: '$25,377',
              xpath:
                '/html/body/div[1]/main/div[1]/section[2]/div[1]/div/div[3]/div/table/tbody/tr[1]/td[3]/text()',
            },
          ],
        },
      ];
    }
    runLog.debug('response', { response });

    // Log full inputs if they haven't been logged already and there was an
    // error during inference.
    let table: string;
    if ('error' in response) {
      table = '';
      if (!logInputs) runLog.silly('inputs', { html, visuals });
    } else {
      // Render table with results.
      const pagePred = response[0];
      const rows = [];
      for (const [labelKey, nodePreds] of Object.entries<NodePrediction[]>(
        pagePred
      )) {
        for (const nodePred of nodePreds) {
          rows.push({ labelKey, ...nodePred });
        }
      }
      rows.sort((x, y) => y.confidence - x.confidence);
      table = h`
      <table>
        <tr>
          <th>Key</th>
          <th>Value</th>
          <th>Confidence</th>
        </tr>
        $${rows
          .map(
            (r) => h`
            <tr>
              <td>${r.labelKey}</td>
              <td>${r.text}</td>
              <td>${r.confidence.toFixed(2)}</td>
            </tr>`
          )
          .join('')}
      </table>`;
    }

    // Take screenshot.
    const screenshot = (await page.screenshot({
      fullPage: true,
      encoding: 'base64',
    })) as string;

    res.send(
      h`<h1>AWE results</h1>
      <dl>
        <dt>URL</dt>
        <dd><code>${url}</code></dd>
      </dl>
      $${table}
      <img src="data:image/png;base64,${screenshot}" />`
    );
  });

  // Start demo server.
  log.verbose('starting demo server');
  const server = app.listen(port, async () => {
    console.log(`Listening on http://localhost:${port}/`);
  });

  // Close server when Python closes.
  if (!mockInference) {
    python!.on('close', () => {
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
})();
