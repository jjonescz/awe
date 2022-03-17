import express from 'express';
import puppeteer from 'puppeteer-core';
import { logger } from './lib/logging';

logger.level = 'verbose';

(async () => {
  // Open browser.
  logger.verbose('opening Puppeteer');
  const browser = await puppeteer.launch({
    args: [
      // Allow running as root.
      '--no-sandbox',
    ],
    executablePath: 'google-chrome-stable',
  });
  logger.verbose('opened Puppeteer');

  // Create server.
  const app = express();
  const port = process.env.PORT || 3000;
  const log = logger.child({ server: port });

  app.get(['/'], (req, res) => {
    res.send('<h1>Hello From Node</h1>');
  });

  app.get('/run', async (req, res) => {
    // Parse parameters.
    const url = req.query['url']?.toString() ?? '';
    log.debug('run', { url: url });

    // Run through Puppeteer.
    const page = await browser.newPage();
    await page.goto(url, { waitUntil: 'networkidle0' });

    // Take screenshot.
    const screenshot = await page.screenshot({
      fullPage: true,
      encoding: 'base64',
    });

    res.send(
      `<h1>${url}</h1>
      <img src="data:image/png;base64,${screenshot}" />`
    );
  });

  app.listen(port, async () => {
    console.log(`Listening on http://localhost:${port}/`);
  });
})();
