import puppeteer from 'puppeteer-core';
import path from 'path';

(async () => {
  const browser = await puppeteer.launch({
    args: [
      // Allow running as root.
      '--no-sandbox'
    ],
    executablePath: 'google-chrome-stable'
  });

  // Intercept requests.
  const page = await browser.newPage();
  page.setRequestInterception(true);
  page.on('request', request => {
    console.log('request: ', request.url());
    if (request.url().startsWith('file://'))
      request.continue();
    else
      request.abort();
  });

  // Open a page (hard-coded path for now).
  const fullPath = path.resolve('../data/swde/data/auto/auto-aol(2000)/0000.htm');
  await page.goto(`file://${fullPath}`);

  await browser.close();
})();
