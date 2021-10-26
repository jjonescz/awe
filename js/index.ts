import puppeteer from 'puppeteer-core';

(async () => {
  const browser = await puppeteer.launch({
    args: [
      // Allow running as root.
      '--no-sandbox'
    ],
    executablePath: 'google-chrome-stable'
  });
  await browser.close();
})();
