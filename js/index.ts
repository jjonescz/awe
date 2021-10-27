import path from 'path';
import { Scraper, SwdePage } from './lib/scraper';
import { SWDE_FOLDER } from './lib/constants';
import { replaceExtension } from './lib/utils';

(async () => {
  // Open browser.
  const scraper = await Scraper.create();

  try {
    // Open a page (hard-coded path for now).
    const fullPath = path.join(SWDE_FOLDER, 'auto/auto-aol(2000)/0000.htm');
    const page = await SwdePage.parse(fullPath);
    console.log('goto: ', fullPath);
    await scraper.go(page);

    // Take screenshot.
    const screenshotPath = replaceExtension(fullPath, '.png');
    console.log('screenshot:', screenshotPath);
    await scraper.page.screenshot({ path: screenshotPath, fullPage: true });
  } finally {
    await scraper.dispose();
  }
})();
