import path from 'path';
import { Scraper } from './lib/scraper';
import { SWDE_FOLDER } from './lib/constants';
import { Controller } from './lib/controller';

(async () => {
  // Open browser.
  const scraper = await Scraper.create();

  // Enable offline mode.
  // scraper.allowLive = false;

  // Force refresh.
  // scraper.forceLive = true;

  try {
    // Scrape a page (hard-coded path for now).
    const fullPath = path.join(SWDE_FOLDER, 'auto/auto-aol(2000)/0000.htm');
    const controller = new Controller(scraper);
    await controller.scrapeBoth(fullPath);
  } finally {
    await scraper.dispose();
  }
})();
