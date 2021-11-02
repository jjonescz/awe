import { Command, flags } from '@oclif/command';
import path from 'path';
import { SWDE_DIR } from './lib/constants';
import { Controller } from './lib/controller';
import { Scraper } from './lib/scraper';

class Program extends Command {
  static flags = {
    version: flags.version(),
    help: flags.help(),
  };

  async run() {
    // Open browser.
    const scraper = await Scraper.create();

    // Enable offline mode.
    // scraper.allowLive = false;

    // Force refresh.
    // scraper.forceLive = true;

    try {
      // Scrape a page (hard-coded path for now).
      const fullPath = path.join(SWDE_DIR, 'auto/auto-aol(2000)/0000.htm');
      const controller = new Controller(scraper);
      await controller.scrapeBoth(fullPath);
    } finally {
      await scraper.dispose();
    }
  }
}

Program.run().then(null, require('@oclif/errors/handle'));
