import { Command, flags } from '@oclif/command';
import path from 'path';
import { SWDE_DIR } from './lib/constants';
import { Controller } from './lib/controller';
import { logFile, logger } from './lib/logging';
import { Scraper } from './lib/scraper';

class Program extends Command {
  static flags = {
    version: flags.version(),
    help: flags.help(),
    offlineMode: flags.boolean({
      char: 'o',
      description: 'disable online requests (use only cached)',
    }),
    forceRefresh: flags.boolean({
      char: 'f',
      description: 'force online requests (even if cached)',
    }),
  };

  async run() {
    const { flags } = this.parse(Program);
    logger.info('starting', { logFile, flags });

    // Open browser.
    const scraper = await Scraper.create();

    // Apply CLI flags.
    if (flags.offlineMode) scraper.allowLive = false;
    if (flags.forceRefresh) scraper.forceLive = true;

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
