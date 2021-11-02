import { Command, flags } from '@oclif/command';
import path from 'path';
import { SWDE_DIR } from './lib/constants';
import { Controller } from './lib/controller';
import { logFile, logger } from './lib/logging';
import { Scraper } from './lib/scraper';

function* getLogLevelNames() {
  for (const name in logger.levels) yield name;
}

class Program extends Command {
  static flags = {
    version: flags.version(),
    help: flags.help({ char: 'h' }),
    offlineMode: flags.boolean({
      char: 'o',
      description: 'disable online requests (use only cached)',
    }),
    forceRefresh: flags.boolean({
      char: 'f',
      description: 'force online requests (even if cached)',
    }),
    logLevel: flags.enum({
      char: 'l',
      options: [...getLogLevelNames()],
      default: 'info',
      description: 'console logging level',
    }),
    verbose: flags.boolean({
      char: 'v',
      description: 'equivalent of `--logLevel=verbose`',
      exclusive: ['logLevel'],
    }),
    globPattern: flags.string({
      char: 'g',
      description: 'files to process',
      default: '**/????.htm',
    }),
  };

  async run() {
    const { flags } = this.parse(Program);

    // Configure logger.
    logger.level = flags.verbose ? 'verbose' : flags.logLevel;
    logger.info('starting', { logFile, flags });

    // Open browser.
    const scraper = await Scraper.create();

    // Apply CLI flags.
    if (flags.offlineMode) scraper.allowLive = false;
    if (flags.forceRefresh) scraper.forceLive = true;

    try {
      // Scrape pages.
      const fullGlob = path.join(SWDE_DIR, flags.globPattern);
      const controller = new Controller(scraper);
      await controller.scrapeAll(fullGlob);
    } finally {
      await scraper.dispose();
    }
  }
}

Program.run().then(null, require('@oclif/errors/handle'));
