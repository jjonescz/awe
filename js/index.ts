import { Command, flags } from '@oclif/command';
import glob from 'fast-glob';
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
      default: path.relative('.', path.join(SWDE_DIR, '**/????.htm')),
    }),
    screenshot: flags.boolean({
      char: 's',
      description: 'take screenshot of each page',
    }),
    maxNumber: flags.integer({
      char: 'm',
      description: 'maximum number of pages to process (the rest is skipped)',
    }),
    noProgress: flags.boolean({
      char: 'p',
      description: 'disable progress bar',
    }),
  };

  async run() {
    const { flags } = this.parse(Program);

    // Configure logger.
    logger.level = flags.verbose ? 'verbose' : flags.logLevel;
    logger.info('starting', { logFile, flags });

    // Open browser.
    const scraper = await Scraper.create();
    const controller = new Controller(scraper);

    // Find pages to process.
    const fullGlob = path.resolve('.', flags.globPattern);
    const allFiles = await glob(fullGlob);
    const files = allFiles.slice(0, flags.maxNumber);

    // Apply CLI flags.
    if (flags.offlineMode) scraper.allowLive = false;
    if (flags.forceRefresh) scraper.forceLive = true;
    controller.takeScreenshot = flags.screenshot;

    // Scrape pages.
    try {
      await controller.scrapeAll(files, {
        showProgressBar: !flags.noProgress,
      });
    } finally {
      await scraper.dispose();
    }
  }
}

Program.run().then(null, require('@oclif/errors/handle'));
