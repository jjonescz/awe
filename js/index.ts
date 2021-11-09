import { Command, flags } from '@oclif/command';
import { ExitError } from '@oclif/errors/lib/errors/exit';
import glob from 'fast-glob';
import { unlink } from 'fs/promises';
import path from 'path';
import { SWDE_DIR } from './lib/constants';
import { Controller } from './lib/controller';
import { logFile, logger } from './lib/logging';
import { ScrapeVersion } from './lib/page-controller';
import { Scraper } from './lib/scraper';
import { escapeFilePath, replaceExtension } from './lib/utils';

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
    baseDir: flags.string({
      char: 'd',
      description: 'base directory for `--globPattern`',
      default: path.relative('.', SWDE_DIR),
    }),
    globPattern: flags.string({
      char: 'g',
      description: 'files to process',
      default: '**/????.htm',
      dependsOn: ['baseDir'],
    }),
    screenshot: flags.boolean({
      char: 't',
      description: 'take screenshot of each page',
    }),
    skip: flags.integer({
      char: 's',
      description: 'how many pages to skip',
      default: 0,
    }),
    maxNumber: flags.integer({
      char: 'm',
      description: 'maximum number of pages to process (the rest is skipped)',
    }),
    noProgress: flags.boolean({
      char: 'p',
      description: 'disable progress bar',
    }),
    jobs: flags.integer({
      char: 'j',
      description: 'number of jobs to run in parallel',
      default: 1,
    }),
    continueOnError: flags.boolean({
      char: 'e',
      description: 'log errors but continue with other pages if possible',
    }),
    clean: flags.boolean({
      description: 'clean files from previous runs and exit',
    }),
    dryMode: flags.boolean({
      char: 'n',
      description: 'do not actually remove any files, just list them',
      dependsOn: ['clean'],
    }),
    latest: flags.boolean({
      char: 'L',
      description: 'scrape latest version from WaybackMachine',
    }),
    exact: flags.boolean({
      description: 'scrape exact version from the dataset',
      default: true,
    }),
  };

  async run() {
    const { flags } = this.parse(Program);

    // Configure logger.
    logger.level = flags.verbose ? 'verbose' : flags.logLevel;
    logger.info('starting', { logFile, flags });

    // Find pages to process.
    const fullPattern = path.join(flags.baseDir, flags.globPattern);
    const fullGlob = path.resolve('.', fullPattern);
    const allFiles = await glob(fullGlob);
    const files = allFiles
      .sort()
      .slice(
        flags.skip,
        flags.maxNumber === undefined ? undefined : flags.skip + flags.maxNumber
      );

    // Clean files if asked.
    if (flags.clean) {
      // Detect files for cleaning.
      const patterns = files.map((f) => {
        // Generated files have some suffix after dash and any file extension.
        return replaceExtension(escapeFilePath(f), '-*.*');
      });
      const toClean = await glob(patterns);

      // Delete files.
      logger.info('to clean', { numFiles: toClean.length });
      for (const file of toClean) {
        if (flags.dryMode) {
          logger.verbose('would clean', { file });
        } else {
          logger.verbose('cleaning', { file });
          await unlink(file);
        }
      }

      // Exit.
      return;
    }

    // Open browser.
    const scraper = await Scraper.create();
    const controller = new Controller(scraper);

    // Apply CLI flags.
    if (flags.offlineMode) scraper.allowLive = false;
    if (flags.forceRefresh) scraper.forceLive = true;
    controller.takeScreenshot = flags.screenshot;

    // Scrape pages.
    try {
      await controller.scrapeAll(files, {
        showProgressBar: !flags.noProgress,
        jobs: flags.jobs,
        continueOnError: flags.continueOnError,
        versions: [
          ...(flags.latest ? [ScrapeVersion.Latest] : []),
          ...(flags.exact ? [ScrapeVersion.Exact] : []),
        ],
      });
    } finally {
      await scraper.dispose();
    }
  }
}

(async () => {
  try {
    await Program.run();
  } catch (e) {
    if (e instanceof ExitError) {
      process.exit(e.oclif.exit);
    } else {
      throw e;
    }
  }
})();
