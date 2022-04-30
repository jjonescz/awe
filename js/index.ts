import { Command, flags } from '@oclif/command';
import { ExitError } from '@oclif/errors/lib/errors/exit';
import glob from 'fast-glob';
import { readFile, unlink } from 'fs/promises';
import path from 'path';
import { SWDE_DIR } from './lib/constants';
import { Controller } from './lib/controller';
import { logFile, logger } from './lib/logging';
import { ScrapeVersion } from './lib/scrape-version';
import { Scraper } from './lib/scraper';
import { escapeFilePath } from './lib/utils';

function* getLogLevelNames() {
  for (const name in logger.levels) yield name;
}

/** Entrypoint of the visual extractor CLI app. */
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
      description:
        'base directory for `--globPattern` or paths loaded from `--files`',
      default: path.relative('.', SWDE_DIR),
    }),
    globPattern: flags.string({
      char: 'g',
      description: 'files to process',
      default: '**/????.htm',
      dependsOn: ['baseDir'],
    }),
    screenshot: flags.integer({
      char: 't',
      description: 'take screenshot of every n-th page',
      default: 0,
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
    skipExisting: flags.boolean({
      char: 'x',
      description: 'skip already-scraped pages',
    }),
    skipSave: flags.boolean({
      char: 'H',
      description: 'skip HTML saving',
    }),
    skipExtraction: flags.boolean({
      char: 'R',
      description: 'skip feature extraction',
    }),
    rememberAborted: flags.boolean({
      char: 'a',
      description: 'avoid retrying requests aborted in previous runs',
    }),
    chromeExecutable: flags.string({
      char: 'e',
      description: 'Chrome-like browser that will be controlled by Puppeteer',
      default: 'google-chrome-stable',
    }),
    devtools: flags.boolean({
      char: 'D',
      description: 'Run Chrome in non-headless mode with DevTools open',
    }),
    keepOpen: flags.boolean({
      char: 'K',
      description: 'Keeps browser open (useful for debugging)',
    }),
    timeout: flags.integer({
      char: 'T',
      description: 'milliseconds before scraping of one page is aborted',
      default: 0,
    }),
    validateOnly: flags.boolean({
      char: 'V',
      description: 'only validate existing extraction outcomes',
    }),
    files: flags.string({
      description:
        'path to a text file (relative to working directory) with each line ' +
        'representing file to process (relative to `--baseDir`)',
      exclusive: ['glob'],
    }),
    disableJavaScript: flags.boolean({
      char: 'S',
      description: 'disable JavaScript execution on pages',
    }),
    fromEachFolder: flags.integer({
      char: 'F',
      description: 'take only this many pages from each folder',
    }),
    blendOnly: flags.boolean({
      char: 'B',
      description: 'blend existing JSON and HTML into XML',
    }),
    extractXml: flags.boolean({
      char: 'X',
      description: 'extract visuals together with HTML as XML',
    }),
    freeze: flags.boolean({
      char: 'Z',
      description:
        'freeze HTML before extracting visuals ' +
        '(reloads the page with JavaScript disabled)',
      exclusive: ['disableJavaScript'],
    }),
  };

  async run() {
    const { flags } = this.parse(Program);

    // Configure logger.
    logger.level = flags.verbose ? 'verbose' : flags.logLevel;
    logger.info('starting', { logFile, flags });

    // Find pages to process.
    let allFiles: string[];
    if (flags.files !== undefined) {
      const fullFiles = path.resolve('.', flags.files);
      logger.verbose('reading files', { path: fullFiles });
      const content = await readFile(fullFiles, { encoding: 'utf-8' });
      allFiles = content
        .split(/\r?\n/)
        .filter((v) => v.length !== 0)
        .map((v) => path.resolve('.', path.join(flags.baseDir, v)));
    } else {
      const fullPattern = path.join(flags.baseDir, flags.globPattern);
      const fullGlob = path.resolve('.', fullPattern).replaceAll('\\', '/');
      logger.verbose('executing glob', { pattern: fullGlob });
      allFiles = await glob(fullGlob);
    }
    logger.debug('found files', { count: allFiles.length });
    let files = allFiles
      .sort()
      .slice(
        flags.skip,
        flags.maxNumber === undefined ? undefined : flags.skip + flags.maxNumber
      );

    // Take only N pages from each folder.
    if (flags.fromEachFolder !== undefined) {
      const groups: Map<string, string[]> = new Map();
      if (flags.fromEachFolder > 0) {
        for (const file of files) {
          const directory = path.dirname(file);
          let group = groups.get(directory);
          if (group === undefined) {
            group = [file];
            groups.set(directory, group);
          } else if (group.length < flags.fromEachFolder) {
            group.push(file);
          }
        }
      }
      files = [...groups.values()].flat();
    }

    // Clean files if asked.
    if (flags.clean) {
      // Detect files for cleaning.
      const patterns = files.map((f) => {
        // Generated files have some suffix after dash and any file extension.
        return escapeFilePath(f).replace(/\.htm$/, '-*.*');
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
    const scraper = await Scraper.create({
      poolSize: flags.jobs,
      executablePath: flags.chromeExecutable,
      devtools: flags.devtools,
      timeout: flags.timeout,
      disableJavaScript: flags.disableJavaScript,
    });
    const controller = new Controller(scraper);

    // Apply CLI flags.
    if (flags.offlineMode) scraper.allowLive = false;
    if (flags.forceRefresh) scraper.forceLive = true;
    scraper.rememberAborted = flags.rememberAborted;
    controller.takeScreenshot = flags.screenshot;
    controller.skipExisting = flags.skipExisting;
    controller.skipSave = flags.skipSave;
    controller.skipExtraction = flags.skipExtraction;
    controller.validateOnly = flags.validateOnly;
    controller.blendOnly = flags.blendOnly;
    controller.extractXml = flags.extractXml;
    controller.freeze = flags.freeze;

    // Scrape pages.
    try {
      await controller.scrapeAll(files, {
        showProgressBar: !flags.noProgress,
        jobs: flags.jobs,
        versions: [
          ...(flags.latest ? [ScrapeVersion.Latest] : []),
          ...(flags.exact ? [ScrapeVersion.Exact] : []),
        ],
      });

      if (flags.keepOpen) {
        console.log(
          'Completed but waiting (due to argument `-K`). ' +
            'Press any key to exit...'
        );
        process.stdin.setRawMode(true);
        await new Promise<void>((resolve) =>
          process.stdin.once('data', () => {
            process.stdin.setRawMode(false);
            resolve();
          })
        );
      }
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
