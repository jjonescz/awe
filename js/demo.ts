import { DemoApp, DemoOptions } from './lib/demo/app';
import { logger } from './lib/logging';

const options = new DemoOptions();
logger.level = options.debug ? 'debug' : 'verbose';
DemoApp.start(options, logger);
