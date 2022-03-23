import { DemoApp } from './lib/demo/app';
import { logger } from './lib/logging';

logger.level = process.env.DEBUG ? 'debug' : 'verbose';
DemoApp.start(logger);
