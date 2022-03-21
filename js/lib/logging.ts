import path from 'path';
import winston from 'winston';
import { SCRAPING_LOG_DIR } from './constants';

const timestamp = new Date().toISOString().replaceAll(':', '-');
export const logFile = path.join(SCRAPING_LOG_DIR, `${timestamp}.txt`);

export const logger = winston.createLogger({
  format: winston.format.combine(
    winston.format.timestamp({ alias: '_timestamp' }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({
      filename: logFile,
      level: 'silly',
    }),
    new winston.transports.Console({ format: winston.format.simple() }),
  ],
});
