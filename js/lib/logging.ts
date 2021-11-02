import path from 'path';
import winston from 'winston';
import { SCRAPING_LOG_DIR } from './constants';

const timestamp = new Date().toISOString();
export const logFile = path.join(SCRAPING_LOG_DIR, `${timestamp}.txt`);

export const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({
      filename: logFile,
      level: 'debug',
    }),
    new winston.transports.Console({ format: winston.format.simple() }),
  ],
});
