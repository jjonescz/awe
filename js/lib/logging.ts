import path from 'path';
import winston from 'winston';
import { DATA_FOLDER } from './constants';

const logFile = path.join(DATA_FOLDER, 'scraping-logs.txt');

export const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: logFile }),
    new winston.transports.Console({ format: winston.format.simple() }),
  ],
});
