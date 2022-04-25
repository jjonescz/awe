import path from 'path';

export const DATA_DIR = path.resolve(__dirname, '../../data');
export const CACHE_DIR = path.join(DATA_DIR, 'cache');
export const SWDE_DIR = path.join(DATA_DIR, 'swde/data');
export const WAYBACK_CLOSEST_FILE = path.join(DATA_DIR, 'wayback-closest.json');
export const SWDE_TIMESTAMP = '20110601000000'; // half of year 2011
export const SCRAPING_LOG_DIR = path.join(DATA_DIR, 'scraping-logs');
export const LOG_DIR = path.resolve(__dirname, '../../logs');
export const TEMPORARY_DIR = path.join(DATA_DIR, 'tmp');
