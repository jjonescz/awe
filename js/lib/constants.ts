import path from 'path';

export const DATA_DIR = path.resolve(__dirname, '../../data');
export const ARCHIVE_DIR = path.join(DATA_DIR, 'archive');
export const SWDE_DIR = path.join(DATA_DIR, 'swde/data');
export const WAYBACK_CLOSEST_FILE = path.join(DATA_DIR, 'wayback-closest.json');
export const SWDE_TIMESTAMP = '20110601000000'; // half of year 2011
