import path from 'path';

export const DATA_FOLDER = path.resolve(__dirname, '../../data');
export const ARCHIVE_FOLDER = path.join(DATA_FOLDER, 'archive');
export const SWDE_FOLDER = path.join(DATA_FOLDER, 'swde/data');
export const WAYBACK_CLOSEST_FILE = path.join(
  DATA_FOLDER,
  'wayback-closest.json'
);
export const SWDE_TIMESTAMP = '20110601000000'; // half of year 2011
