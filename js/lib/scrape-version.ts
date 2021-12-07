import { SwdeHandling } from './page-scraper';

export const enum ScrapeVersion {
  /** Scrapes version that preserves HTML from the SWDE dataset. */
  Exact,
  /**
   * Scrapes latest available version from the WaybackMachine. This doesn't
   * consider HTML from the SWDE dataset and therefore gold labels might be
   * inconsistent.
   */
  Latest,
}

export function scrapeVersionToSwdeHandling(version: ScrapeVersion) {
  switch (version) {
    case ScrapeVersion.Exact:
      return SwdeHandling.Offline;
    case ScrapeVersion.Latest:
      return SwdeHandling.Wayback;
  }
}

export function scrapeVersionToString(version: ScrapeVersion) {
  switch (version) {
    case ScrapeVersion.Exact:
      return 'exact';
    case ScrapeVersion.Latest:
      return 'latest';
  }
}
