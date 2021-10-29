import { HTTPRequest } from 'puppeteer-core';
import { SWDE_TIMESTAMP } from './constants';

const ARCHIVE_URL_REGEX = /^https:\/\/web.archive.org\/web\/(\d{14})id_\/(.*)$/;

/** Functionality related to WaybackMachine. */
export class Wayback {
  public getArchiveUrl(url: string) {
    // For URL scheme, see
    // https://en.wikipedia.org/wiki/Help:Using_the_Wayback_Machine#Specific_archive_copy.
    return `https://web.archive.org/web/${SWDE_TIMESTAMP}id_/${url}`;
  }

  public parseArchiveUrl(url: string) {
    const match = url.match(ARCHIVE_URL_REGEX);
    if (match === null) return null;
    const [_full, date, pageUrl] = match;
    return [date, pageUrl] as const;
  }

  public isArchiveRedirect(request: HTTPRequest) {
    const archive = this.parseArchiveUrl(request.url());
    if (archive == null) return null;
    const [_date, url] = archive;
    const chain = request.redirectChain();
    if (chain.length !== 1) return null;
    const prev = chain[0];
    if (prev.url() !== url) return null;
    return url;
  }
}
