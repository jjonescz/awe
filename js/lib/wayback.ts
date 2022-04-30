import { HTTPRequest } from 'puppeteer-core';
import { WAYBACK_CLOSEST_FILE } from './constants';
import {
  getHttps,
  normalizeUrl,
  tryReadFile,
  urlsEqual,
  writeFileSafe,
} from './utils';

const ARCHIVE_URL_REGEX =
  /^https?:\/\/web.archive.org\/web\/(\d{14})([a-z]{2}_)?\/(.*)$/;

/** Functionality related to WaybackMachine. */
export class Wayback {
  /**
   * Map from URL to its closest timestamp as obtained by
   * {@link closestTimestamp}.
   */
  private responses: Record<string, string | null> = {};

  /** Variant to use for {@link getArchiveUrl}. */
  public variant = 'id_';

  /**
   * Constructs URL to Wayback Machine displaying {@link url} at
   * {@link timestamp}.
   */
  public getArchiveUrl(url: string, timestamp: string) {
    url = normalizeUrl(url);
    // For URL scheme, see
    // https://en.wikipedia.org/wiki/Help:Using_the_Wayback_Machine#Specific_archive_copy.
    return `https://web.archive.org/web/${timestamp}${this.variant}/${url}`;
  }

  /** Parses Wayback Machine {@link url} scheme. */
  public parseArchiveUrl(url: string) {
    url = normalizeUrl(url);
    const match = url.match(ARCHIVE_URL_REGEX);
    if (match === null) return null;
    const [_full, date, _modifier, pageUrl] = match;
    return [date, pageUrl] as const;
  }

  /**
   * Determines whether {@link request} corresponds to redirect of Wayback
   * Machine.
   */
  public isArchiveRedirect(request: HTTPRequest) {
    const archive = this.parseArchiveUrl(request.url());
    if (archive == null) return null;
    const [_date, url] = archive;
    const chain = request.redirectChain();
    if (chain.length !== 1) return null;
    const prev = chain[0];
    if (!urlsEqual(prev.url(), url)) return null;
    return url;
  }

  /** Loads {@link closestTimestamp} cache. */
  public async loadResponses() {
    const responsesJson = await tryReadFile(WAYBACK_CLOSEST_FILE, '{}');
    this.responses = { ...this.responses, ...JSON.parse(responsesJson) };
  }

  /** Saves {@link closestTimestamp} cache. */
  public saveResponses() {
    writeFileSafe(WAYBACK_CLOSEST_FILE, JSON.stringify(this.responses));
  }

  /**
   * Obtains closest timestamp for {@link url} either from local cache if
   * available, or WaybackMachine API.
   *
   * @see https://archive.org/help/wayback_api.php.
   */
  public async closestTimestamp(url: string) {
    url = normalizeUrl(url);

    // Try cache first.
    const cached = this.responses[url];
    if (cached !== undefined) {
      return cached;
    }

    // Ask WaybackMachine API for the latest available version.
    const response = await getHttps('https://archive.org/wayback/available/', {
      url,
    });

    // Get timestamp from the response.
    const date = this.parseWaybackResponse(response, url);

    // Save to cache.
    this.responses[url] = date;

    return date;
  }

  /** Parses timestamp out of WaybackMachine API response. */
  private parseWaybackResponse(response: string, url: string) {
    const json = JSON.parse(response);
    if (json['url'] !== url) {
      throw new Error(
        `URL in response (${json['url']}) doesn't match requested URL (${url}).`
      );
    }
    const snapshots = json['archived_snapshots'];
    if (!snapshots) {
      // This page is not archived.
      return null;
    }
    const snapshot = snapshots['closest'];
    if (snapshot['status'] !== '200') {
      throw new Error(
        `Non-OK status (${snapshot['status']}) returned (${url}).`
      );
    }
    if (snapshot['available'] !== true) {
      throw new Error(`Snapshot not available (${url}).`);
    }
    const archiveUrl = snapshot['url'];
    const parsedUrl = this.parseArchiveUrl(archiveUrl);
    if (parsedUrl === null) {
      throw new Error(`Cannot parse Wayback URL (${archiveUrl}).`);
    }
    const [date, pageUrl] = parsedUrl;
    if (urlsEqual(pageUrl, url)) {
      throw new Error(
        `Wayback URL (${archiveUrl}) inconsistent with requested URL (${url}).`
      );
    }
    return date;
  }
}
