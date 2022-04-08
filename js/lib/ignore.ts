import { ResponseForRequest } from 'puppeteer-core';
import { normalizeUrl } from './utils';

/** Patterns of URLs that will be ignored while scraping. */
export const ignorePatterns: RegExp[] = [
  // From `auto-automotive` and `auto-motortrend`.
  /^http:\/\/[a-z]+\.112\.2o7\.net\/b\/ss/,
];

/** Prefixes of URLs that will be ignored while scraping. */
export const ignoreStartingWith = [
  // From `auto-aol`.
  'http://a.vast.com/impressions',
  'http://tacoda.at.atwola.com/rtx/r.js',
  'http://aol.tt.omtrdc.net/m2/aol/mbox/',
  'http://img.vast.com/',
  // From `auto-autobytel`.
  'http://www.google-analytics.com/__utm.gif',
  'http://dp.specificclick.net/',
  'http://tags.bluekai.com/',
  'http://www.facebook.com/plugins/like.php',
  'http://ad.doubleclick.net/',
  'http://smp.specificmedia.com/smp/',
  // From `auto-automotive`.
  'http://secure-us.imrworldwide.com/cgi-bin/j',
  'http://pbid.pro-market.net/engine',
  'http://b.scorecardresearch.com/b',
  'http://pix04.revsci.net/',
  // From `auto-cars`.
  'http://fls.doubleclick.net/activityi',
  // From `auto-kbb`.
  'http://metrics.kbb.com/',
  'http://mxptint.net/2/1/0/',
  'http://by.optimost.com/counter/422/',
  'http://l.addthiscdn.com/live/',
  // From `auto-yahoo`.
  'http://autos.yahoo.com/darla/fc.php',
  // From `notinoEn`.
  'https://publicapi.databreakers.com/',
];

export function ignoreUrl(url: string) {
  url = normalizeUrl(url);
  return (
    ignoreStartingWith.some((p) => url.startsWith(p)) ||
    ignorePatterns.some((r) => r.test(url))
  );
}

function ignoreHeader(name: string) {
  // These headers can cause errors in Puppeteer, remove them.
  return name.startsWith('x-archive-orig-');
}

/** Removes problematic headers. */
export function cleanHeaders(request: Partial<ResponseForRequest>) {
  if (request.headers !== undefined) {
    for (const name in request.headers) {
      if (ignoreHeader(name)) {
        delete request.headers[name];
      }
    }
  }
}
