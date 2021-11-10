/** Patterns of URLs that will be ignored while scraping. */
export const ignorePatterns: RegExp[] = [];

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
  'http://aumoautomotivecom.112.2o7.net/b/ss/aumoautomotivecom/1/H.17/',
  'http://pix04.revsci.net/C07585/b3/0/3/1008211/',
];

export function ignoreUrl(url: string) {
  return (
    ignoreStartingWith.some((p) => url.startsWith(p)) ||
    ignorePatterns.some((r) => r.test(url))
  );
}
