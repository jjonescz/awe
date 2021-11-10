/** Patterns of URLs that will be ignored while scraping. */
export const ignorePatterns: RegExp[] = [];

/** Prefixes of URLs that will be ignored while scraping. */
export const ignoreStartingWith = [
  // From `auto-aol`.
  'http://a.vast.com/impressions',
  'http://tacoda.at.atwola.com/rtx/r.js',
  'http://aol.tt.omtrdc.net/m2/aol/mbox/',
];

export function ignoreUrl(url: string) {
  return (
    ignoreStartingWith.some((p) => url.startsWith(p)) ||
    ignorePatterns.some((r) => r.test(url))
  );
}
