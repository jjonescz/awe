/** Patterns of URLs that will be ignored while scraping. */
export const ignorePatterns = [
  // Analytics from `auto-aol` website.
  /&mboxTime=\d+&/,
  /^http:\/\/a\.vast\.com\/impressions/,
];

export function ignoreUrl(url: string) {
  return ignorePatterns.some((r) => r.test(url));
}
