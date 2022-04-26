# Node.js app

This folder contains
- visual extractor
  (see [`docs/extractor.md`](../docs/extractor.md) for instructions) and
- inference demo
  (see [`docs/demo/run.md`](../docs/demo/run.md) for instructions).

Further documentation is available in code.

## Architecture

1. HTML dataset is expected to be downloaded and available locally.
2. A page from the dataset is loaded into Chromium browser using Puppeteer.
3. The browser starts downloading other assets that are not available locally
   like images, CSS and JavaScript files.
   These requests are intercepted and replaced with links to Wayback Machine
   if necessary
   Responses are stored offline so they don't need to be requested again later.
4. Visual attributes are computed for each element in the page and saved
   alongside each page to a JSON file which is later loaded by the Python
   machine learning code.

## Code overview

- 📄 `index.ts`: visual extractor CLI entrypoint.
- 📄 `demo.ts` demo server app entrypoint.
- 📂 `lib/`:
  - 📄 `page-scraper.ts`: controls headless browser.
  - 📄 `extractor.ts`: extracts a set of visual attributes.
  - 📄 `page-controller.ts`: high-level control of one page extraction
    (wraps `page-scraper` and uses the `extractor`).
  - 📄 `controller.ts`: can extract from several pages in parallel
    (wraps `page-controller`s).
  - 📄 `cache.ts`: offline asset caching.
