# JavaScript

This folder contains code for extracting visual attributes from SWDE dataset.

## Architecture

### Phase 1

1. SWDE dataset is expected to be downloaded and available locally.
2. A page from the dataset is loaded into Chromium browser using Puppeteer.
3. The browser starts downloading other assets that are not available locally
   (SWDE dataset doesn't provide them) like images, CSS and JavaScript files.
   These requests are intercepted and replaced with links to WaybackMachine if
   necessary.
4. A snapshot of the page is saved, so following phases can be performed
   repeatedly, even offline.

### Phase 2

1. Snapshot of a page is loaded into Chromium browser using Puppeteer.
2. Visual attributes are computed and saved.
