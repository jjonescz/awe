# JavaScript

This folder contains code for extracting visual attributes from SWDE dataset. To
get started, execute:

```bash
pnpm start -- --help
```

## Architecture

1. SWDE dataset is expected to be downloaded and available locally.
2. A page from the dataset is loaded into Chromium browser using Puppeteer.
3. The browser starts downloading other assets that are not available locally
   (SWDE dataset doesn't provide them) like images, CSS and JavaScript files.
   These requests are intercepted and replaced with links to WaybackMachine if
   necessary. Responses are stored offline so they don't need to be requested
   again later.
4. Visual attributes are computed for each element in the page and saved
   alongside each page to a JSON file which is later loaded by the Python
   machine learning code.

## Development

To debug with parameters, open JavaScript Debug Terminal in Visual Studio Code
and start your command with:

```bash
node -r ts-node/register/transpile-only index.ts
```

To run with type-checking, run `pnpm test`.

## Execution

Scraping was executed on a Windows computer using command:

```ps1
pnpm start -- -j=8 -e="C:\Program Files\Google\Chrome\Application\chrome.exe" -T=1000 -S -x
```

Re-scraping invalid pages (as determined by script `../validate.py`) was
performed via:

```ps1
pnpm start -- -j=8 -e="C:\Program Files\Google\Chrome\Application\chrome.exe" -T=1000 -S --files="..\data\swde\invalid_pages.txt"
```

Taking screenshots of 10 pages of each website of `auto` vertical:

```ps1
pnpm start -- -j=8 -e="C:\Program Files\Google\Chrome\Application\chrome.exe" -RSot -T=1000 -F 10 -g "auto/*/????.htm"
```

Extracting visuals from Apify dataset (note that some pages require enabled
JavaScript, so beware of that `-S` option):

```bash
pnpm start -- -d ../data/apify/alzaEn -g 'pages/localized_html_*.htm' -o -T=1000 -j=8 -SH
pnpm start -- -d ../data/apify/bestbuyEn -g 'pages/localized_html_*.htm' -o -T=1000 -j=8 -SH
pnpm start -- -d ../data/apify/conradEn -g 'pages/localized_html_*.htm' -o -T=1000 -j=8 -SH
pnpm start -- -d ../data/apify/ikeaEn -g 'pages/localized_html_*.htm' -o -T=1000 -j=8 -SH
pnpm start -- -d ../data/apify/notinoEn -g 'pages/localized_html_*.htm' -o -T=1000 -j=8
pnpm start -- -d ../data/apify/tescoEn -g 'pages/localized_html_*.htm' -o -T=1000 -j=8 -SH
```

And validating them:

```bash
(cd .. && python -m awe.data.validate --no-labels --visuals --max-errors=1 [<website_name>])
```

To save a list of invalid, add argument `--save-list=data/invalid_pages.txt -q`
and re-scrape using:

```bash
pnpm start -- -d ../ --files=../data/invalid_pages.txt -o -T=1000 -j=8 -SH
```

To blend JSON and HTML into XML:

```bash
pnpm start -- -d ../data/apify/notinoEn -g 'pages/localized_html_*.htm' -B
```
