# JavaScript

This folder contains code for extracting visual attributes from datasets (SWDE
and Apify) and the demo (which can extract visual attributes from live pages and
call Python for inference).

First, install Node.js packages (this and following commands assume the current
working directory is `js`):

```bash
pnpm install
```

To get started with the extraction, execute:

```bash
pnpm start -- --help
```

To start the demo locally (see [`demo/README.md`](../demo/README.md) for a
containerized version):

1. Make sure there is a pre-trained model in `logs`.

   ```bash
   cd ..
   gh auth login
   gh release download v0.1 --pattern logs.tar.gz
   tar xvzf logs.tar.gz
   rm logs.tar.gz
   cd js
   ```

2. Start the demo. See `DemoOptions` in `app.ts` for more options (passed as
   environment variables).

   ```bash
   DEBUG=1 pnpm run server
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

Taking screenshots of 3 pages of each website of `auto` vertical:

```bash
pnpm start -- -g 'auto/*/????.htm' -oRH t=1 -T=10000 -F=3 -S
```

Extracting visuals from Apify dataset (note that some pages require enabled
JavaScript, so beware of that `-S` option):

```bash
pnpm start -- -d ../data/apify/alzaEn -g 'pages/localized_html_*.htm' -o -T=1000 -j=8 -SH
pnpm start -- -d ../data/apify/asosEn -g 'pages/localized_html_*.htm' -o -T=1000 -j=8 -SH
pnpm start -- -d ../data/apify/bestbuyEn -g 'pages/localized_html_*.htm' -o -T=1000 -j=8 -SH
pnpm start -- -d ../data/apify/bloomingdalesEn -g 'pages/localized_html_*.htm' -o -T=1000 -j=8 -SH
pnpm start -- -d ../data/apify/conradEn -g 'pages/localized_html_*.htm' -o -T=1000 -j=8 -SH
pnpm start -- -d ../data/apify/etsyEn -g 'pages/localized_html_*.htm' -o -T=1000 -j=8 -SH
pnpm start -- -d ../data/apify/ikeaEn -g 'pages/localized_html_*.htm' -o -T=1000 -j=8 -SH
pnpm start -- -d ../data/apify/notinoEn -g 'pages/localized_html_*.htm' -o -T=1000 -j=8 -Z
pnpm start -- -d ../data/apify/tescoEn -g 'pages/localized_html_*.htm' -o -T=1000 -j=8 -SH
```

And validating them:

```bash
(cd .. && python -m awe.data.validate --visuals --max-errors=1 [<website_name>])
```

To save a list of invalid, replace `--max-errors=1` with
`--save-list=data/invalid_pages.txt -q` and re-scrape using `-d ../
--files=../data/invalid_pages.txt` instead of `-d -g` arguments, e.g.:

```bash
pnpm start -- -d ../ --files=../data/invalid_pages.txt -o -T=1000 -j=8 -SH
```

To blend JSON and HTML into XML:

```bash
pnpm start -- -d ../data/apify/alzaEn -g 'pages/localized_html_*.htm' -B
```

To add new website to the above list (and determine which parameters are needed),
try extracting a few pages with screenshots:

```bash
pnpm start -- -d ../data/apify/alzaEn -g 'pages/localized_html_*.htm' -o -T=1000 -SH -t=1 -m=2
```

And validate them (including manually looking at the screenshots):

```bash
(cd .. && python -m awe.data.validate --visuals --skip-without-visuals --max-errors=1 alzaEn)
```

To take 3 screenshots of a website:

```bash
pnpm start -- -d ../data/apify/alzaEn -g 'pages/localized_html_*.htm' -oRH -t=1 -T=1000 -m=3 -S
```
