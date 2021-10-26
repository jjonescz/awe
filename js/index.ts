import path from "path";
import { Scraper, SwdePage } from "./lib/scraper";

(async () => {
  // Open browser.
  const scraper = await Scraper.create();

  // Open a page (hard-coded path for now).
  const fullPath = path.resolve(
    "../data/swde/data/auto/auto-aol(2000)/0000.htm"
  );
  const page = await SwdePage.parse(fullPath);
  console.log("goto: ", fullPath);
  await scraper.go(page);

  // Take screenshot.
  const screenshotPath = path.format({
    ...path.parse(fullPath),
    base: undefined,
    ext: ".png",
  });
  console.log("screenshot: ", screenshotPath);
  await scraper.page.screenshot({ path: screenshotPath, fullPage: true });

  await scraper.browser.close();
})();
