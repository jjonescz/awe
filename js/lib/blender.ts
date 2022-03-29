import * as cheerio from 'cheerio';
import { readFile } from 'fs/promises';
import { DomData } from './extractor';
import { PageRecipe } from './page-recipe';

/** Can blend JSON with visuals (generated by `Extractor`) and original HTML of
 * the page (as seen by the browser, simulated by `cheerio`) into one XML.
 *
 * Thanks to this, visuals should always match HTML DOM, unlike previously when
 * Python HTML parser would parse the DOM differently then Puppeteer visuals
 * extractor, leading to inconsistencies. This happened mainly when HTML was
 * broken.
 *
 * Furthermore, can also load labels, so Python does not have to evaluate CSS
 * selectors (as the used HTML parser Lexbor has some problems with them).
 */
export class Blender {
  private data: DomData = { timestamp: null };
  private dom: cheerio.CheerioAPI = cheerio.load('');

  public constructor(public readonly recipe: PageRecipe) {}

  public async loadJsonData() {
    const json = await readFile(this.recipe.jsonPath, { encoding: 'utf-8' });
    this.data = JSON.parse(json);
  }

  public loadHtmlDom() {
    this.dom = cheerio.load(this.recipe.page.html);
  }
}
