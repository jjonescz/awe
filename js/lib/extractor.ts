import { writeFile } from 'fs/promises';
import { ElementHandle, Page } from 'puppeteer-core';
import { logger } from './logging';
import { SwdePage } from './swde-page';
import { addSuffix, replaceExtension } from './utils';

type TreeData = {
  /** Data for child element. */
  [key: `/${string}`]: NodeData;
};

/** Visual attributes for one DOM node. */
type ElementData = {
  /** Bounding box (x, y, width, height). */
  box?: readonly [number, number, number, number];
  /** Value of `id` attribute (for debugging purposes). */
  id?: string;
};

type NodeData = TreeData & ElementData;

type ElementInfo = ElementData & {
  tagName: string;
};

/**
 * Structure in which extracted visual attributes are stored for an element and
 * its descendants.
 */
type DomData = TreeData;

const CHILD_SELECTOR = '*';

/** Can extract visual attributes from a Puppeteer-controlled page. */
export class Extractor {
  public readonly data: DomData = {};

  constructor(public readonly page: Page, public readonly swdePage: SwdePage) {}

  /** Extracts visual attributes for all DOM nodes in the {@link page}. */
  public async extract() {
    // Start a queue with root elements.
    const rootElements = await this.page.$x(CHILD_SELECTOR);
    const queue = rootElements.map((e) => ({
      element: e,
      parent: this.data,
    }));
    while (queue.length !== 0) {
      const { element, parent } = queue.shift()!;

      // Extract data for an element.
      const { tagName, ...info } = await this.extractFor(element);

      // Append this element's data to parent `DomData`.
      const container: NodeData = { ...info };
      const key = `/${tagName}` as const;
      const indexedKey = (i: number) => `${key}[${i}]` as const;
      let finalKey = key;
      // If parent already contains this child, we have to add indices.
      if (parent[key] !== undefined) {
        parent[indexedKey(1)] = parent[key];
        delete parent[key];
      }
      // If parent already contains an indexed child, find new available index.
      if (parent[indexedKey(1)] !== undefined) {
        for (let i = 2; ; i++) {
          if (parent[indexedKey(i)] === undefined) {
            finalKey = indexedKey(i);
            break;
          }
        }
      }
      parent[finalKey] = container;

      // Add children to the queue.
      const children = await element.$x(CHILD_SELECTOR);
      queue.push(...children.map((e) => ({ element: e, parent: container })));
    }
  }

  /** Extracts visual attributes for one {@link element}. */
  public async extractFor(
    element: ElementHandle<Element>
  ): Promise<ElementInfo> {
    // Run code in browser to get element's attributes.
    const evaluated = await element.evaluate((e) => {
      // Note that we cannot reference outside functions easily, hence we define
      // them here.
      const check = <T>(value: T) => {
        if (value === undefined)
          throw new Error('Ambiguous attribute value undefined.');
        return value;
      };
      const except = <T>(value: T, defaultValue: T) => {
        if (value === defaultValue) return undefined;
        return check(value);
      };
      const unless = <T>(value: T, condition: boolean) => {
        if (condition) return undefined;
        return check(value);
      };

      // Pick some properties from element's computed style.
      const style = getComputedStyle(e);
      const picked = {
        fontFamily: style.fontFamily,
        fontSize: except(style.fontSize, '16px'),
        fontWeight: except(style.fontWeight, '400'),
        fontStyle: except(style.fontStyle, 'normal'),
        textAlign: except(style.textAlign, 'start'),
        textDecoration: unless(
          style.textDecoration,
          style.textDecorationStyle === 'none'
        ),
        color: style.color,
        backgroundColor: except(style.backgroundColor, 'rgba(0, 0, 0, 0)'),
        backgroundImage: except(style.backgroundImage, 'none'),
        border: unless(style.border, style.borderStyle === 'none'),
      };

      // Construct `ElementInfo`.
      return {
        id: e.getAttribute('id') ?? undefined,
        tagName: e.nodeName.toLowerCase(),
        ...picked,
      };
    });

    // Get other attributes that don't directly need browser context.
    const box = await element.boundingBox();

    // Combine all element attributes together.
    return {
      ...evaluated,
      box: box === null ? undefined : [box.x, box.y, box.width, box.height],
    };
  }

  public get filePath() {
    return replaceExtension(this.swdePage.fullPath, '.json');
  }

  public async save({ suffix = '' } = {}) {
    const fullPath = addSuffix(this.filePath, suffix);
    logger.verbose('data', { path: fullPath });
    const json = JSON.stringify(this.data, null, 1);
    await writeFile(fullPath, json, { encoding: 'utf-8' });
  }
}
