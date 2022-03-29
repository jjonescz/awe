import { writeFile } from 'fs/promises';
import { ElementHandle, Page } from 'puppeteer-core';
import winston from 'winston';
import { logger } from './logging';
import { PageRecipe } from './page-recipe';

type TreeData = {
  /** Data for child element. */
  [key: `/${string}`]: NodeData;
};

/** A-RGB color. */
type Color = `#${string}${string}${string}${string}`;

/** Visual attributes for one DOM node. */
type ElementData = {
  /** Bounding box (x, y, width, height). */
  box?: readonly [number, number, number, number];
  /** Value of `id` attribute (for debugging purposes). */
  id?: string;
  /** Whether this text element is only white-space. */
  whiteSpace?: true;
};

type NodeData = TreeData & ElementData;

type ElementInfo = ElementData & {
  tagName: string;
};

/**
 * Structure in which extracted visual attributes are stored for an element and
 * its descendants.
 */
export type DomData = TreeData & {
  /** From {@link SwdePage.timestamp}. */
  timestamp: string;
};

const CHILD_SELECTOR = '* | text()';

async function tryGetXPath(element: ElementHandle<Element>) {
  try {
    return await element.evaluate((e) => {
      // Inspired by https://stackoverflow.com/a/30227178.
      const getPathTo = (element: Element): string => {
        let ix = 0;
        const parent = element.parentNode;
        const tagName = element.nodeName.toLowerCase();
        if (parent === null || !(parent instanceof Element))
          return `/${tagName}`;
        const siblings = parent.childNodes;
        for (let i = 0; i < siblings.length; i++) {
          const sibling = siblings[i];
          if (sibling === element)
            return `${getPathTo(parent)}/${tagName}[${ix + 1}]`;
          if ((sibling as Element)?.tagName === element.tagName) ix++;
        }
        return `invalid(${tagName})`;
      };
      return getPathTo(e);
    });
  } catch (e) {
    return `error(${(e as Error)?.message})`;
  }
}

/** Can extract visual attributes from a Puppeteer-controlled page. */
export class Extractor {
  public readonly data: DomData;

  constructor(
    public readonly page: Page,
    public readonly recipe: PageRecipe,
    public readonly logger: winston.Logger
  ) {
    this.data = {
      timestamp: recipe.page.timestamp!,
    };
  }

  /** Extracts visual attributes for all DOM nodes in the {@link page}. */
  public async extract() {
    // Start a queue with root elements.
    const rootElements = await this.page.$x(CHILD_SELECTOR);
    const queue = rootElements.map((e) => ({
      element: e,
      parent: <TreeData>this.data,
    }));
    while (queue.length !== 0) {
      const { element, parent } = queue.shift()!;

      // Extract data for an element.
      const result = await this.tryExtractFor(element);
      if (result === null) continue;
      const { tagName, ...info } = result;

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

  public async tryExtractFor(element: ElementHandle<Element>) {
    try {
      return await this.extractFor(element);
    } catch (e) {
      this.logger.error('extract failed', {
        error: (e as Error)?.message,
        element: await tryGetXPath(element),
      });
      return null;
    }
  }

  /** Extracts visual attributes for one {@link element}. */
  public async extractFor(
    element: ElementHandle<Element>
  ): Promise<ElementInfo> {
    // Run code in browser to get element's attributes.
    const evaluated = await this.evaluate(element);

    // Get other attributes that don't directly need browser context.
    const box = await element.boundingBox();

    // Combine all element attributes together.
    return {
      ...evaluated,
      box: box === null ? undefined : [box.x, box.y, box.width, box.height],
    };
  }

  /** Runs code inside browser for an {@link element}. */
  public evaluate(element: ElementHandle<Element>) {
    return element.evaluate((e) => {
      // Ignore text fragments, they don't have computed style.
      if (e.nodeName === '#text') {
        return {
          tagName: 'text()',
          whiteSpace: /^\s*$/.test(e.textContent ?? '')
            ? (true as const)
            : undefined,
        };
      }

      // Note that we cannot reference outside functions easily, hence we define
      // them here.

      /** Omits {@link value} if {@link condition} is met. */
      const unless = <T>(value: T, condition: boolean) => {
        if (condition) return undefined;
        if (value === undefined)
          throw new Error('Ambiguous attribute value undefined.');
        return value;
      };

      /** Omits {@link value} if it's equal to {@link defaultValue}. */
      const except = <T>(value: T, defaultValue: T) => {
        return unless(value, value === defaultValue);
      };

      /** Parses CSS pixels. */
      const pixels = (value?: string) => {
        if (value === undefined) return undefined;

        const match = value.match(/^(.+)(px)?$/);
        if (match === null) throw new Error(`Cannot parse pixels: '${value}'`);
        return parseInt(match[1]);
      };

      const toHex = (value: number) => {
        return value.toString(16).padStart(2, '0') as `${string}`;
      };

      /** Parses CSS color. */
      const color = (value: string) => {
        // Remove whitespace.
        value = value.replaceAll(/\s+/g, '');

        // Parse `rgb(r,g,b)` and `rgba(r,g,b,a)` patterns.
        const match = value.match(/^rgba?\((\d+),(\d+),(\d+)(,([0-9.]+))?\)$/);
        if (match === null) throw new Error(`Cannot parse color: '${value}'`);
        const [_full, r, g, b, _last, a] = match;

        // Stringify color to common hex format.
        const rh = toHex(parseInt(r));
        const gh = toHex(parseInt(g));
        const bh = toHex(parseInt(b));
        const ah = toHex(Math.floor(255 * parseFloat(a ?? '1')));
        return `#${rh}${gh}${bh}${ah}` as const;
      };

      const isTransparent = (value: string) => {
        return isTransparentHex(color(value));
      };

      const isTransparentHex = (value: Color) => {
        return value.endsWith('00');
      };

      /** Parses CSS color, but returns `undefined` if it's transparent. */
      const visibleColor = (value: string) => {
        const parsed = color(value);
        if (isTransparentHex(parsed)) return undefined;
        return parsed;
      };

      /** Parses one side of a border. */
      const borderSide = (
        style: CSSStyleDeclaration,
        prefix: string,
        name = prefix
      ) => {
        if (
          style.getPropertyValue(`${prefix}-width`) === '0px' ||
          style.getPropertyValue(`${prefix}-style`) === 'none' ||
          isTransparent(style.getPropertyValue(`${prefix}-color`))
        ) {
          // Ignore invisible borders.
          return {};
        }
        return {
          [name]: style.getPropertyValue(prefix),
        };
      };

      /** Parses border. */
      const border = (style: CSSStyleDeclaration, prefix = 'border') => {
        // If the border is same on each side, it will be in `border` property.
        if (style.getPropertyValue(prefix) !== '') {
          return borderSide(style, prefix);
        }

        // Otherwise, `border` will be empty string and we must process each
        // side separately.
        return {
          ...borderSide(style, `${prefix}-left`, `${prefix}Left`),
          ...borderSide(style, `${prefix}-top`, `${prefix}Top`),
          ...borderSide(style, `${prefix}-right`, `${prefix}Right`),
          ...borderSide(style, `${prefix}-bottom`, `${prefix}Bottom`),
        };
      };

      // Pick some properties from element's computed style.
      const style = getComputedStyle(e);
      const picked = {
        fontFamily: except(style.fontFamily, '"Times New Roman"'),
        fontSize: except(pixels(style.fontSize), 16),
        fontWeight: except(style.fontWeight, '400'),
        fontStyle: except(style.fontStyle, 'normal'),
        textAlign: except(style.textAlign, 'start'),
        textDecoration: unless(
          style.textDecoration,
          style.textDecorationLine === 'none'
        ),
        color: except(color(style.color), '#000000ff'),
        backgroundColor: visibleColor(style.backgroundColor),
        backgroundImage: except(style.backgroundImage, 'none'),
        ...border(style),
        boxShadow: except(style.boxShadow, 'none'),
        cursor: except(style.cursor, 'auto'),
        letterSpacing: pixels(except(style.letterSpacing, 'normal')),
        lineHeight: pixels(except(style.lineHeight, 'normal')),
        opacity: except(style.opacity, '1'),
        ...borderSide(style, 'outline'),
        overflow: except(style.overflow, 'visible'),
        pointerEvents: except(style.pointerEvents, 'auto'),
        textShadow: except(style.textShadow, 'none'),
        textOverflow: except(style.textOverflow, 'clip'),
        textTransform: except(style.textTransform, 'none'),
        zIndex: except(style.zIndex, 'auto'),
      };

      // Construct `ElementInfo`.
      return {
        id: e.getAttribute('id') ?? undefined,
        tagName: e.nodeName.toLowerCase(),
        ...picked,
      };
    });
  }

  public async save() {
    logger.verbose('data', { path: this.recipe.jsonPath });
    const json = JSON.stringify(this.data, null, 1);
    await writeFile(this.recipe.jsonPath, json, { encoding: 'utf-8' });
  }
}
