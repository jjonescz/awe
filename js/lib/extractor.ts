import { writeFile } from 'fs/promises';
import { ElementHandle, JSONObject, Page } from 'puppeteer-core';
import winston from 'winston';
import { logger } from './logging';
import { PageInfo } from './page-info';
import { PageRecipe } from './page-recipe';

type TreeData = {
  /** Data for child element. */
  [key: `/${string}`]: NodeData;
};

/** A-RGB color. */
type Color = `#${string}${string}${string}${string}`;

/** Bounding box (x, y, width, height). */
export type BoundingBox = readonly [number, number, number, number];

/** Visual attributes for one DOM node. */
type ElementData = {
  box?: BoundingBox;
  /** Value of `id` attribute (for debugging purposes). */
  id?: string;
  /** Whether this text element is only white-space. */
  whiteSpace?: true;

  // The following are present only in XML mode.
  tagName?: string;
  args?: Record<string, string>;
  text?: string;
};

export type NodeData = TreeData & ElementData;

/**
 * Structure in which extracted visual attributes are stored for an element and
 * its descendants.
 */
export type DomData = TreeData & Pick<PageInfo, 'timestamp'>;

// Note that the following objects need to be serializable because they are
// passed to browser context during evaluation. Extending `JSONObject` ensures
// that.

/** Options for extraction. */
export interface ExtractorOptions extends JSONObject {
  readonly extractXml: boolean;
  readonly onlyTextFragments: boolean;
  readonly alsoHtmlTags: string[];
}

/** Convenience methods for creating {@link ExtractorOptions}. */
export class ExtractorOptions {
  public static readonly default: ExtractorOptions = {
    extractXml: false,
    onlyTextFragments: false,
    alsoHtmlTags: [],
  };

  public static create(partial: Partial<ExtractorOptions>) {
    return { ...ExtractorOptions.default, ...partial } as ExtractorOptions;
  }

  public static fromModelParams(params: Record<string, any>): ExtractorOptions {
    return ExtractorOptions.create({
      onlyTextFragments: !!params['classify_only_text_nodes'],
      alsoHtmlTags: params['classify_also_html_tags'],
    });
  }
}

/** Results of in-browser evaluation. */
export interface EvaluationResult extends JSONObject, ExtractionStats {
  readonly errors: EvaluationError[];
  readonly data: TreeData;
}

export interface EvaluationError extends JSONObject {
  readonly xpath: string;
  readonly error: string;
}

export interface ExtractionStats {
  evaluated: number;
  skipped: number;
}

/** Can extract visual attributes from a Puppeteer-controlled page. */
export class Extractor {
  public readonly data: DomData;
  public readonly options: ExtractorOptions;

  constructor(
    public readonly page: Page,
    public readonly recipe: PageRecipe,
    public readonly logger: winston.Logger,
    options: Partial<ExtractorOptions> = ExtractorOptions.default
  ) {
    this.data = {
      timestamp: recipe.page.timestamp!,
    };
    this.options = ExtractorOptions.create(options);
  }

  /** Extracts visual attributes for all DOM nodes in the {@link page}. */
  public async extract(): Promise<ExtractionStats> {
    // Obtain the `document` element.
    const root = (await this.page.$x('.'))[0];

    // Run extraction inside the browser.
    const result = await this.evaluate(root);

    // Log errors.
    if (result.errors.length !== 0) {
      this.logger.error('extraction failed', { errors: result.errors });
    }

    // Save data.
    for (const [key, value] of Object.entries(result.data)) {
      this.data[key as `/${string}`] = value;
    }

    // Return stats.
    const { evaluated, skipped } = result;
    return { evaluated, skipped };
  }

  /** Extracts visual attributes for all DOM nodes in the tree rooted at
   * {@link root}.
   */
  public evaluate(root: ElementHandle<Element>) {
    return root.evaluate(Extractor.evaluateClient, this.options);
  }

  /** This method runs inside the browser. */
  private static evaluateClient(root: Element, options: ExtractorOptions) {
    const result: EvaluationResult = {
      evaluated: 0,
      skipped: 0,
      errors: [],
      data: {},
    };

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

    const isElement = (node: Node): node is Element => {
      return node.nodeType === Node.ELEMENT_NODE;
    };

    const isTextFragment = (node: Node) => {
      return node.nodeType === Node.TEXT_NODE;
    };

    const getTagName = (node: Node) => node.nodeName.toLowerCase();

    const getTextBoundingRect = (node: Node) => {
      const range = document.createRange();
      range.selectNode(node);
      const rect = range.getBoundingClientRect();
      range.detach();
      return rect;
    };

    const rectToBox = (rect: DOMRect): BoundingBox | undefined => {
      if (rect.x === 0 && rect.y === 0 && rect.width === 0 && rect.height === 0)
        return undefined;
      return [rect.x, rect.y, rect.width, rect.height] as const;
    };

    /** Common code used by multiple branches of `extractOne` below. */
    const extractNonText = (node: Element) => {
      const tagName = getTagName(node);
      const box = rectToBox(node.getBoundingClientRect());

      if (options.extractXml) {
        return { tagName, box };
      }

      return {
        id: node.getAttribute('id')?.toString() ?? undefined,
        tagName,
        box,
      };
    };

    /** Picks some properties from element's computed style. */
    const pickStyle = (node: Element) => {
      const style = getComputedStyle(node);
      return {
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
    };

    /** Extracts visual attributes for one {@link node}. */
    const extractOne = (node: Node) => {
      // Extract text fragments separately. They don't have computed style.
      if (isTextFragment(node)) {
        const tagName = 'text()';
        const box = rectToBox(getTextBoundingRect(node));

        if (options.extractXml) {
          return {
            tagName,
            box,
            text: node.textContent ?? '',
          };
        }
        return {
          tagName,
          box,
          whiteSpace: /^\s*$/.test(node.textContent ?? '')
            ? (true as const)
            : undefined,
        };
      }

      // Ignore everything except elements and text fragments.
      if (!isElement(node)) return null;

      // Skip extraction if not needed.
      if (
        // If only text fragments are requested.
        options.onlyTextFragments &&
        // And this is not one of the other requested tag names.
        options.alsoHtmlTags.indexOf(getTagName(node)) < 0
      ) {
        // And it does not have a text fragment child.
        let hasTextFragmentChild = false;
        node.childNodes.forEach((n) => {
          if (isTextFragment(n)) hasTextFragmentChild = true;
        });
        if (!hasTextFragmentChild) {
          result.skipped++;
          return extractNonText(node);
        }
      }
      result.evaluated++;

      // Construct `ElementInfo`.
      const picked = pickStyle(node);
      if (options.extractXml) {
        const attrs: Record<string, string> = {};
        for (let i = 0; i < node.attributes.length; i++) {
          const attr = node.attributes.item(0)!;
          attrs[attr.name] = attr.value;
        }
        return { ...extractNonText(node), attrs, ...picked };
      }
      return {
        ...extractNonText(node),
        ...picked,
      };
    };

    /** Constructs {@link target}'s XPath. */
    const tryGetXPath = (target: Node) => {
      try {
        // Inspired by https://stackoverflow.com/a/30227178.
        const getPathTo = (node: Node): string => {
          let ix = 0;
          const parent = node.parentNode;
          const tagName = node.nodeName.toLowerCase();
          if (parent === null || !(parent instanceof Element))
            return `/${tagName}`;
          const siblings = parent.childNodes;
          for (let i = 0; i < siblings.length; i++) {
            const sibling = siblings[i];
            if (sibling === node)
              return `${getPathTo(parent)}/${tagName}[${ix + 1}]`;
            if (isElement(sibling) && sibling.tagName.toLowerCase() === tagName)
              ix++;
          }
          return `invalid(${tagName})`;
        };
        return getPathTo(target);
      } catch (e) {
        return `error(${(e as Error)?.message})`;
      }
    };

    /** Catches extraction errors. */
    const tryExtractOne = (node: Node) => {
      try {
        return extractOne(node);
      } catch (e) {
        result.errors.push({
          xpath: tryGetXPath(node),
          error: (e as Error)?.message,
        });
        return null;
      }
    };

    // Start a queue with the root's children.
    const queue: { node: ChildNode; parent: TreeData }[] = [];
    root.childNodes.forEach((node) =>
      queue.push({ node, parent: result.data })
    );

    while (queue.length !== 0) {
      const { node, parent } = queue.shift()!;

      // Extract data for an element.
      const result = tryExtractOne(node);
      if (result === null) continue;
      const { tagName, ...info } = result;

      // Append this element's data to parent `DomData`.
      const container: NodeData = { ...info };
      const key = `/${tagName}` as const;
      const indexedKey = (i: number) => `${key}[${i}]` as const;
      let finalKey = key;
      // If parent already contains an indexed child, find new available index.
      if (parent[indexedKey(1)] !== undefined) {
        for (let i = 2; ; i++) {
          if (parent[indexedKey(i)] === undefined) {
            finalKey = indexedKey(i);
            break;
          }
        }
      }
      // Otherwise, determine whether we have to add index `[1]`.
      else {
        for (let s = node.nextSibling; s !== null; s = s.nextSibling) {
          if (getTagName(s) === getTagName(node)) {
            finalKey = indexedKey(1);
            break;
          }
        }
      }

      // Note that we never change `parent[key]`, we only *append*, therefore
      // the order should match the original order. This is actually not
      // required (and the processing code should not count on that), but it
      // helps when looking at the JSON manually.
      parent[finalKey] = container;

      // Add children to the queue.
      node.childNodes.forEach((node) =>
        queue.push({ node, parent: container })
      );
    }

    return result;
  }

  public async save() {
    logger.verbose('data', { path: this.recipe.jsonPath });
    const json = JSON.stringify(this.data, null, 1);
    await writeFile(this.recipe.jsonPath, json, { encoding: 'utf-8' });
  }
}
