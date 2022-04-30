import { nameOf } from './utils';

/** Simple numerical statistics of scraping. */
export class ScrapingStats {
  /** Map from status code to number of occurrences. */
  public readonly status: Record<number, number> = {};
  public undef = 0;
  public aborted = 0;
  public unhandled = 0;
  public offline = 0;
  public live = 0;
  public ignored = 0;
  public disabled = 0;
  public timeout = 0;

  public increment(statusCode: number) {
    this.status[statusCode] = (this.status[statusCode] ?? 0) + 1;
  }

  public *iterateStrings() {
    for (const key in this) {
      if (key !== nameOf<ScrapingStats>('status')) {
        const value = this[key] as unknown as number;
        if (value !== 0) {
          yield `${key}: ${value}`;
        }
      }
    }

    for (const [code, count] of Object.entries(this.status)) {
      if (count !== 0) {
        yield `${code}: ${count}`;
      }
    }
  }

  public toString() {
    return [...this.iterateStrings()].join(', ');
  }
}
