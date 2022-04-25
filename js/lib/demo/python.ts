import { PythonShell, PythonShellError } from 'python-shell';
import { Logger } from 'winston';
import { BoundingBox, DomData } from '../extractor';
import { DemoOptions } from './app';

export interface InferenceInput {
  url: string;
  html: string;
  visuals: DomData;
  /** Base64-encoded image. */
  screenshot: string;
}

export interface InferenceOutput {
  pages: { [labelKey: string]: NodePrediction[] }[];
  /** Base64-encoded image. */
  screenshot: string;
}

export interface NodePrediction {
  text: string | null;
  url: string | null;
  xpath: string;
  confidence: number;
  probability: number;
  box: BoundingBox | null;
}

type PromiseResolve<T> = (value: T | PromiseLike<T>) => void;

export class Inference {
  public readonly shell: PythonShell;
  private readonly queue: PromiseResolve<string>[] = [];
  public loading: Promise<void> | null;
  private resolve: PromiseResolve<void>;
  private reject: (reason?: any) => void;

  public constructor(
    private readonly options: DemoOptions,
    private readonly log: Logger
  ) {
    // Create a promise that will be fulfilled when Python inference is fully
    // loaded.
    this.resolve = () => {};
    this.reject = () => {};
    this.loading = new Promise<void>((resolve, reject) => {
      this.resolve = resolve;
      this.reject = reject;
    });

    // Start Python inference in a background shell.
    log.verbose('opening Python');
    this.shell = new PythonShell('awe.inference', {
      pythonOptions: ['-u', '-m'],
      cwd: '..',
    });

    // Register listeners.
    this.shell.on('message', this.onMessage);
    this.shell.on('stderr', this.onStderr);
    this.shell.on('close', this.onClose);
    this.shell.on('pythonError', this.onPythonError);
    this.shell.on('error', this.onError);
  }

  public async send(input: InferenceInput) {
    const responseStr = await this.sendStr(JSON.stringify(input));
    // Avoid logging this as debug, it contains a screenshot in Base64, making
    // it a very long string.
    this.log.silly('response', { json: responseStr });
    return JSON.parse(responseStr) as InferenceOutput;
  }

  private async sendStr(message: string) {
    this.shell.send(message);
    // Append new promise to the queue, it is resolved once a response from the
    // Python interpreter arrives. Thanks to this, correspondence of
    // request-response pairs is preserved even if requests are made in
    // parallel.
    return await new Promise<string>((resolve) => {
      this.queue.push(resolve);
    });
  }

  private shouldForward() {
    return this.loading !== null;
  }

  private onMessage = (data: string) => {
    if (this.shouldForward()) console.log(`PYTHON: ${data}`);
    this.log.silly('python stdout', { data });
    if (this.loading === null) {
      const resolve = this.queue.shift();
      if (resolve === undefined) {
        this.log.error('unmatched Python response', { data });
      } else {
        resolve(data);
      }
    } else if (data === 'Inference started.') {
      this.log.verbose('opened Python');
      this.resolve();
      this.loading = null;
    }
  };

  private onStderr = (data: string) => {
    if (this.shouldForward() || this.options.debug)
      console.error(`PYTERR: ${data}`);
    this.log.silly('python stderr', { data });
  };

  private onClose = () => {
    this.log.verbose('python closed');
    this.reject();
  };

  private onPythonError = (error: PythonShellError) => {
    this.log.error('python killed', { error });
  };

  private onError = (error: NodeJS.ErrnoException) => {
    this.log.error('python failure', { error });
    this.reject(error);
  };
}
