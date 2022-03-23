import { PythonShell, PythonShellError } from 'python-shell';
import { Logger } from 'winston';
import { DomData } from '../extractor';

export interface InferenceInput {
  url: string;
  html: string;
  visuals: DomData;
}

export interface NodePrediction {
  text: string;
  xpath: string;
  confidence: number;
}

export class Inference {
  public readonly shell: PythonShell;
  public loading: Promise<void> | null;
  private resolve: (value: void | PromiseLike<void>) => void;
  private reject: (reason?: any) => void;

  public constructor(private readonly log: Logger) {
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
    return JSON.parse(responseStr);
  }

  private async sendStr(message: string) {
    this.shell.send(message);
    return await new Promise<string>((resolve) =>
      this.shell.once('message', resolve)
    );
  }

  private shouldForward() {
    return this.loading !== null;
  }

  private onMessage = (data: string) => {
    if (this.shouldForward()) console.log(`PYTHON: ${data}`);
    this.log.silly('python stdout', { data });
    if (data === 'Inference started.') {
      this.log.verbose('opened Python');
      this.resolve();
      this.loading = null;
    }
  };

  private onStderr = (data: string) => {
    if (this.shouldForward()) console.error(`PYTERR: ${data}`);
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
