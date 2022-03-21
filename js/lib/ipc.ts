import { spawn } from 'child_process';
import {
  createReadStream,
  createWriteStream,
  openSync,
  ReadStream,
  WriteStream,
} from 'fs';
import winston from 'winston';

// IMPORTANT: Keep consistent with constants in `awe/inference.py`.
const INCOMING_PATH = 'awe_to_js';
const OUTGOING_PATH = 'js_to_awe';

/** Communicates via IPC with Python module `awe.inference`. */
export class Communicator {
  private constructor(
    public readonly log: winston.Logger,
    private readonly readStream: ReadStream,
    private readonly writeStream: WriteStream
  ) {}

  static async open(parentLog: winston.Logger) {
    const log = parentLog.child({ cls: 'ipc' });
    const incomingPipe = spawn('mkfifo', [INCOMING_PATH]);
    const [readStream, writeStream] = await new Promise<
      readonly [ReadStream, WriteStream]
    >((resolve) =>
      incomingPipe.on('exit', (status) => {
        log.debug('created incoming pipe', { status });
        const incomingFd = openSync('pipe_b', 'r+');
        const readStream = createReadStream('', { fd: incomingFd });
        const writeStream = createWriteStream(OUTGOING_PATH);
        log.debug('ready to write');
        resolve([readStream, writeStream] as const);
      })
    );
    return new Communicator(log, readStream, writeStream);
  }
}
