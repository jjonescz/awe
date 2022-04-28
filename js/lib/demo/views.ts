import h from 'html-template-tag';
import { ExtractionStats } from '../extractor';
import { Wayback } from '../wayback';
import { DemoOptions } from './app';
import { ModelInfo } from './model-info';
import { PageInference } from './page-inference';
import { NodePrediction } from './python';

export function info(model: ModelInfo, options: DemoOptions) {
  return h`
  <h1><a href="/">AWE</a></h1>
  <p>
    AI-based web extractor (more details on
    <a href="${options.githubUrl}#readme" rel="external">GitHub</a>).
  </p>
  <details>
  <summary>Model training details</summary>
  $${model.description !== undefined ? h`<p>${model.description}</p>` : ''}
  <dl>
    <dt>Vertical <small>(which type of websites is the model tailored for)</small></dt>
    <dd>${model.vertical}</dd>
    <dt>Label keys <small>(what the model can extract)</small></dt>
    <dd>$${model.labels.map((l) => h`<code>${l}</code>`).join(', ')}</dd>
    <dt>Trained on</dt>
    <dd>$${model.websites
      .map((w) => h`<a rel="external" href="${w}">${w}</a>`)
      .join('<br />')}</dd>
  </dl>
  </details>`;
}

/** Encodes URL parameter. */
function withUrl(url: string) {
  const params = new URLSearchParams();
  params.set('url', url);
  return `/?${params.toString()}`;
}

export function form(page: PageInference) {
  // Gather Wayback Machine URLs.
  const examples = page.app.getExamples().map((e) => ({
    original: e,
    archived: <string | null>null,
  }));
  const model = page.app.model.info;
  if (examples.length !== 0 && model.timestamp !== undefined) {
    const wayback = new Wayback();
    wayback.variant = 'if_';
    for (const e of examples)
      e.archived = wayback.getArchiveUrl(e.original, model.timestamp);
  }

  return h`
  <details $${page.url === '' ? 'open' : ''}>
  <summary>Try extracting data from any page</summary>
  <p>
    💡 The page should display <em>${model.vertical}</em> details.
    See the section above for details about the extractor's capabilities.
  </p>
  <form method="get">
    <p>
      <label>
        URL<br />
        <input type="url" name="url" value="${page.url}" list="examples"
               placeholder="https://example.com/" />
        <datalist id="examples">
          $${examples.map((e) => h`<option value="${e.original}" />`).join('')}
        </datalist>
      </label>
    </p>
    <p>
      <label>
        Timeout (seconds)<br />
        <input type="number" name="timeout"
               value="${page.timeout.toString()}"
               onchange="updateTimeout()" />
      </label>
    </p>
    <p><button type="submit">Submit</button></p>
    $${
      examples === undefined
        ? ''
        : h`
    <p>
      Or pick an example:
      <ul>
      $${examples
        .map(
          (e) =>
            h`<li><span>
            <a href="${withUrl(e.original)}">${e.original}</a>
            $${
              e.archived === null
                ? ''
                : h`<small>
                  (<a href="${withUrl(e.archived)}">archived</a>)
                </small>`
            }
            </span></li>`
        )
        .join('')}
      </ul>
    </p>
    `
    }
  </form>
  </details>
  `;
}

export function logStart() {
  return h`
  <details open>
  <summary>Log</summary>
  <table>
    <tr>
      <th>Time</th>
      <th>Message</th>
    </tr>`;
}

export function logEntry(message: string) {
  return h`
  <tr>
    <td>${new Date().toISOString()}</td>
    <td>${message}</td>
  </tr>`;
}

export function logEnd() {
  return h`</table></details>`;
}

export function results(
  rows: ({
    labelKey: string;
  } & NodePrediction)[],
  screenshot: string,
  stats: ExtractionStats
) {
  return h`
  <details open>
  <summary>Results</summary>
  <h2>Labels</h2>
  <table>
    <tr>
      <th></th>
      <th>Value</th>
      <th>Confidence</th>
      <th>Box</th>
    </tr>
    $${rows
      .map((r) => {
        return h`
        <tr>
          <th>${r.labelKey}</th>
          <td>
            $${
              r.url === null
                ? r.text === null
                  ? h`<em>empty</em>`
                  : h`${r.text}`
                : h`<a href="${r.url}">$${
                    r.text === null ? h`<em>picture</em>` : h`${r.text}`
                  }</a>`
            }
          </td>
          <td>${(r.probability * 100).toFixed(2)}%</td>
          <td>${r.box?.join(', ') ?? ''}</td>
        </tr>`;
      })
      .join('')}
  </table>
  <h2>Screenshot</h2>
  <img src="data:image/png;base64,${screenshot}" />
  <h2>Stats</h2>
  <table>
    <tr>
      <th>all nodes</th>
      <td>${(stats.evaluated + stats.skipped).toLocaleString()}</td>
    </tr>
    <tr>
      <th>candidate nodes</th>
      <td>${stats.evaluated.toLocaleString()}</td>
    </tr>
  </table>
  </details>`;
}

export function layoutStart() {
  return h`
  <!DOCTYPE html>
  <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <meta http-equiv="X-UA-Compatible" content="ie=edge">
      <title>AWE</title>
      <link
        rel="stylesheet"
        type="text/css"
        href="https://cdn.jsdelivr.net/gh/alvaromontoro/almond.css@8698060/dist/almond.min.css"
      />
      <style>
        ul {
          padding-left: 1rem;
        }
        ${
          ''
          /* Make links in the example section shortened by ellipsis
           * if they are too long. */
        }
        ul > li > span {
          display: flex;
        }
        ul > li > span > a {
          overflow-x: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          margin-right: 0.5rem;
        }
      </style>
      <script>
        ${'' /* Updates timeout of examples upon form input change. */}
        function updateTimeout() {
          const timeout = document.querySelector('form input[type=number]').value;
          document.querySelectorAll('form > ul > li > span a').forEach(a => {
            const url = new URL(a.href);
            url.searchParams.set('timeout', 1);
            a.href = url.toString();
          });
        }
      </script>
    </head>
    <body>`;
}

function toYyyyMmDd(date: Date) {
  return date.toISOString().split('T')[0];
}

export function layoutEnd(options: DemoOptions) {
  const { url, display } = options.githubInfo;

  return h`
    <details open>
      <summary>About</summary>
      <p>
        Copyright &copy; 2022
        <a href="https://github.com/jjonescz" rel="external">Jan Joneš</a>.
      </p>
        Built from <a href="${url}" rel="external">${display}</a>.
      </p>
      $${
        options.commitTimestamp === null
          ? ''
          : h`
      <p>
        Last updated on ${toYyyyMmDd(options.commitTimestamp)}.
      </p>
      `
      }
    </details>
    </body>
  </html>
  `;
}
