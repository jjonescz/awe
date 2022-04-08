import h from 'html-template-tag';
import { ModelInfo } from './model-info';
import { NodePrediction } from './python';

export function info(model: ModelInfo) {
  return h`
  <h1><a href="/">AWE</a></h1>
  <details>
  <summary>Model</summary>
  $${model.description !== undefined ? h`<p>${model.description}</p>` : ''}
  <dl>
    <dt>Vertical</dt>
    <dd>${model.vertical}</dd>
    <dt>Label keys</dt>
    <dd>$${model.labels.map((l) => h`<code>${l}</code>`).join(', ')}</dd>
    <dt>Trained on</dt>
    <dd>$${model.websites
      .map((w) => h`<a rel="external" href="${w}">${w}</a>`)
      .join('<br />')}</dd>
  </dl>
  </details>`;
}

export function form(model: ModelInfo, { url = '' } = {}) {
  return h`
  <details $${url === '' ? 'open' : ''}>
  <summary>Inputs</summary>
  <form method="get">
    <p>
      <label>
        URL<br />
        <input type="url" name="url" value="${url}" list="examples" />
        <datalist id="examples">
          $${(model.examples ?? [])
            .map((e) => h`<option value="${e}" />`)
            .join('')}
        </datalist>
      </label>
    </p>
    $${
      model.examples === undefined
        ? ''
        : h`
    <p>
      Examples
      <ul>
      $${model.examples
        .map((e) => h`<li><a href="/?url=${e}">${e}</a></li>`)
        .join('')}
      </ul>
    </p>
    `
    }
    <p><button type="submit">Submit</button></p>
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
  screenshot: string
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
    </tr>
    $${rows
      .map((r) => {
        const prob = (r.probability * 100).toFixed(2);
        const conf = r.confidence.toFixed(2);
        return h`
        <tr>
          <th>${r.labelKey}</th>
          <td>${r.text}</td>
          <td>${prob} % (${conf})</td>
        </tr>`;
      })
      .join('')}
  </table>
  <h2>Screenshot</h2>
  <img src="data:image/png;base64,${screenshot}" />
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
        ul > li {
          overflow-x: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          list-style-position: inside;
        }
      </style>
    </head>
    <body>`;
}

export function layoutEnd() {
  return h`
    <details open>
      <summary>About</summary>
      <p>
        Created by <a href="https://github.com/jjonescz" rel="external">Jan Joneš</a>
        for thesis <a href="https://is.cuni.cz/studium/dipl_st/index.php?id=&tid=&do=main&doo=detail&did=241832" rel="external">AI-based Structured Web Data Extraction</a>.
      </p>
      <p>
        Last updated 2022-04-08.
      </p>
    </details>
    </body>
  </html>
  `;
}
