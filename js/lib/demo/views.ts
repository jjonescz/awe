import h from 'html-template-tag';
import { ModelInfo } from './model-info';

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
      Examples<br />
      $${model.examples
        .map((e) => h`<a href="/?url=${e}">${e}</a>`)
        .join('<br />')}
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
  rows: { labelKey: string; text: string; confidence: number }[],
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
      .map(
        (r) => h`
        <tr>
          <th>${r.labelKey}</th>
          <td>${r.text}</td>
          <td>${r.confidence.toFixed(2)}</td>
        </tr>`
      )
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
    </head>
    <body>`;
}

export function layoutEnd() {
  return h`
    </body>
  </html>
  `;
}
