import express from 'express';
import { logger } from './lib/logging';
import { Scraper } from './lib/scraper';

logger.level = 'verbose';

(async () => {
  // Open browser (Puppeteer).
  const scraper = await Scraper.create({
    poolSize: 1,
    executablePath: 'google-chrome-stable',
    devtools: false,
    timeout: 1000,
    disableJavaScript: false,
  });

  // Create server.
  const app = express();
  const port = process.env.PORT || 3000;
  const log = logger.child({ server: port });

  app.get(['/', '/:name'], (req, res) => {
    const greeting = '<h1>Hello From Node</h1>';
    const name = req.params['name'];
    if (name) {
      res.send(`${greeting}<p>Welcome, ${name}.</p>`);
    } else {
      res.send(greeting);
    }
  });

  app.get('/run/:url', async (req, res) => {
    // Parse parameters.
    const url = req.params['url'];
    log.debug('run', { url: url });

    // TODO: Run through Puppeteer.
  });

  app.listen(port, async () => {
    console.log(`Listening on http://localhost:${port}/`);
  });
})();
