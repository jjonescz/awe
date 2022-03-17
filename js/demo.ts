import express from 'express';

const app = express();
const port = process.env.PORT || 3000;

app.get(['/', '/:name'], (req, res) => {
  const greeting = '<h1>Hello From Node</h1>';
  const name = req.params['name'];
  if (name) {
    res.send(`${greeting}<p>Welcome, ${name}.</p>`);
  } else {
    res.send(greeting);
  }
});

app.listen(port, () => console.log(`Listening on http://localhost:${port}/`));
