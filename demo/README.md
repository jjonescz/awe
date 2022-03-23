# AWE demo

This folder contains sources for deploying demo.

## Running locally

```bash
git clean -fXd # optional, removes all Git-ignored files
docker build -t awe-demo -f demo/Dockerfile --build-arg GITHUB_API_TOKEN=<token> .
docker run --rm -it -p 3000:3000 awe-demo
```

## Deploying to Fly.io

1. Download and install [`flyctl`
   tool](https://fly.io/docs/getting-started/installing-flyctl/).

2. Login:

   ```bash
   flyctl auth login
   ```

3. Build and deploy the app:

   ```bash
   git clean -fXd # optional, removes all Git-ignored files
   flyctl deploy --build-arg GITHUB_API_TOKEN=<token>
   ```

Alternatively, run GitHub Action `fly-deploy`.

## Deploying to Heroku

1. Install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) and
   [Docker](https://www.docker.com/products/docker-desktop/).

2. Login:

   ```bash
   heroku login
   ```

3. Build and deploy the app:

   ```bash
   heroku container:login
   git clean -fXd # optional, removes all Git-ignored files
   (cd demo && heroku container:push web --context-path .. --app awe-demo --arg GITHUB_API_TOKEN=<token>)
   heroku container:release web --app awe-demo
   ```

Alternatively, run GitHub Action `heroku-deploy`.
