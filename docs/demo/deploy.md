# Demo deployment

These instructions are for deploying [the demo](run.md)
to various cloud hosting providers.
Alternatively, its Docker image can be used on other hosting providers
such as [DigitalOcean](https://www.digitalocean.com/).

## Deploying to Gradient.run

1. Install [Gradient CLI](https://docs.paperspace.com/gradient/cli/).

   ```bash
   pip install -U gradient
   ```

2. Login:

   ```bash
   gradient apiKey <key>
   ```

3. Ensure Docker image `janjones/awe-demo` exists on
   [Docker Hub](https://hub.docker.com/)
   (see [demo running instructions](run.md)).

4. Deploy:

   ```bash
   gradient deployments create --name awe-demo --spec demo/deployment.yaml
   ```

To stop the deployment, number of replicas can be set to 0 from the UI.

## Deploying to Fly.io

1. Download and install
   [`flyctl` CLI](https://fly.io/docs/getting-started/installing-flyctl/).

2. Login:

   ```bash
   flyctl auth login
   ```

3. Build and deploy the app:

   ```bash
   git clean -fXd # optional, removes all Git-ignored files
   flyctl deploy --remote-only --build-arg GITHUB_API_TOKEN=<token>
   ```

Alternatively, run GitHub Action
[Fly.io app](../../.github/workflows/fly-deploy.yml).

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

Alternatively, run GitHub Action
[Heroku app](../../.github/workflows/heroku-deploy.yml)
