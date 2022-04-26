# Demo

The demo is completely self-contained.
Both build and runtime happens in a Docker container.
Note that currently it needs around 1 GiB of RAM at runtime.

## Running as a Docker image

```bash
git clean -fXd # optional, removes all Git-ignored files
docker build -t janjones/awe-demo -f demo/Dockerfile --build-arg GITHUB_API_TOKEN=<token> .
docker run --rm -it -p 3000:3000 janjones/awe-demo
```

Alternatively, run GitHub Action
[Demo Docker Image](../../.github/workflows/demo-docker-image.yml)
and use image `janjones/awe-demo` from [Docker Hub](https://hub.docker.com/).

## Development

Demo can be started during [development](../dev/env.md).

1. Make sure there is a pre-trained model in `logs`.

   ```bash
   gh auth login
   gh release download v0.1 --pattern logs.tar.gz
   tar xvzf logs.tar.gz
   rm logs.tar.gz
   ```

2. Install packages and start the server.

   ```bash
   cd js
   pnpm install
   DEBUG=1 pnpm run server
   ```

## Options

Options are passed as environment variables.
See `DemoOptions` in `app.ts`.
