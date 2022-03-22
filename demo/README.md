# AWE demo

This folder contains sources for deploying demo.

## Running locally

```bash
git clean -fXd # optional, removes all Git-ignored files
docker build -t awe-demo --build-arg GITHUB_API_TOKEN=<token> -f demo/Dockerfile .
docker run --rm -it -p 3000:3000 awe-demo
```
