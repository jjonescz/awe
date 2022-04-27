FROM janjones/awe-gradient

COPY . ./

# Download latest release asset (pre-trained model).
ARG AWE_VERSION=latest
ARG GITHUB_API_TOKEN
RUN curl -s -H "Authorization: token $GITHUB_API_TOKEN" -s \
    https://api.github.com/repos/jjonescz/awe/releases/$AWE_VERSION \
    | jq -r '.assets[0].url' \
    | xargs -n 1 curl -sL -H "Authorization: token $GITHUB_API_TOKEN" \
    -H 'Accept: application/octet-stream' \
    | tar xvz

# Download GloVe embeddings.
RUN python -m awe.prepare

WORKDIR /storage/awe/src/js

# Install JavaScript packages and run type checking.
RUN pnpm install --frozen-lockfile && pnpm test

ARG GIT_COMMIT_HASH \
    GIT_COMMIT_TIMESTAMP
ENV PORT=3000 \
    GIT_COMMIT_HASH=${GIT_COMMIT_HASH} \
    GIT_COMMIT_TIMESTAMP=${GIT_COMMIT_TIMESTAMP}
EXPOSE $PORT
CMD pnpm run server
