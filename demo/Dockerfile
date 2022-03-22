FROM janjones/awe-gradient

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
COPY awe/ ./awe/
RUN python -m awe.prepare

# Install JavaScript packages.
COPY js/package.json js/pnpm-lock.yaml ./js/
RUN (cd js && pnpm install --frozen-lockfile)

# Run type checking.
COPY js/ ./js/
RUN (cd js && pnpm test)

WORKDIR /storage/awe/src/js
ENV PORT=3000
EXPOSE $PORT
CMD pnpm run server
