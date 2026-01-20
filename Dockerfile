FROM us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1

RUN apt-get update -y -qq \
    && apt-get install -y -qq google-perftools curl ca-certificates \
    && apt-get clean -y -qq \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

CMD ["sleep", "infinity"]