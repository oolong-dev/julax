ARG BASEIMAGE=python:3.12-slim-bookworm
FROM ${BASEIMAGE}

# Install system dependencies
# Combined apt-get commands for layer optimization and cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    google-perftools \
    ca-certificates \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-sdk \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

# Set working directory
WORKDIR /workspace/julax

# Enable bytecode compilation and copying for uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock README.md ./
COPY examples/01_mnist/pyproject.toml examples/01_mnist/
COPY examples/02_mini_transformer/pyproject.toml examples/02_mini_transformer/
COPY examples/03_Llama_3.2_1B/pyproject.toml examples/03_Llama_3.2_1B/

# Create a dummy source structure to satisfy build backend checks
# This allows installing dependencies without copying the full source code yet
RUN mkdir -p src/julax && touch src/julax/__init__.py

# Install dependencies only (no project source)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --all-packages --extra tpu

# Copy the rest of the application
COPY . .

# Install the project itself
# This step is fast as dependencies are already installed
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-packages --extra tpu

CMD ["sleep", "infinity"]