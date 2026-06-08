FROM python:3.11-slim-bookworm

# Create non-root user (HF recommendation)
RUN useradd -m -u 1000 user

WORKDIR /code

# Install uv
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && apt-get remove -y curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock README.md ./

# Create venv and install locked dependencies
RUN uv sync --frozen --no-dev

# Copy app last
COPY app ./app

# Fix permissions
RUN chown -R user:user /code

USER 1000

EXPOSE 7860

CMD ["uv", "run", "python", "app/main.py"]