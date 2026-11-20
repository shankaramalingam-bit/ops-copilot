# Ops Copilot app image. Build context is the repository root:
#   docker compose -f deploy/docker-compose.yml build
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 app
WORKDIR /app

# Dependencies first (cached layer). No .env, no data/cache, no tests, no notebooks.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY data/docs ./data/docs
COPY data/sim ./data/sim
COPY eval/latest.json eval/report.md ./eval/
COPY web ./web
RUN uv sync --frozen --no-dev \
    && mkdir -p data/cache \
    && chown -R app:app /app

USER app
EXPOSE 8000
# Configuration comes from the environment (docker-compose.yml); never from a baked-in .env.
CMD ["/app/.venv/bin/python", "-m", "ops_copilot"]
