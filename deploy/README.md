# Deploying Ops Copilot with Docker Compose

Two services: `app` (the FastAPI API plus the portal) and `ollama` (models, kept in the named volume `ollama_models`).

## Start

From the repository root:

```bash
docker compose --env-file .env -f deploy/docker-compose.yml up --build -d
# pull the models inside the ollama container (first run only; they persist in the volume)
docker compose -f deploy/docker-compose.yml exec ollama ollama pull nomic-embed-text
docker compose -f deploy/docker-compose.yml exec ollama ollama pull llama3.1:8b
```

Open http://localhost:8000. Stop with `docker compose -f deploy/docker-compose.yml down` (add `-v` to also delete the models).

The default provider is Ollama with `llama3.1:8b`. Compose reads `LLM_PROVIDER`, `OLLAMA_MODEL` and, only for
`LLM_PROVIDER=groq`, `GROQ_API_KEY` from the shell environment or from the repository-root `.env`, which you must pass explicitly with
`--env-file .env` (compose uses that file only to resolve `${...}` in the compose file). Compose's project directory is
`deploy/`, so without `--env-file` it looks for `deploy/.env` and silently falls back to the defaults (`ollama`).
If you have no `.env`, drop the `--env-file .env` flag. Only these three variables are passed to the
container, by explicit `environment:` entries; no other key from the shell or `.env` reaches it. Never commit `.env`;
the Docker build context excludes `**/.env*`, so it is never baked into the image.

Both services have health checks (`app`: `GET /api/docs` with `Host: localhost`; `ollama`: `ollama list`), and `app`
starts only once `ollama` is healthy.

## Secret warning

`docker compose config` and `docker inspect` print `GROQ_API_KEY` in clear text when it is set. Never paste their
output into chat, issues or logs.

## Why the port is bound to 127.0.0.1

Inside the container the app must listen on `0.0.0.0` so Docker can forward to it, which needs
`OPS_COPILOT_ALLOW_NONLOCAL=1` (the app is loopback-only otherwise). That is safe only because compose publishes
`"127.0.0.1:8000:8000"`: the portal is reachable from this machine and nothing else. `OPS_COPILOT_ALLOWED_HOSTS=localhost,127.0.0.1`
makes the app reject requests with any other `Host` header. The `ollama` service publishes no port at all.

## Warning: no authentication

The portal and API have **no authentication**. Do not change the port mapping to `8000:8000` or `0.0.0.0:8000:8000`,
and do not put the service behind a public proxy, until the owner has decided on and built auth.

## Image notes

Non-root user (`app`), dependencies from `uv.lock` (`--frozen --no-dev`). Images are pinned to explicit tags:
`python:3.12.14-slim`, `ghcr.io/astral-sh/uv:0.12.5`, `ollama/ollama:0.40.16` (no `:latest`). Only `src/`, `data/docs`,
`data/sim`, `web/` and the latest eval result are copied. The root `.dockerignore`, `deploy/.dockerignore` and
`deploy/Dockerfile.dockerignore` (BuildKit prefers the last for a Dockerfile in a subdirectory; keep all three identical)
exclude `**/.env*`, `.venv`, `data/cache`, `.git` and notebooks.
