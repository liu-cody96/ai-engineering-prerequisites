FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install dependencies from the lockfile so the container matches the local env
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

COPY src ./src

# Run as a non-root user
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    mkdir -p /home/app/.streamlit && \
    chown -R app:app /app /home/app
ENV HOME=/home/app
USER app

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
