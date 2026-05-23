FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml ./
COPY hermes_trading ./hermes_trading
# Bundled defaults — copied to the volume on first boot by entrypoint.sh
COPY state ./state_init
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

RUN uv sync --no-dev

ENV HERMES_TRADING_MODE=paper
ENV HERMES_TRADING_I_ACCEPT_RISK=false

CMD ["./entrypoint.sh"]
