FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_ROOT=/var/lib/hedge-fund \
    LOG_DIR=/var/log/hedge-fund \
    OPTIMAL_PARAMS_PATH=/var/lib/hedge-fund/optimal_params.json

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    requests \
    optuna \
    xgboost \
    pandas_ta \
    alpaca-trade-api \
    pyarrow \
    yfinance \
    websockets \
    tzdata \
    rich \
    joblib \
    scikit-learn \
    lightgbm \
    shap \
    python-dotenv

RUN mkdir -p /var/lib/hedge-fund /var/log/hedge-fund

RUN chmod +x /app/docker/entrypoint.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
  CMD pgrep -f "python bot.py" >/dev/null || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
