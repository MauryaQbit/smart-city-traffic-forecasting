FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY models ./models
COPY reports ./reports

RUN pip install --no-cache-dir -e ".[lstm,dashboard]"

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/traffic_forecast/dashboard/app.py", \
     "--server.headless", "true", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501"]
