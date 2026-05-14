FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-local.txt .
RUN pip install --no-cache-dir -r requirements-local.txt

COPY app/ app/
COPY src/ src/
COPY models/ models/
COPY scaler.pkl .

EXPOSE 8501

ENV APP_MODE=mlflow
ENV FRAUD_THRESHOLD=0.45

HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
