FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data /app/models /app/logs

COPY main.py /app/
COPY data_processor.py /app/
COPY models.py /app/
COPY collaborative_filter_recommender.py /app/

COPY app/ /app/

ENV MODEL_PATH="/app/models/recommender_model.pkl"
ENV REDIS_HOST="redis"
ENV REDIS_PORT=6379
ENV LOG_LEVEL="INFO"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
