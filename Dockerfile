FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.cache/hf \
    PYTHONPATH=/app/src:/app/vendor/dronedocs-rag/src DRONEDOCS_ROOT=/app/vendor/dronedocs-rag
WORKDIR /app
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY vendor/dronedocs-rag/requirements.txt vendor/dronedocs-rag/requirements.txt
RUN pip install -r vendor/dronedocs-rag/requirements.txt opentelemetry-sdk==1.45.2 \
    opentelemetry-exporter-otlp-proto-http==1.45.2 matplotlib==3.11.2 streamlit==1.67.0
COPY . .
RUN pip install -e vendor/dronedocs-rag --no-deps && pip install -e . --no-deps \
 && python -m dronedocs.cli ingest
EXPOSE 8000 8501
CMD ["uvicorn", "ragobs.api:app", "--host", "0.0.0.0", "--port", "8000"]
