FROM python:3.14-slim

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# The whole `ml` package is required at runtime: the serialized pipelines
# reference ml.preprocessing.column_transformer by module path when
# unpickled, and the inference service reuses ml.data.features directly.
COPY ml ml
COPY backend backend

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
